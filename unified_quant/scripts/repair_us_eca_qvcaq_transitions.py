#!/usr/bin/env python3
"""Repair the exact ECA->OVV and QVCGA->QVCAQ market transitions.

The repair deliberately separates acquisition, review, planning, and apply:

* ``--fetch-stage1`` makes exactly three official-source requests and six
  EODHD requests.  Every request is one-shot with no retry.  The EODHD calls
  are protected by the persistent cap/reserve ledger and all nine raw bodies
  are written to a content-addressed quarantine.
* ``--resume-quarantine`` accepts only the one reviewed official-only failure
  pinned in this file, reuses its three exact official raws, and makes only
  the six remaining one-shot EODHD requests.
* ``--promote-quarantine`` is offline.  It succeeds only after a reviewer has
  filled every exact SHA-256 pin in ``REVIEWED_ARTIFACT_SHA256``.
* ``--offline-plan`` (the default) and ``--apply`` never construct an HTTP
  client.  Apply is writer-locked, CAS-guarded, journaled, and rolls release
  and dataset pointers back on any failure.

ECA closed on 2020-01-24.  The official reorganization delivered one OVV
share for every five ECA shares, and OVV first traded on the NYSE on
2020-01-27.  ECA provider rows after 2020-01-24 are therefore contamination,
not a continuing ECA listing.  QVCGA and QVCAQ, by contrast, are the same
legal common-share identity: QVCGA left Nasdaq after 2026-04-23 and continued
on OTCID as QVCAQ on 2026-04-24.  The existing 2025 1-for-50 split remains on
that same identity.

This script never accesses R2.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import html
import json
import math
import os
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from supertrend_quant.env import load_env
from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.ingest import EodhdCallBudget, SourceArtifact
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


DEFAULT_CACHE_ROOT = Path("data/cache")
OPERATION = "repair_us_eca_qvcaq_transitions"
POLICY = "exact_eca_reorganization_and_qvc_otcid_continuity/v1"
TRANSACTION_DIR = "transactions/us-eca-qvcaq-transitions"
RECOVERY_DIR = "recovery/us-eca-qvcaq-transitions"

ECA_ID = "US:EODHD:53a2ef22-39f2-506c-80b4-bb0f698f43dd"
OVV_ID = "US:EODHD:9e7f1eca-2773-5c4d-8bb5-0443d36be2ea"
QVCGA_ID = "US:EODHD:5eca6dac-4c4c-50af-9fc2-17c5839a4efc"

# Keep successor identity generation identical to us_bootstrap.py:280.  The
# provider code is ``OVV`` (without the transport suffix ``.US``); including
# that suffix would silently create a non-canonical duplicate identity.
_CANONICAL_OVV_ID = "US:EODHD:" + str(
    uuid.uuid5(uuid.NAMESPACE_URL, "eodhd:US:OVV:symbol:OVV")
)
if OVV_ID != _CANONICAL_OVV_ID:
    raise RuntimeError("OVV canonical bootstrap security_id formula changed.")

ECA_SYMBOL = "ECA"
OVV_SYMBOL = "OVV"
QVCGA_SYMBOL = "QVCGA"
QVCAQ_SYMBOL = "QVCAQ"
ECA_ISSUER_NAME = "Encana Corporation"
OVV_ISSUER_NAME = "Ovintiv Inc"
ECA_LAST = "2020-01-24"
OVV_FIRST = "2020-01-27"
QVCGA_LAST = "2026-04-23"
QVCAQ_FIRST = "2026-04-24"
FETCH_END = "2026-07-15"
ECA_RATIO = 0.2
QVC_2025_SPLIT_EVENT_ID = (
    "985d7f16eda88208c7d4c898ecd57d86a3d57e0247d678a4b2e052da25e2277c"
)
QVC_2025_SPLIT_DATE = "2025-05-23"
QVC_2025_SPLIT_RATIO = 0.02
QVC_2025_SPLIT_URL = (
    "https://eodhd.com/api/splits/QVCGA.US?from=2015-01-01&to=2026-07-15"
)
QVC_2025_SPLIT_SHA256 = (
    "2692b402f69995c6a1bd17d85014e3a34716a74dddeb2b5ee1a410a28175965c"
)
QVC_2025_SPLIT_RETRIEVED_AT = "2026-07-17T20:37:21.435411Z"
ECA_EVENT_ID = canonical_lifecycle_event_id(ECA_ID, "stock_merger", OVV_FIRST)
QVC_EVENT_ID = canonical_lifecycle_event_id(QVCGA_ID, "ticker_change", QVCAQ_FIRST)

ECA_PRIMARY_URL = (
    "https://investor.ovintiv.com/2020-01-24-Encana-Completes-"
    "Reorganization-and-Establishes-Corporate-Domicile-in-the-U-S"
)
ECA_SEC_URL = (
    "https://www.sec.gov/Archives/edgar/data/876661/"
    "000087666120000056/ruleprovisionnotice.htm"
)
QVC_OFFICIAL_URL = "https://investors.qvcgrp.com/investors/stock-cost-basis"
OFFICIAL_URLS = (ECA_PRIMARY_URL, ECA_SEC_URL, QVC_OFFICIAL_URL)
OFFICIAL_SOURCES = (
    "ovintiv_issuer_reorganization",
    "sec_rule_provision_notice",
    "qvc_issuer_stock_cost_basis",
)

EODHD_ENDPOINTS = ("eod", "div", "splits")
PROVIDER_SPECS = {
    OVV_SYMBOL: {
        "provider_symbol": "OVV.US",
        "security_id": OVV_ID,
        "start": OVV_FIRST,
        "end": FETCH_END,
    },
    QVCAQ_SYMBOL: {
        "provider_symbol": "QVCAQ.US",
        "security_id": QVCGA_ID,
        "start": QVCAQ_FIRST,
        "end": FETCH_END,
    },
}
EXPECTED_OFFICIAL_CALLS = 3
EXPECTED_EODHD_CALLS = 6
EXPECTED_TOTAL_CALLS = 9


def _request_url(symbol: str, endpoint: str) -> str:
    spec = PROVIDER_SPECS[symbol]
    return (
        f"https://eodhd.com/api/{endpoint}/{spec['provider_symbol']}"
        f"?from={spec['start']}&to={spec['end']}"
    )


REQUEST_ORDER = tuple(
    (symbol, endpoint)
    for symbol in (OVV_SYMBOL, QVCAQ_SYMBOL)
    for endpoint in EODHD_ENDPOINTS
)
REQUEST_URLS = {
    (symbol, endpoint): _request_url(symbol, endpoint)
    for symbol, endpoint in REQUEST_ORDER
}

# Stage 1 only emits hashes.  A human reviewer must inspect the exact raw
# bodies and fill all nine pins before offline promotion is possible.  The
# three official bodies below were captured and reviewed on 2026-07-18; the
# six provider pins remain pending.  Keeping these values in code prevents a
# newly fetched page or revised provider response from silently authorizing
# itself.
REVIEWED_ARTIFACT_SHA256: Mapping[str, str] = {
    ECA_PRIMARY_URL: "cb6cdb670b3a30d38f0529d242f4ea470052c04204e3101537627f7df3955bef",
    ECA_SEC_URL: "58d199861b620211b63c846e3184baf1ff7982adb124e085c5f726e2fd06af59",
    QVC_OFFICIAL_URL: "55829c9064eee534b6f79027648172494a507f8b9be16e9598dc57cdd58c165b",
    REQUEST_URLS[(OVV_SYMBOL, "eod")]: "2911e9b1eb3e59f3649f1a7ccef3b3a62b6b2667ed910aca8e335001afceafca",
    REQUEST_URLS[(OVV_SYMBOL, "div")]: "63d125e117f9eeb8dcfb65833216553b46010f91abec2992ccc5e28c290f7fa6",
    REQUEST_URLS[(OVV_SYMBOL, "splits")]: "195def5749f8d07f7311576b9470a2cf2c22a8b866bb2e263763311e828793b5",
    REQUEST_URLS[(QVCAQ_SYMBOL, "eod")]: "66a03be49bab3e158b6133fb2e49897008e90acbd4629ff11c812d6ee46f76aa",
    REQUEST_URLS[(QVCAQ_SYMBOL, "div")]: "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    REQUEST_URLS[(QVCAQ_SYMBOL, "splits")]: "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
}

RESUMABLE_OFFICIAL_SHA256: Mapping[str, str] = {
    url: REVIEWED_ARTIFACT_SHA256[url] for url in OFFICIAL_URLS
}
RESUMABLE_OFFICIAL_QUARANTINE_ID = (
    "d784da1588c64351e9eb673884be793635f250c2b6b0fa3f1cb18080fe614ce5"
)
RESUMABLE_OFFICIAL_ERROR = (
    "ValueError: Reviewed Ovintiv issuer evidence lost required claim groups: "
    "[('one ovintiv', 'one common share of ovintiv'), "
    "('every five encana', 'for each five encana')]"
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
REQUIRED_DATASETS = WRITE_DATASETS


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _date(value: Any) -> str:
    value = _text(value)
    if not value:
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _safe_hash(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", _text(value).lower()))


def _normalized_html(content: bytes) -> str:
    decoded = content.decode("utf-8", errors="replace")
    decoded = re.sub(r"(?is)<script\b.*?</script>|<style\b.*?</style>", " ", decoded)
    decoded = re.sub(r"(?s)<[^>]+>", " ", decoded)
    return " ".join(html.unescape(decoded).lower().split())


def _event_id(source: str, security_id: str, action_type: str, date: str) -> str:
    return hashlib.sha256(
        f"{source}|{security_id}|{action_type}|{date}".encode("utf-8")
    ).hexdigest()


def _adjustment_source_version(price_version: str, action_version: str) -> str:
    price = _text(price_version)
    action = _text(action_version)
    if not price or not action:
        raise ValueError("Adjustment-factor source versions must be non-empty.")
    return f"{price}+{action}"


@dataclass(frozen=True)
class ReviewedBundle:
    artifacts: tuple[SourceArtifact, ...]
    prices: pd.DataFrame
    provider_actions: pd.DataFrame
    official_http_attempts: int
    eodhd_http_attempts: int
    budget_receipt: Mapping[str, Any]
    evidence_claims: Mapping[str, Any]


@dataclass(frozen=True)
class RawQuarantine:
    quarantine_id: str
    path: Path
    artifacts: tuple[SourceArtifact, ...]
    budget_receipt: Mapping[str, Any]


@dataclass
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: dict[str, str | None]
    planned_versions: dict[str, str]
    frames: dict[str, pd.DataFrame]
    artifacts: tuple[SourceArtifact, ...]
    summary: dict[str, Any]


class ExactStage1Client:
    """One exact HTTP attempt per reviewed request, with no retry path."""

    def __init__(
        self,
        *,
        session: Any,
        token: str,
        user_agent: str,
        budget: EodhdCallBudget,
    ):
        if not token:
            raise RuntimeError("EODHD_API_TOKEN is required for stage-1 collection.")
        normalized_user_agent = _text(user_agent)
        if not normalized_user_agent or "@" not in normalized_user_agent:
            raise RuntimeError(
                "SEC_USER_AGENT with a truthful contact email is required for "
                "stage-1 official-source collection."
            )
        self.session = session
        self.token = token
        self.user_agent = normalized_user_agent
        self.budget = budget
        self.official_attempts: list[str] = []
        self.eodhd_attempts: list[tuple[str, str]] = []
        self.eodhd_claim_positions: list[int] = []

    def get_official(self, url: str, *, retrieved_at: str) -> SourceArtifact:
        position = len(self.official_attempts)
        if position >= EXPECTED_OFFICIAL_CALLS or url != OFFICIAL_URLS[position]:
            raise RuntimeError("Stage-1 client refused an unreviewed official request.")
        self.official_attempts.append(url)
        response = self.session.get(
            url,
            headers={
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Encoding": "identity",
                "User-Agent": self.user_agent,
            },
            timeout=45,
            allow_redirects=False,
        )
        status = int(getattr(response, "status_code", 0) or 0)
        if 300 <= status < 400:
            raise RuntimeError(
                f"Stage-1 official source returned forbidden redirect HTTP {status}."
            )
        response.raise_for_status()
        return SourceArtifact(
            source=OFFICIAL_SOURCES[position],
            source_url=url,
            retrieved_at=retrieved_at,
            content=bytes(response.content),
            content_type=str(
                getattr(response, "headers", {}).get("Content-Type", "text/html")
            ),
        )

    def get_eodhd(
        self, symbol: str, endpoint: str, *, retrieved_at: str
    ) -> SourceArtifact:
        position = len(self.eodhd_attempts)
        if position >= EXPECTED_EODHD_CALLS or (symbol, endpoint) != REQUEST_ORDER[position]:
            raise RuntimeError("Stage-1 client refused an unreviewed EODHD request.")
        spec = PROVIDER_SPECS[symbol]
        claim_position = int(self.budget.claim())
        self.eodhd_claim_positions.append(claim_position)
        self.eodhd_attempts.append((symbol, endpoint))
        try:
            response = self.session.get(
                f"https://eodhd.com/api/{endpoint}/{spec['provider_symbol']}",
                params={
                    "from": spec["start"],
                    "to": spec["end"],
                    "api_token": self.token,
                    "fmt": "json",
                },
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                    "User-Agent": self.user_agent,
                },
                timeout=120,
                allow_redirects=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Stage-1 EODHD request failed before a response: "
                f"{symbol}/{endpoint} ({type(exc).__name__})."
            ) from None
        status = int(getattr(response, "status_code", 0) or 0)
        if 300 <= status < 400:
            raise RuntimeError(
                f"Stage-1 EODHD source returned forbidden redirect HTTP {status}."
            )
        try:
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                "Stage-1 EODHD response failed: "
                f"{symbol}/{endpoint}, HTTP {status or 'unknown'} "
                f"({type(exc).__name__})."
            ) from None
        return SourceArtifact(
            source=f"eodhd_{symbol.lower()}_{endpoint}",
            source_url=REQUEST_URLS[(symbol, endpoint)],
            retrieved_at=retrieved_at,
            content=bytes(response.content),
            content_type=str(
                getattr(response, "headers", {}).get("Content-Type", "application/json")
            ),
        )


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


def _assert_budget_capacity(budget: EodhdCallBudget, used: int) -> None:
    if used + EXPECTED_EODHD_CALLS > budget.ceiling:
        raise RuntimeError(
            "EODHD stage-1 preflight refused a partial six-call acquisition: "
            f"used={used}, required={EXPECTED_EODHD_CALLS}, "
            f"safety_ceiling={budget.ceiling}, daily_limit={budget.limit}, "
            f"reserve={budget.reserve}."
        )


def _budget_receipt(
    budget: EodhdCallBudget,
    *,
    used_before: int,
    used_after: int,
    claim_positions: Iterable[int],
) -> dict[str, Any]:
    positions = [int(value) for value in claim_positions]
    return {
        "schema": "eodhd_budget_receipt/v2",
        "period": budget.period,
        "used_before": int(used_before),
        "used_after": int(used_after),
        "delta": int(used_after) - int(used_before),
        # Global ledger growth may include interleaved claims from another
        # collector.  These positions prove only this client's own reservations.
        "own_claim_count": len(positions),
        "claim_positions": positions,
        "daily_limit": int(budget.limit),
        "reserve": int(budget.reserve),
        "safety_ceiling": int(budget.ceiling),
    }


def _signature() -> dict[str, Any]:
    return {
        "schema": "us_eca_qvcaq_stage1_signature/v1",
        "policy": POLICY,
        "official_urls": list(OFFICIAL_URLS),
        "eodhd_request_order": [
            {
                "symbol": symbol,
                "endpoint": endpoint,
                "url": REQUEST_URLS[(symbol, endpoint)],
            }
            for symbol, endpoint in REQUEST_ORDER
        ],
        "official_http_attempts": EXPECTED_OFFICIAL_CALLS,
        "eodhd_http_attempts": EXPECTED_EODHD_CALLS,
        "retry_count": 0,
    }


def _artifact_rows(artifacts: Iterable[SourceArtifact]) -> list[dict[str, Any]]:
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


def _validate_receipt(receipt: Mapping[str, Any], *, complete: bool) -> dict[str, Any]:
    value = dict(receipt)
    required = {
        "schema",
        "period",
        "used_before",
        "used_after",
        "delta",
        "own_claim_count",
        "claim_positions",
        "daily_limit",
        "reserve",
        "safety_ceiling",
    }
    if set(value) != required or value.get("schema") != "eodhd_budget_receipt/v2":
        raise ValueError("Stage-1 EODHD budget receipt schema changed.")
    if int(value["used_after"]) - int(value["used_before"]) != int(value["delta"]):
        raise ValueError("Stage-1 EODHD budget receipt arithmetic changed.")
    positions = value["claim_positions"]
    if not isinstance(positions, list):
        raise ValueError("Stage-1 EODHD claim positions changed.")
    try:
        normalized_positions = [int(item) for item in positions]
    except (TypeError, ValueError) as exc:
        raise ValueError("Stage-1 EODHD claim positions changed.") from exc
    own_claim_count = int(value["own_claim_count"])
    if (
        own_claim_count != len(normalized_positions)
        or normalized_positions != sorted(set(normalized_positions))
        or any(
            position <= int(value["used_before"])
            or position > int(value["used_after"])
            or position > int(value["safety_ceiling"])
            for position in normalized_positions
        )
        or own_claim_count > int(value["delta"])
    ):
        raise ValueError("Stage-1 EODHD own-claim proof changed.")
    if complete and own_claim_count != EXPECTED_EODHD_CALLS:
        raise ValueError("Complete stage-1 receipt must prove exactly six own EODHD calls.")
    if int(value["safety_ceiling"]) != int(value["daily_limit"]) - int(value["reserve"]):
        raise ValueError("Stage-1 EODHD budget safety ceiling changed.")
    return value


def _write_quarantine(
    cache_root: Path,
    artifacts: Iterable[SourceArtifact],
    receipt: Mapping[str, Any],
    *,
    status: str,
    error: str = "",
) -> tuple[str, Path]:
    items = tuple(artifacts)
    if status not in {"complete_unreviewed", "incomplete"}:
        raise ValueError("Unknown stage-1 quarantine status.")
    if status == "complete_unreviewed" and len(items) != EXPECTED_TOTAL_CALLS:
        raise ValueError("Complete stage-1 quarantine must contain exactly nine raws.")
    envelope = {
        "schema": "us_eca_qvcaq_raw_quarantine/v1",
        "signature": _signature(),
        "status": status,
        "error": _text(error),
        "budget_receipt": dict(receipt),
        "artifacts": _artifact_rows(items),
    }
    content = _canonical_json_bytes(envelope)
    quarantine_id = sha256_bytes(content)
    path = (
        cache_root
        / "state/us-eca-qvcaq-transitions/quarantine"
        / f"{quarantine_id}.json.gz"
    )
    encoded = gzip.compress(content, mtime=0)
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Immutable quarantine collision: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(path, encoded)
    return quarantine_id, path


def _quarantine_path(cache_root: Path, quarantine_id: str) -> Path:
    normalized = _text(quarantine_id).lower()
    if not _safe_hash(normalized):
        raise ValueError("Quarantine id must be an exact lowercase SHA-256.")
    return (
        cache_root
        / "state/us-eca-qvcaq-transitions/quarantine"
        / f"{normalized}.json.gz"
    )


def _read_quarantine_envelope(
    cache_root: Path, quarantine_id: str
) -> tuple[Path, dict[str, Any]]:
    path = _quarantine_path(cache_root, quarantine_id)
    if not path.is_file():
        raise FileNotFoundError(f"Stage-1 quarantine does not exist: {path}")
    try:
        content = gzip.decompress(path.read_bytes())
        envelope = json.loads(content)
    except Exception as exc:
        raise ValueError(f"Stage-1 quarantine is unreadable: {path}") from exc
    if content != _canonical_json_bytes(envelope):
        raise ValueError("Stage-1 quarantine is not canonical JSON.")
    if sha256_bytes(content) != _text(quarantine_id).lower():
        raise ValueError("Stage-1 quarantine content-address hash changed.")
    if set(envelope) != {
        "schema",
        "signature",
        "status",
        "error",
        "budget_receipt",
        "artifacts",
    } or envelope.get("schema") != "us_eca_qvcaq_raw_quarantine/v1":
        raise ValueError("Stage-1 quarantine wrapper schema changed.")
    if envelope.get("signature") != _signature():
        raise ValueError("Stage-1 quarantine acquisition signature changed.")
    return path, envelope


def _artifacts_from_rows(
    rows: Any,
    expected_urls: Iterable[str],
    *,
    expected_sources: Iterable[str] | None = None,
) -> tuple[SourceArtifact, ...]:
    urls = tuple(expected_urls)
    sources = tuple(expected_sources) if expected_sources is not None else None
    if not isinstance(rows, list) or len(rows) != len(urls):
        raise ValueError("Stage-1 quarantine raw inventory changed.")
    if sources is not None and len(sources) != len(urls):
        raise ValueError("Stage-1 expected source inventory changed.")
    artifacts: list[SourceArtifact] = []
    for index, (row, expected_url) in enumerate(zip(rows, urls, strict=True)):
        if not isinstance(row, Mapping) or set(row) != {
            "source",
            "source_url",
            "retrieved_at",
            "content_type",
            "content_sha256",
            "content_base64",
        } or row.get("source_url") != expected_url:
            raise ValueError("Stage-1 artifact URL/order changed.")
        if sources is not None and row.get("source") != sources[index]:
            raise ValueError("Stage-1 artifact source/order changed.")
        try:
            raw = base64.b64decode(row["content_base64"], validate=True)
        except Exception as exc:
            raise ValueError("Stage-1 raw body is not valid base64.") from exc
        if sha256_bytes(raw) != row.get("content_sha256"):
            raise ValueError("Stage-1 raw body hash changed.")
        artifacts.append(
            SourceArtifact(
                source=str(row["source"]),
                source_url=str(row["source_url"]),
                retrieved_at=str(row["retrieved_at"]),
                content=raw,
                content_type=str(row["content_type"]),
            )
        )
    return tuple(artifacts)


def read_quarantine(cache_root: Path, quarantine_id: str) -> RawQuarantine:
    path, envelope = _read_quarantine_envelope(cache_root, quarantine_id)
    if envelope.get("status") != "complete_unreviewed":
        raise ValueError("Only a complete stage-1 quarantine can be promoted.")
    receipt = _validate_receipt(envelope["budget_receipt"], complete=True)
    rows = envelope.get("artifacts")
    expected_urls = (*OFFICIAL_URLS, *(REQUEST_URLS[key] for key in REQUEST_ORDER))
    artifacts = _artifacts_from_rows(rows, expected_urls)
    return RawQuarantine(
        quarantine_id=_text(quarantine_id).lower(),
        path=path,
        artifacts=artifacts,
        budget_receipt=receipt,
    )


def _validate_reviewer_pins(artifacts: Iterable[SourceArtifact]) -> None:
    if set(REVIEWED_ARTIFACT_SHA256) != {
        *OFFICIAL_URLS,
        *REQUEST_URLS.values(),
    }:
        raise ValueError("Reviewed artifact URL inventory changed.")
    pending = sorted(
        url for url, digest in REVIEWED_ARTIFACT_SHA256.items() if not _safe_hash(digest)
    )
    if pending:
        raise ValueError(
            "Reviewer SHA-256 pins are still pending for: " + ", ".join(pending)
        )
    observed = {item.source_url: item.source_hash for item in artifacts}
    if observed != {url: digest.lower() for url, digest in REVIEWED_ARTIFACT_SHA256.items()}:
        raise ValueError("Reviewed exact artifact SHA-256 pin mismatch.")


def _required_groups(text: str, groups: Iterable[Iterable[str]], label: str) -> None:
    missing = [tuple(group) for group in groups if not any(term in text for term in group)]
    if missing:
        raise ValueError(f"Reviewed {label} evidence lost required claim groups: {missing}")


def _validate_official_evidence(
    artifacts: Mapping[str, SourceArtifact],
) -> dict[str, Any]:
    primary = _normalized_html(artifacts[ECA_PRIMARY_URL].content)
    sec = _normalized_html(artifacts[ECA_SEC_URL].content)
    qvc = _normalized_html(artifacts[QVC_OFFICIAL_URL].content)
    _required_groups(
        primary,
        (
            ("encana completes reorganization", "completed its reorganization"),
            ("ovintiv",),
            ("january 24, 2020", "jan. 24, 2020", "jan 24, 2020"),
            (
                "one ovintiv",
                "one common share of ovintiv",
                "one share of common stock of ovintiv",
            ),
            (
                "every five encana",
                "for each five encana",
                "for every five common shares of encana",
            ),
            ("new york stock exchange", "nyse"),
            ("january 27, 2020", "jan. 27, 2020", "jan 27, 2020"),
            ("ovv",),
        ),
        "Ovintiv issuer",
    )
    _required_groups(
        sec,
        (
            ("encana",),
            ("ovintiv",),
            ("rule 12d2-2", "rule 12d2 2", "17 cfr 240.12d2-2"),
            ("january 24, 2020", "jan. 24, 2020", "jan 24, 2020"),
            ("january 27, 2020", "jan. 27, 2020", "jan 27, 2020"),
            ("every five shares", "every five common shares"),
            ("exchanged for one share", "one share of common stock of ovintiv"),
            ("continued listing on the nyse", "continued listing on nyse"),
        ),
        "SEC rule-provision",
    )
    _required_groups(
        qvc,
        (
            ("qvcga",),
            ("qvcgp",),
            ("qvcaq",),
            ("qvcpq",),
            ("nasdaq",),
            ("otcid",),
            ("removed", "delisted", "no longer listed"),
            (
                "began trading",
                "commenced trading",
                "will begin trading",
                "trade on",
                "trading on",
            ),
            ("april 24, 2026", "april 24 2026", "4/24/2026"),
        ),
        "QVC issuer",
    )
    return {
        "eca_legal_completion": ECA_LAST,
        "eca_exchange_ratio": ECA_RATIO,
        "ovv_first_nyse_session": OVV_FIRST,
        "qvcga_last_nasdaq_session": QVCGA_LAST,
        "qvcaq_first_otcid_session": QVCAQ_FIRST,
        "qvc_identity_decision": "same_legal_security_same_security_id",
    }


def _split_ratio(value: Any) -> float | None:
    text = _text(value)
    if not text:
        return None
    for separator in ("/", ":"):
        if separator in text:
            left, right = text.split(separator, 1)
            try:
                result = float(left) / float(right)
            except (TypeError, ValueError, ZeroDivisionError):
                return None
            return result if math.isfinite(result) and result > 0 else None
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _price_frame(
    symbol: str, rows: Any, artifact: SourceArtifact
) -> pd.DataFrame:
    if not isinstance(rows, list):
        raise ValueError(f"{symbol} EOD payload is not a row list.")
    spec = PROVIDER_SPECS[symbol]
    records: list[dict[str, Any]] = []
    for row in rows:
        session = _date(row.get("date"))
        if not session:
            continue
        if not spec["start"] <= session <= spec["end"]:
            raise ValueError(f"{symbol} EOD row is outside the exact request range.")
        records.append(
            {
                "security_id": spec["security_id"],
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
    columns = tuple(
        dict.fromkeys((*dataset_spec("daily_price_raw").required_columns, "source_url"))
    )
    frame = pd.DataFrame(records, columns=columns)
    if frame.empty:
        raise ValueError(f"{symbol} EOD payload is empty.")
    sessions = tuple(frame["session"].astype(str).sort_values())
    if sessions[0] != spec["start"] or sessions[-1] != spec["end"]:
        raise ValueError(f"{symbol} EOD boundary does not match the exact request.")
    if frame["session"].astype(str).duplicated().any():
        raise ValueError(f"{symbol} EOD payload contains duplicate sessions.")
    numeric = frame[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite = numeric.map(lambda item: math.isfinite(float(item))).all().all()
    if not finite:
        raise ValueError(f"{symbol} EOD payload contains non-finite OHLCV.")
    if (
        numeric[["open", "high", "low", "close"]].le(0).any().any()
        or numeric["volume"].lt(0).any()
        or numeric["low"].gt(numeric[["open", "close"]].min(axis=1)).any()
        or numeric["high"].lt(numeric[["open", "close"]].max(axis=1)).any()
        or numeric["low"].gt(numeric["high"]).any()
    ):
        raise ValueError(f"{symbol} EOD payload contains invalid OHLCV envelopes.")
    return frame


def _provider_action(
    symbol: str,
    artifact: SourceArtifact,
    *,
    action_type: str,
    date: str,
    cash_amount: float | None = None,
    ratio: float | None = None,
    row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = dict(row or {})
    security_id = str(PROVIDER_SPECS[symbol]["security_id"])
    return {
        "event_id": _event_id(artifact.source, security_id, action_type, date),
        "security_id": security_id,
        "action_type": action_type,
        "effective_date": date,
        "ex_date": date,
        "announcement_date": _date(row.get("declarationDate")),
        "record_date": _date(row.get("recordDate")),
        "payment_date": _date(row.get("paymentDate")),
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


def _action_frame(
    symbol: str, endpoint: str, rows: Any, artifact: SourceArtifact
) -> tuple[pd.DataFrame, int]:
    if not isinstance(rows, list):
        raise ValueError(f"{symbol} {endpoint} payload is not a row list.")
    spec = PROVIDER_SPECS[symbol]
    records: list[dict[str, Any]] = []
    transition_splits_removed = 0
    for row in rows:
        date = _date(row.get("date"))
        if not date:
            continue
        if not spec["start"] <= date <= spec["end"]:
            raise ValueError(f"{symbol} {endpoint} row is outside the exact request range.")
        if endpoint == "div":
            try:
                amount = float(row.get("unadjustedValue", row.get("value")))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{symbol} dividend amount is invalid on {date}.") from exc
            if not math.isfinite(amount) or amount <= 0:
                raise ValueError(f"{symbol} dividend amount is invalid on {date}.")
            records.append(
                _provider_action(
                    symbol,
                    artifact,
                    action_type="cash_dividend",
                    date=date,
                    cash_amount=amount,
                    row=row,
                )
            )
        else:
            ratio = _split_ratio(row.get("split"))
            if ratio is None:
                raise ValueError(f"{symbol} split ratio is invalid on {date}.")
            if symbol == OVV_SYMBOL and date == OVV_FIRST:
                if abs(ratio - ECA_RATIO) > 1e-12:
                    raise ValueError("OVV transition split conflicts with official 1-for-5 ratio.")
                transition_splits_removed += 1
                continue
            records.append(
                _provider_action(
                    symbol,
                    artifact,
                    action_type="split",
                    date=date,
                    ratio=ratio,
                    row=row,
                )
            )
    columns = tuple(
        dict.fromkeys((*dataset_spec("corporate_actions").required_columns, "metadata"))
    )
    return pd.DataFrame(records, columns=columns), transition_splits_removed


def bundle_from_artifacts(
    artifacts: Iterable[SourceArtifact],
    *,
    official_http_attempts: int,
    eodhd_http_attempts: int,
    budget_receipt: Mapping[str, Any],
    require_reviewer_pins: bool,
) -> ReviewedBundle:
    items = tuple(artifacts)
    expected_urls = (*OFFICIAL_URLS, *(REQUEST_URLS[key] for key in REQUEST_ORDER))
    if (
        len(items) != EXPECTED_TOTAL_CALLS
        or tuple(item.source_url for item in items) != expected_urls
    ):
        raise ValueError("Reviewed bundle URL inventory/order is not exactly nine.")
    if official_http_attempts != EXPECTED_OFFICIAL_CALLS:
        raise ValueError("Reviewed bundle must prove exactly three official calls.")
    if eodhd_http_attempts != EXPECTED_EODHD_CALLS:
        raise ValueError("Reviewed bundle must prove exactly six EODHD calls.")
    receipt = _validate_receipt(budget_receipt, complete=True)
    if require_reviewer_pins:
        _validate_reviewer_pins(items)
    official = {item.source_url: item for item in items[:EXPECTED_OFFICIAL_CALLS]}
    claims = _validate_official_evidence(official)
    by_request = {
        key: artifact
        for key, artifact in zip(REQUEST_ORDER, items[EXPECTED_OFFICIAL_CALLS:], strict=True)
    }
    price_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []
    removed = 0
    for symbol in (OVV_SYMBOL, QVCAQ_SYMBOL):
        eod = by_request[(symbol, "eod")]
        price_frames.append(_price_frame(symbol, json.loads(eod.content), eod))
        for endpoint in ("div", "splits"):
            artifact = by_request[(symbol, endpoint)]
            frame, count = _action_frame(
                symbol, endpoint, json.loads(artifact.content), artifact
            )
            action_frames.append(frame)
            removed += count
    bundle = ReviewedBundle(
        artifacts=items,
        prices=pd.concat(price_frames, ignore_index=True, sort=False),
        provider_actions=pd.concat(action_frames, ignore_index=True, sort=False),
        official_http_attempts=official_http_attempts,
        eodhd_http_attempts=eodhd_http_attempts,
        budget_receipt=receipt,
        evidence_claims={
            **claims,
            "ovv_transition_provider_split_rows_suppressed": removed,
        },
    )
    if bundle.provider_actions["event_id"].astype(str).duplicated().any():
        raise ValueError("Reviewed provider actions contain duplicate event IDs.")
    return bundle


def _bundle_cache_path(cache_root: Path) -> Path:
    digest = sha256_bytes(_canonical_json_bytes(_signature()))
    return cache_root / "state/us-eca-qvcaq-transitions" / f"{digest}.reviewed.json.gz"


def _write_bundle_cache(cache_root: Path, bundle: ReviewedBundle) -> Path:
    _validate_reviewer_pins(bundle.artifacts)
    payload = {
        "signature": _signature(),
        "official_http_attempts": bundle.official_http_attempts,
        "eodhd_http_attempts": bundle.eodhd_http_attempts,
        "budget_receipt": dict(bundle.budget_receipt),
        "reviewed_pins": dict(REVIEWED_ARTIFACT_SHA256),
        "artifacts": _artifact_rows(bundle.artifacts),
    }
    envelope = {
        "schema": "us_eca_qvcaq_reviewed_bundle/v1",
        "payload": payload,
        "payload_sha256": sha256_bytes(_canonical_json_bytes(payload)),
    }
    content = _canonical_json_bytes(envelope)
    path = _bundle_cache_path(cache_root)
    encoded = gzip.compress(content, mtime=0)
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Immutable reviewed bundle conflict: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(path, encoded)
    return path


def _read_bundle_cache(cache_root: Path) -> ReviewedBundle | None:
    path = _bundle_cache_path(cache_root)
    if not path.is_file():
        return None
    try:
        content = gzip.decompress(path.read_bytes())
        envelope = json.loads(content)
    except Exception as exc:
        raise ValueError(f"Reviewed bundle cache is unreadable: {path}") from exc
    if content != _canonical_json_bytes(envelope):
        raise ValueError("Reviewed bundle cache is not canonical JSON.")
    if set(envelope) != {"schema", "payload", "payload_sha256"} or envelope.get(
        "schema"
    ) != "us_eca_qvcaq_reviewed_bundle/v1":
        raise ValueError("Reviewed bundle cache wrapper changed.")
    payload = envelope["payload"]
    if envelope.get("payload_sha256") != sha256_bytes(_canonical_json_bytes(payload)):
        raise ValueError("Reviewed bundle cache payload hash changed.")
    if payload.get("signature") != _signature():
        raise ValueError("Reviewed bundle acquisition signature changed.")
    if payload.get("reviewed_pins") != dict(REVIEWED_ARTIFACT_SHA256):
        raise ValueError("Reviewed bundle code pins changed.")
    artifacts: list[SourceArtifact] = []
    for row in payload.get("artifacts", []):
        raw = base64.b64decode(row["content_base64"], validate=True)
        if sha256_bytes(raw) != row.get("content_sha256"):
            raise ValueError("Reviewed bundle raw body hash changed.")
        artifacts.append(
            SourceArtifact(
                source=str(row["source"]),
                source_url=str(row["source_url"]),
                retrieved_at=str(row["retrieved_at"]),
                content=raw,
                content_type=str(row["content_type"]),
            )
        )
    return bundle_from_artifacts(
        artifacts,
        official_http_attempts=int(payload.get("official_http_attempts", 0)),
        eodhd_http_attempts=int(payload.get("eodhd_http_attempts", 0)),
        budget_receipt=payload.get("budget_receipt", {}),
        require_reviewer_pins=True,
    )


def _collect_eodhd_artifacts(
    client: ExactStage1Client,
    artifacts: list[SourceArtifact],
    *,
    retrieved_at: str,
) -> None:
    for symbol, endpoint in REQUEST_ORDER:
        artifact = client.get_eodhd(symbol, endpoint, retrieved_at=retrieved_at)
        artifacts.append(artifact)
        # Validate each body immediately so a bad EOD response does not spend
        # calls on div/splits, and a bad action response does not spend calls
        # that follow it.
        rows = json.loads(artifact.content)
        if endpoint == "eod":
            _price_frame(symbol, rows, artifact)
        else:
            _action_frame(symbol, endpoint, rows, artifact)


def _resumable_official_quarantine(
    cache_root: Path,
    quarantine_id: str,
    budget: EodhdCallBudget,
) -> tuple[tuple[SourceArtifact, ...], int]:
    if _text(quarantine_id).lower() != RESUMABLE_OFFICIAL_QUARANTINE_ID:
        raise ValueError("Resume quarantine id is not the exact reviewed failure.")
    _, envelope = _read_quarantine_envelope(cache_root, quarantine_id)
    if envelope.get("status") != "incomplete":
        raise ValueError("Resume requires an incomplete stage-1 quarantine.")
    if envelope.get("error") != RESUMABLE_OFFICIAL_ERROR:
        raise ValueError(
            "Resume requires the exact reviewed official-term pre-EODHD failure."
        )
    receipt = _validate_receipt(envelope.get("budget_receipt", {}), complete=False)
    expected_budget = {
        "period": budget.period,
        "daily_limit": int(budget.limit),
        "reserve": int(budget.reserve),
        "safety_ceiling": int(budget.ceiling),
    }
    if any(
        _text(receipt.get(key)) != _text(value)
        for key, value in expected_budget.items()
    ):
        raise ValueError("Resume quarantine does not match the current EODHD budget.")
    if not (
        int(receipt["used_before"]) == int(receipt["used_after"])
        and int(receipt["delta"]) == 0
        and int(receipt["own_claim_count"]) == 0
        and receipt["claim_positions"] == []
    ):
        raise ValueError("Resume quarantine must prove zero prior EODHD claims.")
    current_used = _budget_used(budget)
    if current_used < int(receipt["used_after"]):
        raise ValueError("Current EODHD budget regressed behind resume evidence.")
    _assert_budget_capacity(budget, current_used)

    artifacts = _artifacts_from_rows(
        envelope.get("artifacts"),
        OFFICIAL_URLS,
        expected_sources=OFFICIAL_SOURCES,
    )
    observed_hashes = {item.source_url: item.source_hash for item in artifacts}
    if observed_hashes != dict(RESUMABLE_OFFICIAL_SHA256):
        raise ValueError("Resume official raw-body hashes are not the reviewed pins.")
    _validate_official_evidence(
        {item.source_url: item for item in artifacts}
    )
    return artifacts, current_used


def collect_stage1(
    cache_root: Path,
    *,
    session_factory: Callable[[], Any] | None = None,
    budget_factory: Callable[[], EodhdCallBudget] = EodhdCallBudget,
) -> dict[str, Any]:
    load_env()
    budget = budget_factory()
    before = _budget_used(budget)
    _assert_budget_capacity(budget, before)
    if session_factory is None:
        import requests

        session_factory = requests.Session
    client = ExactStage1Client(
        session=session_factory(),
        token=os.getenv("EODHD_API_TOKEN", ""),
        user_agent=os.getenv("SEC_USER_AGENT", ""),
        budget=budget,
    )
    artifacts: list[SourceArtifact] = []
    retrieved_at = utc_now_iso()
    try:
        for url in OFFICIAL_URLS:
            artifacts.append(client.get_official(url, retrieved_at=retrieved_at))
        # Reject a changed/blocked official page before spending any of the six
        # provider calls.  Reviewer hash pins remain a later, separate gate;
        # this is only a minimum semantic preflight on the raw bodies.
        _validate_official_evidence(
            {item.source_url: item for item in artifacts}
        )
        _collect_eodhd_artifacts(
            client,
            artifacts,
            retrieved_at=retrieved_at,
        )
    except BaseException as exc:
        after = _budget_used(budget)
        receipt = _budget_receipt(
            budget,
            used_before=before,
            used_after=after,
            claim_positions=client.eodhd_claim_positions,
        )
        _, path = _write_quarantine(
            cache_root,
            artifacts,
            receipt,
            status="incomplete",
            error=f"{type(exc).__name__}: {exc}",
        )
        if hasattr(exc, "add_note"):
            exc.add_note(f"Partial exact raw responses were preserved at {path}.")
        raise
    after = _budget_used(budget)
    receipt = _budget_receipt(
        budget,
        used_before=before,
        used_after=after,
        claim_positions=client.eodhd_claim_positions,
    )
    try:
        _validate_receipt(receipt, complete=True)
    except BaseException as exc:
        _, path = _write_quarantine(
            cache_root,
            artifacts,
            receipt,
            status="incomplete",
            error=f"{type(exc).__name__}: {exc}",
        )
        if hasattr(exc, "add_note"):
            exc.add_note(
                f"All exact raw responses were preserved as incomplete at {path}."
            )
        raise
    quarantine_id, path = _write_quarantine(
        cache_root,
        artifacts,
        receipt,
        status="complete_unreviewed",
    )
    return {
        "status": "stage1_fetched_needs_reviewer_hash_pins",
        "network_accessed": True,
        "official_http_attempts": len(client.official_attempts),
        "eodhd_http_attempts": len(client.eodhd_attempts),
        "retry_count": 0,
        "quarantine_id": quarantine_id,
        "quarantine_path": str(path),
        "budget_receipt": receipt,
        "artifact_sha256": {
            item.source_url: item.source_hash for item in artifacts
        },
    }


def resume_stage1(
    cache_root: Path,
    quarantine_id: str,
    *,
    session_factory: Callable[[], Any] | None = None,
    budget_factory: Callable[[], EodhdCallBudget] = EodhdCallBudget,
) -> dict[str, Any]:
    """Reuse one exact reviewed official-only failure and fetch provider raws."""

    load_env()
    budget = budget_factory()
    official_artifacts, before = _resumable_official_quarantine(
        cache_root, quarantine_id, budget
    )
    if session_factory is None:
        import requests

        session_factory = requests.Session
    client = ExactStage1Client(
        session=session_factory(),
        token=os.getenv("EODHD_API_TOKEN", ""),
        user_agent=os.getenv("SEC_USER_AGENT", ""),
        budget=budget,
    )
    artifacts = list(official_artifacts)
    retrieved_at = utc_now_iso()
    try:
        _collect_eodhd_artifacts(
            client,
            artifacts,
            retrieved_at=retrieved_at,
        )
    except BaseException as exc:
        after = _budget_used(budget)
        receipt = _budget_receipt(
            budget,
            used_before=before,
            used_after=after,
            claim_positions=client.eodhd_claim_positions,
        )
        _, path = _write_quarantine(
            cache_root,
            artifacts,
            receipt,
            status="incomplete",
            error=f"{type(exc).__name__}: {exc}",
        )
        if hasattr(exc, "add_note"):
            exc.add_note(f"Resumed partial raw responses were preserved at {path}.")
        raise

    after = _budget_used(budget)
    receipt = _budget_receipt(
        budget,
        used_before=before,
        used_after=after,
        claim_positions=client.eodhd_claim_positions,
    )
    try:
        _validate_receipt(receipt, complete=True)
    except BaseException as exc:
        _, path = _write_quarantine(
            cache_root,
            artifacts,
            receipt,
            status="incomplete",
            error=f"{type(exc).__name__}: {exc}",
        )
        if hasattr(exc, "add_note"):
            exc.add_note(
                f"All resumed raw responses were preserved as incomplete at {path}."
            )
        raise
    resumed_id, path = _write_quarantine(
        cache_root,
        artifacts,
        receipt,
        status="complete_unreviewed",
    )
    return {
        "status": "stage1_resumed_fetched_needs_reviewer_hash_pins",
        "network_accessed": True,
        "resumed_from_quarantine_id": _text(quarantine_id).lower(),
        "official_raws_reused": EXPECTED_OFFICIAL_CALLS,
        "official_http_attempts_this_run": 0,
        "eodhd_http_attempts": len(client.eodhd_attempts),
        "retry_count": 0,
        "quarantine_id": resumed_id,
        "quarantine_path": str(path),
        "budget_receipt": receipt,
        "artifact_sha256": {
            item.source_url: item.source_hash for item in artifacts
        },
    }


def promote_quarantine(cache_root: Path, quarantine_id: str) -> dict[str, Any]:
    quarantine = read_quarantine(cache_root, quarantine_id)
    bundle = bundle_from_artifacts(
        quarantine.artifacts,
        official_http_attempts=EXPECTED_OFFICIAL_CALLS,
        eodhd_http_attempts=EXPECTED_EODHD_CALLS,
        budget_receipt=quarantine.budget_receipt,
        require_reviewer_pins=True,
    )
    path = _write_bundle_cache(cache_root, bundle)
    return {
        "status": "reviewed_bundle_promoted",
        "network_accessed": False,
        "quarantine_id": quarantine.quarantine_id,
        "reviewed_bundle_path": str(path),
        "artifact_sha256": {
            item.source_url: item.source_hash for item in bundle.artifacts
        },
    }


def _official_action_rows(bundle: ReviewedBundle) -> pd.DataFrame:
    artifacts = {item.source_url: item for item in bundle.artifacts}
    eca = artifacts[ECA_PRIMARY_URL]
    qvc = artifacts[QVC_OFFICIAL_URL]
    rows = [
        {
            "event_id": ECA_EVENT_ID,
            "security_id": ECA_ID,
            "action_type": "stock_merger",
            "effective_date": OVV_FIRST,
            "ex_date": OVV_FIRST,
            "announcement_date": ECA_LAST,
            "record_date": ECA_LAST,
            "payment_date": OVV_FIRST,
            "cash_amount": None,
            "ratio": ECA_RATIO,
            "currency": "USD",
            "new_security_id": OVV_ID,
            "new_symbol": OVV_SYMBOL,
            "official": True,
            "source_url": ECA_PRIMARY_URL,
            "source_kind": "official_issuer_plus_sec_crosscheck",
            "source": eca.source,
            "retrieved_at": eca.retrieved_at,
            "source_hash": eca.source_hash,
            "metadata": json.dumps(
                {
                    "legal_completion": ECA_LAST,
                    "first_successor_session": OVV_FIRST,
                    "supplemental_sec_url": ECA_SEC_URL,
                    "supplemental_sec_sha256": artifacts[ECA_SEC_URL].source_hash,
                },
                sort_keys=True,
            ),
        },
        {
            "event_id": QVC_EVENT_ID,
            "security_id": QVCGA_ID,
            "action_type": "ticker_change",
            "effective_date": QVCAQ_FIRST,
            "ex_date": QVCAQ_FIRST,
            # The issuer page proves the market-transition boundary, not a
            # separate announcement date.  Do not repurpose the last Nasdaq
            # session as announcement provenance.
            "announcement_date": "",
            "record_date": "",
            "payment_date": "",
            "cash_amount": None,
            "ratio": None,
            "currency": "USD",
            "new_security_id": QVCGA_ID,
            "new_symbol": QVCAQ_SYMBOL,
            "official": True,
            "source_url": QVC_OFFICIAL_URL,
            "source_kind": "official_issuer_market_transition",
            "source": qvc.source,
            "retrieved_at": qvc.retrieved_at,
            "source_hash": qvc.source_hash,
            "metadata": json.dumps(
                {
                    "identity_policy": "same_legal_security_same_security_id",
                    "old_market": "NASDAQ",
                    "new_market": "OTCID",
                },
                sort_keys=True,
            ),
        },
    ]
    columns = tuple(
        dict.fromkeys((*dataset_spec("corporate_actions").required_columns, "metadata"))
    )
    return pd.DataFrame(rows, columns=columns)


def _archive_id(item: SourceArtifact) -> str:
    return sha256_bytes(f"{item.source}|{item.source_url}|{item.source_hash}".encode())


def _archive_row(item: SourceArtifact, completed_session: str) -> dict[str, Any]:
    content_type = item.content_type.lower()
    suffix = "json" if "json" in content_type else "html" if "html" in content_type else "bin"
    return {
        "archive_id": _archive_id(item),
        "dataset": item.source,
        "object_path": f"archives/{completed_session}/{item.source_hash}.{suffix}.gz",
        "content_type": item.content_type,
        "effective_date": completed_session,
        "source": item.source,
        "retrieved_at": item.retrieved_at,
        "source_hash": item.source_hash,
        "source_url": item.source_url,
    }


def _one(frame: pd.DataFrame, column: str, value: str, label: str) -> pd.Series:
    rows = frame.loc[frame[column].astype(str).eq(value)]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one {label}; observed={len(rows)}")
    return rows.iloc[0]


def _assert_qvc_2025_split(actions: pd.DataFrame) -> pd.Series:
    rows = actions.loc[
        actions["security_id"].astype(str).eq(QVCGA_ID)
        & actions["action_type"].astype(str).eq("split")
        & actions["effective_date"].map(_date).eq(QVC_2025_SPLIT_DATE)
    ]
    if len(rows) != 1:
        raise ValueError(
            "QVC 2025 reverse split must remain exactly one canonical row."
        )
    row = rows.iloc[0]
    try:
        ratio = float(row.get("ratio"))
    except (TypeError, ValueError) as exc:
        raise ValueError("QVC 2025 reverse split ratio is not numeric.") from exc
    expected = {
        "event_id": QVC_2025_SPLIT_EVENT_ID,
        "security_id": QVCGA_ID,
        "action_type": "split",
        "effective_date": QVC_2025_SPLIT_DATE,
        "ex_date": QVC_2025_SPLIT_DATE,
        "currency": "USD",
        "source_url": QVC_2025_SPLIT_URL,
        "source_kind": "provider",
        "source": "eodhd_splits",
        "retrieved_at": QVC_2025_SPLIT_RETRIEVED_AT,
        "source_hash": QVC_2025_SPLIT_SHA256,
    }
    changed = {
        field: _text(row.get(field))
        for field, value in expected.items()
        if _text(row.get(field)) != value
    }
    if changed or abs(ratio - QVC_2025_SPLIT_RATIO) > 1e-12 or bool(
        row.get("official")
    ):
        raise ValueError(
            "QVC 2025 reverse split exact ratio/provenance changed: "
            f"fields={sorted(changed)}, ratio={ratio}."
        )
    return row


def _assert_series_preserved(before: pd.Series, after: pd.Series, label: str) -> None:
    columns = tuple(dict.fromkeys((*before.index.tolist(), *after.index.tolist())))
    left = pd.DataFrame([{column: before.get(column) for column in columns}])
    right = pd.DataFrame([{column: after.get(column) for column in columns}])
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    except AssertionError as exc:
        raise ValueError(f"{label} changed during the narrow repair.") from exc


def _factor_economics_and_lineage_are_exact(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    factors: pd.DataFrame,
) -> bool:
    touched = {ECA_ID, OVV_ID, QVCGA_ID}
    touched_prices = prices.loc[prices["security_id"].astype(str).isin(touched)]
    touched_actions = actions.loc[actions["security_id"].astype(str).isin(touched)]
    observed = factors.loc[factors["security_id"].astype(str).isin(touched)].copy()
    expected = build_adjustment_factors(
        touched_prices,
        touched_actions,
        source_version="eca-qvcaq-invariant-rebuild",
    )
    columns = ("security_id", "session", "split_factor", "total_return_factor")
    left = observed[list(columns)].copy()
    right = expected[list(columns)].copy()
    for frame in (left, right):
        frame["security_id"] = frame["security_id"].astype(str)
        frame["session"] = frame["session"].map(_date)
        frame["split_factor"] = pd.to_numeric(frame["split_factor"], errors="coerce")
        frame["total_return_factor"] = pd.to_numeric(
            frame["total_return_factor"], errors="coerce"
        )
        frame.sort_values(["security_id", "session"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError:
        return False
    required = {
        "source_version",
        "source_hash",
        "source",
        "retrieved_at",
        "calculated_at",
    }
    if not required.issubset(factors.columns):
        return False
    versions = set(factors["source_version"].map(_text))
    if len(versions) != 1 or "" in versions:
        return False
    lineage = next(iter(versions))
    return bool(
        not factors.empty
        and factors["source"].astype(str).eq("derived").all()
        and factors["source_hash"].map(_text).eq(lineage).all()
        and factors["retrieved_at"].map(_text).ne("").all()
        and factors["calculated_at"].map(_text).ne("").all()
    )


def _history_row(
    security_id: str,
    symbol: str,
    exchange: str,
    effective_from: str,
    effective_to: str,
    artifact: SourceArtifact,
) -> dict[str, Any]:
    return {
        "security_id": security_id,
        "symbol": symbol,
        "exchange": exchange,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "source": artifact.source,
        "retrieved_at": artifact.retrieved_at,
        "source_hash": artifact.source_hash,
        "source_url": artifact.source_url,
    }


def prepare_frames(
    existing: Mapping[str, pd.DataFrame],
    bundle: ReviewedBundle,
    *,
    completed_session: str,
    factor_source_version: str = "eca-qvcaq-offline-plan-lineage",
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if completed_session != FETCH_END:
        raise ValueError("ECA/QVCAQ repair is pinned to completed session 2026-07-15.")
    master = existing["security_master"].copy()
    history = existing["symbol_history"].copy()
    prices = existing["daily_price_raw"].copy()
    actions = existing["corporate_actions"].copy()
    resolutions = existing["lifecycle_resolutions"].copy()
    archive = existing["source_archive"].copy()
    artifacts = {item.source_url: item for item in bundle.artifacts}
    eca_evidence = artifacts[ECA_PRIMARY_URL]
    qvc_evidence = artifacts[QVC_OFFICIAL_URL]

    eca_master = _one(master, "security_id", ECA_ID, "ECA master row")
    qvc_master = _one(master, "security_id", QVCGA_ID, "QVCGA master row")
    if _text(eca_master.get("primary_symbol")).upper() != ECA_SYMBOL:
        raise ValueError("ECA master identity changed before repair.")
    if _text(qvc_master.get("primary_symbol")).upper() != QVCGA_SYMBOL:
        raise ValueError("QVCGA master identity changed before repair.")
    if master["security_id"].astype(str).eq(OVV_ID).any():
        raise ValueError("OVV security already exists in a partial, non-idempotent state.")

    eca_prices = prices.loc[prices["security_id"].astype(str).eq(ECA_ID)].copy()
    eca_sessions = eca_prices["session"].map(_date)
    if ECA_LAST not in set(eca_sessions) or not eca_sessions.gt(ECA_LAST).any():
        raise ValueError("ECA exact contaminated tail precondition changed.")
    qvc_prices = prices.loc[prices["security_id"].astype(str).eq(QVCGA_ID)].copy()
    qvc_sessions = qvc_prices["session"].map(_date)
    if QVCGA_LAST not in set(qvc_sessions) or qvc_sessions.gt(QVCGA_LAST).any():
        raise ValueError("QVCGA exact terminal boundary precondition changed.")
    if prices["security_id"].astype(str).eq(OVV_ID).any():
        raise ValueError("OVV prices already exist in a partial state.")
    qvc_2025_split_before = _assert_qvc_2025_split(actions)

    eca_tail = prices["security_id"].astype(str).eq(ECA_ID) & prices["session"].map(
        _date
    ).gt(ECA_LAST)
    prices = prices.loc[~eca_tail].copy()
    prices = pd.concat([prices, bundle.prices], ignore_index=True, sort=False)
    if prices.duplicated(["security_id", "session"]).any():
        raise ValueError("Prepared transition prices contain duplicate keys.")

    eca_action_tail = actions["security_id"].astype(str).eq(ECA_ID) & actions[
        "effective_date"
    ].map(_date).gt(ECA_LAST)
    qvc_action_tail = actions["security_id"].astype(str).eq(QVCGA_ID) & actions[
        "effective_date"
    ].map(_date).ge(QVCAQ_FIRST)
    actions = actions.loc[
        ~(eca_action_tail | qvc_action_tail)
        & ~actions["event_id"].astype(str).isin({ECA_EVENT_ID, QVC_EVENT_ID})
    ].copy()
    actions = pd.concat(
        [actions, bundle.provider_actions, _official_action_rows(bundle)],
        ignore_index=True,
        sort=False,
    )
    if actions["event_id"].astype(str).duplicated().any():
        raise ValueError("Prepared transition actions contain duplicate event IDs.")
    qvc_2025_split_after = _assert_qvc_2025_split(actions)
    _assert_series_preserved(
        qvc_2025_split_before,
        qvc_2025_split_after,
        "QVC 2025 reverse split",
    )

    # Rewrite only the three exact identities.  QVCAQ intentionally reuses
    # QVCGA_ID; OVV intentionally does not reuse ECA_ID because the official
    # 1-for-5 reorganization changes the share unit and is modeled as a stock
    # successor action.
    master = master.loc[~master["security_id"].astype(str).isin({ECA_ID, QVCGA_ID})]
    eca_row = dict(eca_master)
    eca_row.update(
        {
            "name": ECA_ISSUER_NAME,
            "active_to": ECA_LAST,
            "source": eca_evidence.source,
            "source_url": eca_evidence.source_url,
            "retrieved_at": eca_evidence.retrieved_at,
            "source_hash": eca_evidence.source_hash,
        }
    )
    ovv_row = dict(eca_row)
    ovv_row.update(
        {
            "security_id": OVV_ID,
            "primary_symbol": OVV_SYMBOL,
            "provider_symbol": "OVV.US",
            "action_provider_symbol": "OVV.US",
            "name": OVV_ISSUER_NAME,
            "exchange": "NYSE",
            "active_from": OVV_FIRST,
            "active_to": "",
        }
    )
    qvc_row = dict(qvc_master)
    qvc_row.update(
        {
            "primary_symbol": QVCAQ_SYMBOL,
            "provider_symbol": "QVCAQ.US",
            "action_provider_symbol": "QVCAQ.US",
            "exchange": "OTCID",
            "active_to": "",
            "source": qvc_evidence.source,
            "source_url": qvc_evidence.source_url,
            "retrieved_at": qvc_evidence.retrieved_at,
            "source_hash": qvc_evidence.source_hash,
        }
    )
    master = pd.concat(
        [master, pd.DataFrame([eca_row, ovv_row, qvc_row], columns=master.columns)],
        ignore_index=True,
        sort=False,
    )

    old_eca_history = _one(history, "security_id", ECA_ID, "ECA symbol history")
    old_qvc_history = _one(history, "security_id", QVCGA_ID, "QVCGA symbol history")
    history = history.loc[
        ~history["security_id"].astype(str).isin({ECA_ID, OVV_ID, QVCGA_ID})
    ].copy()
    history_rows = [
        _history_row(
            ECA_ID,
            ECA_SYMBOL,
            "NYSE",
            _date(old_eca_history.get("effective_from")),
            ECA_LAST,
            eca_evidence,
        ),
        _history_row(
            OVV_ID, OVV_SYMBOL, "NYSE", OVV_FIRST, "", eca_evidence
        ),
        _history_row(
            QVCGA_ID,
            QVCGA_SYMBOL,
            "NASDAQ",
            _date(old_qvc_history.get("effective_from")),
            QVCGA_LAST,
            qvc_evidence,
        ),
        _history_row(
            QVCGA_ID, QVCAQ_SYMBOL, "OTCID", QVCAQ_FIRST, "", qvc_evidence
        ),
    ]
    history = pd.concat(
        [history, pd.DataFrame(history_rows, columns=history.columns)],
        ignore_index=True,
        sort=False,
    )

    resolutions = resolutions.loc[
        ~resolutions["security_id"].astype(str).isin({ECA_ID, QVCGA_ID})
    ].copy()
    # Neither identity is a direct canonical index-terminal candidate in the
    # current graph.  The official actions close their predecessors' missing
    # successor chains; adding a self-authorizing lifecycle resolution here
    # would create an unexpected resolution at the publication gate.

    # Rebuild every row, not only the three economically touched identities.
    # A release has one adjustment-factor lineage binding the exact planned
    # daily-price and corporate-action versions; retaining old lineage on
    # out-of-scope rows would fail the publication contract even if their
    # numeric factors did not change.
    factors = build_adjustment_factors(
        prices,
        actions,
        source_version=factor_source_version,
    )

    archive_ids = {_archive_id(item) for item in bundle.artifacts}
    archive = archive.loc[~archive["archive_id"].astype(str).isin(archive_ids)]
    archive = pd.concat(
        [
            archive,
            pd.DataFrame(
                [_archive_row(item, completed_session) for item in bundle.artifacts],
                columns=archive.columns,
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
        raise RuntimeError("Prepared ECA/QVCAQ state failed its exact invariant.")

    return frames, {
        "status": "validated_offline_plan",
        "network_accessed_this_run": False,
        "r2_accessed": False,
        "eca_tail_price_rows_removed": int(eca_tail.sum()),
        "eca_tail_action_rows_removed": int(eca_action_tail.sum()),
        "eca_canonical_ratio": ECA_RATIO,
        "eca_successor_security_id": OVV_ID,
        "qvc_identity_decision": "same_legal_security_same_security_id",
        "qvc_security_id": QVCGA_ID,
        "qvc_2025_split_rows_preserved": 1,
        "qvc_2025_split_event_ids": [QVC_2025_SPLIT_EVENT_ID],
        "qvc_2025_split_ratio": QVC_2025_SPLIT_RATIO,
        "qvc_2025_split_source_hash": QVC_2025_SPLIT_SHA256,
        "official_archive_rows_added": EXPECTED_OFFICIAL_CALLS,
        "eodhd_archive_rows_added": EXPECTED_EODHD_CALLS,
        "original_official_http_attempts": bundle.official_http_attempts,
        "original_eodhd_http_attempts": bundle.eodhd_http_attempts,
        "adjustment_source_version": factor_source_version,
        "adjustment_factor_rows_rebound": len(factors),
        **dict(bundle.evidence_claims),
    }


def identity_is_repaired(frames: Mapping[str, pd.DataFrame]) -> bool:
    try:
        master = frames["security_master"]
        history = frames["symbol_history"]
        prices = frames["daily_price_raw"]
        actions = frames["corporate_actions"]
        resolutions = frames["lifecycle_resolutions"]
        factors = frames["adjustment_factors"]
        eca = _one(master, "security_id", ECA_ID, "ECA master")
        ovv = _one(master, "security_id", OVV_ID, "OVV master")
        qvc = _one(master, "security_id", QVCGA_ID, "QVC master")
        if not (
            _text(eca.get("primary_symbol")).upper() == ECA_SYMBOL
            and _text(eca.get("name")) == ECA_ISSUER_NAME
            and _date(eca.get("active_to")) == ECA_LAST
            and _text(ovv.get("primary_symbol")).upper() == OVV_SYMBOL
            and _text(ovv.get("name")) == OVV_ISSUER_NAME
            and _date(ovv.get("active_from")) == OVV_FIRST
            and not _date(ovv.get("active_to"))
            and _text(qvc.get("primary_symbol")).upper() == QVCAQ_SYMBOL
            and _text(qvc.get("provider_symbol")).upper() == "QVCAQ.US"
            and not _date(qvc.get("active_to"))
        ):
            return False
        observed_history = {
            (
                _text(row.security_id),
                _text(row.symbol).upper(),
                _date(row.effective_from),
                _date(row.effective_to),
            )
            for row in history.loc[
                history["security_id"].astype(str).isin({ECA_ID, OVV_ID, QVCGA_ID})
            ].itertuples(index=False)
        }
        expected_history = {
            (ECA_ID, ECA_SYMBOL, next(x[2] for x in observed_history if x[0] == ECA_ID), ECA_LAST),
            (OVV_ID, OVV_SYMBOL, OVV_FIRST, ""),
            (QVCGA_ID, QVCGA_SYMBOL, next(x[2] for x in observed_history if x[0] == QVCGA_ID and x[1] == QVCGA_SYMBOL), QVCGA_LAST),
            (QVCGA_ID, QVCAQ_SYMBOL, QVCAQ_FIRST, ""),
        }
        if observed_history != expected_history:
            return False
        eca_sessions = prices.loc[prices["security_id"].astype(str).eq(ECA_ID), "session"].map(_date)
        ovv_sessions = prices.loc[prices["security_id"].astype(str).eq(OVV_ID), "session"].map(_date)
        qvc_sessions = prices.loc[prices["security_id"].astype(str).eq(QVCGA_ID), "session"].map(_date)
        if (
            eca_sessions.max() != ECA_LAST
            or ovv_sessions.min() != OVV_FIRST
            or ovv_sessions.max() != FETCH_END
            or QVCGA_LAST not in set(qvc_sessions)
            or QVCAQ_FIRST not in set(qvc_sessions)
            or qvc_sessions.max() != FETCH_END
        ):
            return False
        eca_action = _one(actions, "event_id", ECA_EVENT_ID, "ECA action")
        qvc_action = _one(actions, "event_id", QVC_EVENT_ID, "QVC action")
        _assert_qvc_2025_split(actions)
        if not (
            _text(eca_action.get("action_type")) == "stock_merger"
            and abs(float(eca_action.get("ratio")) - ECA_RATIO) <= 1e-12
            and _text(eca_action.get("new_security_id")) == OVV_ID
            and bool(eca_action.get("official"))
            and _text(qvc_action.get("action_type")) == "ticker_change"
            and _text(qvc_action.get("new_security_id")) == QVCGA_ID
            and bool(qvc_action.get("official"))
        ):
            return False
        if resolutions["security_id"].astype(str).isin({ECA_ID, QVCGA_ID}).any():
            return False
        touched = {ECA_ID, OVV_ID, QVCGA_ID}
        price_keys = {
            (_text(row.security_id), _date(row.session))
            for row in prices.loc[prices["security_id"].astype(str).isin(touched)].itertuples(index=False)
        }
        factor_keys = {
            (_text(row.security_id), _date(row.session))
            for row in factors.loc[factors["security_id"].astype(str).isin(touched)].itertuples(index=False)
        }
        return bool(
            price_keys == factor_keys
            and _factor_economics_and_lineage_are_exact(prices, actions, factors)
        )
    except (KeyError, StopIteration, TypeError, ValueError):
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


def _snapshot_issue_fingerprint(report: Any) -> tuple[str, ...]:
    """Bind a repair to the exact pre-existing repository issue inventory."""

    return tuple(
        sorted(
            _canonical_json_bytes(
                {
                    "code": issue.code,
                    "message": issue.message,
                    "severity": issue.severity,
                    "row_count": issue.row_count,
                    "fingerprints": list(issue.fingerprints),
                }
            ).decode("utf-8")
            for issue in report.issues
        )
    )


def _assert_snapshot_issues_preserved(
    repository: LocalDatasetRepository,
    versions: Mapping[str, str],
    before: Mapping[str, pd.DataFrame],
    after: Mapping[str, pd.DataFrame],
) -> tuple[str, ...]:
    before_report = validate_repository_snapshot(
        _CandidateRepository(repository, versions, before)
    )
    after_report = validate_repository_snapshot(
        _CandidateRepository(repository, versions, after)
    )
    before_fingerprint = _snapshot_issue_fingerprint(before_report)
    if before_fingerprint != _snapshot_issue_fingerprint(after_report):
        raise RuntimeError(
            "ECA/QVCAQ repair changed unrelated repository snapshot issues."
        )
    return before_fingerprint


def _capture_pointer_etags(
    repository: LocalDatasetRepository, release: DataRelease
) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for dataset in WRITE_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != release.dataset_versions.get(dataset):
            raise RuntimeError(f"Release/pointer mismatch before ECA/QVCAQ repair: {dataset}")
        output[dataset] = etag
    return output


def _planned_versions(release: DataRelease) -> dict[str, str]:
    transaction_id = uuid.uuid4().hex
    session = release.completed_session.replace("-", "")
    return {
        dataset: f"eca-qvcaq-transition-{session}-{transaction_id}-{dataset}"
        for dataset in WRITE_DATASETS
    }


def _validate_planned_versions(
    release: DataRelease, planned_versions: Mapping[str, str]
) -> tuple[dict[str, str], str]:
    planned = {str(key): _text(value) for key, value in planned_versions.items()}
    if set(planned) != set(WRITE_DATASETS):
        raise RuntimeError("Prepared ECA/QVCAQ version inventory changed.")
    session = release.completed_session.replace("-", "")
    prefix = f"eca-qvcaq-transition-{session}-"
    daily_version = planned["daily_price_raw"]
    suffix = "-daily_price_raw"
    if not daily_version.startswith(prefix) or not daily_version.endswith(suffix):
        raise RuntimeError("Prepared ECA/QVCAQ version format changed.")
    transaction_id = daily_version[len(prefix) : -len(suffix)]
    if not re.fullmatch(r"[0-9a-f]{32}", transaction_id):
        raise RuntimeError("Prepared ECA/QVCAQ transaction id changed.")
    expected = {
        dataset: f"{prefix}{transaction_id}-{dataset}" for dataset in WRITE_DATASETS
    }
    if planned != expected:
        raise RuntimeError("Prepared ECA/QVCAQ versions do not share one transaction.")
    return planned, transaction_id


def _assert_release_factor_lineage(
    repository: LocalDatasetRepository,
    release: DataRelease,
    factors: pd.DataFrame,
) -> str:
    price_version = release.dataset_versions["daily_price_raw"]
    action_version = release.dataset_versions["corporate_actions"]
    lineage = _adjustment_source_version(price_version, action_version)
    manifest = repository.manifest_for_version(
        "adjustment_factors", release.dataset_versions["adjustment_factors"]
    )
    expected_metadata = {
        "source_version": lineage,
        "source_daily_price_version": price_version,
        "source_corporate_actions_version": action_version,
    }
    if any(
        _text(manifest.metadata.get(key)) != value
        for key, value in expected_metadata.items()
    ):
        raise RuntimeError(
            "Already-repaired adjustment-factor manifest is not release-exact."
        )
    if (
        set(factors["source_version"].map(_text)) != {lineage}
        or set(factors["source_hash"].map(_text)) != {lineage}
        or set(factors["source"].astype(str)) != {"derived"}
    ):
        raise RuntimeError(
            "Already-repaired adjustment-factor rows are not release-exact."
        )
    return lineage


def prepare_run(repository: LocalDatasetRepository, bundle: ReviewedBundle) -> PreparedRepair:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    missing = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks datasets: " + ", ".join(missing))
    existing = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in REQUIRED_DATASETS
    }
    pointer_etags = _capture_pointer_etags(repository, release)
    if identity_is_repaired(existing):
        lineage = _assert_release_factor_lineage(
            repository, release, existing["adjustment_factors"]
        )
        snapshot_issues = _snapshot_issue_fingerprint(
            validate_repository_snapshot(repository)
        )
        return PreparedRepair(
            release=release,
            release_etag=release_etag,
            pointer_etags=pointer_etags,
            planned_versions={},
            frames=existing,
            artifacts=(),
            summary={
                "status": "already_repaired",
                "release_version": release.version,
                "network_accessed_this_run": False,
                "r2_accessed": False,
                "adjustment_source_version": lineage,
                "repository_snapshot_issues_preserved": True,
                "repository_snapshot_issue_fingerprints": list(snapshot_issues),
            },
        )
    planned = _planned_versions(release)
    factor_source_version = _adjustment_source_version(
        planned["daily_price_raw"], planned["corporate_actions"]
    )
    frames, summary = prepare_frames(
        existing,
        bundle,
        completed_session=release.completed_session,
        factor_source_version=factor_source_version,
    )
    candidate = _CandidateRepository(repository, release.dataset_versions, frames)
    snapshot_issues = _assert_snapshot_issues_preserved(
        repository, release.dataset_versions, existing, frames
    )
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions=planned,
        frames=frames,
        artifacts=bundle.artifacts,
        summary={
            **summary,
            "release_version": release.version,
            "repository_snapshot_issues_preserved": True,
            "repository_snapshot_issue_fingerprints": list(snapshot_issues),
        },
    )


def _persist_artifacts(
    repository: LocalDatasetRepository,
    artifacts: Iterable[SourceArtifact],
    completed_session: str,
) -> None:
    for item in artifacts:
        content_type = item.content_type.lower()
        suffix = "json" if "json" in content_type else "html" if "html" in content_type else "bin"
        path = repository.root / f"archives/{completed_session}/{item.source_hash}.{suffix}.gz"
        encoded = gzip.compress(item.content, mtime=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            if gzip.decompress(path.read_bytes()) != item.content:
                raise RuntimeError(f"Immutable archive content changed: {path}")
        else:
            write_atomic(path, encoded)


@contextmanager
def _exclusive_lock(repository: LocalDatasetRepository):
    path = repository.root / ".locks/market-store-write.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery = repository.root / RECOVERY_DIR
        if recovery.exists() and tuple(recovery.glob("*.json")):
            raise RuntimeError("An ECA/QVCAQ recovery marker blocks writes.")
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
            observed = DataRelease.from_bytes(current.data)
            if any(
                observed.dataset_versions.get(dataset) != planned[dataset]
                for dataset in WRITE_DATASETS
            ):
                raise RuntimeError(
                    f"unexpected release during ECA/QVCAQ rollback: {observed.version}"
                )
            repository.objects.put(
                "releases/current.json", old_release, if_match=current.etag
            )
    except Exception as exc:
        errors.append(f"release: {type(exc).__name__}: {exc}")
    for dataset in reversed(WRITE_DATASETS):
        try:
            key = repository.current_key(dataset)
            current = repository.objects.get(key)
            if current.data != old_pointers[dataset]:
                pointer = CurrentPointer.from_bytes(current.data)
                if pointer.version != planned[dataset]:
                    raise RuntimeError(f"unexpected pointer {pointer.version}")
                repository.objects.put(key, old_pointers[dataset], if_match=current.etag)
        except Exception as exc:
            errors.append(f"{dataset}: {type(exc).__name__}: {exc}")
    return tuple(errors)


FailureInjector = Callable[[str], None]


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    failure_injector: FailureInjector | None = None,
) -> dict[str, Any]:
    if prepared.summary["status"] == "already_repaired":
        return dict(prepared.summary)
    if prepared.summary.get("network_accessed_this_run"):
        raise RuntimeError("Fetch and apply must be separate invocations.")
    inject = failure_injector or (lambda _stage: None)
    with _exclusive_lock(repository):
        release, release_etag = repository.current_release()
        if (
            release is None
            or release.version != prepared.release.version
            or release_etag != prepared.release_etag
        ):
            raise RuntimeError("Current release changed after offline preflight.")
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != release.dataset_versions[dataset]
                or value.etag != prepared.pointer_etags[dataset]
            ):
                raise RuntimeError(f"Dataset pointer changed before apply: {dataset}")
            old_pointers[dataset] = value.data

        # The factor rows were built against these exact versions during the
        # offline prepare phase.  Generating a second transaction here would
        # publish factors whose lineage names inputs that were never committed.
        planned, transaction_id = _validate_planned_versions(
            release, prepared.planned_versions
        )
        factor_lineage = _adjustment_source_version(
            planned["daily_price_raw"], planned["corporate_actions"]
        )
        prepared_factors = prepared.frames["adjustment_factors"]
        if (
            set(prepared_factors["source_version"].map(_text)) != {factor_lineage}
            or set(prepared_factors["source_hash"].map(_text)) != {factor_lineage}
            or set(prepared_factors["source"].astype(str)) != {"derived"}
        ):
            raise RuntimeError(
                "Prepared adjustment-factor rows do not match planned inputs."
            )
        journal_path = repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        journal = {
            "schema": "us_eca_qvcaq_transition_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": release.version,
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
            _persist_artifacts(repository, prepared.artifacts, release.completed_session)
            inject("after_archive")
            versions = dict(release.dataset_versions)
            for dataset in WRITE_DATASETS:
                metadata = dict(
                    repository.manifest_for_version(
                        dataset, release.dataset_versions[dataset]
                    ).metadata
                )
                metadata.update(
                    {
                        "operation": OPERATION,
                        "policy": POLICY,
                        "network_accessed_this_run": False,
                        "r2_accessed": False,
                        "eca_ratio": ECA_RATIO,
                        "qvc_identity_policy": "same_legal_security_same_security_id",
                        "needs_lifecycle_refinalization": True,
                    }
                )
                if dataset == "adjustment_factors":
                    metadata.update(
                        {
                            "source_version": factor_lineage,
                            "source_daily_price_version": planned["daily_price_raw"],
                            "source_corporate_actions_version": planned[
                                "corporate_actions"
                            ],
                            "provenance_rows_rebound": len(prepared_factors),
                        }
                    )
                result = repository.write_frame(
                    dataset,
                    prepared.frames[dataset],
                    completed_session=release.completed_session,
                    incomplete_action_policy="warn",
                    metadata=metadata,
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict:
                    raise RuntimeError(f"ECA/QVCAQ write conflicted: {dataset}")
                versions[dataset] = result.manifest.version
                inject(f"after_{dataset}")
            written = {
                dataset: repository.read_frame(dataset, planned[dataset])
                for dataset in WRITE_DATASETS
            }
            if not identity_is_repaired(written):
                raise RuntimeError("Written ECA/QVCAQ state failed its invariant.")
            _assert_snapshot_issues_preserved(
                repository,
                release.dataset_versions,
                {},
                written,
            )
            quality = (
                DataQuality.DEGRADED
                if release.quality == DataQuality.DEGRADED.value
                else DataQuality.VALID
            )
            committed = repository.commit_release(
                release.completed_session,
                versions,
                quality=quality,
                warnings=release.warnings,
                expected_etag=prepared.release_etag,
            )
            current, _ = repository.current_release()
            if current is None or current.to_bytes() != committed.to_bytes():
                raise RuntimeError("Committed ECA/QVCAQ release is not current.")
            for dataset in WRITE_DATASETS:
                pointer, _ = repository.current_pointer(dataset)
                if pointer is None or pointer.version != committed.dataset_versions[dataset]:
                    raise RuntimeError(
                        f"Committed ECA/QVCAQ pointer mismatch: {dataset}."
                    )
            _assert_release_factor_lineage(
                repository, committed, written["adjustment_factors"]
            )
            inject("after_release")
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
            }
        except BaseException as original:
            errors = _restore(
                repository,
                old_release=old_release.data,
                old_pointers=old_pointers,
                planned=planned,
            )
            journal.update(
                {
                    "status": "rollback_failed" if errors else "rolled_back",
                    "original_error": f"{type(original).__name__}: {original}",
                    "rollback_errors": list(errors),
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            if errors:
                recovery = repository.root / RECOVERY_DIR / f"{transaction_id}.json"
                _write_journal(recovery, journal)
                raise RuntimeError(
                    f"ECA/QVCAQ rollback failed; recovery marker created: {recovery}"
                ) from original
            raise


def requirements_plan() -> dict[str, Any]:
    return {
        "status": "requirements_plan",
        "network_accessed": False,
        "r2_accessed": False,
        "official_http_attempts": EXPECTED_OFFICIAL_CALLS,
        "eodhd_http_attempts": EXPECTED_EODHD_CALLS,
        "total_http_attempts": EXPECTED_TOTAL_CALLS,
        "retry_count": 0,
        "resume_quarantine_id": RESUMABLE_OFFICIAL_QUARANTINE_ID,
        "resume_official_http_attempts": 0,
        "resume_eodhd_http_attempts": EXPECTED_EODHD_CALLS,
        "official_requests": [
            {"url": url, "attempts": 1, "retries": 0} for url in OFFICIAL_URLS
        ],
        "eodhd_requests": [
            {
                "symbol": symbol,
                "endpoint": endpoint,
                "url": REQUEST_URLS[(symbol, endpoint)],
                "attempts": 1,
                "retries": 0,
            }
            for symbol, endpoint in REQUEST_ORDER
        ],
        "reviewer_hash_pins_pending": sorted(
            url
            for url, digest in REVIEWED_ARTIFACT_SHA256.items()
            if not _safe_hash(digest)
        ),
        "eca_policy": "ECA terminal 2020-01-24; OVV new security 2020-01-27 at 0.2",
        "qvc_policy": "QVCGA/QVCAQ same security_id; Nasdaq then OTCID",
        "qvc_2025_split_policy": "preserve existing action unchanged",
    }


def _current_state_summary(repository: LocalDatasetRepository) -> dict[str, Any]:
    release, _ = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    prices = repository.read_frame(
        "daily_price_raw", release.dataset_versions["daily_price_raw"]
    )
    actions = repository.read_frame(
        "corporate_actions", release.dataset_versions["corporate_actions"]
    )
    eca = prices.loc[prices["security_id"].astype(str).eq(ECA_ID)]
    qvc = prices.loc[prices["security_id"].astype(str).eq(QVCGA_ID)]
    split = _assert_qvc_2025_split(actions)
    return {
        "release_version": release.version,
        "completed_session": release.completed_session,
        "eca_current_last_session": max(eca["session"].map(_date), default=""),
        "eca_tail_rows_after_2020_01_24": int(eca["session"].map(_date).gt(ECA_LAST).sum()),
        "qvcga_current_last_session": max(qvc["session"].map(_date), default=""),
        "qvc_2025_split_rows_found": 1,
        "qvc_2025_split_ratios": [float(split["ratio"])],
        "qvc_2025_split_source_hash": _text(split["source_hash"]),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--requirements-plan", action="store_true")
    parser.add_argument("--fetch-stage1", action="store_true")
    parser.add_argument("--resume-quarantine", metavar="SHA256")
    parser.add_argument("--promote-quarantine", metavar="SHA256")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline-plan", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    cache_root = Path(args.cache_root)
    selected = sum(
        bool(item)
        for item in (
            args.requirements_plan,
            args.fetch_stage1,
            args.resume_quarantine,
            args.promote_quarantine,
            args.offline_plan,
            args.apply,
        )
    )
    if selected > 1:
        raise ValueError("Fetch, promote, plan, and apply must be separate invocations.")
    if args.requirements_plan:
        return requirements_plan()
    if args.fetch_stage1:
        return collect_stage1(cache_root)
    if args.resume_quarantine:
        return resume_stage1(cache_root, args.resume_quarantine)
    if args.promote_quarantine:
        return promote_quarantine(cache_root, args.promote_quarantine)

    repository = LocalDatasetRepository(cache_root)
    bundle = _read_bundle_cache(cache_root)
    if bundle is None:
        if args.apply:
            raise RuntimeError(
                "Reviewed ECA/QVCAQ bundle is absent; stage1 fetch, hash review, "
                "and offline promotion must finish before apply."
            )
        return {
            **requirements_plan(),
            **_current_state_summary(repository),
            "status": "offline_plan_blocked_pending_reviewed_bundle",
            "network_accessed": False,
        }
    prepared = prepare_run(repository, bundle)
    if not args.apply:
        return prepared.summary
    return apply_repair(repository, prepared)


def main() -> int:
    print(json.dumps(run(_parse_args()), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
