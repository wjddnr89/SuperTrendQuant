#!/usr/bin/env python3
"""Collect and plan the SIVB and AVP terminal-transition repairs safely.

The default path is a read-only offline planner.  The only network path is the
explicit SIVB evidence collector: one exact SEC filing request and one exact
EODHD SIVBQ request, with a persistent budget claim and no retry.  OCC memo
52179 is represented by a deterministic reviewed extraction because the
official PDF endpoint rejects direct archive clients.  The apply path is
explicit, offline, CAS-guarded, and transactional; it never calls SEC, EODHD,
or R2.

The planner reads the current local release and hash-pinned evidence, then:

* AVP: plan the stock-merger engine date on the first priceable NTCO market
  session while retaining the January 3 legal completion date in metadata.
* SIVB: preserve the November 7, 2024 zero-distribution legal cancellation,
  add the official March 10 halt and March 28 SIVB-to-SIVBQ OTC transition,
  append the exact-hash SIVBQ price path, and schedule the engine zero exit on
  the first XNYS session after the final observed OTC close.

The JSON printed by this script is a reviewable plan, not an applied repair.
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
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from supertrend_quant.env import load_env
from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.ingest import EodhdCallBudget
from supertrend_quant.market_store.lifecycle import canonical_lifecycle_event_id
from supertrend_quant.market_store.lifecycle_coverage import (
    lifecycle_candidate_id,
    lifecycle_candidate_set_sha256,
    lifecycle_resolution_set_sha256,
)
from supertrend_quant.market_store.manifest import write_atomic
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    utc_now_iso,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.terminal_readiness import (
    audit_terminal_transitions,
)
from supertrend_quant.market_store.validation import validate_dataset
from supertrend_quant.market_store.validation import validate_repository_snapshot


DEFAULT_CACHE_ROOT = Path("data/cache")
SCHEMA = "us_sivb_avp_terminal_transition_plan/v1"
POLICY = "legal_cancellation_and_market_transition_separation/v1"
REVIEWED_AT = "2026-07-18T15:30:00Z"
REPAIR_REVIEWER = "sivb_avp_terminal_transition_planner_v1"
REPAIR_SOURCE = "official_market_transition_repair"
OPERATION = "repair_us_sivb_avp_terminal_transitions"
TRANSACTION_DIR = "transactions/us-sivb-avp-terminal-transitions"
RECOVERY_DIR = "recovery/us-sivb-avp-terminal-transitions"

REQUIRED_DATASETS = (
    "corporate_actions",
    "lifecycle_resolutions",
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "adjustment_factors",
    "source_archive",
    "index_constituent_anchors",
    "index_membership_events",
)
WRITE_DATASETS = (
    "corporate_actions",
    "lifecycle_resolutions",
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "adjustment_factors",
    "source_archive",
)

SIVB_ID = "US:EODHD:181d58e2-2b82-5f5d-8c33-311c5d7b5129"
SIVB_SYMBOL = "SIVB"
SIVB_LAST_SESSION = "2023-03-09"
SIVB_EXPECTED_MARKET_SESSION = "2023-03-10"
SIVB_LEGAL_CANCELLATION = "2024-11-07"
SIVB_EVENT_ID = (
    "1f4a23cffdf2decb8c26be93d94318d6d5a2be7fc045c33ff9e5abd4e9c69c82"
)
SIVB_CANDIDATE_ID = (
    "919fffde38cef854109fdd3fa98df2af3b5d039cd76d06d733d7fc8bc2da6601"
)
SIVB_OFFICIAL_URL = (
    "https://www.sec.gov/Archives/edgar/data/719739/"
    "000119312524254186/d904756d8k.htm"
)
SIVB_OFFICIAL_HASH = (
    "14371aef1566bfdcda9ca3171b1ced46d095adf34d899a8bde6d8e038d68e231"
)
SIVB_OFFICIAL_BYTES = 54_478
SIVB_OFFICIAL_RETRIEVED_AT = "2026-07-18T10:31:24.972232Z"
SIVB_RAW_URL = "https://eodhd.com/api/eod/SIVB.US?from=2015-01-01&to=2026-07-15"
SIVB_RAW_HASH = (
    "b143fda394b979fbccfbacd8f729e8f8b9f1a6d878d3312498d0c9a300005228"
)
SIVB_RAW_BYTES = 241_882
SIVB_RAW_ROWS = 2_060
SIVB_RAW_RETRIEVED_AT = "2026-07-16T15:56:23.032810Z"
SIVB_TERMINAL_OHLCV = (176.55, 177.7499, 100.0, 106.04, 39_006_960.0)
SIVB_TRANSITION_EVIDENCE_SUBDIR = Path(
    "state/issuer_lifecycle/sivb_otc_transition"
)
SIVB_TRANSITION_REPORT = "sivb_otc_transition_evidence.json"
SIVB_TRANSITION_EVIDENCE_SCHEMA = "us_sivb_otc_transition_evidence/v1"
SIVB_SEC_MARKET_URL = (
    "https://www.sec.gov/Archives/edgar/data/719739/"
    "000119312523073665/d485308d8k.htm"
)
SIVB_OCC_MEMO_URL = "https://infomemo.theocc.com/infomemos?number=52179"
SIVBQ_PROVIDER_SYMBOL = "SIVBQ.US"
SIVBQ_FETCH_START = "2023-03-28"
SIVBQ_FETCH_END = "2024-11-08"
SIVBQ_SAFE_URL = (
    f"https://eodhd.com/api/eod/{SIVBQ_PROVIDER_SYMBOL}"
    f"?from={SIVBQ_FETCH_START}&to={SIVBQ_FETCH_END}"
)
SIVB_TRANSITION_MAX_HTTP_ATTEMPTS = 2
SIVB_TRANSITION_MAX_EODHD_ATTEMPTS = 1
SIVB_TRANSITION_MAX_RESPONSE_BYTES = {
    "sec": 2_000_000,
    "occ": 2_000_000,
    "eodhd": 4_000_000,
}
SIVB_TRANSITION_TIMEOUT_SECONDS = 45

# Exact pins from the authorized one-shot collection.  The OCC hash binds the
# deterministic reviewed extraction because its official PDF endpoint rejected
# direct archival clients; it must not be described as a raw-PDF hash.
SIVB_TRANSITION_RETRIEVED_AT = "2026-07-18T14:11:49.785762Z"
SIVB_SEC_MARKET_COLLECTED_HASH = (
    "69f3b20dfab4c9c43641a3c38a99f288129665af40e5ae3e6993ec36ccf4fcef"
)
SIVB_SEC_MARKET_COLLECTED_BYTES = 33_250
SIVB_OCC_MEMO_COLLECTED_HASH = (
    "11e986df7ea010021dd343353e24da8009673053b555b32cce95d4aae1598c6f"
)
SIVB_OCC_MEMO_COLLECTED_BYTES = 659
SIVBQ_EOD_COLLECTED_HASH = (
    "038c5a1ab7a5b439835a12507ebacc8bd8342ba73005479a0c57acc60ff04a1f"
)
SIVBQ_EOD_COLLECTED_BYTES = 44_932
SIVBQ_EOD_ROWS = 409
SIVBQ_NON_XNYS_EXCLUSIONS = ("2024-09-02",)
SIVBQ_STORED_ROWS = 408
SIVBQ_FIRST_SESSION = "2023-03-28"
SIVBQ_FIRST_OHLCV = (0.53, 0.74, 0.01, 0.4, 84_502_118.0)
SIVBQ_LAST_SESSION = "2024-11-07"
SIVBQ_LAST_OHLCV = (0.005, 0.006, 0.005, 0.006, 797.0)
SIVB_MARKET_TERMINATION = "2024-11-08"
SIVB_TICKER_EVENT_ID = (
    "01419d978e03e608512e4e898e695fdb39953278b08dc8138d97e0d0e21e4caa"
)
SIVB_MARKET_EXIT_EVENT_ID = (
    "f8cbecc851776e882ac85872972ac5c1680a672138276c1a9301d532c2d8ad3f"
)
SIVBQ_CANDIDATE_ID = (
    "ecc7e3f3b03853a8370f40c979ad08d8167989f6a83f7c9a68d09a308a3b2e16"
)

AVP_ID = "US:EODHD:529d8af8-043b-542e-8eeb-e8651009a2a8"
NTCO_ID = "US:EODHD:7e3cea59-409f-5cf0-8429-9d4245013622"
AVP_SYMBOL = "AVP"
NTCO_SYMBOL = "NTCO"
AVP_LAST_SESSION = "2020-01-03"
AVP_LEGAL_COMPLETION = "2020-01-03"
AVP_MARKET_TRANSITION = "2020-01-06"
AVP_RATIO = 0.3
AVP_OLD_EVENT_ID = (
    "7fd31cc07d0f1fff0c4b17a7b821acfd20e584a770c3119ff18ad074803e67d3"
)
AVP_NEW_EVENT_ID = (
    "825ea0640b20da42dcfa1c516ff921f272b0fd0a0fd4020de509674832391806"
)
AVP_CANDIDATE_ID = (
    "c0b08e4292606de3101f8de3234555743a348ae99ded84fdfbce3d7900e88339"
)
AVP_OFFICIAL_URL = (
    "https://www.sec.gov/Archives/edgar/data/8868/"
    "000095015720000022/0000950157-20-000022.txt"
)
AVP_OFFICIAL_HASH = (
    "12ca5855e19d9c0c0542f393964ef1e9ee0b1f831c26296f389f143d4bad42a4"
)
AVP_OFFICIAL_BYTES = 365_795
AVP_OFFICIAL_RETRIEVED_AT = "2026-07-18T10:30:42.442453Z"
AVP_RAW_URL = "https://eodhd.com/api/eod/AVP.US?from=2015-01-01&to=2026-07-15"
AVP_RAW_HASH = (
    "b7a04462fb63c48d389d5a03300e4ba0b6bc0307e4f03c06babd75e10f7ebbe4"
)
AVP_RAW_BYTES = 139_550
AVP_RAW_ROWS = 1_260
AVP_RAW_RETRIEVED_AT = "2026-07-16T15:57:00.046989Z"
AVP_TERMINAL_OHLCV = (5.64, 5.92, 5.05, 5.6, 236_653_175.0)

NTCO_RAW_URL = "https://eodhd.com/api/eod/NTCO.US?from=2015-01-01&to=2026-07-15"
NTCO_ENVELOPE_HASH = (
    "e88684de37208bd947df3140593aff81082126aefbc353d545f3ef0ae9fd8883"
)
NTCO_ENVELOPE_BYTES = 161_099
NTCO_RAW_HASH = (
    "91cb9baec50c86d49447d78f2882256a991884e46fda1a6019f5df792cb02dde"
)
NTCO_RAW_BYTES = 120_644
NTCO_RAW_ROWS = 1_075
NTCO_RAW_RETRIEVED_AT = "2026-07-17T20:37:19.646249Z"
NTCO_FIRST_OHLCV = (20.6, 20.73, 19.06, 19.46, 9_007_021.0)

TRUSTED_REPAIR_REGISTRY_SHA256 = (
    "41733c97fe1815b0611b827af8ad34aad0b015e30300f94b77f8fd22565d50d8"
)
TRUSTED_EVIDENCE_INVENTORY_SHA256 = (
    "61807fde7abccc27ee7d4aa598d3a6998a13fb6b4a0ae0fb5603a1857bbdeb99"
)


class EvidenceError(ValueError):
    """Raised when a hash-pinned prerequisite has changed or disappeared."""


def repair_registry_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "case": "AVP_to_NTCO",
            "security_id": AVP_ID,
            "successor_security_id": NTCO_ID,
            "old_event_id": AVP_OLD_EVENT_ID,
            "new_event_id": AVP_NEW_EVENT_ID,
            "candidate_id": AVP_CANDIDATE_ID,
            "legal_completion_date": AVP_LEGAL_COMPLETION,
            "market_transition_session": AVP_MARKET_TRANSITION,
            "ratio": AVP_RATIO,
            "official_source_hash": AVP_OFFICIAL_HASH,
            "successor_price_source_hash": NTCO_RAW_HASH,
        },
        {
            "case": "SIVB_to_SIVBQ",
            "security_id": SIVB_ID,
            "old_event_id": SIVB_EVENT_ID,
            "ticker_event_id": SIVB_TICKER_EVENT_ID,
            "market_exit_event_id": SIVB_MARKET_EXIT_EVENT_ID,
            "old_candidate_id": SIVB_CANDIDATE_ID,
            "new_candidate_id": SIVBQ_CANDIDATE_ID,
            "nasdaq_halt_date": SIVB_EXPECTED_MARKET_SESSION,
            "otc_transition_session": SIVBQ_FIRST_SESSION,
            "last_otc_price_session": SIVBQ_LAST_SESSION,
            "legal_cancellation_date": SIVB_LEGAL_CANCELLATION,
            "market_terminal_session": SIVB_MARKET_TERMINATION,
            "same_security_identity": True,
            "ratio": 1.0,
            "sec_market_source_hash": SIVB_SEC_MARKET_COLLECTED_HASH,
            "occ_reviewed_extraction_hash": SIVB_OCC_MEMO_COLLECTED_HASH,
            "otc_price_source_hash": SIVBQ_EOD_COLLECTED_HASH,
            "raw_otc_rows": SIVBQ_EOD_ROWS,
            "stored_otc_rows": SIVBQ_STORED_ROWS,
            "non_xnys_exclusions": list(SIVBQ_NON_XNYS_EXCLUSIONS),
        },
    ]
    output = []
    for row in rows:
        item = dict(row)
        item["registry_item_sha256"] = _canonical_json_sha256(row)
        output.append(item)
    return output


def repair_registry_inventory_sha256() -> str:
    return _canonical_json_sha256(repair_registry_inventory())


def evidence_inventory() -> list[dict[str, Any]]:
    return [
        {
            "label": "sivb_nasdaq_halt_and_suspension",
            "dataset": "sec_edgar_filing",
            "source": "sec_edgar_filing",
            "source_url": SIVB_SEC_MARKET_URL,
            "source_hash": SIVB_SEC_MARKET_COLLECTED_HASH,
            "size": SIVB_SEC_MARKET_COLLECTED_BYTES,
            "suffix": "html",
            "content_type": "text/html",
            "retrieved_at": SIVB_TRANSITION_RETRIEVED_AT,
            "raw_payload": True,
        },
        {
            "label": "sivb_occ_memo_52179_reviewed_extraction",
            "dataset": "occ_reviewed_memo_extraction",
            "source": "occ_reviewed_memo_extraction",
            "source_url": SIVB_OCC_MEMO_URL,
            "source_hash": SIVB_OCC_MEMO_COLLECTED_HASH,
            "size": SIVB_OCC_MEMO_COLLECTED_BYTES,
            "suffix": "json",
            "content_type": "application/json",
            "retrieved_at": SIVB_TRANSITION_RETRIEVED_AT,
            "raw_payload": False,
        },
        {
            "label": "sivbq_eodhd_otc_price_path",
            "dataset": "eodhd_eod",
            "source": "eodhd_eod",
            "source_url": SIVBQ_SAFE_URL,
            "source_hash": SIVBQ_EOD_COLLECTED_HASH,
            "size": SIVBQ_EOD_COLLECTED_BYTES,
            "suffix": "json",
            "content_type": "application/json",
            "retrieved_at": SIVB_TRANSITION_RETRIEVED_AT,
            "raw_payload": True,
        },
    ]


def evidence_inventory_sha256() -> str:
    return _canonical_json_sha256(evidence_inventory())


def _verify_code_pins() -> None:
    observed = {
        "repair_registry": repair_registry_inventory_sha256(),
        "evidence_inventory": evidence_inventory_sha256(),
    }
    expected = {
        "repair_registry": TRUSTED_REPAIR_REGISTRY_SHA256,
        "evidence_inventory": TRUSTED_EVIDENCE_INVENTORY_SHA256,
    }
    if observed != expected:
        raise RuntimeError(
            "SIVB/AVP repair code pins changed: "
            + _canonical_json({"observed": observed, "expected": expected})
        )


@dataclass(frozen=True)
class PreparedPlan:
    release: DataRelease
    release_etag: str | None
    pointer_etags: Mapping[str, str | None]
    planned_versions: Mapping[str, str]
    frame_hashes: Mapping[str, str]
    frames: Mapping[str, pd.DataFrame]
    plan: Mapping[str, Any]

    @property
    def release_version(self) -> str:
        return self.release.version


FailureInjector = Callable[[str], None]


def _frame_content_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "columns": [str(column) for column in frame.columns],
                "dtypes": [str(dtype) for dtype in frame.dtypes],
                "rows": len(frame),
            }
        ).encode()
    )
    digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return digest.hexdigest()


def _candidate_frame_hashes(
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, str]:
    return {
        dataset: _frame_content_sha256(frames[dataset])
        for dataset in WRITE_DATASETS
    }


def _semantic_candidate_hashes(
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, str]:
    output: dict[str, str] = {}
    lineage_columns = {
        "source_version",
        "source_hash",
        "calculated_at",
        "retrieved_at",
    }
    for dataset in WRITE_DATASETS:
        frame = frames[dataset]
        if dataset == "adjustment_factors":
            frame = frame.drop(
                columns=[
                    column for column in frame.columns if column in lineage_columns
                ]
            )
        output[dataset] = _frame_content_sha256(frame)
    return output


def _text(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def _date(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _number(value: Any) -> float | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _require_sivb_transition_url(label: str, url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    expected = {
        "sec": (
            "www.sec.gov",
            "/Archives/edgar/data/719739/000119312523073665/d485308d8k.htm",
            "",
        ),
        "occ": ("infomemo.theocc.com", "/infomemos", "number=52179"),
        "eodhd": ("eodhd.com", f"/api/eod/{SIVBQ_PROVIDER_SYMBOL}", ""),
    }
    if label not in expected:
        raise ValueError(f"Unknown SIVB transition source label: {label}.")
    host, path, query = expected[label]
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() != host
        or parsed.path != path
        or parsed.fragment
        or (label != "eodhd" and parsed.query != query)
    ):
        raise ValueError(f"SIVB {label} source URL envelope changed.")
    if label == "sec" and url != SIVB_SEC_MARKET_URL:
        raise ValueError("SIVB SEC source is not the exact reviewed URL.")
    if label == "occ" and url != SIVB_OCC_MEMO_URL:
        raise ValueError("SIVB OCC source is not the exact reviewed URL.")
    if label == "eodhd":
        query_values = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        if set(query_values) != {"from", "to", "api_token", "fmt"}:
            raise ValueError("SIVBQ EODHD query fields changed.")
        if (
            query_values["from"] != [SIVBQ_FETCH_START]
            or query_values["to"] != [SIVBQ_FETCH_END]
            or query_values["fmt"] != ["json"]
            or len(query_values["api_token"]) != 1
            or not query_values["api_token"][0]
        ):
            raise ValueError("SIVBQ EODHD query envelope changed.")


def _validate_sec_user_agent(value: str) -> str:
    user_agent = value.strip()
    if not user_agent or "@" not in user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT with a contact email is required for SIVB collection."
        )
    return user_agent


def _fetch_exact_once(
    url: str,
    headers: Mapping[str, str],
    max_bytes: int,
) -> bytes:
    request = urllib.request.Request(url, headers=dict(headers))
    with urllib.request.urlopen(
        request, timeout=SIVB_TRANSITION_TIMEOUT_SECONDS
    ) as response:
        status = int(getattr(response, "status", response.getcode()))
        final_url = str(response.geturl())
        if status != 200:
            raise RuntimeError(f"SIVB evidence returned HTTP {status}.")
        if final_url != url:
            raise RuntimeError(
                "SIVB evidence redirected outside the exact reviewed URL: "
                + final_url
            )
        content = response.read(max_bytes + 1)
    if not content or len(content) > max_bytes:
        raise RuntimeError("SIVB evidence response size is outside its envelope.")
    return content


def _verify_sivb_sec_market_payload(payload: bytes) -> None:
    if not payload or len(payload) > SIVB_TRANSITION_MAX_RESPONSE_BYTES["sec"]:
        raise EvidenceError("SIVB SEC market payload size is invalid.")
    text = _normalized_official_text(payload)
    _require_patterns(
        "SIVB SEC market transition",
        text,
        {
            "issuer": r"\bSVB Financial Group\b",
            "nasdaq_halt_2023-03-10": (
                r"SIVB:NASDAQ.{0,500}?on Nasdaq was halted on March\s+10,\s+2023"
            ),
            "nasdaq_suspension_2023-03-28": (
                r"will be suspended on March\s+28,\s+2023"
            ),
            "otc_pink_eligibility": r"OTC Pink Quotation System",
        },
    )


def _sivb_occ_reviewed_extraction() -> dict[str, Any]:
    """Deterministic extraction of official OCC memo 52179.

    OCC serves the memo at the pinned PDF URL but rejects non-browser archive
    clients.  As with the existing FRCB repair, the planner therefore stores a
    narrow reviewed extraction, not a claim that the raw PDF was archived.
    """

    return {
        "schema": "occ_reviewed_memo_extraction/v1",
        "memo_number": "52179",
        "source_url": SIVB_OCC_MEMO_URL,
        "subject": "SVB Financial Group - Symbol Change",
        "announcement_date": "2023-03-27",
        "effective_date": "2023-03-28",
        "old_symbol": SIVB_SYMBOL,
        "new_symbol": "SIVBQ",
        "market": "OTC",
        "opening_of_business": True,
        "underlying_security_change": "SIVB changes to SIVBQ",
        "cusip": "78486Q101",
        "contract_multiplier": 1,
        "deliverable_per_contract": "100 SVB Financial Group (SIVBQ) Common Shares",
        "reviewed_claim": (
            "SIVB and SIVBQ are the same SVB Financial Group common-share "
            "identity; the market and ticker changed on 2023-03-28."
        ),
        "raw_pdf_archived": False,
    }


def _sivb_occ_reviewed_payload() -> bytes:
    return (_canonical_json(_sivb_occ_reviewed_extraction()) + "\n").encode()


def _verify_sivb_occ_payload(payload: bytes) -> None:
    if not payload or len(payload) > SIVB_TRANSITION_MAX_RESPONSE_BYTES["occ"]:
        raise EvidenceError("SIVB OCC reviewed extraction size is invalid.")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("SIVB OCC reviewed extraction is invalid JSON.") from exc
    if value != _sivb_occ_reviewed_extraction():
        raise EvidenceError("SIVB OCC reviewed extraction fields changed.")


def _verify_sivbq_eod_payload(payload: bytes) -> list[Mapping[str, Any]]:
    if not payload or len(payload) > SIVB_TRANSITION_MAX_RESPONSE_BYTES["eodhd"]:
        raise EvidenceError("SIVBQ EOD payload size is invalid.")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("SIVBQ EOD payload is not JSON.") from exc
    if not isinstance(value, list) or not value or not all(
        isinstance(row, Mapping) for row in value
    ):
        raise EvidenceError("SIVBQ EOD payload is not a non-empty row list.")
    sessions: list[str] = []
    for row in value:
        session = _date(row.get("date"))
        if not session or session < SIVBQ_FETCH_START or session > SIVBQ_FETCH_END:
            raise EvidenceError("SIVBQ EOD row lies outside the reviewed date range.")
        numbers = {
            field: _number(row.get(field))
            for field in ("open", "high", "low", "close", "volume")
        }
        if any(numbers[field] is None for field in numbers):
            raise EvidenceError(f"SIVBQ EOD row has invalid OHLCV: {session}.")
        o, h, low, close, volume = (numbers[field] for field in numbers)
        assert o is not None and h is not None and low is not None
        assert close is not None and volume is not None
        if (
            min(o, h, low, close) <= 0
            or h < max(o, low, close)
            or low > min(o, h, close)
            or volume < 0
        ):
            raise EvidenceError(f"SIVBQ EOD row has invalid price geometry: {session}.")
        sessions.append(session)
    if sessions != sorted(set(sessions)):
        raise EvidenceError("SIVBQ EOD sessions are duplicated or unsorted.")
    return list(value)


def _sivb_transition_cache_dir(cache_root: Path) -> Path:
    return cache_root / SIVB_TRANSITION_EVIDENCE_SUBDIR


def _sivb_transition_report_path(cache_root: Path) -> Path:
    return _sivb_transition_cache_dir(cache_root) / SIVB_TRANSITION_REPORT


def _safe_sivb_transition_payload_path(cache_root: Path, filename: str) -> Path:
    base = _sivb_transition_cache_dir(cache_root).resolve()
    path = (base / filename).resolve()
    if path == base or base not in path.parents:
        raise EvidenceError("SIVB transition payload path escapes its cache directory.")
    return path


def _budget_used(budget: EodhdCallBudget) -> int:
    used = int(budget.seed_used)
    if budget.state_path.is_file():
        try:
            value = json.loads(budget.state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
            value = {}
        if _text(value.get("period")) == budget.period:
            used = max(used, int(value.get("used", 0)))
    return used


def _eod_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    last = rows[-1]
    return {
        "row_count": len(rows),
        "first_session": _date(first.get("date")),
        "first_ohlcv": [
            _number(first.get(field))
            for field in ("open", "high", "low", "close", "volume")
        ],
        "last_session": _date(last.get("date")),
        "last_ohlcv": [
            _number(last.get(field))
            for field in ("open", "high", "low", "close", "volume")
        ],
    }


def verify_sivb_transition_cache(cache_root: Path) -> dict[str, Any] | None:
    report_path = _sivb_transition_report_path(cache_root)
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("SIVB transition cache report is unreadable.") from exc
    required = {
        "schema",
        "status",
        "evidence",
        "http_attempts_total",
        "eodhd_calls",
        "budget_receipt",
        "r2_accessed",
    }
    if set(report) != required:
        raise EvidenceError("SIVB transition cache report fields are not exact.")
    if not (
        report.get("schema") == SIVB_TRANSITION_EVIDENCE_SCHEMA
        and report.get("status") == "collected"
        and report.get("http_attempts_total") == SIVB_TRANSITION_MAX_HTTP_ATTEMPTS
        and report.get("eodhd_calls") == SIVB_TRANSITION_MAX_EODHD_ATTEMPTS
        and report.get("r2_accessed") is False
    ):
        raise EvidenceError("SIVB transition cache report contract changed.")
    receipt = report.get("budget_receipt")
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "period",
        "used_before",
        "used_after",
        "delta",
        "daily_limit",
        "reserve",
        "safety_ceiling",
    }:
        raise EvidenceError("SIVB transition budget receipt is invalid.")
    if (
        int(receipt["delta"]) != 1
        or int(receipt["used_after"]) - int(receipt["used_before"]) != 1
        or int(receipt["safety_ceiling"])
        != int(receipt["daily_limit"]) - int(receipt["reserve"])
    ):
        raise EvidenceError("SIVB transition budget delta is not exactly one.")

    evidence = report.get("evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != {"sec", "occ", "eodhd"}:
        raise EvidenceError("SIVB transition evidence inventory changed.")
    expected_sources = {
        "sec": (SIVB_SEC_MARKET_URL, "html"),
        "occ": (SIVB_OCC_MEMO_URL, "json"),
        "eodhd": (SIVBQ_SAFE_URL, "json"),
    }
    payloads: dict[str, bytes] = {}
    for label, (source_url, suffix) in expected_sources.items():
        item = evidence.get(label)
        if not isinstance(item, Mapping) or set(item) != {
            "source_url",
            "source_hash",
            "size",
            "filename",
            "retrieved_at",
        }:
            raise EvidenceError(f"SIVB {label} cache metadata fields changed.")
        digest = _text(item.get("source_hash"))
        filename = _text(item.get("filename"))
        size = item.get("size")
        if (
            _text(item.get("source_url")) != source_url
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or filename != f"{digest}.{suffix}"
            or not isinstance(size, int)
            or size <= 0
            or size > SIVB_TRANSITION_MAX_RESPONSE_BYTES[label]
            or not _text(item.get("retrieved_at")).endswith("Z")
        ):
            raise EvidenceError(f"SIVB {label} cache metadata is invalid.")
        path = _safe_sivb_transition_payload_path(cache_root, filename)
        if not path.is_file():
            raise EvidenceError(f"SIVB {label} cached payload is missing: {path}.")
        payload = path.read_bytes()
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
            raise EvidenceError(f"SIVB {label} cached payload hash/size changed.")
        payloads[label] = payload
    _verify_sivb_sec_market_payload(payloads["sec"])
    _verify_sivb_occ_payload(payloads["occ"])
    rows = _verify_sivbq_eod_payload(payloads["eodhd"])
    return {
        **report,
        "payloads": payloads,
        "eod_rows": rows,
        "eod_summary": _eod_summary(rows),
    }


def _verify_reviewed_sivb_transition_cache(cache_root: Path) -> dict[str, Any]:
    cached = verify_sivb_transition_cache(cache_root)
    if cached is None:
        raise EvidenceError("Reviewed SIVB transition cache is missing.")
    expected = {
        "sec": (
            SIVB_SEC_MARKET_COLLECTED_HASH,
            SIVB_SEC_MARKET_COLLECTED_BYTES,
        ),
        "occ": (
            SIVB_OCC_MEMO_COLLECTED_HASH,
            SIVB_OCC_MEMO_COLLECTED_BYTES,
        ),
        "eodhd": (SIVBQ_EOD_COLLECTED_HASH, SIVBQ_EOD_COLLECTED_BYTES),
    }
    for label, (digest, size) in expected.items():
        item = cached["evidence"][label]
        observed = (
            _text(item.get("source_hash")),
            int(item.get("size", -1)),
            _text(item.get("retrieved_at")),
        )
        wanted = (digest, size, SIVB_TRANSITION_RETRIEVED_AT)
        if observed != wanted:
            raise EvidenceError(
                f"Reviewed SIVB {label} transition pin changed: {observed!r}."
            )
    rows = cached["eod_rows"]
    summary = cached["eod_summary"]
    expected_summary = {
        "row_count": SIVBQ_EOD_ROWS,
        "first_session": SIVBQ_FIRST_SESSION,
        "first_ohlcv": list(SIVBQ_FIRST_OHLCV),
        "last_session": SIVBQ_LAST_SESSION,
        "last_ohlcv": list(SIVBQ_LAST_OHLCV),
    }
    if summary != expected_summary or len(rows) != SIVBQ_EOD_ROWS:
        raise EvidenceError("Reviewed SIVBQ EOD boundary or row inventory changed.")
    receipt = cached["budget_receipt"]
    if not (
        int(receipt["used_before"]) == 8839
        and int(receipt["used_after"]) == 8840
        and int(receipt["delta"]) == 1
        and int(receipt["daily_limit"]) == 100000
        and int(receipt["reserve"]) == 5000
    ):
        raise EvidenceError("Reviewed SIVBQ collection budget receipt changed.")
    return cached


def _evidence_object_path(entry: Mapping[str, Any], completed_session: str) -> str:
    return (
        f"archives/{completed_session}/{entry['source_hash']}."
        f"{entry['suffix']}.gz"
    )


def _evidence_payload_for_entry(
    transition: Mapping[str, Any], entry: Mapping[str, Any]
) -> bytes:
    label_to_cache_key = {
        "sivb_nasdaq_halt_and_suspension": "sec",
        "sivb_occ_memo_52179_reviewed_extraction": "occ",
        "sivbq_eodhd_otc_price_path": "eodhd",
    }
    key = label_to_cache_key[_text(entry.get("label"))]
    payload = transition["payloads"][key]
    if (
        len(payload) != int(entry["size"])
        or hashlib.sha256(payload).hexdigest() != entry["source_hash"]
    ):
        raise EvidenceError(f"Prepared evidence bytes changed: {entry['label']}.")
    return payload


def _source_archive_evidence_row(
    archive: pd.DataFrame,
    entry: Mapping[str, Any],
    *,
    completed_session: str,
) -> dict[str, Any]:
    row = {column: None for column in archive.columns}
    values = {
        "archive_id": entry["source_hash"],
        "dataset": entry["dataset"],
        "object_path": _evidence_object_path(entry, completed_session),
        "content_type": entry["content_type"],
        "effective_date": completed_session,
        "source": entry["source"],
        "retrieved_at": entry["retrieved_at"],
        "source_hash": entry["source_hash"],
        "source_url": entry["source_url"],
    }
    missing = sorted(set(values) - set(row))
    if missing:
        raise EvidenceError(
            "source_archive lacks evidence fields: " + ", ".join(missing)
        )
    row.update(values)
    return row


def _rewrite_source_archive(
    archive: pd.DataFrame,
    *,
    completed_session: str,
) -> tuple[pd.DataFrame, int]:
    output = archive.copy(deep=True)
    additions: list[dict[str, Any]] = []
    states: list[str] = []
    for entry in evidence_inventory():
        expected = _source_archive_evidence_row(
            archive, entry, completed_session=completed_session
        )
        related = (
            output["archive_id"].astype(str).eq(entry["source_hash"])
            | output["source_hash"].astype(str).eq(entry["source_hash"])
            | output["object_path"].astype(str).eq(expected["object_path"])
            | output["source_url"].fillna("").astype(str).eq(entry["source_url"])
        )
        rows = output.loc[related]
        if rows.empty:
            additions.append(expected)
            states.append("old")
            continue
        exact = len(rows) == 1 and all(
            _text(rows.iloc[0].get(field)) == _text(value)
            for field, value in expected.items()
        )
        if not exact:
            raise EvidenceError(
                f"Conflicting source_archive evidence row: {entry['label']}."
            )
        states.append("repaired")
    if len(set(states)) != 1:
        raise RuntimeError(
            "SIVB evidence source_archive is partially applied: " + repr(states)
        )
    if additions:
        output = pd.concat(
            [output, pd.DataFrame(additions, columns=output.columns)],
            ignore_index=True,
        )
    primary_key = list(dataset_spec("source_archive").primary_key)
    if output.duplicated(primary_key, keep=False).any():
        raise EvidenceError("SIVB evidence duplicates source_archive primary keys.")
    return output.reset_index(drop=True), len(additions)


def _safe_repository_path(root: Path, object_path: str) -> Path:
    base = root.resolve()
    path = (base / object_path).resolve()
    if path == base or base not in path.parents:
        raise EvidenceError("SIVB evidence archive path escapes repository root.")
    return path


def _verify_persisted_evidence(
    repository: LocalDatasetRepository,
    transition: Mapping[str, Any],
    *,
    completed_session: str,
) -> None:
    for entry in evidence_inventory():
        expected = _evidence_payload_for_entry(transition, entry)
        path = _safe_repository_path(
            repository.root,
            _evidence_object_path(entry, completed_session),
        )
        if not path.is_file():
            raise EvidenceError(f"Persisted evidence is missing: {path}.")
        try:
            observed = gzip.decompress(path.read_bytes())
        except (OSError, EOFError) as exc:
            raise EvidenceError(f"Persisted evidence is not valid gzip: {path}.") from exc
        if observed != expected:
            raise EvidenceError(f"Persisted evidence conflicts with cache: {path}.")


def _persist_evidence(
    repository: LocalDatasetRepository,
    transition: Mapping[str, Any],
    *,
    completed_session: str,
) -> None:
    for entry in evidence_inventory():
        payload = _evidence_payload_for_entry(transition, entry)
        path = _safe_repository_path(
            repository.root,
            _evidence_object_path(entry, completed_session),
        )
        if path.is_file():
            try:
                existing = gzip.decompress(path.read_bytes())
            except (OSError, EOFError) as exc:
                raise EvidenceError(
                    f"Existing immutable evidence is not valid gzip: {path}."
                ) from exc
            if existing != payload:
                raise EvidenceError(
                    f"Existing immutable evidence conflicts with cache: {path}."
                )
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(path, gzip.compress(payload, mtime=0))
    _verify_persisted_evidence(
        repository, transition, completed_session=completed_session
    )


SivbTransitionFetcher = Callable[[str, Mapping[str, str], int], bytes]


def collect_sivb_transition_evidence(
    cache_root: Path,
    *,
    sec_user_agent: str,
    eodhd_token: str,
    budget: EodhdCallBudget,
    fetcher: SivbTransitionFetcher = _fetch_exact_once,
) -> dict[str, Any]:
    cached = verify_sivb_transition_cache(cache_root)
    if cached is not None:
        return {
            **cached,
            "status": "cache_verified",
            "mode": "fetch",
            "http_attempts_this_run": 0,
            "eodhd_calls_this_run": 0,
            "network_accessed": False,
            "writes_performed": False,
        }
    user_agent = _validate_sec_user_agent(sec_user_agent)
    token = eodhd_token.strip()
    if not token:
        raise RuntimeError("EODHD_API_TOKEN is required for SIVBQ collection.")
    _require_sivb_transition_url("sec", SIVB_SEC_MARKET_URL)
    _require_sivb_transition_url("occ", SIVB_OCC_MEMO_URL)
    eodhd_url = "https://eodhd.com/api/eod/" + SIVBQ_PROVIDER_SYMBOL + "?" + urllib.parse.urlencode(
        {
            "from": SIVBQ_FETCH_START,
            "to": SIVBQ_FETCH_END,
            "api_token": token,
            "fmt": "json",
        }
    )
    _require_sivb_transition_url("eodhd", eodhd_url)

    sec_payload = fetcher(
        SIVB_SEC_MARKET_URL,
        {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Encoding": "identity",
        },
        SIVB_TRANSITION_MAX_RESPONSE_BYTES["sec"],
    )
    _verify_sivb_sec_market_payload(sec_payload)
    occ_payload = _sivb_occ_reviewed_payload()
    _verify_sivb_occ_payload(occ_payload)

    used_before = _budget_used(budget)
    budget.claim()
    eodhd_payload = fetcher(
        eodhd_url,
        {"Accept": "application/json", "Accept-Encoding": "identity"},
        SIVB_TRANSITION_MAX_RESPONSE_BYTES["eodhd"],
    )
    eod_rows = _verify_sivbq_eod_payload(eodhd_payload)
    used_after = _budget_used(budget)
    if used_after - used_before != SIVB_TRANSITION_MAX_EODHD_ATTEMPTS:
        raise RuntimeError("SIVBQ collection did not consume exactly one budget call.")

    retrieved_at = _now()
    raw_sources = {
        "sec": (SIVB_SEC_MARKET_URL, "html", sec_payload),
        "occ": (SIVB_OCC_MEMO_URL, "json", occ_payload),
        "eodhd": (SIVBQ_SAFE_URL, "json", eodhd_payload),
    }
    evidence: dict[str, Any] = {}
    cache_dir = _sivb_transition_cache_dir(cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for label, (source_url, suffix, payload) in raw_sources.items():
        digest = hashlib.sha256(payload).hexdigest()
        filename = f"{digest}.{suffix}"
        path = _safe_sivb_transition_payload_path(cache_root, filename)
        if path.is_file() and path.read_bytes() != payload:
            raise RuntimeError(f"Immutable SIVB transition cache collision: {path}.")
        if not path.is_file():
            write_atomic(path, payload)
        evidence[label] = {
            "source_url": source_url,
            "source_hash": digest,
            "size": len(payload),
            "filename": filename,
            "retrieved_at": retrieved_at,
        }
    receipt = {
        "period": budget.period,
        "used_before": used_before,
        "used_after": used_after,
        "delta": used_after - used_before,
        "daily_limit": int(budget.limit),
        "reserve": int(budget.reserve),
        "safety_ceiling": int(budget.ceiling),
    }
    report = {
        "schema": SIVB_TRANSITION_EVIDENCE_SCHEMA,
        "status": "collected",
        "evidence": evidence,
        "http_attempts_total": SIVB_TRANSITION_MAX_HTTP_ATTEMPTS,
        "eodhd_calls": SIVB_TRANSITION_MAX_EODHD_ATTEMPTS,
        "budget_receipt": receipt,
        "r2_accessed": False,
    }
    write_atomic(
        _sivb_transition_report_path(cache_root),
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )
    verified = verify_sivb_transition_cache(cache_root)
    if verified is None:
        raise RuntimeError("SIVB transition cache disappeared after collection.")
    return {
        **verified,
        "status": "collected",
        "mode": "fetch",
        "http_attempts_this_run": SIVB_TRANSITION_MAX_HTTP_ATTEMPTS,
        "eodhd_calls_this_run": SIVB_TRANSITION_MAX_EODHD_ATTEMPTS,
        "network_accessed": True,
        "writes_performed": True,
        "eod_summary": _eod_summary(eod_rows),
    }


def sivb_transition_collection_plan(cache_root: Path) -> dict[str, Any]:
    cached = verify_sivb_transition_cache(cache_root)
    if cached is not None:
        return {
            **cached,
            "status": "cache_verified",
            "mode": "offline_collection_plan",
            "http_attempts_this_run": 0,
            "eodhd_calls_this_run": 0,
            "network_accessed": False,
            "writes_performed": False,
        }
    return {
        "schema": SIVB_TRANSITION_EVIDENCE_SCHEMA,
        "status": "ready_for_authorized_fetch",
        "mode": "offline_collection_plan",
        "sources": {
            "sec": SIVB_SEC_MARKET_URL,
            "occ": SIVB_OCC_MEMO_URL,
            "eodhd": SIVBQ_SAFE_URL,
        },
        "max_http_attempts": SIVB_TRANSITION_MAX_HTTP_ATTEMPTS,
        "max_eodhd_attempts": SIVB_TRANSITION_MAX_EODHD_ATTEMPTS,
        "http_attempts_this_run": 0,
        "eodhd_calls_this_run": 0,
        "network_accessed": False,
        "writes_performed": False,
        "r2_accessed": False,
    }


def _normalized_official_text(payload: bytes) -> str:
    decoded = payload.decode("utf-8", errors="replace")
    decoded = html.unescape(re.sub(r"<[^>]+>", " ", decoded))
    return re.sub(r"\s+", " ", decoded).strip()


def _archive_path(completed_session: str, digest: str, suffix: str) -> str:
    return f"archives/{completed_session}/{digest}.{suffix}.gz"


def _safe_archive_file(repository: LocalDatasetRepository, object_path: str) -> Path:
    root = repository.root.resolve()
    path = (root / object_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise EvidenceError(f"Archive path escapes repository root: {object_path}") from exc
    if not path.is_file():
        raise EvidenceError(f"Pinned archive object is missing: {object_path}")
    return path


def _archive_row(
    archive: pd.DataFrame,
    *,
    completed_session: str,
    digest: str,
    suffix: str,
    dataset: str,
    content_type: str,
    source_url: str,
    retrieved_at: str,
) -> Mapping[str, Any]:
    expected_path = _archive_path(completed_session, digest, suffix)
    related = archive.loc[
        archive["archive_id"].astype(str).eq(digest)
        | archive["source_hash"].astype(str).eq(digest)
        | archive["object_path"].astype(str).eq(expected_path)
    ]
    if len(related) != 1:
        raise EvidenceError(
            f"Pinned archive inventory for {digest} is not exactly one row."
        )
    row = related.iloc[0]
    expected = {
        "archive_id": digest,
        "dataset": dataset,
        "object_path": expected_path,
        "content_type": content_type,
        "effective_date": completed_session,
        "source": dataset,
        "retrieved_at": retrieved_at,
        "source_hash": digest,
        "source_url": source_url,
    }
    changed = {
        field: (_text(row.get(field)), value)
        for field, value in expected.items()
        if (_date(row.get(field)) if field == "effective_date" else _text(row.get(field)))
        != value
    }
    if changed:
        raise EvidenceError(
            f"Pinned archive metadata changed for {digest}: {_canonical_json(changed)}"
        )
    return row


def _archive_payload(
    repository: LocalDatasetRepository,
    row: Mapping[str, Any],
    *,
    digest: str,
    expected_bytes: int,
) -> bytes:
    path = _safe_archive_file(repository, _text(row.get("object_path")))
    try:
        with gzip.open(path, "rb") as handle:
            payload = handle.read()
    except (OSError, EOFError) as exc:
        raise EvidenceError(f"Pinned archive object is not valid gzip: {path}") from exc
    actual_digest = hashlib.sha256(payload).hexdigest()
    if actual_digest != digest or len(payload) != expected_bytes:
        raise EvidenceError(
            f"Pinned archive bytes changed for {digest}: "
            f"sha256={actual_digest}; bytes={len(payload)}."
        )
    return payload


def _official_payload(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    completed_session: str,
    digest: str,
    suffix: str,
    content_type: str,
    source_url: str,
    retrieved_at: str,
    expected_bytes: int,
) -> bytes:
    row = _archive_row(
        archive,
        completed_session=completed_session,
        digest=digest,
        suffix=suffix,
        dataset="sec_edgar_filing",
        content_type=content_type,
        source_url=source_url,
        retrieved_at=retrieved_at,
    )
    return _archive_payload(
        repository, row, digest=digest, expected_bytes=expected_bytes
    )


def _plain_eod_records(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    completed_session: str,
    digest: str,
    source_url: str,
    retrieved_at: str,
    expected_bytes: int,
    expected_rows: int,
) -> list[Mapping[str, Any]]:
    row = _archive_row(
        archive,
        completed_session=completed_session,
        digest=digest,
        suffix="json",
        dataset="eodhd_eod",
        content_type="application/json",
        source_url=source_url,
        retrieved_at=retrieved_at,
    )
    payload = _archive_payload(
        repository, row, digest=digest, expected_bytes=expected_bytes
    )
    try:
        records = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Archived EOD JSON is invalid: {digest}") from exc
    if (
        not isinstance(records, list)
        or len(records) != expected_rows
        or not all(isinstance(record, Mapping) for record in records)
    ):
        raise EvidenceError(f"Archived EOD row inventory changed: {digest}")
    return list(records)


def _enveloped_eod_records(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    completed_session: str,
) -> list[Mapping[str, Any]]:
    row = _archive_row(
        archive,
        completed_session=completed_session,
        digest=NTCO_ENVELOPE_HASH,
        suffix="json",
        dataset="eodhd_eod",
        content_type="application/vnd.supertrendquant.source-envelope+json",
        source_url=NTCO_RAW_URL,
        retrieved_at=NTCO_RAW_RETRIEVED_AT,
    )
    payload = _archive_payload(
        repository,
        row,
        digest=NTCO_ENVELOPE_HASH,
        expected_bytes=NTCO_ENVELOPE_BYTES,
    )
    try:
        envelope = json.loads(payload)
        raw = base64.b64decode(envelope["content_base64"], validate=True)
        records = json.loads(raw)
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("Archived NTCO source envelope is invalid.") from exc
    expected_envelope = {
        "content_sha256": NTCO_RAW_HASH,
        "content_type": "application/json",
        "source": "eodhd_eod",
        "source_url": NTCO_RAW_URL,
    }
    observed_envelope = {
        field: _text(envelope.get(field)) for field in expected_envelope
    }
    if observed_envelope != expected_envelope:
        raise EvidenceError("Archived NTCO source-envelope metadata changed.")
    if hashlib.sha256(raw).hexdigest() != NTCO_RAW_HASH or len(raw) != NTCO_RAW_BYTES:
        raise EvidenceError("Archived NTCO raw payload changed inside its envelope.")
    if (
        not isinstance(records, list)
        or len(records) != NTCO_RAW_ROWS
        or not all(isinstance(record, Mapping) for record in records)
    ):
        raise EvidenceError("Archived NTCO raw row inventory changed.")
    return list(records)


def _require_patterns(label: str, text: str, patterns: Mapping[str, str]) -> None:
    missing = [name for name, pattern in patterns.items() if not re.search(pattern, text, re.I)]
    if missing:
        raise EvidenceError(
            f"{label} official evidence no longer proves: {', '.join(missing)}."
        )


def _verify_official_evidence(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    completed_session: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    avp_payload = _official_payload(
        repository,
        archive,
        completed_session=completed_session,
        digest=AVP_OFFICIAL_HASH,
        suffix="txt",
        content_type="text/plain",
        source_url=AVP_OFFICIAL_URL,
        retrieved_at=AVP_OFFICIAL_RETRIEVED_AT,
        expected_bytes=AVP_OFFICIAL_BYTES,
    )
    avp_text = _normalized_official_text(avp_payload)
    _require_patterns(
        "AVP",
        avp_text,
        {
            "legal_completion_2020-01-03": (
                r"On January 3, 2020.{0,500}?consummated the previously announced "
                r"business combination"
            ),
            "exchange_ratio_0.300_ntco_ads": (
                r"each share.{0,800}?automatically converted.{0,300}?0\.300 "
                r"validly issued.{0,250}?American Depositary Shares"
            ),
            "avp_suspended_before_2020-01-06_open": (
                r"AVP.{0,250}?NYSE.{0,250}?suspended from trading.{0,250}?prior "
                r"to the opening of the market on January 6, 2020"
            ),
            "ntco_expected_to_begin_2020-01-06": (
                r"expects to begin trading.{0,120}?NYSE \(NTCO\) on January 6"
            ),
        },
    )

    sivb_payload = _official_payload(
        repository,
        archive,
        completed_session=completed_session,
        digest=SIVB_OFFICIAL_HASH,
        suffix="html",
        content_type="text/html",
        source_url=SIVB_OFFICIAL_URL,
        retrieved_at=SIVB_OFFICIAL_RETRIEVED_AT,
        expected_bytes=SIVB_OFFICIAL_BYTES,
    )
    sivb_text = _normalized_official_text(sivb_payload)
    _require_patterns(
        "SIVB",
        sivb_text,
        {
            "legal_effective_date_2024-11-07": (
                r"Confirmed Plan became effective.{0,120}?November 7, 2024"
            ),
            "common_equity_no_distribution": (
                r"No holders of Common Equity Interests.{0,100}?received any "
                r"distributions"
            ),
            "common_equity_cancelled_on_effective_date": (
                r"All Common Equity Interests.{0,120}?were canceled on the "
                r"Effective Date"
            ),
            "otc_trading_ceased_after_effective_date": (
                r"trading on the over-the-counter markets has ceased after the "
                r"Effective Date"
            ),
        },
    )
    market_patterns = {
        "official_nasdaq_name": r"\bNasdaq\b",
        "official_2023-03-10_date": r"\bMarch 10, 2023\b",
        "official_halt_or_suspension": r"\b(?:halted|suspended)\b.{0,180}?\btrading\b",
        "official_sivbq_continuity": r"\bSIVBQ\b",
    }
    missing_market_patterns = [
        name
        for name, pattern in market_patterns.items()
        if not re.search(pattern, sivb_text, re.I)
    ]
    return (
        {
            "source_url": AVP_OFFICIAL_URL,
            "source_hash": AVP_OFFICIAL_HASH,
            "exact_bytes": AVP_OFFICIAL_BYTES,
            "legal_completion_date": AVP_LEGAL_COMPLETION,
            "market_transition_session": AVP_MARKET_TRANSITION,
            "ratio": AVP_RATIO,
            "all_required_patterns_passed": True,
        },
        {
            "source_url": SIVB_OFFICIAL_URL,
            "source_hash": SIVB_OFFICIAL_HASH,
            "exact_bytes": SIVB_OFFICIAL_BYTES,
            "legal_cancellation_date": SIVB_LEGAL_CANCELLATION,
            "zero_distribution_proven": True,
            "otc_continuation_until_legal_effective_date_indicated": True,
            "market_boundary_proven": not missing_market_patterns,
            "missing_market_patterns": missing_market_patterns,
        },
    )


def _record(records: list[Mapping[str, Any]], session: str) -> Mapping[str, Any]:
    rows = [record for record in records if _date(record.get("date")) == session]
    if len(rows) != 1:
        raise EvidenceError(f"Archived price row is not unique for {session}.")
    return rows[0]


def _ohlcv_matches(row: Mapping[str, Any], expected: tuple[float, ...]) -> bool:
    observed = tuple(_number(row.get(field)) for field in ("open", "high", "low", "close", "volume"))
    return all(
        value is not None
        and math.isclose(float(value), float(wanted), rel_tol=0, abs_tol=1e-8)
        for value, wanted in zip(observed, expected, strict=True)
    )


def _verify_parquet_price_series(
    prices: pd.DataFrame,
    *,
    security_id: str,
    records: list[Mapping[str, Any]],
    expected_first: str,
    expected_last: str,
    expected_source_hash: str,
    expected_retrieved_at: str,
) -> None:
    rows = prices.loc[prices["security_id"].astype(str).eq(security_id)].copy()
    if len(rows) != len(records):
        raise EvidenceError(f"Stored price count differs from raw archive: {security_id}.")
    rows["_session"] = pd.to_datetime(rows["session"], errors="coerce").dt.date.astype(str)
    if rows["_session"].eq("NaT").any() or rows["_session"].duplicated().any():
        raise EvidenceError(f"Stored price sessions are invalid: {security_id}.")
    if rows["_session"].min() != expected_first or rows["_session"].max() != expected_last:
        raise EvidenceError(f"Stored price boundary changed: {security_id}.")
    raw = {_date(record.get("date")): record for record in records}
    if set(rows["_session"]) != set(raw):
        raise EvidenceError(f"Stored price sessions differ from raw archive: {security_id}.")
    for row in rows.to_dict("records"):
        if not _ohlcv_matches(row, tuple(float(raw[row["_session"]][field]) for field in ("open", "high", "low", "close", "volume"))):
            raise EvidenceError(
                f"Stored OHLCV differs from raw archive: {security_id}/{row['_session']}."
            )
    lineage = {
        "source": set(rows["source"].astype(str)),
        "retrieved_at": set(rows["retrieved_at"].astype(str)),
        "source_hash": set(rows["source_hash"].astype(str)),
        "currency": set(rows["currency"].astype(str)),
    }
    expected_lineage = {
        "source": {"eodhd_eod"},
        "retrieved_at": {expected_retrieved_at},
        "source_hash": {expected_source_hash},
        "currency": {"USD"},
    }
    if lineage != expected_lineage:
        raise EvidenceError(f"Stored price lineage changed: {security_id}.")


def _verify_price_evidence(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    completed_session: str,
    sivb_repaired: bool = False,
) -> dict[str, Any]:
    sivb = _plain_eod_records(
        repository,
        archive,
        completed_session=completed_session,
        digest=SIVB_RAW_HASH,
        source_url=SIVB_RAW_URL,
        retrieved_at=SIVB_RAW_RETRIEVED_AT,
        expected_bytes=SIVB_RAW_BYTES,
        expected_rows=SIVB_RAW_ROWS,
    )
    avp = _plain_eod_records(
        repository,
        archive,
        completed_session=completed_session,
        digest=AVP_RAW_HASH,
        source_url=AVP_RAW_URL,
        retrieved_at=AVP_RAW_RETRIEVED_AT,
        expected_bytes=AVP_RAW_BYTES,
        expected_rows=AVP_RAW_ROWS,
    )
    ntco = _enveloped_eod_records(
        repository, archive, completed_session=completed_session
    )
    if not _ohlcv_matches(_record(sivb, SIVB_LAST_SESSION), SIVB_TERMINAL_OHLCV):
        raise EvidenceError("SIVB terminal raw OHLCV changed.")
    if not _ohlcv_matches(_record(avp, AVP_LAST_SESSION), AVP_TERMINAL_OHLCV):
        raise EvidenceError("AVP terminal raw OHLCV changed.")
    if not _ohlcv_matches(_record(ntco, AVP_MARKET_TRANSITION), NTCO_FIRST_OHLCV):
        raise EvidenceError("NTCO first-session raw OHLCV changed.")
    if min(_date(record.get("date")) for record in ntco) != AVP_MARKET_TRANSITION:
        raise EvidenceError("NTCO archive now contains a price before 2020-01-06.")
    if sivb_repaired:
        sessions = pd.to_datetime(prices["session"], errors="coerce").dt.date.astype(str)
        original_sivb_prices = prices.loc[
            prices["security_id"].astype(str).eq(SIVB_ID)
            & sessions.le(SIVB_LAST_SESSION)
        ].copy()
        _verify_parquet_price_series(
            original_sivb_prices,
            security_id=SIVB_ID,
            records=sivb,
            expected_first="2015-01-02",
            expected_last=SIVB_LAST_SESSION,
            expected_source_hash=SIVB_RAW_HASH,
            expected_retrieved_at=SIVB_RAW_RETRIEVED_AT,
        )
    else:
        _verify_parquet_price_series(
            prices,
            security_id=SIVB_ID,
            records=sivb,
            expected_first="2015-01-02",
            expected_last=SIVB_LAST_SESSION,
            expected_source_hash=SIVB_RAW_HASH,
            expected_retrieved_at=SIVB_RAW_RETRIEVED_AT,
        )
    _verify_parquet_price_series(
        prices,
        security_id=AVP_ID,
        records=avp,
        expected_first="2015-01-02",
        expected_last=AVP_LAST_SESSION,
        expected_source_hash=AVP_RAW_HASH,
        expected_retrieved_at=AVP_RAW_RETRIEVED_AT,
    )
    _verify_parquet_price_series(
        prices,
        security_id=NTCO_ID,
        records=ntco,
        expected_first=AVP_MARKET_TRANSITION,
        expected_last="2024-04-12",
        expected_source_hash=NTCO_RAW_HASH,
        expected_retrieved_at=NTCO_RAW_RETRIEVED_AT,
    )
    return {
        "SIVB": {
            "last_price_session": SIVB_LAST_SESSION,
            "terminal_ohlcv": list(SIVB_TERMINAL_OHLCV),
            "raw_source_hash": SIVB_RAW_HASH,
        },
        "AVP": {
            "last_price_session": AVP_LAST_SESSION,
            "terminal_ohlcv": list(AVP_TERMINAL_OHLCV),
            "raw_source_hash": AVP_RAW_HASH,
        },
        "NTCO": {
            "first_price_session": AVP_MARKET_TRANSITION,
            "first_ohlcv": list(NTCO_FIRST_OHLCV),
            "raw_source_hash": NTCO_RAW_HASH,
            "source_envelope_hash": NTCO_ENVELOPE_HASH,
        },
    }


def _one_row(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise EvidenceError(f"{label} is missing or duplicated.")
    return rows.iloc[0]


def _verify_static_contract() -> None:
    checks = {
        "SIVB event": (
            canonical_lifecycle_event_id(SIVB_ID, "delisting", SIVB_LEGAL_CANCELLATION),
            SIVB_EVENT_ID,
        ),
        "SIVB candidate": (
            lifecycle_candidate_id(SIVB_ID, SIVB_LAST_SESSION),
            SIVB_CANDIDATE_ID,
        ),
        "SIVB ticker event": (
            canonical_lifecycle_event_id(
                SIVB_ID, "ticker_change", SIVBQ_FIRST_SESSION
            ),
            SIVB_TICKER_EVENT_ID,
        ),
        "SIVB market-exit event": (
            canonical_lifecycle_event_id(
                SIVB_ID, "delisting", SIVB_MARKET_TERMINATION
            ),
            SIVB_MARKET_EXIT_EVENT_ID,
        ),
        "SIVBQ candidate": (
            lifecycle_candidate_id(SIVB_ID, SIVBQ_LAST_SESSION),
            SIVBQ_CANDIDATE_ID,
        ),
        "AVP old event": (
            canonical_lifecycle_event_id(AVP_ID, "stock_merger", AVP_LEGAL_COMPLETION),
            AVP_OLD_EVENT_ID,
        ),
        "AVP new event": (
            canonical_lifecycle_event_id(AVP_ID, "stock_merger", AVP_MARKET_TRANSITION),
            AVP_NEW_EVENT_ID,
        ),
        "AVP candidate": (
            lifecycle_candidate_id(AVP_ID, AVP_LAST_SESSION),
            AVP_CANDIDATE_ID,
        ),
    }
    changed = {label: values for label, values in checks.items() if values[0] != values[1]}
    if changed:
        raise RuntimeError(f"Canonical lifecycle IDs changed: {_canonical_json(changed)}")


def _verify_current_rows(frames: Mapping[str, pd.DataFrame]) -> None:
    actions = frames["corporate_actions"]
    avp = _one_row(
        actions,
        actions["event_id"].astype(str).eq(AVP_OLD_EVENT_ID),
        "AVP stock-merger action",
    )
    avp_expected = {
        "security_id": AVP_ID,
        "action_type": "stock_merger",
        "effective_date": AVP_LEGAL_COMPLETION,
        "ex_date": AVP_LEGAL_COMPLETION,
        "announcement_date": AVP_LEGAL_COMPLETION,
        "record_date": "",
        "payment_date": "",
        "currency": "USD",
        "new_security_id": NTCO_ID,
        "new_symbol": NTCO_SYMBOL,
        "source_url": AVP_OFFICIAL_URL,
        "source_kind": "official_crosscheck",
        "source": "sec_edgar+stored_price_crosscheck",
        "retrieved_at": AVP_OFFICIAL_RETRIEVED_AT,
        "source_hash": AVP_OFFICIAL_HASH,
    }
    for field, value in avp_expected.items():
        observed = _date(avp.get(field)) if "date" in field else _text(avp.get(field))
        if observed != value:
            raise EvidenceError(f"AVP action field changed: {field}={observed!r}.")
    if (
        _number(avp.get("cash_amount")) is not None
        or not math.isclose(_number(avp.get("ratio")) or 0.0, AVP_RATIO, abs_tol=1e-12)
        or _text(avp.get("official")).lower() != "true"
        or _text(avp.get("metadata")) != ""
    ):
        raise EvidenceError("AVP action economics or provenance changed.")

    sivb = _one_row(
        actions,
        actions["event_id"].astype(str).eq(SIVB_EVENT_ID),
        "SIVB legal-cancellation action",
    )
    if not (
        _text(sivb.get("security_id")) == SIVB_ID
        and _text(sivb.get("action_type")) == "delisting"
        and _date(sivb.get("effective_date")) == SIVB_LEGAL_CANCELLATION
        and _date(sivb.get("ex_date")) == SIVB_LEGAL_CANCELLATION
        and math.isclose(_number(sivb.get("cash_amount")) or 0.0, 0.0, abs_tol=0.0)
        and _text(sivb.get("source_url")) == SIVB_OFFICIAL_URL
        and _text(sivb.get("source_hash")) == SIVB_OFFICIAL_HASH
    ):
        raise EvidenceError("SIVB legal-cancellation action changed.")

    resolutions = frames["lifecycle_resolutions"]
    expected_resolutions = {
        AVP_ID: (AVP_CANDIDATE_ID, AVP_LAST_SESSION, AVP_OLD_EVENT_ID),
        SIVB_ID: (SIVB_CANDIDATE_ID, SIVB_LAST_SESSION, SIVB_EVENT_ID),
    }
    for security_id, (candidate_id, last_price, event_id) in expected_resolutions.items():
        row = _one_row(
            resolutions,
            resolutions["security_id"].astype(str).eq(security_id),
            f"{security_id} lifecycle resolution",
        )
        if not (
            _text(row.get("candidate_id")) == candidate_id
            and _date(row.get("last_price_date")) == last_price
            and _text(row.get("resolution")) == "applied"
            and _text(row.get("event_id")) == event_id
        ):
            raise EvidenceError(f"Lifecycle resolution changed: {security_id}.")

    master = frames["security_master"]
    identity_expectations = {
        AVP_ID: (AVP_SYMBOL, "NYSE", AVP_LAST_SESSION),
        NTCO_ID: (NTCO_SYMBOL, "NYSE", "2024-04-12"),
        SIVB_ID: (SIVB_SYMBOL, "NASDAQ", SIVB_LAST_SESSION),
    }
    for security_id, (symbol, exchange, active_to) in identity_expectations.items():
        row = _one_row(
            master,
            master["security_id"].astype(str).eq(security_id),
            f"{symbol} security master",
        )
        expected_start = AVP_MARKET_TRANSITION if security_id == NTCO_ID else "2015-01-02"
        if not (
            _text(row.get("primary_symbol")).upper() == symbol
            and _text(row.get("exchange")).upper() == exchange
            and _date(row.get("active_from")) == expected_start
            and _date(row.get("active_to")) == active_to
        ):
            raise EvidenceError(f"Security-master boundary changed: {symbol}.")

    history = frames["symbol_history"]
    expected_history = {
        AVP_ID: (AVP_SYMBOL, "2015-01-01", ""),
        NTCO_ID: (NTCO_SYMBOL, AVP_LEGAL_COMPLETION, "2024-04-12"),
    }
    for security_id, (symbol, start, end) in expected_history.items():
        row = _one_row(
            history,
            history["security_id"].astype(str).eq(security_id)
            & history["symbol"].astype(str).str.upper().eq(symbol),
            f"{symbol} symbol history",
        )
        if _date(row.get("effective_from")) != start or _date(row.get("effective_to")) != end:
            raise EvidenceError(f"Symbol-history boundary changed: {symbol}.")


def _repair_state(
    frames: Mapping[str, pd.DataFrame],
    *,
    source_archive_rows_to_add: int,
) -> str:
    actions = set(frames["corporate_actions"]["event_id"].astype(str))
    candidates = set(frames["lifecycle_resolutions"]["candidate_id"].astype(str))
    master = _one_row(
        frames["security_master"],
        frames["security_master"]["security_id"].astype(str).eq(SIVB_ID),
        "SIVB security master state",
    )
    history_symbols = set(
        frames["symbol_history"]
        .loc[frames["symbol_history"]["security_id"].astype(str).eq(SIVB_ID), "symbol"]
        .astype(str)
        .str.upper()
    )
    sivb_prices = frames["daily_price_raw"].loc[
        frames["daily_price_raw"]["security_id"].astype(str).eq(SIVB_ID)
    ]
    last_price = _date(pd.to_datetime(sivb_prices["session"]).max())
    old = (
        AVP_OLD_EVENT_ID in actions
        and AVP_NEW_EVENT_ID not in actions
        and SIVB_EVENT_ID in actions
        and SIVB_TICKER_EVENT_ID not in actions
        and SIVB_MARKET_EXIT_EVENT_ID not in actions
        and AVP_CANDIDATE_ID in candidates
        and SIVB_CANDIDATE_ID in candidates
        and SIVBQ_CANDIDATE_ID not in candidates
        and _text(master.get("primary_symbol")).upper() == SIVB_SYMBOL
        and "SIVBQ" not in history_symbols
        and last_price == SIVB_LAST_SESSION
        and source_archive_rows_to_add == 3
    )
    repaired = (
        AVP_OLD_EVENT_ID not in actions
        and AVP_NEW_EVENT_ID in actions
        and SIVB_EVENT_ID not in actions
        and SIVB_TICKER_EVENT_ID in actions
        and SIVB_MARKET_EXIT_EVENT_ID in actions
        and AVP_CANDIDATE_ID in candidates
        and SIVB_CANDIDATE_ID not in candidates
        and SIVBQ_CANDIDATE_ID in candidates
        and _text(master.get("primary_symbol")).upper() == "SIVBQ"
        and "SIVBQ" in history_symbols
        and last_price == SIVBQ_LAST_SESSION
        and source_archive_rows_to_add == 0
    )
    if old == repaired:
        raise RuntimeError(
            "SIVB/AVP repair is partially applied or has an unknown state."
        )
    return "old" if old else "repaired"


def _verify_repaired_rows(
    frames: Mapping[str, pd.DataFrame],
    transition: Mapping[str, Any],
    release: DataRelease,
    repository: LocalDatasetRepository,
) -> None:
    actions = frames["corporate_actions"]
    avp = _one_row(
        actions,
        actions["event_id"].astype(str).eq(AVP_NEW_EVENT_ID),
        "repaired AVP action",
    )
    if not (
        _text(avp.get("security_id")) == AVP_ID
        and _text(avp.get("action_type")) == "stock_merger"
        and _date(avp.get("effective_date")) == AVP_MARKET_TRANSITION
        and _date(avp.get("ex_date")) == AVP_MARKET_TRANSITION
        and _date(avp.get("announcement_date")) == AVP_LEGAL_COMPLETION
        and math.isclose(_number(avp.get("ratio")) or 0.0, AVP_RATIO, abs_tol=1e-12)
        and _text(avp.get("new_security_id")) == NTCO_ID
        and _text(avp.get("new_symbol")) == NTCO_SYMBOL
        and _text(avp.get("metadata")) == _avp_metadata()
    ):
        raise EvidenceError("Repaired AVP action changed.")
    ticker = _one_row(
        actions,
        actions["event_id"].astype(str).eq(SIVB_TICKER_EVENT_ID),
        "repaired SIVB ticker action",
    )
    if not (
        _text(ticker.get("security_id")) == SIVB_ID
        and _text(ticker.get("action_type")) == "ticker_change"
        and _date(ticker.get("effective_date")) == SIVBQ_FIRST_SESSION
        and _date(ticker.get("ex_date")) == SIVBQ_FIRST_SESSION
        and _text(ticker.get("new_security_id")) == SIVB_ID
        and _text(ticker.get("new_symbol")) == "SIVBQ"
        and _text(ticker.get("source_hash")) == SIVB_OCC_MEMO_COLLECTED_HASH
        and _text(ticker.get("metadata")) == _sivb_ticker_metadata()
    ):
        raise EvidenceError("Repaired SIVB ticker action changed.")
    exit_action = _one_row(
        actions,
        actions["event_id"].astype(str).eq(SIVB_MARKET_EXIT_EVENT_ID),
        "repaired SIVB terminal action",
    )
    if not (
        _text(exit_action.get("security_id")) == SIVB_ID
        and _text(exit_action.get("action_type")) == "delisting"
        and _date(exit_action.get("effective_date")) == SIVB_MARKET_TERMINATION
        and _date(exit_action.get("ex_date")) == SIVB_MARKET_TERMINATION
        and math.isclose(_number(exit_action.get("cash_amount")) or 0.0, 0.0, abs_tol=0.0)
        and _text(exit_action.get("metadata")) == _sivb_exit_metadata()
    ):
        raise EvidenceError("Repaired SIVB terminal action changed.")

    resolutions = frames["lifecycle_resolutions"]
    expected_resolutions = {
        AVP_ID: (AVP_CANDIDATE_ID, AVP_NEW_EVENT_ID, AVP_SYMBOL, AVP_LAST_SESSION),
        SIVB_ID: (
            SIVBQ_CANDIDATE_ID,
            SIVB_MARKET_EXIT_EVENT_ID,
            "SIVBQ",
            SIVBQ_LAST_SESSION,
        ),
    }
    for security_id, (candidate_id, event_id, symbol, last_price) in expected_resolutions.items():
        row = _one_row(
            resolutions,
            resolutions["security_id"].astype(str).eq(security_id),
            f"repaired resolution {security_id}",
        )
        if not (
            _text(row.get("candidate_id")) == candidate_id
            and _text(row.get("event_id")) == event_id
            and _text(row.get("symbol")).upper() == symbol
            and _date(row.get("last_price_date")) == last_price
            and _text(row.get("resolution")) == "applied"
        ):
            raise EvidenceError(f"Repaired resolution changed: {security_id}.")

    master = _one_row(
        frames["security_master"],
        frames["security_master"]["security_id"].astype(str).eq(SIVB_ID),
        "repaired SIVBQ master",
    )
    if not (
        _text(master.get("primary_symbol")).upper() == "SIVBQ"
        and _text(master.get("provider_symbol")).upper() == SIVBQ_PROVIDER_SYMBOL
        and _text(master.get("exchange")).upper() == "PINK"
        and _date(master.get("active_to")) == SIVBQ_LAST_SESSION
        and _text(master.get("source_hash")) == SIVB_OCC_MEMO_COLLECTED_HASH
    ):
        raise EvidenceError("Repaired SIVBQ security master changed.")
    history = frames["symbol_history"]
    old_alias = _one_row(
        history,
        history["security_id"].astype(str).eq(SIVB_ID)
        & history["symbol"].astype(str).str.upper().eq(SIVB_SYMBOL),
        "repaired SIVB alias",
    )
    new_alias = _one_row(
        history,
        history["security_id"].astype(str).eq(SIVB_ID)
        & history["symbol"].astype(str).str.upper().eq("SIVBQ"),
        "repaired SIVBQ alias",
    )
    if not (
        _date(old_alias.get("effective_to")) == "2023-03-27"
        and _date(new_alias.get("effective_from")) == SIVBQ_FIRST_SESSION
        and _date(new_alias.get("effective_to")) == SIVBQ_LAST_SESSION
        and _text(new_alias.get("source_hash")) == SIVB_OCC_MEMO_COLLECTED_HASH
    ):
        raise EvidenceError("Repaired SIVB/SIVBQ symbol history changed.")

    prices = frames["daily_price_raw"]
    sessions = pd.to_datetime(prices["session"], errors="coerce").dt.date.astype(str)
    otc_prices = prices.loc[
        prices["security_id"].astype(str).eq(SIVB_ID)
        & sessions.ge(SIVBQ_FIRST_SESSION)
    ].copy()
    raw_otc = [
        row
        for row in transition["eod_rows"]
        if _date(row.get("date")) not in SIVBQ_NON_XNYS_EXCLUSIONS
    ]
    _verify_parquet_price_series(
        otc_prices,
        security_id=SIVB_ID,
        records=raw_otc,
        expected_first=SIVBQ_FIRST_SESSION,
        expected_last=SIVBQ_LAST_SESSION,
        expected_source_hash=SIVBQ_EOD_COLLECTED_HASH,
        expected_retrieved_at=SIVB_TRANSITION_RETRIEVED_AT,
    )
    if set(otc_prices["source_url"].astype(str)) != {SIVBQ_SAFE_URL}:
        raise EvidenceError("Repaired SIVBQ price source URL changed.")

    factors = frames["adjustment_factors"]
    expected_lineage = _factor_source_version(
        release.dataset_versions["daily_price_raw"],
        release.dataset_versions["corporate_actions"],
    )
    if not (
        len(factors) == len(prices)
        and set(factors["source_version"].astype(str)) == {expected_lineage}
        and set(factors["source_hash"].astype(str)) == {expected_lineage}
        and set(factors["source"].astype(str)) == {"derived"}
        and set(factors["calculated_at"].astype(str)) == {REVIEWED_AT}
        and set(factors["retrieved_at"].astype(str)) == {REVIEWED_AT}
    ):
        raise EvidenceError("Repaired full adjustment-factor lineage changed.")
    _verify_persisted_evidence(
        repository,
        transition,
        completed_session=release.completed_session,
    )


def _avp_metadata() -> str:
    return _canonical_json(
        {
            "legal_completion_date": AVP_LEGAL_COMPLETION,
            "market_transition_session": AVP_MARKET_TRANSITION,
            "date_relation": "first_xnys_session_after_terminal_close",
            "market_evidence_source_hash": AVP_OFFICIAL_HASH,
            "successor_price_source_hash": NTCO_RAW_HASH,
            "original_event_id": AVP_OLD_EVENT_ID,
            "policy": POLICY,
        }
    )


def _rewrite_avp_action(actions: pd.DataFrame) -> pd.DataFrame:
    output = actions.copy(deep=True)
    mask = output["event_id"].astype(str).eq(AVP_OLD_EVENT_ID)
    if int(mask.sum()) != 1:
        raise EvidenceError("Cannot rewrite a non-exact AVP action inventory.")
    index = output.index[mask][0]
    output.at[index, "event_id"] = AVP_NEW_EVENT_ID
    output.at[index, "effective_date"] = AVP_MARKET_TRANSITION
    output.at[index, "ex_date"] = AVP_MARKET_TRANSITION
    output.at[index, "metadata"] = _avp_metadata()
    return output


def _rewrite_avp_resolution(resolutions: pd.DataFrame) -> pd.DataFrame:
    output = resolutions.copy(deep=True)
    mask = output["candidate_id"].astype(str).eq(AVP_CANDIDATE_ID)
    if int(mask.sum()) != 1:
        raise EvidenceError("Cannot rewrite a non-exact AVP resolution inventory.")
    index = output.index[mask][0]
    output.at[index, "event_id"] = AVP_NEW_EVENT_ID
    output.at[index, "reviewed_by"] = REPAIR_REVIEWER
    output.at[index, "reviewed_at"] = REVIEWED_AT
    output.at[index, "source"] = REPAIR_SOURCE
    output.at[index, "retrieved_at"] = REVIEWED_AT
    return output


def _rewrite_avp_history(history: pd.DataFrame) -> pd.DataFrame:
    output = history.copy(deep=True)
    changes = {
        AVP_ID: (AVP_SYMBOL, "effective_to", AVP_LAST_SESSION),
        NTCO_ID: (NTCO_SYMBOL, "effective_from", AVP_MARKET_TRANSITION),
    }
    for security_id, (symbol, field, value) in changes.items():
        mask = output["security_id"].astype(str).eq(security_id) & output[
            "symbol"
        ].astype(str).str.upper().eq(symbol)
        if int(mask.sum()) != 1:
            raise EvidenceError(f"Cannot rewrite a non-exact {symbol} history inventory.")
        index = output.index[mask][0]
        output.at[index, field] = value
        output.at[index, "source"] = REPAIR_SOURCE
        output.at[index, "source_url"] = AVP_OFFICIAL_URL
        output.at[index, "retrieved_at"] = REVIEWED_AT
        output.at[index, "source_hash"] = AVP_OFFICIAL_HASH
    return output


def _sivb_ticker_metadata() -> str:
    return _canonical_json(
        {
            "memo_number": "52179",
            "cusip": "78486Q101",
            "nasdaq_halt_date": SIVB_EXPECTED_MARKET_SESSION,
            "nasdaq_suspension_date": SIVBQ_FIRST_SESSION,
            "otc_open_date": SIVBQ_FIRST_SESSION,
            "same_common_share_identity": True,
            "ratio": 1.0,
            "sec_market_source_hash": SIVB_SEC_MARKET_COLLECTED_HASH,
            "occ_reviewed_extraction_hash": SIVB_OCC_MEMO_COLLECTED_HASH,
            "policy": POLICY,
        }
    )


def _sivb_exit_metadata() -> str:
    return _canonical_json(
        {
            "legal_cancellation_date": SIVB_LEGAL_CANCELLATION,
            "last_observed_otc_price_session": SIVBQ_LAST_SESSION,
            "engine_terminal_session": SIVB_MARKET_TERMINATION,
            "date_relation": "first_xnys_session_after_last_otc_close",
            "legal_zero_distribution_preserved": True,
            "original_event_id": SIVB_EVENT_ID,
            "otc_price_source_hash": SIVBQ_EOD_COLLECTED_HASH,
            "policy": POLICY,
        }
    )


def _rewrite_sivb_actions(actions: pd.DataFrame) -> pd.DataFrame:
    output = actions.copy(deep=True)
    old_mask = output["event_id"].astype(str).eq(SIVB_EVENT_ID)
    if int(old_mask.sum()) != 1:
        raise EvidenceError("Cannot rewrite a non-exact SIVB legal action inventory.")
    if output["event_id"].astype(str).isin(
        {SIVB_TICKER_EVENT_ID, SIVB_MARKET_EXIT_EVENT_ID}
    ).any():
        raise EvidenceError("SIVB reviewed transition actions already exist unexpectedly.")
    index = output.index[old_mask][0]
    output.at[index, "event_id"] = SIVB_MARKET_EXIT_EVENT_ID
    output.at[index, "effective_date"] = SIVB_MARKET_TERMINATION
    output.at[index, "ex_date"] = SIVB_MARKET_TERMINATION
    output.at[index, "metadata"] = _sivb_exit_metadata()

    ticker = {column: np.nan for column in output.columns}
    ticker.update(
        {
            "event_id": SIVB_TICKER_EVENT_ID,
            "security_id": SIVB_ID,
            "action_type": "ticker_change",
            "effective_date": SIVBQ_FIRST_SESSION,
            "ex_date": SIVBQ_FIRST_SESSION,
            "announcement_date": "2023-03-27",
            "record_date": "",
            "payment_date": "",
            "cash_amount": np.nan,
            "ratio": np.nan,
            "currency": "USD",
            "new_security_id": SIVB_ID,
            "new_symbol": "SIVBQ",
            "official": True,
            "source_url": SIVB_OCC_MEMO_URL,
            "source_kind": "clearing_notice_reviewed_extraction",
            "source": "occ_reviewed_memo_extraction",
            "retrieved_at": SIVB_TRANSITION_RETRIEVED_AT,
            "source_hash": SIVB_OCC_MEMO_COLLECTED_HASH,
            "metadata": _sivb_ticker_metadata(),
        }
    )
    return pd.concat([output, pd.DataFrame([ticker], columns=output.columns)], ignore_index=True)


def _rewrite_sivb_resolution(resolutions: pd.DataFrame) -> pd.DataFrame:
    output = resolutions.copy(deep=True)
    mask = output["candidate_id"].astype(str).eq(SIVB_CANDIDATE_ID)
    if int(mask.sum()) != 1:
        raise EvidenceError("Cannot rewrite a non-exact SIVB resolution inventory.")
    if output["candidate_id"].astype(str).eq(SIVBQ_CANDIDATE_ID).any():
        raise EvidenceError("SIVBQ reviewed resolution already exists unexpectedly.")
    index = output.index[mask][0]
    output.at[index, "candidate_id"] = SIVBQ_CANDIDATE_ID
    output.at[index, "symbol"] = "SIVBQ"
    output.at[index, "last_price_date"] = SIVBQ_LAST_SESSION
    output.at[index, "event_id"] = SIVB_MARKET_EXIT_EVENT_ID
    output.at[index, "reviewed_by"] = REPAIR_REVIEWER
    output.at[index, "reviewed_at"] = REVIEWED_AT
    output.at[index, "source"] = REPAIR_SOURCE
    output.at[index, "retrieved_at"] = REVIEWED_AT
    return output


def _rewrite_sivb_master(master: pd.DataFrame) -> pd.DataFrame:
    output = master.copy(deep=True)
    mask = output["security_id"].astype(str).eq(SIVB_ID)
    if int(mask.sum()) != 1:
        raise EvidenceError("Cannot rewrite a non-exact SIVB security master.")
    index = output.index[mask][0]
    changes = {
        "exchange": "PINK",
        "active_to": SIVBQ_LAST_SESSION,
        "source": "occ_reviewed_memo_extraction",
        "source_url": SIVB_OCC_MEMO_URL,
        "retrieved_at": SIVB_TRANSITION_RETRIEVED_AT,
        "source_hash": SIVB_OCC_MEMO_COLLECTED_HASH,
        "primary_symbol": "SIVBQ",
        "provider_symbol": SIVBQ_PROVIDER_SYMBOL,
        "action_provider_symbol": SIVBQ_PROVIDER_SYMBOL,
    }
    for field, value in changes.items():
        output.at[index, field] = value
    return output


def _rewrite_sivb_history(history: pd.DataFrame) -> pd.DataFrame:
    output = history.copy(deep=True)
    old_mask = output["security_id"].astype(str).eq(SIVB_ID) & output[
        "symbol"
    ].astype(str).str.upper().eq(SIVB_SYMBOL)
    if int(old_mask.sum()) != 1:
        raise EvidenceError("Cannot rewrite a non-exact SIVB symbol history.")
    if (
        output["security_id"].astype(str).eq(SIVB_ID)
        & output["symbol"].astype(str).str.upper().eq("SIVBQ")
    ).any():
        raise EvidenceError("SIVBQ symbol history already exists unexpectedly.")
    index = output.index[old_mask][0]
    output.at[index, "effective_to"] = "2023-03-27"
    output.at[index, "source"] = "occ_reviewed_memo_extraction"
    output.at[index, "source_url"] = SIVB_OCC_MEMO_URL
    output.at[index, "retrieved_at"] = SIVB_TRANSITION_RETRIEVED_AT
    output.at[index, "source_hash"] = SIVB_OCC_MEMO_COLLECTED_HASH
    successor = {column: np.nan for column in output.columns}
    successor.update(
        {
            "security_id": SIVB_ID,
            "symbol": "SIVBQ",
            "exchange": "PINK",
            "effective_from": SIVBQ_FIRST_SESSION,
            "effective_to": SIVBQ_LAST_SESSION,
            "source": "occ_reviewed_memo_extraction",
            "retrieved_at": SIVB_TRANSITION_RETRIEVED_AT,
            "source_hash": SIVB_OCC_MEMO_COLLECTED_HASH,
            "source_url": SIVB_OCC_MEMO_URL,
        }
    )
    return pd.concat(
        [output, pd.DataFrame([successor], columns=output.columns)],
        ignore_index=True,
    )


def _sivbq_price_frame(rows: list[Mapping[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        if _date(row.get("date")) in SIVBQ_NON_XNYS_EXCLUSIONS:
            continue
        records.append(
            {
                "security_id": SIVB_ID,
                "session": _date(row.get("date")),
                "open": _number(row.get("open")),
                "high": _number(row.get("high")),
                "low": _number(row.get("low")),
                "close": _number(row.get("close")),
                "volume": _number(row.get("volume")),
                "currency": "USD",
                "source": "eodhd_eod",
                "retrieved_at": SIVB_TRANSITION_RETRIEVED_AT,
                "source_hash": SIVBQ_EOD_COLLECTED_HASH,
                "source_url": SIVBQ_SAFE_URL,
            }
        )
    return pd.DataFrame(records)


def _append_sivbq_prices(
    prices: pd.DataFrame,
    rows: list[Mapping[str, Any]],
) -> pd.DataFrame:
    output = prices.copy(deep=True)
    existing = output.loc[output["security_id"].astype(str).eq(SIVB_ID)].copy()
    existing_sessions = set(
        pd.to_datetime(existing["session"], errors="coerce").dt.date.astype(str)
    )
    new = _sivbq_price_frame(rows).reindex(columns=output.columns)
    new_sessions = set(new["session"].astype(str))
    if existing_sessions & new_sessions:
        raise EvidenceError("SIVBQ price cache overlaps existing SIVB sessions.")
    if len(new) != SIVBQ_STORED_ROWS or len(new_sessions) != SIVBQ_STORED_ROWS:
        raise EvidenceError("SIVBQ candidate price inventory changed.")
    return pd.concat([output, new], ignore_index=True)


def _rebuild_sivb_factors(
    prices: pd.DataFrame,
    old_factors: pd.DataFrame,
    candidate_actions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    price_rows = prices.loc[prices["security_id"].astype(str).eq(SIVB_ID)].copy()
    action_rows = candidate_actions.loc[
        candidate_actions["security_id"].astype(str).eq(SIVB_ID)
    ].copy()
    old_rows = old_factors.loc[
        old_factors["security_id"].astype(str).eq(SIVB_ID)
    ].copy()
    source_version = "sivb-otc-transition-" + SIVBQ_EOD_COLLECTED_HASH
    rebuilt = build_adjustment_factors(
        price_rows,
        action_rows,
        source_version=source_version,
    )
    rebuilt["calculated_at"] = REVIEWED_AT
    rebuilt["retrieved_at"] = REVIEWED_AT
    rebuilt["source_version"] = source_version
    rebuilt["source_hash"] = source_version
    rebuilt = rebuilt.reindex(columns=old_factors.columns)
    for frame in (old_rows, rebuilt):
        frame["session"] = pd.to_datetime(frame["session"], errors="coerce").dt.normalize()
    old_economic = old_rows[
        ["session", "split_factor", "total_return_factor"]
    ].sort_values("session")
    retained = rebuilt.loc[
        rebuilt["session"].isin(set(old_economic["session"]))
    ][["session", "split_factor", "total_return_factor"]].sort_values("session")
    old_economic = old_economic.reset_index(drop=True)
    retained = retained.reset_index(drop=True)
    if not old_economic["session"].equals(retained["session"]):
        raise EvidenceError("SIVB retained adjustment-factor keys changed.")
    changed = ~(
        np.isclose(
            old_economic["split_factor"],
            retained["split_factor"],
            rtol=0,
            atol=1e-12,
        )
        & np.isclose(
            old_economic["total_return_factor"],
            retained["total_return_factor"],
            rtol=0,
            atol=1e-12,
        )
    )
    if int(changed.sum()):
        raise EvidenceError("SIVB retained adjustment-factor economics changed.")
    if len(rebuilt) != len(old_rows) + SIVBQ_STORED_ROWS:
        raise EvidenceError("SIVB rebuilt adjustment-factor inventory changed.")
    other = old_factors.loc[
        ~old_factors["security_id"].astype(str).eq(SIVB_ID)
    ].copy()
    output = pd.concat([other, rebuilt], ignore_index=True)
    return output, {
        "old_rows_checked": len(old_rows),
        "new_rows_added": SIVBQ_STORED_ROWS,
        "rebuilt_rows": len(rebuilt),
        "retained_economic_rows_changed": int(changed.sum()),
    }


def _factor_source_version(price_version: str, action_version: str) -> str:
    if not price_version or not action_version:
        raise RuntimeError("SIVB/AVP factor lineage requires exact input versions.")
    return f"{price_version}+{action_version}"


def _rebuild_all_factors(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    old_factors: pd.DataFrame,
    *,
    source_version: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rebuilt = build_adjustment_factors(
        prices,
        actions,
        source_version=source_version,
    ).reindex(columns=old_factors.columns)
    rebuilt["source_version"] = source_version
    rebuilt["calculated_at"] = REVIEWED_AT
    rebuilt["source"] = "derived"
    rebuilt["retrieved_at"] = REVIEWED_AT
    rebuilt["source_hash"] = source_version

    old = old_factors.copy()
    for frame in (old, rebuilt):
        frame["security_id"] = frame["security_id"].astype(str)
        frame["session"] = pd.to_datetime(
            frame["session"], errors="coerce"
        ).dt.normalize()
    old_keys = set(old[["security_id", "session"]].itertuples(index=False, name=None))
    new_keys = set(
        rebuilt[["security_id", "session"]].itertuples(index=False, name=None)
    )
    price_keys = set(
        zip(
            prices["security_id"].astype(str),
            pd.to_datetime(prices["session"], errors="coerce").dt.normalize(),
        )
    )
    if new_keys != price_keys or not old_keys.issubset(new_keys):
        raise EvidenceError("Full factor keys do not exactly cover candidate prices.")
    columns = ["security_id", "session", "split_factor", "total_return_factor"]
    left = old[columns].sort_values(["security_id", "session"]).reset_index(drop=True)
    retained = rebuilt.loc[
        [
            (str(row.security_id), pd.Timestamp(row.session).normalize()) in old_keys
            for row in rebuilt.itertuples(index=False)
        ],
        columns,
    ].sort_values(["security_id", "session"]).reset_index(drop=True)
    if not left[["security_id", "session"]].equals(
        retained[["security_id", "session"]]
    ):
        raise EvidenceError("Retained full factor key order changed.")
    changed = ~(
        np.isclose(
            left["split_factor"], retained["split_factor"], rtol=0, atol=1e-12
        )
        & np.isclose(
            left["total_return_factor"],
            retained["total_return_factor"],
            rtol=0,
            atol=1e-12,
        )
    )
    if int(changed.sum()):
        raise EvidenceError("Full factor lineage rebind changed economic values.")
    added = len(new_keys - old_keys)
    if added != SIVBQ_STORED_ROWS:
        raise EvidenceError("Full factor lineage added an unexpected key inventory.")
    return rebuilt, {
        "retained_rows_checked": len(old),
        "new_rows_added": added,
        "rebuilt_rows": len(rebuilt),
        "retained_economic_rows_changed": int(changed.sum()),
        "provenance_rows_rebound": len(rebuilt),
    }


def _factor_economic_changes(
    prices: pd.DataFrame,
    old_factors: pd.DataFrame,
    candidate_actions: pd.DataFrame,
) -> tuple[int, int]:
    price_rows = prices.loc[prices["security_id"].astype(str).eq(AVP_ID)].copy()
    action_rows = candidate_actions.loc[
        candidate_actions["security_id"].astype(str).eq(AVP_ID)
    ].copy()
    current = old_factors.loc[old_factors["security_id"].astype(str).eq(AVP_ID)].copy()
    rebuilt = build_adjustment_factors(
        price_rows, action_rows, source_version="offline-avp-transition-plan"
    )
    columns = ["security_id", "session", "split_factor", "total_return_factor"]
    for frame in (current, rebuilt):
        frame["session"] = pd.to_datetime(frame["session"], errors="coerce").dt.normalize()
    left = current[columns].sort_values(["security_id", "session"]).reset_index(drop=True)
    right = rebuilt[columns].sort_values(["security_id", "session"]).reset_index(drop=True)
    if not left[["security_id", "session"]].equals(right[["security_id", "session"]]):
        raise EvidenceError("AVP adjustment-factor key inventory changed in the plan.")
    changed = ~(
        np.isclose(left["split_factor"], right["split_factor"], rtol=0, atol=1e-12)
        & np.isclose(
            left["total_return_factor"],
            right["total_return_factor"],
            rtol=0,
            atol=1e-12,
        )
    )
    return len(left), int(changed.sum())


def _terminal_report(frames: Mapping[str, pd.DataFrame], release_version: str):
    return audit_terminal_transitions(
        corporate_actions=frames["corporate_actions"],
        lifecycle_resolutions=frames["lifecycle_resolutions"],
        daily_price_raw=frames["daily_price_raw"],
        index_constituent_anchors=frames["index_constituent_anchors"],
        index_membership_events=frames["index_membership_events"],
        symbol_history=frames["symbol_history"],
        security_master=frames["security_master"],
        release_version=release_version,
    )


def _target_issues(report: Any) -> list[dict[str, Any]]:
    target_ids = {SIVB_ID, AVP_ID}
    return [issue.to_dict() for issue in report.issues if issue.security_id in target_ids]


def _row_diff_count(old: pd.DataFrame, new: pd.DataFrame) -> int:
    if list(old.columns) != list(new.columns) or len(old) != len(new):
        raise AssertionError("Planner comparison requires identical table shape.")
    equal = old.eq(new) | (old.isna() & new.isna())
    return int((~equal.all(axis=1)).sum())


def _pointer_etags(
    repository: LocalDatasetRepository, release: DataRelease
) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for dataset in REQUIRED_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != release.dataset_versions.get(dataset):
            raise RuntimeError(f"Release/current pointer mismatch: {dataset}.")
        output[dataset] = etag
    return output


def _new_versions(release: DataRelease) -> dict[str, str]:
    token = uuid.uuid4().hex
    session = release.completed_session.replace("-", "")
    return {
        dataset: f"sivb-avp-transition-{session}-{token}-{dataset}"
        for dataset in WRITE_DATASETS
    }


def _prepare_plan_legacy(repository: LocalDatasetRepository) -> PreparedPlan:
    """Build and validate the read-only plan against the current release."""

    _verify_static_contract()
    release, _ = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    missing = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks required datasets: " + ", ".join(missing))
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in REQUIRED_DATASETS
    }
    _verify_current_rows(frames)
    avp_official, sivb_official = _verify_official_evidence(
        repository,
        frames["source_archive"],
        completed_session=release.completed_session,
    )
    price_evidence = _verify_price_evidence(
        repository,
        frames["source_archive"],
        frames["daily_price_raw"],
        completed_session=release.completed_session,
    )
    transition = _verify_reviewed_sivb_transition_cache(repository.root)

    candidate = {dataset: frame.copy(deep=True) for dataset, frame in frames.items()}
    candidate["corporate_actions"] = _rewrite_sivb_actions(
        _rewrite_avp_action(frames["corporate_actions"])
    )
    candidate["lifecycle_resolutions"] = _rewrite_sivb_resolution(
        _rewrite_avp_resolution(frames["lifecycle_resolutions"])
    )
    candidate["symbol_history"] = _rewrite_sivb_history(
        _rewrite_avp_history(frames["symbol_history"])
    )
    candidate["security_master"] = _rewrite_sivb_master(
        frames["security_master"]
    )
    candidate["daily_price_raw"] = _append_sivbq_prices(
        frames["daily_price_raw"], transition["eod_rows"]
    )
    candidate["adjustment_factors"], sivb_factor_validation = (
        _rebuild_sivb_factors(
            candidate["daily_price_raw"],
            frames["adjustment_factors"],
            candidate["corporate_actions"],
        )
    )

    for dataset in (
        "corporate_actions",
        "lifecycle_resolutions",
        "symbol_history",
        "security_master",
        "daily_price_raw",
        "adjustment_factors",
    ):
        validate_dataset(
            dataset,
            candidate[dataset],
            completed_session=release.completed_session,
            incomplete_action_policy="block",
        ).raise_for_errors()

    factor_rows, factor_changes = _factor_economic_changes(
        frames["daily_price_raw"],
        frames["adjustment_factors"],
        candidate["corporate_actions"],
    )
    if factor_changes:
        raise EvidenceError("AVP transition plan changed adjustment-factor economics.")

    before_report = _terminal_report(frames, release.version)
    after_report = _terminal_report(candidate, release.version + ":offline-plan")
    before_target = _target_issues(before_report)
    after_target = _target_issues(after_report)
    before_by_symbol = {
        symbol: [issue for issue in before_target if issue["symbol"] == symbol]
        for symbol in (SIVB_SYMBOL, AVP_SYMBOL)
    }
    after_by_symbol = {
        symbol: [issue for issue in after_target if issue["symbol"] == symbol]
        for symbol in (SIVB_SYMBOL, AVP_SYMBOL)
    }
    if [issue["code"] for issue in before_by_symbol[AVP_SYMBOL]] != [
        "successor_not_ready_on_transition"
    ] or after_by_symbol[AVP_SYMBOL]:
        raise EvidenceError("AVP terminal-readiness issue did not resolve exactly.")
    if [issue["code"] for issue in before_by_symbol[SIVB_SYMBOL]] != [
        "terminal_action_after_expected_session"
    ]:
        raise EvidenceError("SIVB pre-repair terminal issue changed unexpectedly.")
    if after_target:
        raise EvidenceError(
            "SIVB/AVP target terminal issues remain after the offline plan: "
            + _canonical_json(after_target)
        )

    row_deltas = {
        dataset: len(candidate[dataset]) - len(frames[dataset])
        for dataset in REQUIRED_DATASETS
    }
    expected_deltas = {
        "corporate_actions": 1,
        "lifecycle_resolutions": 0,
        "security_master": 0,
        "symbol_history": 1,
        "daily_price_raw": SIVBQ_STORED_ROWS,
        "adjustment_factors": SIVBQ_STORED_ROWS,
        "source_archive": 0,
        "index_constituent_anchors": 0,
        "index_membership_events": 0,
    }
    if row_deltas != expected_deltas:
        raise AssertionError(f"SIVB/AVP plan row scope changed: {row_deltas}")
    if not frames["source_archive"].equals(candidate["source_archive"]):
        raise AssertionError("Offline planner mutated the release source archive.")

    avp_mutations = [
        {
            "dataset": "corporate_actions",
            "row_selector": {"event_id": AVP_OLD_EVENT_ID},
            "changes": {
                "event_id": AVP_NEW_EVENT_ID,
                "effective_date": AVP_MARKET_TRANSITION,
                "ex_date": AVP_MARKET_TRANSITION,
                "metadata": json.loads(_avp_metadata()),
            },
        },
        {
            "dataset": "lifecycle_resolutions",
            "row_selector": {"candidate_id": AVP_CANDIDATE_ID},
            "changes": {
                "event_id": AVP_NEW_EVENT_ID,
                "reviewed_by": REPAIR_REVIEWER,
                "reviewed_at": REVIEWED_AT,
                "source": REPAIR_SOURCE,
                "retrieved_at": REVIEWED_AT,
            },
        },
        {
            "dataset": "symbol_history",
            "row_selector": {"security_id": AVP_ID, "symbol": AVP_SYMBOL},
            "changes": {"effective_to": AVP_LAST_SESSION},
        },
        {
            "dataset": "symbol_history",
            "row_selector": {"security_id": NTCO_ID, "symbol": NTCO_SYMBOL},
            "changes": {"effective_from": AVP_MARKET_TRANSITION},
        },
    ]
    sivb_mutations = [
        {
            "dataset": "corporate_actions",
            "operation": "add",
            "event_id": SIVB_TICKER_EVENT_ID,
            "action_type": "ticker_change",
            "effective_date": SIVBQ_FIRST_SESSION,
            "old_symbol": SIVB_SYMBOL,
            "new_symbol": "SIVBQ",
            "new_security_id": SIVB_ID,
            "ratio": 1.0,
        },
        {
            "dataset": "corporate_actions",
            "row_selector": {"event_id": SIVB_EVENT_ID},
            "changes": {
                "event_id": SIVB_MARKET_EXIT_EVENT_ID,
                "effective_date": SIVB_MARKET_TERMINATION,
                "ex_date": SIVB_MARKET_TERMINATION,
                "cash_amount": 0.0,
                "metadata": json.loads(_sivb_exit_metadata()),
            },
        },
        {
            "dataset": "lifecycle_resolutions",
            "row_selector": {"candidate_id": SIVB_CANDIDATE_ID},
            "changes": {
                "candidate_id": SIVBQ_CANDIDATE_ID,
                "symbol": "SIVBQ",
                "last_price_date": SIVBQ_LAST_SESSION,
                "event_id": SIVB_MARKET_EXIT_EVENT_ID,
            },
        },
        {
            "dataset": "security_master",
            "row_selector": {"security_id": SIVB_ID},
            "changes": {
                "primary_symbol": "SIVBQ",
                "provider_symbol": SIVBQ_PROVIDER_SYMBOL,
                "exchange": "PINK",
                "active_to": SIVBQ_LAST_SESSION,
            },
        },
        {
            "dataset": "symbol_history",
            "operation": "close_and_add_alias",
            "old_symbol_effective_to": "2023-03-27",
            "new_symbol": "SIVBQ",
            "new_symbol_effective_from": SIVBQ_FIRST_SESSION,
            "new_symbol_effective_to": SIVBQ_LAST_SESSION,
        },
        {
            "dataset": "daily_price_raw",
            "operation": "append",
            "raw_rows": SIVBQ_EOD_ROWS,
            "stored_xnys_rows": SIVBQ_STORED_ROWS,
            "exact_hash_non_xnys_exclusions": list(SIVBQ_NON_XNYS_EXCLUSIONS),
            "first_session": SIVBQ_FIRST_SESSION,
            "last_session": SIVBQ_LAST_SESSION,
            "security_id": SIVB_ID,
            "source_hash": SIVBQ_EOD_COLLECTED_HASH,
        },
        {
            "dataset": "adjustment_factors",
            "operation": "rebuild_security_lineage",
            **sivb_factor_validation,
        },
    ]
    plan: dict[str, Any] = {
        "schema": SCHEMA,
        "policy": POLICY,
        "status": "ready_offline_plan",
        "base_release_version": release.version,
        "completed_session": release.completed_session,
        "offline_guards": {
            "read_only": True,
            "apply_supported": False,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
            "source_archive_mutated": False,
        },
        "cases": {
            "AVP": {
                "status": "ready_offline_plan",
                "security_id": AVP_ID,
                "successor_security_id": NTCO_ID,
                "legal_completion_date": AVP_LEGAL_COMPLETION,
                "last_source_price_session": AVP_LAST_SESSION,
                "market_transition_session": AVP_MARKET_TRANSITION,
                "successor_first_price_session": AVP_MARKET_TRANSITION,
                "ratio": AVP_RATIO,
                "economics_changed": False,
                "official_evidence": avp_official,
                "price_evidence": {
                    "source": price_evidence["AVP"],
                    "successor": price_evidence["NTCO"],
                },
                "proposed_mutations": avp_mutations,
            },
            "SIVB": {
                "status": "ready_offline_plan",
                "security_id": SIVB_ID,
                "last_nasdaq_price_session_observed": SIVB_LAST_SESSION,
                "official_nasdaq_halt_date": SIVB_EXPECTED_MARKET_SESSION,
                "official_nasdaq_suspension_date": SIVBQ_FIRST_SESSION,
                "otc_symbol": "SIVBQ",
                "otc_first_price_session": SIVBQ_FIRST_SESSION,
                "otc_last_price_session": SIVBQ_LAST_SESSION,
                "market_terminal_session": SIVB_MARKET_TERMINATION,
                "legal_cancellation_date": SIVB_LEGAL_CANCELLATION,
                "legal_zero_distribution_action_preserved": True,
                "zero_cash_moved_to_2023_03_10": False,
                "economics_invented": False,
                "same_security_identity": True,
                "same_security_id": SIVB_ID,
                "ticker_change_ratio": 1.0,
                "halted_nonpriceable_interval": {
                    "from": SIVB_EXPECTED_MARKET_SESSION,
                    "through": "2023-03-27",
                },
                "official_evidence": {
                    "legal_cancellation": sivb_official,
                    "market_halt_and_suspension": transition["evidence"]["sec"],
                    "ticker_and_otc_continuity": transition["evidence"]["occ"],
                    "occ_raw_pdf_archived": False,
                },
                "price_evidence": {
                    "nasdaq": price_evidence["SIVB"],
                    "otc": transition["eod_summary"],
                    "otc_source_hash": SIVBQ_EOD_COLLECTED_HASH,
                },
                "collection_budget_receipt": transition["budget_receipt"],
                "missing_evidence": [],
                "proposed_mutations": sivb_mutations,
            },
        },
        "validation": {
            "terminal_issues_before": before_target,
            "terminal_issues_after_plan": after_target,
            "avp_target_issue_resolved": True,
            "sivb_target_issue_resolved": True,
            "avp_adjustment_factor_rows_checked": factor_rows,
            "avp_adjustment_factor_economic_rows_changed": factor_changes,
            "sivb_adjustment_factors": sivb_factor_validation,
            "factor_lineage_rebuild_required_on_future_apply": True,
            "row_deltas": row_deltas,
            "daily_price_rows_added": SIVBQ_STORED_ROWS,
            "raw_price_rows_reviewed": SIVBQ_EOD_ROWS,
            "exact_hash_non_xnys_exclusions": list(SIVBQ_NON_XNYS_EXCLUSIONS),
            "security_master_rows_modified": 1,
            "source_archive_rows_changed": 0,
            "future_source_archive_rows_required": 3,
            "eodhd_calls_in_offline_plan": 0,
            "r2_accessed": False,
        },
    }
    plan["plan_sha256"] = _canonical_json_sha256(plan)
    return PreparedPlan(
        release_version=release.version,
        frames=candidate,
        plan=plan,
    )


class _CandidateRepository:
    """Read-only repository view with candidate frames over the base release."""

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
        return self.base.read_frame(dataset, version) if version else pd.DataFrame()


def _validation_issue_fingerprint(report: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            _canonical_json(
                {
                    "code": issue.code,
                    "message": issue.message,
                    "severity": issue.severity,
                    "row_count": issue.row_count,
                    "fingerprints": list(issue.fingerprints),
                }
            )
            for issue in report.issues
        )
    )


def _non_target_terminal_fingerprint(report: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            _canonical_json(issue.to_dict())
            for issue in report.issues
            if issue.security_id not in {SIVB_ID, AVP_ID}
        )
    )


def prepare_plan(repository: LocalDatasetRepository) -> PreparedPlan:
    """Build and validate an offline, replay-safe plan against current state."""

    _verify_static_contract()
    _verify_code_pins()
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    missing = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError(
            "Current release lacks required datasets: " + ", ".join(missing)
        )
    pointer_etags = _pointer_etags(repository, release)
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in REQUIRED_DATASETS
    }
    transition = _verify_reviewed_sivb_transition_cache(repository.root)
    rewritten_archive, archive_additions = _rewrite_source_archive(
        frames["source_archive"], completed_session=release.completed_session
    )
    state = _repair_state(frames, source_archive_rows_to_add=archive_additions)
    avp_official, sivb_official = _verify_official_evidence(
        repository,
        frames["source_archive"],
        completed_session=release.completed_session,
    )
    price_evidence = _verify_price_evidence(
        repository,
        frames["source_archive"],
        frames["daily_price_raw"],
        completed_session=release.completed_session,
        sivb_repaired=state == "repaired",
    )

    candidate = {dataset: frame.copy(deep=True) for dataset, frame in frames.items()}
    if state == "old":
        _verify_current_rows(frames)
        planned_versions = _new_versions(release)
        candidate["corporate_actions"] = _rewrite_sivb_actions(
            _rewrite_avp_action(frames["corporate_actions"])
        )
        candidate["lifecycle_resolutions"] = _rewrite_sivb_resolution(
            _rewrite_avp_resolution(frames["lifecycle_resolutions"])
        )
        candidate["symbol_history"] = _rewrite_sivb_history(
            _rewrite_avp_history(frames["symbol_history"])
        )
        candidate["security_master"] = _rewrite_sivb_master(
            frames["security_master"]
        )
        candidate["daily_price_raw"] = _append_sivbq_prices(
            frames["daily_price_raw"], transition["eod_rows"]
        )
        candidate["source_archive"] = rewritten_archive
        factor_lineage = _factor_source_version(
            planned_versions["daily_price_raw"],
            planned_versions["corporate_actions"],
        )
        candidate["adjustment_factors"], factor_validation = _rebuild_all_factors(
            candidate["daily_price_raw"],
            candidate["corporate_actions"],
            frames["adjustment_factors"],
            source_version=factor_lineage,
        )
        status = "ready_offline_plan"
    else:
        planned_versions = {}
        _verify_repaired_rows(frames, transition, release, repository)
        factor_lineage = _factor_source_version(
            release.dataset_versions["daily_price_raw"],
            release.dataset_versions["corporate_actions"],
        )
        factor_validation = {
            "retained_rows_checked": len(frames["adjustment_factors"]),
            "new_rows_added": 0,
            "rebuilt_rows": len(frames["adjustment_factors"]),
            "retained_economic_rows_changed": 0,
            "provenance_rows_rebound": 0,
        }
        status = "already_repaired"

    for dataset in WRITE_DATASETS:
        validate_dataset(
            dataset,
            candidate[dataset],
            completed_session=release.completed_session,
            incomplete_action_policy="block",
        ).raise_for_errors()

    before_snapshot = validate_repository_snapshot(
        _CandidateRepository(repository, release.dataset_versions, frames)
    )
    after_snapshot = validate_repository_snapshot(
        _CandidateRepository(repository, release.dataset_versions, candidate)
    )
    if _validation_issue_fingerprint(before_snapshot) != _validation_issue_fingerprint(
        after_snapshot
    ):
        raise EvidenceError(
            "SIVB/AVP repair changed unrelated snapshot validation issues."
        )

    factor_rows, factor_changes = _factor_economic_changes(
        frames["daily_price_raw"],
        frames["adjustment_factors"],
        candidate["corporate_actions"],
    )
    if factor_changes:
        raise EvidenceError("AVP transition plan changed adjustment-factor economics.")

    before_report = _terminal_report(frames, release.version)
    after_report = _terminal_report(candidate, release.version + ":offline-plan")
    before_target = _target_issues(before_report)
    after_target = _target_issues(after_report)
    if state == "old":
        before_by_symbol = {
            symbol: [issue for issue in before_target if issue["symbol"] == symbol]
            for symbol in (SIVB_SYMBOL, AVP_SYMBOL)
        }
        if [issue["code"] for issue in before_by_symbol[AVP_SYMBOL]] != [
            "successor_not_ready_on_transition"
        ]:
            raise EvidenceError("AVP pre-repair terminal issue changed unexpectedly.")
        if [issue["code"] for issue in before_by_symbol[SIVB_SYMBOL]] != [
            "terminal_action_after_expected_session"
        ]:
            raise EvidenceError("SIVB pre-repair terminal issue changed unexpectedly.")
    elif before_target:
        raise EvidenceError("Already-repaired SIVB/AVP rows are not terminal-ready.")
    if after_target:
        raise EvidenceError(
            "SIVB/AVP target terminal issues remain after the offline plan: "
            + _canonical_json(after_target)
        )
    if _non_target_terminal_fingerprint(before_report) != _non_target_terminal_fingerprint(
        after_report
    ):
        raise EvidenceError("SIVB/AVP plan changed non-target terminal issues.")

    row_deltas = {
        dataset: len(candidate[dataset]) - len(frames[dataset])
        for dataset in REQUIRED_DATASETS
    }
    expected_deltas = {
        "corporate_actions": 1,
        "lifecycle_resolutions": 0,
        "security_master": 0,
        "symbol_history": 1,
        "daily_price_raw": SIVBQ_STORED_ROWS,
        "adjustment_factors": SIVBQ_STORED_ROWS,
        "source_archive": 3,
        "index_constituent_anchors": 0,
        "index_membership_events": 0,
    }
    if state == "repaired":
        expected_deltas = {dataset: 0 for dataset in REQUIRED_DATASETS}
    if row_deltas != expected_deltas:
        raise AssertionError(f"SIVB/AVP plan row scope changed: {row_deltas}")

    plan: dict[str, Any] = {
        "schema": SCHEMA,
        "policy": POLICY,
        "status": status,
        "base_release_version": release.version,
        "completed_session": release.completed_session,
        "write_datasets": list(WRITE_DATASETS),
        "repair_registry": repair_registry_inventory(),
        "repair_registry_sha256": repair_registry_inventory_sha256(),
        "evidence_inventory": evidence_inventory(),
        "evidence_inventory_sha256": evidence_inventory_sha256(),
        "offline_guards": {
            "read_only": True,
            "apply_supported": True,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
            "source_archive_mutated": False,
        },
        "cases": {
            "AVP": {
                "status": status,
                "security_id": AVP_ID,
                "successor_security_id": NTCO_ID,
                "legal_completion_date": AVP_LEGAL_COMPLETION,
                "last_source_price_session": AVP_LAST_SESSION,
                "market_transition_session": AVP_MARKET_TRANSITION,
                "successor_first_price_session": AVP_MARKET_TRANSITION,
                "ratio": AVP_RATIO,
                "economics_changed": False,
                "official_evidence": avp_official,
                "price_evidence": {
                    "source": price_evidence["AVP"],
                    "successor": price_evidence["NTCO"],
                },
            },
            "SIVB": {
                "status": status,
                "security_id": SIVB_ID,
                "last_nasdaq_price_session_observed": SIVB_LAST_SESSION,
                "official_nasdaq_halt_date": SIVB_EXPECTED_MARKET_SESSION,
                "official_nasdaq_suspension_date": SIVBQ_FIRST_SESSION,
                "otc_symbol": "SIVBQ",
                "otc_first_price_session": SIVBQ_FIRST_SESSION,
                "otc_last_price_session": SIVBQ_LAST_SESSION,
                "market_terminal_session": SIVB_MARKET_TERMINATION,
                "legal_cancellation_date": SIVB_LEGAL_CANCELLATION,
                "legal_zero_distribution_action_preserved": True,
                "zero_cash_moved_to_2023_03_10": False,
                "economics_invented": False,
                "same_security_identity": True,
                "same_security_id": SIVB_ID,
                "ticker_change_ratio": 1.0,
                "halted_nonpriceable_interval": {
                    "from": SIVB_EXPECTED_MARKET_SESSION,
                    "through": "2023-03-27",
                },
                "official_evidence": {
                    "legal_cancellation": sivb_official,
                    "market_halt_and_suspension": transition["evidence"]["sec"],
                    "ticker_and_otc_continuity": transition["evidence"]["occ"],
                    "occ_raw_pdf_archived": False,
                },
                "price_evidence": {
                    "nasdaq": price_evidence["SIVB"],
                    "otc": transition["eod_summary"],
                    "otc_source_hash": SIVBQ_EOD_COLLECTED_HASH,
                },
                "collection_budget_receipt": transition["budget_receipt"],
                "missing_evidence": [],
            },
        },
        "validation": {
            "terminal_issues_before": before_target,
            "terminal_issues_after_plan": after_target,
            "avp_target_issue_resolved": True,
            "sivb_target_issue_resolved": True,
            "avp_adjustment_factor_rows_checked": factor_rows,
            "avp_adjustment_factor_economic_rows_changed": factor_changes,
            "full_adjustment_factor_lineage": factor_validation,
            "factor_lineage": {
                "binding": "daily_price_raw_version+corporate_actions_version",
                "economic_values_changed": 0,
            },
            "factor_lineage_rebuild_required_on_apply": state == "old",
            "row_deltas": row_deltas,
            "daily_price_rows_added": SIVBQ_STORED_ROWS if state == "old" else 0,
            "raw_price_rows_reviewed": SIVBQ_EOD_ROWS,
            "exact_hash_non_xnys_exclusions": list(SIVBQ_NON_XNYS_EXCLUSIONS),
            "security_master_rows_modified": 1 if state == "old" else 0,
            "source_archive_rows_changed": archive_additions,
            "source_archive_rows_required": 3,
            "repository_snapshot_issues_preserved": True,
            "eodhd_calls_in_offline_plan": 0,
            "r2_accessed": False,
        },
    }
    plan["plan_sha256"] = _canonical_json_sha256(plan)
    return PreparedPlan(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions=planned_versions,
        frame_hashes=_candidate_frame_hashes(candidate),
        frames=candidate,
        plan=plan,
    )


@contextmanager
def _exclusive_repository_lock(repository: LocalDatasetRepository):
    lock_path = repository.root / ".locks/market-store-write.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery = repository.root / RECOVERY_DIR
        if recovery.exists() and tuple(recovery.glob("*.json")):
            raise RuntimeError("Unresolved SIVB/AVP recovery marker blocks writes.")
        transactions = repository.root / TRANSACTION_DIR
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = _text(json.loads(journal.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        f"Interrupted SIVB/AVP transaction blocks writes: {journal}."
                    )
        yield


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(
        path,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )


def _verify_plan_hash(prepared: PreparedPlan) -> None:
    payload = dict(prepared.plan)
    observed = _text(payload.pop("plan_sha256", ""))
    expected = _canonical_json_sha256(payload)
    if observed != expected:
        raise RuntimeError("Prepared SIVB/AVP plan hash changed before apply.")
    if tuple(prepared.plan.get("write_datasets", ())) != WRITE_DATASETS:
        raise RuntimeError("Prepared SIVB/AVP write scope changed before apply.")


def _verify_prepared_frame_hashes(prepared: PreparedPlan) -> None:
    if not set(WRITE_DATASETS).issubset(prepared.frames):
        raise RuntimeError("Prepared SIVB/AVP candidate frames are incomplete.")
    observed = _candidate_frame_hashes(prepared.frames)
    if observed != dict(prepared.frame_hashes):
        raise RuntimeError("Prepared SIVB/AVP candidate content changed before apply.")
    if prepared.planned_versions:
        if set(prepared.planned_versions) != set(WRITE_DATASETS) or len(
            set(prepared.planned_versions.values())
        ) != len(WRITE_DATASETS):
            raise RuntimeError("Prepared SIVB/AVP versions are invalid.")
        lineage = _factor_source_version(
            prepared.planned_versions["daily_price_raw"],
            prepared.planned_versions["corporate_actions"],
        )
        factors = prepared.frames["adjustment_factors"]
        if not (
            set(factors["source_version"].astype(str)) == {lineage}
            and set(factors["source_hash"].astype(str)) == {lineage}
        ):
            raise RuntimeError("Prepared SIVB/AVP factor/version binding changed.")


def _assert_inputs_unchanged(
    repository: LocalDatasetRepository,
    prepared: PreparedPlan,
) -> None:
    release, release_etag = repository.current_release()
    if (
        release is None
        or release.version != prepared.release.version
        or release_etag != prepared.release_etag
    ):
        raise RuntimeError("Current release changed after SIVB/AVP planning.")
    for dataset in REQUIRED_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if (
            pointer is None
            or pointer.version != prepared.release.dataset_versions[dataset]
            or etag != prepared.pointer_etags[dataset]
        ):
            raise RuntimeError(f"{dataset} pointer changed after SIVB/AVP planning.")


def _metadata_for_write(
    repository: LocalDatasetRepository,
    prepared: PreparedPlan,
    dataset: str,
) -> dict[str, Any]:
    current = repository.manifest_for_version(
        dataset, prepared.release.dataset_versions[dataset]
    )
    metadata = dict(current.metadata)
    input_versions = dict(prepared.release.dataset_versions)
    output_versions = dict(input_versions)
    output_versions.update(prepared.planned_versions)
    factor_lineage = _factor_source_version(
        prepared.planned_versions["daily_price_raw"],
        prepared.planned_versions["corporate_actions"],
    )
    metadata.update(
        {
            "operation": OPERATION,
            "repair_reviewed_at": REVIEWED_AT,
            "input_release_version": prepared.release.version,
            "input_versions": input_versions,
            "output_versions": output_versions,
            "plan_sha256": prepared.plan["plan_sha256"],
            "candidate_content_sha256": prepared.frame_hashes[dataset],
            "repair_registry": repair_registry_inventory(),
            "repair_registry_sha256": repair_registry_inventory_sha256(),
            "evidence_inventory": evidence_inventory(),
            "evidence_inventory_sha256": evidence_inventory_sha256(),
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        }
    )
    if dataset == "daily_price_raw":
        metadata.update(
            {
                "added_security_id": SIVB_ID,
                "added_source_hash": SIVBQ_EOD_COLLECTED_HASH,
                "added_rows": SIVBQ_STORED_ROWS,
            }
        )
    elif dataset == "adjustment_factors":
        factors = prepared.frames["adjustment_factors"]
        expected = {
            "source_version": {factor_lineage},
            "source_hash": {factor_lineage},
            "source": {"derived"},
            "calculated_at": {REVIEWED_AT},
            "retrieved_at": {REVIEWED_AT},
        }
        observed = {
            field: set(factors[field].astype(str)) for field in expected
        }
        if observed != expected:
            raise RuntimeError("Prepared full adjustment-factor lineage is stale.")
        metadata.update(
            {
                "source_version": factor_lineage,
                "source_daily_price_version": prepared.planned_versions[
                    "daily_price_raw"
                ],
                "source_corporate_actions_version": prepared.planned_versions[
                    "corporate_actions"
                ],
                "economic_rows_changed": 0,
                "provenance_rows_rebound": len(factors),
            }
        )
    elif dataset == "lifecycle_resolutions":
        resolutions = prepared.frames["lifecycle_resolutions"]
        metadata.update(
            {
                "candidate_set_sha256": lifecycle_candidate_set_sha256(
                    resolutions[["security_id", "last_price_date"]]
                ),
                "resolution_set_sha256": lifecycle_resolution_set_sha256(
                    resolutions
                ),
                "adjustment_source_version": factor_lineage,
            }
        )
    elif dataset == "source_archive":
        metadata.update(
            {
                "immutable_evidence_rows_added": 3,
                "immutable_evidence_hashes": [
                    entry["source_hash"] for entry in evidence_inventory()
                ],
            }
        )
    return metadata


def _restore_transaction(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: Mapping[str, bytes],
    planned_versions: Mapping[str, str],
    committed_release_version: str,
    old_versions: Mapping[str, str],
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        current = repository.objects.get("releases/current.json")
        if current.data != old_release_bytes:
            observed = DataRelease.from_bytes(current.data)
            expected_versions = {**dict(old_versions), **dict(planned_versions)}
            belongs = (
                bool(committed_release_version)
                and observed.version == committed_release_version
            ) or observed.dataset_versions == expected_versions
            if not belongs:
                raise RuntimeError(
                    f"unexpected release during SIVB/AVP rollback: {observed.version}"
                )
            repository.objects.put(
                "releases/current.json", old_release_bytes, if_match=current.etag
            )
    except Exception as exc:
        errors.append(f"releases/current.json: {type(exc).__name__}: {exc}")
    for dataset in reversed(WRITE_DATASETS):
        key = repository.current_key(dataset)
        try:
            current = repository.objects.get(key)
            old = old_pointer_bytes[dataset]
            if current.data != old:
                pointer = CurrentPointer.from_bytes(current.data)
                if pointer.version != planned_versions[dataset]:
                    raise RuntimeError(
                        f"unexpected {dataset} pointer during SIVB/AVP rollback: "
                        f"{pointer.version}"
                    )
                repository.objects.put(key, old, if_match=current.etag)
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _assert_applied_release(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> None:
    current, _ = repository.current_release()
    if current is None or current.to_bytes() != release.to_bytes():
        raise RuntimeError("Committed SIVB/AVP release is not current.")
    for dataset, version in release.dataset_versions.items():
        pointer, _ = repository.current_pointer(dataset)
        if pointer is None or pointer.version != version:
            raise RuntimeError(f"Applied SIVB/AVP pointer mismatch: {dataset}.")
    lineage = _factor_source_version(
        release.dataset_versions["daily_price_raw"],
        release.dataset_versions["corporate_actions"],
    )
    factor_manifest = repository.manifest_for_version(
        "adjustment_factors", release.dataset_versions["adjustment_factors"]
    )
    if any(
        _text(factor_manifest.metadata.get(key)) != value
        for key, value in {
            "source_version": lineage,
            "source_daily_price_version": release.dataset_versions[
                "daily_price_raw"
            ],
            "source_corporate_actions_version": release.dataset_versions[
                "corporate_actions"
            ],
        }.items()
    ):
        raise RuntimeError("SIVB/AVP factor manifest lineage is not release-exact.")
    archive_manifest = repository.manifest_for_version(
        "source_archive", release.dataset_versions["source_archive"]
    )
    if (
        _text(archive_manifest.metadata.get("evidence_inventory_sha256"))
        != evidence_inventory_sha256()
    ):
        raise RuntimeError("SIVB/AVP source archive manifest is not evidence-pinned.")
    replay = prepare_plan(repository)
    if replay.plan["status"] != "already_repaired":
        raise RuntimeError("SIVB/AVP repair is not idempotent.")
    if replay.plan["validation"]["terminal_issues_after_plan"]:
        raise RuntimeError("Applied SIVB/AVP targets are not terminal-ready.")


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedPlan,
    *,
    inject_failure: FailureInjector | None = None,
) -> dict[str, Any]:
    """Apply one prepared plan atomically; immutable payloads may survive rollback."""

    inject = inject_failure or (lambda _stage: None)
    with _exclusive_repository_lock(repository):
        _verify_static_contract()
        _verify_code_pins()
        _assert_inputs_unchanged(repository, prepared)
        canonical = prepare_plan(repository)
        _verify_plan_hash(canonical)
        _assert_inputs_unchanged(repository, canonical)
        if canonical.plan["status"] == "already_repaired":
            return {
                **canonical.plan,
                "mode": "apply",
                "writes_performed": False,
            }
        if canonical.plan["status"] != "ready_offline_plan":
            raise RuntimeError(
                "Locked SIVB/AVP re-plan is neither ready nor already repaired."
            )
        _verify_plan_hash(prepared)
        _verify_prepared_frame_hashes(prepared)
        if (
            canonical.release.version != prepared.release.version
            or canonical.plan["plan_sha256"] != prepared.plan["plan_sha256"]
            or _semantic_candidate_hashes(canonical.frames)
            != _semantic_candidate_hashes(prepared.frames)
        ):
            raise RuntimeError(
                "Locked SIVB/AVP re-plan differs from the caller's reviewed plan."
            )
        prepared = canonical
        _verify_prepared_frame_hashes(prepared)
        transition = _verify_reviewed_sivb_transition_cache(repository.root)
        _verify_official_evidence(
            repository,
            prepared.frames["source_archive"],
            completed_session=prepared.release.completed_session,
        )
        planned = dict(prepared.planned_versions)
        if set(planned) != set(WRITE_DATASETS) or len(set(planned.values())) != len(
            WRITE_DATASETS
        ):
            raise RuntimeError("Prepared SIVB/AVP versions are invalid.")
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != prepared.release.dataset_versions[dataset]
                or value.etag != prepared.pointer_etags[dataset]
            ):
                raise RuntimeError(f"{dataset} pointer changed before SIVB/AVP apply.")
            old_pointers[dataset] = value.data
        transaction_id = uuid.uuid4().hex
        journal_path = repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        journal: dict[str, Any] = {
            "schema": "us_sivb_avp_terminal_transition_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                key: base64.b64encode(value).decode("ascii")
                for key, value in old_pointers.items()
            },
            "planned_versions": planned,
            "write_datasets": list(WRITE_DATASETS),
            "plan_sha256": prepared.plan["plan_sha256"],
            "repair_registry_sha256": repair_registry_inventory_sha256(),
            "evidence_inventory_sha256": evidence_inventory_sha256(),
            "candidate_content_sha256": dict(prepared.frame_hashes),
            "written_datasets": [],
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        try:
            inject("after_journal")
            _persist_evidence(
                repository,
                transition,
                completed_session=prepared.release.completed_session,
            )
            inject("after_evidence_write")
            versions = dict(prepared.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    prepared.frames[dataset],
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy="block",
                    metadata=_metadata_for_write(repository, prepared, dataset),
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"{dataset} write conflicted: {result.conflict_path}."
                    )
                if result.manifest.version != planned[dataset]:
                    raise RuntimeError(f"Unexpected {dataset} version was written.")
                versions[dataset] = result.manifest.version
                journal["written_datasets"] = [
                    *journal["written_datasets"],
                    dataset,
                ]
                _write_journal(journal_path, journal)
                inject(f"after_write:{dataset}")
            for dataset in REQUIRED_DATASETS:
                if dataset not in WRITE_DATASETS:
                    pointer, etag = repository.current_pointer(dataset)
                    if (
                        pointer is None
                        or pointer.version != prepared.release.dataset_versions[dataset]
                        or etag != prepared.pointer_etags[dataset]
                    ):
                        raise RuntimeError(
                            f"Out-of-scope pointer changed during apply: {dataset}."
                        )
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=prepared.release.quality,
                warnings=prepared.release.warnings,
                expected_etag=prepared.release_etag,
            )
            inject("after_release_commit")
            _assert_applied_release(repository, committed)
            journal.update(
                {
                    "status": "committed",
                    "committed_release_version": committed.version,
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            return {
                **prepared.plan,
                "status": "applied",
                "mode": "apply",
                "writes_performed": True,
                "new_release_version": committed.version,
                "transaction_id": transaction_id,
                "quality": str(committed.quality),
                "warnings": list(committed.warnings),
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_versions=planned,
                committed_release_version=committed.version if committed else "",
                old_versions=prepared.release.dataset_versions,
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
                recovery = repository.root / RECOVERY_DIR / f"{transaction_id}.json"
                _write_journal(recovery, journal)
                raise RuntimeError(
                    "SIVB/AVP rollback was incomplete; recovery marker blocks writes: "
                    f"{recovery}; errors={rollback_errors}."
                ) from original
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or transactionally apply the offline SIVB/AVP repair."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--compact", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan-sivb-evidence", action="store_true")
    modes.add_argument("--fetch-sivb-evidence", action="store_true")
    modes.add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fetch_sivb_evidence:
        load_env()
        result = collect_sivb_transition_evidence(
            args.cache_root,
            sec_user_agent=os.getenv("SEC_USER_AGENT", ""),
            eodhd_token=os.getenv("EODHD_API_TOKEN", ""),
            budget=EodhdCallBudget(),
        )
        result = {
            key: value
            for key, value in result.items()
            if key not in {"payloads", "eod_rows"}
        }
    elif args.plan_sivb_evidence:
        result = sivb_transition_collection_plan(args.cache_root)
        result = {
            key: value
            for key, value in result.items()
            if key not in {"payloads", "eod_rows"}
        }
    else:
        repository = LocalDatasetRepository(args.cache_root)
        prepared = prepare_plan(repository)
        result = apply_repair(repository, prepared) if args.apply else prepared.plan
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            indent=None if args.compact else 2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
