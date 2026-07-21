#!/usr/bin/env python3
"""Plan or atomically apply seven exact US identity-price-tail repairs.

The reviewed batch is finite and code-pinned: FLT, CDAY, XEC, HCP, UTX,
COG, and CTRP.  SYMC/NLOK is deliberately excluded because it needs a
separate coupled canonical-identity and index-membership repair.

The default command is a read-only plan.  ``--apply`` is available only for a
later, explicit authorization.  Both paths are offline: this module has no
HTTP, EODHD, Yahoo, R2, or other remote code path.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gc
import gzip
import hashlib
import json
import math
import sys
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import duckdb
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audit_us_identity_tail_repairs import (  # noqa: E402
    CASES as AUDIT_CASES,
    PINNED_DATASET_VERSIONS,
    PINNED_RELEASE_VERSION,
    SIGNAL_COLUMNS,
    TRIPLE_SETTINGS,
    _canonical_json_bytes,
    _date,
    _frame_sha256,
    _read_security_subset,
    _signal_diff,
    _signal_frame,
    _text,
    build_audit,
)
from collect_us_lifecycle_actions import (  # noqa: E402
    _build_price_histories,
    _crosscheck,
)
from supertrend_quant.market_store.lifecycle import (  # noqa: E402
    LifecycleCandidate,
    build_lifecycle_candidates,
)
from supertrend_quant.market_store.lifecycle_coverage import (  # noqa: E402
    lifecycle_candidate_id,
    lifecycle_candidate_set_sha256,
    lifecycle_resolution_set_sha256,
    validate_lifecycle_coverage,
)
from supertrend_quant.market_store.lifecycle_report_provenance import (  # noqa: E402
    REPORT_BINDING_FIELDS,
    build_lifecycle_report_binding,
    validate_lifecycle_report_binding,
)
from supertrend_quant.market_store.manifest import (  # noqa: E402
    CurrentPointer,
    DataRelease,
    sha256_bytes,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.repository import (  # noqa: E402
    LocalDatasetRepository,
)
from supertrend_quant.market_store.official_lifecycle_evidence import (  # noqa: E402
    include_bound_official_applied_event_candidates,
    load_official_lifecycle_exception_evidence,
    validate_official_evidence_content,
)
from supertrend_quant.market_store.validation import (  # noqa: E402
    index_member_identity_gap_fingerprint,
    validate_dataset,
)


DEFAULT_CACHE_ROOT = Path("data/cache")
LIFECYCLE_HINTS_PATH = Path("unified_quant/configs/us_lifecycle_hints.yaml")
BASE_LIFECYCLE_REPORT_PATH = Path(
    "results/data_quality/us_lifecycle/sec_collection_20260719_current_release.json"
)
OPERATION = "repair_us_identity_price_tails"
REPAIR_SCHEMA = "us_identity_price_tail_repair/v1"
REPAIR_REVIEWED_AT = "2026-07-19T00:00:00Z"
REPAIRED_IDENTITY_SOURCE = "official_identity_price_tail_boundary_repair"
WRITE_DATASETS = (
    "daily_price_raw",
    "adjustment_factors",
    "lifecycle_resolutions",
    "security_master",
    "source_archive",
    "symbol_history",
)
# Minimum read/transform inputs.  Pointer CAS protection is intentionally wider:
# every dataset named by the current DataRelease is captured dynamically.
REQUIRED_DATASETS = (
    *WRITE_DATASETS,
    "corporate_actions",
    "index_constituent_anchors",
    "index_membership_events",
)
TRANSACTION_DIR = "transactions/us-identity-price-tails"
RECOVERY_DIR = "recovery/us-identity-price-tails"

CASES = tuple(case for case in AUDIT_CASES if case.symbol != "SYMC")
EXPECTED_SYMBOLS = frozenset({"FLT", "CDAY", "XEC", "HCP", "UTX", "COG", "CTRP"})
EXPECTED_REMOVED_PRICE_ROWS = 616
EXPECTED_OLD_AFFECTED_FACTOR_ECONOMICS_SHA256 = (
    "cca21fd3697832755d41db3fae161864f6123bdd7840c0ae752ba4366fe7139f"
)
EXPECTED_REPAIRED_AFFECTED_FACTOR_ECONOMICS_SHA256 = (
    "1e74df3c540dfc10d6627df355c4bb7dbfe7a4f441145f0f93b1dc54228e2100"
)
EXPECTED_REPAIRED_AFFECTED_PRICE_CONTENT_SHA256 = (
    "55a9a879cea3c6a7df7065ef0234daa0fe3486c1bc4bbc914225d7a6a1dd0eef"
)
EXPECTED_REPAIRED_IDENTITY_MASTER_SHA256 = (
    "d2fda8215bd40a35c7a9ae20cc4c2c02c0ad8bf42cc1d8b9266e1a9ffbf4a027"
)
EXPECTED_REPAIRED_SYMBOL_HISTORY_SHA256 = (
    "2a31ff8d03505f8af69d53c6f81008c5aad5c517b1a94b51b9ea0e10fcb15bfe"
)
EXPECTED_CANDIDATE_CONTENT_SHA256 = (
    "6e41ecd8c75b693860628616f34784b1a1cf339cf4271f504ed9e224b3b62b03"
)
EXPECTED_BASE_LIFECYCLE_REPORT_SHA256 = (
    "14845368ca771dbea5bc82e8fed4077152fd2c92b424059961e02239188c45ae"
)
EXPECTED_BASE_LIFECYCLE_HINTS_SHA256 = (
    "2dc55661d72c9f993cd233339ad14858ff44b86d8ceca3d3bb33059fc30369e6"
)
EXPECTED_CURRENT_LIFECYCLE_HINTS_SHA256 = (
    "3fd60e760466ffba8b00c6ea191944dc4c00ceb21199f0bb45e96f0d5d969866"
)
EXPECTED_BASE_LIFECYCLE_COLLECTOR_CONFIG_SHA256 = (
    "38b79fb80b624ce4faf924d5e462e64ed35b3a032dd1ac17fd04a7c38c5aaeb3"
)
EXPECTED_BASE_LIFECYCLE_COLLECTION_CONTEXT_SHA256 = (
    "dadeb622841c53f973e21a6f847af40cf2407a9d999ebdf3dcf058a6fe98628e"
)
EXPECTED_BASE_LIFECYCLE_CANDIDATE_SET_SHA256 = (
    "3d4af2beeaa5364dc397e60d9ab806eb10e07b77da062935992daf5ca55c070e"
)
EXPECTED_BASE_LIFECYCLE_RESOLUTION_SET_SHA256 = (
    "1f124a67ffb87efe211772b1ae4bd4560db2aa87d1f27332d266c0c9ff6db5eb"
)
EXPECTED_REPAIRED_LIFECYCLE_CANDIDATE_SET_SHA256 = (
    "050ec3de6a06e219c77768a499cb59d6eee7d445a376641ecd36ca0307fcead0"
)
EXPECTED_REPAIRED_LIFECYCLE_RESOLUTION_SET_SHA256 = (
    "be5a0000dcde7ce73306d9d656c66e0457b057668283bab9c7c361d370f61727"
)
EXPECTED_LIFECYCLE_COVERAGE = {
    "candidate_count": 182,
    "resolution_count": 182,
    "applied_count": 170,
    "exception_count": 12,
    "open_count": 0,
}
LIFECYCLE_REPORT_SOURCE = "lifecycle_evidence_report"
UTX_CARR_SECURITY_ID = "US:EODHD:a2ae04c6-0b8a-5b91-9166-0fa28b73926b"
UTX_OTIS_SECURITY_ID = "US:EODHD:db0c2e8e-5060-521f-b68e-b4c750abcd72"
UTX_EXCEPTION_REASON = (
    "UTX-to-RTX raw ticker-change price equality is invalid because the current "
    "canonical action inventory omits the CARR 1.0-share and OTIS 0.5-share "
    "distributions; keep fail-closed until exact spin-off actions are modeled."
)
UTX_RELEASE_WARNING = (
    "UTX is a permanent unsupported_consideration exception, not a passed "
    "economic cross-check: the canonical action model omits the exact 1.0 CARR "
    "+ 0.5 OTIS distributions documented by the pinned SEC filing."
)
UTX_EXCEPTION_REVIEWED_AT = "2026-07-18T00:00:00Z"
UTX_EXCEPTION_RECHECK_AFTER = ""
UTX_DISTRIBUTION_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/101829/"
    "000114036120008397/0001140361-20-008397.txt"
)
UTX_DISTRIBUTION_SOURCE_HASH = (
    "8b3131e8bf46b322c0c7c9e37e32c624c05336e3f3acaddf86c777ce17f7d6a2"
)
UTX_DISTRIBUTION_SOURCE_BYTES = 6_337_925
UTX_DISTRIBUTION_RETRIEVED_AT = "2026-07-18T20:58:38.225941Z"
UTX_HINT_ADDITIVE_BLOCK = (
    b"  utx_2020_carr_otis_distributions:\n"
    b"    candidate_symbols: [UTX]\n"
    b"    candidate_name_contains: [Raytheon Technologies]\n"
    b"    candidate_security_ids:\n"
    b"      - US:EODHD:aefd1dd7-529d-5b6a-80e9-65d0e14102a6\n"
    b"    candidate_last_price_dates: [2020-04-02]\n"
    b"    binding_status: bound\n"
    b"    effective_date: 2020-04-03\n"
    b"    filing_date: 2020-04-03\n"
    b"    resolution_kind: exception\n"
    b"    exception_code: unsupported_consideration\n"
    b"    claim: UTX-to-RTX raw ticker-change price equality is invalid because "
    b"the current canonical action inventory omits the CARR 1.0-share and OTIS "
    b"0.5-share distributions; keep fail-closed until exact spin-off actions are "
    b"modeled.\n"
    b"    source_url: https://www.sec.gov/Archives/edgar/data/101829/"
    b"000114036120008397/0001140361-20-008397.txt\n"
    b"    source_sha256: \"8b3131e8bf46b322c0c7c9e37e32c624c05336e3f3acaddf8"
    b"6c777ce17f7d6a2\"\n"
    b"    required_text_groups:\n"
    b"      - [\"April 2, 2020\"]\n"
    b"      - [Separation and the Distributions]\n"
    b"      - [one share of Carrier common stock for each share of the Company"
    b"\xe2\x80\x99s common stock held as of the Record Date]\n"
    b"      - [one-half share of Otis common stock for each share of the Company"
    b"\xe2\x80\x99s common stock held as of the Record Date]\n"
)

# Cross-validation targets from the exact current archived Yahoo report.  The
# repair does not authorize these exceptions by itself.  It emits the exact
# policy/verifier work that must be code-reviewed after the repaired release is
# committed and a new offline report is generated.
_CROSS_VALIDATION_TARGETS = {
    "FLT": (
        "f1d33b3660eab9c87a6d293a44cd07f559d36fbb7f5250adde9590a94e70e0ee",
        "cbbeaa4a07c0eb4f4b3ad6cd84b8068f9a73d8e5f1d6bf23ae53ab6c1932ecab",
        "passed",
    ),
    "CDAY": (
        "0e8ae718d558c843f3add73eccc7549680728dde024cb664c157d0d3f112c6c0",
        "6474aa4dd0199ae11fcd95db904ce6ae57fdacb9db716e82679520e3550b22d5",
        "finite_chain_required",
    ),
    "XEC": (
        "5a7d992833e6db3ede1850047f5dedf2a806e2055304ff54490e177a60a07bd8",
        "aed2031265aeba347424145a0518f26e021ec0608fb262842f5be5661ca1add6",
        "finite_chain_required",
    ),
    "HCP": (
        "db3ded074c2702bbea5ba8cd96b43422af814773ede30fa2516bc6f1e38f3d25",
        "3f03ec2b1a12bb81f0f1c509f6332b45c4928958926828120535333341a38679",
        "finite_chain_required",
    ),
    "UTX": (
        "b35f720518c5b28fc0aff4b1bdd9a13231da9d9ed57ca4444371dec3aa56dbed",
        "3ef54278e2c2a2432a153d547a254eb32bce5506a8aa5c365d6fda2dbcd00c95",
        "passed",
    ),
    "COG": (
        "aed2031265aeba347424145a0518f26e021ec0608fb262842f5be5661ca1add6",
        "c2b2bae7a161d772a3d93e68d225d16b33bb171730a4190834506ac4031c4020",
        "finite_chain_required",
    ),
    "CTRP": (
        "0043cc09f8d447c3b32e26cebb651aba0118eeca87e1cea5ec0535a8f42cfee9",
        "b4ca2124443b0b05a5ce3fbacfcc5fc1b45dab595e5812273bd5af2ad09a1478",
        "passed",
    ),
}

EXPECTED_SNAPSHOT_IDENTITY_GAP = {
    "code": "index_member_missing_active_symbol",
    "row_count": 1,
    "index_id": "sp500",
    "replay_date": "2020-10-07",
    "security_id": "US:EODHD:3dd6d6ce-e7a1-5078-b258-df5b18404c9d",
    "next_remove_event_id": (
        "59d17bfad7dceb1c4903d45cc083841209982df638ec0f18006f2d3a7987d12d"
    ),
    "next_remove_effective_date": "2020-10-12",
    "next_remove_source": "community_sp500_history",
    "next_remove_source_hash": (
        "39a9202c9ef69a74c0ff07e2113ad41fb6da7c8c5b6cd9541f0185fb4391e717"
    ),
    "fingerprint": (
        "989c5d44ef1b8cf8a682d807b63a62ebe3c3f38eb6f57e6314b3fe381d5c7d04"
    ),
}


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: dict[str, str | None]
    planned_versions: dict[str, str]
    frames: dict[str, pd.DataFrame]
    summary: dict[str, Any]
    planned_release: DataRelease | None = None
    evidence_report_bytes: bytes = b""
    evidence_report_object_path: str = ""


FailureInjector = Callable[[str], None]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_sha256(value: Any) -> str:
    return sha256_bytes(_canonical_json_bytes(value))


def repair_registry() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in CASES:
        target_id, successor_target_id, successor_path = _CROSS_VALIDATION_TARGETS[
            case.symbol
        ]
        row = {
            "symbol": case.symbol,
            "security_id": case.security_id,
            "successor_symbol": case.successor_symbol,
            "successor_security_id": case.successor_security_id,
            "event_id": case.event_id,
            "action_type": case.action_type,
            "transition_date": case.transition_date,
            "old_last_good_session": case.old_last_good_session,
            "tail_end": case.tail_end,
            "tail_rows": case.tail_rows,
            "old_tail_source_hash": case.old_tail_source_hash,
            "old_tail_sha256": case.old_tail_sha256,
            "successor_overlap_source_hash": case.successor_overlap_source_hash,
            "successor_overlap_sha256": case.successor_overlap_sha256,
            "official_source_hash": case.official_source_hash,
            "action_row_sha256": case.action_row_sha256,
            "identity_inventory_sha256": case.identity_inventory_sha256,
            "membership_inventory_sha256": case.membership_inventory_sha256,
            "cross_validation_target_id": target_id,
            "successor_target_id": successor_target_id,
            "successor_path": successor_path,
            "reassign_tail_to_successor": False,
            "hcp_replace_successor_first_session": case.symbol == "HCP",
        }
        row["registry_item_sha256"] = _canonical_sha256(row)
        rows.append(row)
    return rows


def registry_inventory_sha256() -> str:
    return _canonical_sha256(repair_registry())


# Filled with a literal after the registry is reviewed.  This second code pin
# prevents changing imported audit constants and merely recomputing metadata.
TRUSTED_REGISTRY_INVENTORY_SHA256 = (
    "5e5274d6ddec6eec037bdd127ea1c38a93c0e218ef96e3ed5b9e9af5fd3259ee"
)


def _static_contract() -> None:
    _require(len(CASES) == 7, "Identity-price-tail case count changed.")
    _require(
        {case.symbol for case in CASES} == EXPECTED_SYMBOLS,
        "Identity-price-tail symbol inventory changed.",
    )
    _require(
        sum(case.tail_rows for case in CASES) == EXPECTED_REMOVED_PRICE_ROWS,
        "Identity-price-tail row inventory changed.",
    )
    _require(
        registry_inventory_sha256() == TRUSTED_REGISTRY_INVENTORY_SHA256,
        "Identity-price-tail registry fingerprint changed.",
    )
    for case in CASES:
        _require(case.symbol != "SYMC", "SYMC/NLOK must remain out of this batch.")
        _require(
            case.old_last_good_session < case.transition_date <= case.tail_end,
            f"{case.symbol} boundary ordering changed.",
        )
    hcp = next(case for case in CASES if case.symbol == "HCP")
    _require(
        hcp.transition_date == "2019-11-05"
        and hcp.tail_rows == 4
        and hcp.repair_class
        == "replace_successor_first_session_then_delete_old_tail",
        "HCP replacement contract changed.",
    )
    xec = next(case for case in CASES if case.symbol == "XEC")
    cog = next(case for case in CASES if case.symbol == "COG")
    _require(
        xec.successor_security_id == cog.security_id
        and xec.action_type == "stock_merger"
        and xec.tail_rows == 2,
        "XEC/COG distinct-issuer contract changed.",
    )


def cross_validation_change_plan() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for case in CASES:
        target_id, successor_target_id, successor_path = _CROSS_VALIDATION_TARGETS[
            case.symbol
        ]
        entries.append(
            {
                "symbol": case.symbol,
                "target_id": target_id,
                "security_id": case.security_id,
                "old_symbol_effective_to": case.old_last_good_session,
                "old_master_active_to": case.old_last_good_session,
                "event_id": case.event_id,
                "action_type": case.action_type,
                "transition_date": case.transition_date,
                "official_source_hash": case.official_source_hash,
                "successor_symbol": case.successor_symbol,
                "successor_security_id": case.successor_security_id,
                "successor_target_id": successor_target_id,
                "successor_path": successor_path,
                "required_basis": (
                    "code_pinned_identity_tail_transition_to_successor_price/v1"
                ),
            }
        )
    inventory_sha256 = _canonical_sha256(entries)
    return {
        "status": "post_apply_policy_work_required",
        "generic_date_tolerance": False,
        "generic_successor_inheritance": False,
        "exception_count": 7,
        "exception_inventory_sha256": inventory_sha256,
        "policy_changes": [
            "add seven exact reviewed_identity_tail_no_data_transitions entries",
            "add only finite successor chains whose final target has passed price evidence",
            "pin the complete transition inventory SHA in Python and YAML",
        ],
        "verifier_requirements": [
            "require repaired release and manifest operation/registry SHA",
            "require exact old master active_to and symbol effective_to",
            "require no old-SID price or factor key on/after transition",
            "require exact official action and archived official payload hash",
            "require exact successor identity interval and successor target ID",
            "require successor passed price target or a separately code-pinned finite chain",
            "reject cycles, dead ends, no-data inheritance, and unlisted targets",
        ],
        "entries": entries,
    }


def _safe_path(root: Path, object_path: str) -> Path:
    base = root.resolve()
    target = (base / object_path).resolve()
    if target == base or base not in target.parents:
        raise RuntimeError(f"Archive path escapes repository: {object_path}.")
    return target


def _verify_official_evidence(
    repository: LocalDatasetRepository,
    actions: pd.DataFrame,
    archive: pd.DataFrame,
) -> None:
    for case in CASES:
        rows = actions.loc[actions["event_id"].map(_text).eq(case.event_id)]
        _require(len(rows) == 1, f"{case.symbol} official action inventory changed.")
        row = rows.iloc[0]
        _require(
            _text(row.get("security_id")) == case.security_id
            and _text(row.get("action_type")) == case.action_type
            and _date(row.get("effective_date")) == case.transition_date
            and _text(row.get("new_security_id")) == case.successor_security_id
            and _text(row.get("new_symbol")).upper() == case.successor_symbol
            and bool(row.get("official"))
            and _text(row.get("source_hash")).lower() == case.official_source_hash,
            f"{case.symbol} official action binding changed.",
        )
        _require(
            _frame_sha256(rows, sort_by=("event_id",)) == case.action_row_sha256,
            f"{case.symbol} action row hash changed.",
        )
        source_url = _text(row.get("source_url"))
        archived = archive.loc[
            archive["source_hash"].map(_text).str.lower().eq(case.official_source_hash)
            & archive["source_url"].map(_text).eq(source_url)
        ]
        _require(len(archived) == 1, f"{case.symbol} archive binding changed.")
        path = _safe_path(repository.root, _text(archived.iloc[0].get("object_path")))
        _require(path.is_file(), f"{case.symbol} official archive object is missing.")
        try:
            payload = gzip.decompress(path.read_bytes())
        except (OSError, EOFError) as exc:
            raise RuntimeError(f"{case.symbol} official archive gzip is invalid.") from exc
        _require(
            hashlib.sha256(payload).hexdigest() == case.official_source_hash,
            f"{case.symbol} official archive payload changed.",
        )


def _session_series(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["session"], errors="raise").dt.date.astype(str)


def _old_tail(frame: pd.DataFrame, case: Any) -> pd.DataFrame:
    sessions = _session_series(frame)
    return frame.loc[
        frame["security_id"].map(_text).eq(case.security_id)
        & sessions.ge(case.transition_date)
    ]


def _identity_rows(frame: pd.DataFrame, case: Any, *, history: bool) -> pd.DataFrame:
    symbol_field = "symbol" if history else "primary_symbol"
    return frame.loc[
        frame["security_id"].map(_text).eq(case.security_id)
        & frame[symbol_field].map(_text).str.upper().eq(case.symbol)
    ]


def _old_state(frames: Mapping[str, pd.DataFrame]) -> None:
    for case in CASES:
        tail = _old_tail(frames["daily_price_raw"], case)
        _require(
            len(tail) == case.tail_rows
            and _frame_sha256(tail, sort_by=("session",)) == case.old_tail_sha256,
            f"{case.symbol} old price-tail bytes changed.",
        )
        master = _identity_rows(frames["security_master"], case, history=False)
        history = _identity_rows(frames["symbol_history"], case, history=True)
        _require(len(master) == len(history) == 1, f"{case.symbol} identity changed.")
        _require(
            _date(master.iloc[0].get("active_to")) == case.tail_end
            and not _date(history.iloc[0].get("effective_to")),
            f"{case.symbol} old identity boundary changed.",
        )


def _replacement_hcp_row(prices: pd.DataFrame) -> pd.Series:
    case = next(case for case in CASES if case.symbol == "HCP")
    rows = prices.loc[
        prices["security_id"].map(_text).eq(case.security_id)
        & _session_series(prices).eq(case.transition_date)
    ]
    _require(len(rows) == 1, "HCP exact replacement source row changed.")
    row = rows.iloc[0].copy()
    _require(
        _text(row.get("source_hash")) == case.old_tail_source_hash
        and math.isclose(float(row.get("open")), 34.75, rel_tol=0, abs_tol=1e-12)
        and math.isclose(float(row.get("high")), 34.82, rel_tol=0, abs_tol=1e-12)
        and math.isclose(float(row.get("low")), 33.85, rel_tol=0, abs_tol=1e-12)
        and math.isclose(float(row.get("close")), 34.41, rel_tol=0, abs_tol=1e-12)
        and math.isclose(float(row.get("volume")), 8_054_269.0, rel_tol=0, abs_tol=0),
        "HCP exact replacement OHLCV/source hash changed.",
    )
    row["security_id"] = case.successor_security_id
    return row


def _rewrite_prices(
    prices: pd.DataFrame, *, copy_frame: bool = True
) -> pd.DataFrame:
    output = prices.copy(deep=True) if copy_frame else prices
    hcp_replacement = _replacement_hcp_row(output)
    sessions = _session_series(output)
    remove = pd.Series(False, index=output.index)
    for case in CASES:
        remove |= output["security_id"].map(_text).eq(case.security_id) & sessions.ge(
            case.transition_date
        )
    _require(
        int(remove.sum()) == EXPECTED_REMOVED_PRICE_ROWS,
        "Exact old-SID price-tail row inventory changed.",
    )
    hcp = next(case for case in CASES if case.symbol == "HCP")
    successor_first = (
        output["security_id"].map(_text).eq(hcp.successor_security_id)
        & sessions.eq(hcp.transition_date)
    )
    _require(int(successor_first.sum()) == 1, "PEAK first-session key changed.")
    for column in output.columns:
        output.loc[successor_first, column] = hcp_replacement[column]
    return output.loc[~remove].reset_index(drop=True)


def _rewrite_identity(frame: pd.DataFrame, *, history: bool) -> pd.DataFrame:
    output = frame.copy(deep=True)
    end_field = "effective_to" if history else "active_to"
    for case in CASES:
        rows = _identity_rows(output, case, history=history).index
        _require(len(rows) == 1, f"{case.symbol} identity row changed.")
        action_url = ""
        # Source URL is filled later from the exact action row.  The pure
        # rewrite intentionally needs only the imported code pins.
        output.loc[rows, end_field] = case.old_last_good_session
        output.loc[rows, "source"] = REPAIRED_IDENTITY_SOURCE
        output.loc[rows, "retrieved_at"] = REPAIR_REVIEWED_AT
        output.loc[rows, "source_hash"] = case.official_source_hash
        if "source_url" in output.columns:
            output.loc[rows, "source_url"] = action_url
    return output.reset_index(drop=True)


def _bind_identity_source_urls(
    frame: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    history: bool,
) -> pd.DataFrame:
    output = frame.copy(deep=True)
    for case in CASES:
        action = actions.loc[actions["event_id"].map(_text).eq(case.event_id)]
        _require(len(action) == 1, f"{case.symbol} action source URL changed.")
        rows = _identity_rows(output, case, history=history).index
        output.loc[rows, "source_url"] = _text(action.iloc[0].get("source_url"))
    return output


def _candidate_content_projection(
    *,
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    master: pd.DataFrame,
    history: pd.DataFrame,
) -> dict[str, str]:
    projection = {
        "price": _frame_sha256(
            prices, sort_by=("security_id", "session")
        ),
        "factor_economics": _affected_factor_economics_sha256(factors),
        "master": _frame_sha256(master, sort_by=("security_id",)),
        "history": _frame_sha256(
            history, sort_by=("security_id", "symbol", "effective_from")
        ),
    }
    expected = {
        "price": EXPECTED_REPAIRED_AFFECTED_PRICE_CONTENT_SHA256,
        "factor_economics": EXPECTED_REPAIRED_AFFECTED_FACTOR_ECONOMICS_SHA256,
        "master": EXPECTED_REPAIRED_IDENTITY_MASTER_SHA256,
        "history": EXPECTED_REPAIRED_SYMBOL_HISTORY_SHA256,
    }
    _require(projection == expected, "Candidate content projection changed.")
    digest = _canonical_sha256(projection)
    _require(
        digest == EXPECTED_CANDIDATE_CONTENT_SHA256,
        "Candidate content aggregate hash changed.",
    )
    return {**projection, "aggregate": digest}


def _adjustment_source_version(price_version: str, action_version: str) -> str:
    _require(bool(price_version and action_version), "Factor lineage inputs are empty.")
    return f"{price_version}+{action_version}"


def _new_versions(release: DataRelease) -> dict[str, str]:
    token = uuid.uuid4().hex
    session = release.completed_session.replace("-", "")
    return {
        dataset: f"identity-price-tails-{session}-{token}-{dataset}"
        for dataset in WRITE_DATASETS
    }


def _workspace_path(path: Path) -> Path:
    return path if path.is_absolute() else SCRIPT_DIR.parents[1] / path


def _json_document_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _candidate_frame(
    candidates: tuple[LifecycleCandidate, ...],
) -> pd.DataFrame:
    return pd.DataFrame(
        [asdict(candidate) for candidate in candidates],
        columns=(
            "security_id",
            "symbol",
            "name",
            "exchange",
            "last_price_date",
            "active_to",
            "index_remove_dates",
        ),
    )


def _candidate_dict(candidate: LifecycleCandidate) -> dict[str, Any]:
    value = asdict(candidate)
    value["index_remove_dates"] = list(value["index_remove_dates"])
    return value


class _ProjectedCandidateRepository:
    """Minimal in-memory repository used only by the canonical candidate builder."""

    def __init__(self, frames: Mapping[str, pd.DataFrame]):
        self.frames = dict(frames)

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        return self.frames[dataset].copy(deep=False)


def _read_last_price_summary(
    repository: LocalDatasetRepository,
    version: str,
) -> pd.DataFrame:
    """Use DuckDB pushdown so candidate regeneration never loads all daily bars."""

    paths = [str(path) for path in repository.parquet_paths("daily_price_raw", version)]
    _require(bool(paths), "Daily-price Parquet inventory is empty.")
    connection = duckdb.connect()
    try:
        output = connection.execute(
            "SELECT CAST(security_id AS VARCHAR) AS security_id, "
            "MAX(session) AS session "
            "FROM read_parquet(?, union_by_name=true) GROUP BY security_id",
            [paths],
        ).fetchdf()
    finally:
        connection.close()
    _require(
        not output.empty and not output.duplicated("security_id").any(),
        "Daily-price terminal-session summary changed.",
    )
    return output.reset_index(drop=True)


def _read_candidate_context(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, pd.DataFrame]:
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in (
            "security_master",
            "symbol_history",
            "corporate_actions",
            "index_constituent_anchors",
            "index_membership_events",
        )
    }
    frames["daily_price_raw"] = _read_last_price_summary(
        repository, release.dataset_versions["daily_price_raw"]
    )
    return frames


def _project_candidate_context(
    current: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    output = {key: value.copy(deep=True) for key, value in current.items()}
    actions = output["corporate_actions"]
    output["security_master"] = _bind_identity_source_urls(
        _rewrite_identity(output["security_master"], history=False),
        actions,
        history=False,
    )
    output["symbol_history"] = _bind_identity_source_urls(
        _rewrite_identity(output["symbol_history"], history=True),
        actions,
        history=True,
    )
    terminals = output["daily_price_raw"]
    sessions = pd.to_datetime(terminals["session"], errors="raise").dt.normalize()
    for case in CASES:
        rows = terminals.index[terminals["security_id"].map(_text).eq(case.security_id)]
        _require(len(rows) == 1, f"{case.symbol} terminal summary row changed.")
        row = rows[0]
        _require(
            _date(sessions.loc[row]) == case.tail_end,
            f"{case.symbol} baseline terminal summary changed.",
        )
        terminals.loc[row, "session"] = case.old_last_good_session
    return output


def _build_candidate_values(
    frames: Mapping[str, pd.DataFrame],
    release: DataRelease,
) -> tuple[LifecycleCandidate, ...]:
    repository = _ProjectedCandidateRepository(frames)
    specs = load_official_lifecycle_exception_evidence(
        _workspace_path(LIFECYCLE_HINTS_PATH)
    )
    return include_bound_official_applied_event_candidates(
        build_lifecycle_candidates(repository, release=release),
        repository,
        release,
        specs,
    )


def _report_binding(
    report: Mapping[str, Any],
    release: DataRelease,
    candidates: tuple[LifecycleCandidate, ...],
    *,
    hints_path: Path | None = None,
) -> dict[str, Any]:
    return build_lifecycle_report_binding(
        release_version=release.version,
        completed_session=release.completed_session,
        dataset_versions=release.dataset_versions,
        candidates=candidates,
        hints_path=(
            _workspace_path(LIFECYCLE_HINTS_PATH)
            if hints_path is None
            else hints_path
        ),
        sec_fetch_policy=report["sec_fetch_policy"],
        sec_max_http_attempts=report["sec_max_http_attempts"],
        sec_max_http_attempts_per_candidate=report[
            "sec_max_http_attempts_per_candidate"
        ],
        sec_max_http_attempts_per_request=report[
            "sec_max_http_attempts_per_request"
        ],
        sec_http_attempts=report["sec_http_attempts"],
        sec_http_attempts_by_candidate=report["sec_http_attempts_by_candidate"],
    )


def _legacy_lifecycle_hints_bytes(current: bytes) -> bytes:
    _require(
        sha256_bytes(current) == EXPECTED_CURRENT_LIFECYCLE_HINTS_SHA256,
        "Current lifecycle hints bytes changed outside the reviewed UTX delta.",
    )
    _require(
        current.count(UTX_HINT_ADDITIVE_BLOCK) == 1,
        "Reviewed UTX lifecycle hint block is not present exactly once.",
    )
    legacy = current.replace(UTX_HINT_ADDITIVE_BLOCK, b"", 1)
    _require(
        sha256_bytes(legacy) == EXPECTED_BASE_LIFECYCLE_HINTS_SHA256,
        "Removing the exact UTX hint delta did not reproduce report-time hints.",
    )
    return legacy


@contextmanager
def _legacy_lifecycle_hints_path():
    current = _workspace_path(LIFECYCLE_HINTS_PATH).read_bytes()
    legacy = _legacy_lifecycle_hints_bytes(current)
    with tempfile.TemporaryDirectory(prefix="stq-legacy-lifecycle-hints-") as temp_dir:
        path = Path(temp_dir) / "us_lifecycle_hints.yaml"
        path.write_bytes(legacy)
        _require(
            sha256_bytes(path.read_bytes()) == EXPECTED_BASE_LIFECYCLE_HINTS_SHA256,
            "Temporary report-time lifecycle hints changed.",
        )
        yield path


def _validate_report_candidate_inventory(
    report: Mapping[str, Any],
    candidates: tuple[LifecycleCandidate, ...],
) -> None:
    records = report.get("records")
    _require(isinstance(records, dict), "Lifecycle report records are missing.")
    expected = {candidate.security_id: candidate for candidate in candidates}
    _require(
        set(records) == set(expected),
        "Lifecycle report candidate SID inventory changed.",
    )
    for security_id, candidate in expected.items():
        record = records[security_id]
        _require(
            isinstance(record, dict)
            and record.get("candidate") == _candidate_dict(candidate),
            f"Lifecycle report candidate identity changed: {security_id}.",
        )
    summary = report.get("summary")
    _require(isinstance(summary, dict), "Lifecycle report summary is missing.")
    count = len(expected)
    eligible = sum(bool(record.get("eligible_for_apply")) for record in records.values())
    _require(
        int(summary.get("candidate_count", -1)) == count
        and int(summary.get("collected_count", -1)) == count
        and int(summary.get("eligible_count", -1)) == eligible
        and int(summary.get("unresolved_count", -1)) == count - eligible,
        "Lifecycle report summary is stale.",
    )


def _refresh_report_summary(report: dict[str, Any]) -> None:
    records = list(report["records"].values())
    counts: dict[str, int] = {}
    for record in records:
        action_type = _text((record.get("parsed") or {}).get("action_type"))
        key = action_type or "unresolved"
        counts[key] = counts.get(key, 0) + 1
    eligible = sum(bool(record.get("eligible_for_apply")) for record in records)
    summary = dict(report.get("summary") or {})
    summary.update(
        {
            "candidate_count": len(records),
            "collected_count": len(records),
            "eligible_count": eligible,
            "unresolved_count": len(records) - eligible,
            "action_type_counts": dict(sorted(counts.items())),
            "sec_fetch_policy": report["sec_fetch_policy"],
            "sec_max_http_attempts": int(report["sec_max_http_attempts"]),
            "sec_max_http_attempts_per_candidate": int(
                report["sec_max_http_attempts_per_candidate"]
            ),
            "sec_max_http_attempts_per_request": int(
                report["sec_max_http_attempts_per_request"]
            ),
            "sec_http_attempts": int(report["sec_http_attempts"]),
            "sec_http_attempts_remaining": int(report["sec_max_http_attempts"])
            - int(report["sec_http_attempts"]),
            "sec_http_attempts_by_candidate": dict(
                report["sec_http_attempts_by_candidate"]
            ),
        }
    )
    official = report.get("official_exception_evidence") or {}
    if official:
        statuses: dict[str, int] = {}
        for value in official.values():
            status = _text((value or {}).get("status")) or "invalid"
            statuses[status] = statuses.get(status, 0) + 1
        summary["official_exception_evidence"] = {
            "evidence_count": len(official),
            "status_counts": dict(sorted(statuses.items())),
            "http_attempts": int(
                report.get("official_exception_evidence_http_attempts", 0)
            ),
        }
    else:
        summary.pop("official_exception_evidence", None)
    report["summary"] = summary


def _validate_utx_report_evidence(report: Mapping[str, Any]) -> Mapping[str, Any]:
    security_id = next(case.security_id for case in CASES if case.symbol == "UTX")
    records = report.get("records") or {}
    utx = records.get(security_id) if isinstance(records, Mapping) else None
    _require(isinstance(utx, Mapping), "UTX lifecycle report record is missing.")
    crosscheck = utx.get("crosscheck") or {}
    _require(
        utx.get("eligible_for_apply") is False
        and crosscheck.get("passed") is False
        and crosscheck.get("date_passed") is True
        and crosscheck.get("economic_terms_passed") is False
        and crosscheck.get("old_price_session") == "2020-04-02"
        and float(crosscheck.get("old_close")) == 86.01
        and crosscheck.get("successor_price_session") == "2020-04-03"
        and float(crosscheck.get("successor_close")) == 49.93,
        "UTX lifecycle report no longer fails the raw-price economic check.",
    )
    official = (report.get("official_exception_evidence") or {}).get(
        "utx_2020_carr_otis_distributions"
    )
    _require(
        isinstance(official, Mapping)
        and official.get("status") == "verified_pinned_attached"
        and official.get("candidate_security_id") == security_id
        and official.get("candidate_last_price_date") == "2020-04-02"
        and official.get("exception_code") == "unsupported_consideration"
        and official.get("claim") == UTX_EXCEPTION_REASON
        and official.get("source_url") == UTX_DISTRIBUTION_SOURCE_URL
        and official.get("observed_sha256") == UTX_DISTRIBUTION_SOURCE_HASH
        and official.get("pinned_sha256") == UTX_DISTRIBUTION_SOURCE_HASH
        and int(official.get("content_bytes", -1)) == UTX_DISTRIBUTION_SOURCE_BYTES,
        "UTX official exception evidence report binding changed.",
    )
    artifacts = [
        value
        for value in (utx.get("artifacts") or ())
        if isinstance(value, Mapping)
        and value.get("source_hash") == UTX_DISTRIBUTION_SOURCE_HASH
    ]
    _require(
        len(artifacts) == 1
        and artifacts[0].get("source") == "sec_edgar_filing"
        and artifacts[0].get("source_url") == UTX_DISTRIBUTION_SOURCE_URL
        and artifacts[0].get("retrieved_at") == UTX_DISTRIBUTION_RETRIEVED_AT
        and artifacts[0].get("content_type") == "text/plain",
        "UTX official report artifact binding changed.",
    )
    official_summary = (report.get("summary") or {}).get(
        "official_exception_evidence"
    ) or {}
    all_official = report.get("official_exception_evidence") or {}
    status_counts: dict[str, int] = {}
    for value in all_official.values():
        status = _text((value or {}).get("status")) or "invalid"
        status_counts[status] = status_counts.get(status, 0) + 1
    _require(
        official_summary.get("evidence_count") == len(all_official)
        and official_summary.get("status_counts") == dict(sorted(status_counts.items()))
        and int(official_summary.get("http_attempts", -1))
        == int(report.get("official_exception_evidence_http_attempts", 0)),
        "UTX report official-evidence summary is stale.",
    )
    return utx


def _load_and_validate_base_lifecycle_report(
    release: DataRelease,
    candidates: tuple[LifecycleCandidate, ...],
) -> dict[str, Any]:
    path = _workspace_path(BASE_LIFECYCLE_REPORT_PATH)
    _require(path.is_file(), f"Pinned lifecycle report is missing: {path}.")
    content = path.read_bytes()
    _require(
        sha256_bytes(content) == EXPECTED_BASE_LIFECYCLE_REPORT_SHA256,
        "Pinned lifecycle report bytes changed.",
    )
    report = json.loads(content)
    with _legacy_lifecycle_hints_path() as hints_path:
        binding = _report_binding(
            report,
            release,
            candidates,
            hints_path=hints_path,
        )
    _require(
        binding["collector_config_sha256"]
        == EXPECTED_BASE_LIFECYCLE_COLLECTOR_CONFIG_SHA256
        and binding["collection_context_sha256"]
        == EXPECTED_BASE_LIFECYCLE_COLLECTION_CONTEXT_SHA256,
        "Report-time lifecycle collection provenance changed.",
    )
    validate_lifecycle_report_binding(
        report, binding, purpose="identity-tail base-report verification"
    )
    _validate_report_candidate_inventory(report, candidates)
    _require(
        report["candidate_set_sha256"]
        == EXPECTED_BASE_LIFECYCLE_CANDIDATE_SET_SHA256,
        "Pinned lifecycle candidate-set hash changed.",
    )
    return report


def _validate_utx_distribution_archive(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
) -> dict[str, Any]:
    rows = archive.loc[
        archive["archive_id"].map(_text).eq(UTX_DISTRIBUTION_SOURCE_HASH)
        & archive["source_hash"].map(_text).eq(UTX_DISTRIBUTION_SOURCE_HASH)
        & archive["source_url"].map(_text).eq(UTX_DISTRIBUTION_SOURCE_URL)
    ]
    _require(len(rows) == 1, "UTX distribution SEC archive binding changed.")
    row = rows.iloc[0]
    expected_path = (
        "archives/2026-07-15/"
        f"{UTX_DISTRIBUTION_SOURCE_HASH}.txt.gz"
    )
    _require(
        _text(row.get("dataset")) == "sec_edgar_filing"
        and _text(row.get("source")) == "sec_edgar_filing"
        and _text(row.get("object_path")) == expected_path
        and _text(row.get("content_type")) == "text/plain"
        and _date(row.get("effective_date")) == "2026-07-15"
        and _text(row.get("retrieved_at")) == UTX_DISTRIBUTION_RETRIEVED_AT,
        "UTX distribution SEC archive provenance changed.",
    )
    path = _safe_path(repository.root, expected_path)
    _require(path.is_file(), "UTX distribution SEC archive payload is missing.")
    try:
        content = gzip.decompress(path.read_bytes())
    except Exception as exc:
        raise RuntimeError("UTX distribution SEC archive is invalid gzip.") from exc
    _require(
        len(content) == UTX_DISTRIBUTION_SOURCE_BYTES
        and sha256_bytes(content) == UTX_DISTRIBUTION_SOURCE_HASH,
        "UTX distribution SEC archive bytes changed.",
    )
    specs = load_official_lifecycle_exception_evidence(
        _workspace_path(LIFECYCLE_HINTS_PATH)
    )
    spec = specs.get("utx_2020_carr_otis_distributions")
    _require(
        spec is not None
        and spec.candidate_security_ids
        == ("US:EODHD:aefd1dd7-529d-5b6a-80e9-65d0e14102a6",)
        and spec.candidate_last_price_dates == ("2020-04-02",)
        and spec.exception_code == "unsupported_consideration"
        and spec.claim == UTX_EXCEPTION_REASON
        and spec.source_url == UTX_DISTRIBUTION_SOURCE_URL
        and spec.source_sha256 == UTX_DISTRIBUTION_SOURCE_HASH,
        "UTX permanent-exception evidence registry changed.",
    )
    matched = validate_official_evidence_content(spec, content)
    return {
        "artifact": {
            "source": "sec_edgar_filing",
            "source_url": UTX_DISTRIBUTION_SOURCE_URL,
            "retrieved_at": UTX_DISTRIBUTION_RETRIEVED_AT,
            "content_type": "text/plain",
            "source_hash": UTX_DISTRIBUTION_SOURCE_HASH,
        },
        "official_report_entry": {
            "action_type": "",
            "candidate_binding_status": "bound",
            "candidate_last_price_date": "2020-04-02",
            "candidate_security_id": (
                "US:EODHD:aefd1dd7-529d-5b6a-80e9-65d0e14102a6"
            ),
            "candidate_symbol": "UTX",
            "cash_amount": None,
            "claim": UTX_EXCEPTION_REASON,
            "content_bytes": UTX_DISTRIBUTION_SOURCE_BYTES,
            "content_type": "text/plain",
            "effective_date": "2020-04-03",
            "exception_code": "unsupported_consideration",
            "matched_phrases": list(matched),
            "observed_sha256": UTX_DISTRIBUTION_SOURCE_HASH,
            "pinned_sha256": UTX_DISTRIBUTION_SOURCE_HASH,
            "resolution_kind": "exception",
            "retrieved_at": UTX_DISTRIBUTION_RETRIEVED_AT,
            "source_url": UTX_DISTRIBUTION_SOURCE_URL,
            "status": "verified_pinned_attached",
        },
        "content_bytes": len(content),
        "object_path": expected_path,
        "matched_phrases": list(matched),
    }


def _rebind_lifecycle_report(
    base_report: Mapping[str, Any],
    *,
    planned_release: DataRelease,
    base_candidates: tuple[LifecycleCandidate, ...],
    repaired_candidates: tuple[LifecycleCandidate, ...],
    repaired_prices: pd.DataFrame,
    utx_distribution_evidence: Mapping[str, Any],
) -> bytes:
    report = json.loads(json.dumps(base_report, ensure_ascii=False))
    before = {candidate.security_id: candidate for candidate in base_candidates}
    after = {candidate.security_id: candidate for candidate in repaired_candidates}
    _require(set(before) == set(after), "Lifecycle candidate SID inventory changed.")
    repaired_ids = {case.security_id for case in CASES}
    changed_ids = {
        security_id
        for security_id in before
        if _candidate_dict(before[security_id]) != _candidate_dict(after[security_id])
    }
    _require(
        changed_ids == repaired_ids,
        "Lifecycle candidate migration is not exactly the seven reviewed identities.",
    )
    price_histories = _build_price_histories(repaired_prices)
    for case in CASES:
        old = _candidate_dict(before[case.security_id])
        new = _candidate_dict(after[case.security_id])
        old_without_boundary = {
            key: value
            for key, value in old.items()
            if key not in {"last_price_date", "active_to"}
        }
        new_without_boundary = {
            key: value
            for key, value in new.items()
            if key not in {"last_price_date", "active_to"}
        }
        _require(
            old_without_boundary == new_without_boundary
            and new["last_price_date"] == case.old_last_good_session
            and new["active_to"] == case.old_last_good_session,
            f"{case.symbol} lifecycle candidate migration changed extra fields.",
        )
        record = report["records"][case.security_id]
        old_crosscheck = dict(record.get("crosscheck") or {})
        record["candidate"] = new
        record["crosscheck"] = _crosscheck(
            record,
            successor_security_id=_text(record.get("successor_security_id")),
            price_histories=price_histories,
        )
        if case.symbol == "UTX":
            check = record["crosscheck"]
            _require(
                check.get("passed") is False
                and check.get("date_passed") is True
                and check.get("economic_terms_passed") is False
                and check.get("basis") == "eodhd_terminal_price+index_remove"
                and check.get("old_price_session") == "2020-04-02"
                and float(check.get("old_close")) == 86.01
                and check.get("successor_price_session") == "2020-04-03"
                and float(check.get("successor_close")) == 49.93
                and float(check.get("relative_deviation")) > 0.20,
                "UTX projected collector mismatch changed.",
            )
            # The old synthetic 49.93 tail made this raw-level check pass.  The
            # repaired report must state the real mismatch; the already reviewed
            # action/resolution is preserved separately by the lifecycle gate.
            record["eligible_for_apply"] = False
            artifact = dict(utx_distribution_evidence["artifact"])
            artifacts = list(record.get("artifacts") or ())
            _require(
                not any(
                    _text(item.get("source_hash"))
                    == UTX_DISTRIBUTION_SOURCE_HASH
                    for item in artifacts
                    if isinstance(item, dict)
                ),
                "UTX distribution artifact is already present in the stale report.",
            )
            artifacts.append(artifact)
            record["artifacts"] = artifacts
            official = report.setdefault("official_exception_evidence", {})
            _require(
                "utx_2020_carr_otis_distributions" not in official,
                "UTX official exception evidence already exists in the stale report.",
            )
            official["utx_2020_carr_otis_distributions"] = dict(
                utx_distribution_evidence["official_report_entry"]
            )
        else:
            _require(
                record["crosscheck"].get("passed") is True
                and record["crosscheck"].get("date_passed") is True
                and record["crosscheck"].get("economic_terms_passed") is True,
                f"{case.symbol} projected collector cross-check did not pass.",
            )
        _require(
            record["crosscheck"] != old_crosscheck,
            f"{case.symbol} projected collector cross-check remained stale.",
        )
    binding = _report_binding(report, planned_release, repaired_candidates)
    for field in REPORT_BINDING_FIELDS:
        report[field] = binding[field]
    _refresh_report_summary(report)
    _validate_utx_report_evidence(report)
    validate_lifecycle_report_binding(
        report, binding, purpose="identity-tail projected-report verification"
    )
    _validate_report_candidate_inventory(report, repaired_candidates)
    _require(
        report["candidate_set_sha256"]
        == EXPECTED_REPAIRED_LIFECYCLE_CANDIDATE_SET_SHA256,
        "Repaired lifecycle candidate-set hash changed.",
    )
    output = _json_document_bytes(report)
    _require(
        sha256_bytes(output) != EXPECTED_BASE_LIFECYCLE_REPORT_SHA256,
        "Lifecycle evidence report was not freshly rebound.",
    )
    return output


def _migrate_lifecycle_resolutions(
    current: pd.DataFrame,
    *,
    candidates: tuple[LifecycleCandidate, ...],
    actions: pd.DataFrame,
    completed_session: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require(
        lifecycle_resolution_set_sha256(current)
        == EXPECTED_BASE_LIFECYCLE_RESOLUTION_SET_SHA256,
        "Pinned lifecycle resolution inventory changed.",
    )
    output = current.copy(deep=True)
    for case in CASES:
        rows = output.index[output["security_id"].map(_text).eq(case.security_id)]
        _require(len(rows) == 1, f"{case.symbol} lifecycle resolution changed.")
        row = rows[0]
        old_row = output.loc[row].copy(deep=True)
        _require(
            _date(old_row["last_price_date"]) == case.tail_end
            and _text(old_row["candidate_id"])
            == lifecycle_candidate_id(case.security_id, case.tail_end),
            f"{case.symbol} baseline lifecycle resolution binding changed.",
        )
        output.loc[row, "candidate_id"] = lifecycle_candidate_id(
            case.security_id, case.old_last_good_session
        )
        output.loc[row, "last_price_date"] = case.old_last_good_session
        if case.symbol == "UTX":
            output.loc[row, "resolution"] = "exception"
            output.loc[row, "event_id"] = ""
            output.loc[row, "exception_code"] = "unsupported_consideration"
            output.loc[row, "exception_reason"] = UTX_EXCEPTION_REASON
            output.loc[row, "reviewed_by"] = "us_lifecycle_finalizer_v1"
            output.loc[row, "reviewed_at"] = UTX_EXCEPTION_REVIEWED_AT
            output.loc[row, "recheck_after"] = UTX_EXCEPTION_RECHECK_AFTER
            output.loc[row, "successor_security_id"] = ""
            output.loc[row, "successor_symbol"] = ""
            output.loc[row, "source"] = "sec_edgar_filing"
            output.loc[row, "source_url"] = UTX_DISTRIBUTION_SOURCE_URL
            output.loc[row, "retrieved_at"] = UTX_DISTRIBUTION_RETRIEVED_AT
            output.loc[row, "source_hash"] = UTX_DISTRIBUTION_SOURCE_HASH
            expected = {
                "candidate_id": lifecycle_candidate_id(
                    case.security_id, case.old_last_good_session
                ),
                "last_price_date": case.old_last_good_session,
                "resolution": "exception",
                "event_id": "",
                "exception_code": "unsupported_consideration",
                "exception_reason": UTX_EXCEPTION_REASON,
                "reviewed_by": "us_lifecycle_finalizer_v1",
                "reviewed_at": UTX_EXCEPTION_REVIEWED_AT,
                "recheck_after": UTX_EXCEPTION_RECHECK_AFTER,
                "successor_security_id": "",
                "successor_symbol": "",
                "source_url": UTX_DISTRIBUTION_SOURCE_URL,
                "source": "sec_edgar_filing",
                "retrieved_at": UTX_DISTRIBUTION_RETRIEVED_AT,
                "source_hash": UTX_DISTRIBUTION_SOURCE_HASH,
            }
            _require(
                all(_text(output.loc[row, key]) == _text(value) for key, value in expected.items()),
                "UTX fail-closed lifecycle exception changed.",
            )
        else:
            unchanged_columns = [
                column
                for column in output.columns
                if column not in {"candidate_id", "last_price_date"}
            ]
            _require(
                output.loc[row, unchanged_columns].equals(old_row[unchanged_columns]),
                f"{case.symbol} lifecycle resolution changed non-key fields.",
            )
    resolution_sha = lifecycle_resolution_set_sha256(output)
    _require(
        resolution_sha == EXPECTED_REPAIRED_LIFECYCLE_RESOLUTION_SET_SHA256,
        "Repaired lifecycle resolution-set hash changed.",
    )
    coverage = validate_lifecycle_coverage(
        _candidate_frame(candidates),
        output,
        actions,
        completed_session=completed_session,
    )
    metadata = coverage.manifest_metadata()
    _require(
        coverage.valid
        and all(metadata[key] == value for key, value in EXPECTED_LIFECYCLE_COVERAGE.items())
        and metadata["candidate_set_sha256"]
        == EXPECTED_REPAIRED_LIFECYCLE_CANDIDATE_SET_SHA256
        and metadata["resolution_set_sha256"]
        == EXPECTED_REPAIRED_LIFECYCLE_RESOLUTION_SET_SHA256,
        "Repaired lifecycle coverage gate did not close exactly.",
    )
    return output.reset_index(drop=True), metadata


def _validate_utx_unmodeled_distribution_gap(
    repaired_prices: pd.DataFrame,
    distribution_prices: pd.DataFrame,
    actions: pd.DataFrame,
) -> dict[str, Any]:
    case = next(item for item in CASES if item.symbol == "UTX")
    expected = {
        "UTX": {
            "security_id": case.security_id,
            "session": "2020-04-02",
            "ohlcv": (89.63, 92.25, 85.11, 86.01, 13_275_777.0),
            "source_hash": (
                "979bf4b38ed0292cb30e2b3f23018dcfec16c33c36238375a4665d03abdcedba"
            ),
            "weight": 1.0,
        },
        "RTX": {
            "security_id": case.successor_security_id,
            "session": "2020-04-03",
            "ohlcv": (51.0, 53.3, 48.05, 49.93, 18_932_350.0),
            "source_hash": (
                "e7dc3c9e3755ef02c551ca117240511fe47a74b54f1e67b0ea6acd1e4944a78a"
            ),
            "weight": 1.0,
        },
        "CARR": {
            "security_id": UTX_CARR_SECURITY_ID,
            "session": "2020-04-03",
            "ohlcv": (13.75, 17.0, 13.38, 16.92, 66_934_300.0),
            "source_hash": (
                "bd34ce991f534a57158045cc6d41725144d18227c7ca51c4bb76acf820bc8dde"
            ),
            "weight": 1.0,
        },
        "OTIS": {
            "security_id": UTX_OTIS_SECURITY_ID,
            "session": "2020-04-03",
            "ohlcv": (43.75, 49.3, 41.8, 47.32, 22_551_600.0),
            "source_hash": (
                "35e974f1bf433d32ba521562a88518858febdac8730d46a9fa5abfed3da59a71"
            ),
            "weight": 0.5,
        },
    }
    combined = pd.concat(
        [repaired_prices, distribution_prices], ignore_index=True, sort=False
    )
    observed: dict[str, dict[str, Any]] = {}
    for symbol, spec in expected.items():
        rows = combined.loc[
            combined["security_id"].map(_text).eq(spec["security_id"])
            & combined["session"].map(_date).eq(spec["session"])
        ]
        _require(len(rows) == 1, f"UTX distribution boundary changed: {symbol}.")
        row = rows.iloc[0]
        ohlcv = tuple(float(row[field]) for field in ("open", "high", "low", "close", "volume"))
        _require(
            ohlcv == spec["ohlcv"]
            and _text(row.get("source")) == "eodhd_eod"
            and _text(row.get("source_hash")) == spec["source_hash"],
            f"UTX distribution price/provenance changed: {symbol}.",
        )
        observed[symbol] = {
            "security_id": spec["security_id"],
            "session": spec["session"],
            "close": ohlcv[3],
            "weight": spec["weight"],
            "source_hash": spec["source_hash"],
        }
    own_actions = actions.loc[actions["security_id"].map(_text).eq(case.security_id)]
    lifecycle_types = own_actions["action_type"].map(_text).str.lower()
    _require(
        not lifecycle_types.isin({"spinoff", "spin_off", "spin-off"}).any(),
        "UTX canonical action inventory unexpectedly contains a spin-off.",
    )
    ticker = own_actions.loc[own_actions["event_id"].map(_text).eq(case.event_id)]
    _require(
        len(ticker) == 1
        and _text(ticker.iloc[0].get("source_hash")) == case.official_source_hash,
        "UTX official ticker-change evidence changed.",
    )
    modeled_value = (
        observed["RTX"]["close"]
        + observed["CARR"]["close"]
        + 0.5 * observed["OTIS"]["close"]
    )
    _require(abs(modeled_value - 90.51) < 1e-12, "UTX distribution value pin changed.")
    return {
        "status": "fail_closed_unmodeled_spinoff_consideration",
        "canonical_spinoff_action_count": 0,
        "pre_distribution_close": observed["UTX"]["close"],
        "raw_rtx_close": observed["RTX"]["close"],
        "carr_close": observed["CARR"]["close"],
        "otis_close": observed["OTIS"]["close"],
        "known_distribution_weights": {"CARR": 1.0, "OTIS": 0.5},
        "illustrative_combined_value": modeled_value,
        "prices": observed,
        "official_ticker_source_hash": case.official_source_hash,
    }


def _append_lifecycle_report_archive(
    current: pd.DataFrame,
    *,
    report_bytes: bytes,
    planned_release: DataRelease,
) -> tuple[pd.DataFrame, str, str]:
    report_sha = sha256_bytes(report_bytes)
    object_path = (
        f"archives/{planned_release.completed_session}/{report_sha}.json.gz"
    )
    matches = current.loc[current["archive_id"].map(_text).eq(report_sha)]
    _require(matches.empty, "Fresh lifecycle report archive_id already exists.")
    row = {column: "" for column in current.columns}
    row.update(
        {
            "archive_id": report_sha,
            "dataset": LIFECYCLE_REPORT_SOURCE,
            "object_path": object_path,
            "content_type": "application/json",
            "effective_date": planned_release.completed_session,
            "source": LIFECYCLE_REPORT_SOURCE,
            "source_url": f"file://{BASE_LIFECYCLE_REPORT_PATH}",
            "retrieved_at": planned_release.created_at,
            "source_hash": report_sha,
        }
    )
    output = pd.concat([current, pd.DataFrame([row])], ignore_index=True, sort=False)
    _require(
        len(output) == len(current) + 1
        and not output.duplicated("archive_id").any(),
        "Lifecycle report archive append changed extra rows.",
    )
    return output.reset_index(drop=True), report_sha, object_path


def _normalized_factor_economics(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[
        ["security_id", "session", "split_factor", "total_return_factor"]
    ].copy()
    output["security_id"] = output["security_id"].map(_text)
    output["session"] = pd.to_datetime(output["session"], errors="raise").dt.normalize()
    _require(
        not output.duplicated(["security_id", "session"]).any(),
        "Adjustment-factor keys are duplicated.",
    )
    return output.sort_values(["security_id", "session"], ignore_index=True)


def _affected_factor_economics_sha256(frame: pd.DataFrame) -> str:
    security_ids = {
        security_id
        for case in CASES
        for security_id in (case.security_id, case.successor_security_id)
    }
    selected = frame.loc[frame["security_id"].map(_text).isin(security_ids)]
    return _frame_sha256(
        selected,
        columns=(
            "security_id",
            "session",
            "split_factor",
            "total_return_factor",
        ),
        sort_by=("security_id", "session"),
    )


def _factor_economic_change_count(
    current: pd.DataFrame,
    rebuilt: pd.DataFrame,
    removed_keys: set[tuple[str, pd.Timestamp]],
) -> int:
    old = _normalized_factor_economics(current)
    if removed_keys:
        old_keys = list(zip(old["security_id"], old["session"]))
        old = old.loc[[key not in removed_keys for key in old_keys]].reset_index(drop=True)
    new = _normalized_factor_economics(rebuilt)
    _require(
        len(old) == len(new)
        and old[["security_id", "session"]].equals(new[["security_id", "session"]]),
        "Rebuilt factor keys do not exactly match retained factor keys.",
    )
    changed = np.zeros(len(old), dtype=bool)
    for column in ("split_factor", "total_return_factor"):
        left = pd.to_numeric(old[column], errors="raise").to_numpy(dtype=float)
        right = pd.to_numeric(new[column], errors="raise").to_numpy(dtype=float)
        changed |= ~((left == right) | (np.isnan(left) & np.isnan(right)))
    return int(changed.sum())


def _rewrite_factors_minimal(
    current: pd.DataFrame,
    *,
    source_version: str,
    copy_frame: bool = True,
) -> tuple[pd.DataFrame, int]:
    _require(
        _affected_factor_economics_sha256(current)
        == EXPECTED_OLD_AFFECTED_FACTOR_ECONOMICS_SHA256,
        "Affected baseline factor economics changed.",
    )
    factor_keys = set(
        zip(
            current["security_id"].map(_text),
            pd.to_datetime(current["session"], errors="raise").dt.normalize(),
        )
    )
    expected_removed = {
        (case.security_id, pd.Timestamp(value).normalize())
        for case in CASES
        for value in _session_series(
            pd.DataFrame(
                {"session": current.loc[current["security_id"].map(_text).eq(case.security_id), "session"]}
            )
        )
        if value >= case.transition_date
    }
    removed_keys = factor_keys & expected_removed
    _require(
        removed_keys == expected_removed
        and len(removed_keys) == EXPECTED_REMOVED_PRICE_ROWS,
        "Rebuilt factor deletion inventory changed.",
    )
    key_series = list(
        zip(
            current["security_id"].map(_text),
            pd.to_datetime(current["session"], errors="raise").dt.normalize(),
        )
    )
    keep = [key not in removed_keys for key in key_series]
    if copy_frame:
        retained = current.loc[keep].copy(deep=True).reset_index(drop=True)
    else:
        current.drop(index=current.index[[not value for value in keep]], inplace=True)
        current.reset_index(drop=True, inplace=True)
        retained = current
    # The price/action economics are intentionally preserved: deleting COG's
    # post-transition provider tail and then running the generic factor builder
    # would retroactively remove a dividend factor from 1,700 valid pre-change
    # sessions.  The reviewed minimum repair therefore deletes only the exact
    # 616 invalid factor keys and refreshes content-bound lineage; every
    # retained split/total-return value remains byte-for-byte numeric-equal.
    economic_before = _normalized_factor_economics(current)
    economic_before = economic_before.loc[
        [
            key not in removed_keys
            for key in zip(
                economic_before["security_id"], economic_before["session"]
            )
        ]
    ].reset_index(drop=True)
    economic_after = _normalized_factor_economics(retained)
    _require(
        economic_before.equals(economic_after),
        "Minimal factor deletion changed retained factor economics.",
    )
    retained["source_version"] = source_version
    retained["calculated_at"] = REPAIR_REVIEWED_AT
    retained["source"] = "derived"
    retained["retrieved_at"] = REPAIR_REVIEWED_AT
    retained["source_hash"] = source_version
    _require(
        _affected_factor_economics_sha256(retained)
        == EXPECTED_REPAIRED_AFFECTED_FACTOR_ECONOMICS_SHA256,
        "Affected repaired factor economics changed.",
    )
    return retained, 0


def _prepare_factors(
    current: pd.DataFrame,
    current_prices: pd.DataFrame,
    repaired_prices: pd.DataFrame,
    *,
    source_version: str,
) -> tuple[pd.DataFrame, int]:
    current_factor_keys = set(
        zip(
            current["security_id"].map(_text),
            pd.to_datetime(current["session"], errors="raise").dt.normalize(),
        )
    )
    current_price_keys = set(
        zip(
            current_prices["security_id"].map(_text),
            pd.to_datetime(
                current_prices["session"], errors="raise"
            ).dt.normalize(),
        )
    )
    _require(
        current_factor_keys == current_price_keys,
        "Current factor keys do not exactly match current price keys.",
    )
    retained, changed = _rewrite_factors_minimal(
        current, source_version=source_version
    )
    repaired_factor_keys = set(
        zip(
            retained["security_id"].map(_text),
            pd.to_datetime(retained["session"], errors="raise").dt.normalize(),
        )
    )
    repaired_price_keys = set(
        zip(
            repaired_prices["security_id"].map(_text),
            pd.to_datetime(
                repaired_prices["session"], errors="raise"
            ).dt.normalize(),
        )
    )
    _require(
        repaired_factor_keys == repaired_price_keys,
        "Minimally retained factor keys do not match repaired prices.",
    )
    return retained, changed


def _hcp_signal_impact(
    old_prices: pd.DataFrame,
    new_prices: pd.DataFrame,
    old_factors: pd.DataFrame,
    new_factors: pd.DataFrame,
) -> dict[str, Any]:
    case = next(case for case in CASES if case.symbol == "HCP")
    sid = case.successor_security_id
    before_price = old_prices.loc[
        old_prices["security_id"].map(_text).eq(sid)
    ].sort_values("session", kind="stable")
    after_price = new_prices.loc[
        new_prices["security_id"].map(_text).eq(sid)
    ].sort_values("session", kind="stable")
    before_factor = old_factors.loc[
        old_factors["security_id"].map(_text).eq(sid)
    ].sort_values("session", kind="stable")
    after_factor = new_factors.loc[
        new_factors["security_id"].map(_text).eq(sid)
    ].sort_values("session", kind="stable")
    result: dict[str, Any] = {}
    for mode in ("raw", "total_return_adjusted"):
        diff = _signal_diff(
            _signal_frame(before_price, before_factor, mode=mode),
            _signal_frame(after_price, after_factor, mode=mode),
        )
        _require(
            all(diff[column]["count"] == 0 for column in SIGNAL_COLUMNS),
            f"HCP/PEAK replacement changed {mode} Triple Supertrend state.",
        )
        result[mode] = diff
    return result


def _validate_known_snapshot_gap_pin() -> bool:
    """Verify the one pre-existing NBL gap without materializing the snapshot."""

    expected = EXPECTED_SNAPSHOT_IDENTITY_GAP
    fingerprint = index_member_identity_gap_fingerprint(
        index_id=expected["index_id"],
        replay_date=expected["replay_date"],
        security_id=expected["security_id"],
        next_remove_event_id=expected["next_remove_event_id"],
        next_remove_effective_date=expected["next_remove_effective_date"],
        next_remove_source=expected["next_remove_source"],
        next_remove_source_hash=expected["next_remove_source_hash"],
    )
    _require(fingerprint == expected["fingerprint"], "NBL gap pin changed.")
    return True


def _validate_input_manifests(
    repository: LocalDatasetRepository, release: DataRelease
) -> None:
    """Hash-check immutable input files without loading their rows into pandas."""

    for dataset in REQUIRED_DATASETS:
        manifest = repository.current_manifest(dataset)
        _require(
            manifest is not None
            and manifest.version == release.dataset_versions[dataset],
            f"Current {dataset} manifest changed.",
        )


def _pointer_etags(
    repository: LocalDatasetRepository, release: DataRelease
) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for dataset in sorted(release.dataset_versions):
        pointer, etag = repository.current_pointer(dataset)
        _require(
            pointer is not None
            and pointer.version == release.dataset_versions.get(dataset),
            f"Release/current pointer mismatch: {dataset}.",
        )
        output[dataset] = etag
    _require(
        set(output) == set(release.dataset_versions),
        "Release pointer inventory capture is incomplete.",
    )
    return output


def _verify_old_release(
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    _require(release.version == PINNED_RELEASE_VERSION, "Base release pin changed.")
    for dataset, version in PINNED_DATASET_VERSIONS.items():
        _require(
            release.dataset_versions.get(dataset) == version,
            f"Base {dataset} version pin changed.",
        )
    audit = build_audit(repository)
    selected = [case for case in audit["cases"] if case["symbol"] in EXPECTED_SYMBOLS]
    _require(len(selected) == 7, "Pinned audit no longer contains exactly seven cases.")
    _old_state(frames)
    return audit


def _exact_repair_manifests(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> bool:
    if UTX_RELEASE_WARNING not in release.warnings:
        return False
    report_hashes: set[str] = set()
    for dataset in WRITE_DATASETS:
        manifest = repository.manifest_for_version(
            dataset, release.dataset_versions.get(dataset, "")
        )
        metadata = manifest.metadata
        report_hash = _text(metadata.get("lifecycle_evidence_report_sha256"))
        report_hashes.add(report_hash)
        if not (
            metadata.get("operation") == OPERATION
            and metadata.get("schema") == REPAIR_SCHEMA
            and metadata.get("input_release_version") == PINNED_RELEASE_VERSION
            and metadata.get("registry_inventory_sha256")
            == TRUSTED_REGISTRY_INVENTORY_SHA256
            and metadata.get("registry") == repair_registry()
            and metadata.get("candidate_content_sha256")
            == EXPECTED_CANDIDATE_CONTENT_SHA256
            and metadata.get("lifecycle_candidate_set_sha256")
            == EXPECTED_REPAIRED_LIFECYCLE_CANDIDATE_SET_SHA256
            and metadata.get("lifecycle_resolution_set_sha256")
            == EXPECTED_REPAIRED_LIFECYCLE_RESOLUTION_SET_SHA256
            and metadata.get("lifecycle_coverage")
            and all(
                metadata["lifecycle_coverage"].get(key) == value
                for key, value in EXPECTED_LIFECYCLE_COVERAGE.items()
            )
            and all(
                metadata.get(key) == value
                for key, value in {
                    **EXPECTED_LIFECYCLE_COVERAGE,
                    "candidate_set_sha256": (
                        EXPECTED_REPAIRED_LIFECYCLE_CANDIDATE_SET_SHA256
                    ),
                    "resolution_set_sha256": (
                        EXPECTED_REPAIRED_LIFECYCLE_RESOLUTION_SET_SHA256
                    ),
                }.items()
            )
            and metadata.get("evidence_report_sha256") == report_hash
            and metadata.get("evidence_report_object_path")
            == metadata.get("lifecycle_evidence_report_object_path")
            and metadata.get("lifecycle_report_release_version") == release.version
            and metadata.get("utx_release_warning") == UTX_RELEASE_WARNING
            and len(report_hash) == 64
            and all(value in "0123456789abcdef" for value in report_hash)
            and metadata.get("output_versions") == release.dataset_versions
            and metadata.get("network_accessed") is False
            and metadata.get("eodhd_calls") == 0
            and metadata.get("r2_accessed") is False
        ):
            return False
    if len(report_hashes) != 1:
        return False
    factors = repository.manifest_for_version(
        "adjustment_factors", release.dataset_versions["adjustment_factors"]
    )
    lineage = _adjustment_source_version(
        release.dataset_versions["daily_price_raw"],
        release.dataset_versions["corporate_actions"],
    )
    return bool(
        factors.metadata.get("source_version") == lineage
        and factors.metadata.get("source_daily_price_version")
        == release.dataset_versions["daily_price_raw"]
        and factors.metadata.get("source_corporate_actions_version")
        == release.dataset_versions["corporate_actions"]
        and factors.metadata.get("economic_rows_changed") == 0
    )


def _verify_repaired_state(
    frames: Mapping[str, pd.DataFrame],
    release: DataRelease,
) -> None:
    prices = frames["daily_price_raw"]
    sessions = _session_series(prices)
    for case in CASES:
        _require(_old_tail(prices, case).empty, f"{case.symbol} tail still exists.")
        old = prices.loc[prices["security_id"].map(_text).eq(case.security_id)]
        _require(
            not old.empty and _date(old["session"].max()) == case.old_last_good_session,
            f"{case.symbol} repaired final session changed.",
        )
        for history, dataset, end_field in (
            (False, "security_master", "active_to"),
            (True, "symbol_history", "effective_to"),
        ):
            rows = _identity_rows(frames[dataset], case, history=history)
            _require(len(rows) == 1, f"{case.symbol} repaired identity changed.")
            row = rows.iloc[0]
            action = frames["corporate_actions"].loc[
                frames["corporate_actions"]["event_id"].map(_text).eq(case.event_id)
            ].iloc[0]
            _require(
                _date(row.get(end_field)) == case.old_last_good_session
                and _text(row.get("source")) == REPAIRED_IDENTITY_SOURCE
                and _text(row.get("source_hash")).lower() == case.official_source_hash
                and _text(row.get("source_url")) == _text(action.get("source_url")),
                f"{case.symbol} repaired identity provenance changed.",
            )
    hcp = next(case for case in CASES if case.symbol == "HCP")
    peak = prices.loc[
        prices["security_id"].map(_text).eq(hcp.successor_security_id)
        & sessions.eq(hcp.transition_date)
    ]
    _require(len(peak) == 1, "Repaired PEAK first-session key changed.")
    row = peak.iloc[0]
    _require(
        _text(row.get("source_hash")) == hcp.old_tail_source_hash
        and [float(row[field]) for field in ("open", "high", "low", "close", "volume")]
        == [34.75, 34.82, 33.85, 34.41, 8_054_269.0],
        "Repaired PEAK first-session actual HCP row changed.",
    )
    factor_keys = set(
        zip(
            frames["adjustment_factors"]["security_id"].map(_text),
            pd.to_datetime(
                frames["adjustment_factors"]["session"], errors="raise"
            ).dt.normalize(),
        )
    )
    price_keys = set(
        zip(
            prices["security_id"].map(_text),
            pd.to_datetime(prices["session"], errors="raise").dt.normalize(),
        )
    )
    _require(factor_keys == price_keys, "Repaired factor/price keys differ.")
    lineage = _adjustment_source_version(
        release.dataset_versions["daily_price_raw"],
        release.dataset_versions["corporate_actions"],
    )
    factors = frames["adjustment_factors"]
    _require(
        set(factors["source_version"].map(_text)) == {lineage}
        and set(factors["source_hash"].map(_text)) == {lineage}
        and set(factors["source"].map(_text)) == {"derived"},
        "Repaired factor lineage changed.",
    )
    _require(
        _affected_factor_economics_sha256(factors)
        == EXPECTED_REPAIRED_AFFECTED_FACTOR_ECONOMICS_SHA256,
        "Repaired affected factor economics changed.",
    )


def _verify_repaired_lifecycle_state(
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    _require(
        UTX_RELEASE_WARNING in release.warnings,
        "Applied release lost the UTX unsupported-consideration warning.",
    )
    context = _read_candidate_context(repository, release)
    candidates = _build_candidate_values(context, release)
    candidate_sha = lifecycle_candidate_set_sha256(_candidate_frame(candidates))
    _require(
        len(candidates) == EXPECTED_LIFECYCLE_COVERAGE["candidate_count"]
        and candidate_sha == EXPECTED_REPAIRED_LIFECYCLE_CANDIDATE_SET_SHA256,
        "Applied lifecycle candidate inventory changed.",
    )
    coverage = validate_lifecycle_coverage(
        _candidate_frame(candidates),
        frames["lifecycle_resolutions"],
        context["corporate_actions"],
        completed_session=release.completed_session,
    )
    coverage_metadata = coverage.manifest_metadata()
    _require(
        coverage.valid
        and all(
            coverage_metadata.get(key) == value
            for key, value in EXPECTED_LIFECYCLE_COVERAGE.items()
        )
        and coverage_metadata["candidate_set_sha256"]
        == EXPECTED_REPAIRED_LIFECYCLE_CANDIDATE_SET_SHA256
        and coverage_metadata["resolution_set_sha256"]
        == EXPECTED_REPAIRED_LIFECYCLE_RESOLUTION_SET_SHA256,
        "Applied lifecycle coverage gate changed.",
    )
    manifest = repository.manifest_for_version(
        "source_archive", release.dataset_versions["source_archive"]
    )
    report_sha = _text(manifest.metadata.get("lifecycle_evidence_report_sha256"))
    object_path = _text(
        manifest.metadata.get("lifecycle_evidence_report_object_path")
    )
    _require(
        manifest.metadata.get("evidence_report_sha256") == report_sha
        and manifest.metadata.get("evidence_report_object_path") == object_path
        and all(
            manifest.metadata.get(key) == value
            for key, value in coverage_metadata.items()
        ),
        "Applied lifecycle publication metadata is stale.",
    )
    rows = frames["source_archive"].loc[
        frames["source_archive"]["archive_id"].map(_text).eq(report_sha)
        & frames["source_archive"]["source_hash"].map(_text).eq(report_sha)
    ]
    _require(len(rows) == 1, "Applied lifecycle report archive row changed.")
    row = rows.iloc[0]
    _require(
        _text(row.get("dataset")) == LIFECYCLE_REPORT_SOURCE
        and _text(row.get("source")) == LIFECYCLE_REPORT_SOURCE
        and _text(row.get("content_type")) == "application/json"
        and _text(row.get("object_path")) == object_path,
        "Applied lifecycle report archive provenance changed.",
    )
    path = _safe_path(repository.root, object_path)
    _require(path.is_file(), "Applied lifecycle report archive payload is missing.")
    try:
        report_bytes = gzip.decompress(path.read_bytes())
    except Exception as exc:
        raise RuntimeError("Applied lifecycle report archive is invalid gzip.") from exc
    _require(
        sha256_bytes(report_bytes) == report_sha,
        "Applied lifecycle report archive payload hash changed.",
    )
    report = json.loads(report_bytes)
    binding = _report_binding(report, release, candidates)
    validate_lifecycle_report_binding(
        report, binding, purpose="identity-tail applied-report verification"
    )
    _validate_report_candidate_inventory(report, candidates)
    _validate_utx_report_evidence(report)
    utx_security_id = next(
        case.security_id for case in CASES if case.symbol == "UTX"
    )
    resolution_rows = frames["lifecycle_resolutions"].loc[
        frames["lifecycle_resolutions"]["security_id"]
        .map(_text)
        .eq(utx_security_id)
    ]
    _require(len(resolution_rows) == 1, "Applied UTX lifecycle resolution changed.")
    resolution = resolution_rows.iloc[0]
    expected_resolution = {
        "candidate_id": lifecycle_candidate_id(utx_security_id, "2020-04-02"),
        "last_price_date": "2020-04-02",
        "resolution": "exception",
        "event_id": "",
        "exception_code": "unsupported_consideration",
        "exception_reason": UTX_EXCEPTION_REASON,
        "reviewed_by": "us_lifecycle_finalizer_v1",
        "reviewed_at": UTX_EXCEPTION_REVIEWED_AT,
        "recheck_after": "",
        "successor_security_id": "",
        "successor_symbol": "",
        "source": "sec_edgar_filing",
        "source_url": UTX_DISTRIBUTION_SOURCE_URL,
        "retrieved_at": UTX_DISTRIBUTION_RETRIEVED_AT,
        "source_hash": UTX_DISTRIBUTION_SOURCE_HASH,
    }
    _require(
        all(
            _text(resolution.get(field)) == value
            for field, value in expected_resolution.items()
        ),
        "Applied UTX permanent lifecycle exception changed.",
    )
    _validate_utx_distribution_archive(repository, frames["source_archive"])
    gap = _validate_utx_unmodeled_distribution_gap(
        frames["daily_price_raw"],
        frames["utx_distribution_prices"],
        context["corporate_actions"],
    )
    return {
        "candidate_set_sha256": candidate_sha,
        "resolution_set_sha256": coverage_metadata["resolution_set_sha256"],
        "evidence_report_sha256": report_sha,
        "evidence_report_object_path": object_path,
        "coverage": coverage_metadata,
        "utx_distribution_gap": gap,
    }


def _read_affected_frames(
    repository: LocalDatasetRepository, release: DataRelease
) -> dict[str, pd.DataFrame]:
    security_ids = {
        security_id
        for case in CASES
        for security_id in (case.security_id, case.successor_security_id)
    }
    frames = {
        dataset: _read_security_subset(
            repository,
            dataset,
            release.dataset_versions[dataset],
            security_ids,
        )
        for dataset in (
            "daily_price_raw",
            "adjustment_factors",
            "security_master",
            "symbol_history",
            "corporate_actions",
        )
    }
    # The archive is small and is needed to bind each official payload path.
    frames["source_archive"] = repository.read_frame(
        "source_archive", release.dataset_versions["source_archive"]
    )
    frames["lifecycle_resolutions"] = repository.read_frame(
        "lifecycle_resolutions",
        release.dataset_versions["lifecycle_resolutions"],
    )
    frames["utx_distribution_prices"] = _read_security_subset(
        repository,
        "daily_price_raw",
        release.dataset_versions["daily_price_raw"],
        {UTX_CARR_SECURITY_ID, UTX_OTIS_SECURITY_ID},
    )
    return frames


def prepare_repair(repository: LocalDatasetRepository) -> PreparedRepair:
    _static_contract()
    release, release_etag = repository.current_release()
    _require(release is not None, "Current local release is required.")
    missing = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    _require(not missing, "Release lacks datasets: " + ", ".join(missing))
    pointer_etags = _pointer_etags(repository, release)
    _validate_input_manifests(repository, release)
    frames = _read_affected_frames(repository, release)
    _verify_official_evidence(
        repository, frames["corporate_actions"], frames["source_archive"]
    )

    if release.version == PINNED_RELEASE_VERSION:
        audit = _verify_old_release(repository, release, frames)
        planned_versions = _new_versions(release)
        output_versions = dict(release.dataset_versions)
        output_versions.update(planned_versions)
        planned_release = DataRelease.create(
            release.completed_session,
            output_versions,
            quality=release.quality,
            warnings=tuple(dict.fromkeys((*release.warnings, UTX_RELEASE_WARNING))),
        )
        repaired_prices = _rewrite_prices(frames["daily_price_raw"])
        repaired_master = _bind_identity_source_urls(
            _rewrite_identity(frames["security_master"], history=False),
            frames["corporate_actions"],
            history=False,
        )
        repaired_history = _bind_identity_source_urls(
            _rewrite_identity(frames["symbol_history"], history=True),
            frames["corporate_actions"],
            history=True,
        )
        factor_lineage = _adjustment_source_version(
            planned_versions["daily_price_raw"],
            release.dataset_versions["corporate_actions"],
        )
        repaired_factors, economic_changes = _prepare_factors(
            frames["adjustment_factors"],
            frames["daily_price_raw"],
            repaired_prices,
            source_version=factor_lineage,
        )
        candidate_context = _read_candidate_context(repository, release)
        base_candidates = _build_candidate_values(candidate_context, release)
        base_candidate_sha = lifecycle_candidate_set_sha256(
            _candidate_frame(base_candidates)
        )
        _require(
            len(base_candidates) == EXPECTED_LIFECYCLE_COVERAGE["candidate_count"]
            and base_candidate_sha
            == EXPECTED_BASE_LIFECYCLE_CANDIDATE_SET_SHA256,
            "Pinned lifecycle candidate inventory changed.",
        )
        base_report = _load_and_validate_base_lifecycle_report(
            release, base_candidates
        )
        projected_candidate_context = _project_candidate_context(candidate_context)
        repaired_candidates = _build_candidate_values(
            projected_candidate_context, planned_release
        )
        repaired_candidate_sha = lifecycle_candidate_set_sha256(
            _candidate_frame(repaired_candidates)
        )
        _require(
            len(repaired_candidates)
            == EXPECTED_LIFECYCLE_COVERAGE["candidate_count"]
            and repaired_candidate_sha
            == EXPECTED_REPAIRED_LIFECYCLE_CANDIDATE_SET_SHA256,
            "Projected lifecycle candidate inventory changed.",
        )
        utx_distribution_evidence = _validate_utx_distribution_archive(
            repository, frames["source_archive"]
        )
        evidence_report_bytes = _rebind_lifecycle_report(
            base_report,
            planned_release=planned_release,
            base_candidates=base_candidates,
            repaired_candidates=repaired_candidates,
            repaired_prices=repaired_prices,
            utx_distribution_evidence=utx_distribution_evidence,
        )
        utx_distribution_gap = _validate_utx_unmodeled_distribution_gap(
            repaired_prices,
            frames["utx_distribution_prices"],
            projected_candidate_context["corporate_actions"],
        )
        repaired_resolutions, lifecycle_coverage = _migrate_lifecycle_resolutions(
            frames["lifecycle_resolutions"],
            candidates=repaired_candidates,
            actions=projected_candidate_context["corporate_actions"],
            completed_session=release.completed_session,
        )
        repaired_archive, report_sha, report_object_path = (
            _append_lifecycle_report_archive(
                frames["source_archive"],
                report_bytes=evidence_report_bytes,
                planned_release=planned_release,
            )
        )
        overrides = {
            "daily_price_raw": repaired_prices,
            "adjustment_factors": repaired_factors,
            "lifecycle_resolutions": repaired_resolutions,
            "security_master": repaired_master,
            "source_archive": repaired_archive,
            "symbol_history": repaired_history,
        }
        for dataset in WRITE_DATASETS:
            validate_dataset(
                dataset,
                overrides[dataset],
                completed_session=release.completed_session,
                incomplete_action_policy="block",
            ).raise_for_errors()
        snapshot_gap_recorded = _validate_known_snapshot_gap_pin()
        signal_impact = _hcp_signal_impact(
            frames["daily_price_raw"],
            repaired_prices,
            frames["adjustment_factors"],
            repaired_factors,
        )
        _require(
            all(
                case["index_membership"]["old_sid_member_tail_session_count"] == 0
                for case in audit["cases"]
                if case["symbol"] in EXPECTED_SYMBOLS
            ),
            "A repaired old-SID tail gained direct index membership.",
        )
        candidate_content = _candidate_content_projection(
            prices=repaired_prices,
            factors=repaired_factors,
            master=repaired_master,
            history=repaired_history,
        )
        summary = {
            "status": "validated_offline_plan",
            "base_release_version": release.version,
            "target_count": len(CASES),
            "symbols": sorted(EXPECTED_SYMBOLS),
            "removed_daily_price_rows": EXPECTED_REMOVED_PRICE_ROWS,
            "replaced_successor_price_rows": 1,
            "removed_adjustment_factor_rows": EXPECTED_REMOVED_PRICE_ROWS,
            "adjustment_factor_economic_rows_changed": economic_changes,
            "factor_source_version": factor_lineage,
            "planned_versions": planned_versions,
            "output_versions": output_versions,
            "release_pointer_inventory": sorted(pointer_etags),
            "release_pointer_inventory_count": len(pointer_etags),
            "registry": repair_registry(),
            "registry_inventory_sha256": registry_inventory_sha256(),
            "candidate_content_projection": candidate_content,
            "candidate_content_sha256": candidate_content["aggregate"],
            "lifecycle_candidate_set_sha256": repaired_candidate_sha,
            "lifecycle_resolution_set_sha256": lifecycle_coverage[
                "resolution_set_sha256"
            ],
            "lifecycle_evidence_report_sha256": report_sha,
            "lifecycle_evidence_report_object_path": report_object_path,
            "lifecycle_evidence_report_base_sha256": (
                EXPECTED_BASE_LIFECYCLE_REPORT_SHA256
            ),
            "lifecycle_report_release_version": planned_release.version,
            "lifecycle_coverage": lifecycle_coverage,
            "lifecycle_report_crosschecks_replayed": sorted(EXPECTED_SYMBOLS),
            "utx_distribution_gap": utx_distribution_gap,
            "utx_release_warning": UTX_RELEASE_WARNING,
            "utx_distribution_evidence": {
                key: value
                for key, value in utx_distribution_evidence.items()
                if key != "artifact" and key != "official_report_entry"
            },
            "candidate_hashes_are_deterministic": True,
            "planned_versions_are_ephemeral": True,
            "plan_file_sha256_is_reproducible": False,
            "plan_read_mode": "duckdb_affected_security_subset",
            "plan_full_market_frames_materialized": False,
            "apply_materialization": "sequential_one_dataset_at_a_time",
            "affected_daily_price_rows_loaded": len(frames["daily_price_raw"]),
            "affected_adjustment_factor_rows_loaded": len(
                frames["adjustment_factors"]
            ),
            "hcp_peak_triple_supertrend_impact": signal_impact,
            "old_sid_direct_index_tail_sessions": 0,
            "corporate_actions_unchanged": True,
            "index_constituent_anchors_unchanged": True,
            "index_membership_events_unchanged": True,
            "source_archive_unchanged": False,
            "source_archive_rows_added": 1,
            "lifecycle_resolution_rows_rebound": len(CASES),
            "snapshot_identity_gap_recorded": snapshot_gap_recorded,
            "snapshot_invariance_basis": (
                "hash_pinned_zero_direct_membership_tail_for_all_seven_targets"
            ),
            "cross_validation_change_plan": cross_validation_change_plan(),
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        }
        return PreparedRepair(
            release=release,
            release_etag=release_etag,
            pointer_etags=pointer_etags,
            planned_versions=planned_versions,
            frames=overrides,
            summary=summary,
            planned_release=planned_release,
            evidence_report_bytes=evidence_report_bytes,
            evidence_report_object_path=report_object_path,
        )

    _require(
        _exact_repair_manifests(repository, release),
        "Current release is neither the pinned base nor an exact repaired release.",
    )
    _verify_repaired_state(frames, release)
    candidate_content = _candidate_content_projection(
        prices=frames["daily_price_raw"],
        factors=frames["adjustment_factors"],
        master=frames["security_master"],
        history=frames["symbol_history"],
    )
    lifecycle = _verify_repaired_lifecycle_state(repository, release, frames)
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions={},
        frames={},
        summary={
            "status": "already_repaired",
            "base_release_version": PINNED_RELEASE_VERSION,
            "current_release_version": release.version,
            "release_pointer_inventory": sorted(pointer_etags),
            "release_pointer_inventory_count": len(pointer_etags),
            "target_count": len(CASES),
            "symbols": sorted(EXPECTED_SYMBOLS),
            "registry": repair_registry(),
            "registry_inventory_sha256": registry_inventory_sha256(),
            "candidate_content_projection": candidate_content,
            "candidate_content_sha256": candidate_content["aggregate"],
            "lifecycle_candidate_set_sha256": lifecycle[
                "candidate_set_sha256"
            ],
            "lifecycle_resolution_set_sha256": lifecycle[
                "resolution_set_sha256"
            ],
            "lifecycle_evidence_report_sha256": lifecycle[
                "evidence_report_sha256"
            ],
            "lifecycle_evidence_report_object_path": lifecycle[
                "evidence_report_object_path"
            ],
            "lifecycle_coverage": lifecycle["coverage"],
            "utx_distribution_gap": lifecycle["utx_distribution_gap"],
            "candidate_hashes_are_deterministic": True,
            "planned_versions_are_ephemeral": False,
            "plan_file_sha256_is_reproducible": True,
            "plan_read_mode": "duckdb_affected_security_subset",
            "plan_full_market_frames_materialized": False,
            "apply_materialization": "already_repaired_no_write",
            "cross_validation_change_plan": cross_validation_change_plan(),
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        },
    )


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
            raise RuntimeError("Unresolved identity-tail recovery marker blocks writes.")
        transactions = repository.root / TRANSACTION_DIR
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = _text(json.loads(journal.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        f"Interrupted identity-tail transaction blocks writes: {journal}."
                    )
        yield


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(
        path,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )


def _assert_inputs_unchanged(
    repository: LocalDatasetRepository, prepared: PreparedRepair
) -> None:
    release, etag = repository.current_release()
    _require(
        release is not None
        and release.to_bytes() == prepared.release.to_bytes()
        and etag == prepared.release_etag,
        "Current release changed after identity-tail planning.",
    )
    _require(
        set(prepared.pointer_etags) == set(prepared.release.dataset_versions),
        "Prepared release pointer inventory is incomplete.",
    )
    for dataset in sorted(prepared.release.dataset_versions):
        pointer, pointer_etag = repository.current_pointer(dataset)
        _require(
            pointer is not None
            and pointer.version == prepared.release.dataset_versions[dataset]
            and pointer_etag == prepared.pointer_etags[dataset],
            f"{dataset} pointer changed after identity-tail planning.",
        )


def _metadata_for_write(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    dataset: str,
) -> dict[str, Any]:
    manifest = repository.manifest_for_version(
        dataset, prepared.release.dataset_versions[dataset]
    )
    metadata = dict(manifest.metadata)
    output_versions = dict(prepared.release.dataset_versions)
    output_versions.update(prepared.planned_versions)
    report_sha = sha256_bytes(prepared.evidence_report_bytes)
    _require(
        prepared.planned_release is not None
        and prepared.planned_release.dataset_versions == output_versions
        and prepared.summary.get("lifecycle_evidence_report_sha256") == report_sha
        and prepared.summary.get("lifecycle_evidence_report_object_path")
        == prepared.evidence_report_object_path,
        "Prepared lifecycle report/release binding changed.",
    )
    metadata.update(
        {
            "schema": REPAIR_SCHEMA,
            "operation": OPERATION,
            "input_release_version": PINNED_RELEASE_VERSION,
            "output_versions": output_versions,
            "registry": repair_registry(),
            "registry_inventory_sha256": registry_inventory_sha256(),
            "candidate_content_sha256": EXPECTED_CANDIDATE_CONTENT_SHA256,
            "lifecycle_candidate_set_sha256": (
                EXPECTED_REPAIRED_LIFECYCLE_CANDIDATE_SET_SHA256
            ),
            "lifecycle_resolution_set_sha256": (
                EXPECTED_REPAIRED_LIFECYCLE_RESOLUTION_SET_SHA256
            ),
            "lifecycle_evidence_report_sha256": report_sha,
            "lifecycle_evidence_report_object_path": (
                prepared.evidence_report_object_path
            ),
            "lifecycle_report_release_version": prepared.planned_release.version,
            "lifecycle_report_input_versions": output_versions,
            "lifecycle_coverage": prepared.summary["lifecycle_coverage"],
            "utx_release_warning": UTX_RELEASE_WARNING,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        }
    )
    # Publication gates consume these canonical flat lifecycle fields.  Do not
    # leave the base manifest's hashes/counts in place after rebinding.
    metadata.update(prepared.summary["lifecycle_coverage"])
    metadata["evidence_report_sha256"] = report_sha
    metadata["evidence_report_object_path"] = prepared.evidence_report_object_path
    if dataset == "daily_price_raw":
        metadata.update(
            {
                "removed_rows": EXPECTED_REMOVED_PRICE_ROWS,
                "replaced_rows": 1,
                "replacement": "HCP:2019-11-05->PEAK:2019-11-05",
            }
        )
    elif dataset == "adjustment_factors":
        lineage = _adjustment_source_version(
            prepared.planned_versions["daily_price_raw"],
            prepared.release.dataset_versions["corporate_actions"],
        )
        metadata.update(
            {
                "source_version": lineage,
                "source_daily_price_version": prepared.planned_versions[
                    "daily_price_raw"
                ],
                "source_corporate_actions_version": prepared.release.dataset_versions[
                    "corporate_actions"
                ],
                "removed_rows": EXPECTED_REMOVED_PRICE_ROWS,
                "economic_rows_changed": 0,
            }
        )
    elif dataset == "lifecycle_resolutions":
        metadata.update(
            {
                "rebound_candidate_rows": len(CASES),
                "utx_resolution": "fail_closed_unsupported_consideration",
            }
        )
    elif dataset == "source_archive":
        metadata.update(
            {
                "archive_rows_added": 1,
                "archive_payload_sha256": report_sha,
                "archive_payload_object_path": prepared.evidence_report_object_path,
                "archive_payload_content_type": "application/json",
            }
        )
    return metadata


def _materialize_full_write_frame(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    dataset: str,
) -> pd.DataFrame:
    """Build one full output at a time for the later authorized apply path."""

    version = prepared.release.dataset_versions[dataset]
    frame = repository.read_frame(dataset, version)
    if dataset == "daily_price_raw":
        return _rewrite_prices(frame, copy_frame=False)
    if dataset == "adjustment_factors":
        lineage = _adjustment_source_version(
            prepared.planned_versions["daily_price_raw"],
            prepared.release.dataset_versions["corporate_actions"],
        )
        output, changed = _rewrite_factors_minimal(
            frame,
            source_version=lineage,
            copy_frame=False,
        )
        _require(changed == 0, "Materialized factor economics changed.")
        return output
    if dataset in {"security_master", "symbol_history"}:
        history = dataset == "symbol_history"
        actions = _read_security_subset(
            repository,
            "corporate_actions",
            prepared.release.dataset_versions["corporate_actions"],
            {case.security_id for case in CASES},
        )
        output = _bind_identity_source_urls(
            _rewrite_identity(frame, history=history),
            actions,
            history=history,
        )
        del actions
        return output
    if dataset in {"lifecycle_resolutions", "source_archive"}:
        output = prepared.frames.get(dataset)
        _require(output is not None, f"Prepared {dataset} frame is missing.")
        return output.copy(deep=True)
    raise ValueError(f"Unsupported identity-tail write dataset: {dataset}.")


def _persist_lifecycle_report_payload(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> bool:
    if (
        not prepared.evidence_report_bytes
        and not prepared.evidence_report_object_path
        and prepared.planned_release is None
    ):
        # Synthetic transaction tests can exercise pointer atomicity without a
        # lifecycle payload; real validated plans always carry all three.
        return False
    _require(
        bool(prepared.evidence_report_bytes)
        and bool(prepared.evidence_report_object_path)
        and sha256_bytes(prepared.evidence_report_bytes)
        == prepared.summary.get("lifecycle_evidence_report_sha256"),
        "Prepared lifecycle report payload changed.",
    )
    destination = _safe_path(
        repository.root, prepared.evidence_report_object_path
    )
    if destination.is_file():
        try:
            current = gzip.decompress(destination.read_bytes())
        except Exception as exc:
            raise RuntimeError(
                f"Existing lifecycle report archive is invalid: {destination}."
            ) from exc
        _require(
            current == prepared.evidence_report_bytes,
            "Existing lifecycle report archive payload conflicts.",
        )
        return False
    write_atomic(
        destination,
        gzip.compress(prepared.evidence_report_bytes, mtime=0),
    )
    _require(
        gzip.decompress(destination.read_bytes()) == prepared.evidence_report_bytes,
        "Lifecycle report archive payload verification failed.",
    )
    return True


def _remove_created_lifecycle_report_payload(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    created: bool,
) -> tuple[str, ...]:
    if not created:
        return ()
    try:
        destination = _safe_path(
            repository.root, prepared.evidence_report_object_path
        )
        _require(destination.is_file(), "Created lifecycle report archive disappeared.")
        current = gzip.decompress(destination.read_bytes())
        _require(
            current == prepared.evidence_report_bytes,
            "Created lifecycle report archive changed before rollback.",
        )
        destination.unlink()
        return ()
    except Exception as exc:
        return (f"archive_payload: {type(exc).__name__}: {exc}",)


def _commit_planned_release(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    versions: Mapping[str, str],
) -> DataRelease:
    planned = prepared.planned_release
    if planned is None:
        # Compatibility path for small synthetic transaction tests.
        return repository.commit_release(
            prepared.release.completed_session,
            dict(versions),
            quality=prepared.release.quality,
            warnings=prepared.release.warnings,
            expected_etag=prepared.release_etag,
        )
    _require(
        planned.dataset_versions == dict(sorted(versions.items()))
        and planned.completed_session == prepared.release.completed_session
        and planned.quality == prepared.release.quality
        and planned.warnings
        == tuple(
            dict.fromkeys((*prepared.release.warnings, UTX_RELEASE_WARNING))
        ),
        "Planned release/report binding changed before commit.",
    )
    repository.objects.put(
        f"releases/{planned.version}.json",
        planned.to_bytes(),
        if_none_match=True,
    )
    repository.objects.put(
        "releases/current.json",
        planned.to_bytes(),
        if_match=prepared.release_etag,
    )
    return planned


def _restore_transaction(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: Mapping[str, bytes],
    planned_versions: Mapping[str, str],
    planned_release_bytes: bytes | None,
    committed_release_version: str,
) -> tuple[str, ...]:
    # Preflight every mutable pointer before changing any of them.  A current
    # value is ours only when it is the exact old value or the exact planned
    # identity.  If a non-cooperating publisher changed even one value, leave
    # the entire publication untouched and force manual recovery.
    try:
        current_release = repository.objects.get("releases/current.json")
        if current_release.data != old_release_bytes:
            observed = DataRelease.from_bytes(current_release.data)
            belongs = (
                current_release.data == planned_release_bytes
                if planned_release_bytes is not None
                else bool(committed_release_version)
                and observed.version == committed_release_version
            )
            if not belongs:
                raise RuntimeError(
                    f"unexpected release during identity-tail rollback: {observed.version}"
                )
        current_pointers = {}
        for dataset in reversed(WRITE_DATASETS):
            key = repository.current_key(dataset)
            current = repository.objects.get(key)
            old = old_pointer_bytes[dataset]
            if current.data != old:
                pointer = CurrentPointer.from_bytes(current.data)
                if pointer.version != planned_versions[dataset]:
                    raise RuntimeError(
                        f"unexpected {dataset} pointer during rollback: {pointer.version}"
                    )
            current_pointers[dataset] = current
    except Exception as exc:
        return (f"rollback preflight: {type(exc).__name__}: {exc}",)

    errors: list[str] = []
    try:
        if current_release.data != old_release_bytes:
            repository.objects.put(
                "releases/current.json",
                old_release_bytes,
                if_match=current_release.etag,
            )
    except Exception as exc:
        errors.append(f"releases/current.json: {type(exc).__name__}: {exc}")
    for dataset in reversed(WRITE_DATASETS):
        key = repository.current_key(dataset)
        try:
            current = current_pointers[dataset]
            old = old_pointer_bytes[dataset]
            if current.data != old:
                repository.objects.put(key, old, if_match=current.etag)
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _assert_applied_release(
    repository: LocalDatasetRepository,
    committed: DataRelease,
    *,
    expected_out_of_scope_pointer_etags: Mapping[str, str | None],
) -> None:
    _require(
        set(expected_out_of_scope_pointer_etags)
        == set(committed.dataset_versions),
        "Applied release pointer inventory is incomplete.",
    )
    current, _ = repository.current_release()
    _require(
        current is not None and current.to_bytes() == committed.to_bytes(),
        "Committed identity-tail release is not current.",
    )
    for dataset, version in committed.dataset_versions.items():
        pointer, etag = repository.current_pointer(dataset)
        _require(
            pointer is not None and pointer.version == version,
            f"Applied identity-tail pointer mismatch: {dataset}.",
        )
        if dataset not in WRITE_DATASETS:
            _require(
                etag == expected_out_of_scope_pointer_etags[dataset],
                f"Out-of-scope pointer changed: {dataset}.",
            )
    replay = prepare_repair(repository)
    _require(
        replay.summary["status"] == "already_repaired",
        "Applied identity-tail repair is not idempotent.",
    )


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    inject_failure: FailureInjector | None = None,
) -> dict[str, Any]:
    inject = inject_failure or (lambda _stage: None)
    with _exclusive_lock(repository):
        _assert_inputs_unchanged(repository, prepared)
        current_plan = prepared
        if current_plan.summary["status"] == "already_repaired":
            return {
                **current_plan.summary,
                "mode": "apply",
                "writes_performed": False,
            }
        planned = dict(current_plan.planned_versions)
        _require(
            set(planned) == set(WRITE_DATASETS)
            and len(set(planned.values())) == len(WRITE_DATASETS),
            "Prepared identity-tail versions are invalid.",
        )
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            _require(
                pointer.version == current_plan.release.dataset_versions[dataset]
                and value.etag == current_plan.pointer_etags[dataset],
                f"{dataset} changed before identity-tail apply.",
            )
            old_pointers[dataset] = value.data
        transaction_id = uuid.uuid4().hex
        journal_path = repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        journal: dict[str, Any] = {
            "schema": "us_identity_price_tail_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": current_plan.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                key: base64.b64encode(value).decode("ascii")
                for key, value in old_pointers.items()
            },
            "planned_versions": planned,
            "planned_release_version": (
                current_plan.planned_release.version
                if current_plan.planned_release is not None
                else ""
            ),
            "lifecycle_evidence_report_sha256": current_plan.summary.get(
                "lifecycle_evidence_report_sha256", ""
            ),
            "lifecycle_evidence_report_object_path": (
                current_plan.evidence_report_object_path
            ),
            "registry_inventory_sha256": registry_inventory_sha256(),
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        archive_created = False
        try:
            archive_created = _persist_lifecycle_report_payload(
                repository, current_plan
            )
            inject("after_archive_payload")
            versions = dict(current_plan.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                frame = _materialize_full_write_frame(
                    repository, current_plan, dataset
                )
                result = repository.write_frame(
                    dataset,
                    frame,
                    completed_session=current_plan.release.completed_session,
                    incomplete_action_policy="block",
                    metadata=_metadata_for_write(repository, current_plan, dataset),
                    expected_pointer_etag=current_plan.pointer_etags[dataset],
                    version=planned[dataset],
                )
                del frame
                gc.collect()
                _require(
                    not result.conflict and result.manifest.version == planned[dataset],
                    f"Unexpected {dataset} write result.",
                )
                versions[dataset] = result.manifest.version
                inject(f"after_write:{dataset}")
            for dataset in sorted(current_plan.release.dataset_versions):
                if dataset in WRITE_DATASETS:
                    continue
                pointer, etag = repository.current_pointer(dataset)
                _require(
                    pointer is not None
                    and pointer.version == current_plan.release.dataset_versions[dataset]
                    and etag == current_plan.pointer_etags[dataset],
                    f"Out-of-scope pointer changed during apply: {dataset}.",
                )
            committed = _commit_planned_release(
                repository,
                current_plan,
                versions,
            )
            inject("after_release_commit")
            _assert_applied_release(
                repository,
                committed,
                expected_out_of_scope_pointer_etags=current_plan.pointer_etags,
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
                **current_plan.summary,
                "status": "applied",
                "mode": "apply",
                "writes_performed": True,
                "new_release_version": committed.version,
                "transaction_id": transaction_id,
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_versions=planned,
                planned_release_bytes=(
                    current_plan.planned_release.to_bytes()
                    if current_plan.planned_release is not None
                    else None
                ),
                committed_release_version=committed.version if committed else "",
            )
            # Never remove our archive payload when rollback ownership or any
            # pointer restoration is uncertain; the recovery marker retains the
            # evidence needed for manual reconciliation.
            if not rollback_errors:
                rollback_errors = (
                    *rollback_errors,
                    *_remove_created_lifecycle_report_payload(
                        repository,
                        current_plan,
                        created=archive_created,
                    ),
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
                    "Identity-tail rollback was incomplete; recovery marker blocks writes: "
                    f"{recovery}; errors={rollback_errors}."
                ) from original
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan seven exact US identity-price-tail repairs offline."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    repository = LocalDatasetRepository(args.cache_root)
    prepared = prepare_repair(repository)
    result = (
        apply_repair(repository, prepared)
        if args.apply
        else {**prepared.summary, "mode": "plan", "writes_performed": False}
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
