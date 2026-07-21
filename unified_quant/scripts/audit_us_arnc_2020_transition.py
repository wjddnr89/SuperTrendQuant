#!/usr/bin/env python3
"""Prove the missing 2020 ARNC -> HWM transition from local immutable data.

This audit is pinned to release ``20260715-20260718T230255094849Z``.  It reads
the release Parquet files, immutable source-archive objects, the already-built
Yahoo cross-validation report, and the matching S&P 500 backtest summary.  It
never performs HTTP, EODHD, R2, dataset, policy, or release writes.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from supertrend_quant.market_store.cross_validation import (  # noqa: E402
    TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256,
    canonical_json_bytes,
    reviewed_nonterminal_extraction_mismatches,
    reviewed_nonterminal_extraction_sha256,
    reviewed_nonterminal_extractions,
    reviewed_nonterminal_inventory_sha256,
    successor_price_check_binding,
)
from supertrend_quant.market_store.lifecycle import (  # noqa: E402
    canonical_lifecycle_event_id,
)
from supertrend_quant.market_store.lifecycle_coverage import (  # noqa: E402
    lifecycle_candidate_id,
)
from supertrend_quant.market_store.manifest import (  # noqa: E402
    DataRelease,
    sha256_bytes,
)
from supertrend_quant.market_store.repository import (  # noqa: E402
    LocalDatasetRepository,
)


PINNED_RELEASE_VERSION = "20260715-20260718T230255094849Z"
AUDIT_SCHEMA = "us_arnc_2020_transition_audit/v1"

PARENT_SECURITY_ID = "US:EODHD:f5daeed5-d1a2-5279-aa49-8c06c902b97f"
CHILD_SECURITY_ID = "US:EODHD:33cf5387-6cec-598e-84a9-563ca333b0f3"
OLD_SYMBOL = "ARNC"
NEW_PARENT_SYMBOL = "HWM"
EFFECTIVE_DATE = "2020-04-01"
LAST_OLD_SYMBOL_SESSION = "2020-03-31"
SPINOFF_EVENT_ID = (
    "15d7f68b627981221f55f109696085382809e34d4abfb98163704d1b47a45b04"
)
TICKER_CHANGE_EVENT_ID = canonical_lifecycle_event_id(
    PARENT_SECURITY_ID, "ticker_change", EFFECTIVE_DATE
)
OLD_SYMBOL_TARGET_ID = (
    "ef89f10e83177128247e7b62c97338dba1c62fdce831e5840631075781afa79d"
)
NEW_PARENT_TARGET_ID = (
    "5554726d8670f59fd104028a7ef71ab5ea3d2abbd89b2c77c0354d3a3090dc8d"
)

SEC_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/1790982/"
    "000179098221000096/arnc-20210630.htm"
)
SEC_SOURCE_HASH = (
    "6b88966f49fb2a2c7a2bbb832873aad7db2f396148ccf63484849729009adf70"
)
SP_SOURCE_URL = (
    "https://press.spglobal.com/"
    "2020-04-01-Arconic-Set-to-Join-S-P-SmallCap-600"
)
SP_SOURCE_HASH = (
    "99bf852ed888154abdd1754398c1cc33dba31ddc6aee6bb5912c56df22ff24ee"
)

DEFAULT_CROSS_VALIDATION_REPORT = Path("/tmp/crossval-current-final.json")
DEFAULT_POLICY = SCRIPT_DIR.parent / "configs/us_cross_validation.yaml"
DEFAULT_BACKTEST_SUMMARY = Path(
    "results/research/us/backtests/"
    "sp500_triple_supertrend_alone_max_20260715-20260718T230255094849Z/"
    "summary.json"
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _date(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    parsed = pd.Timestamp(text)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_localize(None)
    return parsed.date().isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_release(repository: LocalDatasetRepository) -> DataRelease:
    path = repository.root / "releases" / f"{PINNED_RELEASE_VERSION}.json"
    _require(path.is_file(), "Pinned ARNC audit release is missing.")
    release = DataRelease.from_bytes(path.read_bytes())
    _require(
        release.version == PINNED_RELEASE_VERSION,
        "Pinned ARNC audit release bytes changed.",
    )
    return release


def _archive_payload(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    source_url: str,
    source_hash: str,
) -> tuple[bytes, dict[str, Any]]:
    matches = archive.loc[
        archive["source_url"].map(_text).eq(source_url)
        & archive["source_hash"].map(_text).str.lower().eq(source_hash)
    ]
    _require(len(matches) == 1, "Official ARNC archive pair is not unique.")
    row = matches.iloc[0].to_dict()
    object_path = Path(_text(row.get("object_path")))
    base = repository.root.resolve()
    path = (repository.root / object_path).resolve()
    _require(base in path.parents and path.is_file(), "ARNC archive path is unsafe or missing.")
    encoded = path.read_bytes()
    payload = gzip.decompress(encoded) if path.suffix == ".gz" else encoded
    _require(sha256_bytes(payload) == source_hash, "ARNC archive payload hash changed.")
    _require(
        _text(row.get("archive_id")).lower() == source_hash
        and _text(row.get("dataset")) == _text(row.get("source")),
        "ARNC source_archive row is not an exact immutable object binding.",
    )
    return payload, {
        "source_url": source_url,
        "source_hash": source_hash,
        "object_path": object_path.as_posix(),
        "content_type": _text(row.get("content_type")),
        "payload_bytes": len(payload),
        "payload_sha256_verified": True,
    }


def _plain_html(payload: bytes) -> str:
    value = payload.decode("utf-8", errors="replace")
    value = re.sub(r"<script\b[^>]*>.*?</script>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<style\b[^>]*>.*?</style>", " ", value, flags=re.I | re.S)
    value = html.unescape(re.sub(r"<[^>]+>", " ", value))
    return re.sub(r"\s+", " ", value).strip()


def _official_evidence_checks(sec_text: str, sp_text: str) -> dict[str, bool]:
    sec = sec_text.lower()
    sp = sp_text.lower()
    checks = {
        "sec_separation_effective_2020_04_01": bool(
            re.search(
                r"on april 1, 2020 .*?separation was completed and became effective",
                sec,
            )
        ),
        "sec_distribution_one_for_four": bool(
            re.search(
                r"one share of arconic common stock for every four shares",
                sec,
            )
        ),
        "sec_child_regular_way_arnc_started_2020_04_01": bool(
            re.search(
                r"regular-way.*?trading of arconic.*?began.*?april 1, 2020.*?ticker symbol.*?arnc",
                sec,
            )
        ),
        "sp_parent_hwm_formerly_arconic": bool(
            re.search(
                r"howmet aerospace inc\. \(nyse: hwm -formerly arconic inc\.\)",
                sp,
            )
        ),
        "sp_separation_completed_2020_04_01": bool(
            re.search(
                r"spun off arconic corp\..*?completed today, april 1",
                sp,
            )
        ),
        "sp_hwm_remained_sp500": "post spin-off howmet aerospace will remain in the s&p 500"
        in sp,
    }
    _require(all(checks.values()), "Archived ARNC official evidence text changed.")
    return checks


def proposed_nonterminal_extraction() -> dict[str, Any]:
    return {
        "event_id": TICKER_CHANGE_EVENT_ID,
        "security_id": PARENT_SECURITY_ID,
        "action_type": "ticker_change",
        "effective_date": EFFECTIVE_DATE,
        "new_security_id": PARENT_SECURITY_ID,
        "new_symbol": NEW_PARENT_SYMBOL,
        "ratio": None,
        "cash_amount": None,
        "currency": "USD",
        "source_kind": "official_crosscheck",
        "source_url": SP_SOURCE_URL,
        "source_hash": SP_SOURCE_HASH,
    }


def proposed_action() -> dict[str, Any]:
    return {
        "event_id": TICKER_CHANGE_EVENT_ID,
        "security_id": PARENT_SECURITY_ID,
        "action_type": "ticker_change",
        "effective_date": EFFECTIVE_DATE,
        "ex_date": EFFECTIVE_DATE,
        "announcement_date": "",
        "record_date": "",
        "payment_date": "",
        "cash_amount": None,
        "ratio": None,
        "currency": "USD",
        "new_security_id": PARENT_SECURITY_ID,
        "new_symbol": NEW_PARENT_SYMBOL,
        "official": True,
        "source_url": SP_SOURCE_URL,
        "source_kind": "official_crosscheck",
        "source": "official_identity_repair",
        "source_hash": SP_SOURCE_HASH,
        "metadata": None,
    }


def _action_install_state(actions: pd.DataFrame) -> str:
    candidates = actions.loc[
        actions["security_id"].map(_text).eq(PARENT_SECURITY_ID)
        & actions["action_type"].map(_text).str.lower().eq("ticker_change")
        & actions["effective_date"].map(_date).eq(EFFECTIVE_DATE)
    ]
    if candidates.empty:
        _require(
            not actions["event_id"].map(_text).eq(TICKER_CHANGE_EVENT_ID).any(),
            "Canonical ARNC ticker event ID is attached to a different action.",
        )
        return "missing"
    _require(len(candidates) == 1, "ARNC 2020 ticker action is ambiguous.")
    row = candidates.iloc[0].to_dict()
    mismatches = reviewed_nonterminal_extraction_mismatches(
        row, proposed_nonterminal_extraction()
    )
    _require(not mismatches, "Installed ARNC ticker action differs: " + ", ".join(mismatches))
    _require(_text(row.get("official")).lower() == "true", "ARNC ticker action is not official.")
    return "exact"


def _policy_projection(policy: Mapping[str, Any]) -> dict[str, Any]:
    events = policy.get("events")
    _require(isinstance(events, Mapping), "Cross-validation events policy is missing.")
    current = reviewed_nonterminal_extractions(events)
    proposal = proposed_nonterminal_extraction()
    existing = current.get(TICKER_CHANGE_EVENT_ID)
    if existing is None:
        state = "missing"
        projected = copy.deepcopy(dict(policy))
        projected_events = projected["events"]
        projected_events["reviewed_nonterminal_extractions"] = [
            *projected_events["reviewed_nonterminal_extractions"],
            proposal,
        ]
    else:
        _require(existing == proposal, "ARNC nonterminal policy extraction differs.")
        state = "exact"
        projected = copy.deepcopy(dict(policy))
    current_hash = reviewed_nonterminal_inventory_sha256(events)
    return {
        "state": state,
        "proposal": proposal,
        "proposal_sha256": reviewed_nonterminal_extraction_sha256(proposal),
        "current_inventory_sha256": current_hash,
        "compiled_trusted_inventory_sha256": (
            TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256
        ),
        "current_inventory_code_pin_passed": (
            current_hash == TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256
        ),
        "projected_inventory_sha256": reviewed_nonterminal_inventory_sha256(
            projected["events"]
        ),
    }


def _cross_validation_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        _text(report.get("base_release_version")) == PINNED_RELEASE_VERSION,
        "ARNC cross-validation report is not release pinned.",
    )
    prices = [value for value in report.get("prices", ()) if isinstance(value, Mapping)]
    old = [value for value in prices if _text(value.get("target_id")) == OLD_SYMBOL_TARGET_ID]
    successor = [
        value for value in prices if _text(value.get("target_id")) == NEW_PARENT_TARGET_ID
    ]
    _require(len(old) == 1 and len(successor) == 1, "ARNC/HWM price targets are not unique.")
    _require(
        _text(old[0].get("status")) == "mismatch"
        and not _text(old[0].get("terminal_event_id"))
        and _text(successor[0].get("status")) == "passed",
        "Pinned pre-repair ARNC/HWM cross-validation state changed.",
    )
    event = {**proposed_nonterminal_extraction(), "status": "passed"}
    binding = successor_price_check_binding(
        prices,
        event,
        source_target_id=OLD_SYMBOL_TARGET_ID,
        expected_successor_security_id=PARENT_SECURITY_ID,
        reviewed_successor_chains={},
        event_checks=[event],
    )
    _require(
        binding.get("passed") is True
        and _text(binding.get("target_id")) == NEW_PARENT_TARGET_ID,
        "Exact ARNC -> HWM successor price binding did not pass.",
    )
    return {
        "pre_repair_old_symbol_status": _text(old[0].get("status")),
        "old_symbol_terminal_event_id": _text(old[0].get("terminal_event_id")),
        "hwm_successor_status": _text(successor[0].get("status")),
        "projected_successor_binding": binding,
        "projected_old_symbol_status": "explicit_exception",
        "projected_validation_basis": (
            "exact reviewed nonterminal same-security ticker transition plus "
            "independently passed HWM price target"
        ),
    }


def _backtest_projection(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"summary_present": False}
    payload = path.read_bytes()
    summary = json.loads(payload)
    trades = [
        value
        for value in summary.get("trades", ())
        if isinstance(value, Mapping)
        and _text(value.get("symbol")).upper() in {OLD_SYMBOL, NEW_PARENT_SYMBOL}
    ]
    event_day = pd.Timestamp(EFFECTIVE_DATE)
    event_window = [
        value
        for value in trades
        if pd.Timestamp(value["entry_time"]) <= event_day
        <= pd.Timestamp(value["exit_time"])
    ]
    return {
        "summary_present": True,
        "summary_path": path.as_posix(),
        "summary_sha256": sha256_bytes(payload),
        "arnc_hwm_trade_count": len(trades),
        "event_window_position_count": len(event_window),
        "trades": [
            {
                "symbol": _text(value.get("symbol")),
                "entry_time": _text(value.get("entry_time")),
                "exit_time": _text(value.get("exit_time")),
            }
            for value in trades
        ],
        "expected_existing_run_equity_effect": "none",
        "reason": (
            "The missing row is a quantity/cash-neutral same-security ticker "
            "transition; the 0.25-share spin-off and cost-basis allocation are "
            "already stored, and this run held no ARNC/HWM position on the event date."
        ),
    }


def build_audit(
    repository: LocalDatasetRepository,
    report: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    backtest_summary_path: Path = DEFAULT_BACKTEST_SUMMARY,
) -> dict[str, Any]:
    release = _load_release(repository)
    names = (
        "security_master",
        "symbol_history",
        "corporate_actions",
        "daily_price_raw",
        "adjustment_factors",
        "lifecycle_resolutions",
        "source_archive",
        "index_constituent_anchors",
        "index_membership_events",
    )
    frames = {
        name: repository.read_frame(name, release.dataset_versions[name])
        for name in names
    }

    master = frames["security_master"]
    history = frames["symbol_history"]
    actions = frames["corporate_actions"]
    prices = frames["daily_price_raw"]
    resolutions = frames["lifecycle_resolutions"]

    parent_master = master.loc[master["security_id"].map(_text).eq(PARENT_SECURITY_ID)]
    child_master = master.loc[master["security_id"].map(_text).eq(CHILD_SECURITY_ID)]
    _require(len(parent_master) == 1 and len(child_master) == 1, "ARNC master identities changed.")
    _require(
        _text(parent_master.iloc[0].get("primary_symbol")).upper() == NEW_PARENT_SYMBOL
        and not _date(parent_master.iloc[0].get("active_to"))
        and _text(child_master.iloc[0].get("primary_symbol")).upper() == OLD_SYMBOL
        and _date(child_master.iloc[0].get("active_from")) == EFFECTIVE_DATE,
        "ARNC/HWM security_master identity changed.",
    )

    parent_history = history.loc[
        history["security_id"].map(_text).eq(PARENT_SECURITY_ID)
    ]
    expected_intervals = {
        (OLD_SYMBOL, "2016-11-01", LAST_OLD_SYMBOL_SESSION),
        (NEW_PARENT_SYMBOL, EFFECTIVE_DATE, ""),
    }
    observed_intervals = {
        (
            _text(row.get("symbol")).upper(),
            _date(row.get("effective_from")),
            _date(row.get("effective_to")),
        )
        for row in parent_history.to_dict(orient="records")
    }
    _require(expected_intervals.issubset(observed_intervals), "ARNC/HWM symbol boundary changed.")
    child_history = history.loc[
        history["security_id"].map(_text).eq(CHILD_SECURITY_ID)
        & history["symbol"].map(_text).str.upper().eq(OLD_SYMBOL)
    ]
    _require(
        len(child_history) == 1
        and _date(child_history.iloc[0].get("effective_from")) == EFFECTIVE_DATE,
        "New Arconic child identity boundary changed.",
    )

    parent_prices = prices.loc[prices["security_id"].map(_text).eq(PARENT_SECURITY_ID)].copy()
    parent_prices["_session"] = parent_prices["session"].map(_date)
    child_prices = prices.loc[prices["security_id"].map(_text).eq(CHILD_SECURITY_ID)].copy()
    child_prices["_session"] = child_prices["session"].map(_date)
    _require(
        len(parent_prices.loc[parent_prices["_session"].eq(LAST_OLD_SYMBOL_SESSION)]) == 1
        and len(parent_prices.loc[parent_prices["_session"].eq(EFFECTIVE_DATE)]) == 1
        and child_prices["_session"].min() == EFFECTIVE_DATE,
        "ARNC/HWM price boundary changed.",
    )

    spinoff = actions.loc[actions["event_id"].map(_text).eq(SPINOFF_EVENT_ID)]
    _require(len(spinoff) == 1, "2020 Arconic spin-off action is missing or ambiguous.")
    spin = spinoff.iloc[0]
    _require(
        _text(spin.get("security_id")) == PARENT_SECURITY_ID
        and _text(spin.get("action_type")).lower() == "spinoff"
        and _date(spin.get("effective_date")) == EFFECTIVE_DATE
        and _text(spin.get("new_security_id")) == CHILD_SECURITY_ID
        and _text(spin.get("new_symbol")).upper() == OLD_SYMBOL
        and abs(float(spin.get("ratio")) - 0.25) <= 1e-12
        and _text(spin.get("source_hash")).lower() == SEC_SOURCE_HASH,
        "Stored 2020 Arconic spin-off economics changed.",
    )

    ticker_state = _action_install_state(actions)
    candidate_id = lifecycle_candidate_id(PARENT_SECURITY_ID, LAST_OLD_SYMBOL_SESSION)
    forbidden_resolutions = resolutions.loc[
        resolutions["event_id"].map(_text).eq(TICKER_CHANGE_EVENT_ID)
        | resolutions["candidate_id"].map(_text).eq(candidate_id)
    ]
    _require(
        forbidden_resolutions.empty,
        "ARNC -> HWM is an intermediate same-security ticker transition, not a terminal lifecycle resolution.",
    )

    sec_payload, sec_archive = _archive_payload(
        repository,
        frames["source_archive"],
        source_url=SEC_SOURCE_URL,
        source_hash=SEC_SOURCE_HASH,
    )
    sp_payload, sp_archive = _archive_payload(
        repository,
        frames["source_archive"],
        source_url=SP_SOURCE_URL,
        source_hash=SP_SOURCE_HASH,
    )
    text_checks = _official_evidence_checks(
        _plain_html(sec_payload), _plain_html(sp_payload)
    )

    anchors = frames["index_constituent_anchors"]
    membership = frames["index_membership_events"]
    sp500_anchor = anchors.loc[
        anchors["security_id"].map(_text).eq(PARENT_SECURITY_ID)
        & anchors["index_id"].map(_text).eq("sp500")
        & anchors["anchor_date"].map(_date).le(LAST_OLD_SYMBOL_SESSION)
    ]
    later_events = membership.loc[
        membership["security_id"].map(_text).eq(PARENT_SECURITY_ID)
        & membership["index_id"].map(_text).eq("sp500")
        & membership["effective_date"].map(_date).le(EFFECTIVE_DATE)
    ].sort_values("effective_date")
    member = not sp500_anchor.empty
    if not later_events.empty:
        member = _text(later_events.iloc[-1].get("operation")).upper() == "ADD"
    _require(member, "Parent Arconic/Howmet was not an S&P 500 member at transition.")

    policy_projection = _policy_projection(policy)
    cross_projection = _cross_validation_projection(report)
    proposal = proposed_action()
    return {
        "schema": AUDIT_SCHEMA,
        "release_version": release.version,
        "release_dataset_versions": dict(release.dataset_versions),
        "network_accessed": False,
        "http_attempts": 0,
        "eodhd_calls": 0,
        "r2_accessed": False,
        "dataset_writes_performed": False,
        "policy_writes_performed": False,
        "conclusion": {
            "local_evidence_sufficient": True,
            "actual_event": (
                "The old Arconic Inc. identity continued as Howmet Aerospace "
                "under HWM on 2020-04-01 and simultaneously distributed 0.25 "
                "share of the separate new Arconic Corporation (ARNC)."
            ),
            "mismatch_cause": (
                "The spin-off, identities, and both price histories are present; "
                "only the same-security ARNC -> HWM ticker_change row is absent."
            ),
            "repair_state": (
                "already_exact"
                if ticker_state == "exact" and policy_projection["state"] == "exact"
                else "repair_required"
            ),
        },
        "official_evidence": {
            "sec_child_filing": sec_archive,
            "sp_index_provider_release": sp_archive,
            "text_claims_passed": text_checks,
        },
        "stored_state": {
            "old_symbol_last_session": LAST_OLD_SYMBOL_SESSION,
            "next_xnys_session": EFFECTIVE_DATE,
            "parent_price_present_on_both_sides": True,
            "child_first_price_session": child_prices["_session"].min(),
            "spinoff_event_id": SPINOFF_EVENT_ID,
            "spinoff_ratio": float(spin.get("ratio")),
            "ticker_change_state": ticker_state,
            "sp500_member_at_transition": member,
        },
        "required_changes": {
            "corporate_actions": {
                "change": "add_one_exact_nonterminal_ticker_change",
                "row": proposal,
            },
            "lifecycle_resolutions": {
                "change": "none",
                "reason": (
                    "The parent security did not terminate; it continued under "
                    "HWM. Adding an applied terminal resolution would misclassify "
                    "an intermediate identity transition."
                ),
                "forbidden_candidate_id": candidate_id,
            },
            "security_master": {"change": "none"},
            "symbol_history": {"change": "none"},
            "daily_price_raw": {"change": "none"},
            "adjustment_factors": {
                "change": "none",
                "reason": "A ticker change has no cash, quantity, or cost-basis effect.",
            },
            "existing_spinoff_action": {"change": "none"},
            "index_membership": {
                "change": "none",
                "reason": "The same parent SID remained in the S&P 500 as HWM.",
            },
            "cross_validation_policy": {
                "change": (
                    "add the exact reviewed_nonterminal_extractions row and "
                    "update the compiled aggregate inventory SHA"
                ),
                **policy_projection,
            },
            "cross_validation_report_verifier": {
                "change": (
                    "accept this exact nonterminal same-SID ticker transition "
                    "without requiring an applied lifecycle resolution; retain "
                    "the exact action/policy/archive/date and passed-successor checks"
                )
            },
        },
        "cross_validation": cross_projection,
        "backtest_impact": _backtest_projection(backtest_summary_path),
        "fail_closed_conditions": [
            "Either official archive URL/hash pair or decompressed payload SHA changes.",
            "The SEC separation date, 1-for-4 ratio, or new ARNC first-trade claim changes.",
            "The S&P HWM-formerly-ARNC, completion-date, or S&P 500 continuity claim changes.",
            "The parent SID does not have ARNC through 2020-03-31 and HWM from 2020-04-01.",
            "The child ARNC SID does not start on 2020-04-01 with the exact 0.25 spin-off.",
            "The canonical ticker action or reviewed policy row differs in any normalized field.",
            "An applied terminal lifecycle resolution is added for this continuing parent SID.",
            "The independently checked HWM successor price target is not passed exactly once.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/cache")
    parser.add_argument(
        "--cross-validation-report",
        default=str(DEFAULT_CROSS_VALIDATION_REPORT),
    )
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument(
        "--backtest-summary", default=str(DEFAULT_BACKTEST_SUMMARY)
    )
    args = parser.parse_args()

    report = json.loads(Path(args.cross_validation_report).read_text())
    policy = yaml.safe_load(Path(args.policy).read_text())
    audit = build_audit(
        LocalDatasetRepository(Path(args.data_root)),
        report,
        policy,
        backtest_summary_path=Path(args.backtest_summary),
    )
    payload = canonical_json_bytes(audit)
    print(payload.decode())
    print(
        json.dumps(
            {
                "status": "read_only_audit_complete",
                "audit_sha256": sha256_bytes(payload),
                "repair_state": audit["conclusion"]["repair_state"],
                "network_accessed": False,
                "dataset_writes_performed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
