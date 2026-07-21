from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_ROOT / "unified_quant/scripts/audit_us_arnc_2020_transition.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_us_arnc_2020_transition", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


POLICY_PATH = PROJECT_ROOT / "unified_quant/configs/us_cross_validation.yaml"
RELEASE_PATH = (
    PROJECT_ROOT
    / "data/cache/releases"
    / f"{script.PINNED_RELEASE_VERSION}.json"
)


def _synthetic_cross_validation_report() -> dict[str, object]:
    return {
        "base_release_version": script.PINNED_RELEASE_VERSION,
        "prices": [
            {
                "target_id": script.OLD_SYMBOL_TARGET_ID,
                "security_id": script.PARENT_SECURITY_ID,
                "provider_symbol": script.OLD_SYMBOL,
                "status": "mismatch",
                "terminal_event_id": "",
                "identity_active_from": "2016-11-01",
                "identity_active_to": script.LAST_OLD_SYMBOL_SESSION,
            },
            {
                "target_id": script.NEW_PARENT_TARGET_ID,
                "security_id": script.PARENT_SECURITY_ID,
                "provider_symbol": script.NEW_PARENT_SYMBOL,
                "status": "passed",
                "reason": "",
                "identity_active_from": script.EFFECTIVE_DATE,
                "identity_active_to": "",
            },
        ],
    }


def test_exact_ticker_change_proposal_is_hash_pinned() -> None:
    assert script.TICKER_CHANGE_EVENT_ID == (
        "fb3d264732079815004e26780f47e9c816133970ad35ab903054fa5c97406a48"
    )
    proposal = script.proposed_nonterminal_extraction()
    assert proposal == {
        "event_id": script.TICKER_CHANGE_EVENT_ID,
        "security_id": script.PARENT_SECURITY_ID,
        "action_type": "ticker_change",
        "effective_date": "2020-04-01",
        "new_security_id": script.PARENT_SECURITY_ID,
        "new_symbol": "HWM",
        "ratio": None,
        "cash_amount": None,
        "currency": "USD",
        "source_kind": "official_crosscheck",
        "source_url": script.SP_SOURCE_URL,
        "source_hash": script.SP_SOURCE_HASH,
    }
    assert script.reviewed_nonterminal_extraction_sha256(proposal) == (
        "0121bd4918ff07fbab92be65b4ca12bd5546e83e3804aeb39266b573d2cb0ec5"
    )


def test_official_claim_extraction_is_fail_closed() -> None:
    sec = (
        "On April 1, 2020 the Separation was completed and became effective. "
        "One share of Arconic common stock for every four shares. "
        "Regular-way trading of Arconic began on April 1, 2020 under the "
        "ticker symbol ARNC."
    )
    sp = (
        "Howmet Aerospace Inc. (NYSE: HWM -formerly Arconic Inc.) spun off "
        "Arconic Corp. The separation was completed today, April 1. "
        "Post spin-off Howmet Aerospace will remain in the S&P 500."
    )
    assert all(script._official_evidence_checks(sec, sp).values())

    with pytest.raises(RuntimeError, match="official evidence text changed"):
        script._official_evidence_checks(
            sec.replace("every four shares", "every five shares"), sp
        )


def test_successor_projection_requires_one_passed_hwm_interval() -> None:
    projection = script._cross_validation_projection(
        _synthetic_cross_validation_report()
    )
    assert projection["projected_successor_binding"] == {
        "required": True,
        "passed": True,
        "target_id": script.NEW_PARENT_TARGET_ID,
        "provider_symbol": "HWM",
        "status": "passed",
        "reason": "",
        "candidate_count": 1,
    }

    failed = _synthetic_cross_validation_report()
    failed["prices"][1]["status"] = "mismatch"  # type: ignore[index]
    with pytest.raises(RuntimeError, match="cross-validation state changed"):
        script._cross_validation_projection(failed)


@pytest.mark.skipif(
    not RELEASE_PATH.is_file() or not POLICY_PATH.is_file(),
    reason="Pinned immutable ARNC audit release is not installed.",
)
def test_pinned_release_proves_repair_scope_without_external_access() -> None:
    policy = yaml.safe_load(POLICY_PATH.read_text())
    audit = script.build_audit(
        script.LocalDatasetRepository(PROJECT_ROOT / "data/cache"),
        _synthetic_cross_validation_report(),
        policy,
        backtest_summary_path=PROJECT_ROOT / "not-present-summary.json",
    )

    assert audit["conclusion"] == {
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
        "repair_state": "repair_required",
    }
    assert audit["stored_state"]["ticker_change_state"] == "missing"
    assert audit["stored_state"]["sp500_member_at_transition"] is True
    assert audit["required_changes"]["lifecycle_resolutions"]["change"] == "none"
    assert audit["required_changes"]["security_master"]["change"] == "none"
    assert audit["required_changes"]["symbol_history"]["change"] == "none"
    assert audit["required_changes"]["daily_price_raw"]["change"] == "none"
    policy_result = audit["required_changes"]["cross_validation_policy"]
    assert policy_result["state"] == "exact"
    assert policy_result["current_inventory_code_pin_passed"] is True
    assert policy_result["projected_inventory_sha256"] == (
        "39d4ea8e1852d57f0e47fce8a0e5a80e1ef0e80dd6f3412746f4e3d07adbd6c5"
    )
    assert audit["network_accessed"] is False
    assert audit["eodhd_calls"] == 0
    assert audit["r2_accessed"] is False
    assert audit["dataset_writes_performed"] is False
