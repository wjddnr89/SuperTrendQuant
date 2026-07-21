from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

from supertrend_quant.market_store.repository import LocalDatasetRepository


ROOT = Path(__file__).parents[2]
SCRIPT_PATH = ROOT / "unified_quant/scripts/audit_us_market_exit_gaps.py"
SPEC = importlib.util.spec_from_file_location("audit_us_market_exit_gaps", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Could not load {SCRIPT_PATH}")
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _current_report():
    return audit.build_audit(LocalDatasetRepository(ROOT / "data/cache"))


def test_exact_finite_market_exit_inventory_is_release_and_hash_pinned():
    report = _current_report()
    assert report["release_version"] == audit.PINNED_RELEASE_VERSION
    assert report["summary"] == {
        "case_count": 4,
        "dataset_repair_required_count": 4,
        "exact_otc_start_count": 2,
        "bounded_otc_start_count": 1,
        "unbound_otc_start_count": 1,
        "terminal_index_member_count": 0,
        "expected_triple_supertrend_trade_or_equity_delta": 0,
    }
    assert report["network_accessed"] is False
    assert report["http_attempts"] == 0
    assert report["eodhd_calls"] == 0
    assert report["r2_accessed"] is False
    assert report["dataset_writes_performed"] is False
    assert {item["symbol"] for item in report["cases"]} == {
        "WIN",
        "CHK",
        "FTR",
        "ENDP",
    }
    assert all(
        evidence["verified"]
        for item in report["cases"]
        for evidence in item["transition_evidence"]
    )
    assert all(
        item["official_cancellation_action"]["archive"]["verified"]
        for item in report["cases"]
    )


def test_price_tail_identity_successor_and_index_impact_are_exact():
    by_symbol = {item["symbol"]: item for item in _current_report()["cases"]}
    assert by_symbol["WIN"]["stored_price_tail"] == {
        "row_count": 1118,
        "first_session": "2015-01-02",
        "last_session": "2020-07-10",
        "last_positive_volume_session": "2019-06-28",
        "rows_on_or_after_otc_transition": 0,
        "zero_volume_flat_rows_after_last_positive_volume": 6,
        "rows_sha256": "3fd3dbd065ce606b840e9b62367cbc015851caf6b3e3bb35d833fc5870a2b312",
    }
    assert by_symbol["CHK"]["market_exit"] == {
        "exchange_suspension_date": "2020-06-29",
        "otc_first_date": "2020-06-30",
        "otc_first_date_status": "exact_official",
        "otc_symbol": "CHKAQ",
        "exchange_removal_date": "2020-07-31",
        "legacy_equity_cancellation_date": "2021-02-09",
    }
    assert by_symbol["CHK"]["successor"]["present_in_release"] is True
    assert by_symbol["CHK"]["successor"]["first_price_session"] == "2021-02-10"
    assert by_symbol["FTR"]["stored_price_tail"][
        "zero_volume_flat_rows_after_last_positive_volume"
    ] == 4
    assert by_symbol["FTR"]["market_exit"]["otc_first_date_status"] == (
        "expected_official_symbol_changed_by_confirmation"
    )
    assert by_symbol["ENDP"]["stored_price_tail"][
        "rows_on_or_after_otc_transition"
    ] == 44
    assert by_symbol["ENDP"]["market_exit"]["otc_symbol"] == "ENDPQ"
    assert all(
        scope["member_on_stored_terminal_session"] is False
        and scope["later_event_count"] == 0
        and scope["later_anchor_count"] == 0
        for item in by_symbol.values()
        for scope in item["index_scope"]
    )
    assert all(
        item["triple_supertrend_backtest"]["direct_delta_expected"] == 0
        for item in by_symbol.values()
    )


def test_cached_evidence_hash_or_claim_mutation_fails_closed(tmp_path: Path):
    relative = "state/sec_lifecycle/evidence.bin"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"official claim")
    with pytest.raises(RuntimeError, match="Cached SEC bytes changed"):
        audit.verify_evidence_pin(
            tmp_path,
            audit.EvidencePin(
                cache_object=relative,
                payload_sha256="0" * 64,
                required_patterns=(r"official claim",),
                claim="test",
            ),
        )
    with pytest.raises(RuntimeError, match="Pinned SEC claim changed"):
        audit.verify_evidence_pin(
            tmp_path,
            audit.EvidencePin(
                cache_object=relative,
                payload_sha256=hashlib.sha256(b"official claim").hexdigest(),
                required_patterns=(r"different claim",),
                claim="test",
            ),
        )
