from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from supertrend_quant.config import parse_config
from supertrend_quant.data import MarketData
from supertrend_quant.ledger import PortfolioLedger
from supertrend_quant.market_store.lifecycle_coverage import (
    lifecycle_candidate_id,
    validate_lifecycle_coverage,
)
from supertrend_quant.market_store.manifest import DataRelease
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.portfolio import OrderPlan, Position
from supertrend_quant.runners import run_backtest_on_data


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_abmd_cvr_lower_bound.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_abmd_cvr_lower_bound", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def _source() -> dict[str, object]:
    return {
        "source": "fixture",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": "fixture-source",
    }


def _empty(dataset: str, *extra: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=tuple(dict.fromkeys((*dataset_spec(dataset).required_columns, *extra)))
    )


def _evidence_bytes() -> bytes:
    return b"""
    <html><body>
      December 22, 2022
      $380.00 per Company Share
      one non-tradeable contractual contingent value right per Company Share
      up to $35.00 per Company Share
    </body></html>
    """


def _valuation_bytes() -> bytes:
    return b"%PDF-1.7\nfixture JNJ annual report with remaining CVR balance\n%%EOF\n"


@pytest.fixture
def pinned_evidence(monkeypatch: pytest.MonkeyPatch):
    primary = _evidence_bytes()
    primary_hash = hashlib.sha256(primary).hexdigest()
    primary_spec = script.EvidenceSpec(
        source_url="https://www.sec.gov/Archives/fixture-abmd.txt",
        source_hash=primary_hash,
        uncompressed_size=len(primary),
        archive_object_path=f"archives/{script.POLICY_AS_OF}/{primary_hash}.txt.gz",
        retrieved_at="2026-07-18T00:00:00Z",
        required_text_groups=(
            ("December 22, 2022",),
            ("$380.00 per Company Share",),
            (
                "one non-tradeable contractual contingent value right per Company Share",
            ),
            ("up to $35.00 per Company Share",),
        ),
    )
    valuation = _valuation_bytes()
    valuation_hash = hashlib.sha256(valuation).hexdigest()
    valuation_spec = script.ValuationEvidenceSpec(
        source_url="https://investor.jnj.com/fixture-2025-annual-report.pdf",
        source_hash=valuation_hash,
        size=len(valuation),
        retrieved_at="2026-07-18T00:01:00Z",
        report_as_of="2025-12-28",
        remaining_cvr_liability_usd_billions=0.4,
    )
    monkeypatch.setattr(script, "EVIDENCE", primary_spec)
    monkeypatch.setattr(script, "VALUATION_EVIDENCE", valuation_spec)
    return primary, valuation


def _base_frames(primary: bytes, repository_root: Path) -> dict[str, pd.DataFrame]:
    source = _source()
    evidence = script.EVIDENCE
    archive_path = repository_root / evidence.archive_object_path
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(gzip.compress(primary, mtime=0))

    master = pd.DataFrame(
        [
            {
                "security_id": script.ABMD_SECURITY_ID,
                "primary_symbol": script.ABMD_SYMBOL,
                "name": "ABIOMED, Inc.",
                "exchange": "NASDAQ",
                "asset_type": "Common Stock",
                "currency": "USD",
                "country": "USA",
                "active_from": "2015-01-01",
                "active_to": script.LAST_REAL_SESSION,
                **source,
            },
            {
                "security_id": "CURRENT",
                "primary_symbol": "CURRENT",
                "name": "Current fixture security",
                "exchange": "NYSE",
                "asset_type": "Common Stock",
                "currency": "USD",
                "country": "USA",
                "active_from": "2020-01-01",
                "active_to": "",
                **source,
            },
        ]
    )
    history = pd.DataFrame(
        [
            {
                "security_id": script.ABMD_SECURITY_ID,
                "symbol": script.ABMD_SYMBOL,
                "exchange": "NASDAQ",
                "effective_from": "2015-01-01",
                "effective_to": script.LAST_REAL_SESSION,
                **source,
            },
            {
                "security_id": "CURRENT",
                "symbol": "CURRENT",
                "exchange": "NYSE",
                "effective_from": "2020-01-01",
                "effective_to": "",
                **source,
            },
        ]
    )
    prices = pd.DataFrame(
        [
            {
                "security_id": script.ABMD_SECURITY_ID,
                "session": "2015-01-02",
                "open": 40.0,
                "high": 41.0,
                "low": 39.0,
                "close": 40.0,
                "volume": 100_000,
                "currency": "USD",
                **source,
            },
            {
                "security_id": script.ABMD_SECURITY_ID,
                "session": script.LAST_REAL_SESSION,
                "open": 380.0,
                "high": 382.0,
                "low": 379.0,
                "close": script.LAST_REAL_CLOSE,
                "volume": 1_000_000,
                "currency": "USD",
                **source,
            },
            {
                "security_id": "CURRENT",
                "session": script.POLICY_AS_OF,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 100_000,
                "currency": "USD",
                **source,
            },
        ]
    )
    actions = _empty("corporate_actions", "metadata")
    factors = pd.DataFrame(
        [
            {
                "security_id": script.ABMD_SECURITY_ID,
                "session": "2015-01-02",
                "split_factor": 1.0,
                "total_return_factor": 1.0,
                "source_version": "base-daily_price_raw+base-corporate_actions",
                "calculated_at": "2026-07-18T00:00:00Z",
                "source": "derived",
                "retrieved_at": "2026-07-18T00:00:00Z",
                "source_hash": "base-daily_price_raw+base-corporate_actions",
            },
            {
                "security_id": script.ABMD_SECURITY_ID,
                "session": script.LAST_REAL_SESSION,
                "split_factor": 1.0,
                "total_return_factor": 1.0,
                "source_version": "base-daily_price_raw+base-corporate_actions",
                "calculated_at": "2026-07-18T00:00:00Z",
                "source": "derived",
                "retrieved_at": "2026-07-18T00:00:00Z",
                "source_hash": "base-daily_price_raw+base-corporate_actions",
            },
            {
                "security_id": "CURRENT",
                "session": script.POLICY_AS_OF,
                "split_factor": 1.0,
                "total_return_factor": 1.0,
                "source_version": "base-daily_price_raw+base-corporate_actions",
                "calculated_at": "2026-07-18T00:00:00Z",
                "source": "derived",
                "retrieved_at": "2026-07-18T00:00:00Z",
                "source_hash": "base-daily_price_raw+base-corporate_actions",
            },
        ]
    )
    anchors = pd.DataFrame(
        [
            {
                "index_id": "SP500",
                "anchor_date": "2015-01-01",
                "security_id": script.ABMD_SECURITY_ID,
                "official": True,
                "source_url": "https://www.spglobal.com/fixture",
                "source_kind": "official",
                **source,
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "remove-abmd",
                "index_id": "SP500",
                "announcement_date": script.EFFECTIVE_DATE,
                "effective_date": script.EFFECTIVE_DATE,
                "operation": "REMOVE",
                "security_id": script.ABMD_SECURITY_ID,
                "official": True,
                "source_url": "https://www.spglobal.com/fixture",
                "source_kind": "official",
                **source,
            }
        ]
    )
    resolutions = pd.DataFrame(
        [
            {
                "candidate_id": lifecycle_candidate_id(
                    script.ABMD_SECURITY_ID, script.LAST_REAL_SESSION
                ),
                "security_id": script.ABMD_SECURITY_ID,
                "symbol": script.ABMD_SYMBOL,
                "last_price_date": script.LAST_REAL_SESSION,
                "resolution": "exception",
                "event_id": "",
                "exception_code": "unsupported_consideration",
                "exception_reason": "Cash plus a non-tradeable CVR.",
                "reviewed_by": "fixture-reviewer",
                "reviewed_at": "2026-07-18T00:00:00Z",
                "recheck_after": "",
                "successor_security_id": "",
                "successor_symbol": "",
                "source_url": evidence.source_url,
                "source": "sec_edgar_filing",
                "retrieved_at": evidence.retrieved_at,
                "source_hash": evidence.source_hash,
            }
        ]
    )
    archive = pd.DataFrame(
        [
            {
                "archive_id": evidence.source_hash,
                "dataset": "sec_edgar_filing",
                "object_path": evidence.archive_object_path,
                "content_type": "text/plain",
                "effective_date": script.POLICY_AS_OF,
                "source": "sec_edgar_filing",
                "retrieved_at": evidence.retrieved_at,
                "source_hash": evidence.source_hash,
                "source_url": evidence.source_url,
            }
        ]
    )
    return {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "adjustment_factors": factors,
        "index_constituent_anchors": anchors,
        "index_membership_events": events,
        "lifecycle_resolutions": resolutions,
        "source_archive": archive,
    }


def _repository(
    root: Path,
    primary: bytes,
    valuation: bytes,
) -> LocalDatasetRepository:
    repository = LocalDatasetRepository(root)
    evidence_dir = root / "state/issuer_lifecycle"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / script.VALUATION_EVIDENCE.filename).write_bytes(valuation)
    frames = _base_frames(primary, root)
    candidates = frames["lifecycle_resolutions"][
        ["candidate_id", "security_id", "last_price_date"]
    ]
    coverage = validate_lifecycle_coverage(
        candidates,
        frames["lifecycle_resolutions"],
        frames["corporate_actions"],
        completed_session=script.POLICY_AS_OF,
    )
    assert coverage.valid
    versions: dict[str, str] = {}
    for dataset in script.REQUIRED_DATASETS:
        metadata = (
            {
                **coverage.manifest_metadata(),
                "evidence_report_sha256": "e" * 64,
            }
            if dataset == "lifecycle_resolutions"
            else None
        )
        result = repository.write_frame(
            dataset,
            frames[dataset],
            completed_session=script.POLICY_AS_OF,
            incomplete_action_policy="block",
            metadata=metadata,
            version=f"base-{dataset}",
        )
        versions[dataset] = result.manifest.version
    repository.commit_release(
        script.POLICY_AS_OF,
        versions,
        quality="valid",
    )
    return repository


def test_plan_installs_only_guaranteed_cash_and_audited_zero_lower_bound(
    tmp_path: Path,
    pinned_evidence,
):
    primary, valuation = pinned_evidence
    repository = _repository(tmp_path, primary, valuation)
    prepared = script.prepare_repair(
        repository,
        evidence_dir=tmp_path / "state/issuer_lifecycle",
        official_evidence_specs={},
    )

    assert prepared.summary["status"] == "validated_offline_plan"
    assert prepared.summary["network_accessed"] is False
    assert prepared.summary["eodhd_calls"] == 0
    assert prepared.summary["r2_accessed"] is False
    assert prepared.coverage.open_count == 0
    assert prepared.coverage.applied_count == 1
    assert prepared.coverage.exception_count == 0
    assert not prepared.source_candidate_drift

    action = prepared.frames["corporate_actions"].iloc[0]
    assert action["event_id"] == script.EVENT_ID
    assert action["action_type"] == "cash_merger"
    assert float(action["cash_amount"]) == pytest.approx(380.0)
    metadata = json.loads(action["metadata"])
    assert metadata["cvr"]["quantity_per_share"] == 1.0
    assert metadata["cvr"]["mark_per_right"] == 0.0
    assert metadata["cvr"]["valuation_policy"] == "zero_mark_lower_bound"
    assert metadata["lower_bound_rationale"]["zero_is_fair_value_estimate"] is False
    assert metadata["lower_bound_rationale"][
        "remaining_aggregate_cvr_liability_usd_billions"
    ] == pytest.approx(0.4)
    assert script.LOWER_BOUND_WARNING in prepared.warnings

    resolution = prepared.frames["lifecycle_resolutions"].iloc[0]
    assert resolution["resolution"] == "applied"
    assert resolution["event_id"] == script.EVENT_ID
    assert resolution["exception_code"] == ""


def test_cash_merger_executes_380_but_does_not_fabricate_cvr_value(
    pinned_evidence,
):
    action = script._expected_action()
    action["symbol"] = script.ABMD_SYMBOL
    ledger = PortfolioLedger(
        cash=0.0,
        positions={script.ABMD_SYMBOL: Position(script.ABMD_SYMBOL, 2.0, 300.0)},
    )

    events = ledger.apply_actions([action], through=script.EFFECTIVE_DATE)

    assert len(events) == 1
    assert ledger.cash == pytest.approx(760.0)
    assert script.ABMD_SYMBOL not in ledger.positions
    assert not ledger.unresolved_event_ids
    assert json.loads(action["metadata"])["cvr"]["mark_per_right"] == 0.0


def test_lower_bound_warning_survives_into_final_backtest_result(
    monkeypatch: pytest.MonkeyPatch,
):
    class NoopStrategy:
        def warmup_bars(self):
            return 0

        def build_order_plan(self, _bars, _account, mode, **_kwargs):
            return OrderPlan("noop", mode, ())

    config = parse_config(
        {
            "strategy": {"name": "test", "type": "equal", "params": {}},
            "scoring": {
                "type": "relative_strength",
                "params": {"lookback_bars": 1},
            },
            "market": "US",
            "universe": {"source": "symbols", "symbols": ["AAA"]},
            "period": "max",
        }
    )
    index = pd.date_range("2026-07-13", periods=3, freq="D")
    bars = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.0, 101.0, 102.0],
        },
        index=index,
    )
    market_data = MarketData(
        bars={"AAA": bars},
        execution_bars={"AAA": bars},
        data_quality="degraded",
        warnings=(script.LOWER_BOUND_WARNING,),
    )
    monkeypatch.setattr(
        "supertrend_quant.runners.create_strategy", lambda _config: NoopStrategy()
    )

    result = run_backtest_on_data(config, market_data)

    assert result.data_quality == "degraded"
    assert script.LOWER_BOUND_WARNING in result.data_warnings


def test_apply_is_cas_bound_degraded_warned_and_idempotent(
    tmp_path: Path,
    pinned_evidence,
):
    primary, valuation = pinned_evidence
    repository = _repository(tmp_path, primary, valuation)
    evidence_dir = tmp_path / "state/issuer_lifecycle"
    prepared = script.prepare_repair(
        repository, evidence_dir=evidence_dir, official_evidence_specs={}
    )
    retained_factor_version = prepared.release.dataset_versions[
        "adjustment_factors"
    ]

    applied = script.apply_repair(
        repository, prepared, evidence_dir=evidence_dir
    )

    assert applied["status"] == "applied"
    assert applied["quality"] == "degraded"
    assert script.LOWER_BOUND_WARNING in applied["warnings"]
    release, _ = repository.current_release()
    assert release is not None
    assert release.quality == "degraded"
    assert script.LOWER_BOUND_WARNING in release.warnings
    actions = repository.read_frame(
        "corporate_actions", release.dataset_versions["corporate_actions"]
    )
    assert actions["event_id"].astype(str).eq(script.EVENT_ID).sum() == 1
    resolutions = repository.read_frame(
        "lifecycle_resolutions", release.dataset_versions["lifecycle_resolutions"]
    )
    assert resolutions.iloc[0]["resolution"] == "applied"
    factors = repository.read_frame(
        "adjustment_factors", release.dataset_versions["adjustment_factors"]
    )
    assert release.dataset_versions["adjustment_factors"] == retained_factor_version
    assert set(factors["source_version"].astype(str)) == {
        "base-daily_price_raw+base-corporate_actions"
    }
    action_manifest = repository.manifest_for_version(
        "corporate_actions", release.dataset_versions["corporate_actions"]
    )
    assert action_manifest.quality == "degraded"
    assert script.LOWER_BOUND_WARNING in action_manifest.warnings
    archived = tmp_path / script.VALUATION_EVIDENCE.archive_object_path
    assert gzip.decompress(archived.read_bytes()) == valuation

    second = script.prepare_repair(
        repository, evidence_dir=evidence_dir, official_evidence_specs={}
    )
    assert second.summary["status"] == "already_applied"
    result = script.apply_repair(repository, second, evidence_dir=evidence_dir)
    assert result["writes_performed"] is False


def test_changed_release_cas_prevents_all_dataset_writes(
    tmp_path: Path,
    pinned_evidence,
):
    primary, valuation = pinned_evidence
    repository = _repository(tmp_path, primary, valuation)
    evidence_dir = tmp_path / "state/issuer_lifecycle"
    prepared = script.prepare_repair(
        repository, evidence_dir=evidence_dir, official_evidence_specs={}
    )
    release, etag = repository.current_release()
    assert release is not None
    old_pointers = {
        dataset: repository.objects.get(repository.current_key(dataset)).data
        for dataset in script.WRITE_DATASETS
    }
    repository.commit_release(
        release.completed_session,
        release.dataset_versions,
        quality=release.quality,
        warnings=("intervening release",),
        expected_etag=etag,
    )

    with pytest.raises(RuntimeError, match="Current release changed"):
        script.apply_repair(repository, prepared, evidence_dir=evidence_dir)

    assert all(
        repository.objects.get(repository.current_key(dataset)).data
        == old_pointers[dataset]
        for dataset in script.WRITE_DATASETS
    )


def test_failure_after_release_commit_rolls_back_release_and_pointers(
    tmp_path: Path,
    pinned_evidence,
):
    primary, valuation = pinned_evidence
    repository = _repository(tmp_path, primary, valuation)
    evidence_dir = tmp_path / "state/issuer_lifecycle"
    prepared = script.prepare_repair(
        repository, evidence_dir=evidence_dir, official_evidence_specs={}
    )
    old_release = repository.objects.get("releases/current.json").data
    old_pointers = {
        dataset: repository.objects.get(repository.current_key(dataset)).data
        for dataset in script.WRITE_DATASETS
    }

    def fail(stage: str) -> None:
        if stage == "after_release_commit":
            raise RuntimeError("injected ABMD rollback")

    with pytest.raises(RuntimeError, match="injected ABMD rollback"):
        script.apply_repair(
            repository,
            prepared,
            evidence_dir=evidence_dir,
            inject_failure=fail,
        )

    assert repository.objects.get("releases/current.json").data == old_release
    assert all(
        repository.objects.get(repository.current_key(dataset)).data
        == old_pointers[dataset]
        for dataset in script.WRITE_DATASETS
    )
    journals = tuple(
        (tmp_path / "transactions/us-abmd-cvr-lower-bound").glob("*.json")
    )
    assert len(journals) == 1
    assert json.loads(journals[0].read_bytes())["status"] == "rolled_back"


def test_pinned_valuation_evidence_tamper_is_rejected(
    tmp_path: Path,
    pinned_evidence,
):
    primary, valuation = pinned_evidence
    repository = _repository(tmp_path, primary, valuation)
    evidence_dir = tmp_path / "state/issuer_lifecycle"
    (evidence_dir / script.VALUATION_EVIDENCE.filename).write_bytes(
        valuation + b"tampered"
    )

    with pytest.raises(ValueError, match="hash/size mismatch"):
        script.prepare_repair(
            repository, evidence_dir=evidence_dir, official_evidence_specs={}
        )


def test_cli_defaults_to_strict_read_only_plan():
    args = script._parse_args([])
    assert args.apply is False
    assert args.cache_root == script.DEFAULT_CACHE_ROOT
    assert args.evidence_dir == script.DEFAULT_ISSUER_EVIDENCE_DIR
    assert args.hints == script.DEFAULT_HINTS
