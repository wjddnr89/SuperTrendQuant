from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import exchange_calendars as xcals

from supertrend_quant.ledger import PortfolioLedger
from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.lifecycle_coverage import lifecycle_candidate_id
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.portfolio import Position
from supertrend_quant.runners import _corporate_action_sort_key


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_kraft_special_dividend.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_kraft_special_dividend", SCRIPT_PATH
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


def _document(*phrases: str) -> bytes:
    return ("<html><body>" + " ".join(phrases) + "</body></html>").encode()


@pytest.fixture
def pinned_evidence(monkeypatch: pytest.MonkeyPatch):
    declaration = _document(
        "June 22, 2015",
        "special cash dividend in the amount of $16.50 per share",
        "conditioned upon the closing of the proposed merger",
        "payable to Kraft shareholders of record immediately prior",
    )
    payment = _document(
        "consummated on July 2, 2015",
        "on a one-for-one basis",
        "Upon the completion of the 2015 Merger",
        "received a special cash dividend of $16.50 per share",
    )
    completion = _document(
        "On July 2, 2015",
        "converted into the right to receive one fully paid",
        "on June 22, 2015, Kraft declared a special cash dividend",
        "$16.50 per share of Kraft Common Stock",
        "shareholders of record immediately prior to the closing",
    )

    def make_spec(
        label: str,
        content: bytes,
        required: tuple[tuple[str, ...], ...],
        *,
        archived: bool = False,
    ) -> script.EvidenceSpec:
        digest = hashlib.sha256(content).hexdigest()
        suffix = "txt" if archived else "html"
        return script.EvidenceSpec(
            label=label,
            source_url=f"https://www.sec.gov/Archives/fixture/{label}.{suffix}",
            source_hash=digest,
            size=len(content),
            retrieved_at="2026-07-18T00:00:00Z",
            filename="" if archived else f"{digest}.{suffix}",
            archive_object_path=(
                f"archives/{script.POLICY_AS_OF}/{digest}.{suffix}.gz"
            ),
            content_type="text/plain" if archived else "text/html",
            required_text_groups=required,
            already_archived=archived,
        )

    declaration_spec = make_spec(
        "kraft_special_dividend_declaration",
        declaration,
        (
            ("June 22, 2015",),
            ("special cash dividend in the amount of $16.50 per share",),
            ("conditioned upon the closing of the proposed merger",),
            ("payable to Kraft shareholders of record immediately prior",),
        ),
    )
    payment_spec = make_spec(
        "kraft_special_dividend_completion_payment",
        payment,
        (
            ("consummated on July 2, 2015",),
            ("on a one-for-one basis",),
            ("Upon the completion of the 2015 Merger",),
            ("received a special cash dividend of $16.50 per share",),
        ),
    )
    completion_spec = make_spec(
        "kraft_heinz_merger_completion",
        completion,
        (
            ("On July 2, 2015",),
            ("converted into the right to receive one fully paid",),
            ("on June 22, 2015, Kraft declared a special cash dividend",),
            ("$16.50 per share of Kraft Common Stock",),
            ("shareholders of record immediately prior to the closing",),
        ),
        archived=True,
    )
    monkeypatch.setattr(script, "DECLARATION_EVIDENCE", declaration_spec)
    monkeypatch.setattr(script, "PAYMENT_EVIDENCE", payment_spec)
    monkeypatch.setattr(script, "COMPLETION_EVIDENCE", completion_spec)
    monkeypatch.setattr(
        script,
        "EVIDENCE_SPECS",
        (declaration_spec, payment_spec, completion_spec),
    )
    return {
        declaration_spec.label: declaration,
        payment_spec.label: payment,
        completion_spec.label: completion,
    }


def _sessions(start: str, end: str, rows: int) -> list[pd.Timestamp]:
    candidates = [
        pd.Timestamp(value).tz_localize(None)
        for value in xcals.get_calendar("XNYS").sessions_in_range(start, end)
    ]
    assert len(candidates) == rows
    return candidates


def _price_rows(
    security_id: str,
    sessions: list[pd.Timestamp],
    *,
    default_close: float,
) -> list[dict[str, object]]:
    rows = []
    for session in sessions:
        close = (
            script.PRE_DIVIDEND_LAST_CLOSE
            if security_id == script.KRFT_ID
            and session.date().isoformat() == "2015-07-01"
            else default_close
        )
        rows.append(
            {
                "security_id": security_id,
                "session": session,
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 100_000,
                "currency": "USD",
                **_source(),
            }
        )
    return rows


def _base_frames(
    repository_root: Path,
    evidence: dict[str, bytes],
) -> dict[str, pd.DataFrame]:
    completion_spec = script.COMPLETION_EVIDENCE
    completion_path = repository_root / completion_spec.archive_object_path
    completion_path.parent.mkdir(parents=True, exist_ok=True)
    completion_path.write_bytes(
        gzip.compress(evidence[completion_spec.label], mtime=0)
    )
    master = pd.DataFrame(
        [
            {
                "security_id": script.KRFT_ID,
                "primary_symbol": script.KRFT_SYMBOL,
                "name": "Kraft Foods Group Inc",
                "exchange": "NASDAQ",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "active_from": "2015-01-02",
                "active_to": script.EFFECTIVE_DATE,
                **_source(),
            },
            {
                "security_id": script.KHC_ID,
                "primary_symbol": script.KHC_SYMBOL,
                "name": "Kraft Heinz Co",
                "exchange": "NASDAQ",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "active_from": script.KHC_FIRST_TRADING_SESSION,
                "active_to": "",
                **_source(),
            },
        ]
    )
    history = pd.DataFrame(
        [
            {
                "security_id": script.KRFT_ID,
                "symbol": script.KRFT_SYMBOL,
                "exchange": "NASDAQ",
                "effective_from": "2015-01-01",
                "effective_to": script.EFFECTIVE_DATE,
                **_source(),
            },
            {
                "security_id": script.KHC_ID,
                "symbol": script.KHC_SYMBOL,
                "exchange": "NASDAQ",
                "effective_from": script.EFFECTIVE_DATE,
                "effective_to": "",
                **_source(),
            },
        ]
    )
    krft_sessions = _sessions(
        "2015-01-02", script.EFFECTIVE_DATE, script.EXPECTED_KRFT_PRICE_ROWS
    )
    khc_sessions = _sessions(
        script.KHC_FIRST_TRADING_SESSION,
        script.POLICY_AS_OF,
        script.EXPECTED_KHC_PRICE_ROWS,
    )
    prices = pd.DataFrame(
        _price_rows(script.KRFT_ID, krft_sessions, default_close=88.19)
        + _price_rows(script.KHC_ID, khc_sessions, default_close=90.0)
    )
    actions = _empty("corporate_actions", "source", "retrieved_at", "source_hash", "metadata")
    regular = {column: None for column in actions.columns}
    regular.update(
        {
            "event_id": "regular-krft-dividend",
            "security_id": script.KRFT_ID,
            "action_type": "cash_dividend",
            "effective_date": "2015-04-08",
            "ex_date": "2015-04-08",
            "announcement_date": "2015-03-03",
            "record_date": "2015-04-10",
            "payment_date": "2015-04-24",
            "cash_amount": 0.55,
            "currency": "USD",
            "official": False,
            "source_url": "https://fixture.invalid/dividend",
            "source_kind": "provider",
            **_source(),
        }
    )
    merger = {column: None for column in actions.columns}
    merger.update(
        {
            "event_id": script.STOCK_MERGER_EVENT_ID,
            "security_id": script.KRFT_ID,
            "action_type": "stock_merger",
            "effective_date": script.EFFECTIVE_DATE,
            "ex_date": script.EFFECTIVE_DATE,
            "announcement_date": script.EFFECTIVE_DATE,
            "cash_amount": None,
            "ratio": 1.0,
            "currency": "USD",
            "new_security_id": script.KHC_ID,
            "new_symbol": script.KHC_SYMBOL,
            "official": True,
            "source_url": script.COMPLETION_EVIDENCE.source_url,
            "source_kind": "official_crosscheck",
            "source": "sec_edgar",
            "retrieved_at": script.COMPLETION_EVIDENCE.retrieved_at,
            "source_hash": script.COMPLETION_EVIDENCE.source_hash,
        }
    )
    actions = pd.DataFrame([regular, merger]).loc[:, actions.columns]
    factors = build_adjustment_factors(
        prices,
        actions,
        source_version="base-price+base-actions",
    )
    anchors = _empty(
        "index_constituent_anchors", "source", "retrieved_at", "source_hash"
    )
    events = _empty(
        "index_membership_events", "source", "retrieved_at", "source_hash"
    )
    resolutions = pd.DataFrame(
        [
            {
                "candidate_id": lifecycle_candidate_id(
                    script.KRFT_ID, script.EFFECTIVE_DATE
                ),
                "security_id": script.KRFT_ID,
                "symbol": script.KRFT_SYMBOL,
                "last_price_date": script.EFFECTIVE_DATE,
                "resolution": "applied",
                "event_id": script.STOCK_MERGER_EVENT_ID,
                "exception_code": "",
                "exception_reason": "",
                "reviewed_by": "fixture",
                "reviewed_at": "2026-07-18T00:00:00Z",
                "recheck_after": "",
                "successor_security_id": script.KHC_ID,
                "successor_symbol": script.KHC_SYMBOL,
                "source_url": script.COMPLETION_EVIDENCE.source_url,
                "source": "fixture",
                "retrieved_at": script.COMPLETION_EVIDENCE.retrieved_at,
                "source_hash": script.COMPLETION_EVIDENCE.source_hash,
            }
        ]
    )
    archive = pd.DataFrame(
        [
            {
                "archive_id": completion_spec.source_hash,
                "dataset": "sec_edgar_filing",
                "object_path": completion_spec.archive_object_path,
                "content_type": completion_spec.content_type,
                "effective_date": script.EFFECTIVE_DATE,
                "source": "sec_edgar_filing",
                "retrieved_at": completion_spec.retrieved_at,
                "source_hash": completion_spec.source_hash,
                "source_url": completion_spec.source_url,
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
    evidence: dict[str, bytes],
) -> LocalDatasetRepository:
    evidence_dir = root / "state/issuer_lifecycle"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in (script.DECLARATION_EVIDENCE, script.PAYMENT_EVIDENCE):
        content = evidence[spec.label]
        (evidence_dir / spec.filename).write_bytes(content)
        rows.append(
            {
                "label": spec.label,
                "source_url": spec.source_url,
                "source_hash": spec.source_hash,
                "size": spec.size,
                "filename": spec.filename,
                "retrieved_at": spec.retrieved_at,
            }
        )
    (evidence_dir / script.EVIDENCE_REPORT).write_text(
        json.dumps({"evidence": rows}), encoding="utf-8"
    )
    repository = LocalDatasetRepository(root)
    frames = _base_frames(root, evidence)
    versions = {}
    for dataset in script.REQUIRED_DATASETS:
        result = repository.write_frame(
            dataset,
            frames[dataset],
            completed_session=script.POLICY_AS_OF,
            incomplete_action_policy="block",
            metadata={"parent_marker": dataset},
            version=f"base-{dataset}",
        )
        versions[dataset] = result.manifest.version
    repository.commit_release(script.POLICY_AS_OF, versions, quality="degraded", warnings=("existing warning",))
    return repository


def test_offline_plan_restores_exact_pre_merger_distribution(
    tmp_path: Path,
    pinned_evidence,
):
    repository = _repository(tmp_path, pinned_evidence)
    prepared = script.prepare_repair(
        repository, evidence_dir=tmp_path / "state/issuer_lifecycle"
    )

    assert prepared.summary["status"] == "validated_offline_plan"
    assert prepared.summary["network_accessed"] is False
    assert prepared.summary["eodhd_calls"] == 0
    assert prepared.summary["r2_accessed"] is False
    assert prepared.summary["adjustment_factor_value_changes"] == 125
    assert prepared.summary["source_archive_rows_added"] == 2
    action = prepared.frames["corporate_actions"].loc[
        lambda frame: frame["event_id"].eq(script.SPECIAL_DIVIDEND_EVENT_ID)
    ].iloc[0]
    assert action["security_id"] == script.KRFT_ID
    assert action["action_type"] == "special_dividend"
    assert float(action["cash_amount"]) == pytest.approx(16.50)
    assert str(action["announcement_date"])[:10] == "2015-06-22"
    assert str(action["record_date"])[:10] == "2015-07-02"
    assert str(action["payment_date"])[:10] == "2015-07-02"
    assert str(action["effective_date"])[:10] == "2015-07-02"


def test_same_day_dividend_is_entitled_before_stock_merger(pinned_evidence):
    columns = pd.Index((*dataset_spec("corporate_actions").required_columns, "metadata"))
    dividend = script._expected_action(columns)
    dividend["symbol"] = script.KRFT_SYMBOL
    merger = {
        "event_id": script.STOCK_MERGER_EVENT_ID,
        "security_id": script.KRFT_ID,
        "symbol": script.KRFT_SYMBOL,
        "action_type": "stock_merger",
        "effective_date": script.EFFECTIVE_DATE,
        "ex_date": script.EFFECTIVE_DATE,
        "ratio": 1.0,
        "new_security_id": script.KHC_ID,
        "new_symbol": script.KHC_SYMBOL,
    }
    assert _corporate_action_sort_key(dividend) < _corporate_action_sort_key(merger)
    ledger = PortfolioLedger(
        cash=0.0,
        positions={
            script.KRFT_SYMBOL: Position(script.KRFT_SYMBOL, 2.0, 80.0)
        },
    )
    ledger.apply_actions([merger, dividend], through=script.EFFECTIVE_DATE)
    assert ledger.cash == pytest.approx(33.0)
    assert script.KRFT_SYMBOL not in ledger.positions
    assert ledger.positions[script.KHC_SYMBOL].quantity == pytest.approx(2.0)


def test_apply_is_cas_bound_preserves_quality_and_is_idempotent(
    tmp_path: Path,
    pinned_evidence,
):
    repository = _repository(tmp_path, pinned_evidence)
    evidence_dir = tmp_path / "state/issuer_lifecycle"
    prepared = script.prepare_repair(repository, evidence_dir=evidence_dir)
    applied = script.apply_repair(
        repository, prepared, evidence_dir=evidence_dir
    )
    assert applied["status"] == "applied"
    release, _ = repository.current_release()
    assert release is not None
    for dataset in script.WRITE_DATASETS:
        manifest = repository.manifest_for_version(
            dataset, release.dataset_versions[dataset]
        )
        assert manifest.metadata["parent_marker"] == dataset
        assert manifest.metadata["operation"] == "repair_us_kraft_special_dividend"
    assert release.quality == "degraded"
    assert release.warnings == ("existing warning",)
    actions = repository.read_frame(
        "corporate_actions", release.dataset_versions["corporate_actions"]
    )
    assert actions["event_id"].eq(script.SPECIAL_DIVIDEND_EVENT_ID).sum() == 1
    for spec in (script.DECLARATION_EVIDENCE, script.PAYMENT_EVIDENCE):
        assert gzip.decompress((tmp_path / spec.archive_object_path).read_bytes()) == pinned_evidence[spec.label]
    second = script.prepare_repair(repository, evidence_dir=evidence_dir)
    assert second.summary["status"] == "already_applied"
    result = script.apply_repair(repository, second, evidence_dir=evidence_dir)
    assert result["writes_performed"] is False


def test_apply_rebinds_full_factor_lineage_without_unrelated_economic_changes(
    tmp_path: Path,
    pinned_evidence,
):
    repository = _repository(tmp_path, pinned_evidence)
    evidence_dir = tmp_path / "state/issuer_lifecycle"
    base_release, _ = repository.current_release()
    assert base_release is not None
    before = repository.read_frame(
        "adjustment_factors",
        base_release.dataset_versions["adjustment_factors"],
    )

    prepared = script.prepare_repair(repository, evidence_dir=evidence_dir)
    expected_planned_source = script._adjustment_source_version(
        base_release.dataset_versions["daily_price_raw"],
        prepared.planned_versions["corporate_actions"],
    )
    assert prepared.summary["factor_source_version"] == expected_planned_source
    assert set(
        prepared.frames["adjustment_factors"]["source_version"].astype(str)
    ) == {expected_planned_source}
    assert set(
        prepared.frames["adjustment_factors"]["source_hash"].astype(str)
    ) == {expected_planned_source}

    script.apply_repair(repository, prepared, evidence_dir=evidence_dir)
    release, _ = repository.current_release()
    assert release is not None
    expected_source = script._adjustment_source_version(
        release.dataset_versions["daily_price_raw"],
        release.dataset_versions["corporate_actions"],
    )
    assert expected_source == expected_planned_source
    factor_manifest = repository.manifest_for_version(
        "adjustment_factors",
        release.dataset_versions["adjustment_factors"],
    )
    assert factor_manifest.metadata["parent_marker"] == "adjustment_factors"
    assert factor_manifest.metadata["source_version"] == expected_source
    assert (
        factor_manifest.metadata["source_daily_price_version"]
        == release.dataset_versions["daily_price_raw"]
    )
    assert (
        factor_manifest.metadata["source_corporate_actions_version"]
        == release.dataset_versions["corporate_actions"]
    )

    after = repository.read_frame(
        "adjustment_factors",
        release.dataset_versions["adjustment_factors"],
    )
    assert set(after["source_version"].astype(str)) == {expected_source}
    assert set(after["source_hash"].astype(str)) == {expected_source}
    assert set(after["source"].astype(str)) == {"derived"}

    comparison = before[
        ["security_id", "session", "split_factor", "total_return_factor"]
    ].merge(
        after[["security_id", "session", "split_factor", "total_return_factor"]],
        on=["security_id", "session"],
        how="outer",
        validate="one_to_one",
        suffixes=("_before", "_after"),
        indicator=True,
    )
    assert comparison["_merge"].eq("both").all()
    changed = (
        comparison["split_factor_before"]
        .sub(comparison["split_factor_after"])
        .abs()
        .gt(1e-12)
        | comparison["total_return_factor_before"]
        .sub(comparison["total_return_factor_after"])
        .abs()
        .gt(1e-12)
    )
    assert int(changed.sum()) == script.EXPECTED_FACTOR_VALUE_CHANGES
    assert set(comparison.loc[changed, "security_id"].astype(str)) == {
        script.KRFT_ID
    }

    prices = repository.read_frame(
        "daily_price_raw", release.dataset_versions["daily_price_raw"]
    )
    actions = repository.read_frame(
        "corporate_actions", release.dataset_versions["corporate_actions"]
    )
    rebuilt = build_adjustment_factors(
        prices,
        actions,
        source_version=expected_source,
    )
    reproduced = after[
        ["security_id", "session", "split_factor", "total_return_factor"]
    ].merge(
        rebuilt[["security_id", "session", "split_factor", "total_return_factor"]],
        on=["security_id", "session"],
        how="outer",
        validate="one_to_one",
        suffixes=("_stored", "_rebuilt"),
        indicator=True,
    )
    assert reproduced["_merge"].eq("both").all()
    assert reproduced["split_factor_stored"].sub(
        reproduced["split_factor_rebuilt"]
    ).abs().max() <= 1e-12
    assert reproduced["total_return_factor_stored"].sub(
        reproduced["total_return_factor_rebuilt"]
    ).abs().max() <= 1e-12


def test_changed_release_blocks_all_writes(tmp_path: Path, pinned_evidence):
    repository = _repository(tmp_path, pinned_evidence)
    evidence_dir = tmp_path / "state/issuer_lifecycle"
    prepared = script.prepare_repair(repository, evidence_dir=evidence_dir)
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
        warnings=("intervening",),
        expected_etag=etag,
    )
    with pytest.raises(RuntimeError, match="Current release changed"):
        script.apply_repair(repository, prepared, evidence_dir=evidence_dir)
    assert all(
        repository.objects.get(repository.current_key(dataset)).data
        == old_pointers[dataset]
        for dataset in script.WRITE_DATASETS
    )


def test_failure_after_release_commit_rolls_back(tmp_path: Path, pinned_evidence):
    repository = _repository(tmp_path, pinned_evidence)
    evidence_dir = tmp_path / "state/issuer_lifecycle"
    prepared = script.prepare_repair(repository, evidence_dir=evidence_dir)
    old_release = repository.objects.get("releases/current.json").data
    old_pointers = {
        dataset: repository.objects.get(repository.current_key(dataset)).data
        for dataset in script.WRITE_DATASETS
    }

    def fail(stage: str) -> None:
        if stage == "after_release_commit":
            raise RuntimeError("injected Kraft rollback")

    with pytest.raises(RuntimeError, match="injected Kraft rollback"):
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
        (tmp_path / "transactions/us-kraft-special-dividend").glob("*.json")
    )
    assert len(journals) == 1
    assert json.loads(journals[0].read_bytes())["status"] == "rolled_back"


def test_evidence_tamper_is_rejected(tmp_path: Path, pinned_evidence):
    repository = _repository(tmp_path, pinned_evidence)
    evidence_dir = tmp_path / "state/issuer_lifecycle"
    path = evidence_dir / script.DECLARATION_EVIDENCE.filename
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        script.prepare_repair(repository, evidence_dir=evidence_dir)


def test_cli_is_read_only_by_default():
    args = script._parse_args([])
    assert args.apply is False
    assert args.cache_root == script.DEFAULT_CACHE_ROOT
    assert args.evidence_dir == script.DEFAULT_EVIDENCE_DIR
