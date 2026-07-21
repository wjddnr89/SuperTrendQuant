from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    DatasetManifest,
)
from supertrend_quant.market_store.repository import (
    DatasetWriteResult,
    LocalDatasetRepository,
)
from supertrend_quant.market_store.storage import LocalObjectStore
from supertrend_quant.market_store.validation import ValidationIssue, ValidationReport


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_early_terminal_history_supplements.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_early_terminal_history_supplements", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2] / "data" / "cache"


def _control_hashes(repository: LocalDatasetRepository) -> dict[str, str]:
    keys = ["releases/current.json"] + [
        repository.current_key(dataset) for dataset in script.REQUIRED_DATASETS
    ]
    budget = repository.root / "state/eodhd_call_budget.json"
    output = {
        key: hashlib.sha256(repository.objects.get(key).data).hexdigest()
        for key in keys
    }
    if budget.is_file():
        output[str(budget)] = hashlib.sha256(budget.read_bytes()).hexdigest()
    return output


@pytest.fixture(scope="module")
def actual_plan():
    if not (REPOSITORY_ROOT / "releases/current.json").is_file():
        pytest.skip("Actual current release is unavailable.")
    repository = LocalDatasetRepository(REPOSITORY_ROOT)
    before = _control_hashes(repository)
    prepared = script.prepare_repair(repository)
    after = _control_hashes(repository)
    return repository, prepared, before, after


def test_actual_current_plan_is_read_only_offline_and_minimal(actual_plan) -> None:
    repository, prepared, before, after = actual_plan
    release, _ = repository.current_release()
    assert release is not None
    assert prepared.release.version == release.version
    assert before == after
    summary = prepared.summary
    assert summary["status"] in {
        "awaiting_eodhd_fetch",
        "validated_offline_plan",
        "already_repaired",
    }
    if summary["status"] == "awaiting_eodhd_fetch":
        assert summary["apply_ready"] is False
        assert "PETM" not in summary["missing_symbols"]
        assert set(summary["missing_symbols"]).issubset({"SWY", "CFN", "AGN"})
        assert summary["eodhd_call_cap"] == 4
        assert summary["eodhd_expected_calls"] == len(summary["missing_symbols"])
        assert summary["eodhd_http_attempts_this_plan"] == 0
        assert summary["cov_insert_rows"] == 44
        assert summary["total_insert_rows"] == 118
    assert summary["network_accessed"] is False
    assert summary["r2_accessed"] is False


def test_exact_request_inventory_and_legacy_alias_uncertainty_are_pinned() -> None:
    script._static_contract()
    assert script.request_inventory_sha256() == script.TRUSTED_REQUEST_INVENTORY_SHA256
    expected = {
        "PETM": ("2014-12-12", 14, 13),
        "SWY": ("2014-11-03", 42, 41),
        "CFN": ("2014-12-17", 11, 10),
        "AGN": ("2014-12-17", 11, 10),
    }
    for item in script.request_inventory():
        start, response_rows, inserts = expected[item["symbol"]]
        assert item["endpoint"] == "eod"
        assert item["from"] == start
        assert item["to"] == "2015-01-02"
        assert item["expected_response_rows"] == response_rows
        assert item["expected_insert_rows"] == inserts
        assert item["expected_sessions"][-1] == "2015-01-02"
    agn = next(item for item in script.request_inventory() if item["symbol"] == "AGN")
    assert agn["provider_symbol"] == "AGN_old.US"
    assert "pre-2015" in agn["provider_uncertainty"]


def test_existing_cov_archives_extend_all_60_sessions_without_network(actual_plan) -> None:
    repository, prepared, _, _ = actual_plan
    release = prepared.release
    archive = repository.read_frame(
        "source_archive", release.dataset_versions["source_archive"]
    )
    prices, artifacts, report = script._load_cov_primary_and_crosscheck(
        repository, archive
    )
    assert len(prices) == 60
    assert tuple(prices["session"].astype(str)) == script._terminal_window(
        script.COV_CASE
    )
    assert len(artifacts) == 2
    assert report["status"] == "passed"
    assert report["sessions_compared"] == 60
    assert report["maximum_absolute_delta"]["low"] == pytest.approx(0.0004)
    assert report["maximum_absolute_delta"]["volume"] == 0
    assert report["dividend"]["cash_amount"] == 0.36
    assert round(report["dividend"]["inferred_unrounded_cash_amount"], 2) == 0.36


def test_actual_swy_fallback_is_exact_private_and_bound_to_failed_attempt(
    actual_plan,
) -> None:
    repository, prepared, _, _ = actual_plan
    archive = repository.read_frame(
        "source_archive", prepared.release.dataset_versions["source_archive"]
    )
    reviewed = script._read_swy_wiki_cache(repository, archive)
    assert reviewed is not None
    prices, extract, provenance, report = reviewed
    assert len(prices) == 42
    assert tuple(prices["session"].astype(str)) == script._request_sessions(
        script.SWY_CASE
    )
    assert extract.source_hash == script.SWY_WIKI_EXTRACT_SHA256
    assert report["dividend_ex_date"] == "2014-12-23"
    assert report["dividend_cash_amount"] == 0.23
    assert report["formal_license_name"] == "Unknown"
    assert report["allowed_scope"] == "private_internal_only"
    assert report["publication_allowed"] is False
    assert report["redistribution_allowed"] is False
    assert report["automatic_eodhd_retry_allowed"] is False
    evidence = json.loads(provenance.content)
    assert evidence["eodhd_fail_closed_binding"]["automatic_retry_allowed"] is False
    assert evidence["eodhd_fail_closed_binding"]["reason"] == (
        "undisclosed_historical_adjustment"
    )


def test_swy_cache_wrapper_tampering_fails_closed(
    actual_plan,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, prepared, _, _ = actual_plan
    archive = repository.read_frame(
        "source_archive", prepared.release.dataset_versions["source_archive"]
    )
    original = script._swy_wiki_cache_path(repository.root).read_bytes()
    tampered = tmp_path / "swy-tampered.json.gz"
    tampered.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    monkeypatch.setattr(script, "_swy_wiki_cache_path", lambda _root: tampered)
    with pytest.raises(ValueError, match="cache is invalid"):
        script._read_swy_wiki_cache(repository, archive)


def _response_payload(case: script.HistoryCase, overlap: pd.Series) -> bytes:
    records = []
    base = float(overlap["close"])
    for offset, session in enumerate(script._request_sessions(case)):
        close = base if session == script.EODHD_OVERLAP_SESSION else base - (
            len(script._request_sessions(case)) - offset
        ) * 0.01
        records.append(
            {
                "date": session,
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "adjusted_close": close,
                "volume": 1000 + offset,
            }
        )
    # The single overlap must be byte-for-byte equal in normalized raw fields.
    records[-1].update(
        {
            "open": float(overlap["open"]),
            "high": float(overlap["high"]),
            "low": float(overlap["low"]),
            "close": float(overlap["close"]),
            "adjusted_close": float(overlap["close"]),
            "volume": float(overlap["volume"]),
        }
    )
    return script._canonical_json(records)


def _synthetic_eod_evidence(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, script.SupplementalEvidence]:
    prices = repository.read_frame(
        "daily_price_raw", release.dataset_versions["daily_price_raw"]
    )
    archive = repository.read_frame(
        "source_archive", release.dataset_versions["source_archive"]
    )
    output, _ = script._load_supplemental_evidence(repository, archive)
    for case in script.EODHD_CASES:
        overlap = prices.loc[
            prices["security_id"].astype(str).eq(case.security_id)
            & pd.to_datetime(prices["session"])
            .dt.date.astype(str)
            .eq(script.EODHD_OVERLAP_SESSION)
        ].iloc[0]
        content = _response_payload(case, overlap)
        source_hash = script._sha256(content)
        normalized = script._parse_eodhd_response(
            case,
            content,
            source_hash=source_hash,
            retrieved_at="2026-07-19T00:00:00Z",
        )
        primary = script.EvidenceArtifact(
            dataset="eodhd_eod_history_supplement",
            source="eodhd_eod",
            source_url=case.request_url,
            retrieved_at="2026-07-19T00:00:00Z",
            content=content,
            content_type="application/json",
            effective_date="2026-07-15",
            suffix="json",
        )
        cross = {
            "status": "passed",
            "provider": "eodhd",
            "endpoint": "eod",
            "expected_response_rows": len(script._request_sessions(case)),
            "insert_rows": case.missing_session_count,
            "overlap_session": script.EODHD_OVERLAP_SESSION,
            "overlap_rows": 1,
            "adjusted_close_equals_close": True,
        }
        identity = script._identity_manifest_artifact(
            case, primary=primary, cross_validation=cross
        )
        output[case.symbol] = script.SupplementalEvidence(
            case=case,
            prices=normalized,
            primary_artifact=primary,
            identity_artifact=identity,
            all_artifacts=(primary, identity),
            overlap_rows=1,
            cross_validation=cross,
        )
    return output


def _exact_pre_repair_frames(
    current: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Reconstruct the pinned pre-repair state without mutating repository data."""
    frames = {dataset: frame.copy() for dataset, frame in current.items()}

    prices = frames["daily_price_raw"]
    remove_prices = pd.Series(False, index=prices.index)
    for case in script.CASES:
        remove_prices |= prices["security_id"].astype(str).eq(
            case.security_id
        ) & prices["session"].astype(str).isin(script._missing_sessions(case))
    frames["daily_price_raw"] = prices.loc[~remove_prices].reset_index(drop=True)

    supplement_dividends = {
        script.COV_DIVIDEND_EVENT_ID,
        script.SWY_DIVIDEND_EVENT_ID,
    }
    actions = frames["corporate_actions"]
    frames["corporate_actions"] = actions.loc[
        ~actions["event_id"].astype(str).isin(supplement_dividends)
    ].reset_index(drop=True)

    for dataset, history in (
        ("security_master", False),
        ("symbol_history", True),
    ):
        identity = frames[dataset]
        for case in script.CASES:
            mask = identity["security_id"].astype(str).eq(case.security_id)
            if history:
                mask &= identity["symbol"].astype(str).eq(case.symbol)
            assert int(mask.sum()) == 1
            identity.loc[
                mask, "effective_from" if history else "active_from"
            ] = case.old_history_from if history else case.old_active_from
            identity.loc[mask, "source"] = case.old_identity_source
            identity.loc[mask, "source_url"] = case.old_identity_url
            identity.loc[mask, "source_hash"] = case.old_identity_hash
            identity.loc[mask, "retrieved_at"] = case.old_identity_retrieved_at
        frames[dataset] = identity

    return frames


def test_eod_parser_requires_exact_rows_overlap_and_no_hidden_adjustment(
    actual_plan,
) -> None:
    repository, prepared, _, _ = actual_plan
    release = prepared.release
    prices = repository.read_frame(
        "daily_price_raw", release.dataset_versions["daily_price_raw"]
    )
    case = script.EODHD_CASES[0]
    overlap = prices.loc[
        prices["security_id"].astype(str).eq(case.security_id)
        & pd.to_datetime(prices["session"])
        .dt.date.astype(str)
        .eq(script.EODHD_OVERLAP_SESSION)
    ].iloc[0]
    content = _response_payload(case, overlap)
    frame = script._parse_eodhd_response(
        case,
        content,
        source_hash=script._sha256(content),
        retrieved_at="2026-07-19T00:00:00Z",
    )
    assert len(frame) == 14
    assert frame.iloc[-1]["session"] == "2015-01-02"

    payload = json.loads(content)
    payload.pop(0)
    fewer = script._canonical_json(payload)
    with pytest.raises(ValueError, match="row count is not exact"):
        script._parse_eodhd_response(
            case,
            fewer,
            source_hash=script._sha256(fewer),
            retrieved_at="2026-07-19T00:00:00Z",
        )
    payload = json.loads(content)
    payload[0]["adjusted_close"] += 0.01
    adjusted = script._canonical_json(payload)
    with pytest.raises(ValueError, match="undisclosed historical adjustment"):
        script._parse_eodhd_response(
            case,
            adjusted,
            source_hash=script._sha256(adjusted),
            retrieved_at="2026-07-19T00:00:00Z",
        )


def test_in_memory_candidate_adds_only_118_rows_and_is_exactly_repaired(
    actual_plan,
) -> None:
    repository, prepared, _, _ = actual_plan
    release = prepared.release
    current = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in script.REQUIRED_DATASETS
    }
    evidence = _synthetic_eod_evidence(repository, release)
    frames = _exact_pre_repair_frames(current)
    assert set(script._case_states(frames, evidence).values()) == {"old"}
    rewritten = dict(frames)
    rewritten["daily_price_raw"] = script._rewrite_prices(
        frames["daily_price_raw"], evidence
    )
    rewritten["corporate_actions"] = script._rewrite_actions(
        frames["corporate_actions"], evidence
    )
    rewritten["security_master"] = script._rewrite_identity(
        frames["security_master"], evidence, history=False
    )
    rewritten["symbol_history"] = script._rewrite_identity(
        frames["symbol_history"], evidence, history=True
    )
    assert len(rewritten["daily_price_raw"]) - len(frames["daily_price_raw"]) == 118
    assert len(rewritten["corporate_actions"]) - len(frames["corporate_actions"]) == 2
    assert set(script._case_states(rewritten, evidence).values()) == {"repaired"}


class _TransactionRepository:
    def __init__(self, root: Path):
        self.root = root
        self.objects = LocalObjectStore(root)
        self.manifests: dict[tuple[str, str], DatasetManifest] = {}
        self.frames: dict[tuple[str, str], pd.DataFrame] = {}
        self.write_records: list[tuple[str, str]] = []
        self.release_count = 0
        versions = {
            dataset: f"base-{dataset}" for dataset in script.REQUIRED_DATASETS
        }
        for dataset, version in versions.items():
            manifest = DatasetManifest.create(
                dataset,
                version,
                "2026-07-15",
                (),
                metadata={"preserved": dataset},
            )
            self.manifests[(dataset, version)] = manifest
            manifest_path = f"datasets/{dataset}/versions/{version}/manifest.json"
            self.objects.put(manifest_path, manifest.to_bytes(), if_none_match=True)
            pointer = CurrentPointer.create(manifest, manifest_path)
            self.objects.put(
                self.current_key(dataset), pointer.to_bytes(), if_none_match=True
            )
            self.frames[(dataset, version)] = pd.DataFrame({"base": [dataset]})
        release = DataRelease(
            version="base-release",
            created_at="2026-07-19T00:00:00Z",
            completed_session="2026-07-15",
            dataset_versions=versions,
            quality="valid",
            warnings=(),
        )
        self.objects.put("releases/current.json", release.to_bytes(), if_none_match=True)

    @staticmethod
    def current_key(dataset: str) -> str:
        return LocalDatasetRepository.current_key(dataset)

    def current_release(self):
        value = self.objects.get("releases/current.json")
        return DataRelease.from_bytes(value.data), value.etag

    def current_pointer(self, dataset: str):
        value = self.objects.get(self.current_key(dataset))
        return CurrentPointer.from_bytes(value.data), value.etag

    def manifest_for_version(self, dataset: str, version: str):
        return self.manifests[(dataset, version)]

    def write_frame(
        self,
        dataset: str,
        frame: pd.DataFrame,
        *,
        completed_session: str,
        incomplete_action_policy: str,
        metadata: dict,
        expected_pointer_etag: str | None,
        version: str,
    ):
        manifest = DatasetManifest.create(
            dataset, version, completed_session, (), metadata=metadata
        )
        manifest_path = f"datasets/{dataset}/versions/{version}/manifest.json"
        self.objects.put(manifest_path, manifest.to_bytes(), if_none_match=True)
        pointer = CurrentPointer.create(manifest, manifest_path)
        self.objects.put(
            self.current_key(dataset),
            pointer.to_bytes(),
            if_match=expected_pointer_etag,
        )
        self.manifests[(dataset, version)] = manifest
        self.frames[(dataset, version)] = frame.copy()
        self.write_records.append((dataset, version))
        return DatasetWriteResult(manifest, ValidationReport(dataset))

    def commit_release(
        self,
        completed_session: str,
        dataset_versions: dict[str, str],
        *,
        quality: str,
        warnings: tuple[str, ...],
        expected_etag: str | None,
    ):
        self.release_count += 1
        release = DataRelease(
            version=f"committed-release-{self.release_count}",
            created_at=f"2026-07-19T00:00:0{self.release_count}Z",
            completed_session=completed_session,
            dataset_versions=dict(dataset_versions),
            quality=quality,
            warnings=warnings,
        )
        self.objects.put(
            "releases/current.json", release.to_bytes(), if_match=expected_etag
        )
        return release


def _transaction_plan(
    repository: _TransactionRepository,
    token: str,
    *,
    status: str = "validated_offline_plan",
) -> script.PreparedRepair:
    release, release_etag = repository.current_release()
    pointer_etags = {
        dataset: repository.current_pointer(dataset)[1]
        for dataset in script.REQUIRED_DATASETS
    }
    planned = (
        {
            dataset: f"early-history-{token}-{dataset}"
            for dataset in script.WRITE_DATASETS
        }
        if status == "validated_offline_plan"
        else {}
    )
    frames = (
        {dataset: pd.DataFrame({"plan": [token]}) for dataset in script.WRITE_DATASETS}
        if planned
        else {}
    )
    return script.PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions=planned,
        frames=frames,
        archive_objects={},
        summary={
            "status": status,
            "factor_source_version": f"factor:{token}",
            "source_archive_rows_added": 0,
        },
    )


def _control_bytes(repository: _TransactionRepository) -> dict[str, bytes]:
    keys = ["releases/current.json"] + [
        repository.current_key(dataset) for dataset in script.REQUIRED_DATASETS
    ]
    return {key: repository.objects.get(key).data for key in keys}


class _CountingEodSource:
    def __init__(self) -> None:
        self.claims: list[int] = []
        self.requests: list[str] = []

    def claim(self) -> int:
        usage = 8_850 + len(self.claims)
        self.claims.append(usage)
        return usage

    def request(
        self, case: script.HistoryCase
    ) -> tuple[bytes, str, int, str]:
        self.requests.append(case.provider_symbol)
        payload = []
        for offset, session in enumerate(script._request_sessions(case)):
            close = 100.0 + offset
            payload.append(
                {
                    "date": session,
                    "open": close - 0.1,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "adjusted_close": close,
                    "volume": 1_000 + offset,
                }
            )
        return (
            script._canonical_json(payload),
            "2026-07-19T00:00:00Z",
            200,
            "application/json",
        )


def test_fetch_is_capped_at_four_claimed_attempts_and_never_auto_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _TransactionRepository(tmp_path)
    base = _transaction_plan(repository, "fetch", status="awaiting_eodhd_fetch")
    awaiting = replace(
        base,
        summary={
            **base.summary,
            "missing_symbols": [case.symbol for case in script.EODHD_CASES],
        },
    )
    monkeypatch.setattr(script, "prepare_repair", lambda _repo: awaiting)
    source = _CountingEodSource()

    result = script.fetch_missing(repository, awaiting, source=source)

    assert result["eodhd_http_attempts_this_run"] == 4
    assert source.claims == [8_850, 8_851, 8_852, 8_853]
    assert source.requests == [case.provider_symbol for case in script.EODHD_CASES]
    ledger = script._read_fetch_ledger(tmp_path)
    assert len(ledger["attempts"]) == script.MAX_EODHD_HTTP_ATTEMPTS == 4
    assert [item["status"] for item in ledger["attempts"]] == [
        "cached_valid"
    ] * 4
    assert all(
        script._response_cache_path(tmp_path, case).is_file()
        for case in script.EODHD_CASES
    )

    with pytest.raises(RuntimeError, match="automatic retry is forbidden"):
        script.fetch_missing(repository, awaiting, source=source)
    assert len(source.claims) == len(source.requests) == 4


def _install_apply_guards(
    monkeypatch: pytest.MonkeyPatch,
    plan_provider: Callable[[_TransactionRepository], script.PreparedRepair],
) -> None:
    monkeypatch.setattr(script, "prepare_repair", plan_provider)
    monkeypatch.setattr(
        script, "validate_repository_snapshot", lambda _repo: ValidationReport("snapshot")
    )

    def assert_applied(
        repository: _TransactionRepository,
        committed: DataRelease,
        *,
        expected_out_of_scope_pointer_etags: dict[str, str | None],
    ) -> None:
        current, _ = repository.current_release()
        assert current.to_bytes() == committed.to_bytes()

    monkeypatch.setattr(script, "_assert_applied_release", assert_applied)


@pytest.mark.parametrize(
    "failure_stage", ("after_write:adjustment_factors", "after_release_commit")
)
def test_apply_transaction_rolls_back_all_control_pointers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    repository = _TransactionRepository(tmp_path)
    caller = _transaction_plan(repository, "caller")
    locked = _transaction_plan(repository, "locked")
    _install_apply_guards(monkeypatch, lambda _repo: locked)
    before = _control_bytes(repository)

    def fail(stage: str) -> None:
        if stage == failure_stage:
            raise RuntimeError(f"injected {stage}")

    with pytest.raises(RuntimeError, match="injected"):
        script.apply_repair(repository, caller, inject_failure=fail)
    assert _control_bytes(repository) == before
    journal = next((tmp_path / script.TRANSACTION_DIR).glob("*.json"))
    assert json.loads(journal.read_text())["status"] == "rolled_back"
    assert not (tmp_path / script.RECOVERY_DIR).exists()


def test_apply_replay_is_locked_idempotent_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _TransactionRepository(tmp_path)
    caller = _transaction_plan(repository, "caller")
    locked = _transaction_plan(repository, "locked")
    _install_apply_guards(monkeypatch, lambda _repo: locked)
    result = script.apply_repair(repository, caller)
    assert result["status"] == "applied"
    before = _control_bytes(repository)
    writes = len(repository.write_records)
    noop = _transaction_plan(repository, "noop", status="already_repaired")
    monkeypatch.setattr(script, "prepare_repair", lambda _repo: noop)
    replay = script.apply_repair(repository, noop)
    assert replay["status"] == "already_repaired"
    assert replay["writes_performed"] is False
    assert len(repository.write_records) == writes
    assert _control_bytes(repository) == before


def test_apply_accepts_only_unchanged_preexisting_snapshot_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _TransactionRepository(tmp_path)
    caller = _transaction_plan(repository, "caller")
    locked = _transaction_plan(repository, "locked")
    _install_apply_guards(monkeypatch, lambda _repo: locked)
    existing = ValidationReport(
        "snapshot",
        (
            ValidationIssue(
                "existing_gap",
                "preexisting exact gap",
                row_count=1,
                fingerprints=("pinned",),
            ),
        ),
    )
    monkeypatch.setattr(script, "validate_repository_snapshot", lambda _repo: existing)
    result = script.apply_repair(repository, caller)
    assert result["status"] == "applied"


def test_apply_rolls_back_if_snapshot_error_fingerprint_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _TransactionRepository(tmp_path)
    caller = _transaction_plan(repository, "caller")
    locked = _transaction_plan(repository, "locked")
    _install_apply_guards(monkeypatch, lambda _repo: locked)
    reports = iter(
        (
            ValidationReport(
                "snapshot",
                (ValidationIssue("existing_gap", "preexisting exact gap"),),
            ),
            ValidationReport(
                "snapshot",
                (
                    ValidationIssue("existing_gap", "preexisting exact gap"),
                    ValidationIssue("new_gap", "new error"),
                ),
            ),
        )
    )
    monkeypatch.setattr(
        script, "validate_repository_snapshot", lambda _repo: next(reports)
    )
    before = _control_bytes(repository)
    with pytest.raises(RuntimeError, match="changed repository snapshot errors"):
        script.apply_repair(repository, caller)
    assert _control_bytes(repository) == before


def test_apply_rejects_stale_release_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _TransactionRepository(tmp_path)
    caller = _transaction_plan(repository, "caller")
    current, _ = repository.current_release()
    changed = replace(
        current,
        version="changed-release",
        created_at="2026-07-19T00:01:00Z",
    )
    repository.objects.put("releases/current.json", changed.to_bytes())
    monkeypatch.setattr(
        script,
        "prepare_repair",
        lambda _repo: pytest.fail("stale plan reached locked replan"),
    )
    with pytest.raises(RuntimeError, match="Current release changed"):
        script.apply_repair(repository, caller)
    assert repository.write_records == []
    assert not (tmp_path / script.TRANSACTION_DIR).exists()
