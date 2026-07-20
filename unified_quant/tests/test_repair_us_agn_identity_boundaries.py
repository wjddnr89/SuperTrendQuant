from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.repository import LocalDatasetRepository


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_agn_identity_boundaries.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_agn_identity_boundaries", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


COMPLETED_SESSION = "2026-07-15"
RETRIEVED_AT = "2026-07-18T20:58:16.840234Z"
FIXTURE_HASH = "f" * 64


def _master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "security_id": script.LEGACY_AGN_ID,
                "primary_symbol": "AGN",
                "name": "Allergan Inc",
                "exchange": "NYSE",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "active_from": "2014-12-17",
                "active_to": script.LEGACY_OLD_ACTIVE_TO,
                "source": "reviewed_early_terminal_history_supplement",
                "retrieved_at": RETRIEVED_AT,
                "source_hash": FIXTURE_HASH,
                "source_url": "archive://old-legacy-review",
                "provider_symbol": "AGN_old.US",
                "action_provider_symbol": "AGN_old.US",
                "isin": "",
            },
            {
                "security_id": script.ACTAVIS_AGN_ID,
                "primary_symbol": "AGN",
                "name": "Allergan plc (formerly Actavis plc)",
                "exchange": "NYSE",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "active_from": "2015-01-02",
                "active_to": script.LATER_OLD_ACTIVE_TO,
                "source": "official_identity_repair",
                "retrieved_at": RETRIEVED_AT,
                "source_hash": script.LATER_EVIDENCE.source_hash,
                "source_url": script.LATER_EVIDENCE.source_url,
                "provider_symbol": "AGN.US",
                "action_provider_symbol": "AGN.US",
                "isin": "",
            },
        ]
    )


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "security_id": script.LEGACY_AGN_ID,
                "symbol": "AGN",
                "exchange": "NYSE",
                "effective_from": "2014-12-17",
                "effective_to": script.LEGACY_OLD_ACTIVE_TO,
                "source": "reviewed_early_terminal_history_supplement",
                "retrieved_at": RETRIEVED_AT,
                "source_hash": FIXTURE_HASH,
                "source_url": "archive://old-legacy-review",
            },
            {
                "security_id": script.ACTAVIS_AGN_ID,
                "symbol": "ACT",
                "exchange": "NYSE",
                "effective_from": script.ACT_EFFECTIVE_FROM,
                "effective_to": script.ACT_ACTIVE_TO,
                "source": "official_identity_repair",
                "retrieved_at": RETRIEVED_AT,
                "source_hash": script.ACT_TICKER_EVIDENCE.source_hash,
                "source_url": script.ACT_TICKER_EVIDENCE.source_url,
            },
            {
                "security_id": script.ACTAVIS_AGN_ID,
                "symbol": "AGN",
                "exchange": "NYSE",
                "effective_from": script.AGN_EFFECTIVE_FROM,
                "effective_to": script.LATER_OLD_ACTIVE_TO,
                "source": "official_identity_repair",
                "retrieved_at": RETRIEVED_AT,
                "source_hash": script.LATER_EVIDENCE.source_hash,
                "source_url": script.LATER_EVIDENCE.source_url,
            },
        ]
    )


def _evidence() -> dict[str, dict[str, str]]:
    return {
        script.LEGACY_EVIDENCE.name: {"retrieved_at": RETRIEVED_AT},
        script.ACT_TICKER_EVIDENCE.name: {"retrieved_at": RETRIEVED_AT},
        script.LATER_EVIDENCE.name: {"retrieved_at": RETRIEVED_AT},
    }


def _prepared_repository(
    root: Path,
) -> tuple[LocalDatasetRepository, script.PreparedRepair]:
    repository = LocalDatasetRepository(root)
    master_result = repository.write_frame(
        "security_master",
        _master(),
        completed_session=COMPLETED_SESSION,
        version="base-security-master",
    )
    history_result = repository.write_frame(
        "symbol_history",
        _history(),
        completed_session=COMPLETED_SESSION,
        version="base-symbol-history",
    )
    release = repository.commit_release(
        COMPLETED_SESSION,
        {
            "security_master": master_result.manifest.version,
            "symbol_history": history_result.manifest.version,
        },
        quality=DataQuality.DEGRADED,
        warnings=("fixture warning",),
    )
    current, release_etag = repository.current_release()
    assert current is not None and current.version == release.version
    pointer_etags = {
        dataset: repository.current_pointer(dataset)[1]
        for dataset in script.WRITE_DATASETS
    }
    logical, deltas, changed = script._build_frames(
        _master(), _history(), _evidence()
    )
    assert changed is True
    prepared = script.PreparedRepair(
        release=current,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        logical_frames=logical,
        deltas=deltas,
        summary={
            "status": "validated_offline_plan",
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        },
    )
    return repository, prepared


def _install_prepare_stub(
    monkeypatch: pytest.MonkeyPatch,
    prepared: script.PreparedRepair,
) -> None:
    def replay(repository: LocalDatasetRepository, *, yahoo_cache: Path):
        release, release_etag = repository.current_release()
        assert release is not None
        if release.version == prepared.release.version:
            return prepared
        pointer_etags = {
            dataset: repository.current_pointer(dataset)[1]
            for dataset in script.WRITE_DATASETS
        }
        return dataclasses.replace(
            prepared,
            release=release,
            release_etag=release_etag,
            pointer_etags=pointer_etags,
            deltas={
                dataset: prepared.deltas[dataset].iloc[:0].copy()
                for dataset in script.WRITE_DATASETS
            },
            summary={**prepared.summary, "status": "already_repaired"},
        )

    monkeypatch.setattr(script, "prepare_repair", replay)


def test_build_frames_changes_only_two_boundaries_and_legacy_provenance():
    master = _master()
    history = _history()

    logical, deltas, changed = script._build_frames(master, history, _evidence())

    assert changed is True
    assert len(deltas["security_master"]) == 2
    assert len(deltas["symbol_history"]) == 2
    legacy_master = logical["security_master"].loc[
        logical["security_master"]["security_id"].eq(script.LEGACY_AGN_ID)
    ].iloc[0]
    later_master = logical["security_master"].loc[
        logical["security_master"]["security_id"].eq(script.ACTAVIS_AGN_ID)
    ].iloc[0]
    assert legacy_master["active_to"] == script.LEGACY_ACTIVE_TO
    assert legacy_master["source_hash"] == script.LEGACY_EVIDENCE.source_hash
    assert later_master["active_to"] == script.LATER_ACTIVE_TO
    act_before = history.loc[history["symbol"].eq("ACT")].reset_index(drop=True)
    act_after = logical["symbol_history"].loc[
        logical["symbol_history"]["symbol"].eq("ACT")
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(act_before, act_after)


def test_build_frames_is_idempotent_and_rejects_partial_state():
    logical, _deltas, _changed = script._build_frames(
        _master(), _history(), _evidence()
    )
    replay, deltas, changed = script._build_frames(
        logical["security_master"], logical["symbol_history"], _evidence()
    )
    assert changed is False
    assert all(frame.empty for frame in deltas.values())
    pd.testing.assert_frame_equal(replay["security_master"], logical["security_master"])

    partial = logical["security_master"].copy()
    partial.loc[
        partial["security_id"].eq(script.ACTAVIS_AGN_ID), "active_to"
    ] = script.LATER_OLD_ACTIVE_TO
    with pytest.raises(RuntimeError, match="partial or unexpected"):
        script._build_frames(partial, logical["symbol_history"], _evidence())


def test_act_source_is_explicitly_left_blocked():
    assert "ACT" in script.BLOCKED_TARGETS
    assert "not an ACT-to-AGN ticker notice" in script.BLOCKED_TARGETS["ACT"]
    assert script.ACT_TICKER_EVIDENCE.source_hash != script.LEGACY_EVIDENCE.source_hash


def test_apply_writes_only_inherited_identity_deltas_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repository, prepared = _prepared_repository(tmp_path)
    old_release = prepared.release
    _install_prepare_stub(monkeypatch, prepared)

    result = script.apply_repair(repository, prepared, yahoo_cache=tmp_path / "yahoo")

    current, _ = repository.current_release()
    assert current is not None
    assert result["status"] == "applied"
    assert result["writes_performed"] is True
    assert current.version != old_release.version
    assert current.quality == old_release.quality
    assert current.warnings == old_release.warnings
    for dataset in script.WRITE_DATASETS:
        manifest = repository.current_manifest(dataset)
        assert manifest is not None
        assert manifest.parent_version == old_release.dataset_versions[dataset]
        assert manifest.metadata["inherits_parent"] is True
    master = repository.read_frame(
        "security_master", current.dataset_versions["security_master"]
    )
    history = repository.read_frame(
        "symbol_history", current.dataset_versions["symbol_history"]
    )
    _old, repaired = script._repaired_state(script._identity_rows(master, history))
    assert repaired is True

    already = dataclasses.replace(
        prepared, summary={**prepared.summary, "status": "already_repaired"}
    )
    before = repository.objects.get("releases/current.json").data
    replay = script.apply_repair(repository, already, yahoo_cache=tmp_path / "yahoo")
    assert replay["writes_performed"] is False
    assert repository.objects.get("releases/current.json").data == before


@pytest.mark.parametrize(
    "failure_stage",
    ["after_write:security_master", "after_write:symbol_history", "after_release_commit"],
)
def test_apply_rolls_back_both_pointers_and_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
):
    repository, prepared = _prepared_repository(tmp_path)
    _install_prepare_stub(monkeypatch, prepared)
    old_release = repository.objects.get("releases/current.json").data
    old_pointers = {
        dataset: repository.objects.get(repository.current_key(dataset)).data
        for dataset in script.WRITE_DATASETS
    }

    def inject(stage: str) -> None:
        if stage == failure_stage:
            raise RuntimeError(f"injected:{stage}")

    with pytest.raises(RuntimeError, match=f"injected:{failure_stage}"):
        script.apply_repair(
            repository,
            prepared,
            yahoo_cache=tmp_path / "yahoo",
            inject_failure=inject,
        )

    assert repository.objects.get("releases/current.json").data == old_release
    for dataset in script.WRITE_DATASETS:
        assert (
            repository.objects.get(repository.current_key(dataset)).data
            == old_pointers[dataset]
        )
    journals = tuple((tmp_path / script.TRANSACTION_DIR).glob("*.json"))
    assert len(journals) == 1
    assert json.loads(journals[0].read_text())["status"] == "rolled_back"
    recovery = tmp_path / script.RECOVERY_DIR
    assert not recovery.exists() or not tuple(recovery.glob("*.json"))


def test_apply_fails_closed_on_stale_release_etag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repository, prepared = _prepared_repository(tmp_path)
    _install_prepare_stub(monkeypatch, prepared)
    stale = dataclasses.replace(prepared, release_etag="stale")
    old_release = repository.objects.get("releases/current.json").data

    with pytest.raises(RuntimeError, match="Current release changed"):
        script.apply_repair(repository, stale, yahoo_cache=tmp_path / "yahoo")

    assert repository.objects.get("releases/current.json").data == old_release


def test_cli_defaults_to_read_only_plan():
    args = script._parse_args([])
    assert args.apply is False
    assert args.cache_root == script.DEFAULT_CACHE_ROOT
