from __future__ import annotations

import copy
import gzip
import importlib.util
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from supertrend_quant.market_store import cross_validation as cv
from supertrend_quant.market_store.manifest import CurrentPointer, DataRelease
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "unified_quant/scripts/repair_us_arnc_hwm_ticker_change.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_arnc_hwm_ticker_change", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)

VALIDATOR_PATH = (
    PROJECT_ROOT / "unified_quant/scripts/validate_us_lifecycle_cross_sources.py"
)
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_us_lifecycle_cross_sources_arnc_tests", VALIDATOR_PATH
)
assert VALIDATOR_SPEC is not None and VALIDATOR_SPEC.loader is not None
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
sys.modules[VALIDATOR_SPEC.name] = validator
VALIDATOR_SPEC.loader.exec_module(validator)

POLICY_SOURCE = PROJECT_ROOT / "unified_quant/configs/us_cross_validation.yaml"
REAL_REPOSITORY = LocalDatasetRepository(PROJECT_ROOT / "data/cache")


@pytest.fixture(autouse=True)
def _small_post_symc_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit repositories tiny while exercising the same strict gates."""

    empty_coverage = {
        "coverage_gate_version": 1,
        "selection_rule": "us_terminal_v1",
        "candidate_set_sha256": (
            "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
        ),
        "resolution_set_sha256": (
            "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
        ),
        "candidate_count": 0,
        "resolution_count": 0,
        "applied_count": 0,
        "exception_count": 0,
        "open_count": 0,
    }
    monkeypatch.setattr(script, "EXPECTED_ACTION_ROWS_BEFORE", 1)
    monkeypatch.setattr(script, "EXPECTED_ACTION_ROWS_AFTER", 2)
    monkeypatch.setattr(script, "EXPECTED_LIFECYCLE_COVERAGE", empty_coverage)
    monkeypatch.setattr(script, "_candidate_values", lambda *_: ())


def _source_fields(source_hash: str = "a" * 64) -> dict[str, object]:
    return {
        "source": "fixture",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": source_hash,
    }


def _empty(dataset: str) -> pd.DataFrame:
    return pd.DataFrame(columns=dataset_spec(dataset).required_columns)


def _official_archive_fixture(
    target_root: Path,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    release, _ = REAL_REPOSITORY.current_release()
    if release is None:
        pytest.skip("Current local evidence release is unavailable.")
    archive = REAL_REPOSITORY.read_frame(
        "source_archive", release.dataset_versions["source_archive"]
    )
    rows = []
    copied: dict[str, Path] = {}
    for source_url, source_hash in (
        (script.audit.SEC_SOURCE_URL, script.audit.SEC_SOURCE_HASH),
        (script.audit.SP_SOURCE_URL, script.audit.SP_SOURCE_HASH),
    ):
        matches = archive.loc[
            archive["source_url"].astype(str).eq(source_url)
            & archive["source_hash"].astype(str).eq(source_hash)
        ]
        if len(matches) != 1:
            pytest.skip("Pinned ARNC official archive rows are unavailable.")
        row = matches.iloc[0].to_dict()
        relative = Path(str(row["object_path"]))
        source = REAL_REPOSITORY.root / relative
        if not source.is_file():
            pytest.skip("Pinned ARNC official archive payload is unavailable.")
        destination = target_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied[source_hash] = destination
        rows.append(row)
    return pd.DataFrame(rows), copied


def _base_frames(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    archive, copied = _official_archive_fixture(root)
    parent = script.audit.PARENT_SECURITY_ID
    child = script.audit.CHILD_SECURITY_ID
    source = _source_fields()
    master = pd.DataFrame(
        [
            {
                "security_id": parent,
                "primary_symbol": "HWM",
                "name": "Howmet Aerospace",
                "exchange": "NYSE",
                "asset_type": "Common Stock",
                "currency": "USD",
                "country": "USA",
                "active_from": "2016-11-01",
                "active_to": "",
                **source,
            },
            {
                "security_id": child,
                "primary_symbol": "ARNC",
                "name": "Arconic Corporation",
                "exchange": "NYSE",
                "asset_type": "Common Stock",
                "currency": "USD",
                "country": "USA",
                "active_from": "2020-04-01",
                "active_to": "",
                **source,
            },
        ]
    )
    history = pd.DataFrame(
        [
            {
                "security_id": parent,
                "symbol": "ARNC",
                "exchange": "NYSE",
                "effective_from": "2016-11-01",
                "effective_to": "2020-03-31",
                **source,
            },
            {
                "security_id": parent,
                "symbol": "HWM",
                "exchange": "NYSE",
                "effective_from": "2020-04-01",
                "effective_to": "",
                **source,
            },
            {
                "security_id": child,
                "symbol": "ARNC",
                "exchange": "NYSE",
                "effective_from": "2020-04-01",
                "effective_to": "",
                **source,
            },
        ]
    )
    price_rows = []
    factor_rows = []
    for security_id, session, close in (
        (parent, "2020-03-31", 16.0),
        (parent, "2020-04-01", 12.0),
        (parent, "2026-07-15", 180.0),
        (child, "2020-04-01", 7.0),
    ):
        price_rows.append(
            {
                "security_id": security_id,
                "session": session,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1000.0,
                "currency": "USD",
                **source,
            }
        )
        factor_rows.append(
            {
                "security_id": security_id,
                "session": session,
                "split_factor": 1.0,
                "total_return_factor": 1.0,
                "source_version": "fixture-lineage",
                "calculated_at": "2026-07-18T00:00:00Z",
                **source,
            }
        )
    actions = pd.DataFrame(
        [
            {
                "event_id": script.audit.SPINOFF_EVENT_ID,
                "security_id": parent,
                "action_type": "spinoff",
                "effective_date": "2020-04-01",
                "ex_date": "2020-04-01",
                "announcement_date": "",
                "record_date": "2020-03-19",
                "payment_date": "2020-04-01",
                "cash_amount": None,
                "ratio": 0.25,
                "currency": "USD",
                "new_security_id": child,
                "new_symbol": "ARNC",
                "official": True,
                "source_url": script.audit.SEC_SOURCE_URL,
                "source_kind": "official_filing",
                "source": "official_identity_repair",
                "retrieved_at": "2026-07-18T00:00:00Z",
                "source_hash": script.audit.SEC_SOURCE_HASH,
            }
        ]
    )
    anchors = pd.DataFrame(
        [
            {
                "index_id": "sp500",
                "anchor_date": "2019-12-31",
                "security_id": parent,
                "official": True,
                "source_url": script.audit.SP_SOURCE_URL,
                "source_kind": "official_crosscheck",
                **source,
            }
        ]
    )
    return (
        {
            "corporate_actions": actions,
            "adjustment_factors": pd.DataFrame(factor_rows),
            "security_master": master,
            "symbol_history": history,
            "daily_price_raw": pd.DataFrame(price_rows),
            "lifecycle_resolutions": _empty("lifecycle_resolutions"),
            "source_archive": archive,
            "index_constituent_anchors": anchors,
            "index_membership_events": _empty("index_membership_events"),
        },
        copied,
    )


def _repository_fixture(
    tmp_path: Path,
) -> tuple[LocalDatasetRepository, Path, dict[str, Path]]:
    root = tmp_path / "cache"
    root.mkdir(parents=True)
    frames, copied = _base_frames(root)
    repository = LocalDatasetRepository(root)
    versions: dict[str, str] = {}
    for dataset in script.REQUIRED_DATASETS:
        metadata: dict[str, object] = {"fixture": True}
        if dataset == "lifecycle_resolutions":
            metadata.update(
                {
                    "operation": script.POST_SYMC_OPERATION,
                    **dict(script.EXPECTED_LIFECYCLE_COVERAGE),
                }
            )
        result = repository.write_frame(
            dataset,
            frames[dataset],
            completed_session="2026-07-15",
            incomplete_action_policy="block",
            metadata=metadata,
            version=f"fixture-{dataset}",
        )
        assert not result.conflict
        versions[dataset] = result.manifest.version
    repository.commit_release(
        "2026-07-15",
        versions,
        quality="valid",
        expected_etag=None,
    )
    policy = tmp_path / "us_cross_validation.yaml"
    shutil.copyfile(POLICY_SOURCE, policy)
    return repository, policy, copied


def _reviewed_event() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    policy = yaml.safe_load(POLICY_SOURCE.read_text())
    reviewed = cv.reviewed_nonterminal_extractions(policy["events"])
    extraction = reviewed[script.audit.TICKER_CHANGE_EVENT_ID]
    event = {
        **extraction,
        "status": "passed",
        "validation_kind": cv.NONTERMINAL_EVENT_VALIDATION,
        "candidate_id": "",
        "lifecycle_report_extraction_approved": False,
        "reviewed_extraction_match": True,
        "reviewed_extraction_sha256": (
            cv.reviewed_nonterminal_extraction_sha256(extraction)
        ),
        "evidence_sha256": script.audit.SP_SOURCE_HASH,
    }
    return event, reviewed


def _reviewed_target() -> dict[str, object]:
    return {
        "target_id": script.audit.OLD_SYMBOL_TARGET_ID,
        "security_id": script.audit.PARENT_SECURITY_ID,
        "symbol": "ARNC",
        "provider_symbol": "ARNC",
        "identity_active_from": "2016-11-01",
        "identity_active_to": "2020-03-31",
        "terminal_event_id": script.audit.TICKER_CHANGE_EVENT_ID,
        "successor_security_id": script.audit.PARENT_SECURITY_ID,
    }


def test_plan_adds_one_delta_and_preserves_every_other_version(tmp_path: Path) -> None:
    repository, policy, _ = _repository_fixture(tmp_path)
    prepared = script.prepare_repair(repository, policy_path=policy)
    assert prepared.summary["status"] == "validated_offline_plan"
    assert prepared.summary["corporate_action_rows_added"] == 1
    assert len(prepared.action_delta) == 1
    assert prepared.action_delta.iloc[0]["event_id"] == script.audit.TICKER_CHANGE_EVENT_ID
    assert prepared.summary["adjustment_factor_economic_rows_changed"] == 0
    assert prepared.summary["lifecycle_resolution_change"] == "none"
    assert prepared.summary["network_accessed"] is False
    assert prepared.summary["eodhd_calls"] == 0
    assert prepared.summary["r2_accessed"] is False


def test_plan_never_materializes_full_price_or_factor_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, policy, _ = _repository_fixture(tmp_path)
    original = repository.read_frame

    def guarded(dataset: str, version: str | None = None) -> pd.DataFrame:
        if dataset in {"daily_price_raw", "adjustment_factors"}:
            raise AssertionError(f"full heavy-table read attempted: {dataset}")
        return original(dataset, version)

    monkeypatch.setattr(repository, "read_frame", guarded)
    prepared = script.prepare_repair(repository, policy_path=policy)
    assert prepared.summary["full_daily_price_raw_materialized"] is False
    assert prepared.summary["adjustment_factors_materialized"] is False
    assert prepared.summary["plan_materialization"] == (
        "two_sid_duckdb_price_scope_plus_terminal_session_summary"
    )


def test_apply_uses_inherited_delta_and_is_idempotent(tmp_path: Path) -> None:
    repository, policy, _ = _repository_fixture(tmp_path)
    prepared = script.prepare_repair(repository, policy_path=policy)
    old_release, _ = repository.current_release()
    assert old_release is not None
    old_action_version = old_release.dataset_versions["corporate_actions"]
    old_action_manifest = (
        repository.root
        / repository.version_prefix("corporate_actions", old_action_version)
        / "manifest.json"
    ).read_bytes()
    old_factor_pointer = repository.objects.get(
        repository.current_key("adjustment_factors")
    ).data

    result = script.apply_repair(repository, prepared)
    assert result["status"] == "applied"
    release, _ = repository.current_release()
    assert release is not None
    assert release.dataset_versions["adjustment_factors"] == old_release.dataset_versions[
        "adjustment_factors"
    ]
    assert repository.objects.get(repository.current_key("adjustment_factors")).data == old_factor_pointer
    assert (
        repository.root
        / repository.version_prefix("corporate_actions", old_action_version)
        / "manifest.json"
    ).read_bytes() == old_action_manifest
    manifest = repository.current_manifest("corporate_actions")
    assert manifest is not None
    assert manifest.parent_version == old_action_version
    assert manifest.metadata["inherits_parent"] is True
    actions = repository.read_frame(
        "corporate_actions", release.dataset_versions["corporate_actions"]
    )
    assert len(actions) == 2
    assert script._action_state(
        actions,
        script.expected_action(
            "2026-07-18T01:43:12.149458Z"
        ),
    ) == "exact"
    resolutions = repository.read_frame(
        "lifecycle_resolutions", release.dataset_versions["lifecycle_resolutions"]
    )
    assert resolutions.empty

    replay = script.prepare_repair(repository, policy_path=policy)
    assert replay.summary["status"] == "already_repaired"
    second = script.apply_repair(repository, replay)
    assert second["writes_performed"] is False
    assert repository.current_release()[0].version == release.version


def test_apply_does_not_reenter_full_plan_after_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, policy, _ = _repository_fixture(tmp_path)
    prepared = script.prepare_repair(repository, policy_path=policy)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("post-commit prepare_repair re-entry")

    monkeypatch.setattr(script, "prepare_repair", forbidden)
    result = script.apply_repair(repository, prepared)
    assert result["status"] == "applied"


@pytest.mark.parametrize("foreign_kind", ["release", "protected_pointer"])
def test_foreign_publication_causes_zero_pointer_restore_and_recovery_marker(
    tmp_path: Path,
    foreign_kind: str,
) -> None:
    repository, policy, _ = _repository_fixture(tmp_path)
    prepared = script.prepare_repair(repository, policy_path=policy)
    observed_at_failure: dict[str, bytes] = {}

    def inject(stage: str) -> None:
        if stage != "after_action_write":
            return
        if foreign_kind == "release":
            current = repository.objects.get("releases/current.json")
            foreign = DataRelease(
                version="foreign-release",
                created_at="2026-07-19T00:02:00Z",
                completed_session=prepared.release.completed_session,
                dataset_versions=dict(prepared.release.dataset_versions),
                warnings=("foreign",),
            )
            repository.objects.put(
                "releases/current.json", foreign.to_bytes(), if_match=current.etag
            )
        else:
            dataset = "adjustment_factors"
            key = repository.current_key(dataset)
            current = repository.objects.get(key)
            old = CurrentPointer.from_bytes(current.data)
            foreign = CurrentPointer(
                dataset=dataset,
                version=old.version,
                manifest_path=old.manifest_path,
                manifest_sha256="f" * 64,
                updated_at="2026-07-19T00:02:00Z",
            )
            repository.objects.put(key, foreign.to_bytes(), if_match=current.etag)
        observed_at_failure["release"] = repository.objects.get(
            "releases/current.json"
        ).data
        for dataset in script.REQUIRED_DATASETS:
            observed_at_failure[dataset] = repository.objects.get(
                repository.current_key(dataset)
            ).data
        raise RuntimeError("injected foreign ARNC publication")

    with pytest.raises(RuntimeError, match="rollback was incomplete"):
        script.apply_repair(repository, prepared, inject_failure=inject)
    assert repository.objects.get("releases/current.json").data == observed_at_failure[
        "release"
    ]
    for dataset in script.REQUIRED_DATASETS:
        assert (
            repository.objects.get(repository.current_key(dataset)).data
            == observed_at_failure[dataset]
        )
    recovery = tuple((repository.root / script.RECOVERY_DIR).glob("*.json"))
    assert len(recovery) == 1


@pytest.mark.parametrize("stage", ["after_action_write", "after_release_commit"])
def test_failure_rolls_back_release_and_action_pointer(
    tmp_path: Path, stage: str
) -> None:
    repository, policy, _ = _repository_fixture(tmp_path)
    prepared = script.prepare_repair(repository, policy_path=policy)
    old_release = repository.objects.get("releases/current.json").data
    old_pointer = repository.objects.get(repository.current_key("corporate_actions")).data

    def fail(value: str) -> None:
        if value == stage:
            raise RuntimeError("injected ARNC failure")

    with pytest.raises(RuntimeError, match="injected ARNC failure"):
        script.apply_repair(repository, prepared, inject_failure=fail)
    assert repository.objects.get("releases/current.json").data == old_release
    assert repository.objects.get(repository.current_key("corporate_actions")).data == old_pointer
    assert script.prepare_repair(repository, policy_path=policy).summary[
        "status"
    ] == "validated_offline_plan"


def test_release_cas_blocks_stale_plan_before_dataset_write(tmp_path: Path) -> None:
    repository, policy, _ = _repository_fixture(tmp_path)
    prepared = script.prepare_repair(repository, policy_path=policy)
    old_pointer = repository.objects.get(repository.current_key("corporate_actions")).data
    release, etag = repository.current_release()
    assert release is not None
    repository.commit_release(
        release.completed_session,
        dict(release.dataset_versions),
        quality=release.quality,
        warnings=release.warnings,
        expected_etag=etag,
    )
    with pytest.raises(RuntimeError, match="release changed"):
        script.apply_repair(repository, prepared)
    assert repository.objects.get(repository.current_key("corporate_actions")).data == old_pointer


def test_already_repaired_noop_rejects_stale_release_plan(tmp_path: Path) -> None:
    repository, policy, _ = _repository_fixture(tmp_path)
    first = script.prepare_repair(repository, policy_path=policy)
    script.apply_repair(repository, first)
    replay = script.prepare_repair(repository, policy_path=policy)
    assert replay.summary["status"] == "already_repaired"
    action_pointer = repository.objects.get(
        repository.current_key("corporate_actions")
    ).data
    release, etag = repository.current_release()
    assert release is not None
    changed = repository.commit_release(
        release.completed_session,
        dict(release.dataset_versions),
        quality=release.quality,
        warnings=release.warnings,
        expected_etag=etag,
    )

    with pytest.raises(RuntimeError, match="release changed"):
        script.apply_repair(repository, replay)
    assert repository.current_release()[0].version == changed.version
    assert (
        repository.objects.get(repository.current_key("corporate_actions")).data
        == action_pointer
    )


def test_factor_parent_parquet_tamper_after_release_commit_fails_and_rolls_back(
    tmp_path: Path,
) -> None:
    repository, policy, _ = _repository_fixture(tmp_path)
    prepared = script.prepare_repair(repository, policy_path=policy)
    old_release = repository.objects.get("releases/current.json").data
    old_action_pointer = repository.objects.get(
        repository.current_key("corporate_actions")
    ).data
    factor_version = prepared.release.dataset_versions["adjustment_factors"]
    factor_manifest = repository.manifest_chain(
        "adjustment_factors", factor_version
    )[0]
    factor_path = (
        repository.root
        / repository.version_prefix(
            "adjustment_factors", factor_manifest.version
        )
        / factor_manifest.files[0].path
    )
    original_factor_bytes = factor_path.read_bytes()

    def tamper(stage: str) -> None:
        if stage == "after_release_commit":
            factor_path.write_bytes(original_factor_bytes + b"tamper")

    try:
        with pytest.raises(ValueError, match="Size mismatch|SHA-256 mismatch"):
            script.apply_repair(repository, prepared, inject_failure=tamper)
        assert repository.objects.get("releases/current.json").data == old_release
        assert (
            repository.objects.get(
                repository.current_key("corporate_actions")
            ).data
            == old_action_pointer
        )
    finally:
        factor_path.write_bytes(original_factor_bytes)
    assert script.prepare_repair(repository, policy_path=policy).summary[
        "status"
    ] == "validated_offline_plan"


def test_official_archive_tamper_fails_closed(tmp_path: Path) -> None:
    repository, policy, copied = _repository_fixture(tmp_path)
    copied[script.audit.SP_SOURCE_HASH].write_bytes(
        gzip.compress(b"tampered S&P payload", mtime=0)
    )
    with pytest.raises(RuntimeError, match="payload hash changed"):
        script.prepare_repair(repository, policy_path=policy)


def test_policy_tamper_fails_closed(tmp_path: Path) -> None:
    repository, policy, _ = _repository_fixture(tmp_path)
    value = yaml.safe_load(policy.read_text())
    for item in value["events"]["reviewed_nonterminal_extractions"]:
        if item["event_id"] == script.audit.TICKER_CHANGE_EVENT_ID:
            item["new_symbol"] = "ARNC"
    policy.write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(RuntimeError, match="policy row is missing or differs"):
        script.prepare_repair(repository, policy_path=policy)


def test_code_pinned_nonterminal_binding_is_exact_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event, reviewed = _reviewed_event()
    target = _reviewed_target()
    binding = cv.reviewed_nonterminal_same_sid_no_data_binding(
        target, event, reviewed
    )
    assert binding is not None
    assert binding["source_target_id"] == script.audit.OLD_SYMBOL_TARGET_ID
    assert binding["successor_target_id"] == script.audit.NEW_PARENT_TARGET_ID
    assert binding["same_security_id_continuation"] is True
    assert binding["terminal_resolution_required"] is False
    assert binding["terminal_resolution_forbidden"] is True

    for mapping, key, value in (
        (target, "target_id", "f" * 64),
        (target, "identity_active_to", "2020-03-30"),
        (target, "successor_security_id", script.audit.CHILD_SECURITY_ID),
        (event, "validation_kind", cv.TERMINAL_EVENT_VALIDATION),
        (event, "new_security_id", script.audit.CHILD_SECURITY_ID),
        (event, "evidence_sha256", "f" * 64),
    ):
        changed_target = dict(target)
        changed_event = dict(event)
        if mapping is target:
            changed_target[key] = value
        else:
            changed_event[key] = value
        assert (
            cv.reviewed_nonterminal_same_sid_no_data_binding(
                changed_target, changed_event, reviewed
            )
            is None
        )

    tampered = copy.deepcopy(
        cv.TRUSTED_REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_SPECS
    )
    tampered[script.audit.TICKER_CHANGE_EVENT_ID]["old_symbol"] = "AA"
    monkeypatch.setattr(
        cv,
        "TRUSTED_REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_SPECS",
        tampered,
    )
    with pytest.raises(RuntimeError, match="not code-pinned"):
        cv.reviewed_nonterminal_same_sid_no_data_binding(target, event, reviewed)


def test_price_target_projection_links_old_arnc_to_passed_hwm_without_resolution() -> None:
    parent = script.audit.PARENT_SECURITY_ID
    master = pd.DataFrame(
        [{"security_id": parent, "primary_symbol": "HWM", "active_from": "2016-11-01", "active_to": ""}]
    )
    history = pd.DataFrame(
        [
            {"security_id": parent, "symbol": "ARNC", "effective_from": "2016-11-01", "effective_to": "2020-03-31"},
            {"security_id": parent, "symbol": "HWM", "effective_from": "2020-04-01", "effective_to": ""},
        ]
    )
    action = pd.DataFrame([script.expected_action("2026-07-18T01:43:12.149458Z")])
    resolutions = _empty("lifecycle_resolutions")
    prices = pd.DataFrame(
        [
            {"security_id": parent, "session": "2020-03-31", "source": "eodhd"},
            {"security_id": parent, "session": "2020-04-01", "source": "eodhd"},
        ]
    )
    targets = validator.build_price_targets(master, history, action, resolutions, prices)
    by_id = {item.target_id: item for item in targets}
    old = by_id[script.audit.OLD_SYMBOL_TARGET_ID]
    successor = by_id[script.audit.NEW_PARENT_TARGET_ID]
    assert old.terminal_event_id == script.audit.TICKER_CHANGE_EVENT_ID
    assert old.successor_security_id == parent
    assert successor.security_id == parent
    assert successor.provider_symbol == "HWM"

    event, reviewed = _reviewed_event()
    binding = cv.reviewed_nonterminal_same_sid_no_data_binding(
        {
            "target_id": old.target_id,
            "security_id": old.security_id,
            "provider_symbol": old.provider_symbol,
            "active_from": old.active_from,
            "active_to": old.active_to,
            "terminal_event_id": old.terminal_event_id,
            "successor_security_id": old.successor_security_id,
        },
        event,
        reviewed,
    )
    assert binding is not None
    successor_check = cv.successor_price_check_binding(
        [
            {
                "target_id": old.target_id,
                "security_id": old.security_id,
                "provider_symbol": old.provider_symbol,
                "status": "explicit_exception",
            },
            {
                "target_id": successor.target_id,
                "security_id": successor.security_id,
                "provider_symbol": successor.provider_symbol,
                "identity_active_from": successor.active_from,
                "status": "passed",
            },
        ],
        event,
        source_target_id=old.target_id,
        expected_successor_security_id=parent,
        reviewed_successor_chains={},
        event_checks=[event],
    )
    assert successor_check["passed"] is True
    assert successor_check["target_id"] == script.audit.NEW_PARENT_TARGET_ID
