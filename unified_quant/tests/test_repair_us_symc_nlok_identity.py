from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml


PROJECT_ROOT = Path(__file__).parents[2]
SCRIPT_PATH = PROJECT_ROOT / "unified_quant/scripts/repair_us_symc_nlok_identity.py"
DRAFT_PATH = (
    PROJECT_ROOT
    / "unified_quant/configs/drafts/us_symc_nlok_identity_integration.yaml"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_symc_nlok_identity_for_test", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot import {SCRIPT_PATH}")
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


@pytest.fixture(scope="module")
def actual_state():
    repository = script.LocalDatasetRepository(PROJECT_ROOT / "data/cache")
    release, release_etag = repository.current_release()
    assert release is not None
    before_bytes = release.to_bytes()
    affected = {script.OLD_SECURITY_ID, script.CANONICAL_SECURITY_ID}
    frames: dict[str, pd.DataFrame] = {}
    for dataset in (
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "corporate_actions",
        "adjustment_factors",
        "lifecycle_resolutions",
    ):
        frames[dataset] = script.identity_tails._read_security_subset(
            repository,
            dataset,
            release.dataset_versions[dataset],
            affected,
        )
    for dataset in ("index_constituent_anchors", "index_membership_events"):
        frame = repository.read_frame(dataset, release.dataset_versions[dataset])
        frames[dataset] = frame.loc[
            frame["security_id"].astype(str).isin(affected)
        ].copy()
    frames["source_archive"] = repository.read_frame(
        "source_archive", release.dataset_versions["source_archive"]
    )
    diagnostic = script.audit._symc_full_identity_diagnostic(
        frames["daily_price_raw"], frames["adjustment_factors"]
    )
    candidate, summary = script.prepare_candidate_frames(
        frames,
        completed_session=script.CANONICAL_SYMBOL_TO,
        require_candidate_pins=True,
        parent_release_kind="identity_price_tails_descendant",
    )
    proof = {
        "audit_sha256": script.sha256_bytes(
            script.audit._canonical_json_bytes(diagnostic)
        ),
        "signal_diagnostic": {
            "pre_transition_triple_supertrend_diff": diagnostic[
                "pre_transition_triple_supertrend_diff"
            ],
        },
    }
    after, after_etag = repository.current_release()
    assert after is not None
    assert after.to_bytes() == before_bytes
    assert after_etag == release_etag
    return SimpleNamespace(
        repository=repository,
        release=release,
        release_etag=release_etag,
        candidate=candidate,
        summary=summary,
        proof=proof,
    )


@pytest.fixture(scope="module")
def descendant_state(actual_state):
    repository = script.LocalDatasetRepository(PROJECT_ROOT / "data/cache")
    before, before_etag = repository.current_release()
    assert before is not None
    parent_context = script.identity_tails._read_candidate_context(
        repository, before
    )
    parent_frames = {
        **parent_context,
        "lifecycle_resolutions": repository.read_frame(
            "lifecycle_resolutions",
            before.dataset_versions["lifecycle_resolutions"],
        ),
        "source_archive": repository.read_frame(
            "source_archive", before.dataset_versions["source_archive"]
        ),
    }
    parent_repository = script._CandidateRepository(
        repository, before.dataset_versions, parent_frames
    )
    parent_candidates = script._candidate_values(parent_repository, before)
    parent_report, parent_payload, parent_digest = (
        script._descendant_lifecycle_report_document(
            repository,
            before,
            parent_frames,
            parent_candidates=parent_candidates,
        )
    )

    affected = {script.OLD_SECURITY_ID, script.CANONICAL_SECURITY_ID}
    candidate_context = {
        key: value.copy(deep=True) for key, value in parent_context.items()
    }
    for dataset in (
        "security_master",
        "symbol_history",
        "corporate_actions",
        "index_constituent_anchors",
        "index_membership_events",
    ):
        parent = candidate_context[dataset]
        exact = actual_state.candidate[dataset]
        parent = parent.loc[~parent["security_id"].astype(str).isin(affected)].copy()
        exact = exact.loc[exact["security_id"].astype(str).isin(affected)].copy()
        candidate_context[dataset] = pd.concat(
            [parent, exact.reindex(columns=parent.columns)], ignore_index=True
        )
    terminals = candidate_context["daily_price_raw"]
    terminals = terminals.loc[
        ~terminals["security_id"].astype(str).isin(affected)
    ].copy()
    canonical_prices = actual_state.candidate["daily_price_raw"].loc[
        actual_state.candidate["daily_price_raw"]["security_id"]
        .astype(str)
        .eq(script.CANONICAL_SECURITY_ID)
    ]
    terminal = pd.DataFrame(
        [
            {
                "security_id": script.CANONICAL_SECURITY_ID,
                "session": pd.to_datetime(canonical_prices["session"]).max(),
            }
        ]
    ).reindex(columns=terminals.columns)
    candidate_context["daily_price_raw"] = pd.concat(
        [terminals, terminal], ignore_index=True
    )
    candidate_frames = {
        **candidate_context,
        "lifecycle_resolutions": parent_frames["lifecycle_resolutions"].loc[
            ~parent_frames["lifecycle_resolutions"]["security_id"]
            .astype(str)
            .eq(script.OLD_SECURITY_ID)
        ].copy(),
        "source_archive": parent_frames["source_archive"],
    }
    planned_versions = script._new_planned_versions(before)
    output_versions = {**before.dataset_versions, **planned_versions}
    planned_release = script.DataRelease.create(
        before.completed_session,
        output_versions,
        quality=before.quality,
        warnings=before.warnings,
    )
    candidate_repository = script._CandidateRepository(
        repository, planned_release.dataset_versions, candidate_frames
    )
    fresh = script._fresh_lifecycle_evidence(
        repository,
        before,
        planned_release,
        candidate_repository,
        candidate_frames,
        parent_release_kind="identity_price_tails_descendant",
        parent_report_payload=(parent_report, parent_payload, parent_digest),
        parent_report_candidates=parent_candidates,
    )
    after, after_etag = repository.current_release()
    assert after is not None
    assert after.to_bytes() == before.to_bytes()
    assert after_etag == before_etag
    return SimpleNamespace(
        repository=repository,
        parent_release=before,
        candidate=candidate_frames,
        planned_release=planned_release,
        parent_report=parent_report,
        parent_payload=parent_payload,
        parent_digest=parent_digest,
        fresh=fresh,
    )


@pytest.fixture(scope="module")
def actual_prepared_plan():
    repository = script.LocalDatasetRepository(PROJECT_ROOT / "data/cache")
    before, before_etag = repository.current_release()
    assert before is not None
    pointer_versions = {
        dataset: repository.current_pointer(dataset)[0].version
        for dataset in script.REQUIRED_DATASETS
    }
    prepared = script.prepare_repair(repository)
    after, after_etag = repository.current_release()
    assert after is not None
    assert after.to_bytes() == before.to_bytes()
    assert after_etag == before_etag
    assert {
        dataset: repository.current_pointer(dataset)[0].version
        for dataset in script.REQUIRED_DATASETS
    } == pointer_versions
    assert not (
        repository.root / prepared.lifecycle_report_object_path
    ).exists()
    return prepared


def test_actual_plan_is_exact_offline_and_write_free(actual_state) -> None:
    summary = actual_state.summary
    assert summary["status"] == "validated_offline_plan"
    assert summary["old_price_rows_retired"] == 1455
    assert summary["canonical_price_rows_preserved"] == 1977
    assert summary["old_factor_rows_retired"] == 1455
    assert summary["old_action_rows_retired"] == 40
    assert summary["old_terminal_resolutions_removed"] == 1
    assert summary["new_terminal_resolutions_added"] == 0
    assert summary["sp500_anchors_rebound"] == 1
    assert summary["redundant_sp500_swap_rows_removed"] == 2
    assert summary["canonical_price_action_factor_resolution_economics_preserved"]
    assert not summary["network_accessed"]
    assert summary["http_attempts"] == 0
    assert summary["eodhd_calls"] == 0
    assert not summary["r2_accessed"]
    assert not summary["writes_performed"]
    assert actual_state.proof["audit_sha256"] == script.DESCENDANT_AUDIT_SHA256


def test_actual_prepared_plan_is_sparse_and_pins_full_output_counts(
    actual_prepared_plan,
) -> None:
    prepared = actual_prepared_plan
    assert prepared.summary["status"] == "validated_offline_plan"
    assert prepared.summary["plan_materialization"] == (
        "affected_heavy_tables_plus_full_small_tables"
    )
    assert prepared.summary["full_market_price_frames_materialized"] is False
    assert prepared.summary["canonical_event_id_preexisting_rows"] == 0
    counts = prepared.summary["expected_output_row_counts"]
    assert counts["daily_price_raw"] == 2_095_793
    assert counts["adjustment_factors"] == 2_095_793
    assert script._expected_output_row_counts(prepared) == counts
    for dataset in script.HEAVY_WRITE_DATASETS:
        frame = prepared.frames[dataset]
        assert len(frame) == script.CANONICAL_PRICE_ROWS
        assert set(frame["security_id"].astype(str)) == {
            script.CANONICAL_SECURITY_ID
        }
        assert len(frame) != counts[dataset]
    assert set(prepared.candidate_frames) == set(
        script.LIFECYCLE_CANDIDATE_DATASETS
    )
    terminals = prepared.candidate_frames["daily_price_raw"]
    assert not terminals["security_id"].astype(str).eq(
        script.OLD_SECURITY_ID
    ).any()
    assert int(
        terminals["security_id"].astype(str).eq(
            script.CANONICAL_SECURITY_ID
        ).sum()
    ) == 1


def test_global_canonical_event_id_collision_is_fail_closed(actual_state) -> None:
    frames = script._load_plan_frames(
        actual_state.repository,
        actual_state.release,
    )
    actions = frames["corporate_actions"]
    collision = actions.iloc[0].to_dict()
    collision.update(
        {
            "event_id": script.CANONICAL_EVENT_ID,
            "security_id": "US:EODHD:foreign-collision",
        }
    )
    frames["corporate_actions"] = pd.concat(
        [actions, pd.DataFrame([collision]).reindex(columns=actions.columns)],
        ignore_index=True,
    )
    with pytest.raises(RuntimeError, match="already exists globally"):
        script.prepare_candidate_frames(
            frames,
            completed_session=actual_state.release.completed_session,
            parent_release_kind="identity_price_tails_descendant",
            consume_inputs=True,
        )


def test_canonical_identity_and_index_rebind_are_exact(actual_state) -> None:
    frames = actual_state.candidate
    for dataset in script.IDENTITY_WRITE_DATASETS:
        assert not frames[dataset]["security_id"].astype(str).eq(
            script.OLD_SECURITY_ID
        ).any()

    history = frames["symbol_history"].loc[
        frames["symbol_history"]["security_id"].astype(str).eq(
            script.CANONICAL_SECURITY_ID
        )
    ]
    assert {
        (
            str(row.symbol),
            script._date(row.effective_from),
            script._date(row.effective_to),
        )
        for row in history.itertuples(index=False)
    } == {
        ("SYMC", "2015-01-01", "2019-11-01"),
        ("NLOK", "2019-11-04", "2022-11-07"),
    }

    anchors = frames["index_constituent_anchors"]
    sp500 = anchors.loc[
        anchors["index_id"].astype(str).eq("sp500")
        & anchors["anchor_date"].map(script._date).eq("2015-01-07")
        & anchors["security_id"].astype(str).eq(script.CANONICAL_SECURITY_ID)
    ]
    assert len(sp500) == 1
    assert str(sp500.iloc[0]["security_id"]) == script.CANONICAL_SECURITY_ID
    nasdaq = anchors.loc[
        anchors["index_id"].astype(str).eq("nasdaq100")
        & anchors["anchor_date"].map(script._date).eq("2015-01-01")
        & anchors["security_id"].astype(str).eq(script.CANONICAL_SECURITY_ID)
    ]
    assert len(nasdaq) == 1
    assert str(nasdaq.iloc[0]["security_id"]) == script.CANONICAL_SECURITY_ID

    events = frames["index_membership_events"]
    assert not events["event_id"].astype(str).isin(
        {script.SP500_SWAP_REMOVE_EVENT_ID, script.SP500_SWAP_ADD_EVENT_ID}
    ).any()


def test_action_resolution_and_economic_inventory_are_preserved(actual_state) -> None:
    frames = actual_state.candidate
    action = frames["corporate_actions"].loc[
        frames["corporate_actions"]["event_id"].astype(str).eq(
            script.CANONICAL_EVENT_ID
        )
    ]
    assert len(action) == 1
    row = action.iloc[0]
    assert row["security_id"] == script.CANONICAL_SECURITY_ID
    assert row["new_security_id"] == script.CANONICAL_SECURITY_ID
    assert row["new_symbol"] == "NLOK"
    assert script._date(row["effective_date"]) == "2019-11-04"
    assert pd.isna(row["ratio"])
    assert pd.isna(row["cash_amount"])
    assert row["source_url"] == script.OFFICIAL_SOURCE_URL
    assert row["source_hash"] == script.OFFICIAL_SOURCE_HASH

    resolutions = frames["lifecycle_resolutions"]
    assert not resolutions["event_id"].astype(str).isin(
        {script.OLD_EVENT_ID, script.CANONICAL_EVENT_ID}
    ).any()
    later = resolutions.loc[
        resolutions["security_id"].astype(str).eq(script.CANONICAL_SECURITY_ID)
    ]
    assert len(later) == 1
    assert later.iloc[0]["event_id"] == script.NLOK_TO_GEN_EVENT_ID
    assert later.iloc[0]["successor_security_id"] == script.GEN_SECURITY_ID

    before = actual_state.summary[
        "canonical_economic_inventory_sha256_before"
    ]
    after = actual_state.summary["canonical_economic_inventory_sha256_after"]
    assert before == after
    assert after["prices"] == script.CANONICAL_PRICE_SHA256
    assert script._factor_economics_sha256(frames["adjustment_factors"]) == (
        script.CANDIDATE_FACTOR_ECONOMICS_SHA256
    )


def test_pretransition_triple_supertrend_impact_is_exact(actual_state) -> None:
    impact = actual_state.proof["signal_diagnostic"][
        "pre_transition_triple_supertrend_diff"
    ]
    for mode in ("raw", "total_return_adjusted"):
        assert impact[mode]["TripleBuySignal"]["count"] == 0
        assert impact[mode]["TripleSellSignal"]["count"] == 0
    assert impact["raw"]["TripleST1_Trend"]["count"] == 0
    assert impact["total_return_adjusted"]["TripleST1_Trend"] == {
        "count": 1,
        "sessions": ["2015-08-26"],
    }


def test_pinned_base_plan_is_fail_closed_before_any_write(actual_state) -> None:
    base = script.DataRelease.from_bytes(
        actual_state.repository.objects.get(
            f"releases/{script.PINNED_RELEASE_VERSION}.json"
        ).data
    )
    with pytest.raises(RuntimeError, match="apply the exact reviewed seven-identity"):
        script._parent_release_kind(actual_state.repository, base, {})
    report, _payload = script._current_lifecycle_report_document(
        actual_state.repository,
        base,
        {
            "source_archive": actual_state.repository.read_frame(
                "source_archive",
                base.dataset_versions["source_archive"],
            )
        },
    )
    assert report["hints_sha256"] == script.LEGACY_LIFECYCLE_HINTS_SHA256
    assert script.sha256_bytes(script.DEFAULT_HINTS_PATH.read_bytes()) == (
        script.CURRENT_LIFECYCLE_HINTS_SHA256
    )
    assert report["hints_sha256"] != script.CURRENT_LIFECYCLE_HINTS_SHA256


def test_exact_simple7_descendant_lifecycle_plan_closes_181(descendant_state) -> None:
    fresh = descendant_state.fresh
    report = json.loads(fresh.content)
    assert descendant_state.parent_report["hints_sha256"] == (
        script.CURRENT_LIFECYCLE_HINTS_SHA256
    )
    assert fresh.metadata["parent_release_kind"] == (
        "identity_price_tails_descendant"
    )
    assert fresh.metadata["current_source_report_sha256"] == (
        descendant_state.parent_digest
    )
    assert fresh.metadata["lifecycle_hints_sha256"] == (
        script.CURRENT_LIFECYCLE_HINTS_SHA256
    )
    assert fresh.metadata["full_lifecycle_finalizer_gate_passed"] is True
    assert fresh.metadata["evidence_report_sha256"] == fresh.sha256
    assert fresh.metadata["lifecycle_evidence_report_sha256"] == fresh.sha256
    assert fresh.metadata["evidence_report_object_path"] == fresh.object_path
    assert fresh.metadata["lifecycle_evidence_report_object_path"] == (
        fresh.object_path
    )
    assert report["release_version"] == descendant_state.planned_release.version
    assert report["input_dataset_versions"] == (
        descendant_state.planned_release.dataset_versions
    )
    assert report["candidate_count"] == 181
    assert len(report["records"]) == 181
    assert script.OLD_SECURITY_ID not in report["records"]
    assert script.CANONICAL_SECURITY_ID in report["records"]
    assert report["summary"]["eligible_count"] == 160
    assert report["summary"]["unresolved_count"] == 21
    assert report["summary"]["action_type_counts"] == {
        "cash_merger": 52,
        "delisting": 10,
        "stock_merger": 87,
        "ticker_change": 29,
        "unresolved": 3,
    }
    expected_coverage = {
        "coverage_gate_version": 1,
        "selection_rule": "us_terminal_v1",
        "candidate_set_sha256": script.DESCENDANT_LIFECYCLE_CANDIDATE_SET_SHA256,
        "resolution_set_sha256": (
            script.DESCENDANT_LIFECYCLE_RESOLUTION_SET_SHA256
        ),
        "candidate_count": 181,
        "resolution_count": 181,
        "applied_count": 169,
        "exception_count": 12,
        "open_count": 0,
    }
    assert fresh.metadata["lifecycle_coverage"] == expected_coverage
    assert fresh.metadata["lifecycle_candidate_set_sha256"] == (
        script.DESCENDANT_LIFECYCLE_CANDIDATE_SET_SHA256
    )
    assert fresh.metadata["lifecycle_resolution_set_sha256"] == (
        script.DESCENDANT_LIFECYCLE_RESOLUTION_SET_SHA256
    )


def test_simple7_parent_report_tampering_is_fail_closed(descendant_state) -> None:
    tampered = json.loads(json.dumps(descendant_state.parent_report))
    tampered["hints_sha256"] = script.LEGACY_LIFECYCLE_HINTS_SHA256
    tampered_payload = (
        json.dumps(tampered, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    with pytest.raises(RuntimeError, match="binding changed"):
        script._validate_descendant_lifecycle_report_payload(
            descendant_state.repository,
            descendant_state.parent_release,
            {},
            report=tampered,
            payload=tampered_payload,
            digest=script.sha256_bytes(tampered_payload),
        )

    with pytest.raises(RuntimeError, match="payload hash changed"):
        script._validate_descendant_lifecycle_report_payload(
            descendant_state.repository,
            descendant_state.parent_release,
            {},
            report=descendant_state.parent_report,
            payload=descendant_state.parent_payload + b" ",
            digest=descendant_state.parent_digest,
        )


def test_candidate_price_tampering_fails_closed(actual_state) -> None:
    prices = script._scope(
        actual_state.candidate["daily_price_raw"], "daily_price_raw"
    )
    mask = prices["security_id"].astype(str).eq(script.CANONICAL_SECURITY_ID) & prices[
        "session"
    ].map(script._date).eq("2015-01-02")
    assert int(mask.sum()) == 1
    index = prices.index[mask][0]
    original = prices.at[index, "close"]
    prices.at[index, "close"] = float(original) + 0.01
    try:
        assert script._frame_sha256(
            prices, sort_by=("security_id", "session")
        ) != script.CANONICAL_PRICE_SHA256
    finally:
        prices.at[index, "close"] = original


@pytest.mark.parametrize(
    ("dataset", "mutate"),
    [
        (
            "corporate_actions",
            lambda frame: frame.assign(
                source_hash=frame["source_hash"].where(
                    ~frame["event_id"].astype(str).eq(script.CANONICAL_EVENT_ID),
                    "0" * 64,
                )
            ),
        ),
        (
            "index_membership_events",
            lambda frame: frame.assign(
                effective_date=frame["effective_date"].where(
                    ~frame["security_id"]
                    .astype(str)
                    .eq(script.CANONICAL_SECURITY_ID),
                    "2019-11-06",
                )
            ),
        ),
    ],
)
def test_candidate_tampering_fails_closed(actual_state, dataset, mutate) -> None:
    tampered = dict(actual_state.candidate)
    tampered[dataset] = mutate(actual_state.candidate[dataset].copy())
    assert script._projection_sha256(tampered, dataset) != (
        script.CANDIDATE_PROJECTION_SHA256[dataset]
    )


def test_candidate_tampering_and_replay_are_fail_closed(actual_state) -> None:
    tampered = dict(actual_state.candidate)
    history = tampered["symbol_history"].copy()
    mask = history["symbol"].astype(str).eq("SYMC") & history[
        "security_id"
    ].astype(str).eq(script.CANONICAL_SECURITY_ID)
    history.loc[mask, "effective_to"] = "2019-11-04"
    tampered["symbol_history"] = history
    assert script._projection_sha256(tampered, "symbol_history") != (
        script.CANDIDATE_PROJECTION_SHA256["symbol_history"]
    )


def test_code_pinned_nonterminal_binding_rejects_generic_mutations() -> None:
    identity = script.reviewed_same_sid_no_data_spec()
    extraction = script.reviewed_nonterminal_extraction()
    target = {
        "target_id": identity["source_target_id"],
        "security_id": identity["security_id"],
        "provider_symbol": "SYMC",
        "identity_active_from": identity["old_active_from"],
        "identity_active_to": identity["old_active_to"],
        "terminal_event_id": identity["event_id"],
        "successor_security_id": identity["security_id"],
    }
    event = {
        "event_id": identity["event_id"],
        "security_id": identity["security_id"],
        "action_type": "ticker_change",
        "effective_date": identity["effective_date"],
        "new_security_id": identity["security_id"],
        "new_symbol": "NLOK",
        "evidence_sha256": identity["official_source_hash"],
        "status": "passed",
        "reviewed_extraction_match": True,
        "candidate_id": "",
    }
    assert script.reviewed_nonterminal_extraction() == extraction
    assert script.reviewed_same_sid_no_data_spec() == identity
    assert script.reviewed_nonterminal_extraction()["ratio"] is None
    assert script.reviewed_nonterminal_extraction()["cash_amount"] is None
    assert script.reviewed_same_sid_no_data_spec()["source_target_id"] == (
        "76cfddc97b878414119dfd9db08e356216cffc4ddc2839188451df534e11296f"
    )
    verifier = sys.modules[
        "supertrend_quant.market_store.symc_nlok_identity"
    ].exact_nonterminal_binding_inputs
    assert verifier(target, event, extraction)
    for field, value in (
        ("target_id", "0" * 64),
        ("identity_active_to", "2019-11-04"),
        ("successor_security_id", script.OLD_SECURITY_ID),
    ):
        changed = dict(target)
        changed[field] = value
        assert not verifier(changed, event, extraction)
    changed_event = dict(event, candidate_id=script.OLD_CANDIDATE_ID)
    assert not verifier(target, changed_event, extraction)
    changed_extraction = dict(extraction, effective_date="2019-11-05")
    assert not verifier(target, event, changed_extraction)


def test_exact_simple7_descendant_handoff_contract(monkeypatch) -> None:
    versions = dict(script.PINNED_DATASET_VERSIONS)
    for dataset in script.identity_tails.WRITE_DATASETS:
        versions[dataset] = f"simple7-{dataset}"
    release = script.DataRelease(
        version="simple7-descendant",
        created_at="2026-07-19T00:00:00Z",
        completed_session="2026-07-15",
        dataset_versions=versions,
        quality="valid",
        warnings=(),
    )
    summary = {
        "status": "already_repaired",
        "registry_inventory_sha256": (
            script.identity_tails.TRUSTED_REGISTRY_INVENTORY_SHA256
        ),
        "candidate_content_sha256": (
            script.identity_tails.EXPECTED_CANDIDATE_CONTENT_SHA256
        ),
        "lifecycle_candidate_set_sha256": (
            script.identity_tails.EXPECTED_REPAIRED_LIFECYCLE_CANDIDATE_SET_SHA256
        ),
        "lifecycle_resolution_set_sha256": (
            script.identity_tails.EXPECTED_REPAIRED_LIFECYCLE_RESOLUTION_SET_SHA256
        ),
    }
    monkeypatch.setattr(
        script.identity_tails, "_exact_repair_manifests", lambda *_: True
    )
    monkeypatch.setattr(
        script.identity_tails,
        "prepare_repair",
        lambda *_: SimpleNamespace(release=release, summary=summary),
    )
    assert script._parent_release_kind(object(), release, {}) == (
        "identity_price_tails_descendant"
    )

    changed = dict(summary, lifecycle_resolution_set_sha256="0" * 64)
    monkeypatch.setattr(
        script.identity_tails,
        "prepare_repair",
        lambda *_: SimpleNamespace(release=release, summary=changed),
    )
    with pytest.raises(RuntimeError, match="exact replay failed"):
        script._parent_release_kind(object(), release, {})


def test_projected_repository_routes_prior_versions_to_immutable_base() -> None:
    calls: list[tuple[str, str | None]] = []

    class _Base:
        def read_frame(self, dataset, version=None):
            calls.append((dataset, version))
            return pd.DataFrame({"security_id": [f"base:{version}"]})

    projected = pd.DataFrame({"security_id": ["projected"]})
    repository = script._CandidateRepository(
        _Base(),
        {"security_master": "planned-v2"},
        {"security_master": projected},
    )
    assert repository.read_frame("security_master") is projected
    assert repository.read_frame("security_master", "planned-v2") is projected
    prior = repository.read_frame("security_master", "immutable-v1")
    assert prior.iloc[0]["security_id"] == "base:immutable-v1"
    assert calls == [("security_master", "immutable-v1")]


def test_integration_draft_is_explicitly_not_publication_ready() -> None:
    draft = yaml.safe_load(DRAFT_PATH.read_bytes())
    assert draft["schema"] == "us_symc_nlok_identity_integration_draft/v1"
    assert draft["publication_ready"] is False
    assert draft["required_application_order"] == [
        "repair_us_identity_price_tails",
        "repair_us_symc_nlok_identity",
        "repair_us_arnc_hwm_ticker_change",
    ]
    assert draft["parent_release_contract"]["pinned_base_plan_and_apply"] == (
        "fail_closed"
    )
    assert draft["parent_release_contract"]["current_hints_sha256"] == (
        script.CURRENT_LIFECYCLE_HINTS_SHA256
    )
    assert draft["reviewed_nonterminal_extraction"] == (
        script.reviewed_nonterminal_extraction()
    )
    assert draft["reviewed_same_sid_no_data"]["terminal_resolution_forbidden"]
    chain = draft["reviewed_no_data_successor_chain"]
    assert [node["provider_symbol"] for node in chain["nodes"]] == [
        "SYMC",
        "NLOK",
    ]
    assert chain["final"]["provider_symbol"] == "GEN"
    assert all(
        node["cache_wrapper_sha256"] == "PENDING_POST_REPAIR_EXACT_REQUEST"
        for node in chain["nodes"]
    )
    finalizer = draft["finalizer_supersession"]
    assert finalizer["legacy_pinned_base_hints_sha256"] == (
        script.LEGACY_LIFECYCLE_HINTS_SHA256
    )
    assert finalizer["exact_simple7_parent_hints_sha256"] == (
        script.CURRENT_LIFECYCLE_HINTS_SHA256
    )
    assert finalizer["candidate_count_before"] == 182
    assert finalizer["candidate_count_after"] == 181
    assert finalizer["stale_report_fails_before_write"] is True
    assert finalizer["generic_candidate_tolerance"] is False
    lifecycle = draft["lifecycle_transaction_projection"]
    assert lifecycle["applied_count"] == 169
    assert lifecycle["exception_count"] == 12
    assert lifecycle["candidate_set_sha256"] == (
        script.DESCENDANT_LIFECYCLE_CANDIDATE_SET_SHA256
    )
    assert lifecycle["resolution_set_sha256"] == (
        script.DESCENDANT_LIFECYCLE_RESOLUTION_SET_SHA256
    )
    assert lifecycle["current_base_plan_gate"] == "fail_closed"
    observed = draft["actual_no_write_plan_2026_07_19"]
    assert observed["parent_release_version"] == "20260715-20260719T031437502501Z"
    assert observed["plan_materialization"] == (
        "affected_heavy_tables_plus_full_small_tables"
    )
    assert observed["full_market_price_frames_materialized"] is False
    assert observed["max_resident_set_kb"] < 1024 * 1024
    assert observed["expected_output_row_counts"] == {
        "daily_price_raw": 2_095_793,
        "adjustment_factors": 2_095_793,
    }
    assert observed["canonical_event_id_preexisting_rows"] == 0


def test_release_cas_stops_before_object_writes(actual_state, tmp_path: Path) -> None:
    fake = SimpleNamespace(
        root=tmp_path,
        current_release=lambda: (None, "changed-etag"),
    )
    prepared = script.PreparedRepair(
        release=actual_state.release,
        release_etag=actual_state.release_etag,
        pointer_etags={},
        version_state={},
        frames={},
        planned_versions={},
        planned_release=None,
        lifecycle_report_content=b"",
        lifecycle_report_object_path="",
        lifecycle_metadata={},
        warnings=(),
        summary={"status": "validated_offline_plan"},
    )
    with pytest.raises(RuntimeError, match="Current release changed"):
        script.apply_repair(fake, prepared)


@dataclass
class _ObjectValue:
    data: bytes
    etag: str


class _MemoryObjects:
    def __init__(self, values: dict[str, bytes]):
        self.values = dict(values)
        self.puts: list[str] = []

    def get(self, key: str) -> _ObjectValue:
        data = self.values[key]
        return _ObjectValue(data=data, etag=script.sha256_bytes(data))

    def put(self, key: str, data: bytes, *, if_match: str | None = None):
        current = self.get(key)
        if if_match is not None and current.etag != if_match:
            raise RuntimeError("CAS mismatch")
        self.values[key] = data
        self.puts.append(key)
        return self.get(key)


def test_rollback_restores_release_and_all_dataset_pointers(actual_state) -> None:
    old_release = actual_state.release
    planned = {dataset: f"planned-{dataset}" for dataset in script.WRITE_DATASETS}
    planned_release = script.DataRelease(
        version="planned-release",
        created_at=old_release.created_at,
        completed_session=old_release.completed_session,
        dataset_versions={**old_release.dataset_versions, **planned},
        quality=old_release.quality,
        warnings=old_release.warnings,
    )
    old_pointers: dict[str, bytes] = {}
    owned_pointers: dict[str, bytes] = {}
    values = {"releases/current.json": planned_release.to_bytes()}
    for dataset in script.WRITE_DATASETS:
        old_pointer = script.CurrentPointer(
            dataset=dataset,
            version=old_release.dataset_versions[dataset],
            manifest_path=f"datasets/{dataset}/old/manifest.json",
            manifest_sha256="1" * 64,
            updated_at="2026-07-19T00:00:00Z",
        ).to_bytes()
        planned_pointer = script.CurrentPointer(
            dataset=dataset,
            version=planned[dataset],
            manifest_path=f"datasets/{dataset}/planned/manifest.json",
            manifest_sha256="2" * 64,
            updated_at="2026-07-19T00:00:01Z",
        ).to_bytes()
        old_pointers[dataset] = old_pointer
        owned_pointers[dataset] = planned_pointer
        values[f"current/{dataset}.json"] = planned_pointer
    objects = _MemoryObjects(values)
    fake = SimpleNamespace(
        objects=objects,
        current_key=lambda dataset: f"current/{dataset}.json",
    )
    errors = script._restore_transaction(
        fake,
        old_release_bytes=old_release.to_bytes(),
        old_pointer_bytes=old_pointers,
        planned_release_bytes=planned_release.to_bytes(),
        owned_pointer_bytes=owned_pointers,
    )
    assert errors == ()
    assert objects.values["releases/current.json"] == old_release.to_bytes()
    for dataset in script.WRITE_DATASETS:
        assert objects.values[f"current/{dataset}.json"] == old_pointers[dataset]
    assert len(objects.puts) == 1 + len(script.WRITE_DATASETS)


def test_source_has_no_external_or_r2_execution_path() -> None:
    source = SCRIPT_PATH.read_text()
    for forbidden in (
        "import requests",
        "import httpx",
        "from urllib",
        "import boto3",
        "EODHD_API_KEY",
        "R2_ACCESS_KEY",
        "--fetch",
    ):
        assert forbidden not in source


def _synthetic_daily_price_repository(tmp_path: Path):
    repository = script.LocalDatasetRepository(tmp_path / "cache")
    rows = []
    identities = (
        (script.OLD_SECURITY_ID, "2020-01-02"),
        (script.OLD_SECURITY_ID, "2020-01-03"),
        (script.CANONICAL_SECURITY_ID, "2020-01-02"),
        (script.CANONICAL_SECURITY_ID, "2020-01-03"),
        ("US:EODHD:OTHER", "2020-02-03"),
        ("US:EODHD:OTHER", "2020-02-04"),
    )
    for number, (security_id, session) in enumerate(identities, start=1):
        rows.append(
            {
                "security_id": security_id,
                "session": session,
                "open": 10.0 + number,
                "high": 11.0 + number,
                "low": 9.0 + number,
                "close": 10.5 + number,
                "volume": 1_000.0 + number,
                "currency": "USD",
                "source": "synthetic",
                "retrieved_at": "2026-07-19T00:00:00Z",
                "source_hash": f"{number:064x}",
            }
        )
    result = repository.write_frame(
        "daily_price_raw",
        pd.DataFrame(rows),
        completed_session="2020-02-04",
        version="parent-daily-price",
        metadata={
            "_logical_quality": script.DataQuality.DEGRADED,
            "_logical_warnings": ("synthetic warning",),
            "source_mode": "synthetic_fixture",
            "official_coverage_start": "2020-01-02",
            "official_coverage_end": "2020-02-04",
        },
    )
    assert not result.conflict
    pointer, etag = repository.current_pointer("daily_price_raw")
    assert pointer is not None and etag is not None
    expected_files = [
        {
            "path": item.path,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
            "row_count": item.row_count,
        }
        for item in result.manifest.files
    ]
    return repository, result.manifest, etag, expected_files


def _parquet_rows(path: Path):
    _pc, pq = script._arrow_modules()
    table = pq.ParquetFile(path).read()
    return table.schema, table.to_pylist()


def test_heavy_delete_only_writer_is_bounded_and_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, parent, pointer_etag, expected_files = (
        _synthetic_daily_price_repository(tmp_path)
    )
    parent_root = repository.root / repository.version_prefix(
        "daily_price_raw", parent.version
    )
    parent_bytes = {
        item.path: (parent_root / item.path).read_bytes()
        for item in parent.files
    }

    def forbid_pandas_path(*_args, **_kwargs):
        raise AssertionError("heavy writer must not use repository pandas I/O")

    monkeypatch.setattr(repository, "read_frame", forbid_pandas_path)
    monkeypatch.setattr(repository, "write_frame", forbid_pandas_path)
    result = script._write_delete_only_heavy_dataset(
        repository,
        dataset="daily_price_raw",
        parent_version=parent.version,
        version="child-daily-price",
        completed_session="2020-02-04",
        metadata={"operation": "synthetic_delete_only"},
        expected_pointer_etag=pointer_etag,
        expected_parent_files=expected_files,
        retired_security_id=script.OLD_SECURITY_ID,
        expected_removed_rows=2,
        expected_output_rows=4,
    )
    assert not result.conflict and result.validation.valid
    child = result.manifest
    assert child.metadata["inherits_parent"] is False
    assert child.parent_version == parent.version
    assert child.quality == parent.quality
    assert child.warnings == parent.warnings
    assert child.source_mode == parent.source_mode
    assert child.official_coverage_start == parent.official_coverage_start
    assert child.official_coverage_end == parent.official_coverage_end
    assert sum(item.row_count for item in child.files) == 4
    current, _etag = repository.current_pointer("daily_price_raw")
    assert current is not None and current.version == child.version

    child_root = repository.root / repository.version_prefix(
        "daily_price_raw", child.version
    )
    parent_by_path = {item.path: item for item in parent.files}
    for child_item in child.files:
        parent_item = parent_by_path[child_item.path]
        parent_path = parent_root / parent_item.path
        child_path = child_root / child_item.path
        parent_schema, parent_rows = _parquet_rows(parent_path)
        child_schema, child_rows = _parquet_rows(child_path)
        expected_rows = [
            row
            for row in parent_rows
            if row["security_id"] != script.OLD_SECURITY_ID
        ]
        assert child_rows == expected_rows
        assert child_schema.equals(parent_schema, check_metadata=True)
        inspected = script._inspect_heavy_parquet_file(
            "daily_price_raw",
            child_path,
            child_item.path,
            retired_security_id=script.OLD_SECURITY_ID,
        )
        script._assert_manifest_file_matches_inspection(
            "daily_price_raw", child_item, inspected
        )
        if len(expected_rows) == len(parent_rows):
            assert child_path.read_bytes() == parent_bytes[child_item.path]
            assert child_item == parent_item


def test_heavy_writer_mismatch_fails_before_staging_or_pointer(
    tmp_path: Path,
) -> None:
    repository, parent, pointer_etag, expected_files = (
        _synthetic_daily_price_repository(tmp_path)
    )
    pointer_before = repository.objects.get(
        repository.current_key("daily_price_raw")
    ).data
    with pytest.raises(RuntimeError, match="retired row count changed"):
        script._write_delete_only_heavy_dataset(
            repository,
            dataset="daily_price_raw",
            parent_version=parent.version,
            version="mismatch-child",
            completed_session="2020-02-04",
            metadata={},
            expected_pointer_etag=pointer_etag,
            expected_parent_files=expected_files,
            retired_security_id=script.OLD_SECURITY_ID,
            expected_removed_rows=3,
            expected_output_rows=3,
        )
    assert repository.objects.get(
        repository.current_key("daily_price_raw")
    ).data == pointer_before
    assert not (
        repository.root
        / repository.version_prefix("daily_price_raw", "mismatch-child")
    ).exists()
    staging = repository.root / ".staging/daily_price_raw"
    assert not staging.exists() or not tuple(staging.rglob("mismatch-child*"))


def test_heavy_writer_snapshot_closes_live_parent_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, parent, pointer_etag, expected_files = (
        _synthetic_daily_price_repository(tmp_path)
    )
    pointer_before = repository.objects.get(
        repository.current_key("daily_price_raw")
    ).data
    parent_root = repository.root / repository.version_prefix(
        "daily_price_raw", parent.version
    )
    affected = next(
        item
        for item in parent.files
        if any(
            row["security_id"] == script.OLD_SECURITY_ID
            for row in _parquet_rows(parent_root / item.path)[1]
        )
    )
    original_rows = _parquet_rows(parent_root / affected.path)[1]
    expected_rows = [
        row
        for row in original_rows
        if row["security_id"] != script.OLD_SECURITY_ID
    ]
    original_rewrite = script._rewrite_affected_heavy_file
    captured: dict[str, object] = {}

    def mutate_live_parent_then_rewrite(
        dataset,
        inspected,
        destination,
        *,
        retired_security_id,
        source_path,
    ):
        assert script.sha256_file(source_path) == affected.sha256
        inspected.path.write_bytes(inspected.path.read_bytes() + b"foreign")
        assert script.sha256_file(source_path) == affected.sha256
        original_rewrite(
            dataset,
            inspected,
            destination,
            retired_security_id=retired_security_id,
            source_path=source_path,
        )
        captured["rows"] = _parquet_rows(destination)[1]

    monkeypatch.setattr(
        script,
        "_rewrite_affected_heavy_file",
        mutate_live_parent_then_rewrite,
    )
    with pytest.raises(RuntimeError, match="source changed after pre-scan"):
        script._write_delete_only_heavy_dataset(
            repository,
            dataset="daily_price_raw",
            parent_version=parent.version,
            version="raced-child",
            completed_session="2020-02-04",
            metadata={},
            expected_pointer_etag=pointer_etag,
            expected_parent_files=expected_files,
            retired_security_id=script.OLD_SECURITY_ID,
            expected_removed_rows=2,
            expected_output_rows=4,
        )
    assert captured["rows"] == expected_rows
    assert repository.objects.get(
        repository.current_key("daily_price_raw")
    ).data == pointer_before
    assert not (
        repository.root
        / repository.version_prefix("daily_price_raw", "raced-child")
    ).exists()


def test_heavy_writer_pointer_cas_conflict_is_reported(
    tmp_path: Path,
) -> None:
    repository, parent, _pointer_etag, expected_files = (
        _synthetic_daily_price_repository(tmp_path)
    )
    pointer_before = repository.objects.get(
        repository.current_key("daily_price_raw")
    ).data
    result = script._write_delete_only_heavy_dataset(
        repository,
        dataset="daily_price_raw",
        parent_version=parent.version,
        version="conflicted-child",
        completed_session="2020-02-04",
        metadata={},
        expected_pointer_etag="0" * 64,
        expected_parent_files=expected_files,
        retired_security_id=script.OLD_SECURITY_ID,
        expected_removed_rows=2,
        expected_output_rows=4,
    )
    assert result.conflict
    assert result.conflict_path == (
        "conflicts/daily_price_raw/conflicted-child/manifest.json"
    )
    assert repository.objects.get(
        repository.current_key("daily_price_raw")
    ).data == pointer_before
    assert repository.objects.get(result.conflict_path).data == (
        result.manifest.to_bytes()
    )


def test_heavy_inspector_rejects_null_security_id(tmp_path: Path) -> None:
    repository, parent, _etag, _files = _synthetic_daily_price_repository(
        tmp_path
    )
    _pc, pq = script._arrow_modules()
    import pyarrow as pa

    item = parent.files[0]
    source = (
        repository.root
        / repository.version_prefix("daily_price_raw", parent.version)
        / item.path
    )
    table = pq.ParquetFile(source).read()
    values = table.column("security_id").to_pylist()
    values[0] = None
    changed = table.set_column(
        table.schema.get_field_index("security_id"),
        table.schema.field("security_id"),
        pa.array(values, type=table.schema.field("security_id").type),
    )
    destination = tmp_path / item.path
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(changed, destination, compression="zstd")
    with pytest.raises(RuntimeError, match="null security_id"):
        script._inspect_heavy_parquet_file(
            "daily_price_raw",
            destination,
            item.path,
            retired_security_id=script.OLD_SECURITY_ID,
        )
