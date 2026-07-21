from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest

from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.ingest import EodhdCallBudget


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_sivb_avp_terminal_transitions.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_sivb_avp_terminal_transitions", SCRIPT_PATH
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
    return {
        key: hashlib.sha256(repository.objects.get(key).data).hexdigest()
        for key in keys
    }


@pytest.fixture(scope="module")
def actual_plan() -> dict[str, object]:
    if not (REPOSITORY_ROOT / "releases/current.json").is_file():
        pytest.skip("Actual current-release cache is unavailable.")
    repository = LocalDatasetRepository(REPOSITORY_ROOT)
    release, _ = repository.current_release()
    if release is None or not set(script.REQUIRED_DATASETS).issubset(
        release.dataset_versions
    ):
        pytest.skip("Actual release lacks SIVB/AVP planner datasets.")
    before = _control_hashes(repository)
    old = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in script.REQUIRED_DATASETS
    }
    # These five integration assertions describe the pre-repair planning
    # state.  Once the shared cache has advanced through the SIVB/AVP writer,
    # raw-OCC rebinding, NTCO/NTCOY repair, and lifecycle finalization, its
    # exact current-state preservation is covered by the finalizer regression
    # suite instead of replaying this historical pre-repair fixture.
    if not old["corporate_actions"]["event_id"].astype(str).eq(
        script.AVP_OLD_EVENT_ID
    ).any():
        pytest.skip("Actual cache is already beyond the pre-SIVB/AVP plan state.")
    prepared = script.prepare_plan(repository)
    after = _control_hashes(repository)
    return {
        "release": release,
        "before": before,
        "after": after,
        "old": old,
        "prepared": prepared,
        "plan": prepared.plan,
    }


def _security_rows(frame: pd.DataFrame, security_id: str) -> pd.DataFrame:
    return frame.loc[frame["security_id"].astype(str).eq(security_id)].copy()


def _sec_market_fixture() -> bytes:
    return (
        "<html><body>SVB Financial Group. Trading of the Company’s common stock "
        "(SIVB:NASDAQ) on Nasdaq was halted on March 10, 2023 and will be "
        "suspended on March 28, 2023. OTC Pink Quotation System.</body></html>"
    ).encode()


def _eod_fixture() -> bytes:
    return json.dumps(
        [
            {
                "date": "2023-03-28",
                "open": 0.53,
                "high": 0.74,
                "low": 0.01,
                "close": 0.4,
                "volume": 84502118,
            },
            {
                "date": "2024-11-07",
                "open": 0.005,
                "high": 0.006,
                "low": 0.005,
                "close": 0.006,
                "volume": 797,
            },
        ]
    ).encode()


def _budget(root: Path) -> EodhdCallBudget:
    return EodhdCallBudget(
        state_path=root / "state/eodhd_call_budget.json",
        limit=100,
        reserve=5,
        seed_used=10,
        period="2026-07-18",
    )


def test_actual_plan_is_read_only_offline_and_ready(actual_plan: dict) -> None:
    plan = actual_plan["plan"]
    assert actual_plan["before"] == actual_plan["after"]
    assert plan["status"] == "ready_offline_plan"
    assert plan["offline_guards"] == {
        "read_only": True,
        "apply_supported": True,
        "network_accessed": False,
        "eodhd_calls": 0,
        "r2_accessed": False,
        "source_archive_mutated": False,
    }
    assert plan["cases"]["AVP"]["status"] == "ready_offline_plan"
    assert plan["cases"]["SIVB"]["status"] == "ready_offline_plan"
    assert plan["validation"]["row_deltas"] == {
        "corporate_actions": 1,
        "lifecycle_resolutions": 0,
        "security_master": 0,
        "symbol_history": 1,
        "daily_price_raw": script.SIVBQ_STORED_ROWS,
        "adjustment_factors": script.SIVBQ_STORED_ROWS,
        "source_archive": 3,
        "index_constituent_anchors": 0,
        "index_membership_events": 0,
    }
    assert plan["validation"]["daily_price_rows_added"] == 408
    assert plan["validation"]["security_master_rows_modified"] == 1
    assert plan["validation"]["source_archive_rows_changed"] == 3
    assert plan["write_datasets"] == list(script.WRITE_DATASETS)
    assert plan["repair_registry_sha256"] == script.TRUSTED_REPAIR_REGISTRY_SHA256
    assert (
        plan["evidence_inventory_sha256"]
        == script.TRUSTED_EVIDENCE_INVENTORY_SHA256
    )
    assert plan["validation"]["terminal_issues_after_plan"] == []


def test_avp_official_and_price_evidence_pin_first_priceable_session(
    actual_plan: dict,
) -> None:
    avp = actual_plan["plan"]["cases"]["AVP"]
    assert avp["legal_completion_date"] == "2020-01-03"
    assert avp["last_source_price_session"] == "2020-01-03"
    assert avp["market_transition_session"] == "2020-01-06"
    assert avp["successor_first_price_session"] == "2020-01-06"
    assert avp["ratio"] == 0.3
    assert avp["economics_changed"] is False
    assert avp["official_evidence"]["all_required_patterns_passed"] is True
    assert avp["official_evidence"]["source_hash"] == script.AVP_OFFICIAL_HASH
    assert avp["price_evidence"]["successor"] == {
        "first_price_session": "2020-01-06",
        "first_ohlcv": list(script.NTCO_FIRST_OHLCV),
        "raw_source_hash": script.NTCO_RAW_HASH,
        "source_envelope_hash": script.NTCO_ENVELOPE_HASH,
    }


def test_avp_candidate_mutations_are_narrow_and_keep_legal_date(
    actual_plan: dict,
) -> None:
    old = actual_plan["old"]
    new = actual_plan["prepared"].frames
    old_action = old["corporate_actions"].loc[
        old["corporate_actions"]["event_id"].astype(str).eq(script.AVP_OLD_EVENT_ID)
    ].iloc[0]
    new_action = new["corporate_actions"].loc[
        new["corporate_actions"]["event_id"].astype(str).eq(script.AVP_NEW_EVENT_ID)
    ].iloc[0]
    changed_action_fields = {
        column
        for column in old_action.index
        if not (
            old_action[column] == new_action[column]
            or (pd.isna(old_action[column]) and pd.isna(new_action[column]))
        )
    }
    assert changed_action_fields == {
        "event_id",
        "effective_date",
        "ex_date",
        "metadata",
    }
    assert script._date(new_action["announcement_date"]) == script.AVP_LEGAL_COMPLETION
    assert script._date(new_action["effective_date"]) == script.AVP_MARKET_TRANSITION
    assert script._date(new_action["ex_date"]) == script.AVP_MARKET_TRANSITION
    assert float(new_action["ratio"]) == 0.3
    metadata = script.json.loads(str(new_action["metadata"]))
    assert metadata["legal_completion_date"] == script.AVP_LEGAL_COMPLETION
    assert metadata["market_transition_session"] == script.AVP_MARKET_TRANSITION

    old_avp_history = _security_rows(old["symbol_history"], script.AVP_ID).iloc[0]
    new_avp_history = _security_rows(new["symbol_history"], script.AVP_ID).iloc[0]
    old_ntco_history = _security_rows(old["symbol_history"], script.NTCO_ID).iloc[0]
    new_ntco_history = _security_rows(new["symbol_history"], script.NTCO_ID).iloc[0]
    assert script._date(old_avp_history["effective_to"]) == ""
    assert script._date(new_avp_history["effective_to"]) == "2020-01-03"
    assert script._date(old_ntco_history["effective_from"]) == "2020-01-03"
    assert script._date(new_ntco_history["effective_from"]) == "2020-01-06"
    assert _security_rows(old["daily_price_raw"], script.AVP_ID).equals(
        _security_rows(new["daily_price_raw"], script.AVP_ID)
    )
    for security_id in (script.AVP_ID, script.NTCO_ID):
        assert _security_rows(old["security_master"], security_id).equals(
            _security_rows(new["security_master"], security_id)
        )


def test_avp_terminal_issue_resolves_without_factor_economic_change(
    actual_plan: dict,
) -> None:
    validation = actual_plan["plan"]["validation"]
    before = {
        (issue["symbol"], issue["code"])
        for issue in validation["terminal_issues_before"]
    }
    after = {
        (issue["symbol"], issue["code"])
        for issue in validation["terminal_issues_after_plan"]
    }
    assert ("AVP", "successor_not_ready_on_transition") in before
    assert not any(symbol == "AVP" for symbol, _ in after)
    assert validation["avp_target_issue_resolved"] is True
    assert validation["avp_adjustment_factor_rows_checked"] == script.AVP_RAW_ROWS
    assert validation["avp_adjustment_factor_economic_rows_changed"] == 0
    assert validation["factor_lineage_rebuild_required_on_apply"] is True


def test_sivb_plan_bridges_same_identity_and_preserves_legal_zero_economics(
    actual_plan: dict,
) -> None:
    old = actual_plan["old"]
    new = actual_plan["prepared"].frames
    sivb = actual_plan["plan"]["cases"]["SIVB"]
    assert sivb["last_nasdaq_price_session_observed"] == "2023-03-09"
    assert sivb["official_nasdaq_halt_date"] == "2023-03-10"
    assert sivb["official_nasdaq_suspension_date"] == "2023-03-28"
    assert sivb["otc_first_price_session"] == "2023-03-28"
    assert sivb["otc_last_price_session"] == "2024-11-07"
    assert sivb["market_terminal_session"] == "2024-11-08"
    assert sivb["legal_cancellation_date"] == "2024-11-07"
    assert sivb["legal_zero_distribution_action_preserved"] is True
    assert sivb["zero_cash_moved_to_2023_03_10"] is False
    assert sivb["economics_invented"] is False
    assert sivb["same_security_identity"] is True
    assert sivb["same_security_id"] == script.SIVB_ID
    assert sivb["ticker_change_ratio"] == 1.0
    assert sivb["missing_evidence"] == []
    assert (
        sivb["official_evidence"]["legal_cancellation"]["zero_distribution_proven"]
        is True
    )
    assert sivb["official_evidence"]["occ_raw_pdf_archived"] is False

    actions = _security_rows(new["corporate_actions"], script.SIVB_ID)
    assert not actions["event_id"].astype(str).eq(script.SIVB_EVENT_ID).any()
    ticker = actions.loc[
        actions["event_id"].astype(str).eq(script.SIVB_TICKER_EVENT_ID)
    ].iloc[0]
    assert ticker["action_type"] == "ticker_change"
    assert script._date(ticker["effective_date"]) == "2023-03-28"
    assert ticker["new_security_id"] == script.SIVB_ID
    assert ticker["new_symbol"] == "SIVBQ"
    assert ticker["source_hash"] == script.SIVB_OCC_MEMO_COLLECTED_HASH
    action = actions.loc[
        actions["event_id"].astype(str).eq(script.SIVB_MARKET_EXIT_EVENT_ID)
    ].iloc[0]
    assert script._date(action["effective_date"]) == "2024-11-08"
    assert script._date(action["ex_date"]) == "2024-11-08"
    assert float(action["cash_amount"]) == 0.0
    metadata = script.json.loads(str(action["metadata"]))
    assert metadata["legal_cancellation_date"] == "2024-11-07"
    assert metadata["last_observed_otc_price_session"] == "2024-11-07"
    assert metadata["legal_zero_distribution_preserved"] is True

    resolution = _security_rows(new["lifecycle_resolutions"], script.SIVB_ID).iloc[0]
    assert resolution["candidate_id"] == script.SIVBQ_CANDIDATE_ID
    assert resolution["symbol"] == "SIVBQ"
    assert script._date(resolution["last_price_date"]) == "2024-11-07"
    assert resolution["event_id"] == script.SIVB_MARKET_EXIT_EVENT_ID

    master = _security_rows(new["security_master"], script.SIVB_ID).iloc[0]
    assert master["primary_symbol"] == "SIVBQ"
    assert master["provider_symbol"] == "SIVBQ.US"
    assert master["exchange"] == "PINK"
    assert script._date(master["active_to"]) == "2024-11-07"

    history = _security_rows(new["symbol_history"], script.SIVB_ID)
    old_alias = history.loc[history["symbol"].astype(str).eq("SIVB")].iloc[0]
    new_alias = history.loc[history["symbol"].astype(str).eq("SIVBQ")].iloc[0]
    assert script._date(old_alias["effective_to"]) == "2023-03-27"
    assert script._date(new_alias["effective_from"]) == "2023-03-28"
    assert script._date(new_alias["effective_to"]) == "2024-11-07"

    old_prices = _security_rows(old["daily_price_raw"], script.SIVB_ID).copy()
    new_prices = _security_rows(new["daily_price_raw"], script.SIVB_ID).copy()
    assert len(old_prices) == script.SIVB_RAW_ROWS
    assert len(new_prices) == script.SIVB_RAW_ROWS + script.SIVBQ_STORED_ROWS
    new_prices["_session"] = pd.to_datetime(new_prices["session"]).dt.date.astype(str)
    assert not new_prices["_session"].eq("2024-09-02").any()
    first_otc = new_prices.loc[new_prices["_session"].eq("2023-03-28")].iloc[0]
    last_otc = new_prices.loc[new_prices["_session"].eq("2024-11-07")].iloc[0]
    assert tuple(first_otc[field] for field in ("open", "high", "low", "close", "volume")) == script.SIVBQ_FIRST_OHLCV
    assert tuple(last_otc[field] for field in ("open", "high", "low", "close", "volume")) == script.SIVBQ_LAST_OHLCV
    assert first_otc["source_hash"] == script.SIVBQ_EOD_COLLECTED_HASH

    factors = _security_rows(new["adjustment_factors"], script.SIVB_ID)
    assert len(factors) == len(new_prices)
    assert set(factors["split_factor"].astype(float)) == {1.0}
    assert set(factors["total_return_factor"].astype(float)) == {1.0}
    lineage = script._factor_source_version(
        actual_plan["prepared"].planned_versions["daily_price_raw"],
        actual_plan["prepared"].planned_versions["corporate_actions"],
    )
    assert set(new["adjustment_factors"]["source_version"].astype(str)) == {
        lineage
    }
    assert set(new["adjustment_factors"]["source_hash"].astype(str)) == {
        lineage
    }
    assert (
        actual_plan["plan"]["validation"]["full_adjustment_factor_lineage"]
        ["provenance_rows_rebound"]
        == len(new["adjustment_factors"])
    )
    assert actual_plan["plan"]["validation"]["sivb_target_issue_resolved"] is True


def test_actual_sivb_transition_cache_is_exactly_pinned() -> None:
    cached = script._verify_reviewed_sivb_transition_cache(REPOSITORY_ROOT)

    assert cached["budget_receipt"]["used_before"] == 8839
    assert cached["budget_receipt"]["used_after"] == 8840
    assert cached["budget_receipt"]["delta"] == 1
    assert cached["evidence"]["sec"]["source_hash"] == script.SIVB_SEC_MARKET_COLLECTED_HASH
    assert cached["evidence"]["occ"]["source_hash"] == script.SIVB_OCC_MEMO_COLLECTED_HASH
    assert cached["evidence"]["eodhd"]["source_hash"] == script.SIVBQ_EOD_COLLECTED_HASH
    assert cached["eod_summary"] == {
        "row_count": 409,
        "first_session": "2023-03-28",
        "first_ohlcv": list(script.SIVBQ_FIRST_OHLCV),
        "last_session": "2024-11-07",
        "last_ohlcv": list(script.SIVBQ_LAST_OHLCV),
    }


def test_sivb_collector_uses_one_eodhd_attempt_and_replays_cache(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fetcher(url: str, _headers: dict[str, str], _max_bytes: int) -> bytes:
        calls.append(url)
        if url == script.SIVB_SEC_MARKET_URL:
            return _sec_market_fixture()
        assert url.startswith(
            f"https://eodhd.com/api/eod/{script.SIVBQ_PROVIDER_SYMBOL}?"
        )
        assert "api_token=fake-secret" in url
        return _eod_fixture()

    budget = _budget(tmp_path)
    first = script.collect_sivb_transition_evidence(
        tmp_path,
        sec_user_agent="Researcher researcher@example.com",
        eodhd_token="fake-secret",
        budget=budget,
        fetcher=fetcher,
    )
    second = script.collect_sivb_transition_evidence(
        tmp_path,
        sec_user_agent="Researcher researcher@example.com",
        eodhd_token="fake-secret",
        budget=budget,
        fetcher=lambda *_args: pytest.fail("cache replay attempted network"),
    )

    assert len(calls) == 2
    assert calls[0] == script.SIVB_SEC_MARKET_URL
    assert calls[1].startswith("https://eodhd.com/api/eod/SIVBQ.US?")
    assert not any(script.SIVB_OCC_MEMO_URL == url for url in calls)
    assert first["http_attempts_this_run"] == 2
    assert first["eodhd_calls_this_run"] == 1
    assert first["budget_receipt"]["used_before"] == 10
    assert first["budget_receipt"]["used_after"] == 11
    assert first["budget_receipt"]["delta"] == 1
    assert second["status"] == "cache_verified"
    assert second["http_attempts_this_run"] == 0
    assert second["eodhd_calls_this_run"] == 0
    assert json.loads(budget.state_path.read_text())["used"] == 11


def test_sivb_collector_does_not_retry_failed_eodhd_request(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetcher(url: str, _headers: dict[str, str], _max_bytes: int) -> bytes:
        calls.append(url)
        if url == script.SIVB_SEC_MARKET_URL:
            return _sec_market_fixture()
        raise RuntimeError("one-shot EOD failure")

    budget = _budget(tmp_path)
    with pytest.raises(RuntimeError, match="one-shot EOD failure"):
        script.collect_sivb_transition_evidence(
            tmp_path,
            sec_user_agent="Researcher researcher@example.com",
            eodhd_token="fake-secret",
            budget=budget,
            fetcher=fetcher,
        )

    eod_calls = [url for url in calls if "/api/eod/SIVBQ.US?" in url]
    assert len(eod_calls) == 1
    assert json.loads(budget.state_path.read_text())["used"] == 11
    assert not script._sivb_transition_report_path(tmp_path).exists()


def test_sivb_collector_cache_tampering_fails_closed(tmp_path: Path) -> None:
    def fetcher(url: str, _headers: dict[str, str], _max_bytes: int) -> bytes:
        return _sec_market_fixture() if url == script.SIVB_SEC_MARKET_URL else _eod_fixture()

    result = script.collect_sivb_transition_evidence(
        tmp_path,
        sec_user_agent="Researcher researcher@example.com",
        eodhd_token="fake-secret",
        budget=_budget(tmp_path),
        fetcher=fetcher,
    )
    eod_path = script._safe_sivb_transition_payload_path(
        tmp_path, result["evidence"]["eodhd"]["filename"]
    )
    eod_path.write_bytes(eod_path.read_bytes() + b"tampered")

    with pytest.raises(script.EvidenceError, match="hash/size changed"):
        script.verify_sivb_transition_cache(tmp_path)


def test_tampered_or_incomplete_official_evidence_fails_closed() -> None:
    with pytest.raises(script.EvidenceError, match="no longer proves"):
        script._require_patterns(
            "fixture",
            "The transaction completed, but no market date is present.",
            {"market_transition": r"January 6, 2020"},
        )


def test_cli_has_explicit_mutually_exclusive_apply_option() -> None:
    parser = script.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--apply" in option_strings
    with pytest.raises(SystemExit):
        parser.parse_args(["--apply", "--fetch-sivb-evidence"])


def _transaction_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LocalDatasetRepository, object, dict[str, str]]:
    if not (REPOSITORY_ROOT / "releases/current.json").is_file():
        pytest.skip("Actual current-release cache is unavailable.")
    source_repository = LocalDatasetRepository(REPOSITORY_ROOT)
    source_release, _ = source_repository.current_release()
    assert source_release is not None
    cache_source = REPOSITORY_ROOT / script.SIVB_TRANSITION_EVIDENCE_SUBDIR
    cache_target = tmp_path / script.SIVB_TRANSITION_EVIDENCE_SUBDIR
    cache_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cache_source, cache_target)

    repository = LocalDatasetRepository(tmp_path)
    versions: dict[str, str] = {}
    frames: dict[str, pd.DataFrame] = {}
    for dataset in script.REQUIRED_DATASETS:
        session_bound = (
            source_release.completed_session
            if dataset in {"daily_price_raw", "adjustment_factors"}
            else ""
        )
        paths = source_repository.parquet_paths(
            dataset,
            source_release.dataset_versions[dataset],
            min_session=session_bound,
            max_session=session_bound,
        )
        assert paths
        columns = (
            None
            if dataset == "source_archive"
            else list(script.dataset_spec(dataset).required_columns)
        )
        available = pd.read_parquet(paths[0], columns=columns)
        if dataset == "source_archive":
            available = available.drop(
                columns=[
                    column
                    for column in script.dataset_spec(dataset).partition_columns
                    if column in available.columns
                    and column
                    not in script.dataset_spec(dataset).required_columns
                ],
                errors="ignore",
            )
        if session_bound:
            sessions = pd.to_datetime(available["session"]).dt.date.astype(str)
            available = available.loc[sessions.eq(session_bound)]
        frame = available.head(1).copy()
        assert not frame.empty
        version = f"seed-{dataset}"
        result = repository.write_frame(
            dataset,
            frame,
            completed_session=source_release.completed_session,
            incomplete_action_policy="block",
            version=version,
        )
        assert result.conflict is False
        versions[dataset] = version
        frames[dataset] = frame
    release = repository.commit_release(
        source_release.completed_session,
        versions,
        quality="degraded",
        warnings=("transaction-fixture",),
    )
    release, release_etag = repository.current_release()
    assert release is not None
    pointer_etags = {
        dataset: repository.current_pointer(dataset)[1]
        for dataset in script.REQUIRED_DATASETS
    }
    planned = {
        dataset: f"planned-{dataset}" for dataset in script.WRITE_DATASETS
    }
    candidate = {
        dataset: frames[dataset].copy(deep=True)
        for dataset in script.WRITE_DATASETS
    }
    candidate["source_archive"], additions = script._rewrite_source_archive(
        candidate["source_archive"], completed_session=release.completed_session
    )
    assert additions == 3
    lineage = script._factor_source_version(
        planned["daily_price_raw"], planned["corporate_actions"]
    )
    for field, value in {
        "source_version": lineage,
        "source_hash": lineage,
        "source": "derived",
        "calculated_at": script.REVIEWED_AT,
        "retrieved_at": script.REVIEWED_AT,
    }.items():
        candidate["adjustment_factors"][field] = value
    plan = {
        "schema": script.SCHEMA,
        "status": "ready_offline_plan",
        "write_datasets": list(script.WRITE_DATASETS),
    }
    plan["plan_sha256"] = script._canonical_json_sha256(plan)
    prepared = script.PreparedPlan(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions=planned,
        frame_hashes=script._candidate_frame_hashes(candidate),
        frames=candidate,
        plan=plan,
    )
    monkeypatch.setattr(
        script,
        "_verify_official_evidence",
        lambda *_args, **_kwargs: ({}, {}),
    )
    monkeypatch.setattr(script, "prepare_plan", lambda _repository: prepared)

    def assert_applied(repo: LocalDatasetRepository, committed) -> None:
        current, _ = repo.current_release()
        assert current is not None and current.to_bytes() == committed.to_bytes()
        for dataset, version in committed.dataset_versions.items():
            pointer, _ = repo.current_pointer(dataset)
            assert pointer is not None and pointer.version == version
        transition = script._verify_reviewed_sivb_transition_cache(repo.root)
        script._verify_persisted_evidence(
            repo, transition, completed_session=committed.completed_session
        )

    monkeypatch.setattr(script, "_assert_applied_release", assert_applied)
    return repository, prepared, versions


def _transaction_journals(root: Path) -> list[dict]:
    return [
        json.loads(path.read_text())
        for path in sorted((root / script.TRANSACTION_DIR).glob("*.json"))
    ]


def _already_repaired_prepared(
    repository: LocalDatasetRepository,
) -> object:
    release, release_etag = repository.current_release()
    assert release is not None
    plan = {
        "schema": script.SCHEMA,
        "status": "already_repaired",
        "write_datasets": list(script.WRITE_DATASETS),
    }
    plan["plan_sha256"] = script._canonical_json_sha256(plan)
    return script.PreparedPlan(
        release=release,
        release_etag=release_etag,
        pointer_etags={
            dataset: repository.current_pointer(dataset)[1]
            for dataset in script.REQUIRED_DATASETS
        },
        planned_versions={},
        frame_hashes={},
        frames={},
        plan=plan,
    )


def test_transaction_apply_writes_exact_scope_and_replay_is_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, prepared, old_versions = _transaction_fixture(tmp_path, monkeypatch)
    old_index = {
        dataset: repository.objects.get(repository.current_key(dataset)).data
        for dataset in ("index_constituent_anchors", "index_membership_events")
    }
    result = script.apply_repair(repository, prepared)

    assert result["status"] == "applied"
    assert result["writes_performed"] is True
    release, release_etag = repository.current_release()
    assert release is not None
    assert {
        dataset
        for dataset in script.REQUIRED_DATASETS
        if release.dataset_versions[dataset] != old_versions[dataset]
    } == set(script.WRITE_DATASETS)
    assert all(
        repository.objects.get(repository.current_key(dataset)).data == value
        for dataset, value in old_index.items()
    )
    assert _transaction_journals(tmp_path)[0]["status"] == "committed"
    for entry in script.evidence_inventory():
        path = script._safe_repository_path(
            tmp_path,
            script._evidence_object_path(entry, release.completed_session),
        )
        assert path.is_file()

    replay = _already_repaired_prepared(repository)
    fresh_calls = 0

    def fresh_plan(_repository):
        nonlocal fresh_calls
        fresh_calls += 1
        return replay

    monkeypatch.setattr(script, "prepare_plan", fresh_plan)
    controls = _control_hashes(repository)
    replay_result = script.apply_repair(repository, replay)
    assert replay_result["writes_performed"] is False
    assert replay_result["status"] == "already_repaired"
    assert fresh_calls == 1
    assert _control_hashes(repository) == controls


@pytest.mark.parametrize(
    "stage",
    [
        "after_journal",
        "after_evidence_write",
        "after_write:corporate_actions",
        "after_release_commit",
    ],
)
def test_transaction_failure_injection_restores_release_and_all_pointers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    repository, prepared, _ = _transaction_fixture(tmp_path, monkeypatch)
    controls = _control_hashes(repository)

    def inject(observed: str) -> None:
        if observed == stage:
            raise RuntimeError(f"injected:{stage}")

    with pytest.raises(RuntimeError, match="injected"):
        script.apply_repair(repository, prepared, inject_failure=inject)

    assert _control_hashes(repository) == controls
    journals = _transaction_journals(tmp_path)
    assert len(journals) == 1
    assert journals[0]["status"] == "rolled_back"
    assert journals[0]["rollback_errors"] == []
    assert not tuple((tmp_path / script.RECOVERY_DIR).glob("*.json"))


@pytest.mark.parametrize("target", ["release", "corporate_actions"])
def test_transaction_cas_rejects_stale_release_or_pointer_before_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    repository, prepared, _ = _transaction_fixture(tmp_path, monkeypatch)
    if target == "release":
        value = repository.objects.get("releases/current.json")
        raw = json.loads(value.data)
        raw["created_at"] = "2099-01-01T00:00:00Z"
        repository.objects.put(
            "releases/current.json",
            (json.dumps(raw, sort_keys=True, indent=2) + "\n").encode(),
            if_match=value.etag,
        )
    else:
        key = repository.current_key(target)
        value = repository.objects.get(key)
        raw = json.loads(value.data)
        raw["updated_at"] = "2099-01-01T00:00:00Z"
        repository.objects.put(
            key,
            (json.dumps(raw, sort_keys=True, indent=2) + "\n").encode(),
            if_match=value.etag,
        )

    with pytest.raises(RuntimeError, match="changed after SIVB/AVP planning"):
        script.apply_repair(repository, prepared)
    assert _transaction_journals(tmp_path) == []


def test_already_repaired_caller_cannot_bypass_locked_canonical_ready_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, canonical_ready, _ = _transaction_fixture(tmp_path, monkeypatch)
    forged_noop = _already_repaired_prepared(repository)
    fresh_calls = 0

    def fresh_plan(_repository):
        nonlocal fresh_calls
        fresh_calls += 1
        return canonical_ready

    monkeypatch.setattr(script, "prepare_plan", fresh_plan)
    controls = _control_hashes(repository)

    with pytest.raises(RuntimeError, match="candidate frames are incomplete"):
        script.apply_repair(repository, forged_noop)
    assert fresh_calls == 1
    assert _control_hashes(repository) == controls
    assert _transaction_journals(tmp_path) == []


@pytest.mark.parametrize("target", ["release", "corporate_actions"])
def test_stale_already_repaired_caller_cannot_bypass_base_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    repository, _prepared, _ = _transaction_fixture(tmp_path, monkeypatch)
    stale_noop = _already_repaired_prepared(repository)
    if target == "release":
        key = "releases/current.json"
        value = repository.objects.get(key)
        raw = json.loads(value.data)
        raw["created_at"] = "2099-01-01T00:00:00Z"
    else:
        key = repository.current_key(target)
        value = repository.objects.get(key)
        raw = json.loads(value.data)
        raw["updated_at"] = "2099-01-01T00:00:00Z"
    repository.objects.put(
        key,
        (json.dumps(raw, sort_keys=True, indent=2) + "\n").encode(),
        if_match=value.etag,
    )
    monkeypatch.setattr(
        script,
        "prepare_plan",
        lambda _repository: pytest.fail("stale no-op reached locked re-plan"),
    )

    with pytest.raises(RuntimeError, match="changed after SIVB/AVP planning"):
        script.apply_repair(repository, stale_noop)
    assert _transaction_journals(tmp_path) == []


def test_already_repaired_noop_still_requires_writer_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _prepared, _ = _transaction_fixture(tmp_path, monkeypatch)
    canonical_noop = _already_repaired_prepared(repository)
    monkeypatch.setattr(script, "prepare_plan", lambda _repository: canonical_noop)
    lock_path = tmp_path / ".locks/market-store-write.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        script.fcntl.flock(handle.fileno(), script.fcntl.LOCK_EX | script.fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="writer lock is already held"):
            script.apply_repair(repository, canonical_noop)
    assert _transaction_journals(tmp_path) == []


def test_transaction_rejects_mutated_caller_frame_before_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, prepared, _ = _transaction_fixture(tmp_path, monkeypatch)
    prepared.frames["daily_price_raw"].loc[
        prepared.frames["daily_price_raw"].index[0], "close"
    ] = float(prepared.frames["daily_price_raw"].iloc[0]["close"]) + 1.0

    with pytest.raises(RuntimeError, match="candidate content changed"):
        script.apply_repair(repository, prepared)
    assert _transaction_journals(tmp_path) == []


def test_transaction_rejects_locked_replan_semantic_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, prepared, _ = _transaction_fixture(tmp_path, monkeypatch)
    canonical_frames = {
        dataset: frame.copy(deep=True)
        for dataset, frame in prepared.frames.items()
    }
    canonical_frames["daily_price_raw"].loc[
        canonical_frames["daily_price_raw"].index[0], "close"
    ] = float(canonical_frames["daily_price_raw"].iloc[0]["close"]) + 1.0
    canonical = script.PreparedPlan(
        release=prepared.release,
        release_etag=prepared.release_etag,
        pointer_etags=prepared.pointer_etags,
        planned_versions=prepared.planned_versions,
        frame_hashes=script._candidate_frame_hashes(canonical_frames),
        frames=canonical_frames,
        plan=prepared.plan,
    )
    monkeypatch.setattr(script, "prepare_plan", lambda _repository: canonical)

    with pytest.raises(RuntimeError, match="Locked SIVB/AVP re-plan differs"):
        script.apply_repair(repository, prepared)
    assert _transaction_journals(tmp_path) == []


def test_transaction_lock_contention_fails_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, prepared, _ = _transaction_fixture(tmp_path, monkeypatch)
    lock_path = tmp_path / ".locks/market-store-write.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        script.fcntl.flock(handle.fileno(), script.fcntl.LOCK_EX | script.fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="writer lock is already held"):
            script.apply_repair(repository, prepared)
    assert _transaction_journals(tmp_path) == []


def test_transaction_tampered_review_cache_fails_before_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, prepared, _ = _transaction_fixture(tmp_path, monkeypatch)
    report = script._verify_reviewed_sivb_transition_cache(tmp_path)
    eod_path = script._safe_sivb_transition_payload_path(
        tmp_path, report["evidence"]["eodhd"]["filename"]
    )
    eod_path.write_bytes(eod_path.read_bytes() + b"tampered")

    with pytest.raises(script.EvidenceError, match="hash/size changed"):
        script.apply_repair(repository, prepared)
    assert _transaction_journals(tmp_path) == []


def test_transaction_conflicting_immutable_evidence_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, prepared, _ = _transaction_fixture(tmp_path, monkeypatch)
    controls = _control_hashes(repository)
    entry = script.evidence_inventory()[0]
    path = script._safe_repository_path(
        tmp_path,
        script._evidence_object_path(entry, prepared.release.completed_session),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(script.gzip.compress(b"conflict", mtime=0))

    with pytest.raises(script.EvidenceError, match="conflicts with cache"):
        script.apply_repair(repository, prepared)
    assert _control_hashes(repository) == controls
    assert _transaction_journals(tmp_path)[0]["status"] == "rolled_back"
