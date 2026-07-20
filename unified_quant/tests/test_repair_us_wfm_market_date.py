from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.repository import LocalDatasetRepository


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_wfm_market_date.py"
)
SPEC = importlib.util.spec_from_file_location("repair_us_wfm_market_date", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


COMPLETED_SESSION = "2026-07-15"
OTHER_ID = "US:FIXTURE:OTHER"


def _source(source: str = "fixture", source_hash: str = "fixture-hash") -> dict:
    return {
        "source": source,
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": source_hash,
    }


@pytest.fixture
def official_evidence(monkeypatch: pytest.MonkeyPatch) -> bytes:
    payload = (
        "On August 28, 2017, Amazon.com, Inc. completed its previously "
        "announced acquisition. The acquisition subsidiary merged with and "
        "into Whole Foods Market on August 28, 2017. Each share was converted "
        "into the right to receive $42.00 in cash. NASDAQ will suspend trading "
        "of Whole Foods Market Shares prior to market open on August 28, 2017. "
        "On August 23, 2017, the Company held a special meeting of shareholders."
    ).encode()
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(script, "OFFICIAL_SOURCE_HASH", digest)
    monkeypatch.setattr(script, "OFFICIAL_SOURCE_BYTES", len(payload))
    return payload


def _frames(root: Path, payload: bytes) -> dict[str, pd.DataFrame]:
    archive_path = root / script._archive_path(COMPLETED_SESSION)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(gzip.compress(payload, mtime=0))

    master = pd.DataFrame(
        [
            {
                "security_id": script.WFM_SECURITY_ID,
                "primary_symbol": script.WFM_SYMBOL,
                "name": "Whole Foods Market Inc",
                "exchange": "NASDAQ",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "active_from": "2015-01-02",
                "active_to": script.WFM_LAST_SESSION,
                **_source(),
            },
            {
                "security_id": OTHER_ID,
                "primary_symbol": "OTHER",
                "name": "Other Inc",
                "exchange": "NYSE",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "active_from": COMPLETED_SESSION,
                "active_to": "",
                **_source(),
            },
        ]
    )
    history = pd.DataFrame(
        [
            {
                "security_id": script.WFM_SECURITY_ID,
                "symbol": script.WFM_SYMBOL,
                "exchange": "NASDAQ",
                "effective_from": "2015-01-01",
                "effective_to": "",
                "source": script.OLD_HISTORY_SOURCE,
                "source_url": script.OLD_HISTORY_URL,
                "retrieved_at": script.OLD_HISTORY_RETRIEVED_AT,
                "source_hash": script.OLD_HISTORY_HASH,
            },
            {
                "security_id": OTHER_ID,
                "symbol": "OTHER",
                "exchange": "NYSE",
                "effective_from": COMPLETED_SESSION,
                "effective_to": "",
                "source_url": "https://example.test/other",
                **_source(),
            },
        ]
    )
    prices = pd.DataFrame(
        [
            {
                "security_id": script.WFM_SECURITY_ID,
                "session": "2017-08-24",
                "open": 41.90,
                "high": 42.00,
                "low": 41.80,
                "close": 41.98,
                "volume": 1_000,
                "currency": "USD",
                **_source(),
            },
            {
                "security_id": script.WFM_SECURITY_ID,
                "session": script.WFM_LAST_SESSION,
                "open": 41.99,
                "high": 42.00,
                "low": 41.98,
                "close": 41.99,
                "volume": 2_000,
                "currency": "USD",
                **_source(),
            },
            {
                "security_id": OTHER_ID,
                "session": COMPLETED_SESSION,
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": 10.0,
                "volume": 100,
                "currency": "USD",
                **_source(),
            },
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "event_id": script.OLD_EVENT_ID,
                "security_id": script.WFM_SECURITY_ID,
                "action_type": "cash_merger",
                "effective_date": script.WRONG_PARSED_DATE,
                "ex_date": script.WRONG_PARSED_DATE,
                "announcement_date": script.CORRECTED_MARKET_DATE,
                "record_date": "",
                "payment_date": "",
                "cash_amount": script.CASH_PER_SHARE,
                "ratio": None,
                "currency": "USD",
                "new_security_id": "",
                "new_symbol": "",
                "official": True,
                "source_url": script.OFFICIAL_SOURCE_URL,
                "source_kind": script.ACTION_SOURCE_KIND,
                "source": script.ACTION_SOURCE,
                "retrieved_at": script.OFFICIAL_RETRIEVED_AT,
                "source_hash": script.OFFICIAL_SOURCE_HASH,
                "metadata": "",
            }
        ]
    )
    resolutions = pd.DataFrame(
        [
            {
                "candidate_id": script.WFM_CANDIDATE_ID,
                "security_id": script.WFM_SECURITY_ID,
                "symbol": script.WFM_SYMBOL,
                "last_price_date": script.WFM_LAST_SESSION,
                "resolution": "applied",
                "event_id": script.OLD_EVENT_ID,
                "exception_code": "",
                "exception_reason": "",
                "reviewed_by": script.RESOLUTION_REVIEWED_BY,
                "reviewed_at": script.RESOLUTION_REVIEWED_AT,
                "recheck_after": "",
                "successor_security_id": "",
                "successor_symbol": "",
                "source_url": script.OFFICIAL_SOURCE_URL,
                "source": script.RESOLUTION_SOURCE,
                "retrieved_at": script.OFFICIAL_RETRIEVED_AT,
                "source_hash": script.OFFICIAL_SOURCE_HASH,
            }
        ]
    )
    membership = pd.DataFrame(
        [
            {
                "event_id": "wfm-sp500-remove",
                "index_id": "sp500",
                "announcement_date": "2017-08-24",
                "effective_date": script.CORRECTED_MARKET_DATE,
                "operation": "REMOVE",
                "security_id": script.WFM_SECURITY_ID,
                "official": True,
                "source_url": "https://example.test/sp500",
                "source_kind": "official_primary",
                **_source(),
            }
        ]
    )
    anchors = pd.DataFrame(
        [
            {
                "index_id": "sp500",
                "anchor_date": script.WFM_LAST_SESSION,
                "security_id": script.WFM_SECURITY_ID,
                "official": True,
                "source_url": "https://example.test/sp500",
                "source_kind": "official_primary",
                **_source(),
            }
        ]
    )
    archive = pd.DataFrame(
        [
            {
                "archive_id": script.OFFICIAL_SOURCE_HASH,
                "dataset": script.ARCHIVE_DATASET,
                "object_path": script._archive_path(COMPLETED_SESSION),
                "content_type": script.ARCHIVE_CONTENT_TYPE,
                "effective_date": COMPLETED_SESSION,
                "source": script.ARCHIVE_DATASET,
                "retrieved_at": script.OFFICIAL_RETRIEVED_AT,
                "source_hash": script.OFFICIAL_SOURCE_HASH,
                "source_url": script.OFFICIAL_SOURCE_URL,
            }
        ]
    )
    factors = build_adjustment_factors(prices, actions, source_version="old-lineage")
    return {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "lifecycle_resolutions": resolutions,
        "adjustment_factors": factors,
        "source_archive": archive,
        "index_membership_events": membership,
        "index_constituent_anchors": anchors,
    }


def _repository(root: Path, payload: bytes) -> LocalDatasetRepository:
    repository = LocalDatasetRepository(root)
    versions: dict[str, str] = {}
    for dataset, frame in _frames(root, payload).items():
        result = repository.write_frame(
            dataset,
            frame,
            completed_session=COMPLETED_SESSION,
            incomplete_action_policy="block",
            version=f"fixture-{dataset}",
        )
        assert not result.conflict
        versions[dataset] = result.manifest.version
    repository.commit_release(
        COMPLETED_SESSION,
        versions,
        quality="degraded",
        warnings=("fixture-warning",),
    )
    return repository


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_plan_is_read_only_and_changes_only_reviewed_cells(
    tmp_path: Path, official_evidence: bytes
) -> None:
    repository = _repository(tmp_path, official_evidence)
    before_tree = _tree_hashes(tmp_path)
    release, _ = repository.current_release()
    assert release is not None
    old_actions = repository.read_frame(
        "corporate_actions", release.dataset_versions["corporate_actions"]
    )
    old_resolutions = repository.read_frame(
        "lifecycle_resolutions", release.dataset_versions["lifecycle_resolutions"]
    )
    old_history = repository.read_frame(
        "symbol_history", release.dataset_versions["symbol_history"]
    )
    old_factors = repository.read_frame(
        "adjustment_factors", release.dataset_versions["adjustment_factors"]
    )

    prepared = script.prepare_repair(repository)

    assert prepared.summary["status"] == "validated_offline_plan"
    assert prepared.summary["network_accessed"] is False
    assert prepared.summary["eodhd_calls"] == 0
    assert prepared.summary["r2_accessed"] is False
    assert _tree_hashes(tmp_path) == before_tree

    action = prepared.frames["corporate_actions"].iloc[0]
    assert action["event_id"] == script.NEW_EVENT_ID
    assert script._date(action["effective_date"]) == script.CORRECTED_MARKET_DATE
    assert script._date(action["ex_date"]) == script.CORRECTED_MARKET_DATE
    assert script._date(action["announcement_date"]) == script.CORRECTED_MARKET_DATE
    assert script._date(action["payment_date"]) == ""
    changed_action_columns = {
        column
        for column in old_actions.columns
        if not old_actions[column].equals(prepared.frames["corporate_actions"][column])
    }
    assert changed_action_columns == {"event_id", "effective_date", "ex_date"}

    changed_resolution_columns = {
        column
        for column in old_resolutions.columns
        if not old_resolutions[column].equals(
            prepared.frames["lifecycle_resolutions"][column]
        )
    }
    assert changed_resolution_columns == {"event_id"}
    changed_history_columns = {
        column
        for column in old_history.columns
        if not old_history[column].equals(prepared.frames["symbol_history"][column])
    }
    assert changed_history_columns == {
        "effective_to",
        "source",
        "source_url",
        "retrieved_at",
        "source_hash",
    }

    new_factors = prepared.frames["adjustment_factors"]
    keys = ["security_id", "session"]
    values = ["split_factor", "total_return_factor"]
    pd.testing.assert_frame_equal(
        old_factors[keys + values].sort_values(keys).reset_index(drop=True),
        new_factors[keys + values].sort_values(keys).reset_index(drop=True),
        check_dtype=False,
    )
    assert set(new_factors["source_version"].astype(str)) == {
        prepared.summary["factor_source_version"]
    }


def test_archive_payload_tamper_fails_closed(
    tmp_path: Path, official_evidence: bytes
) -> None:
    repository = _repository(tmp_path, official_evidence)
    path = tmp_path / script._archive_path(COMPLETED_SESSION)
    path.write_bytes(gzip.compress(official_evidence + b"tampered", mtime=0))
    with pytest.raises(ValueError, match="hash/size changed"):
        script.prepare_repair(repository)


@pytest.mark.parametrize(
    ("dataset", "column", "value", "message"),
    [
        ("security_master", "active_to", "2017-08-24", "terminal boundary"),
        ("daily_price_raw", "session", "2017-08-23", "last price session"),
        (
            "index_membership_events",
            "effective_date",
            "2017-08-29",
            "S&P 500 removal",
        ),
    ],
)
def test_terminal_boundaries_fail_closed(
    tmp_path: Path,
    official_evidence: bytes,
    dataset: str,
    column: str,
    value: str,
    message: str,
) -> None:
    repository = _repository(tmp_path, official_evidence)
    release, release_etag = repository.current_release()
    assert release is not None
    frame = repository.read_frame(dataset, release.dataset_versions[dataset])
    if dataset == "daily_price_raw":
        mask = frame["security_id"].astype(str).eq(
            script.WFM_SECURITY_ID
        ) & pd.to_datetime(frame["session"]).dt.date.astype(str).eq(
            script.WFM_LAST_SESSION
        )
        frame.loc[mask, column] = value
    else:
        frame.loc[0, column] = value
    _, pointer_etag = repository.current_pointer(dataset)
    result = repository.write_frame(
        dataset,
        frame,
        completed_session=COMPLETED_SESSION,
        incomplete_action_policy="block",
        expected_pointer_etag=pointer_etag,
        version=f"tampered-{dataset}",
    )
    versions = dict(release.dataset_versions)
    versions[dataset] = result.manifest.version
    repository.commit_release(
        COMPLETED_SESSION,
        versions,
        quality=release.quality,
        warnings=release.warnings,
        expected_etag=release_etag,
    )
    with pytest.raises(ValueError, match=message):
        script.prepare_repair(repository)


def test_partial_state_is_rejected(tmp_path: Path, official_evidence: bytes) -> None:
    repository = _repository(tmp_path, official_evidence)
    release, release_etag = repository.current_release()
    assert release is not None
    actions = repository.read_frame(
        "corporate_actions", release.dataset_versions["corporate_actions"]
    )
    actions.loc[0, ["event_id", "effective_date", "ex_date"]] = [
        script.NEW_EVENT_ID,
        script.CORRECTED_MARKET_DATE,
        script.CORRECTED_MARKET_DATE,
    ]
    _, pointer_etag = repository.current_pointer("corporate_actions")
    result = repository.write_frame(
        "corporate_actions",
        actions,
        completed_session=COMPLETED_SESSION,
        incomplete_action_policy="block",
        expected_pointer_etag=pointer_etag,
        version="partial-corporate-actions",
    )
    versions = dict(release.dataset_versions)
    versions["corporate_actions"] = result.manifest.version
    repository.commit_release(
        COMPLETED_SESSION,
        versions,
        quality=release.quality,
        warnings=release.warnings,
        expected_etag=release_etag,
    )
    with pytest.raises(RuntimeError, match="partially applied"):
        script.prepare_repair(repository)


def test_temp_apply_is_atomic_and_replay_is_idempotent(
    tmp_path: Path, official_evidence: bytes
) -> None:
    repository = _repository(tmp_path, official_evidence)
    old_release, _ = repository.current_release()
    assert old_release is not None
    prepared = script.prepare_repair(repository)

    result = script.apply_repair(repository, prepared)

    assert result["status"] == "applied"
    assert result["writes_performed"] is True
    new_release, _ = repository.current_release()
    assert new_release is not None
    assert new_release.quality == old_release.quality
    assert new_release.warnings == old_release.warnings
    assert {
        dataset
        for dataset, version in new_release.dataset_versions.items()
        if version != old_release.dataset_versions[dataset]
    } == set(script.WRITE_DATASETS)
    replay = script.prepare_repair(repository)
    assert replay.summary["status"] == "already_repaired"
    replay_result = script.apply_repair(repository, replay)
    assert replay_result["writes_performed"] is False


@pytest.mark.parametrize(
    "stage",
    [
        *(f"after_write:{dataset}" for dataset in script.WRITE_DATASETS),
        "after_release_commit",
    ],
)
def test_failure_injection_restores_release_and_pointers(
    tmp_path: Path, official_evidence: bytes, stage: str
) -> None:
    repository = _repository(tmp_path, official_evidence)
    before_release = repository.objects.get("releases/current.json").data
    before_pointers = {
        dataset: repository.objects.get(repository.current_key(dataset)).data
        for dataset in script.WRITE_DATASETS
    }
    prepared = script.prepare_repair(repository)

    def fail(observed: str) -> None:
        if observed == stage:
            raise RuntimeError(f"injected at {stage}")

    with pytest.raises(RuntimeError, match="injected at"):
        script.apply_repair(repository, prepared, inject_failure=fail)
    assert repository.objects.get("releases/current.json").data == before_release
    for dataset, value in before_pointers.items():
        assert repository.objects.get(repository.current_key(dataset)).data == value
    journals = tuple((tmp_path / script.TRANSACTION_DIR).glob("*.json"))
    assert len(journals) == 1
    assert json.loads(journals[0].read_text())["status"] == "rolled_back"


def test_stale_release_cas_blocks_apply(
    tmp_path: Path, official_evidence: bytes
) -> None:
    repository = _repository(tmp_path, official_evidence)
    prepared = script.prepare_repair(repository)
    release, etag = repository.current_release()
    assert release is not None
    repository.commit_release(
        COMPLETED_SESSION,
        release.dataset_versions,
        quality=release.quality,
        warnings=release.warnings,
        expected_etag=etag,
    )
    with pytest.raises(RuntimeError, match="Current release changed"):
        script.apply_repair(repository, prepared)
