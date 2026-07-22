from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Callable

import pandas as pd
import pytest

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.repository import LocalDatasetRepository


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_krft_yhoo_market_transitions.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_krft_yhoo_market_transitions", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)

CROSS_VALIDATION_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_us_lifecycle_cross_sources.py"
)
CROSS_SPEC = importlib.util.spec_from_file_location(
    "validate_us_lifecycle_cross_sources_for_krft_yhoo_test",
    CROSS_VALIDATION_SCRIPT_PATH,
)
assert CROSS_SPEC is not None and CROSS_SPEC.loader is not None
cross_script = importlib.util.module_from_spec(CROSS_SPEC)
sys.modules[CROSS_SPEC.name] = cross_script
CROSS_SPEC.loader.exec_module(cross_script)


COMPLETED_SESSION = "2026-07-15"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2] / "data" / "cache"


def _source(
    source: str = "fixture", source_hash: str = "fixture-hash", **extra: object
) -> dict[str, object]:
    return {
        "source": source,
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": source_hash,
        **extra,
    }


@pytest.fixture
def evidence(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    krft = (
        "On July 2, 2015, Kraft became a wholly owned subsidiary. "
        "Kraft Common Stock ceased trading on, and was delisted from, NASDAQ. "
        "Shares under KHC will begin trading on July 6, 2015. "
        "Kraft declared a special cash dividend in the amount of $16.50 per share."
    ).encode()
    yhoo = (
        "On June 13, 2017, Yahoo completed the sale of its operating business. "
        "On June 16, 2017, Yahoo changed its name to Altaba Inc. "
        "Altaba shares will trade under AABA as of the open of trading on "
        "June 19, 2017. Previously, through June 16, 2017, Yahoo common stock "
        "traded under YHOO. No action is required to be taken by stockholders."
    ).encode()
    monkeypatch.setattr(script, "KRFT_SOURCE_HASH", hashlib.sha256(krft).hexdigest())
    monkeypatch.setattr(script, "KRFT_SOURCE_BYTES", len(krft))
    monkeypatch.setattr(script, "YHOO_SOURCE_HASH", hashlib.sha256(yhoo).hexdigest())
    monkeypatch.setattr(script, "YHOO_SOURCE_BYTES", len(yhoo))
    monkeypatch.setattr(
        script,
        "EXPECTED_PRICE_BOUNDARIES",
        {
            script.KRFT_ID: (4, "2015-01-02", script.KRFT_LAST_SESSION, 88.0, 88.19),
            script.KHC_ID: (
                2,
                script.KHC_FIRST_SESSION,
                COMPLETED_SESSION,
                71.0,
                72.96,
            ),
            script.YHOO_ID: (
                3,
                "2015-01-02",
                script.YHOO_LAST_SESSION,
                52.79,
                52.58,
            ),
            script.AABA_ID: (
                4,
                "2015-01-02",
                script.AABA_LAST_SESSION,
                19.56,
                19.63,
            ),
        },
    )
    return {"krft": krft, "yhoo": yhoo}


def _master_row(
    security_id: str,
    symbol: str,
    name: str,
    active_from: str,
    active_to: str,
    *,
    source: str,
    source_url: str,
    retrieved_at: str,
    source_hash: str,
) -> dict[str, object]:
    provider_symbol = f"{symbol}.US"
    return {
        "security_id": security_id,
        "exchange": "NASDAQ",
        "active_from": active_from,
        "active_to": active_to,
        "source": source,
        "source_url": source_url,
        "retrieved_at": retrieved_at,
        "source_hash": source_hash,
        "primary_symbol": symbol,
        "provider_symbol": provider_symbol,
        "name": name,
        "asset_type": "STOCK",
        "currency": "USD",
        "country": "US",
        "action_provider_symbol": provider_symbol,
        "isin": "",
    }


def _history_row(
    security_id: str,
    symbol: str,
    start: str,
    end: str,
    *,
    source: str,
    source_url: str,
    retrieved_at: str,
    source_hash: str,
) -> dict[str, object]:
    return {
        "security_id": security_id,
        "symbol": symbol,
        "exchange": "NASDAQ",
        "effective_from": start,
        "effective_to": end,
        "source": source,
        "source_url": source_url,
        "retrieved_at": retrieved_at,
        "source_hash": source_hash,
    }


def _price(
    security_id: str,
    session: str,
    open_: float,
    close: float,
) -> dict[str, object]:
    return {
        "security_id": security_id,
        "session": session,
        "open": open_,
        "high": max(open_, close) + 1.0,
        "low": min(open_, close) - 1.0,
        "close": close,
        "volume": 1_000.0,
        "currency": "USD",
        "source_url": "https://example.test/eod",
        **_source("fixture_eod", "fixture-eod-hash"),
    }


def _action_common() -> dict[str, object]:
    return {
        "record_date": "",
        "payment_date": "",
        "cash_amount": None,
        "currency": "USD",
        "official": True,
        "source_kind": script.ACTION_SOURCE_KIND,
        "source": script.ACTION_SOURCE,
        "metadata": "",
    }


def _special_dividend() -> dict[str, object]:
    return {
        "event_id": script.KRFT_SPECIAL_DIVIDEND_EVENT_ID,
        "security_id": script.KRFT_ID,
        "action_type": "special_dividend",
        "effective_date": script.KRFT_LEGAL_COMPLETION,
        "ex_date": script.KRFT_LEGAL_COMPLETION,
        "announcement_date": "2015-06-22",
        "record_date": script.KRFT_LEGAL_COMPLETION,
        "payment_date": script.KRFT_LEGAL_COMPLETION,
        "cash_amount": 16.5,
        "ratio": None,
        "currency": "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": True,
        "source_url": script.SPECIAL_DIVIDEND_SOURCE_URL,
        "source_kind": script.ACTION_SOURCE_KIND,
        "source": "sec_edgar+reviewed_special_dividend",
        "retrieved_at": script.SPECIAL_DIVIDEND_RETRIEVED_AT,
        "source_hash": script.SPECIAL_DIVIDEND_SOURCE_HASH,
        "metadata": script._special_dividend_metadata(),
    }


def _resolution(
    candidate_id: str,
    security_id: str,
    symbol: str,
    last_price: str,
    event_id: str,
    successor_id: str,
    successor_symbol: str,
    source_url: str,
    retrieved_at: str,
    source_hash: str,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "security_id": security_id,
        "symbol": symbol,
        "last_price_date": last_price,
        "resolution": "applied",
        "event_id": event_id,
        "exception_code": "",
        "exception_reason": "",
        "reviewed_by": script.RESOLUTION_REVIEWED_BY,
        "reviewed_at": script.RESOLUTION_REVIEWED_AT,
        "recheck_after": "",
        "successor_security_id": successor_id,
        "successor_symbol": successor_symbol,
        "source_url": source_url,
        "source": script.RESOLUTION_SOURCE,
        "retrieved_at": retrieved_at,
        "source_hash": source_hash,
    }


def _anchor(
    index_id: str, anchor_date: str, security_id: str
) -> dict[str, object]:
    sp = index_id == "sp500"
    return {
        "index_id": index_id,
        "anchor_date": anchor_date,
        "security_id": security_id,
        "official": False,
        "source": script.SP_SOURCE if sp else script.NASDAQ_SOURCE,
        "source_url": script.SP_SOURCE_URL if sp else script.NASDAQ_SOURCE_URL,
        "source_kind": "community",
        "retrieved_at": (
            script.SP_ANCHOR_RETRIEVED_AT
            if sp
            else script.NASDAQ_ANCHOR_RETRIEVED_AT
        ),
        "source_hash": script.SP_SOURCE_HASH if sp else script.NASDAQ_SOURCE_HASH,
    }


def _event(
    event_id: str,
    index_id: str,
    date: str,
    operation: str,
    security_id: str,
) -> dict[str, object]:
    sp = index_id == "sp500"
    return {
        "event_id": event_id,
        "index_id": index_id,
        "announcement_date": "",
        "effective_date": date,
        "operation": operation,
        "security_id": security_id,
        "official": False,
        "source": script.SP_SOURCE if sp else script.NASDAQ_SOURCE,
        "source_url": script.SP_SOURCE_URL if sp else script.NASDAQ_SOURCE_URL,
        "source_kind": "community",
        "retrieved_at": (
            script.SP_EVENT_RETRIEVED_AT
            if sp
            else script.NASDAQ_EVENT_RETRIEVED_AT
        ),
        "source_hash": script.SP_SOURCE_HASH if sp else script.NASDAQ_SOURCE_HASH,
    }


def _frames(root: Path, evidence: dict[str, bytes]) -> dict[str, pd.DataFrame]:
    krft_archive_path = root / script._archive_path(
        COMPLETED_SESSION, script.KRFT_SOURCE_HASH
    )
    krft_archive_path.parent.mkdir(parents=True, exist_ok=True)
    krft_archive_path.write_bytes(gzip.compress(evidence["krft"], mtime=0))
    yhoo_cache = root / script.YHOO_STATE_CACHE_PATH
    yhoo_cache.parent.mkdir(parents=True, exist_ok=True)
    yhoo_cache.write_bytes(evidence["yhoo"])

    master = pd.DataFrame(
        [
            _master_row(
                script.KRFT_ID,
                script.KRFT_SYMBOL,
                "Kraft Foods Group Inc",
                "2015-01-02",
                script.KRFT_LAST_SESSION,
                source=script.CONFIRMED_IDENTITY_SOURCE,
                source_url=script.KRFT_SOURCE_URL,
                retrieved_at=script.KRFT_HISTORY_RETRIEVED_AT,
                source_hash=script.KRFT_SOURCE_HASH,
            ),
            _master_row(
                script.KHC_ID,
                script.KHC_SYMBOL,
                "Kraft Heinz Co",
                script.KHC_FIRST_SESSION,
                "",
                source=script.CONFIRMED_IDENTITY_SOURCE,
                source_url=script.KRFT_SOURCE_URL,
                retrieved_at=script.KRFT_HISTORY_RETRIEVED_AT,
                source_hash=script.KRFT_SOURCE_HASH,
            ),
            _master_row(
                script.YHOO_ID,
                script.YHOO_SYMBOL,
                "Yahoo! Inc",
                "2015-01-02",
                script.YHOO_LAST_SESSION,
                source=script.EOD_SYMBOL_SOURCE,
                source_url=script.EOD_SYMBOL_URL,
                retrieved_at=script.EOD_SYMBOL_RETRIEVED_AT,
                source_hash=script.EOD_SYMBOL_HASH,
            ),
            _master_row(
                script.AABA_ID,
                script.AABA_SYMBOL,
                "Altaba Inc",
                "2015-01-02",
                script.AABA_LAST_SESSION,
                source=script.EOD_SYMBOL_SOURCE,
                source_url=script.EOD_SYMBOL_URL,
                retrieved_at=script.EOD_SYMBOL_RETRIEVED_AT,
                source_hash=script.EOD_SYMBOL_HASH,
            ),
        ]
    )
    history = pd.DataFrame(
        [
            _history_row(
                script.KRFT_ID,
                script.KRFT_SYMBOL,
                "2015-01-01",
                script.KRFT_LAST_SESSION,
                source=script.CONFIRMED_IDENTITY_SOURCE,
                source_url=script.KRFT_SOURCE_URL,
                retrieved_at=script.KRFT_HISTORY_RETRIEVED_AT,
                source_hash=script.KRFT_SOURCE_HASH,
            ),
            _history_row(
                script.KHC_ID,
                script.KHC_SYMBOL,
                script.KRFT_LEGAL_COMPLETION,
                "",
                source=script.CONFIRMED_IDENTITY_SOURCE,
                source_url=script.KRFT_SOURCE_URL,
                retrieved_at=script.KRFT_HISTORY_RETRIEVED_AT,
                source_hash=script.KRFT_SOURCE_HASH,
            ),
            _history_row(
                script.YHOO_ID,
                script.YHOO_SYMBOL,
                "2015-01-01",
                "",
                source=script.EOD_SYMBOL_SOURCE,
                source_url=script.EOD_SYMBOL_URL,
                retrieved_at=script.EOD_SYMBOL_RETRIEVED_AT,
                source_hash=script.EOD_SYMBOL_HASH,
            ),
            _history_row(
                script.AABA_ID,
                script.AABA_SYMBOL,
                "2015-01-01",
                "",
                source=script.EOD_SYMBOL_SOURCE,
                source_url=script.EOD_SYMBOL_URL,
                retrieved_at=script.EOD_SYMBOL_RETRIEVED_AT,
                source_hash=script.EOD_SYMBOL_HASH,
            ),
        ]
    )
    prices = pd.DataFrame(
        [
            _price(script.KRFT_ID, "2015-01-02", 62.73, 62.64),
            _price(script.KRFT_ID, "2015-01-07", 63.0, 63.2),
            _price(script.KRFT_ID, "2015-07-01", 85.49, 88.30),
            _price(script.KRFT_ID, script.KRFT_LAST_SESSION, 88.0, 88.19),
            _price(script.KHC_ID, script.KHC_FIRST_SESSION, 71.0, 72.96),
            _price(script.KHC_ID, COMPLETED_SESSION, 24.97, 25.45),
            _price(script.YHOO_ID, "2015-01-02", 50.66, 50.17),
            _price(script.YHOO_ID, "2015-01-07", 49.8, 50.0),
            _price(script.YHOO_ID, script.YHOO_LAST_SESSION, 52.79, 52.58),
            _price(script.AABA_ID, "2015-01-02", 50.66, 50.17),
            _price(script.AABA_ID, script.YHOO_LAST_SESSION, 52.79, 52.58),
            _price(script.AABA_ID, script.AABA_FIRST_SESSION, 54.0, 54.46),
            _price(script.AABA_ID, script.AABA_LAST_SESSION, 19.56, 19.63),
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "event_id": script.KRFT_OLD_EVENT_ID,
                "security_id": script.KRFT_ID,
                "action_type": "stock_merger",
                "effective_date": script.KRFT_LEGAL_COMPLETION,
                "ex_date": script.KRFT_LEGAL_COMPLETION,
                "announcement_date": script.KRFT_LEGAL_COMPLETION,
                "ratio": 1.0,
                "new_security_id": script.KHC_ID,
                "new_symbol": script.KHC_SYMBOL,
                "source_url": script.KRFT_SOURCE_URL,
                "retrieved_at": script.KRFT_ACTION_RETRIEVED_AT,
                "source_hash": script.KRFT_SOURCE_HASH,
                **_action_common(),
            },
            _special_dividend(),
            {
                "event_id": script.YHOO_OLD_EVENT_ID,
                "security_id": script.YHOO_ID,
                "action_type": "ticker_change",
                "effective_date": script.YHOO_WRONG_PARSED_DATE,
                "ex_date": script.YHOO_WRONG_PARSED_DATE,
                "announcement_date": "2017-07-31",
                "ratio": None,
                "new_security_id": script.AABA_ID,
                "new_symbol": script.AABA_SYMBOL,
                "source_url": script.YHOO_OLD_SOURCE_URL,
                "retrieved_at": script.YHOO_OLD_RETRIEVED_AT,
                "source_hash": script.YHOO_OLD_SOURCE_HASH,
                **_action_common(),
            },
        ]
    )
    resolutions = pd.DataFrame(
        [
            _resolution(
                script.KRFT_CANDIDATE_ID,
                script.KRFT_ID,
                script.KRFT_SYMBOL,
                script.KRFT_LAST_SESSION,
                script.KRFT_OLD_EVENT_ID,
                script.KHC_ID,
                script.KHC_SYMBOL,
                script.KRFT_SOURCE_URL,
                script.KRFT_ACTION_RETRIEVED_AT,
                script.KRFT_SOURCE_HASH,
            ),
            _resolution(
                script.YHOO_CANDIDATE_ID,
                script.YHOO_ID,
                script.YHOO_SYMBOL,
                script.YHOO_LAST_SESSION,
                script.YHOO_OLD_EVENT_ID,
                script.AABA_ID,
                script.AABA_SYMBOL,
                script.YHOO_OLD_SOURCE_URL,
                script.YHOO_OLD_RETRIEVED_AT,
                script.YHOO_OLD_SOURCE_HASH,
            ),
        ]
    )
    anchors = pd.DataFrame(
        [
            _anchor("nasdaq100", "2015-01-01", script.YHOO_ID),
            _anchor("nasdaq100", "2015-01-01", script.KRFT_ID),
            _anchor("sp500", "2015-01-07", script.AABA_ID),
            _anchor("sp500", "2015-01-07", script.KRFT_ID),
        ]
    )
    events = pd.DataFrame(
        [
            _event(
                script.KRFT_NASDAQ_ADD_OLD_ID,
                "nasdaq100",
                script.KRFT_LEGAL_COMPLETION,
                "ADD",
                script.KHC_ID,
            ),
            _event(
                script.KRFT_NASDAQ_REMOVE_OLD_ID,
                "nasdaq100",
                script.KRFT_LEGAL_COMPLETION,
                "REMOVE",
                script.KRFT_ID,
            ),
            _event(
                script.KRFT_SP_ADD_ID,
                "sp500",
                script.KHC_FIRST_SESSION,
                "ADD",
                script.KHC_ID,
            ),
            _event(
                script.KRFT_SP_REMOVE_ID,
                "sp500",
                script.KHC_FIRST_SESSION,
                "REMOVE",
                script.KRFT_ID,
            ),
            _event(
                script.YHOO_NASDAQ_REMOVE_ID,
                "nasdaq100",
                script.AABA_FIRST_SESSION,
                "REMOVE",
                script.YHOO_ID,
            ),
            _event(
                script.YHOO_SP_REMOVE_OLD_ID,
                "sp500",
                script.AABA_FIRST_SESSION,
                "REMOVE",
                script.AABA_ID,
            ),
        ]
    )
    archive = pd.DataFrame(
        [
            script._archive_row_expected(
                completed_session=COMPLETED_SESSION,
                source_url=script.KRFT_SOURCE_URL,
                source_hash=script.KRFT_SOURCE_HASH,
                retrieved_at=script.KRFT_ACTION_RETRIEVED_AT,
            )
        ]
    )
    factors = build_adjustment_factors(prices, actions, source_version="old-lineage")
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
    mutate: Callable[[dict[str, pd.DataFrame]], None] | None = None,
) -> LocalDatasetRepository:
    frames = _frames(root, evidence)
    if mutate is not None:
        mutate(frames)
    repository = LocalDatasetRepository(root)
    versions: dict[str, str] = {}
    for dataset, frame in frames.items():
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


def _control_bytes(repository: LocalDatasetRepository) -> dict[str, bytes]:
    keys = ["releases/current.json"] + [
        repository.current_key(dataset) for dataset in script.REQUIRED_DATASETS
    ]
    return {key: repository.objects.get(key).data for key in keys}


def _current(repository: LocalDatasetRepository, dataset: str) -> pd.DataFrame:
    release, _ = repository.current_release()
    assert release is not None
    return repository.read_frame(dataset, release.dataset_versions[dataset])


def test_plan_is_read_only_and_repairs_only_reviewed_state(
    tmp_path: Path, evidence: dict[str, bytes]
) -> None:
    repository = _repository(tmp_path, evidence)
    before = _tree_hashes(tmp_path)
    old_actions = _current(repository, "corporate_actions")
    old_master = _current(repository, "security_master")
    old_factors = _current(repository, "adjustment_factors")

    prepared = script.prepare_repair(repository)

    assert _tree_hashes(tmp_path) == before
    assert prepared.summary["status"] == "validated_offline_plan"
    assert prepared.summary["writes_performed"] is False
    assert prepared.summary["network_accessed"] is False
    assert prepared.summary["eodhd_calls"] == 0
    assert prepared.summary["r2_accessed"] is False
    assert prepared.summary["corporate_action_rows_rekeyed"] == 2
    assert prepared.summary["index_membership_rows_rekeyed"] == 3
    assert prepared.summary["krft_special_dividend_preserved_exactly"] is True
    assert hasattr(script, "apply_repair")
    assert prepared.release_etag
    assert set(prepared.pointer_etags) == set(script.REQUIRED_DATASETS)

    actions = prepared.frames["corporate_actions"]
    special_before = old_actions.loc[
        old_actions["event_id"].astype(str).eq(script.KRFT_SPECIAL_DIVIDEND_EVENT_ID)
    ].reset_index(drop=True)
    special_after = actions.loc[
        actions["event_id"].astype(str).eq(script.KRFT_SPECIAL_DIVIDEND_EVENT_ID)
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(special_before, special_after, check_dtype=False)
    krft = actions.loc[actions["event_id"].astype(str).eq(script.KRFT_NEW_EVENT_ID)]
    yhoo = actions.loc[actions["event_id"].astype(str).eq(script.YHOO_NEW_EVENT_ID)]
    assert len(krft) == len(yhoo) == 1
    assert script._date(krft.iloc[0]["effective_date"]) == script.KHC_FIRST_SESSION
    assert script._date(krft.iloc[0]["announcement_date"]) == script.KRFT_LEGAL_COMPLETION
    assert script._canonical_metadata(krft.iloc[0]["metadata"]) == script._transition_metadata(
        "krft"
    )
    assert script._date(yhoo.iloc[0]["effective_date"]) == script.AABA_FIRST_SESSION
    assert script._canonical_metadata(yhoo.iloc[0]["metadata"]) == script._transition_metadata(
        "yhoo"
    )

    master = prepared.frames["security_master"]
    aaba_before = old_master.loc[old_master["security_id"].astype(str).eq(script.AABA_ID)]
    aaba_after = master.loc[master["security_id"].astype(str).eq(script.AABA_ID)]
    assert script._date(aaba_before.iloc[0]["active_from"]) == "2015-01-02"
    assert script._date(aaba_after.iloc[0]["active_from"]) == script.AABA_FIRST_SESSION
    yhoo_before = old_master.loc[old_master["security_id"].astype(str).eq(script.YHOO_ID)]
    yhoo_after = master.loc[master["security_id"].astype(str).eq(script.YHOO_ID)]
    pd.testing.assert_frame_equal(
        yhoo_before.reset_index(drop=True), yhoo_after.reset_index(drop=True), check_dtype=False
    )

    history = prepared.frames["symbol_history"]
    expected_history = {
        script.KRFT_ID: ("2015-01-01", script.KRFT_LAST_SESSION),
        script.KHC_ID: (script.KHC_FIRST_SESSION, ""),
        script.YHOO_ID: ("2015-01-01", script.YHOO_LAST_SESSION),
        script.AABA_ID: (script.AABA_FIRST_SESSION, script.AABA_LAST_SESSION),
    }
    for security_id, (start, end) in expected_history.items():
        row = history.loc[history["security_id"].astype(str).eq(security_id)].iloc[0]
        assert script._date(row["effective_from"]) == start
        assert script._date(row["effective_to"]) == end

    anchors = prepared.frames["index_constituent_anchors"]
    sp_yahoo = anchors.loc[
        anchors["index_id"].astype(str).eq("sp500")
        & anchors["anchor_date"].astype(str).eq("2015-01-07")
    ]
    assert set(sp_yahoo["security_id"].astype(str)) == {script.YHOO_ID, script.KRFT_ID}
    events = prepared.frames["index_membership_events"]
    assert {
        script.KRFT_NASDAQ_ADD_NEW_ID,
        script.KRFT_NASDAQ_REMOVE_NEW_ID,
        script.YHOO_SP_REMOVE_NEW_ID,
    }.issubset(set(events["event_id"].astype(str)))

    factors = prepared.frames["adjustment_factors"]
    economic = ["security_id", "session", "split_factor", "total_return_factor"]
    pd.testing.assert_frame_equal(
        old_factors[economic].reset_index(drop=True),
        factors[economic].reset_index(drop=True),
        check_dtype=False,
    )
    assert set(factors["source_version"].astype(str)) == {
        prepared.summary["factor_source_version"]
    }
    assert len(prepared.frames["source_archive"]) == 2
    assert script.YHOO_SOURCE_HASH in set(
        prepared.frames["source_archive"]["archive_id"].astype(str)
    )


def test_planned_frames_replay_idempotently(
    tmp_path: Path, evidence: dict[str, bytes]
) -> None:
    repository = _repository(tmp_path, evidence)
    prepared = script.prepare_repair(repository)
    script._assert_nonfactor_replay(
        prepared.frames, completed_session=COMPLETED_SESSION
    )
    factors, changed = script._rebind_factors(
        prepared.frames["adjustment_factors"],
        _current(repository, "daily_price_raw"),
        prepared.frames["corporate_actions"],
        prepared.frames["corporate_actions"],
        source_version=prepared.summary["factor_source_version"],
    )
    assert changed is False
    pd.testing.assert_frame_equal(
        prepared.frames["adjustment_factors"], factors, check_dtype=False
    )


def test_partial_krft_transition_is_fail_closed(
    tmp_path: Path, evidence: dict[str, bytes]
) -> None:
    def mutate(frames: dict[str, pd.DataFrame]) -> None:
        mask = frames["symbol_history"]["security_id"].astype(str).eq(script.KHC_ID)
        frames["symbol_history"].loc[mask, "effective_from"] = script.KHC_FIRST_SESSION

    repository = _repository(tmp_path, evidence, mutate)
    with pytest.raises(RuntimeError, match="KRFT->KHC.*partially applied"):
        script.prepare_repair(repository)


def test_tampered_yhoo_evidence_is_fail_closed(
    tmp_path: Path, evidence: dict[str, bytes]
) -> None:
    repository = _repository(tmp_path, evidence)
    (tmp_path / script.YHOO_STATE_CACHE_PATH).write_bytes(evidence["yhoo"] + b"tamper")
    with pytest.raises(ValueError, match="YHOO evidence hash/size changed"):
        script.prepare_repair(repository)


def test_special_dividend_change_is_fail_closed(
    tmp_path: Path, evidence: dict[str, bytes]
) -> None:
    def mutate(frames: dict[str, pd.DataFrame]) -> None:
        mask = frames["corporate_actions"]["event_id"].astype(str).eq(
            script.KRFT_SPECIAL_DIVIDEND_EVENT_ID
        )
        frames["corporate_actions"].loc[mask, "cash_amount"] = 16.49
        frames["adjustment_factors"] = build_adjustment_factors(
            frames["daily_price_raw"],
            frames["corporate_actions"],
            source_version="old-lineage",
        )

    repository = _repository(tmp_path, evidence, mutate)
    with pytest.raises(ValueError, match="special-dividend action changed"):
        script.prepare_repair(repository)


def test_apply_commits_one_coherent_release_and_replays_idempotently(
    tmp_path: Path, evidence: dict[str, bytes]
) -> None:
    repository = _repository(tmp_path, evidence)
    before_release, _ = repository.current_release()
    assert before_release is not None
    before_prices = _current(repository, "daily_price_raw")
    before_factors = _current(repository, "adjustment_factors")
    prepared = script.prepare_repair(repository)

    result = script.apply_repair(repository, prepared)

    after_release, _ = repository.current_release()
    assert after_release is not None
    assert result["status"] == "applied"
    assert result["writes_performed"] is True
    assert result["network_accessed"] is False
    assert result["eodhd_calls"] == 0
    assert result["r2_accessed"] is False
    assert after_release.version == result["new_release_version"]
    assert (
        after_release.dataset_versions["daily_price_raw"]
        == before_release.dataset_versions["daily_price_raw"]
    )
    for dataset in script.WRITE_DATASETS:
        assert (
            after_release.dataset_versions[dataset]
            != before_release.dataset_versions[dataset]
        )
    pd.testing.assert_frame_equal(
        before_prices.reset_index(drop=True),
        _current(repository, "daily_price_raw").reset_index(drop=True),
        check_dtype=False,
    )
    economic = ["security_id", "session", "split_factor", "total_return_factor"]
    pd.testing.assert_frame_equal(
        before_factors[economic].reset_index(drop=True),
        _current(repository, "adjustment_factors")[economic].reset_index(drop=True),
        check_dtype=False,
    )
    archived_path = tmp_path / script._archive_path(
        COMPLETED_SESSION, script.YHOO_SOURCE_HASH
    )
    assert gzip.decompress(archived_path.read_bytes()) == evidence["yhoo"]
    journal = tmp_path / script.TRANSACTION_DIR / f"{result['transaction_id']}.json"
    assert '"status": "committed"' in journal.read_text()

    replay = script.prepare_repair(repository)
    assert replay.summary["status"] == "already_repaired"
    controls = _control_bytes(repository)
    no_op = script.apply_repair(repository, replay)
    assert no_op["status"] == "already_repaired"
    assert no_op["writes_performed"] is False
    assert _control_bytes(repository) == controls


def test_apply_rolls_back_release_and_all_pointers_after_commit_failure(
    tmp_path: Path, evidence: dict[str, bytes]
) -> None:
    repository = _repository(tmp_path, evidence)
    prepared = script.prepare_repair(repository)
    before = _control_bytes(repository)

    def fail(stage: str) -> None:
        if stage == "after_release_commit":
            raise RuntimeError("injected post-commit failure")

    with pytest.raises(RuntimeError, match="injected post-commit failure"):
        script.apply_repair(repository, prepared, inject_failure=fail)

    assert _control_bytes(repository) == before
    journals = tuple((tmp_path / script.TRANSACTION_DIR).glob("*.json"))
    assert len(journals) == 1
    assert '"status": "rolled_back"' in journals[0].read_text()
    assert not (tmp_path / script.RECOVERY_DIR).exists()
    retry = script.prepare_repair(repository)
    assert retry.summary["status"] == "validated_offline_plan"
    assert set(retry.planned_versions.values()).isdisjoint(
        set(prepared.planned_versions.values())
    )


def test_apply_rejects_stale_release_before_any_transition_write(
    tmp_path: Path, evidence: dict[str, bytes]
) -> None:
    repository = _repository(tmp_path, evidence)
    prepared = script.prepare_repair(repository)
    release, etag = repository.current_release()
    assert release is not None
    repository.commit_release(
        release.completed_session,
        release.dataset_versions,
        quality=release.quality,
        warnings=release.warnings,
        expected_etag=etag,
    )
    current_before = _control_bytes(repository)

    with pytest.raises(RuntimeError, match="Current release changed"):
        script.apply_repair(repository, prepared)

    assert _control_bytes(repository) == current_before
    assert not (tmp_path / script.TRANSACTION_DIR).exists()




def test_cli_supports_explicit_apply_mode() -> None:
    args = script._parse_args(["--apply"])
    assert args.apply is True
    assert args.dry_run is False
    with pytest.raises(SystemExit):
        script._parse_args(["--dry-run", "--apply"])


def test_cli_rejects_unknown_mode() -> None:
    with pytest.raises(SystemExit):
        script._parse_args(["--write-without-cas"])
