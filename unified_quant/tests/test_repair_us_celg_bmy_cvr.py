from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from supertrend_quant.ledger import PortfolioLedger
from supertrend_quant.market_store.ingest import EodhdCallBudget, SourceArtifact
from supertrend_quant.market_store.lifecycle_coverage import lifecycle_candidate_id
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "repair_us_celg_bmy_cvr.py"
)
SPEC = importlib.util.spec_from_file_location("repair_us_celg_bmy_cvr", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)

BMY_SECURITY_ID = script.BMY_SECURITY_ID
BMY_SYMBOL = script.BMY_SYMBOL
ACTIVE_CATALOG_SHA256 = script.ACTIVE_CATALOG_SHA256
ACTIVE_CATALOG_URL = script.ACTIVE_CATALOG_URL
CELG_LAST_SESSION = script.CELG_LAST_SESSION
CELG_SECURITY_ID = script.CELG_SECURITY_ID
CELG_STOCK_MERGER_EVENT_ID = script.CELG_STOCK_MERGER_EVENT_ID
CELG_SYMBOL = script.CELG_SYMBOL
CVR_DISTRIBUTION_EVENT_ID = script.CVR_DISTRIBUTION_EVENT_ID
CVR_ECONOMIC_BASIS_FRACTION = script.CVR_ECONOMIC_BASIS_FRACTION
CVR_LAST_SESSION = script.CVR_LAST_SESSION
CVR_REFERENCE_CLOSE = script.CVR_REFERENCE_CLOSE
CVR_PROVIDER_CODE = script.CVR_PROVIDER_CODE
CVR_PROVIDER_SYMBOL = script.CVR_PROVIDER_SYMBOL
CVR_SYMBOL = script.CVR_SYMBOL
CVR_TERMINATION_DATE = script.CVR_TERMINATION_DATE
EXPECTED_CVR_SESSIONS = script.EXPECTED_CVR_SESSIONS
EXACT_CATALOG_ROW = script.EXACT_CATALOG_ROW
MERGER_SESSION = script.MERGER_SESSION
MERGER_TERMS_SHA256 = script.MERGER_TERMS_SHA256
MERGER_TERMS_URL = script.MERGER_TERMS_URL
CvrBundle = script.CvrBundle
ExactThreeEodhdClient = script.ExactThreeEodhdClient
FrozenCatalogSelection = script.FrozenCatalogSelection
_bundle_from_artifacts = script._bundle_from_artifacts
_endpoint_url = script._endpoint_url
_expected_sessions = script._expected_sessions
_official_actions = script._official_actions
_parse_args = script._parse_args
_source_artifact = script._source_artifact
_target_is_exact = script._target_is_exact
_validate_catalog_selection = script._validate_catalog_selection
fetch_exact_bundle = script.fetch_exact_bundle
prepare_frames = script.prepare_frames
read_bundle_cache = script.read_bundle_cache
validate_bundle = script.validate_bundle
write_bundle_cache = script.write_bundle_cache


def _selection() -> FrozenCatalogSelection:
    return FrozenCatalogSelection(
        provider_code=CVR_PROVIDER_CODE,
        source_url=ACTIVE_CATALOG_URL,
        source_hash=ACTIVE_CATALOG_SHA256,
        row=dict(EXACT_CATALOG_ROW),
        secondary_evidence=(
            {
                "role": "secondary_ambiguous",
                "reason": "Provider alias lacks the SEC submitter ticker and ISIN binding.",
                "source_url": script.DELISTED_CATALOG_URL,
                "source_hash": script.DELISTED_CATALOG_SHA256,
                "row": {
                    "Code": "BMY-R",
                    "Country": "USA",
                    "Currency": "USD",
                    "Exchange": "NYSE",
                    "Isin": None,
                    "Name": "BMY-R",
                    "Type": "Common Stock",
                },
            },
            {
                "role": "secondary_ambiguous",
                "reason": "Provider alias lacks the SEC submitter ticker and ISIN binding.",
                "source_url": script.DELISTED_CATALOG_URL,
                "source_hash": script.DELISTED_CATALOG_SHA256,
                "row": {
                    "Code": "BMY-RI",
                    "Country": "USA",
                    "Currency": "USD",
                    "Exchange": "NYSE",
                    "Isin": None,
                    "Name": "BMY-RI",
                    "Type": "Common Stock",
                },
            },
        ),
    )


def _raw_artifact(source: str, endpoint: str, payload, retrieved_at: str) -> SourceArtifact:
    return SourceArtifact(
        source=source,
        source_url=_endpoint_url(endpoint, CVR_PROVIDER_CODE),
        retrieved_at=retrieved_at,
        content=json.dumps(payload, ensure_ascii=False, indent=1).encode(),
        content_type="application/json; charset=utf-8",
    )


def _bundle(*, release: str = "release-v1") -> CvrBundle:
    retrieved_at = "2026-07-18T09:00:00Z"
    eod = [
        {
            "date": session,
            "open": 2.30,
            "high": 2.40,
            "low": 2.20,
            "close": CVR_REFERENCE_CLOSE,
            "volume": 1_000 + index,
        }
        for index, session in enumerate(_expected_sessions())
    ]
    artifacts = [
        _raw_artifact(f"eodhd_{endpoint}", endpoint, payload, retrieved_at)
        for endpoint, payload in (("eod", eod), ("div", []), ("splits", []))
    ]
    return _bundle_from_artifacts(
        artifacts,
        catalog_selection=_selection(),
        http_attempts=3,
        fetched_against_release=release,
        budget_used_before=20,
        budget_used_after=23,
    )


def _terms():
    return SimpleNamespace(
        source="sec_edgar_filing",
        source_url=MERGER_TERMS_URL,
        retrieved_at="2026-07-18T07:25:37Z",
        source_hash=MERGER_TERMS_SHA256,
    )


def _termination() -> SourceArtifact:
    return SourceArtifact(
        source="sec_bmy_2020_10k",
        source_url="https://www.sec.gov/Archives/edgar/data/14272/bmy-20201231.htm",
        retrieved_at="2026-07-18T08:39:00Z",
        content=b"reviewed termination evidence",
        content_type="text/html",
    )


def _empty(dataset: str, *, extras: tuple[str, ...] = ()) -> pd.DataFrame:
    columns = tuple(dataset_spec(dataset).required_columns) + extras
    return pd.DataFrame(columns=tuple(dict.fromkeys(columns)))


def _base_frames() -> dict[str, pd.DataFrame]:
    master = pd.DataFrame(
        [
            {
                "security_id": CELG_SECURITY_ID,
                "primary_symbol": CELG_SYMBOL,
                "provider_symbol": "CELG.US",
                "action_provider_symbol": "CELG.US",
                "name": "Celgene Corporation",
                "exchange": "NASDAQ",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "active_from": "2015-01-02",
                "active_to": CELG_LAST_SESSION,
                "isin": "",
                "source": "eodhd_exchange_symbols",
                "source_url": "https://eodhd.com/api/exchange-symbol-list/US",
                "retrieved_at": "2026-07-16T00:00:00Z",
                "source_hash": "a" * 64,
            },
            {
                "security_id": BMY_SECURITY_ID,
                "primary_symbol": BMY_SYMBOL,
                "provider_symbol": "BMY.US",
                "action_provider_symbol": "BMY.US",
                "name": "Bristol-Myers Squibb Company",
                "exchange": "NYSE",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "active_from": "2015-01-02",
                "active_to": "",
                "isin": "",
                "source": "eodhd_exchange_symbols",
                "source_url": "https://eodhd.com/api/exchange-symbol-list/US",
                "retrieved_at": "2026-07-16T00:00:00Z",
                "source_hash": "a" * 64,
            },
        ]
    )
    history = pd.DataFrame(
        [
            {
                "security_id": CELG_SECURITY_ID,
                "symbol": CELG_SYMBOL,
                "exchange": "NASDAQ",
                "effective_from": "2015-01-01",
                "effective_to": "",
                "source": "eodhd_exchange_symbols",
                "source_url": "https://eodhd.com/api/exchange-symbol-list/US",
                "retrieved_at": "2026-07-16T00:00:00Z",
                "source_hash": "a" * 64,
            },
            {
                "security_id": BMY_SECURITY_ID,
                "symbol": BMY_SYMBOL,
                "exchange": "NYSE",
                "effective_from": "2015-01-01",
                "effective_to": "",
                "source": "eodhd_exchange_symbols",
                "source_url": "https://eodhd.com/api/exchange-symbol-list/US",
                "retrieved_at": "2026-07-16T00:00:00Z",
                "source_hash": "a" * 64,
            },
        ]
    )
    prices = pd.DataFrame(
        [
            {
                "security_id": CELG_SECURITY_ID,
                "session": CELG_LAST_SESSION,
                "open": 108.0,
                "high": 109.0,
                "low": 107.0,
                "close": 108.5,
                "volume": 10_000,
                "currency": "USD",
                "source": "eodhd_eod",
                "source_url": "https://eodhd.com/api/eod/CELG.US",
                "retrieved_at": "2026-07-16T00:00:00Z",
                "source_hash": "b" * 64,
            },
            {
                "security_id": BMY_SECURITY_ID,
                "session": MERGER_SESSION,
                "open": 56.0,
                "high": 57.0,
                "low": 55.0,
                "close": 56.48,
                "volume": 10_000,
                "currency": "USD",
                "source": "eodhd_eod",
                "source_url": "https://eodhd.com/api/eod/BMY.US",
                "retrieved_at": "2026-07-16T00:00:00Z",
                "source_hash": "c" * 64,
            },
            {
                "security_id": BMY_SECURITY_ID,
                "session": "2026-07-15",
                "open": 47.0,
                "high": 48.0,
                "low": 46.0,
                "close": 47.5,
                "volume": 10_000,
                "currency": "USD",
                "source": "eodhd_eod",
                "source_url": "https://eodhd.com/api/eod/BMY.US",
                "retrieved_at": "2026-07-16T00:00:00Z",
                "source_hash": "c" * 64,
            },
        ]
    )
    factors = pd.DataFrame(
        [
            {
                "security_id": row.security_id,
                "session": row.session,
                "split_factor": 1.0,
                "total_return_factor": 1.0,
                "source_version": "old-prices+old-actions",
                "calculated_at": "2026-07-16T00:00:00Z",
                "source": "derived",
                "retrieved_at": "2026-07-16T00:00:00Z",
                "source_hash": "old-prices+old-actions",
            }
            for row in prices.itertuples(index=False)
        ]
    )
    resolution = pd.DataFrame(
        [
            {
                "candidate_id": lifecycle_candidate_id(
                    CELG_SECURITY_ID, CELG_LAST_SESSION
                ),
                "security_id": CELG_SECURITY_ID,
                "symbol": CELG_SYMBOL,
                "last_price_date": CELG_LAST_SESSION,
                "resolution": "exception",
                "event_id": "",
                "exception_code": "unsupported_consideration",
                "exception_reason": "CELG consideration included a tradable CVR.",
                "reviewed_by": "finalizer",
                "reviewed_at": "2026-07-18T00:00:00Z",
                "recheck_after": "",
                "successor_security_id": "",
                "successor_symbol": "",
                "source_url": MERGER_TERMS_URL,
                "source": "sec_edgar_filing",
                "retrieved_at": "2026-07-18T07:25:37Z",
                "source_hash": MERGER_TERMS_SHA256,
            }
        ]
    )
    return {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": _empty("corporate_actions", extras=("metadata",)),
        "lifecycle_resolutions": resolution,
        "adjustment_factors": factors,
        "source_archive": _empty("source_archive", extras=("source_url",)),
    }


def _official_repository(root: Path) -> LocalDatasetRepository:
    repository = LocalDatasetRepository(root)
    frames = {
        **_base_frames(),
        "index_constituent_anchors": pd.DataFrame(
            [
                {
                    "index_id": "SP500",
                    "anchor_date": CELG_LAST_SESSION,
                    "security_id": CELG_SECURITY_ID,
                    "official": True,
                    "source_url": "https://example.test/sp500-anchor",
                    "source_kind": "fixture",
                    "source": "fixture",
                    "retrieved_at": "2026-07-18T00:00:00Z",
                    "source_hash": "fixture-anchor",
                }
            ]
        ),
        "index_membership_events": pd.DataFrame(
            [
                {
                    "event_id": "remove-celg",
                    "index_id": "SP500",
                    "announcement_date": CELG_LAST_SESSION,
                    "effective_date": MERGER_SESSION,
                    "operation": "REMOVE",
                    "security_id": CELG_SECURITY_ID,
                    "official": True,
                    "source_url": "https://example.test/remove-celg",
                    "source_kind": "fixture",
                    "source": "fixture",
                    "retrieved_at": "2026-07-18T00:00:00Z",
                    "source_hash": "fixture-remove",
                }
            ]
        ),
    }
    versions: dict[str, str] = {}
    for dataset, frame in frames.items():
        result = repository.write_frame(
            dataset,
            frame,
            completed_session="2026-07-15",
            incomplete_action_policy="block",
            metadata=(
                {"evidence_report_sha256": "e" * 64}
                if dataset == "lifecycle_resolutions"
                else None
            ),
            version=f"base-{dataset}",
        )
        versions[dataset] = result.manifest.version
    repository.commit_release(
        "2026-07-15",
        versions,
        quality="valid",
    )
    return repository


def _patch_official_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        script,
        "_load_merger_terms",
        lambda _repository, _release: _terms(),
    )
    monkeypatch.setattr(
        script,
        "_load_termination_evidence",
        lambda _evidence_dir: _termination(),
    )
    monkeypatch.setattr(
        script,
        "_frozen_catalog_selection",
        lambda _repository, _release: _selection(),
    )


def test_catalog_selection_is_exact_celg_ri_and_bmy_aliases_stay_secondary():
    selection = _selection()
    _validate_catalog_selection(selection)
    assert selection.provider_code == "CELG-RI"
    assert selection.row["Isin"] == "US1101221406"
    assert [item["row"]["Code"] for item in selection.secondary_evidence] == [
        "BMY-R",
        "BMY-RI",
    ]
    promoted = FrozenCatalogSelection(
        **{
            **selection.__dict__,
            "provider_code": "BMY-RI",
            "row": {**dict(selection.row), "Code": "BMY-RI"},
        }
    )
    with pytest.raises(ValueError, match="exact CELG-RI"):
        _validate_catalog_selection(promoted)


def test_bundle_requires_exact_280_sessions_and_three_claimed_calls():
    bundle = _bundle()
    summary = validate_bundle(bundle)
    assert summary["price_rows"] == EXPECTED_CVR_SESSIONS
    assert summary["actual_eodhd_calls"] == 3
    assert summary["first_close"] == CVR_REFERENCE_CLOSE
    missing = CvrBundle(
        **{
            **bundle.__dict__,
            "prices": bundle.prices.iloc[1:].copy(),
        }
    )
    with pytest.raises(ValueError, match="session coverage is not exact"):
        validate_bundle(missing)
    wrong_calls = CvrBundle(**{**bundle.__dict__, "http_attempts": 4})
    with pytest.raises(ValueError, match="exactly three"):
        validate_bundle(wrong_calls)


def test_bundle_cache_hash_wrapper_detects_tampering(tmp_path: Path):
    path = tmp_path / "bmyrt.json.gz"
    bundle = _bundle()
    write_bundle_cache(path, bundle)
    cached = read_bundle_cache(path)
    assert cached.security_id == bundle.security_id
    assert cached.catalog_source_hash == ACTIVE_CATALOG_SHA256
    assert [item.content for item in cached.artifacts] == [
        item.content for item in bundle.artifacts
    ]
    wrapper = json.loads(gzip.decompress(path.read_bytes()))
    payload = json.loads(
        __import__("base64").b64decode(wrapper["payload_base64"], validate=True)
    )
    assert "diagnostic_budget_note" not in payload
    assert "prior_failed_search_calls" not in payload
    wrapper["payload_sha256"] = "0" * 64
    path.write_bytes(gzip.compress(json.dumps(wrapper).encode(), mtime=0))
    with pytest.raises(ValueError, match="wrapper hash mismatch"):
        read_bundle_cache(path)


def test_official_actions_distribute_cvr_before_merger_and_terminate_at_zero():
    bundle = _bundle()
    actions = _official_actions(bundle, _terms(), _termination())
    same_day = actions.loc[actions["effective_date"].eq(MERGER_SESSION)]
    assert list(same_day["action_type"]) == ["spinoff", "stock_merger"]
    spin = same_day.iloc[0]
    merger = same_day.iloc[1]
    terminal = actions.iloc[2]
    assert spin["new_security_id"] == bundle.security_id
    assert spin["ratio"] == 1.0
    assert json.loads(spin["metadata"])["cost_basis_fraction"] == pytest.approx(
        CVR_ECONOMIC_BASIS_FRACTION
    )
    assert merger["new_security_id"] == BMY_SECURITY_ID
    assert merger["cash_amount"] == 50.0
    assert terminal["action_type"] == "delisting"
    assert terminal["effective_date"] == CVR_TERMINATION_DATE
    assert terminal["cash_amount"] == 0.0


def test_ledger_tracks_cvr_separately_then_removes_it_at_zero_idempotently():
    actions = _official_actions(_bundle(), _terms(), _termination()).to_dict(
        orient="records"
    )
    for action in actions:
        action["symbol"] = (
            CELG_SYMBOL if action["security_id"] == CELG_SECURITY_ID else CVR_SYMBOL
        )
    ledger = PortfolioLedger(cash=1_000.0)
    ledger.buy(CELG_SYMBOL, 10.0, 100.0, 1_000.0)
    first_events = ledger.apply_actions(actions, through=MERGER_SESSION)
    assert [event.action_type for event in first_events] == [
        "spinoff",
        "stock_merger",
    ]
    assert ledger.cash == 500.0
    assert ledger.positions[CVR_SYMBOL].quantity == 10.0
    assert ledger.positions[BMY_SYMBOL].quantity == 10.0
    assert ledger.positions[CVR_SYMBOL].avg_price == pytest.approx(
        100.0 * CVR_ECONOMIC_BASIS_FRACTION
    )
    assert ledger.positions[BMY_SYMBOL].avg_price == pytest.approx(
        100.0 * (1.0 - CVR_ECONOMIC_BASIS_FRACTION)
    )
    terminal = ledger.apply_actions(actions, through=CVR_TERMINATION_DATE)
    assert [event.action_type for event in terminal] == ["delisting"]
    assert CVR_SYMBOL not in ledger.positions
    assert ledger.cash == 500.0
    before = ledger.snapshot()
    assert ledger.apply_actions(actions, through="2021-01-04") == ()
    assert ledger.snapshot() == before


def test_prepare_frames_is_idempotent_and_rejects_partial_price_state():
    bundle = _bundle()
    frames = _base_frames()
    first, _, counts = prepare_frames(
        frames,
        bundle,
        _terms(),
        _termination(),
        completed_session="2026-07-15",
        factor_source_version="planned-prices+planned-actions",
    )
    assert counts["price_rows_added"] == EXPECTED_CVR_SESSIONS
    assert counts["corporate_action_rows_added"] == 3
    assert counts["lifecycle_resolution_rows_changed"] == 1
    assert _target_is_exact(first, bundle, _terms(), _termination())
    second, _, second_counts = prepare_frames(
        first,
        bundle,
        _terms(),
        _termination(),
        completed_session="2026-07-15",
        factor_source_version="planned-prices+planned-actions",
    )
    assert all(value == 0 for value in second_counts.values())
    for dataset in first:
        pd.testing.assert_frame_equal(first[dataset], second[dataset])
    partial = {name: value.copy() for name, value in first.items()}
    target = partial["daily_price_raw"]["security_id"].astype(str).eq(
        bundle.security_id
    )
    partial["daily_price_raw"] = partial["daily_price_raw"].drop(
        partial["daily_price_raw"].index[target][0]
    )
    with pytest.raises(ValueError, match="price history is partial"):
        prepare_frames(
            partial,
            bundle,
            _terms(),
            _termination(),
            completed_session="2026-07-15",
            factor_source_version="planned-prices+planned-actions",
        )


def test_official_exit_mark_uses_one_policy_row_and_no_provider_price_claim():
    model = script._official_exit_model(_selection())
    termination = _termination()
    policy = script._official_exit_policy_artifact(model, termination)
    price = script._official_exit_price(model, policy)
    identity = script._official_exit_identity_artifact(
        model, _terms(), termination, policy
    )
    identity_payload = json.loads(identity.content)

    assert len(price) == 1
    assert price.iloc[0]["session"] == MERGER_SESSION
    assert float(price.iloc[0]["close"]) == pytest.approx(2.30)
    assert float(price.iloc[0]["volume"]) == 0.0
    assert price.iloc[0]["source"] == "official_exit_mark_policy"
    assert json.loads(policy.content)["row_encoding"]["meaning"] == (
        "valuation_mark_not_observed_provider_ohlcv"
    )
    assert identity_payload["provider_price_artifact_claimed"] is False
    assert "eodhd_eod_url" not in identity_payload
    assert script.PRIOR_FAILED_EOD_NOTE["raw_response_preserved"] is False


def test_official_exit_mark_frames_are_exact_and_idempotent():
    model = script._official_exit_model(_selection())
    frames = _base_frames()
    first, _, counts = script.prepare_official_exit_frames(
        frames,
        model,
        _terms(),
        _termination(),
        completed_session="2026-07-15",
        factor_source_version="official-prices+official-actions",
    )
    assert counts["price_rows_added"] == 1
    assert counts["corporate_action_rows_added"] == 4
    assert counts["lifecycle_resolution_rows_changed"] == 2
    assert counts["adjustment_factor_rows_added"] == 1
    prices = first["daily_price_raw"].loc[
        first["daily_price_raw"]["security_id"].astype(str).eq(model.security_id)
    ]
    assert len(prices) == 1
    assert set(
        first["corporate_actions"].loc[
            first["corporate_actions"]["security_id"].astype(str).eq(model.security_id),
            "event_id",
        ]
    ) == {
        script.OFFICIAL_EXIT_EVENT_ID,
        script.OFFICIAL_RESIDUAL_TERMINATION_EVENT_ID,
    }
    child_resolution = first["lifecycle_resolutions"].loc[
        first["lifecycle_resolutions"]["security_id"]
        .astype(str)
        .eq(model.security_id)
    ]
    assert len(child_resolution) == 1
    assert script._official_exit_child_resolution_is_exact(
        child_resolution.iloc[0].to_dict(), _termination()
    )
    assert script._official_exit_target_is_exact(
        first,
        model,
        _terms(),
        _termination(),
        release_warnings=(script.OFFICIAL_EXIT_WARNING,),
    )
    second, _, second_counts = script.prepare_official_exit_frames(
        first,
        model,
        _terms(),
        _termination(),
        completed_session="2026-07-15",
        factor_source_version="official-prices+official-actions",
    )
    assert all(value == 0 for value in second_counts.values())
    for dataset in first:
        pd.testing.assert_frame_equal(first[dataset], second[dataset])


def test_actual_current_release_factor_session_is_arrow_writable(tmp_path: Path):
    cache_root = Path("data/cache")
    if not (cache_root / "releases/current.json").is_file():
        pytest.skip("Actual current-release cache is unavailable.")
    repository = LocalDatasetRepository(cache_root)
    release, _ = repository.current_release()
    if release is None or "adjustment_factors" not in release.dataset_versions:
        pytest.skip("Actual current release has no adjustment_factors dataset.")
    current = repository.read_frame(
        "adjustment_factors",
        release.dataset_versions["adjustment_factors"],
    )
    model = script._official_exit_model(_selection())
    current_without_target = current.loc[
        ~current["security_id"].astype(str).eq(model.security_id)
    ].copy()
    if current_without_target.empty:
        pytest.skip("Actual current adjustment_factors dataset is empty.")
    assert isinstance(current_without_target.iloc[0]["session"], pd.Timestamp)

    rewritten, added = script._rewrite_official_exit_factors(
        current_without_target,
        model,
        source_version="actual-current-regression-prices+actions",
    )

    assert added == 1
    assert pd.api.types.is_datetime64_any_dtype(rewritten["session"])
    assert not rewritten["session"].map(lambda value: isinstance(value, str)).any()
    destination = tmp_path / "actual-current-adjustment-factors.parquet"
    rewritten.to_parquet(
        destination,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    restored = pd.read_parquet(destination)
    assert len(restored) == len(current_without_target) + 1
    target = restored.loc[
        restored["security_id"].astype(str).eq(model.security_id)
    ]
    assert len(target) == 1
    assert pd.Timestamp(target.iloc[0]["session"]) == pd.Timestamp(MERGER_SESSION)


def test_official_exit_mark_liquidates_at_close_and_cash_settles_next_session():
    model = script._official_exit_model(_selection())
    actions = script._official_exit_actions(
        model, _terms(), _termination()
    ).to_dict(orient="records")
    for action in actions:
        action["symbol"] = (
            CELG_SYMBOL if action["security_id"] == CELG_SECURITY_ID else CVR_SYMBOL
        )
    ledger = PortfolioLedger(cash=1_000.0)
    ledger.buy(CELG_SYMBOL, 10.0, 100.0, 1_000.0)

    events = ledger.apply_actions(actions, through=MERGER_SESSION)

    assert [event.action_type for event in events] == [
        "spinoff",
        "stock_merger",
        "delisting",
    ]
    assert CVR_SYMBOL not in ledger.positions
    assert ledger.positions[BMY_SYMBOL].quantity == 10.0
    assert ledger.cash == pytest.approx(500.0)
    assert ledger.receivable_value == pytest.approx(23.0)
    assert script.OFFICIAL_EXIT_EVENT_ID in ledger.entitled_event_ids
    settlement = ledger.apply_actions((), through="2019-11-22")
    assert len(settlement) == 1
    assert ledger.cash == pytest.approx(523.0)
    assert ledger.receivable_value == 0.0
    terminal = ledger.apply_actions(actions, through=CVR_TERMINATION_DATE)
    assert [event.action_type for event in terminal] == ["delisting"]
    assert ledger.cash == pytest.approx(523.0)


def test_official_exit_apply_is_degraded_warned_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_official_inputs(monkeypatch)
    repository = _official_repository(tmp_path)
    prepared = script.prepare_official_exit_repair(
        repository,
        evidence_dir=tmp_path / "unused-evidence",
    )

    assert prepared.summary["status"] == "validated_offline_plan"
    assert prepared.summary["apply_ready"] is True
    assert prepared.summary["new_global_issue_count"] == 0
    assert prepared.summary["candidate_set_unchanged"] is False
    assert prepared.summary["expected_successor_candidate_added"] is True
    assert prepared.summary["network_accessed"] is False
    assert prepared.summary["eodhd_calls"] == 0
    assert prepared.summary["r2_accessed"] is False

    applied = script.apply_repair(repository, prepared)

    assert applied["status"] == "applied"
    assert applied["quality"] == "degraded"
    assert script.OFFICIAL_EXIT_WARNING in applied["warnings"]
    release, _ = repository.current_release()
    assert release is not None
    assert release.quality == "degraded"
    assert script.OFFICIAL_EXIT_WARNING in release.warnings
    factors = repository.read_frame(
        "adjustment_factors",
        release.dataset_versions["adjustment_factors"],
    )
    expected_factor_source = (
        release.dataset_versions["daily_price_raw"]
        + "+"
        + release.dataset_versions["corporate_actions"]
    )
    assert set(factors["source_version"].astype(str)) == {
        expected_factor_source
    }
    lifecycle_manifest = repository.manifest_for_version(
        "lifecycle_resolutions",
        release.dataset_versions["lifecycle_resolutions"],
    )
    assert lifecycle_manifest.metadata["evidence_report_sha256"] == "e" * 64
    second = script.prepare_official_exit_repair(
        repository,
        evidence_dir=tmp_path / "unused-evidence",
    )
    assert second.summary["status"] == "already_applied"
    assert second.summary["candidate_set_unchanged"] is True
    assert second.summary["expected_successor_candidate_added"] is False
    assert script.apply_repair(repository, second)["writes_performed"] is False


def test_official_exit_coverage_rejects_any_second_successor_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_official_inputs(monkeypatch)
    repository = _official_repository(tmp_path)
    release, _ = repository.current_release()
    assert release is not None
    before = {
        dataset: repository.read_frame(
            dataset, release.dataset_versions[dataset]
        )
        for dataset in script.REQUIRED_DATASETS
    }
    extra_security_id = "US:EODHD:unexpected-successor"
    extra_master = before["security_master"].loc[
        before["security_master"]["security_id"].astype(str).eq(BMY_SECURITY_ID)
    ].iloc[0].copy()
    extra_master["security_id"] = extra_security_id
    extra_master["primary_symbol"] = "EXTRA"
    extra_master["provider_symbol"] = "EXTRA.US"
    extra_master["name"] = "Unexpected Successor"
    extra_master["active_to"] = MERGER_SESSION
    before["security_master"] = pd.concat(
        [before["security_master"], pd.DataFrame([extra_master])],
        ignore_index=True,
    )
    extra_price = before["daily_price_raw"].loc[
        before["daily_price_raw"]["security_id"]
        .astype(str)
        .eq(CELG_SECURITY_ID)
    ].iloc[-1].copy()
    extra_price["security_id"] = extra_security_id
    before["daily_price_raw"] = pd.concat(
        [before["daily_price_raw"], pd.DataFrame([extra_price])],
        ignore_index=True,
    )
    model = script._official_exit_model(_selection())
    after, _, _ = script.prepare_official_exit_frames(
        before,
        model,
        _terms(),
        _termination(),
        completed_session=release.completed_session,
        factor_source_version="planned-prices+planned-actions",
    )
    malicious = after["corporate_actions"].iloc[-1].copy()
    malicious["event_id"] = "unexpected-successor-edge"
    malicious["security_id"] = model.security_id
    malicious["action_type"] = "stock_merger"
    malicious["effective_date"] = "2020-01-02"
    malicious["ex_date"] = "2020-01-02"
    malicious["new_security_id"] = extra_security_id
    malicious["new_symbol"] = "EXTRA"
    after["corporate_actions"] = pd.concat(
        [after["corporate_actions"], pd.DataFrame([malicious])],
        ignore_index=True,
    )
    with pytest.raises(
        ValueError,
        match="candidate expansion is not exactly BMYRT",
    ):
        script._official_exit_coverage_delta(
            repository,
            release,
            before,
            after,
            expected_transition=True,
        )


def test_official_exit_apply_cas_and_release_commit_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_official_inputs(monkeypatch)
    cas_repository = _official_repository(tmp_path / "cas")
    cas_prepared = script.prepare_official_exit_repair(
        cas_repository,
        evidence_dir=tmp_path / "unused-evidence",
    )
    old_pointers = {
        dataset: cas_repository.objects.get(
            cas_repository.current_key(dataset)
        ).data
        for dataset in script.WRITE_DATASETS
    }
    release, release_etag = cas_repository.current_release()
    assert release is not None
    cas_repository.commit_release(
        release.completed_session,
        release.dataset_versions,
        quality=release.quality,
        warnings=("intervening release",),
        expected_etag=release_etag,
    )

    with pytest.raises(RuntimeError, match="Current release changed"):
        script.apply_repair(cas_repository, cas_prepared)
    assert all(
        cas_repository.objects.get(cas_repository.current_key(dataset)).data
        == old_pointers[dataset]
        for dataset in script.WRITE_DATASETS
    )

    rollback_repository = _official_repository(tmp_path / "rollback")
    rollback_prepared = script.prepare_official_exit_repair(
        rollback_repository,
        evidence_dir=tmp_path / "unused-evidence",
    )
    old_release = rollback_repository.objects.get("releases/current.json").data
    rollback_pointers = {
        dataset: rollback_repository.objects.get(
            rollback_repository.current_key(dataset)
        ).data
        for dataset in script.WRITE_DATASETS
    }

    def fail(stage: str) -> None:
        if stage == "after_release_commit":
            raise RuntimeError("injected CELG official-exit rollback")

    with pytest.raises(RuntimeError, match="injected CELG official-exit rollback"):
        script.apply_repair(
            rollback_repository,
            rollback_prepared,
            inject_failure=fail,
        )
    assert (
        rollback_repository.objects.get("releases/current.json").data
        == old_release
    )
    assert all(
        rollback_repository.objects.get(
            rollback_repository.current_key(dataset)
        ).data
        == rollback_pointers[dataset]
        for dataset in script.WRITE_DATASETS
    )
    journals = tuple(
        (tmp_path / "rollback/transactions/us-celg-bmy-cvr").glob("*.json")
    )
    assert len(journals) == 1
    assert json.loads(journals[0].read_bytes())["status"] == "rolled_back"


class _Response:
    def __init__(self, content: bytes):
        self.content = content
        self.headers = {"Content-Type": "application/json; charset=utf-8"}

    def raise_for_status(self):
        return None

class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response(self.responses.pop(0))


def test_exact_client_forbids_search_and_followups_before_eod_gate(tmp_path: Path):
    bundle = _bundle()
    by_source = {
        artifact.source: artifact.content for artifact in bundle.artifacts
    }
    session = _Session(
        [
            by_source["eodhd_eod"],
            by_source["eodhd_div"],
            by_source["eodhd_splits"],
        ]
    )
    budget = EodhdCallBudget(
        tmp_path / "budget.json",
        limit=100,
        reserve=1,
        seed_used=0,
        period="2026-07-18",
    )
    client = ExactThreeEodhdClient(session=session, token="secret", budget=budget)
    with pytest.raises(RuntimeError, match="forbids get_json/search"):
        client.get_json("search/BMYRT", params={"limit": 10})
    eod = client.fetch_artifact(
        f"eod/{CVR_PROVIDER_SYMBOL}",
        params={"from": MERGER_SESSION, "to": CVR_LAST_SESSION},
    )
    with pytest.raises(RuntimeError, match="before the exact EOD gate"):
        client.fetch_artifact(
            f"div/{CVR_PROVIDER_SYMBOL}",
            params={"from": MERGER_SESSION, "to": CVR_LAST_SESSION},
        )
    client.validate_and_authorize_eod(eod, security_id=bundle.security_id)
    for endpoint in ("div", "splits"):
        client.fetch_artifact(
            f"{endpoint}/{CVR_PROVIDER_SYMBOL}",
            params={"from": MERGER_SESSION, "to": CVR_LAST_SESSION},
        )
    assert len(session.calls) == 3
    assert all("secret" not in url for url, _ in session.calls)
    with pytest.raises(RuntimeError, match="fourth"):
        client.fetch_artifact(
            f"eod/{CVR_PROVIDER_SYMBOL}",
            params={"from": MERGER_SESSION, "to": CVR_LAST_SESSION},
        )
    fresh = ExactThreeEodhdClient(
        session=_Session([]), token="secret", budget=budget
    )
    with pytest.raises(RuntimeError, match="non-reviewed"):
        fresh.fetch_artifact(f"div/{CVR_PROVIDER_SYMBOL}", params={})


def test_fetch_stops_after_one_call_when_eod_gate_fails(tmp_path: Path):
    bundle = _bundle()
    by_source = {artifact.source: artifact for artifact in bundle.artifacts}
    short_eod = json.loads(by_source["eodhd_eod"].content)[1:]
    session = _Session(
        [
            json.dumps(short_eod).encode(),
            by_source["eodhd_div"].content,
            by_source["eodhd_splits"].content,
        ]
    )
    budget = EodhdCallBudget(
        tmp_path / "budget.json",
        limit=100,
        reserve=1,
        seed_used=0,
        period="2026-07-18",
    )
    client = ExactThreeEodhdClient(session=session, token="secret", budget=budget)
    with pytest.raises(ValueError, match="session coverage is not exact"):
        fetch_exact_bundle(
            client,
            catalog_selection=_selection(),
            release_version="release-v1",
            budget_used_before=0,
        )
    assert len(session.calls) == 1
    assert json.loads((tmp_path / "budget.json").read_text())["used"] == 1


def test_cli_modes_are_mutually_exclusive_and_default_to_plan():
    args = _parse_args([])
    assert not args.fetch and not args.apply
    with pytest.raises(SystemExit):
        _parse_args(["--fetch", "--apply"])
