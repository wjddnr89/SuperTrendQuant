from __future__ import annotations

import base64
import gzip
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from supertrend_quant.ledger import PortfolioLedger
from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.ingest import EodhdCallBudget, SourceArtifact
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    sha256_bytes,
)
from supertrend_quant.portfolio import Position


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_alcoa_spinoffs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_alcoa_spinoffs", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


class _ValidReport:
    issues: tuple[object, ...] = ()

    def raise_for_errors(self) -> None:
        return None


def _catalog() -> object:
    return script.FrozenCatalogSelection(
        row=dict(script.EXPECTED_CATALOG_ROW),
        source_url="https://eodhd.com/api/exchange-symbol-list/US?delisted=0",
        retrieved_at="2026-07-16T15:56:01Z",
        source_hash=sha256_bytes(b"frozen-catalog"),
        object_path="archives/2026-07-15/catalog.json.gz",
    )


def _eod_rows() -> list[dict[str, object]]:
    return [
        {
            "date": session,
            "open": 20.0 + number / 10_000,
            "high": 21.0 + number / 10_000,
            "low": 19.0 + number / 10_000,
            "close": 20.5 + number / 10_000,
            "adjusted_close": 20.5 + number / 10_000,
            "volume": 1_000_000 + number,
        }
        for number, session in enumerate(script._expected_sessions())
    ]


def _raw_artifacts() -> tuple[SourceArtifact, ...]:
    rows: dict[str, list[dict[str, object]]] = {
        "eod": _eod_rows(),
        "div": [
            {
                "date": "2024-05-20",
                "unadjustedValue": 0.10,
                "currency": "USD",
                "declarationDate": "2024-05-02",
                "recordDate": "2024-05-21",
                "paymentDate": "2024-06-07",
            }
        ],
        "splits": [
            {"date": script.FETCH_START, "split": "1/3"},
            {"date": "2022-01-03", "split": "2/1"},
        ],
    }
    return tuple(
        script._source_artifact(
            endpoint, rows[endpoint], "2026-07-18T00:00:00Z"
        )
        for endpoint in script.ENDPOINTS
    )


def _bundle(*, before: int = 8_393) -> object:
    return script._bundle_from_artifacts(
        _raw_artifacts(),
        http_attempts=3,
        budget_used_before=before,
        budget_used_after=before + 3,
    )


def _source(label: str) -> dict[str, object]:
    return {
        "source": label,
        "source_url": f"https://example.test/{label}",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": sha256_bytes(label.encode()),
    }


def _master(
    security_id: str,
    symbol: str,
    provider_symbol: str,
    start: str,
    end: str,
) -> dict[str, object]:
    return {
        "security_id": security_id,
        "primary_symbol": symbol,
        "provider_symbol": provider_symbol,
        "action_provider_symbol": provider_symbol,
        "name": symbol,
        "exchange": "NYSE",
        "asset_type": "STOCK",
        "currency": "USD",
        "country": "US",
        "active_from": start,
        "active_to": end,
        "isin": "",
        **_source(f"master-{symbol}"),
    }


def _history(
    security_id: str,
    symbol: str,
    start: str,
    end: str,
) -> dict[str, object]:
    return {
        "security_id": security_id,
        "symbol": symbol,
        "exchange": "NYSE",
        "effective_from": start,
        "effective_to": end,
        **_source(f"history-{symbol}-{start}"),
    }


def _price(
    security_id: str,
    session: str,
    close: float = 30.0,
) -> dict[str, object]:
    return {
        "security_id": security_id,
        "session": session,
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1_000_000,
        "currency": "USD",
        **_source(f"price-{security_id}"),
    }


def _action(
    event_id: str,
    action_type: str,
    effective_date: str,
    *,
    ratio: float | None,
    new_security_id: str = "",
    new_symbol: str = "",
    official: bool,
    source_url: str,
    source_hash: str,
    source: str,
    source_kind: str,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "security_id": script.HWM_SECURITY_ID,
        "action_type": action_type,
        "effective_date": effective_date,
        "ex_date": effective_date,
        "announcement_date": "",
        "record_date": "",
        "payment_date": "",
        "cash_amount": None,
        "ratio": ratio,
        "currency": "USD",
        "new_security_id": new_security_id,
        "new_symbol": new_symbol,
        "official": official,
        "source_url": source_url,
        "source_kind": source_kind,
        "source": source,
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": source_hash,
        "metadata": None,
    }


def _archive_row(url: str, source_hash: str, suffix: str = "html") -> dict[str, object]:
    return {
        "archive_id": source_hash,
        "dataset": "official_identity_evidence_raw",
        "object_path": f"archives/{script.FETCH_END}/{source_hash}.{suffix}.gz",
        "content_type": "text/html",
        "effective_date": script.FETCH_END,
        "source": "official_identity_evidence_raw",
        "source_url": url,
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": source_hash,
    }


def _base_frames() -> dict[str, pd.DataFrame]:
    master = pd.DataFrame(
        [
            _master(
                script.HWM_SECURITY_ID,
                script.HWM_SYMBOL,
                "HWM.US",
                "2015-01-02",
                "",
            ),
            _master(
                script.ARNC_SECURITY_ID,
                script.ARNC_SYMBOL,
                "ARNC.US",
                "2020-04-01",
                "2023-08-17",
            ),
        ]
    )
    history = pd.DataFrame(
        [
            _history(script.HWM_SECURITY_ID, "AA", "2015-01-01", "2016-10-31"),
            _history(script.HWM_SECURITY_ID, "ARNC", "2016-11-01", "2020-03-31"),
            _history(script.HWM_SECURITY_ID, "HWM", "2020-04-01", ""),
            _history(
                script.ARNC_SECURITY_ID,
                "ARNC",
                "2020-04-01",
                "2023-08-17",
            ),
        ]
    )
    hwm_sessions = tuple(
        pd.Timestamp(value).date().isoformat()
        for value in script.xcals.get_calendar("XNYS").sessions_in_range(
            "2015-01-02", script.FETCH_END
        )
    )
    assert len(hwm_sessions) == 2_899
    hwm_prices = pd.DataFrame(
        [
            _price(script.HWM_SECURITY_ID, session, 30.0)
            for session in hwm_sessions
        ]
    )
    arnc_sessions = tuple(
        pd.Timestamp(value).date().isoformat()
        for value in script.xcals.get_calendar("XNYS").sessions_in_range(
            "2020-04-01", "2023-08-17"
        )
    )
    assert len(arnc_sessions) == 851
    arnc_prices = pd.DataFrame(
        [
            _price(script.ARNC_SECURITY_ID, session, 20.0)
            for session in arnc_sessions
        ]
    )
    prices = pd.concat([hwm_prices, arnc_prices], ignore_index=True)
    actions = pd.DataFrame(
        [
            _action(
                script.LEGAL_REVERSE_SPLIT_EVENT_ID,
                "split",
                script.LEGAL_REVERSE_SPLIT_DATE,
                ratio=1.0 / 3.0,
                official=True,
                source_url="https://www.sec.gov/official-reverse-split",
                source_hash=sha256_bytes(b"official reverse split"),
                source="official_identity_repair",
                source_kind="official_filing",
            ),
            _action(
                script.PSEUDO_SPLIT_EVENT_ID,
                "split",
                script.PSEUDO_SPLIT_EFFECTIVE_DATE,
                ratio=script.PSEUDO_SPLIT_RATIO,
                official=False,
                source_url=script.PSEUDO_SPLIT_SOURCE_URL,
                source_hash=script.PSEUDO_SPLIT_RAW_SHA256,
                source="eodhd_splits",
                source_kind="provider",
            ),
            _action(
                script.SPINOFF_2016_EVENT_ID,
                "spinoff",
                script.SPINOFF_2016_EFFECTIVE_DATE,
                ratio=script.SPINOFF_2016_RATIO,
                new_symbol="AA",
                official=True,
                source_url=script.SPINOFF_2016_SOURCE_URL,
                source_hash=script.SPINOFF_2016_SOURCE_SHA256,
                source="official_identity_repair",
                source_kind="official_filing",
            ),
            _action(
                script.SPINOFF_2020_EVENT_ID,
                "spinoff",
                script.SPINOFF_2020_EFFECTIVE_DATE,
                ratio=script.SPINOFF_2020_RATIO,
                new_security_id=script.ARNC_SECURITY_ID,
                new_symbol="ARNC",
                official=True,
                source_url=script.SPINOFF_2020_SOURCE_URL,
                source_hash=script.SPINOFF_2020_SOURCE_SHA256,
                source="official_identity_repair",
                source_kind="official_filing",
            ),
        ]
    )
    factors = pd.concat(
        [
            build_adjustment_factors(
                hwm_prices,
                actions,
                source_version="base-hwm",
            ),
            build_adjustment_factors(
                arnc_prices,
                actions.iloc[:0],
                source_version="base-arnc",
            ),
        ],
        ignore_index=True,
    )
    archive = pd.DataFrame(
        [
            _archive_row(
                script.SPINOFF_2016_SOURCE_URL,
                script.SPINOFF_2016_SOURCE_SHA256,
            ),
            _archive_row(
                script.SPINOFF_2020_SOURCE_URL,
                script.SPINOFF_2020_SOURCE_SHA256,
            ),
        ]
    )
    return {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "adjustment_factors": factors,
        "source_archive": archive,
    }


def _official_evidence(monkeypatch: pytest.MonkeyPatch) -> dict[str, SourceArtifact]:
    first = SourceArtifact(
        source="howmet_tax_basis_alcoa_2016",
        source_url=script.TAX_BASIS_2016_URL,
        retrieved_at="2026-07-18T00:00:00Z",
        content=b"test-pdf-2016",
        content_type="application/pdf",
    )
    second = SourceArtifact(
        source="howmet_tax_basis_arconic_2020",
        source_url=script.TAX_BASIS_2020_URL,
        retrieved_at="2026-07-18T00:00:00Z",
        content=b"test-pdf-2020",
        content_type="application/pdf",
    )
    monkeypatch.setattr(script, "TAX_BASIS_2016_SHA256", first.source_hash)
    monkeypatch.setattr(script, "TAX_BASIS_2020_SHA256", second.source_hash)
    monkeypatch.setitem(script.COST_BASIS_2016, "source_hash", first.source_hash)
    monkeypatch.setitem(script.COST_BASIS_2020, "source_hash", second.source_hash)
    return {"2016": first, "2020": second}


class _PrepareRepository:
    def __init__(self, root: Path, versions: dict[str, str]):
        self.root = root
        self.versions = versions

    def current_pointer(self, dataset: str):
        return SimpleNamespace(version=self.versions[dataset]), f"etag-{dataset}"


def _release() -> DataRelease:
    return DataRelease(
        version="base-release",
        created_at="2026-07-18T00:00:00Z",
        completed_session=script.FETCH_END,
        dataset_versions={
            dataset: f"base-{dataset}" for dataset in script.WRITE_DATASETS
        },
    )


def test_pinned_identity_and_exact_session_inventory() -> None:
    expected = script.uuid.uuid5(
        script.uuid.NAMESPACE_URL, "eodhd:US:AA:symbol:AA"
    )
    assert script.AA_SECURITY_ID == f"US:EODHD:{expected}"
    assert script.EXPECTED_CATALOG_ROW["Isin"] == "US0138721065"
    assert len(script._expected_sessions()) == 2_437
    assert script._expected_sessions()[0] == script.FETCH_START
    assert script._expected_sessions()[-1] == script.FETCH_END


def test_bundle_filters_only_inception_pseudo_split_and_is_strict() -> None:
    bundle = _bundle()
    summary = script.validate_fetched_bundle(bundle)

    assert summary["aa_price_rows"] == 2_437
    assert summary["aa_dividend_rows"] == 1
    assert summary["aa_split_rows"] == 1
    assert summary["aa_inception_pseudo_split_rows_archived_only"] == 1
    assert not (
        bundle.corporate_actions["effective_date"].eq(script.FETCH_START)
        & bundle.corporate_actions["action_type"].eq("split")
    ).any()

    tampered = script.FetchedBundle(
        prices=bundle.prices.iloc[:-1].copy(),
        corporate_actions=bundle.corporate_actions,
        artifacts=bundle.artifacts,
        http_attempts=3,
        inception_pseudo_split_rows=1,
        budget_used_before=8_393,
        budget_used_after=8_396,
    )
    with pytest.raises(ValueError, match="2,437"):
        script.validate_fetched_bundle(tampered)


def test_exact_client_claims_three_and_refuses_fourth(tmp_path: Path) -> None:
    payloads = [_eod_rows(), [], []]

    class Response:
        def __init__(self, value):
            self.value = value

        def raise_for_status(self):
            return None

        def json(self):
            return self.value

    class Session:
        def __init__(self):
            self.calls: list[tuple[str, dict[str, object]]] = []

        def get(self, url, *, params, timeout):
            assert timeout == 120
            self.calls.append((url, dict(params)))
            return Response(payloads[len(self.calls) - 1])

    budget = EodhdCallBudget(
        tmp_path / "budget.json",
        limit=100_000,
        reserve=5_000,
        seed_used=8_393,
        period="test-period",
    )
    session = Session()
    client = script.ExactThreeAaEodhdClient(
        session=session,
        token="secret-token",
        budget=budget,
    )
    bundle = script.fetch_exact_bundle(client, budget_used_before=8_393)

    assert client.attempted_endpoints == [
        "eod/AA.US",
        "div/AA.US",
        "splits/AA.US",
    ]
    assert len(session.calls) == 3
    assert script._budget_used(budget) == 8_396
    assert bundle.budget_used_after == 8_396
    with pytest.raises(RuntimeError, match="fourth"):
        client.get_json("eod/AA.US", params=script.REQUEST_PARAMS)
    assert script._budget_used(budget) == 8_396


def test_bundle_cache_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    catalog = _catalog()
    bundle = _bundle()
    path = script._bundle_cache_path(tmp_path, catalog)
    script._write_bundle_cache(path, catalog, bundle)
    replay = script._read_bundle_cache(path, catalog)
    assert replay is not None
    assert script.validate_fetched_bundle(replay)["actual_eodhd_calls"] == 3

    value = json.loads(gzip.decompress(path.read_bytes()))
    value["artifacts"][0]["content_base64"] = base64.b64encode(b"[]").decode()
    path.write_bytes(gzip.compress(script._canonical_json_bytes(value), mtime=0))
    with pytest.raises(ValueError, match="hash mismatch"):
        script._read_bundle_cache(path, catalog)


def test_request_envelope_avoids_empty_payload_archive_collisions() -> None:
    dividend = script._source_artifact(
        "div", [], "2026-07-18T00:00:00Z"
    )
    splits = script._source_artifact(
        "splits", [], "2026-07-18T00:00:00Z"
    )
    assert dividend.source_hash == splits.source_hash

    archived_dividend = script._request_archive_artifact(dividend)
    archived_splits = script._request_archive_artifact(splits)
    assert archived_dividend.source_hash != archived_splits.source_hash
    envelope = json.loads(archived_dividend.content)
    assert envelope["source_url"] == script.REQUEST_URLS["div"]
    assert envelope["content_sha256"] == dividend.source_hash
    assert base64.b64decode(envelope["content_base64"]) == dividend.content


def test_installed_request_envelopes_replay_back_to_exact_bundle(
    tmp_path: Path,
) -> None:
    original = _bundle()
    archived = tuple(
        script._request_archive_artifact(artifact)
        for artifact in original.artifacts
    )
    columns = tuple(
        dict.fromkeys(
            (
                *script.dataset_spec("source_archive").required_columns,
                "source_url",
            )
        )
    )
    archive = script._append_source_archive(
        pd.DataFrame(columns=columns),
        archived,
        completed_session=script.FETCH_END,
    )
    repository = SimpleNamespace(root=tmp_path)
    script._persist_archive_payloads(
        repository,
        archived,
        completed_session=script.FETCH_END,
    )

    replay = script._bundle_from_installed_archive(repository, archive)

    assert script.validate_fetched_bundle(replay)["aa_price_rows"] == 2_437
    assert [item.source_hash for item in replay.artifacts] == [
        item.source_hash for item in original.artifacts
    ]


def test_official_evidence_partial_cache_fetches_only_missing_reviewed_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_content = b"first-reviewed-pdf"
    second_content = b"second-reviewed-pdf"
    specs = {
        "2016": {
            "url": "https://example.test/2016.pdf",
            "sha256": sha256_bytes(first_content),
            "size": len(first_content),
            "source": "basis-2016",
        },
        "2020": {
            "url": "https://example.test/2020.pdf",
            "sha256": sha256_bytes(second_content),
            "size": len(second_content),
            "source": "basis-2020",
        },
    }
    monkeypatch.setattr(script, "OFFICIAL_SPECS", specs)
    first_path = script._official_cache_path(tmp_path, specs["2016"])
    first_path.parent.mkdir(parents=True)
    first_path.write_bytes(first_content)

    class Response:
        content = second_content

        def raise_for_status(self):
            return None

    class Session:
        def __init__(self):
            self.urls: list[str] = []

        def get(self, url, **_kwargs):
            self.urls.append(url)
            return Response()

    session = Session()
    client = script.ExactOfficialEvidenceClient(session=session)
    result = script.fetch_missing_official_evidence(tmp_path, client=client)

    assert result["official_http_attempts"] == 1
    assert result["fetched"] == ["2020"]
    assert session.urls == [specs["2020"]["url"]]
    assert script._official_cache_path(tmp_path, specs["2020"]).read_bytes() == second_content
    with pytest.raises(RuntimeError, match="unreviewed"):
        client.fetch("2020")


def test_prepare_repair_installs_both_basis_events_and_rebuilds_factors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _base_frames()
    official = _official_evidence(monkeypatch)
    release = _release()
    repository = _PrepareRepository(tmp_path, release.dataset_versions)
    monkeypatch.setattr(script, "validate_repository_snapshot", lambda _repo: _ValidReport())

    prepared = script.prepare_repair(
        repository,
        release,
        "release-etag",
        frames,
        _catalog(),
        _bundle(),
        official,
    )

    assert prepared.summary["fake_split_rows_removed"] == 1
    assert prepared.summary["aa_price_rows"] == 2_437
    actions = prepared.frames["corporate_actions"]
    assert not actions.event_id.astype(str).eq(script.PSEUDO_SPLIT_EVENT_ID).any()
    first = actions.loc[actions.event_id.eq(script.SPINOFF_2016_EVENT_ID)].iloc[0]
    second = actions.loc[actions.event_id.eq(script.SPINOFF_2020_EVENT_ID)].iloc[0]
    assert first.new_security_id == script.AA_SECURITY_ID
    assert json.loads(first.metadata)["cost_basis_fraction"] == pytest.approx(0.2686)
    assert json.loads(second.metadata)["cost_basis_fraction"] == pytest.approx(0.114)
    early = prepared.frames["adjustment_factors"].loc[
        prepared.frames["adjustment_factors"].security_id.eq(script.HWM_SECURITY_ID)
        & (
            pd.to_datetime(prepared.frames["adjustment_factors"].session)
            < pd.Timestamp(script.LEGAL_REVERSE_SPLIT_DATE)
        )
    ]
    assert set(early.split_factor) == {3.0}
    assert len(
        prepared.frames["adjustment_factors"].loc[
            prepared.frames["adjustment_factors"].security_id.eq(script.AA_SECURITY_ID)
        ]
    ) == 2_437

    tampered = {key: value.copy() for key, value in prepared.frames.items()}
    index = tampered["corporate_actions"].index[
        tampered["corporate_actions"].event_id.eq(script.SPINOFF_2016_EVENT_ID)
    ].item()
    metadata = json.loads(tampered["corporate_actions"].at[index, "metadata"])
    metadata["cost_basis_fraction"] = 0.25
    tampered["corporate_actions"].at[index, "metadata"] = script._metadata(metadata)
    with pytest.raises(ValueError, match="terms are not exact"):
        script.validate_candidate_frames(
            tampered,
            catalog=_catalog(),
            bundle=_bundle(),
            official=official,
            completed_session=script.FETCH_END,
        )


def test_pseudo_split_removal_fails_closed_on_provenance_change() -> None:
    actions = _base_frames()["corporate_actions"].copy()
    index = actions.index[actions.event_id.eq(script.PSEUDO_SPLIT_EVENT_ID)].item()
    actions.at[index, "source_hash"] = "0" * 64
    with pytest.raises(ValueError, match="provenance changed"):
        script._validated_pseudo_split_mask(actions)


def test_canonical_basis_metadata_reaches_ledger() -> None:
    ledger_2016 = PortfolioLedger(
        cash=0.0,
        positions={"ARNC": Position("ARNC", 12.0, 100.0)},
    )
    action_2016 = {
        "event_id": script.SPINOFF_2016_EVENT_ID,
        "action_type": "spinoff",
        "symbol": "ARNC",
        "effective_date": script.SPINOFF_2016_EFFECTIVE_DATE,
        "ratio": script.SPINOFF_2016_RATIO,
        "new_symbol": "AA",
        "metadata": script._metadata(script.COST_BASIS_2016),
    }
    ledger_2016.apply_actions([action_2016], through=script.SPINOFF_2016_EFFECTIVE_DATE)
    assert ledger_2016.positions["ARNC"].avg_price == pytest.approx(73.14)
    assert ledger_2016.positions["AA"].quantity == pytest.approx(4.0)
    assert ledger_2016.positions["AA"].avg_price == pytest.approx(80.58)

    ledger_2020 = PortfolioLedger(
        cash=0.0,
        positions={"HWM": Position("HWM", 12.0, 100.0)},
    )
    action_2020 = {
        "event_id": script.SPINOFF_2020_EVENT_ID,
        "action_type": "spinoff",
        "symbol": "HWM",
        "effective_date": script.SPINOFF_2020_EFFECTIVE_DATE,
        "ratio": script.SPINOFF_2020_RATIO,
        "new_symbol": "ARNC",
        "metadata": script._metadata(script.COST_BASIS_2020),
    }
    ledger_2020.apply_actions([action_2020], through=script.SPINOFF_2020_EFFECTIVE_DATE)
    assert ledger_2020.positions["HWM"].avg_price == pytest.approx(88.6)
    assert ledger_2020.positions["ARNC"].quantity == pytest.approx(3.0)
    assert ledger_2020.positions["ARNC"].avg_price == pytest.approx(45.6)


def test_cli_plan_is_read_only_and_modes_are_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert script._parse_args([]).mode == "plan"
    assert script._parse_args(["--fetch-official-evidence"]).mode == "fetch_official_evidence"
    assert script._parse_args(["--fetch-missing"]).mode == "fetch_missing"
    assert script._parse_args(["--offline-plan"]).mode == "offline_plan"
    assert script._parse_args(["--apply"]).mode == "apply"
    with pytest.raises(SystemExit):
        script._parse_args(["--fetch-missing", "--apply"])

    release = _release()
    frames = {
        "security_master": pd.DataFrame([{"security_id": script.HWM_SECURITY_ID}]),
        "corporate_actions": pd.DataFrame([{"event_id": "base"}]),
    }
    repository = SimpleNamespace(root=tmp_path)
    monkeypatch.setattr(
        script,
        "_base_context",
        lambda _repository: (release, "etag", frames, _catalog()),
    )
    result = script.run(
        SimpleNamespace(cache_root=tmp_path, mode="plan"),
        repository_factory=lambda _root: repository,
        budget_factory=lambda: pytest.fail("plan constructed an EODHD budget"),
        client_factory=lambda **_kwargs: pytest.fail("plan constructed an EODHD client"),
        official_client_factory=lambda **_kwargs: pytest.fail(
            "plan constructed an official HTTP client"
        ),
    )
    assert result["network_accessed"] is False
    assert result["would_write"] is False
    assert result["expected_eodhd_calls"] == 3


def test_restore_transaction_state_restores_release_and_all_pointers() -> None:
    class Value:
        def __init__(self, data: bytes, etag: str):
            self.data = data
            self.etag = etag

    class Objects:
        def __init__(self, values: dict[str, bytes]):
            self.values = values
            self.revisions = {key: 0 for key in values}

        def get(self, key: str):
            return Value(self.values[key], f"{key}-{self.revisions[key]}")

        def put(self, key: str, data: bytes, *, if_match: str):
            assert if_match == f"{key}-{self.revisions[key]}"
            self.values[key] = data
            self.revisions[key] += 1

    old_versions = {dataset: f"old-{dataset}" for dataset in script.WRITE_DATASETS}
    planned = {dataset: f"new-{dataset}" for dataset in script.WRITE_DATASETS}
    old_release = DataRelease(
        "old-release", "now", script.FETCH_END, old_versions
    ).to_bytes()
    new_release = DataRelease(
        "new-release", "later", script.FETCH_END, planned
    ).to_bytes()
    old_pointers = {
        dataset: CurrentPointer(
            dataset,
            old_versions[dataset],
            f"old/{dataset}",
            "a" * 64,
            "now",
        ).to_bytes()
        for dataset in script.WRITE_DATASETS
    }
    values = {"releases/current.json": new_release}
    values.update(
        {
            f"datasets/{dataset}/current.json": CurrentPointer(
                dataset,
                planned[dataset],
                f"new/{dataset}",
                "b" * 64,
                "later",
            ).to_bytes()
            for dataset in script.WRITE_DATASETS
        }
    )

    class Repository:
        def __init__(self):
            self.objects = Objects(values)

        @staticmethod
        def current_key(dataset: str) -> str:
            return f"datasets/{dataset}/current.json"

    repository = Repository()
    errors = script._restore_transaction_state(
        repository,
        old_release_bytes=old_release,
        old_pointer_bytes=old_pointers,
        planned_versions=planned,
        committed_release_version="new-release",
    )
    assert errors == ()
    assert repository.objects.values["releases/current.json"] == old_release
    for dataset in script.WRITE_DATASETS:
        assert repository.objects.values[repository.current_key(dataset)] == old_pointers[dataset]


def test_already_applied_path_is_idempotent_and_detects_release_cas() -> None:
    release = _release()

    class Repository:
        def __init__(self):
            self.etag = "release-etag"
            self.write_calls = 0

        def current_release(self):
            return release, self.etag

    repository = Repository()
    prepared = script.PreparedRepair(
        release=release,
        release_etag="release-etag",
        pointer_etags={},
        frames={},
        archive_artifacts=(),
        warnings=(),
        summary={"status": "already_applied"},
    )
    first = script.apply_repair(repository, prepared)
    second = script.apply_repair(repository, prepared)
    assert first["writes_performed"] is False
    assert second["status"] == "already_applied"
    repository.etag = "changed"
    with pytest.raises(RuntimeError, match="Current release changed"):
        script.apply_repair(repository, prepared)
