from __future__ import annotations

import importlib.util
import io
import json
import shutil
import sys
import urllib.request
import urllib.response
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import exchange_calendars as xcals
import pandas as pd
import pytest
import yaml

from supertrend_quant.market_store.ingest import EodhdCallBudget, SourceArtifact
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    DatasetManifest,
)
from supertrend_quant.market_store.official_lifecycle_evidence import (
    OFFICIAL_EXCEPTION_EVIDENCE_URL_ALLOWLIST,
)
from supertrend_quant.market_store.repository import (
    DatasetWriteResult,
    LocalDatasetRepository,
)
from supertrend_quant.market_store.storage import LocalObjectStore
from supertrend_quant.market_store.validation import ValidationReport
from supertrend_quant.ledger import PortfolioLedger
from supertrend_quant.portfolio import Position


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2] / "data/cache"
REAL_OFFICIAL_DIR = (
    REPOSITORY_ROOT / "state/issuer_lifecycle/ntco_ntcoy_transition/official"
)
REAL_BNY_TERMINATION_DIR = Path("tmp/pdfs/ntco_bny_ad1140774")
REAL_BNY_BOOKS_CLOSED_DIR = Path("tmp/pdfs/ntco_bny_books_closed")


def _load(name: str):  # type: ignore[no-untyped-def]
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load("plan_us_ntco_ntcoy_transition")
legacy = _load("repair_us_ntco_nyse_boundary")


def _tree(path: Path) -> dict[str, bytes]:
    if not path.exists():
        return {}
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _content(source_key: str) -> bytes:
    if source_key in {"cboe", "bny", "bny_termination", "bny_books_closed"}:
        return f"%PDF-1.7\nfixture-{source_key}\n%%EOF".encode()
    return b"<html><body>OCC memo 54105 fixture</body></html>"


def _price(session: str, close: str = "10") -> dict[str, object]:
    return {
        "date": session,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 100,
    }


def _dividend(session: str, value: str) -> dict[str, object]:
    return {"date": session, "value": value}


def _real_official_artifacts() -> tuple[SourceArtifact, ...]:
    return script._official_artifacts(REAL_OFFICIAL_DIR)


def _actual_overlap() -> tuple[
    LocalDatasetRepository,
    DataRelease,
    list[dict[str, object]],
    list[dict[str, object]],
]:
    repository = LocalDatasetRepository(REPOSITORY_ROOT)
    release, _ = repository.current_release()
    assert release is not None
    prices, dividends = script.current_overlap_records(repository, release)
    return repository, release, list(prices), list(dividends)


def _provider_raw_artifacts(
    current_prices: list[dict[str, object]],
    current_dividends: list[dict[str, object]],
) -> tuple[SourceArtifact, SourceArtifact, SourceArtifact]:
    by_date = {str(row["date"]): dict(row) for row in current_prices}
    template = dict(current_prices[-1])
    for timestamp in xcals.get_calendar("XNYS").sessions_in_range(
        "2024-04-15", "2024-09-03"
    ):
        session = timestamp.date().isoformat()
        row = dict(template)
        row["date"] = session
        by_date[session] = row
    prices = [by_date[key] for key in sorted(by_date)]
    rows = (prices, current_dividends, [])
    retrieved_at = "2026-07-19T00:00:00Z"
    return tuple(
        SourceArtifact(
            source=f"eodhd_{endpoint}",
            source_url=script.EODHD_REQUEST_URLS[endpoint],
            retrieved_at=retrieved_at,
            content=json.dumps(value, sort_keys=True, separators=(",", ":")).encode(),
            content_type="application/json",
        )
        for endpoint, value in zip(script.EODHD_ENDPOINTS, rows, strict=True)
    )  # type: ignore[return-value]


def _receipt(*, before: int = 10) -> dict[str, object]:
    return {
        "schema": "eodhd_budget_receipt/v2",
        "period": "2026-07-19",
        "used_before": before,
        "used_after": before + 3,
        "delta": 3,
        "own_claim_count": 3,
        "claim_positions": [before + 1, before + 2, before + 3],
        "daily_limit": 100000,
        "reserve": 5000,
        "safety_ceiling": 95000,
    }


def _write_pins(
    path: Path,
    artifacts: tuple[SourceArtifact, ...],
    *,
    include_provider: bool,
) -> Path:
    document = yaml.safe_load(script.DEFAULT_PINS.read_text(encoding="utf-8"))
    by_url = {item.source_url: item.source_hash for item in artifacts}
    for key, url in script.OFFICIAL_URLS.items():
        document["official_sources"][key]["source_sha256"] = by_url[url]
    for endpoint, url in script.EODHD_REQUEST_URLS.items():
        document["provider"]["requests"][endpoint]["source_sha256"] = (
            by_url[url] if include_provider else ""
        )
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _stage_real_official(cache_root: Path) -> tuple[SourceArtifact, ...]:
    evidence_dir = cache_root / script.STATE_DIR / "official"
    artifacts = _real_official_artifacts()
    by_url = {item.source_url: item for item in artifacts}
    for key, url in script.OFFICIAL_URLS.items():
        script.fetch_official(
            evidence_dir,
            key,
            user_agent="Researcher researcher@example.com",
            fetcher=lambda requested, _agent, expected=url: by_url[expected].content
            if requested == expected
            else pytest.fail("wrong official URL"),
        )
    return artifacts


def _reviewed_fixture_bundle() -> tuple[
    LocalDatasetRepository,
    DataRelease,
    script.ReviewedBundle,
    tuple[SourceArtifact, ...],
]:
    repository, release, current_prices, current_dividends = _actual_overlap()
    artifacts = (*_real_official_artifacts(), *_provider_raw_artifacts(current_prices, current_dividends))
    bundle = script.bundle_from_artifacts(
        artifacts,
        current_prices=current_prices,
        current_dividends=current_dividends,
        budget_receipt=_receipt(),
        base_release_version=release.version,
    )
    return repository, release, bundle, artifacts


def test_pin_contract_and_transition_model_are_exact() -> None:
    document = script.validate_pin_contract()

    assert set(document["official_sources"]) == {"cboe", "occ", "bny"}
    assert {
        key: document["official_sources"][key]["source_sha256"]
        for key in script.OFFICIAL_URLS
    } == {
        "cboe": "e67d8046a4c532daa11dac1faacb9e573238812c90d15be9792b5d6a46e67928",
        "occ": "5eb3c0058195d6d6c22361e362f1623c4eae7dd7262e9497ef7b53f8a564e913",
        "bny": "057a95798bd3b9ff4f0f0be6fd34d7ed42d6ad1d27a12ad1c41e30700c8da28b",
    }
    assert {
        key: document["supplemental_official_sources"][key]["source_sha256"]
        for key in script.SUPPLEMENTAL_OFFICIAL_URLS
    } == {
        "bny_termination": "abc23f86378f82f5efb475fc23e0a82e398db4201dc7e52fd4064d088eb47b83",
        "bny_books_closed": "b5d449fcad19cd38fb461c8f9bae4c1856c0fb7ccd4903e062f87840315ba675",
    }
    assert document["provider"]["provider_symbol"] == "NTCOY.US"
    assert document["provider"]["max_http_attempts"] == 3
    assert list(document["provider"]["requests"]) == ["eod", "div", "splits"]

    model = script.transition_model()
    assert model["identity_policy"] == "same_security_id"
    assert model["ticker_change"] == {
        "event_id": script.TICKER_CHANGE_EVENT_ID,
        "action_type": "ticker_change",
        "effective_date": "2024-02-12",
        "old_symbol": "NTCO",
        "new_symbol": "NTCOY",
        "new_security_id": script.SECURITY_ID,
        "new_exchange": "OTC",
        "official_destination_market": "Other-OTC",
        "cash_amount": None,
        "ratio": None,
        "official_source_keys": ["cboe", "occ"],
        "official_identity_terms": {
            "cusip": "63884N108",
            "deliverable": "100 American Depositary Shares",
        },
    }
    assert model["terminal"]["action_type"] == "delisting"
    assert model["terminal"]["effective_date"] == "2024-09-04"
    assert model["terminal"]["cash_amount"] == "5.043659"
    assert model["terminal"]["currency"] == "USD"
    assert model["tradability_boundary"] == {
        "last_trade_session": "2024-08-07",
        "facility_termination_time": "5:00 PM ET",
        "next_state": "pending_cash_conversion",
        "forward_fill_allowed": False,
        "primary_official_source_key": "bny_termination",
        "corroborating_official_source_key": "bny_books_closed",
    }


def test_default_plan_is_read_only_and_lists_all_six_future_requests(
    tmp_path: Path,
) -> None:
    before = _tree(tmp_path)
    result = script.readiness_plan(evidence_dir=tmp_path)

    assert result["status"] == "blocked_pending_evidence"
    assert result["apply_allowed"] is False
    assert result["network_accessed"] is False
    assert result["writes_performed"] is False
    assert result["eodhd_calls_this_run"] == 0
    assert result["eodhd_future_call_cap"] == 3
    assert set(result["official_sources"]) == {"cboe", "occ", "bny"}
    assert [row["endpoint"] for row in result["eodhd_requests"]] == [
        "eod",
        "div",
        "splits",
    ]
    assert all(row["provider_symbol"] == "NTCOY.US" for row in result["eodhd_requests"])
    assert all(
        row["params"] == {"from": "2024-02-12", "to": "2024-09-03"}
        for row in result["eodhd_requests"]
    )
    urls = {
        *[row["source_url"] for row in result["eodhd_requests"]],
        *[row["source_url"] for row in result["official_sources"].values()],
    }
    assert urls == {*script.OFFICIAL_URLS.values(), *script.EODHD_REQUEST_URLS.values()}
    inventory = result["six_request_inventory"]
    assert [row["order"] for row in inventory] == [1, 2, 3, 4, 5, 6]
    assert [row["source_key"] for row in inventory] == [
        "cboe",
        "occ",
        "bny",
        "eod",
        "div",
        "splits",
    ]
    assert result["current_ntco_tail"]["price_rows"] == 43
    assert result["current_ntco_tail"]["dividend_rows"] == 2
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("source_key", ("cboe", "occ", "bny"))
def test_each_official_fetch_is_one_exact_attempt_then_cache_replay(
    tmp_path: Path,
    source_key: str,
) -> None:
    calls: list[tuple[str, str]] = []

    def fetcher(url: str, user_agent: str) -> bytes:
        calls.append((url, user_agent))
        return _content(source_key)

    first = script.fetch_official(
        tmp_path,
        source_key,
        user_agent="Researcher researcher@example.com",
        fetcher=fetcher,
    )
    second = script.fetch_official(
        tmp_path,
        source_key,
        user_agent="Researcher researcher@example.com",
        fetcher=lambda *_args: pytest.fail("cache replay attempted network"),
    )

    assert calls == [
        (script.OFFICIAL_URLS[source_key], "Researcher researcher@example.com")
    ]
    assert first["source_url"] == script.OFFICIAL_URLS[source_key]
    assert first["http_attempts_this_run"] == 1
    assert second["http_attempts_this_run"] == 0
    staged = script.verify_staged_official(tmp_path, source_key)
    assert staged is not None
    assert staged.content == _content(source_key)


def test_fetch_requires_contact_before_attempt_and_does_not_retry(tmp_path: Path) -> None:
    attempts = 0

    def fetcher(_key: str, _user_agent: str) -> bytes:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("one-shot failure")

    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        script.fetch_official(
            tmp_path,
            "cboe",
            user_agent="missing-contact",
            fetcher=fetcher,
        )
    assert attempts == 0
    assert _tree(tmp_path) == {}

    with pytest.raises(RuntimeError, match="one-shot failure"):
        script.fetch_official(
            tmp_path,
            "cboe",
            user_agent="Researcher researcher@example.com",
            fetcher=fetcher,
        )
    assert attempts == 1
    assert _tree(tmp_path) == {".locks/cboe.lock": b""}


def test_no_redirect_handler_stops_before_hidden_second_request() -> None:
    attempts: list[str] = []

    class RedirectingTransport(urllib.request.BaseHandler):
        handler_order = 100

        def https_open(self, request: urllib.request.Request):  # type: ignore[no-untyped-def]
            attempts.append(request.full_url)
            headers = Message()
            headers["Location"] = request.full_url + "&redirected=1"
            response = urllib.response.addinfourl(
                io.BytesIO(b"redirect"), headers, request.full_url, 302
            )
            response.msg = "Found"
            return response

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        script._NoRedirectHandler(),
        RedirectingTransport(),
    )
    request = urllib.request.Request(script.OFFICIAL_URLS["occ"])

    with pytest.raises(RuntimeError, match="automatic follow-up requests are disabled"):
        opener.open(request, timeout=1)
    assert attempts == [script.OFFICIAL_URLS["occ"]]


def test_staged_hash_cannot_self_approve_and_tampering_is_detected(tmp_path: Path) -> None:
    script.fetch_official(
        tmp_path,
        "bny",
        user_agent="Researcher researcher@example.com",
        fetcher=lambda _url, _agent: _content("bny"),
    )
    document = yaml.safe_load(script.DEFAULT_PINS.read_text(encoding="utf-8"))
    document["official_sources"]["bny"]["source_sha256"] = ""
    pins = tmp_path / "pins.yaml"
    pins.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    plan = script.readiness_plan(evidence_dir=tmp_path, pins_path=pins)
    assert plan["official_sources"]["bny"]["state"] == "pending_reviewer_pin"
    assert "official_pin_missing:bny" in plan["blockers"]

    payload = tmp_path / script.OFFICIAL_FILENAMES["bny"]
    payload.write_bytes(payload.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash/size changed"):
        script.verify_staged_official(tmp_path, "bny")


def test_apply_without_a_release_or_promoted_bundle_fails_before_writes(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="current release"):
        script.main(["--cache-root", str(tmp_path), "--apply"])
    assert _tree(tmp_path) == {}


def test_exact_overlap_is_eligible_for_transaction_design_but_not_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_prices = [_price("2024-02-12", "10"), _price("2024-02-13", "11")]
    current_dividends = [_dividend("2024-03-21", "0.28427")]
    provider_prices = [*current_prices, _price("2024-02-14", "12")]
    provider_dividends = [*current_dividends, _dividend("2024-04-09", "0.01099")]
    monkeypatch.setattr(script, "CURRENT_OVERLAP_PRICE_ROWS", len(current_prices))
    monkeypatch.setattr(
        script, "CURRENT_OVERLAP_PRICE_SHA256", script._canonical_sha256(current_prices)
    )
    monkeypatch.setattr(script, "CURRENT_OVERLAP_FIRST_SESSION", "2024-02-12")
    monkeypatch.setattr(script, "CURRENT_OVERLAP_LAST_SESSION", "2024-02-13")
    monkeypatch.setattr(script, "CURRENT_OVERLAP_DIVIDEND_ROWS", len(current_dividends))
    monkeypatch.setattr(
        script,
        "CURRENT_OVERLAP_DIVIDEND_SHA256",
        script._canonical_sha256(current_dividends),
    )
    monkeypatch.setattr(script, "MIN_PROVIDER_PRICE_ROWS", len(provider_prices))
    monkeypatch.setattr(script, "MIN_PROVIDER_TERMINAL_SESSION", "2024-02-14")

    result = script.assess_provider_overlap(
        current_prices=current_prices,
        current_dividends=current_dividends,
        ntcoy_prices=provider_prices,
        ntcoy_dividends=provider_dividends,
        ntcoy_splits=[],
    )

    assert result["status"] == "ready_for_transaction_design_review"
    assert result["apply_allowed"] is False
    assert result["blockers"] == []
    assert result["price_overlap_rows"] == 2
    assert result["dividend_overlap_rows"] == 1
    assert result["transition_model"]["terminal"]["action_type"] == "delisting"


def test_any_overlap_mismatch_or_split_blocks_and_quarantines_both_bundles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_prices = [_price("2024-02-12", "10"), _price("2024-02-13", "11")]
    provider_prices = [_price("2024-02-12", "10"), _price("2024-02-13", "12")]
    current_dividends = [_dividend("2024-03-21", "0.28427")]
    provider_dividends = [_dividend("2024-03-21", "0.3")]
    monkeypatch.setattr(script, "CURRENT_OVERLAP_PRICE_ROWS", len(current_prices))
    monkeypatch.setattr(
        script, "CURRENT_OVERLAP_PRICE_SHA256", script._canonical_sha256(current_prices)
    )
    monkeypatch.setattr(script, "CURRENT_OVERLAP_FIRST_SESSION", "2024-02-12")
    monkeypatch.setattr(script, "CURRENT_OVERLAP_LAST_SESSION", "2024-02-13")
    monkeypatch.setattr(script, "CURRENT_OVERLAP_DIVIDEND_ROWS", len(current_dividends))
    monkeypatch.setattr(
        script,
        "CURRENT_OVERLAP_DIVIDEND_SHA256",
        script._canonical_sha256(current_dividends),
    )
    monkeypatch.setattr(script, "MIN_PROVIDER_PRICE_ROWS", len(provider_prices))
    monkeypatch.setattr(script, "MIN_PROVIDER_TERMINAL_SESSION", "2024-02-13")

    result = script.assess_provider_overlap(
        current_prices=current_prices,
        current_dividends=current_dividends,
        ntcoy_prices=provider_prices,
        ntcoy_dividends=provider_dividends,
        ntcoy_splits=[{"date": "2024-03-01", "split": "2/1"}],
    )

    assert result["status"] == "blocked_provider_mismatch"
    assert result["apply_allowed"] is False
    assert result["price_overlap_mismatch_sessions"] == ["2024-02-13"]
    assert result["dividend_overlap_mismatch_sessions"] == ["2024-03-21"]
    assert "ntcoy_split_requires_separate_official_review" in result["blockers"]
    assert result["accepted_overlap_policy"].startswith("none; quarantine")


def test_provider_window_and_official_terminal_cash_cannot_be_blended_as_dividend() -> None:
    result = script.assess_provider_overlap(
        current_prices=[],
        current_dividends=[],
        ntcoy_prices=[_price("2024-02-13")],
        ntcoy_dividends=[_dividend("2024-09-04", "5.043659")],
        ntcoy_splits=[],
    )

    assert "ntcoy_first_session_is_not_transition_date" in result["blockers"]
    assert "provider_dividend_on_or_after_official_cash_termination" in result["blockers"]
    assert result["transition_model"]["terminal"]["cash_amount"] == "5.043659"


def test_superseded_nyse_only_entrypoints_never_fetch_or_apply(tmp_path: Path) -> None:
    attempts = 0

    def fetcher(_url: str, _agent: str) -> bytes:
        nonlocal attempts
        attempts += 1
        return b"should-not-run"

    result = legacy.readiness_plan(SimpleNamespace())
    assert result["status"] == "superseded_do_not_apply"
    with pytest.raises(RuntimeError, match="permanently disabled"):
        legacy.fetch_official(
            tmp_path,
            user_agent="Researcher researcher@example.com",
            fetcher=fetcher,
        )
    with pytest.raises(RuntimeError, match="permanently disabled"):
        legacy.prepare_repair(SimpleNamespace())
    with pytest.raises(RuntimeError, match="permanently disabled"):
        legacy.apply_repair(SimpleNamespace(), None)
    assert attempts == 0
    assert _tree(tmp_path) == {}


def test_planned_ticker_change_and_cash_delisting_are_ledger_complete() -> None:
    ledger = PortfolioLedger(
        cash=0.0,
        positions={"NTCO": Position("NTCO", 10.0, 6.5)},
    )
    model = script.transition_model()
    ticker = {
        **model["ticker_change"],
        "symbol": "NTCO",
    }
    terminal = {
        **model["terminal"],
        "symbol": "NTCOY",
        "cash_amount": float(model["terminal"]["cash_amount"]),
    }

    ticker_events = ledger.apply_actions((ticker,), through="2024-02-12")
    terminal_events = ledger.apply_actions((terminal,), through="2024-09-04")

    assert len(ticker_events) == 1
    assert set(ledger.positions) == set()
    assert len(terminal_events) == 1
    assert terminal_events[0].cash_delta == pytest.approx(50.43659)
    assert ledger.cash == pytest.approx(50.43659)
    assert ledger.unresolved_event_ids == set()


def test_stale_ntco_unsupported_exception_was_removed_from_shared_registry() -> None:
    assert "ntco_2024_nyse_ads_delisting" not in OFFICIAL_EXCEPTION_EVIDENCE_URL_ALLOWLIST


def test_real_official_caches_match_reviewer_pins_and_required_claims() -> None:
    artifacts = _real_official_artifacts()
    document = script.validate_pin_contract()

    pins = script._validate_artifact_pins(
        artifacts, document, official_only=True
    )
    claims = script._validate_official_semantics(artifacts)

    assert pins == {
        script.OFFICIAL_URLS[key]: document["official_sources"][key]["source_sha256"]
        for key in script.OFFICIAL_URLS
    }
    assert claims["cboe"]["new_symbol"] == "NTCOY"
    assert claims["occ"]["cusip"] == "63884N108"
    assert claims["bny"]["net_cash_usd_per_ads"] == "5.043659"


def test_bny_boundary_raws_match_exact_pins_and_semantics() -> None:
    document = script.validate_pin_contract()
    staged = {
        "bny_termination": script._supplemental_official_artifact(
            REAL_BNY_TERMINATION_DIR,
            source_key="bny_termination",
        ),
        "bny_books_closed": script._supplemental_official_artifact(
            REAL_BNY_BOOKS_CLOSED_DIR,
            source_key="bny_books_closed",
        ),
    }

    termination = script._validate_supplemental_official_semantics(
        staged["bny_termination"],
        source_key="bny_termination",
    )
    books_closed = script._validate_supplemental_official_semantics(
        staged["bny_books_closed"],
        source_key="bny_books_closed",
    )
    for key, artifact in staged.items():
        assert script._validate_supplemental_artifact_pin(
            artifact,
            document,
            source_key=key,
        ) == document["supplemental_official_sources"][key]["source_sha256"]

    assert termination["termination_date"] == "2024-08-07"
    assert termination["termination_time"] == "5:00 PM ET"
    assert books_closed["ads_symbol"] == "NTCOY"
    assert books_closed["cusip"] == "63884N108"
    assert books_closed["issuance_close_date"] == "2024-08-08"
    assert books_closed["cancellation_close_date"] == "2024-08-13"


def test_eodhd_stage_uses_exact_three_budget_claims_and_never_self_promotes(
    tmp_path: Path,
) -> None:
    official = _stage_real_official(tmp_path)
    _, _, current_prices, current_dividends = _actual_overlap()
    provider = _provider_raw_artifacts(current_prices, current_dividends)
    pins = _write_pins(tmp_path / "pins.yaml", (*official, *provider), include_provider=False)
    budget = EodhdCallBudget(
        tmp_path / "budget.json",
        limit=100,
        reserve=5,
        seed_used=10,
        period="2026-07-19",
    )

    class FakeClient:
        def __init__(self, *, budget: EodhdCallBudget):
            self.budget = budget
            self.attempted_endpoints: list[str] = []
            self.claim_positions: list[int] = []

        def get_raw_artifact(self, endpoint: str, *, params, retrieved_at):  # type: ignore[no-untyped-def]
            position = len(self.attempted_endpoints)
            expected = f"{script.EODHD_ENDPOINTS[position]}/{script.PROVIDER_SYMBOL}"
            assert endpoint == expected
            assert params == script.EODHD_REQUEST_PARAMS
            claim = self.budget.claim()
            self.claim_positions.append(claim)
            self.attempted_endpoints.append(script.EODHD_ENDPOINTS[position])
            item = provider[position]
            return SourceArtifact(
                source=item.source,
                source_url=item.source_url,
                retrieved_at=retrieved_at,
                content=item.content,
                content_type=item.content_type,
            )

    result = script.collect_eodhd_stage(
        tmp_path,
        pins_path=pins,
        client_factory=FakeClient,
        budget_factory=lambda: budget,
    )

    assert result["status"] == "eodhd_stage_fetched_needs_reviewer_pins"
    assert result["eodhd_http_attempts_this_run"] == 3
    assert result["budget_receipt"]["claim_positions"] == [11, 12, 13]
    quarantine = script.read_quarantine(tmp_path, result["quarantine_id"])
    assert [item.source_url for item in quarantine.artifacts] == [
        *script.OFFICIAL_URLS.values(),
        *script.EODHD_REQUEST_URLS.values(),
    ]
    assert not script._reviewed_cache_path(tmp_path).exists()


def test_exact_client_is_one_shot_no_redirect_and_refuses_unreviewed_calls(
    tmp_path: Path,
) -> None:
    class Response:
        def __init__(self, status: int, content: bytes):
            self.status_code = status
            self.content = content
            self.headers = {"Content-Type": "application/json"}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    class Session:
        def __init__(self):
            self.responses = [
                Response(200, b"[]"),
                Response(302, b"redirect"),
                Response(200, b"[]"),
            ]
            self.calls: list[tuple[str, dict[str, object]]] = []

        def get(self, url: str, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append((url, kwargs))
            return self.responses[len(self.calls) - 1]

    session = Session()
    budget = EodhdCallBudget(
        tmp_path / "budget.json",
        limit=100,
        reserve=5,
        seed_used=0,
        period="2026-07-19",
    )
    client = script.ExactNtcoyEodhdClient(
        session=session,
        token="secret-token",
        budget=budget,
    )

    with pytest.raises(RuntimeError, match="non-reviewed provider request"):
        client.get_raw_artifact(
            f"div/{script.PROVIDER_SYMBOL}",
            params=script.EODHD_REQUEST_PARAMS,
            retrieved_at="2026-07-19T00:00:00Z",
        )
    assert not (tmp_path / "budget.json").exists()

    first = client.get_raw_artifact(
        f"eod/{script.PROVIDER_SYMBOL}",
        params=script.EODHD_REQUEST_PARAMS,
        retrieved_at="2026-07-19T00:00:00Z",
    )
    assert first.source_url == script.EODHD_REQUEST_URLS["eod"]
    with pytest.raises(RuntimeError, match="forbidden redirect HTTP 302"):
        client.get_raw_artifact(
            f"div/{script.PROVIDER_SYMBOL}",
            params=script.EODHD_REQUEST_PARAMS,
            retrieved_at="2026-07-19T00:00:00Z",
        )
    client.get_raw_artifact(
        f"splits/{script.PROVIDER_SYMBOL}",
        params=script.EODHD_REQUEST_PARAMS,
        retrieved_at="2026-07-19T00:00:00Z",
    )
    with pytest.raises(RuntimeError, match="fourth EODHD request"):
        client.get_raw_artifact(
            f"eod/{script.PROVIDER_SYMBOL}",
            params=script.EODHD_REQUEST_PARAMS,
            retrieved_at="2026-07-19T00:00:00Z",
        )

    assert client.claim_positions == [1, 2, 3]
    assert len(session.calls) == 3
    for position, (url, kwargs) in enumerate(session.calls):
        assert url == (
            f"https://eodhd.com/api/{script.EODHD_ENDPOINTS[position]}/"
            f"{script.PROVIDER_SYMBOL}"
        )
        assert kwargs["allow_redirects"] is False
        assert kwargs["params"] == {
            **script.EODHD_REQUEST_PARAMS,
            "api_token": "secret-token",
            "fmt": "json",
        }
        assert "secret-token" not in script.EODHD_REQUEST_URLS[
            script.EODHD_ENDPOINTS[position]
        ]


def test_stage_lock_is_acquired_before_budget_preflight(tmp_path: Path) -> None:
    official = _stage_real_official(tmp_path)
    pins = _write_pins(tmp_path / "pins.yaml", official, include_provider=False)
    budget_constructed = False

    def budget_factory():  # type: ignore[no-untyped-def]
        nonlocal budget_constructed
        budget_constructed = True
        raise AssertionError("budget must not be observed outside the stage lock")

    lock_path = tmp_path / script.STATE_DIR / ".locks/eodhd-stage.lock"
    with script._exclusive_file_lock(lock_path, label="test"):
        with pytest.raises(RuntimeError, match="lock is already held"):
            script.collect_eodhd_stage(
                tmp_path,
                pins_path=pins,
                budget_factory=budget_factory,
            )
    assert budget_constructed is False


def test_failed_second_provider_call_is_counted_once_and_partial_raws_are_quarantined(
    tmp_path: Path,
) -> None:
    official = _stage_real_official(tmp_path)
    _, _, current_prices, current_dividends = _actual_overlap()
    provider = _provider_raw_artifacts(current_prices, current_dividends)
    pins = _write_pins(tmp_path / "pins.yaml", (*official, *provider), include_provider=False)
    budget = EodhdCallBudget(
        tmp_path / "budget.json",
        limit=100,
        reserve=5,
        seed_used=20,
        period="2026-07-19",
    )

    class FailingClient:
        def __init__(self, *, budget: EodhdCallBudget):
            self.budget = budget
            self.attempted_endpoints: list[str] = []
            self.claim_positions: list[int] = []

        def get_raw_artifact(self, endpoint: str, *, params, retrieved_at):  # type: ignore[no-untyped-def]
            del endpoint, params
            position = len(self.attempted_endpoints)
            claim = self.budget.claim()
            self.claim_positions.append(claim)
            self.attempted_endpoints.append(script.EODHD_ENDPOINTS[position])
            if position == 1:
                raise RuntimeError("second call failed")
            item = provider[position]
            return SourceArtifact(
                source=item.source,
                source_url=item.source_url,
                retrieved_at=retrieved_at,
                content=item.content,
                content_type=item.content_type,
            )

    with pytest.raises(RuntimeError, match="second call failed"):
        script.collect_eodhd_stage(
            tmp_path,
            pins_path=pins,
            client_factory=FailingClient,
            budget_factory=lambda: budget,
        )

    state = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert state["used"] == 22
    quarantine_files = list((tmp_path / script.QUARANTINE_DIR).glob("*.json.gz"))
    assert len(quarantine_files) == 1
    import gzip

    envelope = json.loads(gzip.decompress(quarantine_files[0].read_bytes()))
    assert envelope["status"] == "incomplete"
    assert envelope["budget_receipt"]["own_claim_count"] == 2
    assert len(envelope["artifacts"]) == 4  # three official + completed EOD
    assert not script._reviewed_cache_path(tmp_path).exists()


def test_budget_preflight_refuses_all_provider_calls_before_partial_acquisition(
    tmp_path: Path,
) -> None:
    official = _stage_real_official(tmp_path)
    _, _, current_prices, current_dividends = _actual_overlap()
    provider = _provider_raw_artifacts(current_prices, current_dividends)
    pins = _write_pins(tmp_path / "pins.yaml", (*official, *provider), include_provider=False)
    budget = EodhdCallBudget(
        tmp_path / "budget.json",
        limit=100,
        reserve=5,
        seed_used=93,
        period="2026-07-19",
    )
    constructed = 0

    def factory(**_kwargs):  # type: ignore[no-untyped-def]
        nonlocal constructed
        constructed += 1
        raise AssertionError("client must not be constructed")

    with pytest.raises(RuntimeError, match="refused a partial three-call"):
        script.collect_eodhd_stage(
            tmp_path,
            pins_path=pins,
            client_factory=factory,
            budget_factory=lambda: budget,
        )
    assert constructed == 0
    assert not (tmp_path / "budget.json").exists()
    assert not (tmp_path / script.QUARANTINE_DIR).exists()


def test_promotion_requires_all_six_pins_and_is_tamper_evident(tmp_path: Path) -> None:
    repository, release, current_prices, current_dividends = _actual_overlap()
    artifacts = (*_real_official_artifacts(), *_provider_raw_artifacts(current_prices, current_dividends))
    quarantine_id, quarantine_path = script._write_quarantine(
        tmp_path,
        artifacts,
        _receipt(),
        status="complete_unreviewed",
    )
    pending = _write_pins(tmp_path / "pending.yaml", artifacts, include_provider=False)
    with pytest.raises(ValueError, match="pins are pending"):
        script.promote_quarantine(
            tmp_path,
            quarantine_id,
            pins_path=pending,
            repository=repository,
        )
    assert not script._reviewed_cache_path(tmp_path).exists()

    reviewed = _write_pins(tmp_path / "reviewed.yaml", artifacts, include_provider=True)
    result = script.promote_quarantine(
        tmp_path,
        quarantine_id,
        pins_path=reviewed,
        repository=repository,
    )
    assert result["status"] == "reviewed_bundle_promoted"
    assert result["provider_last_session"] == "2024-09-03"
    assert script._reviewed_cache_path(tmp_path).is_file()

    quarantine_path.write_bytes(quarantine_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="unreadable"):
        script.read_quarantine(tmp_path, quarantine_id)


def test_provider_overlap_mismatch_and_nonempty_splits_cannot_promote(tmp_path: Path) -> None:
    repository, _, current_prices, current_dividends = _actual_overlap()
    provider = list(_provider_raw_artifacts(current_prices, current_dividends))
    rows = json.loads(provider[0].content)
    rows[0]["close"] = float(rows[0]["close"]) + 1.0
    provider[0] = SourceArtifact(
        source=provider[0].source,
        source_url=provider[0].source_url,
        retrieved_at=provider[0].retrieved_at,
        content=json.dumps(rows, sort_keys=True, separators=(",", ":")).encode(),
        content_type="application/json",
    )
    provider[2] = SourceArtifact(
        source=provider[2].source,
        source_url=provider[2].source_url,
        retrieved_at=provider[2].retrieved_at,
        content=b'[{"date":"2024-03-01","split":"2/1"}]',
        content_type="application/json",
    )
    artifacts = (*_real_official_artifacts(), *provider)
    quarantine_id, _ = script._write_quarantine(
        tmp_path, artifacts, _receipt(), status="complete_unreviewed"
    )
    pins = _write_pins(tmp_path / "pins.yaml", artifacts, include_provider=True)

    with pytest.raises(ValueError, match="overlap validation failed"):
        script.promote_quarantine(
            tmp_path,
            quarantine_id,
            pins_path=pins,
            repository=repository,
        )
    assert not script._reviewed_cache_path(tmp_path).exists()


def test_actual_quarantine_is_exactly_profiled_for_price_only_promotion() -> None:
    repository, release, _, _ = _actual_overlap()
    control_keys = ["releases/current.json"] + [
        repository.current_key(dataset) for dataset in script.REQUIRED_DATASETS
    ]
    before = {key: repository.objects.get(key).data for key in control_keys}

    report = script.assess_quarantine_decision(
        REPOSITORY_ROOT,
        script.OBSERVED_UNREVIEWED_QUARANTINE_ID,
        repository=repository,
    )

    assert report["base_release_version"] == release.version
    assert report["status"] == "ready_for_price_identity_terminal_review"
    assert report["blockers"] == []
    assert report["decision_mode"] == script.PRICE_IDENTITY_TERMINAL_ONLY
    assert report["provider_raw_sha256"] == dict(
        script.OBSERVED_PROVIDER_RAW_SHA256
    )
    assert report["field_overlap_exact_rows"] == {
        "open": 43,
        "high": 41,
        "low": 41,
        "close": 43,
        "adjusted_close": 43,
        "volume": 0,
    }
    inventory = report["difference_inventory"]
    assert inventory["high_low"] == script._expected_high_low_diff_inventory()
    assert inventory["high_low_sha256"] == script.OBSERVED_HIGH_LOW_DIFF_SHA256
    assert inventory["volume"] == script._expected_volume_diff_inventory()
    assert inventory["volume_sha256"] == script.OBSERVED_VOLUME_DIFF_SHA256
    assert inventory["dividends"] == script._expected_dividend_diff_inventory()
    assert inventory["dividends_sha256"] == script.OBSERVED_DIVIDEND_DIFF_SHA256
    assert report["dividend_ambiguity"] == {
        "classification": "exact_provider_alias_conflict_rejected",
        "automatic_selection_allowed": False,
        "provider_economics_accepted": False,
        "policy": script.REJECTED_DIVIDEND_CONFLICT_POLICY,
        "preserved_event_ids": sorted(script.CURRENT_DIVIDEND_EVENT_IDS),
        "maximum_absolute_sensitivity_usd_per_ads": "0.01585",
        "records": [
            {
                "session": "2024-03-21",
                "ntco_alias_amount": "0.28427",
                "ntcoy_canonical_amount": "0.27036",
                "accepted_amount": "0.28427",
                "accepted_source": "preserved_current_ntco_corporate_action",
                "rejected_amount": "0.27036",
                "rejection_reason": "conflicts_with_preserved_ntco_economics",
            },
            {
                "session": "2024-04-09",
                "ntco_alias_amount": "0.01099",
                "ntcoy_canonical_amount": "0.01293",
                "accepted_amount": "0.01099",
                "accepted_source": "preserved_current_ntco_corporate_action",
                "rejected_amount": "0.01293",
                "rejection_reason": "conflicts_with_preserved_ntco_economics",
            },
        ],
    }
    exceptional = next(
        row for row in inventory["volume"] if row["session"] == "2024-03-25"
    )
    assert exceptional == {
        "session": "2024-03-25",
        "current": 5_026_884,
        "canonical": 10_026_900,
        "delta": 5_000_016,
    }
    assert report["ntcoy_price_rows"] == 123
    assert report["ntcoy_last_session"] == "2024-08-07"
    assert len(
        report["terminal_boundary"][
            "unobserved_xnys_sessions_before_request_end"
        ]
    ) == 18
    assert report["terminal_boundary"][
        "official_cash_conversion_effective_date"
    ] == "2024-09-04"
    assert report["terminal_boundary"]["official_boundary_confirmed"] is True
    assert report["terminal_boundary"][
        "official_facility_termination_session"
    ] == "2024-08-07"
    supplemental = report["supplemental_official_evidence"]
    assert supplemental["boundary_gate_closed"] is True
    assert supplemental["books_closed_corroboration_valid"] is True
    assert supplemental["sources"]["bny_termination"]["source_sha256"] == (
        "abc23f86378f82f5efb475fc23e0a82e398db4201dc7e52fd4064d088eb47b83"
    )
    assert supplemental["sources"]["bny_books_closed"]["source_sha256"] == (
        "b5d449fcad19cd38fb461c8f9bae4c1856c0fb7ccd4903e062f87840315ba675"
    )
    assert report["canonical_price_replacement_candidate"] is True
    assert report["provider_pins_written"] is False
    assert report["provider_pins_validated"] is True
    assert report["promotion_eligible"] is True
    assert report["promotion_performed"] is False
    assert report["apply_allowed"] is False
    scope = report["release_scope_audit"]
    assert scope["release_version"] == release.version
    assert scope["target_row_count"] == 0
    assert scope["absence_proven"] is True
    assert set(scope["datasets"]) == set(script.OUT_OF_SCOPE_CAS_DATASETS)
    assert all(
        row["target_security_rows"] == 0
        for row in scope["datasets"].values()
    )
    assert {key: repository.objects.get(key).data for key in control_keys} == before


@pytest.mark.parametrize("mode", ("missing", "tampered"))
def test_primary_boundary_raw_must_validate_or_last_trade_gate_reopens(
    tmp_path: Path,
    mode: str,
) -> None:
    repository, _, _, _ = _actual_overlap()
    primary_dir = tmp_path / "primary"
    if mode == "tampered":
        primary_dir.mkdir()
        shutil.copy2(
            REAL_BNY_TERMINATION_DIR / "bny_termination.json",
            primary_dir / "bny_termination.json",
        )
        shutil.copy2(
            REAL_BNY_TERMINATION_DIR / "ad1140774.pdf",
            primary_dir / "ad1140774.pdf",
        )
        payload = primary_dir / "ad1140774.pdf"
        payload.write_bytes(payload.read_bytes() + b"tamper")

    report = script.assess_quarantine_decision(
        REPOSITORY_ROOT,
        script.OBSERVED_UNREVIEWED_QUARANTINE_ID,
        repository=repository,
        supplemental_evidence_dir=primary_dir,
        supplemental_books_closed_evidence_dir=REAL_BNY_BOOKS_CLOSED_DIR,
    )

    assert report["terminal_boundary"]["official_boundary_confirmed"] is False
    assert report["blockers"] == [
        "independent_last_trade_boundary_evidence_required",
    ]
    evidence = report["supplemental_official_evidence"]
    assert evidence["sources"]["bny_termination"]["state"] == "missing_or_invalid"
    assert evidence["books_closed_corroboration_valid"] is True


@pytest.mark.parametrize(
    ("target", "expected_blocker"),
    [
        ("adjusted_close", "ntco_ntcoy_ohlcv_overlap_mismatch"),
        ("high", "ntco_ntcoy_ohlcv_overlap_mismatch"),
        ("volume", "ntco_ntcoy_ohlcv_overlap_mismatch"),
        ("dividend", "ntco_ntcoy_dividend_overlap_mismatch"),
        ("last_trade", "ntcoy_terminal_price_too_early"),
    ],
)
def test_observed_profile_tampering_falls_back_to_generic_block(
    target: str,
    expected_blocker: str,
) -> None:
    repository, release, current_prices, current_dividends = _actual_overlap()
    quarantine = script.read_quarantine(
        REPOSITORY_ROOT,
        script.OBSERVED_UNREVIEWED_QUARANTINE_ID,
    )
    prices = json.loads(quarantine.artifacts[3].content)
    dividends = json.loads(quarantine.artifacts[4].content)
    splits = json.loads(quarantine.artifacts[5].content)
    if target == "dividend":
        dividends[0]["value"] += 0.001
        dividends[0]["unadjustedValue"] += 0.001
    elif target == "last_trade":
        prices.append({**prices[-1], "date": "2024-08-08"})
    else:
        prices[0][target] += 0.001 if target != "volume" else 1
    mutated = {
        "eod": SourceArtifact(
            source="eodhd_eod",
            source_url=script.EODHD_REQUEST_URLS["eod"],
            retrieved_at="2026-07-19T00:00:00Z",
            content=json.dumps(prices, sort_keys=True, separators=(",", ":")).encode(),
            content_type="application/json",
        ),
        "div": SourceArtifact(
            source="eodhd_div",
            source_url=script.EODHD_REQUEST_URLS["div"],
            retrieved_at="2026-07-19T00:00:00Z",
            content=json.dumps(dividends, sort_keys=True, separators=(",", ":")).encode(),
            content_type="application/json",
        ),
        "splits": SourceArtifact(
            source="eodhd_splits",
            source_url=script.EODHD_REQUEST_URLS["splits"],
            retrieved_at="2026-07-19T00:00:00Z",
            content=json.dumps(splits, sort_keys=True, separators=(",", ":")).encode(),
            content_type="application/json",
        ),
    }
    report = script.assess_provider_overlap(
        current_prices=current_prices,
        current_dividends=current_dividends,
        ntcoy_prices=prices,
        ntcoy_dividends=dividends,
        ntcoy_splits=splits,
        provider_raw_sha256={key: value.source_hash for key, value in mutated.items()},
    )
    assert report["status"] == "blocked_provider_mismatch"
    assert expected_blocker in report["blockers"]
    assert report["observed_actual_profile"] is False
    assert report["canonical_price_replacement_candidate"] is False
    current, _ = repository.current_release()
    assert current.version == release.version






class _TransactionRepository:
    """Small CAS repository for NTCOY transaction-control tests."""

    def __init__(self, root: Path):
        self.root = root
        self.objects = LocalObjectStore(root)
        self.manifests: dict[tuple[str, str], DatasetManifest] = {}
        self.frames: dict[tuple[str, str], pd.DataFrame] = {}
        self.write_count = 0
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
            self.objects.put(
                self.current_key(dataset),
                CurrentPointer.create(manifest, manifest_path).to_bytes(),
                if_none_match=True,
            )
            self.frames[(dataset, version)] = pd.DataFrame(
                {"base_marker": [dataset]}
            )
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

    def current_release(self):  # type: ignore[no-untyped-def]
        value = self.objects.get("releases/current.json")
        return DataRelease.from_bytes(value.data), value.etag

    def current_pointer(self, dataset: str):  # type: ignore[no-untyped-def]
        value = self.objects.get(self.current_key(dataset))
        return CurrentPointer.from_bytes(value.data), value.etag

    def manifest_for_version(self, dataset: str, version: str):  # type: ignore[no-untyped-def]
        return self.manifests[(dataset, version)]

    def read_frame(self, dataset: str, version: str):  # type: ignore[no-untyped-def]
        return self.frames[(dataset, version)].copy(deep=True)

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
    ) -> DatasetWriteResult:
        assert incomplete_action_policy == "block"
        manifest = DatasetManifest.create(
            dataset,
            version,
            completed_session,
            (),
            metadata=metadata,
        )
        manifest_path = f"datasets/{dataset}/versions/{version}/manifest.json"
        self.objects.put(manifest_path, manifest.to_bytes(), if_none_match=True)
        self.objects.put(
            self.current_key(dataset),
            CurrentPointer.create(manifest, manifest_path).to_bytes(),
            if_match=expected_pointer_etag,
        )
        self.manifests[(dataset, version)] = manifest
        self.frames[(dataset, version)] = frame.copy(deep=True)
        self.write_count += 1
        return DatasetWriteResult(manifest, ValidationReport(dataset))

    def commit_release(
        self,
        completed_session: str,
        dataset_versions: dict[str, str],
        *,
        quality: str,
        warnings: tuple[str, ...],
        expected_etag: str | None,
    ) -> DataRelease:
        release = DataRelease(
            version="committed-release",
            created_at="2026-07-19T00:00:01Z",
            completed_session=completed_session,
            dataset_versions=dict(dataset_versions),
            quality=quality,
            warnings=warnings,
        )
        self.objects.put(
            "releases/current.json",
            release.to_bytes(),
            if_match=expected_etag,
        )
        return release

    def replace_pointer(self, dataset: str, version: str) -> None:
        current, etag = self.current_pointer(dataset)
        assert current is not None
        manifest = DatasetManifest.create(dataset, version, "2026-07-15", ())
        path = f"datasets/{dataset}/versions/{version}/manifest.json"
        self.objects.put(path, manifest.to_bytes(), if_none_match=True)
        self.manifests[(dataset, version)] = manifest
        self.frames[(dataset, version)] = pd.DataFrame({"external": [dataset]})
        self.objects.put(
            self.current_key(dataset),
            CurrentPointer.create(manifest, path).to_bytes(),
            if_match=etag,
        )


def _transaction_plan(
    repository: _TransactionRepository,
    token: str,
    *,
    status: str = "validated_offline_plan",
) -> object:
    release, release_etag = repository.current_release()
    pointers = {
        dataset: repository.current_pointer(dataset)[1]
        for dataset in script.REQUIRED_DATASETS
    }
    planned = {
        dataset: f"ntco-ntcoy-20260715-{token}-{dataset}"
        for dataset in script.WRITE_DATASETS
    }
    frames = {
        dataset: pd.DataFrame({"locked_plan": [f"{token}:{dataset}"]})
        for dataset in script.WRITE_DATASETS
    }
    return script.PreparedTransition(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointers,
        planned_versions=planned if status != "already_repaired" else {},
        frames=frames,
        archive_artifacts=(),
        summary={
            "status": status,
            "network_accessed": False,
            "prewrite_allowed_index_identity_gap_fingerprints": [],
        },
    )


def _control_bytes(repository: _TransactionRepository) -> dict[str, bytes]:
    keys = ["releases/current.json"] + [
        repository.current_key(dataset) for dataset in script.REQUIRED_DATASETS
    ]
    return {key: repository.objects.get(key).data for key in keys}


def _patch_transaction_economics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(script, "_persist_artifacts", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(script, "_read_reviewed_bundle", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(script, "_is_repaired", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        script,
        "validate_repository_snapshot",
        lambda *_args, **_kwargs: ValidationReport("repository"),
    )
    monkeypatch.setattr(script, "validate_pin_contract", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(script, "_pin_map", lambda *_args, **_kwargs: {})


@pytest.mark.parametrize(
    "failure_stage",
    [
        "after_artifacts",
        *(f"after_dataset:{dataset}" for dataset in script.WRITE_DATASETS),
        "after_release_commit",
    ],
)
def test_transaction_failure_rolls_back_all_owned_control_pointers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    repository = _TransactionRepository(tmp_path / "repo")
    caller = _transaction_plan(repository, "a" * 32)
    locked = _transaction_plan(repository, "b" * 32)
    before = _control_bytes(repository)
    monkeypatch.setattr(script, "prepare_repair", lambda *_args, **_kwargs: locked)
    _patch_transaction_economics(monkeypatch)

    def fail(stage: str) -> None:
        if stage == failure_stage:
            raise RuntimeError(f"synthetic failure at {stage}")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        script.apply_repair(repository, caller, inject_failure=fail)

    assert _control_bytes(repository) == before
    journals = tuple((repository.root / script.TRANSACTION_DIR).glob("*.json"))
    assert len(journals) == 1
    journal = json.loads(journals[0].read_text(encoding="utf-8"))
    assert journal["status"] == "rolled_back"
    assert journal["rollback_errors"] == []
    assert not (repository.root / script.RECOVERY_DIR).exists()


def test_external_release_is_never_clobbered_by_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _TransactionRepository(tmp_path / "repo")
    caller = _transaction_plan(repository, "a" * 32)
    locked = _transaction_plan(repository, "b" * 32)
    monkeypatch.setattr(script, "prepare_repair", lambda *_args, **_kwargs: locked)
    _patch_transaction_economics(monkeypatch)
    external: DataRelease | None = None

    def racing_commit(
        completed_session: str,
        dataset_versions: dict[str, str],
        *,
        quality: str,
        warnings: tuple[str, ...],
        expected_etag: str | None,
    ) -> DataRelease:
        nonlocal external
        external = DataRelease(
            version="external-release",
            created_at="2026-07-19T00:00:02Z",
            completed_session=completed_session,
            dataset_versions=dict(dataset_versions),
            quality=quality,
            warnings=warnings,
        )
        repository.objects.put(
            "releases/current.json", external.to_bytes(), if_match=expected_etag
        )
        raise RuntimeError("synthetic competing release")

    monkeypatch.setattr(repository, "commit_release", racing_commit)
    with pytest.raises(RuntimeError, match="rollback failed"):
        script.apply_repair(repository, caller)

    current, _ = repository.current_release()
    assert external is not None and current.to_bytes() == external.to_bytes()
    recovery = tuple((repository.root / script.RECOVERY_DIR).glob("*.json"))
    assert len(recovery) == 1
    journal = json.loads(recovery[0].read_text(encoding="utf-8"))
    assert journal["status"] == "rollback_failed"
    assert any("not owned" in value for value in journal["rollback_errors"])


@pytest.mark.parametrize("journal_body", ['{"status":"prepared"}', '{"status":"rollback_failed"}', "not-json"])
def test_unfinished_or_unreadable_transaction_journal_blocks_writes(
    tmp_path: Path,
    journal_body: str,
) -> None:
    repository = _TransactionRepository(tmp_path / "repo")
    caller = _transaction_plan(repository, "a" * 32)
    before = _control_bytes(repository)
    journal = repository.root / script.TRANSACTION_DIR / "orphan.json"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(journal_body, encoding="utf-8")

    with pytest.raises(RuntimeError, match="journal blocks writes"):
        script.apply_repair(repository, caller)
    assert repository.write_count == 0
    assert _control_bytes(repository) == before


def test_forged_already_repaired_cannot_bypass_writer_lock(tmp_path: Path) -> None:
    repository = _TransactionRepository(tmp_path / "repo")
    forged = _transaction_plan(
        repository,
        "a" * 32,
        status="already_repaired",
    )
    lock_path = repository.root / ".locks/market-store-write.lock"
    with script._exclusive_file_lock(lock_path, label="test"):
        with pytest.raises(RuntimeError, match="lock is already held"):
            script.apply_repair(repository, forged)
    assert repository.write_count == 0


def test_success_uses_locked_replan_and_replay_is_write_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _TransactionRepository(tmp_path / "repo")
    caller = _transaction_plan(repository, "a" * 32)
    locked = _transaction_plan(repository, "b" * 32)
    _patch_transaction_economics(monkeypatch)
    calls = 0

    def replan(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            return locked
        return _transaction_plan(
            repository,
            "c" * 32,
            status="already_repaired",
        )

    monkeypatch.setattr(script, "prepare_repair", replan)
    result = script.apply_repair(repository, caller)
    assert result["status"] == "applied"
    assert result["writes_performed"] is True
    assert repository.write_count == len(script.WRITE_DATASETS)
    release, _ = repository.current_release()
    for dataset in script.WRITE_DATASETS:
        assert release.dataset_versions[dataset] == locked.planned_versions[dataset]
        written = repository.read_frame(dataset, locked.planned_versions[dataset])
        assert written.iloc[0, 0] == f"{'b' * 32}:{dataset}"
    for dataset in script.OUT_OF_SCOPE_CAS_DATASETS:
        assert release.dataset_versions[dataset] == f"base-{dataset}"

    before_replay = _control_bytes(repository)
    already = _transaction_plan(
        repository,
        "d" * 32,
        status="already_repaired",
    )
    replay = script.apply_repair(repository, already)
    assert replay["status"] == "already_repaired"
    assert replay["writes_performed"] is False
    assert repository.write_count == len(script.WRITE_DATASETS)
    assert _control_bytes(repository) == before_replay


def test_out_of_scope_pointer_race_fails_closed_and_preserves_external_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _TransactionRepository(tmp_path / "repo")
    caller = _transaction_plan(repository, "a" * 32)
    locked = _transaction_plan(repository, "b" * 32)
    monkeypatch.setattr(script, "prepare_repair", lambda *_args, **_kwargs: locked)
    _patch_transaction_economics(monkeypatch)
    raced = script.OUT_OF_SCOPE_CAS_DATASETS[0]

    def inject(stage: str) -> None:
        if stage == f"after_dataset:{script.WRITE_DATASETS[-1]}":
            repository.replace_pointer(raced, "external-index-version")

    with pytest.raises(RuntimeError, match="rollback failed"):
        script.apply_repair(repository, caller, inject_failure=inject)
    pointer, _ = repository.current_pointer(raced)
    release, _ = repository.current_release()
    assert pointer.version == "external-index-version"
    assert release.version == "base-release"
    assert (repository.root / script.RECOVERY_DIR).exists()
