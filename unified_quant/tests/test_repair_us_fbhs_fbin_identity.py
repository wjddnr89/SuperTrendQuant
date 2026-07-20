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
import yaml

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.cross_validation import (
    TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256,
    reviewed_nonterminal_inventory_sha256,
)
from supertrend_quant.market_store.ingest import SourceArtifact
from supertrend_quant.market_store.lifecycle import canonical_lifecycle_event_id
from supertrend_quant.market_store.manifest import sha256_bytes
from supertrend_quant.market_store.schemas import dataset_spec


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/repair_us_fbhs_fbin_identity.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_fbhs_fbin_identity", SCRIPT_PATH
)
script = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


RETRIEVED_AT = "2026-07-18T00:00:00Z"


def _source(label: str, source_hash: str | None = None) -> dict:
    return {
        "source": label,
        "source_url": f"https://example.test/{label}",
        "retrieved_at": RETRIEVED_AT,
        "source_hash": source_hash or sha256_bytes(label.encode()),
    }


def _evidence() -> script.EvidenceBundle:
    eod_content = json.dumps(
        [
            {
                "adjusted_close": 9.5,
                "close": 10.0,
                "date": "2022-12-13",
                "high": 10.2,
                "low": 9.7,
                "open": 9.9,
                "volume": 1000,
            },
            {
                "adjusted_close": 9.7,
                "close": 10.2,
                "date": "2022-12-14",
                "high": 10.4,
                "low": 9.9,
                "open": 10.1,
                "volume": 1100,
            },
            {
                "adjusted_close": 8.8,
                "close": 9.0,
                "date": "2022-12-15",
                "high": 9.2,
                "low": 8.5,
                "open": 8.7,
                "volume": 1200,
            },
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    dividend_content = json.dumps(
        [
            {"date": "2022-11-23", "unadjustedValue": 0.28},
            {"date": "2023-02-23", "unadjustedValue": 0.23},
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    split_content = b'[{"date":"2022-12-15","split":"117.000000/100.000000"}]'
    return script.EvidenceBundle(
        sec=SourceArtifact(
            source="sec_edgar_filing",
            source_url=script.SEC_URL,
            retrieved_at=script.SEC_RETRIEVED_AT,
            content=b"fixture SEC",
            content_type="text/plain",
        ),
        eod=SourceArtifact(
            source="eodhd_eod",
            source_url=script.FBIN_EOD_URL,
            retrieved_at=script.FBIN_RETRIEVED_AT,
            content=eod_content,
            content_type="application/json",
        ),
        dividends=SourceArtifact(
            source="eodhd_div",
            source_url=script.FBIN_DIV_URL,
            retrieved_at=script.FBIN_RETRIEVED_AT,
            content=dividend_content,
            content_type="application/json",
        ),
        splits=SourceArtifact(
            source="eodhd_splits",
            source_url=script.FBIN_SPLITS_URL,
            retrieved_at=script.FBIN_RETRIEVED_AT,
            content=split_content,
            content_type="application/json",
        ),
    )


def _action(
    security_id: str,
    event_id: str,
    effective_date: str,
    amount: float,
    source_hash: str,
) -> dict:
    return {
        "event_id": event_id,
        "security_id": security_id,
        "action_type": "cash_dividend",
        "effective_date": effective_date,
        "ex_date": effective_date,
        "announcement_date": "2022-10-01",
        "record_date": effective_date,
        "payment_date": effective_date,
        "cash_amount": amount,
        "ratio": None,
        "currency": "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": False,
        "source_kind": "provider",
        **_source(f"action-{event_id}", source_hash),
    }


def _frames(evidence: script.EvidenceBundle) -> dict[str, pd.DataFrame]:
    source_eod = pd.DataFrame(json.loads(evidence.eod.content)).rename(
        columns={"date": "session"}
    )
    canonical_prices = source_eod[
        ["session", "open", "high", "low", "close", "volume"]
    ].copy()
    canonical_prices.insert(0, "security_id", script.CANONICAL_SECURITY_ID)
    canonical_prices["currency"] = "USD"
    canonical_prices["source"] = "eodhd_eod"
    canonical_prices["source_url"] = script.FBIN_EOD_URL
    canonical_prices["retrieved_at"] = script.FBIN_RETRIEVED_AT
    canonical_prices["source_hash"] = evidence.eod.source_hash
    old_prices = canonical_prices.iloc[:2].copy()
    old_prices["security_id"] = script.OLD_SECURITY_ID
    old_prices["source_url"] = script.FBHS_EOD_URL
    old_prices["source_hash"] = "fixture-old-eod"
    prices = pd.concat([canonical_prices, old_prices], ignore_index=True)

    old_dividend = _action(
        script.OLD_SECURITY_ID,
        "old-dividend",
        "2022-11-23",
        0.28,
        "fixture-old-div",
    )
    new_dividend = _action(
        script.CANONICAL_SECURITY_ID,
        "new-dividend-1",
        "2022-11-23",
        0.28,
        evidence.dividends.source_hash,
    )
    later_dividend = _action(
        script.CANONICAL_SECURITY_ID,
        "new-dividend-2",
        "2023-02-23",
        0.23,
        evidence.dividends.source_hash,
    )
    provider_split = {
        "event_id": script.PROVIDER_SPLIT_EVENT_ID,
        "security_id": script.CANONICAL_SECURITY_ID,
        "action_type": "split",
        "effective_date": script.TRANSITION_DATE,
        "ex_date": script.TRANSITION_DATE,
        "announcement_date": "",
        "record_date": "",
        "payment_date": "",
        "cash_amount": None,
        "ratio": script.PROVIDER_SPLIT_RATIO,
        "currency": "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": False,
        "source_kind": "provider",
        "source": "eodhd_splits",
        "source_url": script.FBIN_SPLITS_URL,
        "retrieved_at": script.FBIN_RETRIEVED_AT,
        "source_hash": evidence.splits.source_hash,
    }
    actions = pd.DataFrame(
        [old_dividend, new_dividend, later_dividend, provider_split]
    )
    factors = build_adjustment_factors(prices, actions, source_version="fixture")
    master = pd.DataFrame(
        [
            {
                "security_id": script.OLD_SECURITY_ID,
                "exchange": "NYSE",
                "active_from": script.PRICE_START,
                "active_to": script.SOURCE_OLD_LAST_SESSION,
                "primary_symbol": script.OLD_SYMBOL,
                "provider_symbol": "FBHS.US",
                "action_provider_symbol": "FBHS.US",
                "name": "Fortune Brands Home & Security Inc",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "isin": "",
                **_source("catalog"),
            },
            {
                "security_id": script.CANONICAL_SECURITY_ID,
                "exchange": "NYSE",
                "active_from": script.PRICE_START,
                "active_to": "",
                "primary_symbol": script.NEW_SYMBOL,
                "provider_symbol": "FBIN.US",
                "action_provider_symbol": "FBIN.US",
                "name": "Fortune Brands Innovations Inc.",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "isin": "",
                **_source("catalog"),
            },
        ]
    )
    history = pd.DataFrame(
        [
            {
                "security_id": script.OLD_SECURITY_ID,
                "symbol": script.OLD_SYMBOL,
                "exchange": "NYSE",
                "effective_from": script.HISTORY_START,
                "effective_to": script.SOURCE_OLD_LAST_SESSION,
                **_source("catalog"),
            },
            {
                "security_id": script.CANONICAL_SECURITY_ID,
                "symbol": script.NEW_SYMBOL,
                "exchange": "NYSE",
                "effective_from": script.SOURCE_NEW_FIRST_SESSION,
                "effective_to": "",
                **_source("catalog"),
            },
        ]
    )
    anchors = pd.DataFrame(
        columns=tuple(dataset_spec("index_constituent_anchors").required_columns)
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "add-old",
                "index_id": "sp500",
                "announcement_date": "",
                "effective_date": "2016-06-24",
                "operation": "ADD",
                "security_id": script.OLD_SECURITY_ID,
                "official": False,
                "source_kind": "community",
                "source": script.SP500_SOURCE,
                "source_url": script.SP500_SOURCE_URL,
                "retrieved_at": RETRIEVED_AT,
                "source_hash": script.SP500_SOURCE_SHA256,
            },
            {
                "event_id": "remove-old",
                "index_id": "sp500",
                "announcement_date": "",
                "effective_date": "2022-12-19",
                "operation": "REMOVE",
                "security_id": script.OLD_SECURITY_ID,
                "official": False,
                "source_kind": "community",
                "source": script.SP500_SOURCE,
                "source_url": script.SP500_SOURCE_URL,
                "retrieved_at": RETRIEVED_AT,
                "source_hash": script.SP500_SOURCE_SHA256,
            },
        ]
    )
    archive_columns = tuple(
        dict.fromkeys((*dataset_spec("source_archive").required_columns, "source_url"))
    )
    return {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "adjustment_factors": factors,
        "index_constituent_anchors": anchors,
        "index_membership_events": events,
        "source_archive": pd.DataFrame(columns=archive_columns),
    }


def _patch_fixture_pins(monkeypatch: pytest.MonkeyPatch, evidence: script.EvidenceBundle):
    values = {
        "PRICE_START": "2022-12-13",
        "PRICE_END": "2022-12-15",
        "EXPECTED_OLD_PRICE_ROWS": 2,
        "EXPECTED_CANONICAL_PRICE_ROWS": 3,
        "EXPECTED_OVERLAP_ROWS": 2,
        "EXPECTED_CLOSE_EXACT_OVERLAP_ROWS": 2,
        "EXPECTED_OLD_DIVIDENDS": 1,
        "EXPECTED_CANONICAL_DIVIDENDS": 2,
        "FBHS_EOD_SHA256": "fixture-old-eod",
        "FBHS_DIV_SHA256": "fixture-old-div",
        "FBIN_EOD_SHA256": evidence.eod.source_hash,
        "FBIN_DIV_SHA256": evidence.dividends.source_hash,
        "FBIN_SPLITS_SHA256": evidence.splits.source_hash,
        "SEC_SHA256": evidence.sec.source_hash,
        "PROVIDER_SPLIT_DISPOSITION": "remove_pseudo_split",
    }
    for name, value in values.items():
        monkeypatch.setattr(script, name, value)


def test_repair_merges_same_identity_removes_synthetic_split_and_preserves_index_provenance(
    monkeypatch: pytest.MonkeyPatch,
):
    evidence = _evidence()
    _patch_fixture_pins(monkeypatch, evidence)
    frames = _frames(evidence)
    prior_prices = frames["daily_price_raw"].loc[
        frames["daily_price_raw"].security_id.eq(script.CANONICAL_SECURITY_ID),
        ["session", "open", "high", "low", "close", "volume"],
    ].reset_index(drop=True)
    provenance = [
        "official",
        "source",
        "source_url",
        "source_kind",
        "retrieved_at",
        "source_hash",
    ]
    prior_index_provenance = frames["index_membership_events"][provenance].copy()

    repaired, summary = script.prepare_repair_frames(
        frames,
        evidence,
        completed_session="2022-12-15",
        source_version="fixture-repair",
    )

    assert summary["status"] == "validated_offline_plan"
    assert summary["provider_split_disposition"] == "remove_pseudo_split"
    assert summary["provider_split_rows"] == 0
    assert summary["pending_followup"] == script.PENDING_FOLLOWUP
    assert summary["publication_ready"] is False
    for dataset in (
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "corporate_actions",
        "adjustment_factors",
        "index_membership_events",
    ):
        assert not repaired[dataset].security_id.astype(str).eq(
            script.OLD_SECURITY_ID
        ).any()
    actual_prices = repaired["daily_price_raw"].loc[
        repaired["daily_price_raw"].security_id.eq(script.CANONICAL_SECURITY_ID),
        prior_prices.columns,
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_prices, prior_prices, check_dtype=False)
    intervals = script._target_history_signature(repaired["symbol_history"])
    assert intervals == {
        ("FBHS", "2015-01-01", "2022-12-14"),
        ("FBIN", "2022-12-15", ""),
    }
    ticker = repaired["corporate_actions"].loc[
        repaired["corporate_actions"].action_type.eq("ticker_change")
    ].iloc[0]
    assert ticker.event_id == canonical_lifecycle_event_id(
        script.CANONICAL_SECURITY_ID, "ticker_change", script.TRANSITION_DATE
    )
    assert ticker.new_security_id == script.CANONICAL_SECURITY_ID
    assert not repaired["corporate_actions"].event_id.eq(
        script.PROVIDER_SPLIT_EVENT_ID
    ).any()
    factors = repaired["adjustment_factors"].loc[
        repaired["adjustment_factors"].security_id.eq(script.CANONICAL_SECURITY_ID)
    ]
    assert factors.split_factor.eq(1.0).all()
    assert set(repaired["index_membership_events"].security_id) == {
        script.CANONICAL_SECURITY_ID
    }
    pd.testing.assert_frame_equal(
        repaired["index_membership_events"][provenance],
        prior_index_provenance,
        check_dtype=False,
    )


def test_unreviewed_provider_split_decision_fails_before_rewrite(
    monkeypatch: pytest.MonkeyPatch,
):
    evidence = _evidence()
    _patch_fixture_pins(monkeypatch, evidence)
    monkeypatch.setattr(script, "PROVIDER_SPLIT_DISPOSITION", "pending_review")
    with pytest.raises(RuntimeError, match="differs from the reviewed"):
        script.prepare_repair_frames(
            _frames(evidence),
            evidence,
            completed_session="2026-07-15",
            source_version="fixture-repair",
        )


def test_changed_provider_split_ratio_fails_closed(monkeypatch: pytest.MonkeyPatch):
    evidence = _evidence()
    _patch_fixture_pins(monkeypatch, evidence)
    frames = _frames(evidence)
    split = frames["corporate_actions"].event_id.eq(script.PROVIDER_SPLIT_EVENT_ID)
    frames["corporate_actions"].loc[split, "ratio"] = 1.2
    with pytest.raises(ValueError, match="117/100 split row changed"):
        script.prepare_repair_frames(
            frames,
            evidence,
            completed_session="2026-07-15",
            source_version="fixture-repair",
        )


def test_exact_cached_evidence_bundle_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    evidence = _evidence()
    sec_content = b"accession central identity ticker"
    monkeypatch.setattr(script, "SEC_CACHE_KEY", "fixture-sec")
    monkeypatch.setattr(script, "SEC_SHA256", sha256_bytes(sec_content))
    monkeypatch.setattr(script, "SEC_EXACT_BYTES", len(sec_content))
    monkeypatch.setattr(
        script,
        "SEC_REQUIRED_TEXT_GROUPS",
        (("accession",), ("identity",), ("ticker",)),
    )
    sec_path = tmp_path / "state/sec_lifecycle/fixture-sec.bin"
    sec_path.parent.mkdir(parents=True)
    sec_path.write_bytes(sec_content)

    artifacts = []
    for artifact in (evidence.eod, evidence.dividends, evidence.splits):
        artifacts.append(
            {
                "source": artifact.source,
                "source_url": artifact.source_url,
                "retrieved_at": script.FBIN_RETRIEVED_AT,
                "content_type": artifact.content_type,
                "content_base64": base64.b64encode(artifact.content).decode("ascii"),
            }
        )
    payload = json.dumps({"artifacts": artifacts}, sort_keys=True).encode()
    wrapper = json.dumps(
        {
            "payload_base64": base64.b64encode(payload).decode("ascii"),
            "payload_sha256": sha256_bytes(payload),
        },
        sort_keys=True,
    ).encode()
    encoded = gzip.compress(wrapper, mtime=0)
    bundle_path = tmp_path / "bundle.json.gz"
    bundle_path.write_bytes(encoded)
    pins = {
        "SUCCESSOR_BUNDLE_PATH": Path("bundle.json.gz"),
        "SUCCESSOR_BUNDLE_SHA256": sha256_bytes(encoded),
        "SUCCESSOR_BUNDLE_EXACT_BYTES": len(encoded),
        "SUCCESSOR_PAYLOAD_SHA256": sha256_bytes(payload),
        "SUCCESSOR_PAYLOAD_EXACT_BYTES": len(payload),
        "SUCCESSOR_ARTIFACT_COUNT": 3,
        "FBIN_EOD_SHA256": evidence.eod.source_hash,
        "FBIN_EOD_EXACT_BYTES": len(evidence.eod.content),
        "FBIN_DIV_SHA256": evidence.dividends.source_hash,
        "FBIN_DIV_EXACT_BYTES": len(evidence.dividends.content),
        "FBIN_SPLITS_SHA256": evidence.splits.source_hash,
        "FBIN_SPLITS_EXACT_BYTES": len(evidence.splits.content),
    }
    for name, value in pins.items():
        monkeypatch.setattr(script, name, value)
    loaded = script.load_evidence(tmp_path)
    assert loaded.sec.source_hash == sha256_bytes(sec_content)
    bundle_path.write_bytes(encoded + b"tamper")
    with pytest.raises(ValueError, match="bundle hash/size changed"):
        script.load_evidence(tmp_path)


def test_apply_refuses_split_decision_drift_before_repository_access(
    monkeypatch: pytest.MonkeyPatch,
):
    prepared = SimpleNamespace(
        summary={
            "status": "validated_offline_plan",
            "provider_split_disposition": "remove_pseudo_split",
        }
    )
    monkeypatch.setattr(script, "PROVIDER_SPLIT_DISPOSITION", "pending_review")
    with pytest.raises(RuntimeError, match="differs from the reviewed"):
        script.apply_repair(object(), prepared)


def test_writer_lock_rejects_prepared_transaction(tmp_path: Path):
    marker = tmp_path / "transactions/other/prepared.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"status":"prepared"}', encoding="utf-8")
    repository = SimpleNamespace(root=tmp_path)
    with pytest.raises(RuntimeError, match="interrupted transaction blocks"):
        with script._exclusive_repository_lock(repository):
            pytest.fail("prepared transaction must block the writer")


def test_reviewed_nonterminal_registry_exactly_contains_canonical_fbin_event():
    policy_path = Path(__file__).resolve().parents[1] / "configs/us_cross_validation.yaml"
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    extractions = policy["events"]["reviewed_nonterminal_extractions"]
    assert script.REVIEWED_NONTERMINAL_EXTRACTION in extractions
    assert (
        reviewed_nonterminal_inventory_sha256(policy["events"])
        == TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256
    )
