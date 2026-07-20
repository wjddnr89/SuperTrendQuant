from __future__ import annotations

import gzip
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest
import yaml

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.cross_validation import (
    TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256,
    reviewed_nonterminal_inventory_sha256,
)
from supertrend_quant.market_store.ingest import SourceArtifact
from supertrend_quant.market_store.manifest import DataRelease, sha256_bytes
from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.yahoo_chart import (
    YahooChartCachedResponse,
    parse_yahoo_chart_json,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/repair_us_vip_veon_identity.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_vip_veon_identity", SCRIPT_PATH
)
script = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


RETRIEVED_AT = "2026-07-18T00:00:00Z"
EXPECTED_SESSIONS = script._expected_sessions()
CANONICAL_EOD_HASH = sha256_bytes(b"fixture canonical VEON eod")
REJECTED_VIP_HASH = sha256_bytes(b"fixture contaminated VIP eod")


def _source(label: str) -> dict:
    return {
        "source": label,
        "source_url": f"https://example.test/{label}",
        "retrieved_at": RETRIEVED_AT,
        "source_hash": sha256_bytes(label.encode()),
    }


def _base_bars() -> pd.DataFrame:
    rows = []
    for index, session in enumerate(EXPECTED_SESSIONS):
        close = 4.0 + (index % 37) * 0.01
        if session == script.SOURCE_OLD_LAST_SESSION:
            close = 3.95
        elif session in {script.OLD_LAST_SESSION, script.TRANSITION_DATE}:
            close = 4.05
        rows.append(
            {
                "session": session,
                "open": close - 0.02,
                "high": close + 0.04,
                "low": close - 0.05,
                "close": close,
                "volume": 100_000.0 + index,
            }
        )
    return pd.DataFrame(rows)


def _chart_content(
    *,
    scale: float = 25.0,
    drop_session: str = "",
    granularity: str = "1d",
) -> bytes:
    bars = _base_bars()
    if drop_session:
        bars = bars.loc[~bars["session"].eq(drop_session)].copy()
    timestamps = [
        int(
            (
                pd.Timestamp(value, tz="America/New_York")
                + pd.Timedelta(hours=16)
            ).timestamp()
        )
        for value in bars["session"]
    ]
    quote = {
        column: (bars[column] * (scale if column != "volume" else 1.0)).tolist()
        for column in ("open", "high", "low", "close", "volume")
    }
    payload = {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {
                        "symbol": script.NEW_SYMBOL,
                        "currency": "USD",
                        "instrumentType": "EQUITY",
                        "exchangeName": "NMS",
                        "exchangeTimezoneName": "America/New_York",
                        "dataGranularity": granularity,
                    },
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [quote],
                        "adjclose": [{"adjclose": quote["close"]}],
                    },
                }
            ],
        }
    }
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _response(
    *,
    scale: float = 25.0,
    drop_session: str = "",
    granularity: str = "1d",
) -> YahooChartCachedResponse:
    content = _chart_content(
        scale=scale,
        drop_session=drop_session,
        granularity=granularity,
    )
    cache = script._yahoo_cache(Path("unused"))
    return YahooChartCachedResponse(
        symbol=script.NEW_SYMBOL,
        source_url=cache.url(
            script.NEW_SYMBOL,
            period1=script.YAHOO_PERIOD1,
            period2=script.YAHOO_PERIOD2,
        ),
        retrieved_at=RETRIEVED_AT,
        content=content,
        content_type="application/json",
        http_status=200,
        wrapper_hash=sha256_bytes(b"fixture wrapper"),
        request_period1=script.YAHOO_PERIOD1,
        request_period2=script.YAHOO_PERIOD2,
    )


def _price_rows() -> pd.DataFrame:
    canonical = _base_bars()
    canonical.insert(0, "security_id", script.CANONICAL_SECURITY_ID)
    canonical["currency"] = "USD"
    canonical["source"] = "eodhd_eod"
    canonical["source_url"] = "https://eodhd.com/api/eod/VEON.US?fixture"
    canonical["retrieved_at"] = RETRIEVED_AT
    canonical["source_hash"] = CANONICAL_EOD_HASH

    old = canonical.loc[
        canonical["session"].le(script.SOURCE_OLD_LAST_SESSION)
    ].copy()
    old["security_id"] = script.OLD_SECURITY_ID
    old[["open", "high", "low", "close"]] = (
        old[["open", "high", "low", "close"]] * 74.0
    )
    terminal = old["session"].eq(script.SOURCE_OLD_LAST_SESSION)
    old.loc[terminal, ["open", "high", "low", "close"]] = [
        294.0,
        294.4,
        290.8,
        292.8,
    ]
    old["source_url"] = "https://eodhd.com/api/eod/VIP.US?fixture"
    old["source_hash"] = REJECTED_VIP_HASH
    return pd.concat([canonical, old], ignore_index=True, sort=False)


def _action(
    security_id: str,
    *,
    cash_amount: float,
) -> dict:
    return {
        "event_id": sha256_bytes(f"{security_id}|dividend".encode()),
        "security_id": security_id,
        "action_type": "cash_dividend",
        "effective_date": "2016-11-16",
        "ex_date": "2016-11-16",
        "announcement_date": "2016-11-04",
        "record_date": "2016-11-18",
        "payment_date": "2016-12-07",
        "cash_amount": cash_amount,
        "ratio": None,
        "currency": "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": False,
        "source_kind": "provider",
        **_source(f"action-{security_id}"),
    }


def _existing_frames() -> dict[str, pd.DataFrame]:
    prices = _price_rows()
    actions = pd.DataFrame(
        [
            _action(script.OLD_SECURITY_ID, cash_amount=0.0014),
            _action(script.CANONICAL_SECURITY_ID, cash_amount=0.02618),
        ]
    )
    master = pd.DataFrame(
        [
            {
                "security_id": script.OLD_SECURITY_ID,
                "primary_symbol": script.OLD_SYMBOL,
                "provider_symbol": "VIP.US",
                "action_provider_symbol": "VIP.US",
                "name": "VimpelCom Ltd",
                "exchange": "NASDAQ",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "active_from": script.PRICE_START,
                "active_to": script.SOURCE_OLD_LAST_SESSION,
                **_source("catalog"),
            },
            {
                "security_id": script.CANONICAL_SECURITY_ID,
                "primary_symbol": script.NEW_SYMBOL,
                "provider_symbol": "VEON.US",
                "action_provider_symbol": "VEON.US",
                "name": "VEON Ltd",
                "exchange": "NASDAQ",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "active_from": script.PRICE_START,
                "active_to": "",
                **_source("catalog"),
            },
        ]
    )
    history = pd.DataFrame(
        [
            {
                "security_id": script.OLD_SECURITY_ID,
                "symbol": script.OLD_SYMBOL,
                "exchange": "NASDAQ",
                "effective_from": script.HISTORY_START,
                "effective_to": script.SOURCE_OLD_LAST_SESSION,
                **_source("catalog"),
            },
            {
                "security_id": script.CANONICAL_SECURITY_ID,
                "symbol": script.NEW_SYMBOL,
                "exchange": "NASDAQ",
                "effective_from": script.SOURCE_NEW_FIRST_SESSION,
                "effective_to": "",
                **_source("catalog"),
            },
        ]
    )
    factors = build_adjustment_factors(
        prices,
        actions,
        source_version="fixture",
    )
    anchors = pd.DataFrame(
        [
            {
                "index_id": "nasdaq100",
                "anchor_date": "2015-01-01",
                "security_id": script.OLD_SECURITY_ID,
                "official": False,
                "source_kind": "community",
                **_source("index"),
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": sha256_bytes(b"old index event"),
                "index_id": "nasdaq100",
                "announcement_date": "",
                "effective_date": "2015-12-21",
                "operation": "REMOVE",
                "security_id": script.OLD_SECURITY_ID,
                "official": False,
                "source_kind": "community",
                **_source("index"),
            }
        ]
    )
    archive = pd.DataFrame(
        [
            {
                "archive_id": script.WIKI_FULL_SHA256,
                "dataset": "kaggle_frozen_quandl_wiki_mirror",
                "object_path": (
                    f"archives/2017-03-30/{script.WIKI_FULL_SHA256}.zip.gz"
                ),
                "content_type": "application/zip",
                "effective_date": "2017-03-30",
                "source": "kaggle_frozen_quandl_wiki_mirror",
                "source_url": script.WIKI_URL,
                "retrieved_at": RETRIEVED_AT,
                "source_hash": script.WIKI_FULL_SHA256,
            }
        ],
        columns=tuple(
            dict.fromkeys(
                (*dataset_spec("source_archive").required_columns, "source_url")
            )
        ),
    )
    return {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "adjustment_factors": factors,
        "index_constituent_anchors": anchors,
        "index_membership_events": events,
        "source_archive": archive,
    }


def _evidence(response: YahooChartCachedResponse | None = None) -> script.EvidenceBundle:
    response = response or _response()
    sec_old_symbol = SourceArtifact(
        source="sec_edgar_filing",
        source_url=script.OLD_SYMBOL_SEC_URL,
        retrieved_at=RETRIEVED_AT,
        content=b"fixture official old-symbol SEC bytes",
        content_type="text/plain",
    )
    sec = SourceArtifact(
        source="sec_edgar_filing",
        source_url=script.SEC_URL,
        retrieved_at=RETRIEVED_AT,
        content=b"fixture official SEC bytes",
        content_type="text/plain",
    )
    yahoo = SourceArtifact(
        source=script.YAHOO_SOURCE,
        source_url=response.source_url,
        retrieved_at=response.retrieved_at,
        content=response.content,
        content_type=response.content_type,
    )
    audit_content = script._canonical_json_bytes(
        {
            "schema": script.WIKI_AUDIT_SCHEMA,
            "exact_ticker_rows": {"VIP": 0, "VEON": 0},
        }
    )
    audit = SourceArtifact(
        source=script.WIKI_AUDIT_SOURCE,
        source_url=script.WIKI_URL,
        retrieved_at=RETRIEVED_AT,
        content=audit_content,
        content_type="application/json",
    )
    wiki = script.WikiRejectedAudit(
        artifact=audit,
        full_archive_path=Path("unused"),
        full_response_hash=script.WIKI_FULL_SHA256,
        full_response_size=script.WIKI_FULL_SIZE,
        total_data_rows=script.WIKI_TOTAL_DATA_ROWS,
        ticker_rows={"VIP": 0, "VEON": 0},
    )
    return script.EvidenceBundle(
        sec_old_symbol=sec_old_symbol,
        sec=sec,
        yahoo=yahoo,
        yahoo_response=response,
        yahoo_data=parse_yahoo_chart_json(response.content, script.NEW_SYMBOL),
        wiki=wiki,
        metrics={},
    )


def _evidence_patches(evidence: script.EvidenceBundle):
    return (
        mock.patch.object(
            script,
            "OLD_SYMBOL_SEC_SHA256",
            evidence.sec_old_symbol.source_hash,
        ),
        mock.patch.object(script, "SEC_SHA256", evidence.sec.source_hash),
        mock.patch.object(script, "YAHOO_SHA256", evidence.yahoo.source_hash),
        mock.patch.object(
            script,
            "YAHOO_WRAPPER_SHA256",
            evidence.yahoo_response.wrapper_hash,
        ),
        mock.patch.object(script, "EODHD_VEON_EOD_SHA256", CANONICAL_EOD_HASH),
        mock.patch.object(
            script,
            "EODHD_VIP_REJECTED_EOD_SHA256",
            REJECTED_VIP_HASH,
        ),
    )


def test_exact_bounded_yahoo_inventory_scale_and_boundary_pass():
    response = _response(scale=25.0)
    cache = script._yahoo_cache(Path("unused"))
    with (
        mock.patch.object(script, "YAHOO_SHA256", response.source_hash),
        mock.patch.object(script, "YAHOO_WRAPPER_SHA256", response.wrapper_hash),
        mock.patch.object(script, "EODHD_VEON_EOD_SHA256", CANONICAL_EOD_HASH),
    ):
        parsed, metrics = script.validate_yahoo_crosscheck(
            response,
            _price_rows(),
            require_pin=True,
            cache=cache,
        )
    assert len(parsed.bars) == script.YAHOO_EXPECTED_ROWS
    assert metrics["all_sessions_compared"] is True
    assert metrics["median_yahoo_to_eodhd_close_scale"] == 25.0
    assert metrics["ohlc_mismatch_counts"] == {
        "open": 0,
        "high": 0,
        "low": 0,
        "close": 0,
    }
    assert metrics["close_return_correlation"] == pytest.approx(1.0)
    assert metrics["eodhd_boundary_return"] == pytest.approx(
        metrics["yahoo_boundary_return"]
    )


def test_yahoo_missing_one_xnys_session_fails_closed():
    response = _response(drop_session="2016-03-10")
    with (
        mock.patch.object(script, "YAHOO_SHA256", response.source_hash),
        mock.patch.object(script, "YAHOO_WRAPPER_SHA256", response.wrapper_hash),
        mock.patch.object(script, "EODHD_VEON_EOD_SHA256", CANONICAL_EOD_HASH),
        pytest.raises(ValueError, match="exact bounded XNYS inventory"),
    ):
        script.validate_yahoo_crosscheck(
            response,
            _price_rows(),
            require_pin=True,
            cache=script._yahoo_cache(Path("unused")),
        )


def test_yahoo_hash_or_granularity_change_fails_closed():
    response = _response()
    with (
        mock.patch.object(script, "YAHOO_SHA256", "0" * 64),
        mock.patch.object(script, "YAHOO_WRAPPER_SHA256", response.wrapper_hash),
        pytest.raises(ValueError, match="pin changed"),
    ):
        script.validate_yahoo_crosscheck(
            response,
            _price_rows(),
            require_pin=True,
            cache=script._yahoo_cache(Path("unused")),
        )
    bad = _response(granularity="3mo")
    with (
        mock.patch.object(script, "YAHOO_SHA256", bad.source_hash),
        mock.patch.object(script, "YAHOO_WRAPPER_SHA256", bad.wrapper_hash),
        mock.patch.object(script, "EODHD_VEON_EOD_SHA256", CANONICAL_EOD_HASH),
        pytest.raises(ValueError, match="dataGranularity must be exactly 1d"),
    ):
        script.validate_yahoo_crosscheck(
            bad,
            _price_rows(),
            require_pin=True,
            cache=script._yahoo_cache(Path("unused")),
        )


def test_repair_discards_vip_prices_actions_and_rekeys_same_lineage():
    existing = _existing_frames()
    evidence = _evidence()
    provenance_columns = [
        "official",
        "source",
        "source_url",
        "source_kind",
        "retrieved_at",
        "source_hash",
    ]
    original_anchor_provenance = existing["index_constituent_anchors"].loc[
        :, provenance_columns
    ].copy()
    original_event_provenance = existing["index_membership_events"].loc[
        :, provenance_columns
    ].copy()
    original_canonical = existing["daily_price_raw"].loc[
        existing["daily_price_raw"]["security_id"].astype(str).eq(
            script.CANONICAL_SECURITY_ID
        )
    ].sort_values("session").reset_index(drop=True)
    patches = _evidence_patches(evidence)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        frames, summary = script.prepare_repair_frames(
            existing,
            evidence,
            completed_session=script.TRANSITION_DATE,
            source_version="fixture-repair",
        )

    assert summary["status"] == "validated_offline_plan"
    assert summary["contaminated_vip_price_rows_removed"] == 564
    assert summary["contaminated_vip_action_rows_removed"] == 1
    for dataset in (
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "corporate_actions",
        "adjustment_factors",
        "index_constituent_anchors",
        "index_membership_events",
    ):
        assert not frames[dataset]["security_id"].astype(str).eq(
            script.OLD_SECURITY_ID
        ).any()
    repaired_canonical = frames["daily_price_raw"].loc[
        frames["daily_price_raw"]["security_id"].astype(str).eq(
            script.CANONICAL_SECURITY_ID
        )
    ].sort_values("session").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        repaired_canonical.loc[:, original_canonical.columns],
        original_canonical,
        check_dtype=False,
    )
    intervals = frames["symbol_history"].loc[
        frames["symbol_history"]["security_id"].astype(str).eq(
            script.CANONICAL_SECURITY_ID
        ),
        ["symbol", "effective_from", "effective_to"],
    ]
    assert {
        tuple("" if pd.isna(value) else str(value) for value in row)
        for row in intervals.to_numpy()
    } == {
        ("VIP", "2015-01-01", "2017-03-30"),
        ("VEON", "2017-03-31", ""),
    }
    official = frames["corporate_actions"].loc[
        frames["corporate_actions"]["action_type"].astype(str).eq("ticker_change")
    ].iloc[0]
    assert official.new_security_id == script.CANONICAL_SECURITY_ID
    assert official.new_symbol == "VEON"
    assert str(official.effective_date) == "2017-03-31"
    assert str(official.announcement_date) == "2017-03-30"
    assert pd.isna(official.ratio)
    assert set(frames["index_constituent_anchors"].security_id) == {
        script.CANONICAL_SECURITY_ID
    }
    assert set(frames["index_membership_events"].security_id) == {
        script.CANONICAL_SECURITY_ID
    }
    pd.testing.assert_frame_equal(
        frames["index_constituent_anchors"].loc[:, provenance_columns],
        original_anchor_provenance,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        frames["index_membership_events"].loc[:, provenance_columns],
        original_event_provenance,
        check_dtype=False,
    )
    assert script._looks_repaired(frames)


def test_preflight_rejects_when_vip_contamination_signature_disappears():
    frames = _existing_frames()
    mask = frames["daily_price_raw"]["security_id"].astype(str).eq(
        script.OLD_SECURITY_ID
    )
    frames["daily_price_raw"].loc[mask, ["open", "high", "low", "close"]] = 4.0
    with (
        mock.patch.object(script, "EODHD_VEON_EOD_SHA256", CANONICAL_EOD_HASH),
        mock.patch.object(
            script, "EODHD_VIP_REJECTED_EOD_SHA256", REJECTED_VIP_HASH
        ),
        pytest.raises(ValueError, match="contamination signature changed"),
    ):
        script._identity_preflight(frames)


class _RootOnlyRepository:
    def __init__(self, root: Path):
        self.root = root


def _archive_frame_for_artifact(artifact: SourceArtifact) -> pd.DataFrame:
    empty = pd.DataFrame(
        columns=tuple(
            dict.fromkeys(
                (*dataset_spec("source_archive").required_columns, "source_url")
            )
        )
    )
    return script._append_source_archive(
        empty,
        (artifact,),
        completed_session=script.TRANSITION_DATE,
    )


def test_persisted_target_archive_missing_fails_closed(tmp_path: Path):
    artifact = SourceArtifact(
        source="fixture_evidence",
        source_url="https://example.test/evidence",
        retrieved_at=RETRIEVED_AT,
        content=b"exact reviewed evidence",
        content_type="text/plain",
    )
    archive = _archive_frame_for_artifact(artifact)
    with pytest.raises(ValueError, match="Missing/escaping VIP/VEON archive object"):
        script._verify_persisted_archive_artifact(
            _RootOnlyRepository(tmp_path), archive, artifact
        )


def test_persisted_target_archive_corruption_fails_closed(tmp_path: Path):
    artifact = SourceArtifact(
        source="fixture_evidence",
        source_url="https://example.test/evidence",
        retrieved_at=RETRIEVED_AT,
        content=b"exact reviewed evidence",
        content_type="text/plain",
    )
    archive = _archive_frame_for_artifact(artifact)
    path = tmp_path / str(archive.iloc[0].object_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(gzip.compress(b"corrupted evidence", mtime=0))
    with pytest.raises(ValueError, match="differs from evidence"):
        script._verify_persisted_archive_artifact(
            _RootOnlyRepository(tmp_path), archive, artifact
        )


def _write_wiki_fixture(root: Path, rows: list[str]) -> tuple[pd.DataFrame, str, int]:
    raw_zip = root / "raw.zip"
    csv_content = (
        "ticker,date,open,high,low,close,volume,ex-dividend,split_ratio,"
        "adj_open,adj_high,adj_low,adj_close,adj_volume\n"
        + "".join(rows)
    ).encode()
    with zipfile.ZipFile(raw_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(script.WIKI_MEMBER_NAME, csv_content)
    raw = raw_zip.read_bytes()
    digest = sha256_bytes(raw)
    destination = root / "archives/fixture.zip.gz"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(gzip.compress(raw, mtime=0))
    archive_frame = pd.DataFrame(
        [
            {
                "archive_id": digest,
                "dataset": "wiki",
                "object_path": "archives/fixture.zip.gz",
                "content_type": "application/zip",
                "effective_date": "2017-03-30",
                "source": "wiki",
                "source_url": script.WIKI_URL,
                "retrieved_at": RETRIEVED_AT,
                "source_hash": digest,
            }
        ]
    )
    return archive_frame, digest, len(raw)


def test_final_wiki_zero_rows_is_preserved_as_rejected_audit():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        frame, digest, size = _write_wiki_fixture(
            root,
            ["AA,2017-03-29,1,1,1,1,1,0,1,1,1,1,1,1\n"],
        )
        with (
            mock.patch.object(script, "WIKI_FULL_SHA256", digest),
            mock.patch.object(script, "WIKI_FULL_SIZE", size),
            mock.patch.object(script, "WIKI_TOTAL_DATA_ROWS", 1),
        ):
            audit = script.audit_rejected_wiki_source(
                _RootOnlyRepository(root), frame
            )
        assert audit.ticker_rows == {"VIP": 0, "VEON": 0}
        assert audit.artifact.source == script.WIKI_AUDIT_SOURCE
        assert b'"permitted_as_price_source":false' in audit.artifact.content


def test_final_wiki_target_row_is_never_silently_accepted():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        frame, digest, size = _write_wiki_fixture(
            root,
            ["VIP,2017-03-29,1,1,1,1,1,0,1,1,1,1,1,1\n"],
        )
        with (
            mock.patch.object(script, "WIKI_FULL_SHA256", digest),
            mock.patch.object(script, "WIKI_FULL_SIZE", size),
            mock.patch.object(script, "WIKI_TOTAL_DATA_ROWS", 1),
            pytest.raises(ValueError, match="no longer a rejected zero-row"),
        ):
            script.audit_rejected_wiki_source(_RootOnlyRepository(root), frame)


def test_offline_evidence_load_never_calls_yahoo_fetch():
    response = _response()
    evidence = _evidence(response)

    class OfflineCache:
        def __init__(self, *_args, **_kwargs):
            self.http_attempts = 0

        def get(self, *_args, **_kwargs):
            return response

        def fetch(self, *_args, **_kwargs):
            raise AssertionError("offline path called Yahoo fetch")

        def url(self, *_args, **_kwargs):
            return response.source_url

    repository = SimpleNamespace(root=Path("unused"))
    patches = _evidence_patches(evidence)
    with (
        mock.patch.object(
            script,
            "load_sec_evidence",
            return_value=(evidence.sec_old_symbol, evidence.sec),
        ),
        mock.patch.object(
            script, "audit_rejected_wiki_source", return_value=evidence.wiki
        ),
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
    ):
        loaded = script.load_evidence(
            repository,
            pd.DataFrame(),
            _price_rows(),
            yahoo_cache_root=Path("unused"),
            require_yahoo_pin=True,
            yahoo_factory=OfflineCache,
        )
    assert loaded.yahoo_response.source_hash == response.source_hash


def test_acquisition_mode_is_capped_to_one_call_and_does_not_mutate_release():
    response = _response()
    evidence = _evidence(response)
    release = DataRelease(
        version="fixture-release",
        created_at=RETRIEVED_AT,
        completed_session=script.TRANSITION_DATE,
        dataset_versions={"daily_price_raw": "prices", "source_archive": "archive"},
        quality=DataQuality.VALID,
        warnings=(),
    )

    class Repository:
        root = Path("unused")

        def current_release(self):
            return release, "same-etag"

        def read_frame(self, dataset, _version):
            return _price_rows() if dataset == "daily_price_raw" else pd.DataFrame()

    instances = []

    class OneCallCache:
        def __init__(self, root, **_kwargs):
            self.root = Path(root)
            self.http_attempts = 0
            instances.append(self)

        def fetch(self, *_args, **_kwargs):
            if self.http_attempts:
                raise AssertionError("Yahoo retried")
            self.http_attempts += 1
            return response

        def url(self, *_args, **_kwargs):
            return response.source_url

        def path(self, *_args, **_kwargs):
            return self.root / "fixture.json.gz"

    with (
        mock.patch.object(
            script,
            "load_sec_evidence",
            return_value=(evidence.sec_old_symbol, evidence.sec),
        ),
        mock.patch.object(
            script, "audit_rejected_wiki_source", return_value=evidence.wiki
        ),
        mock.patch.object(script, "EODHD_VEON_EOD_SHA256", CANONICAL_EOD_HASH),
        mock.patch.object(script, "YAHOO_SHA256", ""),
        mock.patch.object(script, "YAHOO_WRAPPER_SHA256", ""),
    ):
        summary = script.acquire_yahoo_evidence_only(
            Repository(),
            yahoo_cache_root=Path("unused"),
            yahoo_factory=OneCallCache,
        )
    assert len(instances) == 1
    assert instances[0].http_attempts == 1
    assert summary["yahoo_http_attempts"] == 1
    assert summary["release_mutated"] is False
    assert summary["status"] == "yahoo_evidence_observed_unpinned"
    assert summary["old_symbol_sec_sha256"] == evidence.sec_old_symbol.source_hash
    assert summary["sec_sha256"] == evidence.sec.source_hash


def test_reviewed_nonterminal_registry_exactly_contains_vip_veon_handoff():
    policy_path = Path(__file__).resolve().parents[1] / "configs/us_cross_validation.yaml"
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    events = policy["events"]
    matches = [
        item
        for item in events["reviewed_nonterminal_extractions"]
        if item["event_id"] == script.REVIEWED_NONTERMINAL_EXTRACTION["event_id"]
    ]
    assert matches == [script.REVIEWED_NONTERMINAL_EXTRACTION]
    assert (
        reviewed_nonterminal_inventory_sha256(events)
        == TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256
    )


def test_already_repaired_apply_is_idempotent_and_writes_nothing():
    prepared = SimpleNamespace(
        summary={"status": "already_repaired", "release_version": "fixture"}
    )
    repository = mock.Mock()
    assert script.apply_repair(repository, prepared) == prepared.summary
    repository.assert_not_called()


def test_cli_prevents_fetch_apply_combination_by_construction():
    with pytest.raises(SystemExit):
        script._parse_args(["--fetch-yahoo-evidence", "--apply"])
