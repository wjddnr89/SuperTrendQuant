from __future__ import annotations

import gzip
import importlib.util
import json
import sys
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from supertrend_quant.market_store.ingest import SourceArtifact
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_dwdp_identity_spinoffs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_dwdp_identity_spinoffs", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def _artifact(*, source: str, url: str, content: bytes) -> SourceArtifact:
    return SourceArtifact(
        source=source,
        source_url=url,
        retrieved_at="2026-07-18T00:00:00Z",
        content=content,
        content_type="application/json",
    )


def _official_evidence() -> object:
    specs = tuple(
        script.EvidenceSpec(key, f"https://sec.test/{key}", "0" * 64, "now", ())
        for key in ("legacy_dd_merger", "dow_distribution", "corteva_dd_completion")
    )
    artifacts = tuple(
        _artifact(source="sec_edgar_filing", url=spec.url, content=spec.key.encode())
        for spec in specs
    )
    return script.EvidenceBundle(artifacts, specs)


def _full_legacy_prices(*, scale: float = 1.0) -> pd.DataFrame:
    sessions = script._expected_xnys_sessions(
        script.LEGACY_DD_FIRST, script.LEGACY_DD_LAST
    )
    rows = []
    for number, session in enumerate(sessions):
        close = (70.0 + number / 100.0) * scale
        rows.append(
            {
                "security_id": script.LEGACY_DD_ID,
                "session": session,
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1_000_000 + number,
                "currency": "USD",
                "source": "test_legacy_dd",
                "source_url": "https://legacy.test/dd",
                "retrieved_at": "2026-07-18T00:00:00Z",
                "source_hash": "a" * 64,
            }
        )
    return pd.DataFrame(rows)


def _wiki_overlap_from(provider: pd.DataFrame) -> pd.DataFrame:
    return provider.loc[
        provider["session"].between(script.WIKI_DD_FIRST, script.WIKI_DD_LAST)
        & ~provider["session"].isin(script.WIKI_DD_MISSING_SESSIONS),
        ["session", "open", "high", "low", "close", "volume"],
    ].reset_index(drop=True)


def _yahoo_payload(*, granularity: str = "1d", drop_tail: bool = False) -> bytes:
    sessions = script._expected_xnys_sessions(
        script.LEGACY_DD_FIRST, script.LEGACY_DD_LAST
    )
    if drop_tail:
        sessions = sessions[:-1]
    timestamps = [
        int(
            (
                pd.Timestamp(session, tz="America/New_York")
                + pd.Timedelta(hours=16)
            ).timestamp()
        )
        for session in sessions
    ]
    closes = [70.0 + number / 100.0 for number in range(len(sessions))]
    result = {
        "meta": {
            "symbol": "DD",
            "currency": "USD",
            "instrumentType": "EQUITY",
            "exchangeName": "NYQ",
            "exchangeTimezoneName": "America/New_York",
            "dataGranularity": granularity,
        },
        "timestamp": timestamps,
        "indicators": {
            "quote": [
                {
                    "open": [value - 0.2 for value in closes],
                    "high": [value + 0.5 for value in closes],
                    "low": [value - 0.5 for value in closes],
                    "close": closes,
                    "volume": [1_000_000 + number for number in range(len(closes))],
                }
            ]
        },
    }
    return json.dumps({"chart": {"result": [result], "error": None}}).encode()


def test_current_dd_and_known_boris_payloads_are_explicitly_forbidden(tmp_path: Path):
    current = _artifact(
        source="eodhd_eod",
        url=script.DD_EOD_URL,
        content=b"[]",
    )
    with pytest.raises(ValueError, match="URL is not exact"):
        script._legacy_dd_prices(current)

    boris_sized = SourceArtifact(
        source=script.STOOQ_DD_SOURCE,
        source_url=script.STOOQ_DD_URL,
        retrieved_at="now",
        content=b"x" * script.BORIS_DD_REJECTED_SIZE,
        content_type="text/csv",
    )
    value = script.StooqLegacyDdArtifact(boris_sized, 200, "text/csv")
    with pytest.raises(ValueError, match="Boris"):
        script._stooq_legacy_dd_prices(value)

    cache = script.LegacyDdEndpointCache(tmp_path, allow_http=False)
    empty = _artifact(
        source="eodhd_eod",
        url=script._legacy_dd_public_url("eod"),
        content=b"[]",
    )
    cache._store("eod", empty)
    with pytest.raises(ValueError, match="no legacy price rows"):
        script._legacy_dd_prices(cache.get("eod"))


def test_http_negative_result_is_cached_once_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class Response:
        status_code = 404
        content = b"not found"
        headers = {"Content-Type": "text/plain"}

    class Session:
        def __init__(self):
            self.calls = 0

        def get(self, *_args, **_kwargs):
            self.calls += 1
            return Response()

    session = Session()
    monkeypatch.setattr(script, "STOOQ_DD_FETCH_DISABLED_AFTER_AUDIT", False)
    cache = script.StooqLegacyDdCache(
        tmp_path, allow_http=True, session_factory=lambda: session
    )
    value, attempts = cache.load()
    assert attempts == 1
    assert session.calls == 1
    assert value.http_status == 404
    with pytest.raises(ValueError, match="HTTP 404"):
        script._stooq_legacy_dd_prices(value)

    replay, replay_attempts = cache.load()
    assert replay_attempts == 0
    assert replay.artifact.source_hash == value.artifact.source_hash
    assert session.calls == 1


def test_overlap_accepts_raw_scale_and_rejects_silent_ratio_normalization():
    provider = _full_legacy_prices()
    wiki = _wiki_overlap_from(provider)
    summary = script._validate_legacy_dd_overlap(provider, wiki)
    assert summary["overlap_rows"] == script.WIKI_DD_EXPECTED_ROWS
    assert summary["median_level_ratio"] == pytest.approx(1.0)

    scaled = provider.copy()
    scaled[["open", "high", "low", "close"]] *= 1.282
    with pytest.raises(ValueError, match="price-level/return cross-validation"):
        script._validate_legacy_dd_overlap(scaled, wiki)


def test_bounded_yahoo_requires_daily_granularity_and_exact_tail():
    assert "range=max" not in script.YAHOO_DD_URL
    assert "period1=" in script.YAHOO_DD_URL and "period2=" in script.YAHOO_DD_URL

    artifact = _artifact(
        source=script.YAHOO_DD_SOURCE,
        url=script.YAHOO_DD_URL,
        content=_yahoo_payload(),
    )
    value = script.YahooLegacyDdArtifact(artifact, 200, "application/json")
    prices = script._yahoo_legacy_dd_prices(value)
    assert len(prices) == 672
    assert prices.iloc[-1]["session"] == script.LEGACY_DD_LAST

    weekly = _artifact(
        source=script.YAHOO_DD_SOURCE,
        url=script.YAHOO_DD_URL,
        content=_yahoo_payload(granularity="1wk"),
    )
    with pytest.raises(ValueError, match="daily granularity"):
        script._yahoo_legacy_dd_prices(
            script.YahooLegacyDdArtifact(weekly, 200, "application/json")
        )

    short = _artifact(
        source=script.YAHOO_DD_SOURCE,
        url=script.YAHOO_DD_URL,
        content=_yahoo_payload(drop_tail=True),
    )
    with pytest.raises(ValueError, match="exactly cover"):
        script._yahoo_legacy_dd_prices(
            script.YahooLegacyDdArtifact(short, 200, "application/json")
        )


def _write_wiki_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sessions: tuple[str, ...],
) -> tuple[SimpleNamespace, pd.DataFrame]:
    rows = ["ticker,date,open,high,low,close,volume"]
    for session in sessions:
        if session == "2020-01-07":
            rows.append("DD,2020-01-07,59.16,60,60,59.77,6462524")
        else:
            rows.append(f"DD,{session},70,71,69,70.5,1000000")
    content = ("\n".join(rows) + "\n").encode()
    source_hash = script.sha256_bytes(content)
    object_path = Path("archives/test/wiki.csv.gz")
    destination = tmp_path / object_path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(gzip.compress(content, mtime=0))
    monkeypatch.setattr(script, "WIKI_DD_URL", "https://wiki.test/DD.csv")
    monkeypatch.setattr(script, "WIKI_DD_SHA256", source_hash)
    monkeypatch.setattr(script, "WIKI_DD_SIZE", len(content))
    monkeypatch.setattr(script, "WIKI_DD_FIRST", "2020-01-02")
    monkeypatch.setattr(script, "WIKI_DD_LAST", "2020-01-08")
    monkeypatch.setattr(script, "WIKI_DD_EXPECTED_ROWS", 4)
    monkeypatch.setattr(script, "WIKI_DD_MISSING_SESSIONS", ("2020-01-03",))
    monkeypatch.setattr(
        script,
        "WIKI_DD_KNOWN_INCOHERENT_BAR",
        {
            "session": "2020-01-07",
            "open": 59.16,
            "high": 60.0,
            "low": 60.0,
            "close": 59.77,
            "volume": 6_462_524.0,
        },
    )
    monkeypatch.setattr(
        script,
        "_expected_xnys_sessions",
        lambda _start, _end: (
            "2020-01-02",
            "2020-01-03",
            "2020-01-06",
            "2020-01-07",
            "2020-01-08",
        ),
    )
    archive = pd.DataFrame(
        [
            {
                "source_url": script.WIKI_DD_URL,
                "source_hash": source_hash,
                "object_path": str(object_path),
            }
        ]
    )
    return SimpleNamespace(root=tmp_path), archive


def test_wiki_inventory_allows_only_the_six_pinned_omissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    accepted = ("2020-01-02", "2020-01-06", "2020-01-07", "2020-01-08")
    repository, archive = _write_wiki_fixture(
        tmp_path, monkeypatch, accepted
    )
    frame = script._load_wiki_dd_rows(repository, archive)
    assert tuple(frame["session"]) == accepted

    # A different missing session fails even when the changed bytes/hash are
    # independently re-pinned; this exercises the semantic inventory gate.
    other_missing = ("2020-01-02", "2020-01-07", "2020-01-08")
    repository, archive = _write_wiki_fixture(
        tmp_path / "other", monkeypatch, other_missing
    )
    with pytest.raises(ValueError, match="row count|coverage"):
        script._load_wiki_dd_rows(repository, archive)

    extra = (
        "2020-01-02",
        "2020-01-03",
        "2020-01-06",
        "2020-01-07",
        "2020-01-08",
    )
    repository, archive = _write_wiki_fixture(
        tmp_path / "extra", monkeypatch, extra
    )
    with pytest.raises(ValueError, match="row count|coverage"):
        script._load_wiki_dd_rows(repository, archive)


def test_official_actions_pin_same_lineage_and_exact_distribution_terms():
    empty_actions = pd.DataFrame(
        columns=dataset_spec("corporate_actions").required_columns
    )
    evidence = _official_evidence()
    legacy = script.LegacyDdEvidence(
        prices=pd.DataFrame(),
        actions=empty_actions.copy(),
        artifacts=(),
        wiki_url="https://wiki.test",
        wiki_hash="f" * 64,
        overlap_rows=490,
    )
    actions = script._rewrite_actions(
        {"corporate_actions": empty_actions}, evidence, legacy
    )
    official = actions.loc[
        actions["source"].eq("official_dwdp_identity_repair")
    ]
    assert len(official) == 5
    ticker = official.loc[official["action_type"].eq("ticker_change")].iloc[0]
    assert ticker["security_id"] == script.DWDP_ID
    assert ticker["new_security_id"] == script.DWDP_ID
    assert ticker["effective_date"] == script.DD_FIRST_REGULAR_WAY
    split = official.loc[official["action_type"].eq("split")].iloc[0]
    assert split["effective_date"] == script.CTVA_DISTRIBUTION
    assert split["ex_date"] == script.DD_FIRST_REGULAR_WAY
    assert float(split["ratio"]) == pytest.approx(1 / 3)


def test_archive_remains_content_addressed_for_equal_endpoint_bytes():
    content = b"[]"
    artifacts = tuple(
        _artifact(
            source=f"eodhd_{endpoint}",
            url=script._legacy_dd_public_url(endpoint),
            content=content,
        )
        for endpoint in script.LEGACY_DD_ENDPOINTS
    )
    base = pd.DataFrame(columns=dataset_spec("source_archive").required_columns)
    result = script._append_source_archive(
        base, artifacts, completed_session="2026-07-15"
    )
    assert len(result) == 1
    assert result["archive_id"].nunique() == 1
    assert result["source_hash"].nunique() == 1
    assert result.iloc[0]["archive_id"] == result.iloc[0]["source_hash"]


def test_large_kaggle_wiki_cache_streams_full_zip_and_requires_reviewed_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sessions = ("2020-01-02", "2020-01-03", "2020-01-06")
    lines = [
        "ticker,date,open,high,low,close,volume,ex-dividend,split_ratio",
        "DD,2020-01-02,70,71,69,70.5,1000000,0,1",
        "DD,2020-01-03,71,72,70,71.5,1000001,0,1",
        "DD,2020-01-06,72,73,71,72.5,1000002,0,1",
    ]
    csv_content = ("\n".join(lines) + "\n").encode()
    segment = ("\n".join(lines[1:]) + "\n").encode()
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("WIKI_PRICES.csv", csv_content)
    response_bytes = archive_buffer.getvalue()

    monkeypatch.setattr(script, "LEGACY_DD_FIRST", sessions[0])
    monkeypatch.setattr(script, "LEGACY_DD_LAST", sessions[-1])
    monkeypatch.setattr(script, "KAGGLE_WIKI_DD_TOTAL_ROWS", 3)
    monkeypatch.setattr(script, "KAGGLE_WIKI_DD_SEGMENT_ROWS", 3)
    monkeypatch.setattr(script, "KAGGLE_WIKI_DD_FIRST_AVAILABLE", sessions[0])
    monkeypatch.setattr(script, "KAGGLE_WIKI_DD_LAST_AVAILABLE", sessions[-1])
    monkeypatch.setattr(
        script, "KAGGLE_WIKI_DD_SEGMENT_SHA256", script.sha256_bytes(segment)
    )
    monkeypatch.setattr(
        script,
        "KAGGLE_WIKI_DD_TERMINAL",
        {"open": 72.0, "high": 73.0, "low": 71.0, "close": 72.5, "volume": 1_000_002.0},
    )
    monkeypatch.setattr(
        script, "_expected_xnys_sessions", lambda _start, _end: sessions
    )

    class Response:
        status_code = 200
        headers = {"Content-Type": "application/zip"}

        @staticmethod
        def iter_content(chunk_size: int):
            assert chunk_size > 0
            midpoint = len(response_bytes) // 2
            yield response_bytes[:midpoint]
            yield response_bytes[midpoint:]

    class Session:
        calls = 0

        def get(self, *_args, **kwargs):
            assert kwargs["stream"] is True
            self.calls += 1
            return Response()

    session = Session()
    cache = script.KaggleWikiLegacyDdCache(
        tmp_path, allow_http=True, session_factory=lambda: session
    )
    value, attempts = cache.load()
    assert attempts == 1
    assert session.calls == 1
    assert value.artifact.source_hash == script.sha256_bytes(response_bytes)
    assert value.artifact.content_size == len(response_bytes)

    prices, segment_artifact, audit = script._kaggle_wiki_legacy_dd_prices(
        value, require_full_pin=False
    )
    assert tuple(prices["session"]) == sessions
    assert segment_artifact.content == segment
    assert audit["segment_rows"] == 3
    monkeypatch.setattr(script, "KAGGLE_WIKI_FULL_SHA256", "")
    monkeypatch.setattr(script, "KAGGLE_WIKI_FULL_SIZE", 0)
    with pytest.raises(ValueError, match="operator review"):
        script._kaggle_wiki_legacy_dd_prices(value, require_full_pin=True)

    monkeypatch.setattr(
        script, "KAGGLE_WIKI_FULL_SHA256", script.sha256_bytes(response_bytes)
    )
    monkeypatch.setattr(script, "KAGGLE_WIKI_FULL_SIZE", len(response_bytes))
    strict_prices, _, _ = script._kaggle_wiki_legacy_dd_prices(
        value, require_full_pin=True
    )
    assert len(strict_prices) == 3
    replay, replay_attempts = cache.load()
    assert replay_attempts == 0
    assert replay.artifact.source_hash == value.artifact.source_hash
    assert session.calls == 1


@pytest.mark.skipif(
    not Path("data/cache/releases/current.json").is_file(),
    reason="local release fixture is unavailable",
)
def test_current_release_dry_run_rewrites_identity_without_apply():
    repository = LocalDatasetRepository("data/cache")
    release, _ = repository.current_release()
    assert release is not None
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in script.REQUIRED_DATASETS
    }
    if frames["security_master"]["security_id"].astype(str).eq(
        script.LEGACY_DD_ID
    ).any():
        pytest.skip("current release is already repaired")

    prices = frames["daily_price_raw"]
    legacy_prices = prices.loc[
        prices["security_id"].astype(str).eq(script.CURRENT_DD_ID)
        & prices["session"].astype(str).le(script.LEGACY_DD_LAST)
    ].copy()
    legacy_prices["security_id"] = script.LEGACY_DD_ID
    legacy_prices[["open", "high", "low", "close"]] *= 4.0
    artifacts = tuple(
        _artifact(
            source=f"eodhd_{endpoint}",
            url=script._legacy_dd_public_url(endpoint),
            content=json.dumps({"fixture": endpoint}).encode(),
        )
        for endpoint in script.LEGACY_DD_ENDPOINTS
    )
    legacy_prices["source"] = artifacts[0].source
    legacy_prices["source_url"] = artifacts[0].source_url
    legacy_prices["retrieved_at"] = artifacts[0].retrieved_at
    legacy_prices["source_hash"] = artifacts[0].source_hash
    legacy = script.LegacyDdEvidence(
        prices=legacy_prices,
        actions=frames["corporate_actions"].iloc[0:0].copy(),
        artifacts=artifacts,
        wiki_url=script.WIKI_DD_URL,
        wiki_hash=script.WIKI_DD_SHA256,
        overlap_rows=script.WIKI_DD_EXPECTED_ROWS,
    )
    prepared = script.prepare_dwdp_repair(
        frames,
        script.load_official_evidence(script.DEFAULT_SEC_CACHE),
        legacy,
        completed_session=release.completed_session,
        source_version="test-dwdp-repair",
    )
    assert prepared.summary["status"] == "validated_dry_run"
    assert prepared.summary["unsafe_dd_us_backcast_price_rows_removed"] == 672
    assert not prepared.frames["security_master"]["security_id"].astype(str).eq(
        script.CURRENT_DD_ID
    ).any()
    canonical = prepared.frames["security_master"].loc[
        prepared.frames["security_master"]["security_id"].astype(str).eq(
            script.DWDP_ID
        )
    ].iloc[0]
    assert canonical["primary_symbol"] == "DD"
    assert canonical["active_to"] == ""
    assert set(
        prepared.frames["symbol_history"].loc[
            prepared.frames["symbol_history"]["security_id"].astype(str).eq(
                script.DWDP_ID
            ),
            "symbol",
        ]
    ) == {"DWDP", "DD"}
