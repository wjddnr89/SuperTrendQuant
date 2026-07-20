from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from supertrend_quant.market_store.ingest import SourceArtifact
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    DatasetManifest,
    sha256_bytes,
)
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.storage import LocalObjectStore


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts/repair_us_bbby_identity.py"
)
SPEC = importlib.util.spec_from_file_location("repair_us_bbby_identity", SCRIPT_PATH)
script = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


OLD_SESSIONS = (
    "2015-01-02",
    "2015-01-05",
    "2015-01-06",
    "2023-05-02",
)
WIKI_SESSIONS = OLD_SESSIONS[:3]
CURRENT_SESSIONS = (
    *OLD_SESSIONS,
    "2023-11-06",
    "2025-08-29",
)
COMPLETED_SESSION = "2025-08-29"


def _source_fields(label: str) -> dict:
    return {
        "source": label,
        "source_url": f"https://example.test/{label}",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": sha256_bytes(label.encode()),
    }


def _master(security_id: str, symbol: str, provider_symbol: str, name: str) -> dict:
    return {
        "security_id": security_id,
        "primary_symbol": symbol,
        "provider_symbol": provider_symbol,
        "action_provider_symbol": provider_symbol,
        "name": name,
        "exchange": "NASDAQ",
        "asset_type": "STOCK",
        "currency": "USD",
        "country": "US",
        "active_from": "2015-01-02",
        "active_to": "",
        **_source_fields("catalog"),
    }


def _history(security_id: str, symbol: str) -> dict:
    return {
        "security_id": security_id,
        "symbol": symbol,
        "exchange": "NASDAQ",
        "effective_from": "2015-01-01",
        "effective_to": "",
        **_source_fields("catalog"),
    }


def _price(
    security_id: str,
    session: str,
    close: float,
    *,
    source: str,
    source_url: str,
    source_hash: str,
) -> dict:
    return {
        "security_id": security_id,
        "session": session,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1000,
        "currency": "USD",
        "source": source,
        "source_url": source_url,
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": source_hash,
    }


def _factor(security_id: str, session: str) -> dict:
    return {
        "security_id": security_id,
        "session": session,
        "split_factor": 1.0,
        "total_return_factor": 1.0,
        "source_version": "fixture",
        "calculated_at": "2026-07-18T00:00:00Z",
        "source": "derived",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": "fixture",
    }


def _anchor(index_id: str, date: str, security_id: str) -> dict:
    return {
        "index_id": index_id,
        "anchor_date": date,
        "security_id": security_id,
        "official": False,
        "source_url": "https://example.test/index",
        "source_kind": "community",
        "source": "community_history",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": "index-hash",
    }


def _event(index_id: str, date: str, security_id: str) -> dict:
    return {
        "event_id": sha256_bytes(f"{index_id}|{date}|{security_id}".encode()),
        "index_id": index_id,
        "announcement_date": "",
        "effective_date": date,
        "operation": "REMOVE",
        "security_id": security_id,
        "official": False,
        "source_url": "https://example.test/index",
        "source_kind": "community",
        "source": "community_history",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": "index-hash",
    }


def _archive_row(
    archive_id: str, source_url: str, object_path: str, *, dataset: str
) -> dict:
    return {
        "archive_id": archive_id,
        "dataset": dataset,
        "object_path": object_path,
        "content_type": "application/json",
        "effective_date": COMPLETED_SESSION,
        "source": dataset,
        "source_url": source_url,
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": archive_id,
    }


def _write_catalog(root: Path, rows: list[dict], source_url: str) -> dict:
    content = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    source_hash = sha256_bytes(content)
    relative = f"archives/{COMPLETED_SESSION}/{source_hash}.json.gz"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(content, mtime=0))
    return _archive_row(source_hash, source_url, relative, dataset="eodhd_exchange_symbols")


def _frames(root: Path) -> dict[str, pd.DataFrame]:
    contaminated_hash = "contaminated-old-price"
    current_hash = "current-price"
    current_closes = {
        session: 24.0 - number for number, session in enumerate(CURRENT_SESSIONS)
    }
    prices = []
    for session, close in current_closes.items():
        prices.append(
            _price(
                script.CURRENT_SECURITY_ID,
                session,
                close,
                source="eodhd_eod",
                source_url="https://eodhd.com/api/eod/BBBY.US",
                source_hash=current_hash,
            )
        )
        prices.append(
            _price(
                script.OLD_SECURITY_ID,
                session,
                close,
                source="eodhd_eod",
                source_url="https://eodhd.com/api/eod/BBBY_old.US",
                source_hash=contaminated_hash,
            )
        )
    factors = [
        _factor(row["security_id"], row["session"])
        for row in prices
    ]
    active_rows = [
        {
            "Code": "BBBY",
            "Name": "Bed Bath & Beyond, Inc.",
            "Exchange": "NYSE",
            "Currency": "USD",
            "Type": "Common Stock",
            "Isin": script.CURRENT_ISIN,
        }
    ]
    delisted_rows = [
        {
            "Code": "BBBY_old",
            "Name": "Bed Bath & Beyond Inc",
            "Exchange": "NASDAQ",
            "Currency": "USD",
            "Type": "Common Stock",
            "Isin": script.OLD_ISIN,
        },
        {
            "Code": "BBBYQ",
            "Name": "Bed Bath & Beyond Inc",
            "Exchange": "PINK",
            "Currency": "USD",
            "Type": "Common Stock",
            "Isin": script.OLD_ISIN,
        },
        {
            "Code": "OSTK",
            "Name": "Overstockcom Inc",
            "Exchange": "NASDAQ",
            "Currency": "USD",
            "Type": "Common Stock",
            "Isin": script.CURRENT_ISIN,
        },
        {
            "Code": "BYON",
            "Name": "Beyond Inc",
            "Exchange": "NASDAQ",
            "Currency": "USD",
            "Type": "Common Stock",
            "Isin": script.CURRENT_ISIN,
        },
    ]
    archive = pd.DataFrame(
        [
            _write_catalog(root, active_rows, script.ACTIVE_CATALOG_URL),
            _write_catalog(root, delisted_rows, script.DELISTED_CATALOG_URL),
        ]
    )
    return {
        "security_master": pd.DataFrame(
            [
                {
                    **_master(
                        script.OLD_SECURITY_ID,
                        "BBBY",
                        "BBBY_old.US",
                        "Bed Bath & Beyond Inc",
                    ),
                    "active_to": "2025-08-29",
                },
                _master(
                    script.CURRENT_SECURITY_ID,
                    "BBBY",
                    "BBBY.US",
                    "Bed Bath & Beyond, Inc.",
                ),
            ]
        ),
        "symbol_history": pd.DataFrame(
            [
                _history(script.OLD_SECURITY_ID, "BBBY"),
                _history(script.CURRENT_SECURITY_ID, "BBBY"),
            ]
        ),
        "daily_price_raw": pd.DataFrame(prices),
        "corporate_actions": pd.DataFrame(
            columns=dataset_spec("corporate_actions").required_columns
        ),
        "adjustment_factors": pd.DataFrame(factors),
        "index_constituent_anchors": pd.DataFrame(
            [
                _anchor("nasdaq100", "2015-01-01", script.CURRENT_SECURITY_ID),
                _anchor("sp500", "2015-01-07", script.OLD_SECURITY_ID),
            ]
        ),
        "index_membership_events": pd.DataFrame(
            [
                _event("nasdaq100", "2016-12-19", script.CURRENT_SECURITY_ID),
                _event("sp500", "2017-07-26", script.OLD_SECURITY_ID),
            ]
        ),
        "source_archive": archive,
    }


def _release() -> DataRelease:
    return DataRelease(
        version="base-release",
        created_at="2026-07-18T00:00:00Z",
        completed_session=COMPLETED_SESSION,
        dataset_versions={name: f"old-{name}" for name in script.WRITE_DATASETS},
        warnings=(),
    )


class _FrameRepository:
    def __init__(self, root: Path, frames: dict[str, pd.DataFrame]):
        self.root = root
        self.frames = frames
        self.release = _release()
        self.release_etag = "release-etag"

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        return self.frames[dataset].copy()

    def current_pointer(self, dataset: str):
        return SimpleNamespace(version=self.release.dataset_versions[dataset]), f"etag-{dataset}"

    def current_release(self):
        return self.release, self.release_etag


def _official_bundle() -> script.OfficialEvidenceBundle:
    values = {
        script.OLD_DELISTING_URL: (
            b"Bed Bath & Beyond Inc trading will be suspended at the opening of "
            b"business on May 3, 2023"
        ),
        script.OSTK_TO_BYON_URL: (
            b"Overstock.com OSTK trading ends November 3, 2023 and BYON trading "
            b"commences November 6, 2023"
        ),
        script.BYON_TO_BBBY_URL: (
            b"Beyond, Inc BYON trades through August 28, 2025 and BBBY begins "
            b"August 29, 2025"
        ),
    }
    return script.OfficialEvidenceBundle(
        tuple(
            SourceArtifact(
                source="sec_bbby_identity_evidence",
                source_url=url,
                retrieved_at="2026-07-18T00:00:00Z",
                content=values[url],
                content_type="text/html",
            )
            for url in script.OFFICIAL_URLS
        )
    )


def _wiki_fixture_constants(artifact: SourceArtifact):
    return mock.patch.multiple(
        script,
        WIKI_SHA256=artifact.source_hash,
        WIKI_FIRST_DATE=WIKI_SESSIONS[0],
        WIKI_LAST_DATE=WIKI_SESSIONS[-1],
        WIKI_EXPECTED_ROWS=len(WIKI_SESSIONS),
    )


def _bundles(*, partial_eodhd: bool = False, mismatch: bool = False):
    closes = (75.0, 73.0, 72.0, 0.10)
    eod_rows = [
        {
            "date": session,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000,
        }
        for session, close in zip(OLD_SESSIONS, closes)
    ]
    if partial_eodhd:
        eod_rows = eod_rows[:2]
    eod_artifacts = (
        SourceArtifact(
            source="eodhd_eod",
            source_url="https://eodhd.com/api/eod/BBBYQ.US",
            retrieved_at="2026-07-18T00:00:00Z",
            content=script._canonical_json_bytes(eod_rows),
            content_type="application/json",
        ),
        SourceArtifact(
            source="eodhd_div",
            source_url="https://eodhd.com/api/div/BBBYQ.US",
            retrieved_at="2026-07-18T00:00:00Z",
            content=script._canonical_json_bytes([]),
            content_type="application/json",
        ),
        SourceArtifact(
            source="eodhd_splits",
            source_url="https://eodhd.com/api/splits/BBBYQ.US",
            retrieved_at="2026-07-18T00:00:00Z",
            content=script._canonical_json_bytes([]),
            content_type="application/json",
        ),
    )
    eod_prices = script._eodhd_price_frame(eod_artifacts[0])
    wiki_scale = 0.80
    csv_rows = ["date,open,high,low,close,volume"]
    for number, (session, close) in enumerate(zip(WIKI_SESSIONS, closes)):
        scale = wiki_scale * (1.20 if mismatch and number == 2 else 1.0)
        adjusted_close = close * scale
        csv_rows.append(
            f"{session},{adjusted_close},{adjusted_close * 1.01},"
            f"{adjusted_close * 0.99},{adjusted_close},1000"
        )
    wiki_artifact = SourceArtifact(
        source="quandl_wiki_adjusted_git_csv",
        source_url=script.WIKI_URL,
        retrieved_at="2026-07-18T00:00:00Z",
        content=("\n".join(csv_rows) + "\n").encode(),
        content_type="text/csv",
    )
    with _wiki_fixture_constants(wiki_artifact):
        wiki_prices = script._wiki_price_frame(wiki_artifact)
    return (
        script.PriceSourceBundle(
            prices=eod_prices,
            actions=pd.DataFrame(columns=dataset_spec("corporate_actions").required_columns),
            artifacts=eod_artifacts,
            http_attempts=3,
        ),
        script.PriceSourceBundle(
            prices=wiki_prices,
            actions=pd.DataFrame(columns=dataset_spec("corporate_actions").required_columns),
            artifacts=(wiki_artifact,),
            http_attempts=1,
        ),
    )


def _expected_sessions(start: str, end: str):
    if start == WIKI_SESSIONS[0] and end == WIKI_SESSIONS[-1]:
        return WIKI_SESSIONS
    if end == script.OLD_LAST_TRADING_DATE:
        return OLD_SESSIONS
    if start == script.OLD_START and end == COMPLETED_SESSION:
        return CURRENT_SESSIONS
    raise AssertionError((start, end))


class BbbyIdentityRepairTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.frames = _frames(self.root)
        self.repository = _FrameRepository(self.root, self.frames)

    def tearDown(self):
        self.temp.cleanup()

    def _preflight(self):
        return script.build_local_preflight(self.repository, self.repository.release)

    def test_preflight_proves_two_isins_and_price_contamination(self):
        preflight = self._preflight()
        self.assertFalse(preflight.already_repaired)
        self.assertEqual(preflight.catalog.old_row["Isin"], script.OLD_ISIN)
        self.assertEqual(preflight.catalog.bbbyq_row["Isin"], script.OLD_ISIN)
        self.assertEqual(preflight.catalog.active_row["Isin"], script.CURRENT_ISIN)
        self.assertEqual(preflight.contamination_overlap_rows, len(CURRENT_SESSIONS))
        self.assertIn("contaminated-old-price", preflight.contaminated_price_hashes)

    def test_offline_plan_constructs_no_network_source_factory(self):
        args = argparse.Namespace(
            cache_root=str(self.root),
            offline_plan=True,
            apply=False,
            fetch_eodhd_bbbyq=False,
            fetch_wiki=False,
            fetch_official_evidence=False,
        )

        def forbidden(*_args, **_kwargs):
            raise AssertionError("network source factory was constructed")

        result = script.run(
            args,
            repository_factory=lambda _root: self.repository,
            eodhd_source_factory=forbidden,
            wiki_source_factory=forbidden,
            official_source_factory=forbidden,
        )
        self.assertEqual(result["status"], "offline_plan")
        self.assertEqual(result["http_attempts"], 0)
        self.assertEqual(result["network_clients_constructed"], 0)
        self.assertEqual(result["maximum_total_http_attempts"], 7)

    @mock.patch.object(script, "_expected_sessions", side_effect=_expected_sessions)
    def test_full_bbbyq_history_is_primary_after_whole_overlap_pass(self, _mocked):
        preflight = self._preflight()
        eodhd, wiki = _bundles()
        with _wiki_fixture_constants(wiki.artifacts[0]):
            prepared = script.prepare_repair(
                self.repository,
                self.repository.release,
                self.repository.release_etag,
                preflight,
                eodhd=eodhd,
                wiki=wiki,
                official=_official_bundle(),
            )
        self.assertEqual(prepared.summary["selected_primary"], "eodhd_bbbyq")
        self.assertEqual(
            prepared.summary["independent_cross_validation"]["overlap_session_count"],
            len(WIKI_SESSIONS),
        )
        self.assertTrue(
            prepared.summary["independent_cross_validation"][
                "all_overlap_sessions_compared"
            ]
        )
        self.assertEqual(prepared.warnings, ())
        prices = prepared.frames["daily_price_raw"]
        old = prices.loc[prices["security_id"].eq(script.OLD_SECURITY_ID)]
        self.assertEqual(set(old["source"]), {"eodhd_eod"})
        self.assertFalse(old["source_url"].str.contains("BBBY_old|/BBBY.US").any())
        current = prices.loc[prices["security_id"].eq(script.CURRENT_SECURITY_ID)]
        self.assertEqual(len(current), len(CURRENT_SESSIONS))
        master = prepared.frames["security_master"].set_index("security_id")
        self.assertEqual(master.loc[script.OLD_SECURITY_ID, "isin"], script.OLD_ISIN)
        self.assertEqual(
            master.loc[script.CURRENT_SECURITY_ID, "isin"], script.CURRENT_ISIN
        )
        history = prepared.frames["symbol_history"]
        current_symbols = history.loc[
            history["security_id"].eq(script.CURRENT_SECURITY_ID), "symbol"
        ].tolist()
        self.assertEqual(current_symbols, ["OSTK", "BYON", "BBBY"])
        ndx = prepared.frames["index_constituent_anchors"].query(
            "index_id == 'nasdaq100'"
        )
        self.assertEqual(ndx.iloc[0]["security_id"], script.OLD_SECURITY_ID)

    @mock.patch.object(script, "_expected_sessions", side_effect=_expected_sessions)
    def test_partial_bbbyq_fails_and_wiki_never_becomes_primary(self, _mocked):
        preflight = self._preflight()
        eodhd, wiki = _bundles(partial_eodhd=True)
        with _wiki_fixture_constants(wiki.artifacts[0]):
            with self.assertRaisesRegex(ValueError, "full-history gate failed"):
                script.prepare_repair(
                    self.repository,
                    self.repository.release,
                    self.repository.release_etag,
                    preflight,
                    eodhd=eodhd,
                    wiki=wiki,
                    official=_official_bundle(),
                )

    @mock.patch.object(script, "_expected_sessions", side_effect=_expected_sessions)
    def test_full_but_mismatched_bbbyq_fails_closed(self, _mocked):
        preflight = self._preflight()
        eodhd, wiki = _bundles(mismatch=True)
        with _wiki_fixture_constants(wiki.artifacts[0]):
            with self.assertRaisesRegex(ValueError, "adjusted overlap comparison failed"):
                script.prepare_repair(
                    self.repository,
                    self.repository.release,
                    self.repository.release_etag,
                    preflight,
                    eodhd=eodhd,
                    wiki=wiki,
                    official=_official_bundle(),
                )

    def test_volume_crosscheck_uses_full_history_match_ratio(self):
        sessions = tuple(
            value.date().isoformat()
            for value in pd.bdate_range("2020-01-02", periods=100)
        )
        base = pd.DataFrame(
            {
                "session": sessions,
                "open": [10.0] * len(sessions),
                "high": [10.5] * len(sessions),
                "low": [9.5] * len(sessions),
                "close": [10.1] * len(sessions),
                "volume": [1_000] * len(sessions),
            }
        )
        actions = pd.DataFrame(
            columns=dataset_spec("corporate_actions").required_columns
        )
        independent = base.copy()
        independent.loc[10, "volume"] = 1_300
        with (
            mock.patch.object(script, "OLD_START", sessions[0]),
            mock.patch.object(script, "OLD_LAST_TRADING_DATE", sessions[-1]),
            mock.patch.object(script, "WIKI_FIRST_DATE", sessions[0]),
            mock.patch.object(script, "WIKI_LAST_DATE", sessions[-1]),
            mock.patch.object(script, "_expected_sessions", return_value=sessions),
        ):
            result = script._cross_validate_prices(
                base, independent, actions, sessions
            )
            self.assertEqual(result["volume_mismatch_sessions"], 1)
            self.assertAlmostEqual(result["volume_match_ratio"], 0.99)
            self.assertEqual(result["mismatch_sessions"], 0)

            below_floor = independent.copy()
            below_floor.loc[[10, 20, 30], "volume"] = 1_300
            with self.assertRaisesRegex(
                ValueError, "volume match ratio below the audited floor"
            ):
                script._cross_validate_prices(
                    base, below_floor, actions, sessions
                )

    def test_official_evidence_requires_all_three_exact_boundary_facts(self):
        bundle = _official_bundle()
        script.validate_official_evidence(bundle)
        changed = list(bundle.artifacts)
        changed[-1] = SourceArtifact(
            source=changed[-1].source,
            source_url=changed[-1].source_url,
            retrieved_at=changed[-1].retrieved_at,
            content=b"Beyond Inc BYON only",
            content_type="text/html",
        )
        with self.assertRaisesRegex(ValueError, "lacks audited facts"):
            script.validate_official_evidence(
                script.OfficialEvidenceBundle(tuple(changed))
            )

    @mock.patch.object(script, "_expected_sessions", side_effect=_expected_sessions)
    def test_replay_gate_rejects_current_issuer_in_historical_ndx_anchor(self, _mocked):
        preflight = self._preflight()
        eodhd, wiki = _bundles()
        with _wiki_fixture_constants(wiki.artifacts[0]):
            prepared = script.prepare_repair(
                self.repository,
                self.repository.release,
                self.repository.release_etag,
                preflight,
                eodhd=eodhd,
                wiki=wiki,
                official=_official_bundle(),
            )
        broken = {name: frame.copy() for name, frame in prepared.frames.items()}
        anchors = broken["index_constituent_anchors"]
        mask = anchors["index_id"].eq("nasdaq100")
        anchors.loc[mask, "security_id"] = script.CURRENT_SECURITY_ID
        with self.assertRaisesRegex(ValueError, "canonical old BBBY anchor"):
            script.validate_index_replay_gate(
                preflight.existing,
                broken,
                completed_session=COMPLETED_SESSION,
            )

    @mock.patch.object(script, "_expected_sessions", side_effect=_expected_sessions)
    def test_full_history_gate_rejects_a_missing_legacy_session(self, _mocked):
        preflight = self._preflight()
        eodhd, wiki = _bundles()
        with _wiki_fixture_constants(wiki.artifacts[0]):
            prepared = script.prepare_repair(
                self.repository,
                self.repository.release,
                self.repository.release_etag,
                preflight,
                eodhd=eodhd,
                wiki=wiki,
                official=_official_bundle(),
            )
        broken = {name: frame.copy() for name, frame in prepared.frames.items()}
        prices = broken["daily_price_raw"]
        broken["daily_price_raw"] = prices.loc[
            ~(
                prices["security_id"].eq(script.OLD_SECURITY_ID)
                & prices["session"].astype(str).eq(OLD_SESSIONS[1])
            )
        ].copy()
        with self.assertRaisesRegex(ValueError, "full-history gate failed"):
            script.validate_full_history_gate(
                preflight,
                broken,
                completed_session=COMPLETED_SESSION,
                expected_old=OLD_SESSIONS,
            )


class EodhdCacheTest(unittest.TestCase):
    def test_three_endpoint_cache_is_immutable_and_reused_without_client(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            class Client:
                def __init__(self):
                    self.attempt_count = 0

                def safe_url(self, endpoint, *, params=None):
                    return f"https://eodhd.com/api/{endpoint}"

                def get_json(self, endpoint, *, params=None):
                    self.attempt_count += 1
                    if endpoint.startswith("eod/"):
                        return [
                            {
                                "date": "2023-05-02",
                                "open": 0.1,
                                "high": 0.11,
                                "low": 0.09,
                                "close": 0.1,
                                "volume": 100,
                            }
                        ]
                    return []

            client = Client()
            source = script.EodhdBbbyqSource(
                root,
                allow_http=True,
                client_factory=lambda: client,
            )
            first = source.fetch()
            self.assertEqual(client.attempt_count, 3)
            self.assertEqual(first.http_attempts, 3)
            self.assertTrue(all(source.path(item).is_file() for item in script.EODHD_ENDPOINTS))

            def forbidden():
                raise AssertionError("cached EODHD source constructed a client")

            cached = script.EodhdBbbyqSource(
                root,
                allow_http=False,
                client_factory=forbidden,
            ).fetch()
            self.assertEqual(cached.http_attempts, 0)
            self.assertEqual(len(cached.artifacts), 3)
            self.assertEqual(cached.artifacts[0].content, first.artifacts[0].content)

    def test_failed_endpoint_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []

            class Client:
                attempt_count = 0

                def safe_url(self, endpoint, *, params=None):
                    return f"https://eodhd.com/api/{endpoint}"

                def get_json(self, endpoint, *, params=None):
                    self.attempt_count += 1
                    calls.append(endpoint)
                    raise RuntimeError("single failure")

            with self.assertRaisesRegex(RuntimeError, "single failure"):
                script.EodhdBbbyqSource(
                    Path(temp),
                    allow_http=True,
                    client_factory=Client,
                ).fetch()
            self.assertEqual(calls, ["eod/BBBYQ.US"])


class PinnedWikiCacheTest(unittest.TestCase):
    @staticmethod
    def _content() -> bytes:
        rows = ["date,open,high,low,close,volume"]
        for session, close in zip(WIKI_SESSIONS, (60.0, 58.4, 57.6)):
            rows.append(f"{session},{close},{close * 1.01},{close * 0.99},{close},1000")
        return ("\n".join(rows) + "\n").encode()

    def test_one_shot_pinned_cache_is_immutable_and_reused(self):
        content = self._content()
        artifact = SourceArtifact(
            source="quandl_wiki_adjusted_git_csv",
            source_url=script.WIKI_URL,
            retrieved_at="2026-07-18T00:00:00Z",
            content=content,
            content_type="text/plain",
        )
        calls = []

        class Response:
            status = 200
            headers = {"Content-Type": "text/plain"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return content

        def opener(request, timeout):
            calls.append((request.full_url, timeout))
            return Response()

        with tempfile.TemporaryDirectory() as temp:
            with _wiki_fixture_constants(artifact), mock.patch.object(
                script, "_expected_sessions", side_effect=_expected_sessions
            ):
                source = script.PinnedWikiBbbySource(
                    Path(temp), allow_http=True, opener=opener
                )
                first = source.fetch()
                self.assertEqual(source.http_attempts, 1)
                self.assertEqual(calls, [(script.WIKI_URL, 60)])
                self.assertTrue(source.path().is_file())

                def forbidden(*_args, **_kwargs):
                    raise AssertionError("cached WIKI source made another request")

                cached = script.PinnedWikiBbbySource(
                    Path(temp), allow_http=False, opener=forbidden
                ).fetch()
                self.assertEqual(cached.http_attempts, 0)
                self.assertEqual(cached.artifacts[0].content, first.artifacts[0].content)

    def test_hash_mismatch_consumes_one_attempt_without_retry_or_cache(self):
        content = self._content()
        calls = []

        class Response:
            status = 200
            headers = {"Content-Type": "text/plain"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                calls.append("read")
                return content

        with tempfile.TemporaryDirectory() as temp:
            source = script.PinnedWikiBbbySource(
                Path(temp), allow_http=True, opener=lambda *_args, **_kwargs: Response()
            )
            with self.assertRaisesRegex(ValueError, "hash differs"):
                source.fetch()
            self.assertEqual(source.http_attempts, 1)
            self.assertEqual(calls, ["read"])
            self.assertFalse(source.path().exists())


class OfficialCacheTest(unittest.TestCase):
    def test_three_sec_urls_use_one_attempt_each_then_reuse_cache(self):
        contents = {item.source_url: item.content for item in _official_bundle().artifacts}
        calls = []

        class Response:
            status = 200
            headers = {"Content-Type": "text/html"}

            def __init__(self, content):
                self.content = content

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return self.content

        def fake_urlopen(request, timeout):
            self.assertEqual(timeout, 60)
            calls.append(request.full_url)
            return Response(contents[request.full_url])

        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(script, "urlopen", side_effect=fake_urlopen):
                source = script.OfficialEvidenceSource(Path(temp), allow_http=True)
                first = source.load()
            self.assertEqual(calls, list(script.OFFICIAL_URLS))
            self.assertEqual(source.http_attempts, 3)
            with mock.patch.object(
                script, "urlopen", side_effect=AssertionError("cache miss")
            ):
                cached = script.OfficialEvidenceSource(
                    Path(temp), allow_http=False
                ).load()
            self.assertEqual(
                [item.source_hash for item in cached.artifacts],
                [item.source_hash for item in first.artifacts],
            )


class _TransactionRepository:
    def __init__(self, root: Path, *, fail_dataset: str):
        self.root = root
        self.objects = LocalObjectStore(root)
        self.fail_dataset = fail_dataset
        versions = {}
        for dataset in script.WRITE_DATASETS:
            version = f"old-{dataset}"
            versions[dataset] = version
            manifest = DatasetManifest.create(dataset, version, COMPLETED_SESSION, ())
            manifest_path = f"datasets/{dataset}/versions/{version}/manifest.json"
            self.objects.put(manifest_path, manifest.to_bytes())
            self.objects.put(
                self.current_key(dataset),
                CurrentPointer.create(manifest, manifest_path).to_bytes(),
            )
        self.release = DataRelease(
            version="base-release",
            created_at="2026-07-18T00:00:00Z",
            completed_session=COMPLETED_SESSION,
            dataset_versions=versions,
        )
        self.objects.put(f"releases/{self.release.version}.json", self.release.to_bytes())
        self.objects.put("releases/current.json", self.release.to_bytes())

    @staticmethod
    def current_key(dataset: str) -> str:
        return f"datasets/{dataset}/current.json"

    def current_release(self):
        value = self.objects.get("releases/current.json")
        return DataRelease.from_bytes(value.data), value.etag

    def current_pointer(self, dataset: str):
        value = self.objects.get(self.current_key(dataset))
        return CurrentPointer.from_bytes(value.data), value.etag

    def write_frame(
        self,
        dataset,
        _frame,
        *,
        completed_session,
        expected_pointer_etag,
        version,
        **_kwargs,
    ):
        manifest = DatasetManifest.create(dataset, version, completed_session, ())
        manifest_path = f"datasets/{dataset}/versions/{version}/manifest.json"
        self.objects.put(manifest_path, manifest.to_bytes())
        pointer = CurrentPointer.create(manifest, manifest_path)
        self.objects.put(
            self.current_key(dataset),
            pointer.to_bytes(),
            if_match=expected_pointer_etag,
        )
        if dataset == self.fail_dataset:
            raise RuntimeError("injected write failure")
        return SimpleNamespace(manifest=manifest, conflict=False, conflict_path="")


class AtomicRollbackTest(unittest.TestCase):
    def test_apply_restores_release_and_all_pointers_after_partial_write(self):
        with tempfile.TemporaryDirectory() as temp:
            repository = _TransactionRepository(
                Path(temp), fail_dataset="symbol_history"
            )
            release, release_etag = repository.current_release()
            old_release = repository.objects.get("releases/current.json").data
            old_pointers = {
                dataset: repository.objects.get(repository.current_key(dataset)).data
                for dataset in script.WRITE_DATASETS
            }
            pointer_etags = {
                dataset: repository.current_pointer(dataset)[1]
                for dataset in script.WRITE_DATASETS
            }
            prepared = script.PreparedRepair(
                release=release,
                release_etag=release_etag,
                pointer_etags=pointer_etags,
                frames={name: pd.DataFrame() for name in script.WRITE_DATASETS},
                archive_artifacts=(),
                warnings=(),
                summary={"status": "validated_dry_run"},
            )
            with self.assertRaisesRegex(RuntimeError, "injected write failure"):
                script.apply_repair(repository, prepared)
            self.assertEqual(
                repository.objects.get("releases/current.json").data, old_release
            )
            for dataset in script.WRITE_DATASETS:
                self.assertEqual(
                    repository.objects.get(repository.current_key(dataset)).data,
                    old_pointers[dataset],
                )
            journals = list(
                (Path(temp) / "transactions/bbby-identity-repair").glob("*.json")
            )
            self.assertEqual(len(journals), 1)
            self.assertEqual(json.loads(journals[0].read_bytes())["status"], "rolled_back")
            self.assertFalse((Path(temp) / "recovery/bbby-identity-repair").exists())


if __name__ == "__main__":
    unittest.main()
