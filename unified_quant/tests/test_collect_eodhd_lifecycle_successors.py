from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from supertrend_quant.market_store.ingest import SourceArtifact
from supertrend_quant.market_store.lifecycle import build_lifecycle_candidates
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    DatasetManifest,
)
from supertrend_quant.market_store.storage import LocalObjectStore
from supertrend_quant.market_store.schemas import dataset_spec


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "collect_eodhd_lifecycle_successors.py"
)
SPEC = importlib.util.spec_from_file_location(
    "collect_eodhd_lifecycle_successors", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Could not load {SCRIPT_PATH}")
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def _catalogs(specs):
    by_kind = {"active": [], "delisted": []}
    for item in specs:
        by_kind[item.catalog_kind].append(
            {
                "Code": item.provider_code,
                "Name": " ".join(item.name_tokens),
                "Type": "Common Stock",
                "Exchange": "NYSE",
                "Currency": "USD",
                "Isin": (
                    "USG2554F1134" if item.provider_code == "COV" else ""
                ),
            }
        )
    by_kind["active"].extend(
        [
            {
                "Code": "ACT",
                "Name": "Enact Holdings Inc",
                "Type": "Common Stock",
            },
            {
                "Code": "VAL",
                "Name": "Valaris Ltd",
                "Type": "Common Stock",
            },
        ]
    )
    return {
        kind: script.CatalogArchive(
            kind=kind,
            rows=tuple(rows),
            source_url=f"https://eodhd.test/catalog?delisted={int(kind == 'delisted')}",
            retrieved_at="2026-07-18T00:00:00Z",
            source_hash=f"{kind}-hash",
        )
        for kind, rows in by_kind.items()
    }


def _price(security_id: str, session: str, close: float = 10.0) -> dict:
    return {
        "security_id": security_id,
        "session": session,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 100,
        "currency": "USD",
        "source": "eodhd_eod",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": f"price-{security_id}-{session}",
    }


def _action(
    security_id: str,
    effective_date: str,
    *,
    source: str = "eodhd_div",
) -> dict:
    return {
        "event_id": f"{security_id}-{effective_date}-{source}",
        "security_id": security_id,
        "action_type": "cash_dividend",
        "effective_date": effective_date,
        "ex_date": effective_date,
        "announcement_date": "",
        "record_date": "",
        "payment_date": "",
        "cash_amount": 0.1,
        "ratio": None,
        "currency": "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": False,
        "source_url": "https://eodhd.test/div",
        "source_kind": "provider",
        "source": source,
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": f"action-{security_id}-{effective_date}-{source}",
    }


def _master(
    security_id: str,
    symbol: str,
    provider_symbol: str,
    name: str,
) -> dict:
    return {
        "security_id": security_id,
        "primary_symbol": symbol,
        "provider_symbol": provider_symbol,
        "action_provider_symbol": provider_symbol,
        "name": name,
        "exchange": "NYSE",
        "asset_type": "STOCK",
        "currency": "USD",
        "country": "US",
        "active_from": "2015-01-01",
        "active_to": "",
        "source": "eodhd_exchange_symbols",
        "source_url": "https://eodhd.test/catalog",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": f"master-{security_id}",
    }


def _history(security_id: str, symbol: str) -> dict:
    return {
        "security_id": security_id,
        "symbol": symbol,
        "exchange": "NYSE",
        "effective_from": "2015-01-01",
        "effective_to": "",
        "source": "eodhd_exchange_symbols",
        "source_url": "https://eodhd.test/catalog",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": f"history-{security_id}",
    }


def _artifacts(specs):
    output = []
    for item in specs:
        for endpoint in ("eod", "div", "splits"):
            output.append(
                SourceArtifact(
                    source=f"eodhd_{endpoint}",
                    source_url=(
                        f"https://eodhd.test/api/{endpoint}/{item.provider_symbol}"
                        f"?from=2015-01-01&to=2026-07-15"
                    ),
                    retrieved_at="2026-07-18T00:00:00Z",
                    content=f"{endpoint}-{item.provider_symbol}".encode(),
                    content_type="application/json",
                )
            )
    return tuple(output)


def _cov_artifacts():
    output = []
    for endpoint in ("eod", "div", "splits"):
        output.append(
            SourceArtifact(
                source=f"eodhd_{endpoint}",
                source_url=(
                    f"https://eodhd.test/api/{endpoint}/COV.US"
                    "?from=2015-01-01&to=2015-01-26"
                ),
                retrieved_at="2026-07-18T00:00:00Z",
                content=f"{endpoint}-COV.US".encode(),
                content_type="application/json",
            )
        )
    return tuple(output)


def _valid_cov_fetch():
    direct = _valid_cov_direct_fetch()
    wiki = _valid_cov_wiki_fetch()
    return script.FetchedBundle(
        prices=direct.prices.copy(),
        corporate_actions=pd.DataFrame(
            columns=dataset_spec("corporate_actions").required_columns
        ),
        artifacts=(*_cov_artifacts(), *direct.artifacts, *wiki.artifacts),
        missing_symbols=(),
    )


def _cov_directindex_csv() -> bytes:
    rows = [",Date,Open,High,Low,Close,Adjusted_close,Volume"]
    for offset, session in enumerate(script.COV_EXPECTED_SESSIONS):
        close = 100.0 + offset
        rows.append(
            f"{offset},{session},{close},{close + 1},{close - 1},{close},{close},{1000 + offset}"
        )
    return ("\n".join(rows) + "\n").encode()


def _cov_directindex_artifacts():
    contents = {
        "cov_csv": _cov_directindex_csv(),
        "readme": b"DirectIndexing historical price and volume data.\n",
        "license": b"MIT License\n",
    }
    return tuple(
        SourceArtifact(
            source=f"cov_directindex_{role}",
            source_url=script.COV_DIRECTINDEX_URLS[role],
            retrieved_at="2026-07-18T00:00:00Z",
            content=contents[role],
            content_type="text/csv" if role == "cov_csv" else "text/plain",
        )
        for role in script.COV_DIRECTINDEX_URLS
    )


def _valid_cov_direct_fetch():
    artifacts = _cov_directindex_artifacts()
    hashes = {
        role: artifact.source_hash
        for role, artifact in zip(script.COV_DIRECTINDEX_URLS, artifacts)
    }
    with patch.object(script, "COV_DIRECTINDEX_SHA256", hashes):
        prices = script._parse_cov_directindex_csv(artifacts[0])
    return script.CovDirectIndexBundle(
        prices=prices,
        artifacts=artifacts,
        http_attempts=3,
    )


def _cov_direct_test_hashes(bundle):
    return {
        role: artifact.source_hash
        for role, artifact in zip(script.COV_DIRECTINDEX_URLS, bundle.artifacts)
    }


def _cov_eodhd_incomplete_fetch():
    return script.FetchedBundle(
        prices=pd.DataFrame(columns=dataset_spec("daily_price_raw").required_columns),
        corporate_actions=pd.DataFrame(
            columns=dataset_spec("corporate_actions").required_columns
        ),
        artifacts=_cov_artifacts(),
        missing_symbols=("COV.US",),
    )


def _cov_wiki_generate_py() -> bytes:
    return b'''\
def fix_ticker(ticker):
    return ticker.replace(".", "_")

def quandl_data(wiki, ticker, start, end):
    if ticker in wiki:
        df = wiki[ticker][start:end]
    else:
        ticker = fix_ticker(ticker)
        if ticker in wiki:
            df = wiki[ticker][start:end]
        else:
            return None
    df = df.rename(columns={
        "adj_open": "open", "adj_high": "high", "adj_low": "low",
        "adj_close": "close", "adj_volume": "volume",
    })
    return df

def build(wiki, ticker, start, end):
    df = quandl_data(wiki, ticker, start, end)
    if df is None:
        df = yahoo_data(ticker, start, end)
    return df
'''


def _cov_wiki_csv() -> bytes:
    rows = ["date,open,high,low,close,volume"]
    for offset, session in enumerate(script.COV_EXPECTED_SESSIONS):
        close = 100.0 + offset
        rows.append(
            f"{session},{close},{close + 1},{close - 1},{close},{1000 + offset}"
        )
    return ("\n".join(rows) + "\n").encode()


def _cov_wiki_artifacts():
    contents = {
        "cov_csv": _cov_wiki_csv(),
        "readme": b"Historical prices use Quandl WIKI data.\n",
        "generate_py": _cov_wiki_generate_py(),
    }
    content_types = {
        "cov_csv": "text/csv",
        "readme": "text/plain",
        "generate_py": "text/plain",
    }
    return tuple(
        SourceArtifact(
            source=f"cov_quandl_wiki_{role}",
            source_url=script.COV_WIKI_URLS[role],
            retrieved_at="2026-07-18T00:00:00Z",
            content=contents[role],
            content_type=content_types[role],
        )
        for role in script.COV_WIKI_URLS
    )


def _valid_cov_wiki_fetch():
    artifacts = _cov_wiki_artifacts()
    hashes = {
        role: artifact.source_hash
        for role, artifact in zip(script.COV_WIKI_URLS, artifacts)
    }
    with patch.object(script, "COV_WIKI_SHA256", hashes):
        prices = script._parse_cov_wiki_csv(artifacts[0])
    return script.CovWikiEvidenceBundle(
        prices=prices,
        artifacts=artifacts,
        http_attempts=3,
    )


def _cov_wiki_test_hashes(bundle):
    return {
        role: artifact.source_hash
        for role, artifact in zip(script.COV_WIKI_URLS, bundle.artifacts)
    }


def _valid_primary_fetch():
    rows = []
    for item in script.SUCCESSOR_SPECS:
        dates = {item.event_date}
        if item.first_price_not_after.startswith("2015-01"):
            dates.add("2015-01-02")
        else:
            dates.add(item.history_start)
        if item.require_recent:
            dates.add("2026-07-15")
        rows.extend(_price(item.security_id, value) for value in sorted(dates))
    return SimpleNamespace(
        prices=pd.DataFrame(rows),
        corporate_actions=pd.DataFrame(
            columns=dataset_spec("corporate_actions").required_columns
        ),
        artifacts=_artifacts(script.SUCCESSOR_SPECS),
        missing_symbols=(),
    )


def _valid_enb_fetch(*, enb_close: float = 40.65):
    rows = [
        _price(script.ENB_SPEC.security_id, "2015-01-02", enb_close),
        _price(script.ENB_SPEC.security_id, script.ENB_EFFECTIVE_DATE, enb_close),
        _price(script.ENB_SPEC.security_id, "2026-07-15", enb_close),
    ]
    return script.FetchedBundle(
        prices=pd.DataFrame(rows),
        corporate_actions=pd.DataFrame(
            columns=dataset_spec("corporate_actions").required_columns
        ),
        artifacts=_artifacts((script.ENB_SPEC,)),
        missing_symbols=(),
    )


def _enb_preflight_frames(*, spectra_security_id: str = script.SPECTRA_SECURITY_ID):
    master = _master(
        spectra_security_id,
        "SE",
        "SE1.US",
        "Spectra Energy Corp",
    )
    master["active_to"] = script.SPECTRA_MASTER_ACTIVE_TO
    history = _history(spectra_security_id, "SE")
    history["effective_to"] = script.SPECTRA_SYMBOL_HISTORY_END
    return {
        "security_master": pd.DataFrame([master]),
        "symbol_history": pd.DataFrame([history]),
        "daily_price_raw": pd.DataFrame(
            [
                _price(
                    spectra_security_id,
                    script.SPECTRA_LAST_TRADING_DATE,
                    40.0,
                )
            ]
        ),
        "corporate_actions": pd.DataFrame(
            columns=dataset_spec("corporate_actions").required_columns
        ),
        "source_archive": pd.DataFrame(
            [
                {
                    "archive_id": script.ENB_SEC_COMPLETION_SHA256,
                    "dataset": "official_identity_evidence_raw",
                    "object_path": (
                        "archives/2026-07-15/"
                        f"{script.ENB_SEC_COMPLETION_SHA256}.html.gz"
                    ),
                    "content_type": "text/html",
                    "effective_date": "2026-07-15",
                    "source": "official_identity_evidence_raw",
                    "source_url": script.ENB_SEC_COMPLETION_URL,
                    "retrieved_at": "2026-07-18T00:00:00Z",
                    "source_hash": script.ENB_SEC_COMPLETION_SHA256,
                }
            ]
        ),
    }


def _loaded_report(version: str):
    return script.LoadedEvidenceReport(
        data={"release_version": version, "records": {}},
        artifact=SourceArtifact(
            source="sec_lifecycle_evidence_report",
            source_url="file:///evidence.json",
            retrieved_at="2026-07-18T00:00:00Z",
            content=b"{}",
            content_type="application/json",
        ),
    )


class _TransactionRepository:
    def __init__(
        self,
        root: Path,
        *,
        fail_after_dataset: str = "",
        fail_after_commit: bool = False,
        release_warnings: tuple[str, ...] = (),
    ):
        self.root = root
        self.objects = LocalObjectStore(root)
        self.fail_after_dataset = fail_after_dataset
        self.fail_after_commit = fail_after_commit
        versions = {}
        for dataset in script.WRITE_DATASETS:
            version = f"old-{dataset}"
            versions[dataset] = version
            manifest = DatasetManifest.create(
                dataset,
                version,
                "2026-07-15",
                (),
            )
            manifest_path = f"datasets/{dataset}/versions/{version}/manifest.json"
            self.objects.put(manifest_path, manifest.to_bytes())
            self.objects.put(
                f"datasets/{dataset}/current.json",
                CurrentPointer.create(manifest, manifest_path).to_bytes(),
            )
        self.release = DataRelease(
            version="base-release",
            created_at="2026-07-18T00:00:00Z",
            completed_session="2026-07-15",
            dataset_versions=versions,
            warnings=release_warnings,
        )
        self.objects.put(
            f"releases/{self.release.version}.json",
            self.release.to_bytes(),
        )
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
        if dataset == self.fail_after_dataset:
            raise RuntimeError(f"injected {dataset} failure")
        return SimpleNamespace(
            manifest=manifest,
            conflict=False,
            conflict_path="",
        )

    def commit_release(
        self,
        completed_session,
        dataset_versions,
        *,
        quality,
        warnings,
        expected_etag,
    ):
        release = DataRelease.create(
            completed_session,
            dataset_versions,
            quality=quality,
            warnings=warnings,
        )
        self.objects.put(
            f"releases/{release.version}.json",
            release.to_bytes(),
            if_none_match=True,
        )
        self.objects.put(
            "releases/current.json",
            release.to_bytes(),
            if_match=expected_etag,
        )
        if self.fail_after_commit:
            raise RuntimeError("injected commit failure")
        return release


def _transaction_prepared(
    repository: _TransactionRepository,
    *,
    clear_missing_warning: bool = False,
    clear_external_warning: bool = False,
):
    _release, release_etag = repository.current_release()
    pointer_etags = {
        dataset: repository.current_pointer(dataset)[1]
        for dataset in script.WRITE_DATASETS
    }
    return script.PreparedCollection(
        release=repository.release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        frames={dataset: pd.DataFrame() for dataset in script.WRITE_DATASETS},
        artifacts=(),
        archive_artifacts=(),
        warnings=(),
        summary={
            "status": "validated_dry_run",
            "active_missing_or_stale_after_supplement": (
                0 if clear_missing_warning else None
            ),
            "repairs": {
                "cov_identity_closed_on": (
                    script.COV_LAST_TRADING_DATE
                    if clear_missing_warning
                    else ""
                )
            },
            "cov_cross_validated": clear_external_warning,
        },
        cleared_release_warnings=tuple(
            warning
            for warning, enabled in (
                (script.MISSING_PROVIDER_WARNING, clear_missing_warning),
                (
                    script.COV_PENDING_INDEPENDENT_VALIDATION_WARNING,
                    clear_external_warning,
                ),
            )
            if enabled
        ),
    )


class CatalogSelectionTests(unittest.TestCase):
    def test_selects_only_the_exact_audited_codes(self):
        selected = script.select_catalog_entries(_catalogs(script.SUCCESSOR_SPECS))

        self.assertEqual(
            [item.spec.provider_symbol for item in selected.values()],
            [item.provider_symbol for item in script.SUCCESSOR_SPECS],
        )
        self.assertNotIn("ACT", selected)
        self.assertNotIn("VAL", selected)
        self.assertEqual(
            selected["ECA"].archive.kind,
            "delisted",
        )
        self.assertEqual(
            selected["DINO"].spec.security_id,
            "US:EODHD:636ef90b-6f62-589e-8cbe-368e89552f16",
        )

    def test_rejects_a_reused_code_with_the_wrong_company_name(self):
        specs = (script.SUCCESSOR_SPECS[0],)
        catalogs = _catalogs(specs)
        wrong = dict(catalogs["active"].rows[0])
        wrong["Name"] = "Litian Group Inc"
        catalogs["active"] = script.CatalogArchive(
            **{
                **catalogs["active"].__dict__,
                "rows": (wrong,),
            }
        )

        with self.assertRaisesRegex(ValueError, "Expected one exact"):
            script.select_catalog_entries(catalogs, specs)

    def test_cov_is_exactly_one_delisted_catalog_identity(self):
        selected = script.select_catalog_entries(
            _catalogs((script.COV_SPEC,)),
            (script.COV_SPEC,),
        )["COV"]

        self.assertEqual(selected.archive.kind, "delisted")
        self.assertEqual(selected.row["Name"], "covidien")
        self.assertEqual(selected.row["Isin"], "USG2554F1134")
        self.assertEqual(
            selected.spec.security_id,
            "US:EODHD:e03a169c-f7e7-539c-9dde-a7da5a8e861c",
        )

    def test_enb_is_an_isolated_exact_active_catalog_identity(self):
        selected = script.select_catalog_entries(
            _catalogs((script.ENB_SPEC,)),
            (script.ENB_SPEC,),
        )["ENB"]

        self.assertEqual(selected.archive.kind, "active")
        self.assertEqual(selected.row["Name"], "enbridge")
        self.assertEqual(
            selected.spec.security_id,
            "US:EODHD:8b62832f-27a7-5139-a199-62f9632c21bd",
        )
        self.assertNotIn(script.ENB_SPEC, script.SUCCESSOR_SPECS)


class FetchValidationTests(unittest.TestCase):
    def _valid_fetch(self):
        rows = []
        for item in script.SUCCESSOR_SPECS:
            dates = {item.event_date}
            if item.first_price_not_after.startswith("2015-01"):
                dates.add("2015-01-02")
            else:
                dates.add(item.history_start)
            if item.require_recent:
                dates.add("2026-07-15")
            rows.extend(_price(item.security_id, value) for value in sorted(dates))
        prices = pd.DataFrame(rows)
        actions = pd.DataFrame(
            columns=dataset_spec("corporate_actions").required_columns
        )
        return SimpleNamespace(
            prices=prices,
            corporate_actions=actions,
            artifacts=_artifacts(script.SUCCESSOR_SPECS),
            missing_symbols=(),
        )

    def test_requires_a_price_within_ten_days_after_every_event(self):
        fetched = self._valid_fetch()
        gap_id = next(
            item.security_id for item in script.SUCCESSOR_SPECS if item.symbol == "GAP"
        )
        fetched.prices = fetched.prices.loc[
            ~fetched.prices["security_id"].astype(str).eq(gap_id)
        ]

        with self.assertRaisesRegex(ValueError, "GAP.US has no valid price"):
            script.validate_fetched_result(
                fetched,
                completed_session="2026-07-15",
            )

    def test_missing_provider_symbol_is_a_hard_failure(self):
        fetched = self._valid_fetch()
        fetched.missing_symbols = ("GAP.US",)

        with self.assertRaisesRegex(ValueError, "fetch is incomplete"):
            script.validate_fetched_result(
                fetched,
                completed_session="2026-07-15",
            )

    def test_one_event_row_per_symbol_is_rejected_as_truncated_history(self):
        fetched = SimpleNamespace(
            prices=pd.DataFrame(
                [_price(item.security_id, item.event_date) for item in script.SUCCESSOR_SPECS]
            ),
            corporate_actions=pd.DataFrame(
                columns=dataset_spec("corporate_actions").required_columns
            ),
            artifacts=_artifacts(script.SUCCESSOR_SPECS),
            missing_symbols=(),
        )

        with self.assertRaisesRegex(ValueError, "history is truncated|active history is stale"):
            script.validate_fetched_result(
                fetched,
                completed_session="2026-07-15",
            )

    def test_request_artifacts_are_exactly_thirteen_times_three(self):
        fetched = self._valid_fetch()

        script.validate_fetched_result(
            fetched,
            completed_session="2026-07-15",
        )
        self.assertEqual(len(script.SUCCESSOR_SPECS), 13)
        self.assertEqual(len(fetched.artifacts), 39)

    def test_cov_requires_all_sixteen_sessions_and_exactly_nine_artifacts(self):
        fetched = _valid_cov_fetch()
        direct = _valid_cov_direct_fetch()
        wiki = _valid_cov_wiki_fetch()
        with (
            patch.object(
                script, "COV_DIRECTINDEX_SHA256", _cov_direct_test_hashes(direct)
            ),
            patch.object(script, "COV_WIKI_SHA256", _cov_wiki_test_hashes(wiki)),
        ):
            script.validate_cov_fetched_result(fetched)

        self.assertEqual(len(fetched.prices), 16)
        self.assertEqual(len(fetched.artifacts), 9)

    def test_one_row_cov_response_is_rejected_and_cannot_be_cached(self):
        valid = _valid_cov_fetch()
        fetched = script.FetchedBundle(
            prices=valid.prices.iloc[[-1]].copy(),
            corporate_actions=valid.corporate_actions,
            artifacts=valid.artifacts,
            missing_symbols=(),
        )

        with self.assertRaisesRegex(ValueError, "complete 2015-01-02"):
            script.validate_cov_fetched_result(fetched)

    def test_enb_requires_exactly_three_endpoint_artifacts_and_effective_close(self):
        fetched = _valid_enb_fetch()

        script.validate_fetched_result(
            fetched,
            completed_session="2026-07-15",
            specs=(script.ENB_SPEC,),
        )
        self.assertEqual(len(fetched.artifacts), 3)
        self.assertIn(
            script.ENB_EFFECTIVE_DATE,
            set(fetched.prices["session"].astype(str)),
        )


class CallAndArchiveSafetyTests(unittest.TestCase):
    def test_cov_window_is_exactly_sixteen_xnys_sessions(self):
        self.assertEqual(len(script.COV_EXPECTED_SESSIONS), 16)
        self.assertEqual(script.COV_EXPECTED_SESSIONS, script._cov_xnys_sessions())
        script._validate_spec_invariants()

    def test_directindex_three_pinned_requests_and_second_run_is_offline(self):
        bundle = _valid_cov_direct_fetch()
        by_url = {item.source_url: item for item in bundle.artifacts}

        def response_for(url, *, timeout):
            self.assertEqual(timeout, 120)
            artifact = by_url[url]
            return SimpleNamespace(
                status_code=200,
                content=artifact.content,
                headers={"Content-Type": artifact.content_type},
            )

        with tempfile.TemporaryDirectory() as directory, patch.object(
            script,
            "COV_DIRECTINDEX_SHA256",
            _cov_direct_test_hashes(bundle),
        ):
            session = SimpleNamespace(get=Mock(side_effect=response_for))
            first = script.CappedCovDirectIndexSource(
                Path(directory), allow_http=True, session=session
            )
            fetched = first.fetch()
            self.assertEqual(first.attempt_count, 3)
            self.assertEqual(session.get.call_count, 3)
            self.assertEqual(len(fetched.prices), 16)
            self.assertEqual(len(fetched.artifacts), 3)
            self.assertEqual(len(tuple(Path(directory).glob("*.json.gz"))), 3)

            second = script.CappedCovDirectIndexSource(
                Path(directory), allow_http=False
            )
            cached = second.fetch()
            self.assertEqual(second.attempt_count, 0)
            self.assertEqual(len(cached.prices), 16)

    def test_directindex_network_failure_has_no_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            session = SimpleNamespace(get=Mock(side_effect=OSError("offline")))
            source = script.CappedCovDirectIndexSource(
                Path(directory), allow_http=True, session=session
            )
            with self.assertRaisesRegex(RuntimeError, "single attempt failed"):
                source.fetch()
            with self.assertRaisesRegex(RuntimeError, "already used"):
                source.fetch()
            self.assertEqual(session.get.call_count, 1)
            self.assertEqual(source.attempt_count, 1)

    def test_cov_final_bundle_keeps_all_incomplete_eodhd_raw_responses(self):
        fetched = _valid_cov_fetch()
        direct = _valid_cov_direct_fetch()
        wiki = _valid_cov_wiki_fetch()
        with (
            patch.object(
                script, "COV_DIRECTINDEX_SHA256", _cov_direct_test_hashes(direct)
            ),
            patch.object(script, "COV_WIKI_SHA256", _cov_wiki_test_hashes(wiki)),
        ):
            script.validate_cov_fetched_result(fetched)
        self.assertEqual(len(fetched.artifacts), 9)
        self.assertEqual(tuple(fetched.artifacts[:3]), _cov_artifacts())
        self.assertEqual(set(fetched.prices["source"]), {"directindex_pinned_csv"})

    def test_directindex_rejects_wrong_url_hash_and_html(self):
        valid = _valid_cov_direct_fetch()
        first = valid.artifacts[0]
        wrong_url = SourceArtifact(
            first.source,
            first.source_url.replace(script.COV_DIRECTINDEX_COMMIT, "main"),
            first.retrieved_at,
            first.content,
            first.content_type,
        )
        bad_content = SourceArtifact(
            first.source,
            first.source_url,
            first.retrieved_at,
            b"<html>challenge</html>",
            "text/html",
        )
        with patch.object(
            script, "COV_DIRECTINDEX_SHA256", _cov_direct_test_hashes(valid)
        ):
            with self.assertRaisesRegex(ValueError, "pinned URL"):
                script._parse_cov_directindex_csv(wrong_url)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                script._parse_cov_directindex_csv(bad_content)

    def test_quandl_wiki_cross_validation_is_independent_and_exact(self):
        direct = _valid_cov_direct_fetch()
        wiki = _valid_cov_wiki_fetch()
        with (
            patch.object(
                script, "COV_DIRECTINDEX_SHA256", _cov_direct_test_hashes(direct)
            ),
            patch.object(script, "COV_WIKI_SHA256", _cov_wiki_test_hashes(wiki)),
        ):
            result = script.cross_validate_cov_directindex_with_wiki(
                direct, wiki
            )
        self.assertTrue(result["cov_cross_validated"])
        self.assertEqual(result["sessions_compared"], 16)
        self.assertEqual(result["primary_source"], "directindex_pinned_csv")
        self.assertEqual(result["independent_source"], "quandl_wiki_pinned_csv")

    def test_quandl_wiki_cross_validation_blocks_ohlc_over_tolerance(self):
        direct = _valid_cov_direct_fetch()
        wiki = _valid_cov_wiki_fetch()
        raw = pd.read_csv(__import__("io").BytesIO(wiki.artifacts[0].content))
        raw.loc[0, "close"] += 0.0011
        changed = SourceArtifact(
            wiki.artifacts[0].source,
            wiki.artifacts[0].source_url,
            wiki.artifacts[0].retrieved_at,
            raw.to_csv(index=False).encode(),
            wiki.artifacts[0].content_type,
        )
        wiki = script.CovWikiEvidenceBundle(
            prices=wiki.prices.copy(),
            artifacts=(changed, *wiki.artifacts[1:]),
            http_attempts=3,
        )
        hashes = _cov_wiki_test_hashes(wiki)
        with patch.object(script, "COV_WIKI_SHA256", hashes):
            wiki = script.CovWikiEvidenceBundle(
                prices=script._parse_cov_wiki_csv(changed),
                artifacts=wiki.artifacts,
                http_attempts=3,
            )
        with (
            patch.object(
                script, "COV_DIRECTINDEX_SHA256", _cov_direct_test_hashes(direct)
            ),
            patch.object(script, "COV_WIKI_SHA256", hashes),
        ):
            with self.assertRaisesRegex(ValueError, "close cross-validation"):
                script.cross_validate_cov_directindex_with_wiki(direct, wiki)

    def test_quandl_wiki_cross_validation_requires_exact_volume(self):
        direct = _valid_cov_direct_fetch()
        wiki = _valid_cov_wiki_fetch()
        raw = pd.read_csv(__import__("io").BytesIO(wiki.artifacts[0].content))
        raw.loc[0, "volume"] += 1
        changed = SourceArtifact(
            wiki.artifacts[0].source,
            wiki.artifacts[0].source_url,
            wiki.artifacts[0].retrieved_at,
            raw.to_csv(index=False).encode(),
            wiki.artifacts[0].content_type,
        )
        artifacts = (changed, *wiki.artifacts[1:])
        hashes = {
            role: artifact.source_hash
            for role, artifact in zip(script.COV_WIKI_URLS, artifacts)
        }
        with patch.object(script, "COV_WIKI_SHA256", hashes):
            wiki = script.CovWikiEvidenceBundle(
                prices=script._parse_cov_wiki_csv(changed),
                artifacts=artifacts,
                http_attempts=3,
            )
        with (
            patch.object(
                script, "COV_DIRECTINDEX_SHA256", _cov_direct_test_hashes(direct)
            ),
            patch.object(script, "COV_WIKI_SHA256", hashes),
        ):
            with self.assertRaisesRegex(ValueError, "volume cross-validation"):
                script.cross_validate_cov_directindex_with_wiki(direct, wiki)

    def test_quandl_wiki_three_pinned_requests_then_offline_cache(self):
        bundle = _valid_cov_wiki_fetch()
        by_url = {item.source_url: item for item in bundle.artifacts}

        def response_for(url, *, timeout):
            self.assertEqual(timeout, 120)
            artifact = by_url[url]
            return SimpleNamespace(
                status_code=200,
                content=artifact.content,
                headers={"Content-Type": artifact.content_type},
            )

        with tempfile.TemporaryDirectory() as directory, patch.object(
            script,
            "COV_WIKI_SHA256",
            _cov_wiki_test_hashes(bundle),
        ):
            session = SimpleNamespace(get=Mock(side_effect=response_for))
            first = script.CappedCovQuandlWikiSource(
                Path(directory), allow_http=True, session=session
            )
            fetched = first.fetch()
            self.assertEqual(first.attempt_count, 3)
            self.assertEqual(session.get.call_count, 3)
            self.assertEqual(len(fetched.artifacts), 3)

            second = script.CappedCovQuandlWikiSource(
                Path(directory), allow_http=False
            )
            cached = second.fetch()
            self.assertEqual(second.attempt_count, 0)
            self.assertEqual(len(cached.prices), 16)

    def test_quandl_wiki_hash_mismatch_is_cached_and_never_retried(self):
        bad = SimpleNamespace(
            status_code=200,
            content=b"wrong pinned bytes",
            headers={"Content-Type": "text/csv"},
        )
        with tempfile.TemporaryDirectory() as directory:
            session = SimpleNamespace(get=Mock(return_value=bad))
            first = script.CappedCovQuandlWikiSource(
                Path(directory), allow_http=True, session=session
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                first.fetch()
            self.assertEqual(session.get.call_count, 1)
            self.assertTrue((Path(directory) / "cov_csv.json.gz").is_file())

            second = script.CappedCovQuandlWikiSource(
                Path(directory), allow_http=False
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                second.fetch()
            self.assertEqual(second.attempt_count, 0)

    def test_single_attempt_client_never_retries_and_enforces_cap(self):
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: [],
        )
        session = SimpleNamespace(get=Mock(return_value=response))
        budget = SimpleNamespace(claim=Mock())
        client = script.CappedSingleAttemptEodhdClient(
            session=session,
            token="secret",
            budget=budget,
            max_attempts=1,
        )

        self.assertEqual(client.get_json("eod/ABC.US"), [])
        with self.assertRaisesRegex(RuntimeError, "call cap reached"):
            client.get_json("div/ABC.US")

        self.assertEqual(session.get.call_count, 1)
        self.assertEqual(budget.claim.call_count, 1)
        self.assertEqual(client.attempt_count, 1)

    def test_enb_official_completion_cache_is_exact_url_hash_and_text_bound(self):
        content = b"""<html><body>
        Enbridge and Spectra Energy Complete Merger February 27, 2017.
        This stock-for-stock merger transaction completed today. Trading in
        shares of Spectra Energy common stock will be suspended effective as
        of the opening of trading today and will be delisted from the NYSE.
        Enbridge continues under the symbol ENB.
        </body></html>"""
        source_hash = script.sha256_bytes(content)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            script,
            "ENB_SEC_COMPLETION_SHA256",
            source_hash,
        ):
            root = Path(directory)
            key = script.sha256_bytes(script.ENB_SEC_COMPLETION_URL.encode())
            path = root / "state/official-us-index-identity" / f"{key}.json.gz"
            path.parent.mkdir(parents=True)
            payload = {
                "schema": script.ENB_OFFICIAL_EVIDENCE_SCHEMA,
                "source_url": script.ENB_SEC_COMPLETION_URL,
                "source_hash": source_hash,
                "retrieved_at": "2026-07-18T00:00:00Z",
                "content_type": "text/html",
                "content_base64": script.base64.b64encode(content).decode("ascii"),
            }
            path.write_bytes(
                script.gzip.compress(
                    json.dumps(payload, sort_keys=True).encode(),
                    mtime=0,
                )
            )

            artifact = script._load_enb_official_evidence(root)

            self.assertEqual(artifact.source_url, script.ENB_SEC_COMPLETION_URL)
            self.assertEqual(artifact.source_hash, source_hash)
            self.assertEqual(artifact.content, content)

            payload["source_url"] = "https://www.sec.gov/wrong"
            path.write_bytes(
                script.gzip.compress(
                    json.dumps(payload, sort_keys=True).encode(),
                    mtime=0,
                )
            )
            with self.assertRaisesRegex(ValueError, "URL is not exact"):
                script._load_enb_official_evidence(root)

    def test_identical_empty_responses_keep_distinct_request_provenance(self):
        first = SourceArtifact(
            source="eodhd_div",
            source_url="https://eodhd.test/div/A.US",
            retrieved_at="2026-07-18T00:00:00Z",
            content=b"[]",
            content_type="application/json",
        )
        second = SourceArtifact(
            source="eodhd_div",
            source_url="https://eodhd.test/div/B.US",
            retrieved_at="2026-07-18T00:00:00Z",
            content=b"[]",
            content_type="application/json",
        )
        archived = tuple(map(script._request_archive_artifact, (first, second)))
        rows = script._artifact_rows(archived, "2026-07-15")

        self.assertNotEqual(archived[0].source_hash, archived[1].source_hash)
        self.assertEqual(len(rows), 2)
        for artifact in archived:
            envelope = __import__("json").loads(artifact.content)
            self.assertEqual(envelope["content_sha256"], first.source_hash)

    def test_existing_thirty_nine_call_cache_signature_is_unchanged(self):
        repository = SimpleNamespace(root=Path("data/cache"))
        release = DataRelease(
            version="20260715-20260717T155657288079Z",
            created_at="2026-07-17T15:56:57.288094Z",
            completed_session="2026-07-15",
            dataset_versions={},
        )

        primary = script._bundle_cache_path(repository, release)
        cov = script._cov_bundle_cache_path(repository, release)
        enb = script._enb_bundle_cache_path(repository, release)

        self.assertEqual(
            primary.name,
            "e4fcf8fee787073f9b583ff693f5f955d64985bfb83b904e0158690efcc34bc2.json.gz",
        )
        self.assertNotEqual(primary, cov)
        self.assertIn("eodhd_lifecycle_successors_cov", str(cov))
        self.assertNotEqual(primary, enb)
        self.assertNotEqual(cov, enb)
        self.assertIn("eodhd_lifecycle_successors_enb", str(enb))
        self.assertEqual(script.ENB_MAX_EODHD_HTTP_ATTEMPTS, 3)


class LocalPreflightTests(unittest.TestCase):
    def _frames(self, *, dow_date: str = "2017-08-31", include_dow_history: bool = True):
        master = pd.DataFrame(
            [
                _master("DOW-ID", "DOW", "DOW.US", "Dow Inc"),
                _master("ACT-ID", "AGN", "AGN.US", "Allergan plc"),
                _master("DWDP-ID", "DWDP", "DWDP.US", "DowDuPont Inc"),
            ]
        )
        history_rows = [
            _history("ACT-ID", "AGN"),
            _history("DWDP-ID", "DWDP"),
        ]
        if include_dow_history:
            history_rows.append(_history("DOW-ID", "DOW"))
        return {
            "security_master": master,
            "symbol_history": pd.DataFrame(history_rows),
            "daily_price_raw": pd.DataFrame(
                [
                    _price("DOW-ID", dow_date),
                    _price("ACT-ID", "2015-03-17"),
                    _price("DWDP-ID", "2017-09-01"),
                ]
            ),
        }

    def test_dow_august_31_is_a_hard_pre_provider_requirement(self):
        with (
            patch.object(
                script,
                "PURGE_SPECS",
                (script.PurgeSpec("DOW", "2017-09-01"),),
            ),
            patch.object(script, "_preflight_identity_ids", return_value={}),
        ):
            with self.assertRaisesRegex(ValueError, "2017-08-31"):
                script.validate_local_preflight(
                    self._frames(dow_date="2017-08-30"),
                    {"DOW": "DOW-ID"},
                )

    def test_every_purge_requires_existing_symbol_history(self):
        with (
            patch.object(
                script,
                "PURGE_SPECS",
                (script.PurgeSpec("DOW", "2017-09-01"),),
            ),
            patch.object(script, "_preflight_identity_ids", return_value={}),
        ):
            with self.assertRaisesRegex(ValueError, "no symbol history"):
                script.validate_local_preflight(
                    self._frames(include_dow_history=False),
                    {"DOW": "DOW-ID"},
                )

    def _cov_frames(self):
        cov_id = script.COV_SPEC.security_id
        mdt_id = "MDT-ID"
        return {
            "security_master": pd.DataFrame(
                [
                    _master(cov_id, "COV", "COV.US", "Covidien plc"),
                    _master(mdt_id, "MDT", "MDT.US", "Medtronic PLC"),
                ]
            ),
            "symbol_history": pd.DataFrame(
                [
                    _history(cov_id, "COV"),
                    _history(mdt_id, "MDT"),
                ]
            ),
            "daily_price_raw": pd.DataFrame(
                [
                    _price(mdt_id, "2015-01-26", 75.59),
                    _price(mdt_id, "2015-01-27", 75.26),
                    _price(mdt_id, "2026-07-15", 90.0),
                ]
            ),
            "corporate_actions": pd.DataFrame(
                columns=dataset_spec("corporate_actions").required_columns
            ),
        }

    def test_cov_preflight_proves_it_is_the_only_active_price_gap(self):
        selection = script.select_catalog_entries(
            _catalogs((script.COV_SPEC,)),
            (script.COV_SPEC,),
        )["COV"]

        cov_id, mdt_id = script.validate_cov_local_preflight(
            self._cov_frames(),
            selection,
            completed_session="2026-07-15",
            release_warnings=(script.MISSING_PROVIDER_WARNING,),
        )

        self.assertEqual(cov_id, script.COV_SPEC.security_id)
        self.assertEqual(mdt_id, "MDT-ID")

    def test_cov_preflight_rejects_any_second_active_stale_identity(self):
        frames = self._cov_frames()
        frames["security_master"] = pd.concat(
            [
                frames["security_master"],
                pd.DataFrame([_master("STALE-ID", "BAD", "BAD.US", "Bad Inc")]),
            ],
            ignore_index=True,
        )
        frames["symbol_history"] = pd.concat(
            [
                frames["symbol_history"],
                pd.DataFrame([_history("STALE-ID", "BAD")]),
            ],
            ignore_index=True,
        )
        selection = script.select_catalog_entries(
            _catalogs((script.COV_SPEC,)),
            (script.COV_SPEC,),
        )["COV"]

        with self.assertRaisesRegex(ValueError, "not exactly COV"):
            script.validate_cov_local_preflight(
                frames,
                selection,
                completed_session="2026-07-15",
                release_warnings=(script.MISSING_PROVIDER_WARNING,),
            )

    def test_enb_preflight_requires_the_exact_repaired_spectra_identity(self):
        selection = script.select_catalog_entries(
            _catalogs((script.ENB_SPEC,)),
            (script.ENB_SPEC,),
        )["ENB"]

        script.validate_enb_local_preflight(
            _enb_preflight_frames(),
            selection,
            completed_session="2026-07-15",
            release_warnings=(),
        )

        with self.assertRaisesRegex(ValueError, "exact repaired Spectra"):
            script.validate_enb_local_preflight(
                _enb_preflight_frames(spectra_security_id="REUSED-SEA-ID"),
                selection,
                completed_session="2026-07-15",
                release_warnings=(),
            )

    def test_enb_preflight_refuses_to_run_before_identity_repair(self):
        selection = script.select_catalog_entries(
            _catalogs((script.ENB_SPEC,)),
            (script.ENB_SPEC,),
        )["ENB"]

        with self.assertRaisesRegex(ValueError, "after the audited US identity repair"):
            script.validate_enb_local_preflight(
                _enb_preflight_frames(),
                selection,
                completed_session="2026-07-15",
                release_warnings=(script.IDENTITY_REPAIR_MIGRATION_WARNING,),
            )


class RewriteTests(unittest.TestCase):
    def test_enb_supplement_adds_one_identity_and_crosschecks_point_984_ratio(self):
        frames = _enb_preflight_frames()
        selection = script.select_catalog_entries(
            _catalogs((script.ENB_SPEC,)),
            (script.ENB_SPEC,),
        )["ENB"]

        output, stats = script.apply_enb_supplement(
            frames,
            selection,
            _valid_enb_fetch(),
            completed_session="2026-07-15",
        )

        self.assertEqual(
            set(
                output["security_master"].loc[
                    output["security_master"]["primary_symbol"].astype(str).eq("ENB"),
                    "security_id",
                ]
            ),
            {script.ENB_SPEC.security_id},
        )
        self.assertEqual(
            output["symbol_history"].loc[
                output["symbol_history"]["security_id"]
                .astype(str)
                .eq(script.ENB_SPEC.security_id),
                "effective_from",
            ].iloc[0],
            script.FETCH_START,
        )
        self.assertTrue(stats["economic_crosscheck"]["passed"])
        self.assertEqual(stats["economic_crosscheck"]["ratio"], 0.984)

        bad = _valid_enb_fetch(enb_close=10.0)
        with self.assertRaisesRegex(ValueError, "0.984-share economic crosscheck failed"):
            script.apply_enb_supplement(
                frames,
                selection,
                bad,
                completed_session="2026-07-15",
            )

    def test_pending_identity_migration_keeps_symbol_open_until_membership_rewrite(self):
        security_id = next(iter(script.IDENTITY_REPAIR_MIGRATION_GAP_IDS))
        master = pd.DataFrame(
            [_master(security_id, "AGN", "AGN_old.US", "Allergan Inc")]
        )
        history = pd.DataFrame([_history(security_id, "AGN")])
        prices = pd.DataFrame([_price(security_id, "2015-03-16")])
        specs = (
            script.PurgeSpec(
                "AGN",
                "2015-03-17",
                ("allergan inc",),
                "2015-03-22",
            ),
        )

        closed_master, transitional_history, _last = script._close_old_identities(
            master,
            history,
            prices,
            {"AGN": security_id},
            specs,
            preserve_history_ids={security_id},
        )

        self.assertEqual(closed_master.iloc[0]["active_to"], "2015-03-16")
        self.assertEqual(transitional_history.iloc[0]["effective_to"], "")

    def test_new_reused_ticker_cutoffs_remove_contaminated_prices_and_actions(self):
        values = {
            "GPS": ("2024-08-21", "2024-08-22"),
            "PKI": ("2023-05-15", "2023-05-16"),
            "CTL": ("2020-09-17", "2020-09-18"),
            "BK": ("2026-05-20", "2026-05-21"),
            "COR": ("2021-12-27", "2021-12-28"),
            "QRTEA": ("2025-02-21", "2025-02-24"),
        }
        specs = tuple(
            item for item in script.PURGE_SPECS if item.symbol in values
        )
        purge_ids = {symbol: f"OLD-{symbol}" for symbol in values}
        prices = pd.DataFrame(
            [
                _price(purge_ids[symbol], session)
                for symbol, dates in values.items()
                for session in dates
            ]
        )
        actions = pd.DataFrame(
            [
                _action(purge_ids[symbol], session)
                for symbol, dates in values.items()
                for session in dates
            ]
        )

        trimmed_prices, trimmed_actions, _stats = script._trim_old_candidates(
            prices,
            actions,
            purge_ids,
            specs,
        )

        for symbol, (retained, removed) in values.items():
            security_id = purge_ids[symbol]
            self.assertEqual(
                set(
                    trimmed_prices.loc[
                        trimmed_prices.security_id.eq(security_id), "session"
                    ].astype(str)
                ),
                {retained},
            )
            self.assertFalse(
                (
                    trimmed_actions.security_id.eq(security_id)
                    & trimmed_actions.effective_date.eq(removed)
                ).any()
            )

    def test_repairs_cutoffs_actavis_and_dwdp_without_truncating_successor_warmup(self):
        vip_id = "OLD-VIP"
        dow_id = "OLD-DOW"
        old_agn_id = "OLD-AGN"
        actavis_id = "ACTAVIS-AGN"
        dwdp_id = "DWDP-ID"
        rvty_id = "RVTY-ID"
        lumn_id = "LUMN-ID"
        bny_id = "BNY-ID"
        bny_old_id = "BNY-OLD-ID"
        successor = next(
            item for item in script.SUCCESSOR_SPECS if item.symbol == "VEON"
        )
        purge_specs = (
            script.PurgeSpec("VIP", "2017-03-30"),
            script.PurgeSpec("DOW", "2017-09-01"),
            script.PurgeSpec(
                "AGN",
                "2015-03-17",
                ("allergan inc",),
                "2015-03-22",
            ),
        )
        purge_ids = {"VIP": vip_id, "DOW": dow_id, "AGN": old_agn_id}
        master = pd.DataFrame(
            [
                _master(vip_id, "VIP", "VIP.US", "VimpelCom Ltd"),
                _master(dow_id, "DOW", "DOW.US", "Dow Inc"),
                _master(old_agn_id, "AGN", "AGN_old.US", "Allergan Inc"),
                _master(actavis_id, "AGN", "AGN.US", "Allergan plc"),
                _master(dwdp_id, "DWDP", "DWDP.US", "DowDuPont Inc"),
                _master(rvty_id, "RVTY", "RVTY.US", "Revvity Inc"),
                _master(lumn_id, "LUMN", "LUMN.US", "Lumen Technologies Inc"),
                _master(
                    bny_id,
                    "BNY",
                    "BNY.US",
                    "The Bank of New York Mellon Corporation",
                ),
                _master(
                    bny_old_id,
                    "BNY",
                    "BNY_old.US",
                    "BlackRock New York Municipal Income Trust",
                ),
            ]
        )
        history = pd.DataFrame(
            [
                _history(vip_id, "VIP"),
                _history(dow_id, "DOW"),
                _history(old_agn_id, "AGN"),
                _history(actavis_id, "AGN"),
                _history(dwdp_id, "DWDP"),
                _history(rvty_id, "RVTY"),
                _history(lumn_id, "LUMN"),
                _history(bny_id, "BNY"),
                _history(bny_old_id, "BNY"),
            ]
        )
        prices = pd.DataFrame(
            [
                _price(vip_id, "2017-03-29"),
                _price(vip_id, "2017-03-30"),
                _price(dow_id, "2017-08-31", 66.65),
                _price(dow_id, "2017-09-01", 66.70),
                _price(old_agn_id, "2015-03-16"),
                _price(old_agn_id, "2015-03-17"),
                _price(actavis_id, "2015-03-17"),
                _price(actavis_id, "2015-06-15"),
                _price(dwdp_id, "2017-08-31", 94.8299),
                _price(dwdp_id, "2017-09-01", 67.18),
                _price(dwdp_id, "2017-09-05", 68.00),
            ]
        )
        actions = pd.DataFrame(
            [
                _action(vip_id, "2017-03-29"),
                _action(vip_id, "2017-03-30"),
                _action(dow_id, "2017-08-31"),
                _action(dow_id, "2017-09-01"),
                _action(old_agn_id, "2015-03-17"),
                _action(dwdp_id, "2017-08-31"),
                _action(dwdp_id, "2017-09-01"),
            ]
        )
        fetched = SimpleNamespace(
            prices=pd.DataFrame(
                [
                    _price(successor.security_id, "2015-01-02", 4.0),
                    _price(successor.security_id, "2017-03-30", 5.0),
                ]
            ),
            corporate_actions=pd.DataFrame(
                [_action(successor.security_id, "2016-01-04")]
            ),
        )
        selection = script.select_catalog_entries(
            _catalogs((successor,)), (successor,)
        )

        rewritten, stats = script.rewrite_market_frames(
            {
                "security_master": master,
                "symbol_history": history,
                "daily_price_raw": prices,
                "corporate_actions": actions,
            },
            selection,
            purge_ids,
            fetched,
            completed_session="2026-07-15",
            stamp="2026-07-18T00:00:00Z",
            successor_specs=(successor,),
            purge_specs=purge_specs,
        )

        output_prices = rewritten["daily_price_raw"]
        vip_sessions = set(
            output_prices.loc[output_prices.security_id.eq(vip_id), "session"].astype(str)
        )
        self.assertEqual(vip_sessions, {"2017-03-29"})
        dow_rows = output_prices.loc[output_prices.security_id.eq(dow_id)]
        self.assertEqual(set(dow_rows.session.astype(str)), {"2017-08-31"})
        self.assertAlmostEqual(float(dow_rows.iloc[0].close), 66.65)
        self.assertEqual(
            set(output_prices.loc[output_prices.security_id.eq(dwdp_id), "session"].astype(str)),
            {"2017-09-01", "2017-09-05"},
        )
        successor_sessions = set(
            output_prices.loc[
                output_prices.security_id.eq(successor.security_id), "session"
            ].astype(str)
        )
        self.assertIn("2015-01-02", successor_sessions)

        output_master = rewritten["security_master"]
        self.assertEqual(
            output_master.loc[output_master.security_id.eq(dwdp_id), "active_from"].iloc[0],
            "2017-09-01",
        )
        self.assertEqual(
            output_master.loc[
                output_master.security_id.eq(successor.security_id), "active_from"
            ].iloc[0],
            "2015-01-02",
        )
        output_history = rewritten["symbol_history"]
        successor_history = output_history.loc[
            output_history.security_id.eq(successor.security_id)
        ].iloc[0]
        self.assertEqual(successor_history.effective_from, "2017-03-30")
        self.assertEqual(
            output_history.loc[
                output_history.security_id.eq(dwdp_id), "effective_from"
            ].iloc[0],
            "2017-09-01",
        )
        actavis = output_history.loc[
            output_history.security_id.eq(actavis_id),
            ["symbol", "effective_from", "effective_to"],
        ].sort_values("effective_from")
        self.assertEqual(
            actavis.to_dict("records"),
            [
                {
                    "symbol": "ACT",
                    "effective_from": "2015-01-01",
                    "effective_to": "2015-06-14",
                },
                {
                    "symbol": "AGN",
                    "effective_from": "2015-06-15",
                    "effective_to": "2020-05-08",
                },
            ],
        )
        self.assertEqual(
            output_history.loc[
                output_history.security_id.eq(old_agn_id), "effective_to"
            ].iloc[0],
            "2015-03-22",
        )
        expected_starts = {
            rvty_id: "2023-05-16",
            lumn_id: "2020-09-18",
            bny_id: "2026-05-21",
        }
        for security_id, expected in expected_starts.items():
            value = output_history.loc[
                output_history.security_id.eq(security_id), "effective_from"
            ].iloc[0]
            self.assertEqual(value, expected)
        self.assertEqual(
            output_history.loc[
                output_history.security_id.eq(bny_old_id), "effective_to"
            ].iloc[0],
            "2026-02-09",
        )
        self.assertEqual(
            output_master.loc[
                output_master.security_id.eq(bny_old_id), "active_to"
            ].iloc[0],
            "2026-02-09",
        )
        output_actions = rewritten["corporate_actions"]
        self.assertTrue(
            (
                output_actions.security_id.eq(successor.security_id)
                & output_actions.effective_date.eq("2016-01-04")
            ).any()
        )
        self.assertEqual(stats["dwdp_price_rows_removed"], 1)

    def test_cov_supplement_fills_existing_identity_and_closes_on_last_trade(self):
        cov_id = script.COV_SPEC.security_id
        mdt_id = "MDT-ID"
        frames = {
            "security_master": pd.DataFrame(
                [
                    _master(cov_id, "COV", "COV.US", "Covidien plc"),
                    _master(mdt_id, "MDT", "MDT.US", "Medtronic PLC"),
                ]
            ),
            "symbol_history": pd.DataFrame(
                [
                    _history(cov_id, "COV"),
                    _history(mdt_id, "MDT"),
                ]
            ),
            "daily_price_raw": pd.DataFrame(
                [
                    _price(mdt_id, "2015-01-26", 75.59),
                    _price(mdt_id, "2015-01-27", 75.26),
                    _price(mdt_id, "2026-07-15", 90.0),
                ]
            ),
            "corporate_actions": pd.DataFrame(
                columns=dataset_spec("corporate_actions").required_columns
            ),
        }
        valid = _valid_cov_fetch()
        fetched = script.FetchedBundle(
            prices=valid.prices,
            corporate_actions=pd.DataFrame([_action(cov_id, "2015-01-16")]),
            artifacts=valid.artifacts,
            missing_symbols=(),
        )
        direct = _valid_cov_direct_fetch()
        wiki = _valid_cov_wiki_fetch()
        with (
            patch.object(
                script, "COV_DIRECTINDEX_SHA256", _cov_direct_test_hashes(direct)
            ),
            patch.object(script, "COV_WIKI_SHA256", _cov_wiki_test_hashes(wiki)),
        ):
            output, stats = script.apply_cov_supplement(
                frames,
                fetched,
                cov_security_id=cov_id,
                stamp="2026-07-18T00:00:00Z",
            )

        own_prices = output["daily_price_raw"].loc[
            output["daily_price_raw"].security_id.eq(cov_id)
        ]
        self.assertEqual(
            tuple(sorted(own_prices.session.astype(str))),
            script.COV_EXPECTED_SESSIONS,
        )
        self.assertEqual(
            output["security_master"].loc[
                output["security_master"].security_id.eq(cov_id), "active_to"
            ].iloc[0],
            script.COV_LAST_TRADING_DATE,
        )
        self.assertEqual(
            output["symbol_history"].loc[
                output["symbol_history"].security_id.eq(cov_id), "effective_to"
            ].iloc[0],
            script.COV_LAST_TRADING_DATE,
        )
        own_actions = output["corporate_actions"].loc[
            output["corporate_actions"].security_id.eq(cov_id)
        ]
        self.assertEqual(set(own_actions.action_type), {"cash_dividend"})
        self.assertNotIn("stock_merger", set(own_actions.action_type))
        self.assertEqual(stats["cov_price_rows"], 16)
        self.assertEqual(
            script.active_price_gaps(
                output["security_master"],
                output["daily_price_raw"],
                completed_session="2026-07-15",
            ),
            (),
        )

        lifecycle_frames = {
            "security_master": output["security_master"],
            "daily_price_raw": output["daily_price_raw"],
            "corporate_actions": output["corporate_actions"],
            "index_constituent_anchors": pd.DataFrame(
                [{"security_id": cov_id}]
            ),
            "index_membership_events": pd.DataFrame(
                [
                    {
                        "security_id": cov_id,
                        "operation": "REMOVE",
                        "effective_date": "2015-01-27",
                    }
                ]
            ),
        }
        repository = SimpleNamespace(
            read_frame=lambda dataset, _version=None: lifecycle_frames[dataset]
        )
        release = DataRelease(
            version="candidate-release",
            created_at="2026-07-18T00:00:00Z",
            completed_session="2026-07-15",
            dataset_versions={name: name for name in lifecycle_frames},
        )
        candidates = build_lifecycle_candidates(repository, release=release)
        cov_candidate = next(item for item in candidates if item.symbol == "COV")
        self.assertEqual(cov_candidate.last_price_date, "2015-01-26")
        self.assertEqual(cov_candidate.active_to, "2015-01-26")


class MigrationValidationTests(unittest.TestCase):
    @staticmethod
    def _report():
        return SimpleNamespace(issues=(), raise_for_errors=lambda: None)

    def test_successor_transaction_uses_only_the_frozen_identity_gap_allowlist(self):
        frames = {dataset: pd.DataFrame() for dataset in script.WRITE_DATASETS}
        with patch.object(
            script, "validate_dataset", return_value=self._report()
        ), patch.object(
            script,
            "validate_repository_snapshot",
            return_value=self._report(),
        ) as validate_snapshot:
            warnings = script.validate_candidate_frames(
                frames, completed_session="2026-07-15"
            )

        self.assertEqual(len(script.IDENTITY_REPAIR_MIGRATION_GAP_IDS), 11)
        self.assertIn(script.IDENTITY_REPAIR_MIGRATION_WARNING, warnings)
        self.assertEqual(
            validate_snapshot.call_args.kwargs["allowed_index_price_gap_ids"],
            script.IDENTITY_REPAIR_MIGRATION_GAP_IDS,
        )


class AtomicApplyTests(unittest.TestCase):
    @staticmethod
    def _report():
        return SimpleNamespace(issues=(), raise_for_errors=lambda: None)

    @staticmethod
    def _snapshot(repository):
        return {
            "release": repository.objects.get("releases/current.json").data,
            **{
                dataset: repository.objects.get(
                    repository.current_key(dataset)
                ).data
                for dataset in script.WRITE_DATASETS
            },
        }

    def _assert_snapshot(self, repository, snapshot):
        self.assertEqual(
            repository.objects.get("releases/current.json").data,
            snapshot["release"],
        )
        for dataset in script.WRITE_DATASETS:
            self.assertEqual(
                repository.objects.get(repository.current_key(dataset)).data,
                snapshot[dataset],
            )

    def test_every_dataset_write_failure_rolls_back_all_current_pointers(self):
        for failed_dataset in script.WRITE_DATASETS:
            with self.subTest(dataset=failed_dataset), tempfile.TemporaryDirectory() as directory:
                repository = _TransactionRepository(
                    Path(directory),
                    fail_after_dataset=failed_dataset,
                )
                prepared = _transaction_prepared(repository)
                before = self._snapshot(repository)
                with patch.object(
                    script,
                    "validate_repository_snapshot",
                    return_value=self._report(),
                ):
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        script.apply_collection(repository, prepared)

                self._assert_snapshot(repository, before)
                self.assertFalse(
                    (repository.root / "recovery/lifecycle-successors").exists()
                )

    def test_failure_after_release_commit_restores_release_and_all_pointers(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = _TransactionRepository(
                Path(directory),
                fail_after_commit=True,
            )
            prepared = _transaction_prepared(repository)
            before = self._snapshot(repository)
            with patch.object(
                script,
                "validate_repository_snapshot",
                return_value=self._report(),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected commit"):
                    script.apply_collection(repository, prepared)

            self._assert_snapshot(repository, before)

    def test_archive_payload_failure_happens_before_pointer_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = _TransactionRepository(Path(directory))
            prepared = _transaction_prepared(repository)
            before = self._snapshot(repository)
            with patch.object(
                script,
                "_persist_archive_payloads",
                side_effect=RuntimeError("injected archive failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "archive failure"):
                    script.apply_collection(repository, prepared)

            self._assert_snapshot(repository, before)

    def test_rollback_failure_writes_recovery_marker_and_blocks_future_apply(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = _TransactionRepository(
                Path(directory),
                fail_after_dataset="security_master",
            )
            prepared = _transaction_prepared(repository)
            old_master = repository.objects.get(
                repository.current_key("security_master")
            ).data
            original_put = repository.objects.put

            def fail_restore(key, data, **kwargs):
                if (
                    key == repository.current_key("security_master")
                    and data == old_master
                    and kwargs.get("if_match") is not None
                ):
                    raise RuntimeError("injected rollback CAS failure")
                return original_put(key, data, **kwargs)

            repository.objects.put = fail_restore
            with self.assertRaisesRegex(RuntimeError, "recovery marker"):
                script.apply_collection(repository, prepared)

            markers = tuple(
                (repository.root / "recovery/lifecycle-successors").glob("*.json")
            )
            self.assertEqual(len(markers), 1)
            with self.assertRaisesRegex(RuntimeError, "recovery marker blocks"):
                script.apply_collection(repository, prepared)

    def test_success_release_names_every_new_dataset_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = _TransactionRepository(Path(directory))
            prepared = _transaction_prepared(repository)
            with patch.object(
                script,
                "validate_repository_snapshot",
                return_value=self._report(),
            ):
                result = script.apply_collection(repository, prepared)

            release, _etag = repository.current_release()
            self.assertEqual(result["status"], "applied")
            for dataset in script.WRITE_DATASETS:
                pointer, _pointer_etag = repository.current_pointer(dataset)
                self.assertEqual(pointer.version, release.dataset_versions[dataset])

    def test_complete_cov_supplement_clears_only_the_stale_provider_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = _TransactionRepository(
                Path(directory),
                release_warnings=(
                    script.MISSING_PROVIDER_WARNING,
                    "preserve this unrelated warning",
                ),
            )
            prepared = _transaction_prepared(
                repository,
                clear_missing_warning=True,
            )
            with patch.object(
                script,
                "validate_repository_snapshot",
                return_value=self._report(),
            ):
                script.apply_collection(repository, prepared)

            release, _etag = repository.current_release()
            self.assertEqual(
                release.warnings,
                ("preserve this unrelated warning",),
            )

    def test_provider_warning_is_preserved_without_cov_completion_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = _TransactionRepository(
                Path(directory),
                release_warnings=(script.MISSING_PROVIDER_WARNING,),
            )
            prepared = _transaction_prepared(repository)
            with patch.object(
                script,
                "validate_repository_snapshot",
                return_value=self._report(),
            ):
                script.apply_collection(repository, prepared)

            release, _etag = repository.current_release()
            self.assertEqual(
                release.warnings,
                (script.MISSING_PROVIDER_WARNING,),
            )

    def test_external_validation_warning_clears_only_with_passed_crosscheck(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = _TransactionRepository(
                Path(directory),
                release_warnings=(
                    script.COV_PENDING_INDEPENDENT_VALIDATION_WARNING,
                ),
            )
            prepared = _transaction_prepared(
                repository,
                clear_external_warning=True,
            )
            with patch.object(
                script,
                "validate_repository_snapshot",
                return_value=self._report(),
            ):
                script.apply_collection(repository, prepared)

            release, _etag = repository.current_release()
            self.assertEqual(release.warnings, ())

    def test_forged_warning_clear_rolls_back_every_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = _TransactionRepository(
                Path(directory),
                release_warnings=(script.MISSING_PROVIDER_WARNING,),
            )
            prepared = _transaction_prepared(
                repository,
                clear_missing_warning=True,
            )
            prepared.summary["active_missing_or_stale_after_supplement"] = 1
            before = self._snapshot(repository)
            with patch.object(
                script,
                "validate_repository_snapshot",
                return_value=self._report(),
            ):
                with self.assertRaisesRegex(RuntimeError, "only be cleared"):
                    script.apply_collection(repository, prepared)

            self._assert_snapshot(repository, before)


class ModeGateTests(unittest.TestCase):
    def _args(self, *, offline: bool, apply: bool):
        return SimpleNamespace(
            cache_root="unused",
            evidence_report="unused.json",
            workers=2,
            offline_plan=offline,
            apply=apply,
            fetch_cov_directindex=False,
            fetch_cov_wiki=False,
            cov_eodhd_full_us_failure_response="",
            enb_only=False,
            fetch_enb=False,
        )

    def test_enb_offline_plan_never_constructs_any_provider_source(self):
        repository = SimpleNamespace(
            current_release=lambda: (SimpleNamespace(version="r1"), "etag")
        )
        args = self._args(offline=True, apply=False)
        args.enb_only = True
        enb_source_factory = Mock(side_effect=AssertionError("constructed ENB source"))
        with patch.object(
            script,
            "build_enb_offline_plan",
            return_value={"status": "offline_plan", "mode": "enb_only"},
        ) as plan_mock:
            result = script.run(
                args,
                repository_factory=lambda _: repository,
                source_factory=Mock(side_effect=AssertionError("constructed primary source")),
                cov_direct_source_factory=Mock(
                    side_effect=AssertionError("constructed COV source")
                ),
                cov_wiki_source_factory=Mock(
                    side_effect=AssertionError("constructed WIKI source")
                ),
                enb_source_factory=enb_source_factory,
            )

        self.assertEqual(result, {"status": "offline_plan", "mode": "enb_only"})
        plan_mock.assert_called_once_with(repository, repository.current_release()[0])
        enb_source_factory.assert_not_called()

    def test_enb_dry_run_constructs_only_capped_source_when_explicitly_allowed(self):
        release = SimpleNamespace(version="r1")
        repository = SimpleNamespace(current_release=lambda: (release, "etag"))
        args = self._args(offline=False, apply=False)
        args.enb_only = True
        args.fetch_enb = True
        source = object()
        enb_source_factory = Mock(return_value=source)
        prepared = SimpleNamespace(summary={"status": "validated_dry_run"})
        with patch.object(
            script,
            "prepare_enb_collection",
            return_value=prepared,
        ) as prepare_mock:
            result = script.run(
                args,
                repository_factory=lambda _: repository,
                source_factory=Mock(side_effect=AssertionError("primary source used")),
                cov_direct_source_factory=Mock(side_effect=AssertionError("COV used")),
                cov_wiki_source_factory=Mock(side_effect=AssertionError("WIKI used")),
                enb_source_factory=enb_source_factory,
            )

        self.assertEqual(result["status"], "validated_dry_run")
        enb_source_factory.assert_called_once_with(workers=2, max_attempts=3)
        prepare_mock.assert_called_once_with(repository, release, "etag", source)

    def test_fetch_enb_flag_is_rejected_outside_enb_mode(self):
        repository = SimpleNamespace(
            current_release=lambda: (SimpleNamespace(version="r1"), "etag")
        )
        args = self._args(offline=False, apply=False)
        args.fetch_enb = True

        with self.assertRaisesRegex(ValueError, "only valid together"):
            script.run(args, repository_factory=lambda _: repository)

    def test_offline_plan_never_constructs_the_eodhd_source(self):
        repository = SimpleNamespace(
            current_release=lambda: (SimpleNamespace(version="r1"), "etag")
        )
        source_calls = []
        direct_source_calls = []
        wiki_source_calls = []

        def source_factory(**kwargs):
            source_calls.append(kwargs)
            raise AssertionError("offline plan constructed a source")

        def direct_source_factory(*args, **kwargs):
            direct_source_calls.append((args, kwargs))
            raise AssertionError("offline plan constructed the DirectIndex source")

        def wiki_source_factory(*args, **kwargs):
            wiki_source_calls.append((args, kwargs))
            raise AssertionError("offline plan constructed the COV WIKI source")

        with (
            patch.object(
                script,
                "_load_evidence_report",
                return_value=_loaded_report("r1"),
            ),
            patch.object(
                script,
                "build_offline_plan",
                return_value={"status": "offline_plan"},
            ),
        ):
            result = script.run(
                self._args(offline=True, apply=False),
                repository_factory=lambda _: repository,
                source_factory=source_factory,
                cov_direct_source_factory=direct_source_factory,
                cov_wiki_source_factory=wiki_source_factory,
            )

        self.assertEqual(result["status"], "offline_plan")
        self.assertEqual(source_calls, [])
        self.assertEqual(direct_source_calls, [])
        self.assertEqual(wiki_source_calls, [])

    def test_validated_dry_run_does_not_enter_apply(self):
        repository = SimpleNamespace(
            current_release=lambda: (SimpleNamespace(version="r1"), "etag")
        )
        prepared = SimpleNamespace(summary={"status": "validated_dry_run"})
        primary_source = object()
        direct_source = object()
        wiki_source = object()
        source_factory = Mock(return_value=primary_source)
        direct_source_factory = Mock(return_value=direct_source)
        wiki_source_factory = Mock(return_value=wiki_source)
        with (
            patch.object(
                script,
                "_load_evidence_report",
                return_value=_loaded_report("r1"),
            ),
            patch.object(
                script,
                "prepare_collection",
                return_value=prepared,
            ) as prepare_mock,
            patch.object(script, "apply_collection") as apply_mock,
        ):
            result = script.run(
                self._args(offline=False, apply=False),
                repository_factory=lambda _: repository,
                source_factory=source_factory,
                cov_direct_source_factory=direct_source_factory,
                cov_wiki_source_factory=wiki_source_factory,
            )

        self.assertEqual(result["status"], "validated_dry_run")
        source_factory.assert_called_once_with(workers=2)
        direct_source_factory.assert_called_once_with(
            Path("unused") / "state/cov-directindex",
            allow_http=False,
        )
        wiki_source_factory.assert_called_once_with(
            Path("unused") / "state/cov-quandl-wiki",
            allow_http=False,
        )
        self.assertIs(prepare_mock.call_args.args[-4], primary_source)
        self.assertIs(prepare_mock.call_args.args[-3], direct_source)
        self.assertIs(prepare_mock.call_args.args[-2], wiki_source)
        self.assertEqual(prepare_mock.call_args.args[-1], "")
        apply_mock.assert_not_called()

    def test_stale_evidence_report_is_rejected_before_provider_construction(self):
        repository = SimpleNamespace(
            current_release=lambda: (SimpleNamespace(version="current-r2"), "etag")
        )
        source_factory = unittest.mock.Mock()
        direct_source_factory = unittest.mock.Mock()
        wiki_source_factory = unittest.mock.Mock()
        with patch.object(
            script,
            "_load_evidence_report",
            return_value=_loaded_report("stale-r1"),
        ):
            with self.assertRaisesRegex(RuntimeError, "not for the current release"):
                script.run(
                    self._args(offline=False, apply=True),
                    repository_factory=lambda _: repository,
                    source_factory=source_factory,
                    cov_direct_source_factory=direct_source_factory,
                    cov_wiki_source_factory=wiki_source_factory,
                )

        source_factory.assert_not_called()
        direct_source_factory.assert_not_called()
        wiki_source_factory.assert_not_called()

    def test_failed_local_preflight_spends_no_provider_call(self):
        release = DataRelease(
            version="r1",
            created_at="2026-07-18T00:00:00Z",
            completed_session="2026-07-15",
            dataset_versions={},
        )
        source = SimpleNamespace(fetch=Mock())
        with patch.object(
            script,
            "build_local_preflight",
            side_effect=ValueError("DOW preflight failed"),
        ):
            with self.assertRaisesRegex(ValueError, "DOW preflight"):
                script.prepare_collection(
                    SimpleNamespace(),
                    release,
                    "etag",
                    {},
                    _loaded_report("r1").artifact,
                    source,
                )

        source.fetch.assert_not_called()

    def test_enb_cache_replay_uses_zero_provider_calls_and_retains_three_call_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = SimpleNamespace(root=Path(directory))
            release = DataRelease(
                version="identity-repaired-r1",
                created_at="2026-07-18T00:00:00Z",
                completed_session="2026-07-15",
                dataset_versions={},
            )
            fetched = _valid_enb_fetch()
            script._write_fetched_bundle_cache(
                script._enb_bundle_cache_path(repository, release),
                release,
                fetched,
                http_attempts=script.ENB_MAX_EODHD_HTTP_ATTEMPTS,
                signature=script._enb_bundle_signature(release),
            )
            frames = {
                dataset: pd.DataFrame(
                    columns=dataset_spec(dataset).required_columns
                )
                for dataset in script.WRITE_DATASETS
            }
            official = SourceArtifact(
                source="official_identity_evidence_raw",
                source_url=script.ENB_SEC_COMPLETION_URL,
                retrieved_at="2026-07-18T00:00:00Z",
                content=b"official",
                content_type="text/html",
            )
            preflight = script.EnbPreflight(
                selection=script.select_catalog_entries(
                    _catalogs((script.ENB_SPEC,)),
                    (script.ENB_SPEC,),
                )["ENB"],
                existing=frames,
                pointer_etags={},
                spectra_security_id=script.SPECTRA_SECURITY_ID,
                official_artifact=official,
            )
            source = SimpleNamespace(fetch=Mock())
            with (
                patch.object(script, "build_enb_preflight", return_value=preflight),
                patch.object(
                    script,
                    "apply_enb_supplement",
                    return_value=(frames, {"economic_crosscheck": {"passed": True}}),
                ),
                patch.object(
                    script,
                    "build_adjustment_factors",
                    return_value=frames["adjustment_factors"],
                ),
                patch.object(script, "validate_enb_candidate_frames", return_value=()),
                patch.object(script, "active_price_gaps", return_value=()),
                patch.object(script, "_assert_release_unchanged"),
            ):
                prepared = script.prepare_enb_collection(
                    repository,
                    release,
                    "etag",
                    source,
                )

            source.fetch.assert_not_called()
            self.assertTrue(prepared.summary["fetched_bundle_reused"])
            self.assertEqual(prepared.summary["actual_eodhd_http_attempts_this_run"], 0)
            self.assertEqual(prepared.summary["bundle_eodhd_http_attempts"], 3)

    def test_missing_enb_cache_without_fetch_authority_fails_before_provider_use(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = SimpleNamespace(root=Path(directory))
            release = DataRelease(
                version="identity-repaired-r1",
                created_at="2026-07-18T00:00:00Z",
                completed_session="2026-07-15",
                dataset_versions={},
            )
            preflight = SimpleNamespace()
            with patch.object(
                script,
                "build_enb_preflight",
                return_value=preflight,
            ):
                with self.assertRaisesRegex(FileNotFoundError, "--fetch-enb"):
                    script.prepare_enb_collection(
                        repository,
                        release,
                        "etag",
                        None,
                    )

    def test_valid_primary_and_cov_eodhd_caches_are_reused_without_eodhd_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = SimpleNamespace(root=Path(directory))
            release = DataRelease(
                version="r1",
                created_at="2026-07-18T00:00:00Z",
                completed_session="2026-07-15",
                dataset_versions={},
                warnings=(script.MISSING_PROVIDER_WARNING,),
            )
            primary = _valid_primary_fetch()
            script._write_fetched_bundle_cache(
                script._bundle_cache_path(repository, release),
                release,
                primary,
                http_attempts=script.MAX_EODHD_HTTP_ATTEMPTS,
            )
            script._write_fetched_bundle_cache(
                script._cov_bundle_cache_path(repository, release),
                release,
                _cov_eodhd_incomplete_fetch(),
                http_attempts=script.COV_MAX_EODHD_HTTP_ATTEMPTS,
                signature=script._cov_bundle_signature(release),
            )
            primary_source = SimpleNamespace(fetch=Mock())
            direct_fetched = _valid_cov_direct_fetch()
            direct_source = SimpleNamespace(
                fetch=Mock(return_value=direct_fetched), attempt_count=3
            )
            wiki_fetched = _valid_cov_wiki_fetch()
            wiki_source = SimpleNamespace(
                fetch=Mock(return_value=wiki_fetched), attempt_count=3
            )
            preflight = SimpleNamespace(
                cov_security_id=script.COV_SPEC.security_id,
                existing={},
                selections={},
                purge_ids={},
            )

            with (
                patch.object(
                    script,
                    "COV_DIRECTINDEX_SHA256",
                    _cov_direct_test_hashes(direct_fetched),
                ),
                patch.object(
                    script, "COV_WIKI_SHA256", _cov_wiki_test_hashes(wiki_fetched)
                ),
                patch.object(script, "build_local_preflight", return_value=preflight),
                patch.object(
                    script,
                    "_read_cov_eodhd_symbol_failure_cache",
                    return_value=SourceArtifact(
                        "symbol-failure", "https://test", "2026-07-18Z", b"a", "application/json"
                    ),
                ),
                patch.object(
                    script,
                    "_load_or_import_cov_eodhd_full_us_failure",
                    return_value=SourceArtifact(
                        "full-failure", "https://test", "2026-07-18Z", b"b", "application/json"
                    ),
                ),
                patch.object(
                    script,
                    "rewrite_market_frames",
                    side_effect=RuntimeError("stop after cache decisions"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after"):
                    script.prepare_collection(
                        repository,
                        release,
                        "etag",
                        {},
                        _loaded_report("r1").artifact,
                        primary_source,
                        direct_source,
                        wiki_source,
                    )

            primary_source.fetch.assert_not_called()
            direct_source.fetch.assert_called_once_with()
            wiki_source.fetch.assert_called_once_with()
            cov_cache = script._cov_bundle_cache_path(repository, release)
            self.assertTrue(cov_cache.is_file())
            cached = script._read_fetched_bundle_cache(
                cov_cache,
                release,
                signature=script._cov_bundle_signature(release),
            )
            self.assertIsNotNone(cached)
            self.assertEqual(cached[1], 3)
            self.assertEqual(len(cached[0].artifacts), 3)
            self.assertEqual(cached[0].missing_symbols, ("COV.US",))

    def test_invalid_preserved_cov_eodhd_cache_is_never_refetched(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = SimpleNamespace(root=Path(directory))
            release = DataRelease(
                version="r1",
                created_at="2026-07-18T00:00:00Z",
                completed_session="2026-07-15",
                dataset_versions={},
                warnings=(script.MISSING_PROVIDER_WARNING,),
            )
            primary = _valid_primary_fetch()
            script._write_fetched_bundle_cache(
                script._bundle_cache_path(repository, release),
                release,
                primary,
                http_attempts=script.MAX_EODHD_HTTP_ATTEMPTS,
            )
            script._write_fetched_bundle_cache(
                script._cov_bundle_cache_path(repository, release),
                release,
                _cov_eodhd_incomplete_fetch(),
                http_attempts=2,
                signature=script._cov_bundle_signature(release),
            )
            primary_source = SimpleNamespace(fetch=Mock())
            direct_source = SimpleNamespace(fetch=Mock())
            preflight = SimpleNamespace(cov_security_id=script.COV_SPEC.security_id)

            with patch.object(
                script,
                "build_local_preflight",
                return_value=preflight,
            ):
                with self.assertRaisesRegex(ValueError, "exactly one audited"):
                    script.prepare_collection(
                        repository,
                        release,
                        "etag",
                        {},
                        _loaded_report("r1").artifact,
                        primary_source,
                        direct_source,
                    )

            primary_source.fetch.assert_not_called()
            direct_source.fetch.assert_not_called()
            self.assertTrue(script._cov_bundle_cache_path(repository, release).is_file())

    def test_full_us_failure_response_is_imported_once_then_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = json.dumps([{"code": "ABC", "date": "2015-01-02"}]).encode()
            import_path = root / "full-us.json"
            import_path.write_bytes(content)
            with (
                patch.object(
                    script, "COV_EODHD_FULL_US_FAILURE_SHA256", script.sha256_bytes(content)
                ),
                patch.object(script, "COV_EODHD_FULL_US_FAILURE_ROWS", 1),
            ):
                first = script._load_or_import_cov_eodhd_full_us_failure(
                    root, import_path
                )
                second = script._load_or_import_cov_eodhd_full_us_failure(root, None)
            self.assertEqual(first.content, content)
            self.assertEqual(second.content, content)
            self.assertTrue(script._cov_eodhd_full_us_failure_cache_path(root).is_file())

    def test_successful_prepare_rebuilds_factors_archives_all_requests_and_clears_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            release = DataRelease(
                version="r1",
                created_at="2026-07-18T00:00:00Z",
                completed_session="2026-07-15",
                dataset_versions={},
                warnings=(script.MISSING_PROVIDER_WARNING,),
            )
            repository = SimpleNamespace(
                root=Path(directory),
                current_release=lambda: (release, "etag"),
            )
            primary = _valid_primary_fetch()
            script._write_fetched_bundle_cache(
                script._bundle_cache_path(repository, release),
                release,
                primary,
                http_attempts=script.MAX_EODHD_HTTP_ATTEMPTS,
            )
            script._write_fetched_bundle_cache(
                script._cov_bundle_cache_path(repository, release),
                release,
                _cov_eodhd_incomplete_fetch(),
                http_attempts=script.COV_MAX_EODHD_HTTP_ATTEMPTS,
                signature=script._cov_bundle_signature(release),
            )
            cov_id = script.COV_SPEC.security_id
            mdt_id = "MDT-ID"
            base_frames = {
                "security_master": pd.DataFrame(
                    [
                        _master(cov_id, "COV", "COV.US", "Covidien plc"),
                        _master(mdt_id, "MDT", "MDT.US", "Medtronic PLC"),
                    ]
                ),
                "symbol_history": pd.DataFrame(
                    [
                        _history(cov_id, "COV"),
                        _history(mdt_id, "MDT"),
                    ]
                ),
                "daily_price_raw": pd.DataFrame(
                    [
                        _price(mdt_id, "2015-01-26", 75.59),
                        _price(mdt_id, "2015-01-27", 75.26),
                        _price(mdt_id, "2026-07-15", 90.0),
                    ]
                ),
                "corporate_actions": pd.DataFrame(
                    columns=dataset_spec("corporate_actions").required_columns
                ),
                "source_archive": pd.DataFrame(
                    columns=dataset_spec("source_archive").required_columns
                ),
            }
            preflight = SimpleNamespace(
                existing=base_frames,
                selections={},
                purge_ids={},
                cov_security_id=cov_id,
                pointer_etags={},
            )
            primary_source = SimpleNamespace(fetch=Mock())
            direct_fetched = _valid_cov_direct_fetch()
            direct_source = SimpleNamespace(
                fetch=Mock(return_value=direct_fetched),
                attempt_count=3,
            )
            wiki_fetched = _valid_cov_wiki_fetch()
            wiki_source = SimpleNamespace(
                fetch=Mock(return_value=wiki_fetched),
                attempt_count=3,
            )

            with (
                patch.object(
                    script,
                    "COV_DIRECTINDEX_SHA256",
                    _cov_direct_test_hashes(direct_fetched),
                ),
                patch.object(
                    script,
                    "COV_WIKI_SHA256",
                    _cov_wiki_test_hashes(wiki_fetched),
                ),
                patch.object(script, "build_local_preflight", return_value=preflight),
                patch.object(
                    script,
                    "_read_cov_eodhd_symbol_failure_cache",
                    return_value=SourceArtifact(
                        "symbol-failure", "https://test/symbol", "2026-07-18Z", b"failure-a", "application/json"
                    ),
                ),
                patch.object(
                    script,
                    "_load_or_import_cov_eodhd_full_us_failure",
                    return_value=SourceArtifact(
                        "full-failure", "https://test/full", "2026-07-18Z", b"failure-b", "application/json"
                    ),
                ),
                patch.object(
                    script,
                    "_cov_eodhd_failure_manifest_artifact",
                    return_value=SourceArtifact(
                        "failure-manifest", "https://test/manifest", "2026-07-18Z", b"failure-manifest", "application/json"
                    ),
                ),
                patch.object(
                    script,
                    "rewrite_market_frames",
                    return_value=(base_frames, {}),
                ),
                patch.object(script, "validate_candidate_frames", return_value=()),
            ):
                prepared = script.prepare_collection(
                    repository,
                    release,
                    "etag",
                    {},
                    _loaded_report("r1").artifact,
                    primary_source,
                    direct_source,
                    wiki_source,
                )

            primary_source.fetch.assert_not_called()
            direct_source.fetch.assert_called_once_with()
            wiki_source.fetch.assert_called_once_with()
            self.assertEqual(
                prepared.summary["active_missing_or_stale_after_supplement"],
                0,
            )
            self.assertTrue(prepared.summary["cov_cross_validated"])
            self.assertEqual(prepared.summary["cov_price_source"], "directindex_pinned_csv")
            self.assertEqual(prepared.summary["combined_maximum_eodhd_http_attempts"], 39)
            self.assertEqual(prepared.summary["artifact_count"], 50)
            self.assertEqual(prepared.summary["archived_artifact_count"], 54)
            self.assertEqual(len(prepared.frames["source_archive"]), 54)
            self.assertTrue(prepared.summary["cov_cross_validation_evidence_hash"])
            self.assertNotIn(
                script.COV_PENDING_INDEPENDENT_VALIDATION_WARNING,
                prepared.warnings,
            )
            cov_factors = prepared.frames["adjustment_factors"].loc[
                prepared.frames["adjustment_factors"].security_id.eq(cov_id)
            ]
            self.assertEqual(len(cov_factors), len(script.COV_EXPECTED_SESSIONS))
            self.assertEqual(
                prepared.cleared_release_warnings,
                (script.MISSING_PROVIDER_WARNING,),
            )

    def test_invalid_cached_bundle_is_quarantined_before_one_refetch(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = SimpleNamespace(root=Path(directory))
            release = DataRelease(
                version="r1",
                created_at="2026-07-18T00:00:00Z",
                completed_session="2026-07-15",
                dataset_versions={},
            )
            invalid = SimpleNamespace(
                prices=pd.DataFrame(
                    [
                        _price(item.security_id, item.event_date)
                        for item in script.SUCCESSOR_SPECS
                    ]
                ),
                corporate_actions=pd.DataFrame(
                    columns=dataset_spec("corporate_actions").required_columns
                ),
                artifacts=_artifacts(script.SUCCESSOR_SPECS),
                missing_symbols=(),
            )
            cache_path = script._bundle_cache_path(repository, release)
            script._write_fetched_bundle_cache(
                cache_path,
                release,
                invalid,
                http_attempts=script.MAX_EODHD_HTTP_ATTEMPTS,
            )
            source = SimpleNamespace(fetch=Mock(return_value=invalid))

            with patch.object(
                script,
                "build_local_preflight",
                return_value=SimpleNamespace(),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "history is truncated|active history is stale",
                ):
                    script.prepare_collection(
                        repository,
                        release,
                        "etag",
                        {},
                        _loaded_report("r1").artifact,
                        source,
                    )

            source.fetch.assert_called_once()
            self.assertFalse(cache_path.exists())
            self.assertEqual(
                len(tuple(cache_path.parent.glob(f"{cache_path.name}.invalid-*"))),
                1,
            )


if __name__ == "__main__":
    unittest.main()
