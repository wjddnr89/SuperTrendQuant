from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import yaml

from supertrend_quant.market_store.cross_validation import (
    TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256,
    reviewed_nonterminal_inventory_sha256,
)
from supertrend_quant.market_store.ingest import SourceArtifact
from supertrend_quant.market_store.manifest import sha256_bytes
from supertrend_quant.market_store.schemas import dataset_spec


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/repair_us_formula_one_identity.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_formula_one_identity", SCRIPT_PATH
)
script = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


COMPLETED_SESSION = "2017-01-26"


def _source(label: str) -> dict:
    return {
        "source": label,
        "source_url": f"https://example.test/{label}",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": sha256_bytes(label.encode()),
    }


def _master(lineage) -> dict:
    return {
        "security_id": lineage.security_id,
        "primary_symbol": lineage.old_symbol,
        "provider_symbol": f"{lineage.old_symbol}.US",
        "action_provider_symbol": f"{lineage.old_symbol}.US",
        "name": "Liberty Media Corp",
        "exchange": "NASDAQ",
        "asset_type": "STOCK",
        "currency": "USD",
        "country": "US",
        "active_from": "2015-01-02",
        "active_to": script.LEGAL_EFFECTIVE_DATE,
        "isin": "",
        **_source("catalog"),
    }


def _history(lineage) -> dict:
    return {
        "security_id": lineage.security_id,
        "symbol": lineage.old_symbol,
        "exchange": "NASDAQ",
        "effective_from": "2015-01-01",
        "effective_to": "",
        **_source("catalog"),
    }


def _price(
    lineage,
    session: str,
    close: float,
    *,
    new_symbol: bool = False,
) -> dict:
    symbol = lineage.new_symbol if new_symbol else lineage.old_symbol
    return {
        "security_id": lineage.security_id,
        "session": session,
        "open": close - 0.2,
        "high": close + 0.4,
        "low": close - 0.5,
        "close": close,
        "volume": 1_000.0,
        "currency": "USD",
        "source": "eodhd_eod",
        "source_url": (
            f"https://eodhd.com/api/eod/{symbol}.US?"
            f"from={script.FETCH_START}&to={COMPLETED_SESSION}"
            if new_symbol
            else f"https://example.test/{symbol}/old"
        ),
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": sha256_bytes(f"{symbol}-eod".encode()),
    }


def _factor(lineage, session: str) -> dict:
    return {
        "security_id": lineage.security_id,
        "session": session,
        "split_factor": 1.0,
        "total_return_factor": 1.0,
        "source_version": "fixture",
        "calculated_at": "2026-07-18T00:00:00Z",
        "source": "derived",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": "fixture",
    }


def _official() -> script.OfficialEvidence:
    legal_terms = SourceArtifact(
        source="sec_edgar_filing",
        source_url=script.LEGAL_TERMS_URL,
        retrieved_at="2026-07-18T00:00:00Z",
        content=b"fixture exact cached legal terms",
        content_type="text/plain",
    )
    market_boundary = SourceArtifact(
        source="sec_edgar_filing",
        source_url=script.MARKET_BOUNDARY_URL,
        retrieved_at="2026-07-18T00:00:00Z",
        content=b"fixture exact Jan24 legal and Jan25 trading boundary",
        content_type="text/plain",
    )
    return script.OfficialEvidence(
        legal_terms=legal_terms,
        market_boundary=market_boundary,
    )


def test_source_archive_rows_preserve_optional_source_url():
    artifact = SourceArtifact(
        source="official_identity_evidence_raw",
        source_url=script.MARKET_BOUNDARY_URL,
        retrieved_at="2026-07-18T00:00:00Z",
        content=b"reviewed Formula One SEC filing",
        content_type="text/html",
    )

    frame = script._archive_rows([artifact], completed_session=COMPLETED_SESSION)

    assert "source_url" in frame.columns
    assert frame.iloc[0]["source_url"] == script.MARKET_BOUNDARY_URL


def _catalog() -> script.CatalogProof:
    rows = {
        lineage.new_symbol: {
            "Code": lineage.new_symbol,
            "Name": (
                "Liberty Media Corporation Series A Liberty Formula One Common Stock"
                if lineage.new_symbol == "FWONA"
                else "Liberty Media Corporation Series C Liberty Formula One Common Stock"
            ),
            "Exchange": "NASDAQ",
            "Currency": "USD",
            "Type": "Common Stock",
            "Isin": lineage.isin,
        }
        for lineage in script.LINEAGES
    }
    artifact = SourceArtifact(
        source="eodhd_exchange_symbols",
        source_url=script.ACTIVE_CATALOG_URL,
        retrieved_at="2026-07-18T00:00:00Z",
        content=json.dumps(list(rows.values()), sort_keys=True).encode(),
        content_type="application/json",
    )
    return script.CatalogProof(rows=rows, artifact=artifact)


def _artifact(lineage, endpoint: str, payload: list[dict]) -> SourceArtifact:
    return SourceArtifact(
        source=f"eodhd_{endpoint}",
        source_url=(
            f"https://eodhd.com/api/{endpoint}/{lineage.provider_symbol}?"
            f"from={script.FETCH_START}&to={COMPLETED_SESSION}"
        ),
        retrieved_at="2026-07-18T00:00:00Z",
        content=script._canonical_json_bytes(payload),
        content_type="application/json",
    )


def _provider() -> script.ProviderBundle:
    prices = []
    artifacts = []
    for offset, lineage in enumerate(script.LINEAGES):
        eod_payload = [
            {
                "date": script.TRANSITION_DATE,
                "open": 29.8 + offset,
                "high": 30.4 + offset,
                "low": 29.5 + offset,
                "close": 30.0 + offset,
                "volume": 1_000,
            },
            {
                "date": COMPLETED_SESSION,
                "open": 30.8 + offset,
                "high": 31.4 + offset,
                "low": 30.5 + offset,
                "close": 31.0 + offset,
                "volume": 1_000,
            },
        ]
        eod = _artifact(lineage, "eod", eod_payload)
        artifacts.extend(
            (eod, _artifact(lineage, "div", []), _artifact(lineage, "splits", []))
        )
        for row in eod_payload:
            prices.append(
                {
                    "security_id": lineage.security_id,
                    "session": row["date"],
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                    "currency": "USD",
                    "source": eod.source,
                    "source_url": eod.source_url,
                    "retrieved_at": eod.retrieved_at,
                    "source_hash": eod.source_hash,
                }
            )
    return script.ProviderBundle(
        prices=pd.DataFrame(
            prices,
            columns=tuple(
                dict.fromkeys(
                    (*dataset_spec("daily_price_raw").required_columns, "source_url")
                )
            ),
        ),
        corporate_actions=pd.DataFrame(
            columns=dataset_spec("corporate_actions").required_columns
        ),
        artifacts=tuple(artifacts),
        http_attempts=0,
    )


def _existing() -> dict[str, pd.DataFrame]:
    prices = []
    factors = []
    for offset, lineage in enumerate(script.LINEAGES):
        for session, close in (
            ("2017-01-20", 29.0 + offset),
            (script.OLD_PREVIOUS_SESSION, 29.5 + offset),
            (script.OLD_LAST_SESSION, 30.0 + offset),
        ):
            prices.append(_price(lineage, session, close))
            factors.append(_factor(lineage, session))
    return {
        "security_master": pd.DataFrame([_master(value) for value in script.LINEAGES]),
        "symbol_history": pd.DataFrame([_history(value) for value in script.LINEAGES]),
        "daily_price_raw": pd.DataFrame(prices),
        "corporate_actions": pd.DataFrame(
            columns=dataset_spec("corporate_actions").required_columns
        ),
        "adjustment_factors": pd.DataFrame(factors),
        "source_archive": pd.DataFrame(
            columns=dataset_spec("source_archive").required_columns
        ),
    }


class FormulaOneRepairTests(unittest.TestCase):
    def test_prepared_repair_preserves_ids_and_splits_symbol_history(self):
        evidence = _official()
        with (
            mock.patch.object(
                script,
                "MARKET_BOUNDARY_SHA256",
                evidence.market_boundary.source_hash,
            ),
            mock.patch.object(
                script,
                "LEGAL_TERMS_SHA256",
                evidence.legal_terms.source_hash,
            ),
        ):
            frames, summary, artifacts = script.prepare_formula_one_repair(
                _existing(),
                catalog=_catalog(),
                evidence=evidence,
                provider=_provider(),
                completed_session=COMPLETED_SESSION,
            )

        master = frames["security_master"].set_index("security_id")
        history = frames["symbol_history"]
        actions = frames["corporate_actions"]
        prices = frames["daily_price_raw"]
        for lineage in script.LINEAGES:
            self.assertEqual(
                master.loc[lineage.security_id, "primary_symbol"], lineage.new_symbol
            )
            self.assertEqual(
                master.loc[lineage.security_id, "provider_symbol"],
                lineage.provider_symbol,
            )
            self.assertNotIn(
                lineage.forbidden_new_security_id, set(master.index.astype(str))
            )
            intervals = history.loc[
                history.security_id.astype(str).eq(lineage.security_id)
            ]
            self.assertEqual(
                {
                    (row.symbol, str(row.effective_from), str(row.effective_to or ""))
                    for row in intervals.itertuples(index=False)
                },
                {
                    (lineage.old_symbol, "2015-01-01", script.OLD_LAST_SESSION),
                    (lineage.new_symbol, script.TRANSITION_DATE, ""),
                },
            )
            transition = prices.loc[
                prices.security_id.astype(str).eq(lineage.security_id)
                & pd.to_datetime(prices.session).eq(pd.Timestamp(script.TRANSITION_DATE))
            ]
            self.assertEqual(len(transition), 1)
            self.assertIn(lineage.provider_symbol, str(transition.iloc[0].source_url))
            retained = prices.loc[
                prices.security_id.astype(str).eq(lineage.security_id)
                & pd.to_datetime(prices.session).eq(
                    pd.Timestamp(script.OLD_LAST_SESSION)
                )
            ]
            self.assertEqual(len(retained), 1)
            self.assertIn(lineage.old_symbol, str(retained.iloc[0].source_url))
            action = actions.loc[
                actions.event_id.astype(str).eq(
                    script.canonical_lifecycle_event_id(
                        lineage.security_id, "ticker_change", script.TRANSITION_DATE
                    )
                )
            ].iloc[0]
            self.assertEqual(action.new_security_id, lineage.security_id)
            self.assertEqual(action.new_symbol, lineage.new_symbol)
            self.assertTrue(pd.isna(action.ratio))
        self.assertEqual(summary["price_rows_removed"], 0)
        self.assertEqual(summary["official_ticker_change_rows"], 2)
        self.assertEqual(len(artifacts), 8)
        self.assertEqual(
            len(frames["adjustment_factors"]), len(frames["daily_price_raw"])
        )

    def test_implausible_adjacent_session_bridge_fails_closed(self):
        evidence = _official()
        provider = _provider()
        mismatch = (
            provider.prices.security_id.astype(str).eq(script.LINEAGES[0].security_id)
            & pd.to_datetime(provider.prices.session).eq(
                pd.Timestamp(script.TRANSITION_DATE)
            )
        )
        provider.prices.loc[mismatch, ["open", "high", "low", "close"]] = [
            98.0,
            100.0,
            97.0,
            99.0,
        ]
        with (
            mock.patch.object(
                script,
                "MARKET_BOUNDARY_SHA256",
                evidence.market_boundary.source_hash,
            ),
            self.assertRaisesRegex(ValueError, "economic transition return"),
        ):
            script.prepare_formula_one_repair(
                _existing(),
                catalog=_catalog(),
                evidence=evidence,
                provider=provider,
                completed_session=COMPLETED_SESSION,
            )

    def test_forbidden_new_fwon_security_id_fails_closed(self):
        evidence = _official()
        existing = _existing()
        extra = _master(script.LINEAGES[0])
        extra["security_id"] = script.LINEAGES[0].forbidden_new_security_id
        existing["security_master"] = pd.concat(
            [existing["security_master"], pd.DataFrame([extra])], ignore_index=True
        )
        with (
            mock.patch.object(
                script,
                "MARKET_BOUNDARY_SHA256",
                evidence.market_boundary.source_hash,
            ),
            self.assertRaisesRegex(ValueError, "incorrectly assigned new security_ids"),
        ):
            script.prepare_formula_one_repair(
                existing,
                catalog=_catalog(),
                evidence=evidence,
                provider=_provider(),
                completed_session=COMPLETED_SESSION,
            )

    def test_reviewed_extractions_are_exact_twelve_field_one_for_one_rows(self):
        self.assertEqual(len(script.REVIEWED_NONTERMINAL_EXTRACTIONS), 2)
        expected_fields = {
            "event_id",
            "security_id",
            "action_type",
            "effective_date",
            "new_security_id",
            "new_symbol",
            "ratio",
            "cash_amount",
            "currency",
            "source_kind",
            "source_url",
            "source_hash",
        }
        for lineage, value in zip(
            script.LINEAGES, script.REVIEWED_NONTERMINAL_EXTRACTIONS, strict=True
        ):
            self.assertEqual(set(value), expected_fields)
            self.assertEqual(value["security_id"], lineage.security_id)
            self.assertEqual(value["new_security_id"], lineage.security_id)
            self.assertEqual(value["new_symbol"], lineage.new_symbol)
            self.assertIsNone(value["ratio"])
            self.assertIsNone(value["cash_amount"])

    def test_reviewed_extractions_match_independently_pinned_policy_inventory(self):
        policy_path = Path(__file__).parents[1] / "configs/us_cross_validation.yaml"
        policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        events = policy["events"]
        expected_ids = {
            value["event_id"] for value in script.REVIEWED_NONTERMINAL_EXTRACTIONS
        }
        selected = [
            value
            for value in events["reviewed_nonterminal_extractions"]
            if value["event_id"] in expected_ids
        ]
        self.assertEqual(selected, list(script.REVIEWED_NONTERMINAL_EXTRACTIONS))
        self.assertEqual(
            reviewed_nonterminal_inventory_sha256(events),
            TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256,
        )


class FormulaOneSourceTests(unittest.TestCase):
    @staticmethod
    def _rows(endpoint: str, symbol: str):
        if endpoint == "eod":
            offset = 0 if symbol == "FWONA.US" else 1
            return [
                {
                    "date": script.TRANSITION_DATE,
                    "open": 29.8 + offset,
                    "high": 30.4 + offset,
                    "low": 29.5 + offset,
                    "close": 30.0 + offset,
                    "volume": 1000,
                },
                {
                    "date": COMPLETED_SESSION,
                    "open": 30.8 + offset,
                    "high": 31.4 + offset,
                    "low": 30.5 + offset,
                    "close": 31.0 + offset,
                    "volume": 1000,
                },
            ]
        return []

    def test_six_one_shot_fetches_fill_immutable_cache_and_replay_is_zero_call(self):
        class FakeClient:
            def __init__(self):
                self.attempt_count = 0
                self.budget_claims = ()

            def get_json(self, endpoint, *, params):
                self.attempt_count += 1
                self.budget_claims = tuple(range(1, self.attempt_count + 1))
                self.params = dict(params)
                kind, symbol = endpoint.split("/", 1)
                return FormulaOneSourceTests._rows(kind, symbol)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            made = []

            def factory():
                value = FakeClient()
                made.append(value)
                return value

            source = script.FormulaOneEodhdSource(
                root,
                completed_session=COMPLETED_SESSION,
                allow_http=True,
                client_factory=factory,
            )
            fetched = source.fetch()
            self.assertEqual(fetched.http_attempts, 6)
            self.assertEqual(fetched.budget_claims, (1, 2, 3, 4, 5, 6))
            self.assertEqual(len(tuple(root.glob("*.json.gz"))), 6)
            self.assertEqual(len(made), 1)
            self.assertEqual(
                made[0].params,
                {"from": script.FETCH_START, "to": COMPLETED_SESSION},
            )
            self.assertNotIn("range", made[0].params)

            replay = script.FormulaOneEodhdSource(
                root,
                completed_session=COMPLETED_SESSION,
                allow_http=False,
                client_factory=lambda: self.fail("cache replay constructed HTTP client"),
            ).fetch()
            self.assertEqual(replay.http_attempts, 0)
            self.assertEqual(replay.budget_claims, ())
            pd.testing.assert_frame_equal(
                fetched.prices.reset_index(drop=True),
                replay.prices.reset_index(drop=True),
            )

    def test_offline_cache_miss_does_not_construct_client(self):
        with tempfile.TemporaryDirectory() as temp:
            source = script.FormulaOneEodhdSource(
                Path(temp),
                completed_session=COMPLETED_SESSION,
                allow_http=False,
                client_factory=lambda: self.fail("offline miss constructed client"),
            )
            with self.assertRaisesRegex(FileNotFoundError, "explicitly use"):
                source.fetch()

    def test_tampered_cache_is_rejected(self):
        class FakeClient:
            def __init__(self):
                self.attempt_count = 0
                self.budget_claims = ()

            def get_json(self, endpoint, *, params):
                self.attempt_count += 1
                self.budget_claims = tuple(range(1, self.attempt_count + 1))
                kind, symbol = endpoint.split("/", 1)
                return FormulaOneSourceTests._rows(kind, symbol)

        with tempfile.TemporaryDirectory() as temp:
            source = script.FormulaOneEodhdSource(
                Path(temp),
                completed_session=COMPLETED_SESSION,
                allow_http=True,
                client_factory=FakeClient,
            )
            source.fetch()
            source.path("eod", script.LINEAGES[0]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "Unreadable"):
                source.get("eod", script.LINEAGES[0])

    def test_capped_client_claims_budget_once_and_never_retries(self):
        class Budget:
            def __init__(self):
                self.calls = 0

            def claim(self):
                self.calls += 1
                return self.calls

        class Session:
            def __init__(self):
                self.calls = 0

            def get(self, *args, **kwargs):
                self.calls += 1
                raise OSError("network failed")

        budget = Budget()
        session = Session()
        client = script.CappedSingleAttemptEodhdClient(
            session=session,
            token="not-a-secret",
            budget=budget,
        )
        with self.assertRaisesRegex(RuntimeError, "single attempt failed"):
            client.get_json(
                "eod/FWONA.US", params={"from": script.FETCH_START}
            )
        self.assertEqual(budget.calls, 1)
        self.assertEqual(session.calls, 1)
        self.assertEqual(client.attempt_count, 1)
        self.assertEqual(client.budget_claims, (1,))

    def test_offline_plan_and_fetch_flag_are_mutually_safe(self):
        args = SimpleNamespace(
            cache_root="unused",
            offline_plan=True,
            fetch_missing_eodhd=True,
            apply=False,
        )
        with self.assertRaisesRegex(ValueError, "cannot enable"):
            script.run(
                args,
                repository_factory=lambda _: self.fail("repository constructed"),
            )


class FormulaOneOfficialEvidenceTests(unittest.TestCase):
    @staticmethod
    def _legal_content() -> bytes:
        return b"""
        <html><body>
        On January 17, 2017 the holders approved a proposal to reclassify each
        share of each existing series into one share of the corresponding
        series solely to effect the name change.
        </body></html>
        """

    @staticmethod
    def _boundary_content() -> bytes:
        return b"""
        <html><body>
        The current charter was filed with the Secretary of State on January
        24, 2017 and gave effect to the group name change, which reclassified
        each share of each existing series into one share of the corresponding
        series solely to effect the name change. Shares of FWONK are currently
        listed on Nasdaq under the symbol LMCK, although we expect shares of
        FWONK to trade under the symbol FWONK beginning on January 25, 2017.
        </body></html>
        """

    def _source(self, root: Path, *, allow_http: bool, opener):
        source = script.FormulaOneOfficialEvidenceSource(
            root / "market",
            legal_cache_root=root / "legal",
            allow_http=allow_http,
            opener=opener,
        )
        source.legal_terms_path.parent.mkdir(parents=True, exist_ok=True)
        source.legal_terms_path.write_bytes(self._legal_content())
        return source

    def test_one_shot_acquisition_observes_then_pinned_cache_replays_offline(self):
        content = self._boundary_content()

        class Response:
            status = 200
            headers = {"Content-Type": "text/html"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, limit):
                return content

        calls = []

        def opener(request, *, timeout):
            calls.append((request.full_url, timeout))
            return Response()

        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(
            "os.environ", {"SEC_USER_AGENT": "Tester test@example.com"}
        ):
            root = Path(temp)
            source = self._source(root, allow_http=True, opener=opener)
            with mock.patch.object(
                script,
                "LEGAL_TERMS_SHA256",
                sha256_bytes(self._legal_content()),
            ):
                observed = source.acquire(require_pinned=False)
            self.assertEqual(source.http_attempts, 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], script.MARKET_BOUNDARY_URL)
            self.assertEqual(observed.missing_reviewed_claims, ())
            self.assertEqual(
                observed.market_boundary.source_hash, sha256_bytes(content)
            )
            self.assertTrue(source.path.is_file())

            with (
                mock.patch.object(
                    script,
                    "MARKET_BOUNDARY_SHA256",
                    observed.market_boundary.source_hash,
                ),
                mock.patch.object(
                    script,
                    "LEGAL_TERMS_SHA256",
                    sha256_bytes(self._legal_content()),
                ),
            ):
                replay = script.FormulaOneOfficialEvidenceSource(
                    root / "market",
                    legal_cache_root=root / "legal",
                    allow_http=False,
                    opener=lambda *args, **kwargs: self.fail(
                        "pinned replay performed HTTP"
                    ),
                ).acquire(require_pinned=True)
            self.assertEqual(replay.http_attempts, 0)
            self.assertEqual(replay.market_boundary.content, content)
            self.assertEqual(replay.legal_terms.content, self._legal_content())

    def test_unpinned_hash_blocks_normal_evidence_load_before_http(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            script, "MARKET_BOUNDARY_SHA256", ""
        ):
            with self.assertRaisesRegex(RuntimeError, "not code-pinned"):
                script.load_official_evidence(Path(temp))

    def test_pinned_wrong_hash_and_missing_claims_fail_closed(self):
        weak = b"<html>LMCK FWONK January 24, 2017</html>"

        class Response:
            status = 200
            headers = {"Content-Type": "text/html"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, limit):
                return weak

        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(
            "os.environ", {"SEC_USER_AGENT": "Tester test@example.com"}
        ):
            root = Path(temp)
            source = self._source(
                root,
                allow_http=True,
                opener=lambda *args, **kwargs: Response(),
            )
            with mock.patch.object(
                script,
                "LEGAL_TERMS_SHA256",
                sha256_bytes(self._legal_content()),
            ):
                observed = source.acquire(require_pinned=False)
            self.assertIn(
                "market_one_for_one_corresponding_series",
                observed.missing_reviewed_claims,
            )
            with (
                mock.patch.object(
                    script,
                    "MARKET_BOUNDARY_SHA256",
                    observed.market_boundary.source_hash,
                ),
                mock.patch.object(
                    script,
                    "LEGAL_TERMS_SHA256",
                    sha256_bytes(self._legal_content()),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "lacks directly reviewed"):
                    source.acquire(require_pinned=True)

    def test_missing_cached_legal_terms_blocks_before_market_http(self):
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            source = script.FormulaOneOfficialEvidenceSource(
                Path(temp) / "market",
                legal_cache_root=Path(temp) / "legal",
                allow_http=True,
                opener=lambda *args, **kwargs: calls.append(args),
            )
            with self.assertRaisesRegex(FileNotFoundError, "legal-terms filing"):
                source.acquire(require_pinned=False)
        self.assertEqual(calls, [])

    def test_official_acquisition_flag_cannot_be_combined_with_apply(self):
        args = SimpleNamespace(
            cache_root="unused",
            offline_plan=False,
            fetch_missing_eodhd=False,
            fetch_official_evidence=True,
            apply=True,
        )
        repository = SimpleNamespace(root=Path("unused"))
        with self.assertRaisesRegex(ValueError, "acquisition-only"):
            script.run(args, repository_factory=lambda _: repository)


if __name__ == "__main__":
    unittest.main()
