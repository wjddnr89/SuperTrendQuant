from __future__ import annotations

import base64
import gzip
import importlib.util
import json
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from supertrend_quant.market_store.ingest import SourceArtifact
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    DatasetManifest,
)
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.storage import LocalObjectStore


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "collect_us_index_identity_repairs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "collect_us_index_identity_repairs", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Could not load {SCRIPT_PATH}")
script = importlib.util.module_from_spec(SPEC)
__import__("sys").modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def _ids() -> script.IdentityIds:
    values = {item.name: f"ID:{item.name}" for item in fields(script.IdentityIds)}
    return script.IdentityIds(**values)


def _artifact(
    *,
    source: str = "fixture",
    source_url: str = "https://official.test/evidence",
    content: bytes = b"evidence",
) -> SourceArtifact:
    return SourceArtifact(
        source=source,
        source_url=source_url,
        retrieved_at="2026-07-18T00:00:00Z",
        content=content,
        content_type="application/octet-stream",
    )


def _yahoo_payload(symbol: str) -> dict:
    spec = script.YAHOO_CHART_REQUESTS[symbol]
    sessions = script._expected_sessions(spec["raw_start"], spec["raw_end"])
    timestamps = [
        int(pd.Timestamp(f"{session}T14:30:00Z").timestamp())
        for session in sessions
    ]
    close = [20.0 + number / 100.0 for number in range(len(sessions))]
    return {
        "chart": {
            "result": [
                {
                    "meta": {
                        "symbol": symbol,
                        "currency": "USD",
                        "instrumentType": "EQUITY",
                        "exchangeName": "NMS",
                        "exchangeTimezoneName": "America/New_York",
                    },
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": [value - 0.1 for value in close],
                                "high": [value + 0.2 for value in close],
                                "low": [value - 0.2 for value in close],
                                "close": close,
                                "volume": [1_000 + number for number in range(len(close))],
                            }
                        ],
                        # Deliberately different: the identity supplement must use
                        # the raw quote close and never Yahoo's adjusted-close stream.
                        "adjclose": [{"adjclose": [1.0] * len(close)}],
                    },
                }
            ],
            "error": None,
        }
    }


def _yahoo_content(symbol: str, payload: dict | None = None) -> bytes:
    return json.dumps(payload or _yahoo_payload(symbol), separators=(",", ":")).encode()


def _yahoo_cached_response(
    symbol: str,
    *,
    payload: dict | None = None,
    content: bytes | None = None,
    content_type: str = "application/json;charset=utf-8",
    http_status: int = 200,
) -> script.YahooChartCachedResponse:
    return script.YahooChartCachedResponse(
        symbol=symbol,
        source_url=script._yahoo_chart_url(symbol),
        retrieved_at="2026-07-18T00:00:00Z",
        content=content if content is not None else _yahoo_content(symbol, payload),
        content_type=content_type,
        http_status=http_status,
    )


def _wiki_fixture() -> tuple[bytes, bytes]:
    header = ",".join(script.WIKI_ARNC_HEADER) + "\n"
    lines = [
        "A,2014-01-02,1,1,1,1,100,0,1,1,1,1,1,100\n"
    ]
    for number, session in enumerate(
        script._expected_sessions(script.AA_CROSSCHECK_START, script.AA_CROSSCHECK_END)
    ):
        close = 20.0 + number / 100.0
        dividend = 0.03 if session in script.WIKI_ARNC_DIVIDEND_DATES else 0.0
        split = 1 / 3 if session == "2016-10-06" else 1.0
        lines.append(
            f"ARNC,{session},{close - .1:.2f},{close + .2:.2f},"
            f"{close - .2:.2f},{close:.2f},{1000 + number},{dividend},"
            f"{split},{close - .1:.2f},{close + .2:.2f},"
            f"{close - .2:.2f},{close:.2f},{1000 + number}\n"
        )
    lines.append(
        "ZUMZ,2016-12-19,2,2,2,2,200,0,1,2,2,2,2,200\n"
    )
    content = (header + "".join(lines)).encode("ascii")
    segment = (header + "".join(line for line in lines if line.startswith("ARNC,"))).encode(
        "ascii"
    )
    return content, segment


def _wiki_pins(content: bytes, segment: bytes) -> dict[str, object]:
    return {
        "WIKI_ARNC_FULL_SHA256": script.sha256_bytes(content),
        "WIKI_ARNC_FULL_SIZE": len(content),
        "WIKI_ARNC_FULL_DATA_ROWS": len(content.splitlines()) - 1,
        "WIKI_ARNC_SEGMENT_SHA256": script.sha256_bytes(segment),
        "WIKI_ARNC_SEGMENT_SIZE": len(segment),
    }


class _YahooHttpResponse:
    status = 200

    def __init__(self, content: bytes, content_type: str = "application/json"):
        self.content = content
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit: int) -> bytes:
        return self.content[:limit]


def _empty(dataset: str) -> pd.DataFrame:
    return pd.DataFrame(columns=dataset_spec(dataset).required_columns)


def _price(
    security_id: str,
    session: str,
    close: float,
    *,
    volume: float = 100.0,
    source: str = "fixture",
) -> dict:
    return {
        "security_id": security_id,
        "session": session,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": volume,
        "currency": "USD",
        "source": source,
        "source_url": "https://prices.test/raw",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": "a" * 64,
    }


def _action(security_id: str, event_id: str) -> dict:
    return {
        "event_id": event_id,
        "security_id": security_id,
        "action_type": "cash_dividend",
        "effective_date": "2025-02-21",
        "ex_date": "2025-02-21",
        "announcement_date": "",
        "record_date": "",
        "payment_date": "",
        "cash_amount": 1.0,
        "ratio": None,
        "currency": "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": False,
        "source": "fixture",
        "source_url": "https://actions.test/raw",
        "source_kind": "provider",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": "b" * 64,
    }


def _master(security_id: str, provider_symbol: str = "X.US") -> dict:
    symbol = provider_symbol.removesuffix(".US").removesuffix("_old")
    return {
        "security_id": security_id,
        "primary_symbol": symbol,
        "provider_symbol": provider_symbol,
        "action_provider_symbol": provider_symbol,
        "name": f"{symbol} Corp",
        "exchange": "NYSE",
        "asset_type": "STOCK",
        "currency": "USD",
        "country": "US",
        "active_from": "2015-01-02",
        "active_to": "",
        "source": "fixture",
        "source_url": "https://catalog.test/raw",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": "c" * 64,
    }


def _history(security_id: str, symbol: str = "X") -> dict:
    return {
        "security_id": security_id,
        "symbol": symbol,
        "exchange": "NYSE",
        "effective_from": "2015-01-01",
        "effective_to": "",
        "source": "fixture",
        "source_url": "https://catalog.test/raw",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": "c" * 64,
    }


def _index_event(
    security_id: str,
    effective_date: str,
    operation: str,
) -> dict:
    return {
        "event_id": f"old-{security_id}-{effective_date}-{operation}",
        "index_id": "nasdaq100",
        "announcement_date": "",
        "effective_date": effective_date,
        "operation": operation,
        "security_id": security_id,
        "official": False,
        "source": "community_nasdaq100_history",
        "source_url": "https://community.test/history.yaml",
        "source_kind": "community",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": "d" * 64,
    }


class FrozenCallContractTests(unittest.TestCase):
    def test_network_caps_are_exact_and_inventory_backed(self):
        self.assertEqual(len(script.PRICE_PROBE_CODES), 12)
        self.assertIn("VALPQ", script.PRICE_PROBE_CODES)
        self.assertNotIn("VAL_old", script.PRICE_PROBE_CODES)
        self.assertEqual(script.SELECTED_ACTION_CODE_COUNT, 10)
        self.assertEqual(script.MAX_EODHD_HTTP_ATTEMPTS, 32)
        self.assertEqual(script.YAHOO_SUPPLEMENT_SYMBOLS, ("LILA", "LILAK"))
        self.assertEqual(script.MAX_YAHOO_HTTP_ATTEMPTS, 2)
        self.assertEqual(tuple(script.BORIS_KAGGLE_FILES), ("LILA", "LILAK"))
        self.assertEqual(script.MAX_BORIS_HTTP_ATTEMPTS, 2)
        self.assertEqual(script.MAX_WIKI_ARNC_HTTP_ATTEMPTS, 1)
        self.assertEqual(script.WIKI_ARNC_COMMIT, "ce85e08888de5b8c4f6fd8c2d03bba85a9034f64")
        self.assertEqual(
            script.WIKI_ARNC_FULL_SHA256,
            "dd5127aae478d270150904fcbad6e96a42e461e13c3d48a1587edb9b89cea43e",
        )
        self.assertEqual(len(script.OFFICIAL_EVIDENCE_URLS), 26)
        self.assertEqual(script.MAX_OFFICIAL_HTTP_ATTEMPTS, 26)

    def test_yahoo_urls_are_frozen_https_requests_without_credentials(self):
        expected = {
            "LILA": (
                "https://query1.finance.yahoo.com/v8/finance/chart/LILA"
                "?period1=1434931200&period2=1514764800&interval=1d&events=history"
            ),
            "LILAK": (
                "https://query1.finance.yahoo.com/v8/finance/chart/LILAK"
                "?period1=1434931200&period2=1514764800&interval=1d&events=history"
            ),
        }
        self.assertEqual(
            {symbol: script._yahoo_chart_url(symbol) for symbol in expected},
            expected,
        )
        for url in expected.values():
            self.assertNotIn("token", url.lower())
            self.assertNotIn("crumb", url.lower())
            self.assertNotIn("cookie", url.lower())

    def test_yahoo_network_requires_the_explicit_cli_flag(self):
        args = script._parse_args(["--fetch-yahoo-supplement"])
        self.assertTrue(args.fetch_yahoo_supplement)
        self.assertFalse(args.fetch_official_evidence)

    def test_wiki_network_requires_the_explicit_cli_flag(self):
        args = script._parse_args(["--fetch-aa-wiki-crosscheck"])
        self.assertTrue(args.fetch_aa_wiki_crosscheck)
        self.assertFalse(args.fetch_boris_crosscheck)

    def test_old_alcoa_official_identity_facts_are_bound_to_2016_sec_documents(self):
        identity = script.OFFICIAL_EVIDENCE["old_alcoa_arnc_identity"]
        separation = script.OFFICIAL_EVIDENCE["alcoa_2016_separation"]
        new_aa = script.OFFICIAL_EVIDENCE["new_alcoa_2016_10k"]
        self.assertEqual(
            identity["url"],
            "https://www.sec.gov/Archives/edgar/data/4281/000000428119000031/form10k_4q18.htm",
        )
        self.assertEqual(identity["facts"]["old_aa_to_arnc_same_issuer"], "2016-11-01")
        self.assertEqual(identity["facts"]["reverse_split_adjusted_trading"], "2016-10-06")
        self.assertTrue(separation["facts"]["new_aa_is_separate_security"])
        self.assertEqual(
            separation["facts"]["distribution_ratio_per_old_post_split_share"],
            1 / 3,
        )
        self.assertEqual(new_aa["facts"]["parent_close_2016_10_31"], 28.72)
        self.assertEqual(new_aa["facts"]["new_alcoa_when_issued_close_2016_10_31"], 21.44)

    def test_offline_plan_does_not_construct_any_network_source(self):
        release = DataRelease(
            version="base",
            created_at="2026-07-18T00:00:00Z",
            completed_session="2026-07-15",
            dataset_versions={},
        )
        repository = SimpleNamespace(current_release=lambda: (release, "etag"))
        source_factory = Mock(side_effect=AssertionError("EODHD constructed"))
        yahoo_factory = Mock(side_effect=AssertionError("Yahoo constructed"))
        boris_factory = Mock(side_effect=AssertionError("Boris constructed"))
        wiki_factory = Mock(side_effect=AssertionError("WIKI constructed"))
        evidence_factory = Mock(side_effect=AssertionError("official source constructed"))
        args = SimpleNamespace(
            cache_root="unused",
            offline_plan=True,
            apply=False,
            fetch_yahoo_supplement=False,
            fetch_boris_crosscheck=False,
            fetch_aa_wiki_crosscheck=False,
            fetch_official_evidence=False,
            supplement_bundle="",
        )
        with patch.object(script, "build_offline_plan", return_value={"status": "offline_plan"}):
            result = script.run(
                args,
                repository_factory=lambda _root: repository,
                source_factory=source_factory,
                yahoo_source_factory=yahoo_factory,
                boris_source_factory=boris_factory,
                wiki_source_factory=wiki_factory,
                evidence_source_factory=evidence_factory,
            )
        self.assertEqual(result["status"], "offline_plan")
        source_factory.assert_not_called()
        yahoo_factory.assert_not_called()
        boris_factory.assert_not_called()
        wiki_factory.assert_not_called()
        evidence_factory.assert_not_called()

    def test_offline_plan_reports_zero_calls_and_only_yahoo_supplement_keys(self):
        release = DataRelease(
            version="base",
            created_at="2026-07-18T00:00:00Z",
            completed_session="2026-07-15",
            dataset_versions={},
        )
        kind = {code.upper(): "common_stock" for code in script.PRICE_PROBE_CODES}
        catalogs = {
            "__kind__": kind,
            **{
                code.upper(): {"Name": code, "Isin": f"ISIN:{code}"}
                for code in script.PRICE_PROBE_CODES
            },
        }
        preflight = SimpleNamespace(catalogs=catalogs)
        with tempfile.TemporaryDirectory() as raw:
            repository = SimpleNamespace(root=Path(raw))
            with patch.object(
                script, "build_local_preflight", return_value=preflight
            ) as local:
                plan = script.build_offline_plan(repository, release)
        local.assert_called_once_with(repository, release)
        self.assertEqual(plan["eodhd_http_attempts_this_run"], 0)
        self.assertEqual(plan["yahoo_http_attempts_this_run"], 0)
        self.assertEqual(plan["official_http_attempts_this_run"], 0)
        self.assertFalse(plan["yahoo_accessed"])
        self.assertEqual(
            plan["network_opt_in_flags"]["yahoo"],
            "--fetch-yahoo-supplement",
        )
        self.assertEqual(plan["maximum_yahoo_http_attempts"], 2)
        self.assertEqual(plan["maximum_eodhd_http_attempts"], 32)
        self.assertEqual(plan["maximum_boris_kaggle_http_attempts"], 2)
        self.assertEqual(plan["maximum_aa_wiki_http_attempts"], 1)
        self.assertEqual(
            plan["network_opt_in_flags"]["aa_wiki"],
            "--fetch-aa-wiki-crosscheck",
        )
        self.assertEqual(
            plan["aa_wiki_cache"]["full_sha256"],
            script.WIKI_ARNC_FULL_SHA256,
        )
        self.assertEqual(
            plan["network_opt_in_flags"]["boris_kaggle"],
            "--fetch-boris-crosscheck",
        )
        self.assertFalse(
            plan["partial_apply_assessment"]["supported_by_current_collector"]
        )
        self.assertIn(
            "LILA or LILAK failure blocks every other repair",
            plan["partial_apply_assessment"]["current_behavior"],
        )
        self.assertIn(
            "do not skip LILA/LILAK checks",
            plan["partial_apply_assessment"]["unsafe_shortcut_rejected"],
        )

    def test_eodhd_client_is_single_attempt_and_hard_capped(self):
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
        self.assertEqual(client.get_json("eod/AA.US"), [])
        with self.assertRaisesRegex(RuntimeError, "call cap reached"):
            client.get_json("div/AA.US")
        self.assertEqual(session.get.call_count, 1)
        self.assertEqual(budget.claim.call_count, 1)
        self.assertEqual(client.attempt_count, 1)

    def test_official_network_is_fail_closed_without_explicit_flag(self):
        with tempfile.TemporaryDirectory() as raw:
            source = script.OfficialEvidenceSource(Path(raw), allow_http=False)
            with patch.object(script, "urlopen") as request:
                with self.assertRaisesRegex(FileNotFoundError, "explicitly allowed"):
                    source.load()
            request.assert_not_called()
            self.assertEqual(source.http_attempts, 0)

    def test_raw_official_response_is_cached_with_exact_url_and_hash(self):
        class Response:
            status = 200
            headers = {"Content-Type": "text/html"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def read(_limit):
                return b"raw official bytes"

        with tempfile.TemporaryDirectory() as raw:
            source = script.OfficialEvidenceSource(Path(raw), allow_http=True)
            url = script.OFFICIAL_EVIDENCE_URLS[0]
            with patch.object(script, "urlopen", return_value=Response()) as request:
                artifact = source._fetch(url)
            self.assertEqual(artifact.source_url, url)
            self.assertEqual(artifact.content, b"raw official bytes")
            self.assertEqual(source.get(url).source_hash, artifact.source_hash)
            self.assertEqual(source.http_attempts, 1)
            request.assert_called_once()


class YahooSupplementTests(unittest.TestCase):
    def test_network_is_fail_closed_without_explicit_flag(self):
        with tempfile.TemporaryDirectory() as raw:
            source = script.YahooIdentitySupplementSource(
                Path(raw), allow_http=False
            )
            with patch.object(script, "urlopen") as request:
                with self.assertRaisesRegex(
                    FileNotFoundError, "--fetch-yahoo-supplement"
                ):
                    source.fetch(_ids())
            request.assert_not_called()
            self.assertEqual(source.http_attempts, 0)

    def test_two_one_shot_responses_fill_roles_and_retain_raw_artifacts(self):
        contents = {
            symbol: _yahoo_content(symbol)
            for symbol in script.YAHOO_SUPPLEMENT_SYMBOLS
        }

        def respond(request, *, timeout):
            self.assertEqual(timeout, 30.0)
            symbol = request.full_url.split("/chart/", 1)[1].split("?", 1)[0]
            return _YahooHttpResponse(contents[symbol])

        with tempfile.TemporaryDirectory() as raw:
            source = script.YahooIdentitySupplementSource(
                Path(raw), allow_http=True
            )
            with patch.object(script, "urlopen", side_effect=respond) as request:
                fetched = source.fetch(_ids())
            self.assertEqual(request.call_count, 2)
            self.assertEqual(source.http_attempts, 2)
            self.assertEqual(fetched.http_attempts, 0)
            self.assertEqual(len(fetched.prices), 1_260)
            self.assertEqual(len(fetched.crosscheck_prices), 0)
            self.assertEqual(
                set(fetched.prices["source"]),
                {script.YAHOO_LILA_PRIMARY_SOURCE},
            )
            self.assertEqual(
                float(fetched.prices.iloc[0]["close"]), 20.0
            )
            self.assertEqual(
                fetched.role_codes,
                {
                    "old_lila_regular_way_yahoo_primary": "YAHOO_CHART:LILA",
                    "old_lilak_regular_way_yahoo_primary": "YAHOO_CHART:LILAK",
                },
            )
            self.assertEqual(
                [artifact.content for artifact in fetched.artifacts],
                [contents[symbol] for symbol in script.YAHOO_SUPPLEMENT_SYMBOLS],
            )
            self.assertEqual(
                {artifact.source for artifact in fetched.artifacts},
                {script.YAHOO_LILA_PRIMARY_SOURCE},
            )

            # A second source is cache-only and must make zero network calls.
            cached = script.YahooIdentitySupplementSource(
                Path(raw), allow_http=False
            )
            with patch.object(script, "urlopen") as second_request:
                second = cached.fetch(_ids())
            second_request.assert_not_called()
            self.assertEqual(cached.http_attempts, 0)
            self.assertEqual(len(second.prices), 1_260)
            self.assertEqual(len(second.crosscheck_prices), 0)

    def test_cache_envelope_validates_payload_and_raw_content_hashes(self):
        with tempfile.TemporaryDirectory() as raw:
            cache = script.YahooChartSupplementCache(Path(raw))
            with patch.object(
                script,
                "urlopen",
                return_value=_YahooHttpResponse(_yahoo_content("LILA")),
            ):
                cached = cache.fetch("LILA")
            path = cache.path("LILA")
            encoded = path.read_bytes()
            envelope = json.loads(gzip.decompress(encoded))
            self.assertEqual(
                envelope["payload_sha256"],
                script.sha256_bytes(
                    script._canonical_json_bytes(envelope["payload"])
                ),
            )
            self.assertEqual(
                envelope["payload"]["content_sha256"], cached.source_hash
            )

            envelope["payload"]["content_type"] = "text/html"
            path.write_bytes(
                gzip.compress(script._canonical_json_bytes(envelope), mtime=0)
            )
            with self.assertRaisesRegex(RuntimeError, "payload hash mismatch"):
                cache.get("LILA")

            path.write_bytes(encoded)
            envelope = json.loads(gzip.decompress(encoded))
            envelope["payload"]["content_base64"] = base64.b64encode(
                b"different raw bytes"
            ).decode("ascii")
            envelope["payload_sha256"] = script.sha256_bytes(
                script._canonical_json_bytes(envelope["payload"])
            )
            path.write_bytes(
                gzip.compress(script._canonical_json_bytes(envelope), mtime=0)
            )
            with self.assertRaisesRegex(RuntimeError, "content hash mismatch"):
                cache.get("LILA")

    def test_existing_cache_entry_is_never_overwritten_by_changed_response(self):
        first_content = _yahoo_content("LILA")
        changed = _yahoo_payload("LILA")
        changed["chart"]["result"][0]["indicators"]["quote"][0]["close"][0] += 1
        second_content = _yahoo_content("LILA", changed)
        with tempfile.TemporaryDirectory() as raw:
            cache = script.YahooChartSupplementCache(Path(raw))
            with patch.object(
                script,
                "urlopen",
                return_value=_YahooHttpResponse(first_content),
            ):
                cache.fetch("LILA")
            path = cache.path("LILA")
            original = path.read_bytes()
            with patch.object(
                script,
                "urlopen",
                return_value=_YahooHttpResponse(second_content),
            ):
                with self.assertRaisesRegex(RuntimeError, "changed for one request URL"):
                    cache.fetch("LILA")
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(cache.get("LILA").content, first_content)

    def test_single_attempt_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as raw:
            cache = script.YahooChartSupplementCache(Path(raw))
            with patch.object(
                script,
                "urlopen",
                side_effect=script.URLError("offline"),
            ) as request:
                with self.assertRaisesRegex(RuntimeError, "single HTTP attempt failed"):
                    cache.fetch("LILA")
            request.assert_called_once()
            self.assertEqual(cache.http_attempts, 1)

    def test_run_wide_http_cap_is_exactly_two(self):
        contents = {
            symbol: _yahoo_content(symbol)
            for symbol in script.YAHOO_SUPPLEMENT_SYMBOLS
        }

        def respond(request, *, timeout):
            del timeout
            symbol = request.full_url.split("/chart/", 1)[1].split("?", 1)[0]
            return _YahooHttpResponse(contents[symbol])

        with tempfile.TemporaryDirectory() as raw:
            cache = script.YahooChartSupplementCache(Path(raw))
            with patch.object(script, "urlopen", side_effect=respond) as request:
                cache.fill_missing(script.YAHOO_SUPPLEMENT_SYMBOLS)
                with self.assertRaisesRegex(RuntimeError, "attempt cap reached"):
                    cache.fetch("LILA")
            self.assertEqual(request.call_count, 2)
            self.assertEqual(cache.http_attempts, 2)

    def test_parser_rejects_html_non_json_api_identity_and_bar_errors(self):
        invalid: list[tuple[str, script.YahooChartCachedResponse, str]] = [
            (
                "html",
                _yahoo_cached_response(
                    "LILA", content=b"<html>verification</html>", content_type="text/html"
                ),
                "HTML or non-JSON",
            ),
            (
                "non-json",
                _yahoo_cached_response("LILA", content=b"{not json"),
                "not valid JSON",
            ),
            (
                "http",
                _yahoo_cached_response("LILA", http_status=429),
                "HTTP 429",
            ),
        ]
        api_error = _yahoo_payload("LILA")
        api_error["chart"]["error"] = {"code": "Not Found"}
        invalid.append(
            ("api", _yahoo_cached_response("LILA", payload=api_error), "API error")
        )
        wrong_symbol = _yahoo_payload("LILA")
        wrong_symbol["chart"]["result"][0]["meta"]["symbol"] = "AAL"
        invalid.append(
            (
                "symbol",
                _yahoo_cached_response("LILA", payload=wrong_symbol),
                "symbol mismatch",
            )
        )
        wrong_currency = _yahoo_payload("LILA")
        wrong_currency["chart"]["result"][0]["meta"]["currency"] = "EUR"
        invalid.append(
            (
                "currency",
                _yahoo_cached_response("LILA", payload=wrong_currency),
                "currency must be USD",
            )
        )
        bad_timestamp = _yahoo_payload("LILA")
        bad_timestamp["chart"]["result"][0]["timestamp"][0] = "bad"
        invalid.append(
            (
                "timestamp",
                _yahoo_cached_response("LILA", payload=bad_timestamp),
                "timestamps are invalid",
            )
        )
        bad_ohlcv = _yahoo_payload("LILA")
        bad_ohlcv["chart"]["result"][0]["indicators"]["quote"][0]["close"][0] = None
        invalid.append(
            (
                "ohlcv",
                _yahoo_cached_response("LILA", payload=bad_ohlcv),
                "invalid OHLCV",
            )
        )
        wrong_length = _yahoo_payload("LILA")
        wrong_length["chart"]["result"][0]["indicators"]["quote"][0]["volume"].pop()
        invalid.append(
            (
                "one-to-one",
                _yahoo_cached_response("LILA", payload=wrong_length),
                "not one-to-one with timestamps",
            )
        )
        for label, response, message in invalid:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    script._parse_yahoo_chart_response(response)

    def test_identity_parser_rejects_non_equity_yahoo_metadata(self):
        wrong = _yahoo_payload("LILA")
        wrong["chart"]["result"][0]["meta"]["instrumentType"] = "MUTUALFUND"
        with self.assertRaisesRegex(ValueError, "instrument type must be EQUITY"):
            script._parse_yahoo_chart_response(
                _yahoo_cached_response("LILA", payload=wrong)
            )

    def test_exact_exchange_sessions_and_lilak_regular_way_slice_are_enforced(self):
        lila = script._yahoo_segment(
            _yahoo_cached_response("LILA"), security_id="OLD-LILA"
        )
        lilak_response = _yahoo_cached_response("LILAK")
        raw_lilak = script._parse_yahoo_chart_response(lilak_response)
        lilak = script._yahoo_segment(lilak_response, security_id="OLD-LILAK")
        self.assertEqual(len(lila), 630)
        self.assertEqual(len(raw_lilak), 637)
        self.assertEqual(len(lilak), 630)
        self.assertEqual(lilak.iloc[0]["session"], "2015-07-02")
        self.assertEqual(lilak.iloc[-1]["session"], "2017-12-29")

        missing = _yahoo_payload("LILA")
        result = missing["chart"]["result"][0]
        result["timestamp"].pop(10)
        quote = result["indicators"]["quote"][0]
        for column in ("open", "high", "low", "close", "volume"):
            quote[column].pop(10)
        with self.assertRaisesRegex(ValueError, "exact exchange-session coverage"):
            script._yahoo_segment(
                _yahoo_cached_response("LILA", payload=missing),
                security_id="OLD-LILA",
            )

    def test_external_supplement_bundle_escape_hatch_is_removed(self):
        with self.assertRaises(SystemExit):
            script._parse_args(["--supplement-bundle", "unreviewed.json"])


class WikiArncPinnedSourceTests(unittest.TestCase):
    def test_network_is_fail_closed_without_explicit_flag(self):
        with tempfile.TemporaryDirectory() as raw:
            source = script.WikiArncPinnedSource(Path(raw), allow_http=False)
            with patch.object(script, "urlopen") as request:
                with self.assertRaisesRegex(
                    FileNotFoundError, "--fetch-aa-wiki-crosscheck"
                ):
                    source.fetch(_ids())
            request.assert_not_called()
            self.assertEqual(source.http_attempts, 0)

    def test_one_full_blob_is_cached_parsed_and_reused_offline(self):
        content, segment = _wiki_fixture()
        pins = _wiki_pins(content, segment)
        with patch.multiple(script, **pins):
            with tempfile.TemporaryDirectory() as raw:
                source = script.WikiArncPinnedSource(Path(raw), allow_http=True)
                with patch.object(
                    script,
                    "urlopen",
                    return_value=_YahooHttpResponse(content, "application/octet-stream"),
                ) as request:
                    fetched = source.fetch(_ids())
                request.assert_called_once()
                self.assertEqual(source.http_attempts, 1)
                self.assertEqual(len(fetched.prices), 462)
                self.assertEqual(set(fetched.prices["source"]), {script.WIKI_ARNC_SOURCE})
                self.assertEqual(len(fetched.artifacts), 1)
                self.assertEqual(fetched.artifacts[0].content, content)
                report = script.validate_wiki_arnc_bundle(fetched, _ids())
                self.assertEqual(report["segment_rows"], 462)
                self.assertTrue(report["full_blob_archived_on_apply"])

                cached = script.WikiArncPinnedSource(Path(raw), allow_http=False)
                with patch.object(script, "urlopen") as second_request:
                    second = cached.fetch(_ids())
                second_request.assert_not_called()
                self.assertEqual(cached.http_attempts, 0)
                self.assertEqual(second.artifacts[0].content, content)

                archive = script.append_source_archive(
                    _empty("source_archive"),
                    fetched.artifacts,
                    completed_session="2026-07-15",
                )
                self.assertEqual(len(archive), 1)
                self.assertEqual(archive.iloc[0]["source_url"], script.WIKI_ARNC_URL)
                self.assertEqual(
                    archive.iloc[0]["source_hash"], script.sha256_bytes(content)
                )
                repository = SimpleNamespace(root=Path(raw) / "archive-store")
                script._persist_archive_payloads(
                    repository, fetched.artifacts, "2026-07-15"
                )
                archived_path = (
                    repository.root
                    / "archives/2026-07-15"
                    / f"{script.sha256_bytes(content)}.csv.gz"
                )
                self.assertEqual(gzip.decompress(archived_path.read_bytes()), content)

    def test_full_hash_row_count_segment_hash_and_actions_are_fail_closed(self):
        content, segment = _wiki_fixture()
        pins = _wiki_pins(content, segment)
        response = script.WikiArncCachedResponse(
            source_url=script.WIKI_ARNC_URL,
            retrieved_at="2026-07-18T00:00:00Z",
            content=content,
            content_type="text/csv",
            http_status=200,
        )
        with patch.multiple(script, **pins):
            frame, report = script._parse_wiki_arnc_response(
                response, security_id="OLD-AA"
            )
            self.assertEqual(len(frame), 462)
            self.assertEqual(report["split_session"], "2016-10-06")

            wrong_rows = {**pins, "WIKI_ARNC_FULL_DATA_ROWS": pins["WIKI_ARNC_FULL_DATA_ROWS"] + 1}
            with patch.multiple(script, **wrong_rows):
                with self.assertRaisesRegex(ValueError, "row count changed"):
                    script._parse_wiki_arnc_response(response, security_id="OLD-AA")

            changed = content.replace(b"ARNC,2016-10-06,", b"ARNC,2016-10-06,", 1)
            bad_segment = {**pins, "WIKI_ARNC_SEGMENT_SHA256": "0" * 64}
            with patch.multiple(script, **bad_segment):
                with self.assertRaisesRegex(ValueError, "segment hash changed"):
                    script._parse_wiki_arnc_response(
                        script.WikiArncCachedResponse(
                            **{**response.__dict__, "content": changed}
                        ),
                        security_id="OLD-AA",
                    )

    def test_merge_replaces_eodhd_publication_but_retains_raw_crosscheck(self):
        ids = _ids()
        content, segment = _wiki_fixture()
        pins = _wiki_pins(content, segment)
        with patch.multiple(script, **pins):
            response = script.WikiArncCachedResponse(
                source_url=script.WIKI_ARNC_URL,
                retrieved_at="2026-07-18T00:00:00Z",
                content=content,
                content_type="text/csv",
                http_status=200,
            )
            wiki_prices, _ = script._parse_wiki_arnc_response(
                response, security_id=ids.hwm
            )
            wiki = script.FetchedHistories(
                prices=wiki_prices,
                crosscheck_prices=_empty("daily_price_raw"),
                corporate_actions=_empty("corporate_actions"),
                artifacts=(script._wiki_arnc_artifact(response),),
                role_codes={
                    script.WIKI_ARNC_ROLE: f"WIKI/PRICES:ARNC@{script.WIKI_ARNC_COMMIT}"
                },
                http_attempts=0,
            )
            eodhd_artifact = _artifact(
                source="eodhd_eod",
                source_url=(
                    "https://eodhd.com/api/eod/AA.US"
                    "?from=2015-01-01&to=2016-10-31"
                ),
                content=b"raw eodhd aa",
            )
            eodhd_rows = []
            for row in wiki_prices.itertuples(index=False):
                value = _price(ids.hwm, str(row.session), float(row.close), source="eodhd_eod")
                value.update(
                    {
                        "open": row.open,
                        "high": row.high,
                        "low": row.low,
                        "volume": row.volume,
                        "source_url": eodhd_artifact.source_url,
                        "source_hash": eodhd_artifact.source_hash,
                    }
                )
                eodhd_rows.append(value)
            primary = script.FetchedHistories(
                prices=pd.DataFrame(eodhd_rows),
                crosscheck_prices=_empty("daily_price_raw"),
                corporate_actions=_empty("corporate_actions"),
                artifacts=(eodhd_artifact,),
                role_codes={"old_aa": "AA"},
                http_attempts=script.MAX_EODHD_HTTP_ATTEMPTS,
            )
            merged = script.merge_wiki_arnc_primary(primary, wiki, ids=ids)
        self.assertEqual(set(merged.prices["source"]), {script.WIKI_ARNC_SOURCE})
        self.assertEqual(set(merged.crosscheck_prices["source"]), {"eodhd_eod"})
        self.assertEqual(len(merged.prices), 462)
        self.assertEqual(len(merged.crosscheck_prices), 462)


class BorisKaggleCrosscheckTests(unittest.TestCase):
    @staticmethod
    def _contents() -> dict[str, bytes]:
        output: dict[str, bytes] = {}
        for symbol, prefix in {
            "LILA": ("2015-06-22", "2015-07-01"),
            "LILAK": ("2015-06-23", "2015-07-01"),
        }.items():
            spec = script.BORIS_KAGGLE_FILES[symbol]
            regular = script._expected_sessions(
                str(spec["segment_start"]), str(spec["segment_end"])
            )
            lines = ["Date,Open,High,Low,Close,Volume,OpenInt"]
            for number, session in enumerate((*prefix, *regular)):
                close = 20.0 + number / 100.0
                lines.append(
                    f"{session},{close:.2f},{close + 0.2:.2f},"
                    f"{close - 0.2:.2f},{close + 0.1:.2f},{1000 + number},0"
                )
            output[symbol] = ("\r\n".join(lines) + "\r\n").encode()
        return output

    @staticmethod
    def _patched_files(contents: dict[str, bytes]) -> dict[str, dict[str, object]]:
        return {
            symbol: {
                **script.BORIS_KAGGLE_FILES[symbol],
                "sha256": script.sha256_bytes(content),
                "raw_rows": len(content.splitlines()) - 1,
            }
            for symbol, content in contents.items()
        }

    def test_real_version_three_urls_and_hashes_are_frozen(self):
        self.assertNotIn("AA", script.BORIS_KAGGLE_FILES)
        self.assertEqual(
            script.BORIS_KAGGLE_FILES["LILA"]["sha256"],
            "9885111c20ca809ce8791c429cd8eb66a62470b53ab71f7c2ac6a573d576f73c",
        )
        self.assertEqual(
            script.BORIS_KAGGLE_FILES["LILAK"]["sha256"],
            "b5a56cc0c1b5a478354d85149c2370ccde6146f7f43d94566dcc76382db610e4",
        )
        self.assertTrue(
            all(
                "datasetVersionNumber=3" in spec["url"]
                for spec in script.BORIS_KAGGLE_FILES.values()
            )
        )
        self.assertEqual(script.BORIS_KAGGLE_LICENSE, "CC0: Public Domain")

    def test_two_pinned_files_parse_to_exact_597_session_crosschecks(self):
        contents = self._contents()
        files = self._patched_files(contents)

        def respond(request, *, timeout):
            self.assertEqual(timeout, 30.0)
            if "lilak.us.txt" in request.full_url:
                symbol = "LILAK"
            else:
                symbol = "LILA"
            return _YahooHttpResponse(contents[symbol], "text/plain")

        with patch.object(script, "BORIS_KAGGLE_FILES", files):
            with tempfile.TemporaryDirectory() as raw:
                source = script.BorisKaggleCrosscheckSource(
                    Path(raw), allow_http=True
                )
                with patch.object(script, "urlopen", side_effect=respond) as request:
                    fetched = source.fetch(_ids())
                self.assertEqual(request.call_count, 2)
                self.assertEqual(source.http_attempts, 2)
                report = script.validate_boris_crosscheck_bundle(fetched, _ids())
                self.assertEqual(len(fetched.prices), 0)
                self.assertEqual(len(fetched.crosscheck_prices), 1_194)
                self.assertEqual(report["lila"]["sessions"], 597)
                self.assertEqual(report["lilak"]["sessions"], 597)

                cached = script.BorisKaggleCrosscheckSource(
                    Path(raw), allow_http=False
                )
                with patch.object(script, "urlopen") as second_request:
                    second = cached.fetch(_ids())
                second_request.assert_not_called()
                self.assertEqual(cached.http_attempts, 0)
                self.assertEqual(len(second.crosscheck_prices), 1_194)

    def test_hash_schema_and_missing_session_are_fail_closed(self):
        contents = self._contents()
        files = self._patched_files(contents)
        response = script.BorisKaggleCachedResponse(
            symbol="LILA",
            source_url=files["LILA"]["url"],
            retrieved_at="2026-07-18T00:00:00Z",
            content=contents["LILA"],
            content_type="text/plain",
            http_status=200,
        )
        with patch.object(script, "BORIS_KAGGLE_FILES", files):
            frame = script._parse_boris_kaggle_response(
                response, security_id="OLD-LILA"
            )
            self.assertEqual(len(frame), 597)

            missing = contents["LILA"].splitlines()
            missing.pop(10)
            changed = b"\r\n".join(missing) + b"\r\n"
            changed_files = {
                **files,
                "LILA": {
                    **files["LILA"],
                    "sha256": script.sha256_bytes(changed),
                },
            }
            bad = script.BorisKaggleCachedResponse(
                **{**response.__dict__, "content": changed}
            )
            with patch.object(script, "BORIS_KAGGLE_FILES", changed_files):
                with self.assertRaisesRegex(ValueError, "raw row count changed"):
                    script._parse_boris_kaggle_response(
                        bad, security_id="OLD-LILA"
                    )


class AznExactDuplicateTests(unittest.TestCase):
    def _duplicate_prices(self) -> pd.DataFrame:
        ids = _ids()
        sessions = pd.bdate_range("2015-01-02", periods=2_701)
        rows = []
        for number, session in enumerate(sessions, start=1):
            date = session.date().isoformat()
            rows.append(_price(ids.azn, date, float(number), volume=100.0))
            rows.append(
                _price(ids.azn_duplicate, date, float(number), volume=200.0)
            )
        return pd.DataFrame(rows)

    def test_exact_ohlc_and_distinct_volume_proves_duplicate(self):
        report = script.validate_azn_exact_duplicate(
            self._duplicate_prices(), _ids()
        )
        self.assertEqual(report["overlap_sessions"], 2_701)
        self.assertEqual(report["ohlc_equal_sessions"], 2_701)
        self.assertEqual(report["volume_equal_sessions"], 0)

    def test_any_ohlc_divergence_blocks_deduplication(self):
        prices = self._duplicate_prices()
        ids = _ids()
        row = prices.index[
            prices["security_id"].astype(str).eq(ids.azn_duplicate)
        ][-1]
        prices.loc[row, "close"] += 0.01
        with self.assertRaisesRegex(ValueError, "exact AZN close"):
            script.validate_azn_exact_duplicate(prices, ids)

    def test_price_action_rewrite_keeps_canonical_and_drops_old_id(self):
        ids = _ids()
        prices = pd.DataFrame(
            [
                _price(ids.azn, "2025-01-02", 10.0, volume=100),
                _price(ids.azn_duplicate, "2025-01-02", 10.0, volume=200),
            ]
        )
        actions = pd.DataFrame(
            [_action(ids.azn, "azn-action"), _action(ids.azn_duplicate, "old-action")]
        )
        fetched = script.FetchedHistories(
            prices=_empty("daily_price_raw"),
            crosscheck_prices=_empty("daily_price_raw"),
            corporate_actions=_empty("corporate_actions"),
            artifacts=(),
            role_codes={},
            http_attempts=32,
        )
        rewritten_prices, rewritten_actions, affected, stats = (
            script.rewrite_prices_and_actions(
                prices,
                actions,
                fetched,
                ids=ids,
                completed_session="2026-07-15",
                evidence=_artifact(),
            )
        )
        self.assertEqual(
            set(rewritten_prices["security_id"].astype(str)) & {ids.azn, ids.azn_duplicate},
            {ids.azn},
        )
        self.assertNotIn(
            ids.azn_duplicate, set(rewritten_actions["security_id"].astype(str))
        )
        self.assertTrue({ids.azn, ids.azn_duplicate}.issubset(affected))
        self.assertEqual(stats["azn_duplicate_price_rows_removed"], 1)
        self.assertEqual(stats["azn_duplicate_action_rows_removed"], 1)

    def test_index_events_move_to_canonical_id_without_relabeling_source(self):
        ids = _ids()
        anchor_rows = []
        for number in range(498):
            anchor_rows.append(
                {
                    "index_id": "sp500",
                    "anchor_date": "2015-01-02",
                    "security_id": f"DUMMY:{number}",
                    "official": False,
                    "source": "fixture",
                    "source_url": "https://community.test/sp500",
                    "source_kind": "community",
                    "retrieved_at": "2026-07-18T00:00:00Z",
                    "source_hash": "e" * 64,
                }
            )
        anchor_rows.append(
            {
                **anchor_rows[0],
                "security_id": ids.agn_legacy,
            }
        )
        anchors = pd.DataFrame(anchor_rows)
        events = pd.DataFrame(
            [
                _index_event(ids.azn_duplicate, "2022-02-22", "ADD"),
                _index_event(ids.azn_duplicate, "2026-01-20", "REMOVE"),
            ]
        )
        output_anchors, output_events, stats = script.rewrite_index_references(
            anchors,
            events,
            ids=ids,
            evidence=_artifact(),
        )
        azn_events = output_events.loc[
            output_events["security_id"].astype(str).eq(ids.azn)
        ]
        self.assertEqual(len(azn_events), 2)
        self.assertEqual(set(azn_events["source"]), {"community_nasdaq100_history"})
        self.assertEqual(set(azn_events["source_hash"]), {"d" * 64})
        self.assertFalse(
            output_events["security_id"].astype(str).eq(ids.azn_duplicate).any()
        )
        self.assertEqual(stats["azn_events_identity_remapped"], 2)
        self.assertEqual(len(output_anchors), 500)


class ValarisDeferredOutcomeTests(unittest.TestCase):
    def _fetched_valpq(self) -> script.FetchedHistories:
        ids = _ids()
        start = "2019-07-31"
        end = "2021-04-27"
        artifact = _artifact(
            source="eodhd_eod",
            source_url=(
                "https://eodhd.com/api/eod/VALPQ.US"
                "?from=2019-07-31&to=2021-04-27"
            ),
            content=b"VALPQ exact raw payload",
        )
        rows = []
        for number, session in enumerate(script._valaris_expected_sessions()):
            row = _price(
                ids.esv,
                session,
                10.0 + number / 100.0,
                source="eodhd_eod",
            )
            row["source_url"] = artifact.source_url
            row["source_hash"] = artifact.source_hash
            rows.append(row)
        return script.FetchedHistories(
            prices=pd.DataFrame(rows),
            crosscheck_prices=_empty("daily_price_raw"),
            corporate_actions=_empty("corporate_actions"),
            artifacts=(artifact,),
            role_codes={"valaris": "VALPQ"},
            http_attempts=script.MAX_EODHD_HTTP_ATTEMPTS,
        )

    def test_valpq_requires_exact_price_window_and_raw_endpoint_provenance(self):
        fetched = self._fetched_valpq()

        report = script.validate_valaris_fetched_history(fetched, _ids())

        self.assertEqual(report["provider_code"], "VALPQ")
        self.assertEqual(report["start"], "2019-07-31")
        self.assertEqual(report["end"], "2021-04-27")
        self.assertEqual(
            report["session_count"],
            len(script._valaris_expected_sessions()),
        )
        self.assertEqual(
            report["documented_halt_sessions"],
            sorted(script.VALARIS_DOCUMENTED_HALT_SESSIONS),
        )

        missing = self._fetched_valpq()
        missing = script.FetchedHistories(
            **{**missing.__dict__, "prices": missing.prices.iloc[:-1].copy()}
        )
        with self.assertRaisesRegex(ValueError, "exact 2019-07-31..2021-04-27"):
            script.validate_valaris_fetched_history(missing, _ids())

        wrong_code = self._fetched_valpq()
        wrong_code.role_codes["valaris"] = "VAL_old"
        with self.assertRaisesRegex(ValueError, "must use VALPQ"):
            script.validate_valaris_fetched_history(wrong_code, _ids())

    def test_bundle_roundtrip_preserves_price_source_url(self):
        fetched = self._fetched_valpq()
        value = script._bundle_value(fetched, signature={"fixture": True})

        restored = script._fetched_from_value(value)

        self.assertIn("source_url", restored.prices.columns)
        self.assertEqual(
            set(restored.prices["source_url"]),
            {fetched.artifacts[0].source_url},
        )

    def test_official_2021_cancellation_boundary_is_exact_and_archived(self):
        ids = _ids()
        source_url = script.OFFICIAL_EVIDENCE["valaris_emergence"]["url"]
        source_hash = "f" * 64
        sessions = script._valaris_expected_sessions()
        prices = pd.DataFrame(
            [_price(ids.esv, session, 10.0) for session in sessions]
        )
        frames = {
            "security_master": pd.DataFrame(
                [
                    {
                        **_master(ids.esv, "VALPQ.US"),
                        "primary_symbol": "VAL",
                        "active_to": "2021-04-30",
                        "source": "official_identity_repair",
                        "source_url": source_url,
                        "source_hash": source_hash,
                    }
                ]
            ),
            "symbol_history": pd.DataFrame(
                [
                    {
                        **_history(ids.esv, "VAL"),
                        "effective_from": "2019-07-31",
                        "effective_to": "2021-04-30",
                        "source": "official_identity_repair",
                        "source_url": source_url,
                        "source_hash": source_hash,
                    }
                ]
            ),
            "daily_price_raw": prices,
            "source_archive": pd.DataFrame(
                [
                    {
                        "archive_id": source_hash,
                        "source_url": source_url,
                        "source_hash": source_hash,
                    }
                ]
            ),
        }

        report = script.validate_valaris_cancellation_boundary(frames, ids=_ids())

        self.assertEqual(report["last_price_session"], "2021-04-27")
        self.assertEqual(report["official_cancellation_date"], "2021-04-30")
        self.assertEqual(report["official_evidence_sha256"], source_hash)

        frames["source_archive"].loc[0, "source_url"] = (
            "https://www.sec.gov/Archives/different.htm"
        )
        with self.assertRaisesRegex(ValueError, "exact official cancellation URL/hash"):
            script.validate_valaris_cancellation_boundary(frames, ids=_ids())

    def test_2019_ticker_continuity_remains_but_zero_cash_2021_action_is_not_invented(self):
        ids = _ids()
        fetched = script.FetchedHistories(
            prices=_empty("daily_price_raw"),
            crosscheck_prices=_empty("daily_price_raw"),
            corporate_actions=_empty("corporate_actions"),
            artifacts=(),
            role_codes={},
            http_attempts=32,
        )
        _prices, actions, _affected, stats = script.rewrite_prices_and_actions(
            _empty("daily_price_raw"),
            _empty("corporate_actions"),
            fetched,
            ids=ids,
            completed_session="2026-07-15",
            evidence=_artifact(),
        )
        valaris = actions.loc[actions["security_id"].astype(str).eq(ids.esv)]
        ticker_change = valaris.loc[
            valaris["effective_date"].astype(str).eq("2019-07-31")
        ]
        self.assertEqual(len(ticker_change), 1)
        self.assertEqual(ticker_change.iloc[0]["action_type"], "ticker_change")
        self.assertEqual(ticker_change.iloc[0]["new_security_id"], ids.esv)
        self.assertEqual(ticker_change.iloc[0]["new_symbol"], "VAL")
        self.assertFalse(
            valaris["effective_date"].astype(str).eq("2021-04-30").any()
        )
        self.assertEqual(
            stats["legacy_valaris_2021_outcome"],
            script.VALARIS_2021_OUTCOME_STATUS,
        )
        self.assertIn("unsupported_consideration", script.VALARIS_2021_OUTCOME_STATUS)


class CrossSourceAndGateTests(unittest.TestCase):
    def _aa_frames(self):
        ids = _ids()
        sessions = script._expected_sessions("2015-01-02", "2016-10-31")
        primary = pd.DataFrame(
            [
                _price(
                    ids.hwm,
                    date,
                    20.0 + number / 10,
                    volume=10_000 + number,
                    source=script.WIKI_ARNC_SOURCE,
                )
                for number, date in enumerate(sessions)
            ]
        )
        external = pd.DataFrame(
            [
                _price(
                    ids.hwm,
                    date,
                    20.0 + number / 10,
                    volume=(10_000 + number)
                    * (0.41618 if date < "2016-10-06" else 1.24854),
                    source="eodhd_eod",
                )
                for number, date in enumerate(sessions)
            ]
        )
        return ids, primary, external

    def test_aa_crosscheck_accepts_raw_ohlc_and_reports_volume_basis_mismatch(self):
        ids, primary, eodhd = self._aa_frames()
        report = script._validate_aa_raw_ohlc_crosscheck(
            primary, eodhd, security_id=ids.hwm
        )
        self.assertEqual(report["sessions"], 462)
        self.assertFalse(report["adjusted_close_used"])
        self.assertGreaterEqual(report["ohlc"]["close"]["return_correlation"], 0.99999)
        self.assertTrue(report["volume_mismatch"]["basis_mismatch_confirmed"])
        self.assertFalse(report["volume_mismatch"]["used_for_publication"])
        self.assertAlmostEqual(
            report["volume_mismatch"]["post_to_pre_basis_jump"], 3.0, places=4
        )
        self.assertFalse(hasattr(script, "_aa_adjusted_close_validation_prices"))

    def test_aa_crosscheck_rejects_missing_or_wrong_external_evidence(self):
        ids, primary, eodhd = self._aa_frames()
        with self.assertRaisesRegex(ValueError, "exact one-to-one sessions"):
            script._validate_aa_raw_ohlc_crosscheck(
                primary, eodhd.iloc[1:].copy(), security_id=ids.hwm
            )
        extra = pd.concat(
            [
                eodhd,
                pd.DataFrame(
                    [
                        _price(
                            ids.hwm,
                            "2015-01-03",
                            20.1,
                            source="eodhd_eod",
                        )
                    ]
                ),
            ],
            ignore_index=True,
        ).sort_values("session")
        with self.assertRaisesRegex(ValueError, "exact one-to-one sessions"):
            script._validate_aa_raw_ohlc_crosscheck(
                primary, extra, security_id=ids.hwm
            )
        eodhd["source"] = script.BORIS_KAGGLE_SOURCE
        with self.assertRaisesRegex(ValueError, "retain EODHD provenance"):
            script._validate_aa_raw_ohlc_crosscheck(
                primary, eodhd, security_id=ids.hwm
            )

    def test_old_alcoa_official_actions_keep_arnc_same_and_new_aa_separate(self):
        ids = _ids()
        fetched = script.FetchedHistories(
            prices=_empty("daily_price_raw"),
            crosscheck_prices=_empty("daily_price_raw"),
            corporate_actions=_empty("corporate_actions"),
            artifacts=(),
            role_codes={},
            http_attempts=script.MAX_EODHD_HTTP_ATTEMPTS,
        )
        _prices, actions, _affected, _stats = script.rewrite_prices_and_actions(
            _empty("daily_price_raw"),
            _empty("corporate_actions"),
            fetched,
            ids=ids,
            completed_session="2026-07-15",
            evidence=_artifact(),
        )
        hwm = actions.loc[actions["security_id"].astype(str).eq(ids.hwm)]
        split = hwm.loc[
            hwm["action_type"].astype(str).eq("split")
            & hwm["effective_date"].astype(str).eq("2016-10-06")
        ]
        ticker = hwm.loc[
            hwm["action_type"].astype(str).eq("ticker_change")
            & hwm["effective_date"].astype(str).eq("2016-11-01")
        ]
        new_aa = hwm.loc[
            hwm["action_type"].astype(str).eq("spinoff")
            & hwm["effective_date"].astype(str).eq("2016-11-01")
        ]
        self.assertEqual(len(split), 1)
        self.assertAlmostEqual(float(split.iloc[0]["ratio"]), 1 / 3)
        self.assertEqual(ticker.iloc[0]["new_security_id"], ids.hwm)
        self.assertEqual(ticker.iloc[0]["new_symbol"], "ARNC")
        self.assertEqual(new_aa.iloc[0]["new_security_id"], "")
        self.assertEqual(new_aa.iloc[0]["new_symbol"], "AA")
        self.assertEqual(
            new_aa.iloc[0]["source_url"],
            script.OFFICIAL_EVIDENCE["alcoa_2016_separation"]["url"],
        )

    def test_boundary_rows_require_the_exact_archived_url_hash_pair(self):
        repaired = pd.DataFrame(
            [
                {
                    "source": "official_identity_repair",
                    "source_url": "https://official.test/document",
                    "source_hash": "f" * 64,
                }
            ]
        )
        frames = {
            dataset: repaired.copy()
            for dataset in (
                "security_master",
                "symbol_history",
                "corporate_actions",
                "index_constituent_anchors",
                "index_membership_events",
            )
        }
        frames["source_archive"] = pd.DataFrame(
            [{"source_url": "https://official.test/document", "source_hash": "0" * 64}]
        )
        with self.assertRaisesRegex(ValueError, "exact raw URL/hash"):
            script.validate_boundary_provenance(frames)
        frames["source_archive"].loc[0, "source_hash"] = "f" * 64
        script.validate_boundary_provenance(frames)

    def test_successor_warning_and_complete_cov_are_hard_preconditions(self):
        ids = _ids()
        frames = {dataset: _empty(dataset) for dataset in script.WRITE_DATASETS}
        release = DataRelease(
            version="base",
            created_at="2026-07-18T00:00:00Z",
            completed_session="2026-07-15",
            dataset_versions={dataset: "v1" for dataset in script.WRITE_DATASETS},
            warnings=(),
        )
        repository = SimpleNamespace()
        common = (
            patch.object(script, "_read_release_frames", return_value=frames),
            patch.object(script, "_resolve_identity_ids", return_value=ids),
            patch.object(script, "_assert_local_shape"),
        )
        with common[0], common[1], common[2]:
            with self.assertRaisesRegex(ValueError, "run after the successor"):
                script.build_local_preflight(
                    repository, release, require_successor_snapshot=True
                )

        release = DataRelease(
            **{
                **release.__dict__,
                "warnings": (script.PENDING_IDENTITY_WARNING,),
            }
        )
        with (
            patch.object(script, "_read_release_frames", return_value=frames),
            patch.object(script, "_resolve_identity_ids", return_value=ids),
            patch.object(script, "_assert_local_shape"),
            patch.object(script, "_coverage_missing", return_value=("2015-01-05",)),
        ):
            with self.assertRaisesRegex(ValueError, "incomplete COV history"):
                script.build_local_preflight(
                    repository, release, require_successor_snapshot=True
                )

    def test_full_history_gate_fails_on_one_missing_repaired_session(self):
        ids = _ids()
        frames = {"daily_price_raw": _empty("daily_price_raw")}

        def missing(_prices, security_id, _start, _end, **_kwargs):
            return ("2016-09-21",) if security_id == ids.hot else ()

        with patch.object(script, "_coverage_missing", side_effect=missing):
            with self.assertRaisesRegex(ValueError, "Full-history gate failed for hot"):
                script.validate_full_history_gate(
                    frames,
                    completed_session="2026-07-15",
                    ids=ids,
                )

    def test_replay_gate_requires_warning_free_active_membership_and_continuity(self):
        ids = _ids()
        anchors = pd.DataFrame(
            [
                {"index_id": "sp500", "anchor_date": "2015-01-02"},
                {"index_id": "nasdaq100", "anchor_date": "2015-01-02"},
            ]
        )
        events = pd.DataFrame(
            columns=dataset_spec("index_membership_events").required_columns
        )
        history = pd.DataFrame(
            [
                {
                    **_history(security_id),
                    "effective_from": "2015-01-01",
                    "effective_to": "",
                }
                for security_id in (ids.cor, ids.bkr, ids.hwm)
            ]
        )
        frames = {
            "index_constituent_anchors": anchors,
            "index_membership_events": events,
            "symbol_history": history,
        }

        class Replayer:
            warnings = ()

            @staticmethod
            def members_on(_index, _date):
                return SimpleNamespace(
                    warnings=Replayer.warnings,
                    security_ids=(ids.cor, ids.bkr, ids.hwm),
                )

        with patch.object(script, "IndexEventReplayer", return_value=Replayer()):
            report = script.validate_replay_gate(
                frames, completed_session="2026-07-15", ids=ids
            )
        self.assertEqual(
            report["canonical_position_continuity"], ["bkr", "cor", "hwm"]
        )

        Replayer.warnings = ("duplicate remove",)
        with patch.object(script, "IndexEventReplayer", return_value=Replayer()):
            with self.assertRaisesRegex(ValueError, "Index replay warnings remain"):
                script.validate_replay_gate(
                    frames, completed_session="2026-07-15", ids=ids
                )

class _TransactionRepository:
    def __init__(self, root: Path, *, fail_dataset: str = ""):
        self.root = root
        self.objects = LocalObjectStore(root)
        self.fail_dataset = fail_dataset
        versions = {}
        for dataset in script.WRITE_DATASETS:
            version = f"old-{dataset}"
            versions[dataset] = version
            manifest = DatasetManifest.create(dataset, version, "2026-07-15", ())
            manifest_path = f"datasets/{dataset}/versions/{version}/manifest.json"
            self.objects.put(manifest_path, manifest.to_bytes())
            self.objects.put(
                self.current_key(dataset),
                CurrentPointer.create(manifest, manifest_path).to_bytes(),
            )
        self.release = DataRelease(
            version="base-release",
            created_at="2026-07-18T00:00:00Z",
            completed_session="2026-07-15",
            dataset_versions=versions,
            warnings=(script.PENDING_IDENTITY_WARNING, "keep-me"),
        )
        self.objects.put(
            f"releases/{self.release.version}.json", self.release.to_bytes()
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
        self.objects.put(
            self.current_key(dataset),
            CurrentPointer.create(manifest, manifest_path).to_bytes(),
            if_match=expected_pointer_etag,
        )
        if dataset == self.fail_dataset:
            raise KeyboardInterrupt(f"injected failure after {dataset}")
        return SimpleNamespace(manifest=manifest, conflict=False, conflict_path="")

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
            f"releases/{release.version}.json", release.to_bytes(), if_none_match=True
        )
        self.objects.put(
            "releases/current.json", release.to_bytes(), if_match=expected_etag
        )
        return release


def _prepared(repository: _TransactionRepository) -> script.PreparedCollection:
    _, release_etag = repository.current_release()
    return script.PreparedCollection(
        release=repository.release,
        release_etag=release_etag,
        pointer_etags={
            dataset: repository.current_pointer(dataset)[1]
            for dataset in script.WRITE_DATASETS
        },
        frames={dataset: pd.DataFrame() for dataset in script.WRITE_DATASETS},
        archive_artifacts=(),
        warnings=("new-warning",),
        summary={"status": "validated_dry_run"},
    )


class AtomicApplyTests(unittest.TestCase):
    @staticmethod
    def _valid_report():
        return SimpleNamespace(issues=(), raise_for_errors=lambda: None)

    def test_baseexception_rolls_back_release_and_all_eight_pointers(self):
        with tempfile.TemporaryDirectory() as raw:
            repository = _TransactionRepository(
                Path(raw), fail_dataset=script.WRITE_DATASETS[3]
            )
            prepared = _prepared(repository)
            old_release = repository.objects.get("releases/current.json").data
            old_pointers = {
                dataset: repository.objects.get(repository.current_key(dataset)).data
                for dataset in script.WRITE_DATASETS
            }
            with patch.object(
                script,
                "validate_repository_snapshot",
                return_value=self._valid_report(),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    script.apply_collection(repository, prepared)
            self.assertEqual(
                repository.objects.get("releases/current.json").data, old_release
            )
            for dataset in script.WRITE_DATASETS:
                self.assertEqual(
                    repository.objects.get(repository.current_key(dataset)).data,
                    old_pointers[dataset],
                )

    def test_success_clears_only_pending_warning_after_strict_validation(self):
        with tempfile.TemporaryDirectory() as raw:
            repository = _TransactionRepository(Path(raw))
            prepared = _prepared(repository)
            with patch.object(
                script,
                "validate_repository_snapshot",
                return_value=self._valid_report(),
            ):
                result = script.apply_collection(repository, prepared)
            current, _ = repository.current_release()
            self.assertEqual(result["status"], "applied")
            self.assertNotIn(script.PENDING_IDENTITY_WARNING, current.warnings)
            self.assertIn("keep-me", current.warnings)
            self.assertIn("new-warning", current.warnings)
            self.assertEqual(
                set(current.dataset_versions), set(script.WRITE_DATASETS)
            )

    def test_repository_lock_rejects_a_second_writer(self):
        with tempfile.TemporaryDirectory() as raw:
            repository = SimpleNamespace(root=Path(raw))
            with script._exclusive_repository_lock(repository):
                with self.assertRaisesRegex(RuntimeError, "writer lock is already held"):
                    with script._exclusive_repository_lock(repository):
                        self.fail("second writer entered critical section")

    def test_release_cas_change_blocks_all_dataset_writes(self):
        with tempfile.TemporaryDirectory() as raw:
            repository = _TransactionRepository(Path(raw))
            prepared = _prepared(repository)
            current = repository.objects.get("releases/current.json")
            changed = DataRelease(
                version="other-release",
                created_at="2026-07-18T00:01:00Z",
                completed_session="2026-07-15",
                dataset_versions=repository.release.dataset_versions,
                warnings=repository.release.warnings,
            )
            repository.objects.put(
                "releases/current.json", changed.to_bytes(), if_match=current.etag
            )
            with self.assertRaisesRegex(RuntimeError, "Current release changed"):
                script.apply_collection(repository, prepared)
            for dataset in script.WRITE_DATASETS:
                pointer, _ = repository.current_pointer(dataset)
                self.assertEqual(
                    pointer.version, repository.release.dataset_versions[dataset]
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
