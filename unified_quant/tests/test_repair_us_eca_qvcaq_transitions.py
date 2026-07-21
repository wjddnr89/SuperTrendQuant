from __future__ import annotations

import base64
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import uuid

import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.ingest import SourceArtifact
from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.validation import ValidationIssue, ValidationReport


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_eca_qvcaq_transitions.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_eca_qvcaq_transitions", SCRIPT_PATH
)
assert SPEC and SPEC.loader
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)

OTHER_ID = "US:EODHD:00000000-0000-5000-8000-000000000001"


def _frame(dataset: str, rows: list[dict] | None = None) -> pd.DataFrame:
    extras = {
        "security_master": [
            "source_url",
            "provider_symbol",
            "action_provider_symbol",
            "isin",
        ],
        "symbol_history": ["source_url"],
        "daily_price_raw": ["source_url"],
        "corporate_actions": ["metadata"],
        "source_archive": ["source_url"],
    }.get(dataset, [])
    columns = list(
        dict.fromkeys((*dataset_spec(dataset).required_columns, *extras))
    )
    return pd.DataFrame(rows or [], columns=columns)


def _source() -> dict[str, str]:
    return {
        "source": "fixture",
        "retrieved_at": "2026-07-19T00:00:00Z",
        "source_hash": "f" * 64,
        "source_url": "https://example.test/source",
    }


def _master(
    security_id: str,
    symbol: str,
    *,
    exchange: str,
    active_from: str,
    active_to: str,
) -> dict:
    return {
        "security_id": security_id,
        "primary_symbol": symbol,
        "provider_symbol": f"{symbol}.US",
        "action_provider_symbol": f"{symbol}.US",
        "name": symbol,
        "exchange": exchange,
        "asset_type": "STOCK",
        "currency": "USD",
        "country": "US",
        "active_from": active_from,
        "active_to": active_to,
        "isin": "",
        **_source(),
    }


def _history(
    security_id: str,
    symbol: str,
    exchange: str,
    effective_from: str,
    effective_to: str,
) -> dict:
    return {
        "security_id": security_id,
        "symbol": symbol,
        "exchange": exchange,
        "effective_from": effective_from,
        "effective_to": effective_to,
        **_source(),
    }


def _price(security_id: str, session: str, close: float) -> dict:
    return {
        "security_id": security_id,
        "session": session,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1000,
        "currency": "USD",
        **_source(),
    }


def _split() -> dict:
    return {
        "event_id": script.QVC_2025_SPLIT_EVENT_ID,
        "security_id": script.QVCGA_ID,
        "action_type": "split",
        "effective_date": script.QVC_2025_SPLIT_DATE,
        "ex_date": script.QVC_2025_SPLIT_DATE,
        "announcement_date": "",
        "record_date": "",
        "payment_date": "",
        "cash_amount": None,
        "ratio": script.QVC_2025_SPLIT_RATIO,
        "currency": "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": False,
        "source_kind": "provider",
        "metadata": None,
        "source": "eodhd_splits",
        "retrieved_at": script.QVC_2025_SPLIT_RETRIEVED_AT,
        "source_hash": script.QVC_2025_SPLIT_SHA256,
        "source_url": script.QVC_2025_SPLIT_URL,
    }


def _existing() -> dict[str, pd.DataFrame]:
    master = _frame(
        "security_master",
        [
            _master(
                script.ECA_ID,
                "ECA",
                exchange="NYSE",
                active_from="2015-01-02",
                active_to="2020-10-08",
            ),
            _master(
                script.QVCGA_ID,
                "QVCGA",
                exchange="NASDAQ",
                active_from="2015-01-02",
                active_to=script.QVCGA_LAST,
            ),
            _master(
                OTHER_ID,
                "OTHER",
                exchange="NYSE",
                active_from="2015-01-02",
                active_to="",
            ),
        ],
    )
    history = _frame(
        "symbol_history",
        [
            _history(script.ECA_ID, "ECA", "NYSE", "2015-01-01", "2020-10-08"),
            _history(
                script.QVCGA_ID,
                "QVCGA",
                "NASDAQ",
                "2025-02-24",
                script.QVCGA_LAST,
            ),
            _history(OTHER_ID, "OTHER", "NYSE", "2015-01-01", ""),
        ],
    )
    prices = _frame(
        "daily_price_raw",
        [
            _price(script.ECA_ID, "2019-12-31", 4.69),
            _price(script.ECA_ID, script.ECA_LAST, 3.79),
            _price(script.ECA_ID, "2020-01-27", 0.59),
            _price(script.ECA_ID, "2020-10-08", 0.80),
            _price(script.QVCGA_ID, "2025-05-22", 0.50),
            _price(script.QVCGA_ID, "2025-05-23", 25.0),
            _price(script.QVCGA_ID, script.QVCGA_LAST, 0.34),
            _price(OTHER_ID, script.FETCH_END, 10.0),
        ],
    )
    actions = _frame("corporate_actions", [_split()])
    factors = build_adjustment_factors(prices, actions, source_version="fixture")
    return {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "lifecycle_resolutions": _frame("lifecycle_resolutions"),
        "adjustment_factors": factors,
        "source_archive": _frame("source_archive"),
    }


def _official_contents() -> tuple[bytes, bytes, bytes]:
    return (
        (
            "Encana Completes Reorganization. On January 24, 2020, completing a "
            "consolidation and share exchange for effectively one share of common "
            "stock of Ovintiv for every five common shares of Encana. "
            "The shares began trading on NYSE under OVV on January 27, 2020."
        ).encode(),
        (
            "SEC Rule Provision Notice under Rule 12d2-2. The Encana Corporation "
            "change became effective after the close on January 24, 2020; every "
            "five shares was exchanged for one share of common stock of Ovintiv "
            "Inc. The security was suspended January 27, 2020 and the Ovintiv "
            "common stock had continued listing on the NYSE."
        ).encode(),
        (
            "QVCGA and QVCGP were removed from Nasdaq. QVCAQ and QVCPQ began "
            "trading on OTCID on April 24, 2026."
        ).encode(),
    )


def _artifact(source: str, url: str, content: bytes, content_type: str) -> SourceArtifact:
    return SourceArtifact(
        source=source,
        source_url=url,
        retrieved_at="2026-07-19T00:00:00Z",
        content=content,
        content_type=content_type,
    )


def _artifacts(
    *,
    ovv_split: str = "1/5",
    qvcaq_end: str = script.FETCH_END,
) -> tuple[SourceArtifact, ...]:
    official = tuple(
        _artifact(source, url, content, "text/html")
        for source, url, content in zip(
            script.OFFICIAL_SOURCES,
            script.OFFICIAL_URLS,
            _official_contents(),
            strict=True,
        )
    )
    payloads = {
        (script.OVV_SYMBOL, "eod"): [
            {
                "date": script.OVV_FIRST,
                "open": 19.0,
                "high": 20.0,
                "low": 18.5,
                "close": 19.5,
                "volume": 100000,
            },
            {
                "date": script.FETCH_END,
                "open": 45.0,
                "high": 46.0,
                "low": 44.0,
                "close": 45.5,
                "volume": 200000,
            },
        ],
        (script.OVV_SYMBOL, "div"): [
            {"date": "2020-03-12", "unadjustedValue": 0.09375}
        ],
        (script.OVV_SYMBOL, "splits"): [
            {"date": script.OVV_FIRST, "split": ovv_split}
        ],
        (script.QVCAQ_SYMBOL, "eod"): [
            {
                "date": script.QVCAQ_FIRST,
                "open": 0.30,
                "high": 0.40,
                "low": 0.25,
                "close": 0.35,
                "volume": 50000,
            },
            {
                "date": qvcaq_end,
                "open": 0.20,
                "high": 0.25,
                "low": 0.19,
                "close": 0.22,
                "volume": 40000,
            },
        ],
        (script.QVCAQ_SYMBOL, "div"): [],
        (script.QVCAQ_SYMBOL, "splits"): [],
    }
    provider = tuple(
        _artifact(
            f"eodhd_{symbol.lower()}_{endpoint}",
            script.REQUEST_URLS[(symbol, endpoint)],
            script._canonical_json_bytes(payloads[(symbol, endpoint)]),
            "application/json",
        )
        for symbol, endpoint in script.REQUEST_ORDER
    )
    return (*official, *provider)


def _receipt(before: int = 8840) -> dict:
    return {
        "schema": "eodhd_budget_receipt/v2",
        "period": "2026-07-19",
        "used_before": before,
        "used_after": before + 6,
        "delta": 6,
        "own_claim_count": 6,
        "claim_positions": list(range(before + 1, before + 7)),
        "daily_limit": 100000,
        "reserve": 5000,
        "safety_ceiling": 95000,
    }


def _zero_receipt(before: int = 8840) -> dict:
    return {
        "schema": "eodhd_budget_receipt/v2",
        "period": "2026-07-19",
        "used_before": before,
        "used_after": before,
        "delta": 0,
        "own_claim_count": 0,
        "claim_positions": [],
        "daily_limit": 100000,
        "reserve": 5000,
        "safety_ceiling": 95000,
    }


def _bundle() -> script.ReviewedBundle:
    return script.bundle_from_artifacts(
        _artifacts(),
        official_http_attempts=3,
        eodhd_http_attempts=6,
        budget_receipt=_receipt(),
        require_reviewer_pins=False,
    )


def _pins(artifacts: tuple[SourceArtifact, ...]) -> dict[str, str]:
    return {item.source_url: item.source_hash for item in artifacts}


class _FakeBudget:
    def __init__(self, path: Path, *, used: int = 8840, ceiling: int = 95000):
        self.state_path = path
        self.period = "2026-07-19"
        self.seed_used = used
        self.used = used
        self.limit = 100000
        self.reserve = self.limit - ceiling
        self.ceiling = ceiling

    def claim(self) -> int:
        if self.used >= self.ceiling:
            raise RuntimeError("budget exhausted")
        self.used += 1
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"period": self.period, "used": self.used}),
            encoding="utf-8",
        )
        return self.used


class _FakeResponse:
    def __init__(self, content: bytes, content_type: str, *, status_code: int = 200):
        self.content = content
        self.headers = {"Content-Type": content_type}
        self.status_code = status_code

    def raise_for_status(self):
        return None


class _FakeSession:
    def __init__(self, artifacts: tuple[SourceArtifact, ...]):
        self.by_url = {item.source_url: item for item in artifacts}
        self.calls: list[dict] = []

    def get(
        self,
        url,
        params=None,
        headers=None,
        timeout=None,
        allow_redirects=None,
    ):
        if params:
            safe = (
                f"{url}?from={params['from']}&to={params['to']}"
            )
        else:
            safe = url
        self.calls.append(
            {
                "url": url,
                "params": params,
                "headers": headers,
                "timeout": timeout,
                "allow_redirects": allow_redirects,
            }
        )
        item = self.by_url[safe]
        return _FakeResponse(item.content, item.content_type)


def _write_resume_source(
    root: Path,
    *,
    artifacts: tuple[SourceArtifact, ...] | None = None,
    receipt: dict | None = None,
    error: str = script.RESUMABLE_OFFICIAL_ERROR,
) -> tuple[str, Path]:
    return script._write_quarantine(
        root,
        artifacts or _artifacts()[: script.EXPECTED_OFFICIAL_CALLS],
        receipt or _zero_receipt(),
        status="incomplete",
        error=error,
    )


class RepairUsEcaQvcaqTransitionsTests(unittest.TestCase):
    def test_snapshot_validation_preserves_preexisting_issue_inventory(self):
        baseline = ValidationReport(
            dataset="repository",
            issues=(
                ValidationIssue(
                    "existing_issue",
                    "Existing unrelated issue.",
                    row_count=1,
                    fingerprints=("a" * 64,),
                ),
            ),
        )
        changed = ValidationReport(
            dataset="repository",
            issues=(
                ValidationIssue(
                    "existing_issue",
                    "Existing unrelated issue.",
                    row_count=2,
                    fingerprints=("a" * 64, "b" * 64),
                ),
            ),
        )
        with mock.patch.object(
            script,
            "validate_repository_snapshot",
            side_effect=(baseline, baseline),
        ):
            observed = script._assert_snapshot_issues_preserved(
                mock.sentinel.repository, {}, {}, {}
            )
        self.assertEqual(len(observed), 1)

        with mock.patch.object(
            script,
            "validate_repository_snapshot",
            side_effect=(baseline, changed),
        ):
            with self.assertRaisesRegex(RuntimeError, "unrelated repository"):
                script._assert_snapshot_issues_preserved(
                    mock.sentinel.repository, {}, {}, {}
                )

    def test_ovv_id_uses_canonical_provider_code_without_dot_us_suffix(self):
        expected = "US:EODHD:" + str(
            uuid.uuid5(uuid.NAMESPACE_URL, "eodhd:US:OVV:symbol:OVV")
        )
        wrong_transport_suffix_id = "US:EODHD:" + str(
            uuid.uuid5(uuid.NAMESPACE_URL, "eodhd:US:OVV.US:symbol:OVV")
        )
        self.assertEqual(script.OVV_ID, expected)
        self.assertNotEqual(script.OVV_ID, wrong_transport_suffix_id)

    def test_requirements_are_three_official_and_six_eodhd_one_shot_calls(self):
        plan = script.requirements_plan()

        self.assertEqual(plan["official_http_attempts"], 3)
        self.assertEqual(plan["eodhd_http_attempts"], 6)
        self.assertEqual(plan["total_http_attempts"], 9)
        self.assertEqual(plan["retry_count"], 0)
        self.assertEqual(
            plan["resume_quarantine_id"],
            script.RESUMABLE_OFFICIAL_QUARANTINE_ID,
        )
        self.assertEqual(plan["resume_official_http_attempts"], 0)
        self.assertEqual(plan["resume_eodhd_http_attempts"], 6)
        self.assertEqual(
            [(row["symbol"], row["endpoint"]) for row in plan["eodhd_requests"]],
            list(script.REQUEST_ORDER),
        )
        self.assertFalse(plan["network_accessed"])
        self.assertFalse(plan["r2_accessed"])

    def test_stage1_collection_uses_budget_delta_six_and_never_promotes_itself(self):
        artifacts = _artifacts()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = _FakeBudget(root / "state/eodhd_call_budget.json")
            session = _FakeSession(artifacts)
            with mock.patch.dict(
                "os.environ",
                {
                    "EODHD_API_TOKEN": "secret",
                    "SEC_USER_AGENT": "Tester test@example.com",
                },
            ):
                summary = script.collect_stage1(
                    root,
                    session_factory=lambda: session,
                    budget_factory=lambda: budget,
                )

            self.assertEqual(summary["status"], "stage1_fetched_needs_reviewer_hash_pins")
            self.assertEqual(summary["official_http_attempts"], 3)
            self.assertEqual(summary["eodhd_http_attempts"], 6)
            self.assertEqual(summary["budget_receipt"]["delta"], 6)
            self.assertEqual(summary["budget_receipt"]["own_claim_count"], 6)
            self.assertFalse(script._bundle_cache_path(root).exists())
            self.assertEqual(len(session.calls), 9)
            self.assertNotIn("secret", json.dumps(summary))
            self.assertTrue(
                all(call["allow_redirects"] is False for call in session.calls)
            )
            for call in session.calls[:3]:
                self.assertEqual(
                    call["headers"]["User-Agent"], "Tester test@example.com"
                )
                self.assertEqual(
                    call["headers"]["Accept"],
                    "text/html,application/xhtml+xml",
                )
                self.assertEqual(call["headers"]["Accept-Encoding"], "identity")

    def test_stage1_receipt_uses_own_claims_when_other_collectors_interleave(self):
        artifacts = _artifacts()

        class InterleavingBudget(_FakeBudget):
            def claim(self) -> int:
                # Simulate one foreign claim immediately before each claim made
                # by this collector.  The global ledger grows by twelve while
                # this exact request plan still owns exactly six positions.
                self.used += 1
                return super().claim()

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = InterleavingBudget(root / "state/eodhd_call_budget.json")
            session = _FakeSession(artifacts)
            with mock.patch.dict(
                "os.environ",
                {
                    "EODHD_API_TOKEN": "secret",
                    "SEC_USER_AGENT": "Tester test@example.com",
                },
            ):
                summary = script.collect_stage1(
                    root,
                    session_factory=lambda: session,
                    budget_factory=lambda: budget,
                )

            receipt = summary["budget_receipt"]
            self.assertEqual(receipt["delta"], 12)
            self.assertEqual(receipt["own_claim_count"], 6)
            self.assertEqual(receipt["claim_positions"], [8842, 8844, 8846, 8848, 8850, 8852])
            self.assertEqual(len(session.calls), 9)

    def test_reviewed_official_resume_pins_match_the_actual_stage1_capture(self):
        self.assertEqual(
            script.RESUMABLE_OFFICIAL_QUARANTINE_ID,
            "d784da1588c64351e9eb673884be793635f250c2b6b0fa3f1cb18080fe614ce5",
        )
        self.assertEqual(
            dict(script.RESUMABLE_OFFICIAL_SHA256),
            {
                script.ECA_PRIMARY_URL: (
                    "cb6cdb670b3a30d38f0529d242f4ea470052c04204e3101537627f7df3955bef"
                ),
                script.ECA_SEC_URL: (
                    "58d199861b620211b63c846e3184baf1ff7982adb124e085c5f726e2fd06af59"
                ),
                script.QVC_OFFICIAL_URL: (
                    "55829c9064eee534b6f79027648172494a507f8b9be16e9598dc57cdd58c165b"
                ),
            },
        )

    def test_resume_reuses_three_official_raws_and_calls_only_six_eodhd_urls(self):
        all_artifacts = _artifacts()
        official = all_artifacts[: script.EXPECTED_OFFICIAL_CALLS]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_id, _ = _write_resume_source(root, artifacts=official)
            budget = _FakeBudget(root / "state/eodhd_call_budget.json")
            session = _FakeSession(all_artifacts)
            with mock.patch.object(
                script, "RESUMABLE_OFFICIAL_QUARANTINE_ID", source_id
            ), mock.patch.object(
                script, "RESUMABLE_OFFICIAL_SHA256", _pins(official)
            ), mock.patch.dict(
                "os.environ",
                {
                    "EODHD_API_TOKEN": "secret",
                    "SEC_USER_AGENT": "Tester test@example.com",
                },
            ):
                summary = script.resume_stage1(
                    root,
                    source_id,
                    session_factory=lambda: session,
                    budget_factory=lambda: budget,
                )

            self.assertEqual(
                summary["status"],
                "stage1_resumed_fetched_needs_reviewer_hash_pins",
            )
            self.assertEqual(summary["resumed_from_quarantine_id"], source_id)
            self.assertEqual(summary["official_raws_reused"], 3)
            self.assertEqual(summary["official_http_attempts_this_run"], 0)
            self.assertEqual(summary["eodhd_http_attempts"], 6)
            self.assertEqual(summary["budget_receipt"]["own_claim_count"], 6)
            self.assertEqual(len(session.calls), 6)
            self.assertTrue(
                all(call["url"].startswith("https://eodhd.com/api/") for call in session.calls)
            )
            resumed = script.read_quarantine(root, summary["quarantine_id"])
            self.assertEqual(len(resumed.artifacts), 9)
            self.assertEqual(
                tuple(item.source_hash for item in resumed.artifacts[:3]),
                tuple(item.source_hash for item in official),
            )

    def test_resume_rejects_content_address_tamper_before_any_http(self):
        official = _artifacts()[: script.EXPECTED_OFFICIAL_CALLS]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_id, path = _write_resume_source(root, artifacts=official)
            envelope = json.loads(gzip.decompress(path.read_bytes()))
            envelope["artifacts"][0]["content_base64"] = base64.b64encode(
                b"tampered"
            ).decode("ascii")
            path.write_bytes(
                gzip.compress(script._canonical_json_bytes(envelope), mtime=0)
            )
            session = _FakeSession(_artifacts())
            with mock.patch.object(
                script, "RESUMABLE_OFFICIAL_QUARANTINE_ID", source_id
            ), self.assertRaisesRegex(ValueError, "content-address hash"):
                script.resume_stage1(
                    root,
                    source_id,
                    session_factory=lambda: session,
                    budget_factory=lambda: _FakeBudget(root / "budget.json"),
                )
            self.assertEqual(session.calls, [])

    def test_resume_rejects_official_artifact_order_before_any_http(self):
        official = _artifacts()[: script.EXPECTED_OFFICIAL_CALLS]
        reordered = (official[1], official[0], official[2])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_id, _ = _write_resume_source(root, artifacts=reordered)
            session = _FakeSession(_artifacts())
            with mock.patch.object(
                script, "RESUMABLE_OFFICIAL_QUARANTINE_ID", source_id
            ), self.assertRaisesRegex(ValueError, "URL/order"):
                script.resume_stage1(
                    root,
                    source_id,
                    session_factory=lambda: session,
                    budget_factory=lambda: _FakeBudget(root / "budget.json"),
                )
            self.assertEqual(session.calls, [])

    def test_resume_rejects_nonzero_prior_claim_receipt_before_any_http(self):
        receipt = _zero_receipt()
        receipt.update(
            {
                "used_after": receipt["used_before"] + 1,
                "delta": 1,
                "own_claim_count": 1,
                "claim_positions": [receipt["used_before"] + 1],
            }
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_id, _ = _write_resume_source(root, receipt=receipt)
            session = _FakeSession(_artifacts())
            with mock.patch.object(
                script, "RESUMABLE_OFFICIAL_QUARANTINE_ID", source_id
            ), self.assertRaisesRegex(ValueError, "zero prior EODHD claims"):
                script.resume_stage1(
                    root,
                    source_id,
                    session_factory=lambda: session,
                    budget_factory=lambda: _FakeBudget(root / "budget.json"),
                )
            self.assertEqual(session.calls, [])

    def test_resume_rejects_noncurrent_budget_period_before_any_http(self):
        receipt = _zero_receipt()
        receipt["period"] = "2026-07-18"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_id, _ = _write_resume_source(root, receipt=receipt)
            session = _FakeSession(_artifacts())
            with mock.patch.object(
                script, "RESUMABLE_OFFICIAL_QUARANTINE_ID", source_id
            ), self.assertRaisesRegex(ValueError, "current EODHD budget"):
                script.resume_stage1(
                    root,
                    source_id,
                    session_factory=lambda: session,
                    budget_factory=lambda: _FakeBudget(root / "budget.json"),
                )
            self.assertEqual(session.calls, [])

    def test_resume_rejects_wrong_pre_eodhd_error_before_any_http(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_id, _ = _write_resume_source(
                root,
                error="ValueError: unrelated provider validation failure",
            )
            session = _FakeSession(_artifacts())
            with mock.patch.object(
                script, "RESUMABLE_OFFICIAL_QUARANTINE_ID", source_id
            ), self.assertRaisesRegex(ValueError, "official-term pre-EODHD"):
                script.resume_stage1(
                    root,
                    source_id,
                    session_factory=lambda: session,
                    budget_factory=lambda: _FakeBudget(root / "budget.json"),
                )
            self.assertEqual(session.calls, [])

    def test_resume_rejects_redirect_raw_even_with_valid_quarantine_hash(self):
        official = list(_artifacts()[: script.EXPECTED_OFFICIAL_CALLS])
        official[0] = _artifact(
            script.OFFICIAL_SOURCES[0],
            script.ECA_PRIMARY_URL,
            b"<html>302 Found Location: https://redirect.example</html>",
            "text/html",
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_id, _ = _write_resume_source(root, artifacts=tuple(official))
            session = _FakeSession(_artifacts())
            with mock.patch.object(
                script, "RESUMABLE_OFFICIAL_QUARANTINE_ID", source_id
            ), self.assertRaisesRegex(ValueError, "reviewed pins"):
                script.resume_stage1(
                    root,
                    source_id,
                    session_factory=lambda: session,
                    budget_factory=lambda: _FakeBudget(root / "budget.json"),
                )
            self.assertEqual(session.calls, [])

    def test_post_fetch_receipt_failure_preserves_all_nine_raws_as_incomplete(self):
        artifacts = _artifacts()

        class BadClaimProofBudget(_FakeBudget):
            def claim(self) -> int:
                actual = super().claim()
                return actual if actual == self.seed_used + 1 else self.seed_used + 1

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = BadClaimProofBudget(root / "state/eodhd_call_budget.json")
            session = _FakeSession(artifacts)
            with mock.patch.dict(
                "os.environ",
                {
                    "EODHD_API_TOKEN": "secret",
                    "SEC_USER_AGENT": "Tester test@example.com",
                },
            ):
                with self.assertRaisesRegex(ValueError, "own-claim proof"):
                    script.collect_stage1(
                        root,
                        session_factory=lambda: session,
                        budget_factory=lambda: budget,
                    )

            paths = list(
                (root / "state/us-eca-qvcaq-transitions/quarantine").glob(
                    "*.json.gz"
                )
            )
            self.assertEqual(len(paths), 1)
            envelope = json.loads(gzip.decompress(paths[0].read_bytes()))
            self.assertEqual(envelope["status"], "incomplete")
            self.assertEqual(len(envelope["artifacts"]), 9)
            self.assertEqual(envelope["budget_receipt"]["own_claim_count"], 6)

    def test_stage1_requires_contactable_user_agent_before_any_http(self):
        artifacts = _artifacts()
        with tempfile.TemporaryDirectory() as temp:
            budget = _FakeBudget(Path(temp) / "budget.json")
            session = _FakeSession(artifacts)
            with self.assertRaisesRegex(RuntimeError, "SEC_USER_AGENT"):
                script.ExactStage1Client(
                    session=session,
                    token="secret",
                    user_agent="not-contactable",
                    budget=budget,
                )
            self.assertEqual(session.calls, [])

    def test_redirect_is_fail_closed_after_one_call_with_no_hidden_followup(self):
        artifacts = _artifacts()

        class RedirectSession(_FakeSession):
            def get(
                self,
                url,
                params=None,
                headers=None,
                timeout=None,
                allow_redirects=None,
            ):
                self.calls.append(
                    {
                        "url": url,
                        "params": params,
                        "headers": headers,
                        "timeout": timeout,
                        "allow_redirects": allow_redirects,
                    }
                )
                return _FakeResponse(b"redirect", "text/html", status_code=302)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = _FakeBudget(root / "state/eodhd_call_budget.json")
            session = RedirectSession(artifacts)
            with mock.patch.dict(
                "os.environ",
                {
                    "EODHD_API_TOKEN": "secret",
                    "SEC_USER_AGENT": "Tester test@example.com",
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "forbidden redirect HTTP 302"):
                    script.collect_stage1(
                        root,
                        session_factory=lambda: session,
                        budget_factory=lambda: budget,
                    )
            self.assertEqual(len(session.calls), 1)
            self.assertIs(session.calls[0]["allow_redirects"], False)
            quarantine = next(
                (root / "state/us-eca-qvcaq-transitions/quarantine").glob(
                    "*.json.gz"
                )
            )
            envelope = json.loads(gzip.decompress(quarantine.read_bytes()))
            self.assertEqual(envelope["budget_receipt"]["delta"], 0)
            self.assertEqual(envelope["artifacts"], [])

    def test_bad_official_terms_stop_before_all_six_eodhd_calls(self):
        artifacts = list(_artifacts())
        artifacts[2] = _artifact(
            "official_2",
            script.QVC_OFFICIAL_URL,
            b"generic investor page without transition terms",
            "text/html",
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = _FakeBudget(root / "state/eodhd_call_budget.json")
            session = _FakeSession(tuple(artifacts))
            with mock.patch.dict(
                "os.environ",
                {
                    "EODHD_API_TOKEN": "secret",
                    "SEC_USER_AGENT": "Tester test@example.com",
                },
            ):
                with self.assertRaisesRegex(ValueError, "QVC issuer"):
                    script.collect_stage1(
                        root,
                        session_factory=lambda: session,
                        budget_factory=lambda: budget,
                    )
            self.assertEqual(len(session.calls), 3)
            quarantine = next(
                (root / "state/us-eca-qvcaq-transitions/quarantine").glob(
                    "*.json.gz"
                )
            )
            envelope = json.loads(gzip.decompress(quarantine.read_bytes()))
            self.assertEqual(envelope["budget_receipt"]["delta"], 0)
            self.assertEqual(len(envelope["artifacts"]), 3)

    def test_bad_ovv_eod_stops_before_dividend_and_split_calls(self):
        artifacts = list(_artifacts())
        artifacts[3] = _artifact(
            "eodhd_ovv_eod",
            script.REQUEST_URLS[(script.OVV_SYMBOL, "eod")],
            b"[]",
            "application/json",
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = _FakeBudget(root / "state/eodhd_call_budget.json")
            session = _FakeSession(tuple(artifacts))
            with mock.patch.dict(
                "os.environ",
                {
                    "EODHD_API_TOKEN": "secret",
                    "SEC_USER_AGENT": "Tester test@example.com",
                },
            ):
                with self.assertRaisesRegex(ValueError, "OVV EOD payload is empty"):
                    script.collect_stage1(
                        root,
                        session_factory=lambda: session,
                        budget_factory=lambda: budget,
                    )
            self.assertEqual(len(session.calls), 4)
            quarantine = next(
                (root / "state/us-eca-qvcaq-transitions/quarantine").glob(
                    "*.json.gz"
                )
            )
            envelope = json.loads(gzip.decompress(quarantine.read_bytes()))
            self.assertEqual(envelope["budget_receipt"]["delta"], 1)
            self.assertEqual(len(envelope["artifacts"]), 4)

    def test_bad_ovv_dividend_stops_before_ovv_split_and_qvcaq_calls(self):
        artifacts = list(_artifacts())
        artifacts[4] = _artifact(
            "eodhd_ovv_div",
            script.REQUEST_URLS[(script.OVV_SYMBOL, "div")],
            script._canonical_json_bytes(
                [{"date": "2020-03-12", "unadjustedValue": "bad"}]
            ),
            "application/json",
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = _FakeBudget(root / "state/eodhd_call_budget.json")
            session = _FakeSession(tuple(artifacts))
            with mock.patch.dict(
                "os.environ",
                {
                    "EODHD_API_TOKEN": "secret",
                    "SEC_USER_AGENT": "Tester test@example.com",
                },
            ):
                with self.assertRaisesRegex(ValueError, "dividend amount is invalid"):
                    script.collect_stage1(
                        root,
                        session_factory=lambda: session,
                        budget_factory=lambda: budget,
                    )
            self.assertEqual(len(session.calls), 5)
            quarantine = next(
                (root / "state/us-eca-qvcaq-transitions/quarantine").glob(
                    "*.json.gz"
                )
            )
            envelope = json.loads(gzip.decompress(quarantine.read_bytes()))
            self.assertEqual(envelope["budget_receipt"]["delta"], 2)
            self.assertEqual(len(envelope["artifacts"]), 5)
            for call in session.calls[3:]:
                self.assertEqual(call["headers"]["Accept"], "application/json")
                self.assertEqual(call["headers"]["Accept-Encoding"], "identity")

    def test_budget_preflight_refuses_before_any_partial_request(self):
        artifacts = _artifacts()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = _FakeBudget(
                root / "state/eodhd_call_budget.json", used=94995, ceiling=95000
            )
            session = _FakeSession(artifacts)
            with mock.patch.dict(
                "os.environ",
                {
                    "EODHD_API_TOKEN": "secret",
                    "SEC_USER_AGENT": "Tester test@example.com",
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "refused a partial six-call"):
                    script.collect_stage1(
                        root,
                        session_factory=lambda: session,
                        budget_factory=lambda: budget,
                    )
            self.assertEqual(session.calls, [])

    def test_failed_provider_attempt_is_counted_once_and_partial_raws_are_preserved(self):
        artifacts = _artifacts()

        class FailingResponse(_FakeResponse):
            def raise_for_status(self):
                raise RuntimeError("single attempted failure")

        class FailingSession(_FakeSession):
            def get(
                self,
                url,
                params=None,
                headers=None,
                timeout=None,
                allow_redirects=None,
            ):
                if len(self.calls) == 4:
                    self.calls.append(
                        {
                            "url": url,
                            "params": params,
                            "headers": headers,
                            "timeout": timeout,
                            "allow_redirects": allow_redirects,
                        }
                    )
                    return FailingResponse(b"failure", "application/json")
                return super().get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=allow_redirects,
                )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = _FakeBudget(root / "state/eodhd_call_budget.json")
            session = FailingSession(artifacts)
            with mock.patch.dict(
                "os.environ",
                {
                    "EODHD_API_TOKEN": "secret",
                    "SEC_USER_AGENT": "Tester test@example.com",
                },
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "EODHD response failed: OVV/div"
                ):
                    script.collect_stage1(
                        root,
                        session_factory=lambda: session,
                        budget_factory=lambda: budget,
                    )

            self.assertEqual(len(session.calls), 5)
            paths = list(
                (root / "state/us-eca-qvcaq-transitions/quarantine").glob(
                    "*.json.gz"
                )
            )
            self.assertEqual(len(paths), 1)
            envelope = json.loads(gzip.decompress(paths[0].read_bytes()))
            self.assertEqual(envelope["status"], "incomplete")
            self.assertEqual(envelope["budget_receipt"]["delta"], 2)
            self.assertEqual(len(envelope["artifacts"]), 4)
            self.assertNotIn("secret", envelope["error"])

    def test_quarantine_cannot_promote_until_all_exact_hashes_are_code_pinned(self):
        artifacts = _artifacts()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            qid, _ = script._write_quarantine(
                root, artifacts, _receipt(), status="complete_unreviewed"
            )
            pending_pins = _pins(artifacts)
            pending_pins[script.REQUEST_URLS[(script.OVV_SYMBOL, "eod")]] = ""
            with mock.patch.object(
                script, "REVIEWED_ARTIFACT_SHA256", pending_pins
            ):
                with self.assertRaisesRegex(ValueError, "pins are still pending"):
                    script.promote_quarantine(root, qid)

            with mock.patch.object(
                script, "REVIEWED_ARTIFACT_SHA256", _pins(artifacts)
            ):
                summary = script.promote_quarantine(root, qid)
                self.assertEqual(summary["status"], "reviewed_bundle_promoted")
                replay = script._read_bundle_cache(root)
                self.assertIsNotNone(replay)

    def test_quarantine_content_address_and_review_pin_tampering_fail_closed(self):
        artifacts = _artifacts()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            qid, path = script._write_quarantine(
                root, artifacts, _receipt(), status="complete_unreviewed"
            )
            envelope = json.loads(gzip.decompress(path.read_bytes()))
            envelope["artifacts"][0]["content_base64"] = base64.b64encode(
                b"changed"
            ).decode()
            path.write_bytes(gzip.compress(script._canonical_json_bytes(envelope), mtime=0))
            with self.assertRaisesRegex(ValueError, "content-address hash"):
                script.read_quarantine(root, qid)

        pins = _pins(artifacts)
        pins[script.QVC_OFFICIAL_URL] = "0" * 64
        with mock.patch.object(script, "REVIEWED_ARTIFACT_SHA256", pins):
            with self.assertRaisesRegex(ValueError, "pin mismatch"):
                script.bundle_from_artifacts(
                    artifacts,
                    official_http_attempts=3,
                    eodhd_http_attempts=6,
                    budget_receipt=_receipt(),
                    require_reviewer_pins=True,
                )

    def test_prepare_removes_eca_tail_adds_ovv_and_continues_qvc_same_id(self):
        frames, summary = script.prepare_frames(
            _existing(), _bundle(), completed_session=script.FETCH_END
        )

        self.assertTrue(script.identity_is_repaired(frames))
        self.assertEqual(summary["eca_tail_price_rows_removed"], 2)
        self.assertEqual(summary["eca_canonical_ratio"], 0.2)
        self.assertEqual(summary["qvc_identity_decision"], "same_legal_security_same_security_id")
        self.assertEqual(summary["qvc_2025_split_rows_preserved"], 1)

        master = frames["security_master"].set_index("security_id")
        self.assertEqual(master.loc[script.ECA_ID, "active_to"], script.ECA_LAST)
        self.assertEqual(master.loc[script.ECA_ID, "name"], "Encana Corporation")
        self.assertEqual(master.loc[script.OVV_ID, "primary_symbol"], "OVV")
        self.assertEqual(master.loc[script.OVV_ID, "name"], "Ovintiv Inc")
        self.assertEqual(master.loc[script.OVV_ID, "active_from"], script.OVV_FIRST)
        self.assertEqual(master.loc[script.QVCGA_ID, "primary_symbol"], "QVCAQ")
        self.assertEqual(master.loc[script.QVCGA_ID, "provider_symbol"], "QVCAQ.US")

        history = frames["symbol_history"]
        qvc_history = history.loc[history["security_id"].eq(script.QVCGA_ID)]
        self.assertEqual(set(qvc_history["symbol"]), {"QVCGA", "QVCAQ"})
        self.assertEqual(qvc_history.loc[qvc_history.symbol.eq("QVCGA"), "effective_to"].iloc[0], script.QVCGA_LAST)
        self.assertEqual(qvc_history.loc[qvc_history.symbol.eq("QVCAQ"), "effective_from"].iloc[0], script.QVCAQ_FIRST)

        actions = frames["corporate_actions"].set_index("event_id")
        self.assertEqual(float(actions.loc[script.ECA_EVENT_ID, "ratio"]), 0.2)
        self.assertEqual(actions.loc[script.ECA_EVENT_ID, "new_security_id"], script.OVV_ID)
        self.assertEqual(actions.loc[script.QVC_EVENT_ID, "new_security_id"], script.QVCGA_ID)
        self.assertIn(script.QVC_2025_SPLIT_EVENT_ID, actions.index)

    def test_qvc_2025_split_ratio_and_provenance_are_exact_preconditions(self):
        existing = _existing()
        existing["corporate_actions"].loc[0, "ratio"] = 0.01
        existing["adjustment_factors"] = build_adjustment_factors(
            existing["daily_price_raw"],
            existing["corporate_actions"],
            source_version="fixture",
        )
        with self.assertRaisesRegex(ValueError, "ratio/provenance changed"):
            script.prepare_frames(
                existing, _bundle(), completed_session=script.FETCH_END
            )

        existing = _existing()
        existing["corporate_actions"].loc[0, "source_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "ratio/provenance changed"):
            script.prepare_frames(
                existing, _bundle(), completed_session=script.FETCH_END
            )

    def test_repaired_identity_recomputes_factor_economics_and_checks_lineage(self):
        frames, _ = script.prepare_frames(
            _existing(), _bundle(), completed_session=script.FETCH_END
        )
        self.assertTrue(script.identity_is_repaired(frames))

        economic_tamper = {name: frame.copy() for name, frame in frames.items()}
        target = economic_tamper["adjustment_factors"].index[
            economic_tamper["adjustment_factors"]["security_id"].astype(str).eq(
                script.QVCGA_ID
            )
        ][0]
        economic_tamper["adjustment_factors"].loc[
            target, "total_return_factor"
        ] *= 1.01
        self.assertFalse(script.identity_is_repaired(economic_tamper))

        lineage_tamper = {name: frame.copy() for name, frame in frames.items()}
        lineage_tamper["adjustment_factors"].loc[target, "source_hash"] = "bad"
        self.assertFalse(script.identity_is_repaired(lineage_tamper))

    def test_ovv_transition_split_must_match_official_point_two(self):
        with self.assertRaisesRegex(ValueError, "conflicts with official"):
            script.bundle_from_artifacts(
                _artifacts(ovv_split="1/4"),
                official_http_attempts=3,
                eodhd_http_attempts=6,
                budget_receipt=_receipt(),
                require_reviewer_pins=False,
            )

    def test_qvcaq_provider_history_must_cover_exact_first_and_last_dates(self):
        with self.assertRaisesRegex(ValueError, "boundary"):
            script.bundle_from_artifacts(
                _artifacts(qvcaq_end="2026-07-14"),
                official_http_attempts=3,
                eodhd_http_attempts=6,
                budget_receipt=_receipt(),
                require_reviewer_pins=False,
            )

    def test_official_qvc_claim_must_bind_both_old_and_new_markets_and_tickers(self):
        artifacts = list(_artifacts())
        artifacts[2] = _artifact(
            "official_2",
            script.QVC_OFFICIAL_URL,
            b"QVCGA was removed from Nasdaq on April 24, 2026.",
            "text/html",
        )
        with self.assertRaisesRegex(ValueError, "QVC issuer"):
            script.bundle_from_artifacts(
                artifacts,
                official_http_attempts=3,
                eodhd_http_attempts=6,
                budget_receipt=_receipt(),
                require_reviewer_pins=False,
            )

    def test_source_archive_contains_all_three_official_and_six_provider_hashes(self):
        frames, summary = script.prepare_frames(
            _existing(), _bundle(), completed_session=script.FETCH_END
        )
        archive = frames["source_archive"]
        self.assertEqual(len(archive), 9)
        self.assertEqual(set(archive["source_url"]), set(script.REVIEWED_ARTIFACT_SHA256))
        self.assertEqual(summary["official_archive_rows_added"], 3)
        self.assertEqual(summary["eodhd_archive_rows_added"], 6)

    def test_offline_plan_on_current_release_never_requires_network(self):
        args = script._parse_args(["--offline-plan", "--cache-root", "data/cache"])
        with mock.patch.object(script, "_read_bundle_cache", return_value=None):
            summary = script.run(args)
        self.assertEqual(summary["status"], "offline_plan_blocked_pending_reviewed_bundle")
        self.assertFalse(summary["network_accessed"])
        tail_rows = summary["eca_tail_rows_after_2020_01_24"]
        self.assertGreaterEqual(tail_rows, 0)
        if tail_rows == 0:
            self.assertEqual(summary["eca_current_last_session"], script.ECA_LAST)
        self.assertEqual(summary["qvc_2025_split_ratios"], [0.02])

    def test_fetch_promote_plan_and_apply_modes_cannot_be_combined(self):
        args = script._parse_args(
            ["--fetch-stage1", "--offline-plan", "--cache-root", "data/cache"]
        )
        with self.assertRaisesRegex(ValueError, "separate invocations"):
            script.run(args)

        args = script._parse_args(
            [
                "--resume-quarantine",
                "0" * 64,
                "--apply",
                "--cache-root",
                "data/cache",
            ]
        )
        with self.assertRaisesRegex(ValueError, "separate invocations"):
            script.run(args)

    def test_successful_apply_commits_exact_planned_factor_lineage_and_replays(self):
        bundle = _bundle()
        existing = _existing()
        original_other = existing["adjustment_factors"].loc[
            existing["adjustment_factors"]["security_id"].astype(str).eq(OTHER_ID),
            ["security_id", "session", "split_factor", "total_return_factor"],
        ].reset_index(drop=True)
        with tempfile.TemporaryDirectory() as temp:
            repository = LocalDatasetRepository(temp)
            versions: dict[str, str] = {}
            for dataset, frame in existing.items():
                result = repository.write_frame(
                    dataset,
                    frame,
                    completed_session=script.FETCH_END,
                    incomplete_action_policy="warn",
                    metadata={"fixture": True},
                    version=f"fixture-{dataset}",
                )
                versions[dataset] = result.manifest.version
            repository.commit_release(
                script.FETCH_END,
                versions,
                quality=DataQuality.VALID,
            )

            prepared = script.prepare_run(repository, bundle)
            planned = dict(prepared.planned_versions)
            expected_lineage = script._adjustment_source_version(
                planned["daily_price_raw"], planned["corporate_actions"]
            )
            result = script.apply_repair(repository, prepared)

            self.assertEqual(result["status"], "applied")
            current, _ = repository.current_release()
            self.assertIsNotNone(current)
            assert current is not None
            for dataset in script.WRITE_DATASETS:
                self.assertEqual(current.dataset_versions[dataset], planned[dataset])
            factors = repository.read_frame(
                "adjustment_factors",
                current.dataset_versions["adjustment_factors"],
            )
            self.assertEqual(set(factors["source_version"]), {expected_lineage})
            self.assertEqual(set(factors["source_hash"]), {expected_lineage})
            manifest = repository.manifest_for_version(
                "adjustment_factors",
                current.dataset_versions["adjustment_factors"],
            )
            self.assertEqual(manifest.metadata["source_version"], expected_lineage)
            self.assertEqual(
                manifest.metadata["source_daily_price_version"],
                planned["daily_price_raw"],
            )
            self.assertEqual(
                manifest.metadata["source_corporate_actions_version"],
                planned["corporate_actions"],
            )
            applied_other = factors.loc[
                factors["security_id"].astype(str).eq(OTHER_ID),
                ["security_id", "session", "split_factor", "total_return_factor"],
            ].reset_index(drop=True)
            pd.testing.assert_frame_equal(
                applied_other,
                original_other,
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
            replay = script.prepare_run(repository, bundle)
            self.assertEqual(replay.summary["status"], "already_repaired")
            self.assertEqual(
                replay.summary["adjustment_source_version"], expected_lineage
            )

    def test_transaction_rolls_all_pointers_and_release_back_after_failure(self):
        artifacts = _artifacts()
        bundle = _bundle()
        with tempfile.TemporaryDirectory() as temp:
            repository = LocalDatasetRepository(temp)
            versions: dict[str, str] = {}
            for dataset, frame in _existing().items():
                result = repository.write_frame(
                    dataset,
                    frame,
                    completed_session=script.FETCH_END,
                    incomplete_action_policy="warn",
                    metadata={"fixture": True},
                    version=f"fixture-{dataset}",
                )
                versions[dataset] = result.manifest.version
            release = repository.commit_release(
                script.FETCH_END,
                versions,
                quality=DataQuality.VALID,
            )
            old_release_bytes = repository.objects.get("releases/current.json").data
            old_pointer_bytes = {
                dataset: repository.objects.get(repository.current_key(dataset)).data
                for dataset in script.WRITE_DATASETS
            }

            prepared = script.prepare_run(repository, bundle)
            with self.assertRaisesRegex(RuntimeError, "injected rollback"):
                script.apply_repair(
                    repository,
                    prepared,
                    failure_injector=lambda stage: (
                        (_ for _ in ()).throw(RuntimeError("injected rollback"))
                        if stage == "after_daily_price_raw"
                        else None
                    ),
                )

            self.assertEqual(
                repository.objects.get("releases/current.json").data,
                old_release_bytes,
            )
            for dataset in script.WRITE_DATASETS:
                self.assertEqual(
                    repository.objects.get(repository.current_key(dataset)).data,
                    old_pointer_bytes[dataset],
                )
            journals = list((Path(temp) / script.TRANSACTION_DIR).glob("*.json"))
            self.assertEqual(len(journals), 1)
            journal = json.loads(journals[0].read_text(encoding="utf-8"))
            self.assertEqual(journal["status"], "rolled_back")
            self.assertEqual(journal["rollback_errors"], [])


if __name__ == "__main__":
    unittest.main()
