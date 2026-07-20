from __future__ import annotations

import json
import multiprocessing
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from supertrend_quant.market_store.ingest import (
    DailyDataSynchronizer,
    EodhdCallBudget,
    EodhdClient,
    EodhdDailySource,
    EodhdQuotaExceeded,
    SecNasdaqSecurityMasterSource,
    SecuritySourceResult,
    SourceArtifact,
    YahooFetchResult,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository


def _claim_budget_in_process(state_path: str, start_event, result_queue) -> None:
    try:
        if not start_event.wait(timeout=10):
            raise RuntimeError("budget test start event timed out")
        used = EodhdCallBudget(
            state_path,
            limit=100,
            reserve=1,
            seed_used=0,
            period="2026-07-18",
        ).claim()
        result_queue.put(("ok", used))
    except BaseException as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


class _Response:
    def __init__(self, content, content_type="text/plain"):
        self.content = content
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        return None


class _Session:
    def __init__(self, responses):
        self.responses = iter(responses)

    def get(self, *args, **kwargs):
        return next(self.responses)


class SecuritySourceTest(unittest.TestCase):
    def test_sec_cik_is_preferred_and_unmatched_listing_gets_persistent_internal_id(self):
        sec = b'{"fields":["cik","name","ticker","exchange"],"data":[[320193,"Apple","AAPL","Nasdaq"]]}'
        nasdaq = (
            "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
            "AAPL|Apple Inc|Q|N|N|100|N|N\n"
            "NOSEC|No SEC ETF|G|N|N|100|Y|N\n"
            "File Creation Time: 0715202621|||||||\n"
        ).encode()
        other = (
            "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
            "SPY|SPDR S&P 500 ETF|P|SPY|Y|100|N|SPY\n"
            "File Creation Time: 0715202621|||||||\n"
        ).encode()
        source = SecNasdaqSecurityMasterSource(
            _Session(
                [
                    _Response(sec, "application/json"),
                    _Response(nasdaq),
                    _Response(other),
                ]
            ),
            user_agent="SuperTrendQuant test@example.com",
        )

        result = source.fetch()

        by_symbol = result.security_master.set_index("primary_symbol")
        self.assertEqual(by_symbol.loc["AAPL", "security_id"], "US:CIK:0000320193")
        self.assertTrue(by_symbol.loc["NOSEC", "security_id"].startswith("US:STQ:"))
        self.assertEqual(by_symbol.loc["SPY", "asset_type"], "ETF")
        self.assertEqual(len(result.artifacts), 3)


class _SecuritySource:
    def fetch(self):
        artifact = SourceArtifact("fixture_master", "fixture://master", "2026-07-15T00:00:00Z", b"master", "text/plain")
        master = pd.DataFrame(
            [
                {
                    "security_id": "SEC-A",
                    "primary_symbol": "AAA",
                    "name": "AAA",
                    "exchange": "XNAS",
                    "asset_type": "STOCK",
                    "currency": "USD",
                    "country": "US",
                    "active_from": "2000-01-01",
                    "active_to": "",
                    "source": "fixture_master",
                    "retrieved_at": artifact.retrieved_at,
                    "source_hash": artifact.source_hash,
                }
            ]
        )
        history = pd.DataFrame(
            [
                {
                    "security_id": "SEC-A",
                    "symbol": "AAA",
                    "exchange": "XNAS",
                    "effective_from": "2000-01-01",
                    "effective_to": "",
                    "source": "fixture_master",
                    "retrieved_at": artifact.retrieved_at,
                    "source_hash": artifact.source_hash,
                }
            ]
        )
        return SecuritySourceResult(master, history, (artifact,))


class _PriceSource:
    def __init__(self):
        self.calls = 0

    def fetch(self, securities, *, start, end, batch_size=100):
        self.calls += 1
        artifact = SourceArtifact("fixture_prices", "fixture://prices", f"2026-07-{14 + self.calls}T00:00:00Z", f"prices-{self.calls}".encode(), "text/csv")
        session = end
        close = 100.0 if self.calls == 1 else 50.0
        prices = pd.DataFrame(
            [
                {
                    "security_id": "SEC-A",
                    "session": session,
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": 100.0,
                    "currency": "USD",
                    "source": artifact.source,
                    "retrieved_at": artifact.retrieved_at,
                    "source_hash": artifact.source_hash,
                }
            ]
        )
        actions = pd.DataFrame(
            columns=[
                "event_id", "security_id", "action_type", "effective_date", "ex_date",
                "cash_amount", "ratio", "currency", "new_security_id", "new_symbol",
                "official", "source", "retrieved_at", "source_hash",
            ]
        )
        if self.calls == 2:
            actions.loc[0] = [
                "split", "SEC-A", "split", session, session, None, 2.0, "USD", "", "",
                False, artifact.source, artifact.retrieved_at, artifact.source_hash,
            ]
        return YahooFetchResult(prices, actions, (artifact,), (),)


class DailySynchronizerTest(unittest.TestCase):
    def test_initial_backfill_then_daily_delta_and_factor_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = LocalDatasetRepository(directory)
            synchronizer = DailyDataSynchronizer(
                repository,
                security_source=_SecuritySource(),
                price_source=_PriceSource(),
            )

            first = synchronizer.sync("2026-07-14", backfill_start="2026-07-14")
            second = synchronizer.sync("2026-07-15")

            self.assertEqual(first.completed_session, "2026-07-14")
            self.assertEqual(second.completed_session, "2026-07-15")
            self.assertEqual(first.row_counts["corporate_actions"], 0)
            self.assertIsNotNone(repository.current_manifest("corporate_actions"))
            self.assertEqual(len(repository.manifest_chain("daily_price_raw")), 2)
            self.assertEqual(len(repository.read_frame("daily_price_raw")), 2)
            factors = repository.read_frame("adjustment_factors")
            prior = factors.loc[pd.to_datetime(factors["session"]).dt.date == pd.Timestamp("2026-07-14").date()]
            self.assertEqual(prior.iloc[0]["split_factor"], 0.5)
            self.assertTrue((Path(directory) / "archives").exists())
            self.assertIsNotNone(repository.current_manifest("source_archive"))


class _EodhdClient:
    def __init__(self):
        self.endpoints = []

    def get_json(self, endpoint, *, params=None):
        self.endpoints.append(endpoint)
        if endpoint.startswith("eod/"):
            return [
                {
                    "date": "2020-08-31",
                    "open": 127.58,
                    "high": 131.0,
                    "low": 126.0,
                    "close": 129.04,
                    "adjusted_close": 125.0,
                    "volume": 225702700,
                }
            ]
        if endpoint.startswith("div/"):
            return [
                {
                    "date": "2020-08-07",
                    "unadjustedValue": 0.82,
                    "currency": "USD",
                    "declarationDate": "2020-07-30",
                    "recordDate": "2020-08-10",
                    "paymentDate": "2020-08-13",
                }
            ]
        return [{"date": "2020-08-31", "split": "4.000000/1.000000"}]

    def safe_url(self, endpoint, *, params=None):
        return f"https://eodhd.test/{endpoint}"


class EodhdDailySourceTest(unittest.TestCase):
    def test_raw_prices_dividends_and_splits_are_normalized(self):
        client = _EodhdClient()
        result = EodhdDailySource(client, workers=1).fetch(
            {"SEC-A": "AAPL.US"},
            start="2020-08-01",
            end="2020-09-10",
        )

        self.assertEqual(result.prices.iloc[0]["close"], 129.04)
        self.assertNotIn("adjusted_close", result.prices)
        actions = result.corporate_actions.set_index("action_type")
        self.assertEqual(actions.loc["cash_dividend", "cash_amount"], 0.82)
        self.assertEqual(actions.loc["split", "ratio"], 4.0)
        self.assertTrue(all("api_token" not in item.source_url for item in result.artifacts))

    def test_action_symbol_override_keeps_price_and_actions_on_distinct_tickers(self):
        client = _EodhdClient()
        EodhdDailySource(
            client,
            workers=1,
            action_symbol_overrides={"SEC-PEAK": "PEAK.US"},
        ).fetch(
            {"SEC-PEAK": "DOC.US"},
            start="2021-01-01",
            end="2021-12-31",
        )

        self.assertEqual(
            client.endpoints,
            ["eod/DOC.US", "div/PEAK.US", "splits/PEAK.US"],
        )


class EodhdCallBudgetTest(unittest.TestCase):
    def test_default_budget_period_uses_utc_date(self):
        class _UtcBoundaryDatetime:
            @classmethod
            def now(cls, timezone):
                self.assertIs(timezone, UTC)
                return datetime(2026, 7, 18, 17, 30, tzinfo=UTC)

        with tempfile.TemporaryDirectory() as directory, patch(
            "supertrend_quant.market_store.ingest.datetime",
            _UtcBoundaryDatetime,
        ):
            budget = EodhdCallBudget(Path(directory) / "budget.json")

        self.assertEqual(budget.period, "2026-07-18")

    def test_persistent_budget_counts_attempts_and_blocks_before_ceiling(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "budget.json"
            session = _Session([_JsonResponse([])])
            budget = EodhdCallBudget(
                state_path,
                limit=3,
                reserve=1,
                seed_used=1,
                period="2026-07-18",
            )
            client = EodhdClient(session, token="test-token", budget=budget)

            self.assertEqual(client.get_json("eod/AAA.US"), [])
            with self.assertRaises(EodhdQuotaExceeded):
                client.get_json("eod/BBB.US")

            persisted = EodhdCallBudget(
                state_path,
                limit=3,
                reserve=1,
                period="2026-07-18",
            )
            with self.assertRaises(EodhdQuotaExceeded):
                persisted.claim()

    def test_persistent_budget_refuses_corrupt_current_period_state(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "budget.json"
            state_path.write_text("{not-json", encoding="utf-8")
            budget = EodhdCallBudget(
                state_path,
                limit=100,
                reserve=5,
                seed_used=10,
                period="2026-07-18",
            )

            with self.assertRaisesRegex(RuntimeError, "state is unreadable"):
                budget.claim()

            self.assertEqual(state_path.read_text(encoding="utf-8"), "{not-json")

    def test_persistent_budget_serializes_claims_across_processes(self):
        if "fork" not in multiprocessing.get_all_start_methods():
            self.skipTest("cross-process budget test requires the POSIX fork method")
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "budget.json"
            context = multiprocessing.get_context("fork")
            start_event = context.Event()
            result_queue = context.Queue()
            processes = [
                context.Process(
                    target=_claim_budget_in_process,
                    args=(str(state_path), start_event, result_queue),
                )
                for _ in range(12)
            ]
            for process in processes:
                process.start()
            start_event.set()
            for process in processes:
                process.join(timeout=15)
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, 0)

            results = [result_queue.get(timeout=5) for _ in processes]
            self.assertEqual([status for status, _ in results], ["ok"] * 12)
            self.assertEqual(sorted(value for _, value in results), list(range(1, 13)))
            persisted = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["used"], 12)

class _JsonResponse:
    def __init__(self, value):
        self.value = value

    def raise_for_status(self):
        return None

    def json(self):
        return self.value


if __name__ == "__main__":
    unittest.main()
