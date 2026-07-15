from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from supertrend_quant.market_store.ingest import (
    DailyDataSynchronizer,
    SecNasdaqSecurityMasterSource,
    SecuritySourceResult,
    SourceArtifact,
    YahooFetchResult,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository


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


if __name__ == "__main__":
    unittest.main()
