from __future__ import annotations

import io
import gzip
import json
import os
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

import supertrend_quant.market_store.kr_providers as kr_providers
from supertrend_quant.market_store.adjustments import (
    build_adjustment_factors,
)
from supertrend_quant.market_store.kr_pipeline import (
    KrCheckpointStore,
    KrBenchmarkThresholds,
    _collect_memberships,
    _collect_krx_prices,
    _archive_allowed_artifacts,
    _combine_krx_reference_price_audits,
    _composite_independent_price_metrics,
    _index_coverage_metadata,
    _classify_cached_delisting_effective_absences,
    _load_kr_official_actions,
    _load_ready_benchmark,
    _kr_dividend_ex_date,
    _kr_dividend_sessions,
    _krx_index_price_gap_policy,
    _kr_corporate_action_audit,
    _krx_reference_price_adjustments,
    _krx_tick_sizes,
    _krx_price_checkpoint_covers_symbols,
    _membership_datasets,
    _normalized_timestamp,
    _officialize_opendart_cash_dividends,
    _provider_scorecard,
    _provider_revision_metrics,
    _reconcile_catalog_with_krx_observations,
    _resolve_kr_lifecycle_candidates,
    _sessions_between,
    compare_provider_to_krx,
    krx_tick_size,
    verify_kr_membership_history,
)
from supertrend_quant.market_store.kr_providers import (
    KrDartDividendResult,
    KrOfficialDataUnavailable,
    KrIdentityCatalog,
    KrProviderResult,
    _attach_unambiguous_dart_codes,
    _attach_current_listing_dates,
    _attach_delisted_isins,
    _dart_mapping_from_corp_code_zip,
    _normalize_krx_etf_master,
    _parse_opendart_cash_dividend_document,
    _reconcile_licensed_identity_intervals,
    _validate_krx_membership_count,
    artifact_from_payload,
    fetch_yahoo_prices,
    fetch_kis_prices,
    fetch_krx_membership,
    fetch_krx_session_prices,
    is_valid_kr_symbol,
    krx_price_evidence_by_symbol,
    normalize_kr_symbol,
    validate_krx_official_configuration,
)
from supertrend_quant.market_store.ingest import SourceArtifact
from supertrend_quant.market_store.manifest import utc_now_iso
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.manifest import DataRelease
from supertrend_quant.market_store.markets import (
    exchange_calendar,
    market_spec,
    recent_sessions,
)
from supertrend_quant.market_store.validation import (
    validate_dataset,
    validate_index_price_gap_policy,
)


def _prices(closes: tuple[float, ...]) -> pd.DataFrame:
    sessions = pd.date_range("2026-01-02", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "security_id": ["KR:ISIN"] * len(closes),
            "symbol": ["005930"] * len(closes),
            "session": sessions.date.astype(str),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100] * len(closes),
            "currency": ["KRW"] * len(closes),
            "source": ["test"] * len(closes),
            "source_url": ["https://example.test"] * len(closes),
            "retrieved_at": ["2026-01-10T00:00:00Z"] * len(closes),
            "source_hash": ["a" * 64] * len(closes),
        }
    )


class KrMarketContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self._env_patch = patch.dict(
            os.environ,
            {
                "KRX_ID": "",
                "KRX_PW": "",
                "KRX_DAILY_PRICE_TRANSPORT": "auto",
            },
            clear=False,
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)
        kr_providers._KRX_STOCK_OPENAPI_AVAILABLE = True
        kr_providers._KRX_ETF_OPENAPI_AVAILABLE = True
        kr_providers._KRX_WEB_LOGIN_FAILURE_DETAIL = ""
        kr_providers._KRX_WEB_LOGIN_FAILURE_UNTIL = 0.0
        kr_providers._KRX_WEB_SESSION = None
        kr_providers._KRX_WEB_SESSION_CREDENTIALS = ""

    def test_market_spec_uses_xkrx_and_krw(self) -> None:
        spec = market_spec("KR")

        self.assertEqual(spec.calendar, "XKRX")
        self.assertEqual(spec.timezone, "Asia/Seoul")
        self.assertEqual(spec.currency, "KRW")

    def test_recent_sessions_excludes_weekend(self) -> None:
        sessions = recent_sessions("KR", 2, end="2026-01-12")

        self.assertEqual(sessions[-1], "2026-01-12")
        self.assertNotIn("2026-01-11", sessions)

    def test_recent_sessions_excludes_2026_constitution_day_closure(self) -> None:
        sessions = recent_sessions("KR", 2, end="2026-07-20")

        self.assertEqual(sessions, ("2026-07-16", "2026-07-20"))

    def test_recent_sessions_excludes_2026_local_election_closure(self) -> None:
        sessions = recent_sessions("KR", 2, end="2026-06-04")

        self.assertEqual(sessions, ("2026-06-02", "2026-06-04"))

    def test_long_range_sessions_apply_the_same_closure_overrides(self) -> None:
        sessions = _sessions_between("2026-06-02", "2026-07-20")

        self.assertNotIn("2026-06-03", sessions)
        self.assertNotIn("2026-07-17", sessions)

    def test_duplicate_catalog_rows_for_same_isin_are_one_identity(self) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:KR7005930003",
                        "primary_symbol": "005930",
                        "active_from": "1975-06-11",
                        "active_to": "",
                    },
                    {
                        "security_id": "KR:KR7005930003",
                        "primary_symbol": "005930",
                        "active_from": "1975-06-11",
                        "active_to": "",
                    },
                ]
            ),
            (),
        )

        self.assertEqual(
            catalog.security_id_for("005930", "2026-01-02"),
            "KR:KR7005930003",
        )

    def test_dart_corp_code_zip_restores_delisted_identity(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "CORPCODE.xml",
                (
                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
                    "<result><list><corp_code>00164469</corp_code>"
                    "<corp_name>현대하이스코</corp_name>"
                    "<stock_code>010520</stock_code>"
                    "<modify_date>20170630</modify_date></list></result>"
                ),
            )

        mapping = _dart_mapping_from_corp_code_zip(buffer.getvalue())
        catalog = pd.DataFrame(
            [
                {
                    "security_id": "KR:KR7010520005",
                    "primary_symbol": "010520",
                    "active_to": "2015-07-15",
                    "identity_mapped": True,
                }
            ]
        )
        restored = _attach_unambiguous_dart_codes(catalog, mapping)

        self.assertEqual(mapping, {"010520": "00164469"})
        self.assertEqual(restored.iloc[0]["dart_corp_code"], "00164469")

    def test_dart_ticker_mapping_does_not_relabel_reused_historical_identity(
        self,
    ) -> None:
        catalog = pd.DataFrame(
            [
                {
                    "security_id": "KR:OLD",
                    "primary_symbol": "000010",
                    "active_to": "2020-01-01",
                    "identity_mapped": True,
                },
                {
                    "security_id": "KR:NEW",
                    "primary_symbol": "000010",
                    "active_to": "",
                    "identity_mapped": True,
                },
            ]
        )

        restored = _attach_unambiguous_dart_codes(
            catalog,
            {"000010": "00999999"},
        )

        self.assertEqual(restored.iloc[0]["dart_corp_code"], "")
        self.assertEqual(restored.iloc[1]["dart_corp_code"], "00999999")

    def test_perfect_secondary_provider_passes_hard_gates(self) -> None:
        baseline = _prices((10_000.0, 10_050.0, 10_100.0))
        result = compare_provider_to_krx(
            baseline,
            KrProviderResult("secondary", "ok", baseline.copy()),
            identity_mapping_rate=1.0,
        )

        self.assertTrue(result["hard_gate_passed"])

    def test_composite_providers_fill_each_others_missing_rows(self) -> None:
        baseline = _prices((10_000.0, 10_100.0, 10_200.0))
        provider_results = {
            "first": KrProviderResult(
                "first",
                "ok",
                baseline.iloc[:2].copy(),
            ),
            "second": KrProviderResult(
                "second",
                "ok",
                baseline.iloc[1:].copy(),
            ),
        }
        provider_metrics = {
            provider: {
                "source_artifacts_complete": True,
                "source_artifact_count": 1,
                "source_hash_count": 1,
                "source_hash_inventory_sha256": provider,
                "source_reproducibility_rate": 1.0,
                "ranking": {
                    "revision_observation": {"score": 1.0}
                },
            }
            for provider in provider_results
        }

        result = _composite_independent_price_metrics(
            baseline,
            provider_results,
            provider_metrics,
            identity_mapping_rate=1.0,
            thresholds=KrBenchmarkThresholds(),
        )

        self.assertIsNotNone(result)
        assert result is not None
        name, metrics, _ = result
        self.assertEqual(name, "composite:first+second")
        self.assertTrue(metrics["hard_gate_passed"])
        self.assertEqual(metrics["verified_row_count"], 3)
        self.assertEqual(metrics["single_provider_verified_count"], 2)
        self.assertEqual(metrics["multiple_provider_verified_count"], 1)

    def test_composite_blocks_a_row_no_provider_matches(self) -> None:
        baseline = _prices((10_000.0, 10_100.0, 10_200.0))
        first = baseline.iloc[:2].copy()
        second = baseline.iloc[1:].copy()
        second.loc[
            second["session"].eq(baseline.iloc[-1]["session"]),
            ["open", "high", "low", "close"],
        ] = 20_000.0
        provider_results = {
            "first": KrProviderResult("first", "ok", first),
            "second": KrProviderResult("second", "ok", second),
        }
        provider_metrics = {
            provider: {
                "source_artifacts_complete": True,
                "source_artifact_count": 1,
                "source_hash_count": 1,
                "source_hash_inventory_sha256": provider,
                "source_reproducibility_rate": 1.0,
                "ranking": {
                    "revision_observation": {"score": 1.0}
                },
            }
            for provider in provider_results
        }

        result = _composite_independent_price_metrics(
            baseline,
            provider_results,
            provider_metrics,
            identity_mapping_rate=1.0,
            thresholds=KrBenchmarkThresholds(),
        )

        self.assertIsNotNone(result)
        assert result is not None
        _, metrics, _ = result
        self.assertFalse(metrics["hard_gate_passed"])
        self.assertEqual(metrics["unclassified_missing"], 1)
        self.assertEqual(
            metrics["unresolved_cross_source_disagreement_count"],
            1,
        )

    def test_composite_quarantines_provider_only_anomalies(self) -> None:
        baseline = _prices((10_000.0, 10_100.0))
        first = baseline.copy()
        phantom = first.iloc[[0]].copy()
        phantom["session"] = "2026-01-03"
        first = pd.concat([first, phantom], ignore_index=True)
        provider_results = {
            "first": KrProviderResult("first", "ok", first),
            "second": KrProviderResult(
                "second",
                "ok",
                baseline.copy(),
            ),
        }
        provider_metrics = {
            provider: {
                "source_artifacts_complete": True,
                "source_artifact_count": 1,
                "source_hash_count": 1,
                "source_hash_inventory_sha256": provider,
                "source_reproducibility_rate": 1.0,
                "ranking": {
                    "revision_observation": {"score": 1.0}
                },
            }
            for provider in provider_results
        }

        result = _composite_independent_price_metrics(
            baseline,
            provider_results,
            provider_metrics,
            identity_mapping_rate=1.0,
            thresholds=KrBenchmarkThresholds(),
        )

        self.assertIsNotNone(result)
        assert result is not None
        _, metrics, _ = result
        self.assertTrue(metrics["hard_gate_passed"])
        self.assertEqual(metrics["unexpected_provider_observations"], 1)
        self.assertEqual(
            metrics["quarantined_provider_observation_count"],
            1,
        )

    def test_provider_observation_outside_krx_inventory_fails_gate(self) -> None:
        baseline = _prices((10_000.0, 10_050.0, 10_100.0))
        actual = baseline.copy()
        phantom = actual.iloc[[0]].copy()
        phantom["session"] = "2026-01-03"
        actual = pd.concat([actual, phantom], ignore_index=True)

        result = compare_provider_to_krx(
            baseline,
            KrProviderResult("secondary", "ok", actual),
            identity_mapping_rate=1.0,
        )

        self.assertEqual(result["unexpected_provider_observations"], 1)
        self.assertFalse(result["hard_gate_passed"])

    def test_provider_volume_mismatch_fails_gate(self) -> None:
        baseline = _prices((10_000.0, 10_050.0))
        actual = baseline.copy()
        actual.loc[0, "volume"] = 101

        result = compare_provider_to_krx(
            baseline,
            KrProviderResult("secondary", "ok", actual),
            identity_mapping_rate=1.0,
        )

        self.assertEqual(result["close_within_one_tick_rate"], 1.0)
        self.assertEqual(result["ohlc_within_one_tick_rate"], 1.0)
        self.assertEqual(result["volume_exact_rate"], 0.5)
        self.assertFalse(result["hard_gate_passed"])

    def test_traded_bar_on_official_no_trade_observation_fails_gate(self) -> None:
        baseline = _prices((10_000.0, 10_050.0))
        baseline.loc[0, "observation_status"] = "suspended_or_no_trade"
        actual = _prices((10_000.0, 10_050.0))

        result = compare_provider_to_krx(
            baseline,
            KrProviderResult("secondary", "ok", actual),
            identity_mapping_rate=1.0,
        )

        self.assertEqual(
            result["misclassified_official_no_trade_observations"], 1
        )
        self.assertFalse(result["hard_gate_passed"])
        self.assertEqual(result["unclassified_missing"], 0)
        self.assertEqual(result["unexplained_large_discontinuities"], 0)

    def test_missing_source_hash_fails_reproducibility_gate(self) -> None:
        baseline = _prices((10_000.0, 10_050.0))
        candidate = baseline.copy()
        candidate["source_hash"] = "not-a-sha256"

        result = compare_provider_to_krx(
            baseline,
            KrProviderResult("secondary", "ok", candidate),
            identity_mapping_rate=1.0,
        )

        self.assertFalse(result["hard_gate_passed"])
        self.assertEqual(result["source_reproducibility_rate"], 0.0)

    def test_repeat_provider_sample_measures_raw_revision_stability(self) -> None:
        previous = _prices((10_000.0, 10_050.0, 10_100.0))
        unchanged = _provider_revision_metrics(previous, previous.copy())
        revised_frame = previous.copy()
        revised_frame.loc[1, "close"] = 10_040.0
        revised = _provider_revision_metrics(previous, revised_frame)

        self.assertEqual(unchanged["score"], 1.0)
        self.assertEqual(revised["revised_key_count"], 1)
        self.assertAlmostEqual(revised["score"], 2 / 3)

    def test_krx_tick_uses_market_product_and_rule_date(self) -> None:
        self.assertEqual(
            krx_tick_size(150_000, exchange="KOSPI", session="2026-01-02"),
            500,
        )
        self.assertEqual(
            krx_tick_size(150_000, exchange="KOSDAQ", session="2026-01-02"),
            100,
        )
        self.assertEqual(
            krx_tick_size(15_000, exchange="KOSPI", session="2022-12-30"),
            10,
        )
        self.assertEqual(
            krx_tick_size(
                150_000,
                exchange="KOSPI",
                asset_type="ETF",
                session="2026-01-02",
            ),
            5,
        )

    def test_lifecycle_review_date_normalizes_to_utc_timestamp(self) -> None:
        self.assertEqual(
            _normalized_timestamp("2026-07-23"),
            "2026-07-23T00:00:00.000000Z",
        )

    def test_vectorized_krx_ticks_match_scalar_schedule(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "session": "2026-01-02",
                    "close": 150_000,
                    "exchange": "KOSPI",
                    "asset_type": "STOCK",
                },
                {
                    "session": "2026-01-02",
                    "close": 150_000,
                    "exchange": "KOSDAQ",
                    "asset_type": "STOCK",
                },
                {
                    "session": "2022-12-30",
                    "close": 15_000,
                    "exchange": "KOSPI",
                    "asset_type": "STOCK",
                },
                {
                    "session": "2026-01-02",
                    "close": 150_000,
                    "exchange": "KOSPI",
                    "asset_type": "ETF",
                },
            ]
        )

        vectorized = _krx_tick_sizes(
            frame,
            price_column="close",
            exchange_column="exchange",
            asset_type_column="asset_type",
        )
        scalar = frame.apply(
            lambda row: krx_tick_size(
                row["close"],
                exchange=row["exchange"],
                asset_type=row["asset_type"],
                session=row["session"],
            ),
            axis=1,
        )

        pd.testing.assert_series_equal(
            vectorized,
            scalar,
            check_names=False,
        )

    def test_missing_row_fails_coverage_and_missing_gates(self) -> None:
        baseline = _prices((10_000.0, 10_050.0, 10_100.0))
        result = compare_provider_to_krx(
            baseline,
            KrProviderResult("secondary", "ok", baseline.iloc[:-1].copy()),
            identity_mapping_rate=1.0,
        )

        self.assertFalse(result["hard_gate_passed"])
        self.assertEqual(result["unclassified_missing"], 1)

    def test_matching_raw_jump_is_classified_but_provider_only_jump_is_not(self) -> None:
        baseline = _prices((10_000.0, 5_000.0, 5_100.0))
        matched = compare_provider_to_krx(
            baseline,
            KrProviderResult("secondary", "ok", baseline.copy()),
            identity_mapping_rate=1.0,
        )
        provider_only = _prices((10_000.0, 7_000.0, 7_100.0))
        unmatched = compare_provider_to_krx(
            baseline,
            KrProviderResult("secondary", "ok", provider_only),
            identity_mapping_rate=1.0,
            thresholds=KrBenchmarkThresholds(
                close_within_one_tick_rate=0.0,
                max_large_cross_source_discrepancies=10,
                large_return_threshold=0.25,
            ),
        )

        self.assertEqual(matched["unexplained_large_discontinuities"], 0)
        self.assertEqual(unmatched["unexplained_large_discontinuities"], 1)
        self.assertFalse(unmatched["hard_gate_passed"])

    def test_kis_is_explicitly_skipped_without_credentials(self) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                columns=[
                    "security_id",
                    "primary_symbol",
                    "active_from",
                    "active_to",
                ]
            ),
            (),
        )
        with (
            patch("supertrend_quant.market_store.kr_providers.load_env"),
            patch.dict(
                os.environ,
                {"KIS_APP_KEY": "", "KIS_APP_SECRET": ""},
                clear=False,
            ),
        ):
            result = fetch_kis_prices(catalog, start="2026-01-01", end="2026-01-02")

        self.assertEqual(result.status, "skipped_missing_credentials")

    def test_provider_checkpoint_restores_only_scoped_artifacts(self) -> None:
        used = SourceArtifact(
            source="provider",
            source_url="https://example.test/used",
            retrieved_at=utc_now_iso(),
            content=b'{"used":true}',
            content_type="application/json",
        )
        stale = SourceArtifact(
            source="provider",
            source_url="https://example.test/stale",
            retrieved_at=utc_now_iso(),
            content=b'{"stale":true}',
            content_type="application/json",
        )
        prices = _prices((10_000.0,))
        prices["source_hash"] = used.source_hash
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = KrCheckpointStore(directory)
            checkpoint.save_provider_result(
                KrProviderResult(
                    "secondary",
                    "ok",
                    prices,
                    (used,),
                )
            )
            checkpoint.save_local_artifact(
                stale,
                scope="secondary",
            )

            restored = checkpoint.load_provider_result("secondary")

        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(
            [value.source_hash for value in restored.artifacts],
            [used.source_hash],
        )

    def test_kis_requests_original_unadjusted_prices(self) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:KR7005930003",
                        "primary_symbol": "005930",
                        "active_from": "1975-06-11",
                        "active_to": "",
                        "exchange": "KOSPI",
                        "asset_type": "STOCK",
                    }
                ]
            ),
            (),
        )
        token_response = Mock()
        token_response.raise_for_status.return_value = None
        token_response.json.return_value = {"access_token": "token"}
        price_response = Mock()
        price_response.raise_for_status.return_value = None
        price_response.json.return_value = {"output2": []}
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("supertrend_quant.market_store.kr_providers.load_env"),
                patch.dict(
                    os.environ,
                    {
                        "KIS_APP_KEY": "key",
                        "KIS_APP_SECRET": "secret",
                        "KIS_PROD_APP_KEY": "",
                        "KIS_PROD_APP_SECRET": "",
                        "KIS_MARKET_DATA_MODE": "prod",
                        "KIS_TOKEN_CACHE_PATH": str(Path(directory) / "token.json"),
                    },
                    clear=False,
                ),
                patch("requests.post", return_value=token_response),
                patch("requests.get", return_value=price_response) as get,
                patch("supertrend_quant.market_store.kr_providers.time.sleep"),
            ):
                fetch_kis_prices(catalog, start="2026-01-02", end="2026-01-02")

        self.assertEqual(get.call_args.kwargs["params"]["FID_ORG_ADJ_PRC"], "1")

    def test_kis_full_history_chunks_resume_without_refetch(self) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:KR7005930003",
                        "primary_symbol": "005930",
                        "active_from": "1975-06-11",
                        "active_to": "",
                        "exchange": "KOSPI",
                        "asset_type": "STOCK",
                    }
                ]
            ),
            (),
        )
        token_response = Mock()
        token_response.raise_for_status.return_value = None
        token_response.json.return_value = {
            "access_token": "token",
            "expires_in": 86_400,
        }
        price_response = Mock()
        price_response.raise_for_status.return_value = None
        price_response.json.return_value = {
            "rt_cd": "0",
            "output2": [
                {
                    "stck_bsop_date": "20260102",
                    "stck_oprc": "70000",
                    "stck_hgpr": "71000",
                    "stck_lwpr": "69000",
                    "stck_clpr": "70500",
                    "acml_vol": "1000",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("supertrend_quant.market_store.kr_providers.load_env"),
                patch.dict(
                    os.environ,
                    {
                        "KIS_PROD_APP_KEY": "key",
                        "KIS_PROD_APP_SECRET": "secret",
                        "KIS_MARKET_DATA_MODE": "prod",
                        "KIS_TOKEN_CACHE_PATH": str(root / "token.json"),
                    },
                    clear=False,
                ),
                patch("requests.post", return_value=token_response),
                patch("requests.get", return_value=price_response) as get,
                patch("supertrend_quant.market_store.kr_providers.time.sleep"),
            ):
                first = fetch_kis_prices(
                    catalog,
                    start="2026-01-02",
                    end="2026-01-02",
                    checkpoint_root=root,
                )
                second = fetch_kis_prices(
                    catalog,
                    start="2026-01-02",
                    end="2026-01-02",
                    checkpoint_root=root,
                )

            self.assertEqual(get.call_count, 1)
            self.assertEqual(first.status, "ok")
            self.assertEqual(second.status, "ok")
            self.assertIn("fetched_chunks=1", first.detail)
            self.assertIn("cached_chunks=1", second.detail)
            self.assertEqual(
                first.prices[["security_id", "session", "close"]].to_dict(
                    "records"
                ),
                second.prices[
                    ["security_id", "session", "close"]
                ].to_dict("records"),
            )
            self.assertEqual(
                len(list((root / "evidence_local" / "kis").glob("*.gz"))),
                1,
            )

    def test_kis_reuses_credential_bound_token_cache(self) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                columns=[
                    "security_id",
                    "primary_symbol",
                    "active_from",
                    "active_to",
                ]
            ),
            (),
        )
        token_response = Mock()
        token_response.raise_for_status.return_value = None
        token_response.json.return_value = {
            "access_token": "cached-token",
            "expires_in": 86_400,
        }
        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "token.json"
            with (
                patch("supertrend_quant.market_store.kr_providers.load_env"),
                patch.dict(
                    os.environ,
                    {
                        "KIS_PROD_APP_KEY": "prod-key",
                        "KIS_PROD_APP_SECRET": "prod-secret",
                        "KIS_MARKET_DATA_MODE": "prod",
                        "KIS_TOKEN_CACHE_PATH": str(token_path),
                    },
                    clear=False,
                ),
                patch("requests.post", return_value=token_response) as post,
            ):
                first = fetch_kis_prices(
                    catalog, start="2026-01-01", end="2026-01-02"
                )
                second = fetch_kis_prices(
                    catalog, start="2026-01-01", end="2026-01-02"
                )

            self.assertEqual(first.status, "failed")
            self.assertEqual(second.status, "failed")
            self.assertEqual(post.call_count, 1)
            self.assertEqual(token_path.stat().st_mode & 0o777, 0o600)

    def test_kis_retries_transient_price_http_error(self) -> None:
        import requests

        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:KR7005930003",
                        "primary_symbol": "005930",
                        "active_from": "1975-06-11",
                        "active_to": "",
                        "exchange": "KOSPI",
                        "asset_type": "STOCK",
                    }
                ]
            ),
            (),
        )
        token_response = Mock()
        token_response.raise_for_status.return_value = None
        token_response.json.return_value = {
            "access_token": "token",
            "expires_in": 86_400,
        }
        server_error_response = Mock(status_code=500)
        server_error = requests.HTTPError(response=server_error_response)
        price_response = Mock()
        price_response.raise_for_status.return_value = None
        price_response.json.return_value = {"rt_cd": "0", "output2": []}
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("supertrend_quant.market_store.kr_providers.load_env"),
                patch.dict(
                    os.environ,
                    {
                        "KIS_PROD_APP_KEY": "key",
                        "KIS_PROD_APP_SECRET": "secret",
                        "KIS_MARKET_DATA_MODE": "prod",
                        "KIS_TOKEN_CACHE_PATH": str(Path(directory) / "token.json"),
                    },
                    clear=False,
                ),
                patch("requests.post", return_value=token_response),
                patch(
                    "requests.get",
                    side_effect=[server_error, price_response],
                ) as get,
                patch("supertrend_quant.market_store.kr_providers.time.sleep"),
            ):
                fetch_kis_prices(
                    catalog, start="2026-01-02", end="2026-01-02"
                )

        self.assertEqual(get.call_count, 2)

    def test_kis_bisects_a_persistently_broken_price_window(self) -> None:
        import requests

        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:KR7005930003",
                        "primary_symbol": "005930",
                        "active_from": "1975-06-11",
                        "active_to": "",
                        "exchange": "KOSPI",
                        "asset_type": "STOCK",
                    }
                ]
            ),
            (),
        )
        token_response = Mock()
        token_response.raise_for_status.return_value = None
        token_response.json.return_value = {
            "access_token": "token",
            "expires_in": 86_400,
        }
        server_error_response = Mock(status_code=500)
        server_error = requests.HTTPError(
            response=server_error_response
        )

        def successful_response(session: str) -> Mock:
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "rt_cd": "0",
                "output2": [
                    {
                        "stck_bsop_date": session,
                        "stck_oprc": "70000",
                        "stck_hgpr": "71000",
                        "stck_lwpr": "69000",
                        "stck_clpr": "70500",
                        "acml_vol": "1000",
                    }
                ],
            }
            return response

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "supertrend_quant.market_store.kr_providers.load_env"
                ),
                patch.dict(
                    os.environ,
                    {
                        "KIS_PROD_APP_KEY": "key",
                        "KIS_PROD_APP_SECRET": "secret",
                        "KIS_MARKET_DATA_MODE": "prod",
                        "KIS_TOKEN_CACHE_PATH": str(
                            Path(directory) / "token.json"
                        ),
                    },
                    clear=False,
                ),
                patch("requests.post", return_value=token_response),
                patch(
                    "requests.get",
                    side_effect=[
                        *([server_error] * 8),
                        successful_response("20260102"),
                        successful_response("20260104"),
                    ],
                ) as get,
                patch(
                    "supertrend_quant.market_store.kr_providers.time.sleep"
                ),
            ):
                result = fetch_kis_prices(
                    catalog,
                    start="2026-01-01",
                    end="2026-01-04",
                )

        self.assertEqual(get.call_count, 10)
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.prices), 2)
        self.assertIn(
            "segments",
            json.loads(result.artifacts[0].content),
        )

    def test_kis_zero_volume_bar_is_classified_as_no_trade(self) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:KR7005930003",
                        "primary_symbol": "005930",
                        "active_from": "1975-06-11",
                        "active_to": "",
                        "exchange": "KOSPI",
                        "asset_type": "STOCK",
                    }
                ]
            ),
            (),
        )
        token_response = Mock()
        token_response.raise_for_status.return_value = None
        token_response.json.return_value = {
            "access_token": "token",
            "expires_in": 86_400,
        }
        price_response = Mock()
        price_response.raise_for_status.return_value = None
        price_response.json.return_value = {
            "rt_cd": "0",
            "output2": [
                {
                    "stck_bsop_date": "20260502",
                    "stck_oprc": "53000",
                    "stck_hgpr": "53000",
                    "stck_lwpr": "53000",
                    "stck_clpr": "53000",
                    "acml_vol": "0",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("supertrend_quant.market_store.kr_providers.load_env"),
                patch.dict(
                    os.environ,
                    {
                        "KIS_PROD_APP_KEY": "key",
                        "KIS_PROD_APP_SECRET": "secret",
                        "KIS_MARKET_DATA_MODE": "prod",
                        "KIS_TOKEN_CACHE_PATH": str(Path(directory) / "token.json"),
                    },
                    clear=False,
                ),
                patch("requests.post", return_value=token_response),
                patch("requests.get", return_value=price_response),
                patch("supertrend_quant.market_store.kr_providers.time.sleep"),
            ):
                result = fetch_kis_prices(
                    catalog, start="2026-05-02", end="2026-05-02"
                )

        self.assertEqual(result.status, "ok")
        self.assertEqual(
            result.prices.iloc[0]["observation_status"],
            "suspended_or_no_trade",
        )

    def test_krx_official_inputs_fail_closed_without_credentials(self) -> None:
        with (
            patch("supertrend_quant.market_store.kr_providers.load_env"),
            patch.dict(
                os.environ,
                {
                    "KRX_OPENAPI_AUTH_KEY": "",
                    "KRX_ID": "",
                    "KRX_PW": "",
                    "KRX_PIT_CONSTITUENTS_PATH": "",
                },
                clear=False,
            ),
            self.assertRaisesRegex(KrOfficialDataUnavailable, "KRX_OPENAPI_AUTH_KEY"),
        ):
            validate_krx_official_configuration()

    def test_krx_web_credentials_replace_the_missing_licensed_snapshot(self) -> None:
        with (
            patch("supertrend_quant.market_store.kr_providers.load_env"),
            patch.dict(
                os.environ,
                {
                    "KRX_OPENAPI_AUTH_KEY": "test-key",
                    "KRX_ID": "login-id",
                    "KRX_PW": "login-password",
                    "KRX_PIT_CONSTITUENTS_PATH": "",
                },
                clear=False,
            ),
        ):
            self.assertIsNone(validate_krx_official_configuration())

    def test_krx_web_session_cache_is_private_and_avoids_new_login(self) -> None:
        login_id = "login-id"
        login_pw = "login-password"
        fingerprint = kr_providers.sha256_bytes(
            f"{login_id}\0{login_pw}".encode("utf-8")
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "krx-session.json"
            kr_providers._write_krx_web_session_cache(
                path,
                credential_fingerprint=fingerprint,
                cookies={"JSESSIONID": "cached-cookie"},
            )

            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertIsNone(
                kr_providers._read_krx_web_session_cache(
                    path,
                    credential_fingerprint="wrong-fingerprint",
                )
            )
            with (
                patch("supertrend_quant.market_store.kr_providers.load_env"),
                patch.dict(
                    os.environ,
                    {
                        "KRX_ID": login_id,
                        "KRX_PW": login_pw,
                        "KRX_SESSION_CACHE_PATH": str(path),
                    },
                    clear=False,
                ),
                patch.multiple(
                    kr_providers,
                    _KRX_WEB_SESSION=None,
                    _KRX_WEB_SESSION_CREDENTIALS="",
                    _KRX_WEB_LOGIN_AT=0.0,
                    _KRX_WEB_LAST_USED_AT=0.0,
                    _KRX_WEB_CACHE_SAVED_AT=0.0,
                    _KRX_WEB_LOGIN_FAILURE_UNTIL=0.0,
                    _KRX_WEB_LOGIN_FAILURE_DETAIL="",
                ),
                patch.object(
                    kr_providers,
                    "_login_krx_web_locked",
                ) as login,
            ):
                session = kr_providers._krx_web_session_locked()

            login.assert_not_called()
            self.assertEqual(
                session.cookies.get_dict().get("JSESSIONID"),
                "cached-cookie",
            )
            session.close()

    def test_krx_login_failure_has_cross_worker_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("supertrend_quant.market_store.kr_providers.load_env"),
                patch.dict(
                    os.environ,
                    {
                        "KRX_ID": "login-id",
                        "KRX_PW": "login-password",
                        "KRX_SESSION_CACHE_PATH": str(
                            Path(tmp) / "missing-session.json"
                        ),
                    },
                    clear=False,
                ),
                patch.multiple(
                    kr_providers,
                    _KRX_WEB_SESSION=None,
                    _KRX_WEB_SESSION_CREDENTIALS="",
                    _KRX_WEB_LOGIN_AT=0.0,
                    _KRX_WEB_LAST_USED_AT=0.0,
                    _KRX_WEB_CACHE_SAVED_AT=0.0,
                    _KRX_WEB_LOGIN_FAILURE_UNTIL=0.0,
                    _KRX_WEB_LOGIN_FAILURE_DETAIL="",
                ),
                patch.object(
                    kr_providers,
                    "_login_krx_web_locked",
                    side_effect=KrOfficialDataUnavailable("HTTP 403"),
                ) as login,
            ):
                with self.assertRaisesRegex(
                    KrOfficialDataUnavailable,
                    "HTTP 403",
                ):
                    kr_providers._krx_web_session_locked()
                with self.assertRaisesRegex(
                    KrOfficialDataUnavailable,
                    "retry suppressed",
                ):
                    kr_providers._krx_web_session_locked()

            self.assertEqual(login.call_count, 1)

    def test_authenticated_krx_web_snapshot_is_parsed_and_hashed(self) -> None:
        response = {
            "output": [
                {
                    "ISU_SRT_CD": f"{index:06d}",
                    "ISU_ABBRV": f"종목{index}",
                }
                for index in range(1, 200)
            ]
            + [
                {
                    "ISU_SRT_CD": "0126Z0",
                    "ISU_ABBRV": "삼성에피스홀딩스",
                }
            ]
        }
        with (
            patch("supertrend_quant.market_store.kr_providers.load_env"),
            patch.dict(
                os.environ,
                {
                    "KRX_ID": "login-id",
                    "KRX_PW": "login-password",
                    "KRX_PIT_CONSTITUENTS_PATH": "",
                },
                clear=False,
            ),
            patch(
                "supertrend_quant.market_store.kr_providers._post_krx_web_json",
                return_value=response,
            ) as post,
        ):
            symbols, artifact = fetch_krx_membership(
                "kospi200", "2015-01-02", sleep_seconds=0.25
            )

        self.assertEqual(len(symbols), 200)
        self.assertEqual(symbols[0], "000001")
        self.assertIn("0126Z0", symbols)
        self.assertEqual(
            artifact.source,
            "krx_authenticated_web_index_constituents",
        )
        self.assertRegex(artifact.source_hash, r"^[0-9a-f]{64}$")
        self.assertEqual(
            post.call_args.args[0],
            {
                "bld": "dbms/MDC/STAT/standard/MDCSTAT00601",
                "indIdx": "1",
                "indIdx2": "028",
                "trdDd": "20150102",
            },
        )

    def test_krx_short_code_accepts_numeric_and_alphanumeric_symbols(self) -> None:
        self.assertEqual(normalize_kr_symbol(5930), "005930")
        self.assertEqual(normalize_kr_symbol("0126z0"), "0126Z0")
        self.assertTrue(is_valid_kr_symbol("005930"))
        self.assertTrue(is_valid_kr_symbol("0126Z0"))
        self.assertFalse(is_valid_kr_symbol("삼성01"))

    def test_krx_description_listing_dates_enrich_isin_identity(self) -> None:
        identities = pd.DataFrame(
            [
                {
                    "security_id": "KR:KR70126Z0002",
                    "primary_symbol": "0126Z0",
                    "active_from": "",
                }
            ]
        )
        descriptions = pd.DataFrame(
            [
                {
                    "Code": "0126Z0",
                    "ListingDate": "2025-11-24",
                }
            ]
        )

        enriched = _attach_current_listing_dates(identities, descriptions)

        self.assertEqual(enriched.iloc[0]["active_from"], "2025-11-24")

    def test_krx_delisted_finder_enriches_interval_with_isin(self) -> None:
        identities = pd.DataFrame(
            [
                {
                    "security_id": "KR:UNMAPPED:KOSPI:010620",
                    "primary_symbol": "010620",
                    "name": "HD현대미포",
                    "identity_mapped": False,
                    "isin": "",
                    "source": "krx_security_listing",
                    "source_url": "https://data.krx.co.kr/",
                    "retrieved_at": "2026-07-23T00:00:00Z",
                    "source_hash": "a" * 64,
                }
            ]
        )
        payload = {
            "block1": [
                {
                    "full_code": "KR7010620003",
                    "short_code": "010620",
                    "codeName": "HD현대미포",
                }
            ]
        }

        enriched = _attach_delisted_isins(
            identities,
            payload,
            retrieved_at="2026-07-23T01:00:00Z",
            source_hash="b" * 64,
        )

        self.assertEqual(enriched.iloc[0]["security_id"], "KR:KR7010620003")
        self.assertTrue(enriched.iloc[0]["identity_mapped"])
        self.assertEqual(enriched.iloc[0]["source_hash"], "b" * 64)

    def test_krx_membership_allows_transient_corporate_action_count(self) -> None:
        _validate_krx_membership_count(
            normalized_profile="kospi200",
            count=201,
            context="test snapshot",
        )
        with self.assertRaisesRegex(
            KrOfficialDataUnavailable,
            r"expected 195-205",
        ):
            _validate_krx_membership_count(
                normalized_profile="kospi200",
                count=206,
                context="test snapshot",
            )

    def test_krx_web_pre_inception_kosdaq150_is_explicitly_empty(self) -> None:
        with (
            patch("supertrend_quant.market_store.kr_providers.load_env"),
            patch.dict(
                os.environ,
                {
                    "KRX_ID": "login-id",
                    "KRX_PW": "login-password",
                    "KRX_PIT_CONSTITUENTS_PATH": "",
                },
                clear=False,
            ),
            patch(
                "supertrend_quant.market_store.kr_providers._post_krx_web_json",
                return_value={"output": []},
            ),
            self.assertRaisesRegex(
                KrOfficialDataUnavailable,
                r"announced on 2015-07-13",
            ),
        ):
            fetch_krx_membership("kosdaq150", "2015-01-02")

    def test_krx_http_error_preserves_safe_api_diagnostic(self) -> None:
        import requests

        catalog = KrIdentityCatalog(
            pd.DataFrame(
                columns=[
                    "security_id",
                    "primary_symbol",
                    "active_from",
                    "active_to",
                ]
            ),
            (),
        )
        response = Mock()
        response.json.return_value = {
            "respCode": "401",
            "respMsg": "Unauthorized API Call",
        }
        error = requests.HTTPError(response=response)
        response.raise_for_status.side_effect = error
        with (
            patch("supertrend_quant.market_store.kr_providers.load_env"),
            patch.dict(
                os.environ,
                {"KRX_OPENAPI_AUTH_KEY": "secret"},
                clear=False,
            ),
            patch("requests.get", return_value=response),
            self.assertRaisesRegex(
                KrOfficialDataUnavailable,
                r"401 Unauthorized API Call",
            ),
        ):
            fetch_krx_session_prices(
                "2026-01-02",
                catalog,
                symbols=(),
            )

    def test_licensed_krx_snapshot_replays_latest_complete_membership(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "constituents.csv"
            rows = [
                {
                    "profile": "kospi200",
                    "session": "2026-01-02",
                    "symbol": f"{index:06d}",
                    "isin": f"KR7{index:09d}",
                }
                for index in range(1, 201)
            ]
            pd.DataFrame(rows).to_csv(path, index=False)
            with (
                patch("supertrend_quant.market_store.kr_providers.load_env"),
                patch.dict(
                    os.environ,
                    {
                        "KRX_OPENAPI_AUTH_KEY": "test-key",
                        "KRX_ID": "",
                        "KRX_PW": "",
                        "KRX_PIT_CONSTITUENTS_PATH": str(path),
                    },
                    clear=False,
                ),
            ):
                symbols, artifact = fetch_krx_membership(
                    "kospi200", "2026-01-05"
                )

        self.assertEqual(len(symbols), 200)
        self.assertEqual(symbols[0], "000001")
        self.assertEqual(artifact.source, "krx_licensed_index_constituents")

    def test_authenticated_krx_openapi_prices_are_parsed_as_raw_ohlcv(self) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:KR7005930003",
                        "primary_symbol": "005930",
                        "active_from": "1975-06-11",
                        "active_to": "",
                    },
                    {
                        "security_id": "KR:KR7035720002",
                        "primary_symbol": "035720",
                        "active_from": "2017-07-10",
                        "active_to": "",
                    },
                ]
            ),
            (),
        )
        responses = []
        for symbol in ("005930", "035720"):
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "OutBlock_1": [
                    {
                        "ISU_CD": symbol,
                        "TDD_OPNPRC": "10,000",
                        "TDD_HGPRC": "10,100",
                        "TDD_LWPRC": "9,900",
                        "TDD_CLSPRC": "10,050",
                        "ACC_TRDVOL": "1,234",
                    }
                ]
            }
            responses.append(response)
        with (
            patch("supertrend_quant.market_store.kr_providers.load_env"),
            patch.dict(os.environ, {"KRX_OPENAPI_AUTH_KEY": "secret"}, clear=False),
            patch("requests.get", side_effect=responses) as get,
        ):
            frame, artifact = fetch_krx_session_prices(
                "2026-01-02", catalog, symbols=("005930", "035720")
            )

        self.assertEqual(len(frame), 2)
        self.assertEqual(set(frame["security_id"]), {"KR:KR7005930003", "KR:KR7035720002"})
        self.assertTrue(frame["close"].eq(10_050).all())
        self.assertEqual(artifact.source, "krx_openapi_daily_ohlcv")
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args_list[0].kwargs["headers"], {"AUTH_KEY": "secret"})

    def test_krx_daily_row_uses_its_official_isin_and_historical_market(self) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:KR7035720002",
                        "primary_symbol": "035720",
                        "name": "카카오",
                        "active_from": "2017-07-10",
                        "active_to": "",
                        "exchange": "KOSPI",
                        "asset_type": "STOCK",
                        "identity_mapped": True,
                    }
                ]
            ),
            (),
        )
        with (
            patch("supertrend_quant.market_store.kr_providers.load_env"),
            patch.dict(
                os.environ,
                {
                    "KRX_OPENAPI_AUTH_KEY": "secret",
                    "KRX_DAILY_PRICE_TRANSPORT": "web",
                    "KRX_ID": "login-id",
                    "KRX_PW": "login-password",
                },
                clear=False,
            ),
            patch(
                "supertrend_quant.market_store.kr_providers._post_krx_web_json",
                return_value={
                    "OutBlock_1": [
                        {
                            "ISU_CD": "KR7035720002",
                            "ISU_SRT_CD": "035720",
                            "ISU_ABBRV": "다음카카오",
                            "MKT_NM": "KOSDAQ",
                            "TDD_OPNPRC": "124,400",
                            "TDD_HGPRC": "138,700",
                            "TDD_LWPRC": "124,300",
                            "TDD_CLSPRC": "137,200",
                            "ACC_TRDVOL": "1,162,901",
                        }
                    ]
                },
            ),
        ):
            frame, _ = fetch_krx_session_prices(
                "2015-01-02",
                catalog,
                symbols=("035720",),
            )

        self.assertEqual(frame.iloc[0]["security_id"], "KR:KR7035720002")
        self.assertEqual(frame.iloc[0]["exchange"], "KOSDAQ")
        self.assertEqual(frame.iloc[0]["security_name"], "다음카카오")

    def test_krx_zero_regular_ohl_with_volume_is_classified(self) -> None:
        evidence = krx_price_evidence_by_symbol(
            {
                "responses": {
                    "STOCK": {
                        "OutBlock_1": [
                            {
                                "ISU_CD": "KR7016380008",
                                "ISU_SRT_CD": "016380",
                                "ISU_ABBRV": "동부제철",
                                "MKT_NM": "KOSPI",
                                "TDD_OPNPRC": "0",
                                "TDD_HGPRC": "0",
                                "TDD_LWPRC": "0",
                                "TDD_CLSPRC": "2,500",
                                "ACC_TRDVOL": "437",
                            }
                        ]
                    }
                }
            }
        )

        self.assertEqual(
            evidence["016380"]["observation_status"],
            "no_regular_session_ohlc",
        )

    def test_authenticated_krx_etf_master_supplies_stable_identity(self) -> None:
        frame = _normalize_krx_etf_master(
            {
                "output": [
                    {
                        "ISU_CD": "KR7069500007",
                        "ISU_SRT_CD": "069500",
                        "ISU_ABBRV": "KODEX 200",
                        "LIST_DD": "2002/10/14",
                    }
                ]
            },
            retrieved_at="2026-07-23T00:00:00Z",
            source_hash="a" * 64,
        )

        self.assertEqual(frame.iloc[0]["security_id"], "KR:KR7069500007")
        self.assertEqual(frame.iloc[0]["asset_type"], "ETF")
        self.assertEqual(frame.iloc[0]["active_from"], "2002-10-14")
        self.assertEqual(frame.iloc[0]["yahoo_symbol"], "069500.KS")

    def test_etf_price_uses_authenticated_web_until_openapi_is_approved(self) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:KR7069500007",
                        "primary_symbol": "069500",
                        "active_from": "2002-10-14",
                        "active_to": "",
                        "exchange": "KOSPI",
                        "asset_type": "ETF",
                    }
                ]
            ),
            (),
        )
        stock_responses = []
        for _ in range(2):
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {"OutBlock_1": []}
            stock_responses.append(response)
        with (
            patch("supertrend_quant.market_store.kr_providers.load_env"),
            patch.dict(os.environ, {"KRX_OPENAPI_AUTH_KEY": "secret"}, clear=False),
            patch("requests.get", side_effect=stock_responses),
            patch(
                "supertrend_quant.market_store.kr_providers._fetch_krx_openapi_etf_daily",
                return_value=None,
            ),
            patch(
                "supertrend_quant.market_store.kr_providers._krx_web_credentials",
                return_value=("login-id", "login-password"),
            ),
            patch(
                "supertrend_quant.market_store.kr_providers._post_krx_web_json",
                return_value={
                    "output": [
                        {
                            "ISU_SRT_CD": "069500",
                            "TDD_OPNPRC": "43,660",
                            "TDD_HGPRC": "43,820",
                            "TDD_LWPRC": "42,890",
                            "TDD_CLSPRC": "43,090",
                            "ACC_TRDVOL": "12,362,878",
                        }
                    ]
                },
            ) as web_post,
        ):
            frame, artifact = fetch_krx_session_prices(
                "2025-07-22",
                catalog,
                symbols=("069500",),
            )

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["close"], 43_090)
        self.assertEqual(
            frame.iloc[0]["source"],
            "krx_authenticated_web_daily_ohlcv",
        )
        self.assertEqual(artifact.source, "krx_official_daily_ohlcv")
        self.assertEqual(
            web_post.call_args.args[0],
            {
                "bld": "dbms/MDC/STAT/standard/MDCSTAT04301",
                "trdDd": "20250722",
            },
        )

    def test_authenticated_web_stock_price_prefers_short_code_over_isin(self) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:KR7005930003",
                        "primary_symbol": "005930",
                        "active_from": "1975-06-11",
                        "active_to": "",
                        "exchange": "KOSPI",
                        "asset_type": "STOCK",
                    }
                ]
            ),
            (),
        )
        with (
            patch("supertrend_quant.market_store.kr_providers.load_env"),
            patch.dict(
                os.environ,
                {
                    "KRX_OPENAPI_AUTH_KEY": "secret",
                    "KRX_DAILY_PRICE_TRANSPORT": "web",
                },
                clear=False,
            ),
            patch(
                "supertrend_quant.market_store.kr_providers._krx_web_credentials",
                return_value=("login-id", "login-password"),
            ),
            patch(
                "supertrend_quant.market_store.kr_providers._post_krx_web_json",
                return_value={
                    "OutBlock_1": [
                        {
                            "ISU_CD": "KR7005930003",
                            "ISU_SRT_CD": "005930",
                            "TDD_OPNPRC": "70,000",
                            "TDD_HGPRC": "71,000",
                            "TDD_LWPRC": "69,000",
                            "TDD_CLSPRC": "70,500",
                            "ACC_TRDVOL": "1,000",
                        }
                    ]
                },
            ),
            patch("requests.get") as get,
        ):
            frame, _ = fetch_krx_session_prices(
                "2026-01-02",
                catalog,
                symbols=("005930",),
            )

        get.assert_not_called()
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["security_id"], "KR:KR7005930003")
        self.assertEqual(frame.iloc[0]["close"], 70_500)
        self.assertEqual(frame.iloc[0]["observation_status"], "traded")

    def test_price_checkpoint_is_invalidated_when_etf_identity_is_added(self) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:KR7005930003",
                        "primary_symbol": "005930",
                        "active_from": "1975-06-11",
                        "active_to": "",
                    },
                    {
                        "security_id": "KR:KR7069500007",
                        "primary_symbol": "069500",
                        "active_from": "2002-10-14",
                        "active_to": "",
                    },
                ]
            ),
            (),
        )
        cached = pd.DataFrame({"symbol": ["005930"]})

        self.assertFalse(
            _krx_price_checkpoint_covers_symbols(
                cached,
                ("005930", "069500"),
                catalog,
                "2025-07-22",
            )
        )

    def test_legacy_price_checkpoint_without_schema_version_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = KrCheckpointStore(tmp)
            path = store.price_path("2025-07-22")
            path.parent.mkdir(parents=True)
            pd.DataFrame({"symbol": ["005930"]}).to_parquet(path, index=False)

            self.assertIsNone(store.load_prices("2025-07-22"))

    def test_v2_price_checkpoint_is_upgraded_from_hashed_raw_isin(self) -> None:
        artifact = artifact_from_payload(
            "krx_official_daily_ohlcv",
            "https://data.krx.co.kr/",
            {
                "request": {
                    "session": "2015-01-02",
                    "basDd": "20150102",
                },
                "responses": {
                    "STOCK": {
                        "OutBlock_1": [
                            {
                                "ISU_CD": "KR7035720002",
                                "ISU_SRT_CD": "035720",
                                "ISU_ABBRV": "다음카카오",
                                "MKT_NM": "KOSDAQ",
                                "TDD_OPNPRC": "124,400",
                                "TDD_HGPRC": "138,700",
                                "TDD_LWPRC": "124,300",
                                "TDD_CLSPRC": "137,200",
                                "ACC_TRDVOL": "1,162,901",
                            }
                        ]
                    }
                },
            },
        )
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:KR7035720002",
                        "primary_symbol": "035720",
                        "name": "카카오",
                        "active_from": "2017-07-10",
                        "active_to": "",
                        "exchange": "KOSPI",
                        "asset_type": "STOCK",
                        "identity_mapped": True,
                    }
                ]
            ),
            (),
        )
        cached = _prices((137_200.0,))
        cached["session"] = "2015-01-02"
        cached["symbol"] = "035720"
        cached["security_id"] = ""
        cached["exchange"] = "KOSPI"
        cached["asset_type"] = "STOCK"
        cached["observation_status"] = "traded"
        cached["source"] = "krx_authenticated_web_daily_ohlcv"
        cached["source_url"] = "https://data.krx.co.kr/"
        cached["source_hash"] = artifact.source_hash
        cached["__checkpoint_schema_version"] = 2
        with tempfile.TemporaryDirectory() as tmp:
            store = KrCheckpointStore(tmp)
            path = store.price_path("2015-01-02")
            path.parent.mkdir(parents=True)
            cached.to_parquet(path, index=False)
            store.save_local_artifact(artifact, scope="krx-prices")

            upgraded = _collect_krx_prices(
                ("2015-01-02",),
                ("035720",),
                catalog,
                store,
                sleep_seconds=0.0,
                workers=1,
            )
            stored = pd.read_parquet(path)

        self.assertEqual(upgraded.iloc[0]["security_id"], "KR:KR7035720002")
        self.assertEqual(upgraded.iloc[0]["exchange"], "KOSDAQ")
        self.assertEqual(
            set(stored["__checkpoint_schema_version"]),
            {4},
        )
        self.assertIn("official_reference_price", stored)
        self.assertIn("official_fluctuation_rate", stored)

    def test_daily_isin_history_reconciles_market_transfer_intervals(self) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:KR7035720002",
                        "primary_symbol": "035720",
                        "name": "카카오",
                        "exchange": "KOSPI",
                        "asset_type": "STOCK",
                        "currency": "KRW",
                        "country": "KR",
                        "active_from": "2017-07-10",
                        "active_to": "",
                        "isin": "KR7035720002",
                        "identity_mapped": True,
                        "provider_symbol": "035720.KO",
                        "yahoo_symbol": "035720.KS",
                        "source": "krx_security_listing",
                        "source_url": "https://data.krx.co.kr/",
                        "retrieved_at": "2026-07-23T00:00:00Z",
                        "source_hash": "a" * 64,
                    }
                ]
            ),
            (),
        )
        observations = pd.DataFrame(
            [
                {
                    "security_id": "KR:KR7035720002",
                    "symbol": "035720",
                    "session": "2015-01-02",
                    "exchange": "KOSDAQ",
                    "asset_type": "STOCK",
                    "security_name": "다음카카오",
                    "source": "krx_authenticated_web_daily_ohlcv",
                    "source_url": "https://data.krx.co.kr/",
                    "retrieved_at": "2026-07-23T00:00:00Z",
                    "source_hash": "b" * 64,
                },
                {
                    "security_id": "KR:KR7035720002",
                    "symbol": "035720",
                    "session": "2017-07-06",
                    "exchange": "KOSDAQ",
                    "asset_type": "STOCK",
                    "security_name": "카카오",
                    "source": "krx_authenticated_web_daily_ohlcv",
                    "source_url": "https://data.krx.co.kr/",
                    "retrieved_at": "2026-07-23T00:00:00Z",
                    "source_hash": "c" * 64,
                },
                {
                    "security_id": "KR:KR7035720002",
                    "symbol": "035720",
                    "session": "2017-07-10",
                    "exchange": "KOSPI",
                    "asset_type": "STOCK",
                    "security_name": "카카오",
                    "source": "krx_authenticated_web_daily_ohlcv",
                    "source_url": "https://data.krx.co.kr/",
                    "retrieved_at": "2026-07-23T00:00:00Z",
                    "source_hash": "d" * 64,
                },
                {
                    "security_id": "KR:KR7035720002",
                    "symbol": "035720",
                    "session": "2026-07-22",
                    "exchange": "KOSPI",
                    "asset_type": "STOCK",
                    "security_name": "카카오",
                    "source": "krx_authenticated_web_daily_ohlcv",
                    "source_url": "https://data.krx.co.kr/",
                    "retrieved_at": "2026-07-23T00:00:00Z",
                    "source_hash": "e" * 64,
                },
            ]
        )

        reconciled = _reconcile_catalog_with_krx_observations(
            catalog,
            observations,
            completed_session="2026-07-22",
        )

        self.assertEqual(
            reconciled.row_for("035720", "2015-01-02")["exchange"],
            "KOSDAQ",
        )
        self.assertEqual(
            reconciled.row_for("035720", "2017-07-10")["exchange"],
            "KOSPI",
        )
        self.assertEqual(
            reconciled.security_id_for("035720", "2015-01-02"),
            "KR:KR7035720002",
        )

    def test_membership_events_detect_same_symbol_new_isin(self) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:OLD00000001",
                        "primary_symbol": "123456",
                        "active_from": "2019-01-01",
                        "active_to": "2020-01-02",
                        "identity_mapped": True,
                    },
                    {
                        "security_id": "KR:NEW00000001",
                        "primary_symbol": "123456",
                        "active_from": "2020-01-03",
                        "active_to": "",
                        "identity_mapped": True,
                    },
                ]
            ),
            (),
        )
        common = {
            "symbols": ["123456"],
            "source": "krx_authenticated_web_index_constituents",
            "source_url": "https://data.krx.co.kr/",
            "retrieved_at": "2026-07-23T00:00:00Z",
            "source_hash": "a" * 64,
        }
        memberships = {
            "kospi200": {
                "2020-01-02": {**common, "session": "2020-01-02"},
                "2020-01-03": {**common, "session": "2020-01-03"},
            }
        }

        anchors, events = _membership_datasets(memberships, catalog)

        self.assertEqual(set(anchors["security_id"]), {"KR:OLD00000001"})
        self.assertEqual(set(events["operation"]), {"REMOVE", "ADD"})
        self.assertEqual(
            set(events["security_id"]),
            {"KR:OLD00000001", "KR:NEW00000001"},
        )

    def test_full_membership_verifier_replays_every_official_snapshot(
        self,
    ) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": f"KR:ID{symbol}",
                        "primary_symbol": symbol,
                        "active_from": "2019-01-01",
                        "active_to": "",
                        "identity_mapped": True,
                    }
                    for symbol in ("000001", "000002", "000003")
                ]
            ),
            (),
        )
        common = {
            "source": "krx_authenticated_web_index_constituents",
            "source_url": "https://data.krx.co.kr/",
            "retrieved_at": "2026-07-23T00:00:00Z",
            "source_hash": "a" * 64,
        }
        memberships = {
            "kospi200": {
                "2020-01-02": {
                    **common,
                    "session": "2020-01-02",
                    "symbols": ["000001", "000002"],
                },
                "2020-01-03": {
                    **common,
                    "session": "2020-01-03",
                    "symbols": ["000002", "000003"],
                },
            }
        }
        anchors, events = _membership_datasets(memberships, catalog)
        definition = {
            **kr_providers.KR_INDEX_DEFINITIONS["kospi200"],
            "expected_count": 2,
            "count_tolerance": 0,
        }

        with patch.dict(
            kr_providers.KR_INDEX_DEFINITIONS,
            {"kospi200": definition},
        ):
            report = verify_kr_membership_history(
                memberships,
                catalog,
                anchors,
                events,
                sessions=("2020-01-02", "2020-01-03"),
                profiles=("kospi200",),
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["expected_snapshot_count"], 2)
        self.assertEqual(report["daily_replay_mismatch_count"], 0)
        self.assertEqual(report["profiles"]["kospi200"]["event_count"], 2)

    def test_full_membership_verifier_blocks_tampered_event_history(
        self,
    ) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": f"KR:ID{symbol}",
                        "primary_symbol": symbol,
                        "active_from": "2019-01-01",
                        "active_to": "",
                        "identity_mapped": True,
                    }
                    for symbol in ("000001", "000002", "000003")
                ]
            ),
            (),
        )
        common = {
            "source": "krx_authenticated_web_index_constituents",
            "source_url": "https://data.krx.co.kr/",
            "retrieved_at": "2026-07-23T00:00:00Z",
            "source_hash": "a" * 64,
        }
        memberships = {
            "kospi200": {
                "2020-01-02": {**common, "symbols": ["000001", "000002"]},
                "2020-01-03": {**common, "symbols": ["000002", "000003"]},
            }
        }
        anchors, events = _membership_datasets(memberships, catalog)
        events = events.loc[events["operation"].ne("ADD")].copy()
        definition = {
            **kr_providers.KR_INDEX_DEFINITIONS["kospi200"],
            "expected_count": 2,
            "count_tolerance": 0,
        }

        with patch.dict(
            kr_providers.KR_INDEX_DEFINITIONS,
            {"kospi200": definition},
        ):
            report = verify_kr_membership_history(
                memberships,
                catalog,
                anchors,
                events,
                sessions=("2020-01-02", "2020-01-03"),
                profiles=("kospi200",),
            )

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["daily_replay_mismatch_count"], 1)
        self.assertEqual(report["survivorship_score"], 0.0)

    def test_price_scorecard_never_infers_survivorship_from_coverage(
        self,
    ) -> None:
        metrics = {
            "expected_row_count": 100,
            "unclassified_missing": 0,
            "unexplained_large_discontinuities": 0,
            "close_within_one_tick_rate": 1.0,
            "expected_session_coverage": 1.0,
            "identity_mapping_rate": 1.0,
        }

        scorecard = _provider_scorecard(
            "kis",
            metrics,
            {"score": 1.0},
        )

        self.assertEqual(scorecard["dimensions"]["survivorship"], 0.0)

    def test_membership_success_is_checkpointed_before_later_failure(self) -> None:
        artifact = artifact_from_payload(
            "test_krx_membership",
            "https://data.krx.co.kr/",
            {"output": "test"},
        )

        def fetch(_profile, session, *, sleep_seconds):
            del sleep_seconds
            if session == "2025-07-11":
                raise KrOfficialDataUnavailable("test failure")
            return tuple(f"{index:06d}" for index in range(1, 201)), artifact

        with tempfile.TemporaryDirectory() as tmp:
            store = KrCheckpointStore(tmp)
            with (
                patch(
                    "supertrend_quant.market_store.kr_pipeline.fetch_krx_membership",
                    side_effect=fetch,
                ),
                self.assertRaisesRegex(RuntimeError, "test failure"),
            ):
                _collect_memberships(
                    ("2025-07-10", "2025-07-11"),
                    ("kospi200",),
                    store,
                    sleep_seconds=0.0,
                    allow_pre_inception=False,
                    workers=2,
                )

            self.assertIsNotNone(
                store.load_membership("kospi200", "2025-07-10")
            )

    def test_price_failure_cancels_queued_sessions(self) -> None:
        artifact = artifact_from_payload(
            "test_krx_price",
            "https://data.krx.co.kr/",
            {"output": "test"},
        )
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                columns=[
                    "security_id",
                    "primary_symbol",
                    "active_from",
                    "active_to",
                ]
            ),
            (),
        )
        sessions = tuple(
            pd.date_range("2025-07-10", periods=20, freq="B").date.astype(str)
        )
        release_waiter = threading.Event()
        calls: list[str] = []

        def fetch(session, _catalog, *, symbols, sleep_seconds):
            del _catalog, symbols, sleep_seconds
            calls.append(session)
            if session == sessions[0]:
                raise KrOfficialDataUnavailable("test failure")
            release_waiter.wait(timeout=1)
            return _prices((100.0,)), artifact

        timer = threading.Timer(0.1, release_waiter.set)
        with tempfile.TemporaryDirectory() as tmp:
            timer.start()
            try:
                with (
                    patch(
                        "supertrend_quant.market_store.kr_pipeline.fetch_krx_session_prices",
                        side_effect=fetch,
                    ),
                    self.assertRaisesRegex(RuntimeError, "test failure"),
                ):
                    _collect_krx_prices(
                        sessions,
                        ("005930",),
                        catalog,
                        KrCheckpointStore(tmp),
                        sleep_seconds=0.0,
                        workers=1,
                    )
            finally:
                release_waiter.set()
                timer.cancel()

        self.assertLess(len(calls), len(sessions))

    def test_pre_inception_kosdaq150_sessions_are_not_requested(self) -> None:
        artifact = artifact_from_payload(
            "test_krx_membership",
            "https://data.krx.co.kr/",
            {"output": "test"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = KrCheckpointStore(tmp)
            with patch(
                "supertrend_quant.market_store.kr_pipeline.fetch_krx_membership",
                return_value=(
                    tuple(f"{index:06d}" for index in range(1, 151)),
                    artifact,
                ),
            ) as fetch:
                result = _collect_memberships(
                    ("2015-07-10", "2015-07-13"),
                    ("kosdaq150",),
                    store,
                    sleep_seconds=0.0,
                    allow_pre_inception=True,
                    workers=2,
                )

        self.assertNotIn("2015-07-10", result["kosdaq150"])
        self.assertIn("2015-07-13", result["kosdaq150"])
        self.assertEqual(fetch.call_count, 1)

    def test_missing_bar_on_delisting_effective_date_is_classified(self) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:KR7010620003",
                        "primary_symbol": "010620",
                        "active_from": "1983-12-20",
                        "active_to": "2025-12-15",
                        "identity_mapped": True,
                    }
                ]
            ),
            (),
        )
        frame = pd.DataFrame(
            {
                "symbol": ["010620"],
                "observation_status": ["missing_from_krx_response"],
            }
        )

        classified = _classify_cached_delisting_effective_absences(
            frame,
            catalog,
            "2025-12-15",
        )

        self.assertEqual(
            classified.iloc[0]["observation_status"],
            "delisting_effective_date_no_trade",
        )

    def test_missing_bar_on_market_transfer_is_not_classified_as_delisting(
        self,
    ) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:KR7058970005",
                        "primary_symbol": "058970",
                        "exchange": "STOCK",
                        "active_from": "2016-04-28",
                        "active_to": "2021-08-12",
                        "identity_mapped": True,
                    },
                    {
                        "security_id": "KR:KR7058970005",
                        "primary_symbol": "058970",
                        "exchange": "KOSDAQ",
                        "active_from": "2021-08-13",
                        "active_to": "",
                        "identity_mapped": True,
                    },
                ]
            ),
            (),
        )
        frame = pd.DataFrame(
            {
                "symbol": ["058970"],
                "observation_status": [
                    "delisting_effective_date_no_trade"
                ],
            }
        )

        classified = _classify_cached_delisting_effective_absences(
            frame,
            catalog,
            "2021-08-12",
        )

        self.assertEqual(
            classified.iloc[0]["observation_status"],
            "missing_from_krx_response",
        )

    def test_krx_no_trade_is_classified_without_fabricating_a_bar(self) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:KR7005930003",
                        "primary_symbol": "005930",
                        "active_from": "1975-06-11",
                        "active_to": "",
                    }
                ]
            ),
            (),
        )
        kospi = Mock()
        kospi.raise_for_status.return_value = None
        kospi.json.return_value = {
            "OutBlock_1": [
                {
                    "ISU_CD": "005930",
                    "TDD_OPNPRC": "70000",
                    "TDD_HGPRC": "70000",
                    "TDD_LWPRC": "70000",
                    "TDD_CLSPRC": "70000",
                    "ACC_TRDVOL": "0",
                }
            ]
        }
        kosdaq = Mock()
        kosdaq.raise_for_status.return_value = None
        kosdaq.json.return_value = {"OutBlock_1": []}
        with (
            patch("supertrend_quant.market_store.kr_providers.load_env"),
            patch.dict(os.environ, {"KRX_OPENAPI_AUTH_KEY": "secret"}, clear=False),
            patch("requests.get", side_effect=(kospi, kosdaq)),
        ):
            frame, _ = fetch_krx_session_prices(
                "2026-01-02", catalog, symbols=("005930",)
            )

        self.assertEqual(frame.iloc[0]["observation_status"], "suspended_or_no_trade")
        self.assertTrue(pd.isna(frame.iloc[0]["close"]))

    def test_classified_no_trade_does_not_hide_official_unclassified_rows(self) -> None:
        traded = _prices((10_000.0, 10_050.0))
        no_trade = traded.iloc[[0]].copy()
        no_trade["security_id"] = "KR:NO-TRADE"
        no_trade["observation_status"] = "suspended_or_no_trade"
        no_trade[["open", "high", "low", "close"]] = float("nan")
        baseline = pd.concat([traded, no_trade], ignore_index=True)
        passed = compare_provider_to_krx(
            baseline,
            KrProviderResult("secondary", "ok", traded.copy()),
            identity_mapping_rate=1.0,
        )
        unclassified = no_trade.copy()
        unclassified["observation_status"] = "missing_from_krx_response"
        failed = compare_provider_to_krx(
            pd.concat([traded, unclassified], ignore_index=True),
            KrProviderResult("secondary", "ok", traded.copy()),
            identity_mapping_rate=1.0,
        )

        self.assertTrue(passed["hard_gate_passed"])
        self.assertEqual(passed["krx_classified_no_trade_observations"], 1)
        self.assertFalse(failed["hard_gate_passed"])
        self.assertEqual(failed["krx_unclassified_observations"], 1)

    def test_smoke_benchmark_cannot_authorize_full_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "benchmarks" / "current.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                '{"status":"ready","benchmark_scope":"smoke",'
                '"selection":{"primary":"krx","secondary":"eodhd"}}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "not a full PIT-union"):
                _load_ready_benchmark(Path(tmp))

    def test_yahoo_ticker_reuse_is_split_by_isin_interval(self) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:OLD00000001",
                        "primary_symbol": "000010",
                        "active_from": "2020-01-01",
                        "active_to": "2020-12-31",
                        "yahoo_symbol": "000010.KS",
                    },
                    {
                        "security_id": "KR:NEW00000001",
                        "primary_symbol": "000010",
                        "active_from": "2021-01-01",
                        "active_to": "",
                        "yahoo_symbol": "000010.KS",
                    },
                ]
            ),
            (),
        )
        downloaded = pd.DataFrame(
            {
                "Open": [10, 20],
                "High": [10, 20],
                "Low": [10, 20],
                "Close": [10, 20],
                "Volume": [100, 200],
            },
            index=[pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01")],
        )
        with patch("yfinance.download", return_value=downloaded):
            result = fetch_yahoo_prices(
                catalog,
                start="2020-01-01",
                end="2021-12-31",
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(
            set(result.prices["security_id"]),
            {"KR:OLD00000001", "KR:NEW00000001"},
        )

    def test_official_lifecycle_import_requires_economic_terms(self) -> None:
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:KR7000010000",
                        "primary_symbol": "000010",
                        "active_from": "2010-01-01",
                        "active_to": "2020-06-30",
                    }
                ]
            ),
            (),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "actions.csv"
            pd.DataFrame(
                [
                    {
                        "isin": "KR7000010000",
                        "symbol": "000010",
                        "action_type": "delisting",
                        "effective_date": "2020-06-30",
                        "cash_amount": "0",
                        "source_url": "https://kind.krx.co.kr/example",
                    }
                ]
            ).to_csv(path, index=False)
            with patch.dict(
                os.environ,
                {"KR_OFFICIAL_ACTIONS_PATH": str(path)},
                clear=False,
            ):
                actions, _ = _load_kr_official_actions(
                    catalog,
                    start="2015-01-01",
                    end="2026-01-01",
                )

        report = validate_dataset(
            "corporate_actions",
            actions,
            incomplete_action_policy="block",
            completed_session="2026-01-01",
            market="KR",
        )
        self.assertTrue(report.valid)
        self.assertEqual(actions.iloc[0]["cash_amount"], 0)

    def test_verified_terminal_action_closes_kr_lifecycle_candidate(self) -> None:
        security_id = "KR:KR7000010000"
        source = {
            "source": "fixture",
            "retrieved_at": "2026-01-02T00:00:00Z",
            "source_hash": "a" * 64,
        }
        frames = {
            "security_master": pd.DataFrame(
                [
                    {
                        "security_id": security_id,
                        "primary_symbol": "000010",
                        "name": "Old Co",
                        "exchange": "KOSPI",
                        "asset_type": "STOCK",
                        "currency": "KRW",
                        "country": "KR",
                        "active_from": "2010-01-01",
                        "active_to": "2020-06-30",
                        **source,
                    }
                ]
            ),
            "daily_price_raw": pd.DataFrame(
                [
                    {
                        "security_id": security_id,
                        "session": "2020-06-29",
                        "open": 1,
                        "high": 1,
                        "low": 1,
                        "close": 1,
                        "volume": 100,
                        "currency": "KRW",
                        **source,
                    }
                ]
            ),
            "index_constituent_anchors": pd.DataFrame(
                [{"index_id": "kospi200", "security_id": security_id}]
            ),
            "index_membership_events": pd.DataFrame(
                columns=["security_id", "operation", "effective_date"]
            ),
        }
        actions = pd.DataFrame(
            [
                {
                    "event_id": "event-1",
                    "security_id": security_id,
                    "action_type": "delisting",
                    "effective_date": "2020-06-30",
                    "ex_date": "2020-06-30",
                    "announcement_date": "",
                    "record_date": "",
                    "payment_date": "",
                    "cash_amount": 0,
                    "ratio": None,
                    "currency": "KRW",
                    "new_security_id": "",
                    "new_symbol": "",
                    "official": True,
                    "source_url": "https://kind.krx.co.kr/example",
                    "source_kind": "official",
                    **source,
                }
            ]
        )
        frames["corporate_actions"] = actions

        class FrameRepository:
            def __init__(self, root):
                self.root = root

            def read_frame(self, dataset, version=None):
                return frames[dataset]

        release = DataRelease.create(
            "2026-01-02",
            {name: "v1" for name in frames},
            metadata={"market": "KR"},
        )
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"KR_LIFECYCLE_RESOLUTIONS_PATH": ""},
            clear=False,
        ):
            resolutions, coverage = _resolve_kr_lifecycle_candidates(
                FrameRepository(Path(tmp)),
                release,
                actions,
            )

        self.assertTrue(coverage.valid)
        self.assertEqual(coverage.applied_count, 1)
        self.assertEqual(resolutions.iloc[0]["event_id"], "event-1")

    def test_official_tradable_index_exit_closes_later_terminal_candidate(
        self,
    ) -> None:
        security_id = "KR:KR7000010000"
        source = {
            "source": "krx_authenticated_web_index_constituents",
            "source_url": "https://data.krx.co.kr/",
            "retrieved_at": "2026-01-02T00:00:00Z",
            "source_hash": "a" * 64,
        }
        frames = {
            "security_master": pd.DataFrame(
                [
                    {
                        "security_id": security_id,
                        "primary_symbol": "000010",
                        "name": "Old Co",
                        "exchange": "KOSPI",
                        "asset_type": "STOCK",
                        "currency": "KRW",
                        "country": "KR",
                        "active_from": "2010-01-01",
                        "active_to": "2020-06-30",
                        **source,
                    }
                ]
            ),
            "daily_price_raw": pd.DataFrame(
                [
                    {
                        "security_id": security_id,
                        "session": "2020-06-29",
                        "open": 1,
                        "high": 1,
                        "low": 1,
                        "close": 1,
                        "volume": 100,
                        "currency": "KRW",
                        **source,
                    }
                ]
            ),
            "index_constituent_anchors": pd.DataFrame(
                [{"index_id": "kospi200", "security_id": security_id}]
            ),
            "index_membership_events": pd.DataFrame(
                [
                    {
                        "event_id": "remove-1",
                        "index_id": "kospi200",
                        "effective_date": "2020-01-15",
                        "operation": "REMOVE",
                        "security_id": security_id,
                        **source,
                    }
                ]
            ),
            "corporate_actions": pd.DataFrame(
                columns=[
                    "event_id",
                    "security_id",
                    "action_type",
                    "effective_date",
                    "new_security_id",
                    "new_symbol",
                    "official",
                    "source_url",
                    "source_kind",
                    "retrieved_at",
                    "source_hash",
                ]
            ),
        }

        class FrameRepository:
            def __init__(self, root):
                self.root = root

            def read_frame(self, dataset, version=None):
                return frames[dataset]

        release = DataRelease.create(
            "2026-01-02",
            {name: "v1" for name in frames},
            metadata={"market": "KR"},
        )
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"KR_LIFECYCLE_RESOLUTIONS_PATH": ""},
            clear=False,
        ):
            root = Path(tmp)
            resolutions, coverage = _resolve_kr_lifecycle_candidates(
                FrameRepository(root),
                release,
                frames["corporate_actions"],
            )
            checkpoint = root / "state" / "lifecycle" / "candidates.json"
            self.assertTrue(checkpoint.is_file())

        self.assertTrue(coverage.valid)
        self.assertEqual(coverage.exception_count, 1)
        self.assertEqual(
            resolutions.iloc[0]["exception_code"],
            "already_represented",
        )

    def test_stale_current_krx_identity_gets_expiring_lifecycle_exception(
        self,
    ) -> None:
        security_id = "KR:KR7000010000"
        source = {
            "source": "krx_official_daily_identity",
            "source_url": "https://data.krx.co.kr/",
            "retrieved_at": "2026-01-02T00:00:00Z",
            "source_hash": "b" * 64,
        }
        frames = {
            "security_master": pd.DataFrame(
                [
                    {
                        "security_id": security_id,
                        "primary_symbol": "000010",
                        "name": "Suspended Co",
                        "exchange": "KOSDAQ",
                        "asset_type": "STOCK",
                        "currency": "KRW",
                        "country": "KR",
                        "active_from": "2010-01-01",
                        "active_to": "",
                        **source,
                    }
                ]
            ),
            "daily_price_raw": pd.DataFrame(
                [
                    {
                        "security_id": security_id,
                        "session": "2020-06-29",
                        "open": 1,
                        "high": 1,
                        "low": 1,
                        "close": 1,
                        "volume": 100,
                        "currency": "KRW",
                        **source,
                    }
                ]
            ),
            "index_constituent_anchors": pd.DataFrame(
                [{"index_id": "kosdaq150", "security_id": security_id}]
            ),
            "index_membership_events": pd.DataFrame(
                columns=[
                    "event_id",
                    "index_id",
                    "effective_date",
                    "operation",
                    "security_id",
                    "source",
                    "source_url",
                    "retrieved_at",
                    "source_hash",
                ]
            ),
            "corporate_actions": pd.DataFrame(
                columns=[
                    "event_id",
                    "security_id",
                    "action_type",
                    "effective_date",
                    "new_security_id",
                    "new_symbol",
                    "official",
                    "source_url",
                    "source_kind",
                    "retrieved_at",
                    "source_hash",
                ]
            ),
        }

        class FrameRepository:
            def __init__(self, root):
                self.root = root

            def read_frame(self, dataset, version=None):
                return frames[dataset]

        release = DataRelease.create(
            "2026-01-02",
            {name: "v1" for name in frames},
            metadata={"market": "KR"},
        )
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"KR_LIFECYCLE_RESOLUTIONS_PATH": ""},
            clear=False,
        ):
            resolutions, coverage = _resolve_kr_lifecycle_candidates(
                FrameRepository(Path(tmp)),
                release,
                frames["corporate_actions"],
            )

        self.assertTrue(coverage.valid)
        self.assertEqual(
            resolutions.iloc[0]["exception_code"],
            "insufficient_official_evidence",
        )
        self.assertEqual(resolutions.iloc[0]["recheck_after"], "2026-02-02")

    def test_licensed_isin_borrows_only_matching_listing_interval(self) -> None:
        licensed = pd.DataFrame(
            [
                {
                    "security_id": "KR:KR7000010000",
                    "primary_symbol": "000010",
                    "name": "000010",
                    "exchange": "KOSPI",
                    "asset_type": "STOCK",
                    "currency": "KRW",
                    "country": "KR",
                    "active_from": "2020-01-02",
                    "active_to": "",
                    "isin": "KR7000010000",
                    "identity_mapped": True,
                    "provider_symbol": "000010.KO",
                    "yahoo_symbol": "000010.KS",
                    "source": "krx_licensed_index_constituents",
                    "source_url": "https://example.test",
                    "retrieved_at": "2026-01-01T00:00:00Z",
                    "source_hash": "a" * 64,
                }
            ]
        )
        listing = licensed.copy()
        listing["security_id"] = "KR:UNMAPPED:KOSPI:000010"
        listing["isin"] = ""
        listing["active_from"] = "2010-03-04"
        listing["active_to"] = "2022-06-30"
        listing["name"] = "Historical Co"

        result = _reconcile_licensed_identity_intervals(licensed, listing)

        self.assertEqual(result.iloc[0]["security_id"], "KR:KR7000010000")
        self.assertEqual(result.iloc[0]["active_from"], "2010-03-04")
        self.assertEqual(result.iloc[0]["active_to"], "2022-06-30")
        self.assertEqual(result.iloc[0]["name"], "Historical Co")

    def test_future_listing_interval_does_not_rewrite_older_isin_membership(self) -> None:
        licensed = pd.DataFrame(
            [
                {
                    "security_id": "KR:KR7000010000",
                    "primary_symbol": "000010",
                    "name": "000010",
                    "exchange": "KOSDAQ",
                    "asset_type": "STOCK",
                    "currency": "KRW",
                    "country": "KR",
                    "active_from": "2015-01-02",
                    "active_to": "2019-12-31",
                    "isin": "KR7000010000",
                    "identity_mapped": True,
                    "provider_symbol": "000010.KQ",
                    "yahoo_symbol": "000010.KQ",
                    "source": "licensed",
                    "source_url": "https://example.test",
                    "retrieved_at": "2026-01-01T00:00:00Z",
                    "source_hash": "a" * 64,
                }
            ]
        )
        future = licensed.copy()
        future["exchange"] = "KOSPI"
        future["active_from"] = "2020-01-02"
        future["active_to"] = ""
        future["name"] = "Transferred Co"

        result = _reconcile_licensed_identity_intervals(licensed, future)

        self.assertEqual(result.iloc[0]["active_from"], "2015-01-02")
        self.assertEqual(result.iloc[0]["active_to"], "2019-12-31")

    def test_release_metadata_round_trip_and_legacy_compatibility(self) -> None:
        release = DataRelease.create(
            "2026-01-02",
            {"daily_price_raw": "v1"},
            metadata={"market": "KR", "calendar": "XKRX"},
        )
        restored = DataRelease.from_bytes(release.to_bytes())
        legacy = DataRelease(
            version="legacy",
            created_at="2026-01-02T00:00:00Z",
            completed_session="2026-01-02",
            dataset_versions={"daily_price_raw": "v1"},
        )

        self.assertEqual(restored.metadata["market"], "KR")
        self.assertNotIn(b'"metadata"', legacy.to_bytes())

    def test_index_coverage_keeps_per_profile_inception(self) -> None:
        metadata = _index_coverage_metadata(
            {
                "kospi200": {
                    "2015-01-02": {},
                    "2015-07-13": {},
                },
                "kosdaq150": {"2015-07-13": {}},
            }
        )

        self.assertEqual(metadata["official_coverage_start"], "2015-01-02")
        self.assertEqual(
            metadata["official_coverage_by_index"]["kosdaq150"]["start"],
            "2015-07-13",
        )

    def test_index_price_gap_policy_binds_classified_krx_no_trade_rows(
        self,
    ) -> None:
        anchors = pd.DataFrame(
            [
                {
                    "index_id": "kospi200",
                    "anchor_date": "2026-01-02",
                    "security_id": "KR:TEST",
                }
            ]
        )
        events = pd.DataFrame(
            [
                {
                    "event_id": "remove",
                    "index_id": "kospi200",
                    "security_id": "KR:TEST",
                    "effective_date": "2026-02-03",
                    "operation": "REMOVE",
                }
            ]
        )
        prices = pd.DataFrame(
            [{"security_id": "KR:TEST", "session": "2026-01-02"}]
        )
        observations = pd.DataFrame(
            [
                {
                    "security_id": "KR:TEST",
                    "session": session,
                    "observation_status": "suspended_or_no_trade",
                    "source_url": "https://data.krx.co.kr/",
                    "source_hash": "a" * 64,
                }
                for session in _sessions_between(
                    "2026-01-03",
                    "2026-02-02",
                )
            ]
        )

        policy = _krx_index_price_gap_policy(
            anchors,
            events,
            prices,
            observations,
        )
        cross_reports = pd.DataFrame(
            [
                {
                    "status": "passed",
                    "index_price_gap_policy_sha256": policy["policy_sha256"],
                    "index_price_gap_policy_json": json.dumps(
                        policy,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            ]
        )

        self.assertEqual(policy["gap_count"], 1)
        self.assertEqual(
            policy["observation_count"],
            len(observations),
        )
        self.assertEqual(
            validate_index_price_gap_policy(policy, cross_reports),
            ("KR:TEST",),
        )

    def test_index_price_gap_policy_rejects_unclassified_rows(self) -> None:
        anchors = pd.DataFrame(
            [
                {
                    "index_id": "kospi200",
                    "anchor_date": "2026-01-02",
                    "security_id": "KR:TEST",
                }
            ]
        )
        events = pd.DataFrame(
            [
                {
                    "event_id": "remove",
                    "index_id": "kospi200",
                    "security_id": "KR:TEST",
                    "effective_date": "2026-02-03",
                    "operation": "REMOVE",
                }
            ]
        )
        prices = pd.DataFrame(
            [{"security_id": "KR:TEST", "session": "2026-01-02"}]
        )
        observations = pd.DataFrame(
            [
                {
                    "security_id": "KR:TEST",
                    "session": session,
                    "observation_status": (
                        "missing_from_krx_response"
                        if index == 0
                        else "suspended_or_no_trade"
                    ),
                    "source_url": "https://data.krx.co.kr/",
                    "source_hash": "a" * 64,
                }
                for index, session in enumerate(
                    _sessions_between(
                        "2026-01-03",
                        "2026-02-02",
                    )
                )
            ]
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "lacks complete classified KRX evidence",
        ):
            _krx_index_price_gap_policy(
                anchors,
                events,
                prices,
                observations,
            )

    def test_opendart_dividend_document_parser_reads_euc_kr_table(self) -> None:
        html = """
        <html><meta charset="euc-kr"><table>
          <tr><td>3. 1주당 배당금(원)</td><td>보통주식</td><td>350</td></tr>
          <tr><td>6. 배당기준일</td><td>2015-12-31</td></tr>
          <tr><td>7. 배당금지급 예정일자</td><td>2016-04-08</td></tr>
          <tr><td>10. 이사회결의일(결정일)</td><td>2016-02-25</td></tr>
        </table></html>
        """
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("filing.xml", html.encode("euc-kr"))

        parsed = _parse_opendart_cash_dividend_document(
            buffer.getvalue()
        )

        self.assertEqual(parsed["cash_amount"], 350.0)
        self.assertEqual(parsed["record_date"], "2015-12-31")
        self.assertEqual(parsed["payment_date"], "2016-04-08")
        self.assertEqual(parsed["board_date"], "2016-02-25")
        viewer_parsed = _parse_opendart_cash_dividend_document(
            html.encode("euc-kr")
        )
        self.assertEqual(viewer_parsed, parsed)

    def test_opendart_dividend_parser_uses_final_corrected_form(self) -> None:
        html = """
        <html><meta charset="euc-kr">
          <table>
            <tr><td>정정항목</td><td>정정전</td><td>정정후</td></tr>
            <tr><td>3. 1주당 배당금(원) 보통주식</td><td>150</td><td>200</td></tr>
            <tr><td>6. 배당기준일</td><td>2015.12.29</td><td>2015.12.31</td></tr>
          </table>
          <table>
            <tr><td>3. 1주당 배당금(원)</td><td>보통주식</td><td>200</td></tr>
            <tr><td>6. 배당기준일</td><td>2015-12-31</td></tr>
          </table>
        </html>
        """

        parsed = _parse_opendart_cash_dividend_document(
            html.encode("euc-kr")
        )

        self.assertEqual(parsed["cash_amount"], 200.0)
        self.assertEqual(parsed["record_date"], "2015-12-31")

    def test_kr_dividend_ex_date_uses_t_plus_two_exchange_sessions(self) -> None:
        sessions = _kr_dividend_sessions(
            start="2015-01-01",
            end="2016-01-08",
        )

        self.assertEqual(
            _kr_dividend_ex_date("2015-06-30", sessions),
            "2015-06-29",
        )
        self.assertEqual(
            _kr_dividend_ex_date("2015-12-31", sessions),
            "2015-12-29",
        )

    def test_kr_dividend_sessions_clamp_pre_calendar_listing_dates(self) -> None:
        sessions = _kr_dividend_sessions(
            start="1956-02-01",
            end="2015-01-09",
        )

        self.assertEqual(
            sessions[0].date().isoformat(),
            pd.Timestamp(exchange_calendar("KR").first_session).date().isoformat(),
        )
        self.assertEqual(
            sessions[-1].date().isoformat(),
            "2015-01-09",
        )

    def test_benchmark_report_is_archived_as_private_publishable_evidence(
        self,
    ) -> None:
        payload = b'{"schema_version":5,"status":"ready"}\n'
        artifact = SourceArtifact(
            source="kr_provider_benchmark_report",
            source_url="local://benchmarks/current.json",
            retrieved_at="2026-07-25T00:00:00Z",
            content=payload,
            content_type="application/json",
        )
        with tempfile.TemporaryDirectory() as directory:
            repository = Mock(root=Path(directory))
            archive = _archive_allowed_artifacts(
                repository,
                (artifact,),
                effective_date="2026-07-24",
            )
            archived_path = Path(directory) / str(
                archive.iloc[0]["object_path"]
            )

            self.assertEqual(
                archive.iloc[0]["archive_id"],
                artifact.source_hash,
            )
            self.assertEqual(
                archive.iloc[0]["license_class"],
                "allowed_private",
            )
            self.assertEqual(gzip.decompress(archived_path.read_bytes()), payload)

    def test_krx_reference_reset_generates_official_price_adjustment(self) -> None:
        observations = pd.DataFrame(
            [
                {
                    "security_id": "KR:TEST",
                    "session": "2026-01-02",
                    "close": 100.0,
                    "official_reference_price": 100.0,
                    "observation_status": "traded",
                    "source_url": "https://data.krx.co.kr/",
                    "retrieved_at": "2026-01-03T00:00:00Z",
                    "source_hash": "a" * 64,
                },
                {
                    "security_id": "KR:TEST",
                    "session": "2026-01-05",
                    "close": 52.0,
                    "official_reference_price": 50.0,
                    "observation_status": "traded",
                    "source_url": "https://data.krx.co.kr/",
                    "retrieved_at": "2026-01-06T00:00:00Z",
                    "source_hash": "b" * 64,
                },
            ]
        )
        empty_actions = pd.DataFrame(
            columns=dataset_spec("corporate_actions").required_columns
        )

        generated, report = _krx_reference_price_adjustments(
            observations,
            empty_actions,
        )
        payload = json.loads(report.content)

        self.assertEqual(len(generated), 1)
        self.assertEqual(
            generated.iloc[0]["action_type"],
            "reference_price_adjustment",
        )
        self.assertAlmostEqual(float(generated.iloc[0]["ratio"]), 2.0)
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["unresolved_count"], 0)

    def test_krx_restated_split_generates_price_factor_noop(self) -> None:
        observations = pd.DataFrame(
            [
                {
                    "security_id": "KR:TEST",
                    "session": "2026-01-02",
                    "close": 100.0,
                    "official_reference_price": 100.0,
                    "observation_status": "traded",
                    "source_url": "https://data.krx.co.kr/",
                    "retrieved_at": "2026-01-03T00:00:00Z",
                    "source_hash": "a" * 64,
                },
                {
                    "security_id": "KR:TEST",
                    "session": "2026-01-05",
                    "close": 102.0,
                    "official_reference_price": 100.0,
                    "observation_status": "traded",
                    "source_url": "https://data.krx.co.kr/",
                    "retrieved_at": "2026-01-06T00:00:00Z",
                    "source_hash": "b" * 64,
                },
            ]
        )
        provider_action = {
            column: ""
            for column in dataset_spec(
                "corporate_actions"
            ).required_columns
        }
        provider_action.update(
            {
                "event_id": "provider-five-for-one-split",
                "security_id": "KR:TEST",
                "action_type": "split",
                "effective_date": "2026-01-03",
                "ex_date": "2026-01-03",
                "ratio": 5.0,
                "official": False,
                "source": "provider",
                "source_hash": "c" * 64,
            }
        )
        provider_actions = pd.DataFrame([provider_action])

        generated, report = _krx_reference_price_adjustments(
            observations,
            provider_actions,
        )
        payload = json.loads(report.content)

        self.assertEqual(len(generated), 1)
        self.assertEqual(
            generated.iloc[0]["source"],
            "krx_official_series_restatement_noop",
        )
        self.assertEqual(float(generated.iloc[0]["ratio"]), 1.0)
        self.assertEqual(
            payload["provider_ratio_restatement_noop_count"],
            1,
        )
        self.assertEqual(
            payload["provider_ratio_accounted_action_count"],
            1,
        )
        self.assertEqual(payload["provider_ratio_unaccounted_count"], 0)
        self.assertEqual(payload["unresolved_count"], 0)

        prices = observations[
            ["security_id", "session", "close"]
        ].copy()
        factors = build_adjustment_factors(
            prices,
            pd.concat(
                [provider_actions, generated],
                ignore_index=True,
            ),
            source_version="test",
        )
        self.assertEqual(
            factors["split_factor"].astype(float).tolist(),
            [1.0, 1.0],
        )

    def test_reference_audit_merge_deduplicates_overlap(self) -> None:
        def artifact(payload: dict[str, object]) -> SourceArtifact:
            return SourceArtifact(
                source="krx_reference_price_audit",
                source_url="https://data.krx.co.kr/",
                retrieved_at=utc_now_iso(),
                content=json.dumps(payload).encode(),
                content_type="application/json",
            )

        previous = artifact(
            {
                "schema": "krx_reference_price_audit/v1",
                "status": "passed",
                "observation_count": 4,
                "audit_start": "2026-01-02",
                "audit_end": "2026-01-05",
                "observation_counts_by_session": {
                    "2026-01-02": 2,
                    "2026-01-05": 2,
                },
                "unresolved_count": 0,
                "records": [
                    {
                        "security_id": "KR:A",
                        "session": "2026-01-05",
                        "record_kind": "provider_ratio_restatement_noop",
                        "resolution": "generated_official_restatement_noop",
                        "event_id": "noop",
                        "related_ratio_event_ids": ["provider-a"],
                    }
                ],
            }
        )
        current = artifact(
            {
                "schema": "krx_reference_price_audit/v1",
                "status": "passed",
                "observation_count": 5,
                "audit_start": "2026-01-05",
                "audit_end": "2026-01-06",
                "observation_counts_by_session": {
                    "2026-01-05": 2,
                    "2026-01-06": 3,
                },
                "provider_ratio_outside_audit_window_count": 7,
                "unresolved_count": 0,
                "records": [
                    {
                        "security_id": "KR:A",
                        "session": "2026-01-05",
                        "record_kind": "provider_ratio_restatement_noop",
                        "resolution": "covered_by_existing_restatement_noop",
                        "event_id": "noop",
                        "related_ratio_event_ids": ["provider-a"],
                    },
                    {
                        "security_id": "KR:B",
                        "session": "2026-01-06",
                        "record_kind": "reference_discontinuity",
                        "resolution": "generated_official_reference_adjustment",
                        "event_id": "reference-b",
                        "related_ratio_event_ids": ["provider-b"],
                    },
                    {
                        "security_id": "KR:A",
                        "session": "2020-01-02",
                        "record_kind": "provider_ratio_outside_audit_window",
                        "resolution": "outside_incremental_audit_window",
                        "event_id": "",
                        "related_ratio_event_ids": ["provider-a"],
                    },
                ],
            }
        )

        merged = json.loads(
            _combine_krx_reference_price_audits(
                previous,
                current,
            ).content
        )

        self.assertEqual(merged["observation_count"], 7)
        self.assertEqual(merged["audit_start"], "2026-01-02")
        self.assertEqual(merged["audit_end"], "2026-01-06")
        self.assertEqual(merged["reference_discontinuity_count"], 1)
        self.assertEqual(merged["generated_adjustment_count"], 2)
        self.assertEqual(merged["generated_restatement_noop_count"], 1)
        self.assertEqual(merged["provider_ratio_action_count"], 2)
        self.assertEqual(
            merged["provider_ratio_outside_audit_window_count"],
            0,
        )

    def test_reference_audit_merge_rejects_new_historical_ratio(self) -> None:
        def artifact(payload: dict[str, object]) -> SourceArtifact:
            return SourceArtifact(
                source="krx_reference_price_audit",
                source_url="https://data.krx.co.kr/",
                retrieved_at=utc_now_iso(),
                content=json.dumps(payload).encode(),
                content_type="application/json",
            )

        base = {
            "schema": "krx_reference_price_audit/v1",
            "status": "passed",
            "observation_count": 1,
            "audit_start": "2026-01-02",
            "audit_end": "2026-01-02",
            "observation_counts_by_session": {"2026-01-02": 1},
            "unresolved_count": 0,
        }
        previous = artifact(
            {
                **base,
                "records": [],
            }
        )
        current = artifact(
            {
                **base,
                "records": [
                    {
                        "security_id": "KR:A",
                        "session": "2020-01-02",
                        "record_kind": "provider_ratio_outside_audit_window",
                        "resolution": "outside_incremental_audit_window",
                        "event_id": "",
                        "related_ratio_event_ids": [
                            "new-historical-ratio"
                        ],
                    }
                ],
            }
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "full KR bootstrap",
        ):
            _combine_krx_reference_price_audits(
                previous,
                current,
            )

    def test_corporate_action_audit_matches_official_dividend(self) -> None:
        retrieved_at = utc_now_iso()
        reference_report = SourceArtifact(
            source="krx_reference_price_audit",
            source_url="https://data.krx.co.kr/",
            retrieved_at=retrieved_at,
            content=json.dumps(
                {
                    "schema": "krx_reference_price_audit/v1",
                    "status": "passed",
                    "reference_discontinuity_count": 0,
                    "generated_adjustment_count": 0,
                    "unresolved_count": 0,
                    "records": [],
                }
            ).encode(),
            content_type="application/json",
        )
        dart_report = SourceArtifact(
            source="opendart_action_audit",
            source_url="https://opendart.fss.or.kr/",
            retrieved_at=retrieved_at,
            content=b'{"status":"passed"}',
            content_type="application/json",
        )
        decisions = pd.DataFrame(
            [
                {
                    "security_id": "KR:TEST",
                    "record_date": "2026-03-31",
                    "cash_amount": 100.0,
                    "announcement_date": "2026-02-20",
                    "payment_date": "2026-04-15",
                    "rcept_no": "20260220800001",
                    "source_url": "https://dart.fss.or.kr/example",
                    "source_hash": "d" * 64,
                }
            ]
        )
        dart_result = KrDartDividendResult(
            "passed",
            decisions,
            dart_report,
        )
        action = {
            column: ""
            for column in dataset_spec(
                "corporate_actions"
            ).required_columns
        }
        action.update(
            {
                "event_id": "event",
                "security_id": "KR:TEST",
                "action_type": "cash_dividend",
                "effective_date": "2026-03-30",
                "ex_date": "2026-03-30",
                "cash_amount": 90.0,
                "currency": "KRW",
                "official": False,
                "source_url": "https://example.test/",
                "source_kind": "provider",
                "source": "test_provider",
                "retrieved_at": retrieved_at,
                "source_hash": "e" * 64,
            }
        )
        catalog = KrIdentityCatalog(
            pd.DataFrame(
                [
                    {
                        "security_id": "KR:TEST",
                        "primary_symbol": "005930",
                        "provider_symbol": "005930.KO",
                        "asset_type": "STOCK",
                    }
                ]
            ),
            (),
        )

        unsupported = {
            **action,
            "event_id": "unsupported-provider-event",
            "effective_date": "2025-01-02",
            "ex_date": "2025-01-02",
            "cash_amount": 50.0,
        }
        provider_actions = pd.DataFrame([action, unsupported])
        actions = _officialize_opendart_cash_dividends(
            provider_actions,
            dart_result,
            start="2015-01-01",
            end="2026-07-22",
        )
        report = _kr_corporate_action_audit(
            actions,
            catalog,
            dart_result,
            reference_report,
            start="2015-01-01",
            end="2026-07-22",
            provider_actions=provider_actions,
        )
        payload = json.loads(report.content)

        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["matched_dividend_count"], 1)
        self.assertEqual(
            payload["provider_dividend_amount_mismatch_count"],
            1,
        )
        self.assertNotIn(
            "event",
            set(actions["event_id"].astype(str)),
        )
        self.assertNotIn(
            "unsupported-provider-event",
            set(actions["event_id"].astype(str)),
        )
        self.assertEqual(
            payload["rejected_provider_dividend_count"],
            1,
        )
        self.assertEqual(
            payload["unmatched_provider_dividend_count"],
            0,
        )
        self.assertEqual(payload["blocking_issue_count"], 0)


if __name__ == "__main__":
    unittest.main()
