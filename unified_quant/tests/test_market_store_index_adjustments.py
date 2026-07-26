from __future__ import annotations

import unittest

import pandas as pd
import tempfile

from supertrend_quant.market_store.adjustments import (
    apply_adjustment_factors,
    build_adjustment_factors,
)
from supertrend_quant.market_store.index_membership import IndexEventReplayer
from supertrend_quant.market_store.index_ingest import IndexDataImporter
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.validation import (
    validate_dataset,
    validate_repository_snapshot,
)


class IndexEventReplayTest(unittest.TestCase):
    def setUp(self):
        self.anchors = pd.DataFrame(
            [
                {"index_id": "sp500", "anchor_date": "2020-01-01", "security_id": "A", "official": True},
                {"index_id": "sp500", "anchor_date": "2020-01-01", "security_id": "B", "official": True},
            ]
        )
        self.events = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "index_id": "sp500",
                    "effective_date": "2020-07-20",
                    "operation": "REMOVE",
                    "security_id": "A",
                    "official": True,
                },
                {
                    "event_id": "e2",
                    "index_id": "sp500",
                    "effective_date": "2020-07-20",
                    "operation": "ADD",
                    "security_id": "C",
                    "official": True,
                },
                {
                    "event_id": "e3",
                    "index_id": "sp500",
                    "effective_date": "2020-07-21",
                    "operation": "ADD",
                    "security_id": "D",
                    "official": False,
                },
            ]
        )

    def test_replays_actual_effective_date_not_quarter_boundaries(self):
        replay = IndexEventReplayer(self.anchors, self.events)

        before = replay.members_on("sp500", "2020-07-19")
        effective = replay.members_on("sp500", "2020-07-20")
        next_day = replay.members_on("sp500", "2020-07-21")

        self.assertEqual(before.security_ids, ("A", "B"))
        self.assertEqual(effective.security_ids, ("B", "C"))
        self.assertEqual(next_day.security_ids, ("B", "C", "D"))

    def test_all_three_target_indices_replay_anchor_add_and_remove(self):
        for index_id in ("sp500", "nasdaq100", "russell3000"):
            anchors = self.anchors.assign(index_id=index_id)
            events = self.events.iloc[:2].assign(index_id=index_id)
            membership = IndexEventReplayer(anchors, events).members_on(
                index_id, "2020-07-20", source_mode="official_only"
            )
            self.assertEqual(membership.security_ids, ("B", "C"))

    def test_official_only_rejects_incomplete_official_coverage(self):
        replay = IndexEventReplayer(self.anchors, self.events)
        with self.assertRaisesRegex(ValueError, "official_only coverage is incomplete"):
            replay.members_on("sp500", "2020-07-21", source_mode="official_only")

    def test_official_event_wins_conflict_and_same_grade_conflict_blocks(self):
        conflicting = pd.concat(
            [
                self.events.iloc[:2],
                pd.DataFrame(
                    [
                        {
                            "event_id": "lower-grade-conflict",
                            "index_id": "sp500",
                            "effective_date": "2020-07-20",
                            "operation": "ADD",
                            "security_id": "A",
                            "official": False,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        membership = IndexEventReplayer(self.anchors, conflicting).members_on(
            "sp500", "2020-07-20"
        )
        self.assertEqual(membership.security_ids, ("B", "C"))
        self.assertTrue(any("overrode" in warning for warning in membership.warnings))

        same_grade = conflicting.copy()
        same_grade.loc[same_grade["event_id"] == "lower-grade-conflict", "official"] = True
        with self.assertRaisesRegex(ValueError, "Unresolved same-grade"):
            IndexEventReplayer(self.anchors, same_grade).members_on("sp500", "2020-07-20")

    def test_custom_overlay_is_applied_after_official_membership(self):
        overlays = pd.DataFrame(
            [
                {
                    "overlay_id": "o1",
                    "index_id": "sp500",
                    "effective_from": "2020-07-21",
                    "effective_to": "",
                    "operation": "REMOVE",
                    "security_id": "B",
                },
                {
                    "overlay_id": "o2",
                    "index_id": "sp500",
                    "effective_from": "2020-07-21",
                    "effective_to": "",
                    "operation": "ADD",
                    "security_id": "X",
                },
            ]
        )
        membership = IndexEventReplayer(self.anchors, self.events, overlays).members_on(
            "sp500", "2020-07-21"
        )
        self.assertEqual(membership.security_ids, ("C", "D", "X"))
        self.assertEqual(membership.applied_overlay_ids, ("o1", "o2"))

    def test_importer_maps_historical_symbol_to_stable_security_id(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = LocalDatasetRepository(directory)
            history = pd.DataFrame(
                [
                    {
                        "security_id": "SEC-A",
                        "symbol": "OLD",
                        "exchange": "XNAS",
                        "effective_from": "2000-01-01",
                        "effective_to": "2020-01-31",
                        "source": "fixture",
                        "retrieved_at": "2026-01-01T00:00:00Z",
                        "source_hash": "hash",
                    },
                    {
                        "security_id": "SEC-A",
                        "symbol": "NEW",
                        "exchange": "XNAS",
                        "effective_from": "2020-02-01",
                        "effective_to": "",
                        "source": "fixture",
                        "retrieved_at": "2026-01-01T00:00:00Z",
                        "source_hash": "hash",
                    },
                ]
            )
            repository.write_frame("symbol_history", history, completed_session="2026-01-01")
            importer = IndexDataImporter(repository)
            importer.import_anchor(
                "sp500",
                "2020-01-15",
                pd.DataFrame({"symbol": ["OLD"]}),
                source="official_fixture",
                official=True,
            )
            importer.import_events(
                "sp500",
                pd.DataFrame(
                    {
                        "effective_date": ["2020-02-15"],
                        "operation": ["REMOVE"],
                        "symbol": ["NEW"],
                    }
                ),
                source="official_fixture",
                official=True,
            )
            anchors = repository.read_frame("index_constituent_anchors")
            events = repository.read_frame("index_membership_events")
            self.assertEqual(anchors.iloc[0]["security_id"], "SEC-A")
            self.assertEqual(events.iloc[0]["security_id"], "SEC-A")
            self.assertTrue(events.iloc[0]["official"])


class AdjustmentFactorTest(unittest.TestCase):
    def setUp(self):
        self.prices = pd.DataFrame(
            [
                {"security_id": "A", "session": "2020-01-01", "open": 98.0, "high": 101.0, "low": 97.0, "close": 100.0, "volume": 10.0},
                {"security_id": "A", "session": "2020-01-02", "open": 49.0, "high": 51.0, "low": 48.0, "close": 50.0, "volume": 20.0},
                {"security_id": "A", "session": "2020-01-03", "open": 49.0, "high": 50.0, "low": 48.0, "close": 49.0, "volume": 30.0},
            ]
        )

    def test_split_adjustment_restates_prior_ohlc_and_volume(self):
        actions = pd.DataFrame(
            [
                {
                    "event_id": "split",
                    "security_id": "A",
                    "action_type": "split",
                    "effective_date": "2020-01-02",
                    "ex_date": "2020-01-02",
                    "ratio": 2.0,
                    "cash_amount": None,
                }
            ]
        )
        factors = build_adjustment_factors(self.prices, actions, source_version="v1")
        adjusted = apply_adjustment_factors(self.prices, factors, mode="split_adjusted")

        first = adjusted.loc[adjusted["session"] == pd.Timestamp("2020-01-01")].iloc[0]
        self.assertEqual(first["close"], 50.0)
        self.assertEqual(first["volume"], 20.0)
        self.assertTrue(validate_dataset("adjustment_factors", factors).valid)

    def test_reference_price_adjustment_overrides_same_date_share_ratio(self):
        actions = pd.DataFrame(
            [
                {
                    "event_id": "stock-dividend-share-ratio",
                    "security_id": "A",
                    "action_type": "stock_dividend",
                    "effective_date": "2020-01-02",
                    "ex_date": "2020-01-02",
                    "ratio": 1.05,
                    "cash_amount": None,
                },
                {
                    "event_id": "krx-reference-price-ratio",
                    "security_id": "A",
                    "action_type": "reference_price_adjustment",
                    "effective_date": "2020-01-02",
                    "ex_date": "2020-01-02",
                    "ratio": 100.0 / 96.0,
                    "cash_amount": None,
                },
            ]
        )

        factors = build_adjustment_factors(
            self.prices,
            actions,
            source_version="v1",
        )

        self.assertAlmostEqual(
            factors.iloc[0]["split_factor"],
            0.96,
        )
        self.assertAlmostEqual(
            factors.iloc[0]["total_return_factor"],
            0.96,
        )

    def test_snapshot_validates_reference_price_override_as_one_transition(self):
        cases = (
            (
                "same-date-adjustment",
                self.prices,
                "2020-01-02",
                1.05,
                "2020-01-02",
                100.0 / 96.0,
            ),
            (
                "restatement-after-halt",
                pd.concat(
                    [
                        self.prices.iloc[[0]],
                        self.prices.iloc[[2]].assign(session="2020-01-10"),
                    ],
                    ignore_index=True,
                ),
                "2020-01-06",
                10.0,
                "2020-01-10",
                1.0,
            ),
        )
        for (
            label,
            prices,
            provider_date,
            provider_ratio,
            reference_date,
            reference_ratio,
        ) in cases:
            with self.subTest(label=label):
                actions = pd.DataFrame(
                    [
                        {
                            "event_id": f"{label}-provider-split",
                            "security_id": "A",
                            "action_type": "split",
                            "effective_date": provider_date,
                            "ex_date": provider_date,
                            "ratio": provider_ratio,
                            "cash_amount": None,
                        },
                        {
                            "event_id": f"{label}-krx-reference",
                            "security_id": "A",
                            "action_type": "reference_price_adjustment",
                            "effective_date": reference_date,
                            "ex_date": reference_date,
                            "ratio": reference_ratio,
                            "cash_amount": None,
                        },
                    ]
                )
                factors = build_adjustment_factors(
                    prices,
                    actions,
                    source_version="v1",
                )

                class Snapshot:
                    frames = {
                        "daily_price_raw": prices,
                        "corporate_actions": actions,
                        "adjustment_factors": factors,
                    }

                    def current_manifest(self, dataset):
                        return object() if dataset in self.frames else None

                    def read_frame(self, dataset):
                        return self.frames[dataset]

                report = validate_repository_snapshot(Snapshot())

                self.assertNotIn(
                    "action_factor_mismatch",
                    {issue.code for issue in report.issues},
                )

    def test_opendart_dividend_overrides_same_date_provider_amount(self):
        actions = pd.DataFrame(
            [
                {
                    "event_id": "provider-dividend",
                    "security_id": "A",
                    "action_type": "cash_dividend",
                    "effective_date": "2020-01-02",
                    "ex_date": "2020-01-02",
                    "ratio": None,
                    "cash_amount": 9.0,
                    "official": False,
                    "source": "provider",
                },
                {
                    "event_id": "official-dividend",
                    "security_id": "A",
                    "action_type": "cash_dividend",
                    "effective_date": "2020-01-02",
                    "ex_date": "2020-01-02",
                    "ratio": None,
                    "cash_amount": 10.0,
                    "official": True,
                    "source": "opendart_cash_dividend_decision",
                },
            ]
        )

        factors = build_adjustment_factors(
            self.prices,
            actions,
            source_version="v1",
        )

        self.assertAlmostEqual(
            factors.iloc[0]["total_return_factor"],
            0.9,
        )

    def test_total_return_factor_includes_dividend_and_tax_setting(self):
        actions = pd.DataFrame(
            [
                {
                    "event_id": "dividend",
                    "security_id": "A",
                    "action_type": "cash_dividend",
                    "effective_date": "2020-01-02",
                    "ex_date": "2020-01-02",
                    "ratio": None,
                    "cash_amount": 10.0,
                }
            ]
        )
        gross = build_adjustment_factors(self.prices, actions, source_version="v1", dividend_tax_rate=0.0)
        taxed = build_adjustment_factors(self.prices, actions, source_version="v1", dividend_tax_rate=0.5)
        gross_first = gross.loc[gross["session"] == pd.Timestamp("2020-01-01"), "total_return_factor"].iloc[0]
        taxed_first = taxed.loc[taxed["session"] == pd.Timestamp("2020-01-01"), "total_return_factor"].iloc[0]
        self.assertAlmostEqual(gross_first, 0.9)
        self.assertAlmostEqual(taxed_first, 0.95)

    def test_multiple_same_date_actions_and_later_actions_accumulate_in_date_order(self):
        actions = pd.DataFrame(
            [
                {
                    "event_id": "same-date-stock-dividend",
                    "security_id": "A",
                    "action_type": "stock_dividend",
                    "effective_date": "2020-01-02",
                    "ex_date": "2020-01-02",
                    "ratio": 1.25,
                    "cash_amount": None,
                },
                {
                    "event_id": "later-split",
                    "security_id": "A",
                    "action_type": "split",
                    "effective_date": "2020-01-03",
                    "ex_date": "2020-01-03",
                    "ratio": 4.0,
                    "cash_amount": None,
                },
                {
                    "event_id": "same-date-split",
                    "security_id": "A",
                    "action_type": "split",
                    "effective_date": "2020-01-02",
                    "ex_date": "2020-01-02",
                    "ratio": 2.0,
                    "cash_amount": None,
                },
                {
                    "event_id": "same-date-cash",
                    "security_id": "A",
                    "action_type": "cash_dividend",
                    "effective_date": "2020-01-02",
                    "ex_date": "2020-01-02",
                    "ratio": None,
                    "cash_amount": 10.0,
                },
            ]
        )

        factors = build_adjustment_factors(self.prices, actions, source_version="v1")

        self.assertEqual(list(factors["session"]), list(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])))
        self.assertAlmostEqual(factors.iloc[0]["split_factor"], 0.1)
        self.assertAlmostEqual(factors.iloc[1]["split_factor"], 0.25)
        self.assertAlmostEqual(factors.iloc[2]["split_factor"], 1.0)
        self.assertAlmostEqual(factors.iloc[0]["total_return_factor"], 0.09)
        self.assertAlmostEqual(factors.iloc[1]["total_return_factor"], 0.25)
        self.assertAlmostEqual(factors.iloc[2]["total_return_factor"], 1.0)

    def test_blank_or_missing_ex_date_falls_back_to_effective_date(self):
        actions = pd.DataFrame(
            [
                {
                    "event_id": "blank-ex-date-split",
                    "security_id": "A",
                    "action_type": "split",
                    "effective_date": "2020-01-02",
                    "ex_date": "",
                    "ratio": 2.0,
                    "cash_amount": None,
                },
                {
                    "event_id": "missing-ex-date-dividend",
                    "security_id": "A",
                    "action_type": "cash_dividend",
                    "effective_date": "2020-01-03",
                    "ex_date": None,
                    "ratio": None,
                    "cash_amount": 4.0,
                },
            ]
        )

        factors = build_adjustment_factors(self.prices, actions, source_version="v1")

        self.assertAlmostEqual(factors.iloc[0]["split_factor"], 0.5)
        self.assertAlmostEqual(factors.iloc[1]["split_factor"], 1.0)
        self.assertAlmostEqual(factors.iloc[0]["total_return_factor"], 0.46)
        self.assertAlmostEqual(factors.iloc[1]["total_return_factor"], 0.92)
        self.assertAlmostEqual(factors.iloc[2]["total_return_factor"], 1.0)

    def test_action_during_trading_halt_applies_at_first_later_price(self):
        prices = pd.concat(
            [
                self.prices.iloc[[0]],
                self.prices.iloc[[2]].assign(session="2020-01-10"),
            ],
            ignore_index=True,
        )
        actions = pd.DataFrame(
            [
                {
                    "event_id": "halt-split",
                    "security_id": "A",
                    "action_type": "split",
                    "effective_date": "2020-01-06",
                    "ex_date": "2020-01-06",
                    "ratio": 2.0,
                    "cash_amount": None,
                }
            ]
        )

        factors = build_adjustment_factors(
            prices,
            actions,
            source_version="v1",
        )

        self.assertEqual(list(factors["split_factor"]), [0.5, 1.0])
        self.assertEqual(list(factors["total_return_factor"]), [0.5, 1.0])

    def test_snapshot_groups_actions_at_same_post_halt_price_transition(self):
        prices = pd.concat(
            [
                self.prices.iloc[[0]],
                self.prices.iloc[[2]].assign(session="2020-01-10"),
            ],
            ignore_index=True,
        )
        actions = pd.DataFrame(
            [
                {
                    "event_id": "halt-split",
                    "security_id": "A",
                    "action_type": "split",
                    "effective_date": "2020-01-06",
                    "ex_date": "2020-01-06",
                    "ratio": 2.0,
                    "cash_amount": None,
                },
                {
                    "event_id": "resume-reduction",
                    "security_id": "A",
                    "action_type": "capital_reduction",
                    "effective_date": "2020-01-10",
                    "ex_date": "2020-01-10",
                    "ratio": 0.25,
                    "cash_amount": None,
                },
            ]
        )
        factors = build_adjustment_factors(
            prices,
            actions,
            source_version="v1",
        )

        class Snapshot:
            frames = {
                "daily_price_raw": prices,
                "corporate_actions": actions,
                "adjustment_factors": factors,
            }

            def current_manifest(self, dataset):
                return object() if dataset in self.frames else None

            def read_frame(self, dataset):
                return self.frames[dataset]

        report = validate_repository_snapshot(Snapshot())

        self.assertNotIn(
            "action_factor_mismatch",
            {issue.code for issue in report.issues},
        )

    def test_special_dividends_use_cash_distribution_factor_and_tax(self):
        prices = self.prices.assign(close=[20.0, 19.0, 18.0])
        actions = pd.DataFrame(
            [
                {
                    "event_id": "special-2",
                    "security_id": "A",
                    "action_type": "special_dividend",
                    "effective_date": "2020-01-02",
                    "ex_date": "2020-01-02",
                    "ratio": None,
                    "cash_amount": 0.1462,
                },
                {
                    "event_id": "special-1",
                    "security_id": "A",
                    "action_type": "special_dividend",
                    "effective_date": "2020-01-02",
                    "ex_date": "2020-01-02",
                    "ratio": None,
                    "cash_amount": 0.25,
                },
            ]
        )

        gross = build_adjustment_factors(prices, actions, source_version="v1")
        taxed = build_adjustment_factors(prices, actions, source_version="v1", dividend_tax_rate=0.5)

        self.assertAlmostEqual(
            gross.iloc[0]["total_return_factor"],
            ((20.0 - 0.25) / 20.0) * ((20.0 - 0.1462) / 20.0),
        )
        self.assertAlmostEqual(
            taxed.iloc[0]["total_return_factor"],
            ((20.0 - 0.125) / 20.0) * ((20.0 - 0.0731) / 20.0),
        )
        self.assertEqual(gross.iloc[0]["split_factor"], 1.0)

    def test_no_actions_produces_identity_factors(self):
        actions = pd.DataFrame(
            columns=[
                "event_id",
                "security_id",
                "action_type",
                "effective_date",
                "ex_date",
                "ratio",
                "cash_amount",
            ]
        )

        factors = build_adjustment_factors(self.prices, actions, source_version="v1")

        self.assertEqual(list(factors["split_factor"]), [1.0, 1.0, 1.0])
        self.assertEqual(list(factors["total_return_factor"]), [1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
