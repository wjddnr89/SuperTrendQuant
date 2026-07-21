from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from supertrend_quant.brokers import PaperBroker
from supertrend_quant.ledger import PortfolioLedger
from supertrend_quant.portfolio import Position


class CorporateActionLedgerTest(unittest.TestCase):
    def test_paper_state_persists_receivable_and_exact_once_payment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.json"
            path.write_text(
                json.dumps(
                    {
                        "cash": 100.0,
                        "positions": {"AAA": {"quantity": 10.0, "avg_price": 20.0}},
                        "metadata": {},
                    }
                )
            )
            action = {
                "event_id": "paper-dividend",
                "action_type": "cash_dividend",
                "symbol": "AAA",
                "effective_date": "2026-01-02",
                "ex_date": "2026-01-02",
                "payment_date": "2026-01-10",
                "cash_amount": 2.0,
            }

            first = PaperBroker(path, initial_cash=0.0)
            first.apply_corporate_actions([action], through="2026-01-02")
            pending = json.loads(path.read_text())
            self.assertIn(
                "paper-dividend",
                pending["metadata"]["corporate_action_receivables"],
            )

            restarted = PaperBroker(path, initial_cash=0.0)
            restarted.apply_corporate_actions([action], through="2026-01-10")
            restarted.apply_corporate_actions([action], through="2026-01-10")
            paid = json.loads(path.read_text())
            self.assertEqual(paid["cash"], 120.0)
            self.assertEqual(
                paid["metadata"]["processed_corporate_action_ids"],
                ["paper-dividend"],
            )

    def test_dividend_entitlement_is_receivable_until_payment_date(self):
        ledger = PortfolioLedger(
            cash=100.0,
            positions={"AAA": Position("AAA", 10.0, 20.0)},
            dividend_tax_rate=0.1,
        )
        action = {
            "event_id": "div-pay-later",
            "action_type": "cash_dividend",
            "symbol": "AAA",
            "ex_date": "2026-01-02",
            "effective_date": "2026-01-02",
            "payment_date": "2026-01-10",
            "cash_amount": 2.0,
        }

        entitlement = ledger.apply_actions([action], through="2026-01-02")
        before_payment = ledger.apply_actions([action], through="2026-01-09")
        payment = ledger.apply_actions([action], through="2026-01-10")

        self.assertEqual(ledger.cash, 118.0)
        self.assertEqual(entitlement[0].cash_delta, 0.0)
        self.assertEqual(before_payment, ())
        self.assertEqual(payment[0].cash_delta, 18.0)
        self.assertIn("div-pay-later", ledger.processed_event_ids)
        self.assertNotIn("div-pay-later", ledger.cash_receivables)

    def test_fractional_cash_in_lieu_requires_exact_terms(self):
        ledger = PortfolioLedger(
            cash=0.0,
            positions={"AAA": Position("AAA", 3.0, 10.0)},
        )
        action = {
            "event_id": "reverse-split",
            "action_type": "split",
            "symbol": "AAA",
            "effective_date": "2026-01-02",
            "ratio": 0.5,
            "metadata": {"allow_fractional": False, "cash_in_lieu_price": 20.0},
        }

        event = ledger.apply_actions([action], through="2026-01-02")[0]

        self.assertEqual(ledger.positions["AAA"].quantity, 1.0)
        self.assertEqual(ledger.cash, 10.0)
        self.assertEqual(event.cash_delta, 10.0)

        unresolved = PortfolioLedger(
            cash=0.0,
            positions={"BBB": Position("BBB", 3.0, 10.0)},
        )
        missing_terms = {**action, "event_id": "missing", "symbol": "BBB", "metadata": {"allow_fractional": False}}
        unresolved.apply_actions([missing_terms], through="2026-01-02")
        self.assertEqual(unresolved.positions["BBB"].quantity, 3.0)
        self.assertIn("missing", unresolved.unresolved_event_ids)

        corrected = {
            **missing_terms,
            "metadata": {"allow_fractional": False, "cash_in_lieu_price": 20.0},
        }
        unresolved.apply_actions([corrected], through="2026-01-03")
        self.assertNotIn("missing", unresolved.unresolved_event_ids)
        self.assertIn("missing", unresolved.processed_event_ids)

    def test_dividend_split_and_idempotency(self):
        ledger = PortfolioLedger(
            cash=100.0,
            positions={"AAA": Position("AAA", 10.0, 20.0)},
            dividend_tax_rate=0.1,
        )
        actions = [
            {
                "event_id": "div-1",
                "action_type": "cash_dividend",
                "symbol": "AAA",
                "effective_date": "2026-01-02",
                "cash_amount": 2.0,
            },
            {
                "event_id": "split-1",
                "action_type": "split",
                "symbol": "AAA",
                "effective_date": "2026-01-03",
                "ratio": 2.0,
            },
        ]

        first = ledger.apply_actions(actions, through="2026-01-03")
        second = ledger.apply_actions(actions, through="2026-01-03")

        self.assertEqual(len(first), 2)
        self.assertEqual(second, ())
        self.assertEqual(ledger.cash, 118.0)
        self.assertEqual(ledger.positions["AAA"].quantity, 20.0)
        self.assertEqual(ledger.positions["AAA"].avg_price, 10.0)

    def test_spinoff_stock_merger_ticker_change_cash_merger_and_delisting(self):
        ledger = PortfolioLedger(
            cash=0.0,
            positions={
                "AAA": Position("AAA", 10.0, 100.0),
                "CASH": Position("CASH", 2.0, 10.0),
                "DEAD": Position("DEAD", 3.0, 8.0),
            },
        )
        actions = [
            {
                "event_id": "spin",
                "action_type": "spinoff",
                "symbol": "AAA",
                "effective_date": "2026-01-02",
                "new_symbol": "SPIN",
                "ratio": 0.5,
                "metadata": {"cost_basis_fraction": 0.2},
            },
            {
                "event_id": "rename",
                "action_type": "ticker_change",
                "symbol": "SPIN",
                "effective_date": "2026-01-03",
                "new_symbol": "NEW",
            },
            {
                "event_id": "stock-merger",
                "action_type": "stock_merger",
                "symbol": "AAA",
                "effective_date": "2026-01-04",
                "new_symbol": "NEW",
                "ratio": 0.25,
                "cash_amount": 2.0,
            },
            {
                "event_id": "cash-merger",
                "action_type": "cash_merger",
                "symbol": "CASH",
                "effective_date": "2026-01-04",
                "cash_amount": 12.0,
            },
            {
                "event_id": "delist",
                "action_type": "delisting",
                "symbol": "DEAD",
                "effective_date": "2026-01-04",
                "cash_amount": 1.0,
            },
        ]

        ledger.apply_actions(actions, through="2026-01-04")

        self.assertNotIn("AAA", ledger.positions)
        self.assertNotIn("SPIN", ledger.positions)
        self.assertNotIn("CASH", ledger.positions)
        self.assertNotIn("DEAD", ledger.positions)
        self.assertEqual(ledger.positions["NEW"].quantity, 7.5)
        self.assertAlmostEqual(ledger.positions["NEW"].quantity * ledger.positions["NEW"].avg_price, 1000.0)
        self.assertEqual(ledger.cash, 47.0)

    def test_same_day_spinoff_is_applied_before_ticker_change(self):
        ledger = PortfolioLedger(
            cash=0.0,
            positions={"FBHS": Position("FBHS", 10.0, 100.0)},
        )
        # Event IDs intentionally sort in the wrong semantic order.  The
        # spin-off must still consume the FBHS entitlement before FBHS moves to
        # FBIN, otherwise the MBC child is silently lost.
        actions = [
            {
                "event_id": "a-rename",
                "action_type": "ticker_change",
                "symbol": "FBHS",
                "effective_date": "2026-01-05",
                "new_symbol": "FBIN",
            },
            {
                "event_id": "z-spin",
                "action_type": "spinoff",
                "symbol": "FBHS",
                "effective_date": "2026-01-05",
                "new_symbol": "MBC",
                "ratio": 1.0,
                "metadata": {"cost_basis_fraction": 0.4},
            },
        ]

        events = ledger.apply_actions(actions, through="2026-01-05")

        self.assertEqual([event.event_id for event in events], ["z-spin", "a-rename"])
        self.assertNotIn("FBHS", ledger.positions)
        self.assertEqual(ledger.positions["FBIN"].quantity, 10.0)
        self.assertEqual(ledger.positions["FBIN"].avg_price, 60.0)
        self.assertEqual(ledger.positions["MBC"].quantity, 10.0)
        self.assertEqual(ledger.positions["MBC"].avg_price, 40.0)

    def test_missing_spinoff_basis_is_unresolved_only_for_a_held_parent(self):
        action = {
            "event_id": "spin-no-basis",
            "action_type": "spinoff",
            "symbol": "PARENT",
            "effective_date": "2026-01-05",
            "new_symbol": "CHILD",
            "ratio": 1.0,
        }
        held = PortfolioLedger(
            cash=0.0,
            positions={"PARENT": Position("PARENT", 10.0, 100.0)},
        )
        event = held.apply_actions([action], through="2026-01-05")[0]
        self.assertIn("spin-no-basis", held.unresolved_event_ids)
        self.assertNotIn("spin-no-basis", held.processed_event_ids)
        self.assertNotIn("CHILD", held.positions)
        self.assertIn("cost-basis", event.message)

        unheld = PortfolioLedger(cash=0.0, positions={})
        unheld.apply_actions([action], through="2026-01-05")
        self.assertIn("spin-no-basis", unheld.processed_event_ids)
        self.assertNotIn("spin-no-basis", unheld.unresolved_event_ids)

    def test_delisting_without_settlement_amount_stays_unresolved(self):
        ledger = PortfolioLedger(
            cash=5.0,
            positions={"DEAD": Position("DEAD", 3.0, 8.0)},
        )
        action = {
            "event_id": "delist",
            "action_type": "delisting",
            "symbol": "DEAD",
            "effective_date": "2026-01-04",
            "cash_amount": None,
        }

        event = ledger.apply_actions([action], through="2026-01-04")[0]

        self.assertIn("DEAD", ledger.positions)
        self.assertEqual(ledger.cash, 5.0)
        self.assertIn("delist", ledger.unresolved_event_ids)
        self.assertNotIn("delist", ledger.processed_event_ids)
        self.assertIn("left unapplied", event.message)

        corrected = {**action, "cash_amount": 0.0}
        ledger.apply_actions([corrected], through="2026-01-05")

        self.assertNotIn("DEAD", ledger.positions)
        self.assertNotIn("delist", ledger.unresolved_event_ids)
        self.assertIn("delist", ledger.processed_event_ids)
        self.assertEqual(ledger.cash, 5.0)

    def test_capital_reduction_and_stock_dividend(self):
        ledger = PortfolioLedger(
            cash=0.0,
            positions={"AAA": Position("AAA", 10.0, 10.0)},
        )
        ledger.apply_actions(
            [
                {
                    "event_id": "reduce",
                    "action_type": "capital_reduction",
                    "symbol": "AAA",
                    "effective_date": "2026-01-02",
                    "ratio": 0.5,
                    "cash_amount": 1.0,
                },
                {
                    "event_id": "stock-div",
                    "action_type": "stock_dividend",
                    "symbol": "AAA",
                    "effective_date": "2026-01-03",
                    "ratio": 1.1,
                },
            ],
            through="2026-01-03",
        )
        self.assertAlmostEqual(ledger.positions["AAA"].quantity, 5.5)
        self.assertAlmostEqual(ledger.positions["AAA"].avg_price, 100.0 / 5.5)
        self.assertEqual(ledger.cash, 10.0)


if __name__ == "__main__":
    unittest.main()
