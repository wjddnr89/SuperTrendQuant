from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo
from datetime import datetime

import pandas as pd

from supertrend_quant.config import load_split_config
from supertrend_quant.holdings import HoldingsStore
from supertrend_quant.live_runtime import HybridLiveRuntime, _daily_data_gap
from supertrend_quant.live_state import (
    DailyRiskStateStore,
    LiveOrderLedger,
    SignalPlanStore,
    build_signal_plan,
)
from supertrend_quant.market_store.manifest import DataRelease
from supertrend_quant.market_store.provider import configured_release_identity_issue
from supertrend_quant.portfolio import AccountSnapshot, OrderIntent, Position
from supertrend_quant.runtime import daily_execution_window


def _config():
    base = load_split_config(
        "configs/strategies/leader_rotation.yaml",
        "configs/runtimes/live_toss.yaml",
    )
    return replace(base, symbols=("AAA",), universe=replace(base.universe, symbols=("AAA",)))


class NoCallBroker:
    def get_account(self, market):  # pragma: no cover - kill switch must stop first
        raise AssertionError("broker must not be called")


class LiveEodContractTest(unittest.TestCase):
    def test_xkrx_execution_window_is_first_fifteen_minutes(self) -> None:
        allowed = daily_execution_window(
            "KR",
            datetime(2026, 7, 8, 9, 14, tzinfo=ZoneInfo("Asia/Seoul")),
            minutes=15,
        )
        expired = daily_execution_window(
            "KR",
            datetime(2026, 7, 8, 9, 15, tzinfo=ZoneInfo("Asia/Seoul")),
            minutes=15,
        )

        self.assertIsNotNone(allowed)
        self.assertTrue(allowed.allowed)
        self.assertEqual(allowed.execution_session, "2026-07-08")
        self.assertEqual(allowed.signal_session, "2026-07-07")
        self.assertFalse(expired.allowed)

    def test_non_session_has_no_execution_window(self) -> None:
        value = daily_execution_window(
            "KR",
            datetime(2026, 7, 12, 9, 5, tzinfo=ZoneInfo("Asia/Seoul")),
        )

        self.assertIsNone(value)

    def test_degraded_release_requires_an_explicit_warning_allowlist(self) -> None:
        frame = pd.DataFrame(
            {"Open": [1], "High": [1], "Low": [1], "Close": [1]},
            index=[pd.Timestamp("2026-07-07")],
        )
        market_data = SimpleNamespace(
            bars={"AAA": frame},
            data_quality="degraded",
            warnings=("late_action: reviewed",),
            completed_session="2026-07-07",
        )

        blocked = _daily_data_gap(
            ["AAA"],
            market_data,
            expected_signal_session="2026-07-07",
            required_quality="valid",
        )
        allowed = _daily_data_gap(
            ["AAA"],
            market_data,
            expected_signal_session="2026-07-07",
            required_quality="valid",
            allowed_degraded_warning_codes=("late_action",),
        )

        self.assertIn("non-allowlisted", blocked)
        self.assertIsNone(allowed)

    def test_signal_plan_is_immutable_and_idempotent(self) -> None:
        config = _config()
        account = AccountSnapshot(cash=1_000)
        order = OrderIntent(
            "AAA",
            "buy",
            5,
            client_order_id="stq-us-20260707-b-AAA",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = SignalPlanStore(Path(tmp) / "signal-plan.json")
            plan = build_signal_plan(
                config=config,
                market="US",
                signal_session="2026-07-07",
                execution_session="2026-07-08",
                expires_at="2026-07-08T09:45:00-04:00",
                data_version="release=v1",
                orders=(order,),
                account=account,
            )

            first, created = store.ensure(plan)
            second, created_again = store.ensure(plan)
            changed = {**plan, "data_version": "release=v2"}

            self.assertTrue(created)
            self.assertFalse(created_again)
            self.assertEqual(first["plan_hash"], second["plan_hash"])
            self.assertEqual(len(list((Path(tmp) / "plans").glob("*.json"))), 1)
            with self.assertRaisesRegex(RuntimeError, "different durable live plan"):
                store.ensure(changed)

    def test_order_ledger_blocks_duplicate_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = LiveOrderLedger(Path(tmp) / "orders.jsonl")
            ledger.append(
                {
                    "client_order_id": "cid-1",
                    "symbol": "AAA",
                    "side": "buy",
                    "status": "submitting",
                }
            )
            ledger.append(
                {
                    "client_order_id": "cid-1",
                    "symbol": "AAA",
                    "side": "buy",
                    "status": "accepted",
                }
            )

            self.assertTrue(ledger.already_submitted("cid-1"))
            self.assertEqual(len(ledger.events()), 2)

    def test_reconciliation_uses_official_nested_fill_and_terminal_status(self) -> None:
        class DetailBroker:
            def get_order(self, order_id):
                self.requested = order_id
                return {
                    "orderId": order_id,
                    "status": "FILLED",
                    "execution": {"filledQuantity": "5"},
                }

        config = _config()
        order = OrderIntent(
            "AAA", "buy", 5, client_order_id="stq-us-20260707-b-AAA"
        )
        with tempfile.TemporaryDirectory() as tmp:
            broker = DetailBroker()
            runtime = HybridLiveRuntime(
                config,
                broker=broker,
                holdings=HoldingsStore(Path(tmp) / "holdings.json"),
            )
            runtime.signal_plan_store.ensure(
                build_signal_plan(
                    config=config,
                    market="US",
                    signal_session="2026-07-07",
                    execution_session="2026-07-08",
                    expires_at="2026-07-08T09:45:00-04:00",
                    data_version="release=v1",
                    orders=(order,),
                    account=AccountSnapshot(cash=1_000),
                )
            )
            runtime.order_ledger.append(
                {
                    "client_order_id": order.client_order_id,
                    "symbol": "AAA",
                    "side": "buy",
                    "quantity": 5,
                    "status": "accepted",
                    "broker_order_id": "broker-1",
                }
            )

            runtime._reconcile_order_ledger(
                AccountSnapshot(
                    cash=500,
                    positions={"AAA": Position("AAA", 5, 100)},
                ),
                [],
            )
            latest = runtime.order_ledger.latest_by_client_id()[
                order.client_order_id
            ]

        self.assertEqual(broker.requested, "broker-1")
        self.assertEqual(latest["status"], "filled")
        self.assertEqual(latest["filled_quantity"], 5)

    def test_daily_loss_state_disables_entries_after_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = DailyRiskStateStore(Path(tmp) / "risk.json")
            risk = replace(_config().risk, max_daily_loss_pct=0.03)
            issue, _ = store.update_and_check(
                session="2026-07-08",
                account=AccountSnapshot(cash=1_000, total_asset_value=1_000),
                risk=risk,
            )
            breached, state = store.update_and_check(
                session="2026-07-08",
                account=AccountSnapshot(cash=960, total_asset_value=960),
                risk=risk,
            )

            self.assertIsNone(issue)
            self.assertIn("Daily loss limit reached", breached)
            self.assertAlmostEqual(state["loss_pct"], 0.04)

    def test_max_order_notional_caps_only_buy_quantity(self) -> None:
        config = replace(
            _config(),
            risk=replace(_config().risk, max_order_notional=1_000),
        )
        with tempfile.TemporaryDirectory() as tmp:
            runtime = HybridLiveRuntime(
                config,
                broker=NoCallBroker(),
                holdings=HoldingsStore(Path(tmp) / "holdings.json"),
            )

            quantity = runtime._limit_buy_notional(config, 100, 100)

        self.assertLessEqual(quantity, 9)

    def test_live_release_identity_requires_exact_r2_current_bytes(self) -> None:
        config = _config()
        config = replace(
            config,
            data_store=replace(
                config.data_store,
                provider="parquet",
                r2=replace(config.data_store.r2, enabled=True),
            ),
        )
        local = DataRelease.create(
            "2026-07-07",
            {"daily_price_raw": "prices-v1"},
        )
        remote_store = SimpleNamespace(
            get=lambda key: SimpleNamespace(data=local.to_bytes())
        )
        with (
            patch(
                "supertrend_quant.market_store.provider.LocalDatasetRepository"
            ) as repository,
            patch(
                "supertrend_quant.market_store.storage.R2ObjectStore",
                return_value=remote_store,
            ),
        ):
            repository.return_value.current_release.return_value = (local, "etag")
            matching = configured_release_identity_issue(config)
            remote_store.get = lambda key: SimpleNamespace(
                data=DataRelease.create(
                    "2026-07-07", {"daily_price_raw": "prices-v2"}
                ).to_bytes()
            )
            mismatch = configured_release_identity_issue(config)

        self.assertIsNone(matching)
        self.assertIn("identities differ", mismatch)

    def test_kill_switch_stops_before_broker_or_data_access(self) -> None:
        config = _config()
        with tempfile.TemporaryDirectory() as tmp:
            holdings = HoldingsStore(Path(tmp) / "holdings.json")
            runtime = HybridLiveRuntime(
                config,
                broker=NoCallBroker(),
                holdings=holdings,
            )
            runtime.kill_switch_path.write_text("stop\n", encoding="utf-8")

            plan, results = runtime.run_once(
                ignore_schedule=True,
                assume_yes=True,
            )

        self.assertEqual(plan.orders, ())
        self.assertIn("kill switch", results[0].lower())


if __name__ == "__main__":
    unittest.main()
