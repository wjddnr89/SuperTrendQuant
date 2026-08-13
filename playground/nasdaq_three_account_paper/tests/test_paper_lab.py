from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT))

from reports import PerformanceReporter  # noqa: E402
from intraday_exit import (  # noqa: E402
    IntradayExitSignal,
    build_active_intraday_fence,
    override_held_exit_trends,
    replay_intraday_fence_exit,
)
from toss_data import (  # noqa: E402
    CandlePage,
    TossMarketDataClient,
    regular_session_bars,
)
from lab import (  # noqa: E402
    ThreeAccountPaperLab,
    atr_target_weight,
    confirmed_two_hour_recovery,
    latest_closed_session_cutoff,
    resize_buys_to_atr_risk,
    resize_explicit_buys_to_execution_cash,
)
from supertrend_quant.portfolio import (  # noqa: E402
    AccountSnapshot,
    OrderIntent,
    OrderPlan,
    Position,
    PositionEconomics,
)


class SessionCutoffTests(unittest.TestCase):
    def test_excludes_provider_current_day_before_close(self):
        market_tz = ZoneInfo("America/New_York")

        self.assertEqual(
            latest_closed_session_cutoff(
                datetime(2026, 7, 28, 7, 30, tzinfo=market_tz),
                regular_close="16:00",
            ),
            date(2026, 7, 27),
        )
        self.assertEqual(
            latest_closed_session_cutoff(
                datetime(2026, 7, 28, 16, 30, tzinfo=market_tz),
                regular_close="16:00",
            ),
            date(2026, 7, 28),
        )


class ExecutionSizingTests(unittest.TestCase):
    def test_atr_target_weight_caps_low_volatility_at_full_allocation(self):
        self.assertEqual(atr_target_weight(0.02, 0.025), 1.0)
        self.assertAlmostEqual(atr_target_weight(0.05, 0.025), 0.5)
        self.assertAlmostEqual(atr_target_weight(0.05, 0.02), 0.4)

    def test_atr_sizing_replaces_initial_buy_with_explicit_quantity(self):
        signal_ts = pd.Timestamp("2026-07-31")
        frame = pd.DataFrame(
            {"Close": [100.0], "ATR_pct": [0.05]},
            index=[signal_ts],
        )
        plan = OrderPlan(
            strategy_name="paper",
            mode="paper",
            orders=(
                OrderIntent(
                    symbol="FTNT",
                    side="buy",
                    quantity=99,
                    reason="Top-ranked leader",
                ),
            ),
        )

        resized, notes = resize_buys_to_atr_risk(
            plan,
            {"FTNT": frame},
            signal_ts,
            AccountSnapshot(cash=10_000.0),
            target_atr_risk_pct=0.025,
            fee_rate=0.001,
            slippage_rate=0.0005,
        )

        self.assertEqual(resized.orders[0].quantity, 49)
        self.assertIsNone(resized.orders[0].cash_allocation_pct)
        self.assertIn("ATR risk weight 50.00%", resized.orders[0].reason)
        self.assertIn("quantity 49", notes[0])

    def test_atr_sizing_converts_post_sell_cash_allocation_to_fixed_quantity(self):
        signal_ts = pd.Timestamp("2026-07-31")
        frame = pd.DataFrame(
            {"Close": [100.0], "ATR_pct": [0.05]},
            index=[signal_ts],
        )
        account = AccountSnapshot(
            cash=100.0,
            positions={"AMD": Position("AMD", 100, 90.0)},
            position_economics={
                "AMD": PositionEconomics(
                    entry_cost=9_000.0,
                    raw_mark=100.0,
                    estimated_exit_proceeds=9_985.005,
                )
            },
        )
        plan = OrderPlan(
            strategy_name="paper",
            mode="paper",
            orders=(
                OrderIntent("AMD", "sell", 100, reason="Leader rotation"),
                OrderIntent(
                    "FTNT",
                    "buy",
                    None,
                    cash_allocation_pct=1.0,
                    required_sell_symbols=("AMD",),
                ),
            ),
        )

        resized, _ = resize_buys_to_atr_risk(
            plan,
            {"FTNT": frame},
            signal_ts,
            account,
            target_atr_risk_pct=0.025,
            fee_rate=0.001,
            slippage_rate=0.0005,
        )

        self.assertEqual(resized.orders[1].quantity, 50)
        self.assertIsNone(resized.orders[1].cash_allocation_pct)
        self.assertEqual(resized.orders[1].required_sell_symbols, ("AMD",))

    def test_execution_cash_guard_counts_same_plan_sell_proceeds(self):
        plan = OrderPlan(
            strategy_name="paper",
            mode="paper",
            orders=(
                OrderIntent("AMD", "sell", 10, reason="Leader rotation"),
                OrderIntent(
                    "FTNT",
                    "buy",
                    9,
                    required_sell_symbols=("AMD",),
                ),
            ),
        )
        account = AccountSnapshot(
            cash=0.0,
            positions={"AMD": Position("AMD", 10, 90.0)},
        )

        resized, notes = resize_explicit_buys_to_execution_cash(
            plan,
            account,
            {"AMD": 100.0, "FTNT": 100.0},
            fee_rate=0.001,
            slippage_rate=0.0005,
        )

        self.assertEqual(len(resized.orders), 2)
        self.assertEqual(resized.orders[1].quantity, 9)
        self.assertEqual(notes, ())

    def test_gap_up_resizes_explicit_buy_to_affordable_quantity(self):
        plan = OrderPlan(
            strategy_name="paper",
            mode="paper",
            orders=(
                OrderIntent(
                    symbol="FTNT",
                    side="buy",
                    quantity=65,
                    reason="Top-ranked leader",
                ),
            ),
        )

        resized, notes = resize_explicit_buys_to_execution_cash(
            plan,
            AccountSnapshot(cash=10_000.0),
            {"FTNT": 166.76},
            fee_rate=0.001,
            slippage_rate=0.0005,
        )

        self.assertEqual(resized.orders[0].quantity, 59)
        self.assertIn("65 -> 59", notes[0])

    def test_cash_allocated_buy_remains_execution_sized_by_broker(self):
        plan = OrderPlan(
            strategy_name="paper",
            mode="paper",
            orders=(
                OrderIntent(
                    symbol="FTNT",
                    side="buy",
                    quantity=None,
                    cash_allocation_pct=1.0,
                    reason="Post-sell leader entry",
                ),
            ),
        )

        resized, notes = resize_explicit_buys_to_execution_cash(
            plan,
            AccountSnapshot(cash=10_000.0),
            {"FTNT": 166.76},
            fee_rate=0.001,
            slippage_rate=0.0005,
        )

        self.assertIsNone(resized.orders[0].quantity)
        self.assertEqual(notes, ())


class PagingClient(TossMarketDataClient):
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def candle_page(
        self,
        symbol,
        *,
        interval,
        count=200,
        before=None,
        adjusted=True,
    ):
        self.calls.append((symbol, interval, count, before, adjusted))
        return self.pages.pop(0)


def frame(start: str, periods: int) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="min", tz="America/New_York")
    return pd.DataFrame(
        {
            "Open": range(periods),
            "High": range(periods),
            "Low": range(periods),
            "Close": range(periods),
            "Volume": [1] * periods,
        },
        index=index,
    )


class TossPagingTests(unittest.TestCase):
    def test_fetch_candles_follows_next_before_beyond_200(self):
        first = frame("2026-07-28 12:40", 200)
        second = frame("2026-07-28 09:30", 190)
        client = PagingClient(
            [
                CandlePage(first, "2026-07-28T12:40:00-04:00"),
                CandlePage(second, None),
            ]
        )

        result = client.fetch_candles(
            "QQQ",
            interval="1m",
            minimum_bars=390,
            adjusted=False,
        )

        self.assertEqual(len(result), 390)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(
            client.calls[1][3],
            "2026-07-28T12:40:00-04:00",
        )

    def test_first_regular_minute_pages_and_ignores_premarket(self):
        late = frame("2026-07-28 12:40", 200)
        early = frame("2026-07-28 07:00", 180)
        client = PagingClient(
            [
                CandlePage(late, "cursor-1"),
                CandlePage(early, None),
            ]
        )

        row = client.first_regular_minute("QQQ", date(2026, 7, 28))

        self.assertIsNotNone(row)
        self.assertEqual(float(row["Open"]), 150.0)


class IntradayExitTests(unittest.TestCase):
    def test_reentry_release_requires_a_bullish_bar_completed_after_exit(self):
        self.assertTrue(
            confirmed_two_hour_recovery(
                signal_at="2026-07-31T15:30:00-04:00",
                trend=1,
                blocked_at="2026-07-31T15:45:00-04:00",
                timezone="America/New_York",
                regular_close="16:00",
            )
        )
        self.assertFalse(
            confirmed_two_hour_recovery(
                signal_at="2026-07-31T13:30:00-04:00",
                trend=1,
                blocked_at="2026-07-31T15:45:00-04:00",
                timezone="America/New_York",
                regular_close="16:00",
            )
        )
        self.assertFalse(
            confirmed_two_hour_recovery(
                signal_at="2026-07-31T15:30:00-04:00",
                trend=-1,
                blocked_at="2026-07-31T15:45:00-04:00",
                timezone="America/New_York",
                regular_close="16:00",
            )
        )

    def test_completed_session_replay_fills_the_next_minute_open(self):
        seed_rows = []
        seed_index = []
        price = 100.0
        for session in pd.bdate_range("2026-06-01", periods=30):
            for clock in ("09:30", "11:30", "13:30", "15:30"):
                seed_index.append(
                    pd.Timestamp(
                        f"{session.date()} {clock}",
                        tz="America/New_York",
                    )
                )
                seed_rows.append(
                    {
                        "Open": price,
                        "High": price + 1.0,
                        "Low": price - 0.5,
                        "Close": price + 0.8,
                        "Volume": 1000,
                    }
                )
                price += 1.0
        seed = pd.DataFrame(seed_rows, index=seed_index)
        minute_index = pd.date_range(
            "2026-07-13 09:30",
            periods=390,
            freq="min",
            tz="America/New_York",
        )
        adjusted = pd.DataFrame(
            {
                "Open": 100.0,
                "High": 100.5,
                "Low": 99.5,
                "Close": 100.0,
                "Volume": 1000,
            },
            index=minute_index,
        )
        raw = adjusted.copy()
        raw.loc[minute_index[1], "Open"] = 98.5
        raw.loc[minute_index[2], "Open"] = 97.5

        replay = replay_intraday_fence_exit(
            "FTNT",
            adjusted,
            raw,
            seed_bars=seed,
            session_date=date(2026, 7, 13),
            period=10,
            multiplier=3.0,
        )

        self.assertIsNotNone(replay)
        self.assertTrue(replay.signal_at.startswith("2026-07-13T09:30"))
        self.assertTrue(replay.fill_at.startswith("2026-07-13T09:31"))
        self.assertEqual(replay.raw_fill_open, 98.5)

        confirmed = replay_intraday_fence_exit(
            "FTNT",
            adjusted,
            raw,
            seed_bars=seed,
            session_date=date(2026, 7, 13),
            period=10,
            multiplier=3.0,
            confirm_minutes=2,
        )
        self.assertTrue(confirmed.signal_at.startswith("2026-07-13T09:31"))
        self.assertTrue(confirmed.fill_at.startswith("2026-07-13T09:32"))
        self.assertEqual(confirmed.raw_fill_open, 97.5)

    def test_active_fence_uses_only_completed_two_hour_bars(self):
        rows = []
        timestamps = []
        price = 100.0
        for session in pd.bdate_range("2026-06-01", periods=30):
            for clock in ("09:30", "11:30", "13:30", "15:30"):
                timestamp = pd.Timestamp(
                    f"{session.date()} {clock}",
                    tz="America/New_York",
                )
                timestamps.append(timestamp)
                rows.append(
                    {
                        "Open": price,
                        "High": price + 1.0,
                        "Low": price - 0.5,
                        "Close": price + 0.8,
                        "Volume": 1000,
                    }
                )
                price += 1.0
        seed = pd.DataFrame(rows, index=timestamps)
        as_of = datetime(2026, 7, 13, 10, 15, tzinfo=ZoneInfo("America/New_York"))

        fence = build_active_intraday_fence(
            "FTNT",
            pd.DataFrame(),
            seed_bars=seed,
            as_of=as_of,
            period=10,
            multiplier=3.0,
        )

        self.assertEqual(fence.trend, 1)
        self.assertTrue(fence.bar_at.startswith("2026-07-10T15:30"))
        self.assertGreater(fence.lower_fence, 0.0)

    def test_regular_session_two_hour_bars_are_anchored_at_0930(self):
        minutes = frame("2026-07-31 09:30", 390)

        bars = regular_session_bars(
            minutes,
            bar_minutes=120,
            through=date(2026, 7, 31),
        )

        self.assertEqual(len(bars), 4)
        self.assertEqual(
            [timestamp.strftime("%H:%M") for timestamp in bars.index],
            ["09:30", "11:30", "13:30", "15:30"],
        )
        self.assertEqual(float(bars.iloc[0]["Open"]), 0.0)
        self.assertEqual(float(bars.iloc[-1]["Close"]), 389.0)

    def test_only_latest_held_exit_trend_is_overridden(self):
        index = pd.bdate_range("2026-07-29", periods=3)
        original = pd.DataFrame(
            {"Trend": [1, 1, 1], "Score": [1.0, 2.0, 3.0]},
            index=index,
        )
        signal = IntradayExitSignal(
            symbol="FTNT",
            timeframe="2h",
            signal_at="2026-07-31T15:30:00-04:00",
            trend=-1,
            bar_count=60,
        )

        updated = override_held_exit_trends(
            {"FTNT": original},
            {"FTNT": signal},
        )

        self.assertEqual(updated["FTNT"]["Trend"].tolist(), [1, 1, -1])
        self.assertEqual(original["Trend"].tolist(), [1, 1, 1])


class ReportTests(unittest.TestCase):
    def test_daily_and_weekly_returns_are_written_for_seven_accounts(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = PerformanceReporter(tmp, initial_cash=10_000)
            rows = []
            for day, values in (
                ("2026-07-27", (10_100, 10_000, 9_900, 9_900, 10_050, 10_025, 10_010)),
                ("2026-07-28", (10_201, 10_200, 10_000, 10_000, 10_100, 10_075, 10_030)),
            ):
                for account_id, equity in zip(("A", "B", "C", "D", "E", "F", "G"), values):
                    rows.append(
                        {
                            "execution_date": day,
                            "signal_date": day,
                            "account_id": account_id,
                            "account_name": account_id,
                            "hypothesis": account_id,
                            "status": "processed",
                            "equity": equity,
                            "cash": equity,
                            "positions_value": 0,
                            "daily_return": 0,
                            "cumulative_return": 0,
                            "drawdown": 0,
                            "position_symbol": "",
                            "position_quantity": 0,
                            "position_avg_price": 0,
                            "mark_price": 0,
                            "order_count": 0,
                            "fill_count": 0,
                            "orders": "[]",
                            "fills": "[]",
                            "notes": "[]",
                        }
                    )

            daily = reporter.record_daily(rows)

            self.assertEqual(len(daily), 14)
            account_a = daily.loc[daily["account_id"] == "A"].sort_values(
                "execution_date"
            )
            self.assertAlmostEqual(
                float(account_a.iloc[-1]["daily_return"]),
                0.01,
            )
            weekly = pd.read_csv(Path(tmp) / "weekly_history.csv")
            self.assertEqual(
                set(weekly["account_id"]),
                {"A", "B", "C", "D", "E", "F", "G"},
            )
            self.assertTrue((weekly["start_equity"] == 10_000).all())
            self.assertTrue((Path(tmp) / "dashboard.html").exists())

    def test_config_freezes_the_seven_hypotheses(self):
        config = json.loads((LAB_ROOT / "config.json").read_text(encoding="utf-8"))

        self.assertEqual(
            set(config["accounts"]),
            {"A", "B", "C", "D", "E", "F", "G"},
        )
        self.assertIsNone(config["accounts"]["A"]["stop_loss_pct"])
        self.assertEqual(config["accounts"]["A"]["market_filter"], "1d")
        self.assertEqual(config["accounts"]["B"]["rotation_profit_gate"], "off")
        self.assertEqual(config["accounts"]["B"]["stop_loss_pct"], 0.12)
        self.assertEqual(config["accounts"]["C"]["market_filter"], "1d")
        self.assertEqual(config["accounts"]["C"]["entry_atr_risk_pct"], 0.025)
        self.assertEqual(config["accounts"]["D"]["market_filter"], "none")
        self.assertEqual(
            config["accounts"]["D"]["rotation_profit_gate"],
            config["accounts"]["A"]["rotation_profit_gate"],
        )
        self.assertEqual(
            config["accounts"]["D"]["stop_loss_pct"],
            config["accounts"]["A"]["stop_loss_pct"],
        )
        self.assertEqual(
            config["accounts"]["D"]["late_chase_mode"],
            config["accounts"]["A"]["late_chase_mode"],
        )
        self.assertIsNone(config["accounts"]["D"]["max_extension_atr"])
        for key in (
            "market_filter",
            "rotation_profit_gate",
            "stop_loss_pct",
            "late_chase_mode",
            "max_extension_atr",
        ):
            self.assertEqual(
                config["accounts"]["G"][key],
                config["accounts"]["D"][key],
            )
        self.assertEqual(config["accounts"]["G"]["exit_timeframe"], "2h")
        self.assertEqual(config["accounts"]["G"]["exit_supertrend_period"], 10)
        self.assertEqual(
            config["accounts"]["G"]["exit_supertrend_multiplier"],
            3.0,
        )
        self.assertEqual(config["accounts"]["G"]["exit_trigger_timeframe"], "1m")
        self.assertEqual(config["accounts"]["G"]["exit_confirm_minutes"], 10)
        self.assertEqual(
            config["accounts"]["G"]["reentry_release"],
            "completed_2h_bullish_after_exit",
        )
        self.assertTrue(config["accounts"]["G"]["entry_2h_safety_gate"])
        self.assertEqual(config["accounts"]["E"]["market_filter"], "none")
        self.assertEqual(config["accounts"]["E"]["entry_atr_risk_pct"], 0.025)
        self.assertEqual(config["accounts"]["F"]["market_filter"], "none")
        self.assertEqual(config["accounts"]["F"]["entry_atr_risk_pct"], 0.02)


class FakeLabClient:
    def __init__(self):
        self.index = pd.bdate_range(end="2026-07-28", periods=340)

    def fetch_candles(
        self,
        symbol,
        *,
        interval,
        minimum_bars,
        adjusted,
        before=None,
        max_pages=30,
    ):
        self.assert_daily(interval)
        rank = sum(ord(character) for character in symbol) % 50
        slope = 0.10 if symbol == "QQQ" else 0.16 + rank / 1000
        close = pd.Series(
            [100.0 + slope * index for index in range(len(self.index))],
            index=self.index,
        )
        return pd.DataFrame(
            {
                "Open": close - 0.05,
                "High": close + 0.25,
                "Low": close - 0.25,
                "Close": close,
                "Volume": 1_000_000,
            },
            index=self.index,
        ).tail(max(minimum_bars, 320))

    def first_regular_minute(
        self,
        symbol,
        session_date,
        *,
        timezone="America/New_York",
        regular_open="09:30",
        regular_close="16:00",
        max_pages=30,
    ):
        rank = sum(ord(character) for character in symbol) % 50
        return pd.Series(
            {
                "Open": 154.0 + rank / 10,
                "High": 154.2 + rank / 10,
                "Low": 153.8 + rank / 10,
                "Close": 154.1 + rank / 10,
                "Volume": 1000,
            }
        )

    @staticmethod
    def assert_daily(interval):
        if interval != "1d":
            raise AssertionError(f"Unexpected interval: {interval}")


class EndToEndLabTests(unittest.TestCase):
    def test_seven_accounts_process_one_shared_signal_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = json.loads((LAB_ROOT / "config.json").read_text(encoding="utf-8"))
            raw["storage"] = {
                "state_dir": str(tmp_path / "state"),
                "data_dir": str(tmp_path / "data"),
                "results_dir": str(tmp_path / "results"),
            }
            raw["accounts"]["G"]["exit_timeframe"] = "1d"
            raw["accounts"]["G"]["entry_2h_safety_gate"] = False
            config_path = tmp_path / "config.json"
            config_path.write_text(
                json.dumps(raw, ensure_ascii=False),
                encoding="utf-8",
            )
            lab = ThreeAccountPaperLab(
                config_path,
                client=FakeLabClient(),
            )
            config_a, policy_a = lab._account_strategy(
                "A",
                raw["accounts"]["A"],
            )
            config_d, policy_d = lab._account_strategy(
                "D",
                raw["accounts"]["D"],
            )
            self.assertTrue(config_a.market_trend_filter.enabled)
            self.assertFalse(config_d.market_trend_filter.enabled)
            self.assertEqual(policy_a, policy_d)

            with redirect_stdout(io.StringIO()):
                history = lab.run_daily(date(2026, 7, 28))

            self.assertEqual(
                set(history["account_id"]),
                {"A", "B", "C", "D", "E", "F", "G"},
            )
            self.assertTrue(
                all(
                    (tmp_path / "state" / "accounts" / f"{account}.json").exists()
                    for account in ("A", "B", "C", "D", "E", "F", "G")
                )
            )
            self.assertTrue((tmp_path / "results" / "dashboard.html").exists())
            dashboard = (tmp_path / "results" / "dashboard.html").read_text(
                encoding="utf-8"
            )
            self.assertIn("7계좌 가상실험", dashboard)
            self.assertIn("당일 종가 평가", dashboard)
            states = {
                account: json.loads(
                    (
                        tmp_path / "state" / "accounts" / f"{account}.json"
                    ).read_text(encoding="utf-8")
                )
                for account in ("A", "B", "C", "D", "E", "F", "G")
            }
            self.assertTrue(
                all(
                    state["metadata"]["last_execution_date"] == "2026-07-28"
                    for state in states.values()
                )
            )
            self.assertTrue(states["A"]["positions"])
            self.assertTrue(states["B"]["positions"])
            for account_id in ("A", "B"):
                symbol = next(iter(states[account_id]["positions"]))
                raw_path = (
                    tmp_path / "data" / "daily" / "raw" / f"{symbol}.csv"
                )
                raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
                expected_close = float(
                    raw.loc[pd.Timestamp("2026-07-28"), "Close"]
                )
                row = history.loc[
                    history["account_id"] == account_id
                ].iloc[-1]
                self.assertAlmostEqual(
                    float(row["mark_price"]),
                    expected_close,
                )


if __name__ == "__main__":
    unittest.main()
