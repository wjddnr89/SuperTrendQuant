from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from supertrend_quant.brokers import PaperBroker
from supertrend_quant.config import ComponentConfig, load_split_config, parse_config
from supertrend_quant.data import MarketData
from supertrend_quant.portfolio import (
    AccountSnapshot,
    OrderIntent,
    OrderPlan,
    Position,
    PositionEconomics,
    mark_position_economics,
)
from supertrend_quant.strategies import create_strategy
from supertrend_quant.runners import run_backtest_on_data


def _score_frame(score: float, *, trend: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Close": [100.0, 100.0],
            "Trend": [1, trend],
            "ATR_pct": [0.01, 0.01],
            "BuySignal": [False, True],
            "Score": [float("nan"), score],
        }
    )


def _leader_strategy(
    *,
    minimum_rotation_profit_pct: float | None = 0.01,
    stop_loss_pct: float | None = None,
):
    config = load_split_config(
        "configs/strategies/leader_rotation.yaml",
        "configs/runtimes/research_sp500.yaml",
    )
    components = config.components
    if stop_loss_pct is not None:
        components = (
            *components,
            ComponentConfig(
                type="fixed_stop_loss",
                enabled=True,
                group="exits",
                params={"loss_pct": stop_loss_pct},
            ),
        )
    config = replace(
        config,
        market_trend_filter=replace(config.market_trend_filter, enabled=False),
        leader_rotation=replace(
            config.leader_rotation,
            hurdle_atr_mult=0.0,
            min_rotation_profit_pct=minimum_rotation_profit_pct,
            allow_late_chase=True,
        ),
        execution=replace(config.execution, allocation_pct=1.0),
        components=components,
    )
    return create_strategy(config)


def test_rotation_uses_only_marked_net_economic_return_and_emits_dependent_buy():
    strategy = _leader_strategy()
    prepared = {"HELD": _score_frame(0.1), "NEW": _score_frame(0.5)}
    position = Position("HELD", 10.0, 100.0)

    missing = strategy._build_order_plan_from_prepared(
        prepared,
        AccountSnapshot(cash=0.0, positions={"HELD": position}),
        "backtest",
    )
    below = strategy._build_order_plan_from_prepared(
        prepared,
        AccountSnapshot(
            cash=0.0,
            positions={"HELD": position},
            position_economics={
                "HELD": PositionEconomics(1000.0, net_return_pct=0.009)
            },
        ),
        "backtest",
    )
    met = strategy._build_order_plan_from_prepared(
        prepared,
        AccountSnapshot(
            cash=0.0,
            positions={"HELD": position},
            position_economics={
                "HELD": PositionEconomics(1000.0, net_return_pct=0.01)
            },
        ),
        "backtest",
    )

    assert missing.orders == ()
    assert below.orders == ()
    assert [order.side for order in met.orders] == ["sell", "buy"]
    dependent_buy = met.orders[1]
    assert dependent_buy.quantity is None
    assert dependent_buy.cash_allocation_pct == 1.0
    assert dependent_buy.required_sell_symbols == ("HELD",)


def test_protective_exit_does_not_require_economic_ledger():
    strategy = _leader_strategy()
    plan = strategy._build_order_plan_from_prepared(
        {"HELD": _score_frame(0.1, trend=-1), "NEW": _score_frame(0.5)},
        AccountSnapshot(
            cash=0.0,
            positions={"HELD": Position("HELD", 10.0, 100.0)},
        ),
        "backtest",
    )

    assert plan.orders[0].side == "sell"
    assert plan.orders[0].reason == "Supertrend down"


def test_rotation_without_minimum_profit_gate_does_not_require_economic_ledger():
    strategy = _leader_strategy(minimum_rotation_profit_pct=None)
    plan = strategy._build_order_plan_from_prepared(
        {"HELD": _score_frame(0.1), "NEW": _score_frame(0.5)},
        AccountSnapshot(
            cash=0.0,
            positions={"HELD": Position("HELD", 10.0, 100.0)},
        ),
        "backtest",
    )

    assert [order.side for order in plan.orders] == ["sell", "buy"]
    assert plan.orders[0].reason == "Leader rotation"


def test_fixed_stop_loss_uses_net_economic_return_and_has_exit_priority():
    strategy = _leader_strategy(stop_loss_pct=0.05)
    prepared = {"HELD": _score_frame(0.5, trend=-1)}
    position = Position("HELD", 10.0, 100.0)

    above_stop = strategy._build_order_plan_from_prepared(
        prepared,
        AccountSnapshot(
            cash=0.0,
            positions={"HELD": position},
            position_economics={
                "HELD": PositionEconomics(1000.0, net_return_pct=-0.049)
            },
        ),
        "backtest",
    )
    at_stop = strategy._build_order_plan_from_prepared(
        prepared,
        AccountSnapshot(
            cash=0.0,
            positions={"HELD": position},
            position_economics={
                "HELD": PositionEconomics(1000.0, net_return_pct=-0.05)
            },
        ),
        "backtest",
    )

    assert above_stop.orders[0].reason == "Supertrend down"
    assert at_stop.orders[0].reason == "Fixed stop loss"


def test_raw_mark_economics_include_round_trip_costs_and_distributions():
    entry_cost = 100.0 * 1.0005 * 1.001
    position = Position("AAA", 1.0, 100.05)
    base = AccountSnapshot(
        cash=0.0,
        positions={"AAA": position},
        position_economics={"AAA": PositionEconomics(entry_cost)},
    )
    without_dividend = mark_position_economics(
        base,
        {"AAA": 101.2},
        fee_rate=0.001,
        slippage_rate=0.0005,
    )
    with_dividend = mark_position_economics(
        replace(
            base,
            position_economics={"AAA": PositionEconomics(entry_cost, distributions=0.5)},
        ),
        {"AAA": 101.2},
        fee_rate=0.001,
        slippage_rate=0.0005,
    )

    assert without_dividend.position_economics["AAA"].net_return_pct < 0.01
    assert with_dividend.position_economics["AAA"].net_return_pct > 0.01


def test_split_adjusted_position_preserves_raw_economic_return():
    before = mark_position_economics(
        AccountSnapshot(
            cash=0.0,
            positions={"AAA": Position("AAA", 10.0, 100.0)},
            position_economics={"AAA": PositionEconomics(1000.0)},
        ),
        {"AAA": 110.0},
        fee_rate=0.0,
        slippage_rate=0.0,
    )
    after = mark_position_economics(
        AccountSnapshot(
            cash=0.0,
            positions={"AAA": Position("AAA", 20.0, 50.0)},
            position_economics={"AAA": PositionEconomics(1000.0)},
        ),
        {"AAA": 55.0},
        fee_rate=0.0,
        slippage_rate=0.0,
    )

    assert before.position_economics["AAA"].net_return_pct == after.position_economics["AAA"].net_return_pct


def test_paper_dependent_buy_sizes_from_actual_post_sell_cash_and_gap_prices():
    with tempfile.TemporaryDirectory() as directory:
        broker = PaperBroker(Path(directory) / "paper.json", initial_cash=1000.0)
        broker.execute_plan(
            OrderPlan("leader", "paper", (OrderIntent("HELD", "buy", 10.0),)),
            {"HELD": 100.0},
            0.0,
            0.0,
        )
        plan = OrderPlan(
            "leader",
            "paper",
            (
                OrderIntent("HELD", "sell", 10.0, reason="Leader rotation"),
                OrderIntent(
                    "NEW",
                    "buy",
                    None,
                    reason="Post-sell leader entry",
                    cash_allocation_pct=1.0,
                    required_sell_symbols=("HELD",),
                ),
            ),
        )

        fills = broker.execute_plan(plan, {"HELD": 80.0, "NEW": 40.0}, 0.0, 0.0)
        account = broker.get_account()

    assert fills == ["SELL HELD 10 @ 80.0000", "BUY NEW 20 @ 40.0000"]
    assert account.positions["NEW"].quantity == 20.0
    assert account.cash == 0.0


def test_backtest_dependent_buy_uses_next_open_raw_cash_not_adjusted_signal_price():
    class RotateStrategy:
        def warmup_bars(self):
            return 0

        def build_order_plan(self, bars, account, mode, **kwargs):
            if not account.positions:
                return OrderPlan("rotate", mode, (OrderIntent("HELD", "buy", 10.0),))
            if "HELD" in account.positions:
                return OrderPlan(
                    "rotate",
                    mode,
                    (
                        OrderIntent("HELD", "sell", 10.0, reason="Leader rotation"),
                        OrderIntent(
                            "NEW",
                            "buy",
                            None,
                            reason="Post-sell leader entry",
                            cash_allocation_pct=1.0,
                            required_sell_symbols=("HELD",),
                        ),
                    ),
                )
            return OrderPlan("rotate", mode, ())

    config = parse_config(
        {
            "strategy": {"name": "rotate", "type": "leader_rotation", "params": {}},
            "scoring": {"type": "relative_strength", "params": {"lookback_bars": 1}},
            "market": "US",
            "universe": {"source": "symbols", "symbols": ["HELD", "NEW"]},
            "capital": {"initial_cash": 1000.0},
            "costs": {"fee_rate": 0.0, "slippage_rate": 0.0},
        }
    )
    index = pd.date_range("2026-01-01", periods=4, freq="D")

    def frame(values):
        return pd.DataFrame(
            {"Open": values, "High": values, "Low": values, "Close": values},
            index=index,
        )

    market_data = MarketData(
        bars={"HELD": frame([10.0] * 4), "NEW": frame([1000.0] * 4)},
        execution_bars={
            "HELD": frame([100.0, 100.0, 80.0, 80.0]),
            "NEW": frame([40.0, 40.0, 40.0, 40.0]),
        },
    )

    with patch("supertrend_quant.runners.create_strategy", return_value=RotateStrategy()):
        result = run_backtest_on_data(config, market_data)

    assert result.trade_records[0]["symbol"] == "HELD"
    assert result.trade_records[0]["exit_price"] == 80.0
    assert result.trade_records[1]["symbol"] == "NEW"
    assert result.trade_records[1]["quantity"] == 20


def test_backtest_defers_sell_and_dependent_buy_across_validated_no_trade_day():
    class RotateStrategy:
        def warmup_bars(self):
            return 0

        def build_order_plan(self, bars, account, mode, **kwargs):
            if not account.positions:
                return OrderPlan("rotate", mode, (OrderIntent("HELD", "buy", 10.0),))
            if "HELD" in account.positions:
                return OrderPlan(
                    "rotate",
                    mode,
                    (
                        OrderIntent("HELD", "sell", 10.0, reason="Fixed stop loss"),
                        OrderIntent(
                            "NEW",
                            "buy",
                            None,
                            reason="Post-sell leader entry",
                            cash_allocation_pct=1.0,
                            required_sell_symbols=("HELD",),
                        ),
                    ),
                )
            return OrderPlan("rotate", mode, ())

    config = parse_config(
        {
            "strategy": {"name": "rotate", "type": "leader_rotation", "params": {}},
            "scoring": {"type": "relative_strength", "params": {"lookback_bars": 1}},
            "market": "KR",
            "universe": {"source": "symbols", "symbols": ["HELD", "NEW"]},
            "capital": {"initial_cash": 1000.0},
            "costs": {"fee_rate": 0.0, "slippage_rate": 0.0},
        }
    )
    index = pd.date_range("2026-01-01", periods=5, freq="D")

    def frame(values, frame_index=index):
        return pd.DataFrame(
            {"Open": values, "High": values, "Low": values, "Close": values},
            index=frame_index,
        )

    held_execution_index = index.delete(2)
    market_data = MarketData(
        bars={"HELD": frame([10.0] * 5), "NEW": frame([40.0] * 5)},
        execution_bars={
            "HELD": frame([100.0, 100.0, 80.0, 80.0], held_execution_index),
            "NEW": frame([40.0] * 5),
        },
        allow_no_trade_valuation_carry_forward=True,
    )

    with patch("supertrend_quant.runners.create_strategy", return_value=RotateStrategy()):
        result = run_backtest_on_data(config, market_data)

    held_trade, new_trade = result.trade_records
    assert held_trade["symbol"] == "HELD"
    assert held_trade["exit_time"] == index[3]
    assert held_trade["exit_price"] == 80.0
    assert new_trade["symbol"] == "NEW"
    assert new_trade["entry_time"] == index[3]
    assert new_trade["quantity"] == 20


def test_paper_dependent_buy_is_skipped_when_prerequisite_sell_fails():
    with tempfile.TemporaryDirectory() as directory:
        broker = PaperBroker(Path(directory) / "paper.json", initial_cash=1000.0)
        plan = OrderPlan(
            "leader",
            "paper",
            (
                OrderIntent("HELD", "sell", 10.0),
                OrderIntent(
                    "NEW",
                    "buy",
                    None,
                    cash_allocation_pct=1.0,
                    required_sell_symbols=("HELD",),
                ),
            ),
        )
        fills = broker.execute_plan(plan, {"HELD": 80.0, "NEW": 40.0}, 0.0, 0.0)

    assert fills == [
        "SKIP SELL HELD: no position",
        "SKIP BUY NEW: prerequisite sell not filled",
    ]


def test_paper_ledger_persists_entry_cost_dividend_and_split_state():
    with tempfile.TemporaryDirectory() as directory:
        broker = PaperBroker(Path(directory) / "paper.json", initial_cash=2000.0)
        broker.execute_plan(
            OrderPlan("leader", "paper", (OrderIntent("AAA", "buy", 10.0),)),
            {"AAA": 100.0},
            0.001,
            0.0005,
        )
        broker.apply_corporate_actions(
            (
                {
                    "event_id": "dividend",
                    "action_type": "cash_dividend",
                    "symbol": "AAA",
                    "ex_date": "2026-01-02",
                    "cash_amount": 2.0,
                },
                {
                    "event_id": "split",
                    "action_type": "split",
                    "symbol": "AAA",
                    "effective_date": "2026-01-03",
                    "ratio": 2.0,
                },
            ),
            through="2026-01-03",
        )
        account = broker.get_account()

    assert account.positions["AAA"].quantity == 20.0
    assert account.positions["AAA"].avg_price == 50.025
    economics = account.position_economics["AAA"]
    assert economics.entry_cost == 10.0 * 100.05 * 1.001
    assert economics.distributions == 20.0
