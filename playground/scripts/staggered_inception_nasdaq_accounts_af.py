from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PLAYGROUND_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PLAYGROUND_ROOT.parent
UNIFIED_ROOT = PROJECT_ROOT / "unified_quant"
sys.path.insert(0, str(UNIFIED_ROOT / "src"))
sys.path.insert(0, str(PLAYGROUND_ROOT))

from portfolio_risk_experiment.risk_overlay import (  # noqa: E402
    PortfolioRiskPolicy,
    PortfolioRiskPreparedBacktest,
)
from research_extensions.experimental_leader_rotation import (  # noqa: E402
    ExperimentalLeaderPolicy,
    ExperimentalSignalCache,
    FastExperimentalPreparedLeaderBacktest,
)
from research_extensions.kospi_market_filters import (  # noqa: E402
    build_filter_variant,
)
from scripts.compare_nasdaq_three_accounts_cold_start import (  # noqa: E402
    append_checkpoint,
    first_trade_fields,
    load_checkpoint,
)
from scripts.nested_walk_forward_nasdaq_structure import (  # noqa: E402
    Candidate,
    base_config,
    benchmark_return_for_index,
    config_for_candidate,
)
from supertrend_quant.data import market_index  # noqa: E402
from supertrend_quant.research.data_resolver import download_for_config  # noqa: E402
import supertrend_quant.runners as canonical_runners  # noqa: E402
from supertrend_quant.runners import (  # noqa: E402
    BacktestResult,
    _prepare_backtest,
    run_backtest_on_data,
)
from supertrend_quant.metrics import calculate_metrics  # noqa: E402
from supertrend_quant.portfolio import (  # noqa: E402
    AccountSnapshot,
    Position,
    PositionEconomics,
    estimate_quantity,
    mark_position_economics,
)
from supertrend_quant.strategies import create_strategy  # noqa: E402
from supertrend_quant.strategies.leader_rotation import (  # noqa: E402
    PreparedLeaderBacktest,
)


DEFAULT_CONFIG = (
    PLAYGROUND_ROOT
    / "configs"
    / "nasdaq_accounts_af_staggered_inception.json"
)
DEFAULT_RESULTS = PLAYGROUND_ROOT / "results" / "nasdaq_staggered_inception"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Measure 12-month start-date sensitivity for current Nasdaq "
            "paper accounts A-F at fixed trading-session launch intervals."
        )
    )
    value.add_argument("--config", default=str(DEFAULT_CONFIG))
    value.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    value.add_argument("--run-id", default="")
    value.add_argument("--no-resume", action="store_true")
    value.add_argument("--max-launches", type=int, default=0)
    value.add_argument("--launch-offset", type=int, default=0)
    value.add_argument("--progress-every", type=int, default=10)
    value.add_argument(
        "--engine",
        choices=("fast", "canonical"),
        default="fast",
        help="Fast uses the same cached planner with a lightweight fill loop.",
    )
    return value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_from_account(raw: dict[str, Any]) -> Candidate:
    return Candidate(
        candidate_id=str(raw["account_id"]),
        policy=ExperimentalLeaderPolicy(
            rotation_profit_gate=str(raw["rotation_profit_gate"]),
            stop_loss_pct=(
                None
                if raw.get("stop_loss_pct") is None
                else float(raw["stop_loss_pct"])
            ),
            late_chase_mode=str(raw["late_chase_mode"]),
            max_extension_atr=(
                None
                if raw.get("max_extension_atr") is None
                else float(raw["max_extension_atr"])
            ),
        ),
    )


def launch_windows(
    full_index: pd.Index,
    raw: dict[str, Any],
) -> list[tuple[str, pd.DatetimeIndex]]:
    spec = raw["multi_start"]
    dates = pd.DatetimeIndex(full_index).sort_values().unique()
    first = int(
        dates.searchsorted(pd.Timestamp(str(spec["first_launch"])), side="left")
    )
    last = int(
        dates.searchsorted(pd.Timestamp(str(spec["last_launch"])), side="right")
    ) - 1
    step = int(spec["step_sessions"])
    horizon = int(spec["horizon_sessions"])
    end_limit = pd.Timestamp(str(raw["end"]))
    windows: list[tuple[str, pd.DatetimeIndex]] = []
    for start in range(first, last + 1, step):
        stop = start + horizon
        if stop > len(dates):
            break
        run_index = pd.DatetimeIndex(dates[start:stop])
        if len(run_index) != horizon or run_index[-1] > end_limit:
            continue
        windows.append((str(run_index[0].date()), run_index))
    return windows


def build_backtest(
    *,
    canonical_prepared: PreparedLeaderBacktest,
    market_filter_trends,
    config,
    candidate: Candidate,
    signal_cache: ExperimentalSignalCache,
    atr_risk: float | None,
):
    prepared = PreparedLeaderBacktest(
        create_strategy(config),
        canonical_prepared.prepared,
        market_filter_trends,
        canonical_prepared.universe_schedule,
    )
    backtest = FastExperimentalPreparedLeaderBacktest(
        prepared,
        candidate.policy,
        signal_cache,
    )
    if atr_risk is not None:
        backtest = PortfolioRiskPreparedBacktest(
            backtest,
            PortfolioRiskPolicy(
                name=f"{candidate.candidate_id}_ATR_{atr_risk:.3f}",
                max_positions=1,
                target_portfolio_atr_pct=atr_risk,
            ),
        )
    return backtest


def aggregate_accounts(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (account_id, account_name), group in frame.groupby(
        ["account_id", "account_name"], sort=True
    ):
        returns = pd.to_numeric(group["total_return"], errors="coerce")
        alpha = pd.to_numeric(group["alpha"], errors="coerce")
        mdd = pd.to_numeric(group["mdd"], errors="coerce")
        sharpe = pd.to_numeric(group["sharpe"], errors="coerce")
        first_returns = pd.to_numeric(
            group["first_trade_return"], errors="coerce"
        ).dropna()
        first_symbols = group["first_trade_symbol"].fillna("").astype(str)
        first_symbols = first_symbols[first_symbols.ne("")]
        top_symbol = ""
        top_symbol_ratio = 0.0
        if not first_symbols.empty:
            counts = first_symbols.value_counts()
            top_symbol = str(counts.index[0])
            top_symbol_ratio = float(counts.iloc[0] / len(first_symbols))
        rows.append(
            {
                "account_id": account_id,
                "account_name": account_name,
                "launch_count": int(len(group)),
                "mean_return": float(returns.mean()),
                "median_return": float(returns.median()),
                "return_std": float(returns.std(ddof=1)),
                "return_p10": float(returns.quantile(0.10)),
                "return_p25": float(returns.quantile(0.25)),
                "return_p75": float(returns.quantile(0.75)),
                "return_p90": float(returns.quantile(0.90)),
                "worst_return": float(returns.min()),
                "best_return": float(returns.max()),
                "positive_return_ratio": float(returns.gt(0.0).mean()),
                "benchmark_excess_ratio": float(alpha.gt(0.0).mean()),
                "median_alpha": float(alpha.median()),
                "alpha_p10": float(alpha.quantile(0.10)),
                "median_mdd": float(mdd.median()),
                "worst_mdd": float(mdd.min()),
                "median_sharpe": float(sharpe.median()),
                "sharpe_p10": float(sharpe.quantile(0.10)),
                "median_calmar": float(group["calmar"].median()),
                "median_trade_count": float(group["trade_count"].median()),
                "first_trade_count": int(len(first_returns)),
                "median_first_trade_return": (
                    float(first_returns.median()) if len(first_returns) else 0.0
                ),
                "first_trade_loss_ratio": (
                    float(first_returns.lt(0.0).mean())
                    if len(first_returns)
                    else 0.0
                ),
                "first_trade_severe_loss_ratio": (
                    float(first_returns.le(-0.10).mean())
                    if len(first_returns)
                    else 0.0
                ),
                "unique_first_symbols": int(first_symbols.nunique()),
                "most_common_first_symbol": top_symbol,
                "most_common_first_symbol_ratio": top_symbol_ratio,
            }
        )
    return pd.DataFrame(rows).sort_values("account_id")


def aggregate_first_symbols(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[frame["first_trade_symbol"].fillna("").ne("")].copy()
    if selected.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (account_id, account_name, symbol), group in selected.groupby(
        ["account_id", "account_name", "first_trade_symbol"], sort=True
    ):
        returns = pd.to_numeric(group["total_return"], errors="coerce")
        rows.append(
            {
                "account_id": account_id,
                "account_name": account_name,
                "first_trade_symbol": symbol,
                "launch_count": int(len(group)),
                "launch_ratio_within_account": float(
                    len(group)
                    / len(frame.loc[frame["account_id"].eq(account_id)])
                ),
                "median_12m_return": float(returns.median()),
                "positive_return_ratio": float(returns.gt(0.0).mean()),
                "median_mdd": float(group["mdd"].median()),
                "median_first_trade_return": float(
                    pd.to_numeric(
                        group["first_trade_return"], errors="coerce"
                    ).median()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["account_id", "launch_count", "first_trade_symbol"],
        ascending=[True, False, True],
    )


def finite(value: Any) -> float:
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


def _price_at_or_before(frame: pd.DataFrame | None, timestamp, column: str):
    if frame is None or frame.empty or column not in frame:
        return None
    available = frame.loc[:timestamp, column]
    if available.empty:
        return None
    value = finite(available.iloc[-1])
    return value if value > 0.0 else None


def _open_at(frame: pd.DataFrame | None, timestamp):
    if frame is None or frame.empty or "Open" not in frame or timestamp not in frame.index:
        return None
    value = finite(frame.loc[timestamp, "Open"])
    return value if value > 0.0 else None


def run_fast_backtest(config, data, run_index: pd.Index, backtest) -> BacktestResult:
    """Run the cached paper-account planner with lightweight canonical fills.

    This intentionally mirrors the normal next-session-open fill, fees,
    slippage, cash-allocation dependencies, marked net-return gate, and final
    liquidation. Point-in-time membership and signals remain those of the
    canonical prepared backtest. The full canonical engine is retained as a
    validation option for representative launch windows.
    """

    full_index = pd.DatetimeIndex(market_index(data))
    idx = pd.DatetimeIndex(run_index).intersection(full_index, sort=False)
    if len(idx) < 2:
        raise RuntimeError("Not enough bars for fast inception backtest.")
    execution_bars = getattr(data, "execution_bars", None) or data.bars
    cash = float(config.capital.initial_cash)
    positions: dict[str, Position] = {}
    entry_values: dict[str, float] = {}
    entry_times: dict[str, object] = {}
    equity_points: list[tuple[pd.Timestamp, float]] = []
    trade_returns: list[float] = []
    trade_records: list[dict[str, object]] = []

    for i in range(len(idx) - 1):
        signal_ts = idx[i]
        exec_ts = idx[i + 1]
        raw_marks = {
            symbol: price
            for symbol in positions
            if (
                price := _price_at_or_before(
                    execution_bars.get(symbol), signal_ts, "Close"
                )
            )
            is not None
        }
        marked = mark_position_economics(
            AccountSnapshot(
                cash=cash,
                positions=dict(positions),
                position_economics={
                    symbol: PositionEconomics(entry_cost=entry_values[symbol])
                    for symbol in positions
                    if symbol in entry_values
                },
            ),
            raw_marks,
            fee_rate=config.costs.fee_rate,
            slippage_rate=config.costs.slippage_rate,
        )
        equity = cash + sum(
            float(position.quantity) * float(raw_marks.get(symbol, position.avg_price))
            for symbol, position in positions.items()
        )
        equity_points.append((signal_ts, equity))
        plan = backtest.build_order_plan(signal_ts, marked, mode="backtest")

        filled_sell_symbols: set[str] = set()
        cash_allocation_base: float | None = None
        for order in plan.orders:
            raw_price = _open_at(execution_bars.get(order.symbol), exec_ts)
            if raw_price is None:
                continue
            if order.side.lower() == "sell":
                held = positions.get(order.symbol)
                if held is None:
                    continue
                quantity = min(float(held.quantity), float(order.quantity or 0.0))
                if quantity <= 0.0:
                    continue
                fill = raw_price * (1.0 - config.costs.slippage_rate)
                proceeds = quantity * fill * (1.0 - config.costs.fee_rate)
                cash += proceeds
                entry_value = entry_values.pop(
                    order.symbol, quantity * held.avg_price
                )
                pnl_pct = proceeds / entry_value - 1.0 if entry_value else 0.0
                trade_returns.append(pnl_pct)
                trade_records.append(
                    {
                        "symbol": order.symbol,
                        "entry_time": entry_times.pop(order.symbol, None),
                        "exit_time": exec_ts,
                        "entry_price": held.avg_price,
                        "exit_price": fill,
                        "quantity": quantity,
                        "entry_value": entry_value,
                        "exit_value": proceeds,
                        "pnl_value": proceeds - entry_value,
                        "pnl_pct": pnl_pct,
                        "exit_reason": order.reason,
                    }
                )
                positions.pop(order.symbol, None)
                filled_sell_symbols.add(order.symbol)
                continue

            if order.required_sell_symbols and not set(
                order.required_sell_symbols
            ).issubset(filled_sell_symbols):
                continue
            affordable = estimate_quantity(
                cash,
                raw_price,
                1.0,
                fee_rate=config.costs.fee_rate,
                slippage_rate=config.costs.slippage_rate,
            )
            if order.cash_allocation_pct is not None:
                if cash_allocation_base is None:
                    cash_allocation_base = cash
                target = estimate_quantity(
                    cash_allocation_base,
                    raw_price,
                    float(order.cash_allocation_pct),
                    fee_rate=config.costs.fee_rate,
                    slippage_rate=config.costs.slippage_rate,
                )
                quantity = min(target, affordable)
            elif order.quantity is not None:
                quantity = min(int(order.quantity), affordable)
            else:
                continue
            if quantity <= 0:
                continue
            fill = raw_price * (1.0 + config.costs.slippage_rate)
            cost = quantity * fill * (1.0 + config.costs.fee_rate)
            if cost > cash:
                continue
            cash -= cost
            positions[order.symbol] = Position(order.symbol, quantity, fill)
            entry_values[order.symbol] = cost
            entry_times[order.symbol] = exec_ts

    final_ts = idx[-1]
    final_raw_marks = {
        symbol: price
        for symbol in positions
        if (
            price := _price_at_or_before(
                execution_bars.get(symbol), final_ts, "Close"
            )
        )
        is not None
    }
    final_equity = cash + sum(
        float(position.quantity)
        * float(final_raw_marks.get(symbol, position.avg_price))
        for symbol, position in positions.items()
    )
    equity_points.append((final_ts, final_equity))
    for symbol, held in list(positions.items()):
        raw_close = final_raw_marks.get(symbol)
        if raw_close is None:
            continue
        fill = raw_close * (1.0 - config.costs.slippage_rate)
        proceeds = float(held.quantity) * fill * (1.0 - config.costs.fee_rate)
        entry_value = entry_values.pop(symbol, held.quantity * held.avg_price)
        pnl_pct = proceeds / entry_value - 1.0 if entry_value else 0.0
        trade_returns.append(pnl_pct)
        trade_records.append(
            {
                "symbol": symbol,
                "entry_time": entry_times.pop(symbol, None),
                "exit_time": final_ts,
                "entry_price": held.avg_price,
                "exit_price": fill,
                "quantity": held.quantity,
                "entry_value": entry_value,
                "exit_value": proceeds,
                "pnl_value": proceeds - entry_value,
                "pnl_pct": pnl_pct,
                "exit_reason": "FinalClose",
            }
        )
    equity_series = pd.Series(
        [value for _, value in equity_points],
        index=[timestamp for timestamp, _ in equity_points],
        name="equity",
        dtype=float,
    )
    return BacktestResult(
        equity=equity_series,
        metrics=calculate_metrics(equity_series, trade_returns, config.timeframe),
        trades=trade_returns,
        skipped=data.skipped,
        trade_records=tuple(trade_records),
        universe_snapshot=getattr(data, "universe_snapshot", None),
        data_quality="fast_validated",
    )


def run_start_window_fast(
    config,
    data,
    run_index: pd.Index,
    signal_cache: ExperimentalSignalCache,
    policy: ExperimentalLeaderPolicy,
    atr_risk: float | None,
) -> BacktestResult:
    """Specialized max-one-position cohort path using cached signal arrays."""

    idx = pd.DatetimeIndex(run_index)
    if len(idx) < 2:
        raise RuntimeError("Not enough bars for inception window.")
    # Total-return-adjusted bars keep split/dividend paths continuous in this
    # lightweight research engine. Raw opens are used only to reproduce the
    # canonical ATR order's adjusted-signal/raw-execution sizing cap.
    price_bars = data.bars
    raw_execution_bars = getattr(data, "execution_bars", None) or data.bars
    rankings = signal_cache._rankings_for(policy)
    cash = float(config.capital.initial_cash)
    held: Position | None = None
    entry_cost = 0.0
    entry_time = None
    blocked: set[str] = set()
    equity_points: list[tuple[pd.Timestamp, float]] = []
    trade_returns: list[float] = []
    trade_records: list[dict[str, object]] = []

    def candidate_at(position: int, held_symbol: str | None):
        active = signal_cache.active_by_position[position]
        for symbol in rankings[position]:
            if active is not None and symbol not in active and symbol != held_symbol:
                continue
            if symbol in blocked and symbol != held_symbol:
                continue
            values = signal_cache.symbol_arrays[symbol]
            return {
                "symbol": symbol,
                "score": float(values["score"][position]),
                "atr_pct": float(values["atr_pct"][position]),
                "price": float(values["price"][position]),
            }
        return None

    for i in range(len(idx) - 1):
        signal_ts = idx[i]
        exec_ts = idx[i + 1]
        position = signal_cache.position_at(signal_ts)
        released = {
            symbol
            for symbol in blocked
            if signal_cache.fresh_buy_at(symbol, position)
        }
        blocked.difference_update(released)

        held_close = (
            _price_at_or_before(price_bars.get(held.symbol), signal_ts, "Close")
            if held is not None
            else None
        )
        equity = cash + (
            float(held.quantity) * float(held_close)
            if held is not None and held_close is not None
            else 0.0
        )
        equity_points.append((signal_ts, equity))

        target = candidate_at(position, held.symbol if held else None)
        sell_reason: str | None = None
        defer_replacement = False
        if held is not None:
            estimated_exit = (
                float(held.quantity)
                * float(held_close)
                * (1.0 - config.costs.slippage_rate)
                * (1.0 - config.costs.fee_rate)
                if held_close is not None
                else None
            )
            net_return = (
                estimated_exit / entry_cost - 1.0
                if estimated_exit is not None and entry_cost > 0.0
                else None
            )
            if (
                policy.stop_loss_pct is not None
                and net_return is not None
                and net_return <= -float(policy.stop_loss_pct)
            ):
                sell_reason = f"Fixed stop {float(policy.stop_loss_pct):.1%}"
                blocked.add(held.symbol)
                defer_replacement = True
            elif not signal_cache.has_data_at(held.symbol, position):
                sell_reason = "Held symbol missing from strategy data"
            elif signal_cache.exit_down_at(held.symbol, position):
                sell_reason = "Supertrend down"
            elif target is not None and target["symbol"] != held.symbol:
                current_score = signal_cache.value_at(
                    held.symbol, position, "score"
                )
                hurdle = (
                    float(target["atr_pct"])
                    * config.leader_rotation.hurdle_atr_mult
                )
                gate_passes = (
                    policy.rotation_profit_gate == "off"
                    or (net_return is not None and net_return >= 0.0)
                )
                if (
                    current_score is not None
                    and float(target["score"]) - current_score > hurdle
                    and gate_passes
                ):
                    sell_reason = "Leader rotation"

        sold = False
        if held is not None and sell_reason is not None:
            raw_open = _open_at(price_bars.get(held.symbol), exec_ts)
            if raw_open is not None:
                fill = raw_open * (1.0 - config.costs.slippage_rate)
                proceeds = (
                    float(held.quantity) * fill * (1.0 - config.costs.fee_rate)
                )
                cash += proceeds
                pnl_pct = proceeds / entry_cost - 1.0 if entry_cost else 0.0
                trade_returns.append(pnl_pct)
                trade_records.append(
                    {
                        "symbol": held.symbol,
                        "entry_time": entry_time,
                        "exit_time": exec_ts,
                        "entry_price": held.avg_price,
                        "exit_price": fill,
                        "quantity": held.quantity,
                        "entry_value": entry_cost,
                        "exit_value": proceeds,
                        "pnl_value": proceeds - entry_cost,
                        "pnl_pct": pnl_pct,
                        "exit_reason": sell_reason,
                    }
                )
                held = None
                entry_cost = 0.0
                entry_time = None
                sold = True

        may_buy = held is None and not defer_replacement
        if sell_reason is not None and not sold:
            may_buy = False
        if may_buy and target is not None:
            raw_open = _open_at(price_bars.get(str(target["symbol"])), exec_ts)
            if raw_open is not None:
                if atr_risk is None:
                    quantity = estimate_quantity(
                        cash,
                        raw_open,
                        1.0,
                        fee_rate=config.costs.fee_rate,
                        slippage_rate=config.costs.slippage_rate,
                    )
                else:
                    atr_pct = float(target["atr_pct"])
                    base_weight = (
                        min(1.0, float(atr_risk) / atr_pct)
                        if math.isfinite(atr_pct) and atr_pct > 0.0
                        else 0.0
                    )
                    raw_reference_open = _open_at(
                        raw_execution_bars.get(str(target["symbol"])),
                        exec_ts,
                    )
                    if raw_reference_open is None or float(target["price"]) <= 0.0:
                        effective_weight = base_weight
                    else:
                        effective_weight = min(
                            1.0,
                            base_weight
                            * raw_reference_open
                            / float(target["price"]),
                        )
                    desired = estimate_quantity(
                        equity,
                        raw_open,
                        effective_weight,
                        fee_rate=config.costs.fee_rate,
                        slippage_rate=config.costs.slippage_rate,
                    )
                    affordable = estimate_quantity(
                        cash,
                        raw_open,
                        1.0,
                        fee_rate=config.costs.fee_rate,
                        slippage_rate=config.costs.slippage_rate,
                    )
                    quantity = min(desired, affordable)
                if quantity > 0:
                    fill = raw_open * (1.0 + config.costs.slippage_rate)
                    cost = quantity * fill * (1.0 + config.costs.fee_rate)
                    if cost <= cash:
                        cash -= cost
                        held = Position(str(target["symbol"]), quantity, fill)
                        entry_cost = cost
                        entry_time = exec_ts

    final_ts = idx[-1]
    final_close = (
        _price_at_or_before(price_bars.get(held.symbol), final_ts, "Close")
        if held is not None
        else None
    )
    final_equity = cash + (
        float(held.quantity) * float(final_close)
        if held is not None and final_close is not None
        else 0.0
    )
    equity_points.append((final_ts, final_equity))
    if held is not None and final_close is not None:
        fill = final_close * (1.0 - config.costs.slippage_rate)
        proceeds = float(held.quantity) * fill * (1.0 - config.costs.fee_rate)
        pnl_pct = proceeds / entry_cost - 1.0 if entry_cost else 0.0
        trade_returns.append(pnl_pct)
        trade_records.append(
            {
                "symbol": held.symbol,
                "entry_time": entry_time,
                "exit_time": final_ts,
                "entry_price": held.avg_price,
                "exit_price": fill,
                "quantity": held.quantity,
                "entry_value": entry_cost,
                "exit_value": proceeds,
                "pnl_value": proceeds - entry_cost,
                "pnl_pct": pnl_pct,
                "exit_reason": "FinalClose",
            }
        )
    equity_series = pd.Series(
        [value for _, value in equity_points],
        index=[timestamp for timestamp, _ in equity_points],
        name="equity",
        dtype=float,
    )
    return BacktestResult(
        equity=equity_series,
        metrics=calculate_metrics(equity_series, trade_returns, config.timeframe),
        trades=trade_returns,
        skipped=data.skipped,
        trade_records=tuple(trade_records),
        universe_snapshot=getattr(data, "universe_snapshot", None),
        data_quality="cohort_fast_validated",
    )


def main() -> None:
    args = parser().parse_args()
    raw = load_json(Path(args.config).resolve())
    accounts = list(raw["accounts"])
    variants = {
        str(item["variant_id"]): item
        for item in raw["market_filter_variants"]
    }
    run_id = args.run_id or (
        f"{raw['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(args.results_dir).resolve() / run_id
    checkpoint_path = run_dir / "checkpoint.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.no_resume and checkpoint_path.exists():
        checkpoint_path.unlink()

    started = time.monotonic()
    base = base_config(raw)
    print("[inception] loading canonical Nasdaq data...", flush=True)
    data = download_for_config(base, allow_stale=True)
    full_index = pd.DatetimeIndex(market_index(data))
    windows = launch_windows(full_index, raw)
    if args.launch_offset > 0:
        windows = windows[int(args.launch_offset) :]
    if args.max_launches > 0:
        windows = windows[: args.max_launches]
    if not windows:
        raise RuntimeError("No valid inception windows were generated.")
    print("[inception] preparing canonical indicators...", flush=True)
    canonical_prepared = _prepare_backtest(create_strategy(base), data)
    if not isinstance(canonical_prepared, PreparedLeaderBacktest):
        raise TypeError("Expected canonical PreparedLeaderBacktest.")

    contexts: dict[str, dict[str, Any]] = {}
    used_variants = list(
        dict.fromkeys(str(account["market_filter_variant"]) for account in accounts)
    )
    for position, variant_id in enumerate(used_variants, start=1):
        print(
            f"[inception] filter {position}/{len(used_variants)} "
            f"{variant_id}: building causal regime and signal cache",
            flush=True,
        )
        variant = build_filter_variant(
            variants[variant_id],
            base_config=base,
            data=data,
            canonical_prepared=canonical_prepared,
            full_index=full_index,
        )
        shared = PreparedLeaderBacktest(
            create_strategy(variant.config),
            canonical_prepared.prepared,
            variant.market_filter_trends,
            canonical_prepared.universe_schedule,
        )
        contexts[variant_id] = {
            "variant": variant,
            "signal_cache": ExperimentalSignalCache(shared, full_index),
        }

    checkpoint = load_checkpoint(checkpoint_path)
    total = len(windows) * len(accounts)
    completed = 0
    progress_every = max(1, int(args.progress_every))
    for launch_id, run_index in windows:
        benchmark_return = benchmark_return_for_index(data, run_index)
        for account in accounts:
            candidate = candidate_from_account(account)
            key = f"{candidate.candidate_id}__{launch_id}"
            if key not in checkpoint:
                variant_id = str(account["market_filter_variant"])
                context = contexts[variant_id]
                variant = context["variant"]
                config = config_for_candidate(variant.config, candidate)
                raw_atr = account.get("entry_atr_risk_pct")
                atr_risk = None if raw_atr is None else float(raw_atr)
                backtest = build_backtest(
                    canonical_prepared=canonical_prepared,
                    market_filter_trends=variant.market_filter_trends,
                    config=config,
                    candidate=candidate,
                    signal_cache=context["signal_cache"],
                    atr_risk=atr_risk,
                )
                if args.engine == "fast":
                    result = run_start_window_fast(
                        config,
                        data,
                        run_index,
                        context["signal_cache"],
                        candidate.policy,
                        atr_risk,
                    )
                else:
                    with patch.object(
                        canonical_runners,
                        "_prepare_backtest",
                        return_value=backtest,
                    ):
                        result = run_backtest_on_data(
                            config,
                            data,
                            run_index=run_index,
                        )
                metrics = {
                    metric: finite(value)
                    for metric, value in result.metrics.items()
                }
                row = {
                    "evaluation_key": key,
                    "account_id": candidate.candidate_id,
                    "account_name": str(account["name"]),
                    "market_filter_variant": variant_id,
                    "entry_atr_risk_pct": raw_atr,
                    **asdict(candidate.policy),
                    "launch_id": launch_id,
                    "start_session": str(pd.Timestamp(run_index[0]).date()),
                    "end_session": str(pd.Timestamp(run_index[-1]).date()),
                    "horizon_sessions": int(len(run_index)),
                    "engine": args.engine,
                    **metrics,
                    "benchmark_return": float(benchmark_return),
                    "alpha": float(metrics["total_return"] - benchmark_return),
                    **first_trade_fields(result.trade_records, run_index),
                }
                append_checkpoint(checkpoint_path, row)
                checkpoint[key] = row
                source = "run"
            else:
                row = checkpoint[key]
                source = "cache"
            completed += 1
            if completed % progress_every == 0 or completed == total:
                elapsed = time.monotonic() - started
                rate = completed / elapsed if elapsed > 0 else 0.0
                remaining = (total - completed) / rate if rate > 0 else 0.0
                print(
                    f"[inception] progress {completed}/{total} "
                    f"({completed / total:.1%}) launch={launch_id} "
                    f"account={candidate.candidate_id} "
                    f"return={float(row['total_return']):+.2%} "
                    f"source={source} elapsed={elapsed:.1f}s "
                    f"eta={remaining:.1f}s",
                    flush=True,
                )

    relevant_keys = {
        f"{account['account_id']}__{launch_id}"
        for launch_id, _ in windows
        for account in accounts
    }
    frame = pd.DataFrame(
        checkpoint[key] for key in sorted(relevant_keys)
    ).sort_values(["start_session", "account_id"])
    frame.to_csv(
        run_dir / "launch_results.csv", index=False, encoding="utf-8-sig"
    )
    aggregate_accounts(frame).to_csv(
        run_dir / "account_summary.csv", index=False, encoding="utf-8-sig"
    )
    aggregate_first_symbols(frame).to_csv(
        run_dir / "first_symbol_summary.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "run_id": run_id,
        "launch_count": len(windows),
        "account_count": len(accounts),
        "evaluation_count": total,
        "first_start_session": str(windows[0][1][0].date()),
        "last_start_session": str(windows[-1][1][0].date()),
        "horizon_sessions": int(len(windows[0][1])),
        "engine": args.engine,
        "elapsed_seconds": time.monotonic() - started,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[inception] complete evaluations={total} "
        f"elapsed={metadata['elapsed_seconds']:.1f}s results={run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
