from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import pandas as pd

from .brokers import PaperBroker, TossBroker
from .config import AppConfig
from .data import download_market_data, market_index
from .metrics import calculate_metrics, format_float, format_pct
from .portfolio import AccountSnapshot, OrderPlan, Position, estimate_quantity
from .strategies import build_order_plan, create_strategy
from .strategies.base import PreparedBacktest
from .strategies.common import active_universe_symbols
from .universe import resolve_universe


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    metrics: dict[str, float | int]
    trades: list[float]
    skipped: tuple[str, ...]
    trade_records: tuple[dict[str, object], ...] = field(default_factory=tuple)
    universe_snapshot: dict[str, object] | None = None


def run_backtest(config: AppConfig) -> BacktestResult:
    resolved = resolve_universe(config, mode="backtest")
    symbols = list(resolved.eligible_symbols)
    market_data = download_market_data(config, symbols, resolved_universe=resolved)
    return run_backtest_on_data(config, market_data)


def run_backtest_on_data(
    config: AppConfig,
    market_data,
    run_index: pd.Index | None = None,
) -> BacktestResult:
    """Run the canonical strategy/order path against already loaded market data.

    Research, normal backtests, and acceptance tests all enter here.  Supplying
    ``run_index`` resets the simulated account at the first selected bar while
    retaining earlier bars as indicator warm-up history.  Decisions are made
    from data through the signal bar and filled at the following bar's open.
    """
    if not market_data.bars:
        raise RuntimeError("No market data was downloaded.")

    full_idx = market_index(market_data)
    idx = _select_run_index(full_idx, run_index)
    if len(idx) < 2:
        raise RuntimeError("Not enough common bars to run a backtest.")

    cash = config.capital.initial_cash
    positions: dict[str, Position] = {}
    entry_values: dict[str, float] = {}
    entry_times: dict[str, object] = {}
    equity_points: list[tuple[pd.Timestamp, float]] = []
    trade_returns: list[float] = []
    trade_records: list[dict[str, object]] = []

    strategy = create_strategy(config)
    first_full_position = int(full_idx.get_indexer([idx[0]])[0])
    first_target_position = max(first_full_position, strategy.warmup_bars())
    eligible = full_idx[first_target_position:]
    idx = idx.intersection(eligible, sort=False)
    if len(idx) < 2:
        raise RuntimeError("Not enough bars remain after strategy warm-up.")
    prepared_backtest = _prepare_backtest(strategy, market_data)

    for i in range(0, len(idx) - 1):
        signal_ts = idx[i]
        exec_ts = idx[i + 1]
        equity_points.append((signal_ts, _portfolio_value(cash, positions, market_data.bars, signal_ts)))
        account = AccountSnapshot(cash=cash, positions=positions.copy())
        if prepared_backtest is not None:
            plan = prepared_backtest.build_order_plan(signal_ts, account, mode="backtest")
        else:
            allowed_symbols = _allowed_symbols_for_signal(market_data, signal_ts, positions)
            sliced = {
                symbol: df.loc[:signal_ts].copy()
                for symbol, df in market_data.bars.items()
                if allowed_symbols is None or symbol in allowed_symbols
            }
            benchmark = _slice_benchmark(market_data.benchmark, signal_ts)
            filter_benchmark = _slice_benchmark(market_data.filter_benchmark, signal_ts)
            plan = strategy.build_order_plan(
                sliced,
                account,
                mode="backtest",
                benchmark=benchmark,
                filter_benchmark=filter_benchmark,
            )

        for order in plan.orders:
            df = market_data.bars.get(order.symbol)
            if df is None or exec_ts not in df.index:
                continue
            raw_price = float(df.loc[exec_ts, "Open"])
            if order.side.lower() == "buy":
                affordable_quantity = estimate_quantity(
                    cash,
                    raw_price,
                    1.0,
                    fee_rate=config.costs.fee_rate,
                    slippage_rate=config.costs.slippage_rate,
                )
                quantity = min(order.quantity, affordable_quantity)
                if quantity <= 0:
                    continue
                fill = raw_price * (1.0 + config.costs.slippage_rate)
                cost = quantity * fill * (1.0 + config.costs.fee_rate)
                if cost <= cash:
                    cash -= cost
                    positions[order.symbol] = Position(order.symbol, quantity, fill)
                    entry_values[order.symbol] = cost
                    entry_times[order.symbol] = exec_ts
            else:
                position = positions.get(order.symbol)
                if not position:
                    continue
                qty = min(position.quantity, order.quantity)
                fill = raw_price * (1.0 - config.costs.slippage_rate)
                proceeds = qty * fill * (1.0 - config.costs.fee_rate)
                cash += proceeds
                entry_value = entry_values.pop(order.symbol, qty * position.avg_price)
                pnl_pct = proceeds / entry_value - 1.0 if entry_value else 0.0
                trade_returns.append(pnl_pct)
                trade_records.append(
                    {
                        "symbol": order.symbol,
                        "entry_time": entry_times.pop(order.symbol, None),
                        "exit_time": exec_ts,
                        "entry_price": position.avg_price,
                        "exit_price": fill,
                        "quantity": qty,
                        "pnl_pct": pnl_pct,
                        "exit_reason": order.reason,
                    }
                )
                positions.pop(order.symbol, None)

    if positions:
        final_ts = idx[-1]
        for symbol, position in list(positions.items()):
            final_close = _close_at_or_before(market_data.bars.get(symbol), final_ts)
            if final_close is None:
                continue
            final_price = final_close * (1.0 - config.costs.slippage_rate)
            proceeds = position.quantity * final_price * (1.0 - config.costs.fee_rate)
            cash += proceeds
            entry_value = entry_values.pop(symbol, position.quantity * position.avg_price)
            pnl_pct = proceeds / entry_value - 1.0 if entry_value else 0.0
            trade_returns.append(pnl_pct)
            trade_records.append(
                {
                    "symbol": symbol,
                    "entry_time": entry_times.pop(symbol, None),
                    "exit_time": final_ts,
                    "entry_price": position.avg_price,
                    "exit_price": final_price,
                    "quantity": position.quantity,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "FinalClose",
                }
            )
            positions.pop(symbol, None)

    equity_points.append((idx[-1], _portfolio_value(cash, positions, market_data.bars, idx[-1])))

    equity = pd.Series([point[1] for point in equity_points], index=[point[0] for point in equity_points], name="equity")
    return BacktestResult(
        equity=equity,
        metrics=calculate_metrics(equity, trade_returns, config.timeframe),
        trades=trade_returns,
        skipped=market_data.skipped,
        trade_records=tuple(trade_records),
        universe_snapshot=getattr(market_data, "universe_snapshot", None),
    )


def _select_run_index(full_index: pd.Index, requested: pd.Index | None) -> pd.Index:
    if requested is None:
        return full_index
    selected = full_index.intersection(pd.Index(requested), sort=False)
    if selected.empty:
        raise RuntimeError("Requested backtest segment has no common market bars.")
    positions = full_index.get_indexer(selected)
    if len(positions) > 1 and not bool(((positions[1:] - positions[:-1]) == 1).all()):
        raise ValueError("Requested backtest segment must be contiguous on the common market timeline.")
    return selected


def _prepare_backtest(strategy, market_data) -> PreparedBacktest | None:
    prepare = getattr(strategy, "prepare_backtest", None)
    if not callable(prepare):
        return None
    kwargs = {
        "benchmark": market_data.benchmark,
        "filter_benchmark": market_data.filter_benchmark,
    }
    parameters = inspect.signature(prepare).parameters
    if "universe_schedule" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        kwargs["universe_schedule"] = getattr(market_data, "universe_schedule", ())
    prepared = prepare(market_data.bars, **kwargs)
    if prepared is None:
        return None
    if not callable(getattr(prepared, "build_order_plan", None)):
        raise TypeError("prepare_backtest() must return an object with build_order_plan().")
    return prepared


def _allowed_symbols_for_signal(market_data, signal_ts, positions: dict[str, Position]) -> set[str] | None:
    active = active_universe_symbols(
        getattr(market_data, "universe_schedule", ()),
        signal_ts,
    )
    if active is None:
        return None
    return active | set(positions)


def run_paper_once(config: AppConfig, state_path: str) -> tuple[OrderPlan, list[str]]:
    broker = PaperBroker(state_path=state_path, initial_cash=config.capital.initial_cash)
    account = broker.get_account()
    resolved = resolve_universe(
        config,
        held_symbols=account.positions,
        previously_managed=account.positions,
        mode="paper",
    )
    symbols = list(resolved.symbols if resolved.entries_allowed else resolved.exit_only_symbols)
    market_data = download_market_data(config, symbols, resolved_universe=resolved)
    plan = build_order_plan(
        config,
        market_data.bars,
        account,
        mode="paper",
        benchmark=market_data.benchmark,
        filter_benchmark=market_data.filter_benchmark,
    )
    fills = broker.execute_plan(plan, _latest_prices(market_data.bars), config.costs.fee_rate, config.costs.slippage_rate)
    return plan, fills


def run_live_once(config: AppConfig, assume_yes: bool = False) -> tuple[OrderPlan, list[str]]:
    broker = TossBroker()
    account = broker.get_account(config.market)
    resolved = resolve_universe(
        config,
        held_symbols=account.positions,
        previously_managed=account.positions,
        mode="live",
    )
    symbols = list(resolved.symbols if resolved.entries_allowed else resolved.exit_only_symbols)
    market_data = download_market_data(config, symbols, resolved_universe=resolved)
    plan = build_order_plan(
        config,
        market_data.bars,
        account,
        mode="live",
        benchmark=market_data.benchmark,
        filter_benchmark=market_data.filter_benchmark,
    )
    if not plan.orders:
        return plan, []

    if config.execution.live_confirm_required and not assume_yes:
        _print_order_plan(plan)
        answer = input("Type yes to send live orders: ").strip()
        if answer != "yes":
            return plan, ["Live orders were not sent."]

    results = []
    for order in plan.orders:
        ok = broker.place_order(order)
        results.append(f"{'SENT' if ok else 'FAILED'} {order.side.upper()} {order.symbol} {order.quantity:g}")
    return plan, results


def print_backtest_result(result: BacktestResult) -> None:
    metrics = result.metrics
    print("Backtest Summary")
    print(f"Return      : {format_pct(float(metrics['total_return']))}")
    print(f"MDD         : {format_pct(float(metrics['mdd']))}")
    print(f"Sharpe      : {format_float(float(metrics['sharpe']))}")
    print(f"Win Rate    : {format_pct(float(metrics['win_rate']))}")
    print(f"Payoff      : {format_float(float(metrics['payoff_ratio']))}")
    print(f"Trades      : {metrics['trade_count']}")
    if result.skipped:
        print(f"Skipped     : {', '.join(result.skipped)}")


def _portfolio_value(cash: float, positions: dict[str, Position], bars: dict[str, pd.DataFrame], timestamp) -> float:
    value = cash
    for symbol, position in positions.items():
        close = _close_at_or_before(bars.get(symbol), timestamp)
        if close is not None:
            value += position.quantity * close
    return value


def _close_at_or_before(df: pd.DataFrame | None, timestamp) -> float | None:
    if df is None or df.empty or "Close" not in df:
        return None
    try:
        available = df.loc[:timestamp]
    except TypeError:
        signal_date = pd.Timestamp(timestamp).date()
        available = df.loc[[pd.Timestamp(idx).date() <= signal_date for idx in df.index]]
    if available.empty:
        return None
    return float(available["Close"].iloc[-1])


def _latest_prices(bars: dict[str, pd.DataFrame]) -> dict[str, float]:
    return {symbol: float(df["Close"].iloc[-1]) for symbol, df in bars.items() if not df.empty}


def _slice_benchmark(
    benchmark: pd.DataFrame | dict[str, pd.DataFrame] | None,
    signal_ts,
) -> pd.DataFrame | dict[str, pd.DataFrame] | None:
    if isinstance(benchmark, dict):
        sliced = {
            symbol: sliced_df
            for symbol, df in benchmark.items()
            if (sliced_df := _slice_benchmark_frame(df, signal_ts)) is not None and not sliced_df.empty
        }
        return sliced or None
    return _slice_benchmark_frame(benchmark, signal_ts)


def _slice_benchmark_frame(df: pd.DataFrame | None, signal_ts) -> pd.DataFrame | None:
    if df is None:
        return None
    try:
        return df.loc[:signal_ts].copy()
    except TypeError:
        signal_date = pd.Timestamp(signal_ts).date()
        return df.loc[[pd.Timestamp(idx).date() <= signal_date for idx in df.index]].copy()


def _print_order_plan(plan: OrderPlan) -> None:
    print("Live Order Plan")
    for order in plan.orders:
        print(f"{order.side.upper():4} {order.symbol:8} qty={order.quantity:g} type={order.order_type} reason={order.reason}")
