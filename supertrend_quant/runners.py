from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .brokers import PaperBroker, TossBroker
from .config import AppConfig, load_universe
from .data import common_index, download_market_data
from .metrics import calculate_metrics, format_float, format_pct
from .portfolio import AccountSnapshot, OrderPlan, Position
from .strategies import build_order_plan, create_strategy


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    metrics: dict[str, float | int]
    trades: list[float]
    skipped: tuple[str, ...]


def run_backtest(config: AppConfig) -> BacktestResult:
    symbols = load_universe(config)
    market_data = download_market_data(config, symbols)
    if not market_data.bars:
        raise RuntimeError("No market data was downloaded.")

    idx = common_index(market_data.bars)
    cash = config.capital.initial_cash
    positions: dict[str, Position] = {}
    entry_values: dict[str, float] = {}
    equity_points: list[tuple[pd.Timestamp, float]] = []
    trade_returns: list[float] = []

    strategy = create_strategy(config)
    start_i = strategy.warmup_bars()
    for i in range(start_i, len(idx) - 1):
        signal_ts = idx[i]
        exec_ts = idx[i + 1]
        sliced = {symbol: df.loc[:signal_ts].copy() for symbol, df in market_data.bars.items()}
        benchmark = _slice_benchmark(market_data.benchmark, signal_ts)
        filter_benchmark = _slice_benchmark(market_data.filter_benchmark, signal_ts)
        account = AccountSnapshot(cash=cash, positions=positions.copy())
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
                fill = raw_price * (1.0 + config.costs.slippage_rate)
                cost = order.quantity * fill * (1.0 + config.costs.fee_rate)
                if cost <= cash and order.quantity > 0:
                    cash -= cost
                    positions[order.symbol] = Position(order.symbol, order.quantity, fill)
                    entry_values[order.symbol] = cost
            else:
                position = positions.get(order.symbol)
                if not position:
                    continue
                qty = min(position.quantity, order.quantity)
                fill = raw_price * (1.0 - config.costs.slippage_rate)
                proceeds = qty * fill * (1.0 - config.costs.fee_rate)
                cash += proceeds
                entry_value = entry_values.pop(order.symbol, qty * position.avg_price)
                trade_returns.append(proceeds / entry_value - 1.0 if entry_value else 0.0)
                positions.pop(order.symbol, None)

        equity_points.append((signal_ts, _portfolio_value(cash, positions, market_data.bars, signal_ts)))

    if positions:
        final_ts = idx[-1]
        for symbol, position in list(positions.items()):
            final_price = float(market_data.bars[symbol].loc[final_ts, "Close"]) * (1.0 - config.costs.slippage_rate)
            proceeds = position.quantity * final_price * (1.0 - config.costs.fee_rate)
            cash += proceeds
            entry_value = entry_values.pop(symbol, position.quantity * position.avg_price)
            trade_returns.append(proceeds / entry_value - 1.0 if entry_value else 0.0)
            positions.pop(symbol, None)
        equity_points.append((final_ts, cash))

    equity = pd.Series([point[1] for point in equity_points], index=[point[0] for point in equity_points], name="equity")
    return BacktestResult(
        equity=equity,
        metrics=calculate_metrics(equity, trade_returns, config.timeframe),
        trades=trade_returns,
        skipped=market_data.skipped,
    )


def run_paper_once(config: AppConfig, state_path: str) -> tuple[OrderPlan, list[str]]:
    symbols = load_universe(config)
    market_data = download_market_data(config, symbols)
    broker = PaperBroker(state_path=state_path, initial_cash=config.capital.initial_cash)
    account = broker.get_account()
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
    symbols = load_universe(config)
    market_data = download_market_data(config, symbols)
    broker = TossBroker()
    account = broker.get_account(config.market)
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
        value += position.quantity * float(bars[symbol].loc[timestamp, "Close"])
    return value


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
