from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
import inspect
import math
import re
from dataclasses import dataclass, field, replace

import pandas as pd

from .brokers import PaperBroker, TossBroker
from .config import AppConfig
from .data import market_index
from .metrics import calculate_metrics, format_float, format_pct
from .ledger import PortfolioLedger
from .market_store.provider import ensure_configured_data_ready, load_configured_market_data
from .portfolio import OrderPlan, Position, estimate_quantity
from .strategies import build_order_plan, create_strategy
from .strategies.base import PreparedBacktest
from .strategies.common import active_universe_symbols
from .universe import resolve_universe


_PREPARED_BACKTEST_UNSET = object()


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    metrics: dict[str, float | int]
    trades: list[float]
    skipped: tuple[str, ...]
    trade_records: tuple[dict[str, object], ...] = field(default_factory=tuple)
    universe_snapshot: dict[str, object] | None = None
    data_version: str = ""
    completed_session: str = ""
    data_quality: str = "valid"
    data_warnings: tuple[str, ...] = field(default_factory=tuple)
    processed_corporate_action_ids: tuple[str, ...] = field(default_factory=tuple)
    price_mode: str = "total_return_adjusted"
    dividend_tax_rate: float = 0.0
    corporate_action_cash: float = 0.0
    unresolved_corporate_action_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def validated_session(self) -> str:
        return self.completed_session


@dataclass(frozen=True)
class IntradayStopPolicy:
    """Prior-session signal levels executed against the next raw OHLC bar."""

    signal_levels: Mapping[str, pd.Series]
    catastrophe_loss_pct: float | None = None

    def __post_init__(self) -> None:
        if self.catastrophe_loss_pct is not None and not (
            0.0 < float(self.catastrophe_loss_pct) < 1.0
        ):
            raise ValueError("catastrophe_loss_pct must be between 0 and 1.")


@dataclass(frozen=True)
class _PreOpenOrderTransition:
    successor_symbol: str | None
    quantity_ratio: float
    cancel_buy: bool = False


def run_backtest(config: AppConfig) -> BacktestResult:
    ensure_configured_data_ready(config)
    resolved = resolve_universe(config, mode="backtest")
    symbols = list(resolved.eligible_symbols)
    if resolved.schedule:
        relevant_schedule = _schedule_for_period(
            resolved.schedule,
            config.period,
            _configured_completed_session(config),
        )
        symbols = list(
            dict.fromkeys(
                member.symbol
                for entry in relevant_schedule
                for member in entry.members
            )
        )
    market_data = load_configured_market_data(
        config,
        symbols,
        resolved_universe=resolved,
    )
    return run_backtest_on_data(config, market_data)


def _configured_completed_session(config: AppConfig) -> str:
    from .market_store.repository import LocalDatasetRepository

    repository = LocalDatasetRepository(config.data_store.local_cache_dir)
    release, _ = repository.current_release()
    if release is not None:
        return release.completed_session
    manifest = repository.current_manifest("daily_price_raw")
    return manifest.completed_session if manifest is not None else ""


def _schedule_for_period(schedule, period: str, completed_session: str):
    entries = tuple(sorted(schedule, key=lambda entry: entry.effective_date))
    if not entries or period == "max" or not completed_session:
        return entries
    match = re.fullmatch(r"(\d+)(d|mo|y)", str(period).strip().lower())
    if not match:
        return entries
    end = pd.Timestamp(completed_session)
    amount = int(match.group(1))
    unit = match.group(2)
    start = (
        end - pd.Timedelta(days=amount)
        if unit == "d"
        else end - pd.DateOffset(months=amount)
        if unit == "mo"
        else end - pd.DateOffset(years=amount)
    )
    prior = [entry for entry in entries if pd.Timestamp(entry.effective_date) <= start]
    after = [entry for entry in entries if pd.Timestamp(entry.effective_date) > start]
    return tuple(([prior[-1]] if prior else []) + after)


def run_backtest_on_data(
    config: AppConfig,
    market_data,
    run_index: pd.Index | None = None,
    *,
    prepared_backtest: PreparedBacktest | None | object = _PREPARED_BACKTEST_UNSET,
    intraday_stop_policy: IntradayStopPolicy | None = None,
) -> BacktestResult:
    """Run the canonical strategy/order path against already loaded market data.

    Research, normal backtests, and acceptance tests all enter here.  Supplying
    ``run_index`` resets the simulated account at the first selected bar while
    retaining earlier bars as indicator warm-up history.  Decisions are made
    from data through the signal bar and filled at the following bar's open.
    """
    if not market_data.bars:
        raise RuntimeError("No market data was downloaded.")
    if getattr(market_data, "data_quality", "valid") == "blocked":
        warnings = "; ".join(getattr(market_data, "warnings", ()))
        raise RuntimeError(
            "Market data quality is blocked."
            + (f" {warnings}" if warnings else "")
        )

    full_idx = market_index(market_data)
    idx = _select_run_index(full_idx, run_index)
    if len(idx) < 2:
        raise RuntimeError("Not enough common bars to run a backtest.")

    ledger = PortfolioLedger(
        cash=config.capital.initial_cash,
        dividend_tax_rate=config.data_store.dividend_tax_rate,
    )
    execution_bars = getattr(market_data, "execution_bars", None) or market_data.bars
    corporate_actions = tuple(getattr(market_data, "corporate_actions", ()))
    scheduled_actions = _scheduled_corporate_actions(corporate_actions)
    actions_by_event_id = {
        str(action.get("event_id") or ""): action
        for action in corporate_actions
        if str(action.get("event_id") or "")
    }
    retired_symbols: dict[str, set[str]] = {}
    identity_schedule_dates, identity_schedule_maps = (
        _compile_universe_identity_schedule(
            tuple(getattr(market_data, "universe_schedule", ()) or ())
        )
    )
    action_cursor = 0
    entry_values: dict[str, float] = {}
    entry_times: dict[str, object] = {}
    entry_distributions: dict[str, float] = {}
    corporate_action_cash = 0.0
    equity_points: list[tuple[pd.Timestamp, float]] = []
    trade_returns: list[float] = []
    trade_records: list[dict[str, object]] = []

    def apply_actions_before_open(timestamp) -> dict[str, _PreOpenOrderTransition]:
        nonlocal action_cursor, corporate_action_cash
        due_actions = []
        cutoff = pd.Timestamp(timestamp).normalize()
        while (
            action_cursor < len(scheduled_actions)
            and scheduled_actions[action_cursor][0] <= cutoff
        ):
            due_actions.append(scheduled_actions[action_cursor][2])
            action_cursor += 1

        symbol_transitions: dict[str, _PreOpenOrderTransition] = {}

        def add_transition(
            source_symbol: str,
            successor_symbol: str | None,
            quantity_ratio: float,
            *,
            cancel_buy: bool,
        ) -> None:
            local = _PreOpenOrderTransition(
                successor_symbol,
                quantity_ratio,
                cancel_buy,
            )
            for original_symbol, prior in tuple(symbol_transitions.items()):
                if prior.successor_symbol != source_symbol:
                    continue
                symbol_transitions[original_symbol] = _PreOpenOrderTransition(
                    successor_symbol,
                    prior.quantity_ratio * quantity_ratio,
                    prior.cancel_buy or cancel_buy,
                )
            if source_symbol not in symbol_transitions:
                symbol_transitions[source_symbol] = local

        def record_event(event, held_before: Position | None = None) -> None:
            nonlocal corporate_action_cash
            corporate_action_cash += event.cash_delta
            # Dividend economics are recognized once when entitlement is
            # accrued.  A later payment event moves receivable to cash but
            # must not add the same distribution to trade P&L a second time.
            economic_delta = (
                event.accrual_delta
                if event.action_type in {"cash_dividend", "special_dividend"}
                else event.accrual_delta or event.cash_delta
            )
            terminal_action = event.action_type in {"cash_merger", "delisting"}
            action = actions_by_event_id.get(event.event_id, {})
            action_completed = (
                event.event_id in ledger.processed_event_ids
                or event.event_id in ledger.entitled_event_ids
            )
            if action_completed:
                new_symbol = str(action.get("new_symbol") or "").strip()
                source_symbol = str(event.symbol or "").strip()
                source_security_id = str(
                    action.get("security_id") or ""
                ).strip()
                new_security_id = str(
                    action.get("new_security_id") or ""
                ).strip()
                changes_symbol_or_identity = bool(
                    new_symbol
                    and (
                        new_symbol != source_symbol
                        or (
                            new_security_id
                            and source_security_id
                            and new_security_id != source_security_id
                        )
                    )
                )
                retires_source = event.action_type in {
                    "cash_merger",
                    "delisting",
                    "stock_merger",
                    "ticker_change",
                } and (not new_symbol or changes_symbol_or_identity)
                if (
                    event.action_type in {"stock_merger", "ticker_change"}
                    and changes_symbol_or_identity
                ):
                    # A later reviewed transition may legitimately reactivate
                    # an old display ticker (for example FISV -> FI -> FISV).
                    # The explicit successor identity is authoritative.
                    _reactivate_retired_symbol(
                        retired_symbols,
                        new_symbol,
                        new_security_id,
                    )
                if retires_source and source_symbol:
                    retired_symbols.setdefault(source_symbol, set()).add(
                        source_security_id
                    )
            if (
                not terminal_action
                and event.symbol in entry_values
                and economic_delta
            ):
                entry_distributions[event.symbol] = (
                    entry_distributions.get(event.symbol, 0.0) + economic_delta
                )
            _transfer_corporate_action_entry_state(
                event,
                actions_by_event_id,
                ledger.processed_event_ids,
                entry_values,
                entry_times,
                entry_distributions,
            )
            if event.event_id in ledger.unresolved_event_ids:
                raise RuntimeError(
                    "Held corporate action is incomplete and cannot be valued: "
                    f"{event.event_id}/{event.action_type}"
                )
            if (
                terminal_action
                and held_before is not None
                and (
                    event.event_id in ledger.processed_event_ids
                    or event.event_id in ledger.entitled_event_ids
                )
                and event.symbol not in ledger.positions
            ):
                quantity = held_before.quantity
                entry_value = entry_values.pop(
                    event.symbol,
                    quantity * held_before.avg_price,
                )
                distributions = entry_distributions.pop(event.symbol, 0.0)
                economic_proceeds = (
                    event.accrual_delta or event.cash_delta
                ) + distributions
                pnl_pct = (
                    economic_proceeds / entry_value - 1.0
                    if entry_value
                    else 0.0
                )
                trade_returns.append(pnl_pct)
                trade_records.append(
                    {
                        "symbol": event.symbol,
                        "entry_time": entry_times.pop(event.symbol, None),
                        "exit_time": cutoff,
                        "entry_price": held_before.avg_price,
                        "exit_price": (
                            (event.accrual_delta or event.cash_delta) / quantity
                            if quantity
                            else 0.0
                        ),
                        "quantity": quantity,
                        "corporate_action_cash": distributions,
                        "pnl_pct": pnl_pct,
                        "exit_reason": (
                            "CashMerger"
                            if event.action_type == "cash_merger"
                            else "Delisting"
                        ),
                        "corporate_action_event_id": event.event_id,
                    }
                )

        # Settle cash receivables even on sessions with no newly effective
        # actions.  Applying each due action separately below lets us measure
        # the exact quantity created by a stock merger before remapping a
        # prior-close sell order.
        for event in ledger.apply_actions((), through=timestamp):
            record_event(event)

        for action in due_actions:
            source_symbol = str(
                action.get("symbol") or action.get("old_symbol") or ""
            )
            held_before = ledger.positions.get(source_symbol)
            new_symbol = str(action.get("new_symbol") or "")
            successor_quantity_before = (
                ledger.positions[new_symbol].quantity
                if new_symbol in ledger.positions and new_symbol != source_symbol
                else 0.0
            )
            events = ledger.apply_actions((action,), through=timestamp)
            for event in events:
                record_event(
                    event,
                    (
                        held_before
                        if event.event_id == str(action.get("event_id") or "")
                        else None
                    ),
                )
                if (
                    event.action_type in {"cash_merger", "delisting"}
                    and (
                        event.event_id in ledger.processed_event_ids
                        or event.event_id in ledger.entitled_event_ids
                    )
                    and event.symbol
                ):
                    add_transition(
                        event.symbol,
                        None,
                        0.0,
                        cancel_buy=True,
                    )
                    continue
                if (
                    event.action_type
                    in {"split", "stock_dividend", "capital_reduction"}
                    and event.event_id in ledger.processed_event_ids
                    and event.symbol
                ):
                    resulting_position = ledger.positions.get(event.symbol)
                    if held_before is not None and held_before.quantity > 0:
                        quantity_ratio = (
                            resulting_position.quantity
                            if resulting_position is not None
                            else 0.0
                        ) / held_before.quantity
                    else:
                        try:
                            quantity_ratio = float(action.get("ratio"))
                        except (TypeError, ValueError):
                            quantity_ratio = 0.0
                    add_transition(
                        event.symbol,
                        event.symbol,
                        quantity_ratio,
                        cancel_buy=False,
                    )
                    continue
                if (
                    event.action_type not in {"ticker_change", "stock_merger"}
                    or event.event_id not in ledger.processed_event_ids
                    or not new_symbol
                    or (
                        new_symbol == event.symbol
                        and event.action_type == "ticker_change"
                    )
                ):
                    continue
                successor_position = ledger.positions.get(new_symbol)
                converted_quantity = max(
                    (successor_position.quantity if successor_position else 0.0)
                    - successor_quantity_before,
                    0.0,
                )
                if held_before is not None and held_before.quantity > 0:
                    quantity_ratio = converted_quantity / held_before.quantity
                elif event.action_type == "stock_merger":
                    try:
                        quantity_ratio = float(action.get("ratio"))
                    except (TypeError, ValueError):
                        quantity_ratio = 0.0
                else:
                    quantity_ratio = 1.0
                add_transition(
                    event.symbol,
                    new_symbol,
                    quantity_ratio,
                    cancel_buy=True,
                )
        return symbol_transitions

    strategy = create_strategy(config)
    first_full_position = int(full_idx.get_indexer([idx[0]])[0])
    first_target_position = max(first_full_position, strategy.warmup_bars())
    eligible = full_idx[first_target_position:]
    idx = idx.intersection(eligible, sort=False)
    if len(idx) < 2:
        raise RuntimeError("Not enough bars remain after strategy warm-up.")
    if prepared_backtest is _PREPARED_BACKTEST_UNSET:
        prepared_backtest = _prepare_backtest(strategy, market_data)
    elif prepared_backtest is not None and not callable(
        getattr(prepared_backtest, "build_order_plan", None)
    ):
        raise TypeError("prepared_backtest must provide build_order_plan().")
    apply_actions_before_open(idx[0])

    for i in range(0, len(idx) - 1):
        signal_ts = idx[i]
        exec_ts = idx[i + 1]
        positions = ledger.positions
        equity_points.append(
            (
                signal_ts,
                _portfolio_value(ledger.cash, positions, execution_bars, signal_ts)
                + ledger.receivable_value,
            )
        )
        account = ledger.snapshot()
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

        # Corporate-action entitlement is fixed immediately before the
        # ex-date open.  Apply it to the prior-close holdings before executing
        # orders generated by the prior signal; an ex-date-open buyer must not
        # receive the distribution, while an ex-date-open seller must.
        symbol_transitions = apply_actions_before_open(exec_ts)
        positions = ledger.positions
        for planned_order in plan.orders:
            order = planned_order
            if order.side.lower() == "buy" and _buy_crosses_identity_boundary(
                order.symbol,
                signal_ts,
                exec_ts,
                identity_schedule_dates,
                identity_schedule_maps,
            ):
                # The order was decided from the predecessor issuer's signal.
                # Never fill it against a different issuer merely because the
                # point-in-time schedule reused the same display ticker at the
                # next open.  The successor becomes eligible after producing
                # its own first signal.
                continue
            transition = symbol_transitions.get(order.symbol)
            if transition:
                successor = transition.successor_symbol
                if successor is None:
                    # The security ceased to exist at this open.  The ledger
                    # already records any held terminal settlement.
                    continue
                if order.side.lower() == "buy" and transition.cancel_buy:
                    # A prior-close entry signal for a ticker that ceased to
                    # exist at this open is stale; cancelling is safer than
                    # inventing a post-transition entry decision.
                    continue
                quantity_ratio = (
                    transition.quantity_ratio
                    if order.side.lower() == "sell"
                    or config.data_store.price_mode == "raw"
                    else 1.0
                )
                order = replace(
                    order,
                    symbol=successor,
                    quantity=order.quantity * quantity_ratio,
                )
                if order.quantity <= 0:
                    continue
            if order.side.lower() == "buy" and _retired_symbol_blocks_buy(
                retired_symbols,
                order.symbol,
                exec_ts,
                identity_schedule_dates,
                identity_schedule_maps,
            ):
                # A point-in-time membership schedule can lag a merger,
                # delisting, or ticker transition.  Once the old security has
                # ceased to exist, never re-enter that retired identity merely
                # because a stale schedule still lists its display ticker.
                continue
            df = execution_bars.get(order.symbol)
            if df is None or exec_ts not in df.index:
                if order.side.lower() == "sell" and order.symbol in positions:
                    raise RuntimeError(
                        "Held position has no executable price for a requested sell: "
                        f"{order.symbol}/{pd.Timestamp(exec_ts).date()}"
                    )
                continue
            raw_price = float(df.loc[exec_ts, "Open"])
            if order.side.lower() == "buy":
                if intraday_stop_policy is not None:
                    entry_stop = _raw_intraday_stop_level(
                        intraday_stop_policy,
                        symbol=order.symbol,
                        signal_ts=signal_ts,
                        exec_ts=exec_ts,
                        avg_price=0.0,
                        signal_bars=market_data.bars.get(order.symbol),
                        execution_bars=execution_bars.get(order.symbol),
                    )
                    if entry_stop is not None and raw_price <= entry_stop:
                        continue
                affordable_quantity = estimate_quantity(
                    ledger.cash,
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
                if cost <= ledger.cash:
                    ledger.buy(order.symbol, quantity, fill, cost)
                    entry_values[order.symbol] = cost
                    entry_times[order.symbol] = exec_ts
                    entry_distributions[order.symbol] = 0.0
            else:
                position = positions.get(order.symbol)
                if not position:
                    continue
                qty = min(position.quantity, order.quantity)
                fill = raw_price * (1.0 - config.costs.slippage_rate)
                proceeds = qty * fill * (1.0 - config.costs.fee_rate)
                ledger.sell(order.symbol, qty, proceeds)
                entry_value = entry_values.pop(order.symbol, qty * position.avg_price)
                distributions = entry_distributions.pop(order.symbol, 0.0)
                economic_proceeds = proceeds + distributions
                pnl_pct = economic_proceeds / entry_value - 1.0 if entry_value else 0.0
                trade_returns.append(pnl_pct)
                trade_records.append(
                    {
                        "symbol": order.symbol,
                        "entry_time": entry_times.pop(order.symbol, None),
                        "exit_time": exec_ts,
                        "entry_price": position.avg_price,
                        "exit_price": fill,
                        "quantity": qty,
                        "corporate_action_cash": distributions,
                        "pnl_pct": pnl_pct,
                        "exit_reason": order.reason,
                    }
                )

        if intraday_stop_policy is not None:
            for symbol, position in list(ledger.positions.items()):
                stop_level = _raw_intraday_stop_level(
                    intraday_stop_policy,
                    symbol=symbol,
                    signal_ts=signal_ts,
                    exec_ts=exec_ts,
                    avg_price=position.avg_price,
                    signal_bars=market_data.bars.get(symbol),
                    execution_bars=execution_bars.get(symbol),
                )
                if stop_level is None:
                    continue
                execution_frame = execution_bars.get(symbol)
                if execution_frame is None or exec_ts not in execution_frame.index:
                    continue
                execution_row = execution_frame.loc[exec_ts]
                raw_open = float(execution_row["Open"])
                raw_low = float(execution_row["Low"])
                if raw_open <= stop_level:
                    trigger = "gap_open"
                    raw_fill = raw_open
                elif raw_low <= stop_level:
                    trigger = "intraday"
                    raw_fill = stop_level
                else:
                    continue

                fill = raw_fill * (1.0 - config.costs.slippage_rate)
                proceeds = position.quantity * fill * (1.0 - config.costs.fee_rate)
                ledger.sell(symbol, position.quantity, proceeds)
                entry_value = entry_values.pop(
                    symbol,
                    position.quantity * position.avg_price,
                )
                distributions = entry_distributions.pop(symbol, 0.0)
                economic_proceeds = proceeds + distributions
                pnl_pct = economic_proceeds / entry_value - 1.0 if entry_value else 0.0
                trade_returns.append(pnl_pct)
                trade_records.append(
                    {
                        "symbol": symbol,
                        "entry_time": entry_times.pop(symbol, None),
                        "exit_time": exec_ts,
                        "entry_price": position.avg_price,
                        "exit_price": fill,
                        "quantity": position.quantity,
                        "corporate_action_cash": distributions,
                        "pnl_pct": pnl_pct,
                        "exit_reason": "Intraday protective stop",
                        "stop_level": stop_level,
                        "stop_trigger": trigger,
                    }
                )

    positions = ledger.positions
    if positions:
        final_ts = idx[-1]
        for symbol, position in list(positions.items()):
            final_close = _close_on(execution_bars.get(symbol), final_ts)
            if final_close is None:
                raise RuntimeError(
                    "Held position has no price for final valuation: "
                    f"{symbol}/{pd.Timestamp(final_ts).date()}"
                )
            final_price = final_close * (1.0 - config.costs.slippage_rate)
            proceeds = position.quantity * final_price * (1.0 - config.costs.fee_rate)
            ledger.sell(symbol, position.quantity, proceeds)
            entry_value = entry_values.pop(symbol, position.quantity * position.avg_price)
            distributions = entry_distributions.pop(symbol, 0.0)
            economic_proceeds = proceeds + distributions
            pnl_pct = economic_proceeds / entry_value - 1.0 if entry_value else 0.0
            trade_returns.append(pnl_pct)
            trade_records.append(
                {
                    "symbol": symbol,
                    "entry_time": entry_times.pop(symbol, None),
                    "exit_time": final_ts,
                    "entry_price": position.avg_price,
                    "exit_price": final_price,
                    "quantity": position.quantity,
                    "corporate_action_cash": distributions,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "FinalClose",
                }
            )

    equity_points.append(
        (
            idx[-1],
            _portfolio_value(ledger.cash, ledger.positions, execution_bars, idx[-1])
            + ledger.receivable_value,
        )
    )

    equity = pd.Series([point[1] for point in equity_points], index=[point[0] for point in equity_points], name="equity")
    unresolved_warnings = (
        (
            "Unresolved corporate actions were left unapplied: "
            + ", ".join(sorted(ledger.unresolved_event_ids)),
        )
        if ledger.unresolved_event_ids
        else ()
    )
    return BacktestResult(
        equity=equity,
        metrics=calculate_metrics(equity, trade_returns, config.timeframe),
        trades=trade_returns,
        skipped=market_data.skipped,
        trade_records=tuple(trade_records),
        universe_snapshot=getattr(market_data, "universe_snapshot", None),
        data_version=getattr(market_data, "data_version", ""),
        completed_session=getattr(market_data, "completed_session", ""),
        data_quality=(
            "degraded"
            if ledger.unresolved_event_ids
            and getattr(market_data, "data_quality", "valid") == "valid"
            else getattr(market_data, "data_quality", "valid")
        ),
        data_warnings=tuple(getattr(market_data, "warnings", ())) + unresolved_warnings,
        processed_corporate_action_ids=tuple(sorted(ledger.processed_event_ids)),
        price_mode=config.data_store.price_mode,
        dividend_tax_rate=config.data_store.dividend_tax_rate,
        corporate_action_cash=corporate_action_cash,
        unresolved_corporate_action_ids=tuple(sorted(ledger.unresolved_event_ids)),
    )


def _raw_intraday_stop_level(
    policy: IntradayStopPolicy,
    *,
    symbol: str,
    signal_ts,
    exec_ts,
    avg_price: float,
    signal_bars: pd.DataFrame | None,
    execution_bars: pd.DataFrame | None,
) -> float | None:
    candidates: list[float] = []
    levels = policy.signal_levels.get(symbol)
    if levels is not None and signal_ts in levels.index:
        adjusted_level = pd.to_numeric(
            pd.Series([levels.loc[signal_ts]]),
            errors="coerce",
        ).iloc[0]
        if (
            pd.notna(adjusted_level)
            and signal_bars is not None
            and exec_ts in signal_bars.index
            and execution_bars is not None
            and exec_ts in execution_bars.index
        ):
            adjusted_open = float(signal_bars.loc[exec_ts, "Open"])
            raw_open = float(execution_bars.loc[exec_ts, "Open"])
            if adjusted_open > 0.0 and raw_open > 0.0:
                candidates.append(float(adjusted_level) * raw_open / adjusted_open)

    if policy.catastrophe_loss_pct is not None and avg_price > 0.0:
        candidates.append(avg_price * (1.0 - float(policy.catastrophe_loss_pct)))

    finite = [value for value in candidates if math.isfinite(value) and value > 0.0]
    return max(finite) if finite else None


def _scheduled_corporate_actions(actions):
    grouped: dict[
        pd.Timestamp,
        list[tuple[pd.Timestamp, tuple[int, str], object]],
    ] = {}
    for action in actions:
        raw_effective = action.get("ex_date")
        if pd.isna(raw_effective) or not str(raw_effective).strip():
            raw_effective = action.get("effective_date")
        effective = pd.to_datetime(raw_effective, errors="coerce")
        if pd.isna(effective):
            continue
        item = (
            effective.normalize(),
            _corporate_action_sort_key(action),
            action,
        )
        grouped.setdefault(effective.normalize(), []).append(item)

    scheduled = []
    position_creators = {"spinoff", "stock_merger", "ticker_change"}
    for effective in sorted(grouped):
        nodes = grouped[effective]
        creators: list[tuple[int, str, str]] = []
        for index, (_, _, action) in enumerate(nodes):
            action_type = str(action.get("action_type") or "").lower()
            source_symbol = str(
                action.get("symbol") or action.get("old_symbol") or ""
            ).strip()
            source_security_id = str(action.get("security_id") or "").strip()
            successor_symbol = str(action.get("new_symbol") or "").strip()
            successor_security_id = str(
                action.get("new_security_id") or ""
            ).strip()
            if (
                action_type in position_creators
                and successor_symbol
                and (
                    successor_symbol != source_symbol
                    or (
                        successor_security_id
                        and source_security_id
                        and successor_security_id != source_security_id
                    )
                )
            ):
                creators.append(
                    (index, successor_symbol, successor_security_id)
                )

        edges: list[set[int]] = [set() for _ in nodes]
        indegree = [0 for _ in nodes]
        for target, (_, _, action) in enumerate(nodes):
            source_symbol = str(
                action.get("symbol") or action.get("old_symbol") or ""
            ).strip()
            source_security_id = str(action.get("security_id") or "").strip()
            for creator, successor_symbol, successor_security_id in creators:
                # Exact identities disambiguate same-ticker transitions and
                # ticker reuse.  Legacy action records without IDs retain the
                # conservative symbol dependency used previously.
                if source_security_id and successor_security_id:
                    depends_on_creator = source_security_id == successor_security_id
                else:
                    depends_on_creator = source_symbol == successor_symbol
                if not depends_on_creator:
                    continue
                if creator == target or target in edges[creator]:
                    continue
                edges[creator].add(target)
                indegree[target] += 1

        def node_key(index: int):
            return nodes[index][1], index

        ready = sorted(
            (index for index, degree in enumerate(indegree) if degree == 0),
            key=node_key,
        )
        ordered: list[int] = []
        while ready:
            current = ready.pop(0)
            ordered.append(current)
            for target in sorted(edges[current], key=node_key):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort(key=node_key)
        if len(ordered) != len(nodes):
            cycle_ids = sorted(
                str(nodes[index][2].get("event_id") or "")
                for index, degree in enumerate(indegree)
                if degree > 0
            )
            raise ValueError(
                "Same-session corporate-action successor dependency is cyclic: "
                + ", ".join(cycle_ids)
            )
        scheduled.extend(nodes[index] for index in ordered)
    return tuple(scheduled)


def _compile_universe_identity_schedule(
    schedule: tuple[Mapping[str, object], ...],
) -> tuple[tuple[object, ...], tuple[dict[str, str], ...]]:
    """Compile exact symbol identities for reuse-safe retirement checks."""

    compiled: list[tuple[object, dict[str, str]]] = []
    for entry in sorted(schedule, key=lambda item: str(item.get("effective_date", ""))):
        effective = pd.to_datetime(entry.get("effective_date"), errors="coerce")
        if pd.isna(effective):
            continue
        identities: dict[str, str] = {}
        members = entry.get("members", ()) or ()
        if isinstance(members, (list, tuple)):
            for member in members:
                if not isinstance(member, Mapping):
                    continue
                symbol = str(member.get("symbol") or "").strip()
                security_id = str(member.get("security_id") or "").strip()
                if symbol and security_id:
                    identities[symbol] = security_id
        compiled.append((pd.Timestamp(effective).date(), identities))
    return (
        tuple(item[0] for item in compiled),
        tuple(item[1] for item in compiled),
    )


def _retired_symbol_blocks_buy(
    retired_symbols: dict[str, set[str]],
    symbol: str,
    timestamp,
    identity_schedule_dates: tuple[object, ...],
    identity_schedule_maps: tuple[dict[str, str], ...],
) -> bool:
    retired_ids = retired_symbols.get(str(symbol), set())
    if not retired_ids:
        return False
    if not identity_schedule_dates:
        return True
    active_security_id = _scheduled_security_id(
        symbol,
        timestamp,
        identity_schedule_dates,
        identity_schedule_maps,
    )
    if not active_security_id:
        return True
    # Blank IDs represent legacy/static actions without an exact identity and
    # must remain fail-closed.  A later schedule row with a different exact ID
    # allows a genuinely reused ticker to trade again.
    return "" in retired_ids or active_security_id in retired_ids


def _reactivate_retired_symbol(
    retired_symbols: dict[str, set[str]],
    symbol: str,
    security_id: str,
) -> None:
    """Reactivate only one exact successor identity for a reused alias."""

    retired_ids = retired_symbols.get(str(symbol))
    exact_id = str(security_id).strip()
    if retired_ids is None or not exact_id:
        return
    retired_ids.discard(exact_id)
    if not retired_ids:
        retired_symbols.pop(str(symbol), None)


def _scheduled_security_id(
    symbol: str,
    timestamp,
    identity_schedule_dates: tuple[object, ...],
    identity_schedule_maps: tuple[dict[str, str], ...],
) -> str:
    if not identity_schedule_dates:
        return ""
    position = bisect_right(
        identity_schedule_dates,
        pd.Timestamp(timestamp).date(),
    ) - 1
    if position < 0:
        return ""
    return identity_schedule_maps[position].get(str(symbol), "")


def _buy_crosses_identity_boundary(
    symbol: str,
    signal_timestamp,
    execution_timestamp,
    identity_schedule_dates: tuple[object, ...],
    identity_schedule_maps: tuple[dict[str, str], ...],
) -> bool:
    signal_security_id = _scheduled_security_id(
        symbol,
        signal_timestamp,
        identity_schedule_dates,
        identity_schedule_maps,
    )
    execution_security_id = _scheduled_security_id(
        symbol,
        execution_timestamp,
        identity_schedule_dates,
        identity_schedule_maps,
    )
    return bool(
        signal_security_id
        and execution_security_id
        and signal_security_id != execution_security_id
    )


def _corporate_action_sort_key(action) -> tuple[int, str]:
    action_type = str(action.get("action_type") or "").lower()
    # A distribution belongs to the pre-transition holder.  Apply it before a
    # same-session ticker move so the child cannot disappear merely because
    # opaque event hashes happen to sort in the opposite order.
    priority = {
        "spinoff": 10,
        "split": 20,
        "stock_dividend": 20,
        "capital_reduction": 20,
        "cash_dividend": 30,
        "special_dividend": 30,
        "stock_merger": 40,
        "ticker_change": 90,
    }.get(action_type, 50)
    return priority, str(action.get("event_id") or "")


def _transfer_corporate_action_entry_state(
    event,
    actions_by_event_id,
    processed_event_ids,
    entry_values,
    entry_times,
    entry_distributions,
) -> None:
    if event.event_id not in processed_event_ids:
        return
    action = actions_by_event_id.get(event.event_id, {})
    new_symbol = str(action.get("new_symbol") or "")
    old_symbol = event.symbol
    if not new_symbol or not old_symbol or new_symbol == old_symbol:
        return
    if event.action_type == "spinoff":
        metadata = _corporate_action_metadata(action)
        try:
            cost_fraction = float(metadata.get("cost_basis_fraction", 0.0))
        except (TypeError, ValueError):
            cost_fraction = 0.0
        cost_fraction = min(max(cost_fraction, 0.0), 1.0)
        if old_symbol in entry_values:
            original = entry_values[old_symbol]
            child_value = original * cost_fraction
            entry_values[old_symbol] = original - child_value
            entry_values[new_symbol] = entry_values.get(new_symbol, 0.0) + child_value
        _copy_earliest_entry_time(entry_times, old_symbol, new_symbol)
        return
    if event.action_type not in {"ticker_change", "stock_merger"}:
        return
    _move_additive_entry_value(entry_values, old_symbol, new_symbol)
    _move_earliest_entry_time(entry_times, old_symbol, new_symbol)
    _move_additive_entry_value(entry_distributions, old_symbol, new_symbol)


def _corporate_action_metadata(action) -> dict[str, object]:
    value = action.get("metadata", {})
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        import json

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _move_additive_entry_value(values, old_symbol: str, new_symbol: str) -> None:
    if old_symbol not in values:
        return
    old_value = values.pop(old_symbol)
    values[new_symbol] = values.get(new_symbol, 0.0) + old_value


def _move_earliest_entry_time(values, old_symbol: str, new_symbol: str) -> None:
    if old_symbol not in values:
        return
    old_value = values.pop(old_symbol)
    if new_symbol not in values:
        values[new_symbol] = old_value
        return
    try:
        if pd.Timestamp(old_value) < pd.Timestamp(values[new_symbol]):
            values[new_symbol] = old_value
    except (TypeError, ValueError):
        return


def _copy_earliest_entry_time(values, old_symbol: str, new_symbol: str) -> None:
    if old_symbol not in values:
        return
    old_value = values[old_symbol]
    if new_symbol not in values:
        values[new_symbol] = old_value
        return
    try:
        if pd.Timestamp(old_value) < pd.Timestamp(values[new_symbol]):
            values[new_symbol] = old_value
    except (TypeError, ValueError):
        return


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
    if "execution_bars" in parameters:
        kwargs["execution_bars"] = (
            getattr(market_data, "execution_bars", None) or market_data.bars
        )
    if "universe_schedule" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        kwargs["universe_schedule"] = _strategy_universe_schedule(market_data)
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
        entry_symbols = tuple(getattr(market_data, "entry_symbols", ()) or ())
        if entry_symbols:
            active = set(entry_symbols)
    if active is None:
        return None
    return active | set(positions)


def _strategy_universe_schedule(market_data) -> tuple[dict[str, object], ...]:
    schedule = tuple(getattr(market_data, "universe_schedule", ()) or ())
    if schedule:
        return schedule
    entry_symbols = tuple(getattr(market_data, "entry_symbols", ()) or ())
    if not entry_symbols:
        return ()
    # This schedule is only an entry-eligibility filter for prepared
    # strategies.  ``market_index`` continues to use the original schedule, so
    # static-universe timeline semantics remain unchanged.
    return (
        {
            "effective_date": "1900-01-01",
            "symbols": list(entry_symbols),
        },
    )


def run_paper_once(config: AppConfig, state_path: str) -> tuple[OrderPlan, list[str]]:
    ensure_configured_data_ready(config)
    broker = PaperBroker(state_path=state_path, initial_cash=config.capital.initial_cash)
    account = broker.get_account()
    resolved = resolve_universe(
        config,
        held_symbols=account.positions,
        previously_managed=account.positions,
        mode="paper",
    )
    symbols = list(resolved.symbols if resolved.entries_allowed else resolved.exit_only_symbols)
    market_data = load_configured_market_data(config, symbols, resolved_universe=resolved)
    strategy_bars = _operational_strategy_bars(
        market_data.bars,
        symbols,
        account.positions,
    )
    plan = build_order_plan(
        config,
        strategy_bars,
        account,
        mode="paper",
        benchmark=market_data.benchmark,
        filter_benchmark=market_data.filter_benchmark,
    )
    execution_bars = market_data.execution_bars or market_data.bars
    fills = broker.execute_plan(plan, _latest_prices(execution_bars), config.costs.fee_rate, config.costs.slippage_rate)
    return plan, fills


def run_live_once(config: AppConfig, assume_yes: bool = False) -> tuple[OrderPlan, list[str]]:
    ensure_configured_data_ready(config)
    broker = TossBroker()
    account = broker.get_account(config.market)
    resolved = resolve_universe(
        config,
        held_symbols=account.positions,
        previously_managed=account.positions,
        mode="live",
    )
    symbols = list(resolved.symbols if resolved.entries_allowed else resolved.exit_only_symbols)
    market_data = load_configured_market_data(config, symbols, resolved_universe=resolved)
    strategy_bars = _operational_strategy_bars(
        market_data.bars,
        symbols,
        account.positions,
    )
    plan = build_order_plan(
        config,
        strategy_bars,
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


def _operational_strategy_bars(
    bars: dict[str, pd.DataFrame],
    requested_symbols,
    held_positions,
) -> dict[str, pd.DataFrame]:
    """Hide action-linked-only securities from paper/live entry decisions.

    A held successor remains visible so the strategy can issue its exit, while
    a newly loaded spin-off child cannot become a fresh buy candidate merely
    because the provider needed its prices for lifecycle accounting.
    """

    allowed = {str(symbol) for symbol in requested_symbols}
    allowed.update(str(symbol) for symbol in held_positions)
    return {symbol: frame for symbol, frame in bars.items() if symbol in allowed}


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
        close = _close_on(bars.get(symbol), timestamp)
        if close is None:
            raise RuntimeError(
                "Held position has no price for portfolio valuation: "
                f"{symbol}/{pd.Timestamp(timestamp).date()}"
            )
        value += position.quantity * close
    return value


def _close_on(df: pd.DataFrame | None, timestamp) -> float | None:
    if df is None or df.empty or "Close" not in df:
        return None
    try:
        value = df.loc[timestamp, "Close"]
    except TypeError:
        signal_date = pd.Timestamp(timestamp).date()
        matches = [pd.Timestamp(idx).date() == signal_date for idx in df.index]
        available = df.loc[matches, "Close"]
        if available.empty:
            return None
        value = available.iloc[-1]
    except KeyError:
        return None
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    return float(value)


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
