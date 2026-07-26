from __future__ import annotations

import asyncio
import re
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .brokers import BrokerOrderResult, TossBroker
from .config import AppConfig, load_data_store_config
from .data_cache import YahooStateCache
from .holdings import HoldingsStore
from .live_state import (
    DailyRiskStateStore,
    LiveOrderLedger,
    SignalPlanStore,
    build_signal_plan,
    order_from_dict,
    strategy_config_hash,
)
from .market_store.provider import (
    configured_release_identity_issue,
    ensure_configured_data_ready,
    load_configured_market_data,
)
from .market_store.realtime import QuoteProvider, TossRealtimeQuoteProvider
from .notifications import TelegramNotifier
from .portfolio import AccountSnapshot, OrderIntent, OrderPlan, estimate_quantity
from .runtime import check_market_schedule, daily_execution_window, last_completed_bar_end
from .strategies import build_order_plan
from .universe import resolve_universe


def _quantity_label(order: OrderIntent) -> str:
    if order.quantity is not None:
        return f"{order.quantity:g}"
    if order.cash_allocation_pct is not None:
        return f"cash:{order.cash_allocation_pct:.2%}"
    return "pending"


class HybridLiveRuntime:
    def __init__(
        self,
        config: AppConfig,
        broker: TossBroker | None = None,
        notifier: TelegramNotifier | None = None,
        holdings: HoldingsStore | None = None,
        data_cache: YahooStateCache | None = None,
        quote_provider: QuoteProvider | None = None,
        signal_plan_store: SignalPlanStore | None = None,
        order_ledger: LiveOrderLedger | None = None,
        risk_state_store: DailyRiskStateStore | None = None,
    ):
        self.config = config
        self.broker = broker or TossBroker()
        self.notifier = notifier or TelegramNotifier()
        self.holdings = holdings or HoldingsStore(config.live.holdings_file)
        self.quote_provider = quote_provider or TossRealtimeQuoteProvider(self.broker)
        self.data_cache = (
            data_cache
            if data_cache is not None
            else (YahooStateCache() if config.data_store.provider == "yahoo" else None)
        )
        self.last_briefing_date: dict[str, str | None] = {"KR": None, "US": None}
        self.last_candle_base_time: dict[str, datetime | None] = {"KR": None, "US": None}
        isolated_state_root = self.holdings.path.parent if holdings is not None else None

        def state_path(configured: str) -> Path:
            path = Path(configured)
            return (
                isolated_state_root / path.name
                if isolated_state_root is not None
                else path
            )

        self.signal_plan_store = signal_plan_store or SignalPlanStore(
            state_path(config.live.signal_plan_file)
        )
        self.order_ledger = order_ledger or LiveOrderLedger(
            state_path(config.live.order_ledger_file)
        )
        self.risk_state_store = risk_state_store or DailyRiskStateStore(
            state_path(config.live.risk_state_file)
        )
        self.kill_switch_path = state_path(config.live.kill_switch_file)

    def run_once(
        self,
        ignore_schedule: bool = False,
        assume_yes: bool = False,
        *,
        now: datetime | None = None,
    ) -> tuple[OrderPlan, list[str]]:
        session = (
            check_market_schedule(
                now_kr=now.astimezone(ZoneInfo("Asia/Seoul")),
                now_us=now.astimezone(ZoneInfo("America/New_York")),
            )
            if now is not None and now.tzinfo is not None
            else check_market_schedule()
        )
        if ignore_schedule:
            market = "US" if self.config.market == "AUTO" else self.config.market
            session_market = market if market in {"KR", "US"} else "US"
            is_close_briefing = False
            session_timezone = ZoneInfo("Asia/Seoul") if session_market == "KR" else ZoneInfo("America/New_York")
        else:
            if session.market is None:
                return OrderPlan(self.config.strategy.name, "live", (), ("Market is sleeping.",)), []
            if self.config.market != "AUTO" and session.market != self.config.market:
                return OrderPlan(self.config.strategy.name, "live", (), ("Configured market is closed.",)), []
            session_market = session.market
            is_close_briefing = session.is_close_briefing
            session_timezone = session.timezone

        config = replace(self.config, market=session_market)
        if self.config.market == "AUTO" and self.data_cache is None:
            config = replace(
                config,
                data_store=load_data_store_config(market=session_market),
            )
        if config.timeframe != "1d" or config.live.signal_bar_policy != "completed_daily":
            return (
                OrderPlan(
                    config.strategy.name,
                    "live",
                    (),
                    ("Live trading requires confirmed completed_daily 1d bars.",),
                ),
                ["Live orders blocked by signal-bar policy."],
            )
        market_now = (
            now.astimezone(session_timezone)
            if now is not None and now.tzinfo is not None
            else datetime.now(session_timezone)
            if session_timezone is not None
            else datetime.now()
        )
        execution_window = daily_execution_window(
            session_market,
            market_now,
            minutes=config.live.execution_window_minutes,
        )
        if (
            not ignore_schedule
            and not is_close_briefing
            and (execution_window is None or not execution_window.allowed)
        ):
            return (
                OrderPlan(
                    config.strategy.name,
                    "live",
                    (),
                    (
                        "Daily orders are allowed only during the first "
                        f"{config.live.execution_window_minutes} minutes after the exchange open.",
                    ),
                ),
                ["Live orders blocked outside the D+1 execution window."],
            )
        if not is_close_briefing and self.kill_switch_path.exists():
            return (
                OrderPlan(
                    config.strategy.name,
                    "live",
                    (),
                    (f"Kill switch is active: {self.kill_switch_path}",),
                ),
                ["Live orders blocked by kill switch."],
            )
        if self.data_cache is None:
            ensure_configured_data_ready(config)
            release_issue = configured_release_identity_issue(config)
            if release_issue is not None:
                return (
                    OrderPlan(
                        config.strategy.name,
                        "live",
                        (),
                        (release_issue,),
                    ),
                    ["Live orders blocked by local/R2 release mismatch."],
                )
        account = self.broker.get_account(session_market)
        try:
            open_orders = self.broker.list_open_orders()
        except Exception:
            open_orders = None
        if open_orders is not None:
            self._reconcile_order_ledger(account, open_orders)
        previous_members = self.holdings.member_map(session_market)
        resolved_universe = resolve_universe(
            config,
            market=session_market,
            held_symbols=account.positions,
            previously_managed=previous_members,
            mode="live",
        )
        managed_symbols = list(resolved_universe.symbols)
        if resolved_universe.entries_allowed:
            symbols = managed_symbols
        else:
            symbols = [
                symbol
                for symbol in account.positions
                if resolved_universe.member_for(symbol) is not None
            ]
        synced_holdings = self.holdings.sync_market(
            session_market,
            account,
            managed_symbols,
            resolved_universe.member_map,
        )

        if is_close_briefing:
            close_session = market_now.date().isoformat()
            self.risk_state_store.update_and_check(
                session=close_session,
                account=account,
                risk=config.risk,
            )
            self._send_close_briefing(session_market, account, synced_holdings)
            return OrderPlan(config.strategy.name, "live", (), ("Close briefing sent.",)), []

        account_issue = self._managed_account_issue(config, account, managed_symbols)
        if account_issue is not None:
            plan = OrderPlan(config.strategy.name, "live", (), (account_issue,))
            return plan, ["Live strategy execution blocked by account safety check."]
        managed_account = self._managed_account(account, managed_symbols)
        execution_session = (
            execution_window.execution_session
            if execution_window is not None
            else market_now.date().isoformat()
        )
        daily_loss_issue, _ = self.risk_state_store.update_and_check(
            session=execution_session,
            account=account,
            risk=config.risk,
        )

        data_notes: tuple[str, ...] = ()
        if self.data_cache is None:
            market_data = load_configured_market_data(
                config,
                symbols,
                resolved_universe=resolved_universe,
            )
            bars = market_data.bars
            benchmark = market_data.benchmark
            filter_benchmark = market_data.filter_benchmark
            stale_symbols = list(market_data.skipped)
            current_base = pd.Timestamp(market_data.completed_session).to_pydatetime()
            expected_signal_session = (
                execution_window.signal_session
                if execution_window is not None and not ignore_schedule
                else None
            )
            gap = _daily_data_gap(
                symbols,
                market_data,
                expected_signal_session=expected_signal_session,
                required_quality=config.live.required_data_quality,
                allowed_degraded_warning_codes=(
                    config.live.allowed_degraded_warning_codes
                ),
            )
            if gap:
                return OrderPlan(config.strategy.name, "live", (), (gap,)), ["Live orders blocked by historical data gap."]
            signal_session = str(market_data.completed_session)
            data_version = str(market_data.data_version)
            data_notes = tuple(market_data.warnings) + (
                f"Data version: {market_data.data_version}",
            )
        else:
            filter_timeframe = (
                config.market_trend_filter.timeframe
                if config.market_trend_filter.enabled
                else config.timeframe
            )
            if hasattr(self.data_cache, "configure"):
                self.data_cache.configure(config.timeframe, filter_timeframe, config.period)
            if hasattr(self.data_cache, "configure_universe"):
                self.data_cache.configure_universe(resolved_universe)
            current_base = last_completed_bar_end(market_now, session_market, config.timeframe)
            benchmarks = sorted(
                {resolved_universe.benchmark_for(symbol) for symbol in symbols}
            )
            if self.last_candle_base_time.get(session_market) != current_base:
                self.data_cache.sync(symbols, session_market, config.universe_file, benchmarks, current_candle_base=current_base)
                self.last_candle_base_time[session_market] = current_base
            self.data_cache.retry_missing(session_market, config.universe_file, session_timezone, current_base)
            bars, stale_symbols = self.data_cache.fresh_stock_bars(symbols, session_timezone, current_base)
            benchmark = self.data_cache.fresh_benchmark_map(
                symbols,
                session_market,
                config.universe_file,
                config.timeframe,
                session_timezone,
                current_base,
            )
            current_filter_base = last_completed_bar_end(market_now, session_market, filter_timeframe)
            filter_benchmark = self.data_cache.fresh_benchmark_map(
                symbols,
                session_market,
                config.universe_file,
                filter_timeframe,
                session_timezone,
                current_filter_base,
            )
            signal_session = current_base.date().isoformat()
            data_version = f"legacy-cache:{signal_session}"
        bars = _tail_frames(bars, config.live.history_window_bars)
        benchmark = _tail_benchmark(benchmark, config.live.history_window_bars)
        filter_benchmark = _tail_benchmark(
            filter_benchmark,
            config.live.history_window_bars,
        )
        if not bars:
            return OrderPlan(config.strategy.name, "live", (), ("No fresh market data.",)), []
        notes = data_notes + tuple([f"Skipped stale symbols: {', '.join(stale_symbols)}"] if stale_symbols else [])
        if not resolved_universe.entries_allowed:
            notes += (f"Universe refresh failed; new entries blocked: {resolved_universe.refresh_error}",)

        strategy_hash = strategy_config_hash(config)
        existing_plan = self.signal_plan_store.load()
        reuse_existing = bool(
            existing_plan
            and existing_plan.get("market") == session_market
            and existing_plan.get("execution_session") == execution_session
        )
        if reuse_existing:
            mismatched = [
                label
                for label, expected, actual in (
                    ("signal_session", signal_session, existing_plan.get("signal_session")),
                    ("data_version", data_version, existing_plan.get("data_version")),
                    ("strategy_hash", strategy_hash, existing_plan.get("strategy_hash")),
                )
                if str(expected) != str(actual)
            ]
            if mismatched:
                issue = (
                    "Durable plan mismatch for this execution session: "
                    + ", ".join(mismatched)
                )
                return OrderPlan(config.strategy.name, "live", (), (issue,)), [
                    "Live orders blocked by durable-plan mismatch."
                ]
            plan = OrderPlan(
                config.strategy.name,
                "live",
                tuple(order_from_dict(value) for value in existing_plan.get("orders", ())),
                notes + ("Reusing the durable signal plan for this session.",),
            )
            if daily_loss_issue:
                plan = OrderPlan(
                    plan.strategy_name,
                    plan.mode,
                    tuple(
                        order
                        for order in plan.orders
                        if order.side.lower() == "sell"
                    ),
                    plan.notes
                    + (daily_loss_issue + "; new entries are disabled.",),
                )
            plan = self._apply_live_guards(
                config,
                plan,
                managed_account,
                managed_symbols,
            )
        else:
            plan = build_order_plan(
                config,
                bars,
                managed_account,
                mode="live",
                benchmark=benchmark,
                filter_benchmark=filter_benchmark,
            )
            if notes:
                plan = OrderPlan(
                    plan.strategy_name,
                    plan.mode,
                    plan.orders,
                    plan.notes + notes,
                )
            if not resolved_universe.entries_allowed:
                plan = OrderPlan(
                    plan.strategy_name,
                    plan.mode,
                    tuple(
                        order
                        for order in plan.orders
                        if order.side.lower() != "buy"
                    ),
                    plan.notes,
                )
            if daily_loss_issue:
                plan = OrderPlan(
                    plan.strategy_name,
                    plan.mode,
                    tuple(
                        order
                        for order in plan.orders
                        if order.side.lower() == "sell"
                    ),
                    plan.notes
                    + (daily_loss_issue + "; new entries are disabled.",),
                )
            plan = self._apply_live_guards(
                config,
                plan,
                managed_account,
                managed_symbols,
            )
            plan = OrderPlan(
                plan.strategy_name,
                plan.mode,
                tuple(
                    order
                    if order.client_order_id
                    else replace(
                        order,
                        client_order_id=self._client_order_id(
                            signal_session,
                            order,
                            session_market,
                        ),
                    )
                    for order in plan.orders
                ),
                plan.notes,
            )
            expires_at = (
                execution_window.expires_at.isoformat()
                if execution_window is not None
                else (market_now + timedelta(minutes=config.live.execution_window_minutes)).isoformat()
            )
            durable_plan = build_signal_plan(
                config=config,
                market=session_market,
                signal_session=signal_session,
                execution_session=execution_session,
                expires_at=expires_at,
                data_version=data_version,
                orders=plan.orders,
                account=managed_account,
            )
            self.signal_plan_store.ensure(durable_plan)

        if not plan.orders:
            return plan, ["No live orders."]

        self._print_order_plan(plan)
        if config.execution.live_confirm_required and not assume_yes:
            answer = input("Type yes to send live orders: ").strip()
            if answer != "yes":
                return plan, ["Live orders were not sent."]

        results = []
        required_sell_symbols = {
            order.symbol for order in plan.orders if order.side.lower() == "sell"
        }
        latest_ledger = self.order_ledger.latest_by_client_id()
        accepted_statuses = {
            "accepted",
            "open",
            "partially_filled",
            "filled",
            "inferred_filled",
        }
        accepted_sell_symbols: set[str] = {
            order.symbol
            for order in plan.orders
            if order.side.lower() == "sell"
            and order.client_order_id
            and str(
                latest_ledger.get(order.client_order_id, {}).get("status") or ""
            ).lower()
            in accepted_statuses
        }
        for order in plan.orders:
            if order.client_order_id and self.order_ledger.already_submitted(
                order.client_order_id
            ):
                results.append(
                    f"SKIPPED {order.side.upper()} {order.symbol}: already recorded in order ledger"
                )
                continue
            if order.side.lower() == "buy":
                refreshed_account = self.broker.get_account(session_market)
                is_dependent_buy = bool(order.required_sell_symbols) or (
                    order.reason == "Post-sell leader entry"
                )
                dependencies = set(order.required_sell_symbols) or required_sell_symbols
                if is_dependent_buy:
                    if not dependencies or not dependencies.issubset(accepted_sell_symbols):
                        results.append(f"SKIPPED BUY {order.symbol}: prerequisite sell was not accepted")
                        continue
                    remaining = {
                        symbol
                        for symbol in dependencies
                        if (
                            (position := refreshed_account.positions.get(symbol)) is not None
                            and position.quantity > 0
                        )
                    }
                    if remaining:
                        results.append(
                            f"SKIPPED BUY {order.symbol}: prerequisite sell not filled ({', '.join(sorted(remaining))})"
                        )
                        continue
                refreshed_issue = self._managed_account_issue(config, refreshed_account, managed_symbols)
                if refreshed_issue is not None:
                    results.append(f"SKIPPED BUY {order.symbol}: {refreshed_issue}")
                    continue
                realtime_prices = self._safe_prices([order.symbol])
                current_price = realtime_prices.get(order.symbol)
                if current_price is None or pd.isna(current_price) or current_price <= 0:
                    results.append(f"SKIPPED BUY {order.symbol}: realtime quote unavailable")
                    continue
                allocation = (
                    order.cash_allocation_pct
                    if order.cash_allocation_pct is not None
                    else config.execution.allocation_pct
                    if is_dependent_buy
                    else 1.0
                )
                affordable_qty = estimate_quantity(
                    refreshed_account.cash,
                    current_price,
                    allocation,
                    fee_rate=config.costs.fee_rate,
                    slippage_rate=config.costs.slippage_rate,
                )
                qty = (
                    affordable_qty
                    if is_dependent_buy
                    else min(order.quantity, affordable_qty)
                    if order.quantity is not None
                    else 0
                )
                qty = self._limit_buy_notional(config, qty, current_price)
                if qty <= 0:
                    results.append(
                        f"SKIPPED BUY {order.symbol}: cash or max-order-notional limit"
                    )
                    continue
                order = OrderIntent(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=qty,
                    order_type=order.order_type,
                    price=order.price,
                    reason=order.reason,
                    client_order_id=order.client_order_id,
                    cash_allocation_pct=order.cash_allocation_pct,
                    required_sell_symbols=order.required_sell_symbols,
                )
            if order.client_order_id is None:
                raise RuntimeError("Durable live order has no client_order_id.")
            ledger_base = {
                "market": session_market,
                "signal_session": signal_session,
                "execution_session": execution_session,
                "data_version": data_version,
                "strategy_hash": strategy_hash,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "side": order.side.lower(),
                "quantity": order.quantity,
            }
            self.order_ledger.append({**ledger_base, "status": "submitting"})
            try:
                broker_result = self._place_order(order)
            except Exception as exc:
                self.order_ledger.append(
                    {
                        **ledger_base,
                        "status": "unknown",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )
                results.append(
                    f"UNKNOWN {order.side.upper()} {order.symbol}: broker request raised"
                )
                continue
            ok = broker_result.accepted
            self.order_ledger.append(
                {
                    **ledger_base,
                    "status": "accepted" if ok else "rejected",
                    "broker_order_id": broker_result.broker_order_id,
                    "broker_status": broker_result.status,
                    "detail": broker_result.detail,
                }
            )
            status = "SENT" if ok else "FAILED"
            results.append(
                f"{status} {order.side.upper()} {order.symbol} {_quantity_label(order)}"
            )
            if ok and order.side.lower() == "sell":
                accepted_sell_symbols.add(order.symbol)
            if ok:
                self.notifier.send(self._order_message(order))
                refreshed = self.broker.get_account(session_market)
                self.holdings.sync_market(
                    session_market,
                    refreshed,
                    managed_symbols,
                    resolved_universe.member_map,
                )
        return plan, results

    async def run_loop(self) -> None:
        self.notifier.send("*SuperTrendQuant live runtime started*")
        while True:
            try:
                self.run_once()
            except Exception as exc:
                print(f"Live runtime exception: {exc}")
            await asyncio.sleep(self.config.live.loop_interval_seconds)

    def _apply_live_guards(
        self,
        config: AppConfig,
        plan: OrderPlan,
        account: AccountSnapshot,
        symbols: list[str],
    ) -> OrderPlan:
        try:
            open_orders = self.broker.list_open_orders()
        except Exception as exc:
            return OrderPlan(plan.strategy_name, plan.mode, (), (f"Open order check failed: {exc}",))

        open_symbols = {
            order.get("symbol")
            for order in open_orders
            if order.get("symbol") and (order.get("side", "").lower() == "sell" or order.get("symbol") in symbols)
        }
        guarded_orders: list[OrderIntent] = []
        dependent_buys: list[OrderIntent] = []
        surviving_sell_symbols: set[str] = set()
        notes = list(plan.notes)
        realtime_prices = self._safe_prices(symbols)

        for order in plan.orders:
            side = order.side.lower()
            if side not in {"buy", "sell"}:
                notes.append(f"Skipped invalid live order side for {order.symbol}: {order.side}")
                continue
            if side == "buy" and (
                order.required_sell_symbols or order.reason == "Post-sell leader entry"
            ):
                dependent_buys.append(order)
                continue
            if order.symbol in open_symbols:
                notes.append(f"Skipped {order.symbol}: an open order already exists.")
                continue
            if side == "sell":
                position = account.positions.get(order.symbol)
                if position is None or position.quantity <= 0:
                    notes.append(
                        f"Skipped sell {order.symbol}: no managed position remains."
                    )
                    continue
                if order.quantity is None or order.quantity <= 0:
                    notes.append(
                        f"Skipped sell {order.symbol}: unresolved sell quantity."
                    )
                    continue
                order = replace(
                    order,
                    quantity=min(float(order.quantity), float(position.quantity)),
                )
            if side == "sell" and order.reason == "Leader rotation":
                economics = account.position_economics.get(order.symbol)
                profit_pct = economics.net_return_pct if economics is not None else None
                if profit_pct is None:
                    notes.append(
                        f"Skipped rotation sell {order.symbol}: economic ledger unavailable."
                    )
                    continue
                if profit_pct < config.leader_rotation.min_rotation_profit_pct:
                    notes.append(f"Skipped rotation sell {order.symbol}: minimum profit not met.")
                    continue
            if side == "buy":
                if order.quantity is None:
                    notes.append(f"Skipped buy {order.symbol}: unresolved cash allocation.")
                    continue
                current_price = realtime_prices.get(order.symbol)
                if current_price is None or pd.isna(current_price) or current_price <= 0:
                    notes.append(f"Skipped buy {order.symbol}: realtime quote unavailable.")
                    continue
                maximum_qty = estimate_quantity(
                    account.cash,
                    current_price,
                    config.execution.allocation_pct,
                    fee_rate=config.costs.fee_rate,
                    slippage_rate=config.costs.slippage_rate,
                )
                qty = min(order.quantity, maximum_qty)
                qty = self._limit_buy_notional(config, qty, current_price)
                if qty <= 0:
                    notes.append(
                        f"Skipped buy {order.symbol}: cash or max-order-notional limit."
                    )
                    continue
                order = OrderIntent(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=qty,
                    order_type=order.order_type,
                    price=order.price,
                    reason=order.reason,
                    client_order_id=order.client_order_id,
                    cash_allocation_pct=order.cash_allocation_pct,
                    required_sell_symbols=order.required_sell_symbols,
                )
            guarded_orders.append(order)
            if side == "sell":
                surviving_sell_symbols.add(order.symbol)

        for order in dependent_buys:
            if order.symbol in open_symbols:
                notes.append(f"Skipped {order.symbol}: an open order already exists.")
                continue
            if not surviving_sell_symbols:
                notes.append(f"Skipped dependent buy {order.symbol}: prerequisite sell was guarded out.")
                continue
            current_price = realtime_prices.get(order.symbol)
            if current_price is None or pd.isna(current_price) or current_price <= 0:
                notes.append(f"Skipped dependent buy {order.symbol}: realtime quote unavailable.")
                continue
            guarded_orders.append(order)

        return OrderPlan(plan.strategy_name, plan.mode, tuple(guarded_orders), tuple(notes))

    def _managed_account_issue(
        self,
        config: AppConfig,
        account: AccountSnapshot,
        symbols: list[str],
    ) -> str | None:
        universe = set(symbols)
        active_positions = {
            symbol: position
            for symbol, position in account.positions.items()
            if position.quantity > 0
        }
        unmanaged = sorted(set(active_positions) - universe)
        if unmanaged:
            return f"Unmanaged live holdings detected: {', '.join(unmanaged)}"
        managed_count = len(active_positions)
        if managed_count > config.risk.max_position_count:
            return (
                "Managed live position count exceeds risk limit: "
                f"{managed_count} > {config.risk.max_position_count}"
            )
        return None

    def _managed_account(
        self,
        account: AccountSnapshot,
        symbols: list[str],
    ) -> AccountSnapshot:
        universe = set(symbols)
        positions = {
            symbol: position
            for symbol, position in account.positions.items()
            if symbol in universe and position.quantity > 0
        }
        return AccountSnapshot(
            cash=account.cash,
            positions=positions,
            total_asset_value=account.total_asset_value,
            position_economics={
                symbol: economics
                for symbol, economics in account.position_economics.items()
                if symbol in positions
            },
        )

    def _limit_buy_notional(
        self,
        config: AppConfig,
        quantity: float,
        price: float,
    ) -> int:
        try:
            resolved = max(0, int(quantity))
            quote = float(price)
        except (TypeError, ValueError):
            return 0
        limit = float(config.risk.max_order_notional)
        if limit <= 0 or quote <= 0:
            return resolved
        unit_cost = (
            quote
            * (1.0 + config.costs.slippage_rate)
            * (1.0 + config.costs.fee_rate)
        )
        return min(resolved, int(limit // unit_cost))

    def _place_order(self, order: OrderIntent) -> BrokerOrderResult:
        detailed = getattr(self.broker, "place_order_detailed", None)
        if callable(detailed):
            result = detailed(order)
            if isinstance(result, BrokerOrderResult):
                return result
            if isinstance(result, bool):
                return BrokerOrderResult(
                    accepted=result,
                    status="accepted" if result else "rejected",
                )
            raise TypeError("place_order_detailed returned an unsupported result.")
        accepted = bool(self.broker.place_order(order))
        return BrokerOrderResult(
            accepted=accepted,
            status="accepted" if accepted else "rejected",
        )

    def _reconcile_order_ledger(
        self,
        account: AccountSnapshot,
        open_orders: list[dict],
    ) -> None:
        plan = self.signal_plan_store.load()
        if not plan:
            return
        latest = self.order_ledger.latest_by_client_id()
        open_by_client: dict[str, dict] = {}
        open_by_broker: dict[str, dict] = {}
        for raw in open_orders:
            if not isinstance(raw, dict):
                continue
            client_id = str(
                raw.get("clientOrderId")
                or raw.get("client_order_id")
                or raw.get("clientId")
                or ""
            )
            broker_id = str(raw.get("orderId") or raw.get("id") or "")
            if client_id:
                open_by_client[client_id] = raw
            if broker_id:
                open_by_broker[broker_id] = raw

        starting = {
            str(symbol): float(quantity)
            for symbol, quantity in (plan.get("starting_positions") or {}).items()
        }
        current = {
            symbol: float(position.quantity)
            for symbol, position in account.positions.items()
        }
        reconcilable = {
            "submitting",
            "accepted",
            "open",
            "partially_filled",
            "unknown",
        }
        for raw_order in plan.get("orders", ()):
            order = order_from_dict(raw_order)
            client_id = str(order.client_order_id or "")
            if not client_id:
                continue
            prior = latest.get(client_id)
            if not prior or str(prior.get("status") or "").lower() not in reconcilable:
                continue
            broker_id = str(prior.get("broker_order_id") or "")
            open_order = open_by_client.get(client_id) or open_by_broker.get(broker_id)
            status = ""
            reconciled_broker_id = broker_id
            filled_quantity = 0.0
            if open_order is not None:
                reconciled_broker_id = str(
                    open_order.get("orderId")
                    or open_order.get("id")
                    or broker_id
                )
                filled_quantity = _broker_filled_quantity(open_order)
                status = _broker_ledger_status(open_order)
                if status not in {"open", "partially_filled"}:
                    status = "partially_filled" if filled_quantity > 0 else "open"
            else:
                order_detail = None
                get_order = getattr(self.broker, "get_order", None)
                if broker_id and callable(get_order):
                    try:
                        order_detail = get_order(broker_id)
                    except Exception:
                        order_detail = None
                if isinstance(order_detail, dict):
                    status = _broker_ledger_status(order_detail)
                    filled_quantity = _broker_filled_quantity(order_detail)
                if not status:
                    before = starting.get(order.symbol, 0.0)
                    after = current.get(order.symbol, 0.0)
                    observed_fill = (
                        max(0.0, after - before)
                        if order.side.lower() == "buy"
                        else max(0.0, before - after)
                    )
                    planned = float(order.quantity or 0.0)
                    if observed_fill > 0:
                        filled_quantity = observed_fill
                        status = (
                            "inferred_filled"
                            if planned <= 0 or observed_fill + 1e-9 >= planned
                            else "partially_filled"
                        )
                    else:
                        status = "unknown"
            prior_status = str(prior.get("status") or "").lower()
            if (
                status == prior_status
                and reconciled_broker_id == broker_id
                and float(prior.get("filled_quantity") or 0) == filled_quantity
            ):
                continue
            self.order_ledger.append(
                {
                    "market": plan.get("market"),
                    "signal_session": plan.get("signal_session"),
                    "execution_session": plan.get("execution_session"),
                    "data_version": plan.get("data_version"),
                    "strategy_hash": plan.get("strategy_hash"),
                    "client_order_id": client_id,
                    "symbol": order.symbol,
                    "side": order.side.lower(),
                    "quantity": order.quantity,
                    "status": status,
                    "broker_order_id": reconciled_broker_id,
                    "filled_quantity": filled_quantity,
                    "reconciliation": "broker_open_orders_and_account",
                }
            )

    def _safe_prices(self, symbols: list[str]) -> dict[str, float]:
        try:
            return {
                symbol: quote.price
                for symbol, quote in self.quote_provider.quotes(symbols).items()
            }
        except Exception as exc:
            print(f"Realtime price lookup failed: {exc}")
            return {}

    def _client_order_id(
        self,
        signal_session: str,
        order: OrderIntent,
        market: str,
    ) -> str:
        """Stable D-signal idempotency key accepted by the Toss API."""
        side = "b" if order.side.lower() == "buy" else "s"
        session = str(signal_session).replace("-", "")[:8]
        safe_symbol = re.sub(r"[^a-zA-Z0-9_-]", "_", str(order.symbol))
        value = f"stq-{market.lower()}-{session}-{side}-{safe_symbol}"
        return value[:36]

    def _send_close_briefing(self, market: str, account: AccountSnapshot, holdings: dict[str, dict]) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        if self.last_briefing_date.get(market) == today:
            return
        positions = "\n• ".join(f"{symbol} ({raw['qty']}주)" for symbol, raw in holdings.items()) or "보유 없음"
        total = account.total_asset_value if account.total_asset_value is not None else account.cash
        title = "국내 주식" if market == "KR" else "해외 주식"
        self.notifier.send(f"🏁 *[{title} 마감]*\n• 총자산: {total:,.0f}\n• 포지션:\n• {positions}")
        self.last_briefing_date[market] = today

    def _print_order_plan(self, plan: OrderPlan) -> None:
        print("Live Order Plan")
        for order in plan.orders:
            print(
                f"{order.side.upper():4} {order.symbol:8} "
                f"qty={_quantity_label(order)} type={order.order_type} reason={order.reason}"
            )

    def _order_message(self, order: OrderIntent) -> str:
        if order.side.lower() == "buy":
            return f"🟩 *[추세 주도주 매수 주문 전송]*\n• 종목: {order.symbol} | 수량: {_quantity_label(order)}주"
        return f"🚨 *[매도 주문 전송]*\n• 종목: {order.symbol} | 수량: {_quantity_label(order)}주 | 사유: {order.reason}"


def _daily_data_gap(
    symbols: list[str],
    market_data,
    *,
    expected_signal_session: str | None = None,
    required_quality: str = "valid",
    allowed_degraded_warning_codes: tuple[str, ...] = (),
) -> str | None:
    missing = sorted(set(symbols) - set(market_data.bars))
    if missing:
        return f"Historical data gap; all strategy orders blocked: {', '.join(missing)}"
    quality = str(market_data.data_quality).lower()
    if quality == "blocked":
        return "Historical data quality is blocked; all strategy orders blocked."
    if quality == "degraded" and required_quality == "valid":
        allowed = {
            str(value).strip().lower()
            for value in allowed_degraded_warning_codes
            if str(value).strip()
        }
        warning_codes = {_warning_code(value) for value in market_data.warnings}
        if not warning_codes or not warning_codes.issubset(allowed):
            blocked = ", ".join(sorted(warning_codes - allowed)) or "unclassified"
            return (
                "Historical data is degraded with non-allowlisted warnings: "
                + blocked
            )
    completed = pd.Timestamp(market_data.completed_session).date()
    if (
        expected_signal_session is not None
        and completed.isoformat() != str(expected_signal_session)
    ):
        return (
            "Historical data release is not pinned to the required D signal "
            f"session: got {completed}, expected {expected_signal_session}"
        )
    stale = sorted(
        symbol
        for symbol, frame in market_data.bars.items()
        if frame.empty or pd.Timestamp(frame.index[-1]).date() < completed
    )
    if stale:
        return f"Historical data is incomplete through {completed}; all orders blocked: {', '.join(stale)}"
    return None


def _warning_code(value: str) -> str:
    text = str(value).strip().lower()
    if text.startswith("[") and "]" in text:
        return text[1 : text.index("]")].strip()
    return text.split(":", 1)[0].strip().replace(" ", "_")


def _broker_filled_quantity(order: dict) -> float:
    execution = order.get("execution", {})
    if not isinstance(execution, dict):
        execution = {}
    value = (
        execution.get("filledQuantity")
        or execution.get("filled_quantity")
        or order.get("filledQuantity")
        or order.get("filled_quantity")
        or 0
    )
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _broker_ledger_status(order: dict) -> str:
    value = str(order.get("status") or "").strip().upper()
    return {
        "PENDING": "open",
        "OPEN": "open",
        "PENDING_CANCEL": "open",
        "PENDING_REPLACE": "open",
        "PARTIAL_FILLED": "partially_filled",
        "FILLED": "filled",
        "CANCELED": "canceled",
        "REJECTED": "rejected",
        "REPLACED": "replaced",
        "CANCEL_REJECTED": "cancel_rejected",
        "REPLACE_REJECTED": "replace_rejected",
    }.get(value, "")


def _tail_frames(
    values: dict[str, pd.DataFrame],
    count: int,
) -> dict[str, pd.DataFrame]:
    return {
        symbol: frame.sort_index().tail(count).copy()
        for symbol, frame in values.items()
        if frame is not None and not frame.empty
    }


def _tail_benchmark(values, count: int):
    if isinstance(values, pd.DataFrame):
        return values.sort_index().tail(count).copy()
    if isinstance(values, dict):
        return _tail_frames(values, count)
    return values
