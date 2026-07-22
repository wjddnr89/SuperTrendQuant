from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from .brokers import TossBroker
from .config import AppConfig
from .data_cache import YahooStateCache
from .holdings import HoldingsStore
from .market_store.provider import ensure_configured_data_ready, load_configured_market_data
from .market_store.realtime import QuoteProvider, TossRealtimeQuoteProvider
from .notifications import TelegramNotifier
from .portfolio import AccountSnapshot, OrderIntent, OrderPlan, estimate_quantity
from .runtime import check_market_schedule, last_completed_bar_end
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

    def run_once(self, ignore_schedule: bool = False, assume_yes: bool = False) -> tuple[OrderPlan, list[str]]:
        session = check_market_schedule()
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
        if self.data_cache is None:
            ensure_configured_data_ready(config)
        account = self.broker.get_account(session_market)
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
            self._send_close_briefing(session_market, account, synced_holdings)
            return OrderPlan(config.strategy.name, "live", (), ("Close briefing sent.",)), []

        account_issue = self._managed_account_issue(config, account, managed_symbols)
        if account_issue is not None:
            plan = OrderPlan(config.strategy.name, "live", (), (account_issue,))
            return plan, ["Live strategy execution blocked by account safety check."]
        managed_account = self._managed_account(account, managed_symbols)

        market_now = datetime.now(session_timezone) if session_timezone is not None else datetime.now()
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
            gap = _daily_data_gap(symbols, market_data)
            if gap:
                return OrderPlan(config.strategy.name, "live", (), (gap,)), ["Live orders blocked by historical data gap."]
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
        if not bars:
            return OrderPlan(config.strategy.name, "live", (), ("No fresh market data.",)), []
        notes = data_notes + tuple([f"Skipped stale symbols: {', '.join(stale_symbols)}"] if stale_symbols else [])
        if not resolved_universe.entries_allowed:
            notes += (f"Universe refresh failed; new entries blocked: {resolved_universe.refresh_error}",)

        plan = build_order_plan(
            config,
            bars,
            managed_account,
            mode="live",
            benchmark=benchmark,
            filter_benchmark=filter_benchmark,
        )
        if notes:
            plan = OrderPlan(plan.strategy_name, plan.mode, plan.orders, plan.notes + notes)
        if not resolved_universe.entries_allowed:
            plan = OrderPlan(
                plan.strategy_name,
                plan.mode,
                tuple(order for order in plan.orders if order.side.lower() != "buy"),
                plan.notes,
            )
        plan = self._apply_live_guards(config, plan, managed_account, managed_symbols)
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
        accepted_sell_symbols: set[str] = set()
        for order in plan.orders:
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
                if qty <= 0:
                    results.append(f"SKIPPED BUY {order.symbol}: insufficient refreshed cash")
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
                order = replace(
                    order,
                    client_order_id=self._client_order_id(current_base, order),
                )
            ok = self.broker.place_order(order)
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
                if qty <= 0:
                    notes.append(f"Skipped buy {order.symbol}: insufficient cash.")
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

    def _safe_prices(self, symbols: list[str]) -> dict[str, float]:
        try:
            return {
                symbol: quote.price
                for symbol, quote in self.quote_provider.quotes(symbols).items()
            }
        except Exception as exc:
            print(f"Realtime price lookup failed: {exc}")
            return {}

    def _client_order_id(self, candle_base: datetime, order: OrderIntent) -> str:
        """Stable per-candle idempotency key accepted by the Toss API."""
        side = "b" if order.side.lower() == "buy" else "s"
        value = f"stq-{candle_base:%Y%m%d%H%M}-{side}-{order.symbol}"
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


def _daily_data_gap(symbols: list[str], market_data) -> str | None:
    missing = sorted(set(symbols) - set(market_data.bars))
    if missing:
        return f"Historical data gap; all strategy orders blocked: {', '.join(missing)}"
    if market_data.data_quality == "blocked":
        return "Historical data quality is blocked; all strategy orders blocked."
    completed = pd.Timestamp(market_data.completed_session).date()
    stale = sorted(
        symbol
        for symbol, frame in market_data.bars.items()
        if frame.empty or pd.Timestamp(frame.index[-1]).date() < completed
    )
    if stale:
        return f"Historical data is incomplete through {completed}; all orders blocked: {', '.join(stale)}"
    return None
