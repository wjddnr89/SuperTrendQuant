from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from .brokers import TossBroker
from .config import AppConfig, load_universe_for_market
from .data_cache import YahooStateCache
from .holdings import HoldingsStore
from .notifications import TelegramNotifier
from .portfolio import AccountSnapshot, OrderIntent, OrderPlan
from .runtime import check_market_schedule, current_30m_candle_base
from .strategies import build_order_plan


class HybridLiveRuntime:
    def __init__(
        self,
        config: AppConfig,
        broker: TossBroker | None = None,
        notifier: TelegramNotifier | None = None,
        holdings: HoldingsStore | None = None,
        data_cache: YahooStateCache | None = None,
    ):
        self.config = config
        self.broker = broker or TossBroker()
        self.notifier = notifier or TelegramNotifier()
        self.holdings = holdings or HoldingsStore(config.live.holdings_file)
        self.data_cache = data_cache or YahooStateCache()
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
            session_market = session.market
            is_close_briefing = session.is_close_briefing
            session_timezone = session.timezone

        config = replace(self.config, market=session_market)
        symbols = list(config.symbols) if config.symbols else load_universe_for_market(config.universe_file, session_market)
        account = self.broker.get_account(session_market)
        synced_holdings = self.holdings.sync_market(session_market, account, symbols)

        if is_close_briefing:
            self._send_close_briefing(session_market, account, synced_holdings)
            return OrderPlan(config.strategy.name, "live", (), ("Close briefing sent.",)), []

        market_now = datetime.now(session_timezone) if session_timezone is not None else datetime.now()
        current_base = current_30m_candle_base(market_now)
        benchmarks = ["^KS11", "^KQ11", "QQQ"]
        if self.last_candle_base_time.get(session_market) != current_base:
            self.data_cache.sync(symbols, session_market, config.universe_file, benchmarks, current_candle_base=current_base)
            self.last_candle_base_time[session_market] = current_base
        self.data_cache.retry_missing(session_market, config.universe_file, session_timezone, current_base)
        bars, stale_symbols = self.data_cache.fresh_stock_bars(symbols, session_timezone, current_base)
        benchmark = self.data_cache.fresh_benchmark_map(
            symbols,
            session_market,
            config.universe_file,
            "30m",
            session_timezone,
            current_base,
        )
        current_1h_base = market_now.replace(minute=0, second=0, microsecond=0)
        filter_benchmark = self.data_cache.fresh_benchmark_map(
            symbols,
            session_market,
            config.universe_file,
            "1h",
            session_timezone,
            current_1h_base,
        )
        if not bars:
            return OrderPlan(config.strategy.name, "live", (), ("No fresh market data.",)), []
        notes = tuple([f"Skipped stale symbols: {', '.join(stale_symbols)}"] if stale_symbols else [])

        plan = build_order_plan(
            config,
            bars,
            account,
            mode="live",
            benchmark=benchmark,
            filter_benchmark=filter_benchmark,
        )
        if notes:
            plan = OrderPlan(plan.strategy_name, plan.mode, plan.orders, plan.notes + notes)
        plan = self._apply_live_guards(config, plan, account, symbols)
        if not plan.orders:
            return plan, ["No live orders."]

        self._print_order_plan(plan)
        if config.execution.live_confirm_required and not assume_yes:
            answer = input("Type yes to send live orders: ").strip()
            if answer != "yes":
                return plan, ["Live orders were not sent."]

        results = []
        for order in plan.orders:
            if order.side.lower() == "buy":
                refreshed_account = self.broker.get_account(session_market)
                realtime_prices = self._safe_prices([order.symbol])
                current_price = realtime_prices.get(order.symbol)
                if current_price is not None and current_price > 0:
                    qty = int((refreshed_account.cash * config.execution.allocation_pct) // current_price)
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
                    )
            ok = self.broker.place_order(order)
            status = "SENT" if ok else "FAILED"
            results.append(f"{status} {order.side.upper()} {order.symbol} {order.quantity:g}")
            if ok:
                self.notifier.send(self._order_message(order))
                refreshed = self.broker.get_account(session_market)
                self.holdings.sync_market(session_market, refreshed, symbols)
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
        guarded_orders = []
        realtime_prices = self._safe_prices(symbols)
        has_sell_order = any(order.side.lower() == "sell" for order in plan.orders)

        for order in plan.orders:
            if order.symbol in open_symbols:
                continue
            if order.side.lower() == "sell" and order.reason == "Leader rotation":
                position = account.positions.get(order.symbol)
                current_price = realtime_prices.get(order.symbol)
                if position and position.avg_price > 0 and current_price is not None:
                    profit_pct = (current_price - position.avg_price) / position.avg_price
                    if profit_pct < config.leader_rotation.min_rotation_profit_pct:
                        continue
            if order.side.lower() == "buy" and not has_sell_order:
                current_price = realtime_prices.get(order.symbol)
                if current_price is not None and current_price > 0:
                    qty = int((account.cash * config.execution.allocation_pct) // current_price)
                    if qty <= 0:
                        continue
                    order = OrderIntent(
                        symbol=order.symbol,
                        side=order.side,
                        quantity=qty,
                        order_type=order.order_type,
                        price=order.price,
                        reason=order.reason,
                    )
            guarded_orders.append(order)

        return OrderPlan(plan.strategy_name, plan.mode, tuple(guarded_orders), plan.notes)

    def _safe_prices(self, symbols: list[str]) -> dict[str, float]:
        try:
            return self.broker.get_prices(symbols)
        except Exception as exc:
            print(f"Realtime price lookup failed: {exc}")
            return {}

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
            print(f"{order.side.upper():4} {order.symbol:8} qty={order.quantity:g} type={order.order_type} reason={order.reason}")

    def _order_message(self, order: OrderIntent) -> str:
        if order.side.lower() == "buy":
            return f"🟩 *[추세 주도주 매수 주문 전송]*\n• 종목: {order.symbol} | 수량: {order.quantity:g}주"
        return f"🚨 *[매도 주문 전송]*\n• 종목: {order.symbol} | 수량: {order.quantity:g}주 | 사유: {order.reason}"
