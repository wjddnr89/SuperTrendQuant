from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

from .brokers import PaperBroker
from .config import AppConfig
from .data_cache import YahooStateCache
from .live_runtime import _daily_data_gap
from .market_store.provider import ensure_configured_data_ready, load_configured_market_data
from .market_store.realtime import FrameQuoteProvider, QuoteProvider
from .portfolio import OrderPlan
from .results import PaperRunRecorder
from .runtime import check_market_schedule, last_completed_bar_end
from .strategies import build_order_plan
from .universe import resolve_universe


class PaperRuntime:
    def __init__(
        self,
        config: AppConfig,
        state_path: str | None = None,
        broker: PaperBroker | None = None,
        data_cache: YahooStateCache | None = None,
        recorder: PaperRunRecorder | None = None,
        quote_provider: QuoteProvider | None = None,
    ):
        self.config = config
        self.state_path = state_path or config.paper.state_file
        self.broker = broker or PaperBroker(self.state_path, initial_cash=config.capital.initial_cash)
        self.data_cache = (
            data_cache
            if data_cache is not None
            else (YahooStateCache() if config.data_store.provider == "yahoo" else None)
        )
        self.recorder = recorder or PaperRunRecorder(config.paper.results_dir, config.strategy.name)
        self.quote_provider = quote_provider
        self.recorder.write_metadata(config)
        self.last_candle_base_time: dict[str, datetime | None] = {"KR": None, "US": None}

    def run_once(self, ignore_schedule: bool = False) -> tuple[OrderPlan, list[str]]:
        resolved = self._resolve_market(ignore_schedule)
        if resolved is None:
            plan = OrderPlan(self.config.strategy.name, "paper", (), ("Market is sleeping.",))
            return plan, ["No open market for paper cycle."]
        session_market, session_timezone = resolved

        config = replace(self.config, market=session_market)
        if self.data_cache is None:
            ensure_configured_data_ready(config)
        account_before = self.broker.get_account()
        resolved_universe = resolve_universe(
            config,
            market=session_market,
            held_symbols=account_before.positions,
            previously_managed=account_before.positions,
            mode="paper",
        )
        if resolved_universe.entries_allowed:
            symbols = list(resolved_universe.symbols)
        else:
            symbols = [
                symbol
                for symbol in account_before.positions
                if resolved_universe.member_for(symbol) is not None
            ]
        self.recorder.write_universe_snapshot(resolved_universe.snapshot.to_dict())
        market_now = datetime.now(session_timezone)
        current_base = last_completed_bar_end(market_now, session_market, config.timeframe)
        candle_key = f"last_candle_base:{session_market}"

        data_notes: tuple[str, ...] = ()
        execution_bars = None
        if self.data_cache is None:
            market_data = load_configured_market_data(
                config,
                symbols,
                resolved_universe=resolved_universe,
            )
            gap = _daily_data_gap(symbols, market_data)
            if gap:
                plan = OrderPlan(config.strategy.name, "paper", (), (gap,))
                return plan, ["Paper orders blocked by historical data gap."]
            bars = market_data.bars
            execution_bars = market_data.execution_bars or market_data.bars
            benchmark = market_data.benchmark
            filter_benchmark = market_data.filter_benchmark
            stale_symbols = list(market_data.skipped)
            current_base = datetime.fromisoformat(market_data.completed_session)
            action_notes = self.broker.apply_corporate_actions(
                market_data.corporate_actions,
                through=market_data.completed_session,
                dividend_tax_rate=config.data_store.dividend_tax_rate,
            )
            account_before = self.broker.get_account()
            data_notes = tuple(market_data.warnings) + tuple(action_notes) + (
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
        candle_value = current_base.isoformat()
        if config.paper.run_once_per_candle and self.broker.get_metadata(candle_key) == candle_value:
            plan = OrderPlan(config.strategy.name, "paper", (), (f"Candle already processed: {candle_value}",))
            return plan, ["No paper orders."]
        if not bars:
            plan = OrderPlan(config.strategy.name, "paper", (), ("No fresh market data.",))
            return plan, ["No paper orders."]

        quote_provider = self.quote_provider or FrameQuoteProvider(execution_bars or bars)
        quotes = quote_provider.quotes(symbols)
        missing_quotes = sorted(set(symbols) - set(quotes))
        if missing_quotes:
            plan = OrderPlan(
                config.strategy.name,
                "paper",
                (),
                (
                    "Paper quote gap; all strategy orders blocked: "
                    + ", ".join(missing_quotes),
                ),
            )
            return plan, ["Paper orders blocked by quote gap."]
        prices = {symbol: quote.price for symbol, quote in quotes.items()}
        plan = build_order_plan(
            config,
            bars,
            account_before,
            mode="paper",
            benchmark=benchmark,
            filter_benchmark=filter_benchmark,
        )
        if stale_symbols:
            plan = OrderPlan(
                plan.strategy_name,
                plan.mode,
                plan.orders,
                plan.notes + (f"Skipped stale symbols: {', '.join(stale_symbols)}",),
            )
        if data_notes:
            plan = OrderPlan(plan.strategy_name, plan.mode, plan.orders, plan.notes + data_notes)
        if not resolved_universe.entries_allowed:
            plan = OrderPlan(
                plan.strategy_name,
                plan.mode,
                tuple(order for order in plan.orders if order.side.lower() != "buy"),
                plan.notes + (
                    f"Universe refresh failed; new entries blocked: {resolved_universe.refresh_error}",
                ),
            )
        fills = self.broker.execute_plan(
            plan,
            prices,
            config.costs.fee_rate,
            config.costs.slippage_rate,
            metadata_updates={candle_key: candle_value},
        )
        account_after = self.broker.get_account()
        self.recorder.record_cycle(
            timestamp=market_now,
            market=session_market,
            candle_base=current_base,
            plan=plan,
            fills=fills,
            account_before=account_before,
            account_after=account_after,
            prices=prices,
        )
        return plan, fills or ["No paper orders."]

    async def run_loop(self, ignore_schedule: bool = False) -> None:
        while True:
            try:
                plan, results = self.run_once(ignore_schedule=ignore_schedule)
                print(f"Paper Order Plan: {len(plan.orders)} orders")
                for note in plan.notes:
                    print(note)
                for result in results:
                    print(result)
            except Exception as exc:
                print(f"Paper runtime exception: {exc}")
            await asyncio.sleep(self.config.paper.loop_interval_seconds)

    def _resolve_market(self, ignore_schedule: bool) -> tuple[str, ZoneInfo] | None:
        if ignore_schedule:
            market = "US" if self.config.market == "AUTO" else self.config.market
            session_market = market if market in {"KR", "US"} else "US"
            session_timezone = ZoneInfo("Asia/Seoul") if session_market == "KR" else ZoneInfo("America/New_York")
            return session_market, session_timezone

        session = check_market_schedule()
        if session.market is None or session.timezone is None:
            return None
        if self.config.market != "AUTO" and session.market != self.config.market:
            return None
        return session.market, session.timezone
