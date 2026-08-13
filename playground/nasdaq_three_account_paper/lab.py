from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from bootstrap import PROJECT_ROOT, configure_imports, resolve_lab_path
from reports import PerformanceReporter, daily_row
from toss_data import (
    SessionOpenCache,
    TossDailyCache,
    TossIntradayCache,
    TossMarketDataClient,
    YahooHourlySeedCache,
    close_on,
    latest_completed_signal_date,
    truncate_daily,
)
from intraday_exit import (
    IntradayReplayExit,
    build_intraday_exit_signal,
    replay_intraday_fence_exit,
)


configure_imports()

from research_extensions.experimental_leader_rotation import (  # noqa: E402
    ExperimentalLeaderPolicy,
    ExperimentalPreparedLeaderBacktest,
)
from supertrend_quant.brokers import PaperBroker  # noqa: E402
from supertrend_quant.config import load_split_config  # noqa: E402
from supertrend_quant.portfolio import (  # noqa: E402
    OrderIntent,
    OrderPlan,
    estimate_quantity,
    mark_position_economics,
)
from supertrend_quant.strategies import create_strategy  # noqa: E402


class ThreeAccountPaperLab:
    def __init__(
        self,
        config_path: str | Path,
        *,
        client: TossMarketDataClient | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.experiment = dict(self.raw["experiment"])
        self.storage = dict(self.raw["storage"])
        self.client = client or TossMarketDataClient()
        self.state_dir = resolve_lab_path(self.storage["state_dir"])
        self.data_dir = resolve_lab_path(self.storage["data_dir"])
        self.results_dir = resolve_lab_path(self.storage["results_dir"])
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.daily_cache = TossDailyCache(
            self.client,
            self.data_dir,
            history_bars=int(self.experiment["signal_history_bars"]),
            refresh_bars=int(self.experiment["daily_refresh_bars"]),
        )
        self.open_cache = SessionOpenCache(
            self.client,
            self.data_dir / "session_opens.csv",
            timezone=str(self.experiment["timezone"]),
            regular_open=str(self.experiment["regular_open"]),
            regular_close=str(self.experiment["regular_close"]),
            max_pages=int(self.experiment["minute_max_pages"]),
        )
        self.intraday_cache = TossIntradayCache(
            self.client,
            self.data_dir,
            history_minutes=int(
                self.experiment.get("intraday_history_minutes", 6000)
            ),
            refresh_minutes=int(
                self.experiment.get("intraday_refresh_minutes", 800)
            ),
            max_pages=int(self.experiment["minute_max_pages"]),
            timezone=str(self.experiment["timezone"]),
        )
        self.intraday_seed_cache = YahooHourlySeedCache(
            self.data_dir,
            period=str(self.experiment.get("intraday_seed_period", "60d")),
            timezone=str(self.experiment["timezone"]),
        )
        self.reporter = PerformanceReporter(
            self.results_dir,
            initial_cash=float(self.experiment["initial_cash"]),
        )
        self.base_config = load_split_config(
            resolve_lab_path(self.raw["strategy"]["strategy_file"]),
            resolve_lab_path(self.raw["strategy"]["runtime_file"]),
        )
        self.brokers = {
            account_id: PaperBroker(
                self.state_dir / "accounts" / f"{account_id}.json",
                initial_cash=float(self.experiment["initial_cash"]),
            )
            for account_id in self.raw["accounts"]
        }

    def run_daily(self, execution_date: date | None = None) -> pd.DataFrame:
        market_tz = ZoneInfo(str(self.experiment["timezone"]))
        market_now = datetime.now(market_tz)
        requested_date = execution_date or latest_closed_session_cutoff(
            market_now,
            regular_close=str(self.experiment["regular_close"]),
        )
        benchmark_symbol = str(self.experiment["benchmark"])

        print(f"[paper] requested execution date={requested_date}", flush=True)
        benchmark_adjusted = self.daily_cache.load_symbol(
            benchmark_symbol, adjusted=True
        )
        benchmark_raw = self.daily_cache.load_symbol(
            benchmark_symbol, adjusted=False
        )
        execution_dates = self._execution_dates(
            benchmark_raw,
            requested_date=requested_date,
        )
        if not execution_dates:
            print("[paper] all accounts already processed.", flush=True)
            self.reporter.generate_weekly()
            return self.reporter.load_daily()

        universe_symbols = sorted(
            {
                symbol
                for session_date in execution_dates
                for symbol in self._universe_on(session_date)
            }
            | {
                symbol
                for broker in self.brokers.values()
                for symbol in broker.get_account().positions
            }
        )
        held_symbols = {
            symbol
            for broker in self.brokers.values()
            for symbol in broker.get_account().positions
        }
        adjusted, raw = self.daily_cache.load_universe(
            universe_symbols,
            raw_symbols=held_symbols,
        )

        all_rows: list[dict[str, Any]] = []
        for session_date in execution_dates:
            benchmark_open = self.open_cache.price(
                benchmark_symbol, session_date
            )
            if benchmark_open is None:
                raise RuntimeError(
                    f"No first regular-session minute for {benchmark_symbol} "
                    f"on {session_date}. Run after the US regular session opens."
                )
            signal_date = latest_completed_signal_date(
                benchmark_raw,
                session_date,
            )
            print(
                f"[paper] execution={session_date} signal={signal_date}",
                flush=True,
            )
            active_symbols = set(self._universe_on(session_date))
            account_held = {
                symbol
                for broker in self.brokers.values()
                for symbol in broker.get_account().positions
            }
            eligible_symbols = sorted(active_symbols | account_held)
            bars = self._signal_frames(
                adjusted,
                eligible_symbols,
                signal_date,
                account_held,
            )
            signal_benchmark = truncate_daily(
                benchmark_adjusted,
                signal_date,
            )
            raw_signal_prices = {
                symbol: price
                for symbol in eligible_symbols
                if (price := close_on(raw.get(symbol, pd.DataFrame()), signal_date))
                is not None
            }
            rows = self._run_accounts(
                execution_date=session_date,
                signal_date=signal_date,
                bars=bars,
                benchmark=signal_benchmark,
                raw_bars=raw,
                raw_signal_prices=raw_signal_prices,
            )
            all_rows.extend(rows)

        return self.reporter.record_daily(all_rows)

    def generate_weekly(self) -> pd.DataFrame:
        return self.reporter.generate_weekly()

    def _execution_dates(
        self,
        benchmark_raw: pd.DataFrame,
        *,
        requested_date: date,
    ) -> list[date]:
        known = sorted(
            {
                timestamp.date()
                for timestamp in pd.DatetimeIndex(benchmark_raw.index)
                if timestamp.date() <= requested_date
            }
        )
        if not known:
            return []
        last_dates = {
            account_id: _date_or_none(
                broker.get_metadata("last_execution_date")
            )
            for account_id, broker in self.brokers.items()
        }
        if all(value is None for value in last_dates.values()):
            return [known[-1]]
        earliest = min(
            value for value in last_dates.values() if value is not None
        )
        return [value for value in known if earliest < value <= requested_date]

    def _run_accounts(
        self,
        *,
        execution_date: date,
        signal_date: date,
        bars: dict[str, pd.DataFrame],
        benchmark: pd.DataFrame,
        raw_bars: dict[str, pd.DataFrame],
        raw_signal_prices: dict[str, float],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for account_id, account_raw in self.raw["accounts"].items():
            broker = self.brokers[account_id]
            last_date = _date_or_none(
                broker.get_metadata("last_execution_date")
            )
            if last_date is not None and last_date >= execution_date:
                continue
            print(
                f"[account {account_id}] {account_raw['name']}",
                flush=True,
            )
            account = broker.get_account()
            missing_held = sorted(
                set(account.positions) - set(raw_signal_prices)
            )
            if missing_held:
                raise RuntimeError(
                    f"Account {account_id} held symbols lack raw signal closes: "
                    + ", ".join(missing_held)
                )
            marked = mark_position_economics(
                account,
                raw_signal_prices,
                fee_rate=float(self.experiment["fee_rate"]),
                slippage_rate=float(self.experiment["slippage_rate"]),
            )
            config, policy = self._account_strategy(account_id, account_raw)
            intraday_blocks = self._release_intraday_reentry_blocks(
                account_id=account_id,
                account_raw=account_raw,
                signal_date=signal_date,
            )
            resize_notes: tuple[str, ...] = ()
            prepared = create_strategy(config).prepare_backtest(
                bars,
                benchmark=benchmark,
                filter_benchmark=benchmark,
            )
            if intraday_blocks:
                prepared = replace(
                    prepared,
                    prepared={
                        symbol: frame
                        for symbol, frame in prepared.prepared.items()
                        if symbol not in intraday_blocks
                        or symbol in account.positions
                    },
                )
            entry_gate_notes: list[str] = []
            while True:
                experimental = ExperimentalPreparedLeaderBacktest(prepared, policy)
                experimental.blocked_symbols = set(
                    broker.get_metadata("blocked_symbols", [])
                )
                plan = experimental.build_order_plan(
                    pd.Timestamp(signal_date),
                    marked,
                    mode="paper",
                )
                buy_symbols = {
                    order.symbol
                    for order in plan.orders
                    if str(order.side).lower() == "buy"
                }
                if not bool(account_raw.get("entry_2h_safety_gate", False)):
                    break
                rejected = {
                    symbol
                    for symbol in buy_symbols
                    if not self._two_hour_entry_is_bullish(
                        account_id=account_id,
                        account_raw=account_raw,
                        symbol=symbol,
                        signal_date=signal_date,
                    )
                }
                if not rejected:
                    break
                entry_gate_notes.extend(
                    f"Account {account_id} entry safety veto: {symbol} latest completed 2h trend is not bullish."
                    for symbol in sorted(rejected)
                )
                prepared = replace(
                    prepared,
                    prepared={
                        symbol: frame
                        for symbol, frame in prepared.prepared.items()
                        if symbol not in rejected
                    },
                )
            pending = broker.get_metadata("pending_intraday_exit", None)
            forced_pending = (
                str(account_raw.get("exit_timeframe", "1d")).lower() == "2h"
                and isinstance(pending, dict)
                and str(pending.get("symbol", "")) in account.positions
            )
            if forced_pending:
                plan = force_pending_intraday_exit(plan, account, pending)

            atr_resize_notes: tuple[str, ...] = ()
            entry_atr_risk_pct = account_raw.get("entry_atr_risk_pct")
            if entry_atr_risk_pct is not None:
                plan, atr_resize_notes = resize_buys_to_atr_risk(
                    plan,
                    prepared.prepared,
                    pd.Timestamp(signal_date),
                    marked,
                    target_atr_risk_pct=float(entry_atr_risk_pct),
                    fee_rate=float(self.experiment["fee_rate"]),
                    slippage_rate=float(self.experiment["slippage_rate"]),
                )

            price_symbols = set(account.positions) | {
                order.symbol for order in plan.orders
            }
            execution_prices: dict[str, float] = {}
            for symbol in sorted(price_symbols):
                price = self.open_cache.price(symbol, execution_date)
                if price is None:
                    raise RuntimeError(
                        f"Account {account_id} lacks first-minute price for "
                        f"{symbol} on {execution_date}."
                    )
                execution_prices[symbol] = price

            plan, resize_notes = resize_explicit_buys_to_execution_cash(
                plan,
                marked,
                execution_prices,
                fee_rate=float(self.experiment["fee_rate"]),
                slippage_rate=float(self.experiment["slippage_rate"]),
            )
            replay_candidates = projected_position_symbols(account, plan)
            replays = self._intraday_session_replays(
                account_id=account_id,
                account_raw=account_raw,
                symbols=replay_candidates,
                execution_date=execution_date,
            )
            metadata_updates: dict[str, object] = {
                "last_execution_date": execution_date.isoformat(),
                "last_signal_date": signal_date.isoformat(),
                "blocked_symbols": sorted(experimental.blocked_symbols),
                "strategy_account": account_id,
                "strategy_policy": dict(account_raw),
                "exit_timeframe": str(account_raw.get("exit_timeframe", "1d")),
                "intraday_exit_replay": {},
            }
            if forced_pending:
                metadata_updates["pending_intraday_exit"] = None
                updated_blocks = dict(intraday_blocks)
                pending_symbol = str(pending["symbol"])
                updated_blocks[pending_symbol] = dict(pending)
                metadata_updates["intraday_reentry_blocks"] = updated_blocks
            fills = broker.execute_plan(
                plan,
                execution_prices,
                float(self.experiment["fee_rate"]),
                float(self.experiment["slippage_rate"]),
                metadata_updates=metadata_updates,
            )
            replay = self._apply_intraday_replay(
                account_id=account_id,
                account_raw=account_raw,
                execution_date=execution_date,
                plan=plan,
                fills=fills,
                replays=replays,
            )
            plan = replay[0]
            fills = replay[1]
            account_after = broker.get_account()
            mark_prices: dict[str, float] = {}
            for symbol in account_after.positions:
                raw_frame = raw_bars.get(symbol, pd.DataFrame())
                price = close_on(raw_frame, execution_date)
                if price is None:
                    raw_frame = self.daily_cache.load_symbol(
                        symbol,
                        adjusted=False,
                    )
                    raw_bars[symbol] = raw_frame
                    price = close_on(raw_frame, execution_date)
                if price is None:
                    raise RuntimeError(
                        f"Missing closing mark for {symbol} on {execution_date}."
                    )
                mark_prices[symbol] = price

            intraday_notes = list(replay[2])
            notes = [
                *plan.notes,
                *resize_notes,
                *atr_resize_notes,
                *entry_gate_notes,
                *intraday_notes,
            ]
            status = "processed"
            row = daily_row(
                execution_date=execution_date,
                signal_date=signal_date,
                account_id=account_id,
                account_name=str(account_raw["name"]),
                hypothesis=str(account_raw["hypothesis"]),
                status=status,
                account=account_after,
                mark_prices=mark_prices,
                plan=plan,
                fills=fills,
                notes=notes,
            )
            rows.append(row)
            self._append_event(
                account_id,
                {
                    "execution_date": execution_date.isoformat(),
                    "signal_date": signal_date.isoformat(),
                    "plan": json.loads(row["orders"]),
                    "fills": fills,
                    "equity": row["equity"],
                    "cash": row["cash"],
                    "position": row["position_symbol"],
                    "notes": notes,
                    "intraday_exit_replay": replay[3],
                },
            )
            print(
                f"[account {account_id}] equity=${row['equity']:,.2f} "
                f"position={row['position_symbol'] or '-'} "
                f"fills={row['fill_count']}",
                flush=True,
            )
        return rows

    def _account_strategy(
        self,
        account_id: str,
        account_raw: dict[str, Any],
    ):
        gate = str(account_raw["rotation_profit_gate"])
        minimum_profit = 0.0 if gate == "nonnegative" else None
        market_filter = str(
            account_raw.get(
                "market_filter",
                self.raw["strategy"].get("market_filter", "1d"),
            )
        ).lower()
        if market_filter not in {"none", "1d"}:
            raise ValueError(
                f"Account {account_id} has unsupported market_filter: "
                f"{market_filter}"
            )
        config = replace(
            self.base_config,
            strategy=replace(
                self.base_config.strategy,
                name=f"nasdaq_paper_{account_id}",
            ),
            costs=replace(
                self.base_config.costs,
                fee_rate=float(self.experiment["fee_rate"]),
                slippage_rate=float(self.experiment["slippage_rate"]),
            ),
            capital=replace(
                self.base_config.capital,
                initial_cash=float(self.experiment["initial_cash"]),
            ),
            market_trend_filter=replace(
                self.base_config.market_trend_filter,
                enabled=market_filter != "none",
                timeframe=(
                    self.base_config.market_trend_filter.timeframe
                    if market_filter == "none"
                    else market_filter
                ),
            ),
            execution=replace(
                self.base_config.execution,
                allocation_pct=1.0,
                broker="paper",
            ),
            leader_rotation=replace(
                self.base_config.leader_rotation,
                min_rotation_profit_pct=minimum_profit,
                allow_late_chase=(
                    str(account_raw["late_chase_mode"]) != "fresh_only"
                ),
            ),
        )
        policy = ExperimentalLeaderPolicy(
            rotation_profit_gate=gate,
            stop_loss_pct=account_raw.get("stop_loss_pct"),
            late_chase_mode=str(account_raw["late_chase_mode"]),
            max_extension_atr=account_raw.get("max_extension_atr"),
        )
        return config, policy

    def _two_hour_entry_is_bullish(
        self,
        *,
        account_id: str,
        account_raw: dict[str, Any],
        symbol: str,
        signal_date: date,
    ) -> bool:
        try:
            adjusted = self.intraday_cache.refresh_symbol(
                symbol,
                latest_minutes=int(
                    self.experiment.get("intraday_refresh_minutes", 800)
                ),
                adjusted=True,
            )
            seed = self.intraday_seed_cache.load_symbol(symbol)
            signal = build_intraday_exit_signal(
                symbol,
                adjusted,
                seed_bars=seed,
                through=signal_date,
                period=int(account_raw["exit_supertrend_period"]),
                multiplier=float(account_raw["exit_supertrend_multiplier"]),
                timeframe_minutes=120,
                timezone=str(self.experiment["timezone"]),
                regular_open=str(self.experiment["regular_open"]),
                regular_close=str(self.experiment["regular_close"]),
            )
            return signal.trend == 1
        except (RuntimeError, ValueError) as exc:
            print(
                f"[account {account_id}] veto {symbol} entry because its "
                f"completed 2h state is unavailable: {exc}",
                flush=True,
            )
            return False

    def _release_intraday_reentry_blocks(
        self,
        *,
        account_id: str,
        account_raw: dict[str, Any],
        signal_date: date,
    ) -> dict[str, dict[str, object]]:
        broker = self.brokers[account_id]
        raw_blocks = broker.get_metadata("intraday_reentry_blocks", {})
        if not isinstance(raw_blocks, dict):
            raw_blocks = {}
        blocks = {
            str(symbol): dict(payload)
            for symbol, payload in raw_blocks.items()
            if isinstance(payload, dict)
        }
        if (
            str(account_raw.get("exit_timeframe", "1d")).lower() != "2h"
            or not blocks
        ):
            return blocks
        period = int(account_raw["exit_supertrend_period"])
        multiplier = float(account_raw["exit_supertrend_multiplier"])
        refresh_minutes = int(self.experiment.get("intraday_refresh_minutes", 800))
        released: list[str] = []
        for symbol, payload in sorted(blocks.items()):
            try:
                adjusted = self.intraday_cache.refresh_symbol(
                    symbol,
                    latest_minutes=refresh_minutes,
                    adjusted=True,
                )
                seed = self.intraday_seed_cache.load_symbol(symbol)
                signal = build_intraday_exit_signal(
                    symbol,
                    adjusted,
                    seed_bars=seed,
                    through=signal_date,
                    period=period,
                    multiplier=multiplier,
                    timeframe_minutes=120,
                    timezone=str(self.experiment["timezone"]),
                    regular_open=str(self.experiment["regular_open"]),
                    regular_close=str(self.experiment["regular_close"]),
                )
                if confirmed_two_hour_recovery(
                    signal_at=signal.signal_at,
                    trend=signal.trend,
                    blocked_at=str(payload.get("signal_at", "")),
                    timezone=str(self.experiment["timezone"]),
                    regular_close=str(self.experiment["regular_close"]),
                ):
                    released.append(symbol)
            except (RuntimeError, ValueError) as exc:
                print(
                    f"[account {account_id}] keep {symbol} re-entry block: {exc}",
                    flush=True,
                )
        for symbol in released:
            blocks.pop(symbol, None)
            print(
                f"[account {account_id}] released {symbol} re-entry block "
                "after completed 2h bullish recovery",
                flush=True,
            )
        broker.set_metadata("intraday_reentry_blocks", blocks)
        return blocks

    def _intraday_session_replays(
        self,
        *,
        account_id: str,
        account_raw: dict[str, Any],
        symbols: set[str],
        execution_date: date,
    ) -> dict[str, IntradayReplayExit | None]:
        timeframe = str(account_raw.get("exit_timeframe", "1d")).lower()
        if timeframe == "1d" or not symbols:
            return {}
        if timeframe != "2h":
            raise ValueError(
                f"Account {account_id} has unsupported exit_timeframe: "
                f"{timeframe}"
            )
        period = int(
            account_raw.get(
                "exit_supertrend_period",
                self.raw["strategy"]["supertrend_period"],
            )
        )
        multiplier = float(
            account_raw.get(
                "exit_supertrend_multiplier",
                self.raw["strategy"]["supertrend_multiplier"],
            )
        )
        replays: dict[str, IntradayReplayExit | None] = {}
        refresh_minutes = int(self.experiment.get("intraday_refresh_minutes", 800))
        for symbol in sorted(symbols):
            print(
                f"[account {account_id}] replaying {symbol} 1m session "
                f"for {execution_date}",
                flush=True,
            )
            adjusted = self.intraday_cache.refresh_symbol(
                symbol,
                latest_minutes=refresh_minutes,
                adjusted=True,
            )
            raw = self.intraday_cache.refresh_symbol(
                symbol,
                latest_minutes=refresh_minutes,
                adjusted=False,
            )
            seed_bars = self.intraday_seed_cache.load_symbol(symbol)
            replays[symbol] = replay_intraday_fence_exit(
                symbol,
                adjusted,
                raw,
                seed_bars=seed_bars,
                session_date=execution_date,
                period=period,
                multiplier=multiplier,
                confirm_minutes=int(account_raw.get("exit_confirm_minutes", 1)),
                timeframe_minutes=120,
                timezone=str(self.experiment["timezone"]),
                regular_open=str(self.experiment["regular_open"]),
                regular_close=str(self.experiment["regular_close"]),
            )
        return replays

    def _apply_intraday_replay(
        self,
        *,
        account_id: str,
        account_raw: dict[str, Any],
        execution_date: date,
        plan: OrderPlan,
        fills: list[str],
        replays: dict[str, IntradayReplayExit | None],
    ) -> tuple[OrderPlan, list[str], tuple[str, ...], dict[str, object]]:
        if str(account_raw.get("exit_timeframe", "1d")).lower() != "2h":
            return plan, fills, (), {}
        broker = self.brokers[account_id]
        account = broker.get_account()
        if not account.positions:
            return plan, fills, (f"Account {account_id} 2h replay: no position after the open.",), {}
        if len(account.positions) != 1:
            raise RuntimeError(
                f"Account {account_id} 2h replay expects at most one held symbol."
            )
        symbol, position = next(iter(account.positions.items()))
        if symbol not in replays:
            raise RuntimeError(
                f"Account {account_id} 2h replay was not prepared for {symbol}."
            )
        replay = replays[symbol]
        if replay is None:
            payload = {
                "session_date": execution_date.isoformat(),
                "symbol": symbol,
                "exit_filled": False,
                "status": "no_breach",
            }
            broker.set_metadata("intraday_exit_replay", payload)
            return (
                plan,
                fills,
                (
                    f"Account {account_id} 2h replay: {symbol} had no "
                    "completed-1m fence breach.",
                ),
                payload,
            )
        payload = replay.as_dict()
        if replay.pending_next_session:
            payload.update({"exit_filled": False, "status": "pending_next_open"})
            broker.set_metadata("pending_intraday_exit", payload)
            broker.set_metadata("intraday_exit_replay", payload)
            return (
                plan,
                fills,
                (
                    f"Account {account_id} 2h replay: {symbol} breached at "
                    f"{replay.signal_at}; "
                    "next regular-session open is pending.",
                ),
                payload,
            )

        exit_order = OrderIntent(
            symbol=symbol,
            side="sell",
            quantity=position.quantity,
            reason="completed 1m close breached active completed-2h fence",
        )
        exit_plan = OrderPlan(
            plan.strategy_name,
            "paper",
            (exit_order,),
            (f"Account {account_id} offline intraday replay exit.",),
        )
        exit_fills = broker.execute_plan(
            exit_plan,
            {symbol: float(replay.raw_fill_open)},
            float(self.experiment["fee_rate"]),
            float(self.experiment["slippage_rate"]),
            metadata_updates={
                "pending_intraday_exit": None,
                "last_intraday_exit_session": execution_date.isoformat(),
                "last_intraday_fill_at": replay.fill_at,
            },
        )
        payload.update(
            {
                "exit_filled": True,
                "status": "filled_next_minute_open",
                "fills": exit_fills,
            }
        )
        blocks = broker.get_metadata("intraday_reentry_blocks", {})
        if not isinstance(blocks, dict):
            blocks = {}
        blocks = dict(blocks)
        blocks[symbol] = dict(payload)
        broker.set_metadata("intraday_reentry_blocks", blocks)
        broker.set_metadata("intraday_exit_replay", payload)
        combined = OrderPlan(
            plan.strategy_name,
            plan.mode,
            (*plan.orders, exit_order),
            plan.notes,
        )
        notes = (
            f"Account {account_id} 2h replay: {symbol} close "
            f"{replay.signal_close:.4f} breached "
            f"{replay.lower_fence:.4f} at {replay.signal_at}; "
            f"filled next-minute open {float(replay.raw_fill_open):.4f}.",
        )
        return combined, [*fills, *exit_fills], notes, payload

    def _signal_frames(
        self,
        adjusted: dict[str, pd.DataFrame],
        symbols: list[str],
        signal_date: date,
        held_symbols: set[str],
    ) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        warmup = create_strategy(self.base_config).warmup_bars()
        for symbol in symbols:
            frame = truncate_daily(
                adjusted.get(symbol, pd.DataFrame()),
                signal_date,
            )
            has_signal_bar = any(
                timestamp.date() == signal_date
                for timestamp in pd.DatetimeIndex(frame.index)
            )
            if symbol in held_symbols and not has_signal_bar:
                raise RuntimeError(
                    f"Held symbol {symbol} lacks adjusted signal candle "
                    f"for {signal_date}."
                )
            if has_signal_bar and len(frame) >= warmup:
                frames[symbol] = frame
        if not frames:
            raise RuntimeError("No eligible signal frames after freshness checks.")
        return frames

    def _universe_on(self, session_date: date) -> tuple[str, ...]:
        history_path = resolve_lab_path(self.raw["universe"]["history_file"])
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        eligible = [
            snapshot
            for snapshot in payload.get("snapshots", [])
            if date.fromisoformat(snapshot["effective_date"]) <= session_date
        ]
        if not eligible:
            raise RuntimeError(
                f"No Nasdaq-100 universe snapshot for {session_date}."
            )
        selected = max(eligible, key=lambda value: value["effective_date"])
        symbols = tuple(
            sorted(
                {
                    str(symbol).replace(".", "-").upper()
                    for symbol in selected["symbols"]
                }
            )
        )
        if not symbols:
            raise RuntimeError("Nasdaq-100 universe snapshot is empty.")
        return symbols

    def _append_event(self, account_id: str, payload: dict[str, Any]) -> None:
        path = self.results_dir / "events" / f"account_{account_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _date_or_none(value: Any) -> date | None:
    if not value:
        return None
    return date.fromisoformat(str(value))


def resize_explicit_buys_to_execution_cash(
    plan: OrderPlan,
    account,
    execution_prices: dict[str, float],
    *,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[OrderPlan, tuple[str, ...]]:
    """Cap explicit buy quantities at next-open buying power.

    Initial leader entries carry a quantity estimated from the completed
    signal close.  A gap up at the next open must reduce that quantity rather
    than reject the whole paper order.  Cash-allocation replacement orders are
    left untouched because PaperBroker already sizes those at execution.
    """

    available_cash = float(account.cash)
    orders = []
    notes: list[str] = []
    for order in plan.orders:
        side = str(order.side).lower()
        if side == "sell":
            orders.append(order)
            position = account.positions.get(order.symbol)
            raw_price = execution_prices.get(order.symbol)
            if position is not None and raw_price is not None:
                sell_quantity = min(float(order.quantity), float(position.quantity))
                estimated_proceeds = (
                    sell_quantity
                    * float(raw_price)
                    * (1.0 - float(slippage_rate))
                    * (1.0 - float(fee_rate))
                )
                available_cash += max(0.0, estimated_proceeds)
            continue
        if (
            side != "buy"
            or order.quantity is None
            or order.cash_allocation_pct is not None
        ):
            orders.append(order)
            continue
        raw_price = execution_prices.get(order.symbol)
        if raw_price is None:
            orders.append(order)
            continue
        affordable = estimate_quantity(
            available_cash,
            float(raw_price),
            1.0,
            fee_rate=float(fee_rate),
            slippage_rate=float(slippage_rate),
        )
        desired = float(order.quantity)
        resized = min(desired, float(affordable))
        if resized <= 0.0:
            notes.append(
                f"Execution-price cash resize removed {order.symbol}: "
                f"{desired:g} -> 0"
            )
            continue
        if 0.0 < resized < desired:
            order = replace(order, quantity=resized)
            notes.append(
                f"Execution-price cash resize {order.symbol}: "
                f"{desired:g} -> {resized:g}"
            )
        orders.append(order)
        if resized > 0.0:
            fill_price = float(raw_price) * (1.0 + float(slippage_rate))
            estimated_cost = (
                resized * fill_price * (1.0 + float(fee_rate))
            )
            available_cash = max(0.0, available_cash - estimated_cost)
    return replace(plan, orders=tuple(orders)), tuple(notes)


def atr_target_weight(atr_pct: float, target_atr_risk_pct: float) -> float:
    atr_pct = float(atr_pct)
    target_atr_risk_pct = float(target_atr_risk_pct)
    if not 0.0 < target_atr_risk_pct <= 1.0:
        raise ValueError("entry_atr_risk_pct must be in (0, 1].")
    if not math.isfinite(atr_pct) or atr_pct <= 0.0:
        raise ValueError("ATR_pct must be positive and finite.")
    return min(1.0, target_atr_risk_pct / atr_pct)


def resize_buys_to_atr_risk(
    plan: OrderPlan,
    prepared_frames: dict[str, pd.DataFrame],
    signal_ts: pd.Timestamp,
    account,
    *,
    target_atr_risk_pct: float,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[OrderPlan, tuple[str, ...]]:
    """Size every new buy from the completed signal bar's ATR percentage.

    The result intentionally carries an explicit quantity for both initial and
    post-sell entries.  The execution-price guard later caps that quantity at
    actual next-open buying power, matching the backtest implementation.
    """

    equity = _account_liquidation_value(account)
    orders: list[OrderIntent] = []
    notes: list[str] = []
    for order in plan.orders:
        if str(order.side).lower() != "buy":
            orders.append(order)
            continue
        frame = prepared_frames.get(order.symbol)
        if frame is None or frame.empty:
            raise RuntimeError(f"Missing prepared ATR frame for {order.symbol}.")
        available = frame.loc[:signal_ts]
        if available.empty:
            raise RuntimeError(f"Missing signal-date ATR row for {order.symbol}.")
        row = available.iloc[-1]
        atr_pct = float(row["ATR_pct"])
        signal_close = float(row["Close"])
        weight = atr_target_weight(atr_pct, target_atr_risk_pct)
        quantity = estimate_quantity(
            equity,
            signal_close,
            weight,
            fee_rate=float(fee_rate),
            slippage_rate=float(slippage_rate),
        )
        if quantity <= 0:
            notes.append(
                f"ATR risk sizing removed {order.symbol}: target weight {weight:.2%}"
            )
            continue
        reason = (
            f"{order.reason}; ATR risk weight {weight:.2%}"
            if order.reason
            else f"ATR risk weight {weight:.2%}"
        )
        orders.append(
            replace(
                order,
                quantity=quantity,
                cash_allocation_pct=None,
                reason=reason,
            )
        )
        notes.append(
            f"ATR risk sizing {order.symbol}: ATR {atr_pct:.2%}, "
            f"target {target_atr_risk_pct:.2%}, weight {weight:.2%}, "
            f"quantity {quantity:g}"
        )
    return replace(plan, orders=tuple(orders)), tuple(notes)


def _account_liquidation_value(account) -> float:
    value = float(account.cash)
    for symbol, position in account.positions.items():
        economics = account.position_economics.get(symbol)
        if economics is not None and economics.estimated_exit_proceeds is not None:
            value += float(economics.estimated_exit_proceeds) + float(
                economics.distributions
            )
        elif economics is not None and economics.raw_mark is not None:
            value += float(position.quantity) * float(economics.raw_mark)
        else:
            value += float(position.quantity) * float(position.avg_price)
    return value


def projected_position_symbols(account, plan: OrderPlan) -> set[str]:
    symbols = set(account.positions)
    for order in plan.orders:
        if str(order.side).lower() == "sell":
            symbols.discard(order.symbol)
        elif str(order.side).lower() == "buy":
            symbols.add(order.symbol)
    return symbols


def force_pending_intraday_exit(
    plan: OrderPlan,
    account,
    pending: dict[str, object],
) -> OrderPlan:
    symbol = str(pending["symbol"])
    if any(
        str(order.side).lower() == "sell" and order.symbol == symbol
        for order in plan.orders
    ):
        return plan
    position = account.positions[symbol]
    forced = OrderIntent(
        symbol=symbol,
        side="sell",
        quantity=position.quantity,
        reason="prior-session final-minute 2h fence breach",
    )
    return replace(
        plan,
        orders=(forced, *plan.orders),
        notes=(*plan.notes, "Pending 2h intraday exit filled at this session open."),
    )


def confirmed_two_hour_recovery(
    *,
    signal_at: str,
    trend: int,
    blocked_at: str,
    timezone: str,
    regular_close: str,
) -> bool:
    if int(trend) != 1 or not blocked_at:
        return False
    bar_start = pd.Timestamp(signal_at)
    blocked = pd.Timestamp(blocked_at)
    if bar_start.tzinfo is None:
        bar_start = bar_start.tz_localize(timezone)
    else:
        bar_start = bar_start.tz_convert(timezone)
    if blocked.tzinfo is None:
        blocked = blocked.tz_localize(timezone)
    else:
        blocked = blocked.tz_convert(timezone)
    close_hour, close_minute = (
        int(part) for part in regular_close.split(":", 1)
    )
    session_close = bar_start.normalize() + pd.Timedelta(
        hours=close_hour,
        minutes=close_minute,
    )
    bar_end = min(bar_start + pd.Timedelta(minutes=120), session_close)
    return bar_end > blocked


def latest_closed_session_cutoff(
    market_now: datetime,
    *,
    regular_close: str,
    settlement_minutes: int = 15,
) -> date:
    """Return the latest calendar date that may contain a settled close.

    Before the close plus a short provider-settlement delay, the current
    New York date is excluded. The benchmark calendar then removes weekends
    and exchange holidays.
    """

    hour, minute = (int(part) for part in regular_close.split(":", 1))
    ready_at = market_now.replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    ) + timedelta(minutes=max(0, int(settlement_minutes)))
    if market_now < ready_at:
        return market_now.date() - timedelta(days=1)
    return market_now.date()
