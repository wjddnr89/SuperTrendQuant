from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAYGROUND_ROOT = PROJECT_ROOT / "playground"
UNIFIED_ROOT = PROJECT_ROOT / "unified_quant"
sys.path.insert(0, str(PLAYGROUND_ROOT / "scripts"))
sys.path.insert(0, str(UNIFIED_ROOT / "src"))

from supertrend_quant.config import AppConfig, load_split_config  # noqa: E402
from supertrend_quant.data import MarketData, market_index  # noqa: E402
from supertrend_quant.metrics import format_float, format_pct  # noqa: E402
from supertrend_quant.portfolio import OrderIntent, OrderPlan  # noqa: E402
from supertrend_quant.research import apply_config_overlay  # noqa: E402
from supertrend_quant.research.data_resolver import download_for_config  # noqa: E402
import supertrend_quant.runners as canonical_runners  # noqa: E402
from supertrend_quant.runners import (  # noqa: E402
    BacktestResult,
    _prepare_backtest,
    run_backtest_on_data,
)
from supertrend_quant.strategies import create_strategy  # noqa: E402
from supertrend_quant.strategies.leader_rotation import PreparedLeaderBacktest  # noqa: E402


PARAM_KEYS = (
    "entry",
    "market_filter",
    "asset_filter",
    "rs_method",
    "rs_period",
    "sell_confirm_bars",
    "hurdle",
    "max_positions",
    "st_period",
    "st_multiplier",
    "min_rotation_profit_pct",
)

PREP_KEYS = (
    "entry",
    "market_filter",
    "asset_filter",
    "rs_method",
    "rs_period",
    "st_period",
    "st_multiplier",
)

TEST_ALIASES = {
    "all": "all",
    "fixed": "fixed_walk_forward",
    "fixed_walk_forward": "fixed_walk_forward",
    "expanding": "expanding_walk_forward",
    "expanding_walk_forward": "expanding_walk_forward",
    "stability": "parameter_stability",
    "parameter_stability": "parameter_stability",
    "contribution": "trade_contribution",
    "trade_contribution": "trade_contribution",
    "stress": "cost_execution_stress",
    "cost_execution_stress": "cost_execution_stress",
    "purged": "purged_embargoed_cv",
    "purged_cv": "purged_embargoed_cv",
    "purged_embargoed_cv": "purged_embargoed_cv",
}

DEFAULT_TESTS = (
    "parameter_stability",
    "trade_contribution",
    "cost_execution_stress",
    "fixed_walk_forward",
    "expanding_walk_forward",
    "purged_embargoed_cv",
)


@dataclass
class PreparedBundle:
    config: AppConfig
    prepared_backtest: Any


class DelayedPreparedBacktest:
    """Delay canonical order intents while leaving fills and ledger handling canonical."""

    def __init__(self, delegate: PreparedLeaderBacktest, entry_delay_bars: int, exit_delay_bars: int):
        self.delegate = delegate
        self.entry_delay_bars = int(entry_delay_bars)
        self.exit_delay_bars = int(exit_delay_bars)
        self.step = -1
        self.queued: dict[int, list[OrderIntent]] = {}
        self.pending_buys: set[str] = set()
        self.pending_sells: set[str] = set()

    @property
    def report_frames(self):
        return getattr(self.delegate, "report_frames", None)

    def build_order_plan(self, signal_ts, account, mode="backtest"):
        self.step += 1
        due = self.queued.pop(self.step, [])
        releasable: list[OrderIntent] = []
        for order in due:
            if order.side.lower() != "buy" or not order.required_sell_symbols:
                releasable.append(order)
                continue
            unresolved = set(order.required_sell_symbols).intersection(account.positions)
            if unresolved:
                self.queued.setdefault(self.step + 1, []).append(order)
                continue
            releasable.append(replace(order, required_sell_symbols=()))
        due = releasable
        due_buys = {order.symbol for order in due if order.side.lower() == "buy"}
        due_sells = {order.symbol for order in due if order.side.lower() == "sell"}
        self.pending_buys.difference_update(due_buys)
        self.pending_sells.difference_update(due_sells)

        plan = self.delegate.build_order_plan(signal_ts, account, mode=mode)
        immediate: list[OrderIntent] = []
        plan_sell_symbols = {
            order.symbol for order in plan.orders if order.side.lower() == "sell"
        }
        for order in plan.orders:
            side = order.side.lower()
            if side == "buy":
                if order.symbol in due_buys or order.symbol in self.pending_buys:
                    continue
                delay = self.entry_delay_bars
                if order.required_sell_symbols and plan_sell_symbols.intersection(
                    order.required_sell_symbols
                ):
                    delay = max(delay, self.exit_delay_bars)
                pending = self.pending_buys
            else:
                if order.symbol in due_sells or order.symbol in self.pending_sells:
                    continue
                delay = self.exit_delay_bars
                pending = self.pending_sells
            if delay <= 0:
                immediate.append(order)
            else:
                self.queued.setdefault(self.step + delay, []).append(order)
                pending.add(order.symbol)

        released = sorted(
            [*due, *immediate],
            key=lambda order: 0 if order.side.lower() == "sell" else 1,
        )
        return OrderPlan(plan.strategy_name, mode, tuple(released), plan.notes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify one strategy combo with robustness tests.")
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT
            / "verification"
            / "configs"
            / "canonical_dual_momentum_best.json"
        ),
        help="JSON verification config.",
    )
    parser.add_argument(
        "--tests",
        default="all",
        help=(
            "Comma-separated tests: all, fixed_walk_forward, expanding_walk_forward, "
            "parameter_stability, trade_contribution, cost_execution_stress, purged_embargoed_cv."
        ),
    )
    parser.add_argument("--objective", default=None, help="Override optimization objective.")
    parser.add_argument("--run-id", default="", help="Result folder name.")
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "verification" / "results"),
        help="Directory for verification outputs.",
    )
    parser.add_argument("--max-candidates", type=int, default=None, help="Limit candidate grid size.")
    parser.add_argument("--save-trades", action=argparse.BooleanOptionalAction, default=True)
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | None, default: Path) -> str:
    if not value:
        return str(default)
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def normalize_combo(combo: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(combo)
    normalized["entry"] = str(normalized.get("entry", "single"))
    normalized["market_filter"] = str(normalized.get("market_filter", "1d"))
    normalized["asset_filter"] = str(normalized.get("asset_filter", "ichimoku_cloud+ema_trend"))
    normalized["rs_method"] = str(normalized.get("rs_method", "dual_momentum"))
    normalized["rs_period"] = int(normalized.get("rs_period", 150))
    normalized["sell_confirm_bars"] = int(normalized.get("sell_confirm_bars", 1))
    normalized["hurdle"] = float(normalized.get("hurdle", 2.0))
    normalized["max_positions"] = int(normalized.get("max_positions", 1))
    normalized["st_period"] = int(normalized.get("st_period", 10))
    normalized["st_multiplier"] = float(normalized.get("st_multiplier", 3.0))
    normalized["min_rotation_profit_pct"] = float(
        normalized.get("min_rotation_profit_pct", 0.0)
    )
    return normalized


def combo_key(combo: dict[str, Any], keys: tuple[str, ...] = PARAM_KEYS) -> tuple[Any, ...]:
    normalized = normalize_combo(combo)
    return tuple(normalized[key] for key in keys)


def date_index(full_idx: pd.Index, start: str | None, end: str | None) -> pd.Index:
    selected = full_idx
    if start:
        start_date = pd.Timestamp(start).date()
        selected = selected[[pd.Timestamp(timestamp).date() >= start_date for timestamp in selected]]
    if end:
        end_date = pd.Timestamp(end).date()
        selected = selected[[pd.Timestamp(timestamp).date() <= end_date for timestamp in selected]]
    if len(selected) < 2:
        raise RuntimeError(f"Not enough market bars for period {start} -> {end}.")
    return selected


def year_start(year: int) -> str:
    return f"{int(year):04d}-01-01"


def year_end(year: int) -> str:
    return f"{int(year):04d}-12-31"


def display_metric(value: float, metric: str) -> str:
    if metric in {"sharpe", "payoff", "payoff_ratio", "trade_count"}:
        return format_float(value)
    return format_pct(value)


def metric_value(row: dict[str, Any], objective: str) -> float:
    objective = objective.lower()
    if objective in {"return", "total_return"}:
        return float(row.get("total_return", 0.0))
    if objective == "alpha":
        return float(row.get("alpha", 0.0))
    if objective == "cagr":
        return float(row.get("cagr", 0.0))
    if objective == "mdd":
        return float(row.get("mdd", 0.0))
    if objective == "sharpe":
        return float(row.get("sharpe", 0.0))
    if objective == "calmar":
        return float(row.get("calmar", 0.0))
    if objective == "win_rate":
        return float(row.get("win_rate", 0.0))
    if objective in {"payoff", "payoff_ratio"}:
        return float(row.get("payoff_ratio", 0.0))
    raise ValueError(f"Unsupported objective: {objective}")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def calculate_cagr(equity: pd.Series) -> float:
    if equity.empty or len(equity.index) < 2:
        return 0.0
    years = (pd.Timestamp(equity.index[-1]) - pd.Timestamp(equity.index[0])).days / 365.25
    if years <= 0:
        return 0.0
    return (float(equity.iloc[-1]) / float(equity.iloc[0])) ** (1.0 / years) - 1.0


def benchmark_return_for_index(data: MarketData, index: pd.Index) -> float:
    frames = getattr(data, "benchmark", None) or {}
    frame = next(
        (value for value in frames.values() if value is not None and not value.empty),
        None,
    )
    if frame is None or "Close" not in frame or len(index) < 2:
        return float("nan")
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    selected = close.loc[(close.index >= index[0]) & (close.index <= index[-1])]
    if len(selected) < 2:
        return float("nan")
    return float(selected.iloc[-1] / selected.iloc[0] - 1.0)


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    params = row.get("params", {})
    flat = {key: value for key, value in row.items() if key not in {"params", "result"}}
    flat.update({f"param_{key}": params.get(key) for key in PARAM_KEYS})
    return flat


class VerificationEngine:
    def __init__(self, config_path: Path, cli_args: argparse.Namespace):
        self.config_path = config_path
        self.raw = load_json(config_path)
        self.args = cli_args
        self.objective = str(cli_args.objective or self.raw.get("objective", "total_return"))
        self.start = str(self.raw.get("start", "2015-10-19"))
        self.end = self.raw.get("end")
        self.period = str(self.raw.get("period", "max"))
        self.base_combo = normalize_combo(self.raw.get("base_combo", {}))
        self.strategy_path = resolve_path(
            self.raw.get("strategy"),
            PLAYGROUND_ROOT / "configs" / "strategies" / "leader_rotation.yaml",
        )
        self.runtime_path = resolve_path(
            self.raw.get("runtime"),
            PLAYGROUND_ROOT / "configs" / "runtimes" / "research_us_nasdaq100_rolling.yaml",
        )
        base = load_split_config(self.strategy_path, self.runtime_path)
        base = replace(
            base,
            period=self.period,
            timeframe="1d",
            data_store=replace(base.data_store, provider="parquet"),
        )
        universe = replace(
            base.universe,
            source="index_events",
            profiles={"US": ("nasdaq100",)},
            history_file="",
            file="",
            symbols=(),
            filters=replace(base.universe.filters, enabled=False),
        )
        self.base_config = replace(base, universe=universe, universe_file="", symbols=())
        if "costs" in self.raw:
            self.base_config = apply_config_overlay(self.base_config, self.raw["costs"])

        self.market_data: MarketData | None = None
        self.full_idx: pd.Index | None = None
        self.requested_idx: pd.Index | None = None
        self.prepared_cache: dict[tuple[Any, ...], PreparedBundle] = {}
        self.qqq_cache: dict[tuple[object, object], float] = {}
        self.evaluation_cache: dict[str, dict[str, Any]] = {}
        self.evaluation_checkpoint: Path | None = None

    def attach_run_dir(self, run_dir: Path) -> None:
        self.evaluation_checkpoint = run_dir / "evaluation_checkpoint.jsonl"
        if not self.evaluation_checkpoint.exists():
            return
        for line in self.evaluation_checkpoint.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            self.evaluation_cache[str(record["key"])] = dict(record["row"])
        print(
            f"[verification] resumed evaluations={len(self.evaluation_cache)}",
            flush=True,
        )

    @staticmethod
    def evaluation_key(
        combo: dict[str, Any],
        start: str | None,
        end: str | None,
        stress: dict[str, Any],
    ) -> str:
        payload = {
            "params": normalize_combo(combo),
            "start": start,
            "end": end,
            "stress": stress,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def checkpoint_evaluation(self, key: str, row: dict[str, Any]) -> None:
        self.evaluation_cache[key] = dict(row)
        if self.evaluation_checkpoint is None:
            return
        record = json.dumps(
            json_safe({"key": key, "row": row}),
            ensure_ascii=False,
            sort_keys=True,
        )
        with self.evaluation_checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(record + "\n")

    def load_data(self) -> None:
        print("[verification] downloading shared market data...", flush=True)
        self.market_data = download_for_config(self.base_config, allow_stale=True)
        self.full_idx = market_index(self.market_data)
        self.requested_idx = date_index(self.full_idx, self.start, self.end)
        print(
            f"[verification] timeline {self.full_idx[0]} -> {self.full_idx[-1]}, "
            f"requested {self.requested_idx[0]} -> {self.requested_idx[-1]}",
            flush=True,
        )
        print(
            f"[verification] universe=index_events:nasdaq100 runner=unified_quant "
            f"price_mode={self.base_config.data_store.price_mode} "
            f"data_quality={self.market_data.data_quality}",
            flush=True,
        )

    def ensure_loaded(self) -> None:
        if self.market_data is None:
            self.load_data()

    def test_settings(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}))

    def candidate_grid(self, settings: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return [dict(self.base_combo)]

    def config_for(
        self,
        combo: dict[str, Any],
        *,
        cost_multiplier: float = 1.0,
        adverse_slippage: float = 0.0,
    ) -> AppConfig:
        combo = normalize_combo(combo)
        config = apply_config_overlay(
            self.base_config,
            {
                "entry": combo["entry"],
                "market_filter": combo["market_filter"],
                "asset_filter": combo["asset_filter"],
                "sell_confirm_bars": combo["sell_confirm_bars"],
                "max_positions": combo["max_positions"],
                "st_period": combo["st_period"],
                "st_multiplier": combo["st_multiplier"],
                "fee_rate": self.base_config.costs.fee_rate * cost_multiplier,
                "slippage_rate": (
                    self.base_config.costs.slippage_rate * cost_multiplier
                    + adverse_slippage
                ),
            },
        )
        return replace(
            config,
            scoring=replace(
                config.scoring,
                type=combo["rs_method"],
                params={"lookback_bars": combo["rs_period"]},
            ),
            leader_rotation=replace(
                config.leader_rotation,
                hurdle_atr_mult=combo["hurdle"],
                min_rotation_profit_pct=combo["min_rotation_profit_pct"],
            ),
        )

    def prepare_bundle(self, combo: dict[str, Any]) -> PreparedBundle:
        self.ensure_loaded()
        assert self.market_data is not None
        combo = normalize_combo(combo)
        key = combo_key(combo, PREP_KEYS)
        if key in self.prepared_cache:
            return self.prepared_cache[key]

        config = self.config_for(combo)
        strategy = create_strategy(config)
        print(f"[verification] preparing {dict(zip(PREP_KEYS, key))}", flush=True)
        prepared = _prepare_backtest(strategy, self.market_data)
        if not isinstance(prepared, PreparedLeaderBacktest):
            raise TypeError("Canonical verification requires PreparedLeaderBacktest.")
        bundle = PreparedBundle(
            config=config,
            prepared_backtest=prepared,
        )
        self.prepared_cache[key] = bundle
        return bundle

    @staticmethod
    def prepared_for_config(bundle: PreparedBundle, config: AppConfig) -> PreparedLeaderBacktest:
        source = bundle.prepared_backtest
        return PreparedLeaderBacktest(
            create_strategy(config),
            source.prepared,
            source.market_filter_trends,
            source.universe_schedule,
        )

    def evaluate_combo(
        self,
        combo: dict[str, Any],
        *,
        start: str | None = None,
        end: str | None = None,
        label: str = "",
        stress: dict[str, Any] | None = None,
        include_result: bool = False,
    ) -> tuple[dict[str, Any], BacktestResult | None]:
        self.ensure_loaded()
        assert self.market_data is not None
        assert self.full_idx is not None
        combo = normalize_combo(combo)
        stress = dict(stress or {})
        effective_start = start or self.start
        effective_end = end or self.end
        cache_key = self.evaluation_key(combo, effective_start, effective_end, stress)
        if not include_result and cache_key in self.evaluation_cache:
            cached = dict(self.evaluation_cache[cache_key])
            cached["label"] = label
            print(f"[verification] cache hit {label}", flush=True)
            return cached, None

        run_idx = date_index(self.full_idx, effective_start, effective_end)
        cost_multiplier = float(stress.get("cost_multiplier", 1.0))
        entry_penalty = float(stress.get("entry_price_penalty", 0.0))
        exit_penalty = float(stress.get("exit_price_penalty", 0.0))
        if not math.isclose(entry_penalty, exit_penalty, abs_tol=1e-12):
            raise ValueError(
                "Canonical verification without runner changes supports only equal "
                "entry/exit adverse-fill penalties."
            )
        config = self.config_for(
            combo,
            cost_multiplier=cost_multiplier,
            adverse_slippage=entry_penalty,
        )
        bundle = self.prepare_bundle(combo)
        prepared: Any = self.prepared_for_config(bundle, config)
        entry_delay = int(stress.get("entry_delay_bars", 0))
        exit_delay = int(stress.get("exit_delay_bars", 0))
        if entry_delay or exit_delay:
            prepared = DelayedPreparedBacktest(prepared, entry_delay, exit_delay)
        with patch.object(canonical_runners, "_prepare_backtest", return_value=prepared):
            result = run_backtest_on_data(
                config,
                self.market_data,
                run_index=run_idx,
            )

        qqq_return = benchmark_return_for_index(self.market_data, result.equity.index)
        row = self.result_row(label, combo, result, qqq_return, stress)
        if not include_result:
            self.checkpoint_evaluation(cache_key, row)
        return row, result if include_result else None

    def result_row(
        self,
        label: str,
        combo: dict[str, Any],
        result: BacktestResult,
        qqq_return: float,
        stress: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metrics = result.metrics
        total_return = float(metrics.get("total_return", 0.0))
        mdd = float(metrics.get("mdd", 0.0))
        cagr = float(metrics.get("cagr", calculate_cagr(result.equity)))
        if "cagr" not in metrics:
            cagr = calculate_cagr(result.equity)
        calmar = float(metrics.get("calmar", cagr / abs(mdd) if mdd < 0 else 0.0))
        row = {
            "label": label,
            "params": normalize_combo(combo),
            "start": str(result.equity.index[0]),
            "end": str(result.equity.index[-1]),
            "initial_equity": float(result.equity.iloc[0]),
            "final_equity": float(result.equity.iloc[-1]),
            "total_return": total_return,
            "total_return_display": format_pct(total_return),
            "qqq_return": float(qqq_return),
            "qqq_return_display": format_pct(float(qqq_return)),
            "alpha": total_return - float(qqq_return),
            "alpha_display": format_pct(total_return - float(qqq_return)),
            "cagr": cagr,
            "cagr_display": format_pct(cagr),
            "mdd": mdd,
            "mdd_display": format_pct(mdd),
            "sharpe": float(metrics.get("sharpe", 0.0)),
            "sharpe_display": format_float(float(metrics.get("sharpe", 0.0))),
            "calmar": calmar,
            "calmar_display": format_float(calmar),
            "win_rate": float(metrics.get("win_rate", 0.0)),
            "win_rate_display": format_pct(float(metrics.get("win_rate", 0.0))),
            "payoff_ratio": float(metrics.get("payoff_ratio", 0.0)),
            "payoff_display": format_float(float(metrics.get("payoff_ratio", 0.0))),
            "trade_count": int(metrics.get("trade_count", 0)),
            "data_quality": result.data_quality,
            "data_version": result.data_version,
            "price_mode": result.price_mode,
            "unresolved_corporate_action_count": len(
                result.unresolved_corporate_action_ids
            ),
        }
        for key, value in (stress or {}).items():
            row[f"stress_{key}"] = value
        return row

    def save_rows(self, run_dir: Path, name: str, rows: list[dict[str, Any]]) -> None:
        pd.DataFrame(flatten_row(row) for row in rows).to_csv(run_dir / f"{name}.csv", index=False)

    def select_best(
        self,
        candidates: list[dict[str, Any]],
        *,
        start: str,
        end: str,
        label: str,
        objective: str,
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        for index, combo in enumerate(candidates, start=1):
            print(f"[verification] {label} candidate {index}/{len(candidates)}", flush=True)
            try:
                row, _ = self.evaluate_combo(combo, start=start, end=end, label=label)
                rows.append(row)
            except Exception as exc:  # noqa: BLE001
                rows.append({"label": label, "params": combo, "error": str(exc)})
        valid = [row for row in rows if "error" not in row]
        if not valid:
            raise RuntimeError(f"No valid candidates for {label}.")
        best = max(valid, key=lambda row: metric_value(row, objective))
        return dict(best["params"]), best, rows

    def run_parameter_stability(self, run_dir: Path) -> dict[str, Any]:
        settings = self.test_settings("parameter_stability")
        candidates = self.candidate_grid(settings)
        rows: list[dict[str, Any]] = []
        print(f"[verification] parameter_stability candidates={len(candidates)}", flush=True)
        for index, combo in enumerate(candidates, start=1):
            print(f"[verification] stability {index}/{len(candidates)}", flush=True)
            row, _ = self.evaluate_combo(combo, label="parameter_stability")
            rows.append(row)
        rows.sort(key=lambda row: metric_value(row, self.objective), reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
            row["objective"] = self.objective
            row["objective_value"] = metric_value(row, self.objective)
        self.save_rows(run_dir, "parameter_stability", rows)
        base_key = combo_key(self.base_combo)
        base_rank = next((row["rank"] for row in rows if combo_key(row["params"]) == base_key), None)
        return {
            "rows": len(rows),
            "base_rank": base_rank,
            "best": flatten_row(rows[0]) if rows else None,
        }

    def run_trade_contribution(self, run_dir: Path) -> dict[str, Any]:
        settings = self.test_settings("trade_contribution")
        remove_counts = [int(value) for value in settings.get("remove_top_counts", [1, 3, 5, 10])]
        row, result = self.evaluate_combo(
            self.base_combo,
            label="trade_contribution_base",
            include_result=True,
        )
        assert result is not None
        trades = pd.DataFrame(result.trade_records)
        initial_cash = float(result.equity.iloc[0])
        if not trades.empty:
            trades.insert(0, "trade_no", range(1, len(trades) + 1))
            trades["pnl_pct_display"] = trades["pnl_pct"].astype(float).map(format_pct)
            if "pnl_value" not in trades:
                trades["pnl_value"] = trades.get("pnl_cash", 0.0)
            trades["pnl_value_rank"] = trades["pnl_value"].astype(float).rank(ascending=False, method="first")
        trades.to_csv(run_dir / "trade_contribution_trades.csv", index=False)
        result.equity.rename("equity").to_frame().to_csv(run_dir / "trade_contribution_equity.csv")

        rows: list[dict[str, Any]] = []
        pnl_values = trades["pnl_value"].astype(float) if not trades.empty else pd.Series(dtype=float)
        pnl_pcts = trades["pnl_pct"].astype(float) if not trades.empty else pd.Series(dtype=float)
        sorted_indices = list(pnl_values.sort_values(ascending=False).index)
        original_final = float(result.equity.iloc[-1])
        original_product_final = initial_cash * float((1.0 + pnl_pcts).prod()) if not trades.empty else initial_cash
        for count in [0, *remove_counts]:
            removed = sorted_indices[:count]
            removed_pnl = float(pnl_values.loc[removed].sum()) if removed else 0.0
            value_replay_final = original_final - removed_pnl
            remaining = pnl_pcts.drop(index=removed) if removed else pnl_pcts
            product_final = initial_cash * float((1.0 + remaining).prod()) if not remaining.empty else initial_cash
            rows.append(
                {
                    "removed_top_trades": count,
                    "removed_pnl_value": removed_pnl,
                    "value_replay_final_equity": value_replay_final,
                    "value_replay_return": value_replay_final / initial_cash - 1.0,
                    "value_replay_return_display": format_pct(value_replay_final / initial_cash - 1.0),
                    "pct_product_final_equity": product_final,
                    "pct_product_return": product_final / initial_cash - 1.0,
                    "pct_product_return_display": format_pct(product_final / initial_cash - 1.0),
                    "original_final_equity": original_final,
                    "original_return": float(row["total_return"]),
                    "original_return_display": row["total_return_display"],
                    "original_pct_product_final_equity": original_product_final,
                }
            )
        pd.DataFrame(rows).to_csv(run_dir / "trade_contribution_removed.csv", index=False)
        top_trades = trades.sort_values("pnl_value", ascending=False).head(max(remove_counts or [10], default=10))
        top_trades.to_csv(run_dir / "trade_contribution_top_trades.csv", index=False)
        return {"base": flatten_row(row), "removed_rows": len(rows)}

    def run_cost_execution_stress(self, run_dir: Path) -> dict[str, Any]:
        settings = self.test_settings("cost_execution_stress")
        scenarios = settings.get("scenarios") or default_stress_scenarios(float(settings.get("adverse_open_rate", 0.005)))
        rows: list[dict[str, Any]] = []
        for index, scenario in enumerate(scenarios, start=1):
            scenario = dict(scenario)
            label = str(scenario.get("scenario", f"stress_{index}"))
            print(f"[verification] stress {index}/{len(scenarios)} {label}", flush=True)
            row, result = self.evaluate_combo(
                self.base_combo,
                label=label,
                stress=scenario,
                include_result=bool(self.args.save_trades),
            )
            row["scenario"] = label
            row["description"] = scenario.get("description", "")
            rows.append(row)
            if self.args.save_trades and result is not None:
                save_result_files(run_dir, f"stress_{label}", result)
        self.save_rows(run_dir, "cost_execution_stress", rows)
        return {"rows": len(rows), "baseline": flatten_row(rows[0]) if rows else None}

    def run_fixed_walk_forward(self, run_dir: Path) -> dict[str, Any]:
        return self.run_walk_forward(run_dir, name="fixed_walk_forward", expanding=False)

    def run_expanding_walk_forward(self, run_dir: Path) -> dict[str, Any]:
        return self.run_walk_forward(run_dir, name="expanding_walk_forward", expanding=True)

    def run_walk_forward(self, run_dir: Path, *, name: str, expanding: bool) -> dict[str, Any]:
        settings = self.test_settings(name)
        candidates = self.candidate_grid(settings)
        objective = str(settings.get("objective", self.objective))
        train_years = int(settings.get("train_years", 6))
        test_years = int(settings.get("test_years", 1))
        first_train_year = int(settings.get("first_train_year", 2010))
        first_test_year = int(settings.get("first_test_year", first_train_year + train_years))
        last_test_year = int(settings.get("last_test_year", pd.Timestamp(self.end or datetime.now()).year))
        save_train_candidates = bool(settings.get("save_train_candidates", True))

        fold_rows: list[dict[str, Any]] = []
        train_rows: list[dict[str, Any]] = []
        fold_no = 0
        for test_start_year in range(first_test_year, last_test_year + 1, test_years):
            test_end_year = min(test_start_year + test_years - 1, last_test_year)
            train_start_year = first_train_year if expanding else test_start_year - train_years
            train_end_year = test_start_year - 1
            if train_end_year < train_start_year:
                continue
            fold_no += 1
            train_start = year_start(train_start_year)
            train_end = year_end(train_end_year)
            test_start = year_start(test_start_year)
            test_end = year_end(test_end_year)
            fold_label = f"{name}_fold_{fold_no:02d}"
            print(
                f"[verification] {fold_label}: train {train_start}->{train_end}, "
                f"test {test_start}->{test_end}, candidates={len(candidates)}",
                flush=True,
            )
            best_combo, best_train, fold_train_rows = self.select_best(
                candidates,
                start=train_start,
                end=train_end,
                label=f"{fold_label}_train",
                objective=objective,
            )
            if save_train_candidates:
                for train_row in fold_train_rows:
                    train_row["fold"] = fold_no
                    train_row["train_start"] = train_start
                    train_row["train_end"] = train_end
                    train_rows.append(train_row)
            test_row, _ = self.evaluate_combo(
                best_combo,
                start=test_start,
                end=test_end,
                label=f"{fold_label}_test",
            )
            test_row["fold"] = fold_no
            test_row["train_start"] = train_start
            test_row["train_end"] = train_end
            test_row["test_start"] = test_start
            test_row["test_end"] = test_end
            test_row["objective"] = objective
            test_row["train_objective_value"] = metric_value(best_train, objective)
            test_row["train_total_return"] = best_train.get("total_return")
            test_row["train_sharpe"] = best_train.get("sharpe")
            test_row["train_mdd"] = best_train.get("mdd")
            fold_rows.append(test_row)

        self.save_rows(run_dir, name, fold_rows)
        if save_train_candidates:
            self.save_rows(run_dir, f"{name}_train_candidates", train_rows)
        return {"folds": len(fold_rows), "candidates": len(candidates)}

    def run_purged_embargoed_cv(self, run_dir: Path) -> dict[str, Any]:
        settings = self.test_settings("purged_embargoed_cv")
        candidates = self.candidate_grid(settings)
        objective = str(settings.get("objective", self.objective))
        first_year = int(settings.get("first_year", pd.Timestamp(self.start).year))
        last_year = int(settings.get("last_year", pd.Timestamp(self.end or datetime.now()).year))
        fold_years = int(settings.get("fold_years", 1))
        purge_days = int(settings.get("purge_days", 200))
        embargo_days = int(settings.get("embargo_days", 20))
        save_train_candidates = bool(settings.get("save_train_candidates", True))

        fold_rows: list[dict[str, Any]] = []
        train_rows: list[dict[str, Any]] = []
        fold_no = 0
        for val_start_year in range(first_year, last_year + 1, fold_years):
            val_end_year = min(val_start_year + fold_years - 1, last_year)
            val_start = year_start(val_start_year)
            val_end = year_end(val_end_year)
            train_segments = purged_train_segments(
                global_start=self.start,
                global_end=self.end or year_end(last_year),
                validation_start=val_start,
                validation_end=val_end,
                purge_days=purge_days,
                embargo_days=embargo_days,
            )
            if not train_segments:
                continue
            fold_no += 1
            label = f"purged_cv_fold_{fold_no:02d}"
            print(
                f"[verification] {label}: validation {val_start}->{val_end}, "
                f"train_segments={train_segments}, candidates={len(candidates)}",
                flush=True,
            )
            best_combo = dict(candidates[0])
            best_train: dict[str, Any] | None = None
            if len(candidates) > 1:
                scored_rows: list[dict[str, Any]] = []
                for index, combo in enumerate(candidates, start=1):
                    print(f"[verification] {label} candidate {index}/{len(candidates)}", flush=True)
                    segment_rows = []
                    for segment_start, segment_end in train_segments:
                        try:
                            segment_row, _ = self.evaluate_combo(
                                combo,
                                start=segment_start,
                                end=segment_end,
                                label=f"{label}_train",
                            )
                            segment_rows.append(segment_row)
                        except Exception as exc:  # noqa: BLE001
                            segment_rows.append({"params": combo, "error": str(exc)})
                    valid_segments = [row for row in segment_rows if "error" not in row]
                    if not valid_segments:
                        scored_rows.append({"label": f"{label}_train", "params": combo, "error": "no valid segments"})
                        continue
                    aggregate = aggregate_segment_rows(valid_segments, f"{label}_train")
                    aggregate["params"] = combo
                    scored_rows.append(aggregate)
                valid = [row for row in scored_rows if "error" not in row]
                if not valid:
                    continue
                best_train = max(valid, key=lambda row: metric_value(row, objective))
                best_combo = dict(best_train["params"])
                if save_train_candidates:
                    for row in scored_rows:
                        row["fold"] = fold_no
                        row["validation_start"] = val_start
                        row["validation_end"] = val_end
                        row["purge_days"] = purge_days
                        row["embargo_days"] = embargo_days
                        train_rows.append(row)
            validation_row, _ = self.evaluate_combo(
                best_combo,
                start=val_start,
                end=val_end,
                label=f"{label}_validation",
            )
            validation_row["fold"] = fold_no
            validation_row["validation_start"] = val_start
            validation_row["validation_end"] = val_end
            validation_row["train_segments"] = json.dumps(train_segments)
            validation_row["purge_days"] = purge_days
            validation_row["embargo_days"] = embargo_days
            validation_row["objective"] = objective
            validation_row["train_objective_value"] = (
                metric_value(best_train, objective) if best_train is not None else None
            )
            validation_row["selection_mode"] = (
                "optimized" if best_train is not None else "fixed_single_combo"
            )
            fold_rows.append(validation_row)

        self.save_rows(run_dir, "purged_embargoed_cv", fold_rows)
        if save_train_candidates:
            self.save_rows(run_dir, "purged_embargoed_cv_train_candidates", train_rows)
        return {"folds": len(fold_rows), "candidates": len(candidates)}


def aggregate_segment_rows(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    total_return = 1.0
    qqq_return = 1.0
    total_days = 0
    trade_count = 0
    weighted_win = 0.0
    payoff_values = []
    sharpe_values = []
    mdds = []
    for row in rows:
        total_return *= 1.0 + float(row.get("total_return", 0.0))
        qqq_return *= 1.0 + float(row.get("qqq_return", 0.0))
        days = max(0, (pd.Timestamp(row["end"]) - pd.Timestamp(row["start"])).days)
        total_days += days
        trades = int(row.get("trade_count", 0))
        trade_count += trades
        weighted_win += float(row.get("win_rate", 0.0)) * trades
        payoff_values.append(float(row.get("payoff_ratio", 0.0)))
        sharpe_values.append(float(row.get("sharpe", 0.0)))
        mdds.append(float(row.get("mdd", 0.0)))
    total_return -= 1.0
    qqq_return -= 1.0
    years = total_days / 365.25
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    mdd = min(mdds) if mdds else 0.0
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    win_rate = weighted_win / trade_count if trade_count else 0.0
    return {
        "label": label,
        "start": rows[0]["start"],
        "end": rows[-1]["end"],
        "segments": len(rows),
        "total_return": total_return,
        "total_return_display": format_pct(total_return),
        "qqq_return": qqq_return,
        "qqq_return_display": format_pct(qqq_return),
        "alpha": total_return - qqq_return,
        "alpha_display": format_pct(total_return - qqq_return),
        "cagr": cagr,
        "cagr_display": format_pct(cagr),
        "mdd": mdd,
        "mdd_display": format_pct(mdd),
        "sharpe": sum(sharpe_values) / len(sharpe_values) if sharpe_values else 0.0,
        "sharpe_display": format_float(sum(sharpe_values) / len(sharpe_values) if sharpe_values else 0.0),
        "calmar": calmar,
        "calmar_display": format_float(calmar),
        "win_rate": win_rate,
        "win_rate_display": format_pct(win_rate),
        "payoff_ratio": sum(payoff_values) / len(payoff_values) if payoff_values else 0.0,
        "payoff_display": format_float(sum(payoff_values) / len(payoff_values) if payoff_values else 0.0),
        "trade_count": trade_count,
    }


def purged_train_segments(
    *,
    global_start: str,
    global_end: str,
    validation_start: str,
    validation_end: str,
    purge_days: int,
    embargo_days: int,
) -> list[tuple[str, str]]:
    start = pd.Timestamp(global_start)
    end = pd.Timestamp(global_end)
    val_start = pd.Timestamp(validation_start)
    val_end = pd.Timestamp(validation_end)
    left_end = val_start - pd.Timedelta(days=purge_days + 1)
    right_start = val_end + pd.Timedelta(days=embargo_days + 1)
    segments: list[tuple[str, str]] = []
    if start < left_end:
        segments.append((start.date().isoformat(), left_end.date().isoformat()))
    if right_start < end:
        segments.append((right_start.date().isoformat(), end.date().isoformat()))
    return segments


def default_stress_scenarios(adverse_rate: float) -> list[dict[str, Any]]:
    return [
        {"scenario": "baseline", "description": "Current fee/slippage"},
        {"scenario": "cost_2x", "description": "Fee/slippage multiplied by 2", "cost_multiplier": 2.0},
        {"scenario": "cost_3x", "description": "Fee/slippage multiplied by 3", "cost_multiplier": 3.0},
        {"scenario": "cost_5x", "description": "Fee/slippage multiplied by 5", "cost_multiplier": 5.0},
        {
            "scenario": "entry_delay_1d",
            "description": "Buy execution delayed by one extra trading session",
            "entry_delay_bars": 1,
        },
        {
            "scenario": "exit_delay_1d",
            "description": "Sell execution delayed by one extra trading session",
            "exit_delay_bars": 1,
        },
        {
            "scenario": "entry_exit_delay_1d",
            "description": "Buy and sell execution delayed by one extra trading session",
            "entry_delay_bars": 1,
            "exit_delay_bars": 1,
        },
        {
            "scenario": "entry_exit_open_worse_0p5pct",
            "description": "Buy and sell fills 0.5% worse than open",
            "entry_price_penalty": adverse_rate,
            "exit_price_penalty": adverse_rate,
        },
        {
            "scenario": "execution_full_stress",
            "description": "One-day buy/sell delay and 0.5% worse open fills",
            "entry_delay_bars": 1,
            "exit_delay_bars": 1,
            "entry_price_penalty": adverse_rate,
            "exit_price_penalty": adverse_rate,
        },
        {
            "scenario": "cost_2x_execution_full_stress",
            "description": "2x costs plus full execution stress",
            "cost_multiplier": 2.0,
            "entry_delay_bars": 1,
            "exit_delay_bars": 1,
            "entry_price_penalty": adverse_rate,
            "exit_price_penalty": adverse_rate,
        },
        {
            "scenario": "cost_3x_execution_full_stress",
            "description": "3x costs plus full execution stress",
            "cost_multiplier": 3.0,
            "entry_delay_bars": 1,
            "exit_delay_bars": 1,
            "entry_price_penalty": adverse_rate,
            "exit_price_penalty": adverse_rate,
        },
    ]


def save_result_files(run_dir: Path, stem: str, result: BacktestResult) -> None:
    safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
    trades = pd.DataFrame(result.trade_records)
    if not trades.empty:
        trades.insert(0, "trade_no", range(1, len(trades) + 1))
        trades["pnl_pct_display"] = trades["pnl_pct"].astype(float).map(format_pct)
    trades.to_csv(run_dir / f"{safe_stem}_trades.csv", index=False)
    result.equity.rename("equity").to_frame().to_csv(run_dir / f"{safe_stem}_equity.csv")


def parse_tests(raw: str) -> list[str]:
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    mapped = []
    for item in requested:
        key = TEST_ALIASES.get(item)
        if key is None:
            raise ValueError(f"Unknown test: {item}")
        if key == "all":
            return list(DEFAULT_TESTS)
        mapped.append(key)
    return list(dict.fromkeys(mapped))


def write_report(run_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "Combo Verification Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Objective: {summary.get('objective')}",
        f"Config   : {summary.get('config')}",
        "",
        "Completed Tests",
    ]
    for name, value in summary.get("tests", {}).items():
        lines.append(f"- {name}: {value}")
    (run_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    tests = parse_tests(args.tests)
    run_id = args.run_id.strip() or datetime.now().strftime("verification_%Y%m%d_%H%M%S")
    run_dir = Path(args.results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    engine = VerificationEngine(config_path, args)
    engine.attach_run_dir(run_dir)
    engine.load_data()
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        summary.update(
            {
                "config": str(config_path),
                "objective": engine.objective,
                "base_combo": engine.base_combo,
                "run_dir": str(run_dir),
            }
        )
        summary.setdefault("tests", {})
    else:
        summary = {
            "config": str(config_path),
            "objective": engine.objective,
            "base_combo": engine.base_combo,
            "tests": {},
            "run_dir": str(run_dir),
        }
    runners = {
        "parameter_stability": engine.run_parameter_stability,
        "trade_contribution": engine.run_trade_contribution,
        "cost_execution_stress": engine.run_cost_execution_stress,
        "fixed_walk_forward": engine.run_fixed_walk_forward,
        "expanding_walk_forward": engine.run_expanding_walk_forward,
        "purged_embargoed_cv": engine.run_purged_embargoed_cv,
    }
    for test in tests:
        print(f"[verification] running {test}", flush=True)
        summary["tests"][test] = runners[test](run_dir)
        summary_path.write_text(
            json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_report(run_dir, summary)
    print("[verification] done")
    print(f"saved_dir={run_dir}")
    print(f"summary_json={run_dir / 'summary.json'}")
    print(f"report_txt={run_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
