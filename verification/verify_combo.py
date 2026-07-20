from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAYGROUND_ROOT = PROJECT_ROOT / "playground"
sys.path.insert(0, str(PLAYGROUND_ROOT / "scripts"))
sys.path.insert(0, str(PLAYGROUND_ROOT / "src"))

from search_nasdaq100_daily_3y_grid import (  # noqa: E402
    build_fast_leader_plan,
    close_at_or_before_position,
    open_at_position,
    portfolio_value_fast,
    prepare_active_universe,
    prepare_candidate_lists,
    prepare_exit_down_states,
    prepare_market_filter_states,
    prepare_row_positions,
    qqq_return_for_index,
    run_prepared_backtest,
)
from supertrend_quant.config import AppConfig, load_split_config  # noqa: E402
from supertrend_quant.data import MarketData, market_index  # noqa: E402
from supertrend_quant.metrics import calculate_metrics, format_float, format_pct  # noqa: E402
from supertrend_quant.portfolio import AccountSnapshot, OrderIntent, Position, estimate_quantity  # noqa: E402
from supertrend_quant.research import apply_config_overlay  # noqa: E402
from supertrend_quant.research.data_resolver import download_for_config  # noqa: E402
from supertrend_quant.runners import BacktestResult, _select_run_index  # noqa: E402
from supertrend_quant.strategies import create_strategy  # noqa: E402


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
    strategy: Any
    prepared_backtest: Any
    row_positions: dict[str, Any]
    market_filter_states: dict[str, list[bool]]
    candidates_by_position: list[list[dict[str, float | str]]]
    exit_cache: dict[int, dict[str, list[bool]]] = field(default_factory=dict)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify one strategy combo with robustness tests.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "verification" / "configs" / "dual_momentum_top.json"),
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
    normalized["sell_confirm_bars"] = int(normalized.get("sell_confirm_bars", 5))
    normalized["hurdle"] = float(normalized.get("hurdle", 2.0))
    normalized["max_positions"] = int(normalized.get("max_positions", 1))
    normalized["st_period"] = int(normalized.get("st_period", 10))
    normalized["st_multiplier"] = float(normalized.get("st_multiplier", 3.0))
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
        self.start = str(self.raw.get("start", "2010-01-01"))
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
        self.base_config = base.__class__(**{**base.__dict__, "period": self.period, "timeframe": "1d"})
        if "costs" in self.raw:
            self.base_config = apply_config_overlay(self.base_config, self.raw["costs"])

        self.market_data: MarketData | None = None
        self.full_idx: pd.Index | None = None
        self.requested_idx: pd.Index | None = None
        self.active_by_position: list[set[str] | None] | None = None
        self.prepared_cache: dict[tuple[Any, ...], PreparedBundle] = {}
        self.qqq_cache: dict[tuple[object, object], float] = {}

    def load_data(self) -> None:
        print("[verification] downloading shared market data...", flush=True)
        self.market_data = download_for_config(self.base_config)
        self.full_idx = market_index(self.market_data)
        self.requested_idx = date_index(self.full_idx, self.start, self.end)
        self.active_by_position = prepare_active_universe(self.market_data, self.full_idx)
        print(
            f"[verification] timeline {self.full_idx[0]} -> {self.full_idx[-1]}, "
            f"requested {self.requested_idx[0]} -> {self.requested_idx[-1]}",
            flush=True,
        )

    def ensure_loaded(self) -> None:
        if self.market_data is None:
            self.load_data()

    def test_settings(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}))

    def candidate_grid(self, settings: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        settings = settings or {}
        space = settings.get("space") or self.raw.get("validation_space") or {}
        values = []
        for key in PARAM_KEYS:
            values.append(as_list(space.get(key, self.base_combo.get(key))))
        combos = [normalize_combo(dict(zip(PARAM_KEYS, product))) for product in itertools.product(*values)]
        seen: set[tuple[Any, ...]] = set()
        unique: list[dict[str, Any]] = []
        for combo in combos:
            key = combo_key(combo)
            if key not in seen:
                seen.add(key)
                unique.append(combo)
        limit = self.args.max_candidates or settings.get("max_candidates") or self.raw.get("max_candidates")
        if limit is not None:
            unique = unique[: int(limit)]
        return unique

    def prepare_bundle(self, combo: dict[str, Any]) -> PreparedBundle:
        self.ensure_loaded()
        assert self.market_data is not None
        assert self.full_idx is not None
        assert self.active_by_position is not None
        combo = normalize_combo(combo)
        key = combo_key(combo, PREP_KEYS)
        if key in self.prepared_cache:
            return self.prepared_cache[key]

        config = apply_config_overlay(self.base_config, combo)
        strategy = create_strategy(config)
        print(f"[verification] preparing {dict(zip(PREP_KEYS, key))}", flush=True)
        prepared = strategy.prepare_backtest(
            self.market_data.bars,
            benchmark=self.market_data.benchmark,
            filter_benchmark=self.market_data.filter_benchmark,
            universe_schedule=self.market_data.universe_schedule,
        )
        row_positions = prepare_row_positions(prepared.prepared, self.full_idx)
        market_filter_states = prepare_market_filter_states(prepared, self.full_idx)
        candidates_by_position = prepare_candidate_lists(
            config,
            strategy,
            prepared.prepared,
            market_filter_states,
            self.active_by_position,
            row_positions,
            self.full_idx,
        )
        bundle = PreparedBundle(
            config=config,
            strategy=strategy,
            prepared_backtest=prepared,
            row_positions=row_positions,
            market_filter_states=market_filter_states,
            candidates_by_position=candidates_by_position,
        )
        self.prepared_cache[key] = bundle
        return bundle

    def exit_states(self, bundle: PreparedBundle, config: AppConfig) -> dict[str, list[bool]]:
        assert self.full_idx is not None
        confirm = int(config.exit.sell_confirm_bars)
        if confirm not in bundle.exit_cache:
            bundle.exit_cache[confirm] = prepare_exit_down_states(
                config,
                bundle.prepared_backtest.prepared,
                self.full_idx,
                bundle.row_positions,
            )
        return bundle.exit_cache[confirm]

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
        run_idx = date_index(self.full_idx, start or self.start, end or self.end)
        config = apply_config_overlay(self.base_config, combo)
        stress = dict(stress or {})
        cost_multiplier = float(stress.get("cost_multiplier", 1.0))
        if cost_multiplier != 1.0:
            config = apply_config_overlay(
                config,
                {
                    "fee_rate": config.costs.fee_rate * cost_multiplier,
                    "slippage_rate": config.costs.slippage_rate * cost_multiplier,
                },
            )

        bundle = self.prepare_bundle(combo)
        exit_states = self.exit_states(bundle, config)
        entry_delay = int(stress.get("entry_delay_bars", 0))
        exit_delay = int(stress.get("exit_delay_bars", 0))
        entry_penalty = float(stress.get("entry_price_penalty", 0.0))
        exit_penalty = float(stress.get("exit_price_penalty", 0.0))
        if entry_delay or exit_delay or entry_penalty or exit_penalty:
            result = run_execution_stress_backtest(
                config,
                self.market_data,
                bundle.prepared_backtest.prepared,
                bundle.candidates_by_position,
                bundle.market_filter_states,
                exit_states,
                self.active_by_position or [],
                bundle.row_positions,
                bundle.strategy.warmup_bars(),
                run_idx,
                entry_delay_bars=entry_delay,
                exit_delay_bars=exit_delay,
                entry_price_penalty=entry_penalty,
                exit_price_penalty=exit_penalty,
            )
        else:
            result = run_prepared_backtest(
                config,
                self.market_data,
                bundle.prepared_backtest.prepared,
                bundle.candidates_by_position,
                bundle.market_filter_states,
                exit_states,
                self.active_by_position or [],
                bundle.row_positions,
                bundle.strategy.warmup_bars(),
                run_idx,
            )

        qqq_return = qqq_return_for_index(self.market_data, result.equity.index, self.qqq_cache)
        row = self.result_row(label, combo, result, qqq_return, stress)
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
            validation_row["train_objective_value"] = metric_value(best_train, objective)
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
            "scenario": "entry_open_worse_0p5pct",
            "description": "Buy fills 0.5% worse than open",
            "entry_price_penalty": adverse_rate,
        },
        {
            "scenario": "exit_open_worse_0p5pct",
            "description": "Sell fills 0.5% worse than open",
            "exit_price_penalty": adverse_rate,
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


def run_execution_stress_backtest(
    config: AppConfig,
    market_data: MarketData,
    prepared: dict[str, pd.DataFrame],
    candidates_by_position: list[list[dict[str, float | str]]],
    market_filter_states: dict[str, list[bool]],
    exit_down_states: dict[str, list[bool]],
    active_by_position: list[set[str] | None],
    row_positions: dict[str, Any],
    warmup_bars: int,
    run_index: pd.Index | None = None,
    *,
    entry_delay_bars: int = 0,
    exit_delay_bars: int = 0,
    entry_price_penalty: float = 0.0,
    exit_price_penalty: float = 0.0,
) -> BacktestResult:
    if entry_delay_bars < 0 or exit_delay_bars < 0:
        raise ValueError("Delay bars must be non-negative.")
    if entry_price_penalty < 0 or exit_price_penalty < 0:
        raise ValueError("Price penalties must be non-negative.")
    if config.costs.slippage_rate + exit_price_penalty >= 1.0:
        raise ValueError("Exit slippage plus price penalty must be below 100%.")

    full_idx = market_index(market_data)
    idx = _select_run_index(full_idx, run_index)
    if len(idx) < 2:
        raise RuntimeError("Not enough common bars to run a backtest.")
    first_full_position = int(full_idx.get_indexer([idx[0]])[0])
    first_target_position = max(first_full_position, int(warmup_bars))
    idx = idx.intersection(full_idx[first_target_position:], sort=False)
    if len(idx) < 2:
        raise RuntimeError("Not enough bars remain after strategy warm-up.")

    cash = float(config.capital.initial_cash)
    positions: dict[str, Position] = {}
    entry_values: dict[str, float] = {}
    entry_times: dict[str, object] = {}
    equity_points: list[tuple[pd.Timestamp, float]] = []
    trade_returns: list[float] = []
    trade_records: list[dict[str, object]] = []
    pending_orders: dict[int, list[OrderIntent]] = {}
    pending_buy_symbols: set[str] = set()
    pending_sell_symbols: set[str] = set()
    last_full_position = int(full_idx.get_loc(idx[-1]))
    max_positions = max(1, int(config.risk.max_position_count))
    buy_slippage = config.costs.slippage_rate + entry_price_penalty
    sell_slippage = config.costs.slippage_rate + exit_price_penalty

    def execute_due_orders(exec_position: int, exec_ts: pd.Timestamp) -> None:
        nonlocal cash
        due_orders = pending_orders.pop(exec_position, [])
        due_orders = sorted(due_orders, key=lambda order: 0 if order.side.lower() == "sell" else 1)
        for order in due_orders:
            side = order.side.lower()
            if side == "buy":
                pending_buy_symbols.discard(order.symbol)
            else:
                pending_sell_symbols.discard(order.symbol)

            raw_price = open_at_position(prepared, row_positions, order.symbol, exec_position, exec_ts)
            if raw_price is None:
                continue
            if side == "buy":
                if order.symbol in positions:
                    continue
                affordable_quantity = estimate_quantity(
                    cash,
                    raw_price,
                    1.0,
                    fee_rate=config.costs.fee_rate,
                    slippage_rate=buy_slippage,
                )
                quantity = min(order.quantity, affordable_quantity)
                if quantity <= 0:
                    continue
                fill = raw_price * (1.0 + buy_slippage)
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
                fill = raw_price * (1.0 - sell_slippage)
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
                        "entry_value": entry_value,
                        "exit_value": proceeds,
                        "pnl_value": proceeds - entry_value,
                        "pnl_pct": pnl_pct,
                        "exit_reason": order.reason,
                        "entry_delay_bars": entry_delay_bars,
                        "exit_delay_bars": exit_delay_bars,
                        "entry_price_penalty": entry_price_penalty,
                        "exit_price_penalty": exit_price_penalty,
                    }
                )
                positions.pop(order.symbol, None)

    for signal_ts in idx:
        full_position = int(full_idx.get_loc(signal_ts))
        execute_due_orders(full_position, signal_ts)
        equity_points.append(
            (
                signal_ts,
                portfolio_value_fast(cash, positions, prepared, row_positions, full_position),
            )
        )
        if full_position >= last_full_position:
            continue

        account = AccountSnapshot(cash=cash, positions=positions.copy())
        plan = build_fast_leader_plan(
            config,
            prepared,
            candidates_by_position[full_position],
            market_filter_states,
            exit_down_states,
            active_by_position[full_position],
            row_positions,
            full_position,
            account,
            mode="verification",
        )
        for order in plan.orders:
            side = order.side.lower()
            if side == "buy":
                planned_positions = len(positions) - len(pending_sell_symbols) + len(pending_buy_symbols)
                if planned_positions >= max_positions:
                    continue
                if order.symbol in positions or order.symbol in pending_buy_symbols:
                    continue
                pending_buy_symbols.add(order.symbol)
                delay = entry_delay_bars
            else:
                if order.symbol not in positions or order.symbol in pending_sell_symbols:
                    continue
                pending_sell_symbols.add(order.symbol)
                delay = exit_delay_bars
            exec_position = full_position + 1 + delay
            if exec_position <= last_full_position:
                pending_orders.setdefault(exec_position, []).append(order)

    if positions:
        final_ts = idx[-1]
        final_full_position = int(full_idx.get_loc(final_ts))
        for symbol, position in list(positions.items()):
            final_close = close_at_or_before_position(prepared, row_positions, symbol, final_full_position)
            if final_close is None:
                continue
            final_price = final_close * (1.0 - sell_slippage)
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
                    "entry_value": entry_value,
                    "exit_value": proceeds,
                    "pnl_value": proceeds - entry_value,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "FinalClose",
                    "entry_delay_bars": entry_delay_bars,
                    "exit_delay_bars": exit_delay_bars,
                    "entry_price_penalty": entry_price_penalty,
                    "exit_price_penalty": exit_price_penalty,
                }
            )
            positions.pop(symbol, None)

    equity = pd.Series(
        [point[1] for point in equity_points],
        index=[point[0] for point in equity_points],
        name="equity",
    )
    return BacktestResult(
        equity=equity,
        metrics=calculate_metrics(equity, trade_returns, config.timeframe),
        trades=trade_returns,
        skipped=market_data.skipped,
        trade_records=tuple(trade_records),
        universe_snapshot=getattr(market_data, "universe_snapshot", None),
    )


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
    engine.load_data()
    summary: dict[str, Any] = {
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
    (run_dir / "summary.json").write_text(
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
