from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import optuna
import pandas as pd
from optuna.trial import TrialState


SCRIPT_DIR = Path(__file__).resolve().parent
PLAYGROUND_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PLAYGROUND_ROOT / "src"))

from search_nasdaq100_daily_3y_grid import (  # noqa: E402
    csv_values,
    flat_row,
    group_parameter_grid,
    json_safe,
    ordered_rows,
    prepare_active_universe,
    prepare_candidate_lists,
    prepare_exit_down_states,
    prepare_market_filter_states,
    prepare_row_positions,
    qqq_return_for_index,
    result_table_lines,
    run_index_from_start,
    run_prepared_backtest,
)
from supertrend_quant.config import AppConfig, load_split_config  # noqa: E402
from supertrend_quant.data import MarketData, market_index  # noqa: E402
from supertrend_quant.research import apply_config_overlay  # noqa: E402
from supertrend_quant.research.data_resolver import download_for_config  # noqa: E402
from supertrend_quant.research.scoring import score_metrics  # noqa: E402
from supertrend_quant.strategies.common import (  # noqa: E402
    precompute_market_filter_trends,
    with_strategy_components,
)
from supertrend_quant.strategies import create_strategy  # noqa: E402


OBJECTIVE_METRICS = ("return", "alpha", "mdd", "sharpe", "win_rate", "payoff")
PARAM_FIELDS = (
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optuna search for daily Nasdaq-100 strategy combinations.")
    parser.add_argument(
        "--strategy",
        default=str(PLAYGROUND_ROOT / "configs" / "strategies" / "leader_rotation.yaml"),
    )
    parser.add_argument(
        "--runtime",
        default=str(PLAYGROUND_ROOT / "configs" / "runtimes" / "research_us_nasdaq100_rolling.yaml"),
    )
    parser.add_argument("--period", default="max")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--entries", default="single")
    parser.add_argument("--market-filters", default="none,1d")
    parser.add_argument("--asset-filters", default="none,ichimoku_cloud,ichimoku_cloud+ema_trend")
    parser.add_argument(
        "--rs-methods",
        default="vol_adjusted,composite,skip_recent,beta_adjusted,dual_momentum",
    )
    parser.add_argument("--rs-periods", default="50,100")
    parser.add_argument("--hurdles", default="1.0,1.25,1.5,2.0")
    parser.add_argument("--sell-confirm-bars", default="1,2,3,5,8")
    parser.add_argument("--max-positions", default="3,4,5,6")
    parser.add_argument("--st-periods", default="10")
    parser.add_argument("--st-multipliers", default="3.0")
    parser.add_argument("--objectives", default="return,alpha,mdd,sharpe,win_rate,payoff")
    parser.add_argument("--n-trials", type=int, default=20, help="Additional trials per objective.")
    parser.add_argument("--timeout-minutes", type=float, default=60.0, help="Global wall-clock limit.")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--results-dir",
        default=str(PLAYGROUND_ROOT / "results" / "research" / "us_nasdaq100_rolling" / "optuna"),
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--import-results",
        action="append",
        default=[],
        help="Existing all_results.csv to use as a no-retest cache. Can be repeated.",
    )
    parser.add_argument(
        "--auto-import-searches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Import results/research/us_nasdaq100_rolling/searches/*/all_results.csv as cache.",
    )
    parser.add_argument("--min-trades", type=int, default=0)
    return parser


def metric_value(row: dict[str, Any], metric: str, min_trades: int = 0) -> float:
    metrics = row["metrics"]
    if int(metrics.get("trade_count", 0)) < min_trades:
        return -1.0e9
    if metric == "return":
        value = metrics.get("total_return", 0.0)
    elif metric == "alpha":
        value = row.get("alpha", 0.0)
    elif metric == "mdd":
        value = metrics.get("mdd", 0.0)
    elif metric == "sharpe":
        value = metrics.get("sharpe", 0.0)
    elif metric == "win_rate":
        value = metrics.get("win_rate", 0.0)
    elif metric == "payoff":
        value = metrics.get("payoff_ratio", 0.0)
    else:
        raise ValueError(f"Unsupported objective: {metric}")
    parsed = float(value)
    return parsed if math.isfinite(parsed) else -1.0e9


def canonical_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    normalized.setdefault("rs_method", "relative_strength")
    normalized.setdefault("hurdle", 1.25)
    normalized.setdefault("st_period", None)
    normalized.setdefault("st_multiplier", None)
    normalized["entry"] = str(normalized.get("entry"))
    normalized["market_filter"] = str(normalized.get("market_filter"))
    normalized["asset_filter"] = str(normalized.get("asset_filter"))
    normalized["rs_method"] = str(normalized.get("rs_method"))
    normalized["rs_period"] = int(float(normalized.get("rs_period")))
    normalized["sell_confirm_bars"] = int(float(normalized.get("sell_confirm_bars")))
    normalized["hurdle"] = round(float(normalized.get("hurdle")), 8)
    normalized["max_positions"] = int(float(normalized.get("max_positions")))
    for field in ("st_period", "st_multiplier"):
        value = normalized.get(field)
        if value is None or (isinstance(value, float) and math.isnan(value)) or str(value) in {"", "nan", "None"}:
            normalized[field] = None
    if normalized["st_period"] is not None:
        normalized["st_period"] = int(float(normalized["st_period"]))
    if normalized["st_multiplier"] is not None:
        normalized["st_multiplier"] = round(float(normalized["st_multiplier"]), 8)
    return {field: normalized.get(field) for field in PARAM_FIELDS}


def param_key(params: dict[str, Any]) -> str:
    return json.dumps(canonical_params(params), sort_keys=True, separators=(",", ":"))


def row_from_flat(raw: dict[str, Any], source: str) -> dict[str, Any] | None:
    try:
        params = {
            key.removeprefix("param_"): value
            for key, value in raw.items()
            if key.startswith("param_")
        }
        params = canonical_params(params)
        metrics = {
            "total_return": float(raw.get("total_return", 0.0)),
            "mdd": float(raw.get("mdd", 0.0)),
            "cagr": float(raw.get("cagr", 0.0)),
            "calmar": float(raw.get("calmar", 0.0)),
            "sharpe": float(raw.get("sharpe", 0.0)),
            "sortino": float(raw.get("sortino", 0.0)),
            "win_rate": float(raw.get("win_rate", 0.0)),
            "payoff_ratio": float(raw.get("payoff_ratio", 0.0)),
            "trade_count": int(float(raw.get("trade_count", 0))),
        }
        return {
            "params": params,
            "metrics": metrics,
            "score": float(raw.get("score", 0.0)),
            "qqq_return": float(raw.get("qqq_return", 0.0)),
            "alpha": float(raw.get("alpha", 0.0)),
            "start": str(raw.get("start", "")),
            "end": str(raw.get("end", "")),
            "source": source,
        }
    except (TypeError, ValueError, KeyError):
        return None


def flat_row_with_source(row: dict[str, Any]) -> dict[str, Any]:
    out = flat_row(row)
    out["source"] = row.get("source", "optuna")
    return out


class OptunaBacktestRunner:
    def __init__(self, args: argparse.Namespace, run_dir: Path):
        self.args = args
        self.run_dir = run_dir
        self.cache: dict[str, dict[str, Any]] = {}
        self.run_keys: set[str] = set()
        self.feature_cache: dict[str, dict[str, Any]] = {}
        self.group_cache: dict[str, dict[str, Any]] = {}
        self.exit_cache: dict[tuple[str, int], dict[str, list[bool]]] = {}
        self.qqq_return_cache: dict[tuple[object, object], float] = {}
        self.new_backtests = 0
        self.cache_hits = 0
        self.trial_counter = 0

        base = load_split_config(args.strategy, args.runtime)
        self.base = base.__class__(**{**base.__dict__, "period": args.period, "timeframe": "1d"})
        print("[optuna] Downloading shared market data...", flush=True)
        self.market_data = download_for_config(self.base)
        print("[optuna] Preparing market timeline...", flush=True)
        self.full_idx = market_index(self.market_data)
        self.requested_idx = run_index_from_start(self.full_idx, args.start)
        self.active_by_position = prepare_active_universe(self.market_data, self.full_idx)
        print(
            f"[optuna] Market timeline: {self.full_idx[0]} -> {self.full_idx[-1]} "
            f"({len(self.full_idx)} bars), requested bars={len(self.requested_idx)}",
            flush=True,
        )

    def load_cache_files(self, paths: list[Path], *, source: str) -> int:
        count = 0
        for path in paths:
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            for raw in frame.to_dict("records"):
                row = row_from_flat(raw, source)
                if row is None:
                    continue
                self.cache[param_key(row["params"])] = row
                count += 1
        return count

    def load_run_results(self) -> int:
        path = self.run_dir / "all_results.csv"
        if not path.exists():
            return 0
        count = self.load_cache_files([path], source="resume")
        frame = pd.read_csv(path)
        for raw in frame.to_dict("records"):
            row = row_from_flat(raw, "resume")
            if row is not None:
                self.run_keys.add(param_key(row["params"]))
        return count

    def group_context(self, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        group_params = {
            "entry": params["entry"],
            "market_filter": params["market_filter"],
            "asset_filter": params["asset_filter"],
            "rs_method": params["rs_method"],
            "rs_period": params["rs_period"],
        }
        if params["entry"] in {"single", "single_supertrend", "supertrend"}:
            group_params["st_period"] = params["st_period"]
            group_params["st_multiplier"] = params["st_multiplier"]
        key = json.dumps(group_params, sort_keys=True, separators=(",", ":"))
        if key in self.group_cache:
            return key, self.group_cache[key]

        group_number = len(self.group_cache) + 1
        print(f"[optuna] Prepare group {group_number}: {group_params}", flush=True)
        group_config = apply_config_overlay(self.base, group_params)
        strategy = create_strategy(group_config)
        features = self.feature_context(group_config, group_params)
        prepared_template = SimpleNamespace(
            prepared=strategy.scorer.add_scores(features["featured"], self.market_data.benchmark),
            market_filter_trends=features["market_filter_trends"],
        )
        row_positions = prepare_row_positions(prepared_template.prepared, self.full_idx)
        market_filter_states = prepare_market_filter_states(prepared_template, self.full_idx)
        candidates_by_position = prepare_candidate_lists(
            group_config,
            strategy,
            prepared_template.prepared,
            market_filter_states,
            self.active_by_position,
            row_positions,
            self.full_idx,
        )
        context = {
            "group_config": group_config,
            "strategy": strategy,
            "prepared": prepared_template.prepared,
            "row_positions": row_positions,
            "market_filter_states": market_filter_states,
            "candidates_by_position": candidates_by_position,
        }
        self.group_cache[key] = context
        return key, context

    def feature_context(self, config: AppConfig, group_params: dict[str, Any]) -> dict[str, Any]:
        feature_params = {
            "entry": group_params["entry"],
            "market_filter": group_params["market_filter"],
            "asset_filter": group_params["asset_filter"],
        }
        if group_params["entry"] in {"single", "single_supertrend", "supertrend"}:
            feature_params["st_period"] = group_params.get("st_period")
            feature_params["st_multiplier"] = group_params.get("st_multiplier")
        key = json.dumps(feature_params, sort_keys=True, separators=(",", ":"))
        cached = self.feature_cache.get(key)
        if cached is not None:
            return cached

        feature_number = len(self.feature_cache) + 1
        print(f"[optuna] Prepare features {feature_number}: {feature_params}", flush=True)
        featured = {
            symbol: with_strategy_components(config, symbol, frame)
            for symbol, frame in self.market_data.bars.items()
        }
        market_filter_data = (
            self.market_data.filter_benchmark
            if self.market_data.filter_benchmark is not None
            else self.market_data.benchmark
        )
        context = {
            "featured": featured,
            "market_filter_trends": precompute_market_filter_trends(
                config,
                list(self.market_data.bars),
                market_filter_data,
            ),
        }
        self.feature_cache[key] = context
        return context

    def evaluate(self, params: dict[str, Any]) -> dict[str, Any]:
        params = canonical_params(params)
        key = param_key(params)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            self.run_keys.add(key)
            return cached

        group_key, context = self.group_context(params)
        sell_confirm = int(params["sell_confirm_bars"])
        exit_key = (group_key, sell_confirm)
        if exit_key not in self.exit_cache:
            confirm_config = apply_config_overlay(
                context["group_config"],
                {"sell_confirm_bars": sell_confirm},
            )
            self.exit_cache[exit_key] = prepare_exit_down_states(
                confirm_config,
                context["prepared"],
                self.full_idx,
                context["row_positions"],
            )
        config = apply_config_overlay(
            context["group_config"],
            {
                "sell_confirm_bars": sell_confirm,
                "hurdle": params["hurdle"],
                "max_positions": params["max_positions"],
            },
        )
        result = run_prepared_backtest(
            config,
            self.market_data,
            context["prepared"],
            context["candidates_by_position"],
            context["market_filter_states"],
            self.exit_cache[exit_key],
            self.active_by_position,
            context["row_positions"],
            context["strategy"].warmup_bars(),
            self.requested_idx,
        )
        qqq_return = qqq_return_for_index(self.market_data, result.equity.index, self.qqq_return_cache)
        total_return = float(result.metrics.get("total_return", 0.0))
        row = {
            "params": params,
            "metrics": result.metrics,
            "score": score_metrics(result.metrics),
            "qqq_return": qqq_return,
            "alpha": total_return - qqq_return,
            "start": result.equity.index[0],
            "end": result.equity.index[-1],
            "source": "optuna",
        }
        self.cache[key] = row
        self.run_keys.add(key)
        self.new_backtests += 1
        return row

    def rows_for_report(self) -> list[dict[str, Any]]:
        return [self.cache[key] for key in sorted(self.run_keys) if key in self.cache]

    def write_outputs(self, objectives: list[str], metadata: dict[str, Any]) -> None:
        rows = self.rows_for_report()
        if rows:
            pd.DataFrame(flat_row_with_source(row) for row in rows).to_csv(
                self.run_dir / "all_results.csv",
                index=False,
            )
        ranking_specs = {
            "return": "Top 5 by Return",
            "alpha": "Top 5 by Alpha",
            "mdd": "Top 5 by MDD",
            "sharpe": "Top 5 by Sharpe",
            "win_rate": "Top 5 by Win Rate",
            "payoff": "Top 5 by Payoff",
        }
        top_by_metric = {
            metric: ordered_rows(rows, metric)
            for metric in ranking_specs
            if metric in objectives and rows
        }
        report_lines = [
            "Nasdaq-100 Daily Optuna Results",
            f"Requested    : start={self.args.start}, period={self.args.period}",
            f"Objectives   : {','.join(objectives)}",
            f"Unique rows  : {len(rows)}",
            f"New backtests: {self.new_backtests}",
            f"Cache hits   : {self.cache_hits}",
            f"Entries      : {self.args.entries}",
            f"Market filter: {self.args.market_filters}",
            f"Asset filters: {self.args.asset_filters}",
            f"RS methods   : {self.args.rs_methods}",
            f"RS periods   : {self.args.rs_periods}",
            f"Hurdles      : {self.args.hurdles}",
            f"Sell confirm : {self.args.sell_confirm_bars}",
            f"Max positions: {self.args.max_positions}",
        ]
        for metric, title in ranking_specs.items():
            if metric in top_by_metric:
                report_lines.extend(result_table_lines(title, top_by_metric[metric], self.args.top))
        (self.run_dir / "report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        summary = {
            "metadata": metadata,
            "top": {
                metric: [flat_row_with_source(row) for row in selected[: self.args.top]]
                for metric, selected in top_by_metric.items()
            },
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def suggest_params(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, Any]:
    entry = trial.suggest_categorical("entry", list(csv_values(args.entries)))
    params: dict[str, Any] = {
        "entry": entry,
        "market_filter": trial.suggest_categorical("market_filter", list(csv_values(args.market_filters))),
        "asset_filter": trial.suggest_categorical("asset_filter", list(csv_values(args.asset_filters))),
        "rs_method": trial.suggest_categorical("rs_method", list(csv_values(args.rs_methods))),
        "rs_period": trial.suggest_categorical("rs_period", list(csv_values(args.rs_periods, int))),
        "sell_confirm_bars": trial.suggest_categorical(
            "sell_confirm_bars",
            list(csv_values(args.sell_confirm_bars, int)),
        ),
        "hurdle": trial.suggest_categorical("hurdle", list(csv_values(args.hurdles, float))),
        "max_positions": trial.suggest_categorical("max_positions", list(csv_values(args.max_positions, int))),
    }
    if entry in {"single", "single_supertrend", "supertrend"}:
        params["st_period"] = trial.suggest_categorical("st_period", list(csv_values(args.st_periods, int)))
        params["st_multiplier"] = trial.suggest_categorical(
            "st_multiplier",
            list(csv_values(args.st_multipliers, float)),
        )
    return canonical_params(params)


def completed_trials(study: optuna.Study) -> int:
    return len([trial for trial in study.trials if trial.state == TrialState.COMPLETE])


def main() -> None:
    args = build_parser().parse_args()
    objectives = [item.strip() for item in args.objectives.split(",") if item.strip()]
    unknown = set(objectives) - set(OBJECTIVE_METRICS)
    if unknown:
        raise ValueError(f"Unsupported objectives: {', '.join(sorted(unknown))}")

    run_id = args.run_id.strip() or datetime.now().strftime("optuna_%Y%m%d_%H%M%S")
    run_dir = Path(args.results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    runner = OptunaBacktestRunner(args, run_dir)
    resumed = runner.load_run_results()
    import_paths = [Path(path) for path in args.import_results]
    if args.auto_import_searches:
        import_paths.extend(
            sorted(
                (PLAYGROUND_ROOT / "results" / "research" / "us_nasdaq100_rolling" / "searches").glob(
                    "*/all_results.csv"
                )
            )
        )
    imported = runner.load_cache_files(import_paths, source="imported")

    search_space_groups = group_parameter_grid(args)
    search_space_size = (
        len(search_space_groups)
        * len(csv_values(args.sell_confirm_bars, int))
        * len(csv_values(args.hurdles, float))
        * len(csv_values(args.max_positions, int))
    )
    metadata = {
        "run_id": run_id,
        "search_space_size": search_space_size,
        "group_count": len(search_space_groups),
        "objectives": objectives,
        "n_trials_per_objective": args.n_trials,
        "timeout_minutes": args.timeout_minutes,
        "resumed_rows": resumed,
        "imported_rows": imported,
        "storage": str(run_dir / "optuna.sqlite3"),
    }
    print("Nasdaq-100 Daily Optuna Search")
    print(f"Run dir      : {run_dir}")
    print(f"Search space : {search_space_size} candidates ({len(search_space_groups)} prep groups)")
    print(f"Objectives   : {','.join(objectives)}")
    print(f"Trials       : +{args.n_trials} per objective")
    print(f"Timeout      : {args.timeout_minutes} minutes")
    print(f"Resume rows  : {resumed}")
    print(f"Import rows  : {imported}")

    storage = f"sqlite:///{(run_dir / 'optuna.sqlite3').as_posix()}"
    studies = {
        objective: optuna.create_study(
            study_name=f"nasdaq100_{objective}",
            direction="maximize",
            storage=storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed + index, n_startup_trials=10),
        )
        for index, objective in enumerate(objectives)
    }
    targets = {
        objective: completed_trials(study) + max(0, int(args.n_trials))
        for objective, study in studies.items()
    }

    deadline = time.monotonic() + max(0.0, float(args.timeout_minutes)) * 60.0
    started = time.monotonic()

    def make_objective(metric: str):
        def objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial, args)
            before_new = runner.new_backtests
            row = runner.evaluate(params)
            value = metric_value(row, metric, args.min_trades)
            runner.trial_counter += 1
            source = "new" if runner.new_backtests > before_new else "cached"
            print(
                f"[optuna] {metric:<8} trial={runner.trial_counter} "
                f"value={value:.6f} source={source} params={params}",
                flush=True,
            )
            if args.save_every and runner.trial_counter % args.save_every == 0:
                runner.write_outputs(objectives, metadata)
            return value

        return objective

    try:
        while time.monotonic() < deadline:
            progressed = False
            for metric, study in studies.items():
                if completed_trials(study) >= targets[metric]:
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                study.optimize(make_objective(metric), n_trials=1, timeout=remaining)
                progressed = True
            if not progressed:
                break
    finally:
        metadata["elapsed_seconds"] = time.monotonic() - started
        metadata["new_backtests"] = runner.new_backtests
        metadata["cache_hits"] = runner.cache_hits
        metadata["unique_rows"] = len(runner.run_keys)
        runner.write_outputs(objectives, metadata)

    print("[optuna] Finished")
    print(f"[optuna] Unique rows  : {len(runner.run_keys)}")
    print(f"[optuna] New backtests: {runner.new_backtests}")
    print(f"[optuna] Cache hits   : {runner.cache_hits}")
    print(f"[optuna] Saved results: {run_dir}")


if __name__ == "__main__":
    main()
