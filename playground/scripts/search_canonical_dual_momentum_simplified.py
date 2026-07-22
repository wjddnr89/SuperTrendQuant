from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAYGROUND_ROOT = PROJECT_ROOT / "playground"
UNIFIED_ROOT = PROJECT_ROOT / "unified_quant"
sys.path.insert(0, str(UNIFIED_ROOT / "src"))

# Importing this module registers the Playground dual-momentum scorer with the
# canonical strategy registry.
from compare_nasdaq100_universes_same_prices import DualMomentumScorer  # noqa: E402,F401
from supertrend_quant.config import load_split_config  # noqa: E402
from supertrend_quant.data import market_index  # noqa: E402
from supertrend_quant.research.data_resolver import download_for_config  # noqa: E402
from supertrend_quant.research.overlays import apply_config_overlay  # noqa: E402
from supertrend_quant.runners import _prepare_backtest, run_backtest_on_data  # noqa: E402
from supertrend_quant.strategies import create_strategy  # noqa: E402
from supertrend_quant.strategies.leader_rotation import PreparedLeaderBacktest  # noqa: E402


DEFAULT_STRATEGY = PLAYGROUND_ROOT / "configs" / "strategies" / "leader_rotation.yaml"
DEFAULT_RUNTIME = PLAYGROUND_ROOT / "configs" / "runtimes" / "research_us_nasdaq100_rolling.yaml"
DEFAULT_LEGACY = PLAYGROUND_ROOT / "configs" / "canonical_dual_momentum_legacy_base.json"
DEFAULT_RESULTS = (
    PLAYGROUND_ROOT
    / "results"
    / "research"
    / "us_nasdaq100_rolling"
    / "canonical_searches"
)


def csv_values(raw: str, cast):
    return tuple(cast(value.strip()) for value in raw.split(",") if value.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable simplified dual-momentum search on index_events and the canonical runner."
    )
    parser.add_argument("--strategy", default=str(DEFAULT_STRATEGY))
    parser.add_argument("--runtime", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--legacy-config", default=str(DEFAULT_LEGACY))
    parser.add_argument("--start", default="2015-10-19")
    parser.add_argument("--end", default="2026-07-15")
    parser.add_argument("--rs-periods", default="100,125,150,175,200")
    parser.add_argument("--hurdles", default="1.0,1.5,2.0")
    parser.add_argument(
        "--asset-filters",
        default="none,ema_trend,ichimoku_cloud,ichimoku_cloud+ema_trend",
    )
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--sell-confirm-bars", type=int, default=1)
    parser.add_argument("--st-period", type=int, default=10)
    parser.add_argument("--st-multiplier", type=float, default=3.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--run-id", default="canonical_simplified_grid_20260722")
    return parser


def base_canonical_config(args: argparse.Namespace):
    base = load_split_config(args.strategy, args.runtime)
    base = replace(
        base,
        period="max",
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
    return replace(base, universe=universe, universe_file="", symbols=())


def config_for(base, params: dict[str, Any]):
    config = apply_config_overlay(
        base,
        {
            "entry": "single",
            "market_filter": "1d",
            "asset_filter": params["asset_filter"],
            "sell_confirm_bars": params["sell_confirm_bars"],
            "max_positions": params["max_positions"],
            "st_period": params["st_period"],
            "st_multiplier": params["st_multiplier"],
            "fee_rate": params["fee_rate"],
            "slippage_rate": params["slippage_rate"],
        },
    )
    return replace(
        config,
        scoring=replace(
            config.scoring,
            type="dual_momentum",
            params={"lookback_bars": int(params["rs_period"])},
        ),
        leader_rotation=replace(
            config.leader_rotation,
            hurdle_atr_mult=float(params["hurdle"]),
        ),
    )


def run_key(params: dict[str, Any]) -> str:
    if params["kind"] == "legacy_base":
        return "legacy_base"
    if str(params["kind"]).startswith("ablation_"):
        return str(params["kind"])
    asset = str(params["asset_filter"]).replace("+", "_plus_")
    hurdle = str(params["hurdle"]).replace(".", "p")
    return (
        f"filter-{asset}_rs-{params['rs_period']}_h-{hurdle}"
        f"_pos-{params['max_positions']}_sell-{params['sell_confirm_bars']}"
    )


def search_combinations(args: argparse.Namespace) -> list[dict[str, Any]]:
    common = {
        "rs_method": "dual_momentum",
        "sell_confirm_bars": args.sell_confirm_bars,
        "max_positions": args.max_positions,
        "st_period": args.st_period,
        "st_multiplier": args.st_multiplier,
        "fee_rate": args.fee_rate,
        "slippage_rate": args.slippage_rate,
    }
    combinations = []
    for asset_filter, rs_period, hurdle in itertools.product(
        csv_values(args.asset_filters, str),
        csv_values(args.rs_periods, int),
        csv_values(args.hurdles, float),
    ):
        combinations.append(
            {
                **common,
                "kind": "simplified_grid",
                "asset_filter": asset_filter,
                "rs_period": rs_period,
                "hurdle": hurdle,
            }
        )

    legacy = json.loads(Path(args.legacy_config).read_text(encoding="utf-8"))
    combinations.append(
        {
            "kind": "legacy_base",
            "rs_method": "dual_momentum",
            "asset_filter": legacy["asset_filter"],
            "rs_period": int(legacy["rs_period"]),
            "sell_confirm_bars": int(legacy["sell_confirm_bars"]),
            "hurdle": float(legacy["hurdle"]),
            "max_positions": int(legacy["max_positions"]),
            "st_period": int(legacy["st_period"]),
            "st_multiplier": float(legacy["st_multiplier"]),
            "fee_rate": float(legacy["fee_rate"]),
            "slippage_rate": float(legacy["slippage_rate"]),
        }
    )
    for kind, sell_confirm_bars, max_positions in (
        ("ablation_sell1_pos1", 1, 1),
        ("ablation_sell5_pos3", 5, 3),
    ):
        combinations.append(
            {
                "kind": kind,
                "rs_method": "dual_momentum",
                "asset_filter": legacy["asset_filter"],
                "rs_period": int(legacy["rs_period"]),
                "sell_confirm_bars": sell_confirm_bars,
                "hurdle": float(legacy["hurdle"]),
                "max_positions": max_positions,
                "st_period": int(legacy["st_period"]),
                "st_multiplier": float(legacy["st_multiplier"]),
                "fee_rate": float(legacy["fee_rate"]),
                "slippage_rate": float(legacy["slippage_rate"]),
            }
        )
    return combinations


def benchmark_return(data, index: pd.Index) -> float:
    frames = getattr(data, "benchmark", None) or {}
    frame = next((item for item in frames.values() if item is not None and not item.empty), None)
    if frame is None or "Close" not in frame:
        return float("nan")
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    selected = close.loc[(close.index >= index[0]) & (close.index <= index[-1])]
    if len(selected) < 2:
        return float("nan")
    return float(selected.iloc[-1] / selected.iloc[0] - 1.0)


def annual_stats(equity: pd.Series) -> dict[str, Any]:
    daily = equity.astype(float).pct_change().dropna()
    annual = (1.0 + daily).groupby(daily.index.year).prod() - 1.0
    values = [float(value) for value in annual if math.isfinite(float(value))]
    return {
        "annual_returns": {str(year): float(value) for year, value in annual.items()},
        "worst_year_return": min(values) if values else float("nan"),
        "median_year_return": float(pd.Series(values).median()) if values else float("nan"),
        "positive_year_ratio": (
            sum(value > 0.0 for value in values) / len(values) if values else float("nan")
        ),
    }


def rescore_prepared_frames(
    prepared: dict[str, pd.DataFrame],
    benchmark,
    lookback_bars: int,
) -> None:
    """Update only dual-momentum scores while retaining shared indicator columns."""
    for symbol, frame in prepared.items():
        symbol_benchmark = benchmark.get(symbol) if isinstance(benchmark, dict) else benchmark
        if (
            "Close" not in frame
            or symbol_benchmark is None
            or symbol_benchmark.empty
            or "Close" not in symbol_benchmark
        ):
            frame["Score"] = float("nan")
            continue
        benchmark_return = symbol_benchmark["Close"].pct_change(
            lookback_bars,
            fill_method=None,
        ).reindex(frame.index, method="ffill")
        if "IdentitySegment" in frame and frame["IdentitySegment"].nunique(dropna=False) > 1:
            stock_return = frame.groupby(
                "IdentitySegment",
                sort=False,
                dropna=False,
            )["Close"].transform(
                lambda values: values.pct_change(lookback_bars, fill_method=None)
            )
        else:
            stock_return = frame["Close"].pct_change(lookback_bars, fill_method=None)
        excess = stock_return - benchmark_return
        frame["Score"] = excess.where((stock_return > 0.0) & (excess > 0.0))


def result_row(key: str, params: dict[str, Any], result, qqq_return: float) -> dict[str, Any]:
    metrics = result.metrics
    stability = annual_stats(result.equity)
    return {
        "run_key": key,
        **params,
        "start": str(result.equity.index[0]),
        "end": str(result.equity.index[-1]),
        "initial_equity": float(result.equity.iloc[0]),
        "final_equity": float(result.equity.iloc[-1]),
        "total_return": float(metrics["total_return"]),
        "cagr": float(metrics["cagr"]),
        "qqq_return": qqq_return,
        "alpha": float(metrics["total_return"]) - qqq_return,
        "mdd": float(metrics["mdd"]),
        "sharpe": float(metrics["sharpe"]),
        "calmar": float(metrics["calmar"]),
        "sortino": float(metrics["sortino"]),
        "win_rate": float(metrics["win_rate"]),
        "payoff_ratio": float(metrics["payoff_ratio"]),
        "trade_count": int(metrics["trade_count"]),
        "corporate_action_cash": float(result.corporate_action_cash),
        "data_quality": result.data_quality,
        "data_version": result.data_version,
        "price_mode": result.price_mode,
        "worst_year_return": stability["worst_year_return"],
        "median_year_return": stability["median_year_return"],
        "positive_year_ratio": stability["positive_year_ratio"],
        "annual_returns": stability["annual_returns"],
    }


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["run_key"])] = row
    return rows


def flat_rows(rows: dict[str, dict[str, Any]]) -> pd.DataFrame:
    flattened = []
    for row in rows.values():
        flattened.append({key: value for key, value in row.items() if key != "annual_returns"})
    return pd.DataFrame(flattened)


def format_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:+.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def report_text(
    rows: dict[str, dict[str, Any]],
    *,
    total: int,
    start: str,
    end: str,
) -> str:
    frame = flat_rows(rows)
    lines = [
        "Canonical Dual-Momentum Simplified Search",
        f"Period: {start} -> {end}",
        "Universe: Nasdaq-100 index_events",
        "Runner: canonical (raw next-session open + corporate-action ledger)",
        "Fixed: dual_momentum, sell_confirm=1, max_positions=3, ST=10/3.0",
        "Search: rs_period x hurdle x asset_filter",
        f"Completed: {len(frame)}/{total}",
        "",
    ]
    if frame.empty:
        return "\n".join(lines)

    legacy = frame.loc[frame["kind"] == "legacy_base"]
    if not legacy.empty:
        row = legacy.iloc[0]
        lines.extend(
            [
                "Legacy base control",
                f"  Return {format_pct(row['total_return'])} | CAGR {format_pct(row['cagr'])} "
                f"| MDD {format_pct(row['mdd'])} | Sharpe {row['sharpe']:.2f}",
                "",
            ]
        )

    ablation_keys = (
        "legacy_base",
        "ablation_sell1_pos1",
        "ablation_sell5_pos3",
        "filter-ichimoku_cloud_plus_ema_trend_rs-150_h-2p0_pos-3_sell-1",
    )
    ablation_labels = {
        "legacy_base": "sell=5, positions=1",
        "ablation_sell1_pos1": "sell=1, positions=1",
        "ablation_sell5_pos3": "sell=5, positions=3",
        "filter-ichimoku_cloud_plus_ema_trend_rs-150_h-2p0_pos-3_sell-1": (
            "sell=1, positions=3"
        ),
    }
    controls = frame.set_index("run_key", drop=False)
    if all(key in controls.index for key in ablation_keys):
        lines.append("Sell confirmation x position-count ablation")
        for key in ablation_keys:
            row = controls.loc[key]
            lines.append(
                f"  {ablation_labels[key]} | Return {format_pct(row['total_return'])} "
                f"| CAGR {format_pct(row['cagr'])} | MDD {format_pct(row['mdd'])} "
                f"| Sharpe {row['sharpe']:.2f}"
            )
        lines.append("")

    rankings = (
        ("Return", "total_return", False),
        ("MDD", "mdd", False),
        ("Sharpe", "sharpe", False),
        ("Calmar", "calmar", False),
        ("Win Rate", "win_rate", False),
        ("Payoff", "payoff_ratio", False),
        ("Worst Year", "worst_year_return", False),
    )
    grid = frame.loc[frame["kind"] == "simplified_grid"].copy()
    for title, column, ascending in rankings:
        lines.append(f"Top 5 by {title}")
        ranked = grid.sort_values(column, ascending=ascending).head(5)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            lines.append(
                f"  {rank}. {row['run_key']} | Return {format_pct(row['total_return'])} "
                f"| CAGR {format_pct(row['cagr'])} | MDD {format_pct(row['mdd'])} "
                f"| Sharpe {row['sharpe']:.2f} | Calmar {row['calmar']:.2f} "
                f"| Win {format_pct(row['win_rate'])} | Payoff {row['payoff_ratio']:.2f}"
            )
        lines.append("")
    return "\n".join(lines)


def save_aggregate(
    run_dir: Path,
    rows: dict[str, dict[str, Any]],
    *,
    total: int,
    start: str,
    end: str,
) -> None:
    frame = flat_rows(rows)
    if not frame.empty:
        frame.sort_values("run_key").to_csv(run_dir / "all_results.csv", index=False)
    summary = {
        "completed": len(rows),
        "total": total,
        "start": start,
        "end": end,
        "rows": list(rows.values()),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (run_dir / "report.txt").write_text(
        report_text(rows, total=total, start=start, end=end),
        encoding="utf-8",
    )


def save_run(run_dir: Path, row: dict[str, Any], result) -> None:
    target = run_dir / "runs" / str(row["run_key"])
    target.mkdir(parents=True, exist_ok=True)
    result.equity.rename("equity").to_frame().to_csv(target / "equity.csv")
    pd.DataFrame(result.trade_records).to_csv(target / "trades.csv", index=False)
    (target / "summary.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    combinations = search_combinations(args)
    run_dir = Path(args.results_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.jsonl"
    completed = load_checkpoint(checkpoint_path)

    experiment = {
        "created_at": datetime.now().isoformat(),
        "legacy_config": str(Path(args.legacy_config).resolve()),
        "universe": "index_events:nasdaq100",
        "runner": "canonical",
        "requested_start": args.start,
        "requested_end": args.end,
        "combinations": [{"run_key": run_key(item), **item} for item in combinations],
    }
    experiment_path = run_dir / "experiment.json"
    experiment_path.write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    base_config = base_canonical_config(args)
    print("[canonical-search] loading canonical index_events data once...", flush=True)
    data = download_for_config(base_config, allow_stale=True)
    full_index = market_index(data)
    run_index = full_index[
        (full_index >= pd.Timestamp(args.start)) & (full_index <= pd.Timestamp(args.end))
    ]
    if len(run_index) < 2:
        raise RuntimeError("Requested search period has fewer than two canonical sessions.")

    total = len(combinations)
    print(
        f"[canonical-search] sessions {run_index[0]} -> {run_index[-1]} ({len(run_index)})",
        flush=True,
    )
    print(
        f"[canonical-search] {len(completed)}/{total} already complete; results={run_dir}",
        flush=True,
    )

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for params in combinations:
        groups[
            (
                params["rs_period"],
                params["st_period"],
                params["st_multiplier"],
            )
        ].append(params)

    first_pending = next(
        (params for params in combinations if run_key(params) not in completed),
        None,
    )
    shared_prepared = None
    if first_pending is not None:
        preparation_params = dict(first_pending)
        preparation_params["asset_filter"] = "ichimoku_cloud+ema_trend"
        preparation_config = config_for(base_config, preparation_params)
        print(
            "[canonical-search] preparing shared ST + EMA + Ichimoku indicators once...",
            flush=True,
        )
        shared_prepared = _prepare_backtest(create_strategy(preparation_config), data)
        if not isinstance(shared_prepared, PreparedLeaderBacktest):
            raise TypeError("Expected PreparedLeaderBacktest from canonical leader_rotation strategy.")

    session_start = time.monotonic()
    newly_completed = 0
    pending_total = sum(run_key(params) not in completed for params in combinations)
    for group_number, (group_key, group_params) in enumerate(groups.items(), start=1):
        pending = [params for params in group_params if run_key(params) not in completed]
        if not pending:
            continue
        print(
            f"[canonical-search] score group {group_number}/{len(groups)} rs={group_key[0]}",
            flush=True,
        )
        if shared_prepared is None:
            raise RuntimeError("Shared prepared data was not initialized.")
        rescore_prepared_frames(
            shared_prepared.prepared,
            data.benchmark,
            int(group_key[0]),
        )

        for params in pending:
            key = run_key(params)
            config = config_for(base_config, params)
            strategy = create_strategy(config)
            prepared = PreparedLeaderBacktest(
                strategy,
                shared_prepared.prepared,
                shared_prepared.market_filter_trends,
                shared_prepared.universe_schedule,
            )
            run_started = time.monotonic()
            result = run_backtest_on_data(
                config,
                data,
                run_index=run_index,
                prepared_backtest=prepared,
            )
            qqq = benchmark_return(data, result.equity.index)
            row = result_row(key, params, result, qqq)
            save_run(run_dir, row, result)
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                handle.flush()
            completed[key] = row
            newly_completed += 1
            save_aggregate(
                run_dir,
                completed,
                total=total,
                start=str(run_index[0]),
                end=str(run_index[-1]),
            )

            elapsed = time.monotonic() - session_start
            average = elapsed / newly_completed
            remaining = max(pending_total - newly_completed, 0)
            eta_minutes = average * remaining / 60.0
            run_seconds = time.monotonic() - run_started
            print(
                f"[canonical-search] [{len(completed)}/{total}] {key} "
                f"return={format_pct(row['total_return'])} mdd={format_pct(row['mdd'])} "
                f"sharpe={row['sharpe']:.2f} run={run_seconds:.1f}s "
                f"session_eta={eta_minutes:.1f}m",
                flush=True,
            )

    save_aggregate(
        run_dir,
        completed,
        total=total,
        start=str(run_index[0]),
        end=str(run_index[-1]),
    )
    print(f"[canonical-search] complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
