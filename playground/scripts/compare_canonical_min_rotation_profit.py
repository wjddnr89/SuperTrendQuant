from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAYGROUND_ROOT = PROJECT_ROOT / "playground"
UNIFIED_ROOT = PROJECT_ROOT / "unified_quant"
sys.path.insert(0, str(UNIFIED_ROOT / "src"))

from search_canonical_dual_momentum_simplified import (  # noqa: E402
    DEFAULT_LEGACY,
    DEFAULT_RUNTIME,
    DEFAULT_STRATEGY,
    annual_stats,
    base_canonical_config,
    benchmark_return,
    config_for,
)
from supertrend_quant.data import market_index  # noqa: E402
from supertrend_quant.research.data_resolver import download_for_config  # noqa: E402
from supertrend_quant.runners import _prepare_backtest, run_backtest_on_data  # noqa: E402
from supertrend_quant.strategies import create_strategy  # noqa: E402
from supertrend_quant.strategies.leader_rotation import PreparedLeaderBacktest  # noqa: E402


DEFAULT_RESULTS = (
    PLAYGROUND_ROOT
    / "results"
    / "research"
    / "us_nasdaq100_rolling"
    / "rotation_profit_ablation"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare min-rotation-profit gates on the canonical sell=1 base."
    )
    parser.add_argument("--strategy", default=str(DEFAULT_STRATEGY))
    parser.add_argument("--runtime", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--legacy-config", default=str(DEFAULT_LEGACY))
    parser.add_argument("--start", default="2015-10-19")
    parser.add_argument("--end", default="2026-07-15")
    parser.add_argument("--thresholds", default="-1.0,0.0,0.01")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--run-id", default="sell1_min_rotation_profit_20260722")
    return parser


def fixed_params(args: argparse.Namespace) -> dict[str, Any]:
    legacy = json.loads(Path(args.legacy_config).read_text(encoding="utf-8"))
    return {
        "rs_method": "dual_momentum",
        "asset_filter": legacy["asset_filter"],
        "rs_period": int(legacy["rs_period"]),
        "sell_confirm_bars": 1,
        "hurdle": float(legacy["hurdle"]),
        "max_positions": 1,
        "st_period": int(legacy["st_period"]),
        "st_multiplier": float(legacy["st_multiplier"]),
        "fee_rate": float(legacy["fee_rate"]),
        "slippage_rate": float(legacy["slippage_rate"]),
    }


def variants(raw: str | None = None) -> tuple[dict[str, Any], ...]:
    if raw is None:
        return (
        {
            "run_key": "no_gate",
            "label": "No profit gate",
            "min_rotation_profit_pct": -1.0,
        },
        {
            "run_key": "breakeven_gate",
            "label": "Rotate only at breakeven or better",
            "min_rotation_profit_pct": 0.0,
        },
        {
            "run_key": "one_percent_gate",
            "label": "Rotate only at +1% or better (current)",
            "min_rotation_profit_pct": 0.01,
        },
    )
    known = {
        -1.0: ("no_gate", "No profit gate"),
        0.0: ("breakeven_gate", "Rotate only at breakeven or better"),
        0.01: ("one_percent_gate", "Rotate only at +1% or better (current)"),
    }
    output = []
    for item in raw.split(","):
        threshold = float(item.strip())
        if threshold in known:
            run_key, label = known[threshold]
        else:
            magnitude = f"{abs(threshold) * 100:g}".replace(".", "p")
            direction = "minus" if threshold < 0 else "plus"
            run_key = f"{direction}_{magnitude}pct_gate"
            label = f"Rotate at {threshold * 100:+g}% or better"
        output.append(
            {
                "run_key": run_key,
                "label": label,
                "min_rotation_profit_pct": threshold,
            }
        )
    return tuple(output)


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["run_key"])] = row
    return rows


def trade_stats(result) -> dict[str, Any]:
    records = [dict(record) for record in result.trade_records]
    rotations = [record for record in records if record.get("exit_reason") == "Leader rotation"]
    supertrend = [record for record in records if record.get("exit_reason") == "Supertrend down"]
    rotation_pnl = [float(record["pnl_pct"]) for record in rotations]
    return {
        "rotation_exit_count": len(rotations),
        "supertrend_exit_count": len(supertrend),
        "rotation_loss_count": sum(value < 0.0 for value in rotation_pnl),
        "average_rotation_pnl": (
            sum(rotation_pnl) / len(rotation_pnl) if rotation_pnl else float("nan")
        ),
        "worst_rotation_pnl": min(rotation_pnl) if rotation_pnl else float("nan"),
    }


def result_row(variant, params, result, qqq_return: float) -> dict[str, Any]:
    metrics = result.metrics
    stability = annual_stats(result.equity)
    return {
        "run_key": variant["run_key"],
        "label": variant["label"],
        **params,
        "min_rotation_profit_pct": variant["min_rotation_profit_pct"],
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
        "worst_year_return": stability["worst_year_return"],
        "median_year_return": stability["median_year_return"],
        "positive_year_ratio": stability["positive_year_ratio"],
        **trade_stats(result),
        "data_quality": result.data_quality,
        "data_version": result.data_version,
        "price_mode": result.price_mode,
    }


def pct(value: Any) -> str:
    return f"{float(value) * 100:+.2f}%"


def save_outputs(
    run_dir: Path,
    completed: dict[str, dict[str, Any]],
    all_variants: tuple[dict[str, Any], ...] | None = None,
) -> None:
    selected_variants = variants() if all_variants is None else all_variants
    ordered = [
        completed[item["run_key"]]
        for item in selected_variants
        if item["run_key"] in completed
    ]
    pd.DataFrame(ordered).to_csv(run_dir / "comparison.csv", index=False)
    (run_dir / "summary.json").write_text(
        json.dumps({"rows": ordered}, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    lines = [
        "Canonical Min-Rotation-Profit Ablation",
        "Fixed: sell=1, positions=1, dual_momentum/150, hurdle=2.0, ST=10/3.0",
        "Universe/runner: Nasdaq-100 index_events + canonical",
        "Rotation profit basis: raw signal-date close / raw average entry price",
        "",
    ]
    for row in ordered:
        lines.extend(
            [
                str(row["label"]),
                f"  Threshold: {pct(row['min_rotation_profit_pct'])}",
                f"  Return: {pct(row['total_return'])}",
                f"  CAGR: {pct(row['cagr'])}",
                f"  MDD: {pct(row['mdd'])}",
                f"  Sharpe: {float(row['sharpe']):.2f}",
                f"  Calmar: {float(row['calmar']):.2f}",
                f"  Worst year: {pct(row['worst_year_return'])}",
                f"  Trades: {int(row['trade_count'])}",
                f"  Rotation exits: {int(row['rotation_exit_count'])} "
                f"(losses={int(row['rotation_loss_count'])})",
                f"  Average rotation P&L: {pct(row['average_rotation_pnl'])}",
                f"  Worst rotation P&L: {pct(row['worst_rotation_pnl'])}",
                "",
            ]
        )
    (run_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.results_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoint.jsonl"
    completed = load_checkpoint(checkpoint)
    params = fixed_params(args)
    all_variants = variants(args.thresholds)

    base_config = base_canonical_config(args)
    source_config = config_for(base_config, params)
    print("[rotation-profit] loading canonical index_events data once...", flush=True)
    data = download_for_config(source_config, allow_stale=True)
    full_index = market_index(data)
    run_index = full_index[
        (full_index >= pd.Timestamp(args.start)) & (full_index <= pd.Timestamp(args.end))
    ]
    if len(run_index) < 2:
        raise RuntimeError("Requested comparison period has fewer than two sessions.")

    print("[rotation-profit] preparing shared indicators...", flush=True)
    source_prepared = _prepare_backtest(create_strategy(source_config), data)
    if not isinstance(source_prepared, PreparedLeaderBacktest):
        raise TypeError("Expected PreparedLeaderBacktest for leader rotation.")

    started = time.monotonic()
    pending_count = sum(item["run_key"] not in completed for item in all_variants)
    newly_completed = 0
    for variant in all_variants:
        key = str(variant["run_key"])
        if key in completed:
            continue
        config = replace(
            source_config,
            leader_rotation=replace(
                source_config.leader_rotation,
                min_rotation_profit_pct=float(variant["min_rotation_profit_pct"]),
            ),
        )
        prepared = PreparedLeaderBacktest(
            create_strategy(config),
            source_prepared.prepared,
            source_prepared.market_filter_trends,
            source_prepared.universe_schedule,
        )
        run_started = time.monotonic()
        result = run_backtest_on_data(
            config,
            data,
            run_index=run_index,
            prepared_backtest=prepared,
        )
        row = result_row(variant, params, result, benchmark_return(data, result.equity.index))
        target = run_dir / "runs" / key
        target.mkdir(parents=True, exist_ok=True)
        result.equity.rename("equity").to_frame().to_csv(target / "equity.csv")
        pd.DataFrame(result.trade_records).to_csv(target / "trades.csv", index=False)
        (target / "summary.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()
        completed[key] = row
        newly_completed += 1
        save_outputs(run_dir, completed, all_variants)
        average = (time.monotonic() - started) / newly_completed
        remaining = max(pending_count - newly_completed, 0)
        print(
            f"[rotation-profit] [{len(completed)}/{len(all_variants)}] {key} "
            f"return={pct(row['total_return'])} mdd={pct(row['mdd'])} "
            f"sharpe={row['sharpe']:.2f} rotations={row['rotation_exit_count']} "
            f"run={time.monotonic() - run_started:.1f}s eta={average * remaining / 60.0:.1f}m",
            flush=True,
        )

    save_outputs(run_dir, completed, all_variants)
    (run_dir / "experiment.json").write_text(
        json.dumps(
            {
                "completed_at": datetime.now().isoformat(),
                "requested_start": args.start,
                "requested_end": args.end,
                "fixed_params": params,
                "variants": list(all_variants),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[rotation-profit] complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
