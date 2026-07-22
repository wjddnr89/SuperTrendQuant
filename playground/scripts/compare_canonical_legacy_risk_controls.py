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
from supertrend_quant.runners import (  # noqa: E402
    IntradayStopPolicy,
    _prepare_backtest,
    run_backtest_on_data,
)
from supertrend_quant.strategies import create_strategy  # noqa: E402
from supertrend_quant.strategies.leader_rotation import PreparedLeaderBacktest  # noqa: E402


DEFAULT_SOURCE_RESULTS = (
    PLAYGROUND_ROOT
    / "results"
    / "research"
    / "us_nasdaq100_rolling"
    / "canonical_searches"
    / "canonical_simplified_grid_20260722"
    / "all_results.csv"
)
DEFAULT_RESULTS = (
    PLAYGROUND_ROOT
    / "results"
    / "research"
    / "us_nasdaq100_rolling"
    / "risk_controls"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare legacy-base risk controls on canonical index_events data."
    )
    parser.add_argument("--strategy", default=str(DEFAULT_STRATEGY))
    parser.add_argument("--runtime", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--legacy-config", default=str(DEFAULT_LEGACY))
    parser.add_argument("--source-results", default=str(DEFAULT_SOURCE_RESULTS))
    parser.add_argument("--start", default="2015-10-19")
    parser.add_argument("--end", default="2026-07-15")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--run-id", default="legacy_base_risk_controls_20260722")
    return parser


def legacy_signal_params(args: argparse.Namespace) -> dict[str, Any]:
    legacy = json.loads(Path(args.legacy_config).read_text(encoding="utf-8"))
    return {
        "kind": "risk_control",
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


def variants() -> tuple[dict[str, Any], ...]:
    return (
        {
            "run_key": "st_stop_100pct",
            "label": "Prior-day ST stop, 100% allocation",
            "allocation_pct": 1.0,
            "catastrophe_loss_pct": None,
        },
        {
            "run_key": "st_stop_cat15_100pct",
            "label": "Prior-day ST stop + 15% catastrophe, 100% allocation",
            "allocation_pct": 1.0,
            "catastrophe_loss_pct": 0.15,
        },
        {
            "run_key": "st_stop_cat15_75pct",
            "label": "Prior-day ST stop + 15% catastrophe, 75% allocation",
            "allocation_pct": 0.75,
            "catastrophe_loss_pct": 0.15,
        },
        {
            "run_key": "st_stop_cat15_80pct",
            "label": "Prior-day ST stop + 15% catastrophe, 80% allocation",
            "allocation_pct": 0.80,
            "catastrophe_loss_pct": 0.15,
        },
    )


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["run_key"])] = row
    return rows


def stop_stats(result) -> dict[str, Any]:
    stops = [
        dict(trade)
        for trade in result.trade_records
        if trade.get("exit_reason") == "Intraday protective stop"
    ]
    pnl = [float(trade["pnl_pct"]) for trade in stops]
    return {
        "stop_count": len(stops),
        "stop_gap_count": sum(trade.get("stop_trigger") == "gap_open" for trade in stops),
        "stop_intraday_count": sum(trade.get("stop_trigger") == "intraday" for trade in stops),
        "average_stop_pnl": sum(pnl) / len(pnl) if pnl else float("nan"),
        "worst_stop_pnl": min(pnl) if pnl else float("nan"),
    }


def new_result_row(variant, params, result, qqq_return: float) -> dict[str, Any]:
    metrics = result.metrics
    stability = annual_stats(result.equity)
    return {
        "run_key": variant["run_key"],
        "label": variant["label"],
        "kind": "risk_control",
        **params,
        "allocation_pct": variant["allocation_pct"],
        "catastrophe_loss_pct": variant["catastrophe_loss_pct"],
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
        **stop_stats(result),
        "data_quality": result.data_quality,
        "data_version": result.data_version,
        "price_mode": result.price_mode,
    }


def baseline_rows(source: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(source)
    labels = {
        "legacy_base": "Legacy: sell=5, close-confirmed ST, 100% allocation",
        "ablation_sell1_pos1": "Baseline: sell=1, close-confirmed ST, 100% allocation",
    }
    selected = frame.set_index("run_key", drop=False).loc[list(labels)].copy()
    if len(selected) != 2:
        raise RuntimeError("Source results do not contain both required baseline rows.")
    rows = []
    for _, item in selected.iterrows():
        row = item.to_dict()
        row["label"] = labels[str(row["run_key"])]
        row["allocation_pct"] = 1.0
        row["catastrophe_loss_pct"] = None
        row["stop_count"] = 0
        row["stop_gap_count"] = 0
        row["stop_intraday_count"] = 0
        row["average_stop_pnl"] = float("nan")
        row["worst_stop_pnl"] = float("nan")
        rows.append(row)
    return rows


def pct(value: Any) -> str:
    return f"{float(value) * 100:+.2f}%"


def save_outputs(
    run_dir: Path,
    baselines: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
) -> None:
    rows = baselines + [completed[key] for key in sorted(completed)]
    frame = pd.DataFrame(rows)
    frame.to_csv(run_dir / "comparison.csv", index=False)
    (run_dir / "summary.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    lines = [
        "Legacy Base Risk-Control Comparison",
        "Universe: Nasdaq-100 index_events",
        "Runner: canonical",
        "Signal parameters: dual_momentum/150, hurdle=2.0, positions=1, ST=10/3.0",
        "New variants: sell_confirm=1",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                str(row["label"]),
                f"  Return: {pct(row['total_return'])}",
                f"  CAGR: {pct(row['cagr'])}",
                f"  MDD: {pct(row['mdd'])}",
                f"  Sharpe: {float(row['sharpe']):.2f}",
                f"  Calmar: {float(row['calmar']):.2f}",
                f"  Worst year: {pct(row['worst_year_return'])}",
                f"  Trades: {int(float(row['trade_count']))}",
                f"  Stops: {int(float(row['stop_count']))} "
                f"(intraday={int(float(row['stop_intraday_count']))}, "
                f"gap={int(float(row['stop_gap_count']))})",
                *(
                    [
                        f"  Average stop P&L: {pct(row['average_stop_pnl'])}",
                        f"  Worst stop P&L: {pct(row['worst_stop_pnl'])}",
                    ]
                    if int(float(row["stop_count"])) > 0
                    else []
                ),
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
    baselines = baseline_rows(Path(args.source_results))
    params = legacy_signal_params(args)
    all_variants = variants()

    base_config = base_canonical_config(args)
    source_config = config_for(base_config, params)
    print("[risk-controls] loading canonical index_events data once...", flush=True)
    data = download_for_config(source_config, allow_stale=True)
    full_index = market_index(data)
    run_index = full_index[
        (full_index >= pd.Timestamp(args.start)) & (full_index <= pd.Timestamp(args.end))
    ]
    if len(run_index) < 2:
        raise RuntimeError("Requested risk-control period has fewer than two sessions.")

    print("[risk-controls] preparing shared legacy-base indicators...", flush=True)
    source_prepared = _prepare_backtest(create_strategy(source_config), data)
    if not isinstance(source_prepared, PreparedLeaderBacktest):
        raise TypeError("Expected PreparedLeaderBacktest for legacy leader rotation.")
    signal_levels = {
        symbol: frame["Supertrend_Up"].copy()
        for symbol, frame in source_prepared.prepared.items()
        if "Supertrend_Up" in frame
    }

    started = time.monotonic()
    pending_count = sum(item["run_key"] not in completed for item in all_variants)
    newly_completed = 0
    for variant in all_variants:
        key = str(variant["run_key"])
        if key in completed:
            continue
        config = replace(
            source_config,
            execution=replace(
                source_config.execution,
                allocation_pct=float(variant["allocation_pct"]),
            ),
        )
        strategy = create_strategy(config)
        prepared = PreparedLeaderBacktest(
            strategy,
            source_prepared.prepared,
            source_prepared.market_filter_trends,
            source_prepared.universe_schedule,
        )
        policy = IntradayStopPolicy(
            signal_levels=signal_levels,
            catastrophe_loss_pct=variant["catastrophe_loss_pct"],
        )
        run_started = time.monotonic()
        result = run_backtest_on_data(
            config,
            data,
            run_index=run_index,
            prepared_backtest=prepared,
            intraday_stop_policy=policy,
        )
        row = new_result_row(variant, params, result, benchmark_return(data, result.equity.index))
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
        save_outputs(run_dir, baselines, completed)
        average = (time.monotonic() - started) / newly_completed
        remaining = max(pending_count - newly_completed, 0)
        print(
            f"[risk-controls] [{len(completed)}/{len(all_variants)}] {key} "
            f"return={pct(row['total_return'])} mdd={pct(row['mdd'])} "
            f"sharpe={row['sharpe']:.2f} stops={row['stop_count']} "
            f"run={time.monotonic() - run_started:.1f}s eta={average * remaining / 60.0:.1f}m",
            flush=True,
        )

    save_outputs(run_dir, baselines, completed)
    manifest = {
        "completed_at": datetime.now().isoformat(),
        "requested_start": args.start,
        "requested_end": args.end,
        "variants": list(all_variants),
        "legacy_config": str(Path(args.legacy_config).resolve()),
    }
    (run_dir / "experiment.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[risk-controls] complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
