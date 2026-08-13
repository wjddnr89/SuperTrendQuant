from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
UNIFIED_ROOT = PROJECT_ROOT / "unified_quant"
sys.path.insert(0, str(UNIFIED_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_ROOT))

from supertrend_quant.config import load_split_config  # noqa: E402
from supertrend_quant.data import market_index  # noqa: E402
from supertrend_quant.research.data_resolver import download_for_config  # noqa: E402
from supertrend_quant.research.overlays import apply_config_overlay  # noqa: E402
from supertrend_quant import runners  # noqa: E402
from supertrend_quant.strategies import create_strategy  # noqa: E402
from supertrend_quant.strategies.leader_rotation import PreparedLeaderBacktest  # noqa: E402

from regime_overlay import (  # noqa: E402
    RegimeManagedPreparedBacktest,
    RegimePolicy,
    build_regime_states,
    default_policies,
)


DEFAULT_STRATEGY = UNIFIED_ROOT / "configs" / "strategies" / "leader_rotation_dual_momentum.yaml"
DEFAULT_RUNTIME = UNIFIED_ROOT / "configs" / "runtimes" / "research_nasdaq100.yaml"
DEFAULT_RESULTS = EXPERIMENT_ROOT / "results" / "canonical_2015_2026"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Develop the canonical QQQ market filter into portfolio risk management.")
    parser.add_argument("--start", default="2015-10-19")
    parser.add_argument("--end", default="2026-07-15")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    return parser


def canonical_a_config():
    base = load_split_config(DEFAULT_STRATEGY, DEFAULT_RUNTIME)
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
    base = replace(base, universe=universe, universe_file="", symbols=())
    config = apply_config_overlay(
        base,
        {
            "entry": "single",
            "market_filter": "1d",
            "asset_filter": "ichimoku_cloud+ema_trend",
            "sell_confirm_bars": 1,
            "max_positions": 1,
            "st_period": 10,
            "st_multiplier": 3.0,
            "fee_rate": 0.001,
            "slippage_rate": 0.0005,
        },
    )
    return replace(
        config,
        scoring=replace(config.scoring, type="dual_momentum", params={"lookback_bars": 150}),
        leader_rotation=replace(
            config.leader_rotation,
            hurdle_atr_mult=2.0,
            allow_late_chase=True,
            min_rotation_profit_pct=0.0,
        ),
    )


def benchmark_frame(data) -> pd.DataFrame:
    frames = data.filter_benchmark or data.benchmark or {}
    if isinstance(frames, dict):
        frame = next((item for item in frames.values() if item is not None and not item.empty), None)
    else:
        frame = frames
    if frame is None or frame.empty:
        raise RuntimeError("Canonical market-filter benchmark is missing.")
    return frame


def shared_market_trend(prepared: PreparedLeaderBacktest) -> pd.Series:
    trend = next((item for item in prepared.market_filter_trends.values() if not item.empty), None)
    if trend is None:
        raise RuntimeError("Canonical prepared market trend is missing.")
    return trend


@contextmanager
def inject_prepared(
    shared: PreparedLeaderBacktest,
    policy: RegimePolicy,
    states: pd.Series,
) -> Iterator[None]:
    canonical_prepare = runners._prepare_backtest

    def injected_prepare(strategy, _market_data):
        delegate = PreparedLeaderBacktest(
            strategy,
            shared.prepared,
            shared.market_filter_trends,
            shared.universe_schedule,
        )
        if policy.mode == "canonical":
            return delegate
        return RegimeManagedPreparedBacktest(delegate, policy, states)

    runners._prepare_backtest = injected_prepare
    try:
        yield
    finally:
        runners._prepare_backtest = canonical_prepare


def annual_diagnostics(equity: pd.Series) -> dict[str, Any]:
    daily = equity.astype(float).pct_change().dropna()
    annual = (1.0 + daily).groupby(daily.index.year).prod() - 1.0
    values = [float(value) for value in annual if math.isfinite(float(value))]
    return {
        "annual_returns": {str(year): float(value) for year, value in annual.items()},
        "worst_year_return": min(values) if values else 0.0,
        "median_year_return": float(np.median(values)) if values else 0.0,
        "positive_year_ratio": sum(value > 0 for value in values) / len(values) if values else 0.0,
    }


def period_diagnostics(equity: pd.Series, start: str, end: str) -> dict[str, float]:
    selected = equity.loc[(equity.index >= pd.Timestamp(start)) & (equity.index <= pd.Timestamp(end))]
    if len(selected) < 2:
        return {"return": 0.0, "mdd": 0.0}
    period_return = float(selected.iloc[-1] / selected.iloc[0] - 1.0)
    drawdown = selected / selected.cummax() - 1.0
    return {"return": period_return, "mdd": float(drawdown.min())}


def trade_diagnostics(result) -> dict[str, Any]:
    records = list(result.trade_records)
    market_records = [
        trade
        for trade in records
        if str(trade.get("exit_reason") or "").startswith("Market regime")
    ]
    market_losses = [trade for trade in market_records if float(trade.get("pnl_pct", 0.0) or 0.0) <= 0]
    return {
        "market_exit_count": len(market_records),
        "market_exit_loss_count": len(market_losses),
        "market_exit_loss_rate": len(market_losses) / len(market_records) if market_records else 0.0,
        "market_exit_mean_return": (
            float(np.mean([float(trade.get("pnl_pct", 0.0) or 0.0) for trade in market_records]))
            if market_records
            else 0.0
        ),
    }


def result_row(policy: RegimePolicy, result) -> dict[str, Any]:
    return {
        "variant": policy.name,
        "mode": policy.mode,
        "risk_action": policy.risk_action,
        "bear_confirm_bars": policy.bear_confirm_bars,
        "bull_confirm_bars": policy.bull_confirm_bars,
        **{key: _plain(value) for key, value in result.metrics.items()},
        **trade_diagnostics(result),
        **annual_diagnostics(result.equity),
        "stress_2018": period_diagnostics(result.equity, "2018-01-01", "2018-12-31"),
        "stress_covid_crash": period_diagnostics(result.equity, "2020-02-19", "2020-03-23"),
        "stress_2022": period_diagnostics(result.equity, "2022-01-01", "2022-12-31"),
        "data_quality": result.data_quality,
        "data_version": result.data_version,
    }


def run_variant(config, data, shared, run_index, policy, states, results_dir):
    target = results_dir / "runs" / policy.name
    summary_path = target / "summary.json"
    if summary_path.exists() and (target / "equity.csv").exists() and (target / "trades.csv").exists():
        print(f"[{policy.name}] loaded checkpoint", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    started = time.monotonic()
    with inject_prepared(shared, policy, states):
        result = runners.run_backtest_on_data(config, data, run_index=run_index)
    row = result_row(policy, result)
    target.mkdir(parents=True, exist_ok=True)
    result.equity.rename("equity").to_frame().to_csv(target / "equity.csv")
    pd.DataFrame(result.trade_records).to_csv(target / "trades.csv", index=False)
    summary_path.write_text(json.dumps(row, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        f"[{policy.name}] CAGR={_pct(row['cagr'])} MDD={_pct(row['mdd'])} "
        f"Calmar={float(row['calmar']):.2f} trades={row['trade_count']} "
        f"market_exits={row['market_exit_count']} elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )
    return row


def write_results(results_dir: Path, rows: list[dict[str, Any]], run_index: pd.Index) -> None:
    flat = []
    for row in rows:
        item = {key: value for key, value in row.items() if key not in {"annual_returns", "stress_2018", "stress_covid_crash", "stress_2022"}}
        for period in ("stress_2018", "stress_covid_crash", "stress_2022"):
            item[f"{period}_return"] = row[period]["return"]
            item[f"{period}_mdd"] = row[period]["mdd"]
        flat.append(item)
    pd.DataFrame(flat).to_csv(results_dir / "comparison.csv", index=False)
    best = max(rows, key=lambda row: (float(row["calmar"]), float(row["sharpe"]), float(row["cagr"])))
    summary = {
        "created_at": datetime.now().isoformat(),
        "period": {"start": str(run_index[0]), "end": str(run_index[-1]), "sessions": len(run_index)},
        "universe": "canonical index_events:nasdaq100",
        "best_by_calmar": best["variant"],
        "rows": rows,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (results_dir / "REPORT_KO.md").write_text(build_report(rows, best, run_index), encoding="utf-8")


def build_report(rows: list[dict[str, Any]], best: dict[str, Any], run_index: pd.Index) -> str:
    baseline = next(row for row in rows if row["variant"] == "M0_CURRENT_A")
    table = [
        "| 변형 | CAGR | MDD | Calmar | 거래 | 시장청산 | 2018 | COVID MDD | 2022 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['variant']} | {_pct(row['cagr'])} | {_pct(row['mdd'])} | {float(row['calmar']):.2f} "
            f"| {int(row['trade_count'])} | {int(row['market_exit_count'])} | {_pct(row['stress_2018']['return'])} "
            f"| {_pct(row['stress_covid_crash']['mdd'])} | {_pct(row['stress_2022']['return'])} |"
        )
    return (
        "# Market regime risk-management experiment\n\n"
        f"기간: {run_index[0]} ~ {run_index[-1]} ({len(run_index):,} sessions)  \n"
        "기준: A 계좌 전략, QQQ 1일 Supertrend 시장 필터  \n\n"
        + "\n".join(table)
        + "\n\n"
        f"Calmar 우선 최선은 **{best['variant']}**입니다. "
        f"현재 A의 Calmar는 {float(baseline['calmar']):.2f}입니다.\n\n"
        "M3의 50% 축소/복원은 전량 청산 후 목표 비중으로 재매수하는 실제 리밸런싱이며 비용을 모두 반영합니다.\n"
    )


def main() -> None:
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    config = canonical_a_config()
    print("[market-regime] loading canonical Nasdaq-100 PIT data...", flush=True)
    data = download_for_config(config, allow_stale=True)
    data = replace(data, allow_no_trade_valuation_carry_forward=True)
    full_index = market_index(data)
    run_index = full_index[(full_index >= pd.Timestamp(args.start)) & (full_index <= pd.Timestamp(args.end))]
    if len(run_index) < 2:
        raise RuntimeError("Requested period has fewer than two canonical sessions.")

    print("[market-regime] preparing canonical strategy once...", flush=True)
    prepared = runners._prepare_backtest(create_strategy(config), data)
    if not isinstance(prepared, PreparedLeaderBacktest):
        raise TypeError("Expected canonical PreparedLeaderBacktest.")
    trend = shared_market_trend(prepared)
    benchmark = benchmark_frame(data)
    policies = default_policies()
    rows = []
    for policy in policies:
        states = build_regime_states(trend, benchmark, policy)
        rows.append(run_variant(config, data, prepared, run_index, policy, states, results_dir))
    write_results(results_dir, rows, run_index)
    print(f"[market-regime] complete: {results_dir.resolve()}", flush=True)


def _plain(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _pct(value: Any) -> str:
    return f"{float(value) * 100:+.2f}%"


if __name__ == "__main__":
    main()

