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

from risk_overlay import (  # noqa: E402
    PortfolioRiskPolicy,
    PortfolioRiskPreparedBacktest,
    default_policies,
)


DEFAULT_STRATEGY = UNIFIED_ROOT / "configs" / "strategies" / "leader_rotation_dual_momentum.yaml"
DEFAULT_RUNTIME = UNIFIED_ROOT / "configs" / "runtimes" / "research_nasdaq100.yaml"
DEFAULT_RESULTS = EXPERIMENT_ROOT / "results" / "canonical_2015_2026"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare position count, cash allocation, ATR sizing and an equity drawdown brake.")
    parser.add_argument("--start", default="2015-10-19")
    parser.add_argument("--end", default="2026-07-15")
    parser.add_argument(
        "--account",
        choices=("A", "C"),
        default="A",
        help="A uses the daily QQQ market filter; C uses no market filter.",
    )
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    return parser


def canonical_account_config(account: str = "A"):
    market_filter = "1d" if account == "A" else "none"
    base = load_split_config(DEFAULT_STRATEGY, DEFAULT_RUNTIME)
    base = replace(base, period="max", timeframe="1d", data_store=replace(base.data_store, provider="parquet"))
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
            "market_filter": market_filter,
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


def config_for_policy(base, policy: PortfolioRiskPolicy):
    return replace(
        base,
        risk=replace(base.risk, max_position_count=policy.max_positions),
        execution=replace(base.execution, allocation_pct=policy.allocation_pct),
    )


@contextmanager
def inject_prepared(shared, policy: PortfolioRiskPolicy, wrapper_box: list) -> Iterator[None]:
    canonical_prepare = runners._prepare_backtest

    def injected_prepare(strategy, _market_data):
        delegate = PreparedLeaderBacktest(
            strategy,
            shared.prepared,
            shared.market_filter_trends,
            shared.universe_schedule,
        )
        if policy.target_portfolio_atr_pct is None and policy.drawdown_stop_pct is None:
            wrapper_box.append(None)
            return delegate
        wrapper = PortfolioRiskPreparedBacktest(delegate, policy)
        wrapper_box.append(wrapper)
        return wrapper

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
    drawdown = selected / selected.cummax() - 1.0
    return {
        "return": float(selected.iloc[-1] / selected.iloc[0] - 1.0),
        "mdd": float(drawdown.min()),
    }


def drawdown_episode(equity: pd.Series) -> dict[str, Any]:
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    trough = drawdown.idxmin()
    peak = equity.loc[:trough].idxmax()
    recovery_candidates = equity.loc[trough:]
    recovery_candidates = recovery_candidates[recovery_candidates >= running_peak.loc[trough]]
    return {
        "peak": str(pd.Timestamp(peak).date()),
        "trough": str(pd.Timestamp(trough).date()),
        "recovery": None if recovery_candidates.empty else str(pd.Timestamp(recovery_candidates.index[0]).date()),
    }


def result_row(policy, result, wrapper) -> dict[str, Any]:
    return {
        "variant": policy.name,
        "max_positions": policy.max_positions,
        "allocation_pct": policy.allocation_pct,
        "target_portfolio_atr_pct": policy.target_portfolio_atr_pct,
        "drawdown_stop_pct": policy.drawdown_stop_pct,
        "cooldown_sessions": policy.cooldown_sessions,
        "drawdown_stop_count": wrapper.drawdown_stop_count if wrapper is not None else 0,
        **{key: _plain(value) for key, value in result.metrics.items()},
        **annual_diagnostics(result.equity),
        "max_drawdown_episode": drawdown_episode(result.equity),
        "stress_2021_concentration": period_diagnostics(result.equity, "2021-01-26", "2021-06-16"),
        "stress_covid_crash": period_diagnostics(result.equity, "2020-02-19", "2020-03-23"),
        "stress_2022": period_diagnostics(result.equity, "2022-01-01", "2022-12-31"),
        "data_quality": result.data_quality,
        "data_version": result.data_version,
    }


def run_variant(base_config, data, shared, run_index, policy, results_dir):
    target = results_dir / "runs" / policy.name
    summary_path = target / "summary.json"
    if summary_path.exists() and (target / "equity.csv").exists() and (target / "trades.csv").exists():
        print(f"[{policy.name}] loaded checkpoint", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))
    config = config_for_policy(base_config, policy)
    wrapper_box: list = []
    started = time.monotonic()
    with inject_prepared(shared, policy, wrapper_box):
        result = runners.run_backtest_on_data(config, data, run_index=run_index)
    wrapper = wrapper_box[-1] if wrapper_box else None
    row = result_row(policy, result, wrapper)
    target.mkdir(parents=True, exist_ok=True)
    result.equity.rename("equity").to_frame().to_csv(target / "equity.csv")
    pd.DataFrame(result.trade_records).to_csv(target / "trades.csv", index=False)
    summary_path.write_text(json.dumps(row, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        f"[{policy.name}] CAGR={_pct(row['cagr'])} MDD={_pct(row['mdd'])} "
        f"Calmar={float(row['calmar']):.2f} trades={row['trade_count']} "
        f"dd_stops={row['drawdown_stop_count']} elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )
    return row


def write_results(
    results_dir: Path,
    rows: list[dict[str, Any]],
    run_index: pd.Index,
    account: str,
) -> None:
    flat = []
    for row in rows:
        item = {
            key: value
            for key, value in row.items()
            if key not in {"annual_returns", "max_drawdown_episode", "stress_2021_concentration", "stress_covid_crash", "stress_2022"}
        }
        for period in ("stress_2021_concentration", "stress_covid_crash", "stress_2022"):
            item[f"{period}_return"] = row[period]["return"]
            item[f"{period}_mdd"] = row[period]["mdd"]
        item.update({f"mdd_{key}": value for key, value in row["max_drawdown_episode"].items()})
        flat.append(item)
    pd.DataFrame(flat).to_csv(results_dir / "comparison.csv", index=False)
    best = max(rows, key=lambda row: (float(row["calmar"]), float(row["sharpe"]), float(row["cagr"])))
    summary = {
        "created_at": datetime.now().isoformat(),
        "period": {"start": str(run_index[0]), "end": str(run_index[-1]), "sessions": len(run_index)},
        "universe": "canonical index_events:nasdaq100",
        "account": account,
        "market_filter": "1d" if account == "A" else "none",
        "best_by_calmar": best["variant"],
        "rows": rows,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (results_dir / "REPORT_KO.md").write_text(
        build_report(rows, best, run_index, account), encoding="utf-8"
    )


def build_report(
    rows: list[dict[str, Any]],
    best: dict[str, Any],
    run_index: pd.Index,
    account: str,
) -> str:
    table = [
        "| 변형 | CAGR | MDD | Calmar | 거래 | 2021 집중구간 MDD | COVID MDD | 2022 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['variant']} | {_pct(row['cagr'])} | {_pct(row['mdd'])} | {float(row['calmar']):.2f} "
            f"| {int(row['trade_count'])} | {_pct(row['stress_2021_concentration']['mdd'])} "
            f"| {_pct(row['stress_covid_crash']['mdd'])} | {_pct(row['stress_2022']['return'])} |"
        )
    return (
        f"# 가상계좌 {account} 포트폴리오 위험관리 실험\n\n"
        f"기간: {run_index[0]} ~ {run_index[-1]} ({len(run_index):,} sessions)  \n"
        f"기준: 현재 {account}, 진입·청산·랭킹 신호 동일  \n"
        f"시장필터: {'QQQ 1일' if account == 'A' else '없음'}  \n\n"
        + "\n".join(table)
        + "\n\n"
        f"Calmar 우선 최선은 **{best['variant']}**입니다.\n\n"
        "D4/D5의 ATR 2.5%는 포트폴리오 ATR 위험 예산이며 종목당 예산은 최대 포지션 수로 나눕니다. "
        "D6은 계좌 고점 대비 -15%에서 전량청산하고 20세션 후 고점을 재설정합니다.\n"
    )


def main() -> None:
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    account = args.account.upper()
    base_config = canonical_account_config(account)
    print(f"[portfolio-risk] loading canonical Nasdaq-100 PIT data for account {account}...", flush=True)
    data = download_for_config(base_config, allow_stale=True)
    data = replace(data, allow_no_trade_valuation_carry_forward=True)
    full_index = market_index(data)
    run_index = full_index[(full_index >= pd.Timestamp(args.start)) & (full_index <= pd.Timestamp(args.end))]
    if len(run_index) < 2:
        raise RuntimeError("Requested period has fewer than two canonical sessions.")
    print("[portfolio-risk] preparing canonical indicators once...", flush=True)
    prepared = runners._prepare_backtest(create_strategy(base_config), data)
    if not isinstance(prepared, PreparedLeaderBacktest):
        raise TypeError("Expected canonical PreparedLeaderBacktest.")
    policies = [
        replace(policy, name=policy.name.replace("CURRENT_A", f"CURRENT_{account}"))
        for policy in default_policies()
    ]
    rows = [
        run_variant(base_config, data, prepared, run_index, policy, results_dir)
        for policy in policies
    ]
    write_results(results_dir, rows, run_index, account)
    print(f"[portfolio-risk] complete: {results_dir.resolve()}", flush=True)


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
