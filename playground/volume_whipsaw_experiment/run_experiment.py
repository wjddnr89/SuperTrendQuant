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

from volume_overlay import (  # noqa: E402
    VolumeFilteredPreparedLeaderBacktest,
    VolumeFilterSpec,
    add_volume_features_to_prepared,
    add_sticky_confirmations_to_prepared,
)


DEFAULT_STRATEGY = UNIFIED_ROOT / "configs" / "strategies" / "leader_rotation_dual_momentum.yaml"
DEFAULT_RUNTIME = UNIFIED_ROOT / "configs" / "runtimes" / "research_nasdaq100.yaml"
DEFAULT_RESULTS = EXPERIMENT_ROOT / "results" / "canonical_best_2015_2026"
CANONICAL_REFERENCE = {
    "total_return": 285.73384423088373,
    "cagr": 0.6937969382439999,
    "mdd": -0.40802394666584685,
    "sharpe": 1.3041111915556405,
    "trade_count": 112,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test causal volume confirmations on the canonical Nasdaq-100 best strategy."
    )
    parser.add_argument("--start", default="2015-10-19")
    parser.add_argument("--end", default="2026-07-15")
    parser.add_argument(
        "--market-filter",
        choices=("1d", "none"),
        default="1d",
        help="Use Account A's 1d QQQ filter or Account C's no-market-filter setup.",
    )
    parser.add_argument(
        "--confirmation-window",
        type=int,
        default=0,
        help="Latch volume confirmation within the first N Supertrend-up bars; 0 keeps the daily gate.",
    )
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    return parser


def canonical_best_config(market_filter: str = "1d"):
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


def stage_one_specs(confirmation_window: int = 0) -> list[VolumeFilterSpec]:
    confirmation_bars = confirmation_window or None
    prefix = f"STICKY{confirmation_window}_" if confirmation_window else ""
    return [
        VolumeFilterSpec("BASE"),
        VolumeFilterSpec(f"{prefix}RVOL20_GE_1.0", rvol_min=1.0, confirmation_bars=confirmation_bars),
        VolumeFilterSpec(f"{prefix}RVOL20_GE_1.2", rvol_min=1.2, confirmation_bars=confirmation_bars),
        VolumeFilterSpec(f"{prefix}RVOL20_GE_1.5", rvol_min=1.5, confirmation_bars=confirmation_bars),
        VolumeFilterSpec(
            f"{prefix}CMF20_GT_0",
            cmf_min=np.nextafter(0.0, 1.0),
            confirmation_bars=confirmation_bars,
        ),
        VolumeFilterSpec(
            f"{prefix}OBV_SLOPE10_GT_0",
            obv_slope_min=np.nextafter(0.0, 1.0),
            confirmation_bars=confirmation_bars,
        ),
    ]


@contextmanager
def inject_shared_prepared(
    shared: PreparedLeaderBacktest,
    spec: VolumeFilterSpec,
) -> Iterator[None]:
    canonical_prepare = runners._prepare_backtest

    def injected_prepare(strategy, _market_data):
        delegate = PreparedLeaderBacktest(
            strategy,
            shared.prepared,
            shared.market_filter_trends,
            shared.universe_schedule,
        )
        if not spec.enabled:
            return delegate
        return VolumeFilteredPreparedLeaderBacktest(delegate, spec)

    runners._prepare_backtest = injected_prepare
    try:
        yield
    finally:
        runners._prepare_backtest = canonical_prepare


def trade_diagnostics(result, run_index: pd.Index) -> dict[str, Any]:
    records = list(result.trade_records)
    index_positions = {pd.Timestamp(value): pos for pos, value in enumerate(run_index)}
    session_holds: list[int] = []
    pnls: list[float] = []
    short_losses_5 = 0
    short_losses_10 = 0
    supertrend_whipsaws_10 = 0
    leader_rotation_exits = 0

    for trade in records:
        pnl = float(trade.get("pnl_pct", 0.0) or 0.0)
        pnls.append(pnl)
        entry = pd.Timestamp(trade.get("entry_time"))
        exit_ = pd.Timestamp(trade.get("exit_time"))
        held = max(0, index_positions.get(exit_, 0) - index_positions.get(entry, 0))
        session_holds.append(held)
        is_loss = pnl <= 0.0
        if is_loss and held <= 5:
            short_losses_5 += 1
        if is_loss and held <= 10:
            short_losses_10 += 1
        reason = str(trade.get("exit_reason") or "")
        if is_loss and held <= 10 and reason == "Supertrend down":
            supertrend_whipsaws_10 += 1
        if reason == "Leader rotation":
            leader_rotation_exits += 1

    trade_count = len(records)
    loss_count = sum(value <= 0.0 for value in pnls)
    return {
        "loss_count": loss_count,
        "loss_rate": loss_count / trade_count if trade_count else 0.0,
        "mean_trade_return": float(np.mean(pnls)) if pnls else 0.0,
        "median_trade_return": float(np.median(pnls)) if pnls else 0.0,
        "median_holding_sessions": float(np.median(session_holds)) if session_holds else 0.0,
        "short_loss_5_count": short_losses_5,
        "short_loss_5_rate": short_losses_5 / trade_count if trade_count else 0.0,
        "short_loss_10_count": short_losses_10,
        "short_loss_10_rate": short_losses_10 / trade_count if trade_count else 0.0,
        "supertrend_whipsaw_10_count": supertrend_whipsaws_10,
        "supertrend_whipsaw_10_rate": supertrend_whipsaws_10 / trade_count if trade_count else 0.0,
        "leader_rotation_exit_count": leader_rotation_exits,
    }


def annual_diagnostics(equity: pd.Series) -> dict[str, Any]:
    daily = equity.astype(float).pct_change().dropna()
    annual = (1.0 + daily).groupby(daily.index.year).prod() - 1.0
    finite = [float(value) for value in annual if math.isfinite(float(value))]
    return {
        "annual_returns": {str(year): float(value) for year, value in annual.items()},
        "worst_year_return": min(finite) if finite else 0.0,
        "median_year_return": float(np.median(finite)) if finite else 0.0,
        "positive_year_ratio": sum(value > 0 for value in finite) / len(finite) if finite else 0.0,
    }


def result_row(spec: VolumeFilterSpec, result, run_index: pd.Index) -> dict[str, Any]:
    row = {
        "variant": spec.name,
        "rvol_min": spec.rvol_min,
        "cmf_min": spec.cmf_min,
        "obv_slope_min": spec.obv_slope_min,
        **{key: _plain(value) for key, value in result.metrics.items()},
        **trade_diagnostics(result, run_index),
        **annual_diagnostics(result.equity),
        "data_quality": result.data_quality,
        "data_version": result.data_version,
    }
    return row


def choose_best_rvol(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row.get("rvol_min") is not None
        and row.get("cmf_min") is None
        and row.get("obv_slope_min") is None
    ]
    if not candidates:
        raise RuntimeError("No RVOL candidates were run.")
    # Calmar is the predeclared selection objective: it rewards CAGR while
    # penalizing the drawdown that whipsaws typically amplify.
    return max(
        candidates,
        key=lambda row: (float(row["calmar"]), float(row["sharpe"]), float(row["cagr"])),
    )


def save_variant(results_dir: Path, row: dict[str, Any], result) -> None:
    target = results_dir / "runs" / str(row["variant"])
    target.mkdir(parents=True, exist_ok=True)
    result.equity.rename("equity").to_frame().to_csv(target / "equity.csv")
    pd.DataFrame(result.trade_records).to_csv(target / "trades.csv", index=False)
    (target / "summary.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_aggregate(
    results_dir: Path,
    rows: list[dict[str, Any]],
    best_rvol: dict[str, Any],
    run_index: pd.Index,
    market_filter: str,
    confirmation_window: int,
) -> None:
    flat_rows = [
        {key: value for key, value in row.items() if key not in {"annual_returns"}}
        for row in rows
    ]
    pd.DataFrame(flat_rows).to_csv(results_dir / "comparison.csv", index=False)
    baseline = next(row for row in rows if row["variant"] == "BASE")
    reference_delta = (
        {
            key: float(baseline[key]) - float(value)
            for key, value in CANONICAL_REFERENCE.items()
        }
        if market_filter == "1d"
        else {}
    )
    account_name = "A" if market_filter == "1d" else "C"
    summary = {
        "created_at": datetime.now().isoformat(),
        "period": {"start": str(run_index[0]), "end": str(run_index[-1]), "sessions": len(run_index)},
        "universe": "canonical index_events:nasdaq100",
        "strategy": f"canonical best parameter set / playground paper Account {account_name}",
        "market_filter": market_filter,
        "confirmation_window": confirmation_window,
        "valuation_policy": "carry prior close across isolated missing sessions; exact bar still required for fills",
        "selection_objective": "highest Calmar, then Sharpe, then CAGR among RVOL thresholds",
        "best_rvol_variant": best_rvol["variant"],
        "canonical_reference_delta": reference_delta,
        "rows": rows,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (results_dir / "REPORT_KO.md").write_text(
        build_korean_report(
            rows,
            best_rvol,
            reference_delta,
            run_index,
            market_filter=market_filter,
            confirmation_window=confirmation_window,
        ),
        encoding="utf-8",
    )


def build_korean_report(
    rows: list[dict[str, Any]],
    best_rvol: dict[str, Any],
    reference_delta: dict[str, float],
    run_index: pd.Index,
    *,
    market_filter: str,
    confirmation_window: int,
) -> str:
    baseline = next(row for row in rows if row["variant"] == "BASE")
    headers = "| 변형 | CAGR | MDD | Sharpe | Calmar | 거래 | 10일내 손실 | ST 휩소(10일) |\n|---|---:|---:|---:|---:|---:|---:|---:|"
    body = []
    for row in rows:
        body.append(
            "| {variant} | {cagr} | {mdd} | {sharpe:.2f} | {calmar:.2f} | {trades} | {short} | {st} |".format(
                variant=row["variant"],
                cagr=_pct(row["cagr"]),
                mdd=_pct(row["mdd"]),
                sharpe=float(row["sharpe"]),
                calmar=float(row["calmar"]),
                trades=int(row["trade_count"]),
                short=int(row["short_loss_10_count"]),
                st=int(row["supertrend_whipsaw_10_count"]),
            )
        )
    combined = [row for row in rows if "BEST_RVOL+" in str(row["variant"])]
    best_combined = max(combined, key=lambda row: float(row["calmar"])) if combined else None
    conclusion = (
        f"RVOL 단독 중 사전 기준(Calmar 우선) 최선은 **{best_rvol['variant']}**입니다. "
        f"BASE 대비 CAGR {_delta_pct(best_rvol['cagr'], baseline['cagr'])}, "
        f"MDD {_delta_pct(best_rvol['mdd'], baseline['mdd'])}, "
        f"10세션 이내 손실 거래 {int(best_rvol['short_loss_10_count']) - int(baseline['short_loss_10_count']):+d}건입니다."
    )
    if best_combined is not None:
        conclusion += (
            f" 결합형 중 Calmar가 가장 높은 것은 **{best_combined['variant']}**"
            f"({float(best_combined['calmar']):.2f})입니다."
        )
    account_name = "A" if market_filter == "1d" else "C"
    confirmation_text = (
        f"거래량 확인: Supertrend 상승 전환 후 {confirmation_window}봉 내 확인 시 상승 구간 종료까지 유지  \n"
        if confirmation_window
        else "거래량 확인: 매일 신규 후보에 적용  \n"
    )
    parity_text = (
        "과거 canonical 저장 결과와 BASE의 최대 절대 지표 차이는 "
        f"`{max(abs(float(value)) for value in reference_delta.values()):.12g}`입니다. "
        "차이가 크면 데이터 릴리스 변경 여부를 먼저 확인해야 합니다.\n\n"
        if reference_delta
        else "시장 필터 제거 BASE는 이번 실행의 C 기준선으로 새로 계산했습니다.\n\n"
    )
    return (
        "# Supertrend 거래량 휩소 방지 실험\n\n"
        f"기간: {run_index[0]} ~ {run_index[-1]} ({len(run_index):,} sessions)  \n"
        "유니버스: canonical `index_events:nasdaq100` (시점별 구성종목)  \n"
        f"BASE: playground {account_name} 가상계좌와 동일한 전략 파라미터"
        f"(market_filter={market_filter}). 데이터/실행 환경은 canonical 검증 기준.\n\n"
        + confirmation_text + "\n"
        "## 결과\n\n"
        + headers + "\n" + "\n".join(body) + "\n\n"
        "## 해석\n\n"
        + conclusion + "\n\n"
        "휩소는 `손실이면서 진입 후 10 canonical sessions 이내 청산`으로 정의했고, "
        "그중 `Supertrend down` 청산을 별도로 집계했습니다. 거래량 조건은 신규 후보에만 적용되어 기존 청산 규칙을 바꾸지 않습니다.\n\n"
        "과거 구성종목의 고립된 가격 누락일은 직전 종가로 평가만 하며, 체결에는 여전히 정확한 당일 가격 바가 필요합니다.\n\n"
        + parity_text
        +
        "주의: 같은 전 기간에서 RVOL 임계값을 선택한 탐색적 결과입니다. 실전 채택 전에는 연도별 또는 walk-forward 검증이 필요합니다.\n"
    )


def run_variant(config, data, shared, run_index, spec, results_dir):
    started = time.monotonic()
    with inject_shared_prepared(shared, spec):
        result = runners.run_backtest_on_data(config, data, run_index=run_index)
    row = result_row(spec, result, run_index)
    save_variant(results_dir, row, result)
    print(
        f"[{spec.name}] CAGR={_pct(row['cagr'])} MDD={_pct(row['mdd'])} "
        f"Calmar={float(row['calmar']):.2f} trades={row['trade_count']} "
        f"whipsaw10={row['short_loss_10_count']} elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )
    return row


def run_or_load_variant(config, data, shared, run_index, spec, results_dir):
    summary_path = results_dir / "runs" / spec.name / "summary.json"
    equity_path = results_dir / "runs" / spec.name / "equity.csv"
    trades_path = results_dir / "runs" / spec.name / "trades.csv"
    if summary_path.exists() and equity_path.exists() and trades_path.exists():
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"[{spec.name}] loaded completed checkpoint", flush=True)
        return row
    return run_variant(config, data, shared, run_index, spec, results_dir)


def main() -> None:
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    config = canonical_best_config(args.market_filter)
    print("[volume-experiment] loading canonical Nasdaq-100 PIT data...", flush=True)
    data = download_for_config(config, allow_stale=True)
    # Some historical US constituents have isolated missing sessions.  Carrying
    # the last close is for valuation only; canonical execution still requires
    # an exact bar, so this neither creates a fill nor uses a future price.
    data = replace(data, allow_no_trade_valuation_carry_forward=True)
    full_index = market_index(data)
    run_index = full_index[
        (full_index >= pd.Timestamp(args.start)) & (full_index <= pd.Timestamp(args.end))
    ]
    if len(run_index) < 2:
        raise RuntimeError("Requested period has fewer than two canonical sessions.")

    print("[volume-experiment] preparing canonical indicators and volume features once...", flush=True)
    prepared = runners._prepare_backtest(create_strategy(config), data)
    if not isinstance(prepared, PreparedLeaderBacktest):
        raise TypeError("Expected canonical PreparedLeaderBacktest.")
    stage_specs = stage_one_specs(args.confirmation_window)
    featured = add_volume_features_to_prepared(prepared.prepared)
    featured = add_sticky_confirmations_to_prepared(featured, stage_specs)
    shared = PreparedLeaderBacktest(
        prepared.strategy,
        featured,
        prepared.market_filter_trends,
        prepared.universe_schedule,
    )

    rows = [
        run_or_load_variant(config, data, shared, run_index, spec, results_dir)
        for spec in stage_specs
    ]
    best_rvol = choose_best_rvol(rows)
    threshold = float(best_rvol["rvol_min"])
    combined_prefix = f"STICKY{args.confirmation_window}_" if args.confirmation_window else ""
    confirmation_bars = args.confirmation_window or None
    combined_specs = [
        VolumeFilterSpec(
            f"{combined_prefix}BEST_RVOL+CMF20_GT_0",
            rvol_min=threshold,
            cmf_min=np.nextafter(0.0, 1.0),
            confirmation_bars=confirmation_bars,
        ),
        VolumeFilterSpec(
            f"{combined_prefix}BEST_RVOL+OBV_SLOPE10_GT_0",
            rvol_min=threshold,
            obv_slope_min=np.nextafter(0.0, 1.0),
            confirmation_bars=confirmation_bars,
        ),
    ]
    combined_featured = add_sticky_confirmations_to_prepared(shared.prepared, combined_specs)
    shared = PreparedLeaderBacktest(
        shared.strategy,
        combined_featured,
        shared.market_filter_trends,
        shared.universe_schedule,
    )
    rows.extend(
        run_or_load_variant(config, data, shared, run_index, spec, results_dir)
        for spec in combined_specs
    )
    write_aggregate(
        results_dir,
        rows,
        best_rvol,
        run_index,
        args.market_filter,
        args.confirmation_window,
    )
    print(f"[volume-experiment] complete: {results_dir.resolve()}", flush=True)


def _plain(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _pct(value: Any) -> str:
    return f"{float(value) * 100:+.2f}%"


def _delta_pct(value: Any, baseline: Any) -> str:
    return f"{(float(value) - float(baseline)) * 100:+.2f}%p"


if __name__ == "__main__":
    main()
