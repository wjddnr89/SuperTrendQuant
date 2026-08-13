from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PLAYGROUND_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PLAYGROUND_ROOT.parent
UNIFIED_ROOT = PROJECT_ROOT / "unified_quant"
sys.path.insert(0, str(UNIFIED_ROOT / "src"))
sys.path.insert(0, str(PLAYGROUND_ROOT))
sys.path.insert(0, str(PLAYGROUND_ROOT / "portfolio_risk_experiment"))

from portfolio_risk_experiment.risk_overlay import (  # noqa: E402
    PortfolioRiskPolicy,
    PortfolioRiskPreparedBacktest,
)
from research_extensions.experimental_leader_rotation import (  # noqa: E402
    ExperimentalLeaderPolicy,
    ExperimentalSignalCache,
    FastExperimentalPreparedLeaderBacktest,
)
from research_extensions.kospi_market_filters import build_filter_variant  # noqa: E402
from scripts.nested_walk_forward_nasdaq_structure import (  # noqa: E402
    Candidate,
    base_config,
    benchmark_return_for_index,
    config_for_candidate,
)
from supertrend_quant.data import market_index  # noqa: E402
from supertrend_quant.research.data_resolver import download_for_config  # noqa: E402
import supertrend_quant.runners as canonical_runners  # noqa: E402
from supertrend_quant.runners import _prepare_backtest, run_backtest_on_data  # noqa: E402
from supertrend_quant.strategies import create_strategy  # noqa: E402
from supertrend_quant.strategies.leader_rotation import PreparedLeaderBacktest  # noqa: E402


DEFAULT_CONFIG = PLAYGROUND_ROOT / "configs" / "kospi200_accounts_ag_comparison.json"
DEFAULT_RESULTS = PLAYGROUND_ROOT / "results" / "kospi_accounts_ag"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous KOSPI200 transfer backtest for paper accounts A-G.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--accounts", default="A,B,C,D,E,F,G")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def account_candidate(raw: dict[str, Any]) -> Candidate:
    return Candidate(
        candidate_id=str(raw["account"]),
        policy=ExperimentalLeaderPolicy(
            rotation_profit_gate=str(raw["rotation_profit_gate"]),
            stop_loss_pct=None if raw.get("stop_loss_pct") is None else float(raw["stop_loss_pct"]),
            late_chase_mode=str(raw["late_chase_mode"]),
            max_extension_atr=None,
        ),
    )


def annual_returns(equity: pd.Series) -> dict[str, float]:
    daily = equity.astype(float).pct_change().dropna()
    annual = (1.0 + daily).groupby(daily.index.year).prod() - 1.0
    return {str(int(year)): float(value) for year, value in annual.items()}


def finite(value: Any) -> float | int:
    if isinstance(value, int):
        return value
    number = float(value)
    return number if math.isfinite(number) else 0.0


def save_result(target: Path, result, row: dict[str, Any]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    result.equity.rename("equity").to_frame().to_csv(target / "equity.csv", encoding="utf-8-sig")
    pd.DataFrame(result.trade_records).to_csv(target / "trades.csv", index=False, encoding="utf-8-sig")
    (target / "summary.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    raw = read_json(config_path)
    configured = {str(item["account"]): item for item in raw["accounts"]}
    requested = [value.strip().upper() for value in args.accounts.split(",") if value.strip()]
    unknown = set(requested) - set(configured)
    if unknown:
        raise ValueError(f"Unknown accounts: {sorted(unknown)}")

    run_id = args.run_id or f"{raw['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.results_dir).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    base = base_config(raw)
    print("[KOSPI A-G] 0% loading canonical KOSPI200 PIT daily data...", flush=True)
    data = download_for_config(base, allow_stale=True)
    full_index = market_index(data)
    run_index = full_index[
        (full_index >= pd.Timestamp(str(raw["start"])))
        & (full_index <= pd.Timestamp(str(raw["end"])))
    ]
    if len(run_index) < 2:
        raise RuntimeError("Requested period has fewer than two sessions.")
    print(
        f"[KOSPI A-G] 5% sessions={len(run_index)} symbols={len(data.bars)} "
        f"schedule={len(data.universe_schedule)} benchmark={data.benchmark_symbol}",
        flush=True,
    )
    canonical_prepared = _prepare_backtest(create_strategy(base), data)
    if not isinstance(canonical_prepared, PreparedLeaderBacktest):
        raise TypeError("Expected PreparedLeaderBacktest.")
    print("[KOSPI A-G] 15% shared indicators prepared", flush=True)

    variant_raws = {str(item["variant_id"]): item for item in raw["market_filter_variants"]}
    variants: dict[str, tuple[Any, PreparedLeaderBacktest, ExperimentalSignalCache]] = {}
    required_variants = list(dict.fromkeys(configured[account]["market_filter_variant"] for account in requested))
    for position, variant_id in enumerate(required_variants, start=1):
        variant = build_filter_variant(
            variant_raws[variant_id],
            base_config=base,
            data=data,
            canonical_prepared=canonical_prepared,
            full_index=full_index,
        )
        shared = PreparedLeaderBacktest(
            create_strategy(variant.config),
            canonical_prepared.prepared,
            variant.market_filter_trends,
            canonical_prepared.universe_schedule,
        )
        variants[variant_id] = (variant, shared, ExperimentalSignalCache(shared, full_index))
        print(
            f"[KOSPI A-G] {15 + round(10 * position / len(required_variants))}% "
            f"filter prepared: {variant_id}",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    completed_results: dict[str, tuple[Any, dict[str, Any]]] = {}
    benchmark_return = benchmark_return_for_index(data, run_index)
    started = time.monotonic()
    executable = [account for account in requested if account != "D"]

    for count, account in enumerate(executable, start=1):
        item = configured[account]
        target = run_dir / account
        summary_path = target / "summary.json"
        if not args.no_resume and summary_path.exists() and (target / "equity.csv").exists():
            row = read_json(summary_path)
            completed_results[account] = (None, row)
            rows.append(row)
            print(f"[KOSPI A-G] checkpoint loaded: {account}", flush=True)
            continue

        variant, shared, signal_cache = variants[str(item["market_filter_variant"])]
        candidate = account_candidate(item)
        config = config_for_candidate(variant.config, candidate)
        strategy = create_strategy(config)
        delegate = PreparedLeaderBacktest(
            strategy,
            shared.prepared,
            shared.market_filter_trends,
            shared.universe_schedule,
        )
        experimental = FastExperimentalPreparedLeaderBacktest(delegate, candidate.policy, signal_cache)
        atr_risk = item.get("entry_atr_risk_pct")
        prepared = experimental
        if atr_risk is not None:
            prepared = PortfolioRiskPreparedBacktest(
                experimental,
                PortfolioRiskPolicy(
                    name=f"KOSPI_{account}_ATR_{float(atr_risk):.3f}",
                    max_positions=1,
                    target_portfolio_atr_pct=float(atr_risk),
                ),
            )
        account_started = time.monotonic()
        with patch.object(canonical_runners, "_prepare_backtest", return_value=prepared):
            result = run_backtest_on_data(config, data, run_index=run_index)
        yearly = annual_returns(result.equity)
        metrics = {key: finite(value) for key, value in result.metrics.items()}
        row = {
            "account": account,
            "name": item["name"],
            "market_filter_variant": item["market_filter_variant"],
            "rotation_profit_gate": item["rotation_profit_gate"],
            "stop_loss_pct": item.get("stop_loss_pct"),
            "entry_atr_risk_pct": atr_risk,
            "exact_backtest_supported": True,
            "start": str(pd.Timestamp(run_index[0]).date()),
            "end": str(pd.Timestamp(run_index[-1]).date()),
            "sessions": len(run_index),
            **metrics,
            "benchmark_return": benchmark_return,
            "alpha": float(metrics["total_return"]) - benchmark_return,
            "annual_returns": yearly,
            "positive_years": sum(value > 0 for value in yearly.values()),
            "year_count": len(yearly),
            "worst_year_return": min(yearly.values()),
        }
        save_result(target, result, row)
        completed_results[account] = (result, row)
        rows.append(row)
        pct = 25 + round(70 * count / max(1, len(executable)))
        print(
            f"[KOSPI A-G] {pct}% account={account} return={float(metrics['total_return']):+.2%} "
            f"CAGR={float(metrics['cagr']):+.2%} MDD={float(metrics['mdd']):+.2%} "
            f"trades={int(metrics['trade_count'])} elapsed={time.monotonic() - account_started:.1f}s",
            flush=True,
        )

    if "D" in requested:
        if "C" not in completed_results:
            raise RuntimeError("D proxy requires account C in the requested account set.")
        c_result, c_row = completed_results["C"]
        d_item = configured["D"]
        d_row = {
            **c_row,
            "account": "D",
            "name": d_item["name"],
            "exact_backtest_supported": False,
            "proxy_of": "C",
            "limitation": d_item["limitation"],
        }
        target = run_dir / "D"
        target.mkdir(parents=True, exist_ok=True)
        if c_result is not None:
            save_result(target, c_result, d_row)
        else:
            (target / "summary.json").write_text(json.dumps(d_row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append(d_row)
        print("[KOSPI A-G] 97% D recorded as C daily proxy; intraday exit not tested", flush=True)

    order = {account: index for index, account in enumerate(requested)}
    rows.sort(key=lambda row: order[str(row["account"])])
    flat_rows = [{key: value for key, value in row.items() if key != "annual_returns"} for row in rows]
    pd.DataFrame(flat_rows).to_csv(run_dir / "comparison_summary.csv", index=False, encoding="utf-8-sig")
    yearly_rows = []
    for row in rows:
        for year, value in row["annual_returns"].items():
            yearly_rows.append({"account": row["account"], "year": int(year), "return": value})
    pd.DataFrame(yearly_rows).to_csv(run_dir / "annual_returns.csv", index=False, encoding="utf-8-sig")
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "data_version": data.data_version,
        "data_quality": data.data_quality,
        "benchmark_symbol": data.benchmark_symbol,
        "benchmark_return": benchmark_return,
        "accounts": rows,
        "D_limitation": configured["D"]["limitation"],
    }
    (run_dir / "comparison_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"[KOSPI A-G] 100% complete elapsed={time.monotonic() - started:.1f}s results={run_dir}", flush=True)


if __name__ == "__main__":
    main()
