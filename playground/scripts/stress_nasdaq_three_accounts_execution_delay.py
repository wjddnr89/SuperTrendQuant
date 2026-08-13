from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
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

from research_extensions.execution_delay_overlay import (  # noqa: E402
    OneSessionDelayedPreparedBacktest,
)
from portfolio_risk_experiment.risk_overlay import (  # noqa: E402
    PortfolioRiskPolicy,
    PortfolioRiskPreparedBacktest,
)
from research_extensions.experimental_leader_rotation import (  # noqa: E402
    ExperimentalLeaderPolicy,
    ExperimentalSignalCache,
    FastExperimentalPreparedLeaderBacktest,
)
from research_extensions.kospi_market_filters import (  # noqa: E402
    build_filter_variant,
)
from scripts.nested_walk_forward_nasdaq_structure import (  # noqa: E402
    Candidate,
    base_config,
    benchmark_return_for_index,
    config_for_candidate,
)
from supertrend_quant.data import market_index  # noqa: E402
from supertrend_quant.research.data_resolver import download_for_config  # noqa: E402
import supertrend_quant.runners as canonical_runners  # noqa: E402
from supertrend_quant.runners import (  # noqa: E402
    _prepare_backtest,
    run_backtest_on_data,
)
from supertrend_quant.strategies import create_strategy  # noqa: E402
from supertrend_quant.strategies.leader_rotation import (  # noqa: E402
    PreparedLeaderBacktest,
)


DEFAULT_CONFIG = (
    PLAYGROUND_ROOT
    / "configs"
    / "nasdaq_three_account_execution_delay_stress.json"
)
DEFAULT_RESULTS = PLAYGROUND_ROOT / "results" / "nasdaq_execution_delay_stress"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Compare the current Nasdaq paper accounts at canonical next-open "
            "execution versus one additional trading-session delay."
        )
    )
    value.add_argument("--config", default=str(DEFAULT_CONFIG))
    value.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    value.add_argument("--run-id", default="")
    value.add_argument("--no-resume", action="store_true")
    return value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_from_account(raw: dict[str, Any]) -> Candidate:
    return Candidate(
        candidate_id=str(raw["account_id"]),
        policy=ExperimentalLeaderPolicy(
            rotation_profit_gate=str(raw["rotation_profit_gate"]),
            stop_loss_pct=(
                None
                if raw.get("stop_loss_pct") is None
                else float(raw["stop_loss_pct"])
            ),
            late_chase_mode=str(raw["late_chase_mode"]),
            max_extension_atr=(
                None
                if raw.get("max_extension_atr") is None
                else float(raw["max_extension_atr"])
            ),
        ),
    )


def save_result(root: Path, key: str, result) -> None:
    root.mkdir(parents=True, exist_ok=True)
    result.equity.rename("equity").to_frame().to_csv(
        root / f"{key}_equity.csv",
        encoding="utf-8-sig",
    )
    payload = {
        "metrics": {
            metric: finite(value) for metric, value in result.metrics.items()
        },
        "trades": [float(value) for value in result.trades],
        "trade_records": [dict(value) for value in result.trade_records],
        "data_quality": result.data_quality,
    }
    (root / f"{key}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def load_result(root: Path, key: str) -> dict[str, Any] | None:
    json_path = root / f"{key}.json"
    equity_path = root / f"{key}_equity.csv"
    if not json_path.exists() or not equity_path.exists():
        return None
    payload = load_json(json_path)
    frame = pd.read_csv(equity_path, index_col=0, parse_dates=True)
    payload["equity"] = frame.iloc[:, 0].astype(float)
    return payload


def calendar_year_rows(
    equity: pd.Series,
    *,
    account_id: str,
    account_name: str,
    mode_id: str,
) -> list[dict[str, Any]]:
    series = equity.astype(float).dropna().sort_index()
    if series.empty:
        return []
    rows: list[dict[str, Any]] = []
    prior_equity = float(series.iloc[0])
    for year, values in series.groupby(series.index.year):
        values = values.astype(float)
        end_equity = float(values.iloc[-1])
        year_path = pd.concat(
            [
                pd.Series(
                    [prior_equity],
                    index=[pd.Timestamp(values.index[0]) - pd.Timedelta(days=1)],
                ),
                values,
            ]
        )
        drawdown = year_path / year_path.cummax() - 1.0
        rows.append(
            {
                "account_id": account_id,
                "account_name": account_name,
                "mode_id": mode_id,
                "year": int(year),
                "start_equity": prior_equity,
                "end_equity": end_equity,
                "return": (
                    end_equity / prior_equity - 1.0
                    if prior_equity > 0.0
                    else 0.0
                ),
                "mdd": float(drawdown.min()),
                "partial_year": bool(
                    int(year) == int(series.index[0].year)
                    or int(year) == int(series.index[-1].year)
                ),
            }
        )
        prior_equity = end_equity
    return rows


def stress_delta_rows(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        summary.loc[summary["mode_id"].eq("baseline_next_open")]
        .set_index("account_id")
        .sort_index()
    )
    stress = (
        summary.loc[summary["mode_id"].eq("stress_delay_1d")]
        .set_index("account_id")
        .sort_index()
    )
    rows: list[dict[str, Any]] = []
    for account_id in sorted(set(baseline.index) & set(stress.index)):
        before = baseline.loc[account_id]
        after = stress.loc[account_id]
        base_multiple = 1.0 + float(before["total_return"])
        stress_multiple = 1.0 + float(after["total_return"])
        rows.append(
            {
                "account_id": account_id,
                "account_name": before["account_name"],
                "baseline_total_return": float(before["total_return"]),
                "stress_total_return": float(after["total_return"]),
                "stress_terminal_wealth_ratio": (
                    stress_multiple / base_multiple
                    if base_multiple > 0.0
                    else 0.0
                ),
                "total_return_delta": (
                    float(after["total_return"])
                    - float(before["total_return"])
                ),
                "baseline_cagr": float(before["cagr"]),
                "stress_cagr": float(after["cagr"]),
                "cagr_delta": float(after["cagr"]) - float(before["cagr"]),
                "baseline_mdd": float(before["mdd"]),
                "stress_mdd": float(after["mdd"]),
                "mdd_delta": float(after["mdd"]) - float(before["mdd"]),
                "baseline_calmar": float(before["calmar"]),
                "stress_calmar": float(after["calmar"]),
                "calmar_delta": (
                    float(after["calmar"]) - float(before["calmar"])
                ),
                "baseline_sharpe": float(before["sharpe"]),
                "stress_sharpe": float(after["sharpe"]),
                "sharpe_delta": (
                    float(after["sharpe"]) - float(before["sharpe"])
                ),
                "baseline_trade_count": int(float(before["trade_count"])),
                "stress_trade_count": int(float(after["trade_count"])),
                "trade_count_delta": (
                    int(float(after["trade_count"]))
                    - int(float(before["trade_count"]))
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parser().parse_args()
    raw = load_json(Path(args.config).resolve())
    accounts = list(raw["accounts"])
    variants = {
        str(item["variant_id"]): item
        for item in raw["market_filter_variants"]
    }
    modes = list(raw["execution_modes"])
    run_id = args.run_id or (
        f"{raw['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(args.results_dir).resolve() / run_id
    eval_dir = run_dir / "evaluations"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    base = base_config(raw)
    started = time.monotonic()
    print("[delay-stress] loading canonical Nasdaq data...", flush=True)
    data = download_for_config(base, allow_stale=True)
    full_index = market_index(data)
    run_index = full_index[
        (full_index >= pd.Timestamp(str(raw["start"])))
        & (full_index <= pd.Timestamp(str(raw["end"])))
    ]
    print("[delay-stress] preparing canonical indicators...", flush=True)
    canonical_prepared = _prepare_backtest(create_strategy(base), data)
    if not isinstance(canonical_prepared, PreparedLeaderBacktest):
        raise TypeError("Expected canonical PreparedLeaderBacktest.")
    print(
        f"[delay-stress] sessions={len(run_index)} "
        f"symbols={len(data.bars)} accounts={len(accounts)} "
        f"evaluations={len(accounts) * len(modes)}",
        flush=True,
    )

    accounts_by_variant: dict[str, list[dict[str, Any]]] = {}
    for account in accounts:
        accounts_by_variant.setdefault(
            str(account["market_filter_variant"]),
            [],
        ).append(account)

    summary_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    completed = 0
    total = len(accounts) * len(modes)

    for variant_id, variant_accounts in accounts_by_variant.items():
        variant_raw = variants[variant_id]
        print(
            f"[delay-stress] filter={variant_id}: building causal regime",
            flush=True,
        )
        variant = build_filter_variant(
            variant_raw,
            base_config=base,
            data=data,
            canonical_prepared=canonical_prepared,
            full_index=full_index,
        )
        variant_dir = run_dir / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant.diagnostics.to_csv(
            variant_dir / "filter_diagnostics.csv",
            encoding="utf-8-sig",
        )
        shared_prepared = PreparedLeaderBacktest(
            create_strategy(variant.config),
            canonical_prepared.prepared,
            variant.market_filter_trends,
            canonical_prepared.universe_schedule,
        )
        signal_cache = ExperimentalSignalCache(shared_prepared, full_index)
        comparison_regime = variant.regime.loc[
            (variant.regime.index >= pd.Timestamp(str(raw["start"])))
            & (variant.regime.index <= pd.Timestamp(str(raw["end"])))
        ]
        filter_up_ratio = float(comparison_regime.eq(1).mean())

        for account in variant_accounts:
            candidate = candidate_from_account(account)
            config = config_for_candidate(variant.config, candidate)
            for mode in modes:
                mode_id = str(mode["mode_id"])
                key = f"{candidate.candidate_id}__{mode_id}"
                payload = (
                    None if args.no_resume else load_result(eval_dir, key)
                )
                cached = payload is not None
                if payload is None:
                    prepared = PreparedLeaderBacktest(
                        create_strategy(config),
                        canonical_prepared.prepared,
                        variant.market_filter_trends,
                        canonical_prepared.universe_schedule,
                    )
                    backtest = FastExperimentalPreparedLeaderBacktest(
                        prepared,
                        candidate.policy,
                        signal_cache,
                    )
                    atr_risk = account.get("entry_atr_risk_pct")
                    if atr_risk is not None:
                        backtest = PortfolioRiskPreparedBacktest(
                            backtest,
                            PortfolioRiskPolicy(
                                name=(
                                    f"{candidate.candidate_id}_ATR_"
                                    f"{float(atr_risk):.3f}"
                                ),
                                max_positions=1,
                                target_portfolio_atr_pct=float(atr_risk),
                            ),
                        )
                    if int(mode["extra_delay_sessions"]) == 1:
                        backtest = OneSessionDelayedPreparedBacktest(backtest)
                    elif int(mode["extra_delay_sessions"]) != 0:
                        raise ValueError(
                            "Only zero or one extra delay session is supported."
                        )
                    with patch.object(
                        canonical_runners,
                        "_prepare_backtest",
                        return_value=backtest,
                    ):
                        result = run_backtest_on_data(
                            config,
                            data,
                            run_index=run_index,
                        )
                    save_result(eval_dir, key, result)
                    payload = {
                        "metrics": result.metrics,
                        "trades": result.trades,
                        "trade_records": result.trade_records,
                        "data_quality": result.data_quality,
                        "equity": result.equity,
                    }

                equity = payload["equity"]
                metrics = {
                    metric: finite(value)
                    for metric, value in payload["metrics"].items()
                }
                common = {
                    "account_id": candidate.candidate_id,
                    "account_name": str(account["name"]),
                    "market_filter_variant": variant_id,
                    "market_filter_name": str(variant_raw["name"]),
                    "filter_up_ratio": filter_up_ratio,
                    "mode_id": mode_id,
                    "mode_name": str(mode["name"]),
                    "extra_delay_sessions": int(
                        mode["extra_delay_sessions"]
                    ),
                    "entry_atr_risk_pct": account.get(
                        "entry_atr_risk_pct"
                    ),
                    "exact_backtest_supported": bool(
                        account.get("exact_backtest_supported", True)
                    ),
                    "backtest_limitation": str(
                        account.get("backtest_limitation", "")
                    ),
                    **asdict(candidate.policy),
                }
                benchmark_return = benchmark_return_for_index(data, equity.index)
                summary_rows.append(
                    {
                        **common,
                        **metrics,
                        "benchmark_return": benchmark_return,
                        "alpha": (
                            float(metrics["total_return"]) - benchmark_return
                        ),
                    }
                )
                annual_rows.extend(
                    calendar_year_rows(
                        equity,
                        account_id=candidate.candidate_id,
                        account_name=str(account["name"]),
                        mode_id=mode_id,
                    )
                )
                trade_rows.extend(
                    {**common, **dict(record)}
                    for record in payload["trade_records"]
                )
                completed += 1
                print(
                    f"[delay-stress] progress {completed}/{total} "
                    f"account={candidate.candidate_id} mode={mode_id} "
                    f"return={float(metrics['total_return']):+.2%} "
                    f"mdd={float(metrics['mdd']):.2%} "
                    f"sharpe={float(metrics['sharpe']):.2f} "
                    f"trades={int(float(metrics['trade_count']))} "
                    f"source={'cache' if cached else 'run'} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["account_id", "extra_delay_sessions"]
    )
    summary.to_csv(
        run_dir / "comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stress_delta_rows(summary).to_csv(
        run_dir / "stress_deltas.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(annual_rows).sort_values(
        ["account_id", "mode_id", "year"]
    ).to_csv(
        run_dir / "calendar_year_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(trade_rows).to_csv(
        run_dir / "trade_records.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"[delay-stress] complete elapsed={time.monotonic() - started:.1f}s "
        f"results={run_dir}",
        flush=True,
    )


def finite(value: Any) -> float:
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


if __name__ == "__main__":
    main()
