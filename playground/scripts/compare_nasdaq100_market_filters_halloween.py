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

from research_extensions.experimental_leader_rotation import (  # noqa: E402
    ExperimentalLeaderPolicy,
    ExperimentalSignalCache,
    FastExperimentalPreparedLeaderBacktest,
)
from research_extensions.kospi_market_filters import (  # noqa: E402
    build_filter_variant,
)
from research_extensions.seasonal_overlay import (  # noqa: E402
    HalloweenPreparedBacktest,
)
from scripts.nested_walk_forward_nasdaq_structure import (  # noqa: E402
    Candidate,
    NestedStructureSearch,
    base_config,
    benchmark_return_for_index,
    compound_returns,
    config_for_candidate,
    stitch_equity,
)
from supertrend_quant.data import market_index  # noqa: E402
from supertrend_quant.metrics import calculate_metrics  # noqa: E402
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
    / "nasdaq100_market_filter_halloween_comparison.json"
)
DEFAULT_RESULTS = (
    PLAYGROUND_ROOT / "results" / "nasdaq_market_filter_halloween_comparison"
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Controlled Nasdaq-100 comparison of QQQ cap-weight, "
            "point-in-time equal-weight, and no market filter crossed with "
            "Halloween seasonality OFF/ON."
        )
    )
    value.add_argument("--config", default=str(DEFAULT_CONFIG))
    value.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    value.add_argument("--run-id", default="")
    value.add_argument("--variants", default="")
    value.add_argument("--seasonals", default="")
    value.add_argument("--first-year", type=int, default=0)
    value.add_argument("--last-year", type=int, default=0)
    value.add_argument("--no-resume", action="store_true")
    return value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_values(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def candidate_from(raw: dict[str, Any]) -> Candidate:
    return Candidate(
        candidate_id=str(raw["policy_id"]),
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


def save_evaluation(root: Path, key: str, row: dict[str, Any], result) -> None:
    root.mkdir(parents=True, exist_ok=True)
    result.equity.rename("equity").to_frame().to_csv(
        root / f"{key}_equity.csv",
        encoding="utf-8-sig",
    )
    payload = {
        "row": row,
        "trades": [float(value) for value in result.trades],
        "trade_records": [dict(value) for value in result.trade_records],
    }
    (root / f"{key}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def load_evaluation(
    root: Path,
    key: str,
) -> tuple[dict[str, Any], pd.Series, list[float], list[dict[str, Any]]] | None:
    json_path = root / f"{key}.json"
    equity_path = root / f"{key}_equity.csv"
    if not json_path.exists() or not equity_path.exists():
        return None
    payload = load_json(json_path)
    frame = pd.read_csv(equity_path, index_col=0, parse_dates=True)
    return (
        dict(payload["row"]),
        frame.iloc[:, 0].astype(float),
        [float(value) for value in payload["trades"]],
        [dict(value) for value in payload.get("trade_records", ())],
    )


def evaluate_year(
    *,
    search: NestedStructureSearch,
    candidate: Candidate,
    year: int,
    cost_multiplier: float,
    seasonal: dict[str, Any],
):
    if (
        search.data is None
        or search.full_index is None
        or search.shared_prepared is None
        or search.signal_cache is None
    ):
        raise RuntimeError("Search context is not initialized.")

    start = max(
        pd.Timestamp(f"{year}-01-01"),
        pd.Timestamp(str(search.raw["start"])),
    )
    end = min(
        pd.Timestamp(f"{year}-12-31"),
        pd.Timestamp(str(search.raw["end"])),
    )
    run_index = search.full_index[
        (search.full_index >= start) & (search.full_index <= end)
    ]
    if len(run_index) < 2:
        raise RuntimeError(f"No usable sessions for {year}.")

    config = config_for_candidate(
        search.base,
        candidate,
        cost_multiplier=cost_multiplier,
    )
    prepared = PreparedLeaderBacktest(
        create_strategy(config),
        search.shared_prepared.prepared,
        search.shared_prepared.market_filter_trends,
        search.shared_prepared.universe_schedule,
    )
    experimental = FastExperimentalPreparedLeaderBacktest(
        prepared,
        candidate.policy,
        search.signal_cache,
    )
    backtest = experimental
    if bool(seasonal["enabled"]):
        backtest = HalloweenPreparedBacktest(
            experimental,
            search.full_index,
            active_months=seasonal["active_months"],
        )

    with patch.object(
        canonical_runners,
        "_prepare_backtest",
        return_value=backtest,
    ):
        result = run_backtest_on_data(
            config,
            search.data,
            run_index=run_index,
        )
    row = search._result_row(candidate, year, cost_multiplier, result)
    row["seasonal_exit_count"] = sum(
        str(record.get("exit_reason", "")) == "Halloween seasonal exit"
        for record in result.trade_records
    )
    return row, result


def summarize_window(
    evaluations: list[dict[str, Any]],
    *,
    years: list[int],
    data,
    label: str,
) -> dict[str, Any]:
    selected = [
        item for item in evaluations if int(item["row"]["year"]) in years
    ]
    equity = stitch_equity([item["equity"] for item in selected])
    trades = [value for item in selected for value in item["trades"]]
    trade_records = [
        record for item in selected for record in item["trade_records"]
    ]
    metrics = calculate_metrics(equity, trades, "1d")
    yearly_returns = [
        float(item["row"]["total_return"]) for item in selected
    ]
    benchmark_return = compound_returns(
        [
            benchmark_return_for_index(data, item["equity"].index)
            for item in selected
        ]
    )
    return {
        "window": label,
        "first_year": min(years),
        "last_year": max(years),
        **{key: _finite(value) for key, value in metrics.items()},
        "benchmark_symbol": data.benchmark_symbol,
        "benchmark_return": benchmark_return,
        "alpha": float(metrics["total_return"]) - benchmark_return,
        "positive_years": sum(value > 0.0 for value in yearly_returns),
        "year_count": len(years),
        "worst_year_return": min(yearly_returns),
        "worst_year_mdd": min(
            float(item["row"]["mdd"]) for item in selected
        ),
        "seasonal_exit_count": sum(
            str(record.get("exit_reason", "")) == "Halloween seasonal exit"
            for record in trade_records
        ),
    }


def add_control_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["return_delta_vs_halloween_off"] = math.nan
    output["mdd_delta_vs_halloween_off"] = math.nan
    output["return_delta_vs_none"] = math.nan
    output["mdd_delta_vs_none"] = math.nan

    for _, group in output.groupby(["variant_id", "cost_multiplier", "window"]):
        baseline = group.loc[group["seasonal_id"] == "halloween_off"]
        if baseline.empty:
            continue
        row = baseline.iloc[0]
        output.loc[group.index, "return_delta_vs_halloween_off"] = (
            group["total_return"] - float(row["total_return"])
        )
        output.loc[group.index, "mdd_delta_vs_halloween_off"] = (
            group["mdd"] - float(row["mdd"])
        )

    for _, group in output.groupby(["seasonal_id", "cost_multiplier", "window"]):
        baseline = group.loc[group["variant_id"] == "none"]
        if baseline.empty:
            continue
        row = baseline.iloc[0]
        output.loc[group.index, "return_delta_vs_none"] = (
            group["total_return"] - float(row["total_return"])
        )
        output.loc[group.index, "mdd_delta_vs_none"] = (
            group["mdd"] - float(row["mdd"])
        )
    return output


def main() -> None:
    args = parser().parse_args()
    raw = load_json(Path(args.config).resolve())
    configured_variants = {
        str(item["variant_id"]): item
        for item in raw["market_filter_variants"]
    }
    configured_seasonals = {
        str(item["seasonal_id"]): item
        for item in raw["seasonal_variants"]
    }
    variant_ids = csv_values(args.variants) or list(configured_variants)
    seasonal_ids = csv_values(args.seasonals) or list(configured_seasonals)
    if unknown := set(variant_ids) - set(configured_variants):
        raise ValueError("Unknown variants: " + ", ".join(sorted(unknown)))
    if unknown := set(seasonal_ids) - set(configured_seasonals):
        raise ValueError("Unknown seasonals: " + ", ".join(sorted(unknown)))

    evaluation = raw["evaluation"]
    first_year = args.first_year or int(evaluation["first_year"])
    last_year = args.last_year or int(evaluation["last_year"])
    primary_first = max(first_year, int(evaluation["primary_first_year"]))
    years = list(range(first_year, last_year + 1))
    primary_years = list(range(primary_first, last_year + 1))
    costs = [float(value) for value in evaluation["cost_multipliers"]]
    candidate = candidate_from(raw["policy"])

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
    print("[nasdaq-6] loading canonical Nasdaq-100 data...", flush=True)
    data = download_for_config(base, allow_stale=True)
    full_index = market_index(data)
    print(
        f"[nasdaq-6] data sessions={len(full_index)} "
        f"symbols={len(data.bars)} schedule={len(data.universe_schedule)} "
        f"benchmark={data.benchmark_symbol}",
        flush=True,
    )
    print("[nasdaq-6] preparing shared indicators/scores...", flush=True)
    canonical_prepared = _prepare_backtest(create_strategy(base), data)
    if not isinstance(canonical_prepared, PreparedLeaderBacktest):
        raise TypeError("Expected PreparedLeaderBacktest.")

    total = len(variant_ids) * len(seasonal_ids) * len(costs) * len(years)
    completed = 0
    started = time.monotonic()
    yearly_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    all_trade_records: list[dict[str, Any]] = []
    filter_states = pd.DataFrame(index=pd.DatetimeIndex(full_index))

    for variant_position, variant_id in enumerate(variant_ids, start=1):
        variant_raw = configured_variants[variant_id]
        print(
            f"[nasdaq-6] filter {variant_position}/{len(variant_ids)} "
            f"{variant_id}: building causal regime",
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
        if variant.synthetic_benchmark is not None:
            variant.synthetic_benchmark.to_csv(
                variant_dir / "synthetic_benchmark.csv",
                encoding="utf-8-sig",
            )
        filter_states[variant_id] = variant.regime.reindex(filter_states.index)

        search = NestedStructureSearch(
            raw,
            [candidate],
            variant_dir,
            resume=False,
        )
        search.base = variant.config
        search.data = data
        search.full_index = full_index
        search.shared_prepared = PreparedLeaderBacktest(
            create_strategy(variant.config),
            canonical_prepared.prepared,
            variant.market_filter_trends,
            canonical_prepared.universe_schedule,
        )
        search.signal_cache = ExperimentalSignalCache(
            search.shared_prepared,
            full_index,
        )

        regime_window = variant.regime.loc[
            (variant.regime.index >= pd.Timestamp(str(raw["start"])))
            & (variant.regime.index <= pd.Timestamp(str(raw["end"])))
        ]
        up_ratio = float(regime_window.eq(1).mean())

        for seasonal_id in seasonal_ids:
            seasonal = configured_seasonals[seasonal_id]
            for cost_multiplier in costs:
                collected: list[dict[str, Any]] = []
                for year in years:
                    key = (
                        f"{variant_id}__{seasonal_id}__"
                        f"x{cost_multiplier:g}__{year}"
                    )
                    cached = (
                        None
                        if args.no_resume
                        else load_evaluation(eval_dir, key)
                    )
                    if cached is None:
                        row, result = evaluate_year(
                            search=search,
                            candidate=candidate,
                            year=year,
                            cost_multiplier=cost_multiplier,
                            seasonal=seasonal,
                        )
                        save_evaluation(eval_dir, key, row, result)
                        equity = result.equity
                        trades = [float(value) for value in result.trades]
                        trade_records = [
                            dict(value) for value in result.trade_records
                        ]
                    else:
                        row, equity, trades, trade_records = cached
                    collected.append(
                        {
                            "row": row,
                            "equity": equity,
                            "trades": trades,
                            "trade_records": trade_records,
                        }
                    )
                    yearly_rows.append(
                        {
                            "variant_id": variant_id,
                            "variant_name": variant_raw["name"],
                            "filter_up_ratio": up_ratio,
                            "seasonal_id": seasonal_id,
                            "seasonal_name": seasonal["name"],
                            "seasonal_enabled": bool(seasonal["enabled"]),
                            **row,
                        }
                    )
                    all_trade_records.extend(
                        {
                            "variant_id": variant_id,
                            "seasonal_id": seasonal_id,
                            "year": year,
                            **record,
                        }
                        for record in trade_records
                    )
                    completed += 1
                    print(
                        f"[nasdaq-6] progress {completed}/{total} "
                        f"filter={variant_id} seasonal={seasonal_id} "
                        f"year={year} return={float(row['total_return']):+.2%} "
                        f"mdd={float(row['mdd']):.2%} "
                        f"elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )

                for label, window_years in (
                    ("full_diagnostic", years),
                    ("primary_2021_2026", primary_years),
                ):
                    metrics = summarize_window(
                        collected,
                        years=window_years,
                        data=data,
                        label=label,
                    )
                    summary_rows.append(
                        {
                            "variant_id": variant_id,
                            "variant_name": variant_raw["name"],
                            "variant_type": variant_raw["type"],
                            "filter_up_ratio": up_ratio,
                            "seasonal_id": seasonal_id,
                            "seasonal_name": seasonal["name"],
                            "seasonal_enabled": bool(seasonal["enabled"]),
                            "policy_id": raw["policy"]["policy_id"],
                            "policy_name": raw["policy"]["name"],
                            **asdict(candidate.policy),
                            "cost_multiplier": cost_multiplier,
                            **metrics,
                        }
                    )
        pd.DataFrame(yearly_rows).to_csv(
            run_dir / "yearly_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

    filter_states.index.name = "timestamp"
    filter_states.to_csv(run_dir / "filter_states.csv", encoding="utf-8-sig")
    pd.DataFrame(all_trade_records).to_csv(
        run_dir / "trade_records.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = add_control_deltas(pd.DataFrame(summary_rows))
    summary.to_csv(
        run_dir / "comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "name": raw["name"],
        "generated": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "data_version": data.data_version,
        "data_quality": data.data_quality,
        "benchmark_symbol": data.benchmark_symbol,
        "primary_window": [primary_first, last_year],
        "halloween_definition": (
            "Exposure allowed at executions from November through April. "
            "Positions are liquidated at the first May open and entries become "
            "eligible at the first November open."
        ),
        "note": (
            "Controlled exploratory comparison. Stable Plateau policy B, "
            "point-in-time universe, momentum, asset filters and costs are "
            "fixed; market-filter architecture and Halloween overlay vary."
        ),
        "records": summary.to_dict(orient="records"),
    }
    (run_dir / "comparison_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        f"[nasdaq-6] complete elapsed={time.monotonic() - started:.1f}s "
        f"results={run_dir}",
        flush=True,
    )


def _finite(value: Any) -> float:
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


if __name__ == "__main__":
    main()
