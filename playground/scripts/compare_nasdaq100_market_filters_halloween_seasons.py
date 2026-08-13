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
    ExperimentalSignalCache,
    FastExperimentalPreparedLeaderBacktest,
)
from research_extensions.kospi_market_filters import (  # noqa: E402
    build_filter_variant,
)
from research_extensions.seasonal_overlay import (  # noqa: E402
    HalloweenPreparedBacktest,
)
from scripts.compare_nasdaq100_market_filters_halloween import (  # noqa: E402
    candidate_from,
    csv_values,
    load_json,
)
from scripts.nested_walk_forward_nasdaq_structure import (  # noqa: E402
    NestedStructureSearch,
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
    / "nasdaq100_market_filter_halloween_comparison.json"
)
DEFAULT_RESULTS = (
    PLAYGROUND_ROOT / "results" / "nasdaq_market_filter_halloween_seasons"
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Continuous Nasdaq-100 market-filter/Halloween comparison with "
            "November-April cycle reporting."
        )
    )
    value.add_argument("--config", default=str(DEFAULT_CONFIG))
    value.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    value.add_argument("--run-id", default="")
    value.add_argument("--variants", default="")
    value.add_argument("--seasonals", default="")
    value.add_argument("--no-resume", action="store_true")
    return value


def evaluate_continuous(
    *,
    search: NestedStructureSearch,
    candidate,
    seasonal: dict[str, Any],
    cost_multiplier: float,
):
    if (
        search.data is None
        or search.full_index is None
        or search.shared_prepared is None
        or search.signal_cache is None
    ):
        raise RuntimeError("Search context is not initialized.")
    start = pd.Timestamp(str(search.raw["start"]))
    end = pd.Timestamp(str(search.raw["end"]))
    run_index = search.full_index[
        (search.full_index >= start) & (search.full_index <= end)
    ]
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
        return run_backtest_on_data(
            config,
            search.data,
            run_index=run_index,
        )


def save_result(root: Path, key: str, result) -> None:
    root.mkdir(parents=True, exist_ok=True)
    result.equity.rename("equity").to_frame().to_csv(
        root / f"{key}_equity.csv",
        encoding="utf-8-sig",
    )
    payload = {
        "metrics": {
            key: _finite(value) for key, value in result.metrics.items()
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


def period_return(
    equity: pd.Series,
    *,
    start_boundary: pd.Timestamp,
    end_boundary: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp, float, float] | None:
    series = equity.astype(float).dropna().sort_index()
    starts = series.loc[series.index <= start_boundary]
    ends = series.loc[series.index >= end_boundary]
    if starts.empty or ends.empty:
        return None
    start_ts = pd.Timestamp(starts.index[-1])
    end_ts = pd.Timestamp(ends.index[0])
    if end_ts <= start_ts:
        return None
    window = series.loc[(series.index >= start_ts) & (series.index <= end_ts)]
    if len(window) < 2 or float(window.iloc[0]) <= 0.0:
        return None
    total_return = float(window.iloc[-1] / window.iloc[0] - 1.0)
    drawdown = window / window.cummax() - 1.0
    return start_ts, end_ts, total_return, float(drawdown.min())


def cycle_rows(
    equity: pd.Series,
    *,
    data,
    first_year: int,
    last_year: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    winters: list[dict[str, Any]] = []
    summers: list[dict[str, Any]] = []
    for start_year in range(first_year, last_year):
        winter = period_return(
            equity,
            start_boundary=pd.Timestamp(f"{start_year}-10-31"),
            end_boundary=pd.Timestamp(f"{start_year + 1}-05-01"),
        )
        if winter is not None:
            start_ts, end_ts, value, mdd = winter
            benchmark = benchmark_return_for_index(
                data,
                pd.DatetimeIndex([start_ts, end_ts]),
            )
            winters.append(
                {
                    "cycle": f"{start_year}-{start_year + 1}",
                    "start_session": start_ts,
                    "end_session": end_ts,
                    "total_return": value,
                    "mdd": mdd,
                    "benchmark_return": benchmark,
                    "alpha": value - benchmark,
                }
            )

        summer = period_return(
            equity,
            start_boundary=pd.Timestamp(f"{start_year}-04-30"),
            end_boundary=pd.Timestamp(f"{start_year}-11-01"),
        )
        if summer is not None:
            start_ts, end_ts, value, mdd = summer
            benchmark = benchmark_return_for_index(
                data,
                pd.DatetimeIndex([start_ts, end_ts]),
            )
            summers.append(
                {
                    "cycle": str(start_year),
                    "start_session": start_ts,
                    "end_session": end_ts,
                    "total_return": value,
                    "mdd": mdd,
                    "benchmark_return": benchmark,
                    "alpha": value - benchmark,
                }
            )
    return winters, summers


def aggregate_cycles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [float(row["total_return"]) for row in rows]
    benchmark_returns = [float(row["benchmark_return"]) for row in rows]
    return {
        "cycle_count": len(rows),
        "compound_return": _compound(returns),
        "benchmark_compound_return": _compound(benchmark_returns),
        "compound_alpha": _compound(returns) - _compound(benchmark_returns),
        "positive_cycles": sum(value > 0.0 for value in returns),
        "positive_cycle_ratio": (
            sum(value > 0.0 for value in returns) / len(returns)
            if returns
            else 0.0
        ),
        "median_cycle_return": (
            float(pd.Series(returns).median()) if returns else 0.0
        ),
        "worst_cycle_return": min(returns, default=0.0),
        "worst_cycle_mdd": min(
            (float(row["mdd"]) for row in rows),
            default=0.0,
        ),
    }


def main() -> None:
    args = parser().parse_args()
    raw = load_json(Path(args.config).resolve())
    variants = {
        str(item["variant_id"]): item
        for item in raw["market_filter_variants"]
    }
    seasonals = {
        str(item["seasonal_id"]): item
        for item in raw["seasonal_variants"]
    }
    variant_ids = csv_values(args.variants) or list(variants)
    seasonal_ids = csv_values(args.seasonals) or list(seasonals)
    if unknown := set(variant_ids) - set(variants):
        raise ValueError("Unknown variants: " + ", ".join(sorted(unknown)))
    if unknown := set(seasonal_ids) - set(seasonals):
        raise ValueError("Unknown seasonals: " + ", ".join(sorted(unknown)))

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
    print(
        f"[nasdaq-seasons] account={candidate.candidate_id} "
        "loading canonical data...",
        flush=True,
    )
    data = download_for_config(base, allow_stale=True)
    full_index = market_index(data)
    canonical_prepared = _prepare_backtest(create_strategy(base), data)
    if not isinstance(canonical_prepared, PreparedLeaderBacktest):
        raise TypeError("Expected PreparedLeaderBacktest.")
    print(
        f"[nasdaq-seasons] sessions={len(full_index)} symbols={len(data.bars)} "
        f"schedule={len(data.universe_schedule)}",
        flush=True,
    )

    total = len(variant_ids) * len(seasonal_ids)
    completed = 0
    started = time.monotonic()
    summary_rows: list[dict[str, Any]] = []
    winter_rows: list[dict[str, Any]] = []
    summer_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    filter_states = pd.DataFrame(index=pd.DatetimeIndex(full_index))

    for variant_position, variant_id in enumerate(variant_ids, start=1):
        variant_raw = variants[variant_id]
        print(
            f"[nasdaq-seasons] filter {variant_position}/{len(variant_ids)} "
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
        comparison_window = variant.regime.loc[
            (variant.regime.index >= pd.Timestamp(str(raw["start"])))
            & (variant.regime.index <= pd.Timestamp(str(raw["end"])))
        ]
        up_ratio = float(comparison_window.eq(1).mean())

        for seasonal_id in seasonal_ids:
            seasonal = seasonals[seasonal_id]
            key = f"{variant_id}__{seasonal_id}__x1"
            cached = (
                None if args.no_resume else load_result(eval_dir, key)
            )
            if cached is None:
                result = evaluate_continuous(
                    search=search,
                    candidate=candidate,
                    seasonal=seasonal,
                    cost_multiplier=1.0,
                )
                save_result(eval_dir, key, result)
                payload = {
                    "metrics": result.metrics,
                    "trades": result.trades,
                    "trade_records": result.trade_records,
                    "data_quality": result.data_quality,
                    "equity": result.equity,
                }
            else:
                payload = cached

            equity = payload["equity"]
            winters, summers = cycle_rows(
                equity,
                data=data,
                first_year=pd.Timestamp(str(raw["start"])).year,
                last_year=pd.Timestamp(str(raw["end"])).year,
            )
            common = {
                "account_id": candidate.candidate_id,
                "account_name": raw["policy"]["name"],
                "variant_id": variant_id,
                "variant_name": variant_raw["name"],
                "variant_type": variant_raw["type"],
                "filter_up_ratio": up_ratio,
                "seasonal_id": seasonal_id,
                "seasonal_name": seasonal["name"],
                "seasonal_enabled": bool(seasonal["enabled"]),
            }
            benchmark_return = benchmark_return_for_index(data, equity.index)
            summary_rows.append(
                {
                    **common,
                    **asdict(candidate.policy),
                    **{
                        key: _finite(value)
                        for key, value in payload["metrics"].items()
                    },
                    "benchmark_return": benchmark_return,
                    "alpha": float(payload["metrics"]["total_return"])
                    - benchmark_return,
                    "seasonal_exit_count": sum(
                        str(record.get("exit_reason", ""))
                        == "Halloween seasonal exit"
                        for record in payload["trade_records"]
                    ),
                    **{
                        f"winter_{key}": value
                        for key, value in aggregate_cycles(winters).items()
                    },
                    **{
                        f"summer_{key}": value
                        for key, value in aggregate_cycles(summers).items()
                    },
                }
            )
            winter_rows.extend({**common, **row} for row in winters)
            summer_rows.extend({**common, **row} for row in summers)
            trade_rows.extend(
                {**common, **dict(record)}
                for record in payload["trade_records"]
            )
            completed += 1
            metrics = payload["metrics"]
            print(
                f"[nasdaq-seasons] progress {completed}/{total} "
                f"account={candidate.candidate_id} filter={variant_id} "
                f"seasonal={seasonal_id} "
                f"return={float(metrics['total_return']):+.2%} "
                f"mdd={float(metrics['mdd']):.2%} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    filter_states.index.name = "timestamp"
    filter_states.to_csv(run_dir / "filter_states.csv", encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(
        run_dir / "comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(winter_rows).to_csv(
        run_dir / "november_april_cycles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(summer_rows).to_csv(
        run_dir / "may_october_cycles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(trade_rows).to_csv(
        run_dir / "trade_records.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"[nasdaq-seasons] complete account={candidate.candidate_id} "
        f"elapsed={time.monotonic() - started:.1f}s results={run_dir}",
        flush=True,
    )


def _compound(values: list[float]) -> float:
    factor = 1.0
    for value in values:
        if math.isfinite(float(value)):
            factor *= 1.0 + float(value)
    return factor - 1.0


def _finite(value: Any) -> float:
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


if __name__ == "__main__":
    main()
