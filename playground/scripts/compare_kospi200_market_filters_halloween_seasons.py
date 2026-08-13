from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PLAYGROUND_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PLAYGROUND_ROOT.parent
UNIFIED_ROOT = PROJECT_ROOT / "unified_quant"
sys.path.insert(0, str(UNIFIED_ROOT / "src"))
sys.path.insert(0, str(PLAYGROUND_ROOT))

from research_extensions.experimental_leader_rotation import (  # noqa: E402
    ExperimentalSignalCache,
)
from research_extensions.kospi_market_filters import (  # noqa: E402
    build_filter_variant,
)
from scripts.compare_kospi200_market_filters import (  # noqa: E402
    policies_from,
)
from scripts.compare_nasdaq100_market_filters_halloween import (  # noqa: E402
    csv_values,
    load_json,
)
from scripts.compare_nasdaq100_market_filters_halloween_seasons import (  # noqa: E402
    _finite,
    aggregate_cycles,
    cycle_rows,
    evaluate_continuous,
    load_result,
    save_result,
)
from scripts.nested_walk_forward_nasdaq_structure import (  # noqa: E402
    NestedStructureSearch,
    base_config,
    benchmark_return_for_index,
)
from supertrend_quant.data import market_index  # noqa: E402
from supertrend_quant.research.data_resolver import download_for_config  # noqa: E402
from supertrend_quant.runners import _prepare_backtest  # noqa: E402
from supertrend_quant.strategies import create_strategy  # noqa: E402
from supertrend_quant.strategies.leader_rotation import (  # noqa: E402
    PreparedLeaderBacktest,
)


DEFAULT_CONFIG = (
    PLAYGROUND_ROOT
    / "configs"
    / "kospi200_market_filter_halloween_comparison.json"
)
DEFAULT_RESULTS = (
    PLAYGROUND_ROOT / "results" / "kospi_market_filter_halloween_seasons"
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Continuous KOSPI200 A/B comparison of KODEX200 cap-weight, "
            "point-in-time equal-weight, and no market filter crossed with "
            "Halloween OFF/ON."
        )
    )
    value.add_argument("--config", default=str(DEFAULT_CONFIG))
    value.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    value.add_argument("--run-id", default="")
    value.add_argument("--variants", default="")
    value.add_argument("--accounts", default="")
    value.add_argument("--seasonals", default="")
    value.add_argument("--no-resume", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    raw = load_json(Path(args.config).resolve())
    variants = {
        str(item["variant_id"]): item
        for item in raw["market_filter_variants"]
    }
    policies = {
        str(item["policy_id"]): (item, candidate)
        for item, candidate in policies_from(raw)
    }
    seasonals = {
        str(item["seasonal_id"]): item
        for item in raw["seasonal_variants"]
    }
    variant_ids = csv_values(args.variants) or list(variants)
    account_ids = csv_values(args.accounts) or list(policies)
    seasonal_ids = csv_values(args.seasonals) or list(seasonals)
    _validate_ids("variants", variant_ids, variants)
    _validate_ids("accounts", account_ids, policies)
    _validate_ids("seasonals", seasonal_ids, seasonals)

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
    print("[kospi-seasons] loading canonical KOSPI200 data...", flush=True)
    data = download_for_config(base, allow_stale=True)
    full_index = market_index(data)
    print("[kospi-seasons] preparing shared indicators/scores...", flush=True)
    canonical_prepared = _prepare_backtest(create_strategy(base), data)
    if not isinstance(canonical_prepared, PreparedLeaderBacktest):
        raise TypeError("Expected PreparedLeaderBacktest.")
    print(
        f"[kospi-seasons] sessions={len(full_index)} symbols={len(data.bars)} "
        f"schedule={len(data.universe_schedule)} "
        f"benchmark={data.benchmark_symbol}",
        flush=True,
    )

    costs = [float(value) for value in raw["evaluation"]["cost_multipliers"]]
    total = (
        len(variant_ids)
        * len(account_ids)
        * len(seasonal_ids)
        * len(costs)
    )
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
            f"[kospi-seasons] filter {variant_position}/{len(variant_ids)} "
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
            [policies[account_id][1] for account_id in account_ids],
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

        for account_id in account_ids:
            policy_raw, candidate = policies[account_id]
            for seasonal_id in seasonal_ids:
                seasonal = seasonals[seasonal_id]
                for cost_multiplier in costs:
                    key = (
                        f"{account_id}__{variant_id}__{seasonal_id}__"
                        f"x{cost_multiplier:g}"
                    )
                    cached = (
                        None
                        if args.no_resume
                        else load_result(eval_dir, key)
                    )
                    if cached is None:
                        result = evaluate_continuous(
                            search=search,
                            candidate=candidate,
                            seasonal=seasonal,
                            cost_multiplier=cost_multiplier,
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
                        first_year=int(
                            raw["evaluation"]["first_cycle_year"]
                        ),
                        last_year=int(
                            raw["evaluation"]["last_cycle_year"]
                        ),
                    )
                    common = {
                        "account_id": account_id,
                        "account_name": policy_raw["name"],
                        "variant_id": variant_id,
                        "variant_name": variant_raw["name"],
                        "variant_type": variant_raw["type"],
                        "filter_up_ratio": up_ratio,
                        "seasonal_id": seasonal_id,
                        "seasonal_name": seasonal["name"],
                        "seasonal_enabled": bool(seasonal["enabled"]),
                        "cost_multiplier": cost_multiplier,
                    }
                    benchmark_return = benchmark_return_for_index(
                        data,
                        equity.index,
                    )
                    metrics = payload["metrics"]
                    summary_rows.append(
                        {
                            **common,
                            **asdict(candidate.policy),
                            **{
                                key: _finite(value)
                                for key, value in metrics.items()
                            },
                            "benchmark_symbol": data.benchmark_symbol,
                            "benchmark_return": benchmark_return,
                            "alpha": float(metrics["total_return"])
                            - benchmark_return,
                            "data_quality": payload["data_quality"],
                            "seasonal_exit_count": sum(
                                str(record.get("exit_reason", ""))
                                == "Halloween seasonal exit"
                                for record in payload["trade_records"]
                            ),
                            **{
                                f"winter_{key}": value
                                for key, value in aggregate_cycles(
                                    winters
                                ).items()
                            },
                            **{
                                f"summer_{key}": value
                                for key, value in aggregate_cycles(
                                    summers
                                ).items()
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
                    print(
                        f"[kospi-seasons] progress {completed}/{total} "
                        f"account={account_id} filter={variant_id} "
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
        f"[kospi-seasons] complete elapsed={time.monotonic() - started:.1f}s "
        f"results={run_dir}",
        flush=True,
    )


def _validate_ids(
    label: str,
    requested: list[str],
    configured: dict[str, Any],
) -> None:
    unknown = set(requested) - set(configured)
    if unknown:
        raise ValueError(
            f"Unknown {label}: " + ", ".join(sorted(unknown))
        )


if __name__ == "__main__":
    main()
