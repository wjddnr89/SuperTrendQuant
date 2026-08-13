from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
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
    FastExperimentalPreparedLeaderBacktest,
)
import supertrend_quant.runners as canonical_runners  # noqa: E402
from supertrend_quant.metrics import calculate_metrics  # noqa: E402
from supertrend_quant.runners import run_backtest_on_data  # noqa: E402
from supertrend_quant.strategies import create_strategy  # noqa: E402
from supertrend_quant.strategies.leader_rotation import (  # noqa: E402
    PreparedLeaderBacktest,
)

from scripts.nested_walk_forward_nasdaq_structure import (  # noqa: E402
    Candidate,
    NestedStructureSearch,
    aggregate_inner,
    benchmark_return_for_index,
    compound_returns,
    config_for_candidate,
    finite_json_value,
    is_eligible,
    stitch_equity,
)


DEFAULT_CONFIG = (
    PLAYGROUND_ROOT / "configs" / "rolling_expanding_nasdaq_plateau.json"
)
DEFAULT_RESULTS = PLAYGROUND_ROOT / "results" / "rolling_expanding_plateau"


@dataclass(frozen=True)
class CandidateMeta:
    candidate: Candidate
    family: str
    stop_step: int
    cap_step: int
    is_core: bool
    selectable: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rolling 5-year and expanding Nasdaq structure validation with "
            "local plateau selection."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--first-outer-year", type=int, default=0)
    parser.add_argument("--last-outer-year", type=int, default=0)
    parser.add_argument("--methods", default="")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-continuous", action="store_true")
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _same_optional_float(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def build_candidate_meta(raw: dict[str, Any]) -> list[CandidateMeta]:
    output: list[CandidateMeta] = []
    for family_raw in raw["candidate_families"]:
        family = str(family_raw["family"])
        gate = str(family_raw["rotation_profit_gate"])
        selectable = bool(family_raw.get("selectable", True))
        stops = list(family_raw["stop_loss_pct"])
        chases = list(family_raw["late_chase"])
        core = dict(family_raw["core"])
        for stop_step, stop in enumerate(stops):
            for cap_step, chase in enumerate(chases):
                mode = str(chase["mode"])
                cap = chase.get("max_extension_atr")
                stop_value = None if stop is None else float(stop)
                cap_value = None if cap is None else float(cap)
                stop_label = (
                    "none" if stop_value is None else f"{stop_value:.0%}"
                )
                chase_label = (
                    mode if cap_value is None else f"cap_{cap_value:g}"
                )
                candidate_id = (
                    f"{family}__stop_{stop_label}__late_{chase_label}"
                    .replace("%", "pct")
                    .replace(".", "p")
                )
                policy = ExperimentalLeaderPolicy(
                    rotation_profit_gate=gate,
                    stop_loss_pct=stop_value,
                    late_chase_mode=mode,
                    max_extension_atr=cap_value,
                )
                is_core = bool(
                    _same_optional_float(
                        stop_value,
                        core.get("stop_loss_pct"),
                    )
                    and mode == str(core["late_chase_mode"])
                    and _same_optional_float(
                        cap_value,
                        core.get("max_extension_atr"),
                    )
                )
                output.append(
                    CandidateMeta(
                        candidate=Candidate(candidate_id, policy),
                        family=family,
                        stop_step=stop_step,
                        cap_step=cap_step,
                        is_core=is_core,
                        selectable=selectable,
                    )
                )
    identifiers = [item.candidate.candidate_id for item in output]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Candidate identifiers must be unique.")
    if sum(item.is_core for item in output) != len(raw["candidate_families"]):
        raise ValueError("Each candidate family must resolve exactly one core.")
    if not any(item.selectable for item in output):
        raise ValueError("At least one candidate must be selectable.")
    return output


def train_years_for(
    method: str,
    outer_year: int,
    validation: dict[str, Any],
) -> list[int]:
    first_train = int(validation["first_train_year"])
    if method == "rolling_5y":
        rolling_years = int(validation["rolling_years"])
        return list(range(outer_year - rolling_years, outer_year))
    if method == "expanding":
        return list(range(first_train, outer_year))
    raise ValueError(f"Unsupported validation method: {method}")


def neighbor_ids(
    target: CandidateMeta,
    all_meta: list[CandidateMeta],
    steps: int,
) -> list[str]:
    return [
        item.candidate.candidate_id
        for item in all_meta
        if item.family == target.family
        and abs(item.stop_step - target.stop_step)
        + abs(item.cap_step - target.cap_step)
        <= steps
    ]


def add_plateau_metrics(
    aggregates: list[dict[str, Any]],
    meta: list[CandidateMeta],
    *,
    neighbor_steps: int,
) -> None:
    by_id = {row["candidate_id"]: row for row in aggregates}
    neighbor_counts = {
        item.candidate.candidate_id: len(
            neighbor_ids(item, meta, neighbor_steps)
        )
        for item in meta
    }
    max_neighbors_by_family = {
        family: max(
            neighbor_counts[item.candidate.candidate_id]
            for item in meta
            if item.family == family
        )
        for family in {item.family for item in meta}
    }
    for item in meta:
        row = by_id[item.candidate.candidate_id]
        neighbors = [
            by_id[candidate_id]
            for candidate_id in neighbor_ids(item, meta, neighbor_steps)
        ]
        row["family"] = item.family
        row["is_core"] = item.is_core
        row["selectable"] = item.selectable
        row["neighbor_count"] = len(neighbors)
        row["plateau_neighbor_coverage"] = (
            len(neighbors) / max_neighbors_by_family[item.family]
        )
        row["plateau_eligible_ratio"] = sum(
            bool(value["eligible"]) for value in neighbors
        ) / len(neighbors)
        row["plateau_min_positive_year_ratio"] = min(
            float(value["positive_year_ratio"]) for value in neighbors
        )
        row["plateau_median_calmar"] = float(
            pd.Series(
                [float(value["median_calmar"]) for value in neighbors]
            ).median()
        )
        row["plateau_worst_year_return"] = min(
            float(value["worst_year_return"]) for value in neighbors
        )
        row["plateau_worst_mdd"] = min(
            float(value["worst_mdd"]) for value in neighbors
        )
        row["plateau_median_compound_return"] = float(
            pd.Series(
                [float(value["compound_return"]) for value in neighbors]
            ).median()
        )


def plateau_sort_key(
    row: dict[str, Any],
    *,
    penalize_boundary: bool = False,
) -> tuple[float, ...]:
    return (
        float(row["plateau_eligible_ratio"]),
        (
            float(row["plateau_neighbor_coverage"])
            if penalize_boundary
            else 1.0
        ),
        float(row["plateau_min_positive_year_ratio"]),
        float(row["plateau_median_calmar"]),
        float(row["plateau_worst_year_return"]),
        float(row["plateau_worst_mdd"]),
        float(row["plateau_median_compound_return"]),
        float(row["median_calmar"]),
        float(row["compound_return"]),
    )


class RollingExpandingPlateauStudy:
    def __init__(
        self,
        raw: dict[str, Any],
        meta: list[CandidateMeta],
        run_dir: Path,
        *,
        resume: bool,
    ) -> None:
        self.raw = raw
        self.meta = meta
        self.candidates = [item.candidate for item in meta]
        self.run_dir = run_dir
        self.search = NestedStructureSearch(
            raw,
            self.candidates,
            run_dir,
            resume=resume,
        )
        self.candidate_by_id = {
            item.candidate.candidate_id: item.candidate for item in meta
        }

    def initialize(self) -> None:
        self.search.initialize()

    def run(
        self,
        methods: list[str],
        *,
        first_outer_year: int,
        last_outer_year: int,
        run_continuous: bool,
    ) -> dict[str, Any]:
        selection_records: list[dict[str, Any]] = []
        benchmark_records: list[dict[str, Any]] = []
        outer_records: list[dict[str, Any]] = []
        outer_results: dict[
            str,
            list[tuple[int, Candidate, Any]],
        ] = {method: [] for method in methods}
        selection = self.raw["selection"]
        neighbor_steps = int(selection["plateau_neighbor_steps"])
        penalize_boundary = bool(
            selection.get("penalize_boundary", False)
        )

        for method in methods:
            print(f"[plateau] method={method}", flush=True)
            for outer_year in range(first_outer_year, last_outer_year + 1):
                train_years = train_years_for(
                    method,
                    outer_year,
                    self.raw["validation"],
                )
                print(
                    f"[plateau] {method} outer={outer_year} "
                    f"train={train_years[0]}-{train_years[-1]} "
                    f"({len(train_years)}y)",
                    flush=True,
                )
                aggregates: list[dict[str, Any]] = []
                for candidate in self.candidates:
                    rows = [
                        self.search.evaluate(candidate, year)[0]
                        for year in train_years
                    ]
                    aggregate = aggregate_inner(candidate, rows)
                    aggregate["eligible"] = is_eligible(aggregate, selection)
                    aggregates.append(aggregate)
                add_plateau_metrics(
                    aggregates,
                    self.meta,
                    neighbor_steps=neighbor_steps,
                )
                selection_pool = [
                    row for row in aggregates if bool(row["selectable"])
                ]
                ranked = sorted(
                    selection_pool,
                    key=lambda row: plateau_sort_key(
                        row,
                        penalize_boundary=penalize_boundary,
                    ),
                    reverse=True,
                )
                for row in aggregates:
                    if bool(row["selectable"]):
                        continue
                    benchmark_records.append(
                        {
                            "method": method,
                            "outer_test_year": outer_year,
                            **dict(row),
                        }
                    )
                shortlist = ranked[: int(selection["stress_shortlist"])]
                for row in shortlist:
                    candidate = self.candidate_by_id[row["candidate_id"]]
                    stress_rows = [
                        self.search.evaluate(
                            candidate,
                            year,
                            cost_multiplier=float(
                                selection["cost_stress_multiplier"]
                            ),
                        )[0]
                        for year in train_years
                    ]
                    row["stress_compound_return"] = compound_returns(
                        [
                            float(value["total_return"])
                            for value in stress_rows
                        ]
                    )
                    row["stress_positive"] = (
                        float(row["stress_compound_return"]) > 0.0
                    )
                stress_survivors = [
                    row
                    for row in shortlist
                    if row.get("stress_positive", False)
                ]
                winner_row = (stress_survivors or shortlist)[0]
                winner = self.candidate_by_id[winner_row["candidate_id"]]
                for rank, row in enumerate(ranked, start=1):
                    record = {
                        "method": method,
                        "outer_test_year": outer_year,
                        "plateau_rank": rank,
                        "selected": (
                            row["candidate_id"] == winner.candidate_id
                        ),
                        **dict(row),
                    }
                    selection_records.append(record)

                outer_row, result = self.search.evaluate(
                    winner,
                    outer_year,
                    include_result=True,
                )
                assert result is not None
                benchmark_symbol = (
                    getattr(self.search.data, "benchmark_symbol", "")
                    or "benchmark"
                )
                outer_record = {
                    "method": method,
                    "selected_candidate_id": winner.candidate_id,
                    "selected_family": winner_row["family"],
                    "benchmark_symbol": benchmark_symbol,
                    "benchmark_return": outer_row.get("qqq_return"),
                    "train_years": ",".join(
                        str(value) for value in train_years
                    ),
                    **outer_row,
                }
                outer_records.append(outer_record)
                outer_results[method].append(
                    (outer_year, winner, result)
                )
                print(
                    f"[plateau] selected {method}/{outer_year} "
                    f"{winner.candidate_id} "
                    f"return={float(outer_row['total_return']):+.2%} "
                    f"mdd={float(outer_row['mdd']):.2%}",
                    flush=True,
                )

        selection_frame = pd.DataFrame(selection_records)
        benchmark_frame = pd.DataFrame(benchmark_records)
        outer_frame = pd.DataFrame(outer_records)
        selection_frame.to_csv(
            self.run_dir / "plateau_selection.csv",
            index=False,
            encoding="utf-8-sig",
        )
        outer_frame.to_csv(
            self.run_dir / "outer_test_folds.csv",
            index=False,
            encoding="utf-8-sig",
        )
        benchmark_frame.to_csv(
            self.run_dir / "benchmark_training_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        method_summaries = {
            method: self._save_method_oos(
                method,
                outer_results[method],
            )
            for method in methods
        }
        consensus_frame, consensus_id = consensus_ranking(
            selection_frame,
            self.meta,
        )
        consensus_frame.to_csv(
            self.run_dir / "consensus_ranking.csv",
            index=False,
            encoding="utf-8-sig",
        )

        continuous = {}
        if run_continuous:
            reference_ids = {
                item.candidate.candidate_id
                for item in self.meta
                if item.is_core
            }
            reference_ids.add(consensus_id)
            for candidate_id in sorted(reference_ids):
                continuous[candidate_id] = self._run_continuous(
                    self.candidate_by_id[candidate_id]
                )

        summary = {
            "name": self.raw["name"],
            "generated": datetime.now().isoformat(timespec="seconds"),
            "data_version": getattr(
                self.search.data,
                "data_version",
                None,
            ),
            "data_quality": getattr(
                self.search.data,
                "data_quality",
                None,
            ),
            "effective_scoring_type": self.search.base.scoring.type,
            "effective_scoring_params": dict(
                self.search.base.scoring.params
            ),
            "candidate_count": len(self.candidates),
            "selectable_candidate_count": sum(
                item.selectable for item in self.meta
            ),
            "benchmark_candidate_ids": [
                item.candidate.candidate_id
                for item in self.meta
                if not item.selectable
            ],
            "methods": methods,
            "first_outer_test_year": first_outer_year,
            "last_outer_test_year": last_outer_year,
            "method_oos": method_summaries,
            "consensus_candidate_id": consensus_id,
            "selection_frequency": selection_frequency(
                pd.DataFrame(outer_records)
            ),
            "continuous_fixed_references": continuous,
            "note": (
                "Continuous fixed references are post-selection diagnostics, "
                "not untouched OOS results."
            ),
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        return summary

    def _save_method_oos(
        self,
        method: str,
        results: list[tuple[int, Candidate, Any]],
    ) -> dict[str, Any]:
        equity = stitch_equity(
            [result.equity for _, _, result in results]
        )
        trades = [
            float(value)
            for _, _, result in results
            for value in result.trades
        ]
        metrics = calculate_metrics(equity, trades, "1d")
        equity.rename("equity").to_frame().to_csv(
            self.run_dir / f"{method}_oos_equity.csv",
            encoding="utf-8-sig",
        )
        records = []
        for year, candidate, result in results:
            for record in result.trade_records:
                records.append(
                    {
                        "outer_test_year": year,
                        "candidate_id": candidate.candidate_id,
                        **dict(record),
                    }
                )
        pd.DataFrame(records).to_csv(
            self.run_dir / f"{method}_oos_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
        qqq_return = compound_returns(
            [
                benchmark_return_for_index(
                    self.search.data,
                    result.equity.index,
                )
                for _, _, result in results
            ]
        )
        benchmark_symbol = (
            getattr(self.search.data, "benchmark_symbol", "")
            or "benchmark"
        )
        return {
            **{
                key: finite_json_value(value)
                for key, value in metrics.items()
            },
            "benchmark_symbol": benchmark_symbol,
            "benchmark_return": finite_json_value(qqq_return),
            "qqq_return": finite_json_value(qqq_return),
            "alpha": finite_json_value(
                float(metrics["total_return"]) - qqq_return
            ),
            "fold_count": len(results),
        }

    def _run_continuous(self, candidate: Candidate) -> dict[str, Any]:
        if (
            self.search.data is None
            or self.search.full_index is None
            or self.search.shared_prepared is None
            or self.search.signal_cache is None
        ):
            raise RuntimeError("Study is not initialized.")
        config = config_for_candidate(self.search.base, candidate)
        strategy = create_strategy(config)
        prepared = PreparedLeaderBacktest(
            strategy,
            self.search.shared_prepared.prepared,
            self.search.shared_prepared.market_filter_trends,
            self.search.shared_prepared.universe_schedule,
        )
        experimental = FastExperimentalPreparedLeaderBacktest(
            prepared,
            candidate.policy,
            self.search.signal_cache,
        )
        start = pd.Timestamp(str(self.raw["start"]))
        end = pd.Timestamp(str(self.raw["end"]))
        run_index = self.search.full_index[
            (self.search.full_index >= start)
            & (self.search.full_index <= end)
        ]
        with patch.object(
            canonical_runners,
            "_prepare_backtest",
            return_value=experimental,
        ):
            result = run_backtest_on_data(
                config,
                self.search.data,
                run_index=run_index,
            )
        label = candidate.candidate_id
        result.equity.rename("equity").to_frame().to_csv(
            self.run_dir / f"continuous_{label}_equity.csv",
            encoding="utf-8-sig",
        )
        pd.DataFrame(result.trade_records).to_csv(
            self.run_dir / f"continuous_{label}_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"[plateau] continuous {label} "
            f"return={float(result.metrics['total_return']):+.2%} "
            f"mdd={float(result.metrics['mdd']):.2%}",
            flush=True,
        )
        return {
            **{
                key: finite_json_value(value)
                for key, value in result.metrics.items()
            },
            "data_quality": result.data_quality,
        }


def consensus_ranking(
    selection_frame: pd.DataFrame,
    meta: list[CandidateMeta],
) -> tuple[pd.DataFrame, str]:
    grouped = (
        selection_frame.groupby("candidate_id", as_index=False)
        .agg(
            fold_count=("plateau_rank", "count"),
            mean_plateau_rank=("plateau_rank", "mean"),
            median_plateau_rank=("plateau_rank", "median"),
            top3_count=("plateau_rank", lambda values: int((values <= 3).sum())),
            top5_count=("plateau_rank", lambda values: int((values <= 5).sum())),
            selected_count=("selected", lambda values: int(values.astype(bool).sum())),
            mean_plateau_eligible_ratio=(
                "plateau_eligible_ratio",
                "mean",
            ),
            median_plateau_calmar=("plateau_median_calmar", "median"),
            worst_plateau_year=(
                "plateau_worst_year_return",
                "min",
            ),
        )
    )
    meta_by_id = {
        item.candidate.candidate_id: item for item in meta
    }
    grouped["family"] = grouped["candidate_id"].map(
        lambda value: meta_by_id[str(value)].family
    )
    grouped["is_core"] = grouped["candidate_id"].map(
        lambda value: meta_by_id[str(value)].is_core
    )
    grouped = grouped.sort_values(
        [
            "mean_plateau_rank",
            "median_plateau_rank",
            "top3_count",
            "worst_plateau_year",
            "median_plateau_calmar",
        ],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)
    grouped.insert(0, "consensus_rank", range(1, len(grouped) + 1))
    return grouped, str(grouped.iloc[0]["candidate_id"])


def selection_frequency(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {}
    counts = frame["selected_candidate_id"].value_counts()
    return {
        str(candidate_id): int(count)
        for candidate_id, count in counts.items()
    }


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    raw = load_json(config_path)
    meta = build_candidate_meta(raw)
    if args.candidate_limit > 0:
        meta = meta[: args.candidate_limit]
    validation = raw["validation"]
    methods = (
        [value.strip() for value in args.methods.split(",") if value.strip()]
        if args.methods
        else list(validation["methods"])
    )
    first_outer_year = (
        args.first_outer_year
        or int(validation["first_outer_test_year"])
    )
    last_outer_year = (
        args.last_outer_year
        or int(validation["last_outer_test_year"])
    )
    run_id = args.run_id or (
        f"{raw['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(args.results_dir).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    study = RollingExpandingPlateauStudy(
        raw,
        meta,
        run_dir,
        resume=not args.no_resume,
    )
    study.initialize()
    summary = study.run(
        methods,
        first_outer_year=first_outer_year,
        last_outer_year=last_outer_year,
        run_continuous=not args.skip_continuous,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[plateau] results={run_dir}", flush=True)


if __name__ == "__main__":
    main()
