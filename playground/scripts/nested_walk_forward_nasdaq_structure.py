from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
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
from research_extensions.momentum_scorer_adapter import (  # noqa: E402
    with_research_momentum_scoring,
)
from supertrend_quant.config import AppConfig, load_split_config  # noqa: E402
from supertrend_quant.data import MarketData, market_index  # noqa: E402
from supertrend_quant.metrics import calculate_metrics  # noqa: E402
from supertrend_quant.research import apply_config_overlay  # noqa: E402
from supertrend_quant.research.data_resolver import download_for_config  # noqa: E402
import supertrend_quant.runners as canonical_runners  # noqa: E402
from supertrend_quant.runners import (  # noqa: E402
    BacktestResult,
    _prepare_backtest,
    run_backtest_on_data,
)
from supertrend_quant.strategies import create_strategy  # noqa: E402
from supertrend_quant.strategies.leader_rotation import (  # noqa: E402
    PreparedLeaderBacktest,
)


DEFAULT_CONFIG = PLAYGROUND_ROOT / "configs" / "nested_nasdaq_structure.json"
DEFAULT_RESULTS = PLAYGROUND_ROOT / "results" / "nested_walk_forward"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    policy: ExperimentalLeaderPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nested walk-forward search for Nasdaq experimental structure controls."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--first-outer-year", type=int, default=0)
    parser.add_argument("--last-outer-year", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_candidates(raw: dict[str, Any]) -> list[Candidate]:
    grid = raw["structure_grid"]
    candidates: list[Candidate] = []
    for gate in grid["rotation_profit_gate"]:
        for stop in grid["stop_loss_pct"]:
            for chase in grid["late_chase"]:
                mode = str(chase["mode"])
                cap = chase.get("max_extension_atr")
                stop_label = "none" if stop is None else f"{float(stop):.0%}"
                chase_label = mode if cap is None else f"{mode}_{float(cap):g}"
                candidate_id = (
                    f"gate_{gate}__stop_{stop_label}__late_{chase_label}"
                    .replace("%", "pct")
                    .replace(".", "p")
                )
                candidates.append(
                    Candidate(
                        candidate_id=candidate_id,
                        policy=ExperimentalLeaderPolicy(
                            rotation_profit_gate=str(gate),
                            stop_loss_pct=None if stop is None else float(stop),
                            late_chase_mode=mode,
                            max_extension_atr=None if cap is None else float(cap),
                        ),
                    )
                )
    return candidates


def base_config(raw: dict[str, Any]) -> AppConfig:
    config = replace(
        load_split_config(
            raw["strategy"],
            raw["runtime"],
        ),
        period=str(raw.get("period", "max")),
    )
    overlay = dict(raw["base_combo"])
    hurdle = float(overlay.pop("hurdle"))
    rs_method = overlay.pop("rs_method", None)
    rs_period = overlay.pop("rs_period", None)
    rs_skip_bars = overlay.pop("rs_skip_bars", None)
    overlay["costs"] = raw["costs"]
    updated = apply_config_overlay(config, overlay)
    if (
        rs_method is not None
        or rs_period is not None
        or rs_skip_bars is not None
    ):
        updated = with_research_momentum_scoring(
            updated,
            method=rs_method,
            period=rs_period,
            skip_bars=rs_skip_bars,
        )
    return replace(
        updated,
        leader_rotation=replace(
            updated.leader_rotation,
            hurdle_atr_mult=hurdle,
        ),
    )


def config_for_candidate(
    base: AppConfig,
    candidate: Candidate,
    *,
    cost_multiplier: float = 1.0,
) -> AppConfig:
    gate_value = (
        0.0 if candidate.policy.rotation_profit_gate == "nonnegative" else -1.0
    )
    allow_late_chase = candidate.policy.late_chase_mode != "fresh_only"
    return replace(
        base,
        costs=replace(
            base.costs,
            fee_rate=base.costs.fee_rate * float(cost_multiplier),
            slippage_rate=base.costs.slippage_rate * float(cost_multiplier),
        ),
        leader_rotation=replace(
            base.leader_rotation,
            min_rotation_profit_pct=gate_value,
            allow_late_chase=allow_late_chase,
        ),
    )


class NestedStructureSearch:
    def __init__(
        self,
        raw: dict[str, Any],
        candidates: list[Candidate],
        run_dir: Path,
        *,
        resume: bool,
    ) -> None:
        self.raw = raw
        self.candidates = candidates
        self.run_dir = run_dir
        self.resume = resume
        self.base = base_config(raw)
        self.data: MarketData | None = None
        self.full_index: pd.Index | None = None
        self.shared_prepared: PreparedLeaderBacktest | None = None
        self.signal_cache: ExperimentalSignalCache | None = None
        self.checkpoint_path = run_dir / "evaluation_checkpoint.jsonl"
        self.cache = self._load_checkpoint() if resume else {}
        self.started = time.monotonic()
        self.completed_now = 0

    def initialize(self) -> None:
        print("[nested] loading canonical market data...", flush=True)
        self.data = download_for_config(self.base, allow_stale=True)
        self.full_index = market_index(self.data)
        print("[nested] preparing shared canonical indicators...", flush=True)
        prepared = _prepare_backtest(create_strategy(self.base), self.data)
        if not isinstance(prepared, PreparedLeaderBacktest):
            raise TypeError("Expected canonical PreparedLeaderBacktest.")
        self.shared_prepared = prepared
        print("[nested] caching daily signal states...", flush=True)
        self.signal_cache = ExperimentalSignalCache(
            prepared,
            self.full_index,
        )
        print(
            f"[nested] data sessions={len(self.full_index)} "
            f"symbols={len(self.data.bars)} candidates={len(self.candidates)}",
            flush=True,
        )

    def evaluate(
        self,
        candidate: Candidate,
        year: int,
        *,
        cost_multiplier: float = 1.0,
        include_result: bool = False,
    ) -> tuple[dict[str, Any], BacktestResult | None]:
        if (
            self.data is None
            or self.full_index is None
            or self.shared_prepared is None
            or self.signal_cache is None
        ):
            raise RuntimeError("Search is not initialized.")
        key = self._evaluation_key(candidate, year, cost_multiplier)
        if not include_result and key in self.cache:
            return dict(self.cache[key]), None

        start = max(
            pd.Timestamp(f"{year}-01-01"),
            pd.Timestamp(str(self.raw["start"])),
        )
        end = min(
            pd.Timestamp(f"{year}-12-31"),
            pd.Timestamp(str(self.raw["end"])),
        )
        run_index = self.full_index[
            (self.full_index >= start) & (self.full_index <= end)
        ]
        if len(run_index) < 2:
            raise RuntimeError(f"No usable sessions for {year}.")

        config = config_for_candidate(
            self.base,
            candidate,
            cost_multiplier=cost_multiplier,
        )
        strategy = create_strategy(config)
        prepared = PreparedLeaderBacktest(
            strategy,
            self.shared_prepared.prepared,
            self.shared_prepared.market_filter_trends,
            self.shared_prepared.universe_schedule,
        )
        experimental = FastExperimentalPreparedLeaderBacktest(
            prepared,
            candidate.policy,
            self.signal_cache,
        )
        with patch.object(
            canonical_runners,
            "_prepare_backtest",
            return_value=experimental,
        ):
            result = run_backtest_on_data(
                config,
                self.data,
                run_index=run_index,
            )

        row = self._result_row(
            candidate,
            year,
            cost_multiplier,
            result,
        )
        if not include_result:
            self.cache[key] = row
            self._append_checkpoint(key, row)
            self.completed_now += 1
            if self.completed_now % 10 == 0:
                print(
                    f"[nested] evaluated={self.completed_now} "
                    f"elapsed={time.monotonic() - self.started:.1f}s "
                    f"latest={candidate.candidate_id}/{year}/x{cost_multiplier:g}",
                    flush=True,
                )
        return row, result if include_result else None

    def run(
        self,
        *,
        first_outer_year: int,
        last_outer_year: int,
    ) -> dict[str, Any]:
        nested = self.raw["nested"]
        inner_count = int(nested["inner_validation_years"])
        selection_rows: list[dict[str, Any]] = []
        outer_rows: list[dict[str, Any]] = []
        outer_results: list[tuple[int, Candidate, BacktestResult]] = []

        for outer_year in range(first_outer_year, last_outer_year + 1):
            inner_years = list(
                range(outer_year - inner_count, outer_year)
            )
            print(
                f"[nested] outer={outer_year} inner={inner_years}",
                flush=True,
            )
            aggregates: list[dict[str, Any]] = []
            for candidate in self.candidates:
                rows = [
                    self.evaluate(candidate, year)[0]
                    for year in inner_years
                ]
                aggregate = aggregate_inner(candidate, rows)
                aggregate["outer_test_year"] = outer_year
                aggregate["eligible"] = is_eligible(
                    aggregate,
                    self.raw["selection"],
                )
                aggregates.append(aggregate)

            ranked = sorted(
                aggregates,
                key=selection_sort_key,
                reverse=True,
            )
            eligible = [row for row in ranked if row["eligible"]]
            base_pool = eligible or ranked
            shortlist_count = int(self.raw["selection"]["stress_shortlist"])
            shortlist = base_pool[:shortlist_count]
            candidate_by_id = {
                candidate.candidate_id: candidate
                for candidate in self.candidates
            }
            for row in shortlist:
                candidate = candidate_by_id[row["candidate_id"]]
                stress_rows = [
                    self.evaluate(
                        candidate,
                        year,
                        cost_multiplier=float(
                            self.raw["selection"]["cost_stress_multiplier"]
                        ),
                    )[0]
                    for year in inner_years
                ]
                row["stress_compound_return"] = compound_returns(
                    [float(item["total_return"]) for item in stress_rows]
                )
                row["stress_positive"] = row["stress_compound_return"] > 0.0

            stress_survivors = [
                row for row in shortlist if row.get("stress_positive", False)
            ]
            winner_row = (stress_survivors or shortlist)[0]
            winner = candidate_by_id[winner_row["candidate_id"]]
            for rank, row in enumerate(ranked, start=1):
                row["base_rank"] = rank
                row["selected"] = row["candidate_id"] == winner.candidate_id
                selection_rows.append(dict(row))

            outer_row, result = self.evaluate(
                winner,
                outer_year,
                include_result=True,
            )
            assert result is not None
            outer_row["selected_candidate_id"] = winner.candidate_id
            outer_rows.append(outer_row)
            outer_results.append((outer_year, winner, result))
            print(
                f"[nested] selected outer={outer_year} "
                f"{winner.candidate_id} "
                f"return={float(outer_row['total_return']):+.2%} "
                f"mdd={float(outer_row['mdd']):.2%}",
                flush=True,
            )

        selection_frame = pd.DataFrame(selection_rows)
        outer_frame = pd.DataFrame(outer_rows)
        selection_frame.to_csv(
            self.run_dir / "inner_selection.csv",
            index=False,
            encoding="utf-8-sig",
        )
        outer_frame.to_csv(
            self.run_dir / "outer_test_folds.csv",
            index=False,
            encoding="utf-8-sig",
        )

        nested_summary = self._save_oos(
            "nested_selected",
            outer_results,
        )
        baseline = next(
            candidate
            for candidate in self.candidates
            if candidate.policy
            == ExperimentalLeaderPolicy(
                rotation_profit_gate="nonnegative",
                stop_loss_pct=None,
                late_chase_mode="unlimited",
            )
        )
        baseline_results: list[tuple[int, Candidate, BacktestResult]] = []
        for year in range(first_outer_year, last_outer_year + 1):
            _, result = self.evaluate(baseline, year, include_result=True)
            assert result is not None
            baseline_results.append((year, baseline, result))
        baseline_summary = self._save_oos(
            "legacy_structure_baseline",
            baseline_results,
        )

        summary = {
            "name": self.raw["name"],
            "generated": datetime.now().isoformat(timespec="seconds"),
            "candidate_count": len(self.candidates),
            "first_outer_test_year": first_outer_year,
            "last_outer_test_year": last_outer_year,
            "nested_selected": nested_summary,
            "legacy_structure_baseline": baseline_summary,
            "selected_by_year": {
                str(year): candidate.candidate_id
                for year, candidate, _ in outer_results
            },
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str)
            + "\n",
            encoding="utf-8",
        )
        return summary

    def _save_oos(
        self,
        label: str,
        results: list[tuple[int, Candidate, BacktestResult]],
    ) -> dict[str, Any]:
        equity = stitch_equity([result.equity for _, _, result in results])
        trades = [
            float(value)
            for _, _, result in results
            for value in result.trades
        ]
        metrics = calculate_metrics(equity, trades, "1d")
        equity.rename("equity").to_frame().to_csv(
            self.run_dir / f"{label}_equity.csv",
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
            self.run_dir / f"{label}_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
        qqq_return = compound_returns(
            [
                benchmark_return_for_index(self.data, result.equity.index)
                for _, _, result in results
            ]
        )
        return {
            **{key: finite_json_value(value) for key, value in metrics.items()},
            "qqq_return": finite_json_value(qqq_return),
            "alpha": finite_json_value(float(metrics["total_return"]) - qqq_return),
            "fold_count": len(results),
        }

    def _result_row(
        self,
        candidate: Candidate,
        year: int,
        cost_multiplier: float,
        result: BacktestResult,
    ) -> dict[str, Any]:
        gross_profit = sum(
            max(0.0, float(record.get("pnl_cash", 0.0)))
            for record in result.trade_records
        )
        top_trade_pnl = max(
            (
                max(0.0, float(record.get("pnl_cash", 0.0)))
                for record in result.trade_records
            ),
            default=0.0,
        )
        stop_count = sum(
            str(record.get("exit_reason", "")).startswith("Fixed stop")
            for record in result.trade_records
        )
        return {
            "candidate_id": candidate.candidate_id,
            **asdict(candidate.policy),
            "year": year,
            "cost_multiplier": float(cost_multiplier),
            **{
                key: finite_json_value(value)
                for key, value in result.metrics.items()
            },
            "qqq_return": finite_json_value(
                benchmark_return_for_index(self.data, result.equity.index)
            ),
            "gross_profit": gross_profit,
            "top_trade_pnl": top_trade_pnl,
            "top_trade_gross_profit_share": (
                top_trade_pnl / gross_profit if gross_profit > 0.0 else 0.0
            ),
            "stop_count": stop_count,
            "data_quality": result.data_quality,
        }

    def _load_checkpoint(self) -> dict[str, dict[str, Any]]:
        cache: dict[str, dict[str, Any]] = {}
        if not self.checkpoint_path.exists():
            return cache
        for line in self.checkpoint_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            cache[str(item["cache_key"])] = dict(item["row"])
        return cache

    def _append_checkpoint(self, key: str, row: dict[str, Any]) -> None:
        with self.checkpoint_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"cache_key": key, "row": row},
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    @staticmethod
    def _evaluation_key(
        candidate: Candidate,
        year: int,
        cost_multiplier: float,
    ) -> str:
        return f"{candidate.candidate_id}|{year}|cost={cost_multiplier:g}"


def aggregate_inner(
    candidate: Candidate,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    returns = [float(row["total_return"]) for row in rows]
    calmars = [bounded_number(row["calmar"]) for row in rows]
    gross_profit = sum(float(row["gross_profit"]) for row in rows)
    top_trade_pnl = max(
        (float(row["top_trade_pnl"]) for row in rows),
        default=0.0,
    )
    return {
        "candidate_id": candidate.candidate_id,
        **asdict(candidate.policy),
        "inner_years": ",".join(str(int(row["year"])) for row in rows),
        "positive_year_ratio": sum(value > 0.0 for value in returns) / len(returns),
        "compound_return": compound_returns(returns),
        "median_calmar": float(pd.Series(calmars).median()),
        "worst_year_return": min(returns),
        "worst_mdd": min(float(row["mdd"]) for row in rows),
        "total_trades": sum(int(row["trade_count"]) for row in rows),
        "gross_profit": gross_profit,
        "top_trade_pnl": top_trade_pnl,
        "top_trade_gross_profit_share": (
            top_trade_pnl / gross_profit if gross_profit > 0.0 else 0.0
        ),
    }


def is_eligible(row: dict[str, Any], rules: dict[str, Any]) -> bool:
    positive_year_ratio = float(row["positive_year_ratio"])
    minimum_positive_ratio = float(rules["min_positive_year_ratio"])
    positive_ratio_passes = (
        positive_year_ratio >= minimum_positive_ratio
        or math.isclose(
            positive_year_ratio,
            minimum_positive_ratio,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )
    return bool(
        positive_ratio_passes
        and int(row["total_trades"]) >= int(rules["min_total_trades"])
        and float(row["worst_mdd"]) >= -float(rules["max_abs_mdd"])
        and float(row["top_trade_gross_profit_share"])
        <= float(rules["max_top_trade_gross_profit_share"])
    )


def selection_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        1.0 if row.get("eligible", False) else 0.0,
        float(row["median_calmar"]),
        float(row["worst_year_return"]),
        float(row["worst_mdd"]),
        float(row["compound_return"]),
    )


def compound_returns(values: list[float]) -> float:
    factor = 1.0
    for value in values:
        if not math.isfinite(float(value)):
            continue
        factor *= 1.0 + float(value)
    return factor - 1.0


def stitch_equity(parts: list[pd.Series]) -> pd.Series:
    values: list[float] = []
    index: list[Any] = []
    capital = 1.0
    for part in parts:
        series = part.astype(float).dropna()
        if series.empty or float(series.iloc[0]) <= 0.0:
            continue
        normalized = series / float(series.iloc[0]) * capital
        start = 0 if not values else 1
        values.extend(float(value) for value in normalized.iloc[start:])
        index.extend(normalized.index[start:])
        capital = float(normalized.iloc[-1])
    return pd.Series(values, index=pd.Index(index), dtype=float, name="equity")


def benchmark_return_for_index(
    data: MarketData | None,
    index: pd.Index,
) -> float:
    if data is None or len(index) < 2:
        return float("nan")
    frames = getattr(data, "benchmark", None) or {}
    frame = next(
        (
            value
            for value in frames.values()
            if value is not None and not value.empty
        ),
        None,
    )
    if frame is None or "Close" not in frame:
        return float("nan")
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    selected = close.loc[(close.index >= index[0]) & (close.index <= index[-1])]
    if len(selected) < 2:
        return float("nan")
    return float(selected.iloc[-1] / selected.iloc[0] - 1.0)


def bounded_number(value: Any) -> float:
    number = float(value)
    if math.isinf(number):
        return 1000.0 if number > 0 else -1000.0
    if math.isnan(number):
        return -1000.0
    return number


def finite_json_value(value: Any) -> float | int:
    if isinstance(value, int):
        return value
    number = float(value)
    if math.isnan(number):
        return 0.0
    if math.isinf(number):
        return 1000.0 if number > 0 else -1000.0
    return number


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    raw = load_json(config_path)
    candidates = build_candidates(raw)
    if args.candidate_limit > 0:
        candidates = candidates[: args.candidate_limit]
    nested = raw["nested"]
    first_outer_year = (
        args.first_outer_year
        or int(nested["first_outer_test_year"])
    )
    last_outer_year = (
        args.last_outer_year
        or int(nested["last_outer_test_year"])
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
    search = NestedStructureSearch(
        raw,
        candidates,
        run_dir,
        resume=not args.no_resume,
    )
    search.initialize()
    summary = search.run(
        first_outer_year=first_outer_year,
        last_outer_year=last_outer_year,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[nested] results={run_dir}", flush=True)


if __name__ == "__main__":
    main()
