from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

from ..config import AppConfig, load_split_config
from ..data import MarketData, common_index
from ..metrics import format_float, format_pct
from ..results import make_run_id, save_backtest_result
from ..runners import BacktestResult, run_backtest_on_data
from ..strategies import create_strategy
from .data_resolver import MarketDataCache, MarketDataSource, resolve_market_data


COMPOSITE_METRICS = (
    "total_return",
    "cagr",
    "mdd",
    "calmar",
    "sharpe",
    "sortino",
    "payoff_ratio",
)
CALMAR_TIEBREAKERS = (
    "calmar",
    "sortino",
    "sharpe",
    "cagr",
    "total_return",
    "mdd",
    "payoff_ratio",
)


@dataclass(frozen=True)
class StrategyComparisonError:
    strategy_path: str
    error: str

    def as_dict(self) -> dict[str, str]:
        return {"strategy_path": self.strategy_path, "error": self.error}


@dataclass(frozen=True)
class StrategyComparisonRow:
    rank: int
    is_best: bool
    strategy_path: str
    strategy_name: str
    strategy_type: str
    composite_score: float
    metrics: Mapping[str, float | int]
    config: AppConfig
    result: BacktestResult

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "best": self.is_best,
            "strategy_path": self.strategy_path,
            "strategy_name": self.strategy_name,
            "strategy_type": self.strategy_type,
            "total_return": float(self.metrics.get("total_return", 0.0)),
            "cagr": float(self.metrics.get("cagr", 0.0)),
            "mdd": float(self.metrics.get("mdd", 0.0)),
            "calmar": float(self.metrics.get("calmar", 0.0)),
            "sharpe": float(self.metrics.get("sharpe", 0.0)),
            "sortino": float(self.metrics.get("sortino", 0.0)),
            "win_rate": float(self.metrics.get("win_rate", 0.0)),
            "payoff_ratio": float(self.metrics.get("payoff_ratio", 0.0)),
            "trade_count": int(self.metrics.get("trade_count", 0)),
            "composite_score": self.composite_score,
        }


@dataclass(frozen=True)
class StrategyComparisonResult:
    rows: tuple[StrategyComparisonRow, ...]
    errors: tuple[StrategyComparisonError, ...]
    rank_by: str
    strategies_dir: Path
    runtime_path: Path
    common_index: pd.Index

    @property
    def winner(self) -> StrategyComparisonRow:
        if not self.rows:
            raise RuntimeError("Strategy comparison has no successful result.")
        return self.rows[0]


@dataclass(frozen=True)
class _Candidate:
    path: Path
    label: str
    config: AppConfig
    data: MarketData
    warmup_bars: int


def discover_strategy_files(strategies_dir: str | Path) -> tuple[Path, ...]:
    root = _resolve_directory(strategies_dir)
    files = tuple(
        sorted(
            (
                path
                for path in root.rglob("*.yaml")
                if not any(part.startswith(".") for part in path.relative_to(root).parts)
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    if not files:
        raise FileNotFoundError(f"No strategy YAML files found under: {root}")
    return files


def compare_strategies(
    strategies_dir: str | Path,
    runtime_path: str | Path,
    *,
    rank_by: str = "calmar",
    market_data: MarketDataSource | None = None,
) -> StrategyComparisonResult:
    if rank_by not in {"calmar", "composite"}:
        raise ValueError("rank_by must be calmar or composite.")

    root = _resolve_directory(strategies_dir)
    runtime = _resolve_file(runtime_path)
    source = market_data or MarketDataCache()
    errors: list[StrategyComparisonError] = []
    candidates: list[_Candidate] = []

    for path in discover_strategy_files(root):
        label = path.relative_to(root).as_posix()
        try:
            config = load_split_config(path, runtime)
            strategy = create_strategy(config)
            data = resolve_market_data(source, config)
            index = common_index(data.bars)
            if len(index) < 2:
                raise RuntimeError("Not enough common market bars for comparison.")
            candidates.append(
                _Candidate(
                    path=path,
                    label=label,
                    config=config,
                    data=data,
                    warmup_bars=max(0, int(strategy.warmup_bars())),
                )
            )
        except Exception as exc:
            errors.append(StrategyComparisonError(label, str(exc)))

    if not candidates:
        detail = f" First error: {errors[0].error}" if errors else ""
        raise RuntimeError(f"No strategy configuration could be prepared.{detail}")

    shared_index = _shared_comparison_index(candidates)
    completed: list[tuple[_Candidate, BacktestResult]] = []
    for candidate in candidates:
        try:
            result = run_backtest_on_data(
                candidate.config,
                candidate.data,
                run_index=shared_index,
            )
            completed.append((candidate, result))
        except Exception as exc:
            errors.append(StrategyComparisonError(candidate.label, str(exc)))

    if not completed:
        detail = f" First error: {errors[0].error}" if errors else ""
        raise RuntimeError(f"No strategy backtest completed successfully.{detail}")

    composite_scores = _composite_scores(completed)
    ordered = sorted(
        completed,
        key=lambda item: _ranking_key(
            item[0].label,
            item[1].metrics,
            composite_scores[item[0].label],
            rank_by,
        ),
    )
    rows = tuple(
        StrategyComparisonRow(
            rank=rank,
            is_best=rank == 1,
            strategy_path=candidate.label,
            strategy_name=candidate.config.strategy.name,
            strategy_type=candidate.config.strategy.type,
            composite_score=composite_scores[candidate.label],
            metrics=result.metrics,
            config=candidate.config,
            result=result,
        )
        for rank, (candidate, result) in enumerate(ordered, start=1)
    )
    return StrategyComparisonResult(
        rows=rows,
        errors=tuple(errors),
        rank_by=rank_by,
        strategies_dir=root,
        runtime_path=runtime,
        common_index=shared_index,
    )


def format_comparison_table(result: StrategyComparisonResult) -> str:
    display = []
    for row in result.rows:
        metrics = row.metrics
        display.append(
            {
                "Rank": row.rank,
                "Best": "BEST" if row.is_best else "",
                "Strategy": row.strategy_name,
                "Type": row.strategy_type,
                "YAML": row.strategy_path,
                "Total Return": format_pct(float(metrics.get("total_return", 0.0))),
                "CAGR": format_pct(float(metrics.get("cagr", 0.0))),
                "MDD": format_pct(float(metrics.get("mdd", 0.0))),
                "Calmar": format_float(float(metrics.get("calmar", 0.0))),
                "Sharpe": format_float(float(metrics.get("sharpe", 0.0))),
                "Sortino": format_float(float(metrics.get("sortino", 0.0))),
                "Win Rate": format_pct(float(metrics.get("win_rate", 0.0))),
                "Payoff": format_float(float(metrics.get("payoff_ratio", 0.0))),
                "Trades": int(metrics.get("trade_count", 0)),
                "Composite": format_float(row.composite_score),
            }
        )
    return pd.DataFrame(display).to_string(index=False)


def save_comparison_result(
    result: StrategyComparisonResult,
    root_dir: str | Path = "results/research/comparisons",
    run_id: str | None = None,
) -> Path:
    resolved_run_id = run_id or make_run_id("all_strategies", "comparison")
    run_dir = Path(root_dir) / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    records = [row.as_dict() for row in result.rows]
    pd.DataFrame(records).to_csv(run_dir / "comparison.csv", index=False)
    summary = {
        "run_id": resolved_run_id,
        "mode": "strategy_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rank_by": result.rank_by,
        "strategies_dir": str(result.strategies_dir),
        "runtime_path": str(result.runtime_path),
        "common_start": str(result.common_index[0]),
        "common_end": str(result.common_index[-1]),
        "common_bars": len(result.common_index),
        "successful_count": len(result.rows),
        "failed_count": len(result.errors),
        "winner": result.winner.as_dict(),
        "rows": records,
        "errors": [error.as_dict() for error in result.errors],
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)

    strategy_root = run_dir / "strategies"
    for row in result.rows:
        save_backtest_result(
            row.result,
            row.config,
            strategy_root,
            run_id=_strategy_result_id(row),
        )
    return run_dir


def _shared_comparison_index(candidates: list[_Candidate]) -> pd.Index:
    shared = common_index(candidates[0].data.bars)
    for candidate in candidates[1:]:
        shared = shared.intersection(common_index(candidate.data.bars), sort=False)
    if not shared.is_monotonic_increasing:
        shared = shared.sort_values()
    max_warmup = max(candidate.warmup_bars for candidate in candidates)
    shared = shared[max_warmup:]
    if len(shared) < 2:
        raise RuntimeError(
            "Not enough shared bars remain after the longest strategy warm-up "
            f"({max_warmup} bars)."
        )
    return shared


def _composite_scores(
    completed: list[tuple[_Candidate, BacktestResult]],
) -> dict[str, float]:
    labels = [candidate.label for candidate, _ in completed]
    percentile_columns = []
    for metric in COMPOSITE_METRICS:
        values = pd.Series(
            [_finite_or_worst(result.metrics.get(metric, 0.0)) for _, result in completed],
            index=labels,
            dtype=float,
        )
        percentile_columns.append(values.rank(method="average", pct=True, ascending=True))
    scores = pd.concat(percentile_columns, axis=1).mean(axis=1) * 100.0
    return {label: float(scores.loc[label]) for label in labels}


def _ranking_key(
    label: str,
    metrics: Mapping[str, float | int],
    composite_score: float,
    rank_by: str,
) -> tuple[object, ...]:
    metric_key = tuple(-_finite_or_worst(metrics.get(metric, 0.0)) for metric in CALMAR_TIEBREAKERS)
    if rank_by == "composite":
        return (-composite_score, *metric_key, label)
    return (*metric_key, label)


def _finite_or_worst(value: float | int) -> float:
    number = float(value)
    return float("-inf") if math.isnan(number) else number


def _strategy_result_id(row: StrategyComparisonRow) -> str:
    stem = Path(row.strategy_path).with_suffix("").as_posix().replace("/", "__")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "strategy"
    return f"{row.rank:03d}_{safe}"


def _resolve_directory(path: str | Path) -> Path:
    resolved = _resolve_path(path)
    if not resolved.is_dir():
        raise NotADirectoryError(f"Strategy directory not found: {path}")
    return resolved


def _resolve_file(path: str | Path) -> Path:
    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Runtime file not found: {path}")
    return resolved


def _resolve_path(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    unified_root = Path(__file__).resolve().parents[3]
    repository_root = unified_root.parent
    for candidate in (Path.cwd() / raw, unified_root / raw, repository_root / raw):
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / raw).resolve()


__all__ = [
    "COMPOSITE_METRICS",
    "StrategyComparisonError",
    "StrategyComparisonResult",
    "StrategyComparisonRow",
    "compare_strategies",
    "discover_strategy_files",
    "format_comparison_table",
    "save_comparison_result",
]
