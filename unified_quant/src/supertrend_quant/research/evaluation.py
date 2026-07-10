from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from ..config import AppConfig
from ..data import MarketData, common_index
from ..runners import BacktestResult, run_backtest_on_data
from .benchmarks import BenchmarkResult, build_benchmark_report
from .scoring import score_metrics


@dataclass(frozen=True)
class SegmentEvaluation:
    name: str
    index: pd.Index
    backtest: BacktestResult
    benchmarks: Mapping[str, BenchmarkResult]
    score: float

    @property
    def metrics(self) -> Mapping[str, float | int]:
        return self.backtest.metrics

    @property
    def start(self):
        return self.index[0] if len(self.index) else None

    @property
    def end(self):
        return self.index[-1] if len(self.index) else None

    @property
    def bars(self) -> int:
        return len(self.index)


@dataclass(frozen=True)
class EvaluationResult:
    config: AppConfig
    overall: SegmentEvaluation
    train: SegmentEvaluation | None = None
    validation: SegmentEvaluation | None = None
    test: SegmentEvaluation | None = None
    is_partial: bool = False

    @property
    def segments(self) -> dict[str, SegmentEvaluation]:
        if self.is_partial:
            selected = self.ranking_segment()
            return {selected.name: selected}
        return {
            name: segment
            for name in ("overall", "train", "validation", "test")
            if (segment := getattr(self, name)) is not None
        }

    def ranking_segment(self, preferred: str = "validation") -> SegmentEvaluation:
        selected = getattr(self, preferred, None)
        return selected if selected is not None else self.overall


def split_index(
    index: pd.Index,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    *,
    min_segment_bars: int = 3,
) -> dict[str, pd.Index]:
    """Chronologically split an index while always retaining ``overall``."""

    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("train_ratio and validation_ratio must be positive.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than one.")
    if min_segment_bars < 1:
        raise ValueError("min_segment_bars must be positive.")

    ordered = pd.Index(index)
    if ordered.has_duplicates:
        ordered = ordered.drop_duplicates()
    if not ordered.is_monotonic_increasing:
        ordered = ordered.sort_values()

    n = len(ordered)
    if n < min_segment_bars * 3:
        return {"overall": ordered}

    train_end = int(n * train_ratio)
    validation_end = int(n * (train_ratio + validation_ratio))
    train_end = max(min_segment_bars, train_end)
    validation_end = max(train_end + min_segment_bars, validation_end)
    validation_end = min(validation_end, n - min_segment_bars)
    if train_end < min_segment_bars or validation_end - train_end < min_segment_bars:
        return {"overall": ordered}

    return {
        "overall": ordered,
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }


def evaluate_segment(
    config: AppConfig,
    market_data: MarketData,
    name: str,
    run_index: pd.Index,
    *,
    score_kwargs: Mapping[str, float | int] | None = None,
    include_benchmarks: bool = True,
) -> SegmentEvaluation:
    index = pd.Index(run_index)
    if len(index) < 2:
        raise ValueError(f"{name} segment needs at least two bars.")
    backtest = run_backtest_on_data(config, market_data, run_index=index)
    effective_index = pd.Index(backtest.equity.index)
    benchmarks = (
        build_benchmark_report(config, market_data, effective_index)
        if include_benchmarks
        else {}
    )
    return SegmentEvaluation(
        name=name,
        index=effective_index,
        backtest=backtest,
        benchmarks=benchmarks,
        score=score_metrics(backtest.metrics, **dict(score_kwargs or {})),
    )


def evaluate_ranking_segment(
    config: AppConfig,
    market_data: MarketData,
    *,
    preferred: str = "validation",
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    min_segment_bars: int = 3,
    score_kwargs: Mapping[str, float | int] | None = None,
) -> EvaluationResult:
    """Evaluate only the non-holdout segment needed to rank a candidate."""
    if preferred not in {"overall", "train", "validation"}:
        raise ValueError("Research ranking_split must be overall, train, or validation.")
    splits = split_index(
        common_index(market_data.bars),
        train_ratio,
        validation_ratio,
        min_segment_bars=min_segment_bars,
    )
    name = preferred if preferred in splits else "overall"
    segment = evaluate_segment(
        config,
        market_data,
        name,
        splits[name],
        score_kwargs=score_kwargs,
        include_benchmarks=False,
    )
    return EvaluationResult(
        config=config,
        overall=segment,
        train=segment if name == "train" else None,
        validation=segment if name == "validation" else None,
        is_partial=True,
    )


def evaluate_config(
    config: AppConfig,
    market_data: MarketData,
    *,
    use_splits: bool = True,
    include_test: bool = True,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    min_segment_bars: int = 3,
    run_index: pd.Index | None = None,
    score_kwargs: Mapping[str, float | int] | None = None,
) -> EvaluationResult:
    """Evaluate one production config with the canonical backtest engine."""

    base_index = pd.Index(run_index) if run_index is not None else common_index(market_data.bars)
    splits = (
        split_index(
            base_index,
            train_ratio,
            validation_ratio,
            min_segment_bars=min_segment_bars,
        )
        if use_splits
        else {"overall": base_index}
    )
    overall = evaluate_segment(
        config,
        market_data,
        "overall",
        splits["overall"],
        score_kwargs=score_kwargs,
    )
    train = (
        evaluate_segment(
            config,
            market_data,
            "train",
            splits["train"],
            score_kwargs=score_kwargs,
        )
        if "train" in splits
        else None
    )
    validation = (
        evaluate_segment(
            config,
            market_data,
            "validation",
            splits["validation"],
            score_kwargs=score_kwargs,
        )
        if "validation" in splits
        else None
    )
    test = (
        evaluate_segment(
            config,
            market_data,
            "test",
            splits["test"],
            score_kwargs=score_kwargs,
        )
        if include_test and "test" in splits
        else None
    )
    return EvaluationResult(
        config=config,
        overall=overall,
        train=train,
        validation=validation,
        test=test,
    )


__all__ = [
    "EvaluationResult",
    "SegmentEvaluation",
    "evaluate_config",
    "evaluate_ranking_segment",
    "evaluate_segment",
    "split_index",
]
