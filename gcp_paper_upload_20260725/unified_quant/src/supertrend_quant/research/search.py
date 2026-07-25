from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from ..config import AppConfig
from .data_resolver import (
    MarketDataMismatchError,
    MarketDataSource,
    resolve_market_data,
)
from .evaluation import EvaluationResult, evaluate_config, evaluate_ranking_segment
from .overlays import apply_config_overlay


@dataclass(frozen=True)
class SearchRow:
    rank: int
    parameters: Mapping[str, Any]
    config: AppConfig
    evaluation: EvaluationResult
    score: float
    ranking_split: str = "validation"

    @property
    def ranking_metrics(self) -> Mapping[str, float | int]:
        return self.evaluation.ranking_segment(self.ranking_split).metrics

    def as_dict(self) -> dict[str, Any]:
        ranking = self.evaluation.ranking_segment(self.ranking_split)
        return {
            "rank": self.rank,
            "score": self.score,
            "split": ranking.name,
            "total_return": float(ranking.metrics.get("total_return", 0.0)),
            "mdd": float(ranking.metrics.get("mdd", 0.0)),
            "sharpe": float(ranking.metrics.get("sharpe", 0.0)),
            "trade_count": int(ranking.metrics.get("trade_count", 0)),
            **dict(self.parameters),
        }


@dataclass(frozen=True)
class SearchError:
    parameters: Mapping[str, Any]
    error: str


@dataclass(frozen=True)
class SearchResult:
    rows: tuple[SearchRow, ...]
    best_config: AppConfig
    best_evaluation: EvaluationResult
    errors: tuple[SearchError, ...] = ()

    def ranked_dicts(self) -> list[dict[str, Any]]:
        return [row.as_dict() for row in self.rows]


def cartesian_overlays(grid: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    keys = tuple(grid)
    values = []
    for key in keys:
        raw = grid[key]
        if isinstance(raw, (str, bytes)):
            choices = (raw,)
        else:
            try:
                choices = tuple(raw)
            except TypeError:
                choices = (raw,)
        values.append(choices)
    choices_by_key = tuple(values)
    if any(not choices for choices in choices_by_key):
        return
    for combination in itertools.product(*choices_by_key):
        yield dict(zip(keys, combination))


def search_configs(
    base_config: AppConfig,
    market_data: MarketDataSource,
    grid: Mapping[str, Any],
    *,
    ranking_split: str = "validation",
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    min_segment_bars: int = 3,
    score_kwargs: Mapping[str, float | int] | None = None,
    evaluate_test_for_best: bool = True,
    limit: int | None = None,
) -> SearchResult:
    """Evaluate a Cartesian overlay grid and return numerically ranked rows."""

    rows: list[SearchRow] = []
    errors: list[SearchError] = []
    for number, parameters in enumerate(cartesian_overlays(grid), start=1):
        if limit is not None and number > limit:
            break
        try:
            config = apply_config_overlay(base_config, parameters)
            candidate_data = resolve_market_data(
                market_data,
                config,
                fixed_config=base_config,
            )
            evaluation = evaluate_ranking_segment(
                config,
                candidate_data,
                preferred=ranking_split,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                min_segment_bars=min_segment_bars,
                score_kwargs=score_kwargs,
            )
            segment = evaluation.ranking_segment(ranking_split)
            rows.append(
                SearchRow(
                    rank=0,
                    parameters=dict(parameters),
                    config=config,
                    evaluation=evaluation,
                    score=segment.score,
                    ranking_split=ranking_split,
                )
            )
        except MarketDataMismatchError:
            raise
        except Exception as exc:
            errors.append(SearchError(parameters=dict(parameters), error=str(exc)))

    if not rows:
        detail = f" First error: {errors[0].error}" if errors else ""
        raise RuntimeError(f"No research configuration completed successfully.{detail}")

    rows.sort(
        key=lambda row: (
            row.score,
            float(row.evaluation.ranking_segment(ranking_split).metrics.get("total_return", 0.0)),
        ),
        reverse=True,
    )
    ranked = [replace(row, rank=rank) for rank, row in enumerate(rows, start=1)]
    best = ranked[0]
    best_evaluation = best.evaluation
    if evaluate_test_for_best:
        best_data = resolve_market_data(
            market_data,
            best.config,
            fixed_config=base_config,
        )
        best_evaluation = evaluate_config(
            best.config,
            best_data,
            include_test=True,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            min_segment_bars=min_segment_bars,
            score_kwargs=score_kwargs,
        )
        ranked[0] = replace(best, evaluation=best_evaluation)

    return SearchResult(
        rows=tuple(ranked),
        best_config=best.config,
        best_evaluation=best_evaluation,
        errors=tuple(errors),
    )


def run_search(
    config: AppConfig,
    market_data: MarketDataSource,
    grid: Mapping[str, Any],
    **kwargs: Any,
) -> SearchResult:
    """CLI-friendly callable alias for :func:`search_configs`."""

    return search_configs(config, market_data, grid, **kwargs)


__all__ = [
    "SearchError",
    "SearchResult",
    "SearchRow",
    "cartesian_overlays",
    "run_search",
    "search_configs",
]
