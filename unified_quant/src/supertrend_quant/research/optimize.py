from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..config import AppConfig
from ..data import MarketData, common_index
from .data_resolver import (
    MarketDataMismatchError,
    MarketDataSource,
    resolve_market_data,
)
from .evaluation import EvaluationResult, evaluate_config, evaluate_segment, split_index
from .overlays import apply_config_overlay


SuggestConfig = Callable[[Any, AppConfig], AppConfig]


@dataclass(frozen=True)
class OptimizationSpace:
    timeframes: tuple[str, ...] = ()
    entry_types: tuple[str, ...] = ("single", "triple")
    market_filters: tuple[str, ...] = ("none", "1d")
    asset_filters: tuple[str, ...] = (
        "none",
        "ichimoku_cloud",
        "ema_trend",
        "ichimoku_cloud+ema_trend",
    )
    min_rs_period: int = 10
    max_rs_period: int = 200
    min_sell_confirm_bars: int = 1
    max_sell_confirm_bars: int = 30
    min_st_period: int = 5
    max_st_period: int = 30
    min_st_multiplier: float = 1.0
    max_st_multiplier: float = 5.0
    st_multiplier_step: float = 0.1
    triple_settings: tuple[tuple[int, float], ...] = (
        (10, 1.0),
        (11, 2.0),
        (12, 3.0),
    )


@dataclass(frozen=True)
class OptimizationTrial:
    number: int
    score: float | None
    state: str
    parameters: Mapping[str, Any]
    validation_metrics: Mapping[str, float | int]
    error: str | None = None


@dataclass(frozen=True)
class OptimizationResult:
    best_config: AppConfig
    best_evaluation: EvaluationResult
    best_score: float
    best_parameters: Mapping[str, Any]
    trials: tuple[OptimizationTrial, ...]
    study: Any


def require_optuna():
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Optuna is optional and required only for research optimization. "
            "Install the project's optimization dependencies first."
        ) from exc
    return optuna


def suggest_default_config(
    trial: Any,
    base_config: AppConfig,
    space: OptimizationSpace,
) -> AppConfig:
    """Suggest a conditional overlay without inactive triple/single knobs."""

    timeframes = space.timeframes or (base_config.timeframe,)
    entry_type = trial.suggest_categorical("entry", list(space.entry_types))
    overlay: dict[str, Any] = {
        "timeframe": trial.suggest_categorical("timeframe", list(timeframes)),
        "entry": entry_type,
        "market_filter": trial.suggest_categorical(
            "market_filter", list(space.market_filters)
        ),
        "asset_filter": trial.suggest_categorical(
            "asset_filter", list(space.asset_filters)
        ),
        "rs_period": trial.suggest_int(
            "rs_period", space.min_rs_period, space.max_rs_period
        ),
        "sell_confirm_bars": trial.suggest_int(
            "sell_confirm_bars",
            space.min_sell_confirm_bars,
            space.max_sell_confirm_bars,
        ),
    }
    if entry_type in {"single", "single_supertrend", "supertrend"}:
        overlay["st_period"] = trial.suggest_int(
            "st_period", space.min_st_period, space.max_st_period
        )
        overlay["st_multiplier"] = trial.suggest_float(
            "st_multiplier",
            space.min_st_multiplier,
            space.max_st_multiplier,
            step=space.st_multiplier_step,
        )
    else:
        overlay["triple_settings"] = space.triple_settings
    return apply_config_overlay(base_config, overlay)


def _trial_report(study: Any) -> tuple[OptimizationTrial, ...]:
    rows = []
    for trial in study.trials:
        metrics = trial.user_attrs.get("validation_metrics", {})
        rows.append(
            OptimizationTrial(
                number=int(trial.number),
                score=float(trial.value) if trial.value is not None else None,
                state=str(trial.state.name if hasattr(trial.state, "name") else trial.state),
                parameters=dict(trial.params),
                validation_metrics=dict(metrics) if isinstance(metrics, Mapping) else {},
                error=trial.user_attrs.get("error"),
            )
        )
    return tuple(rows)


def optimize_config(
    base_config: AppConfig,
    market_data: MarketDataSource,
    *,
    n_trials: int = 100,
    timeout: float | None = None,
    seed: int = 42,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    min_segment_bars: int = 3,
    score_kwargs: Mapping[str, float | int] | None = None,
    space: OptimizationSpace | None = None,
    suggest_config: SuggestConfig | None = None,
    study_name: str | None = None,
    storage: str | None = None,
    load_if_exists: bool = False,
    n_jobs: int = 1,
) -> OptimizationResult:
    """Optimize on validation only, then evaluate the selected test once."""

    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    optuna = require_optuna()
    if space is not None:
        resolved_space = space
    else:
        fixed_market_filters = (
            ("none", base_config.market_trend_filter.timeframe)
            if base_config.market_trend_filter.enabled
            else ("none",)
        )
        resolved_space = OptimizationSpace(
            market_filters=(
                fixed_market_filters
                if isinstance(market_data, MarketData)
                else ("none", "1d")
            )
        )
    proposed_configs: dict[int, AppConfig] = {}

    def objective(trial: Any) -> float:
        try:
            candidate = (
                suggest_config(trial, base_config)
                if suggest_config is not None
                else suggest_default_config(trial, base_config, resolved_space)
            )
            proposed_configs[int(trial.number)] = candidate
            candidate_data = resolve_market_data(
                market_data,
                candidate,
                fixed_config=base_config,
            )
            segments = split_index(
                common_index(candidate_data.bars),
                train_ratio,
                validation_ratio,
                min_segment_bars=min_segment_bars,
            )
            validation_index = segments.get("validation")
            if validation_index is None:
                raise ValueError("Not enough data for a validation segment.")
            evaluation = evaluate_segment(
                candidate,
                candidate_data,
                "validation",
                validation_index,
                score_kwargs=score_kwargs,
                include_benchmarks=False,
            )
            trial.set_user_attr("validation_metrics", dict(evaluation.metrics))
            return float(evaluation.score)
        except MarketDataMismatchError:
            raise
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            raise optuna.TrialPruned(str(exc)) from exc

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=False,
    )
    try:
        best_trial = study.best_trial
    except ValueError as exc:
        raise RuntimeError("Optimization produced no completed trial.") from exc

    best_config = proposed_configs.get(int(best_trial.number))
    if best_config is None:
        raise RuntimeError(
            "The best trial was loaded from storage but its AppConfig was not reconstructed. "
            "Provide a deterministic suggest_config callback and run at least one new trial."
        )

    # This is deliberately the first and only evaluation of the holdout test
    # during optimization. Trial objective calls above touch validation only.
    best_data = resolve_market_data(
        market_data,
        best_config,
        fixed_config=base_config,
    )
    best_evaluation = evaluate_config(
        best_config,
        best_data,
        include_test=True,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        min_segment_bars=min_segment_bars,
        score_kwargs=score_kwargs,
    )
    if best_evaluation.test is not None:
        study.set_user_attr("best_test_metrics", dict(best_evaluation.test.metrics))

    return OptimizationResult(
        best_config=best_config,
        best_evaluation=best_evaluation,
        best_score=float(best_trial.value),
        best_parameters=dict(best_trial.params),
        trials=_trial_report(study),
        study=study,
    )


def run_optimize(
    config: AppConfig,
    market_data: MarketDataSource,
    **kwargs: Any,
) -> OptimizationResult:
    """CLI-friendly callable alias for :func:`optimize_config`."""

    return optimize_config(config, market_data, **kwargs)


__all__ = [
    "OptimizationResult",
    "OptimizationSpace",
    "OptimizationTrial",
    "SuggestConfig",
    "optimize_config",
    "require_optuna",
    "run_optimize",
    "suggest_default_config",
]
