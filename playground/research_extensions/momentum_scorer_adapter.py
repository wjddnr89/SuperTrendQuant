from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pandas as pd

from supertrend_quant.config import AppConfig
from supertrend_quant.ranking import (
    BenchmarkInput,
    available_scorers,
    effective_relative_strength_lookback,
    get_scorer_class,
    rank_scores,
    register_scorer,
    validate_scoring_config,
)


RESEARCH_MOMENTUM_ALIASES = {
    "relative_strength": "relative_strength",
    "excess": "relative_strength",
    "excess_rs": "relative_strength",
    "vol_adjusted": "vol_adjusted_relative_strength",
    "vol_adjusted_rs": "vol_adjusted_relative_strength",
    "vol_adjusted_relative_strength": "vol_adjusted_relative_strength",
    "composite": "composite_relative_strength",
    "composite_rs": "composite_relative_strength",
    "composite_relative_strength": "composite_relative_strength",
    "skip_recent": "skip_recent_relative_strength",
    "skip_recent_rs": "skip_recent_relative_strength",
    "skip_1m": "skip_recent_relative_strength",
    "skip_1m_rs": "skip_recent_relative_strength",
    "skip_recent_relative_strength": "skip_recent_relative_strength",
    "beta_adjusted": "beta_adjusted_alpha",
    "beta_adjusted_alpha": "beta_adjusted_alpha",
    "dual_momentum": "dual_momentum",
}


def _validate_lookback_params(
    params: Mapping[str, Any],
    scoring_type: str,
    market: str | None = None,
    *,
    extra: set[str] | None = None,
) -> None:
    allowed = {"lookback_bars"} | set(extra or ())
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(
            f"Unsupported params for scoring type={scoring_type}: "
            + ", ".join(sorted(unknown))
        )
    if "lookback_bars" not in params:
        raise ValueError(
            f"scoring.params.lookback_bars is required for {scoring_type}."
        )
    effective_relative_strength_lookback(
        params,
        str(market or "US"),
    )


def _benchmark_for_symbol(
    symbol: str,
    benchmark: BenchmarkInput,
) -> pd.DataFrame | None:
    if benchmark is None:
        return None
    if isinstance(benchmark, dict):
        return benchmark.get(symbol)
    return benchmark


def _aligned_benchmark_close(
    symbol: str,
    frame: pd.DataFrame,
    benchmark: BenchmarkInput,
) -> pd.Series | None:
    symbol_benchmark = _benchmark_for_symbol(symbol, benchmark)
    if (
        symbol_benchmark is None
        or symbol_benchmark.empty
        or "Close" not in symbol_benchmark
    ):
        return None
    return symbol_benchmark["Close"].reindex(frame.index, method="ffill")


def _identity_safe_pct_change(
    frame: pd.DataFrame,
    periods: int,
) -> pd.Series:
    if "IdentitySegment" in frame and frame["IdentitySegment"].nunique(
        dropna=False
    ) > 1:
        return frame.groupby(
            "IdentitySegment",
            sort=False,
            dropna=False,
        )["Close"].transform(
            lambda values: values.pct_change(
                periods,
                fill_method=None,
            )
        )
    return frame["Close"].pct_change(periods, fill_method=None)


def _identity_safe_daily_returns(frame: pd.DataFrame) -> pd.Series:
    return _identity_safe_pct_change(frame, 1)


def _identity_safe_skip_return(
    frame: pd.DataFrame,
    lookback_bars: int,
    skip_bars: int,
) -> pd.Series:
    def calculate(values: pd.Series) -> pd.Series:
        return (
            values.shift(skip_bars)
            / values.shift(lookback_bars)
            - 1.0
        )

    if "IdentitySegment" in frame and frame["IdentitySegment"].nunique(
        dropna=False
    ) > 1:
        return frame.groupby(
            "IdentitySegment",
            sort=False,
            dropna=False,
        )["Close"].transform(calculate)
    return calculate(frame["Close"])


class _ResearchScorer:
    scoring_type = ""

    def __init__(self, params: Mapping[str, Any], market: str):
        self.params = dict(params)
        self.market = str(market).upper()
        self.validate_params(self.params, self.market)
        self.lookback_bars = effective_relative_strength_lookback(
            self.params,
            self.market,
        )

    @classmethod
    def validate_params(
        cls,
        params: Mapping[str, Any],
        market: str | None = None,
    ) -> None:
        _validate_lookback_params(params, cls.scoring_type, market)

    def warmup_bars(self) -> int:
        return self.lookback_bars + 1

    def rank(self, scores: Mapping[str, Any]) -> tuple[str, ...]:
        return rank_scores(scores)


class VolAdjustedRelativeStrengthScorer(_ResearchScorer):
    scoring_type = "vol_adjusted_relative_strength"

    def add_scores(
        self,
        frames: dict[str, pd.DataFrame],
        benchmark: BenchmarkInput,
    ) -> dict[str, pd.DataFrame]:
        scored: dict[str, pd.DataFrame] = {}
        for symbol, frame in frames.items():
            out = frame.copy()
            out["Score"] = float("nan")
            benchmark_close = _aligned_benchmark_close(
                symbol,
                out,
                benchmark,
            )
            if "Close" not in out or benchmark_close is None:
                scored[symbol] = out
                continue
            stock_return = _identity_safe_pct_change(
                out,
                self.lookback_bars,
            )
            benchmark_return = benchmark_close.pct_change(
                self.lookback_bars,
                fill_method=None,
            )
            period_vol = (
                _identity_safe_daily_returns(out)
                .rolling(self.lookback_bars)
                .std()
                * math.sqrt(self.lookback_bars)
            )
            out["Score"] = (
                (stock_return - benchmark_return)
                / period_vol.replace(0.0, float("nan"))
            )
            scored[symbol] = out
        return scored


class CompositeRelativeStrengthScorer(_ResearchScorer):
    scoring_type = "composite_relative_strength"

    def __init__(self, params: Mapping[str, Any], market: str):
        super().__init__(params, market)
        short = max(2, self.lookback_bars // 2)
        long = max(self.lookback_bars + 1, self.lookback_bars * 2)
        self.lookbacks = (short, self.lookback_bars, long)
        self.weights = (0.3, 0.5, 0.2)

    def warmup_bars(self) -> int:
        return max(self.lookbacks) + 1

    def add_scores(
        self,
        frames: dict[str, pd.DataFrame],
        benchmark: BenchmarkInput,
    ) -> dict[str, pd.DataFrame]:
        scored: dict[str, pd.DataFrame] = {}
        for symbol, frame in frames.items():
            out = frame.copy()
            out["Score"] = float("nan")
            benchmark_close = _aligned_benchmark_close(
                symbol,
                out,
                benchmark,
            )
            if "Close" not in out or benchmark_close is None:
                scored[symbol] = out
                continue
            score = pd.Series(0.0, index=out.index)
            for lookback, weight in zip(self.lookbacks, self.weights):
                stock_return = _identity_safe_pct_change(out, lookback)
                benchmark_return = benchmark_close.pct_change(
                    lookback,
                    fill_method=None,
                )
                score = score + weight * (
                    stock_return - benchmark_return
                )
            out["Score"] = score
            scored[symbol] = out
        return scored


class SkipRecentRelativeStrengthScorer(_ResearchScorer):
    scoring_type = "skip_recent_relative_strength"

    def __init__(self, params: Mapping[str, Any], market: str):
        super().__init__(params, market)
        self.skip_bars = int(self.params.get("skip_bars", 21))
        if self.skip_bars < 1 or self.skip_bars >= self.lookback_bars:
            raise ValueError(
                "skip_recent_relative_strength requires "
                "1 <= skip_bars < lookback_bars."
            )

    @classmethod
    def validate_params(
        cls,
        params: Mapping[str, Any],
        market: str | None = None,
    ) -> None:
        _validate_lookback_params(
            params,
            cls.scoring_type,
            market,
            extra={"skip_bars"},
        )
        lookback = effective_relative_strength_lookback(
            params,
            str(market or "US"),
        )
        skip = int(params.get("skip_bars", 21))
        if skip < 1 or skip >= lookback:
            raise ValueError(
                "skip_recent_relative_strength requires "
                "1 <= skip_bars < lookback_bars."
            )

    def add_scores(
        self,
        frames: dict[str, pd.DataFrame],
        benchmark: BenchmarkInput,
    ) -> dict[str, pd.DataFrame]:
        scored: dict[str, pd.DataFrame] = {}
        for symbol, frame in frames.items():
            out = frame.copy()
            out["Score"] = float("nan")
            benchmark_close = _aligned_benchmark_close(
                symbol,
                out,
                benchmark,
            )
            if "Close" not in out or benchmark_close is None:
                scored[symbol] = out
                continue
            stock_return = _identity_safe_skip_return(
                out,
                self.lookback_bars,
                self.skip_bars,
            )
            benchmark_return = (
                benchmark_close.shift(self.skip_bars)
                / benchmark_close.shift(self.lookback_bars)
                - 1.0
            )
            out["Score"] = stock_return - benchmark_return
            scored[symbol] = out
        return scored


class BetaAdjustedAlphaScorer(_ResearchScorer):
    scoring_type = "beta_adjusted_alpha"

    def add_scores(
        self,
        frames: dict[str, pd.DataFrame],
        benchmark: BenchmarkInput,
    ) -> dict[str, pd.DataFrame]:
        scored: dict[str, pd.DataFrame] = {}
        for symbol, frame in frames.items():
            out = frame.copy()
            out["Score"] = float("nan")
            benchmark_close = _aligned_benchmark_close(
                symbol,
                out,
                benchmark,
            )
            if "Close" not in out or benchmark_close is None:
                scored[symbol] = out
                continue
            stock_daily = _identity_safe_daily_returns(out)
            benchmark_daily = benchmark_close.pct_change(fill_method=None)
            benchmark_var = benchmark_daily.rolling(
                self.lookback_bars
            ).var()
            beta = (
                stock_daily.rolling(self.lookback_bars).cov(benchmark_daily)
                / benchmark_var.replace(0.0, float("nan"))
            )
            stock_return = _identity_safe_pct_change(
                out,
                self.lookback_bars,
            )
            benchmark_return = benchmark_close.pct_change(
                self.lookback_bars,
                fill_method=None,
            )
            out["Score"] = stock_return - beta * benchmark_return
            scored[symbol] = out
        return scored


RESEARCH_SCORER_CLASSES = (
    VolAdjustedRelativeStrengthScorer,
    CompositeRelativeStrengthScorer,
    SkipRecentRelativeStrengthScorer,
    BetaAdjustedAlphaScorer,
)


def register_research_momentum_scorers() -> tuple[str, ...]:
    registered: list[str] = []
    for scorer_cls in RESEARCH_SCORER_CLASSES:
        scoring_type = scorer_cls.scoring_type
        if scoring_type in available_scorers():
            if get_scorer_class(scoring_type) is not scorer_cls:
                raise ValueError(
                    "A different scorer is already registered for "
                    f"{scoring_type}."
                )
        else:
            register_scorer(scorer_cls)
        registered.append(scoring_type)
    return tuple(registered)


def with_research_momentum_scoring(
    config: AppConfig,
    *,
    method: str | None = None,
    period: int | Mapping[str, int] | None = None,
    skip_bars: int | None = None,
) -> AppConfig:
    scoring_type = str(config.scoring.type)
    if method is not None:
        raw_method = str(method).strip().lower()
        try:
            scoring_type = RESEARCH_MOMENTUM_ALIASES[raw_method]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported research rs_method: {raw_method}"
            ) from exc

    if period is None:
        lookback: int | Mapping[str, int] = config.scoring.params.get(
            "lookback_bars",
            100,
        )
    elif isinstance(period, Mapping):
        lookback = {
            str(key).upper(): int(value)
            for key, value in period.items()
        }
    else:
        lookback = int(period)

    params: dict[str, Any] = {"lookback_bars": lookback}
    if scoring_type == "skip_recent_relative_strength":
        params["skip_bars"] = int(
            skip_bars
            if skip_bars is not None
            else config.scoring.params.get("skip_bars", 21)
        )
    elif skip_bars is not None:
        raise ValueError(
            "rs_skip_bars is valid only for skip_recent momentum."
        )

    updated = replace(
        config,
        scoring=replace(
            config.scoring,
            type=scoring_type,
            params=params,
        ),
    )
    validate_scoring_config(updated.scoring, updated.market)
    return updated


register_research_momentum_scorers()


__all__ = [
    "BetaAdjustedAlphaScorer",
    "CompositeRelativeStrengthScorer",
    "RESEARCH_MOMENTUM_ALIASES",
    "RESEARCH_SCORER_CLASSES",
    "SkipRecentRelativeStrengthScorer",
    "VolAdjustedRelativeStrengthScorer",
    "register_research_momentum_scorers",
    "with_research_momentum_scoring",
]
