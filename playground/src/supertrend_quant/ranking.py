from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar, Protocol, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .config import ScoringConfig


BenchmarkInput = pd.DataFrame | dict[str, pd.DataFrame] | None


class Scorer(Protocol):
    """Attach comparable scores to symbol frames and rank higher scores first."""

    scoring_type: ClassVar[str]

    def __init__(self, params: Mapping[str, Any], market: str):
        ...

    @classmethod
    def validate_params(cls, params: Mapping[str, Any], market: str | None = None) -> None:
        ...

    def warmup_bars(self) -> int:
        ...

    def add_scores(
        self,
        frames: dict[str, pd.DataFrame],
        benchmark: BenchmarkInput,
    ) -> dict[str, pd.DataFrame]:
        ...

    def rank(self, scores: Mapping[str, Any]) -> tuple[str, ...]:
        ...


_REGISTRY: dict[str, type[Scorer]] = {}


def register_scorer(scorer_cls: type[Scorer]) -> type[Scorer]:
    scoring_type = str(getattr(scorer_cls, "scoring_type", "")).strip()
    if not scoring_type:
        raise ValueError("Scorer classes must define a non-empty scoring_type.")
    if scoring_type in _REGISTRY:
        raise ValueError(f"Scoring type already registered: {scoring_type}")
    _REGISTRY[scoring_type] = scorer_cls
    return scorer_cls


def available_scorers() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_scorer_class(scoring_type: str) -> type[Scorer]:
    try:
        return _REGISTRY[scoring_type]
    except KeyError as exc:
        available = ", ".join(available_scorers()) or "<none>"
        raise ValueError(
            f"Unsupported scoring type: {scoring_type}. Available scorers: {available}"
        ) from exc


def validate_scoring_config(config: ScoringConfig, market: str | None = None) -> None:
    scoring_type = str(config.type).strip()
    if not scoring_type:
        raise ValueError("scoring.type is required.")
    scorer_cls = get_scorer_class(scoring_type)
    scorer_cls.validate_params(config.params, market)


def create_scorer(config: ScoringConfig, market: str) -> Scorer:
    validate_scoring_config(config, market)
    scorer_cls = get_scorer_class(str(config.type).strip())
    return scorer_cls(config.params, market)


def rank_scores(scores: Mapping[str, Any]) -> tuple[str, ...]:
    """Return finite scores in deterministic best-first order."""

    valid: list[tuple[str, float]] = []
    for symbol, raw_score in scores.items():
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            valid.append((str(symbol), score))
    return tuple(symbol for symbol, _ in sorted(valid, key=lambda item: (-item[1], item[0])))


@register_scorer
class RelativeStrengthScorer:
    scoring_type = "relative_strength"

    def __init__(self, params: Mapping[str, Any], market: str):
        self.params = dict(params)
        self.market = str(market).upper()
        self.validate_params(self.params, self.market)
        self.lookback_bars = effective_relative_strength_lookback(self.params, self.market)

    @classmethod
    def validate_params(cls, params: Mapping[str, Any], market: str | None = None) -> None:
        unknown = set(params) - {"lookback_bars"}
        if unknown:
            raise ValueError(
                f"Unsupported params for scoring type={cls.scoring_type}: {', '.join(sorted(unknown))}"
            )
        if "lookback_bars" not in params:
            raise ValueError("scoring.params.lookback_bars is required for relative_strength.")
        lookback = params["lookback_bars"]
        if isinstance(lookback, Mapping):
            if not lookback:
                raise ValueError("scoring.params.lookback_bars mapping cannot be empty.")
            for key, value in lookback.items():
                _positive_int(value, f"scoring.params.lookback_bars.{key}")
            if market is not None:
                effective_relative_strength_lookback(params, market)
            return
        _positive_int(lookback, "scoring.params.lookback_bars")

    def warmup_bars(self) -> int:
        return self.lookback_bars + 1

    def add_scores(
        self,
        frames: dict[str, pd.DataFrame],
        benchmark: BenchmarkInput,
    ) -> dict[str, pd.DataFrame]:
        scored: dict[str, pd.DataFrame] = {}
        for symbol, frame in frames.items():
            out = frame.copy()
            out["Score"] = float("nan")
            symbol_benchmark = _benchmark_for_symbol(symbol, benchmark)
            if (
                "Close" not in out
                or symbol_benchmark is None
                or symbol_benchmark.empty
                or "Close" not in symbol_benchmark
            ):
                scored[symbol] = out
                continue
            benchmark_return = symbol_benchmark["Close"].pct_change(self.lookback_bars)
            aligned_benchmark_return = benchmark_return.reindex(out.index, method="ffill")
            out["Score"] = out["Close"].pct_change(self.lookback_bars) - aligned_benchmark_return
            scored[symbol] = out
        return scored

    def rank(self, scores: Mapping[str, Any]) -> tuple[str, ...]:
        return rank_scores(scores)


@register_scorer
class VolAdjustedRelativeStrengthScorer:
    scoring_type = "vol_adjusted_relative_strength"

    def __init__(self, params: Mapping[str, Any], market: str):
        self.params = dict(params)
        self.market = str(market).upper()
        _validate_lookback_params(self.params, self.scoring_type, market)
        self.lookback_bars = effective_relative_strength_lookback(self.params, self.market)

    @classmethod
    def validate_params(cls, params: Mapping[str, Any], market: str | None = None) -> None:
        _validate_lookback_params(params, cls.scoring_type, market)

    def warmup_bars(self) -> int:
        return self.lookback_bars + 1

    def add_scores(
        self,
        frames: dict[str, pd.DataFrame],
        benchmark: BenchmarkInput,
    ) -> dict[str, pd.DataFrame]:
        scored: dict[str, pd.DataFrame] = {}
        for symbol, frame in frames.items():
            out = frame.copy()
            out["Score"] = float("nan")
            benchmark_close = _aligned_benchmark_close(symbol, out, benchmark)
            if "Close" not in out or benchmark_close is None:
                scored[symbol] = out
                continue
            stock_return = out["Close"].pct_change(self.lookback_bars)
            benchmark_return = benchmark_close.pct_change(self.lookback_bars)
            period_vol = out["Close"].pct_change().rolling(self.lookback_bars).std() * math.sqrt(self.lookback_bars)
            out["Score"] = (stock_return - benchmark_return) / period_vol.replace(0.0, float("nan"))
            scored[symbol] = out
        return scored

    def rank(self, scores: Mapping[str, Any]) -> tuple[str, ...]:
        return rank_scores(scores)


@register_scorer
class CompositeRelativeStrengthScorer:
    scoring_type = "composite_relative_strength"

    def __init__(self, params: Mapping[str, Any], market: str):
        self.params = dict(params)
        self.market = str(market).upper()
        _validate_lookback_params(self.params, self.scoring_type, market)
        self.lookback_bars = effective_relative_strength_lookback(self.params, self.market)
        short = max(2, self.lookback_bars // 2)
        long = max(self.lookback_bars + 1, self.lookback_bars * 2)
        self.lookbacks = (short, self.lookback_bars, long)
        self.weights = (0.3, 0.5, 0.2)

    @classmethod
    def validate_params(cls, params: Mapping[str, Any], market: str | None = None) -> None:
        _validate_lookback_params(params, cls.scoring_type, market)

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
            benchmark_close = _aligned_benchmark_close(symbol, out, benchmark)
            if "Close" not in out or benchmark_close is None:
                scored[symbol] = out
                continue
            score = pd.Series(0.0, index=out.index)
            for lookback, weight in zip(self.lookbacks, self.weights):
                stock_return = out["Close"].pct_change(lookback)
                benchmark_return = benchmark_close.pct_change(lookback)
                score = score + weight * (stock_return - benchmark_return)
            out["Score"] = score
            scored[symbol] = out
        return scored

    def rank(self, scores: Mapping[str, Any]) -> tuple[str, ...]:
        return rank_scores(scores)


@register_scorer
class SkipRecentRelativeStrengthScorer:
    scoring_type = "skip_recent_relative_strength"

    def __init__(self, params: Mapping[str, Any], market: str):
        self.params = dict(params)
        self.market = str(market).upper()
        _validate_lookback_params(self.params, self.scoring_type, market, extra={"skip_bars"})
        self.lookback_bars = effective_relative_strength_lookback(self.params, self.market)
        self.skip_bars = int(self.params.get("skip_bars", 21))
        if self.skip_bars < 1 or self.skip_bars >= self.lookback_bars:
            raise ValueError("skip_recent_relative_strength requires 1 <= skip_bars < lookback_bars.")

    @classmethod
    def validate_params(cls, params: Mapping[str, Any], market: str | None = None) -> None:
        _validate_lookback_params(params, cls.scoring_type, market, extra={"skip_bars"})
        lookback = effective_relative_strength_lookback(params, str(market or "US"))
        skip = int(params.get("skip_bars", 21))
        if skip < 1 or skip >= lookback:
            raise ValueError("skip_recent_relative_strength requires 1 <= skip_bars < lookback_bars.")

    def warmup_bars(self) -> int:
        return self.lookback_bars + 1

    def add_scores(
        self,
        frames: dict[str, pd.DataFrame],
        benchmark: BenchmarkInput,
    ) -> dict[str, pd.DataFrame]:
        scored: dict[str, pd.DataFrame] = {}
        for symbol, frame in frames.items():
            out = frame.copy()
            out["Score"] = float("nan")
            benchmark_close = _aligned_benchmark_close(symbol, out, benchmark)
            if "Close" not in out or benchmark_close is None:
                scored[symbol] = out
                continue
            stock_return = out["Close"].shift(self.skip_bars) / out["Close"].shift(self.lookback_bars) - 1.0
            benchmark_return = benchmark_close.shift(self.skip_bars) / benchmark_close.shift(self.lookback_bars) - 1.0
            out["Score"] = stock_return - benchmark_return
            scored[symbol] = out
        return scored

    def rank(self, scores: Mapping[str, Any]) -> tuple[str, ...]:
        return rank_scores(scores)


@register_scorer
class BetaAdjustedAlphaScorer:
    scoring_type = "beta_adjusted_alpha"

    def __init__(self, params: Mapping[str, Any], market: str):
        self.params = dict(params)
        self.market = str(market).upper()
        _validate_lookback_params(self.params, self.scoring_type, market)
        self.lookback_bars = effective_relative_strength_lookback(self.params, self.market)

    @classmethod
    def validate_params(cls, params: Mapping[str, Any], market: str | None = None) -> None:
        _validate_lookback_params(params, cls.scoring_type, market)

    def warmup_bars(self) -> int:
        return self.lookback_bars + 1

    def add_scores(
        self,
        frames: dict[str, pd.DataFrame],
        benchmark: BenchmarkInput,
    ) -> dict[str, pd.DataFrame]:
        scored: dict[str, pd.DataFrame] = {}
        for symbol, frame in frames.items():
            out = frame.copy()
            out["Score"] = float("nan")
            benchmark_close = _aligned_benchmark_close(symbol, out, benchmark)
            if "Close" not in out or benchmark_close is None:
                scored[symbol] = out
                continue
            stock_daily = out["Close"].pct_change()
            benchmark_daily = benchmark_close.pct_change()
            benchmark_var = benchmark_daily.rolling(self.lookback_bars).var()
            beta = stock_daily.rolling(self.lookback_bars).cov(benchmark_daily) / benchmark_var.replace(0.0, float("nan"))
            stock_return = out["Close"].pct_change(self.lookback_bars)
            benchmark_return = benchmark_close.pct_change(self.lookback_bars)
            out["Score"] = stock_return - beta * benchmark_return
            scored[symbol] = out
        return scored

    def rank(self, scores: Mapping[str, Any]) -> tuple[str, ...]:
        return rank_scores(scores)


@register_scorer
class DualMomentumScorer:
    scoring_type = "dual_momentum"

    def __init__(self, params: Mapping[str, Any], market: str):
        self.params = dict(params)
        self.market = str(market).upper()
        _validate_lookback_params(self.params, self.scoring_type, market)
        self.lookback_bars = effective_relative_strength_lookback(self.params, self.market)

    @classmethod
    def validate_params(cls, params: Mapping[str, Any], market: str | None = None) -> None:
        _validate_lookback_params(params, cls.scoring_type, market)

    def warmup_bars(self) -> int:
        return self.lookback_bars + 1

    def add_scores(
        self,
        frames: dict[str, pd.DataFrame],
        benchmark: BenchmarkInput,
    ) -> dict[str, pd.DataFrame]:
        scored: dict[str, pd.DataFrame] = {}
        for symbol, frame in frames.items():
            out = frame.copy()
            out["Score"] = float("nan")
            benchmark_close = _aligned_benchmark_close(symbol, out, benchmark)
            if "Close" not in out or benchmark_close is None:
                scored[symbol] = out
                continue
            stock_return = out["Close"].pct_change(self.lookback_bars)
            benchmark_return = benchmark_close.pct_change(self.lookback_bars)
            excess = stock_return - benchmark_return
            out["Score"] = excess.where((stock_return > 0.0) & (excess > 0.0))
            scored[symbol] = out
        return scored

    def rank(self, scores: Mapping[str, Any]) -> tuple[str, ...]:
        return rank_scores(scores)


def effective_relative_strength_lookback(params: Mapping[str, Any], market: str) -> int:
    lookback = params.get("lookback_bars")
    if not isinstance(lookback, Mapping):
        return _positive_int(lookback, "scoring.params.lookback_bars")

    normalized = {str(key).upper(): value for key, value in lookback.items()}
    market_key = str(market).upper()
    selected = normalized.get(market_key)
    if selected is None:
        selected = normalized.get("DEFAULT", normalized.get("US"))
    if selected is None:
        raise ValueError(
            f"scoring.params.lookback_bars requires {market_key}, default, or US."
        )
    return _positive_int(selected, f"scoring.params.lookback_bars.{market_key}")


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer.") from exc
    if parsed < 1 or parsed != value:
        raise ValueError(f"{label} must be a positive integer.")
    return parsed


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
            f"Unsupported params for scoring type={scoring_type}: {', '.join(sorted(unknown))}"
        )
    if "lookback_bars" not in params:
        raise ValueError(f"scoring.params.lookback_bars is required for {scoring_type}.")
    lookback = params["lookback_bars"]
    if isinstance(lookback, Mapping):
        if not lookback:
            raise ValueError("scoring.params.lookback_bars mapping cannot be empty.")
        for key, value in lookback.items():
            _positive_int(value, f"scoring.params.lookback_bars.{key}")
        if market is not None:
            effective_relative_strength_lookback(params, market)
        return
    _positive_int(lookback, "scoring.params.lookback_bars")


def _benchmark_for_symbol(symbol: str, benchmark: BenchmarkInput) -> pd.DataFrame | None:
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
    if symbol_benchmark is None or symbol_benchmark.empty or "Close" not in symbol_benchmark:
        return None
    return symbol_benchmark["Close"].reindex(frame.index, method="ffill")


__all__ = [
    "BetaAdjustedAlphaScorer",
    "BenchmarkInput",
    "CompositeRelativeStrengthScorer",
    "DualMomentumScorer",
    "RelativeStrengthScorer",
    "Scorer",
    "SkipRecentRelativeStrengthScorer",
    "VolAdjustedRelativeStrengthScorer",
    "available_scorers",
    "create_scorer",
    "effective_relative_strength_lookback",
    "get_scorer_class",
    "rank_scores",
    "register_scorer",
    "validate_scoring_config",
]
