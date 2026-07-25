from __future__ import annotations

from collections.abc import Mapping


def score_metrics(
    metrics: Mapping[str, float | int],
    *,
    min_trades: int = 5,
    mdd_weight: float = 0.8,
    sharpe_weight: float = 0.03,
    trade_penalty: float = 0.03,
) -> float:
    """Score canonical backtest metrics for research ranking."""

    trades = int(metrics.get("trade_count", 0))
    too_few_trades = max(0, int(min_trades) - trades)
    return float(
        float(metrics.get("total_return", 0.0))
        - abs(float(metrics.get("mdd", 0.0))) * float(mdd_weight)
        + float(metrics.get("sharpe", 0.0)) * float(sharpe_weight)
        - too_few_trades * float(trade_penalty)
    )


__all__ = ["score_metrics"]
