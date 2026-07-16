# -*- coding: utf-8 -*-
"""
Performance metrics and scoring helpers for modular strategy experiments.

The same metric functions are used for strategy equity, Equal B&H, QQQ B&H, and
KR market-index B&H so optimization results are comparable across configs.
"""

import math
from typing import Dict, Iterable

import numpy as np
import pandas as pd


def annualization_factor(interval: str) -> float:
    mapping = {
        "30m": 13 * 252,
        "1h": 6.5 * 252,
        "2h": 3.25 * 252,
        "4h": 1.625 * 252,
        "1d": 252,
    }
    return mapping.get(interval, 252)


def calculate_metrics(equity: pd.Series, trades: Iterable[dict], interval: str) -> Dict[str, float]:
    equity = equity.dropna()
    if len(equity) < 2:
        return {
            "total_return": 0.0,
            "mdd": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "payoff_ratio": 0.0,
            "trade_count": 0,
        }

    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    drawdown = equity / equity.cummax() - 1.0
    mdd = drawdown.min()

    if len(returns) > 1 and returns.std(ddof=1) > 0:
        sharpe = returns.mean() / returns.std(ddof=1) * math.sqrt(annualization_factor(interval))
    else:
        sharpe = 0.0

    trade_pnl = pd.Series([trade["pnl_pct"] for trade in trades], dtype=float)
    if trade_pnl.empty:
        win_rate = 0.0
        payoff_ratio = 0.0
    else:
        wins = trade_pnl[trade_pnl > 0]
        losses = trade_pnl[trade_pnl <= 0]
        win_rate = len(wins) / len(trade_pnl)
        avg_win = wins.mean() if not wins.empty else 0.0
        avg_loss = abs(losses.mean()) if not losses.empty else 0.0
        payoff_ratio = float("inf") if avg_loss == 0 and avg_win > 0 else (
            avg_win / avg_loss if avg_loss > 0 else 0.0
        )

    return {
        "total_return": float(total_return),
        "mdd": float(mdd),
        "sharpe": float(sharpe),
        "win_rate": float(win_rate),
        "payoff_ratio": float(payoff_ratio),
        "trade_count": int(len(trade_pnl)),
    }


def score_metrics(
    metrics: Dict[str, float],
    min_trades: int = 5,
    mdd_weight: float = 0.8,
    sharpe_weight: float = 0.03,
    trade_penalty: float = 0.03,
) -> float:
    too_few_trades = max(0, min_trades - int(metrics["trade_count"]))
    return (
        metrics["total_return"]
        - abs(metrics["mdd"]) * mdd_weight
        + metrics["sharpe"] * sharpe_weight
        - too_few_trades * trade_penalty
    )


def format_pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def format_float(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}"

