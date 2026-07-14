from __future__ import annotations

import math

import numpy as np
import pandas as pd


def annualization_factor(interval: str) -> float:
    mapping = {
        "1m": 390 * 252,
        "5m": 78 * 252,
        "15m": 26 * 252,
        "30m": 13 * 252,
        "60m": 6.5 * 252,
        "1h": 6.5 * 252,
        "2h": 3.25 * 252,
        "4h": 1.625 * 252,
        "1d": 252,
    }
    return mapping.get(interval, 252)


def calculate_metrics(equity: pd.Series, trade_returns: list[float], interval: str) -> dict[str, float | int]:
    if equity.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "mdd": 0.0,
            "calmar": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "win_rate": 0.0,
            "payoff_ratio": 0.0,
            "trade_count": 0,
        }

    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0 if equity.iloc[0] else 0.0
    cagr = _calculate_cagr(equity, interval)
    drawdown = equity / equity.cummax() - 1.0
    mdd = float(drawdown.min())
    calmar = 0.0
    if mdd < 0:
        calmar = float(cagr / abs(mdd))
    elif cagr > 0:
        calmar = float("inf")

    sharpe = 0.0
    if len(returns) > 1 and returns.std(ddof=1) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(annualization_factor(interval)))

    sortino = 0.0
    if len(returns) > 1:
        downside = np.minimum(returns.to_numpy(dtype=float), 0.0)
        downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
        if downside_deviation > 0:
            sortino = float(
                returns.mean()
                / downside_deviation
                * math.sqrt(annualization_factor(interval))
            )
        elif returns.mean() > 0:
            sortino = float("inf")

    pnl = pd.Series(trade_returns, dtype=float)
    if pnl.empty:
        win_rate = 0.0
        payoff_ratio = 0.0
    else:
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]
        win_rate = len(wins) / len(pnl)
        avg_win = wins.mean() if not wins.empty else 0.0
        avg_loss = abs(losses.mean()) if not losses.empty else 0.0
        payoff_ratio = float("inf") if avg_loss == 0 and avg_win > 0 else (avg_win / avg_loss if avg_loss > 0 else 0.0)

    return {
        "total_return": float(total_return),
        "cagr": cagr,
        "mdd": mdd,
        "calmar": calmar,
        "sharpe": sharpe,
        "sortino": sortino,
        "win_rate": float(win_rate),
        "payoff_ratio": float(payoff_ratio),
        "trade_count": int(len(trade_returns)),
    }


def _calculate_cagr(equity: pd.Series, interval: str) -> float:
    if len(equity) < 2:
        return 0.0
    start_value = float(equity.iloc[0])
    end_value = float(equity.iloc[-1])
    if start_value <= 0 or end_value < 0:
        return 0.0

    elapsed_years = _elapsed_years(equity.index, len(equity) - 1, interval)
    if elapsed_years <= 0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        annualized = np.power(end_value / start_value, 1.0 / elapsed_years) - 1.0
    if np.isnan(annualized):
        return 0.0
    return float(annualized)


def _elapsed_years(index: pd.Index, return_count: int, interval: str) -> float:
    if isinstance(index, pd.DatetimeIndex) and len(index) >= 2:
        elapsed = index[-1] - index[0]
        seconds = float(elapsed.total_seconds())
        if seconds > 0:
            return seconds / (365.25 * 24 * 60 * 60)
    return float(return_count) / annualization_factor(interval)


def format_pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def format_float(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}"
