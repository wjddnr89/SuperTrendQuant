from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from supertrend_quant.portfolio import AccountSnapshot, OrderPlan
from supertrend_quant.strategies.common import sell_all


DEFAULT_HALLOWEEN_MONTHS = frozenset({11, 12, 1, 2, 3, 4})


def halloween_allows_execution(
    execution_ts,
    *,
    active_months: Iterable[int] = DEFAULT_HALLOWEEN_MONTHS,
) -> bool:
    """Return whether the execution session belongs to the Halloween season."""

    months = frozenset(int(value) for value in active_months)
    if not months or any(value < 1 or value > 12 for value in months):
        raise ValueError("active_months must contain valid calendar months.")
    return pd.Timestamp(execution_ts).month in months


class HalloweenPreparedBacktest:
    """Execution-season overlay around a prepared backtest.

    The canonical runner creates orders on the prior session and fills them at
    the next open.  Therefore the next full-index session, rather than the
    signal session, determines whether exposure is allowed.  This liquidates at
    the first May open and permits entry at the first November open.
    """

    def __init__(
        self,
        delegate,
        full_index: pd.Index,
        *,
        active_months: Iterable[int] = DEFAULT_HALLOWEEN_MONTHS,
    ) -> None:
        self.delegate = delegate
        self.full_index = pd.DatetimeIndex(full_index)
        self.active_months = frozenset(int(value) for value in active_months)
        if (
            not self.active_months
            or any(value < 1 or value > 12 for value in self.active_months)
        ):
            raise ValueError("active_months must contain valid calendar months.")

    def report_frames(self, symbols: set[str]):
        return self.delegate.report_frames(symbols)

    def build_order_plan(
        self,
        signal_ts,
        account: AccountSnapshot,
        mode: str = "backtest",
    ) -> OrderPlan:
        signal = pd.Timestamp(signal_ts)
        position = int(self.full_index.searchsorted(signal, side="right")) - 1
        if position < 0 or position + 1 >= len(self.full_index):
            raise IndexError(
                f"Cannot resolve the next execution session for {signal_ts}."
            )
        execution_ts = self.full_index[position + 1]
        if halloween_allows_execution(
            execution_ts,
            active_months=self.active_months,
        ):
            return self.delegate.build_order_plan(signal_ts, account, mode=mode)

        held = [
            position
            for position in account.positions.values()
            if position.quantity > 0
        ]
        return OrderPlan(
            strategy_name=self.delegate.delegate.strategy.config.strategy.name,
            mode=mode,
            orders=tuple(
                sell_all(position, "Halloween seasonal exit")
                for position in sorted(held, key=lambda item: item.symbol)
            ),
            notes=(
                f"Halloween overlay inactive for execution session "
                f"{execution_ts.date()}.",
            ),
        )
