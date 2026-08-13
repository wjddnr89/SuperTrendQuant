from __future__ import annotations

from dataclasses import replace

from supertrend_quant.portfolio import AccountSnapshot, OrderPlan


class OneSessionDelayedPreparedBacktest:
    """Delay every non-empty strategy plan by one additional signal session.

    The canonical runner normally executes a plan at the next session open.
    Holding the plan for one extra signal session changes execution to the
    second session open after the original signal. While an order is pending,
    it is neither cancelled nor replaced by a newer signal.
    """

    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.pending_plan: OrderPlan | None = None
        self.pending_signal_ts = None

    def report_frames(self, symbols: set[str]):
        return self.delegate.report_frames(symbols)

    def build_order_plan(
        self,
        signal_ts,
        account: AccountSnapshot,
        mode: str = "backtest",
    ) -> OrderPlan:
        if self.pending_plan is not None:
            plan = self.pending_plan
            original_signal = self.pending_signal_ts
            self.pending_plan = None
            self.pending_signal_ts = None
            return replace(
                plan,
                mode=mode,
                notes=(
                    *plan.notes,
                    f"Execution delayed one session from {original_signal}.",
                ),
            )

        plan = self.delegate.build_order_plan(signal_ts, account, mode=mode)
        if not plan.orders:
            return plan
        self.pending_plan = plan
        self.pending_signal_ts = signal_ts
        return OrderPlan(
            strategy_name=plan.strategy_name,
            mode=mode,
            orders=(),
            notes=(f"Order plan queued for one-session delay from {signal_ts}.",),
        )
