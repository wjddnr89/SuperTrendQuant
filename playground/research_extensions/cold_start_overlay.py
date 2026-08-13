from __future__ import annotations

from dataclasses import dataclass

from supertrend_quant.portfolio import AccountSnapshot, OrderPlan


@dataclass(frozen=True)
class ColdStartRule:
    mode: str
    label: str

    def __post_init__(self) -> None:
        if self.mode not in {
            "skip_first_leader",
            "fresh_only",
            "kijun_atr_1p5",
            "kijun_atr_2p0",
        }:
            raise ValueError(f"Unsupported cold-start mode: {self.mode}")


class ColdStartPreparedBacktest:
    """Apply an entry guard only until the first actual entry is submitted.

    `normal_delegate` is always used after the cold-start phase. For fresh or
    extension rules, `guarded_delegate` ranks entries with the corresponding
    research policy while the account is still flat.

    `skip_first_leader` suppresses the first selected leader for as long as it
    remains the selected leader. The first different leader is allowed. This
    avoids reducing the rule to a one-session execution delay.
    """

    def __init__(
        self,
        normal_delegate,
        rule: ColdStartRule,
        *,
        guarded_delegate=None,
    ) -> None:
        self.normal_delegate = normal_delegate
        self.guarded_delegate = guarded_delegate
        self.rule = rule
        self.released = False
        self.skipped_symbol: str | None = None

        if (
            rule.mode != "skip_first_leader"
            and guarded_delegate is None
        ):
            raise ValueError(
                "A guarded delegate is required for fresh/extension modes."
            )

    def report_frames(self, symbols: set[str]):
        return self.normal_delegate.report_frames(symbols)

    def build_order_plan(
        self,
        signal_ts,
        account: AccountSnapshot,
        mode: str = "backtest",
    ) -> OrderPlan:
        if account.positions:
            self.released = True
        if self.released:
            return self.normal_delegate.build_order_plan(
                signal_ts,
                account,
                mode=mode,
            )

        if self.rule.mode == "skip_first_leader":
            plan = self.normal_delegate.build_order_plan(
                signal_ts,
                account,
                mode=mode,
            )
            buy_symbols = [
                order.symbol
                for order in plan.orders
                if str(order.side).lower() == "buy"
            ]
            if not buy_symbols:
                return plan
            selected = str(buy_symbols[0])
            if self.skipped_symbol is None:
                self.skipped_symbol = selected
            if selected == self.skipped_symbol:
                return OrderPlan(
                    strategy_name=plan.strategy_name,
                    mode=mode,
                    orders=(),
                    notes=(
                        *plan.notes,
                        f"Cold start skipped initial leader {selected}.",
                    ),
                )
            self.released = True
            return plan

        plan = self.guarded_delegate.build_order_plan(
            signal_ts,
            account,
            mode=mode,
        )
        if any(str(order.side).lower() == "buy" for order in plan.orders):
            self.released = True
        return plan
