from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a saved single-eval trade log against its equity curve.")
    parser.add_argument("run_dir", help="Directory containing trades.csv and equity.csv.")
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    trades = pd.read_csv(run_dir / "trades.csv")
    equity = pd.read_csv(run_dir / "equity.csv")
    initial_equity = float(equity.iloc[0, 1])
    final_equity = float(equity.iloc[-1, 1])
    initial_cash = float(args.initial_cash)
    tolerance = float(args.tolerance)

    formula_errors: list[tuple[int, str, float, float]] = []
    impossible_entries: list[tuple[int, str, float, float]] = []
    overlaps: list[tuple[int, str, str, str]] = []
    replay_rows: list[dict[str, object]] = []

    realized_equity = initial_cash
    previous_exit = None
    for raw in trades.to_dict("records"):
        trade_no = int(raw["trade_no"])
        symbol = str(raw["symbol"])
        entry_value = float(raw["entry_value"])
        exit_value = float(raw["exit_value"])
        pnl_value = float(raw["pnl_value"])
        pnl_pct = float(raw["pnl_pct"])
        expected_pnl_value = exit_value - entry_value
        expected_pnl_pct = exit_value / entry_value - 1.0

        if abs(expected_pnl_value - pnl_value) > max(tolerance, abs(pnl_value) * 1e-9):
            formula_errors.append((trade_no, "pnl_value", expected_pnl_value, pnl_value))
        if abs(expected_pnl_pct - pnl_pct) > 1e-9:
            formula_errors.append((trade_no, "pnl_pct", expected_pnl_pct, pnl_pct))
        if entry_value - realized_equity > max(tolerance, realized_equity * 1e-9):
            impossible_entries.append((trade_no, symbol, entry_value, realized_equity))

        entry_time = pd.Timestamp(raw["entry_time"])
        exit_time = pd.Timestamp(raw["exit_time"])
        if previous_exit is not None and entry_time < previous_exit:
            overlaps.append((trade_no, symbol, str(entry_time), str(previous_exit)))
        previous_exit = exit_time

        before = realized_equity
        realized_equity += pnl_value
        replay_rows.append(
            {
                "trade_no": trade_no,
                "symbol": symbol,
                "equity_before": before,
                "entry_value": entry_value,
                "exit_value": exit_value,
                "pnl_value": pnl_value,
                "equity_after": realized_equity,
            }
        )

    replay_return = realized_equity / initial_cash - 1.0
    equity_return = final_equity / initial_equity - 1.0
    sum_pnl = float(trades["pnl_value"].sum())
    product_trade_returns = float((1.0 + trades["pnl_pct"].astype(float)).prod() - 1.0)

    print("Trade Log Audit")
    print(f"run_dir={run_dir}")
    print(f"trade_count={len(trades)}")
    print(f"initial_equity={initial_equity:.10f}")
    print(f"final_equity={final_equity:.10f}")
    print(f"equity_return={equity_return:.10f} ({equity_return:+.2%})")
    print(f"sum_pnl_value={sum_pnl:.10f}")
    print(f"initial_cash_plus_sum_pnl={initial_cash + sum_pnl:.10f}")
    print(f"replay_final_equity={realized_equity:.10f}")
    print(f"replay_return={replay_return:.10f} ({replay_return:+.2%})")
    print(f"product_of_trade_returns={product_trade_returns:.10f} ({product_trade_returns:+.2%})")
    print(f"formula_errors={len(formula_errors)}")
    print(f"impossible_entries={len(impossible_entries)}")
    print(f"overlaps={len(overlaps)}")
    print()
    print("Last 10 Replay Rows")
    print(pd.DataFrame(replay_rows).tail(10).to_string(index=False))
    if formula_errors:
        print()
        print("Formula Errors")
        print(formula_errors[:10])
    if impossible_entries:
        print()
        print("Impossible Entries")
        print(impossible_entries[:10])
    if overlaps:
        print()
        print("Overlaps")
        print(overlaps[:10])


if __name__ == "__main__":
    main()
