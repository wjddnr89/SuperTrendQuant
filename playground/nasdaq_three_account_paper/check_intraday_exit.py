from __future__ import annotations

import argparse
import json
from datetime import date

from bootstrap import LAB_ROOT
from lab import ThreeAccountPaperLab


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Inspect a paper account's intraday exit signal without trading."
    )
    value.add_argument("--account", default="G")
    value.add_argument("--signal-date", required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    lab = ThreeAccountPaperLab(LAB_ROOT / "config.json")
    account_id = str(args.account)
    account_raw = lab.raw["accounts"][account_id]
    held = set(lab.brokers[account_id].get_account().positions)
    signals = lab._intraday_exit_signals(
        account_id=account_id,
        account_raw=account_raw,
        held_symbols=held,
        signal_date=date.fromisoformat(args.signal_date),
    )
    print(
        json.dumps(
            {symbol: signal.as_dict() for symbol, signal in signals.items()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
