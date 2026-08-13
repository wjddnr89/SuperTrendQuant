from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from bootstrap import LAB_ROOT
from lab import ThreeAccountPaperLab
from toss_data import TossMarketDataClient


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Update the seven Nasdaq-100 paper accounts once."
    )
    value.add_argument(
        "--config",
        default=str(LAB_ROOT / "config.json"),
        help="Experiment JSON path.",
    )
    value.add_argument(
        "--execution-date",
        default="",
        help="US execution session in YYYY-MM-DD. Defaults to today in New York.",
    )
    value.add_argument(
        "--check-credentials",
        action="store_true",
        help="Only verify that Toss credential environment variables exist.",
    )
    return value


def main() -> int:
    args = parser().parse_args()
    client = TossMarketDataClient()
    if args.check_credentials:
        if not client.credentials_available():
            print(
                "MISSING: set TOSS_CLIENT_ID and TOSS_CLIENT_SECRET in the "
                "repository-root .env file."
            )
            return 2
        print("OK: Toss client credentials are present.")
        return 0

    execution_date = (
        date.fromisoformat(args.execution_date)
        if args.execution_date
        else None
    )
    with lab_lock(LAB_ROOT / "state" / "daily.lock"):
        lab = ThreeAccountPaperLab(args.config, client=client)
        history = lab.run_daily(execution_date)
    if history.empty:
        print("No daily performance rows.")
        return 0
    latest = history["execution_date"].max()
    print(f"Daily update complete: {latest}")
    print(f"Dashboard: {LAB_ROOT / 'results' / 'dashboard.html'}")
    return 0


@contextmanager
def lab_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another daily run may still be active: {path}. "
            "Remove the lock only after confirming no process is running."
        ) from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
