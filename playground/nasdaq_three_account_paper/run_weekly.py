from __future__ import annotations

import argparse
import json
from pathlib import Path

from bootstrap import LAB_ROOT
from reports import PerformanceReporter


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate weekly performance from daily paper history."
    )
    parser.add_argument(
        "--config",
        default=str(LAB_ROOT / "config.json"),
    )
    args = parser.parse_args()
    raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    results_dir = LAB_ROOT / raw["storage"]["results_dir"]
    reporter = PerformanceReporter(
        results_dir,
        initial_cash=float(raw["experiment"]["initial_cash"]),
    )
    weekly = reporter.generate_weekly()
    if weekly.empty:
        print("No daily history is available yet.")
        return 0
    latest_week = weekly.iloc[-1]["week"]
    print(f"Weekly report complete: {latest_week}")
    print(results_dir / "weekly" / f"{latest_week}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

