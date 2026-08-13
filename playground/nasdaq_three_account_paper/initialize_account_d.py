from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

from bootstrap import LAB_ROOT
from reports import PerformanceReporter


SOURCE_ACCOUNT = "D"
TARGET_ACCOUNT = "G"
CLONE_NOTE = (
    "G baseline cloned from D; 2h exit divergence begins after the "
    "2026-07-31 close."
)


def main() -> int:
    config = json.loads((LAB_ROOT / "config.json").read_text(encoding="utf-8"))
    source_path = LAB_ROOT / "state" / "accounts" / f"{SOURCE_ACCOUNT}.json"
    target_path = LAB_ROOT / "state" / "accounts" / f"{TARGET_ACCOUNT}.json"
    source_events = LAB_ROOT / "results" / "events" / f"account_{SOURCE_ACCOUNT}.jsonl"
    target_events = LAB_ROOT / "results" / "events" / f"account_{TARGET_ACCOUNT}.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source account state: {source_path}")
    if target_path.exists() or target_events.exists():
        raise FileExistsError(
            "G already exists; refusing to overwrite its independent history."
        )

    source = json.loads(source_path.read_text(encoding="utf-8"))
    target = deepcopy(source)
    target_policy = deepcopy(config["accounts"][TARGET_ACCOUNT])
    effective_date = str(source.get("metadata", {}).get("last_execution_date", ""))
    target_metadata = target.setdefault("metadata", {})
    target_metadata.update(
        {
            "strategy_account": TARGET_ACCOUNT,
            "strategy_generation": (
                "G_v2_mf_off_2h_exit"
            ),
            "strategy_policy": target_policy,
            "exit_timeframe": "2h",
            "intraday_exit_signals": {},
            "clone_source_account": SOURCE_ACCOUNT,
            "clone_effective_date": effective_date,
            "history_basis": f"cloned_from_D_at_{effective_date}",
        }
    )
    _atomic_json(target_path, target)

    copied_events: list[str] = []
    if source_events.exists():
        for line in source_events.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            event["account_id"] = TARGET_ACCOUNT
            event["clone_source_account"] = SOURCE_ACCOUNT
            event["history_basis"] = f"cloned_from_D_at_{effective_date}"
            notes = list(event.get("notes", []))
            if CLONE_NOTE not in notes:
                notes.append(CLONE_NOTE)
            event["notes"] = notes
            copied_events.append(json.dumps(event, ensure_ascii=False))
    target_events.parent.mkdir(parents=True, exist_ok=True)
    target_events.write_text(
        "\n".join(copied_events) + ("\n" if copied_events else ""),
        encoding="utf-8",
    )

    reporter = PerformanceReporter(
        LAB_ROOT / config["storage"]["results_dir"],
        initial_cash=float(config["experiment"]["initial_cash"]),
    )
    daily = reporter.load_daily()
    source_rows = daily.loc[daily["account_id"] == SOURCE_ACCOUNT].copy()
    if source_rows.empty:
        raise RuntimeError("D has no daily history to clone into G.")
    if bool((daily["account_id"] == TARGET_ACCOUNT).any()):
        raise RuntimeError("G daily history already exists.")
    source_rows["account_id"] = TARGET_ACCOUNT
    source_rows["account_name"] = str(target_policy["name"])
    source_rows["hypothesis"] = str(target_policy["hypothesis"])
    source_rows["notes"] = source_rows["notes"].map(_append_clone_note)
    for _, row in source_rows.sort_values("execution_date").iterrows():
        reporter.record_daily([row.to_dict()])

    print(
        f"Initialized G from D at {effective_date}: "
        f"cash=${float(target['cash']):,.2f}, positions={target['positions']}"
    )
    return 0


def _append_clone_note(value: object) -> str:
    try:
        notes = list(json.loads(str(value)))
    except (TypeError, ValueError, json.JSONDecodeError):
        notes = []
    if CLONE_NOTE not in notes:
        notes.append(CLONE_NOTE)
    return json.dumps(notes, ensure_ascii=False)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
