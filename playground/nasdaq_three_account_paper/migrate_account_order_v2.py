from __future__ import annotations

import json
import os
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bootstrap import LAB_ROOT
from reports import PerformanceReporter, _atomic_csv, _write_json


OLD_TO_NEW = {
    "A": "A",
    "B": "B",
    "E": "C",
    "C": "D",
    "F": "E",
    "G": "F",
    "D": "G",
}

NEW_ACCOUNTS = {
    "A": {
        "name": "MF ON - Base",
        "hypothesis": "시장필터 ON 기본형",
        "market_filter": "1d",
        "rotation_profit_gate": "nonnegative",
        "stop_loss_pct": None,
        "late_chase_mode": "unlimited",
        "max_extension_atr": None,
    },
    "B": {
        "name": "MF ON - Stop12 GateOff",
        "hypothesis": "시장필터 ON, 12% 손절 및 회전수익 gate 해제",
        "market_filter": "1d",
        "rotation_profit_gate": "off",
        "stop_loss_pct": 0.12,
        "late_chase_mode": "unlimited",
        "max_extension_atr": None,
    },
    "C": {
        "name": "MF ON - ATR2.5",
        "hypothesis": "시장필터 ON 기본형에 ATR 위험예산 2.5% 적용",
        "market_filter": "1d",
        "rotation_profit_gate": "nonnegative",
        "stop_loss_pct": None,
        "late_chase_mode": "unlimited",
        "max_extension_atr": None,
        "entry_atr_risk_pct": 0.025,
    },
    "D": {
        "name": "MF OFF - Base",
        "hypothesis": "시장필터 OFF 기본형",
        "market_filter": "none",
        "rotation_profit_gate": "nonnegative",
        "stop_loss_pct": None,
        "late_chase_mode": "unlimited",
        "max_extension_atr": None,
    },
    "E": {
        "name": "MF OFF - ATR2.5",
        "hypothesis": "시장필터 OFF 기본형에 ATR 위험예산 2.5% 적용",
        "market_filter": "none",
        "rotation_profit_gate": "nonnegative",
        "stop_loss_pct": None,
        "late_chase_mode": "unlimited",
        "max_extension_atr": None,
        "entry_atr_risk_pct": 0.025,
    },
    "F": {
        "name": "MF OFF - ATR2.0",
        "hypothesis": "시장필터 OFF 기본형에 ATR 위험예산 2.0% 적용",
        "market_filter": "none",
        "rotation_profit_gate": "nonnegative",
        "stop_loss_pct": None,
        "late_chase_mode": "unlimited",
        "max_extension_atr": None,
        "entry_atr_risk_pct": 0.02,
    },
    "G": {
        "name": "MF OFF - 2h Exit",
        "hypothesis": "시장필터 OFF 기본형에 2시간봉 장중 청산 적용",
        "market_filter": "none",
        "rotation_profit_gate": "nonnegative",
        "stop_loss_pct": None,
        "late_chase_mode": "unlimited",
        "max_extension_atr": None,
        "exit_timeframe": "2h",
        "exit_supertrend_period": 10,
        "exit_supertrend_multiplier": 3.0,
        "exit_trigger_timeframe": "1m",
        "exit_confirm_minutes": 10,
        "reentry_release": "completed_2h_bullish_after_exit",
        "entry_2h_safety_gate": True,
    },
}

GENERATIONS = {
    "A": "A_mf_on_base",
    "B": "B_mf_on_stop12_gateoff",
    "C": "C_mf_on_atr25",
    "D": "D_mf_off_base",
    "E": "E_mf_off_atr25",
    "F": "F_mf_off_atr20",
    "G": "G_mf_off_2h_exit",
}


def main() -> int:
    marker = LAB_ROOT / "state" / "account_order_v2.json"
    if marker.exists():
        raise RuntimeError(f"Account-order migration already completed: {marker}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = LAB_ROOT / "archive" / f"account_order_before_v2_{stamp}"
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(LAB_ROOT / "config.json", archive / "config.json")
    shutil.copytree(LAB_ROOT / "state", archive / "state")
    shutil.copytree(LAB_ROOT / "results", archive / "results")

    _migrate_states()
    _migrate_events()
    _migrate_reports()

    _atomic_json(
        marker,
        {
            "migrated_at": datetime.now(timezone.utc).isoformat(),
            "mapping_old_to_new": OLD_TO_NEW,
            "archive": str(archive),
        },
    )
    print(f"Account order migrated. Backup: {archive}")
    return 0


def _migrate_states() -> None:
    root = LAB_ROOT / "state" / "accounts"
    original = {
        account: json.loads((root / f"{account}.json").read_text(encoding="utf-8"))
        for account in OLD_TO_NEW
    }
    for old_id, new_id in OLD_TO_NEW.items():
        state = deepcopy(original[old_id])
        metadata = state.setdefault("metadata", {})
        metadata["strategy_account"] = new_id
        metadata["strategy_generation"] = GENERATIONS[new_id]
        metadata["strategy_policy"] = deepcopy(NEW_ACCOUNTS[new_id])
        if "clone_source_account" in metadata:
            metadata["clone_source_account"] = OLD_TO_NEW.get(
                str(metadata["clone_source_account"]),
                metadata["clone_source_account"],
            )
        state = _rewrite_legacy_intraday_labels(state, old_id)
        _atomic_json(root / f"{new_id}.json", state)


def _migrate_events() -> None:
    root = LAB_ROOT / "results" / "events"
    original = {
        account: (root / f"account_{account}.jsonl").read_text(encoding="utf-8")
        if (root / f"account_{account}.jsonl").exists()
        else ""
        for account in OLD_TO_NEW
    }
    for old_id, new_id in OLD_TO_NEW.items():
        rows = []
        for line in original[old_id].splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if "account_id" in payload:
                payload["account_id"] = new_id
            if "clone_source_account" in payload:
                payload["clone_source_account"] = OLD_TO_NEW.get(
                    str(payload["clone_source_account"]),
                    payload["clone_source_account"],
                )
            payload = _rewrite_legacy_intraday_labels(payload, old_id)
            rows.append(json.dumps(payload, ensure_ascii=False))
        _atomic_text(
            root / f"account_{new_id}.jsonl",
            "\n".join(rows) + ("\n" if rows else ""),
        )


def _migrate_reports() -> None:
    root = LAB_ROOT / "results"
    daily_path = root / "daily_history.csv"
    daily = pd.read_csv(daily_path)
    daily["account_id"] = daily["account_id"].astype(str).map(OLD_TO_NEW)
    if daily["account_id"].isna().any():
        raise RuntimeError("Daily history contains an unknown account id.")
    daily["account_name"] = daily["account_id"].map(
        {key: value["name"] for key, value in NEW_ACCOUNTS.items()}
    )
    daily["hypothesis"] = daily["account_id"].map(
        {key: value["hypothesis"] for key, value in NEW_ACCOUNTS.items()}
    )
    daily["notes"] = [
        _rewrite_text(str(value)) if account == "G" else value
        for account, value in zip(daily["account_id"], daily["notes"])
    ]
    daily = daily.sort_values(["execution_date", "account_id"])
    _atomic_csv(daily_path, daily)
    for execution_date, rows in daily.groupby("execution_date", sort=True):
        _atomic_csv(root / "daily" / f"{execution_date}.csv", rows)

    latest_date = str(daily["execution_date"].max())
    latest = daily.loc[daily["execution_date"].astype(str) == latest_date]
    _write_json(
        root / "latest_daily.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "execution_date": latest_date,
            "accounts": latest.to_dict(orient="records"),
        },
    )
    config = json.loads((LAB_ROOT / "config.json").read_text(encoding="utf-8"))
    reporter = PerformanceReporter(
        root,
        initial_cash=float(config["experiment"]["initial_cash"]),
    )
    reporter.generate_weekly(daily)
    reporter._write_dashboard(daily)


def _rewrite_legacy_intraday_labels(value: Any, old_id: str) -> Any:
    if old_id != "D":
        return value
    if isinstance(value, dict):
        return {
            key: _rewrite_legacy_intraday_labels(item, old_id)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_legacy_intraday_labels(item, old_id) for item in value]
    if isinstance(value, str):
        return _rewrite_text(value)
    return value


def _rewrite_text(value: str) -> str:
    return (
        value.replace("D baseline cloned from C", "G baseline cloned from D")
        .replace("D 2h replay", "G 2h replay")
        .replace("D offline intraday", "G offline intraday")
        .replace("Pending D intraday", "Pending G intraday")
        .replace("cloned_from_C", "cloned_from_D")
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
