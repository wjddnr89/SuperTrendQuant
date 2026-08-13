from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from bootstrap import LAB_ROOT
from lab import ThreeAccountPaperLab
from reports import PerformanceReporter


REFERENCE_ACCOUNTS = ("A", "B", "C")


class CachedHistoricalClient:
    """Read the existing daily cache and use its raw open as a historical proxy."""

    def __init__(self, source_data_dir: Path) -> None:
        self.source_data_dir = source_data_dir

    def fetch_candles(
        self,
        symbol: str,
        *,
        interval: str,
        minimum_bars: int,
        adjusted: bool,
        before=None,
        max_pages: int = 30,
    ) -> pd.DataFrame:
        if interval != "1d":
            raise RuntimeError(
                f"Historical same-inception backfill only supports daily data: {interval}"
            )
        mode = "adjusted" if adjusted else "raw"
        path = self.source_data_dir / "daily" / mode / f"{symbol}.csv"
        if not path.exists() and not adjusted:
            path = self.source_data_dir / "daily" / "adjusted" / f"{symbol}.csv"
        if not path.exists():
            raise RuntimeError(f"Missing cached {mode} daily data for {symbol}: {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def first_regular_minute(
        self,
        symbol: str,
        session_date: date,
        *,
        timezone: str = "America/New_York",
        regular_open: str = "09:30",
        regular_close: str = "16:00",
        max_pages: int = 30,
    ) -> pd.Series | None:
        path = self.source_data_dir / "daily" / "raw" / f"{symbol}.csv"
        if not path.exists():
            path = self.source_data_dir / "daily" / "adjusted" / f"{symbol}.csv"
        if not path.exists():
            raise RuntimeError(f"Missing cached raw daily data for {symbol}: {path}")
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        matches = frame.loc[
            [timestamp.date() == session_date for timestamp in frame.index]
        ]
        if matches.empty:
            return None
        row = matches.iloc[-1].copy()
        row.name = pd.Timestamp(f"{session_date} {regular_open}", tz=timezone)
        return row


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Align newly added paper accounts to the A/B/C inception date."
    )
    value.add_argument("--config", default=str(LAB_ROOT / "config.json"))
    value.add_argument("--apply", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    config_path = Path(args.config).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    state_dir = (LAB_ROOT / raw["storage"]["state_dir"]).resolve()
    data_dir = (LAB_ROOT / raw["storage"]["data_dir"]).resolve()
    results_dir = (LAB_ROOT / raw["storage"]["results_dir"]).resolve()
    daily_path = results_dir / "daily_history.csv"
    daily = pd.read_csv(daily_path)
    reference = daily.loc[daily["account_id"].isin(REFERENCE_ACCOUNTS)].copy()
    if reference.empty:
        raise RuntimeError("A/B/C daily history is empty.")
    coverage = reference.groupby("account_id")["execution_date"].agg(["min", "max"])
    if set(coverage.index) != set(REFERENCE_ACCOUNTS):
        raise RuntimeError("A/B/C do not all have daily history.")
    start = date.fromisoformat(str(coverage["min"].max()))
    end = date.fromisoformat(str(coverage["max"].min()))
    existing_accounts = set(daily["account_id"].astype(str))
    new_accounts = [
        account_id
        for account_id in raw["accounts"]
        if account_id not in existing_accounts
    ]
    if not new_accounts:
        print("No newly added accounts require same-inception backfill.")
        return 0
    print(
        f"New accounts: {', '.join(new_accounts)}; "
        f"reference window: {start} through {end}",
        flush=True,
    )

    sessions = sorted(
        {
            date.fromisoformat(value)
            for value in reference["execution_date"].astype(str)
            if start <= date.fromisoformat(value) <= end
        }
    )
    with tempfile.TemporaryDirectory(prefix="nasdaq-new-account-backfill-") as tmp:
        tmp_root = Path(tmp)
        temp_config = json.loads(json.dumps(raw))
        temp_config["accounts"] = {
            account_id: raw["accounts"][account_id] for account_id in new_accounts
        }
        temp_config["storage"] = {
            "state_dir": str(tmp_root / "state"),
            "data_dir": str(tmp_root / "data"),
            "results_dir": str(tmp_root / "results"),
        }
        temp_config_path = tmp_root / "config.json"
        temp_config_path.write_text(
            json.dumps(temp_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        lab = ThreeAccountPaperLab(
            temp_config_path,
            client=CachedHistoricalClient(data_dir),
        )
        for index, session in enumerate(sessions, start=1):
            print(f"[backfill] {index}/{len(sessions)} execution={session}", flush=True)
            lab.run_daily(session)

        generated = pd.read_csv(tmp_root / "results" / "daily_history.csv")
        expected = {(session.isoformat(), account) for session in sessions for account in new_accounts}
        actual = set(
            zip(
                generated["execution_date"].astype(str),
                generated["account_id"].astype(str),
            )
        )
        missing = sorted(expected - actual)
        if missing:
            raise RuntimeError(f"Generated backfill is incomplete: {missing}")
        print(generated[["execution_date", "account_id", "equity", "position_symbol"]].to_string(index=False))
        if not args.apply:
            print("Dry run complete. Re-run with --apply to merge state and reports.")
            return 0

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = LAB_ROOT / "archive" / f"new_accounts_before_same_inception_{stamp}"
        archive_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, archive_dir / "config.json")
        for account_id in new_accounts:
            current_state = state_dir / "accounts" / f"{account_id}.json"
            if current_state.exists():
                shutil.copy2(current_state, archive_dir / current_state.name)
            source_state = tmp_root / "state" / "accounts" / f"{account_id}.json"
            state = json.loads(source_state.read_text(encoding="utf-8"))
            old_state = (
                json.loads(current_state.read_text(encoding="utf-8"))
                if current_state.exists()
                else {}
            )
            old_metadata = old_state.get("metadata", {})
            metadata = state.setdefault("metadata", {})
            metadata.update(
                {
                    "strategy_generation": old_metadata.get(
                        "strategy_generation", f"{account_id}_same_inception"
                    ),
                    "inception_execution_date": start.isoformat(),
                    "history_basis": (
                        "counterfactual_same_inception_historical_daily_open_proxy"
                    ),
                    "strategy_policy": dict(raw["accounts"][account_id]),
                    "counterfactual_reconstruction_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }
            )
            metadata.pop("cold_start_after_execution_date", None)
            current_state.parent.mkdir(parents=True, exist_ok=True)
            current_state.write_text(
                json.dumps(state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        note = (
            "Counterfactual same-inception reconstruction; historical daily "
            "open proxy used for execution."
        )
        for row_index in generated.index:
            notes = json.loads(generated.at[row_index, "notes"] or "[]")
            if note not in notes:
                notes.append(note)
            generated.at[row_index, "notes"] = json.dumps(notes, ensure_ascii=False)
        reporter = PerformanceReporter(results_dir, raw["experiment"]["initial_cash"])
        for session in sessions:
            rows = generated.loc[
                generated["execution_date"].astype(str) == session.isoformat()
            ].to_dict(orient="records")
            reporter.record_daily(rows)
        print(f"Applied. Archive: {archive_dir}")
        print(f"Dashboard: {results_dir / 'dashboard.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
