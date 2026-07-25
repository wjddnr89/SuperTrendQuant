from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from .manifest import write_atomic


@dataclass(frozen=True)
class PreflightResult:
    expected_session: str
    completed_session: str
    ready: bool
    sync_attempted: bool = False
    warning: str = ""


def expected_completed_us_session(
    now: datetime | None = None,
    *,
    publication_delay: timedelta = timedelta(minutes=90),
) -> str:
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    now_utc = now.astimezone(UTC)
    try:
        import exchange_calendars as xcals
    except ModuleNotFoundError as exc:
        raise RuntimeError("exchange-calendars is required for daily-session preflight.") from exc
    calendar = xcals.get_calendar("XNYS")
    today = pd.Timestamp(now_utc.date())
    sessions = calendar.sessions_in_range(today - pd.Timedelta(days=14), today)
    if len(sessions) == 0:
        raise RuntimeError("Could not resolve a recent XNYS session.")
    last = sessions[-1]
    close = calendar.session_close(last).to_pydatetime()
    if now_utc >= close + publication_delay:
        return last.date().isoformat()
    if len(sessions) < 2:
        raise RuntimeError("Could not resolve the preceding XNYS session.")
    return sessions[-2].date().isoformat()


class DailyPreflight:
    def __init__(self, state_path: str | Path):
        self.state_path = Path(state_path)

    def run(
        self,
        completed_session: str,
        *,
        auto_sync: bool,
        sync: Callable[[str], str] | None = None,
        force: bool = False,
        now: datetime | None = None,
    ) -> PreflightResult:
        expected = expected_completed_us_session(now)
        if completed_session and completed_session >= expected:
            state = self._read_state()
            if state.get("last_validated_session") != completed_session:
                state["last_validated_session"] = completed_session
                self._write_state(state)
            return PreflightResult(expected, completed_session, True)
        state = self._read_state()
        already_attempted = (
            state.get("last_attempted_session") == expected
            or state.get("last_auto_attempt_session") == expected
        )
        should_attempt = sync is not None and (force or (auto_sync and not already_attempted))
        if not should_attempt:
            reason = (
                f"Market data is stale: expected {expected}, current {completed_session or 'missing'}."
            )
            if already_attempted and not force:
                reason += " Automatic sync was already attempted for this session."
            return PreflightResult(expected, completed_session, False, warning=reason)

        attempted_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        state["last_attempted_session"] = expected
        state["last_attempted_at"] = attempted_at
        # Preserve the V1 preview keys for state-file compatibility.
        state["last_auto_attempt_session"] = expected
        state["last_auto_attempt_at"] = attempted_at
        self._write_state(state)
        try:
            updated = sync(expected)
        except Exception as exc:
            return PreflightResult(
                expected,
                completed_session,
                False,
                sync_attempted=True,
                warning=f"Data sync failed: {exc}",
            )
        ready = bool(updated and updated >= expected)
        warning = "" if ready else f"Sync completed but data is still stale: {updated or 'missing'}"
        if ready:
            state = self._read_state()
            state["last_validated_session"] = updated
            self._write_state(state)
        return PreflightResult(expected, updated, ready, sync_attempted=True, warning=warning)

    def _read_state(self) -> dict[str, str]:
        if not self.state_path.is_file():
            return {}
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        return {str(key): str(value) for key, value in raw.items()}

    def _write_state(self, state: dict[str, str]) -> None:
        write_atomic(
            self.state_path,
            (json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
        )
