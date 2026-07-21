from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_terminal_identity_boundaries.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_terminal_identity_boundaries",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def _frames() -> dict[str, pd.DataFrame]:
    master: list[dict[str, object]] = []
    history: list[dict[str, object]] = []
    prices: list[dict[str, object]] = []
    factors: list[dict[str, object]] = []
    archive: list[dict[str, object]] = []
    for number, target in enumerate(script.TARGETS, start=1):
        retrieved_at = f"2026-07-16T15:{number:02d}:00Z"
        old_hash = str(number) * 64
        master.append(
            {
                "security_id": target.security_id,
                "primary_symbol": target.symbol,
                "active_to": target.original_active_to,
                "source": "eodhd_exchange_symbols",
                "source_url": "https://eodhd.test/exchange-symbol-list",
                "source_hash": old_hash,
                "retrieved_at": "2026-07-16T14:00:00Z",
            }
        )
        history.append(
            {
                "security_id": target.security_id,
                "symbol": target.symbol,
                "effective_to": "",
                "source": "eodhd_exchange_symbols",
                "source_url": "https://eodhd.test/exchange-symbol-list",
                "source_hash": old_hash,
                "retrieved_at": "2026-07-16T14:00:00Z",
            }
        )
        close = 100.0 + number
        prices.append(
            {
                "security_id": target.security_id,
                "session": target.final_session,
                "open": close - 1.0,
                "high": close + 1.0,
                "low": close - 2.0,
                "close": close,
                "volume": 1_000_000.0,
                "source": "eodhd_eod",
                "source_url": float("nan"),
                "source_hash": target.source_hash,
                "retrieved_at": retrieved_at,
            }
        )
        factors.append(
            {
                "security_id": target.security_id,
                "session": target.final_session,
            }
        )
        for session in target.overrun_sessions:
            prices.append(
                {
                    "security_id": target.security_id,
                    "session": session,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 0.0,
                    "source": "eodhd_eod",
                    "source_url": float("nan"),
                    "source_hash": target.source_hash,
                    "retrieved_at": retrieved_at,
                }
            )
            factors.append(
                {
                    "security_id": target.security_id,
                    "session": session,
                }
            )
        archive.append(
            {
                "archive_id": target.source_hash,
                "dataset": "eodhd_eod",
                "object_path": (
                    f"archives/2026-07-15/{target.source_hash}.json.gz"
                ),
                "content_type": "application/json",
                "source": "eodhd_eod",
                "source_url": target.source_url,
                "source_hash": target.source_hash,
                "retrieved_at": retrieved_at,
            }
        )
    return {
        "security_master": pd.DataFrame(master),
        "symbol_history": pd.DataFrame(history),
        "daily_price_raw": pd.DataFrame(prices),
        "adjustment_factors": pd.DataFrame(factors),
        "source_archive": pd.DataFrame(archive),
    }


def test_exact_rows_boundaries_and_provenance_are_repaired_idempotently():
    frames = _frames()
    prepared = script.prepare_terminal_boundary_repair(frames)

    assert prepared.summary["status"] == "validated_dry_run"
    assert prepared.summary["price_rows_removed"] == 7
    assert prepared.summary["factor_rows_removed"] == 7
    assert prepared.summary["identity_rows_updated"] == 6

    repaired_master = prepared.frames["security_master"]
    repaired_history = prepared.frames["symbol_history"]
    repaired_prices = prepared.frames["daily_price_raw"]
    repaired_factors = prepared.frames["adjustment_factors"]
    for target in script.TARGETS:
        master_row = repaired_master.loc[
            repaired_master["security_id"].eq(target.security_id)
        ].iloc[0]
        history_row = repaired_history.loc[
            repaired_history["security_id"].eq(target.security_id)
        ].iloc[0]
        for row, boundary_column in (
            (master_row, "active_to"),
            (history_row, "effective_to"),
        ):
            assert row[boundary_column] == target.final_session
            assert row["source"] == "eodhd_eod"
            assert row["source_url"] == target.source_url
            assert row["source_hash"] == target.source_hash
        target_price_sessions = set(
            repaired_prices.loc[
                repaired_prices["security_id"].eq(target.security_id), "session"
            ]
        )
        target_factor_sessions = set(
            repaired_factors.loc[
                repaired_factors["security_id"].eq(target.security_id), "session"
            ]
        )
        assert target_price_sessions == {target.final_session}
        assert target_factor_sessions == {target.final_session}

    replay = script.prepare_terminal_boundary_repair(
        {**frames, **prepared.frames}
    )
    assert replay.summary["status"] == "already_repaired"
    assert replay.summary["price_rows_removed"] == 0


@pytest.mark.parametrize("column,value", [("volume", 1.0), ("close", 999.0)])
def test_non_synthetic_overrun_fails_closed(column: str, value: float):
    frames = _frames()
    target = script.TARGETS[0]
    overrun = (
        frames["daily_price_raw"]["security_id"].eq(target.security_id)
        & frames["daily_price_raw"]["session"].isin(target.overrun_sessions)
    )
    frames["daily_price_raw"].loc[overrun, column] = value
    with pytest.raises(ValueError, match="exact archived flat pseudo-bar"):
        script.prepare_terminal_boundary_repair(frames)


def test_missing_exact_archive_binding_fails_closed():
    frames = _frames()
    frames["source_archive"].loc[0, "source_url"] += "&changed=1"
    with pytest.raises(ValueError, match="exact EODHD source_archive URL/hash"):
        script.prepare_terminal_boundary_repair(frames)


def test_factor_inventory_must_match_every_removed_price_session():
    frames = _frames()
    target = script.TARGETS[1]
    bad = (
        frames["adjustment_factors"]["security_id"].eq(target.security_id)
        & frames["adjustment_factors"]["session"].eq(target.overrun_sessions[-1])
    )
    frames["adjustment_factors"] = frames["adjustment_factors"].loc[~bad]
    with pytest.raises(ValueError, match="adjustment-factor overrun inventory"):
        script.prepare_terminal_boundary_repair(frames)


class _ValidReport:
    def raise_for_errors(self) -> None:
        return None


class _FakeRepository:
    def __init__(self, frames: dict[str, pd.DataFrame], *, stale: bool = False):
        self.frames = {name: frame.copy() for name, frame in frames.items()}
        self.release = SimpleNamespace(
            version="release-v1",
            completed_session="2026-07-15",
            dataset_versions={name: f"{name}-v1" for name in frames},
            warnings=(),
        )
        self.etag = "etag-v1"
        self.stale = stale
        self.current_reads = 0
        self.writes: list[str] = []
        self.commits: list[dict[str, object]] = []

    def current_release(self):
        self.current_reads += 1
        if self.stale and self.current_reads >= 2:
            return self.release, "etag-v2"
        return self.release, self.etag

    def read_frame(self, dataset: str, _version: str | None = None):
        return self.frames[dataset].copy()

    def manifest_for_version(self, dataset: str, version: str):
        return SimpleNamespace(dataset=dataset, version=version)

    def write_frame(self, dataset: str, frame: pd.DataFrame, **_kwargs):
        self.writes.append(dataset)
        self.frames[dataset] = frame.copy()
        return SimpleNamespace(
            manifest=SimpleNamespace(version=f"{dataset}-v2")
        )

    def commit_release(
        self,
        completed_session: str,
        dataset_versions: dict[str, str],
        **kwargs,
    ):
        self.commits.append(
            {
                "completed_session": completed_session,
                "dataset_versions": dict(dataset_versions),
                **kwargs,
            }
        )
        assert kwargs["expected_etag"] == "etag-v1"
        self.release = SimpleNamespace(
            version="release-v2",
            completed_session=completed_session,
            dataset_versions=dict(dataset_versions),
            warnings=(),
            quality=kwargs["quality"],
        )
        self.etag = "etag-v2"
        return self.release


def test_apply_validates_then_cas_commits_one_release(monkeypatch):
    repository = _FakeRepository(_frames())
    snapshot_calls: list[object] = []
    monkeypatch.setattr(script, "validate_dataset", lambda *_a, **_k: _ValidReport())
    monkeypatch.setattr(
        script,
        "validate_repository_snapshot",
        lambda repository: snapshot_calls.append(repository) or _ValidReport(),
    )

    summary = script.run_repair(repository, apply=True)

    assert summary["status"] == "applied"
    assert repository.writes == list(script.WRITE_DATASETS)
    assert len(repository.commits) == 1
    assert repository.commits[0]["expected_etag"] == "etag-v1"
    assert len(snapshot_calls) == 2


def test_changed_current_etag_prevents_all_writes(monkeypatch):
    repository = _FakeRepository(_frames(), stale=True)
    monkeypatch.setattr(script, "validate_dataset", lambda *_a, **_k: _ValidReport())
    monkeypatch.setattr(
        script, "validate_repository_snapshot", lambda *_a, **_k: _ValidReport()
    )

    with pytest.raises(RuntimeError, match="Current release changed"):
        script.run_repair(repository, apply=True)
    assert repository.writes == []
    assert repository.commits == []


def test_plan_is_strictly_read_only(monkeypatch):
    repository = _FakeRepository(_frames())
    snapshot_calls: list[object] = []
    monkeypatch.setattr(script, "validate_dataset", lambda *_a, **_k: _ValidReport())
    monkeypatch.setattr(
        script,
        "validate_repository_snapshot",
        lambda repository: snapshot_calls.append(repository) or _ValidReport(),
    )

    summary = script.run_repair(repository, apply=False)

    assert summary["status"] == "validated_dry_run"
    assert summary["network_accessed"] is False
    assert len(snapshot_calls) == 1
    assert repository.writes == []
    assert repository.commits == []
