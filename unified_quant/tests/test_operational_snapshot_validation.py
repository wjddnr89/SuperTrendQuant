from __future__ import annotations

import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import exchange_calendars as xcals
import pandas as pd
import pytest

from supertrend_quant.cli import data_main
from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.index_ingest import IndexDataImporter
from supertrend_quant.market_store.ingest import (
    DailyDataSynchronizer,
    YahooFetchResult,
)
from supertrend_quant.market_store.operational_validation import (
    TRUSTED_OPERATIONAL_NBL_TERMINAL_STATE,
    reviewed_operational_index_identity_gap_fingerprints,
    validate_operational_repository_snapshot,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.validation import validate_repository_snapshot


EXPECTED_FINGERPRINT = (
    "989c5d44ef1b8cf8a682d807b63a62ebe3c3f38eb6f57e6314b3fe381d5c7d04"
)
FIXTURE_HASH = "a" * 64
FIXTURE_RETRIEVED_AT = "2020-10-13T23:00:00Z"


class _FrameRepository:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames

    def current_manifest(self, dataset: str):
        return SimpleNamespace(version=dataset) if dataset in self.frames else None

    def read_frame(self, dataset: str, _version: str | None = None):
        return self.frames.get(dataset, pd.DataFrame()).copy(deep=True)


def _source(
    row: dict,
    *,
    source: str = "fixture",
    retrieved_at: str = FIXTURE_RETRIEVED_AT,
    source_hash: str = FIXTURE_HASH,
) -> dict:
    return {
        **row,
        "source": source,
        "retrieved_at": retrieved_at,
        "source_hash": source_hash,
    }


def _exact_frames() -> dict[str, pd.DataFrame]:
    expected = TRUSTED_OPERATIONAL_NBL_TERMINAL_STATE
    master = pd.DataFrame(
        [
            _source(
                {
                    "security_id": expected["security_id"],
                    "primary_symbol": "NBL",
                    "name": "Noble Energy Inc",
                    "exchange": expected["identity_exchange"],
                    "asset_type": "STOCK",
                    "currency": "USD",
                    "country": "US",
                    "active_from": expected["master_active_from"],
                    "active_to": expected["last_real_session"],
                },
                source=expected["identity_source"],
                retrieved_at=expected["repair_reviewed_at"],
                source_hash=expected["official_source_hash"],
            ),
            _source(
                {
                    "security_id": expected["successor_security_id"],
                    "primary_symbol": "CVX",
                    "name": "Chevron",
                    "exchange": "NYSE",
                    "asset_type": "STOCK",
                    "currency": "USD",
                    "country": "US",
                    "active_from": "2015-01-01",
                    "active_to": "",
                }
            ),
            _source(
                {
                    "security_id": "OTHER",
                    "primary_symbol": "OTHER",
                    "name": "Other",
                    "exchange": "NYSE",
                    "asset_type": "STOCK",
                    "currency": "USD",
                    "country": "US",
                    "active_from": "2015-01-01",
                    "active_to": "",
                }
            ),
        ]
    )
    master.loc[
        master["security_id"].eq(expected["security_id"]), "source_url"
    ] = expected["official_source_url"]

    history = pd.DataFrame(
        [
            _source(
                {
                    "security_id": expected["security_id"],
                    "symbol": "NBL",
                    "exchange": expected["identity_exchange"],
                    "effective_from": expected["history_effective_from"],
                    "effective_to": expected["last_real_session"],
                    "source_url": expected["official_source_url"],
                },
                source=expected["identity_source"],
                retrieved_at=expected["repair_reviewed_at"],
                source_hash=expected["official_source_hash"],
            ),
            _source(
                {
                    "security_id": expected["successor_security_id"],
                    "symbol": "CVX",
                    "exchange": "NYSE",
                    "effective_from": "2015-01-01",
                    "effective_to": "",
                    "source_url": "memory://cvx",
                }
            ),
            _source(
                {
                    "security_id": "OTHER",
                    "symbol": "OTHER",
                    "exchange": "NYSE",
                    "effective_from": "2015-01-01",
                    "effective_to": "",
                    "source_url": "memory://other",
                }
            ),
        ]
    )
    action = pd.DataFrame(
        [
            _source(
                {
                    "event_id": expected["event_id"],
                    "security_id": expected["security_id"],
                    "action_type": "stock_merger",
                    "effective_date": expected["market_transition_session"],
                    "ex_date": expected["market_transition_session"],
                    "announcement_date": expected[
                        "market_transition_session"
                    ],
                    "record_date": "",
                    "payment_date": "",
                    "cash_amount": None,
                    "ratio": float(expected["ratio"]),
                    "currency": "USD",
                    "new_security_id": expected["successor_security_id"],
                    "new_symbol": expected["successor_symbol"],
                    "official": True,
                    "source_url": expected["official_source_url"],
                    "source_kind": expected["action_source_kind"],
                    "metadata": "",
                },
                source=expected["action_source"],
                retrieved_at=expected["official_retrieved_at"],
                source_hash=expected["official_source_hash"],
            )
        ]
    )
    resolution = pd.DataFrame(
        [
            _source(
                {
                    "candidate_id": expected["candidate_id"],
                    "security_id": expected["security_id"],
                    "symbol": "NBL",
                    "last_price_date": expected["last_real_session"],
                    "resolution": "applied",
                    "event_id": expected["event_id"],
                    "exception_code": "",
                    "exception_reason": "",
                    "reviewed_by": expected["resolution_reviewer"],
                    "reviewed_at": expected["repair_reviewed_at"],
                    "recheck_after": "",
                    "successor_security_id": expected[
                        "successor_security_id"
                    ],
                    "successor_symbol": expected["successor_symbol"],
                    "source_url": expected["official_source_url"],
                },
                source=expected["resolution_source"],
                retrieved_at=expected["repair_reviewed_at"],
                source_hash=expected["official_source_hash"],
            )
        ]
    )
    nbl_price_lineage = {
        "source": "eodhd_eod",
        "retrieved_at": expected["raw_retrieved_at"],
        "source_hash": expected["raw_source_hash"],
    }
    nbl_sessions = xcals.get_calendar("XNYS").sessions_in_range(
        expected["first_price_session"], expected["last_real_session"]
    )
    nbl_prices = [
        {
            "security_id": expected["security_id"],
            "session": pd.Timestamp(session).date().isoformat(),
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.0,
            "volume": 100.0,
            "currency": "USD",
            **nbl_price_lineage,
        }
        for session in nbl_sessions
    ]
    nbl_prices[-1].update(
        {
            "open": 8.14,
            "high": 8.51,
            "low": 8.12,
            "close": 8.46,
            "volume": 13_126_428.0,
        }
    )
    prices = pd.DataFrame(
        [
            *nbl_prices,
            _source(
                {
                    "security_id": expected["successor_security_id"],
                    "session": "2020-10-05",
                    "open": 71.52,
                    "high": 72.73,
                    "low": 70.71,
                    "close": 72.70,
                    "volume": 12_049_800.0,
                    "currency": "USD",
                }
            ),
            _source(
                {
                    "security_id": "OTHER",
                    "session": expected["next_remove_effective_date"],
                    "open": 20.0,
                    "high": 21.0,
                    "low": 19.0,
                    "close": 20.0,
                    "volume": 200.0,
                    "currency": "USD",
                }
            ),
        ]
    )
    anchors = pd.DataFrame(
        [
            _source(
                {
                    "index_id": "sp500",
                    "anchor_date": expected["first_price_session"],
                    "security_id": expected["security_id"],
                    "official": False,
                    "source_url": "memory://anchor",
                    "source_kind": "community",
                }
            )
        ]
    )
    events = pd.DataFrame(
        [
            _source(
                {
                    "event_id": "add-other-at-replay",
                    "index_id": "sp500",
                    "announcement_date": "",
                    "effective_date": expected["replay_date"],
                    "operation": "ADD",
                    "security_id": "OTHER",
                    "official": False,
                    "source_url": "memory://events",
                    "source_kind": "community",
                }
            ),
            _source(
                {
                    "event_id": expected["next_remove_event_id"],
                    "index_id": expected["index_id"],
                    "announcement_date": "",
                    "effective_date": expected[
                        "next_remove_effective_date"
                    ],
                    "operation": "REMOVE",
                    "security_id": expected["security_id"],
                    "official": False,
                    "source_url": expected["next_remove_source_url"],
                    "source_kind": "community",
                },
                source=expected["next_remove_source"],
                source_hash=expected["next_remove_source_hash"],
            ),
        ]
    )
    frames = {
        "security_master": master,
        "symbol_history": history,
        "corporate_actions": action,
        "lifecycle_resolutions": resolution,
        "daily_price_raw": prices,
        "index_constituent_anchors": anchors,
        "index_membership_events": events,
    }
    frames["adjustment_factors"] = build_adjustment_factors(
        prices,
        action,
        source_version="operational-fixture",
    )
    return frames


def _materialize_repository(root: Path) -> LocalDatasetRepository:
    repository = LocalDatasetRepository(root)
    frames = _exact_frames()
    for dataset in (
        "security_master",
        "symbol_history",
        "corporate_actions",
        "lifecycle_resolutions",
        "daily_price_raw",
        "adjustment_factors",
        "index_constituent_anchors",
        "index_membership_events",
    ):
        repository.write_frame(
            dataset,
            frames[dataset],
            completed_session="2020-10-12",
            incomplete_action_policy="block",
        )
    versions = {
        dataset: repository.current_manifest(dataset).version
        for dataset in frames
    }
    repository.commit_release("2020-10-12", versions, quality="valid")
    return repository


def _mutate(frames: dict[str, pd.DataFrame], case: str) -> None:
    expected = TRUSTED_OPERATIONAL_NBL_TERMINAL_STATE
    if case == "action":
        frames["corporate_actions"].loc[0, "ratio"] = 0.12
    elif case == "resolution":
        frames["lifecycle_resolutions"].loc[0, "last_price_date"] = "2020-10-01"
    elif case == "price_tail":
        tail = frames["daily_price_raw"].loc[
            frames["daily_price_raw"]["security_id"].eq(expected["security_id"])
        ].tail(1).copy()
        tail["session"] = "2020-10-05"
        frames["daily_price_raw"] = pd.concat(
            [frames["daily_price_raw"], tail], ignore_index=True
        )
    elif case == "price_row_count":
        prices = frames["daily_price_raw"]
        nbl = prices.index[
            prices["security_id"].eq(expected["security_id"])
        ]
        frames["daily_price_raw"] = prices.drop(nbl[len(nbl) // 2]).reset_index(
            drop=True
        )
    elif case == "master_boundary":
        frames["security_master"].loc[0, "active_to"] = "2020-10-01"
    elif case == "history_boundary":
        frames["symbol_history"].loc[0, "effective_to"] = "2020-10-01"
    else:
        events = frames["index_membership_events"]
        remove = events["event_id"].eq(expected["next_remove_event_id"])
        field, value = {
            "remove_event": ("event_id", "different-remove-event"),
            "remove_date": ("effective_date", "2020-10-13"),
            "remove_source": ("source", "different-community-source"),
            "remove_hash": ("source_hash", "f" * 64),
        }[case]
        events.loc[remove, field] = value


def test_strict_default_stays_blocking_but_exact_operational_state_passes() -> None:
    repository = _FrameRepository(_exact_frames())

    strict = validate_repository_snapshot(repository)
    operational = validate_operational_repository_snapshot(repository)

    error = next(
        issue
        for issue in strict.issues
        if issue.code == "index_member_missing_active_symbol"
    )
    assert error.fingerprints == (EXPECTED_FINGERPRINT,)
    assert not strict.valid
    assert reviewed_operational_index_identity_gap_fingerprints(repository) == (
        EXPECTED_FINGERPRINT,
    )
    assert operational.valid, operational.issues


@pytest.mark.parametrize(
    "case",
    (
        "action",
        "resolution",
        "price_tail",
        "price_row_count",
        "master_boundary",
        "history_boundary",
        "remove_event",
        "remove_date",
        "remove_source",
        "remove_hash",
    ),
)
def test_each_terminal_state_or_remove_lineage_mutation_fails_closed(
    case: str,
) -> None:
    frames = copy.deepcopy(_exact_frames())
    _mutate(frames, case)
    repository = _FrameRepository(frames)

    assert reviewed_operational_index_identity_gap_fingerprints(repository) == ()
    report = validate_operational_repository_snapshot(repository)
    assert not report.valid
    assert "index_member_missing_active_symbol" in {
        issue.code for issue in report.issues
    }


def test_quant_data_validate_uses_operational_gate(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _materialize_repository(cache)
    config = tmp_path / "data.yaml"
    config.write_text(
        "data_store:\n"
        "  provider: parquet\n"
        f"  local_cache_dir: {cache}\n",
        encoding="utf-8",
    )
    output = io.StringIO()
    with patch(
        "sys.argv",
        [
            "quant-data",
            "--data-config",
            str(config),
            "validate",
            "--dataset",
            "security_master",
        ],
    ), redirect_stdout(output):
        data_main()

    result = json.loads(output.getvalue())
    assert result[-1]["dataset"] == "repository_snapshot"
    assert result[-1]["valid"] is True


def test_quant_data_validate_defaults_to_release_inventory(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    repository = _materialize_repository(cache)
    release, _ = repository.current_release()
    assert release is not None
    assert "custom_universe_overlays" not in release.dataset_versions
    assert "cross_validation_reports" not in release.dataset_versions
    config = tmp_path / "data.yaml"
    config.write_text(
        "data_store:\n"
        "  provider: parquet\n"
        f"  local_cache_dir: {cache}\n",
        encoding="utf-8",
    )
    output = io.StringIO()
    with patch(
        "sys.argv",
        ["quant-data", "--data-config", str(config), "validate"],
    ), redirect_stdout(output):
        data_main()

    result = json.loads(output.getvalue())
    validated = {item["dataset"] for item in result}
    assert validated == {*release.dataset_versions, "repository_snapshot"}
    assert all(item["valid"] is True for item in result)


class _NextSessionPriceSource:
    def fetch(self, securities: dict[str, str], *, start: str, end: str):
        expected = TRUSTED_OPERATIONAL_NBL_TERMINAL_STATE
        assert expected["security_id"] not in securities
        rows = []
        for offset, security_id in enumerate(sorted(securities), start=1):
            close = 30.0 + offset
            rows.append(
                _source(
                    {
                        "security_id": security_id,
                        "session": end,
                        "open": close,
                        "high": close + 1.0,
                        "low": close - 1.0,
                        "close": close,
                        "volume": 1_000.0,
                        "currency": "USD",
                    }
                )
            )
        return YahooFetchResult(
            prices=pd.DataFrame(rows),
            corporate_actions=pd.DataFrame(),
            artifacts=(),
            missing_symbols=(),
        )


def test_future_sync_operational_gate_commits_release(tmp_path: Path) -> None:
    repository = _materialize_repository(tmp_path / "cache")
    before, _ = repository.current_release()
    synchronizer = DailyDataSynchronizer(
        repository,
        security_source=SimpleNamespace(),
        price_source=_NextSessionPriceSource(),
    )

    result = synchronizer.sync(
        "2020-10-13",
        refresh_security_master=False,
    )

    after, _ = repository.current_release()
    assert before is not None and after is not None
    assert result.release_version == after.version
    assert after.version != before.version
    assert after.completed_session == "2020-10-13"
    assert validate_operational_repository_snapshot(repository).valid


def test_index_import_final_gate_uses_operational_policy(tmp_path: Path) -> None:
    repository = _materialize_repository(tmp_path / "cache")
    importer = IndexDataImporter(repository)

    result = importer.import_overlays(
        "sp500",
        pd.DataFrame(
            [
                {
                    "effective_from": "2020-10-13",
                    "operation": "ADD",
                    "security_id": "OTHER",
                    "reason": "operational gate fixture",
                }
            ]
        ),
    )

    release, _ = repository.current_release()
    assert release is not None
    assert result.release_version == release.version
    assert validate_operational_repository_snapshot(repository).valid
