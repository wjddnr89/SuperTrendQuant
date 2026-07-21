#!/usr/bin/env python3
"""Remove two synthetic post-identity 21CF bars without network access.

EODHD's retired TFCF/TFCFA responses each contain a 2019-03-20 zero-range,
near-zero-volume bar that repeats the official 2019-03-19 final close.  The
raw responses remain immutable in ``source_archive``; only the published price
and derived adjustment-factor rows beyond the reviewed identity boundary are
removed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.validation import (
    validate_dataset,
    validate_repository_snapshot,
)


BOUNDARY = "2019-03-19"
OVERRUN_SESSION = "2019-03-20"
TARGETS = {
    "TFCF": "US:EODHD:acd9ed55-bf0c-5b15-b624-1a917bf6078e",
    "TFCFA": "US:EODHD:9398e16f-425d-5a51-8720-35fba7433f28",
}
WRITE_DATASETS = ("daily_price_raw", "adjustment_factors")


@dataclass(frozen=True)
class PreparedFoxTerminalRepair:
    frames: dict[str, pd.DataFrame]
    summary: dict[str, Any]


def _session_strings(frame: pd.DataFrame) -> pd.Series:
    parsed = pd.to_datetime(frame["session"], errors="coerce")
    if parsed.isna().any():
        raise ValueError("21CF repair encountered an invalid session.")
    return parsed.dt.date.astype(str)


def _require_identity_boundaries(
    master: pd.DataFrame,
    history: pd.DataFrame,
) -> None:
    for symbol, security_id in TARGETS.items():
        master_rows = master.loc[
            master["security_id"].astype(str).eq(security_id)
        ]
        if not (
            len(master_rows) == 1
            and str(master_rows.iloc[0]["primary_symbol"]) == symbol
            and str(master_rows.iloc[0]["active_to"]) == BOUNDARY
        ):
            raise ValueError(f"{symbol} master identity boundary is not exact.")
        history_rows = history.loc[
            history["security_id"].astype(str).eq(security_id)
            & history["symbol"].astype(str).eq(symbol)
        ]
        if not (
            len(history_rows) == 1
            and str(history_rows.iloc[0]["effective_to"]) == BOUNDARY
        ):
            raise ValueError(f"{symbol} symbol-history boundary is not exact.")


def prepare_fox_terminal_repair(
    frames: Mapping[str, pd.DataFrame],
) -> PreparedFoxTerminalRepair:
    required = {
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "adjustment_factors",
        "source_archive",
    }
    missing = sorted(required - set(frames))
    if missing:
        raise ValueError("21CF repair is missing frames: " + ", ".join(missing))

    _require_identity_boundaries(frames["security_master"], frames["symbol_history"])
    prices = frames["daily_price_raw"].copy()
    factors = frames["adjustment_factors"].copy()
    price_sessions = _session_strings(prices)
    factor_sessions = _session_strings(factors)
    target_ids = set(TARGETS.values())
    price_target = prices["security_id"].astype(str).isin(target_ids)
    factor_target = factors["security_id"].astype(str).isin(target_ids)
    price_overrun = price_target & price_sessions.gt(BOUNDARY)
    factor_overrun = factor_target & factor_sessions.gt(BOUNDARY)

    if not price_overrun.any():
        maxima = {
            symbol: max(
                price_sessions.loc[
                    prices["security_id"].astype(str).eq(security_id)
                ],
                default="",
            )
            for symbol, security_id in TARGETS.items()
        }
        if set(maxima.values()) != {BOUNDARY} or factor_overrun.any():
            raise ValueError("21CF terminal repair is partially applied or inconsistent.")
        return PreparedFoxTerminalRepair(
            frames={dataset: frames[dataset].copy() for dataset in WRITE_DATASETS},
            summary={
                "status": "already_repaired",
                "boundary": BOUNDARY,
                "price_rows_removed": 0,
                "factor_rows_removed": 0,
            },
        )

    overrun_rows = prices.loc[price_overrun].copy()
    if not (
        len(overrun_rows) == 2
        and set(overrun_rows["security_id"].astype(str)) == target_ids
        and set(_session_strings(overrun_rows)) == {OVERRUN_SESSION}
        and set(overrun_rows["source"].astype(str)) == {"eodhd_eod"}
    ):
        raise ValueError("21CF terminal overrun inventory changed.")
    for row in overrun_rows.itertuples(index=False):
        values = [float(row.open), float(row.high), float(row.low), float(row.close)]
        if max(values) != min(values) or float(row.volume) >= 100:
            raise ValueError("21CF terminal overrun is no longer a synthetic flat bar.")
        previous = prices.loc[
            prices["security_id"].astype(str).eq(str(row.security_id))
            & price_sessions.eq(BOUNDARY)
        ]
        if len(previous) != 1 or float(previous.iloc[0]["close"]) != float(row.close):
            raise ValueError("21CF terminal overrun does not repeat the final close.")
        archived = frames["source_archive"].loc[
            frames["source_archive"]["source_url"].astype(str).eq(str(row.source_url))
            & frames["source_archive"]["source_hash"].astype(str).eq(
                str(row.source_hash)
            )
        ]
        if len(archived) != 1:
            raise ValueError("21CF raw provider response is not retained in source_archive.")

    if not (
        int(factor_overrun.sum()) == 2
        and set(factors.loc[factor_overrun, "security_id"].astype(str)) == target_ids
        and set(factor_sessions.loc[factor_overrun]) == {OVERRUN_SESSION}
    ):
        raise ValueError("21CF adjustment-factor overrun inventory changed.")

    repaired_prices = prices.loc[~price_overrun].reset_index(drop=True)
    repaired_factors = factors.loc[~factor_overrun].reset_index(drop=True)
    repaired_sessions = _session_strings(repaired_prices)
    for security_id in target_ids:
        if max(
            repaired_sessions.loc[
                repaired_prices["security_id"].astype(str).eq(security_id)
            ]
        ) != BOUNDARY:
            raise ValueError("21CF repaired history does not end on the official boundary.")
    return PreparedFoxTerminalRepair(
        frames={
            "daily_price_raw": repaired_prices,
            "adjustment_factors": repaired_factors,
        },
        summary={
            "status": "validated_dry_run",
            "boundary": BOUNDARY,
            "overrun_session": OVERRUN_SESSION,
            "price_rows_removed": int(price_overrun.sum()),
            "factor_rows_removed": int(factor_overrun.sum()),
            "raw_artifacts_retained": 2,
            "targets": TARGETS,
        },
    )


class _CandidateRepository:
    def __init__(
        self,
        base: LocalDatasetRepository,
        versions: Mapping[str, str],
        overrides: Mapping[str, pd.DataFrame],
    ):
        self.base = base
        self.versions = dict(versions)
        self.overrides = dict(overrides)

    def current_manifest(self, dataset: str):
        return self.base.manifest_for_version(dataset, self.versions[dataset])

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        if dataset in self.overrides:
            return self.overrides[dataset].copy()
        return self.base.read_frame(dataset, self.versions[dataset])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove synthetic 2019-03-20 TFCF/TFCFA terminal bars offline."
    )
    parser.add_argument("--cache-root", default="data/cache")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    repository = LocalDatasetRepository(args.cache_root)
    release, etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    required = (
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "adjustment_factors",
        "source_archive",
    )
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in required
    }
    prepared = prepare_fox_terminal_repair(frames)
    summary = {
        **prepared.summary,
        "release_version": release.version,
        "network_accessed": False,
    }
    if not args.apply or summary["status"] == "already_repaired":
        print(json.dumps(summary, sort_keys=True))
        return 0

    for dataset, frame in prepared.frames.items():
        report = validate_dataset(
            dataset,
            frame,
            completed_session=release.completed_session,
            incomplete_action_policy="warn",
        )
        report.raise_for_errors()
    candidate = _CandidateRepository(
        repository, release.dataset_versions, prepared.frames
    )
    validate_repository_snapshot(candidate).raise_for_errors()
    current, current_etag = repository.current_release()
    if current is None or current.version != release.version or current_etag != etag:
        raise RuntimeError("Current release changed during 21CF repair validation.")

    versions = dict(release.dataset_versions)
    for dataset, frame in prepared.frames.items():
        result = repository.write_frame(
            dataset,
            frame,
            completed_session=release.completed_session,
            metadata={
                "operation": "repair_us_fox_terminal_prices",
                "official_last_trading_session": BOUNDARY,
                "removed_provider_session": OVERRUN_SESSION,
            },
        )
        versions[dataset] = result.manifest.version
    committed = repository.commit_release(
        release.completed_session,
        versions,
        quality=(
            DataQuality.DEGRADED if release.warnings else DataQuality.VALID
        ),
        warnings=release.warnings,
        expected_etag=etag,
    )
    validate_repository_snapshot(repository).raise_for_errors()
    print(
        json.dumps(
            {
                **summary,
                "status": "applied",
                "new_release_version": committed.version,
                "quality": str(committed.quality),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
