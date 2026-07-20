#!/usr/bin/env python3
"""Repair three reviewed EODHD terminal identity boundaries offline.

The archived EODHD responses for ABMD, legacy DO, and DNR contain zero-volume
flat pseudo-bars after the last real trading session.  This repair keeps those
raw responses immutable in ``source_archive`` while it:

* removes only the exact reviewed pseudo-bars from ``daily_price_raw``;
* removes the matching derived ``adjustment_factors`` rows;
* ends ``security_master`` and ``symbol_history`` on the last real session; and
* binds both edited identity rows to the exact archived EODHD response.

The command has no networking code.  ``--plan`` performs every validation but
does not write.  ``--apply`` writes immutable dataset versions and advances the
release pointer with compare-and-swap only after the candidate snapshot passes.
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


@dataclass(frozen=True)
class TerminalTarget:
    symbol: str
    security_id: str
    final_session: str
    original_active_to: str
    overrun_sessions: tuple[str, ...]
    source_url: str
    source_hash: str


TARGETS: tuple[TerminalTarget, ...] = (
    TerminalTarget(
        symbol="ABMD",
        security_id="US:EODHD:faece1b7-4b1a-5c1f-951b-b1178ed57161",
        final_session="2022-12-21",
        original_active_to="2022-12-23",
        overrun_sessions=("2022-12-22", "2022-12-23"),
        source_url=(
            "https://eodhd.com/api/eod/ABMD.US?from=2015-01-01&to=2026-07-15"
        ),
        source_hash=(
            "17bbbcbdef014339087209771c41fa63b592ef2517d240e4cf034b4384c48b5a"
        ),
    ),
    TerminalTarget(
        symbol="DO",
        security_id="US:EODHD:2826c370-0467-5e82-9617-dcece5be407f",
        final_session="2020-04-24",
        original_active_to="2020-04-29",
        overrun_sessions=("2020-04-27", "2020-04-28", "2020-04-29"),
        source_url=(
            "https://eodhd.com/api/eod/DO_old.US?from=2015-01-01&to=2026-07-15"
        ),
        source_hash=(
            "776fea367dad9318e40454e5a92458aaf6e96dbe3b53965bfc208fe43273c228"
        ),
    ),
    TerminalTarget(
        symbol="DNR",
        security_id="US:EODHD:6d9d4638-4922-5f6c-89fd-6b79db60c1c3",
        final_session="2020-07-28",
        original_active_to="2020-07-30",
        overrun_sessions=("2020-07-29", "2020-07-30"),
        source_url=(
            "https://eodhd.com/api/eod/DNR.US?from=2015-01-01&to=2026-07-15"
        ),
        source_hash=(
            "43b2f83f101e62c808edaa5e4fb53bfdf26be135c44946ddea30c233ec723fbc"
        ),
    ),
)

REQUIRED_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "adjustment_factors",
    "source_archive",
)
WRITE_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "adjustment_factors",
)
EXPECTED_PRICE_ROWS_REMOVED = sum(len(target.overrun_sessions) for target in TARGETS)


@dataclass(frozen=True)
class PreparedTerminalBoundaryRepair:
    frames: dict[str, pd.DataFrame]
    summary: dict[str, Any]


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _date_text(value: Any) -> str:
    if _text(value) == "":
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid date value in terminal repair: {value!r}")
    return pd.Timestamp(parsed).date().isoformat()


def _session_strings(frame: pd.DataFrame) -> pd.Series:
    parsed = pd.to_datetime(frame["session"], errors="coerce")
    if parsed.isna().any():
        raise ValueError("Terminal-boundary repair encountered an invalid session.")
    return parsed.dt.date.astype(str)


def _archive_provenance(
    archive: pd.DataFrame,
    target: TerminalTarget,
) -> dict[str, str]:
    matches = archive.loc[
        archive["source_hash"].astype(str).eq(target.source_hash)
        & archive["source_url"].astype(str).eq(target.source_url)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{target.symbol} exact EODHD source_archive URL/hash binding is missing."
        )
    row = matches.iloc[0]
    expected_path_suffix = f"/{target.source_hash}.json.gz"
    if not (
        _text(row.get("archive_id")) == target.source_hash
        and _text(row.get("dataset")) == "eodhd_eod"
        and _text(row.get("source")) == "eodhd_eod"
        and _text(row.get("content_type")) == "application/json"
        and _text(row.get("object_path")).endswith(expected_path_suffix)
        and _text(row.get("retrieved_at"))
    ):
        raise ValueError(f"{target.symbol} archived EODHD artifact metadata changed.")
    return {
        "source": "eodhd_eod",
        "source_url": target.source_url,
        "source_hash": target.source_hash,
        "retrieved_at": _text(row["retrieved_at"]),
    }


def _identity_rows(
    master: pd.DataFrame,
    history: pd.DataFrame,
    target: TerminalTarget,
) -> tuple[pd.Index, pd.Index]:
    master_index = master.index[
        master["security_id"].astype(str).eq(target.security_id)
    ]
    history_index = history.index[
        history["security_id"].astype(str).eq(target.security_id)
        & history["symbol"].astype(str).eq(target.symbol)
    ]
    if len(master_index) != 1 or len(history_index) != 1:
        raise ValueError(f"{target.symbol} identity row inventory is not exact.")
    master_row = master.loc[master_index[0]]
    if _text(master_row.get("primary_symbol")) != target.symbol:
        raise ValueError(f"{target.symbol} security_master symbol identity changed.")
    return master_index, history_index


def _has_exact_provenance(row: pd.Series, provenance: Mapping[str, str]) -> bool:
    return all(_text(row.get(column)) == value for column, value in provenance.items())


def _target_is_repaired(
    master: pd.DataFrame,
    history: pd.DataFrame,
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    price_sessions: pd.Series,
    factor_sessions: pd.Series,
    target: TerminalTarget,
    provenance: Mapping[str, str],
) -> bool:
    master_index, history_index = _identity_rows(master, history, target)
    target_prices = prices["security_id"].astype(str).eq(target.security_id)
    target_factors = factors["security_id"].astype(str).eq(target.security_id)
    sessions = price_sessions.loc[target_prices]
    return bool(
        _date_text(master.loc[master_index[0], "active_to"])
        == target.final_session
        and _date_text(history.loc[history_index[0], "effective_to"])
        == target.final_session
        and _has_exact_provenance(master.loc[master_index[0]], provenance)
        and _has_exact_provenance(history.loc[history_index[0]], provenance)
        and len(sessions) > 0
        and sessions.max() == target.final_session
        and not factor_sessions.loc[target_factors].gt(target.final_session).any()
    )


def prepare_terminal_boundary_repair(
    frames: Mapping[str, pd.DataFrame],
) -> PreparedTerminalBoundaryRepair:
    """Validate and prepare the exact ABMD/DO/DNR repair in memory."""

    missing = sorted(set(REQUIRED_DATASETS) - set(frames))
    if missing:
        raise ValueError("Terminal repair is missing frames: " + ", ".join(missing))

    master = frames["security_master"].copy()
    history = frames["symbol_history"].copy()
    prices = frames["daily_price_raw"].copy()
    factors = frames["adjustment_factors"].copy()
    archive = frames["source_archive"]
    price_sessions = _session_strings(prices)
    factor_sessions = _session_strings(factors)
    provenance = {
        target.security_id: _archive_provenance(archive, target)
        for target in TARGETS
    }

    repaired = {
        target.security_id: _target_is_repaired(
            master,
            history,
            prices,
            factors,
            price_sessions,
            factor_sessions,
            target,
            provenance[target.security_id],
        )
        for target in TARGETS
    }
    if all(repaired.values()):
        return PreparedTerminalBoundaryRepair(
            frames={dataset: frames[dataset].copy() for dataset in WRITE_DATASETS},
            summary={
                "status": "already_repaired",
                "price_rows_removed": 0,
                "factor_rows_removed": 0,
                "identity_rows_updated": 0,
                "targets": {
                    target.symbol: target.final_session for target in TARGETS
                },
            },
        )
    if any(repaired.values()):
        raise ValueError("Terminal-boundary repair is partially applied.")

    remove_price = pd.Series(False, index=prices.index)
    remove_factor = pd.Series(False, index=factors.index)
    evidence: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        master_index, history_index = _identity_rows(master, history, target)
        master_row = master.loc[master_index[0]]
        history_row = history.loc[history_index[0]]
        if _date_text(master_row["active_to"]) != target.original_active_to:
            raise ValueError(f"{target.symbol} original master boundary changed.")
        if _date_text(history_row["effective_to"]) not in {
            "",
            target.original_active_to,
        }:
            raise ValueError(f"{target.symbol} original symbol-history boundary changed.")

        target_prices = prices["security_id"].astype(str).eq(target.security_id)
        final_rows = prices.loc[target_prices & price_sessions.eq(target.final_session)]
        overrun_mask = target_prices & price_sessions.gt(target.final_session)
        overrun_rows = prices.loc[overrun_mask]
        if len(final_rows) != 1:
            raise ValueError(f"{target.symbol} final real price row is not exact.")
        if tuple(sorted(_session_strings(overrun_rows))) != target.overrun_sessions:
            raise ValueError(f"{target.symbol} terminal pseudo-bar inventory changed.")

        final_close = float(final_rows.iloc[0]["close"])
        for row in overrun_rows.itertuples(index=False):
            ohlc = [float(row.open), float(row.high), float(row.low), float(row.close)]
            if not (
                float(row.volume) == 0.0
                and max(ohlc) == min(ohlc) == final_close
                and _text(row.source) == "eodhd_eod"
                and _text(row.source_hash) == target.source_hash
                and _text(row.retrieved_at)
                == provenance[target.security_id]["retrieved_at"]
            ):
                raise ValueError(
                    f"{target.symbol} overrun is not the exact archived flat pseudo-bar."
                )

        target_factors = factors["security_id"].astype(str).eq(target.security_id)
        factor_overrun = target_factors & factor_sessions.gt(target.final_session)
        if tuple(sorted(factor_sessions.loc[factor_overrun])) != target.overrun_sessions:
            raise ValueError(
                f"{target.symbol} adjustment-factor overrun inventory changed."
            )

        remove_price |= overrun_mask
        remove_factor |= factor_overrun
        row_provenance = provenance[target.security_id]
        for column, value in row_provenance.items():
            master.loc[master_index, column] = value
            history.loc[history_index, column] = value
        master.loc[master_index, "active_to"] = target.final_session
        history.loc[history_index, "effective_to"] = target.final_session
        evidence[target.symbol] = {
            "security_id": target.security_id,
            "final_session": target.final_session,
            "removed_sessions": list(target.overrun_sessions),
            "source_url": target.source_url,
            "source_hash": target.source_hash,
        }

    if int(remove_price.sum()) != EXPECTED_PRICE_ROWS_REMOVED:
        raise ValueError("Terminal price removal count is not exact.")
    if int(remove_factor.sum()) != EXPECTED_PRICE_ROWS_REMOVED:
        raise ValueError("Terminal factor removal count is not exact.")

    repaired_prices = prices.loc[~remove_price].reset_index(drop=True)
    repaired_factors = factors.loc[~remove_factor].reset_index(drop=True)
    repaired_price_sessions = _session_strings(repaired_prices)
    for target in TARGETS:
        mask = repaired_prices["security_id"].astype(str).eq(target.security_id)
        if repaired_price_sessions.loc[mask].max() != target.final_session:
            raise ValueError(f"{target.symbol} repaired price boundary is not exact.")

    return PreparedTerminalBoundaryRepair(
        frames={
            "security_master": master,
            "symbol_history": history,
            "daily_price_raw": repaired_prices,
            "adjustment_factors": repaired_factors,
        },
        summary={
            "status": "validated_dry_run",
            "price_rows_removed": int(remove_price.sum()),
            "factor_rows_removed": int(remove_factor.sum()),
            "identity_rows_updated": len(TARGETS) * 2,
            "raw_artifacts_retained": len(TARGETS),
            "evidence": evidence,
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
        if dataset not in self.versions:
            return None
        return self.base.manifest_for_version(dataset, self.versions[dataset])

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        if dataset in self.overrides:
            return self.overrides[dataset].copy()
        return self.base.read_frame(dataset, self.versions[dataset])


def run_repair(
    repository: LocalDatasetRepository,
    *,
    apply: bool,
) -> dict[str, Any]:
    """Plan or atomically apply the repair against one pinned current release."""

    release, etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    missing_versions = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    if missing_versions:
        raise RuntimeError(
            "Current release lacks terminal-repair datasets: "
            + ", ".join(missing_versions)
        )
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in REQUIRED_DATASETS
    }
    prepared = prepare_terminal_boundary_repair(frames)
    summary = {
        **prepared.summary,
        "release_version": release.version,
        "network_accessed": False,
    }
    if prepared.summary["status"] == "already_repaired":
        validate_repository_snapshot(repository).raise_for_errors()
        return summary

    for dataset, frame in prepared.frames.items():
        validate_dataset(
            dataset,
            frame,
            completed_session=release.completed_session,
            incomplete_action_policy="warn",
        ).raise_for_errors()
    candidate = _CandidateRepository(
        repository,
        release.dataset_versions,
        prepared.frames,
    )
    validate_repository_snapshot(candidate).raise_for_errors()

    if not apply:
        return summary

    current, current_etag = repository.current_release()
    if current is None or current.version != release.version or current_etag != etag:
        raise RuntimeError(
            "Current release changed during terminal-boundary repair validation."
        )

    versions = dict(release.dataset_versions)
    for dataset in WRITE_DATASETS:
        result = repository.write_frame(
            dataset,
            prepared.frames[dataset],
            completed_session=release.completed_session,
            metadata={
                "operation": "repair_us_terminal_identity_boundaries",
                "targets": [target.symbol for target in TARGETS],
                "network_accessed": False,
            },
        )
        versions[dataset] = result.manifest.version

    committed = repository.commit_release(
        release.completed_session,
        versions,
        quality=DataQuality.DEGRADED if release.warnings else DataQuality.VALID,
        warnings=release.warnings,
        expected_etag=etag,
    )
    validate_repository_snapshot(repository).raise_for_errors()
    return {
        **summary,
        "status": "applied",
        "new_release_version": committed.version,
        "quality": str(committed.quality),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair ABMD, legacy DO, and DNR terminal boundaries offline."
    )
    parser.add_argument("--cache-root", default="data/cache")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    summary = run_repair(
        LocalDatasetRepository(args.cache_root),
        apply=bool(args.apply),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
