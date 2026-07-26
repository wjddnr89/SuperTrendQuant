"""Compare one year of strict KRX Open API prices with the local KR release.

The collection checkpoint is intentionally separate from the production KR
cache.  Empty KRX web credentials make the Open API path fail closed instead
of silently falling back to the authenticated Data Marketplace transport.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from supertrend_quant.config import load_split_config
from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.kr_pipeline import (
    KrCheckpointStore,
    _classify_cached_delisting_effective_absences,
    _collect_krx_prices,
    _traded_krx_prices,
)
from supertrend_quant.market_store.kr_providers import (
    KR_BENCHMARK_SECURITIES,
    KrIdentityCatalog,
    normalize_kr_symbol,
)
from supertrend_quant.market_store.manifest import write_atomic
from supertrend_quant.market_store.markets import recent_sessions, sessions_between
from supertrend_quant.market_store.provider import ParquetMarketDataProvider
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.results import save_backtest_result
from supertrend_quant.runners import (
    _schedule_for_period,
    run_backtest_on_data,
)
from supertrend_quant.universe import resolve_universe


OPENAPI_SOURCE = "krx_openapi_daily_ohlcv"
TRADED_STATUS = "traded"
PROFILES = ("kospi200", "kosdaq150")
PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")
    write_atomic(path, encoded)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _reference_symbols(
    reference_root: Path,
    sessions: tuple[str, ...],
) -> set[str]:
    selected: set[str] = set(KR_BENCHMARK_SECURITIES.values())
    allowed_sessions = set(sessions)
    for profile in PROFILES:
        profile_root = reference_root / "memberships" / profile
        for path in profile_root.glob("*.json"):
            if path.stem not in allowed_sessions:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            selected.update(
                normalize_kr_symbol(value)
                for value in payload.get("symbols", ())
            )
    if not selected:
        raise RuntimeError(
            f"No reference membership symbols were found below {reference_root}."
        )
    return {value for value in selected if value}


def _expand_action_linked_symbols(
    symbols: set[str],
    symbol_history: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> set[str]:
    """Include aliases and action successors the backtest may need to value."""

    history = symbol_history.copy()
    history["symbol"] = history["symbol"].map(normalize_kr_symbol)
    managed_ids = set(
        history.loc[history["symbol"].isin(symbols), "security_id"].astype(str)
    )
    actions = corporate_actions.copy()
    action_dates = pd.to_datetime(
        actions["effective_date"], errors="coerce"
    ).dt.normalize()
    actions = actions.loc[
        action_dates.between(pd.Timestamp(start), pd.Timestamp(end))
    ].copy()
    changed = True
    while changed:
        changed = False
        linked = actions.loc[
            actions["security_id"].astype(str).isin(managed_ids)
        ]
        successors = {
            str(value).strip()
            for value in linked.get(
                "new_security_id", pd.Series(dtype=str)
            )
            if str(value).strip()
        }
        if not successors <= managed_ids:
            managed_ids.update(successors)
            changed = True
    aliases = {
        normalize_kr_symbol(value)
        for value in history.loc[
            history["security_id"].astype(str).isin(managed_ids), "symbol"
        ]
    }
    return {value for value in symbols | aliases if value}


def _catalog_from_release_history(
    fallback_catalog: pd.DataFrame,
    symbol_history: pd.DataFrame,
    security_master: pd.DataFrame,
) -> KrIdentityCatalog:
    """Build the session-aware identity catalog used by the released PIT data.

    The bootstrap catalog is captured before official daily observations
    reconcile exchange transfers and older ticker intervals.  The immutable
    release's symbol history is the authoritative result of that reconciliation
    and must therefore drive a full-history Open API comparison.
    """

    fallback_by_id = {
        str(security_id): group.iloc[-1].to_dict()
        for security_id, group in fallback_catalog.groupby(
            fallback_catalog["security_id"].astype(str),
            sort=False,
        )
    }
    master_by_id = {
        str(security_id): group.iloc[-1].to_dict()
        for security_id, group in security_master.groupby(
            security_master["security_id"].astype(str),
            sort=False,
        )
    }
    records: list[dict[str, Any]] = []
    for row in symbol_history.to_dict("records"):
        security_id = str(row.get("security_id") or "").strip()
        symbol = normalize_kr_symbol(row.get("symbol"))
        exchange = str(row.get("exchange") or "").strip().upper()
        base = dict(fallback_by_id.get(security_id, {}))
        base.update(master_by_id.get(security_id, {}))
        base.update(
            {
                "security_id": security_id,
                "primary_symbol": symbol,
                "name": str(base.get("name") or symbol),
                "exchange": exchange,
                "asset_type": str(base.get("asset_type") or "STOCK"),
                "currency": str(base.get("currency") or "KRW"),
                "country": str(base.get("country") or "KR"),
                "active_from": str(row.get("effective_from") or ""),
                "active_to": str(row.get("effective_to") or ""),
                "isin": security_id.removeprefix("KR:"),
                "identity_mapped": True,
                "provider_symbol": (
                    f"{symbol}.{'KO' if exchange == 'KOSPI' else 'KQ'}"
                ),
                "yahoo_symbol": (
                    f"{symbol}.{'KS' if exchange == 'KOSPI' else 'KQ'}"
                ),
                "source": str(
                    row.get("source")
                    or base.get("source")
                    or "krx_official_daily_identity"
                ),
                "source_url": str(
                    row.get("source_url")
                    or base.get("source_url")
                    or "https://data.krx.co.kr/"
                ),
                "retrieved_at": str(
                    row.get("retrieved_at")
                    or base.get("retrieved_at")
                    or ""
                ),
                "source_hash": str(
                    row.get("source_hash")
                    or base.get("source_hash")
                    or ""
                ),
                "dart_corp_code": str(base.get("dart_corp_code") or ""),
            }
        )
        records.append(base)
    frame = pd.DataFrame(
        records,
        columns=fallback_catalog.columns,
    ).sort_values(
        ["primary_symbol", "active_from", "active_to", "security_id"],
        kind="stable",
    ).drop_duplicates(
        ["security_id", "primary_symbol", "active_from", "exchange"],
        keep="last",
    )
    return KrIdentityCatalog(frame.reset_index(drop=True), ())


def _load_checkpoint_prices(
    root: Path,
    sessions: tuple[str, ...],
    catalog: KrIdentityCatalog,
) -> tuple[pd.DataFrame, int]:
    checkpoint = KrCheckpointStore(root)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    reclassified = 0
    for session in sessions:
        frame = checkpoint.load_prices(session)
        if frame is None:
            missing.append(session)
        else:
            before = frame.get(
                "observation_status",
                pd.Series("", index=frame.index),
            ).astype(str)
            frame = _classify_cached_delisting_effective_absences(
                frame,
                catalog,
                session,
            )
            after = frame.get(
                "observation_status",
                pd.Series("", index=frame.index),
            ).astype(str)
            reclassified += int(before.ne(after).sum())
            frames.append(frame)
    if missing:
        sample = ", ".join(missing[:5])
        raise RuntimeError(
            f"Reference checkpoint lacks {len(missing)} sessions: {sample}"
        )
    return (
        pd.concat(frames, ignore_index=True).drop_duplicates(
            ["security_id", "session"], keep="last"
        ),
        reclassified,
    )


def _normalise_observations(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["security_id"] = output["security_id"].astype(str)
    output["session"] = pd.to_datetime(
        output["session"], errors="coerce"
    ).dt.date.astype(str)
    if "symbol" in output:
        output["symbol"] = output["symbol"].map(normalize_kr_symbol)
    if "observation_status" not in output:
        output["observation_status"] = TRADED_STATUS
    output["observation_status"] = (
        output["observation_status"]
        .fillna(TRADED_STATUS)
        .replace("", TRADED_STATUS)
        .astype(str)
    )
    return output


def _compare_observations(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    key = ["security_id", "session"]
    left_frame = _normalise_observations(left).drop_duplicates(key, keep="last")
    right_frame = _normalise_observations(right).drop_duplicates(key, keep="last")
    merged = left_frame.merge(
        right_frame,
        on=key,
        how="outer",
        suffixes=("_left", "_right"),
        indicator=True,
    )
    shared = merged.loc[merged["_merge"].eq("both")].copy()
    left_status = shared["observation_status_left"].astype(str)
    right_status = shared["observation_status_right"].astype(str)
    status_mismatches = shared.loc[left_status.ne(right_status)].copy()
    compared = shared.loc[
        left_status.eq(TRADED_STATUS) & right_status.eq(TRADED_STATUS)
    ].copy()
    field_metrics: dict[str, dict[str, Any]] = {}
    row_match = np.ones(len(compared), dtype=bool)
    for column in PRICE_COLUMNS:
        left_values = pd.to_numeric(
            compared[f"{column}_left"], errors="coerce"
        ).to_numpy(dtype=float)
        right_values = pd.to_numeric(
            compared[f"{column}_right"], errors="coerce"
        ).to_numpy(dtype=float)
        equal = np.isclose(
            left_values,
            right_values,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
        row_match &= equal
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        differences = (
            np.abs(left_values[finite] - right_values[finite])
            if finite.any()
            else np.asarray([], dtype=float)
        )
        field_metrics[column] = {
            "mismatch_count": int((~equal).sum()),
            "max_absolute_difference": (
                float(differences.max()) if len(differences) else 0.0
            ),
        }
    status_pairs = (
        status_mismatches.groupby(
            ["observation_status_left", "observation_status_right"],
            dropna=False,
        )
        .size()
        .sort_values(ascending=False)
    )
    mismatch_symbol_column = (
        "symbol_left" if "symbol_left" in status_mismatches else ""
    )

    def key_sample(frame: pd.DataFrame, side: str) -> list[dict[str, str]]:
        columns = ["security_id", "session"]
        for name in (
            f"symbol_{side}",
            f"observation_status_{side}",
        ):
            if name in frame:
                columns.append(name)
        return (
            frame.loc[:, columns]
            .fillna("")
            .astype(str)
            .head(20)
            .to_dict("records")
        )

    return {
        "left": left_name,
        "right": right_name,
        "left_rows": len(left_frame),
        "right_rows": len(right_frame),
        "shared_keys": len(shared),
        "left_only_keys": int(merged["_merge"].eq("left_only").sum()),
        "right_only_keys": int(merged["_merge"].eq("right_only").sum()),
        "status_mismatch_count": int(left_status.ne(right_status).sum()),
        "status_mismatch_pairs": {
            f"{left_value} -> {right_value}": int(count)
            for (left_value, right_value), count in status_pairs.items()
        },
        "status_mismatch_top_symbols": (
            status_mismatches[mismatch_symbol_column]
            .fillna("")
            .astype(str)
            .value_counts()
            .head(20)
            .astype(int)
            .to_dict()
            if mismatch_symbol_column
            else {}
        ),
        "status_mismatch_sample": key_sample(status_mismatches, "left"),
        "left_only_sample": key_sample(
            merged.loc[merged["_merge"].eq("left_only")],
            "left",
        ),
        "right_only_sample": key_sample(
            merged.loc[merged["_merge"].eq("right_only")],
            "right",
        ),
        "traded_rows_compared": len(compared),
        "exact_ohlcv_rows": int(row_match.sum()),
        "exact_ohlcv_rate": (
            float(row_match.mean()) if len(row_match) else 0.0
        ),
        "fields": field_metrics,
        "left_statuses": left_frame["observation_status"].value_counts(
            dropna=False
        ).to_dict(),
        "right_statuses": right_frame["observation_status"].value_counts(
            dropna=False
        ).to_dict(),
        "left_sources": left_frame.get(
            "source", pd.Series(dtype=str)
        ).value_counts(dropna=False).to_dict(),
        "right_sources": right_frame.get(
            "source", pd.Series(dtype=str)
        ).value_counts(dropna=False).to_dict(),
    }


def _load_canonical_prices(
    repository: LocalDatasetRepository,
    *,
    version: str,
    start: str,
    end: str,
    security_ids: set[str],
) -> pd.DataFrame:
    paths = repository.parquet_paths(
        "daily_price_raw",
        version,
        min_session=start,
        max_session=end,
    )
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        return pd.DataFrame(
            columns=dataset_spec("daily_price_raw").required_columns
        )
    output = pd.concat(frames, ignore_index=True)
    sessions = pd.to_datetime(output["session"], errors="coerce").dt.normalize()
    return output.loc[
        output["security_id"].astype(str).isin(security_ids)
        & sessions.between(pd.Timestamp(start), pd.Timestamp(end))
    ].copy()


def _copy_link_or_file(source: str, target: str) -> str:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return target


def _build_openapi_repository(
    canonical_root: Path,
    temporary_root: Path,
    raw_prices: pd.DataFrame,
    *,
    completed_session: str,
) -> str:
    for name in ("datasets", "releases"):
        shutil.copytree(
            canonical_root / name,
            temporary_root / name,
            copy_function=_copy_link_or_file,
        )
    repository = LocalDatasetRepository(temporary_root, market="KR")
    current_release, _ = repository.current_release()
    if current_release is None:
        raise RuntimeError("The canonical KR repository has no current release.")
    required_price_columns = list(
        dataset_spec("daily_price_raw").required_columns
    )
    prepared = raw_prices.loc[:, required_price_columns].copy()
    daily_write = repository.write_frame(
        "daily_price_raw",
        prepared,
        completed_session=completed_session,
        incomplete_action_policy="warn",
        metadata={
            "operation": "kr_market_pipeline",
            "source_mode": "official_only",
            "primary_provider": "krx",
            "source_transport": "openapi",
            "comparison_only": True,
        },
    )
    actions = repository.read_frame(
        "corporate_actions",
        current_release.dataset_versions["corporate_actions"],
    )
    factors = build_adjustment_factors(
        prepared,
        actions,
        source_version=daily_write.manifest.version,
        dividend_tax_rate=0.0,
    )
    factor_write = repository.write_frame(
        "adjustment_factors",
        factors,
        completed_session=completed_session,
        incomplete_action_policy="warn",
        metadata={
            "operation": "kr_market_pipeline",
            "source_mode": "derived_from_openapi",
            "source_price_version": daily_write.manifest.version,
            "comparison_only": True,
        },
    )
    versions = dict(current_release.dataset_versions)
    versions["daily_price_raw"] = daily_write.manifest.version
    versions["adjustment_factors"] = factor_write.manifest.version
    metadata = dict(current_release.metadata)
    metadata.update(
        {
            "primary_provider": "krx",
            "source_transport": "openapi",
            "operation": "krx_openapi_comparison",
            "comparison_only": True,
        }
    )
    release = repository.commit_release(
        completed_session,
        versions,
        quality=current_release.quality,
        warnings=current_release.warnings,
        metadata=metadata,
    )
    return release.version


def _run_backtest(
    *,
    strategy_path: Path,
    runtime_path: Path,
    data_path: Path,
    repository_root: Path,
    period: str,
    run_id: str,
    completed_session: str,
) -> tuple[dict[str, Any], str]:
    loaded = load_split_config(strategy_path, runtime_path, data_path)
    config = replace(
        loaded,
        period=period,
        data_store=replace(
            loaded.data_store,
            local_cache_dir=str(repository_root),
        ),
    )
    resolved = resolve_universe(config, mode="backtest")
    schedule = _schedule_for_period(
        resolved.schedule,
        config.period,
        completed_session,
    )
    symbols = list(
        dict.fromkeys(
            member.symbol
            for entry in schedule
            for member in entry.members
        )
    )
    market_data = ParquetMarketDataProvider(
        repository_root,
        market="KR",
    ).load(
        config,
        symbols,
        universe_schedule=tuple(entry.to_dict() for entry in schedule),
    )
    market_data = replace(
        market_data,
        universe_snapshot=resolved.snapshot.to_dict(),
        universe_schedule=tuple(entry.to_dict() for entry in schedule),
    )
    result = run_backtest_on_data(
        config,
        market_data,
        capture_artifacts=True,
    )
    run_dir = save_backtest_result(
        result,
        config,
        config.backtest.results_dir,
        run_id=run_id,
    )
    return dict(result.metrics), str(run_dir)


def _metric_comparison(
    canonical: dict[str, Any],
    openapi: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in sorted(set(canonical) | set(openapi)):
        left = canonical.get(key)
        right = openapi.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            delta: Any = right - left
        else:
            delta = None
        output[key] = {
            "canonical": left,
            "openapi": right,
            "delta": delta,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-root",
        default="data/cache/markets/KR",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="data/cache/experiments/KRX_OPENAPI_252",
    )
    parser.add_argument(
        "--reference-root",
        default="",
        help=(
            "Existing KRX checkpoint to compare against. Defaults to the exact "
            "benchmark checkpoint, then state/bootstrap."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="results/research/kr/krx_openapi_comparison",
    )
    parser.add_argument("--session-count", type=int, default=252)
    parser.add_argument(
        "--start-session",
        default="",
        help="Collect every XKRX session from this date instead of recent N sessions.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=0.0,
        help="Per-session delay after the KRX market requests complete.",
    )
    parser.add_argument("--period", default="1y")
    parser.add_argument("--read-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-collection-restarts", type=int, default=50)
    parser.add_argument(
        "--rate-limit-cooldown-seconds",
        type=float,
        default=60.0,
        help="Wait before resuming from checkpoints after an HTTP 429.",
    )
    args = parser.parse_args()

    canonical_root = Path(args.canonical_root)
    canonical_repository = LocalDatasetRepository(
        canonical_root,
        market="KR",
    )
    canonical_release, _ = canonical_repository.current_release()
    if canonical_release is None:
        raise RuntimeError("The canonical KR repository has no current release.")
    completed_session = canonical_release.completed_session
    sessions = (
        sessions_between(
            "KR",
            args.start_session,
            completed_session,
        )
        if args.start_session
        else recent_sessions(
            "KR",
            args.session_count,
            end=completed_session,
        )
    )
    start_session, end_session = sessions[0], sessions[-1]
    benchmark_reference_root = (
        canonical_root
        / "benchmarks"
        / f"{start_session}_{end_session}_{len(sessions)}"
    )
    benchmark_has_prices = (
        benchmark_reference_root / "prices"
    ).is_dir() and any(
        (benchmark_reference_root / "prices").glob("*.parquet")
    )
    reference_root = (
        Path(args.reference_root)
        if args.reference_root
        else (
            benchmark_reference_root
            if benchmark_has_prices
            else canonical_root / "state" / "bootstrap"
        )
    )
    if not reference_root.is_dir():
        raise RuntimeError(
            f"The comparable authenticated-web checkpoint is absent: {reference_root}"
        )

    reference_symbols = _reference_symbols(reference_root, sessions)
    symbol_history = canonical_repository.read_frame(
        "symbol_history",
        canonical_release.dataset_versions["symbol_history"],
    )
    corporate_actions = canonical_repository.read_frame(
        "corporate_actions",
        canonical_release.dataset_versions["corporate_actions"],
    )
    security_master = canonical_repository.read_frame(
        "security_master",
        canonical_release.dataset_versions["security_master"],
    )
    collection_symbols = _expand_action_linked_symbols(
        reference_symbols,
        symbol_history,
        corporate_actions,
        start=start_session,
        end=end_session,
    )
    bootstrap_identity_frame = pd.read_parquet(
        canonical_root / "state" / "bootstrap" / "identity_catalog.parquet"
    )
    catalog = _catalog_from_release_history(
        bootstrap_identity_frame,
        symbol_history,
        security_master,
    )
    checkpoint_root = (
        Path(args.checkpoint_root)
        / f"{start_session}_{end_session}_{len(sessions)}"
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    # Fail closed: a rejected Open API request must not become a valid-looking
    # authenticated-web comparison just because KRX_ID/KRX_PW also exist.
    os.environ["KRX_DAILY_PRICE_TRANSPORT"] = "auto"
    # KRX can leave a reused TLS connection open without returning another
    # response status line.  Fresh connections are slower but keep a
    # multi-year checkpointed collection bounded and recoverable.
    os.environ["KRX_OPENAPI_KEEPALIVE"] = "0"
    os.environ["KRX_ID"] = ""
    os.environ["KRX_PW"] = ""
    os.environ["KRX_OPENAPI_READ_TIMEOUT_SECONDS"] = str(
        max(1.0, args.read_timeout_seconds)
    )
    os.environ["KRX_OPENAPI_HARD_TIMEOUT_SECONDS"] = str(
        max(5.0, args.read_timeout_seconds + 5.0)
    )

    stop_monitor = threading.Event()

    def monitor() -> None:
        previous = -1
        while not stop_monitor.wait(5.0):
            completed = len(tuple((checkpoint_root / "prices").glob("*.parquet")))
            if completed != previous:
                print(
                    json.dumps(
                        {
                            "stage": "collect_openapi",
                            "completed_sessions": completed,
                            "total_sessions": len(sessions),
                        }
                    ),
                    flush=True,
                )
                previous = completed

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        prior_completed = len(
            tuple((checkpoint_root / "prices").glob("*.parquet"))
        )
        stalled_restarts = 0
        openapi_observations = pd.DataFrame()
        for attempt in range(args.max_collection_restarts + 1):
            try:
                openapi_observations = _collect_krx_prices(
                    sessions,
                    collection_symbols,
                    catalog,
                    KrCheckpointStore(checkpoint_root),
                    sleep_seconds=max(0.0, args.request_delay_seconds),
                    workers=args.workers,
                )
                break
            except RuntimeError as exc:
                completed = len(
                    tuple((checkpoint_root / "prices").glob("*.parquet"))
                )
                progressed = completed > prior_completed
                rate_limited = "429" in str(exc)
                if progressed or rate_limited:
                    stalled_restarts = 0
                else:
                    stalled_restarts += 1
                retry_delay = (
                    max(1.0, args.rate_limit_cooldown_seconds)
                    if rate_limited
                    else min(10.0, float(2 ** min(stalled_restarts, 3)))
                )
                print(
                    json.dumps(
                        {
                            "stage": "collect_openapi_retry",
                            "attempt": attempt + 1,
                            "completed_sessions": completed,
                            "progressed": progressed,
                            "rate_limited": rate_limited,
                            "retry_delay_seconds": retry_delay,
                            "error": str(exc),
                        }
                    ),
                    flush=True,
                )
                if (
                    attempt >= args.max_collection_restarts
                    or stalled_restarts >= 3
                ):
                    raise
                prior_completed = completed
                time.sleep(retry_delay)
        else:
            raise RuntimeError("KRX Open API collection retry budget exhausted.")
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=10.0)
    observed_sources = set(
        openapi_observations["source"].dropna().astype(str)
    )
    if observed_sources != {OPENAPI_SOURCE}:
        raise RuntimeError(
            "Strict Open API comparison received unexpected source(s): "
            + ", ".join(sorted(observed_sources))
        )
    print(
        json.dumps(
            {
                "stage": "openapi_complete",
                "rows": len(openapi_observations),
                "sessions": openapi_observations["session"].nunique(),
                "symbols": openapi_observations["symbol"].nunique(),
            }
        ),
        flush=True,
    )

    web_observations, web_status_reclassifications = _load_checkpoint_prices(
        reference_root,
        sessions,
        catalog,
    )
    data_comparison = {
        "openapi_vs_authenticated_web": _compare_observations(
            openapi_observations,
            web_observations,
            left_name="krx_openapi",
            right_name="krx_authenticated_web",
        )
    }
    openapi_traded = _traded_krx_prices(openapi_observations)
    canonical_prices = _load_canonical_prices(
        canonical_repository,
        version=canonical_release.dataset_versions["daily_price_raw"],
        start=start_session,
        end=end_session,
        security_ids=set(openapi_observations["security_id"].astype(str)),
    )
    data_comparison["openapi_vs_canonical_parquet"] = _compare_observations(
        openapi_traded,
        canonical_prices,
        left_name="krx_openapi",
        right_name="canonical_parquet",
    )

    output_root = (
        Path(args.output_root)
        / f"{start_session}_{end_session}_{len(sessions)}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="stq-krx-openapi-",
    ) as temporary:
        temporary_repository = Path(temporary) / "repository"
        temporary_repository.mkdir()
        comparison_release = _build_openapi_repository(
            canonical_root,
            temporary_repository,
            openapi_traded,
            completed_session=completed_session,
        )
        backtests: dict[str, Any] = {}
        package_root = Path(__file__).resolve().parents[1]
        data_path = package_root / "configs" / "data.yaml"
        period_label = "full" if args.period == "max" else args.period
        for profile in PROFILES:
            strategy_path = (
                package_root
                / "configs"
                / "strategies"
                / f"single_supertrend_{profile}.yaml"
            )
            runtime_path = (
                package_root
                / "configs"
                / "runtimes"
                / f"research_{profile}.yaml"
            )
            canonical_metrics, canonical_run_dir = _run_backtest(
                strategy_path=strategy_path,
                runtime_path=runtime_path,
                data_path=data_path,
                repository_root=canonical_root,
                period=args.period,
                run_id=(
                    f"parquet_{period_label}_{end_session.replace('-', '')}"
                    f"_single_supertrend_{profile}"
                ),
                completed_session=completed_session,
            )
            openapi_metrics, openapi_run_dir = _run_backtest(
                strategy_path=strategy_path,
                runtime_path=runtime_path,
                data_path=data_path,
                repository_root=temporary_repository,
                period=args.period,
                run_id=(
                    f"krx_openapi_{period_label}_{end_session.replace('-', '')}"
                    f"_single_supertrend_{profile}"
                ),
                completed_session=completed_session,
            )
            backtests[profile] = {
                "canonical_run_dir": canonical_run_dir,
                "openapi_run_dir": openapi_run_dir,
                "metrics": _metric_comparison(
                    canonical_metrics,
                    openapi_metrics,
                ),
            }
            print(
                json.dumps(
                    {
                        "stage": "backtest_complete",
                        "profile": profile,
                        "canonical_total_return": canonical_metrics[
                            "total_return"
                        ],
                        "openapi_total_return": openapi_metrics[
                            "total_return"
                        ],
                    }
                ),
                flush=True,
            )

    report = {
        "schema_version": 1,
        "market": "KR",
        "profiles": list(PROFILES),
        "sessions": {
            "start": start_session,
            "end": end_session,
            "count": len(sessions),
        },
        "source_contract": {
            "openapi_source": OPENAPI_SOURCE,
            "web_fallback_disabled": True,
            "checkpoint_root": str(checkpoint_root),
            "reference_checkpoint_root": str(reference_root),
            "canonical_release": canonical_release.version,
            "comparison_release": comparison_release,
        },
        "scope": {
            "reference_symbol_count": len(reference_symbols),
            "collection_symbol_count": len(collection_symbols),
            "openapi_observation_rows": len(openapi_observations),
            "openapi_traded_rows": len(openapi_traded),
            "web_checkpoint_delisting_status_reclassifications": (
                web_status_reclassifications
            ),
        },
        "data_comparison": data_comparison,
        "backtests": backtests,
    }
    report_path = output_root / "comparison.json"
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                "stage": "complete",
                "report_path": str(report_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
