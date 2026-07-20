#!/usr/bin/env python3
"""Compare the current release with the planned Kraft special-dividend repair.

This audit is deliberately read-only.  It loads the pinned local Parquet
release twice in memory, overlays only the candidate KRFT adjustment factors
and the canonical $16.50 action, and runs the canonical backtest path.  It has
no network, EODHD, R2, result-persistence, or release-commit code path.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

import repair_us_kraft_special_dividend as repair
from supertrend_quant.config import load_split_config
from supertrend_quant.market_store.provider import ParquetMarketDataProvider
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.runners import (
    _configured_completed_session,
    _schedule_for_period,
    run_backtest_on_data,
)
from supertrend_quant.universe import resolve_universe


DEFAULT_STRATEGY = Path(
    "unified_quant/configs/strategies/triple_supertrend_alone.yaml"
)
DEFAULT_DATA = Path("unified_quant/configs/data.yaml")
DEFAULT_RUNTIMES = (
    Path("unified_quant/configs/runtimes/research_us.yaml"),
    Path("unified_quant/configs/runtimes/research_us_nasdaq100_rolling.yaml"),
)
PRICE_COLUMNS = ("Open", "High", "Low", "Close")


def _json_scalar(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _json_record(record: dict[str, object]) -> dict[str, object]:
    return {key: _json_scalar(value) for key, value in record.items()}


def _candidate_factor_ratios(
    repository: LocalDatasetRepository,
    prepared: repair.PreparedRepair,
) -> pd.Series:
    current = repository.read_frame(
        "adjustment_factors",
        prepared.release.dataset_versions["adjustment_factors"],
    )
    candidate = prepared.frames.get("adjustment_factors")
    if candidate is None:
        return pd.Series(dtype=float)
    columns = ["security_id", "session", "total_return_factor"]
    current = current.loc[
        current["security_id"].astype(str).eq(repair.KRFT_ID), columns
    ].copy()
    candidate = candidate.loc[
        candidate["security_id"].astype(str).eq(repair.KRFT_ID), columns
    ].copy()
    joined = current.merge(
        candidate,
        on=["security_id", "session"],
        how="outer",
        suffixes=("_current", "_candidate"),
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise RuntimeError("Candidate KRFT factor sessions differ from the release.")
    denominator = pd.to_numeric(joined["total_return_factor_current"])
    numerator = pd.to_numeric(joined["total_return_factor_candidate"])
    if denominator.le(0).any() or numerator.le(0).any():
        raise RuntimeError("KRFT factors must be finite and positive.")
    ratios = pd.Series(
        numerator.to_numpy() / denominator.to_numpy(),
        index=pd.to_datetime(joined["session"]).dt.normalize(),
        dtype=float,
    ).sort_index()
    if ratios.index.has_duplicates or not ratios.map(math.isfinite).all():
        raise RuntimeError("Candidate KRFT factor ratios are invalid.")
    return ratios


def _candidate_market_data(market_data, prepared, ratios: pd.Series):
    if prepared.summary["status"] == "already_applied":
        return market_data
    bars = {symbol: frame.copy() for symbol, frame in market_data.bars.items()}
    if repair.KRFT_SYMBOL in bars:
        frame = bars[repair.KRFT_SYMBOL]
        session_index = pd.DatetimeIndex(frame.index).normalize()
        aligned = ratios.reindex(session_index)
        if aligned.isna().any():
            missing = session_index[aligned.isna()].strftime("%Y-%m-%d").tolist()
            raise RuntimeError("Missing candidate KRFT factors: " + ", ".join(missing))
        for column in PRICE_COLUMNS:
            frame[column] = pd.to_numeric(frame[column]).to_numpy() * aligned.to_numpy()
        bars[repair.KRFT_SYMBOL] = frame

    actions = list(market_data.corporate_actions)
    if repair.KRFT_SYMBOL in bars:
        matches = prepared.frames["corporate_actions"].loc[
            lambda frame: frame["event_id"].astype(str).eq(
                repair.SPECIAL_DIVIDEND_EVENT_ID
            )
        ]
        if len(matches) != 1:
            raise RuntimeError("Candidate must contain one exact Kraft dividend.")
        action = matches.iloc[0].to_dict()
        action["symbol"] = repair.KRFT_SYMBOL
        actions.append(action)

    return replace(
        market_data,
        bars=bars,
        corporate_actions=tuple(actions),
        data_version=market_data.data_version + ";in_memory=kraft-special-dividend",
    )


def _metric_delta(baseline, candidate) -> dict[str, object]:
    output: dict[str, object] = {}
    for key in sorted(set(baseline.metrics) | set(candidate.metrics)):
        before = baseline.metrics.get(key)
        after = candidate.metrics.get(key)
        try:
            delta: object = float(after) - float(before)
        except (TypeError, ValueError):
            delta = None if before == after else {"baseline": before, "candidate": after}
        output[key] = _json_scalar(delta)
    return output


def _compare_results(baseline, candidate) -> dict[str, object]:
    baseline_equity, candidate_equity = baseline.equity.align(
        candidate.equity, join="outer"
    )
    if baseline_equity.isna().any() or candidate_equity.isna().any():
        raise RuntimeError("Baseline and candidate equity sessions differ.")
    equity_delta = candidate_equity - baseline_equity
    changed = equity_delta.abs().gt(1e-9)
    baseline_trades = [_json_record(row) for row in baseline.trade_records]
    candidate_trades = [_json_record(row) for row in candidate.trade_records]
    return {
        "baseline": {
            "ending_equity": float(baseline.equity.iloc[-1]),
            "metrics": baseline.metrics,
            "trade_count": len(baseline.trade_records),
            "corporate_action_cash": baseline.corporate_action_cash,
            "processed_corporate_action_ids": list(
                baseline.processed_corporate_action_ids
            ),
        },
        "candidate": {
            "ending_equity": float(candidate.equity.iloc[-1]),
            "metrics": candidate.metrics,
            "trade_count": len(candidate.trade_records),
            "corporate_action_cash": candidate.corporate_action_cash,
            "processed_corporate_action_ids": list(
                candidate.processed_corporate_action_ids
            ),
        },
        "delta": {
            "ending_equity": float(candidate.equity.iloc[-1] - baseline.equity.iloc[-1]),
            "ending_equity_pct_of_baseline": float(
                candidate.equity.iloc[-1] / baseline.equity.iloc[-1] - 1.0
            ),
            "metrics": _metric_delta(baseline, candidate),
            "max_absolute_equity": float(equity_delta.abs().max()),
            "changed_equity_sessions": int(changed.sum()),
            "first_changed_equity_session": (
                equity_delta.index[changed][0].isoformat() if changed.any() else None
            ),
            "last_changed_equity_session": (
                equity_delta.index[changed][-1].isoformat() if changed.any() else None
            ),
            "trade_records_exactly_equal": baseline_trades == candidate_trades,
            "trade_count": len(candidate_trades) - len(baseline_trades),
            "corporate_action_cash": (
                candidate.corporate_action_cash - baseline.corporate_action_cash
            ),
            "special_dividend_processed": (
                repair.SPECIAL_DIVIDEND_EVENT_ID
                in candidate.processed_corporate_action_ids
            ),
        },
        "relevant_trades": {
            "baseline": [
                row for row in baseline_trades if row.get("symbol") in {"KRFT", "KHC"}
            ],
            "candidate": [
                row for row in candidate_trades if row.get("symbol") in {"KRFT", "KHC"}
            ],
        },
    }


def _audit_runtime(
    strategy_path: Path,
    runtime_path: Path,
    data_path: Path,
    repository: LocalDatasetRepository,
    prepared: repair.PreparedRepair,
    ratios: pd.Series,
) -> dict[str, object]:
    config = replace(
        load_split_config(strategy_path, runtime_path, data_path), period="max"
    )
    if config.data_store.price_mode != "total_return_adjusted":
        raise RuntimeError("Impact audit requires total_return_adjusted signal prices.")
    resolved = resolve_universe(config, mode="backtest")
    schedule = _schedule_for_period(
        resolved.schedule,
        config.period,
        _configured_completed_session(config),
    )
    symbols = list(
        dict.fromkeys(
            member.symbol for entry in schedule for member in entry.members
        )
    )
    if not symbols:
        symbols = list(resolved.eligible_symbols)
    print(
        json.dumps(
            {
                "stage": "loading_local_parquet",
                "runtime": str(runtime_path),
                "requested_symbols": len(symbols),
            }
        ),
        flush=True,
    )
    market_data = ParquetMarketDataProvider(config.data_store.local_cache_dir).load(
        config,
        symbols,
        universe_schedule=tuple(entry.to_dict() for entry in schedule),
    )
    loaded_versions = set(market_data.data_version.split(";"))
    if "release=" + prepared.release.version not in loaded_versions:
        raise RuntimeError("Market data did not load the repair's base release.")
    market_data = replace(
        market_data,
        universe_snapshot=resolved.snapshot.to_dict(),
        universe_schedule=tuple(entry.to_dict() for entry in schedule),
    )
    candidate_data = _candidate_market_data(market_data, prepared, ratios)
    print(
        json.dumps(
            {
                "stage": "running_baseline_backtest",
                "runtime": str(runtime_path),
                "loaded_symbols": len(market_data.bars),
            }
        ),
        flush=True,
    )
    baseline = run_backtest_on_data(config, market_data)
    print(
        json.dumps(
            {
                "stage": "running_candidate_backtest",
                "runtime": str(runtime_path),
            }
        ),
        flush=True,
    )
    candidate = run_backtest_on_data(config, candidate_data)
    comparison = _compare_results(baseline, candidate)
    comparison["runtime"] = str(runtime_path)
    comparison["universe_profiles"] = list(config.universe.profiles.get("US", ()))
    comparison["requested_symbols"] = len(symbols)
    comparison["loaded_symbols"] = len(market_data.bars)
    comparison["contains_krft"] = repair.KRFT_SYMBOL in market_data.bars
    comparison["contains_khc"] = repair.KHC_SYMBOL in market_data.bars
    comparison["data_quality"] = market_data.data_quality
    comparison["data_warnings"] = list(market_data.warnings)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--runtime", type=Path, action="append", dest="runtimes"
    )
    parser.add_argument("--cache-root", type=Path, default=repair.DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--evidence-dir", type=Path, default=repair.DEFAULT_EVIDENCE_DIR
    )
    args = parser.parse_args()

    repository = LocalDatasetRepository(args.cache_root)
    prepared = repair.prepare_repair(
        repository, evidence_dir=args.evidence_dir
    )
    if prepared.summary["status"] not in {
        "validated_offline_plan",
        "already_applied",
    }:
        raise RuntimeError("Kraft repair did not produce a validated plan.")
    ratios = _candidate_factor_ratios(repository, prepared)
    changed_ratios = int(ratios.sub(1.0).abs().gt(1e-12).sum())
    if (
        prepared.summary["status"] == "validated_offline_plan"
        and changed_ratios != repair.EXPECTED_FACTOR_VALUE_CHANGES
    ):
        raise RuntimeError(
            "Unexpected in-memory KRFT factor impact: "
            f"expected={repair.EXPECTED_FACTOR_VALUE_CHANGES}; observed={changed_ratios}."
        )

    runtimes = tuple(args.runtimes or DEFAULT_RUNTIMES)
    output = {
        "status": "validated_read_only_backtest_impact",
        "release_version": prepared.release.version,
        "completed_session": prepared.release.completed_session,
        "strategy": str(args.strategy),
        "period": "max",
        "special_dividend_event_id": repair.SPECIAL_DIVIDEND_EVENT_ID,
        "factor_value_changes": changed_ratios,
        "network_accessed": False,
        "eodhd_calls": 0,
        "r2_accessed": False,
        "release_written": False,
        "comparisons": [],
    }
    for runtime in runtimes:
        print(json.dumps({"stage": "running", "runtime": str(runtime)}), flush=True)
        output["comparisons"].append(
            _audit_runtime(
                args.strategy,
                runtime,
                args.data,
                repository,
                prepared,
                ratios,
            )
        )
    current_release, _ = repository.current_release()
    if current_release is None or current_release.version != prepared.release.version:
        raise RuntimeError("Current release changed during the read-only audit.")
    print(json.dumps(output, indent=2, default=_json_scalar))


if __name__ == "__main__":
    main()
