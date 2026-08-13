from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PLAYGROUND_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PLAYGROUND_ROOT.parent
UNIFIED_ROOT = PROJECT_ROOT / "unified_quant"
sys.path.insert(0, str(UNIFIED_ROOT / "src"))
sys.path.insert(0, str(PLAYGROUND_ROOT))

from research_extensions.cold_start_overlay import (  # noqa: E402
    ColdStartPreparedBacktest,
    ColdStartRule,
)
from research_extensions.experimental_leader_rotation import (  # noqa: E402
    ExperimentalLeaderPolicy,
    ExperimentalSignalCache,
    FastExperimentalPreparedLeaderBacktest,
)
from research_extensions.kospi_market_filters import (  # noqa: E402
    build_filter_variant,
)
from scripts.nested_walk_forward_nasdaq_structure import (  # noqa: E402
    Candidate,
    base_config,
    benchmark_return_for_index,
    config_for_candidate,
)
from supertrend_quant.data import market_index  # noqa: E402
from supertrend_quant.research.data_resolver import download_for_config  # noqa: E402
import supertrend_quant.runners as canonical_runners  # noqa: E402
from supertrend_quant.runners import (  # noqa: E402
    _prepare_backtest,
    run_backtest_on_data,
)
from supertrend_quant.strategies import create_strategy  # noqa: E402
from supertrend_quant.strategies.leader_rotation import (  # noqa: E402
    PreparedLeaderBacktest,
)


DEFAULT_CONFIG = (
    PLAYGROUND_ROOT
    / "configs"
    / "nasdaq_three_account_cold_start_comparison.json"
)
DEFAULT_RESULTS = PLAYGROUND_ROOT / "results" / "nasdaq_cold_start_comparison"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Compare five deployment cold-start rules for Nasdaq paper "
            "accounts A/B/C using full-history and staggered 12-month runs."
        )
    )
    value.add_argument("--config", default=str(DEFAULT_CONFIG))
    value.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    value.add_argument("--run-id", default="")
    value.add_argument("--no-resume", action="store_true")
    value.add_argument("--full-only", action="store_true")
    return value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_from_account(raw: dict[str, Any]) -> Candidate:
    return Candidate(
        candidate_id=str(raw["account_id"]),
        policy=ExperimentalLeaderPolicy(
            rotation_profit_gate=str(raw["rotation_profit_gate"]),
            stop_loss_pct=(
                None
                if raw.get("stop_loss_pct") is None
                else float(raw["stop_loss_pct"])
            ),
            late_chase_mode=str(raw["late_chase_mode"]),
            max_extension_atr=(
                None
                if raw.get("max_extension_atr") is None
                else float(raw["max_extension_atr"])
            ),
        ),
    )


def guarded_policy(
    normal: ExperimentalLeaderPolicy,
    mode: dict[str, Any],
) -> ExperimentalLeaderPolicy | None:
    raw = mode.get("guard_policy")
    if raw is None:
        return None
    return replace(
        normal,
        late_chase_mode=str(raw["late_chase_mode"]),
        max_extension_atr=(
            None
            if raw.get("max_extension_atr") is None
            else float(raw["max_extension_atr"])
        ),
    )


def build_backtest(
    *,
    canonical_prepared: PreparedLeaderBacktest,
    market_filter_trends,
    config,
    candidate: Candidate,
    mode: dict[str, Any],
    signal_cache: ExperimentalSignalCache,
):
    prepared = PreparedLeaderBacktest(
        create_strategy(config),
        canonical_prepared.prepared,
        market_filter_trends,
        canonical_prepared.universe_schedule,
    )
    normal = FastExperimentalPreparedLeaderBacktest(
        prepared,
        candidate.policy,
        signal_cache,
    )
    mode_id = str(mode["mode_id"])
    if mode_id == "immediate":
        return normal

    rule = ColdStartRule(mode_id, str(mode["name"]))
    guard = guarded_policy(candidate.policy, mode)
    if guard is None:
        return ColdStartPreparedBacktest(normal, rule)
    guarded = FastExperimentalPreparedLeaderBacktest(
        prepared,
        guard,
        signal_cache,
    )
    return ColdStartPreparedBacktest(
        normal,
        rule,
        guarded_delegate=guarded,
    )


def evaluate(
    *,
    data,
    run_index: pd.Index,
    backtest,
    config,
):
    with patch.object(
        canonical_runners,
        "_prepare_backtest",
        return_value=backtest,
    ):
        return run_backtest_on_data(config, data, run_index=run_index)


def save_full_result(root: Path, key: str, result) -> None:
    root.mkdir(parents=True, exist_ok=True)
    result.equity.rename("equity").to_frame().to_csv(
        root / f"{key}_equity.csv",
        encoding="utf-8-sig",
    )
    payload = {
        "metrics": {
            metric: finite(value) for metric, value in result.metrics.items()
        },
        "trades": [float(value) for value in result.trades],
        "trade_records": [dict(value) for value in result.trade_records],
    }
    (root / f"{key}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def load_full_result(root: Path, key: str) -> dict[str, Any] | None:
    json_path = root / f"{key}.json"
    equity_path = root / f"{key}_equity.csv"
    if not json_path.exists() or not equity_path.exists():
        return None
    payload = load_json(json_path)
    frame = pd.read_csv(equity_path, index_col=0, parse_dates=True)
    payload["equity"] = frame.iloc[:, 0].astype(float)
    return payload


def first_trade_fields(
    records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    run_index: pd.Index,
) -> dict[str, Any]:
    if not records:
        return {
            "first_trade_symbol": "",
            "first_entry_time": "",
            "first_exit_time": "",
            "first_trade_return": None,
            "first_trade_loss": None,
            "first_trade_severe_loss": None,
            "entry_delay_sessions": None,
        }
    ordered = sorted(
        (dict(record) for record in records),
        key=lambda record: str(record.get("entry_time", "")),
    )
    first = ordered[0]
    entry = pd.to_datetime(first.get("entry_time"), errors="coerce")
    entry_delay = None
    if not pd.isna(entry):
        entry_delay = max(
            0,
            int(pd.Index(run_index).searchsorted(entry, side="left")),
        )
    pnl = finite_or_none(first.get("pnl_pct"))
    return {
        "first_trade_symbol": str(first.get("symbol", "")),
        "first_entry_time": str(first.get("entry_time", "")),
        "first_exit_time": str(first.get("exit_time", "")),
        "first_trade_return": pnl,
        "first_trade_loss": None if pnl is None else pnl < 0.0,
        "first_trade_severe_loss": None if pnl is None else pnl <= -0.10,
        "entry_delay_sessions": entry_delay,
    }


def append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["evaluation_key"])] = row
    return rows


def launch_windows(
    full_index: pd.Index,
    raw: dict[str, Any],
) -> list[tuple[str, pd.Index]]:
    spec = raw["multi_start"]
    launches = pd.date_range(
        str(spec["first_launch"]),
        str(spec["last_launch"]),
        freq=str(spec["frequency"]),
    )
    horizon = int(spec["horizon_sessions"])
    end_limit = pd.Timestamp(str(raw["end"]))
    windows: list[tuple[str, pd.Index]] = []
    dates = pd.DatetimeIndex(full_index)
    for requested in launches:
        start_position = int(dates.searchsorted(requested, side="left"))
        end_position = start_position + horizon
        if end_position > len(dates):
            continue
        run_index = dates[start_position:end_position]
        if len(run_index) != horizon or run_index[-1] > end_limit:
            continue
        windows.append((requested.date().isoformat(), run_index))
    return windows


def window_market_data(
    data,
    run_index: pd.Index,
    full_index: pd.Index,
    warmup_sessions: int,
):
    """Slice a launch window while retaining indicator warm-up history.

    The canonical runner determines whether a requested segment has completed
    its warm-up from the market-data timeline.  Starting the slice exactly at
    ``run_index[0]`` therefore postpones trading by ``strategy.warmup_bars()``
    even though the shared signal cache was built from the full history.
    """

    dates = pd.DatetimeIndex(full_index)
    run_start_position = int(dates.searchsorted(run_index[0], side="left"))
    history_start_position = max(0, run_start_position - warmup_sessions)
    start = pd.Timestamp(dates[history_start_position])
    end = pd.Timestamp(run_index[-1])

    def sliced(frames):
        if frames is None:
            return None
        output = {}
        for symbol, frame in frames.items():
            selected = frame.loc[
                (frame.index >= start) & (frame.index <= end)
            ]
            if not selected.empty:
                output[symbol] = selected
        return output

    return replace(
        data,
        bars=sliced(data.bars) or {},
        execution_bars=sliced(data.execution_bars),
        completed_session=str(end.date()),
    )


def add_paired_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    immediate = (
        output.loc[output["mode_id"].eq("immediate")]
        .set_index(["account_id", "launch_id"])
    )
    return_deltas: list[float] = []
    mdd_deltas: list[float] = []
    for row in output.itertuples(index=False):
        baseline = immediate.loc[(row.account_id, row.launch_id)]
        return_deltas.append(
            float(row.total_return) - float(baseline["total_return"])
        )
        mdd_deltas.append(float(row.mdd) - float(baseline["mdd"]))
    output["return_delta_vs_immediate"] = return_deltas
    output["mdd_delta_vs_immediate"] = mdd_deltas
    return output


def aggregate_multistart(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (account_id, account_name, mode_id, mode_name), group in frame.groupby(
        ["account_id", "account_name", "mode_id", "mode_name"],
        sort=True,
    ):
        first_trades = pd.to_numeric(
            group["first_trade_return"], errors="coerce"
        ).dropna()
        entry_delays = pd.to_numeric(
            group["entry_delay_sessions"], errors="coerce"
        ).dropna()
        return_delta = pd.to_numeric(
            group["return_delta_vs_immediate"], errors="coerce"
        )
        mdd_delta = pd.to_numeric(
            group["mdd_delta_vs_immediate"], errors="coerce"
        )
        records.append(
            {
                "account_id": account_id,
                "account_name": account_name,
                "mode_id": mode_id,
                "mode_name": mode_name,
                "launch_count": int(len(group)),
                "mean_return": float(group["total_return"].mean()),
                "median_return": float(group["total_return"].median()),
                "positive_return_ratio": float(group["total_return"].gt(0).mean()),
                "median_mdd": float(group["mdd"].median()),
                "worst_mdd": float(group["mdd"].min()),
                "median_calmar": float(group["calmar"].median()),
                "median_trade_count": float(group["trade_count"].median()),
                "no_trade_ratio": float(group["trade_count"].eq(0).mean()),
                "first_trade_count": int(len(first_trades)),
                "median_first_trade_return": (
                    float(first_trades.median()) if len(first_trades) else 0.0
                ),
                "first_trade_loss_ratio": (
                    float(first_trades.lt(0).mean()) if len(first_trades) else 0.0
                ),
                "first_trade_severe_loss_ratio": (
                    float(first_trades.le(-0.10).mean())
                    if len(first_trades)
                    else 0.0
                ),
                "median_entry_delay_sessions": (
                    float(entry_delays.median()) if len(entry_delays) else 0.0
                ),
                "median_return_delta_vs_immediate": float(return_delta.median()),
                "mean_return_delta_vs_immediate": float(return_delta.mean()),
                "win_ratio_vs_immediate": float(return_delta.gt(0).mean()),
                "median_mdd_delta_vs_immediate": float(mdd_delta.median()),
                "mdd_improvement_ratio_vs_immediate": float(mdd_delta.gt(0).mean()),
            }
        )
    return pd.DataFrame(records).sort_values(["account_id", "mode_id"])


def main() -> None:
    args = parser().parse_args()
    raw = load_json(Path(args.config).resolve())
    accounts = list(raw["accounts"])
    modes = list(raw["cold_start_modes"])
    variants = {
        str(item["variant_id"]): item
        for item in raw["market_filter_variants"]
    }
    run_id = args.run_id or (
        f"{raw['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(args.results_dir).resolve() / run_id
    full_dir = run_dir / "full_period_evaluations"
    checkpoint_path = run_dir / "multistart_checkpoint.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.no_resume and checkpoint_path.exists():
        checkpoint_path.unlink()

    started = time.monotonic()
    base = base_config(raw)
    print("[cold-start] loading canonical Nasdaq data...", flush=True)
    data = download_for_config(base, allow_stale=True)
    full_index = pd.DatetimeIndex(market_index(data))
    full_run_index = full_index[
        (full_index >= pd.Timestamp(str(raw["start"])))
        & (full_index <= pd.Timestamp(str(raw["end"])))
    ]
    print("[cold-start] preparing canonical indicators...", flush=True)
    canonical_prepared = _prepare_backtest(create_strategy(base), data)
    if not isinstance(canonical_prepared, PreparedLeaderBacktest):
        raise TypeError("Expected canonical PreparedLeaderBacktest.")

    contexts: dict[str, dict[str, Any]] = {}
    used_variant_ids = list(
        dict.fromkeys(str(account["market_filter_variant"]) for account in accounts)
    )
    for position, variant_id in enumerate(used_variant_ids, start=1):
        print(
            f"[cold-start] filter {position}/{len(used_variant_ids)} "
            f"{variant_id}: building regime and signal cache",
            flush=True,
        )
        variant = build_filter_variant(
            variants[variant_id],
            base_config=base,
            data=data,
            canonical_prepared=canonical_prepared,
            full_index=full_index,
        )
        shared = PreparedLeaderBacktest(
            create_strategy(variant.config),
            canonical_prepared.prepared,
            variant.market_filter_trends,
            canonical_prepared.universe_schedule,
        )
        contexts[variant_id] = {
            "variant": variant,
            "signal_cache": ExperimentalSignalCache(shared, full_index),
        }

    windows = [] if args.full_only else launch_windows(full_index, raw)
    warmup_sessions = max(
        int(raw["multi_start"].get("warmup_sessions", 252)),
        int(create_strategy(base).warmup_bars()),
    )
    total = len(accounts) * len(modes) * (1 + len(windows))
    completed = 0
    full_rows: list[dict[str, Any]] = []
    full_trade_rows: list[dict[str, Any]] = []

    for account in accounts:
        candidate = candidate_from_account(account)
        variant_id = str(account["market_filter_variant"])
        context = contexts[variant_id]
        variant = context["variant"]
        config = config_for_candidate(variant.config, candidate)
        for mode in modes:
            mode_id = str(mode["mode_id"])
            key = f"{candidate.candidate_id}__{mode_id}"
            payload = None if args.no_resume else load_full_result(full_dir, key)
            cached = payload is not None
            if payload is None:
                backtest = build_backtest(
                    canonical_prepared=canonical_prepared,
                    market_filter_trends=variant.market_filter_trends,
                    config=config,
                    candidate=candidate,
                    mode=mode,
                    signal_cache=context["signal_cache"],
                )
                result = evaluate(
                    data=data,
                    run_index=full_run_index,
                    backtest=backtest,
                    config=config,
                )
                save_full_result(full_dir, key, result)
                payload = {
                    "metrics": result.metrics,
                    "trade_records": result.trade_records,
                    "equity": result.equity,
                }
            metrics = {
                metric: finite(value)
                for metric, value in payload["metrics"].items()
            }
            common = {
                "account_id": candidate.candidate_id,
                "account_name": str(account["name"]),
                "market_filter_variant": variant_id,
                "mode_id": mode_id,
                "mode_name": str(mode["name"]),
                **asdict(candidate.policy),
            }
            full_rows.append(
                {
                    **common,
                    **metrics,
                    **first_trade_fields(
                        payload["trade_records"], full_run_index
                    ),
                    "benchmark_return": benchmark_return_for_index(
                        data, payload["equity"].index
                    ),
                }
            )
            full_trade_rows.extend(
                {**common, **dict(record)}
                for record in payload["trade_records"]
            )
            completed += 1
            print(
                f"[cold-start] full {completed}/{total} "
                f"account={candidate.candidate_id} mode={mode_id} "
                f"return={metrics['total_return']:+.2%} "
                f"mdd={metrics['mdd']:.2%} "
                f"source={'cache' if cached else 'run'} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    full_frame = pd.DataFrame(full_rows).sort_values(["account_id", "mode_id"])
    full_frame.to_csv(
        run_dir / "full_period_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(full_trade_rows).to_csv(
        run_dir / "full_period_trade_records.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if args.full_only:
        print(
            f"[cold-start] complete full-only elapsed="
            f"{time.monotonic() - started:.1f}s results={run_dir}",
            flush=True,
        )
        return

    checkpoint = load_checkpoint(checkpoint_path)
    for launch_id, run_index in windows:
        launch_data = window_market_data(
            data,
            run_index,
            full_index,
            warmup_sessions,
        )
        for account in accounts:
            candidate = candidate_from_account(account)
            variant_id = str(account["market_filter_variant"])
            context = contexts[variant_id]
            variant = context["variant"]
            config = config_for_candidate(variant.config, candidate)
            for mode in modes:
                mode_id = str(mode["mode_id"])
                evaluation_key = (
                    f"{candidate.candidate_id}__{mode_id}__{launch_id}"
                )
                if evaluation_key not in checkpoint:
                    backtest = build_backtest(
                        canonical_prepared=canonical_prepared,
                        market_filter_trends=variant.market_filter_trends,
                        config=config,
                        candidate=candidate,
                        mode=mode,
                        signal_cache=context["signal_cache"],
                    )
                    result = evaluate(
                        data=launch_data,
                        run_index=run_index,
                        backtest=backtest,
                        config=config,
                    )
                    metrics = {
                        metric: finite(value)
                        for metric, value in result.metrics.items()
                    }
                    row = {
                        "evaluation_key": evaluation_key,
                        "account_id": candidate.candidate_id,
                        "account_name": str(account["name"]),
                        "market_filter_variant": variant_id,
                        "mode_id": mode_id,
                        "mode_name": str(mode["name"]),
                        "launch_id": launch_id,
                        "start_session": str(pd.Timestamp(run_index[0]).date()),
                        "end_session": str(pd.Timestamp(run_index[-1]).date()),
                        **metrics,
                        **first_trade_fields(result.trade_records, run_index),
                    }
                    append_checkpoint(checkpoint_path, row)
                    checkpoint[evaluation_key] = row
                    source = "run"
                else:
                    row = checkpoint[evaluation_key]
                    source = "cache"
                completed += 1
                if completed % 5 == 0 or completed == total:
                    print(
                        f"[cold-start] progress {completed}/{total} "
                        f"account={candidate.candidate_id} mode={mode_id} "
                        f"launch={launch_id} "
                        f"return={float(row['total_return']):+.2%} "
                        f"source={source} "
                        f"elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )

    relevant_keys = {
        f"{account['account_id']}__{mode['mode_id']}__{launch_id}"
        for account in accounts
        for mode in modes
        for launch_id, _ in windows
    }
    multi = pd.DataFrame(
        checkpoint[key] for key in sorted(relevant_keys)
    )
    multi = add_paired_deltas(multi)
    multi.to_csv(
        run_dir / "multistart_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate_multistart(multi).to_csv(
        run_dir / "multistart_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"[cold-start] complete evaluations={total} "
        f"elapsed={time.monotonic() - started:.1f}s results={run_dir}",
        flush=True,
    )


def finite(value: Any) -> float:
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


def finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


if __name__ == "__main__":
    main()
