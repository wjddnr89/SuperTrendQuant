from __future__ import annotations

import argparse
import json
import sys
from bisect import bisect_right
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAYGROUND_ROOT = PROJECT_ROOT / "playground"
UNIFIED_ROOT = PROJECT_ROOT / "unified_quant"
sys.path.insert(0, str(UNIFIED_ROOT / "src"))

from supertrend_quant.config import load_split_config  # noqa: E402
from supertrend_quant.data import MarketData, market_index  # noqa: E402
from supertrend_quant.ranking import rank_scores, register_scorer  # noqa: E402
from supertrend_quant.research.data_resolver import download_for_config  # noqa: E402
from supertrend_quant.research.overlays import apply_config_overlay  # noqa: E402
from supertrend_quant.runners import BacktestResult, run_backtest_on_data  # noqa: E402
from supertrend_quant.strategies.common import active_universe_symbols  # noqa: E402
from supertrend_quant.universe import resolve_universe  # noqa: E402


DEFAULT_STRATEGY = PLAYGROUND_ROOT / "configs" / "strategies" / "leader_rotation.yaml"
DEFAULT_RUNTIME = PLAYGROUND_ROOT / "configs" / "runtimes" / "research_us_nasdaq100_rolling.yaml"
DEFAULT_WIKIPEDIA_HISTORY = PLAYGROUND_ROOT / "data" / "universes" / "nasdaq100_quarterly_history.json"
DEFAULT_RESULTS = PLAYGROUND_ROOT / "results" / "research" / "us_nasdaq100_rolling" / "universe_ab"


@register_scorer
class DualMomentumScorer:
    scoring_type = "dual_momentum"

    def __init__(self, params: Mapping[str, Any], market: str):
        self.params = dict(params)
        self.market = str(market).upper()
        self.validate_params(self.params, self.market)
        self.lookback_bars = int(self.params["lookback_bars"])

    @classmethod
    def validate_params(cls, params: Mapping[str, Any], market: str | None = None) -> None:
        unknown = set(params) - {"lookback_bars"}
        if unknown:
            raise ValueError(f"Unsupported dual_momentum params: {sorted(unknown)}")
        lookback = params.get("lookback_bars")
        if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
            raise ValueError("dual_momentum lookback_bars must be a positive integer.")

    def warmup_bars(self) -> int:
        return self.lookback_bars + 1

    def add_scores(self, frames, benchmark):
        scored = {}
        for symbol, frame in frames.items():
            out = frame.copy()
            out["Score"] = float("nan")
            symbol_benchmark = benchmark.get(symbol) if isinstance(benchmark, dict) else benchmark
            if (
                "Close" not in out
                or symbol_benchmark is None
                or symbol_benchmark.empty
                or "Close" not in symbol_benchmark
            ):
                scored[symbol] = out
                continue
            benchmark_return = symbol_benchmark["Close"].pct_change(
                self.lookback_bars,
                fill_method=None,
            ).reindex(out.index, method="ffill")
            if "IdentitySegment" in out and out["IdentitySegment"].nunique(dropna=False) > 1:
                stock_return = out.groupby(
                    "IdentitySegment",
                    sort=False,
                    dropna=False,
                )["Close"].transform(
                    lambda values: values.pct_change(
                        self.lookback_bars,
                        fill_method=None,
                    )
                )
            else:
                stock_return = out["Close"].pct_change(
                    self.lookback_bars,
                    fill_method=None,
                )
            excess = stock_return - benchmark_return
            out["Score"] = excess.where((stock_return > 0.0) & (excess > 0.0))
            scored[symbol] = out
        return scored

    def rank(self, scores):
        return rank_scores(scores)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Nasdaq-100 universe schedules on one canonical local price bundle."
    )
    parser.add_argument("--strategy", default=str(DEFAULT_STRATEGY))
    parser.add_argument("--runtime", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--wikipedia-history", default=str(DEFAULT_WIKIPEDIA_HISTORY))
    parser.add_argument("--period", default="max")
    parser.add_argument("--start", default="2015-10-19")
    parser.add_argument("--end", default="2026-07-15")
    parser.add_argument("--entry", default="single")
    parser.add_argument("--market-filter", default="1d")
    parser.add_argument("--asset-filter", default="ichimoku_cloud+ema_trend")
    parser.add_argument("--rs-method", default="dual_momentum")
    parser.add_argument("--rs-period", type=int, default=150)
    parser.add_argument("--sell-confirm-bars", type=int, default=5)
    parser.add_argument("--hurdle", type=float, default=2.0)
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--st-period", type=int, default=10)
    parser.add_argument("--st-multiplier", type=float, default=3.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--run-id", default="")
    return parser


def canonical_configs(args: argparse.Namespace):
    base = load_split_config(args.strategy, args.runtime)
    base = replace(
        base,
        period=args.period,
        timeframe="1d",
        data_store=replace(base.data_store, provider="parquet"),
    )
    index_universe = replace(
        base.universe,
        source="index_events",
        profiles={"US": ("nasdaq100",)},
        history_file="",
        file="",
        symbols=(),
        filters=replace(base.universe.filters, enabled=False),
    )
    index_config = replace(
        base,
        universe=index_universe,
        universe_file="",
        symbols=(),
    )
    params = {
        "entry": args.entry,
        "market_filter": args.market_filter,
        "asset_filter": args.asset_filter,
        "rs_method": args.rs_method,
        "rs_period": args.rs_period,
        "sell_confirm_bars": args.sell_confirm_bars,
        "hurdle": args.hurdle,
        "max_positions": args.max_positions,
        "st_period": args.st_period,
        "st_multiplier": args.st_multiplier,
        "fee_rate": args.fee_rate,
        "slippage_rate": args.slippage_rate,
    }
    canonical_overlay = {
        key: value
        for key, value in params.items()
        if key not in {"rs_method", "hurdle"}
    }
    index_config = apply_config_overlay(index_config, canonical_overlay)
    index_config = replace(
        index_config,
        scoring=replace(
            index_config.scoring,
            type="dual_momentum",
            params={"lookback_bars": int(args.rs_period)},
        ),
        leader_rotation=replace(
            index_config.leader_rotation,
            hurdle_atr_mult=float(args.hurdle),
        ),
    )

    wikipedia_universe = replace(
        base.universe,
        source="history_file",
        history_file=str(Path(args.wikipedia_history).resolve()),
        file="",
        symbols=(),
        filters=replace(base.universe.filters, enabled=False),
    )
    wikipedia_config = replace(
        index_config,
        universe=wikipedia_universe,
        universe_file="",
        symbols=(),
    )
    return index_config, wikipedia_config, params


def normalized_schedule(
    raw_schedule: tuple[Mapping[str, Any], ...],
    *,
    allowed_symbols: set[str],
    identity_schedule: tuple[Mapping[str, Any], ...],
) -> tuple[tuple[dict[str, Any], ...], set[str]]:
    identities: dict[str, list[tuple[pd.Timestamp, dict[str, Any]]]] = {}
    for entry in identity_schedule:
        effective = pd.Timestamp(entry["effective_date"])
        for member in entry.get("members", ()) or ():
            if not isinstance(member, Mapping):
                continue
            symbol = str(member.get("symbol") or "")
            if symbol:
                identities.setdefault(symbol, []).append((effective, dict(member)))
    for rows in identities.values():
        rows.sort(key=lambda item: item[0])

    missing: set[str] = set()
    normalized: list[dict[str, Any]] = []
    prior_symbols: tuple[str, ...] | None = None
    for entry in sorted(raw_schedule, key=lambda item: str(item.get("effective_date", ""))):
        effective = pd.Timestamp(entry["effective_date"])
        raw_symbols = entry.get("symbols", ()) or ()
        symbols = tuple(sorted({str(symbol) for symbol in raw_symbols if str(symbol)}))
        missing.update(symbol for symbol in symbols if symbol not in allowed_symbols)
        symbols = tuple(symbol for symbol in symbols if symbol in allowed_symbols)
        if not symbols or symbols == prior_symbols:
            continue
        members: list[dict[str, Any]] = []
        for symbol in symbols:
            candidates = identities.get(symbol, ())
            if candidates:
                dates = [item[0] for item in candidates]
                position = bisect_right(dates, effective) - 1
                member = dict(candidates[max(position, 0)][1])
            else:
                member = {"symbol": symbol, "market": "US"}
            member["symbol"] = symbol
            members.append(member)
        normalized.append(
            {
                "effective_date": effective.date().isoformat(),
                "symbols": list(symbols),
                "members": members,
            }
        )
        prior_symbols = symbols
    return tuple(normalized), missing


def benchmark_return(data: MarketData, index: pd.Index) -> float:
    benchmark = data.benchmark
    if not benchmark:
        return float("nan")
    frame = next((value for value in benchmark.values() if value is not None and not value.empty), None)
    if frame is None or "Close" not in frame:
        return float("nan")
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    selected = close.loc[(close.index >= index[0]) & (close.index <= index[-1])]
    if len(selected) < 2 or float(selected.iloc[0]) == 0.0:
        return float("nan")
    return float(selected.iloc[-1] / selected.iloc[0] - 1.0)


def result_payload(label: str, result: BacktestResult, qqq_return: float) -> dict[str, Any]:
    total_return = float(result.metrics["total_return"])
    return {
        "label": label,
        "start": str(result.equity.index[0]),
        "end": str(result.equity.index[-1]),
        "initial_equity": float(result.equity.iloc[0]),
        "final_equity": float(result.equity.iloc[-1]),
        "total_return": total_return,
        "qqq_return": qqq_return,
        "alpha": total_return - qqq_return,
        "metrics": dict(result.metrics),
        "trade_count": int(result.metrics["trade_count"]),
        "data_version": result.data_version,
        "completed_session": result.completed_session,
        "data_quality": result.data_quality,
        "price_mode": result.price_mode,
        "corporate_action_cash": result.corporate_action_cash,
        "processed_corporate_action_count": len(result.processed_corporate_action_ids),
        "unresolved_corporate_action_ids": list(result.unresolved_corporate_action_ids),
        "warnings": list(result.data_warnings),
    }


def universe_difference_rows(
    index: pd.Index,
    index_schedule: tuple[dict[str, Any], ...],
    wikipedia_schedule: tuple[dict[str, Any], ...],
) -> pd.DataFrame:
    rows = []
    for timestamp in index:
        index_symbols = active_universe_symbols(index_schedule, timestamp) or set()
        wikipedia_symbols = active_universe_symbols(wikipedia_schedule, timestamp) or set()
        only_index = sorted(index_symbols - wikipedia_symbols)
        only_wikipedia = sorted(wikipedia_symbols - index_symbols)
        rows.append(
            {
                "date": pd.Timestamp(timestamp).date().isoformat(),
                "index_events_count": len(index_symbols),
                "wikipedia_count": len(wikipedia_symbols),
                "common_count": len(index_symbols & wikipedia_symbols),
                "only_index_events_count": len(only_index),
                "only_wikipedia_count": len(only_wikipedia),
                "only_index_events": ",".join(only_index),
                "only_wikipedia": ",".join(only_wikipedia),
            }
        )
    return pd.DataFrame(rows)


def pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def main() -> None:
    args = build_parser().parse_args()
    index_config, wikipedia_config, params = canonical_configs(args)

    print("[universe-ab] loading one canonical index_events price bundle...", flush=True)
    base_data = download_for_config(index_config, allow_stale=True)
    index_schedule_raw = tuple(base_data.universe_schedule)
    if not index_schedule_raw:
        raise RuntimeError("Canonical index_events schedule is empty.")

    print("[universe-ab] resolving Wikipedia schedule...", flush=True)
    wikipedia_resolved = resolve_universe(wikipedia_config, mode="research")
    wikipedia_schedule_raw = wikipedia_resolved.schedule_as_dicts()
    allowed_symbols = set(base_data.entry_symbols or tuple(base_data.bars))
    index_schedule, index_missing = normalized_schedule(
        index_schedule_raw,
        allowed_symbols=allowed_symbols,
        identity_schedule=index_schedule_raw,
    )
    wikipedia_schedule, wikipedia_missing = normalized_schedule(
        wikipedia_schedule_raw,
        allowed_symbols=allowed_symbols,
        identity_schedule=index_schedule_raw,
    )

    index_data = replace(base_data, universe_schedule=index_schedule)
    wikipedia_data = replace(base_data, universe_schedule=wikipedia_schedule)
    index_timeline = market_index(index_data)
    wikipedia_timeline = market_index(wikipedia_data)
    common_index = index_timeline.intersection(wikipedia_timeline, sort=False)
    common_index = common_index[
        (common_index >= pd.Timestamp(args.start))
        & (common_index <= pd.Timestamp(args.end))
    ]
    if len(common_index) < 2:
        raise RuntimeError("No shared sessions remain in the requested date range.")
    print(
        f"[universe-ab] shared sessions {common_index[0]} -> {common_index[-1]} "
        f"({len(common_index)})",
        flush=True,
    )

    results: dict[str, BacktestResult] = {}
    for label, data in (("index_events", index_data), ("wikipedia_quarterly", wikipedia_data)):
        print(f"[universe-ab] running {label}...", flush=True)
        results[label] = run_backtest_on_data(index_config, data, run_index=common_index)
        metrics = results[label].metrics
        print(
            f"[universe-ab] {label} return={pct(float(metrics['total_return']))} "
            f"mdd={pct(float(metrics['mdd']))} sharpe={float(metrics['sharpe']):.2f} "
            f"trades={int(metrics['trade_count'])}",
            flush=True,
        )

    run_id = args.run_id.strip() or datetime.now().strftime("universe_ab_%Y%m%d_%H%M%S")
    run_dir = Path(args.results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    difference = universe_difference_rows(common_index, index_schedule, wikipedia_schedule)
    difference.to_csv(run_dir / "universe_daily_comparison.csv", index=False)

    payloads: dict[str, dict[str, Any]] = {}
    for label, result in results.items():
        qqq = benchmark_return(base_data, result.equity.index)
        payloads[label] = result_payload(label, result, qqq)
        result.equity.rename("equity").to_frame().to_csv(run_dir / f"{label}_equity.csv")
        pd.DataFrame(result.trade_records).to_csv(run_dir / f"{label}_trades.csv", index=False)

    differing_days = int(
        (
            difference["only_index_events_count"]
            + difference["only_wikipedia_count"]
        ).gt(0).sum()
    )
    summary = {
        "experiment": "same canonical prices and execution; universe schedule only",
        "params": params,
        "requested_start": args.start,
        "requested_end": args.end,
        "shared_start": str(common_index[0]),
        "shared_end": str(common_index[-1]),
        "shared_sessions": len(common_index),
        "canonical_price_symbols": len(allowed_symbols),
        "index_schedule_entries": len(index_schedule),
        "wikipedia_schedule_entries": len(wikipedia_schedule),
        "index_symbols_without_canonical_prices": sorted(index_missing),
        "wikipedia_symbols_without_canonical_prices": sorted(wikipedia_missing),
        "days_with_different_membership": differing_days,
        "average_only_index_events": float(difference["only_index_events_count"].mean()),
        "average_only_wikipedia": float(difference["only_wikipedia_count"].mean()),
        "results": payloads,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    rows = []
    for label, item in payloads.items():
        metrics = item["metrics"]
        rows.append(
            {
                "universe": label,
                "start": item["start"],
                "end": item["end"],
                "total_return": item["total_return"],
                "cagr": metrics["cagr"],
                "qqq_return": item["qqq_return"],
                "alpha": item["alpha"],
                "mdd": metrics["mdd"],
                "sharpe": metrics["sharpe"],
                "win_rate": metrics["win_rate"],
                "payoff_ratio": metrics["payoff_ratio"],
                "trade_count": item["trade_count"],
                "final_equity": item["final_equity"],
                "corporate_action_cash": item["corporate_action_cash"],
            }
        )
    pd.DataFrame(rows).to_csv(run_dir / "comparison.csv", index=False)

    report_lines = [
        "Nasdaq-100 Pure Universe A/B",
        f"Period: {common_index[0]} -> {common_index[-1]}",
        f"Price mode: {index_config.data_store.price_mode}",
        "Prices/execution/corporate actions: identical canonical local bundle",
        f"Membership differs on {differing_days}/{len(common_index)} sessions",
        "",
    ]
    for row in rows:
        report_lines.extend(
            [
                row["universe"],
                f"  Return: {pct(float(row['total_return']))}",
                f"  CAGR: {pct(float(row['cagr']))}",
                f"  QQQ: {pct(float(row['qqq_return']))}",
                f"  Alpha: {pct(float(row['alpha']))}",
                f"  MDD: {pct(float(row['mdd']))}",
                f"  Sharpe: {float(row['sharpe']):.2f}",
                f"  Win rate: {pct(float(row['win_rate']))}",
                f"  Payoff: {float(row['payoff_ratio']):.2f}",
                f"  Trades: {int(row['trade_count'])}",
                "",
            ]
        )
    (run_dir / "report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[universe-ab] saved {run_dir}", flush=True)


if __name__ == "__main__":
    main()
