from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLAYGROUND_ROOT / "scripts"))
sys.path.insert(0, str(PLAYGROUND_ROOT / "src"))

from search_nasdaq100_daily_3y_grid import (  # noqa: E402
    prepare_active_universe,
    prepare_candidate_lists,
    prepare_exit_down_states,
    prepare_market_filter_states,
    prepare_row_positions,
    qqq_return_for_index,
    run_index_from_start,
    run_prepared_backtest,
)
from market_data_source import load_experiment_market_data  # noqa: E402
from supertrend_quant.config import load_split_config  # noqa: E402
from supertrend_quant.data import market_index  # noqa: E402
from supertrend_quant.metrics import format_float, format_pct  # noqa: E402
from supertrend_quant.research import apply_config_overlay  # noqa: E402
from supertrend_quant.strategies import create_strategy  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one Nasdaq-100 daily strategy combination.")
    parser.add_argument("--strategy", default=str(PLAYGROUND_ROOT / "configs" / "strategies" / "leader_rotation.yaml"))
    parser.add_argument(
        "--runtime",
        default=str(PLAYGROUND_ROOT / "configs" / "runtimes" / "research_us_nasdaq100_rolling.yaml"),
    )
    parser.add_argument("--period", default="max")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument("--data-source", choices=("local", "yahoo"), default="local")
    parser.add_argument("--entry", default="single")
    parser.add_argument("--market-filter", default="1d")
    parser.add_argument("--asset-filter", default="ichimoku_cloud")
    parser.add_argument("--rs-method", default="dual_momentum")
    parser.add_argument("--rs-period", type=int, default=100)
    parser.add_argument("--sell-confirm-bars", type=int, default=5)
    parser.add_argument("--hurdle", type=float, default=2.0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--st-period", type=int, default=10)
    parser.add_argument("--st-multiplier", type=float, default=3.0)
    parser.add_argument("--fee-rate", type=float, default=None)
    parser.add_argument("--slippage-rate", type=float, default=None)
    parser.add_argument(
        "--results-dir",
        default=str(PLAYGROUND_ROOT / "results" / "research" / "us_nasdaq100_rolling" / "single_evals"),
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--save-trades", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = load_split_config(args.strategy, args.runtime)
    base = base.__class__(**{**base.__dict__, "period": args.period, "timeframe": "1d"})
    cost_overlay: dict[str, Any] = {}
    if args.fee_rate is not None:
        cost_overlay["fee_rate"] = args.fee_rate
    if args.slippage_rate is not None:
        cost_overlay["slippage_rate"] = args.slippage_rate
    if cost_overlay:
        base = apply_config_overlay(base, cost_overlay)

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
    }

    print(f"[single-eval] loading {args.data_source} data...", flush=True)
    market_data = load_experiment_market_data(
        base,
        data_source=args.data_source,
        strategy_path=args.strategy,
        runtime_path=args.runtime,
    )
    full_idx = market_index(market_data)
    requested_idx = run_index_from_start(full_idx, args.start)
    if args.end:
        requested_idx = requested_idx[requested_idx <= pd.Timestamp(args.end)]
    if len(requested_idx) < 2:
        raise RuntimeError("Not enough bars remain in the requested date range.")
    active_by_position = prepare_active_universe(market_data, full_idx)
    print(
        f"[single-eval] timeline {full_idx[0]} -> {full_idx[-1]}, requested={len(requested_idx)}",
        flush=True,
    )

    config = apply_config_overlay(base, params)
    strategy = create_strategy(config)
    print(f"[single-eval] preparing params={params}", flush=True)
    prepared = strategy.prepare_backtest(
        market_data.bars,
        benchmark=market_data.benchmark,
        filter_benchmark=market_data.filter_benchmark,
        universe_schedule=market_data.universe_schedule,
    )
    row_positions = prepare_row_positions(prepared.prepared, full_idx)
    market_filter_states = prepare_market_filter_states(prepared, full_idx)
    candidates_by_position = prepare_candidate_lists(
        config,
        strategy,
        prepared.prepared,
        market_filter_states,
        active_by_position,
        row_positions,
        full_idx,
    )
    exit_down_states = prepare_exit_down_states(config, prepared.prepared, full_idx, row_positions)
    print("[single-eval] running backtest...", flush=True)
    result = run_prepared_backtest(
        config,
        market_data,
        prepared.prepared,
        candidates_by_position,
        market_filter_states,
        exit_down_states,
        active_by_position,
        row_positions,
        strategy.warmup_bars(),
        requested_idx,
    )
    qqq_return = qqq_return_for_index(market_data, result.equity.index, {})
    metrics = result.metrics
    total_return = float(metrics["total_return"])
    print("[single-eval] done")
    print(f"fee_rate={config.costs.fee_rate}")
    print(f"slippage_rate={config.costs.slippage_rate}")
    print(f"start={result.equity.index[0]}")
    print(f"end={result.equity.index[-1]}")
    print(f"total_return={total_return:.10f}")
    print(f"total_return_pct={format_pct(total_return)}")
    print(f"qqq_return_pct={format_pct(float(qqq_return))}")
    print(f"alpha_pct={format_pct(total_return - float(qqq_return))}")
    print(f"mdd_pct={format_pct(float(metrics['mdd']))}")
    print(f"sharpe={format_float(float(metrics['sharpe']))}")
    print(f"win_rate_pct={format_pct(float(metrics['win_rate']))}")
    print(f"payoff={format_float(float(metrics['payoff_ratio']))}")
    print(f"trades={metrics['trade_count']}")

    if args.save_trades:
        run_id = args.run_id.strip() or datetime.now().strftime("single_eval_%Y%m%d_%H%M%S")
        run_dir = Path(args.results_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        trades = pd.DataFrame(result.trade_records)
        if not trades.empty:
            trades.insert(0, "trade_no", range(1, len(trades) + 1))
            trades["pnl_pct"] = trades["pnl_pct"].astype(float)
            trades["pnl_pct_display"] = trades["pnl_pct"].map(format_pct)
            trades["cumulative_pnl_value"] = trades["pnl_value"].astype(float).cumsum()
            trades["realized_equity_after_trade"] = (
                float(config.capital.initial_cash) + trades["cumulative_pnl_value"]
            )
            trades["cumulative_return"] = (
                trades["realized_equity_after_trade"] / float(config.capital.initial_cash) - 1.0
            )
            trades["cumulative_return_display"] = trades["cumulative_return"].map(format_pct)
            trades["holding_days"] = (
                pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])
            ).dt.days
        trades.to_csv(run_dir / "trades.csv", index=False)
        result.equity.rename("equity").to_frame().to_csv(run_dir / "equity.csv")
        summary = {
            "params": params,
            "data_source": args.data_source,
            "fee_rate": config.costs.fee_rate,
            "slippage_rate": config.costs.slippage_rate,
            "start": str(result.equity.index[0]),
            "end": str(result.equity.index[-1]),
            "metrics": dict(metrics),
            "qqq_return": float(qqq_return),
            "alpha": total_return - float(qqq_return),
            "trade_count": int(metrics["trade_count"]),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"saved_dir={run_dir}")
        print(f"trades_csv={run_dir / 'trades.csv'}")
        print(f"equity_csv={run_dir / 'equity.csv'}")
        print(f"summary_json={run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
