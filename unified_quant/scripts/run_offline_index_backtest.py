"""Run one point-in-time index backtest from the local Parquet release only.

This intentionally bypasses freshness synchronization so an analysis run can
never spend market-data API calls.  The selected local release and its quality
warnings are still recorded in the normal backtest artifacts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from supertrend_quant.config import load_split_config
from supertrend_quant.market_store.provider import ParquetMarketDataProvider
from supertrend_quant.results import save_backtest_result
from supertrend_quant.runners import (
    _configured_completed_session,
    _schedule_for_period,
    run_backtest_on_data,
)
from supertrend_quant.universe import resolve_universe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a local-Parquet-only point-in-time index backtest."
    )
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--period", default="5y")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = replace(
        load_split_config(args.strategy, args.runtime, args.data),
        period=args.period,
    )
    resolved = resolve_universe(config, mode="backtest")
    schedule = _schedule_for_period(
        resolved.schedule,
        config.period,
        _configured_completed_session(config),
    )
    symbols = list(
        dict.fromkeys(
            member.symbol
            for entry in schedule
            for member in entry.members
        )
    )
    if not symbols:
        symbols = list(resolved.eligible_symbols)

    print(
        json.dumps(
            {
                "stage": "loading_local_parquet",
                "schedule_entries": len(schedule),
                "requested_symbols": len(symbols),
            }
        ),
        flush=True,
    )

    market_data = ParquetMarketDataProvider(
        config.data_store.local_cache_dir
    ).load(
        config,
        symbols,
        universe_schedule=tuple(entry.to_dict() for entry in schedule),
    )
    print(
        json.dumps(
            {
                "stage": "running_backtest",
                "loaded_symbols": len(market_data.bars),
                "completed_session": market_data.completed_session,
                "data_quality": market_data.data_quality,
            }
        ),
        flush=True,
    )
    market_data = replace(
        market_data,
        universe_snapshot=resolved.snapshot.to_dict(),
        universe_schedule=tuple(entry.to_dict() for entry in schedule),
    )
    result = run_backtest_on_data(config, market_data)
    run_dir = save_backtest_result(
        result,
        config,
        config.backtest.results_dir,
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "schedule_entries": len(schedule),
                "requested_symbols": len(symbols),
                "loaded_symbols": len(market_data.bars),
                "completed_session": result.completed_session,
                "data_quality": result.data_quality,
                "data_warnings": list(result.data_warnings),
                "metrics": result.metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
