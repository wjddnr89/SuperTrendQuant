from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_fox_terminal_prices.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_fox_terminal_prices", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def _frames() -> dict[str, pd.DataFrame]:
    master = []
    history = []
    prices = []
    factors = []
    archive = []
    for number, (symbol, security_id) in enumerate(script.TARGETS.items(), start=1):
        source_hash = str(number) * 64
        source_url = f"https://eodhd.test/{symbol}"
        master.append(
            {
                "security_id": security_id,
                "primary_symbol": symbol,
                "active_to": script.BOUNDARY,
            }
        )
        history.append(
            {
                "security_id": security_id,
                "symbol": symbol,
                "effective_to": script.BOUNDARY,
            }
        )
        close = 49.0 + number / 10
        prices.extend(
            [
                {
                    "security_id": security_id,
                    "session": script.BOUNDARY,
                    "open": close + 1,
                    "high": close + 2,
                    "low": close - 1,
                    "close": close,
                    "volume": 1_000_000,
                    "source": "eodhd_eod",
                    "source_url": source_url,
                    "source_hash": source_hash,
                },
                {
                    "security_id": security_id,
                    "session": script.OVERRUN_SESSION,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": number,
                    "source": "eodhd_eod",
                    "source_url": source_url,
                    "source_hash": source_hash,
                },
            ]
        )
        factors.extend(
            [
                {"security_id": security_id, "session": pd.Timestamp(script.BOUNDARY)},
                {
                    "security_id": security_id,
                    "session": pd.Timestamp(script.OVERRUN_SESSION),
                },
            ]
        )
        archive.append(
            {
                "source_url": source_url,
                "source_hash": source_hash,
            }
        )
    return {
        "security_master": pd.DataFrame(master),
        "symbol_history": pd.DataFrame(history),
        "daily_price_raw": pd.DataFrame(prices),
        "adjustment_factors": pd.DataFrame(factors),
        "source_archive": pd.DataFrame(archive),
    }


def test_exact_synthetic_rows_are_removed_and_replay_is_idempotent():
    frames = _frames()
    prepared = script.prepare_fox_terminal_repair(frames)
    assert prepared.summary["price_rows_removed"] == 2
    assert prepared.summary["factor_rows_removed"] == 2
    assert set(prepared.frames["daily_price_raw"]["session"]) == {script.BOUNDARY}
    assert set(
        pd.to_datetime(prepared.frames["adjustment_factors"]["session"])
        .dt.date.astype(str)
    ) == {script.BOUNDARY}

    replay_frames = {
        **frames,
        **prepared.frames,
    }
    replay = script.prepare_fox_terminal_repair(replay_frames)
    assert replay.summary["status"] == "already_repaired"


def test_non_synthetic_or_unarchived_overrun_fails_closed():
    frames = _frames()
    frames["daily_price_raw"].loc[
        frames["daily_price_raw"]["session"].eq(script.OVERRUN_SESSION),
        "volume",
    ] = 1_000
    with pytest.raises(ValueError, match="synthetic flat bar"):
        script.prepare_fox_terminal_repair(frames)

    frames = _frames()
    frames["source_archive"] = frames["source_archive"].iloc[:1].copy()
    with pytest.raises(ValueError, match="source_archive"):
        script.prepare_fox_terminal_repair(frames)
