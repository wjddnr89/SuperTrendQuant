from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd


SOURCE_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "universes" / "nasdaq100_quarterly_history.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Nasdaq-100 historical universe snapshots from Wikipedia.")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--source-url", default=SOURCE_URL)
    return parser


def progress(message: str) -> None:
    print(f"[nasdaq100-history] {message}", flush=True)


def normalize_ticker(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return ""
    return text.replace(".", "-")


def fetch_tables(url: str) -> list[pd.DataFrame]:
    progress(f"fetch source: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8")
    progress("parse HTML tables")
    return pd.read_html(StringIO(html))


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    copy = frame.copy()
    if isinstance(copy.columns, pd.MultiIndex):
        copy.columns = [
            "_".join(str(part).strip() for part in column if str(part).strip() and "Unnamed" not in str(part))
            for column in copy.columns
        ]
    else:
        copy.columns = [str(column).strip() for column in copy.columns]
    return copy


def current_symbols(table: pd.DataFrame) -> set[str]:
    frame = flatten_columns(table)
    ticker_column = next((column for column in frame.columns if column.lower().endswith("ticker")), None)
    if ticker_column is None:
        raise ValueError("Could not find current ticker column.")
    symbols = {normalize_ticker(value) for value in frame[ticker_column]}
    symbols.discard("")
    return symbols


def change_rows(table: pd.DataFrame) -> list[dict[str, Any]]:
    frame = flatten_columns(table)
    date_column = next((column for column in frame.columns if column.lower().endswith("date")), None)
    added_column = next((column for column in frame.columns if column.lower().endswith("added_ticker")), None)
    removed_column = next((column for column in frame.columns if column.lower().endswith("removed_ticker")), None)
    if date_column is None or added_column is None or removed_column is None:
        raise ValueError(f"Unexpected change-table columns: {list(frame.columns)}")

    rows: list[dict[str, Any]] = []
    for _, raw in frame.iterrows():
        parsed_date = pd.to_datetime(raw[date_column], errors="coerce")
        if pd.isna(parsed_date):
            continue
        rows.append(
            {
                "date": parsed_date.date(),
                "added": normalize_ticker(raw[added_column]),
                "removed": normalize_ticker(raw[removed_column]),
            }
        )
    return sorted(rows, key=lambda item: item["date"], reverse=True)


def quarter_starts(start: date, end: date) -> list[date]:
    current = date(start.year, ((start.month - 1) // 3) * 3 + 1, 1)
    if current < start:
        month = current.month + 3
        year = current.year + (month - 1) // 12
        current = date(year, ((month - 1) % 12) + 1, 1)

    out: list[date] = []
    while current <= end:
        out.append(current)
        month = current.month + 3
        year = current.year + (month - 1) // 12
        current = date(year, ((month - 1) % 12) + 1, 1)
    if out and out[0] != start:
        out.insert(0, start)
    elif not out:
        out.append(start)
    return out


def symbols_as_of(current: set[str], changes_desc: list[dict[str, Any]], as_of: date) -> list[str]:
    symbols = set(current)
    for change in changes_desc:
        if change["date"] <= as_of:
            continue
        added = change["added"]
        removed = change["removed"]
        if added:
            symbols.discard(added)
        if removed:
            symbols.add(removed)
    return sorted(symbols)


def build_history(start: date, end: date, tables: list[pd.DataFrame]) -> dict[str, Any]:
    current = current_symbols(tables[0])
    changes = change_rows(tables[1])
    progress(f"current constituents: {len(current)}")
    progress(f"component-change rows: {len(changes)}")

    snapshots = []
    dates = quarter_starts(start, end)
    for index, effective_date in enumerate(dates, start=1):
        snapshots.append(
            {
                "effective_date": effective_date.isoformat(),
                "symbols": symbols_as_of(current, changes, effective_date),
            }
        )
        if index == 1 or index == len(dates) or index % 8 == 0:
            progress(f"snapshots built: {index}/{len(dates)}")

    return {
        "market": "US",
        "profile": "nasdaq100",
        "rebalance": "quarterly",
        "generated_at": datetime.now().date().isoformat(),
        "source_url": SOURCE_URL,
        "source_notes": [
            "Current Nasdaq-100 symbols and historical component changes were parsed from Wikipedia's List of NASDAQ-100 companies page.",
            "Snapshots are reconstructed backward from the current constituent list by reversing component-change rows.",
            "Snapshots are quarterly and each snapshot applies from its effective_date until the next snapshot.",
        ],
        "snapshots": snapshots,
    }


def main() -> None:
    args = build_parser().parse_args()
    start = pd.Timestamp(args.start).date()
    end = pd.Timestamp(args.end).date() if args.end else datetime.now().date()
    if end < start:
        raise ValueError("--end must be on or after --start.")

    tables = fetch_tables(args.source_url)
    history = build_history(start, end, tables)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    progress(f"wrote: {output}")
    progress(
        "range: "
        f"{history['snapshots'][0]['effective_date']} -> {history['snapshots'][-1]['effective_date']} "
        f"({len(history['snapshots'])} snapshots)"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[nasdaq100-history] failed: {exc}", file=sys.stderr, flush=True)
        raise
