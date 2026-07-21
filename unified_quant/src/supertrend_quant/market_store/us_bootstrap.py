from __future__ import annotations

import io
import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd
import yaml

from .index_ingest import IndexDataImporter
from .ingest import (
    DailyDataSynchronizer,
    EodhdClient,
    EodhdDailySource,
    SecuritySourceResult,
    SourceArtifact,
)
from .manifest import utc_now_iso
from .repository import LocalDatasetRepository


SP500_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20%28Updated%29.csv"
)
NASDAQ100_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/jmccarrell/n100tickers/main/"
    "src/nasdaq_100_ticker_history/n100-ticker-changes-{year}.yaml"
)
# EODHD can serve a renamed company's continuous history under its latest
# ticker. Keep the index-source label while querying that continuous code.
SYMBOL_ALIASES = {"CDAY": "DAY", "RE": "EG", "PEAK": "DOC"}


@dataclass(frozen=True)
class UsIndexHistory:
    anchors: dict[str, pd.DataFrame]
    events: dict[str, pd.DataFrame]
    raw_content: dict[str, bytes]
    source_urls: dict[str, str]

    @property
    def symbols(self) -> tuple[str, ...]:
        values = {"SPY", "QQQ"}
        for frame in self.anchors.values():
            values.update(frame["symbol"].astype(str))
        for frame in self.events.values():
            values.update(frame["symbol"].astype(str))
        return tuple(sorted(values))


@dataclass(frozen=True)
class CatalogCandidate:
    security_id: str
    symbol: str
    provider_symbol: str
    active_from: str
    active_to: str

    def covers(self, when: pd.Timestamp) -> bool:
        start = pd.Timestamp(self.active_from)
        end = pd.Timestamp(self.active_to) if self.active_to else pd.Timestamp.max
        return start <= when <= end


@dataclass(frozen=True)
class EodhdCatalog:
    source_result: SecuritySourceResult
    candidates: dict[str, tuple[CatalogCandidate, ...]]

    def security_id_for(self, symbol: str, when: str) -> str:
        timestamp = pd.Timestamp(when)
        values = list(self.candidates.get(str(symbol), ()))
        covered = [candidate for candidate in values if candidate.covers(timestamp)]
        if not covered:
            if not values:
                raise ValueError(f"No EODHD identity for {symbol} on {timestamp.date()}.")
            def distance(candidate: CatalogCandidate) -> pd.Timedelta:
                start = pd.Timestamp(candidate.active_from)
                end = pd.Timestamp(candidate.active_to) if candidate.active_to else timestamp
                if timestamp < start:
                    return start - timestamp
                if candidate.active_to and timestamp > end:
                    return timestamp - end
                return pd.Timedelta(0)

            return min(values, key=distance).security_id
        if len(covered) == 1:
            return covered[0].security_id
        finite = [candidate for candidate in covered if candidate.active_to]
        if finite:
            return min(finite, key=lambda item: pd.Timestamp(item.active_to)).security_id
        exact = [
            candidate
            for candidate in covered
            if _canonical_code(candidate.provider_symbol.split(".", 1)[0]) == _canonical_code(symbol)
            and "_old" not in candidate.provider_symbol.lower()
        ]
        return (exact[0] if exact else covered[0]).security_id


def fetch_us_index_history(*, start_date: str = "2015-01-01", session=None) -> UsIndexHistory:
    if session is None:
        import requests

        session = requests.Session()
    sp_content = _download(session, SP500_URL)
    sp_frame = pd.read_csv(io.BytesIO(sp_content))
    sp_frame["date"] = pd.to_datetime(sp_frame["date"], errors="raise")
    sp_frame = sp_frame.loc[sp_frame["date"] >= pd.Timestamp(start_date)].sort_values("date")
    if sp_frame.empty:
        raise RuntimeError("S&P 500 history has no rows in the requested period.")
    sp_anchor_date = sp_frame.iloc[0]["date"].date().isoformat()
    prior = _ticker_set(sp_frame.iloc[0]["tickers"])
    sp_anchor = pd.DataFrame({"symbol": sorted(prior)})
    sp_anchor["anchor_date"] = sp_anchor_date
    sp_events = []
    for row in sp_frame.iloc[1:].itertuples(index=False):
        current = _ticker_set(row.tickers)
        effective = pd.Timestamp(row.date).date().isoformat()
        sp_events.extend(
            {"effective_date": effective, "operation": "ADD", "symbol": symbol}
            for symbol in sorted(current - prior)
        )
        sp_events.extend(
            {"effective_date": effective, "operation": "REMOVE", "symbol": symbol}
            for symbol in sorted(prior - current)
        )
        prior = current

    nasdaq_contents = []
    nasdaq_anchor = None
    nasdaq_events = []
    start_year = pd.Timestamp(start_date).year
    current_year = pd.Timestamp.utcnow().year
    urls = []
    for year in range(start_year, current_year + 1):
        url = NASDAQ100_URL_TEMPLATE.format(year=year)
        content = _download(session, url)
        # BaseLoader is intentional: YAML 1.1 parsers otherwise coerce valid
        # ticker symbols such as ON and NO into booleans.
        payload = yaml.load(content.decode("utf-8"), Loader=yaml.BaseLoader) or {}
        nasdaq_contents.append(content)
        urls.append(url)
        if year == start_year:
            nasdaq_anchor = pd.DataFrame({"symbol": sorted(payload["tickers_on_Jan_1"])})
            nasdaq_anchor["anchor_date"] = f"{year}-01-01"
        for effective, operations in sorted((payload.get("changes") or {}).items()):
            nasdaq_events.extend(
                {"effective_date": effective, "operation": "ADD", "symbol": symbol}
                for symbol in sorted(operations.get("union", ()))
            )
            nasdaq_events.extend(
                {"effective_date": effective, "operation": "REMOVE", "symbol": symbol}
                for symbol in sorted(operations.get("difference", ()))
            )
    if nasdaq_anchor is None:
        raise RuntimeError("Nasdaq-100 history anchor is missing.")
    return UsIndexHistory(
        anchors={"sp500": sp_anchor, "nasdaq100": nasdaq_anchor},
        events={
            "sp500": pd.DataFrame(sp_events, columns=["effective_date", "operation", "symbol"]),
            "nasdaq100": pd.DataFrame(
                nasdaq_events, columns=["effective_date", "operation", "symbol"]
            ),
        },
        raw_content={"sp500": sp_content, "nasdaq100": b"\n---\n".join(nasdaq_contents)},
        source_urls={"sp500": SP500_URL, "nasdaq100": ",".join(urls)},
    )


def build_eodhd_catalog(
    symbols: tuple[str, ...],
    *,
    start_date: str,
    end_date: str,
    client: EodhdClient | None = None,
    workers: int = 8,
) -> EodhdCatalog:
    client = client or EodhdClient()
    retrieved_at = utc_now_iso()
    artifacts = []
    rows_by_code: dict[str, list[dict]] = {}
    for delisted in (0, 1):
        endpoint = "exchange-symbol-list/US"
        params = {"delisted": delisted}
        rows = client.get_json(endpoint, params=params)
        content = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        artifacts.append(
            SourceArtifact(
                source="eodhd_exchange_symbols",
                source_url=client.safe_url(endpoint, params=params),
                retrieved_at=retrieved_at,
                content=content,
                content_type="application/json",
            )
        )
        for row in rows:
            if str(row.get("Type") or "").strip().lower() not in {"common stock", "etf"}:
                continue
            normalized_row = dict(row)
            normalized_row["_delisted"] = bool(delisted)
            rows_by_code.setdefault(_canonical_code(row.get("Code")), []).append(normalized_row)

    candidate_rows: dict[str, list[dict]] = {}
    missing = []
    for symbol in symbols:
        lookup = SYMBOL_ALIASES.get(symbol, symbol)
        values = list(rows_by_code.get(_canonical_code(lookup), ()))
        if lookup != symbol:
            current_code = [
                row
                for row in values
                if _canonical_code(row.get("Code")) == _canonical_code(lookup)
                and "_old" not in str(row.get("Code") or "").lower()
            ]
            values = current_code or values
        if symbol not in {"SPY", "QQQ"}:
            stock_values = [row for row in values if str(row.get("Type")).lower() == "common stock"]
            values = stock_values or values
        if not values:
            missing.append(symbol)
        candidate_rows[symbol] = values
    if missing:
        raise RuntimeError("EODHD symbol catalog is missing: " + ", ".join(sorted(missing)))

    range_targets = {
        symbol: rows
        for symbol, rows in candidate_rows.items()
        if len(rows) > 1 or any(bool(row.get("_delisted")) for row in rows)
    }
    ranges: dict[tuple[str, str], tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {}
        for symbol, rows in range_targets.items():
            for row in rows:
                code = str(row["Code"])
                futures[
                    executor.submit(
                        client.get_json,
                        f"eod/{code}.US",
                        params={"from": start_date, "to": end_date},
                    )
                ] = (symbol, code)
        for future in as_completed(futures):
            symbol, code = futures[future]
            values = future.result()
            ranges[(symbol, code)] = (
                str(values[0]["date"]) if values else "",
                str(values[-1]["date"]) if values else "",
            )

    master_rows = []
    history_rows = []
    candidates: dict[str, tuple[CatalogCandidate, ...]] = {}
    artifact_hash = artifacts[0].source_hash
    for symbol, rows in candidate_rows.items():
        selected = []
        for row in rows:
            code = str(row["Code"])
            active_from, active_to = ranges.get((symbol, code), (start_date, ""))
            if not active_from:
                if len(rows) > 1:
                    continue
                active_from, active_to = start_date, ""
            if active_to and active_to < start_date:
                # Keep a sole catalog identity so the missing price history is
                # measured and reported instead of aborting the whole universe.
                if len(rows) > 1:
                    continue
                active_from, active_to = start_date, ""
            if active_to and pd.Timestamp(active_to) >= pd.Timestamp(end_date) - pd.Timedelta(days=10):
                active_to = ""
            # Keep index-source ticker aliases distinct even when EODHD serves
            # both through one continuous provider code (for example RE -> EG).
            security_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"eodhd:US:{code}:symbol:{symbol}")
            security_id = f"US:EODHD:{security_uuid}"
            provider_symbol = f"{code}.US"
            candidate = CatalogCandidate(
                security_id, symbol, provider_symbol, active_from, active_to
            )
            selected.append(candidate)
            common = {
                "security_id": security_id,
                "exchange": str(row.get("Exchange") or "US"),
                "active_from": active_from,
                "active_to": active_to,
                "source": "eodhd_exchange_symbols",
                "source_url": artifacts[0].source_url,
                "retrieved_at": retrieved_at,
                "source_hash": artifact_hash,
            }
            master_rows.append(
                {
                    **common,
                    "primary_symbol": symbol,
                    "provider_symbol": provider_symbol,
                    "action_provider_symbol": (
                        f"{symbol}.US"
                        if SYMBOL_ALIASES.get(symbol, symbol) != symbol
                        else provider_symbol
                    ),
                    "name": str(row.get("Name") or symbol),
                    "asset_type": "ETF" if str(row.get("Type")).lower() == "etf" else "STOCK",
                    "currency": str(row.get("Currency") or "USD"),
                    "country": "US",
                }
            )
            history_rows.append(
                {
                    "security_id": security_id,
                    "symbol": symbol,
                    "exchange": common["exchange"],
                    # Index dates can precede the provider's first usable
                    # price around spin-offs. Keep the identity resolvable;
                    # the master retains the real availability window.
                    "effective_from": start_date,
                    "effective_to": "",
                    "source": common["source"],
                    "source_url": common["source_url"],
                    "retrieved_at": retrieved_at,
                    "source_hash": artifact_hash,
                }
            )
        if not selected:
            raise RuntimeError(f"EODHD has no usable history candidate for {symbol}.")
        candidates[symbol] = tuple(selected)
    return EodhdCatalog(
        SecuritySourceResult(
            pd.DataFrame(master_rows).drop_duplicates("security_id", keep="last"),
            pd.DataFrame(history_rows).drop_duplicates(
                ["security_id", "symbol", "effective_from"], keep="last"
            ),
            tuple(artifacts),
        ),
        candidates,
    )


def bootstrap_us_market_data(
    repository: LocalDatasetRepository,
    *,
    start_date: str,
    end_date: str,
):
    history = fetch_us_index_history(start_date=start_date)
    catalog = build_eodhd_catalog(
        history.symbols,
        start_date=start_date,
        end_date=end_date,
    )
    repository.write_frame(
        "security_master",
        catalog.source_result.security_master,
        completed_session=end_date,
        metadata={"operation": "eodhd_index_scoped_security_master"},
    )
    repository.write_frame(
        "symbol_history",
        catalog.source_result.symbol_history,
        completed_session=end_date,
        metadata={"operation": "eodhd_index_scoped_symbol_history"},
    )
    action_symbols = {
        str(row.security_id): str(row.action_provider_symbol)
        for row in catalog.source_result.security_master.itertuples(index=False)
        if str(row.action_provider_symbol or "").strip()
        != str(row.provider_symbol or "").strip()
    }
    synchronizer = DailyDataSynchronizer(
        repository,
        price_source=EodhdDailySource(action_symbol_overrides=action_symbols),
    )
    synchronizer._archive(catalog.source_result.artifacts, end_date, {}, [])
    importer = IndexDataImporter(repository)
    index_results = []
    for index_id in ("sp500", "nasdaq100"):
        anchor = history.anchors[index_id].copy()
        anchor_date = str(anchor.iloc[0]["anchor_date"])
        anchor["security_id"] = [
            catalog.security_id_for(symbol, anchor_date) for symbol in anchor["symbol"].astype(str)
        ]
        events = history.events[index_id].copy()
        events = _assign_event_security_ids(anchor, events, catalog)
        index_results.append(
            importer.import_anchor(
                index_id,
                anchor_date,
                anchor,
                source=f"community_{index_id}_history",
                source_url=history.source_urls[index_id],
                official=False,
                raw_content=history.raw_content[index_id],
            )
        )
        index_results.append(
            importer.import_events(
                index_id,
                events,
                source=f"community_{index_id}_history",
                source_url=history.source_urls[index_id],
                official=False,
                raw_content=history.raw_content[index_id],
            )
        )
    synced = synchronizer.sync(
        end_date,
        backfill_start=start_date,
        refresh_security_master=False,
    )
    return history, catalog, tuple(index_results), synced


def _download(session, url: str) -> bytes:
    try:
        response = session.get(url, timeout=120)
        response.raise_for_status()
        return response.content
    except Exception as exc:
        raise RuntimeError(f"Index source download failed: {type(exc).__name__}") from None


def _ticker_set(value) -> set[str]:
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _assign_event_security_ids(
    anchor: pd.DataFrame,
    events: pd.DataFrame,
    catalog: EodhdCatalog,
) -> pd.DataFrame:
    membership = dict(zip(anchor["symbol"].astype(str), anchor["security_id"].astype(str)))
    working = events.copy()
    working["_operation_order"] = working["operation"].astype(str).str.upper().map(
        {"REMOVE": 0, "ADD": 1}
    )
    working = working.sort_values(
        ["effective_date", "_operation_order", "symbol"], kind="stable"
    )
    security_ids = []
    for row in working.itertuples(index=False):
        symbol = str(row.symbol)
        operation = str(row.operation).upper()
        if operation == "REMOVE":
            security_id = membership.pop(symbol, None)
            if security_id is None:
                raise ValueError(
                    f"Index source removes non-member {symbol} on {row.effective_date}."
                )
        else:
            security_id = catalog.security_id_for(symbol, str(row.effective_date))
            membership[symbol] = security_id
        security_ids.append(security_id)
    working["security_id"] = security_ids
    return working.drop(columns="_operation_order")


def _canonical_code(value) -> str:
    code = str(value or "").upper().strip()
    code = re.sub(r"_OLD\d*$", "", code)
    return code.replace("-", ".")
