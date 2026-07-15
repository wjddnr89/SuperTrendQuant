from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd

from .adjustments import build_adjustment_factors
from .manifest import sha256_bytes, utc_now_iso, write_atomic
from .models import DataQuality
from .repository import DatasetWriteResult, LocalDatasetRepository
from .schemas import DATASET_SPECS
from .validation import validate_repository_snapshot


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


@dataclass(frozen=True)
class SourceArtifact:
    source: str
    source_url: str
    retrieved_at: str
    content: bytes
    content_type: str

    @property
    def source_hash(self) -> str:
        return sha256_bytes(self.content)


@dataclass(frozen=True)
class SecuritySourceResult:
    security_master: pd.DataFrame
    symbol_history: pd.DataFrame
    artifacts: tuple[SourceArtifact, ...]
    warnings: tuple[str, ...] = ()


class SecurityMasterSource(Protocol):
    def fetch(self) -> SecuritySourceResult:
        ...


class SecNasdaqSecurityMasterSource:
    """Combine official SEC CIK identities with official US listing directories."""

    def __init__(self, session=None, *, user_agent: str | None = None):
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT", "")
        if not self.user_agent:
            raise RuntimeError(
                "SEC_USER_AGENT is required, for example 'Your Name your-email@example.com'."
            )

    def fetch(self) -> SecuritySourceResult:
        retrieved_at = utc_now_iso()
        sec = self._get(SEC_TICKERS_URL, retrieved_at, "sec_company_tickers")
        nasdaq = self._get(NASDAQ_LISTED_URL, retrieved_at, "nasdaq_symbol_directory")
        other = self._get(OTHER_LISTED_URL, retrieved_at, "nasdaq_other_listed")
        sec_rows = _parse_sec_companies(sec)
        listed_rows = _parse_listing_directory(nasdaq, exchange="NASDAQ")
        listed_rows.extend(_parse_listing_directory(other, exchange="OTHER"))
        by_symbol = {row["primary_symbol"]: row for row in sec_rows}
        warnings: list[str] = []
        for listing in listed_rows:
            symbol = listing["primary_symbol"]
            existing = by_symbol.get(symbol)
            if existing is not None:
                existing["exchange"] = listing["exchange"]
                existing["asset_type"] = listing["asset_type"]
                continue
            stable_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"supertrendquant:us-listing:{listing['exchange']}:{symbol}")
            listing["security_id"] = f"US:STQ:{stable_uuid}"
            listing["source"] = "nasdaq_trader"
            listing["source_url"] = (
                NASDAQ_LISTED_URL if listing["exchange"] == "NASDAQ" else OTHER_LISTED_URL
            )
            listing["retrieved_at"] = retrieved_at
            listing["source_hash"] = nasdaq.source_hash if listing["exchange"] == "NASDAQ" else other.source_hash
            by_symbol[symbol] = listing
            warnings.append(f"No SEC CIK for {symbol}; assigned persistent internal security_id.")
        master = pd.DataFrame(by_symbol.values())
        master = master.sort_values(["primary_symbol", "security_id"]).reset_index(drop=True)
        history = pd.DataFrame(
            [
                {
                    "security_id": row.security_id,
                    "symbol": row.primary_symbol,
                    "exchange": row.exchange,
                    "effective_from": row.active_from,
                    "effective_to": row.active_to,
                    "source": row.source,
                    "source_url": row.source_url,
                    "retrieved_at": row.retrieved_at,
                    "source_hash": row.source_hash,
                }
                for row in master.itertuples(index=False)
            ]
        )
        return SecuritySourceResult(master, history, (sec, nasdaq, other), tuple(warnings))

    def _get(self, url: str, retrieved_at: str, source: str) -> SourceArtifact:
        response = self.session.get(
            url,
            timeout=60,
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
        )
        response.raise_for_status()
        return SourceArtifact(
            source=source,
            source_url=url,
            retrieved_at=retrieved_at,
            content=response.content,
            content_type=response.headers.get("Content-Type", "application/octet-stream"),
        )


@dataclass(frozen=True)
class YahooFetchResult:
    prices: pd.DataFrame
    corporate_actions: pd.DataFrame
    artifacts: tuple[SourceArtifact, ...]
    missing_symbols: tuple[str, ...]


class YahooDailySource:
    def fetch(
        self,
        securities: dict[str, str],
        *,
        start: str,
        end: str,
        batch_size: int = 100,
    ) -> YahooFetchResult:
        try:
            import yfinance as yf
        except ModuleNotFoundError as exc:
            raise RuntimeError("yfinance is required for Yahoo daily ingestion.") from exc
        retrieved_at = utc_now_iso()
        by_symbol = {symbol: security_id for security_id, symbol in securities.items()}
        prices: list[dict] = []
        actions: list[dict] = []
        artifacts: list[SourceArtifact] = []
        seen: set[str] = set()
        symbols = tuple(by_symbol)
        end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
        for offset in range(0, len(symbols), batch_size):
            batch = list(symbols[offset : offset + batch_size])
            raw = yf.download(
                tickers=batch,
                start=start,
                end=end_exclusive,
                interval="1d",
                auto_adjust=False,
                actions=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
            artifact_content = raw.to_csv().encode()
            artifact = SourceArtifact(
                source="yahoo_finance",
                source_url=f"yfinance://download?symbols={','.join(batch)}&start={start}&end={end}",
                retrieved_at=retrieved_at,
                content=artifact_content,
                content_type="text/csv",
            )
            artifacts.append(artifact)
            for symbol in batch:
                frame = _extract_yahoo_symbol(raw, symbol, len(batch))
                if frame.empty or "Close" not in frame:
                    continue
                security_id = by_symbol[symbol]
                valid = frame.loc[frame["Close"].notna()].copy()
                if valid.empty:
                    continue
                seen.add(symbol)
                for timestamp, row in valid.iterrows():
                    prices.append(
                        {
                            "security_id": security_id,
                            "session": pd.Timestamp(timestamp).date().isoformat(),
                            "open": row.get("Open"),
                            "high": row.get("High"),
                            "low": row.get("Low"),
                            "close": row.get("Close"),
                            "volume": row.get("Volume", 0),
                            "currency": "USD",
                            "source": artifact.source,
                            "source_url": artifact.source_url,
                            "retrieved_at": retrieved_at,
                            "source_hash": artifact.source_hash,
                        }
                    )
                actions.extend(
                    _yahoo_actions(
                        valid,
                        security_id=security_id,
                        source_hash=artifact.source_hash,
                        retrieved_at=retrieved_at,
                    )
                )
        return YahooFetchResult(
            prices=pd.DataFrame(prices, columns=_PRICE_COLUMNS),
            corporate_actions=pd.DataFrame(actions, columns=_ACTION_COLUMNS),
            artifacts=tuple(artifacts),
            missing_symbols=tuple(sorted(set(symbols) - seen)),
        )


@dataclass(frozen=True)
class DailySyncResult:
    completed_session: str
    release_version: str
    versions: dict[str, str]
    row_counts: dict[str, int]
    missing_symbols: tuple[str, ...]
    warnings: tuple[str, ...]
    conflicts: tuple[str, ...]


class DailyDataSynchronizer:
    def __init__(
        self,
        repository: LocalDatasetRepository,
        *,
        security_source: SecurityMasterSource | None = None,
        price_source: YahooDailySource | None = None,
    ):
        self.repository = repository
        self.security_source = security_source or SecNasdaqSecurityMasterSource()
        self.price_source = price_source or YahooDailySource()

    def sync(
        self,
        expected_session: str,
        *,
        backfill_start: str = "2000-01-01",
        refresh_security_master: bool = False,
        overlap_days: int = 7,
    ) -> DailySyncResult:
        _, release_etag = self.repository.current_release()
        warnings: list[str] = []
        conflicts: list[str] = []
        versions: dict[str, str] = {}
        row_counts: dict[str, int] = {}
        master = self.repository.current_manifest("security_master")
        history = self.repository.current_manifest("symbol_history")
        if refresh_security_master or master is None or history is None:
            source_result = self.security_source.fetch()
            warnings.extend(source_result.warnings)
            master_result = self.repository.write_frame(
                "security_master",
                source_result.security_master,
                completed_session=expected_session,
                metadata={"operation": "security_master_refresh"},
            )
            history_frame = _merge_symbol_history(
                self._read_optional("symbol_history"),
                source_result.symbol_history,
                expected_session,
            )
            history_result = self.repository.write_frame(
                "symbol_history",
                history_frame,
                completed_session=expected_session,
                metadata={"operation": "symbol_history_refresh"},
            )
            self._record_result("security_master", master_result, versions, conflicts)
            self._record_result("symbol_history", history_result, versions, conflicts)
            self._archive(source_result.artifacts, expected_session, versions, conflicts)

        security_master = self.repository.read_frame("security_master")
        symbol_history = self.repository.read_frame("symbol_history")
        securities = _active_security_symbols(security_master, symbol_history, expected_session)
        current_prices = self.repository.current_manifest("daily_price_raw")
        if current_prices is None or not current_prices.completed_session:
            start = pd.Timestamp(backfill_start).date()
        else:
            start = pd.Timestamp(current_prices.completed_session).date() - timedelta(days=overlap_days)
        end = pd.Timestamp(expected_session).date()
        if start > end:
            start = end
        fetched = self.price_source.fetch(
            securities,
            start=start.isoformat(),
            end=end.isoformat(),
        )
        prices = fetched.prices.loc[
            pd.to_datetime(fetched.prices["session"], errors="coerce").dt.date <= end
        ] if not fetched.prices.empty else fetched.prices
        if prices.empty:
            raise RuntimeError(f"Yahoo returned no daily prices for {start} through {end}.")
        if current_prices is None:
            price_result = self.repository.write_frame(
                "daily_price_raw",
                prices,
                completed_session=expected_session,
                metadata={"operation": "initial_backfill", "start": start.isoformat()},
            )
        else:
            price_result = self.repository.append_frame(
                "daily_price_raw",
                prices,
                completed_session=expected_session,
                metadata={"overlap_days": overlap_days},
            )
        self._record_result("daily_price_raw", price_result, versions, conflicts)
        if price_result.conflict:
            raise RuntimeError(
                "Daily-price revision conflicted with a stored value; candidate was quarantined."
            )
        row_counts["daily_price_raw"] = len(prices)

        if not fetched.corporate_actions.empty:
            if self.repository.current_manifest("corporate_actions") is None:
                action_result = self.repository.write_frame(
                    "corporate_actions",
                    fetched.corporate_actions,
                    completed_session=expected_session,
                    incomplete_action_policy="warn",
                    metadata={"operation": "initial_actions"},
                )
            else:
                action_result = self.repository.append_frame(
                    "corporate_actions",
                    fetched.corporate_actions,
                    completed_session=expected_session,
                    incomplete_action_policy="warn",
                )
            self._record_result("corporate_actions", action_result, versions, conflicts)
            warnings.extend(
                issue.message
                for issue in action_result.validation.issues
                if issue.severity != "error"
            )
            row_counts["corporate_actions"] = len(fetched.corporate_actions)
        elif self.repository.current_manifest("corporate_actions") is None:
            action_result = self.repository.write_frame(
                "corporate_actions",
                pd.DataFrame(columns=_ACTION_COLUMNS),
                completed_session=expected_session,
                incomplete_action_policy="warn",
                metadata={"operation": "initial_actions", "empty": True},
            )
            self._record_result("corporate_actions", action_result, versions, conflicts)
            row_counts["corporate_actions"] = 0
            warnings.append("No corporate actions were returned; adjustment quality is degraded.")

        all_prices = self.repository.read_frame("daily_price_raw")
        all_actions = self._read_optional("corporate_actions")
        factors = build_adjustment_factors(
            all_prices,
            all_actions if not all_actions.empty else pd.DataFrame(columns=["security_id", "event_id", "action_type", "effective_date", "ex_date", "cash_amount", "ratio"]),
            source_version="+".join(
                filter(
                    None,
                    (
                        self.repository.current_manifest("daily_price_raw").version,
                        self.repository.current_manifest("corporate_actions").version
                        if self.repository.current_manifest("corporate_actions") else "no-actions",
                    ),
                )
            ),
        )
        factor_result = self.repository.write_frame(
            "adjustment_factors",
            factors,
            completed_session=expected_session,
            metadata={"operation": "rebuild_after_sync"},
        )
        self._record_result("adjustment_factors", factor_result, versions, conflicts)
        row_counts["adjustment_factors"] = len(factors)
        self._archive(fetched.artifacts, expected_session, versions, conflicts)
        if fetched.missing_symbols:
            warnings.append(f"Yahoo missing symbols: {len(fetched.missing_symbols)}")
        if conflicts:
            raise RuntimeError(
                "One or more current pointers changed during sync; candidate versions were quarantined."
            )
        cross_report = validate_repository_snapshot(self.repository)
        cross_report.raise_for_errors()
        warnings.extend(
            issue.message for issue in cross_report.issues if issue.severity != "error"
        )
        release_versions = {
            dataset: manifest.version
            for dataset in DATASET_SPECS
            if (manifest := self.repository.current_manifest(dataset)) is not None
        }
        release = self.repository.commit_release(
            expected_session,
            release_versions,
            quality=DataQuality.DEGRADED if warnings else DataQuality.VALID,
            warnings=tuple(dict.fromkeys(warnings)),
            expected_etag=release_etag,
        )
        return DailySyncResult(
            completed_session=expected_session,
            release_version=release.version,
            versions=versions,
            row_counts=row_counts,
            missing_symbols=fetched.missing_symbols,
            warnings=tuple(warnings),
            conflicts=tuple(conflicts),
        )

    def _archive(
        self,
        artifacts: tuple[SourceArtifact, ...],
        effective_date: str,
        versions: dict[str, str],
        conflicts: list[str],
    ) -> None:
        if not artifacts:
            return
        rows = []
        for artifact in artifacts:
            archive_id = artifact.source_hash
            content_type = artifact.content_type.lower()
            if "json" in content_type:
                extension = "json"
            elif "pdf" in content_type:
                extension = "pdf"
            elif "html" in content_type:
                extension = "html"
            elif "csv" in content_type:
                extension = "csv"
            else:
                extension = "txt"
            object_path = f"archives/{effective_date}/{archive_id}.{extension}.gz"
            write_atomic(self.repository.root / object_path, gzip.compress(artifact.content))
            rows.append(
                {
                    "archive_id": archive_id,
                    "dataset": artifact.source,
                    "object_path": object_path,
                    "content_type": artifact.content_type,
                    "effective_date": effective_date,
                    "source": artifact.source,
                    "source_url": artifact.source_url,
                    "retrieved_at": artifact.retrieved_at,
                    "source_hash": artifact.source_hash,
                }
            )
        frame = pd.DataFrame(rows).drop_duplicates("archive_id")
        if self.repository.current_manifest("source_archive") is None:
            result = self.repository.write_frame(
                "source_archive",
                frame,
                completed_session=effective_date,
                metadata={"operation": "initial_archive_index"},
            )
        else:
            result = self.repository.append_frame(
                "source_archive",
                frame,
                completed_session=effective_date,
            )
        self._record_result("source_archive", result, versions, conflicts)

    def _read_optional(self, dataset: str) -> pd.DataFrame:
        return (
            self.repository.read_frame(dataset)
            if self.repository.current_manifest(dataset) is not None
            else pd.DataFrame()
        )

    @staticmethod
    def _record_result(
        dataset: str,
        result: DatasetWriteResult,
        versions: dict[str, str],
        conflicts: list[str],
    ) -> None:
        versions[dataset] = result.manifest.version
        if result.conflict:
            conflicts.append(result.conflict_path)


def _parse_sec_companies(artifact: SourceArtifact) -> list[dict]:
    payload = json.loads(artifact.content)
    fields = payload.get("fields", [])
    rows = payload.get("data", [])
    output = []
    for values in rows:
        row = dict(zip(fields, values))
        symbol = str(row.get("ticker") or row.get("Ticker") or "").strip().replace(".", "-")
        cik = row.get("cik") or row.get("CIK")
        if not symbol or cik is None:
            continue
        output.append(
            {
                "security_id": f"US:CIK:{int(cik):010d}",
                "primary_symbol": symbol,
                "name": str(row.get("name") or row.get("Name") or ""),
                "exchange": str(row.get("exchange") or row.get("Exchange") or "US"),
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "active_from": "1900-01-01",
                "active_to": "",
                "source": artifact.source,
                "source_url": artifact.source_url,
                "retrieved_at": artifact.retrieved_at,
                "source_hash": artifact.source_hash,
            }
        )
    return output


def _parse_listing_directory(artifact: SourceArtifact, *, exchange: str) -> list[dict]:
    text = artifact.content.decode("utf-8", errors="replace")
    frame = pd.read_csv(io.StringIO(text), sep="|")
    frame = frame.loc[~frame.iloc[:, 0].astype(str).str.startswith("File Creation Time")]
    symbol_column = "Symbol" if "Symbol" in frame else "ACT Symbol"
    name_column = "Security Name"
    output = []
    for row in frame.to_dict("records"):
        symbol = str(row.get(symbol_column) or "").strip().replace(".", "-")
        if not symbol or str(row.get("Test Issue", "N")).upper() == "Y":
            continue
        row_exchange = exchange
        if exchange == "OTHER":
            row_exchange = {"N": "NYSE", "A": "NYSEAMERICAN", "P": "NYSEARCA", "Z": "BATS", "V": "IEX"}.get(
                str(row.get("Exchange") or ""),
                "OTHER",
            )
        output.append(
            {
                "security_id": "",
                "primary_symbol": symbol,
                "name": str(row.get(name_column) or ""),
                "exchange": row_exchange,
                "asset_type": "ETF" if str(row.get("ETF", "N")).upper() == "Y" else "STOCK",
                "currency": "USD",
                "country": "US",
                "active_from": "1900-01-01",
                "active_to": "",
                "source": artifact.source,
                "source_url": artifact.source_url,
                "retrieved_at": artifact.retrieved_at,
                "source_hash": artifact.source_hash,
            }
        )
    return output


def _extract_yahoo_symbol(raw: pd.DataFrame, symbol: str, batch_count: int) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)
        if symbol in level0:
            return raw[symbol].copy()
        if symbol in level1:
            return raw.xs(symbol, axis=1, level=1).copy()
        return pd.DataFrame()
    return raw.copy() if batch_count == 1 else pd.DataFrame()


def _yahoo_actions(
    frame: pd.DataFrame,
    *,
    security_id: str,
    source_hash: str,
    retrieved_at: str,
) -> list[dict]:
    output = []
    for timestamp, row in frame.iterrows():
        effective = pd.Timestamp(timestamp).date().isoformat()
        dividend = float(row.get("Dividends", 0) or 0)
        split = float(row.get("Stock Splits", 0) or 0)
        if dividend:
            output.append(
                _action_record(
                    security_id,
                    "cash_dividend",
                    effective,
                    cash_amount=dividend,
                    ratio=None,
                    source_hash=source_hash,
                    retrieved_at=retrieved_at,
                )
            )
        if split:
            output.append(
                _action_record(
                    security_id,
                    "split",
                    effective,
                    cash_amount=None,
                    ratio=split,
                    source_hash=source_hash,
                    retrieved_at=retrieved_at,
                )
            )
    return output


def _action_record(
    security_id: str,
    action_type: str,
    effective_date: str,
    *,
    cash_amount: float | None,
    ratio: float | None,
    source_hash: str,
    retrieved_at: str,
) -> dict:
    event_key = f"{security_id}|{action_type}|{effective_date}".encode()
    return {
        "event_id": hashlib.sha256(event_key).hexdigest(),
        "security_id": security_id,
        "action_type": action_type,
        "effective_date": effective_date,
        "ex_date": effective_date,
        "announcement_date": "",
        "record_date": "",
        "payment_date": "",
        "cash_amount": cash_amount,
        "ratio": ratio,
        "currency": "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": False,
        "source": "yahoo_finance",
        "source_url": "https://finance.yahoo.com/",
        "source_kind": "provider",
        "retrieved_at": retrieved_at,
        "source_hash": source_hash,
    }


def _merge_symbol_history(existing: pd.DataFrame, latest: pd.DataFrame, effective_date: str) -> pd.DataFrame:
    if existing.empty:
        return latest.copy()
    output = existing.copy()
    for row in latest.to_dict("records"):
        security_id = str(row["security_id"])
        current = output.loc[
            (output["security_id"].astype(str) == security_id)
            & (output["effective_to"].fillna("").astype(str) == "")
        ]
        if current.empty:
            row["effective_from"] = effective_date
            output = pd.concat([output, pd.DataFrame([row])], ignore_index=True)
            continue
        prior = current.sort_values("effective_from").iloc[-1]
        if str(prior["symbol"]) == str(row["symbol"]):
            continue
        prior_index = prior.name
        output.loc[prior_index, "effective_to"] = (
            pd.Timestamp(effective_date) - pd.Timedelta(days=1)
        ).date().isoformat()
        row["effective_from"] = effective_date
        output = pd.concat([output, pd.DataFrame([row])], ignore_index=True)
    return output


def _active_security_symbols(
    master: pd.DataFrame,
    history: pd.DataFrame,
    as_of: str,
) -> dict[str, str]:
    cutoff = pd.Timestamp(as_of)
    starts = pd.to_datetime(history["effective_from"], errors="coerce")
    ends = pd.to_datetime(history["effective_to"], errors="coerce")
    active = history.loc[(starts <= cutoff) & (ends.isna() | (ends >= cutoff))].copy()
    allowed = master.loc[
        master["asset_type"].astype(str).str.upper().isin({"STOCK", "ETF"}),
        "security_id",
    ].astype(str)
    active = active.loc[active["security_id"].astype(str).isin(set(allowed))]
    active = active.sort_values("effective_from").drop_duplicates("security_id", keep="last")
    return dict(zip(active["security_id"].astype(str), active["symbol"].astype(str)))


_PRICE_COLUMNS = (
    "security_id",
    "session",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "currency",
    "source",
    "retrieved_at",
    "source_hash",
)

_ACTION_COLUMNS = (
    "event_id",
    "security_id",
    "action_type",
    "effective_date",
    "ex_date",
    "announcement_date",
    "record_date",
    "payment_date",
    "cash_amount",
    "ratio",
    "currency",
    "new_security_id",
    "new_symbol",
    "official",
    "source",
    "source_url",
    "source_kind",
    "retrieved_at",
    "source_hash",
)
