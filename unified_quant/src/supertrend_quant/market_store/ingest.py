from __future__ import annotations

import fcntl
import gzip
import hashlib
import io
import json
import os
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Protocol

import pandas as pd

from ..env import load_env
from .adjustments import build_adjustment_factors
from .manifest import sha256_bytes, utc_now_iso, write_atomic
from .models import DataQuality
from .operational_validation import validate_operational_repository_snapshot
from .repository import DatasetWriteResult, LocalDatasetRepository
from .schemas import DATASET_SPECS


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


_EODHD_BUDGET_LOCK = Lock()


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
        load_env()
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
        price_frame = pd.DataFrame(prices, columns=_PRICE_COLUMNS)
        if not price_frame.empty:
            numeric = price_frame[["open", "high", "low", "close"]].apply(
                pd.to_numeric, errors="coerce"
            )
            positive = numeric.notna().all(axis=1) & numeric.gt(0).all(axis=1)
            try:
                import exchange_calendars as xcals

                sessions = xcals.get_calendar("XNYS").sessions_in_range(start, end)
                valid_sessions = {pd.Timestamp(value).date().isoformat() for value in sessions}
                calendar_valid = price_frame["session"].astype(str).isin(valid_sessions)
            except Exception:
                calendar_valid = pd.Series(True, index=price_frame.index)
            price_frame = price_frame.loc[positive & calendar_valid].copy()
        return YahooFetchResult(
            prices=price_frame,
            corporate_actions=pd.DataFrame(actions, columns=_ACTION_COLUMNS),
            artifacts=tuple(artifacts),
            missing_symbols=tuple(sorted(set(symbols) - seen)),
        )


class EodhdQuotaExceeded(RuntimeError):
    """Raised before an HTTP attempt would cross the configured safety ceiling."""


class EodhdCallBudget:
    """Persistent, thread-safe guard around actual EODHD HTTP attempts.

    Failed attempts are counted because providers can charge them.  The
    operator-supplied seed allows the local ledger to start from the usage
    already shown by EODHD instead of assuming a fresh allowance.
    """

    def __init__(
        self,
        state_path: str | Path | None = None,
        *,
        limit: int | None = None,
        reserve: int | None = None,
        seed_used: int | None = None,
        period: str | None = None,
    ):
        self.state_path = Path(
            state_path
            or os.getenv(
                "EODHD_API_CALL_STATE_FILE",
                "data/cache/state/eodhd_call_budget.json",
            )
        )
        self.limit = int(
            limit if limit is not None else os.getenv("EODHD_API_DAILY_LIMIT", "100000")
        )
        self.reserve = int(
            reserve if reserve is not None else os.getenv("EODHD_API_CALL_RESERVE", "5000")
        )
        self.seed_used = int(
            seed_used if seed_used is not None else os.getenv("EODHD_API_CALLS_USED", "0")
        )
        configured_period = (
            str(period).strip()
            if period is not None
            else os.getenv("EODHD_API_BUDGET_PERIOD", "").strip()
        )
        # EODHD resets paid-plan daily limits at midnight GMT/UTC.  Using the
        # host's local date can reset this guard early in positive-offset time
        # zones (for example, Asia/Seoul), so the implicit period is UTC.
        self.period = configured_period or datetime.now(UTC).date().isoformat()
        if self.limit <= 0:
            raise ValueError("EODHD_API_DAILY_LIMIT must be positive.")
        if not 0 <= self.reserve < self.limit:
            raise ValueError("EODHD_API_CALL_RESERVE must be between 0 and the daily limit.")
        if self.seed_used < 0:
            raise ValueError("EODHD_API_CALLS_USED cannot be negative.")

    @property
    def ceiling(self) -> int:
        return self.limit - self.reserve

    def claim(self) -> int:
        """Reserve one HTTP attempt and return the guarded period usage."""

        with _EODHD_BUDGET_LOCK:
            lock_path = self.state_path.with_name(self.state_path.name + ".lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+b") as lock_handle:
                # The in-process lock above protects threads.  This advisory
                # lock protects the persistent read-modify-write transaction
                # when two collectors run in separate processes.
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    used = self.seed_used
                    if self.state_path.exists():
                        try:
                            current = json.loads(
                                self.state_path.read_text(encoding="utf-8")
                            )
                        except (OSError, ValueError, TypeError) as exc:
                            raise RuntimeError(
                                "EODHD call budget state is unreadable; request refused."
                            ) from exc
                        if not isinstance(current, dict):
                            raise RuntimeError(
                                "EODHD call budget state is invalid; request refused."
                            )
                        if str(current.get("period", "")) == self.period:
                            try:
                                persisted_used = int(current["used"])
                            except (KeyError, TypeError, ValueError) as exc:
                                raise RuntimeError(
                                    "EODHD call budget usage is invalid; request refused."
                                ) from exc
                            if persisted_used < 0:
                                raise RuntimeError(
                                    "EODHD call budget usage is negative; request refused."
                                )
                            used = max(used, persisted_used)
                    if used >= self.ceiling:
                        raise EodhdQuotaExceeded(
                            "EODHD call budget stopped the request: "
                            f"period={self.period}, used={used}, "
                            f"safety_ceiling={self.ceiling}, "
                            f"daily_limit={self.limit}, reserve={self.reserve}."
                        )
                    used += 1
                    write_atomic(
                        self.state_path,
                        json.dumps(
                            {
                                "period": self.period,
                                "used": used,
                                "daily_limit": self.limit,
                                "reserve": self.reserve,
                                "updated_at": utc_now_iso(),
                            },
                            sort_keys=True,
                            indent=2,
                        ).encode(),
                    )
                    return used
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


class EodhdClient:
    """Small EODHD client whose errors and provenance never expose the API token."""

    base_url = "https://eodhd.com/api"

    def __init__(
        self,
        session=None,
        *,
        token: str | None = None,
        budget: EodhdCallBudget | None = None,
    ):
        load_env()
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.token = token or os.getenv("EODHD_API_TOKEN", "")
        if not self.token:
            raise RuntimeError("EODHD_API_TOKEN is required.")
        self.budget = budget or EodhdCallBudget()

    def get_json(self, endpoint: str, *, params: dict[str, object] | None = None):
        safe_endpoint = "/" + endpoint.strip("/")
        query = {**(params or {}), "api_token": self.token, "fmt": "json"}
        last_error = "request error"
        for attempt in range(4):
            try:
                self.budget.claim()
                response = self.session.get(
                    self.base_url + safe_endpoint,
                    params=query,
                    timeout=120,
                )
                response.raise_for_status()
                return response.json()
            except EodhdQuotaExceeded:
                raise
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                last_error = f"HTTP {status}" if status else type(exc).__name__
                if attempt < 3:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"EODHD request failed for {safe_endpoint}: {last_error}") from None

    def safe_url(self, endpoint: str, *, params: dict[str, object] | None = None) -> str:
        from urllib.parse import urlencode

        safe_endpoint = "/" + endpoint.strip("/")
        return self.base_url + safe_endpoint + ("?" + urlencode(params) if params else "")


class EodhdDailySource:
    """Fetch raw EOD OHLCV, cash dividends, and splits from EODHD."""

    def __init__(
        self,
        client: EodhdClient | None = None,
        *,
        workers: int = 8,
        action_symbol_overrides: dict[str, str] | None = None,
    ):
        self.client = client or EodhdClient()
        self.workers = max(1, workers)
        self.action_symbol_overrides = dict(action_symbol_overrides or {})

    def fetch(
        self,
        securities: dict[str, str],
        *,
        start: str,
        end: str,
        batch_size: int = 100,
    ) -> YahooFetchResult:
        prices: list[dict] = []
        actions: list[dict] = []
        artifacts: list[SourceArtifact] = []
        missing: list[str] = []
        worker_count = min(self.workers, max(1, batch_size), max(1, len(securities)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(self._fetch_one, security_id, provider_symbol, start, end): provider_symbol
                for security_id, provider_symbol in securities.items()
            }
            for future in as_completed(futures):
                provider_symbol = futures[future]
                try:
                    result = future.result()
                except RuntimeError:
                    missing.append(provider_symbol)
                    continue
                if not result[0]:
                    missing.append(provider_symbol)
                prices.extend(result[0])
                actions.extend(result[1])
                artifacts.extend(result[2])
        price_frame = pd.DataFrame(prices, columns=_PRICE_COLUMNS)
        if not price_frame.empty:
            numeric = price_frame[["open", "high", "low", "close"]].apply(
                pd.to_numeric, errors="coerce"
            )
            positive = numeric.notna().all(axis=1) & numeric.gt(0).all(axis=1)
            try:
                import exchange_calendars as xcals

                sessions = xcals.get_calendar("XNYS").sessions_in_range(start, end)
                valid_sessions = {pd.Timestamp(value).date().isoformat() for value in sessions}
                calendar_valid = price_frame["session"].astype(str).isin(valid_sessions)
            except Exception:
                calendar_valid = pd.Series(True, index=price_frame.index)
            price_frame = price_frame.loc[positive & calendar_valid].copy()
        return YahooFetchResult(
            prices=price_frame,
            corporate_actions=pd.DataFrame(actions, columns=_ACTION_COLUMNS),
            artifacts=tuple(artifacts),
            missing_symbols=tuple(sorted(missing)),
        )

    def _fetch_one(self, security_id: str, provider_symbol: str, start: str, end: str):
        retrieved_at = utc_now_iso()
        params = {"from": start, "to": end}
        endpoint_rows = {}
        artifacts = []
        for endpoint in ("eod", "div", "splits"):
            endpoint_symbol = (
                provider_symbol
                if endpoint == "eod"
                else self.action_symbol_overrides.get(security_id, provider_symbol)
            )
            path = f"{endpoint}/{endpoint_symbol}"
            rows = self.client.get_json(path, params=params)
            if not isinstance(rows, list):
                rows = []
            content = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
            artifact = SourceArtifact(
                source=f"eodhd_{endpoint}",
                source_url=self.client.safe_url(path, params=params),
                retrieved_at=retrieved_at,
                content=content,
                content_type="application/json",
            )
            endpoint_rows[endpoint] = (rows, artifact)
            artifacts.append(artifact)

        eod_rows, eod_artifact = endpoint_rows["eod"]
        prices = [
            {
                "security_id": security_id,
                "session": str(row.get("date") or ""),
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume", 0),
                "currency": "USD",
                "source": eod_artifact.source,
                "source_url": eod_artifact.source_url,
                "retrieved_at": retrieved_at,
                "source_hash": eod_artifact.source_hash,
            }
            for row in eod_rows
            if row.get("date") and row.get("close") is not None
        ]
        actions = []
        dividend_rows, dividend_artifact = endpoint_rows["div"]
        for row in dividend_rows:
            effective = str(row.get("date") or "")
            if not effective:
                continue
            actions.append(
                _provider_action_record(
                    security_id,
                    "cash_dividend",
                    effective,
                    cash_amount=row.get("unadjustedValue", row.get("value")),
                    ratio=None,
                    source=dividend_artifact.source,
                    source_url=dividend_artifact.source_url,
                    source_hash=dividend_artifact.source_hash,
                    retrieved_at=retrieved_at,
                    announcement_date=str(row.get("declarationDate") or ""),
                    record_date=str(row.get("recordDate") or ""),
                    payment_date=str(row.get("paymentDate") or ""),
                    currency=str(row.get("currency") or "USD"),
                )
            )
        split_rows, split_artifact = endpoint_rows["splits"]
        for row in split_rows:
            effective = str(row.get("date") or "")
            ratio = _parse_eodhd_split(row.get("split"))
            if not effective or ratio is None:
                continue
            actions.append(
                _provider_action_record(
                    security_id,
                    "split",
                    effective,
                    cash_amount=None,
                    ratio=ratio,
                    source=split_artifact.source,
                    source_url=split_artifact.source_url,
                    source_hash=split_artifact.source_hash,
                    retrieved_at=retrieved_at,
                )
            )
        return prices, actions, tuple(artifacts)


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
        price_source: YahooDailySource | EodhdDailySource | None = None,
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
        current_prices = self.repository.current_manifest("daily_price_raw")
        if current_prices is None and "provider_symbol" in security_master:
            securities = dict(
                zip(
                    security_master["security_id"].astype(str),
                    security_master["provider_symbol"].astype(str),
                )
            )
        else:
            securities = _active_security_symbols(security_master, symbol_history, expected_session)
            if "provider_symbol" in security_master:
                provider_by_id = dict(
                    zip(
                        security_master["security_id"].astype(str),
                        security_master["provider_symbol"].astype(str),
                    )
                )
                securities = {
                    security_id: provider_by_id.get(security_id, symbol)
                    for security_id, symbol in securities.items()
                }
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
            raise RuntimeError(f"Market-data provider returned no daily prices for {start} through {end}.")
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
            warnings.append(f"Market-data provider missing symbols: {len(fetched.missing_symbols)}")
        if conflicts:
            raise RuntimeError(
                "One or more current pointers changed during sync; candidate versions were quarantined."
            )
        cross_report = validate_operational_repository_snapshot(self.repository)
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


def configured_daily_synchronizer(repository: LocalDatasetRepository, ingest_source: str):
    source = str(ingest_source).strip().lower()
    if source == "eodhd":
        master = (
            repository.read_frame("security_master")
            if repository.current_manifest("security_master") is not None
            else pd.DataFrame()
        )
        action_symbols = {}
        if "action_provider_symbol" in master:
            action_symbols = {
                str(row.security_id): str(row.action_provider_symbol)
                for row in master.itertuples(index=False)
                if pd.notna(row.action_provider_symbol)
                and str(row.action_provider_symbol).strip()
            }
        return DailyDataSynchronizer(
            repository,
            price_source=EodhdDailySource(action_symbol_overrides=action_symbols),
        )
    if source == "yahoo":
        return DailyDataSynchronizer(repository, price_source=YahooDailySource())
    raise ValueError(f"Unsupported ingest source: {ingest_source}")


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


def _parse_eodhd_split(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            denominator_value = float(denominator)
            return float(numerator) / denominator_value if denominator_value else None
        parsed = float(text)
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


def _provider_action_record(
    security_id: str,
    action_type: str,
    effective_date: str,
    *,
    cash_amount,
    ratio,
    source: str,
    source_url: str,
    source_hash: str,
    retrieved_at: str,
    announcement_date: str = "",
    record_date: str = "",
    payment_date: str = "",
    currency: str = "USD",
) -> dict:
    event_key = f"{source}|{security_id}|{action_type}|{effective_date}".encode()
    return {
        "event_id": hashlib.sha256(event_key).hexdigest(),
        "security_id": security_id,
        "action_type": action_type,
        "effective_date": effective_date,
        "ex_date": effective_date,
        "announcement_date": announcement_date,
        "record_date": record_date,
        "payment_date": payment_date,
        "cash_amount": cash_amount,
        "ratio": ratio,
        "currency": currency,
        "new_security_id": "",
        "new_symbol": "",
        "official": False,
        "source": source,
        "source_url": source_url,
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
    master_starts = pd.to_datetime(master["active_from"], errors="coerce")
    master_ends = pd.to_datetime(master["active_to"], errors="coerce")
    allowed = master.loc[
        master["asset_type"].astype(str).str.upper().isin({"STOCK", "ETF"})
        & (master_starts <= cutoff)
        & (master_ends.isna() | (master_ends >= cutoff)),
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
