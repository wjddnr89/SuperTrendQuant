from __future__ import annotations

import io
import gzip
import json
import os
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from ..env import load_env
from .ingest import EodhdClient, SourceArtifact
from .manifest import sha256_bytes, utc_now_iso, write_atomic


KR_INDEX_DEFINITIONS = {
    "kospi200": {
        "market": "KOSPI",
        "name": "코스피 200",
        "index_group": "1",
        "index_code": "028",
        "announcement_date": "1994-06-15",
        "expected_count": 200,
        "count_tolerance": 5,
    },
    "kosdaq150": {
        "market": "KOSDAQ",
        "name": "코스닥 150",
        "index_group": "2",
        "index_code": "203",
        "announcement_date": "2015-07-13",
        "expected_count": 150,
        "count_tolerance": 5,
    },
}
KR_BENCHMARK_SECURITIES = {
    "kospi200": "069500",  # KODEX 200
    "kosdaq150": "229200",  # KODEX KOSDAQ 150
}

KRX_SOURCE_URL = "https://data.krx.co.kr/"
KRX_OPENAPI_SOURCE_URL = "https://openapi.krx.co.kr/"
KRX_OPENAPI_BASE_URL = "https://data-dbg.krx.co.kr/svc/apis"
KRX_INDEX_LICENSE_URL = "https://openapi.krx.co.kr/contents/OPP/DATA/OPPDATA005.jsp"
KRX_WEB_LOGIN_PAGE = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
KRX_WEB_LOGIN_JSP = (
    "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
)
KRX_WEB_LOGIN_URL = (
    "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"
)
KRX_WEB_DATA_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
KRX_WEB_INDEX_BLD = "dbms/MDC/STAT/standard/MDCSTAT00601"
KRX_WEB_DELISTED_FINDER_BLD = "dbms/comm/finder/finder_listdelisu"
KRX_WEB_STOCK_DAILY_BLD = "dbms/MDC/STAT/standard/MDCSTAT01501"
KRX_WEB_ETF_MASTER_BLD = "dbms/MDC/STAT/standard/MDCSTAT04601"
KRX_WEB_ETF_DAILY_BLD = "dbms/MDC/STAT/standard/MDCSTAT04301"
KRX_OPENAPI_ETF_DAILY_ENDPOINT = "etp/etf_bydd_trd"
KRX_WEB_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
NAVER_SOURCE_URL = "https://finance.naver.com/"
YAHOO_SOURCE_URL = "https://finance.yahoo.com/"
KIS_SOURCE_URL = "https://apiportal.koreainvestment.com/"
DART_CORP_CODE_URL = "https://opendart.fss.or.kr/api/corpCode.xml"
DART_DISCLOSURE_LIST_URL = "https://opendart.fss.or.kr/api/list.json"
DART_DOCUMENT_URL = "https://opendart.fss.or.kr/api/document.xml"
DART_FILING_URL = "https://dart.fss.or.kr/dsaf001/main.do"
KIS_PROD_BASE_URL = "https://openapi.koreainvestment.com:9443"
KIS_PAPER_BASE_URL = "https://openapivts.koreainvestment.com:29443"
_KIS_TOKEN_LOCK = threading.Lock()
_KRX_WEB_LOCK = threading.Lock()
_KRX_WEB_SESSION = None
_KRX_WEB_SESSION_CREDENTIALS = ""
_KRX_WEB_LOGIN_AT = 0.0
_KRX_WEB_LAST_USED_AT = 0.0
_KRX_WEB_CACHE_SAVED_AT = 0.0
_KRX_WEB_LOGIN_FAILURE_UNTIL = 0.0
_KRX_WEB_LOGIN_FAILURE_DETAIL = ""
# Active requests prove that the cloned login cookie is still usable. Avoid
# periodic re-login while a long bootstrap is running; an actual LOGOUT/HTTP
# authentication response still triggers an immediate generation-aware refresh.
_KRX_WEB_SESSION_TTL_SECONDS = 30 * 60
_KRX_WEB_CACHE_WRITE_INTERVAL_SECONDS = 30.0
_KRX_WEB_LOGIN_FAILURE_COOLDOWN_SECONDS = 60.0
_KRX_WEB_SESSION_CACHE_SCHEMA_VERSION = 1
_KRX_ETF_OPENAPI_LOCK = threading.Lock()
_KRX_ETF_OPENAPI_AVAILABLE: bool | None = None
_KRX_STOCK_OPENAPI_LOCK = threading.Lock()
_KRX_STOCK_OPENAPI_AVAILABLE = True
_KRX_OPENAPI_HTTP_LOCAL = threading.local()


class KrOfficialDataUnavailable(RuntimeError):
    """Raised when a required authenticated/licensed KRX input is absent."""


@dataclass(frozen=True)
class KrProviderResult:
    provider: str
    status: str
    prices: pd.DataFrame
    artifacts: tuple[SourceArtifact | "KrArtifactReference", ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class KrArtifactReference:
    """Hash-only reference for a raw artifact already saved locally."""

    source_hash: str


@dataclass(frozen=True)
class KrDartDividendResult:
    """Official cash-dividend decisions and their complete filing audit."""

    status: str
    decisions: pd.DataFrame
    report: SourceArtifact
    detail: str = ""


@dataclass(frozen=True)
class KrIdentityCatalog:
    frame: pd.DataFrame
    artifacts: tuple[SourceArtifact, ...]
    dart_status: str = "not_requested"

    def security_id_for(self, symbol: str, session: str) -> str:
        values = self.frame.loc[
            self.frame["primary_symbol"].astype(str).eq(normalize_kr_symbol(symbol))
        ].copy()
        if values.empty:
            raise KeyError(f"No KRX identity for {symbol} on {session}.")
        when = pd.Timestamp(session).normalize()
        starts = pd.to_datetime(values["active_from"], errors="coerce")
        ends = pd.to_datetime(values["active_to"], errors="coerce")
        active = values.loc[
            (starts.isna() | starts.le(when))
            & (ends.isna() | ends.ge(when))
        ]
        security_ids = active["security_id"].astype(str).drop_duplicates()
        if len(security_ids) != 1:
            raise KeyError(
                "Expected one KRX security identity for "
                f"{symbol} on {when.date()}, found {len(security_ids)}."
            )
        return str(security_ids.iloc[0])

    def row_for(self, symbol: str, session: str) -> pd.Series:
        security_id = self.security_id_for(symbol, session)
        values = self.frame.loc[
            self.frame["security_id"].astype(str).eq(security_id)
            & self.frame["primary_symbol"].astype(str).eq(normalize_kr_symbol(symbol))
        ].copy()
        when = pd.Timestamp(session).normalize()
        starts = pd.to_datetime(values["active_from"], errors="coerce")
        ends = pd.to_datetime(values["active_to"], errors="coerce")
        active = values.loc[
            (starts.isna() | starts.le(when))
            & (ends.isna() | ends.ge(when))
        ]
        if len(active) != 1:
            raise KeyError(
                f"Expected one KRX catalog row for {symbol} on {session}, "
                f"found {len(active)}."
            )
        return active.iloc[0]

    def active_rows_for(
        self,
        symbols: Iterable[str],
        session: str,
    ) -> dict[str, dict]:
        """Resolve a session's selected identities with one vectorized scan."""

        selected = {normalize_kr_symbol(symbol) for symbol in symbols}
        if not selected or self.frame.empty:
            return {}
        values = self.frame.loc[
            self.frame["primary_symbol"].astype(str).isin(selected)
        ].copy()
        if "identity_mapped" in values:
            values = values.loc[values["identity_mapped"].eq(True)]  # noqa: E712
        when = pd.Timestamp(session).normalize()
        starts = pd.to_datetime(values["active_from"], errors="coerce")
        ends = pd.to_datetime(values["active_to"], errors="coerce")
        active = values.loc[
            (starts.isna() | starts.le(when))
            & (ends.isna() | ends.ge(when))
        ].copy()
        output: dict[str, dict] = {}
        for symbol, group in active.groupby("primary_symbol", sort=False):
            if group["security_id"].astype(str).nunique() != 1:
                continue
            ordered = group.assign(
                _active_from=pd.to_datetime(
                    group["active_from"],
                    errors="coerce",
                )
            ).sort_values("_active_from", kind="stable", na_position="first")
            row = ordered.iloc[-1].drop(labels=["_active_from"]).to_dict()
            output[str(symbol)] = row
        return output

    def interval_is_terminal(self, identity: dict, session: str) -> bool:
        """Return whether an identity interval has no later same-ISIN interval."""

        security_id = str(identity.get("security_id") or "").strip()
        if not security_id or self.frame.empty:
            return True
        when = pd.Timestamp(session).normalize()
        same_identity = self.frame.loc[
            self.frame["security_id"].astype(str).eq(security_id)
        ]
        later_starts = pd.to_datetime(
            same_identity["active_from"],
            errors="coerce",
        ).gt(when)
        return not bool(later_starts.any())


def canonical_json_bytes(value) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _valid_sha256_text(value: object) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def normalize_kr_symbol(value) -> str:
    """Normalize KRX six-character numeric or alphanumeric short codes."""

    text = str(value or "").strip().upper()
    return text.zfill(6) if text else ""


def is_valid_kr_symbol(value) -> bool:
    symbol = normalize_kr_symbol(value)
    return len(symbol) == 6 and all(
        character in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for character in symbol
    )


def validate_krx_official_configuration(*, require_membership: bool = True) -> Path | None:
    """Validate authenticated KRX price and constituent inputs.

    KRX Open API prices require an ``AUTH_KEY``. Historical constituent files
    can come from an audited licensed snapshot table or from the authenticated
    KRX Data Marketplace historical constituent screen.
    """

    load_env()
    if not os.getenv("KRX_OPENAPI_AUTH_KEY", "").strip():
        raise KrOfficialDataUnavailable(
            "KRX_OPENAPI_AUTH_KEY is required for canonical KRX daily prices. "
            "Apply for the KRX Open API and keep the key in .env."
        )
    if not require_membership:
        return None
    if _krx_web_credentials(required=False) is not None:
        return None
    return _licensed_constituent_path()


def _krx_web_credentials(*, required: bool) -> tuple[str, str] | None:
    login_id = os.getenv("KRX_ID", "").strip()
    login_pw = os.getenv("KRX_PW", "").strip()
    if bool(login_id) != bool(login_pw):
        raise KrOfficialDataUnavailable(
            "KRX_ID and KRX_PW must either both be set or both be empty."
        )
    if login_id and login_pw:
        return login_id, login_pw
    if required:
        raise KrOfficialDataUnavailable(
            "KRX_ID and KRX_PW are required for authenticated historical "
            "KOSPI200/KOSDAQ150 constituent snapshots."
        )
    return None


def _configured_licensed_constituent_path() -> Path | None:
    raw_path = os.getenv("KRX_PIT_CONSTITUENTS_PATH", "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    return path if path.is_file() else None


def _licensed_constituent_path() -> Path:
    load_env()
    path = _configured_licensed_constituent_path()
    if path is None:
        raw_path = os.getenv("KRX_PIT_CONSTITUENTS_PATH", "").strip()
        suffix = f": {raw_path}" if raw_path else ""
        raise KrOfficialDataUnavailable(
            "Historical KRX constituents require either KRX_ID + KRX_PW or an "
            "existing KRX_PIT_CONSTITUENTS_PATH file"
            + suffix
        )
    return path


def _licensed_constituent_table() -> tuple[pd.DataFrame, bytes, Path]:
    path = _licensed_constituent_path()
    stat = path.stat()
    frame, content = _load_licensed_constituent_table(
        str(path.resolve()), stat.st_mtime_ns, stat.st_size
    )
    return frame.copy(), content, path


@lru_cache(maxsize=4)
def _load_licensed_constituent_table(
    path_text: str,
    _mtime_ns: int,
    _size: int,
) -> tuple[pd.DataFrame, bytes]:
    path = Path(path_text)
    content = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path, dtype=str)
    elif suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            frame = pd.read_json(path, lines=True, dtype=False)
        else:
            value = json.loads(content.decode("utf-8"))
            if isinstance(value, dict):
                value = value.get("rows", value.get("constituents", value))
            frame = pd.DataFrame(value)
    else:
        raise KrOfficialDataUnavailable(
            "KRX_PIT_CONSTITUENTS_PATH must end in .csv, .json, .jsonl, or .parquet."
        )
    if frame is None or frame.empty:
        raise KrOfficialDataUnavailable("The licensed KRX constituent file is empty.")

    profile_column = _find_column(frame, ("profile", "index_id", "index"))
    session_column = _find_column(frame, ("session", "effective_date", "date", "기준일"))
    symbol_column = _find_column(
        frame, ("symbol", "ticker", "primary_symbol", "단축코드", "종목코드")
    )
    isin_column = _find_column(
        frame, ("isin", "security_id", "표준코드", "ISU_CD")
    )
    output = pd.DataFrame(
        {
            "profile": frame[profile_column].map(_normalize_profile),
            "session": frame[session_column].map(_date_text),
            "symbol": frame[symbol_column].map(normalize_kr_symbol),
            "isin": frame[isin_column].map(_normalize_isin),
        }
    )
    invalid = output.loc[
        ~output["profile"].isin(KR_INDEX_DEFINITIONS)
        | output["session"].eq("")
        | ~output["symbol"].str.fullmatch(r"[0-9A-Z]{6}")
        | ~output["isin"].str.fullmatch(r"[A-Z0-9]{12}")
    ]
    if not invalid.empty:
        raise KrOfficialDataUnavailable(
            "The licensed KRX constituent file has invalid profile/session/symbol/ISIN "
            f"rows: {len(invalid)}. Every member must carry a stable 12-character ISIN."
        )
    output = output.drop_duplicates(
        ["profile", "session", "symbol", "isin"], keep="last"
    ).sort_values(["profile", "session", "symbol"], kind="stable")
    symbol_conflicts = output.groupby(["profile", "session", "symbol"])["isin"].nunique()
    if symbol_conflicts.gt(1).any():
        raise KrOfficialDataUnavailable(
            "The licensed KRX constituent file maps one symbol to multiple ISINs "
            "within the same snapshot."
        )
    isin_conflicts = output.groupby(["profile", "session", "isin"])["symbol"].nunique()
    if isin_conflicts.gt(1).any():
        raise KrOfficialDataUnavailable(
            "The licensed KRX constituent file maps one ISIN to multiple symbols "
            "within the same snapshot."
        )
    from .markets import exchange_calendar

    calendar = exchange_calendar("KR")
    invalid_sessions = sorted(
        session
        for session in output["session"].drop_duplicates()
        if not calendar.is_session(pd.Timestamp(session))
    )
    if invalid_sessions:
        raise KrOfficialDataUnavailable(
            "The licensed KRX constituent file contains non-XKRX snapshot dates: "
            + ", ".join(invalid_sessions[:20])
        )
    return output.reset_index(drop=True), content


def _normalize_profile(value) -> str:
    text = "".join(str(value or "").lower().replace("_", "").split())
    aliases = {
        "kospi200": "kospi200",
        "코스피200": "kospi200",
        "kosdaq150": "kosdaq150",
        "코스닥150": "kosdaq150",
    }
    return aliases.get(text, text)


def _normalize_isin(value) -> str:
    text = str(value or "").strip().upper()
    return text.removeprefix("KR:")


def _licensed_constituent_identities(
    *,
    start: str,
    end: str,
    retrieved_at: str,
    source_hash: str,
) -> pd.DataFrame:
    table, _, _ = _licensed_constituent_table()
    # Include the latest lineage already in force at ``start``. Restricting
    # this to snapshots physically dated inside the request window would lose
    # identities during short smoke benchmarks between rebalance dates.
    selected = table.loc[table["session"].le(str(end))].copy()
    if selected.empty:
        return pd.DataFrame()
    selected["exchange"] = selected["profile"].map(
        {"kospi200": "KOSPI", "kosdaq150": "KOSDAQ"}
    )
    first_seen = (
        selected.groupby(["symbol", "isin", "exchange"], as_index=False)["session"]
        .min()
        .rename(columns={"session": "active_from"})
    )
    next_starts = first_seen.sort_values(
        ["symbol", "active_from", "isin"], kind="stable"
    ).copy()
    next_starts["next_start"] = next_starts.groupby("symbol")["active_from"].shift(-1)
    next_starts["active_to"] = pd.to_datetime(
        next_starts["next_start"], errors="coerce"
    ).sub(pd.Timedelta(days=1)).dt.date.astype("string").fillna("")
    rows = []
    for row in next_starts.to_dict("records"):
        symbol = str(row["symbol"])
        isin = str(row["isin"])
        exchange_name = str(row["exchange"])
        rows.append(
            {
                "security_id": f"KR:{isin}",
                "primary_symbol": symbol,
                "name": symbol,
                "exchange": exchange_name,
                "asset_type": "STOCK",
                "currency": "KRW",
                "country": "KR",
                "active_from": _date_text(row["active_from"]),
                "active_to": _date_text(row.get("active_to")),
                "isin": isin,
                "identity_mapped": True,
                "provider_symbol": f"{symbol}.{'KO' if exchange_name == 'KOSPI' else 'KQ'}",
                "yahoo_symbol": f"{symbol}.{'KS' if exchange_name == 'KOSPI' else 'KQ'}",
                "source": "krx_licensed_index_constituents",
                "source_url": KRX_INDEX_LICENSE_URL,
                "retrieved_at": retrieved_at,
                "source_hash": source_hash,
            }
        )
    return pd.DataFrame(rows)


def artifact_from_payload(
    source: str,
    source_url: str,
    payload,
    *,
    retrieved_at: str | None = None,
) -> SourceArtifact:
    return SourceArtifact(
        source=source,
        source_url=source_url,
        retrieved_at=retrieved_at or utc_now_iso(),
        content=canonical_json_bytes(payload),
        content_type="application/json",
    )


def fetch_kr_identity_catalog(
    *,
    start: str,
    end: str,
    include_dart: bool = True,
) -> KrIdentityCatalog:
    """Build a KRX ISIN catalog including delisted instruments.

    ``ISU_CD``/``ISIN`` is the stable identity. A missing standard code is
    retained with an explicit unmapped marker so the benchmark hard gate can
    fail with evidence rather than silently using a ticker as identity.
    """

    try:
        import FinanceDataReader as fdr
    except ModuleNotFoundError as exc:
        raise RuntimeError("finance-datareader is required for the KR identity catalog.") from exc

    load_env()
    retrieved_at = utc_now_iso()
    frames: list[pd.DataFrame] = []
    artifacts: list[SourceArtifact] = []
    requests = (
        ("KRX", lambda: fdr.StockListing("KRX")),
        (
            "KRX-DELISTING",
            lambda: fdr.StockListing("KRX-DELISTING", start, end),
        ),
    )
    for source_name, operation in requests:
        try:
            frame = operation()
        except Exception as exc:
            if source_name == "KRX-DELISTING":
                frame = pd.DataFrame()
            else:
                raise RuntimeError(
                    f"KRX identity listing failed: {type(exc).__name__}"
                ) from None
        if frame is None:
            frame = pd.DataFrame()
        frames.append(frame.copy())
        artifacts.append(
            artifact_from_payload(
                "krx_security_listing",
                KRX_SOURCE_URL,
                {
                    "request": {"listing": source_name, "start": start, "end": end},
                    "rows": _frame_records(frame),
                },
                retrieved_at=retrieved_at,
            )
        )

    normalized = pd.concat(
        [_normalize_listing(frame, retrieved_at, artifact.source_hash) for frame, artifact in zip(frames, artifacts)],
        ignore_index=True,
    )
    try:
        description_frame = fdr.StockListing("KRX-DESC")
    except Exception:
        description_frame = pd.DataFrame()
    if description_frame is None:
        description_frame = pd.DataFrame()
    description_artifact = artifact_from_payload(
        "krx_security_description_listing",
        KRX_SOURCE_URL,
        {
            "request": {"listing": "KRX-DESC"},
            "rows": _frame_records(description_frame),
        },
        retrieved_at=retrieved_at,
    )
    artifacts.append(description_artifact)
    normalized = _attach_current_listing_dates(
        normalized,
        description_frame,
    )
    if _krx_web_credentials(required=False) is not None:
        delisted_payload = _post_krx_web_json(
            {
                "bld": KRX_WEB_DELISTED_FINDER_BLD,
                "mktsel": "ALL",
                "searchText": "",
                "typeNo": "0",
            },
            sleep_seconds=0.0,
        )
        delisted_artifact = artifact_from_payload(
            "krx_authenticated_web_delisted_security_finder",
            KRX_SOURCE_URL,
            {
                "request": {
                    "bld": KRX_WEB_DELISTED_FINDER_BLD,
                    "mktsel": "ALL",
                    "searchText": "",
                    "typeNo": "0",
                },
                "response": delisted_payload,
            },
            retrieved_at=retrieved_at,
        )
        artifacts.append(delisted_artifact)
        normalized = _attach_delisted_isins(
            normalized,
            delisted_payload,
            retrieved_at=retrieved_at,
            source_hash=delisted_artifact.source_hash,
        )
        etf_payload = _post_krx_web_json(
            {"bld": KRX_WEB_ETF_MASTER_BLD},
            sleep_seconds=0.0,
        )
        etf_artifact = artifact_from_payload(
            "krx_authenticated_web_etf_master",
            KRX_SOURCE_URL,
            {
                "request": {"bld": KRX_WEB_ETF_MASTER_BLD},
                "response": etf_payload,
            },
            retrieved_at=retrieved_at,
        )
        artifacts.append(etf_artifact)
        etf_identities = _normalize_krx_etf_master(
            etf_payload,
            retrieved_at=retrieved_at,
            source_hash=etf_artifact.source_hash,
        )
        if not etf_identities.empty:
            etf_symbols = set(etf_identities["primary_symbol"].astype(str))
            # The stock listing normally excludes ETFs. If a provider starts
            # returning ticker-only ETF rows, replace only its still-active
            # row so one symbol cannot resolve to two current identities.
            current_etf_placeholders = (
                normalized["primary_symbol"].astype(str).isin(etf_symbols)
                & normalized["active_to"].fillna("").astype(str).str.strip().eq("")
            )
            normalized = pd.concat(
                [normalized.loc[~current_etf_placeholders], etf_identities],
                ignore_index=True,
                sort=False,
            )
    constituent_path = _configured_licensed_constituent_path()
    if constituent_path is not None:
        _, constituent_content, constituent_path = _licensed_constituent_table()
        constituent_artifact = SourceArtifact(
            source="krx_licensed_index_constituents",
            source_url=KRX_INDEX_LICENSE_URL,
            retrieved_at=retrieved_at,
            content=constituent_content,
            content_type={
                ".csv": "text/csv",
                ".json": "application/json",
                ".jsonl": "application/x-ndjson",
                ".parquet": "application/vnd.apache.parquet",
            }.get(constituent_path.suffix.lower(), "application/octet-stream"),
        )
        artifacts.append(constituent_artifact)
        licensed_identities = _licensed_constituent_identities(
            start=start,
            end=end,
            retrieved_at=retrieved_at,
            source_hash=constituent_artifact.source_hash,
        )
        if not licensed_identities.empty:
            licensed_symbols = set(licensed_identities["primary_symbol"].astype(str))
            licensed_identities = _reconcile_licensed_identity_intervals(
                licensed_identities,
                normalized.loc[
                    normalized["primary_symbol"].astype(str).isin(licensed_symbols)
                ],
            )
            normalized = normalized.loc[
                ~normalized["primary_symbol"].astype(str).isin(licensed_symbols)
            ]
            normalized = pd.concat(
                [normalized, licensed_identities], ignore_index=True, sort=False
            )
    if normalized.empty:
        raise RuntimeError("KRX identity listing returned no KOSPI/KOSDAQ securities.")
    normalized = normalized.sort_values(
        ["primary_symbol", "active_from", "active_to", "security_id"],
        kind="stable",
    ).drop_duplicates(
        ["security_id", "primary_symbol", "active_from"], keep="last"
    )

    dart_status = "not_requested"
    if include_dart:
        dart_map, dart_artifact, dart_status = _fetch_dart_corp_codes()
        if dart_artifact is not None:
            artifacts.append(dart_artifact)
        if dart_map:
            normalized = _attach_unambiguous_dart_codes(normalized, dart_map)
        else:
            normalized["dart_corp_code"] = ""
    else:
        normalized["dart_corp_code"] = ""
    return KrIdentityCatalog(normalized.reset_index(drop=True), tuple(artifacts), dart_status)


def restore_kr_identity_catalog_from_evidence(
    evidence_root: str | Path,
) -> KrIdentityCatalog:
    """Rebuild the normalized catalog from already hashed local-only evidence."""

    root = Path(evidence_root)
    listing_frames: dict[str, tuple[pd.DataFrame, str, str]] = {}
    description_frame = pd.DataFrame()
    delisted_payload: dict = {}
    delisted_retrieved_at = ""
    delisted_source_hash = ""
    etf_payload: dict = {}
    etf_retrieved_at = ""
    etf_source_hash = ""
    dart_map: dict[str, str] = {}
    for path in sorted(root.glob("*.gz")):
        try:
            content = gzip.decompress(path.read_bytes())
        except OSError:
            continue
        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            # OpenDART's corp-code inventory is itself a ZIP payload.  Local
            # evidence wraps every artifact in gzip, so it cannot be decoded
            # as the JSON used by the KRX artifacts above.
            restored_dart = _dart_mapping_from_corp_code_zip(content)
            if restored_dart:
                dart_map.update(restored_dart)
            continue
        if not isinstance(payload, dict):
            continue
        request = payload.get("request") or {}
        retrieved_at = datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=UTC,
        ).isoformat()
        source_hash = sha256_bytes(content)
        listing_name = str(request.get("listing") or "")
        if listing_name in {"KRX", "KRX-DELISTING"}:
            listing_frames[listing_name] = (
                pd.DataFrame(payload.get("rows") or []),
                retrieved_at,
                source_hash,
            )
        elif listing_name == "KRX-DESC":
            description_frame = pd.DataFrame(payload.get("rows") or [])
        bld = str(request.get("bld") or "")
        response = payload.get("response")
        if bld == KRX_WEB_DELISTED_FINDER_BLD and isinstance(response, dict):
            delisted_payload = response
            delisted_retrieved_at = retrieved_at
            delisted_source_hash = source_hash
        elif bld == KRX_WEB_ETF_MASTER_BLD and isinstance(response, dict):
            etf_payload = response
            etf_retrieved_at = retrieved_at
            etf_source_hash = source_hash

    if set(listing_frames) != {"KRX", "KRX-DELISTING"}:
        raise KrOfficialDataUnavailable(
            "The KR identity checkpoint lacks KRX/KRX-DELISTING evidence."
        )
    normalized_parts = [
        _normalize_listing(frame, retrieved_at, source_hash)
        for frame, retrieved_at, source_hash in (
            listing_frames["KRX"],
            listing_frames["KRX-DELISTING"],
        )
    ]
    normalized = pd.concat(normalized_parts, ignore_index=True)
    normalized = _attach_current_listing_dates(normalized, description_frame)
    if not delisted_payload:
        raise KrOfficialDataUnavailable(
            "The KR identity checkpoint lacks the delisted-security ISIN finder."
        )
    normalized = _attach_delisted_isins(
        normalized,
        delisted_payload,
        retrieved_at=delisted_retrieved_at,
        source_hash=delisted_source_hash,
    )
    if etf_payload:
        etf_identities = _normalize_krx_etf_master(
            etf_payload,
            retrieved_at=etf_retrieved_at,
            source_hash=etf_source_hash,
        )
        etf_symbols = set(etf_identities["primary_symbol"].astype(str))
        current_etf_placeholders = (
            normalized["primary_symbol"].astype(str).isin(etf_symbols)
            & normalized["active_to"].fillna("").astype(str).str.strip().eq("")
        )
        normalized = pd.concat(
            [normalized.loc[~current_etf_placeholders], etf_identities],
            ignore_index=True,
            sort=False,
        )
    normalized = normalized.sort_values(
        ["primary_symbol", "active_from", "active_to", "security_id"],
        kind="stable",
    ).drop_duplicates(
        ["security_id", "primary_symbol", "active_from"],
        keep="last",
    )
    if normalized.empty:
        raise KrOfficialDataUnavailable(
            "The KR identity checkpoint reconstructed an empty catalog."
        )
    if dart_map:
        normalized = _attach_unambiguous_dart_codes(normalized, dart_map)
        dart_status = "restored_from_local_evidence_with_dart"
    else:
        normalized["dart_corp_code"] = ""
        dart_status = "restored_from_local_evidence_without_dart"
    return KrIdentityCatalog(
        normalized.reset_index(drop=True),
        (),
        dart_status,
    )


def _reconcile_licensed_identity_intervals(
    licensed: pd.DataFrame,
    listings: pd.DataFrame,
) -> pd.DataFrame:
    """Attach listing/delisting intervals without trusting ticker identity.

    The licensed constituent row supplies the ISIN. FinanceDataReader's current
    KRX inventory also carries ISIN, while its delisting inventory supplies the
    historical interval but sometimes omits ISIN. An interval is borrowed only
    when the ISIN matches or exactly one listing row covers the constituent's
    first effective snapshot.
    """

    output: list[dict] = []
    for identity in licensed.to_dict("records"):
        symbol = str(identity["primary_symbol"])
        isin = str(identity["isin"])
        membership_start = pd.Timestamp(identity["active_from"])
        candidates = listings.loc[
            listings["primary_symbol"].astype(str).eq(symbol)
        ].copy()
        exact = candidates.loc[candidates["isin"].astype(str).eq(isin)]
        selected = pd.DataFrame(columns=candidates.columns)
        if not exact.empty:
            exact_starts = pd.to_datetime(exact["active_from"], errors="coerce")
            exact_ends = pd.to_datetime(exact["active_to"], errors="coerce")
            selected = exact.loc[
                (exact_starts.isna() | exact_starts.le(membership_start))
                & (exact_ends.isna() | exact_ends.ge(membership_start))
            ]
        if selected.empty and not candidates.empty:
            starts = pd.to_datetime(candidates["active_from"], errors="coerce")
            ends = pd.to_datetime(candidates["active_to"], errors="coerce")
            overlapping = candidates.loc[
                (starts.isna() | starts.le(membership_start))
                & (ends.isna() | ends.ge(membership_start))
            ]
            if len(overlapping) == 1:
                selected = overlapping
        if not selected.empty:
            listing = selected.sort_values(
                ["active_from", "active_to"], kind="stable"
            ).iloc[-1]
            active_from = _date_text(listing.get("active_from"))
            active_to = _date_text(listing.get("active_to"))
            identity["active_from"] = active_from
            if active_to:
                identity["active_to"] = active_to
            for column in ("name", "asset_type"):
                raw_value = listing.get(column)
                value = "" if pd.isna(raw_value) else str(raw_value).strip()
                if value:
                    identity[column] = value
        output.append(identity)
    return pd.DataFrame(output, columns=licensed.columns)


def _normalize_listing(
    frame: pd.DataFrame,
    retrieved_at: str,
    source_hash: str,
) -> pd.DataFrame:
    columns = [
        "security_id",
        "primary_symbol",
        "name",
        "exchange",
        "asset_type",
        "currency",
        "country",
        "active_from",
        "active_to",
        "isin",
        "identity_mapped",
        "provider_symbol",
        "yahoo_symbol",
        "source",
        "source_url",
        "retrieved_at",
        "source_hash",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    symbol_column = _find_column(frame, ("Code", "Symbol", "종목코드", "단축코드"))
    isin_column = _find_column(
        frame,
        ("ISU_CD", "ISIN", "표준코드", "StandardCode"),
        required=False,
    )
    name_column = _find_column(frame, ("Name", "종목명", "한글 종목명", "한글종목명"))
    market_column = _find_column(frame, ("Market", "MarketId", "시장구분", "시장"))
    listing_column = _find_column(
        frame,
        ("ListingDate", "Listing date", "상장일", "상장일자"),
        required=False,
    )
    delisting_column = _find_column(
        frame,
        ("DelistingDate", "Delisting date", "폐지일", "상장폐지일"),
        required=False,
    )
    type_column = _find_column(
        frame,
        ("SecuGroup", "Type", "종목구분", "증권구분"),
        required=False,
    )

    output = []
    for row in frame.to_dict("records"):
        symbol = normalize_kr_symbol(row.get(symbol_column))
        if not is_valid_kr_symbol(symbol):
            continue
        exchange = _normalize_exchange(row.get(market_column))
        if exchange not in {"KOSPI", "KOSDAQ"}:
            continue
        isin = str(row.get(isin_column) or "").strip().upper() if isin_column else ""
        identity_mapped = len(isin) == 12 and isin.isalnum()
        security_id = f"KR:{isin}" if identity_mapped else f"KR:UNMAPPED:{exchange}:{symbol}"
        asset_type = _normalize_asset_type(row.get(type_column) if type_column else "")
        output.append(
            {
                "security_id": security_id,
                "primary_symbol": symbol,
                "name": str(row.get(name_column) or symbol).strip(),
                "exchange": exchange,
                "asset_type": asset_type,
                "currency": "KRW",
                "country": "KR",
                "active_from": _date_text(row.get(listing_column)) if listing_column else "",
                "active_to": _date_text(row.get(delisting_column)) if delisting_column else "",
                "isin": isin,
                "identity_mapped": identity_mapped,
                "provider_symbol": f"{symbol}.{'KO' if exchange == 'KOSPI' else 'KQ'}",
                "yahoo_symbol": f"{symbol}.{'KS' if exchange == 'KOSPI' else 'KQ'}",
                "source": "krx_security_listing",
                "source_url": KRX_SOURCE_URL,
                "retrieved_at": retrieved_at,
                "source_hash": source_hash,
            }
        )
    return pd.DataFrame(output, columns=columns)


def _normalize_krx_etf_master(
    payload: dict,
    *,
    retrieved_at: str,
    source_hash: str,
) -> pd.DataFrame:
    columns = [
        "security_id",
        "primary_symbol",
        "name",
        "exchange",
        "asset_type",
        "currency",
        "country",
        "active_from",
        "active_to",
        "isin",
        "identity_mapped",
        "provider_symbol",
        "yahoo_symbol",
        "source",
        "source_url",
        "retrieved_at",
        "source_hash",
    ]
    values = payload.get("output")
    if not isinstance(values, list):
        raise KrOfficialDataUnavailable(
            "KRX Data Marketplace changed the ETF master response: "
            "output is not a list."
        )
    rows = []
    for raw_row in values:
        if not isinstance(raw_row, dict):
            continue
        symbol = normalize_kr_symbol(raw_row.get("ISU_SRT_CD"))
        isin = str(raw_row.get("ISU_CD") or "").strip().upper()
        if (
            not is_valid_kr_symbol(symbol)
            or len(isin) != 12
            or not isin.isalnum()
        ):
            continue
        rows.append(
            {
                "security_id": f"KR:{isin}",
                "primary_symbol": symbol,
                "name": str(raw_row.get("ISU_ABBRV") or symbol).strip(),
                "exchange": "KOSPI",
                "asset_type": "ETF",
                "currency": "KRW",
                "country": "KR",
                "active_from": _date_text(raw_row.get("LIST_DD")),
                "active_to": "",
                "isin": isin,
                "identity_mapped": True,
                "provider_symbol": f"{symbol}.KO",
                "yahoo_symbol": f"{symbol}.KS",
                "source": "krx_authenticated_web_etf_master",
                "source_url": KRX_SOURCE_URL,
                "retrieved_at": retrieved_at,
                "source_hash": source_hash,
            }
        )
    if not rows:
        raise KrOfficialDataUnavailable(
            "KRX Data Marketplace ETF master returned no valid ISIN identities."
        )
    return pd.DataFrame(rows, columns=columns)


def _attach_current_listing_dates(
    identities: pd.DataFrame,
    description_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Fill current ISIN rows with KRX-DESC listing dates by short code."""

    output = identities.copy()
    if output.empty or description_frame.empty:
        return output
    symbol_column = _find_column(
        description_frame,
        ("Code", "Symbol", "종목코드", "단축코드"),
        required=False,
    )
    listing_column = _find_column(
        description_frame,
        ("ListingDate", "Listing date", "상장일", "상장일자"),
        required=False,
    )
    if symbol_column is None or listing_column is None:
        return output
    listing_dates = {
        normalize_kr_symbol(row.get(symbol_column)): _date_text(
            row.get(listing_column)
        )
        for row in description_frame.to_dict("records")
        if is_valid_kr_symbol(row.get(symbol_column))
        and _date_text(row.get(listing_column))
    }
    missing_start = output["active_from"].fillna("").astype(str).str.strip().eq("")
    mapped_date = output["primary_symbol"].astype(str).map(listing_dates).fillna("")
    fillable = missing_start & mapped_date.ne("")
    output.loc[fillable, "active_from"] = mapped_date.loc[fillable]
    return output


def _attach_delisted_isins(
    identities: pd.DataFrame,
    finder_payload: dict,
    *,
    retrieved_at: str,
    source_hash: str,
) -> pd.DataFrame:
    """Join KRX delisting intervals to finder ISINs without guessing reuse."""

    values = finder_payload.get("block1")
    if not isinstance(values, list):
        raise KrOfficialDataUnavailable(
            "KRX Data Marketplace changed the delisted-security finder response: "
            "block1 is not a list."
        )
    candidates: dict[str, list[tuple[str, str]]] = {}
    for raw_row in values:
        if not isinstance(raw_row, dict):
            continue
        symbol = normalize_kr_symbol(raw_row.get("short_code"))
        isin = str(raw_row.get("full_code") or "").strip().upper()
        name = str(raw_row.get("codeName") or "").strip()
        if not is_valid_kr_symbol(symbol) or len(isin) != 12 or not isin.isalnum():
            continue
        candidates.setdefault(symbol, []).append((isin, name))

    output = identities.copy()
    for index, row in output.loc[~output["identity_mapped"].eq(True)].iterrows():  # noqa: E712
        symbol = normalize_kr_symbol(row.get("primary_symbol"))
        options = candidates.get(symbol, [])
        unique = sorted(set(options))
        selected: tuple[str, str] | None = None
        if len(unique) == 1:
            selected = unique[0]
        elif unique:
            row_name = str(row.get("name") or "").strip()
            name_matches = sorted(
                {
                    option
                    for option in unique
                    if option[1] and option[1] == row_name
                }
            )
            if len(name_matches) == 1:
                selected = name_matches[0]
        if selected is None:
            continue
        isin, _ = selected
        output.at[index, "security_id"] = f"KR:{isin}"
        output.at[index, "isin"] = isin
        output.at[index, "identity_mapped"] = True
        output.at[index, "source"] = (
            "krx_authenticated_web_delisted_security_finder"
        )
        output.at[index, "source_url"] = KRX_SOURCE_URL
        output.at[index, "retrieved_at"] = retrieved_at
        output.at[index, "source_hash"] = source_hash
    return output


def _krx_web_headers() -> dict[str, str]:
    return {
        "User-Agent": KRX_WEB_USER_AGENT,
        "Referer": "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd",
        "X-Requested-With": "XMLHttpRequest",
    }


def _krx_web_session_cache_path() -> Path:
    return Path(
        os.getenv(
            "KRX_SESSION_CACHE_PATH",
            "data/cache/private/krx_web_session.json",
        )
    ).expanduser()


def _write_krx_web_session_cache(
    path: Path,
    *,
    credential_fingerprint: str,
    cookies: dict[str, str],
) -> None:
    normalized_cookies = {
        str(name): str(value)
        for name, value in cookies.items()
        if str(name) and str(value)
    }
    if not normalized_cookies:
        return
    write_atomic(
        path,
        (
            json.dumps(
                {
                    "schema_version": _KRX_WEB_SESSION_CACHE_SCHEMA_VERSION,
                    "credential_fingerprint": credential_fingerprint,
                    "saved_at": datetime.now(UTC).isoformat(),
                    "cookies": normalized_cookies,
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode(),
    )
    os.chmod(path, 0o600)


def _read_krx_web_session_cache(
    path: Path,
    *,
    credential_fingerprint: str,
    max_age_seconds: float = _KRX_WEB_SESSION_TTL_SECONDS,
) -> tuple[dict[str, str], float] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version")
            != _KRX_WEB_SESSION_CACHE_SCHEMA_VERSION
            or payload.get("credential_fingerprint") != credential_fingerprint
        ):
            return None
        saved_at = pd.Timestamp(payload.get("saved_at"))
        if saved_at.tzinfo is None:
            saved_at = saved_at.tz_localize("UTC")
        age_seconds = (
            pd.Timestamp.now(tz="UTC") - saved_at.tz_convert("UTC")
        ).total_seconds()
        if age_seconds < -300 or age_seconds >= max_age_seconds:
            return None
        raw_cookies = payload.get("cookies")
        if not isinstance(raw_cookies, dict):
            return None
        cookies = {
            str(name): str(value)
            for name, value in raw_cookies.items()
            if str(name) and str(value)
        }
        if not cookies:
            return None
        os.chmod(path, 0o600)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return cookies, max(0.0, float(age_seconds))


def _persist_krx_web_session_locked(*, force: bool = False) -> None:
    global _KRX_WEB_CACHE_SAVED_AT

    if _KRX_WEB_SESSION is None or not _KRX_WEB_SESSION_CREDENTIALS:
        return
    now = time.monotonic()
    if (
        not force
        and _KRX_WEB_CACHE_SAVED_AT
        and now - _KRX_WEB_CACHE_SAVED_AT
        < _KRX_WEB_CACHE_WRITE_INTERVAL_SECONDS
    ):
        return
    _write_krx_web_session_cache(
        _krx_web_session_cache_path(),
        credential_fingerprint=_KRX_WEB_SESSION_CREDENTIALS,
        cookies=_KRX_WEB_SESSION.cookies.get_dict(),
    )
    _KRX_WEB_CACHE_SAVED_AT = now


def _login_krx_web_locked(credentials: tuple[str, str]):
    import requests

    login_id, login_pw = credentials
    for attempt in range(5):
        session = requests.Session()
        try:
            session.get(
                KRX_WEB_LOGIN_PAGE,
                headers={"User-Agent": KRX_WEB_USER_AGENT},
                timeout=20,
            ).raise_for_status()
            session.get(
                KRX_WEB_LOGIN_JSP,
                headers={
                    "User-Agent": KRX_WEB_USER_AGENT,
                    "Referer": KRX_WEB_LOGIN_PAGE,
                },
                timeout=20,
            ).raise_for_status()
            payload = {
                "mbrNm": "",
                "telNo": "",
                "di": "",
                "certType": "",
                "mbrId": login_id,
                "pw": login_pw,
            }
            response = session.post(
                KRX_WEB_LOGIN_URL,
                data=payload,
                headers={
                    "User-Agent": KRX_WEB_USER_AGENT,
                    "Referer": KRX_WEB_LOGIN_PAGE,
                },
                timeout=20,
            )
            response.raise_for_status()
            result = response.json()
            if result.get("_error_code") == "CD011":
                payload["skipDup"] = "Y"
                response = session.post(
                    KRX_WEB_LOGIN_URL,
                    data=payload,
                    headers={
                        "User-Agent": KRX_WEB_USER_AGENT,
                        "Referer": KRX_WEB_LOGIN_PAGE,
                    },
                    timeout=20,
                )
                response.raise_for_status()
                result = response.json()
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            session.close()
            retryable = status in {403, 408, 429, 500, 502, 503, 504} or status is None
            if retryable and attempt < 4:
                time.sleep(0.5 * (2**attempt))
                continue
            detail = f"HTTP {status}" if status is not None else type(exc).__name__
            raise KrOfficialDataUnavailable(
                f"KRX Data Marketplace login request failed: {detail}"
            ) from None
        code = str(result.get("_error_code") or "").strip()
        if code == "CD001":
            return session
        message = str(result.get("_error_message") or "login rejected").strip()
        session.close()
        raise KrOfficialDataUnavailable(
            f"KRX Data Marketplace login failed: {code or 'UNKNOWN'} {message[:200]}"
        )
    raise AssertionError("unreachable")


def _krx_web_session_locked(*, force_refresh: bool = False):
    global _KRX_WEB_CACHE_SAVED_AT
    global _KRX_WEB_LAST_USED_AT
    global _KRX_WEB_LOGIN_AT
    global _KRX_WEB_LOGIN_FAILURE_DETAIL
    global _KRX_WEB_LOGIN_FAILURE_UNTIL
    global _KRX_WEB_SESSION
    global _KRX_WEB_SESSION_CREDENTIALS

    import requests

    credentials = _krx_web_credentials(required=True)
    fingerprint = sha256_bytes(
        f"{credentials[0]}\0{credentials[1]}".encode("utf-8")
    )
    now = time.monotonic()
    expired = (
        not _KRX_WEB_LAST_USED_AT
        or now - _KRX_WEB_LAST_USED_AT >= _KRX_WEB_SESSION_TTL_SECONDS
    )
    if (
        force_refresh
        or _KRX_WEB_SESSION is None
        or _KRX_WEB_SESSION_CREDENTIALS != fingerprint
        or expired
    ):
        if not force_refresh:
            cached = _read_krx_web_session_cache(
                _krx_web_session_cache_path(),
                credential_fingerprint=fingerprint,
            )
            if cached is not None:
                cookies, age_seconds = cached
                if _KRX_WEB_SESSION is not None:
                    try:
                        _KRX_WEB_SESSION.close()
                    except Exception:
                        pass
                _KRX_WEB_SESSION = requests.Session()
                _KRX_WEB_SESSION.cookies.update(cookies)
                _KRX_WEB_SESSION_CREDENTIALS = fingerprint
                _KRX_WEB_LOGIN_AT = now
                _KRX_WEB_LAST_USED_AT = now - age_seconds
                _KRX_WEB_CACHE_SAVED_AT = now - min(
                    age_seconds,
                    _KRX_WEB_CACHE_WRITE_INTERVAL_SECONDS,
                )
                return _KRX_WEB_SESSION
        if _KRX_WEB_SESSION is not None:
            try:
                _KRX_WEB_SESSION.close()
            except Exception:
                pass
        _KRX_WEB_SESSION = None
        _KRX_WEB_SESSION_CREDENTIALS = ""
        remaining = _KRX_WEB_LOGIN_FAILURE_UNTIL - now
        if remaining > 0:
            detail = _KRX_WEB_LOGIN_FAILURE_DETAIL or "recent login failure"
            raise KrOfficialDataUnavailable(
                "KRX Data Marketplace login retry suppressed for "
                f"{max(1, int(remaining))}s after {detail}"
            )
        try:
            _KRX_WEB_SESSION = _login_krx_web_locked(credentials)
        except KrOfficialDataUnavailable as exc:
            _KRX_WEB_LOGIN_FAILURE_UNTIL = (
                time.monotonic() + _KRX_WEB_LOGIN_FAILURE_COOLDOWN_SECONDS
            )
            _KRX_WEB_LOGIN_FAILURE_DETAIL = str(exc)
            raise
        _KRX_WEB_SESSION_CREDENTIALS = fingerprint
        _KRX_WEB_LOGIN_AT = time.monotonic()
        _KRX_WEB_LAST_USED_AT = _KRX_WEB_LOGIN_AT
        _KRX_WEB_LOGIN_FAILURE_UNTIL = 0.0
        _KRX_WEB_LOGIN_FAILURE_DETAIL = ""
        _persist_krx_web_session_locked(force=True)
    return _KRX_WEB_SESSION


def _post_krx_web_json(
    payload: dict[str, str],
    *,
    sleep_seconds: float,
) -> dict:
    global _KRX_WEB_LAST_USED_AT

    import requests

    login_generation = 0.0
    for attempt in range(2):
        with _KRX_WEB_LOCK:
            force_refresh = (
                attempt > 0
                and bool(login_generation)
                and _KRX_WEB_LOGIN_AT == login_generation
            )
            session = _krx_web_session_locked(force_refresh=force_refresh)
            login_generation = _KRX_WEB_LOGIN_AT
            cookies = session.cookies.get_dict()
        request_session = requests.Session()
        request_session.cookies.update(cookies)
        try:
            response = request_session.post(
                KRX_WEB_DATA_URL,
                data=payload,
                headers=_krx_web_headers(),
                timeout=30,
            )
        except Exception as exc:
            raise KrOfficialDataUnavailable(
                f"KRX Data Marketplace request failed: {type(exc).__name__}"
            ) from None
        finally:
            request_session.close()
        body = response.text.strip()
        logged_out = body == "LOGOUT"
        if not logged_out:
            try:
                result = response.json()
            except Exception:
                result = None
            if response.ok and isinstance(result, dict):
                with _KRX_WEB_LOCK:
                    if (
                        _KRX_WEB_SESSION is not None
                        and _KRX_WEB_LOGIN_AT == login_generation
                    ):
                        _KRX_WEB_SESSION.cookies.update(response.cookies)
                        _KRX_WEB_LAST_USED_AT = time.monotonic()
                        _persist_krx_web_session_locked()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                return result
        if attempt == 0 and (logged_out or response.status_code in {400, 401, 403}):
            continue
        detail = "LOGOUT" if logged_out else f"HTTP {response.status_code}"
        raise KrOfficialDataUnavailable(
            f"KRX Data Marketplace request failed: {detail}"
        )
    raise AssertionError("unreachable")


def _fetch_krx_web_membership(
    profile: str,
    session: str,
    *,
    sleep_seconds: float,
) -> tuple[tuple[str, ...], SourceArtifact]:
    definition = KR_INDEX_DEFINITIONS[profile]
    requested = pd.Timestamp(session).date().isoformat()
    date_text = pd.Timestamp(requested).strftime("%Y%m%d")
    request_payload = {
        "bld": KRX_WEB_INDEX_BLD,
        "indIdx": str(definition["index_group"]),
        "indIdx2": str(definition["index_code"]),
        "trdDd": date_text,
    }
    response_payload = _post_krx_web_json(
        request_payload,
        sleep_seconds=sleep_seconds,
    )
    values = response_payload.get("output")
    if not isinstance(values, list):
        raise KrOfficialDataUnavailable(
            f"KRX Data Marketplace changed the {profile} constituent response "
            f"on {requested}: output is not a list."
        )
    symbols = tuple(
        sorted(
            {
                symbol
                for row in values
                if isinstance(row, dict)
                and (
                    symbol := normalize_kr_symbol(row.get("ISU_SRT_CD"))
                )
                and is_valid_kr_symbol(symbol)
            }
        )
    )
    if not symbols:
        inception = str(definition["announcement_date"])
        suffix = (
            f"; the index was announced on {inception}"
            if requested < inception
            else ""
        )
        raise KrOfficialDataUnavailable(
            f"KRX Data Marketplace returned no {profile} constituents on "
            f"{requested}{suffix}."
        )
    _validate_krx_membership_count(
        normalized_profile=profile,
        count=len(symbols),
        context=f"KRX Data Marketplace {profile} snapshot {requested}",
    )
    artifact = artifact_from_payload(
        "krx_authenticated_web_index_constituents",
        KRX_SOURCE_URL,
        {
            "request": {
                "profile": profile,
                "session": requested,
                **request_payload,
            },
            "response": response_payload,
        },
    )
    return symbols, artifact


def _validate_krx_membership_count(
    *,
    normalized_profile: str,
    count: int,
    context: str,
) -> None:
    definition = KR_INDEX_DEFINITIONS[normalized_profile]
    expected = int(definition["expected_count"])
    tolerance = int(definition.get("count_tolerance", 0))
    minimum = expected - tolerance
    maximum = expected + tolerance
    if count < minimum or count > maximum:
        raise KrOfficialDataUnavailable(
            f"{context} has {count} unique members; expected {minimum}-{maximum} "
            f"around the nominal {expected}."
        )


def fetch_krx_membership(
    profile: str,
    session: str,
    *,
    sleep_seconds: float = 0.0,
) -> tuple[tuple[str, ...], SourceArtifact]:
    normalized_profile = str(profile).lower()
    definition = KR_INDEX_DEFINITIONS.get(normalized_profile)
    if definition is None:
        raise ValueError(f"Unsupported KR index profile: {profile}")
    load_env()
    if _krx_web_credentials(required=False) is not None:
        return _fetch_krx_web_membership(
            normalized_profile,
            session,
            sleep_seconds=sleep_seconds,
        )
    table, content, _ = _licensed_constituent_table()
    requested = pd.Timestamp(session).date().isoformat()
    available = table.loc[
        table["profile"].eq(normalized_profile)
        & table["session"].le(requested)
    ]
    if available.empty:
        raise KrOfficialDataUnavailable(
            f"No licensed KRX {profile} constituent snapshot exists on or before {requested}."
        )
    effective = str(available["session"].max())
    snapshot = available.loc[available["session"].eq(effective)]
    symbols = tuple(sorted(snapshot["symbol"].astype(str).unique()))
    _validate_krx_membership_count(
        normalized_profile=normalized_profile,
        count=len(symbols),
        context=f"Licensed KRX {profile} snapshot {effective}",
    )
    artifact = SourceArtifact(
        source="krx_licensed_index_constituents",
        source_url=KRX_INDEX_LICENSE_URL,
        retrieved_at=utc_now_iso(),
        content=content,
        content_type="application/octet-stream",
    )
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return symbols, artifact


def _request_krx_openapi_daily(
    *,
    base_url: str,
    auth_key: str,
    endpoint: str,
    date_text: str,
    market_name: str,
    allow_authorization_fallback: bool = False,
) -> dict | None:
    import requests

    read_timeout = float(
        os.getenv("KRX_OPENAPI_READ_TIMEOUT_SECONDS", "10").strip() or "10"
    )
    hard_timeout = float(
        os.getenv("KRX_OPENAPI_HARD_TIMEOUT_SECONDS", "0").strip() or "0"
    )
    try:
        request_get = requests.get
        if os.getenv("KRX_OPENAPI_KEEPALIVE", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }:
            session = getattr(_KRX_OPENAPI_HTTP_LOCAL, "session", None)
            if session is None:
                session = requests.Session()
                _KRX_OPENAPI_HTTP_LOCAL.session = session
            request_get = session.get
        with _main_thread_hard_timeout(hard_timeout):
            response = request_get(
                f"{base_url}/{endpoint}",
                headers={"AUTH_KEY": auth_key},
                params={"basDd": date_text},
                timeout=(5, max(1.0, read_timeout)),
            )
        if (
            allow_authorization_fallback
            and response.status_code in {401, 403}
        ):
            return None
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        response = getattr(exc, "response", None)
        api_detail = ""
        if response is not None:
            try:
                error_payload = response.json()
            except Exception:
                error_payload = {}
            if isinstance(error_payload, dict):
                code = str(
                    error_payload.get("respCode")
                    or error_payload.get("resultCode")
                    or ""
                ).strip()
                message = str(
                    error_payload.get("respMsg")
                    or error_payload.get("message")
                    or error_payload.get("resultMsg")
                    or ""
                ).strip()
                api_detail = " ".join(value for value in (code, message) if value)
        suffix = f" ({api_detail[:300]})" if api_detail else ""
        raise KrOfficialDataUnavailable(
            f"Authenticated KRX Open API price request failed for {market_name}: "
            f"{type(exc).__name__}{suffix}"
        ) from None
    if not isinstance(payload, dict):
        raise KrOfficialDataUnavailable(
            f"KRX Open API returned a non-object response for {market_name}."
        )
    return payload


@contextmanager
def _main_thread_hard_timeout(seconds: float):
    """Bound a blocking HTTP call when requests' socket timeout is ineffective."""

    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return
    try:
        import signal

        alarm = signal.SIGALRM
        set_timer = signal.setitimer
    except (AttributeError, ImportError):
        yield
        return

    def raise_timeout(_signum, _frame) -> None:
        raise TimeoutError(
            f"KRX Open API request exceeded {seconds:.1f} seconds."
        )

    previous_handler = signal.getsignal(alarm)
    previous_timer = set_timer(signal.ITIMER_REAL, seconds)
    signal.signal(alarm, raise_timeout)
    try:
        yield
    finally:
        set_timer(signal.ITIMER_REAL, 0.0)
        signal.signal(alarm, previous_handler)
        if previous_timer[0] > 0:
            set_timer(signal.ITIMER_REAL, *previous_timer)


def _fetch_krx_openapi_etf_daily(
    *,
    base_url: str,
    auth_key: str,
    date_text: str,
) -> dict | None:
    """Probe ETF authorization once, then reuse the result in this process."""

    global _KRX_ETF_OPENAPI_AVAILABLE

    if _KRX_ETF_OPENAPI_AVAILABLE is False:
        return None
    if _KRX_ETF_OPENAPI_AVAILABLE is True:
        return _request_krx_openapi_daily(
            base_url=base_url,
            auth_key=auth_key,
            endpoint=KRX_OPENAPI_ETF_DAILY_ENDPOINT,
            date_text=date_text,
            market_name="ETF",
            allow_authorization_fallback=False,
        )
    with _KRX_ETF_OPENAPI_LOCK:
        if _KRX_ETF_OPENAPI_AVAILABLE is None:
            payload = _request_krx_openapi_daily(
                base_url=base_url,
                auth_key=auth_key,
                endpoint=KRX_OPENAPI_ETF_DAILY_ENDPOINT,
                date_text=date_text,
                market_name="ETF",
                allow_authorization_fallback=True,
            )
            _KRX_ETF_OPENAPI_AVAILABLE = payload is not None
            return payload
        available = _KRX_ETF_OPENAPI_AVAILABLE
    if not available:
        return None
    return _request_krx_openapi_daily(
        base_url=base_url,
        auth_key=auth_key,
        endpoint=KRX_OPENAPI_ETF_DAILY_ENDPOINT,
        date_text=date_text,
        market_name="ETF",
        allow_authorization_fallback=False,
    )


def _catalog_requires_etf(active_rows: dict[str, dict]) -> bool:
    return any(
        str(identity.get("asset_type") or "").strip().upper() == "ETF"
        for identity in active_rows.values()
    )


def _krx_price_row_evidence(raw_row: dict, market_name: str) -> dict[str, object]:
    """Extract the stable identity and raw-session state from one official row."""

    symbol = normalize_kr_symbol(
        raw_row.get("ISU_SRT_CD")
        or raw_row.get("ISU_CD")
        or ""
    )
    isin = _normalize_isin(
        raw_row.get("ISU_CD")
        or raw_row.get("ISIN")
        or raw_row.get("표준코드")
        or ""
    )
    security_id = (
        f"KR:{isin}"
        if len(isin) == 12 and isin.isalnum()
        else ""
    )
    numeric = [
        _krx_number(raw_row.get(key))
        for key in ("TDD_OPNPRC", "TDD_HGPRC", "TDD_LWPRC", "TDD_CLSPRC")
    ]
    volume = _krx_number(raw_row.get("ACC_TRDVOL"))
    open_price, high, low, close = numeric
    comparison_to_previous = _krx_number(
        raw_row.get("CMPPREVDD_PRC")
    )
    official_reference_price = (
        float(close - comparison_to_previous)
        if close is not None and comparison_to_previous is not None
        else None
    )
    official_fluctuation_rate = _krx_number(raw_row.get("FLUC_RT"))
    if volume == 0:
        observation_status = "suspended_or_no_trade"
    elif all(value is not None and value > 0 for value in numeric):
        observation_status = "traded"
    elif (
        volume is not None
        and volume > 0
        and close is not None
        and close > 0
        and all(value is None or value <= 0 for value in (open_price, high, low))
    ):
        # KRX can report a positive accumulated volume/value and a carried
        # close while regular-session O/H/L are all zero. This is an explicit
        # no-regular-session-price observation, not a usable OHLC bar.
        observation_status = "no_regular_session_ohlc"
    else:
        observation_status = "invalid_official_ohlc"
    raw_exchange = (
        raw_row.get("MKT_NM")
        or raw_row.get("MKT_ID")
        or market_name
    )
    exchange = _normalize_exchange(raw_exchange)
    if exchange not in {"KOSPI", "KOSDAQ"}:
        exchange = "KOSPI" if market_name == "ETF" else str(market_name)
    return {
        "symbol": symbol,
        "security_id": security_id,
        "security_name": str(
            raw_row.get("ISU_ABBRV")
            or raw_row.get("ISU_NM")
            or symbol
        ).strip(),
        "exchange": exchange,
        "asset_type": "ETF" if market_name == "ETF" else "STOCK",
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "official_reference_price": official_reference_price,
        "official_fluctuation_rate": official_fluctuation_rate,
        "observation_status": observation_status,
    }


def krx_price_evidence_by_symbol(payload: dict) -> dict[str, dict[str, object]]:
    """Decode a saved combined KRX price artifact without network access."""

    responses = payload.get("responses")
    if not isinstance(responses, dict):
        raise KrOfficialDataUnavailable(
            "Saved KRX price evidence does not contain responses."
        )
    output: dict[str, dict[str, object]] = {}
    for market_name, response in responses.items():
        if not isinstance(response, dict):
            continue
        values = response.get("OutBlock_1")
        if values is None:
            values = response.get("output")
        if not isinstance(values, list):
            continue
        for raw_row in values:
            if not isinstance(raw_row, dict):
                continue
            evidence = _krx_price_row_evidence(raw_row, str(market_name))
            symbol = str(evidence["symbol"])
            if not is_valid_kr_symbol(symbol):
                continue
            previous = output.get(symbol)
            if (
                previous is not None
                and previous.get("security_id")
                and evidence.get("security_id")
                and previous["security_id"] != evidence["security_id"]
            ):
                raise KrOfficialDataUnavailable(
                    "KRX price evidence maps one symbol to multiple ISINs "
                    f"within one session: {symbol}."
                )
            output[symbol] = evidence
    return output


def fetch_krx_session_prices(
    session: str,
    catalog: KrIdentityCatalog,
    *,
    symbols: Iterable[str] = (),
    sleep_seconds: float = 0.0,
) -> tuple[pd.DataFrame, SourceArtifact]:
    global _KRX_ETF_OPENAPI_AVAILABLE
    global _KRX_STOCK_OPENAPI_AVAILABLE

    validate_krx_official_configuration(require_membership=False)
    load_env()
    auth_key = os.getenv("KRX_OPENAPI_AUTH_KEY", "").strip()
    base_url = os.getenv("KRX_OPENAPI_BASE_URL", KRX_OPENAPI_BASE_URL).rstrip("/")
    price_transport = (
        os.getenv("KRX_DAILY_PRICE_TRANSPORT", "auto").strip().lower() or "auto"
    )
    if price_transport not in {"auto", "web"}:
        raise KrOfficialDataUnavailable(
            "KRX_DAILY_PRICE_TRANSPORT must be 'auto' or 'web'."
        )
    prefer_authenticated_web = price_transport == "web"
    date_text = pd.Timestamp(session).strftime("%Y%m%d")
    selected = {normalize_kr_symbol(value) for value in symbols}
    active_identities = catalog.active_rows_for(selected, session)
    payloads: dict[str, dict] = {}
    payload_metadata: dict[str, dict[str, str]] = {}
    endpoints = {
        "KOSPI": "sto/stk_bydd_trd",
        "KOSDAQ": "sto/ksq_bydd_trd",
    }
    stock_openapi_error: KrOfficialDataUnavailable | None = None
    with _KRX_STOCK_OPENAPI_LOCK:
        stock_openapi_available = (
            _KRX_STOCK_OPENAPI_AVAILABLE and not prefer_authenticated_web
        )
    if stock_openapi_available:
        try:
            for market_name, endpoint in endpoints.items():
                payload = _request_krx_openapi_daily(
                    base_url=base_url,
                    auth_key=auth_key,
                    endpoint=endpoint,
                    date_text=date_text,
                    market_name=f"{market_name} on {session}",
                )
                assert payload is not None
                payloads[market_name] = payload
                payload_metadata[market_name] = {
                    "transport": "openapi",
                    "endpoint": endpoint,
                    "response_key": "OutBlock_1",
                    "source": "krx_openapi_daily_ohlcv",
                    "source_url": KRX_OPENAPI_SOURCE_URL,
                }
        except KrOfficialDataUnavailable as exc:
            stock_openapi_error = exc
            payloads.clear()
            payload_metadata.clear()
            if _krx_web_credentials(required=False) is not None:
                with _KRX_STOCK_OPENAPI_LOCK:
                    _KRX_STOCK_OPENAPI_AVAILABLE = False
    if not payloads:
        if _krx_web_credentials(required=False) is None:
            if stock_openapi_error is not None:
                raise stock_openapi_error
            raise KrOfficialDataUnavailable(
                "KRX stock daily prices require an available Open API or "
                "KRX_ID + KRX_PW for the authenticated Data Marketplace fallback."
            )
        stock_request = {
            "bld": KRX_WEB_STOCK_DAILY_BLD,
            "mktId": "ALL",
            "trdDd": date_text,
            "share": "1",
            "money": "1",
            "csvxls_isNo": "false",
        }
        payloads["STOCK"] = _post_krx_web_json(
            stock_request,
            sleep_seconds=0.0,
        )
        payload_metadata["STOCK"] = {
            "transport": "authenticated_web",
            "endpoint": KRX_WEB_STOCK_DAILY_BLD,
            "response_key": "OutBlock_1",
            "source": "krx_authenticated_web_daily_ohlcv",
            "source_url": KRX_SOURCE_URL,
        }

    if selected and _catalog_requires_etf(active_identities):
        etf_payload = None
        if not prefer_authenticated_web:
            try:
                etf_payload = _fetch_krx_openapi_etf_daily(
                    base_url=base_url,
                    auth_key=auth_key,
                    date_text=date_text,
                )
            except KrOfficialDataUnavailable:
                if _krx_web_credentials(required=False) is None:
                    raise
                with _KRX_ETF_OPENAPI_LOCK:
                    _KRX_ETF_OPENAPI_AVAILABLE = False
        if etf_payload is not None:
            payloads["ETF"] = etf_payload
            payload_metadata["ETF"] = {
                "transport": "openapi",
                "endpoint": KRX_OPENAPI_ETF_DAILY_ENDPOINT,
                "response_key": "OutBlock_1",
                "source": "krx_openapi_daily_ohlcv",
                "source_url": KRX_OPENAPI_SOURCE_URL,
            }
        else:
            if _krx_web_credentials(required=False) is None:
                raise KrOfficialDataUnavailable(
                    "KRX ETF daily prices require either approval for the "
                    "'ETF 일별매매정보' Open API or KRX_ID + KRX_PW for the "
                    "authenticated Data Marketplace fallback."
                )
            request_payload = {
                "bld": KRX_WEB_ETF_DAILY_BLD,
                "trdDd": date_text,
            }
            payloads["ETF"] = _post_krx_web_json(
                request_payload,
                sleep_seconds=0.0,
            )
            payload_metadata["ETF"] = {
                "transport": "authenticated_web",
                "endpoint": KRX_WEB_ETF_DAILY_BLD,
                "response_key": "output",
                "source": "krx_authenticated_web_daily_ohlcv",
                "source_url": KRX_SOURCE_URL,
            }

    used_authenticated_web = any(
        item["transport"] == "authenticated_web"
        for item in payload_metadata.values()
    )
    artifact = artifact_from_payload(
        (
            "krx_official_daily_ohlcv"
            if used_authenticated_web
            else "krx_openapi_daily_ohlcv"
        ),
        KRX_SOURCE_URL if used_authenticated_web else KRX_OPENAPI_SOURCE_URL,
        {
            "request": {
                "session": session,
                "basDd": date_text,
                "endpoints": {
                    market: {
                        "transport": metadata["transport"],
                        "endpoint": metadata["endpoint"],
                    }
                    for market, metadata in payload_metadata.items()
                },
                "adjusted": False,
            },
            "responses": payloads,
        },
    )
    rows = []
    seen_symbols: set[str] = set()
    for market_name, payload in payloads.items():
        metadata = payload_metadata[market_name]
        values = payload.get(metadata["response_key"])
        if values is None:
            detail = str(
                payload.get("message")
                or payload.get("msg1")
                or payload.get("resultMsg")
                or "missing OutBlock_1"
            )
            raise KrOfficialDataUnavailable(
                f"KRX Open API rejected or changed the {market_name} response on "
                f"{session}: {detail}"
            )
        if not isinstance(values, list):
            raise KrOfficialDataUnavailable(
                f"KRX Open API {market_name} OutBlock_1 is not a list on {session}."
            )
        for raw_row in values:
            evidence = _krx_price_row_evidence(raw_row, market_name)
            symbol = str(evidence["symbol"])
            if selected and symbol not in selected:
                continue
            seen_symbols.add(symbol)
            identity = active_identities.get(symbol, {})
            raw_security_id = str(evidence.get("security_id") or "")
            catalog_security_id = str(identity.get("security_id") or "")
            if (
                raw_security_id
                and catalog_security_id
                and raw_security_id != catalog_security_id
            ):
                raise KrOfficialDataUnavailable(
                    "KRX daily ISIN conflicts with the active identity catalog "
                    f"for {symbol} on {session}: "
                    f"{raw_security_id} != {catalog_security_id}."
                )
            security_id = raw_security_id or catalog_security_id
            if not security_id:
                raise KrOfficialDataUnavailable(
                    f"KRX daily row lacks stable ISIN identity for {symbol} on {session}."
                )
            status = str(evidence["observation_status"])
            volume = evidence.get("volume")
            if status != "traded":
                rows.append(
                    _price_record(
                        security_id,
                        symbol,
                        session,
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float(volume or 0),
                        source=metadata["source"],
                        source_url=metadata["source_url"],
                        retrieved_at=artifact.retrieved_at,
                        source_hash=artifact.source_hash,
                        observation_status=status,
                        exchange=str(
                            evidence.get("exchange")
                            or identity.get("exchange")
                            or market_name
                        ),
                        asset_type=str(
                            evidence.get("asset_type")
                            or identity.get("asset_type")
                            or "STOCK"
                        ),
                        security_name=str(
                            evidence.get("security_name")
                            or identity.get("name")
                            or symbol
                        ),
                        official_reference_price=(
                            evidence.get("official_reference_price")
                        ),
                        official_fluctuation_rate=(
                            evidence.get("official_fluctuation_rate")
                        ),
                    )
                )
                continue
            rows.append(
                _price_record(
                    security_id,
                    symbol,
                    session,
                    *(
                        float(evidence[key])
                        for key in ("open", "high", "low", "close")
                    ),
                    float(volume or 0),
                    source=metadata["source"],
                    source_url=metadata["source_url"],
                    retrieved_at=artifact.retrieved_at,
                    source_hash=artifact.source_hash,
                    exchange=str(
                        evidence.get("exchange")
                        or identity.get("exchange")
                        or market_name
                    ),
                    asset_type=str(
                        evidence.get("asset_type")
                        or identity.get("asset_type")
                        or "STOCK"
                    ),
                    security_name=str(
                        evidence.get("security_name")
                        or identity.get("name")
                        or symbol
                    ),
                    official_reference_price=(
                        evidence.get("official_reference_price")
                    ),
                    official_fluctuation_rate=(
                        evidence.get("official_fluctuation_rate")
                    ),
                )
            )
    # A selected security that is active in the identity catalog but absent
    # from both official market payloads cannot be silently treated as a
    # suspension. It remains an explicit hard-gate failure.
    for symbol in sorted(selected - seen_symbols):
        identity = active_identities.get(symbol)
        if identity is None:
            continue
        security_id = str(identity["security_id"])
        observation_status = (
            "delisting_effective_date_no_trade"
            if _date_text(identity.get("active_to"))
            == pd.Timestamp(session).date().isoformat()
            and catalog.interval_is_terminal(identity, session)
            else "missing_from_krx_response"
        )
        rows.append(
            _price_record(
                security_id,
                symbol,
                session,
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                0.0,
                source=artifact.source,
                source_url=artifact.source_url,
                retrieved_at=artifact.retrieved_at,
                source_hash=artifact.source_hash,
                observation_status=observation_status,
                exchange=str(identity.get("exchange") or ""),
                asset_type=str(identity.get("asset_type") or "STOCK"),
                security_name=str(identity.get("name") or symbol),
            )
        )
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return pd.DataFrame(rows, columns=_price_columns()), artifact


def fetch_eodhd_prices(
    catalog: KrIdentityCatalog,
    *,
    start: str,
    end: str,
    workers: int = 6,
) -> KrProviderResult:
    try:
        client = EodhdClient()
    except Exception as exc:
        return KrProviderResult(
            "eodhd",
            "skipped_missing_credentials" if "TOKEN" in str(exc).upper() else "failed",
            _empty_price_frame(),
            detail=str(exc),
        )
    rows: list[dict] = []
    artifacts: list[SourceArtifact] = []
    failures: list[str] = []
    identities = _identities_for_range(catalog.frame, start, end)

    def fetch_one(identity: dict):
        symbol = str(identity["provider_symbol"])
        params = {"from": start, "to": end}
        values = client.get_json(f"eod/{symbol}", params=params)
        if not isinstance(values, list):
            values = []
        artifact = artifact_from_payload(
            "eodhd_kr_eod",
            client.safe_url(f"eod/{symbol}", params=params),
            {"request": {"symbol": symbol, **params}, "rows": values},
        )
        output = []
        for value in values:
            session = str(value.get("date") or "")
            if not session or not _identity_covers(identity, session):
                continue
            try:
                numeric = [float(value.get(key)) for key in ("open", "high", "low", "close")]
                volume = float(value.get("volume") or 0)
            except (TypeError, ValueError):
                continue
            if any(item <= 0 for item in numeric):
                continue
            output.append(
                _price_record(
                    str(identity["security_id"]),
                    str(identity["primary_symbol"]),
                    session,
                    *numeric,
                    volume,
                    source="eodhd_kr_eod",
                    source_url=artifact.source_url,
                    retrieved_at=artifact.retrieved_at,
                    source_hash=artifact.source_hash,
                    observation_status=(
                        "suspended_or_no_trade" if volume == 0 else "traded"
                    ),
                    exchange=str(identity.get("exchange") or ""),
                    asset_type=str(identity.get("asset_type") or "STOCK"),
                )
            )
        return output, artifact

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(fetch_one, identity): str(identity["provider_symbol"])
            for identity in identities
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                values, artifact = future.result()
            except Exception as exc:
                failures.append(f"{symbol}:{type(exc).__name__}")
                continue
            rows.extend(values)
            artifacts.append(artifact)
    status = "ok" if rows else "failed"
    detail = "" if not failures else f"failed_symbols={len(failures)}"
    return KrProviderResult(
        "eodhd",
        status,
        pd.DataFrame(rows, columns=_price_columns()),
        tuple(artifacts),
        detail,
    )


def fetch_eodhd_actions(
    catalog: KrIdentityCatalog,
    *,
    start: str,
    end: str,
    workers: int = 6,
) -> tuple[pd.DataFrame, tuple[SourceArtifact, ...], tuple[str, ...]]:
    """Fetch raw dividend and split facts for the exact KR identities."""

    client = EodhdClient()
    identity_rows = pd.DataFrame(
        _identities_for_range(catalog.frame, start, end)
    )
    identities: list[dict] = []
    if not identity_rows.empty:
        for _, group in identity_rows.groupby("security_id", sort=True):
            ordered = group.assign(
                _active_from=pd.to_datetime(
                    group["active_from"],
                    errors="coerce",
                ),
                _is_current=group["active_to"]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq(""),
            ).sort_values(
                ["_is_current", "_active_from"],
                kind="stable",
            )
            identity = ordered.iloc[-1].drop(
                labels=["_active_from", "_is_current"]
            ).to_dict()
            starts = pd.to_datetime(group["active_from"], errors="coerce").dropna()
            ends = pd.to_datetime(group["active_to"], errors="coerce").dropna()
            identity["active_from"] = (
                starts.min().date().isoformat() if not starts.empty else ""
            )
            identity["active_to"] = (
                ""
                if group["active_to"]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq("")
                .any()
                else ends.max().date().isoformat()
                if not ends.empty
                else ""
            )
            identities.append(identity)

    def fetch_one(identity: dict):
        records: list[dict] = []
        artifacts: list[SourceArtifact] = []
        limitations: list[str] = []
        provider_symbol = str(identity["provider_symbol"])
        params = {"from": start, "to": end}
        for endpoint in ("div", "splits"):
            try:
                values = client.get_json(
                    f"{endpoint}/{provider_symbol}",
                    params=params,
                )
            except RuntimeError as exc:
                active_to = _date_text(identity.get("active_to"))
                known_pre2019_gap = (
                    "HTTP 404" in str(exc)
                    and bool(active_to)
                    and active_to < "2019-01-01"
                )
                if not known_pre2019_gap:
                    raise
                limitation = (
                    "known_unavailable_pre2019_delisted_actions:"
                    f"{provider_symbol}:{endpoint}"
                )
                limitations.append(limitation)
                artifacts.append(
                    artifact_from_payload(
                        "eodhd_kr_action_unavailable",
                        client.safe_url(
                            f"{endpoint}/{provider_symbol}",
                            params=params,
                        ),
                        {
                            "request": {
                                "symbol": provider_symbol,
                                "endpoint": endpoint,
                                **params,
                            },
                            "status": "HTTP 404",
                            "coverage_class": (
                                "pre2019_delisted_actions_unavailable"
                            ),
                        },
                    )
                )
                continue
            if not isinstance(values, list):
                values = []
            artifact = artifact_from_payload(
                f"eodhd_kr_{endpoint}",
                client.safe_url(f"{endpoint}/{provider_symbol}", params=params),
                {
                    "request": {"symbol": provider_symbol, "endpoint": endpoint, **params},
                    "rows": values,
                },
            )
            artifacts.append(artifact)
            for value in values:
                effective = _date_text(value.get("date"))
                if not effective or not _identity_covers(identity, effective):
                    continue
                if endpoint == "div":
                    action_type = "cash_dividend"
                    cash_amount = _optional_float(
                        value.get("unadjustedValue", value.get("value"))
                    )
                    ratio = None
                    if cash_amount is None or cash_amount < 0:
                        continue
                else:
                    action_type = "split"
                    cash_amount = None
                    ratio = _split_ratio(value.get("split"))
                    if ratio is None or ratio <= 0:
                        continue
                identity_key = {
                    "security_id": str(identity["security_id"]),
                    "action_type": action_type,
                    "effective_date": effective,
                    "cash_amount": cash_amount,
                    "ratio": ratio,
                    "source_hash": artifact.source_hash,
                }
                records.append(
                    {
                        "event_id": sha256_bytes(canonical_json_bytes(identity_key)),
                        "security_id": str(identity["security_id"]),
                        "action_type": action_type,
                        "effective_date": effective,
                        "ex_date": effective,
                        "announcement_date": _date_text(value.get("declarationDate")),
                        "record_date": _date_text(value.get("recordDate")),
                        "payment_date": _date_text(value.get("paymentDate")),
                        "cash_amount": cash_amount,
                        "ratio": ratio,
                        "currency": str(value.get("currency") or "KRW").upper(),
                        "new_security_id": "",
                        "new_symbol": "",
                        "official": False,
                        "source_url": artifact.source_url,
                        "source_kind": "provider",
                        "source": artifact.source,
                        "retrieved_at": artifact.retrieved_at,
                        "source_hash": artifact.source_hash,
                    }
                )
        return records, tuple(artifacts), tuple(limitations)

    records: list[dict] = []
    artifacts: list[SourceArtifact] = []
    failures: list[str] = []
    retry_identities: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(fetch_one, identity): identity
            for identity in identities
        }
        for future in as_completed(futures):
            identity = futures[future]
            try:
                values, source_artifacts, source_limitations = future.result()
            except Exception:
                retry_identities.append(identity)
                continue
            records.extend(values)
            artifacts.extend(source_artifacts)
            failures.extend(source_limitations)
    # A small number of EODHD action calls can still exhaust the client's
    # internal transient retry window during a concurrent full-universe run.
    # Retry only those identities once, sequentially, rather than repeating
    # every successful request or silently accepting a hole.
    for identity in retry_identities:
        symbol = str(identity["provider_symbol"])
        try:
            values, source_artifacts, source_limitations = fetch_one(
                identity
            )
        except Exception as exc:
            failures.append(
                f"{symbol}:{type(exc).__name__}:{exc}"
            )
            continue
        records.extend(values)
        artifacts.extend(source_artifacts)
        failures.extend(source_limitations)
    columns = [
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
        "source_url",
        "source_kind",
        "source",
        "retrieved_at",
        "source_hash",
    ]
    return (
        pd.DataFrame(records, columns=columns).drop_duplicates("event_id", keep="last"),
        tuple(artifacts),
        tuple(sorted(failures)),
    )


def fetch_opendart_dividend_decisions(
    catalog: KrIdentityCatalog,
    *,
    start: str,
    end: str,
    checkpoint_root: str | Path,
    workers: int = 4,
) -> KrDartDividendResult:
    """Collect every official cash-dividend decision for scoped KR stocks.

    OpenDART does not expose cash-dividend decisions as a structured endpoint.
    The complete exchange-disclosure inventory is therefore paged per issuer,
    final amended filings are selected, and each matching original filing is
    parsed. Raw list responses and ZIP documents are content-addressed locally
    so a full 2015+ audit resumes without repeating successful requests.
    """

    load_env()
    api_key = os.getenv("DART_API_KEY", "").strip()
    if not api_key:
        report = artifact_from_payload(
            "opendart_action_audit",
            DART_DISCLOSURE_LIST_URL,
            {
                "schema": "opendart_dividend_collection/v1",
                "status": "blocked",
                "issues": ["missing_DART_API_KEY"],
                "records": [],
            },
        )
        return KrDartDividendResult(
            "blocked",
            _empty_opendart_dividend_frame(),
            report,
            "missing_DART_API_KEY",
        )

    import requests

    requested_start = max(
        pd.Timestamp("2015-01-01"),
        pd.Timestamp(start),
    ).date().isoformat()
    requested_end = pd.Timestamp(end).date().isoformat()
    # A decision filed after a fiscal record date still belongs to the audited
    # period. Query through today so late annual decisions are not missed.
    filing_end = min(
        pd.Timestamp.now(tz="UTC").tz_localize(None).normalize(),
        pd.Timestamp(end) + pd.Timedelta(days=370),
    ).date().isoformat()
    root = Path(checkpoint_root)
    request_root = root / "providers" / "opendart" / "actions" / "requests"
    evidence_root = root / "evidence_local" / "opendart-actions"
    request_root.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)

    values = pd.DataFrame(_identities_for_range(catalog.frame, start, end))
    if values.empty:
        report = artifact_from_payload(
            "opendart_action_audit",
            DART_DISCLOSURE_LIST_URL,
            {
                "schema": "opendart_dividend_collection/v1",
                "status": "passed",
                "requested_start": requested_start,
                "requested_end": requested_end,
                "issuer_count": 0,
                "records": [],
                "issues": [],
            },
        )
        return KrDartDividendResult(
            "passed",
            _empty_opendart_dividend_frame(),
            report,
        )

    stock_values = values.loc[
        values.get(
            "asset_type",
            pd.Series("STOCK", index=values.index),
        )
        .fillna("STOCK")
        .astype(str)
        .str.upper()
        .eq("STOCK")
    ].copy()
    identity_rows: list[dict] = []
    missing_dart: list[str] = []
    for security_id, group in stock_values.groupby("security_id", sort=True):
        corp_codes = sorted(
            {
                str(value).strip().zfill(8)
                for value in group.get(
                    "dart_corp_code",
                    pd.Series(dtype=str),
                )
                if str(value).strip()
            }
        )
        if len(corp_codes) != 1:
            missing_dart.append(str(security_id))
            continue
        starts = pd.to_datetime(group["active_from"], errors="coerce").dropna()
        ends = pd.to_datetime(group["active_to"], errors="coerce").dropna()
        identity_rows.append(
            {
                "security_id": str(security_id),
                "symbol": str(group.iloc[-1]["primary_symbol"]),
                "corp_code": corp_codes[0],
                "active_from": (
                    starts.min().date().isoformat()
                    if not starts.empty
                    else requested_start
                ),
                "active_to": (
                    ""
                    if group["active_to"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .eq("")
                    .any()
                    else ends.max().date().isoformat()
                    if not ends.empty
                    else ""
                ),
            }
        )
    identities_by_corp: dict[str, list[dict]] = {}
    for row in identity_rows:
        identities_by_corp.setdefault(row["corp_code"], []).append(row)

    pacing_lock = threading.Lock()
    next_request_at = 0.0
    request_interval = max(
        0.05,
        float(
            os.getenv(
                "OPENDART_MIN_REQUEST_INTERVAL_SECONDS",
                "0.065",
            ).strip()
            or "0.065"
        ),
    )

    def pace_request() -> None:
        nonlocal next_request_at
        with pacing_lock:
            now = time.monotonic()
            wait_seconds = max(0.0, next_request_at - now)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            next_request_at = max(
                next_request_at,
                time.monotonic(),
            ) + request_interval

    def cached_artifact(key: str) -> SourceArtifact | None:
        metadata_path = request_root / f"{key}.json"
        if not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source_hash = str(metadata.get("source_hash") or "").lower()
            suffix = str(metadata.get("suffix") or "bin")
            if not _valid_sha256_text(source_hash):
                return None
            paths = sorted(evidence_root.glob(f"{source_hash}.{suffix}.gz"))
            if len(paths) != 1:
                return None
            content = gzip.decompress(paths[0].read_bytes())
            if sha256_bytes(content) != source_hash:
                return None
            return SourceArtifact(
                source=str(metadata["source"]),
                source_url=str(metadata["source_url"]),
                retrieved_at=str(metadata["retrieved_at"]),
                content=content,
                content_type=str(metadata["content_type"]),
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def save_artifact(
        key: str,
        artifact: SourceArtifact,
        *,
        suffix: str,
    ) -> None:
        artifact_path = evidence_root / (
            f"{artifact.source_hash}.{suffix}.gz"
        )
        if not artifact_path.is_file():
            write_atomic(artifact_path, gzip.compress(artifact.content))
        write_atomic(
            request_root / f"{key}.json",
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "source": artifact.source,
                        "source_url": artifact.source_url,
                        "retrieved_at": artifact.retrieved_at,
                        "content_type": artifact.content_type,
                        "source_hash": artifact.source_hash,
                        "suffix": suffix,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8"),
        )

    def request_json(
        *,
        endpoint: str,
        source_url: str,
        params: dict[str, str],
    ) -> tuple[dict, SourceArtifact]:
        safe_params = {
            key: value
            for key, value in params.items()
            if key != "crtfc_key"
        }
        key = sha256_bytes(
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "endpoint": endpoint,
                    "params": safe_params,
                }
            )
        )
        artifact = cached_artifact(key)
        if artifact is not None:
            payload = json.loads(artifact.content)["response"]
            return payload, artifact
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                pace_request()
                response = requests.get(
                    source_url,
                    params=params,
                    timeout=45,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("OpenDART response is not an object.")
                status = str(payload.get("status") or "")
                if status not in {"000", "013"}:
                    raise RuntimeError(
                        "OpenDART status="
                        f"{status}: {payload.get('message')}"
                    )
                artifact = artifact_from_payload(
                    f"opendart_{endpoint}",
                    source_url,
                    {
                        "request": {
                            "endpoint": endpoint,
                            **safe_params,
                        },
                        "response": payload,
                    },
                )
                save_artifact(key, artifact, suffix="json")
                return payload, artifact
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(0.5 * (2**attempt))
        assert last_error is not None
        raise RuntimeError(
            f"OpenDART {endpoint} failed: {type(last_error).__name__}"
        ) from None

    def request_document(rcept_no: str) -> SourceArtifact:
        safe_url = f"{DART_FILING_URL}?rcpNo={rcept_no}"
        key = sha256_bytes(
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "endpoint": "document.xml",
                    "rcept_no": rcept_no,
                }
            )
        )
        artifact = cached_artifact(key)
        if artifact is not None:
            return artifact
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                pace_request()
                response = requests.get(
                    DART_DOCUMENT_URL,
                    params={
                        "crtfc_key": api_key,
                        "rcept_no": rcept_no,
                    },
                    timeout=45,
                )
                response.raise_for_status()
                content = response.content
                try:
                    with zipfile.ZipFile(io.BytesIO(content)) as archive:
                        if not archive.namelist():
                            raise ValueError(
                                "empty OpenDART document archive"
                            )
                except zipfile.BadZipFile:
                    if b"<status>014</status>" in content:
                        break
                    raise
                artifact = SourceArtifact(
                    source="opendart_document",
                    source_url=safe_url,
                    retrieved_at=utc_now_iso(),
                    content=content,
                    content_type="application/zip",
                )
                save_artifact(key, artifact, suffix="zip")
                return artifact
            except (
                requests.RequestException,
                OSError,
                ValueError,
                zipfile.BadZipFile,
            ) as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(0.5 * (2**attempt))

        # OpenDART returns status 014 for some final correction filings even
        # though the same official filing is available in the DART viewer.
        # Resolve its document number from the official main page and archive
        # the exact viewer HTML instead of falling back to a superseded filing.
        def public_get(url: str):
            public_error: Exception | None = None
            for public_attempt in range(5):
                try:
                    pace_request()
                    response = requests.get(url, timeout=45)
                    response.raise_for_status()
                    return response
                except requests.RequestException as exc:
                    public_error = exc
                    if public_attempt < 4:
                        time.sleep(1.0 * (2**public_attempt))
            assert public_error is not None
            raise public_error

        try:
            main_response = public_get(safe_url)
            main_text = main_response.text
            match = re.search(
                r'viewDoc\(\s*["\']'
                + re.escape(rcept_no)
                + r'["\']\s*,\s*["\'](\d+)["\']',
                main_text,
            )
            if match is None:
                raise ValueError("DART viewer document number is missing.")
            dcm_no = match.group(1)
            viewer_url = (
                "https://dart.fss.or.kr/report/viewer.do"
                f"?rcpNo={rcept_no}&dcmNo={dcm_no}"
                "&eleId=0&offset=0&length=0&dtd=HTML"
            )
            viewer_response = public_get(viewer_url)
            content = viewer_response.content
            if b"<html" not in content.lower() or b"<table" not in content.lower():
                raise ValueError("DART viewer did not return filing HTML.")
            artifact = SourceArtifact(
                source="dart_viewer_document",
                source_url=viewer_url,
                retrieved_at=utc_now_iso(),
                content=content,
                content_type="text/html",
            )
            save_artifact(key, artifact, suffix="html")
            return artifact
        except (requests.RequestException, ValueError) as fallback_error:
            detail = (
                type(last_error).__name__
                if last_error is not None
                else "OpenDARTStatus014"
            )
            raise RuntimeError(
                "OpenDART and DART viewer document failed for "
                f"{rcept_no}: {detail}/"
                f"{type(fallback_error).__name__}"
            ) from None

    def attachment_base_document(
        corp_code: str,
        filing: dict,
    ) -> tuple[
        dict[str, object],
        SourceArtifact,
        str,
        tuple[str, ...],
    ]:
        filing_date = pd.Timestamp(str(filing.get("rcept_dt") or ""))
        if pd.isna(filing_date):
            raise ValueError("attachment correction lacks a filing date")
        start_date = (
            filing_date - pd.Timedelta(days=370)
        ).date().strftime("%Y%m%d")
        end_date = filing_date.date().strftime("%Y%m%d")
        current_rcept = str(filing.get("rcept_no") or "")
        candidates: dict[str, dict] = {}
        history_artifact_hashes: set[str] = set()
        page_no = 1
        while True:
            payload, history_artifact = request_json(
                endpoint="disclosure_list_history",
                source_url=DART_DISCLOSURE_LIST_URL,
                params={
                    "crtfc_key": api_key,
                    "corp_code": corp_code,
                    "bgn_de": start_date,
                    "end_de": end_date,
                    "last_reprt_at": "N",
                    "pblntf_ty": "I",
                    "sort": "date",
                    "sort_mth": "desc",
                    "page_no": str(page_no),
                    "page_count": "100",
                },
            )
            history_artifact_hashes.add(
                history_artifact.source_hash
            )
            for value in payload.get("list") or ():
                report_name = re.sub(
                    r"\s+",
                    "",
                    str(value.get("report_nm") or ""),
                )
                rcept_no = str(value.get("rcept_no") or "")
                if (
                    rcept_no
                    and rcept_no != current_rcept
                    and "배당결정" in report_name
                    and any(
                        label in report_name
                        for label in ("현금", "현물")
                    )
                    and "자회사의주요경영사항" not in report_name
                    and "[첨부정정]" not in report_name
                ):
                    candidates[rcept_no] = dict(value)
            total_page = int(payload.get("total_page") or 0)
            if str(payload.get("status") or "") == "013" or page_no >= total_page:
                break
            page_no += 1
        for rcept_no in sorted(candidates, reverse=True):
            # An attachment-only correction can point at another correction
            # whose original document is no longer available through either
            # OpenDART or the public viewer.  That does not make the filing
            # chain incomplete when an older full-form filing is still
            # available, so keep walking the official history instead of
            # aborting on the first unavailable intermediate document.
            try:
                artifact = request_document(rcept_no)
            except RuntimeError:
                continue
            try:
                parsed = _parse_opendart_cash_dividend_document(
                    artifact.content
                )
            except ValueError:
                continue
            history_artifact_hashes.add(artifact.source_hash)
            return (
                parsed,
                artifact,
                rcept_no,
                tuple(sorted(history_artifact_hashes)),
            )
        raise ValueError(
            "attachment correction has no parseable base filing"
        )

    def collect_corp(
        corp_code: str,
    ) -> tuple[list[dict], list[str], tuple[str, ...]]:
        filings: dict[str, dict] = {}
        artifact_hashes: set[str] = set()
        page_no = 1
        while True:
            payload, artifact = request_json(
                endpoint="disclosure_list",
                source_url=DART_DISCLOSURE_LIST_URL,
                params={
                    "crtfc_key": api_key,
                    "corp_code": corp_code,
                    "bgn_de": requested_start.replace("-", ""),
                    "end_de": filing_end.replace("-", ""),
                    "last_reprt_at": "Y",
                    "pblntf_ty": "I",
                    "sort": "date",
                    "sort_mth": "asc",
                    "page_no": str(page_no),
                    "page_count": "100",
                },
            )
            artifact_hashes.add(artifact.source_hash)
            for filing in payload.get("list") or ():
                report_name = re.sub(
                    r"\s+",
                    "",
                    str(filing.get("report_nm") or ""),
                )
                if (
                    "배당결정" not in report_name
                    or not any(
                        value in report_name
                        for value in ("현금", "현물")
                    )
                    or "자회사의주요경영사항" in report_name
                ):
                    continue
                rcept_no = str(filing.get("rcept_no") or "").strip()
                if rcept_no:
                    filings[rcept_no] = dict(filing)
            total_page = int(payload.get("total_page") or 0)
            if str(payload.get("status") or "") == "013" or page_no >= total_page:
                break
            page_no += 1

        records: list[dict] = []
        issues: list[str] = []
        for rcept_no, filing in sorted(filings.items()):
            data_rcept_no = rcept_no
            try:
                document = request_document(rcept_no)
                artifact_hashes.add(document.source_hash)
                try:
                    parsed = _parse_opendart_cash_dividend_document(
                        document.content
                    )
                except ValueError:
                    if "[첨부정정]" not in str(
                        filing.get("report_nm") or ""
                    ):
                        raise
                    (
                        parsed,
                        document,
                        data_rcept_no,
                        base_artifact_hashes,
                    ) = attachment_base_document(corp_code, filing)
                    artifact_hashes.update(base_artifact_hashes)
            except Exception as exc:
                issues.append(
                    f"{corp_code}:{rcept_no}:{type(exc).__name__}"
                )
                records.append(
                    {
                        "corp_code": corp_code,
                        "rcept_no": rcept_no,
                        "data_rcept_no": "",
                        "report_name": str(filing.get("report_nm") or "").strip(),
                        "filing_date": _date_text(filing.get("rcept_dt")),
                        "record_date": "",
                        "payment_date": "",
                        "board_date": "",
                        "cash_amount": None,
                        "security_id": "",
                        "disposition": "parse_error",
                        "source_url": (
                            f"{DART_FILING_URL}?rcpNo={rcept_no}"
                        ),
                        "source_hash": "",
                    }
                )
                continue
            artifact_hashes.add(document.source_hash)
            record_date = str(parsed.get("record_date") or "")
            cash_amount = parsed.get("cash_amount")
            target_id = ""
            disposition = "zero_or_non_cash"
            if cash_amount is not None and float(cash_amount) > 0:
                if not record_date:
                    disposition = "parse_error"
                    issues.append(
                        f"{corp_code}:{rcept_no}:missing_record_date"
                    )
                elif (
                    record_date < requested_start
                    or record_date > requested_end
                ):
                    disposition = "outside_requested_period"
                else:
                    matches = [
                        value
                        for value in identities_by_corp[corp_code]
                        if (
                            not value["active_from"]
                            or value["active_from"] <= record_date
                        )
                        and (
                            not value["active_to"]
                            or value["active_to"] >= record_date
                        )
                    ]
                    if len(matches) == 1:
                        target_id = str(matches[0]["security_id"])
                        disposition = "in_scope"
                    elif not matches:
                        disposition = "outside_identity_period"
                    else:
                        disposition = "identity_error"
                        issues.append(
                            f"{corp_code}:{rcept_no}:"
                            f"identity_matches={len(matches)}"
                        )
            records.append(
                {
                    "corp_code": corp_code,
                    "rcept_no": rcept_no,
                    "data_rcept_no": data_rcept_no,
                    "report_name": str(filing.get("report_nm") or "").strip(),
                    "filing_date": _date_text(filing.get("rcept_dt")),
                    "record_date": record_date,
                    "payment_date": str(parsed.get("payment_date") or ""),
                    "board_date": str(parsed.get("board_date") or ""),
                    "cash_amount": cash_amount,
                    "security_id": target_id,
                    "disposition": disposition,
                    "source_url": document.source_url,
                    "source_hash": document.source_hash,
                }
            )
        return records, issues, tuple(sorted(artifact_hashes))

    records: list[dict] = []
    issues = [f"missing_dart_corp_code:{value}" for value in missing_dart]
    raw_artifact_hashes: set[str] = set()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(collect_corp, corp_code): corp_code
            for corp_code in sorted(identities_by_corp)
        }
        for future in as_completed(futures):
            corp_code = futures[future]
            try:
                (
                    corp_records,
                    corp_issues,
                    artifact_hashes,
                ) = future.result()
            except Exception as exc:
                issues.append(
                    f"{corp_code}:collection:{type(exc).__name__}"
                )
                continue
            records.extend(corp_records)
            issues.extend(corp_issues)
            raw_artifact_hashes.update(artifact_hashes)

    decisions = pd.DataFrame(
        [
            {
                "security_id": value["security_id"],
                "record_date": value["record_date"],
                "cash_amount": value["cash_amount"],
                "announcement_date": (
                    value["board_date"] or value["filing_date"]
                ),
                "payment_date": value["payment_date"],
                "rcept_no": value["rcept_no"],
                "source_url": value["source_url"],
                "source_hash": value["source_hash"],
            }
            for value in records
            if value["disposition"] == "in_scope"
        ],
        columns=_opendart_dividend_columns(),
    ).drop_duplicates(
        ["security_id", "record_date", "cash_amount"],
        keep="last",
    )
    status = "passed" if not issues else "blocked"
    normalized_records = sorted(
        records,
        key=lambda value: (
            str(value["corp_code"]),
            str(value["rcept_no"]),
        ),
    )
    report = artifact_from_payload(
        "opendart_action_audit",
        DART_DISCLOSURE_LIST_URL,
        {
            "schema": "opendart_dividend_collection/v1",
            "status": status,
            "requested_start": requested_start,
            "requested_end": requested_end,
            "filing_search_end": filing_end,
            "stock_identity_count": len(identity_rows) + len(missing_dart),
            "issuer_count": len(identities_by_corp),
            "missing_dart_identity_count": len(missing_dart),
            "cash_dividend_filing_count": len(records),
            "in_scope_dividend_count": len(decisions),
            "raw_artifact_count": len(raw_artifact_hashes),
            "raw_artifact_inventory_sha256": sha256_bytes(
                canonical_json_bytes(sorted(raw_artifact_hashes))
            ),
            "raw_artifact_hashes": sorted(raw_artifact_hashes),
            "issues": sorted(issues),
            "records": normalized_records,
        },
    )
    return KrDartDividendResult(
        status,
        decisions,
        report,
        (
            f"issuers={len(identities_by_corp)};"
            f"filings={len(records)};"
            f"decisions={len(decisions)};"
            f"issues={len(issues)}"
        ),
    )


def _opendart_dividend_columns() -> list[str]:
    return [
        "security_id",
        "record_date",
        "cash_amount",
        "announcement_date",
        "payment_date",
        "rcept_no",
        "source_url",
        "source_hash",
    ]


def _empty_opendart_dividend_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_opendart_dividend_columns())


def _parse_opendart_cash_dividend_document(content: bytes) -> dict[str, object]:
    """Parse the ordinary-share cash amount and dates from a DART ZIP."""

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            payloads = [
                archive.read(name)
                for name in archive.namelist()
                if not name.endswith("/")
            ]
    except zipfile.BadZipFile:
        payloads = [content]
    if not payloads:
        raise ValueError("OpenDART document archive is empty.")

    decoded: list[str] = []
    for payload in payloads:
        candidates: list[tuple[int, str]] = []
        for encoding in ("euc-kr", "utf-8"):
            text = payload.decode(encoding, errors="replace")
            score = sum(
                text.count(keyword)
                for keyword in (
                    "배당기준일",
                    "1주당 배당금",
                    "보통주",
                    "배당금지급",
                )
            ) - text.count("\ufffd")
            candidates.append((score, text))
        decoded.append(max(candidates, key=lambda value: value[0])[1])

    rows: list[list[str]] = []
    for text in decoded:
        try:
            tables = pd.read_html(io.StringIO(text))
        except (ImportError, ValueError):
            continue
        for table in tables:
            for values in table.astype(object).itertuples(index=False, name=None):
                rows.append(
                    [
                        re.sub(r"\s+", "", str(value or ""))
                        for value in values
                        if str(value or "").strip().lower() != "nan"
                    ]
                )
    if not rows:
        raise ValueError("OpenDART dividend filing contains no readable table.")

    cash_amount: float | None = None
    record_date = ""
    payment_date = ""
    board_date = ""
    for row in rows:
        joined = "|".join(row)
        if (
            "1주당배당금" in joined
            and "보통주" in joined
        ):
            numeric: list[float] = []
            for cell in row:
                if any(
                    label in cell
                    for label in ("1주당배당금", "보통주", "종류주")
                ):
                    continue
                normalized = cell.replace(",", "")
                if re.fullmatch(r"-?\d+(?:\.\d+)?", normalized):
                    numeric.append(float(normalized))
            if numeric:
                cash_amount = numeric[0]
            elif "-" in row:
                cash_amount = 0.0
        # A corrected filing starts with a comparison table containing both
        # the superseded and corrected values, then repeats the complete final
        # form.  Keep the last parseable occurrence so the authoritative main
        # form wins over "정정전"; taking the first occurrence silently
        # resurrects the value that the issuer explicitly corrected.
        if "배당기준일" in joined:
            record_date = _first_date_in_cells(row) or record_date
        if "배당금지급예정일" in joined:
            payment_date = _first_date_in_cells(row) or payment_date
        if "이사회결의일" in joined:
            board_date = _first_date_in_cells(row) or board_date
    if cash_amount is None:
        raise ValueError("ordinary-share cash dividend amount was not found.")
    return {
        "cash_amount": cash_amount,
        "record_date": record_date,
        "payment_date": payment_date,
        "board_date": board_date,
    }


def _first_date_in_cells(values: Iterable[str]) -> str:
    for value in values:
        match = re.search(
            r"(20\d{2})[-./](\d{1,2})[-./](\d{1,2})",
            str(value),
        )
        if not match:
            continue
        try:
            return pd.Timestamp(
                year=int(match.group(1)),
                month=int(match.group(2)),
                day=int(match.group(3)),
            ).date().isoformat()
        except ValueError:
            continue
    return ""


def fetch_yahoo_prices(
    catalog: KrIdentityCatalog,
    *,
    start: str,
    end: str,
    batch_size: int = 80,
) -> KrProviderResult:
    try:
        import yfinance as yf
    except ModuleNotFoundError:
        return KrProviderResult("yahoo", "failed", _empty_price_frame(), detail="yfinance missing")
    identities = _identities_for_range(catalog.frame, start, end)
    by_ticker: dict[str, list[dict]] = {}
    for identity in identities:
        by_ticker.setdefault(str(identity["yahoo_symbol"]), []).append(identity)
    rows: list[dict] = []
    artifacts: list[SourceArtifact] = []
    for tickers in _batched(tuple(by_ticker), batch_size):
        try:
            raw = yf.download(
                tickers=list(tickers),
                start=start,
                end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d",
                auto_adjust=False,
                actions=False,
                group_by="ticker",
                progress=False,
                threads=True,
            )
        except Exception as exc:
            return KrProviderResult(
                "yahoo", "failed", _empty_price_frame(), detail=type(exc).__name__
            )
        for ticker in tickers:
            frame = _extract_yahoo_ticker(raw, ticker, len(tickers))
            artifact = artifact_from_payload(
                "yahoo_kr_daily",
                f"{YAHOO_SOURCE_URL}quote/{ticker}/history",
                {
                    "request": {"ticker": ticker, "start": start, "end": end},
                    "rows": _frame_records(frame.reset_index()),
                },
            )
            artifacts.append(artifact)
            # A six-character ticker can be reused by a different ISIN. Download
            # once, then assign each date only to the identity interval that
            # covers it instead of letting a dict overwrite the old issuer.
            for identity in by_ticker[ticker]:
                rows.extend(
                    _records_from_standard_frame(
                        frame,
                        identity,
                        artifact,
                        "yahoo_kr_daily",
                    )
                )
    return KrProviderResult(
        "yahoo",
        "ok" if rows else "failed",
        pd.DataFrame(rows, columns=_price_columns()),
        tuple(artifacts),
    )


def fetch_naver_prices(
    catalog: KrIdentityCatalog,
    *,
    start: str,
    end: str,
    workers: int = 4,
) -> KrProviderResult:
    try:
        import FinanceDataReader as fdr
    except ModuleNotFoundError:
        return KrProviderResult(
            "naver", "failed", _empty_price_frame(), detail="finance-datareader missing"
        )
    identities = _identities_for_range(catalog.frame, start, end)

    def fetch_one(identity: dict):
        symbol = str(identity["primary_symbol"])
        frame = fdr.DataReader(f"NAVER:{symbol}", start, end)
        artifact = artifact_from_payload(
            "naver_kr_daily",
            f"{NAVER_SOURCE_URL}item/sise_day.naver?code={symbol}",
            {
                "request": {"symbol": symbol, "start": start, "end": end},
                "rows": _frame_records(frame.reset_index()),
            },
        )
        return _records_from_standard_frame(frame, identity, artifact, "naver_kr_daily"), artifact

    rows: list[dict] = []
    artifacts: list[SourceArtifact] = []
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(fetch_one, identity) for identity in identities]
        for future in as_completed(futures):
            try:
                values, artifact = future.result()
            except Exception:
                failures += 1
                continue
            rows.extend(values)
            artifacts.append(artifact)
    return KrProviderResult(
        "naver",
        "ok" if rows else "failed",
        pd.DataFrame(rows, columns=_price_columns()),
        tuple(artifacts),
        f"failed_symbols={failures}" if failures else "",
    )


def fetch_kis_prices(
    catalog: KrIdentityCatalog,
    *,
    start: str,
    end: str,
    checkpoint_root: str | Path | None = None,
    workers: int = 4,
) -> KrProviderResult:
    load_env()
    mode, app_key, app_secret, base_url = _kis_market_data_credentials()
    if not app_key or not app_secret:
        return KrProviderResult(
            "kis",
            "skipped_missing_credentials",
            _empty_price_frame(),
            detail=(
                f"KIS_{mode.upper()}_APP_KEY and KIS_{mode.upper()}_APP_SECRET "
                "are required (legacy KIS_APP_KEY/KIS_APP_SECRET are accepted "
                "for prod)."
            ),
        )
    import requests

    token = _kis_access_token(
        requests,
        mode=mode,
        app_key=app_key,
        app_secret=app_secret,
        base_url=base_url,
    )
    headers = {
        "authorization": f"Bearer {token}",
        "appkey": app_key,
        "appsecret": app_secret,
        "tr_id": "FHKST03010100",
        "custtype": "P",
    }
    chunk_root = Path(checkpoint_root) if checkpoint_root is not None else None
    price_chunk_root = (
        chunk_root / "providers" / "kis" / "chunks"
        if chunk_root is not None
        else None
    )
    evidence_root = (
        chunk_root / "evidence_local" / "kis"
        if chunk_root is not None
        else None
    )
    frames: list[pd.DataFrame] = []
    artifacts: list[SourceArtifact | KrArtifactReference] = []
    fetched_chunk_count = 0
    cached_chunk_count = 0
    request_interval = max(
        0.05,
        float(
            os.getenv(
                "KIS_MARKET_DATA_MIN_REQUEST_INTERVAL_SECONDS",
                "0.065",
            ).strip()
            or "0.065"
        ),
    )
    request_pacing_lock = threading.Lock()
    next_request_at = 0.0

    def pace_request() -> None:
        """Keep aggregate request starts below the documented API limit."""

        nonlocal next_request_at
        with request_pacing_lock:
            now = time.monotonic()
            wait_seconds = max(0.0, next_request_at - now)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            next_request_at = max(
                next_request_at,
                time.monotonic(),
            ) + request_interval

    def chunk_key(params: dict[str, str], security_id: str) -> str:
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "provider": "kis",
                    "security_id": security_id,
                    "params": params,
                }
            )
        )

    def load_chunk(
        key: str,
        params: dict[str, str],
        security_id: str,
    ) -> tuple[pd.DataFrame, KrArtifactReference] | None:
        if price_chunk_root is None or evidence_root is None:
            return None
        parquet_path = price_chunk_root / f"{key}.parquet"
        metadata_path = price_chunk_root / f"{key}.json"
        if not parquet_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("schema_version") != 1
                or metadata.get("provider") != "kis"
                or metadata.get("security_id") != security_id
                or metadata.get("params") != params
            ):
                return None
            source_hash = str(metadata.get("source_hash") or "").lower()
            if not _valid_sha256_text(source_hash):
                return None
            # KIS chunks are always archived as JSON under their content
            # hash.  Using ``glob`` here rescanned a directory containing
            # tens of thousands of files once per chunk, turning a resumable
            # full-history validation into O(N²) directory work.
            artifact_path = evidence_root / f"{source_hash}.json.gz"
            if not artifact_path.is_file():
                return None
            content = gzip.decompress(artifact_path.read_bytes())
            if sha256_bytes(content) != source_hash:
                return None
            frame = pd.read_parquet(parquet_path)
            row_hashes = {
                str(value).strip().lower()
                for value in frame.get("source_hash", pd.Series(dtype=str))
                if str(value).strip()
            }
            if row_hashes and row_hashes != {source_hash}:
                return None
            if int(metadata.get("row_count") or 0) != len(frame):
                return None
            return frame, KrArtifactReference(source_hash)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def save_chunk(
        key: str,
        params: dict[str, str],
        security_id: str,
        frame: pd.DataFrame,
        artifact: SourceArtifact,
    ) -> None:
        if price_chunk_root is None or evidence_root is None:
            return
        price_chunk_root.mkdir(parents=True, exist_ok=True)
        evidence_root.mkdir(parents=True, exist_ok=True)
        artifact_path = evidence_root / (
            f"{artifact.source_hash}.json.gz"
        )
        if not artifact_path.is_file():
            write_atomic(artifact_path, gzip.compress(artifact.content))
        parquet_path = price_chunk_root / f"{key}.parquet"
        temporary = parquet_path.with_name(
            f".{parquet_path.name}.{os.getpid()}.tmp"
        )
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, parquet_path)
        write_atomic(
            price_chunk_root / f"{key}.json",
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "provider": "kis",
                        "security_id": security_id,
                        "params": params,
                        "row_count": len(frame),
                        "source_hash": artifact.source_hash,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8"),
        )

    def request_price_segments(
        params: dict[str, str],
    ) -> list[dict[str, object]]:
        """Fetch one KIS window, bisecting a persistently broken 5xx window."""

        try:
            response = _kis_get_with_retry(
                requests,
                (
                    f"{base_url}/uapi/domestic-stock/v1/quotations/"
                    "inquire-daily-itemchartprice"
                ),
                headers=headers,
                params=params,
                before_request=pace_request,
            )
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            segment_start = pd.Timestamp(
                params["FID_INPUT_DATE_1"]
            )
            segment_end = pd.Timestamp(
                params["FID_INPUT_DATE_2"]
            )
            if (
                status not in {429, 500, 502, 503, 504}
                or segment_start >= segment_end
            ):
                raise
            midpoint = segment_start + pd.Timedelta(
                days=(segment_end - segment_start).days // 2
            )
            left = {
                **params,
                "FID_INPUT_DATE_1": segment_start.strftime(
                    "%Y%m%d"
                ),
                "FID_INPUT_DATE_2": midpoint.strftime("%Y%m%d"),
            }
            right = {
                **params,
                "FID_INPUT_DATE_1": (
                    midpoint + pd.Timedelta(days=1)
                ).strftime("%Y%m%d"),
                "FID_INPUT_DATE_2": segment_end.strftime("%Y%m%d"),
            }
            return [
                *request_price_segments(left),
                *request_price_segments(right),
            ]
        payload = response.json()
        if str(payload.get("rt_cd", "0")) not in {"", "0"}:
            message_code = str(payload.get("msg_cd") or "unknown")
            message = str(
                payload.get("msg1") or "KIS request rejected"
            )
            raise RuntimeError(
                "KIS daily price request failed: "
                f"{message_code}: {message}"
            )
        return [
            {
                "request": {
                    key: value for key, value in params.items()
                },
                "rows": payload.get("output2") or [],
            }
        ]

    def fetch_identity(
        identity: dict,
    ) -> tuple[
        list[pd.DataFrame],
        list[SourceArtifact | KrArtifactReference],
        int,
        int,
    ]:
        identity_frames: list[pd.DataFrame] = []
        identity_artifacts: list[SourceArtifact | KrArtifactReference] = []
        identity_fetched_chunks = 0
        identity_cached_chunks = 0
        symbol = str(identity["primary_symbol"])
        security_id = str(identity["security_id"])
        identity_start = max(
            pd.Timestamp(start),
            pd.Timestamp(_date_text(identity.get("active_from")) or start),
        )
        identity_end = min(
            pd.Timestamp(end),
            pd.Timestamp(_date_text(identity.get("active_to")) or end),
        )
        if identity_end < identity_start:
            return (
                identity_frames,
                identity_artifacts,
                identity_fetched_chunks,
                identity_cached_chunks,
            )
        cursor_end = identity_end
        while cursor_end >= identity_start:
            cursor_start = max(
                identity_start,
                cursor_end - pd.Timedelta(days=99),
            )
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_DATE_1": cursor_start.strftime("%Y%m%d"),
                "FID_INPUT_DATE_2": cursor_end.strftime("%Y%m%d"),
                "FID_PERIOD_DIV_CODE": "D",
                # KIS: 0=adjusted, 1=original.  Cross-provider validation must
                # compare the unadjusted exchange observation.
                "FID_ORG_ADJ_PRC": "1",
            }
            key = chunk_key(params, security_id)
            cached = load_chunk(key, params, security_id)
            if cached is not None:
                frame, artifact_reference = cached
                identity_frames.append(frame)
                identity_artifacts.append(artifact_reference)
                identity_cached_chunks += 1
                cursor_end = cursor_start - pd.Timedelta(days=1)
                continue
            segments = request_price_segments(params)
            values = [
                value
                for segment in segments
                for value in segment["rows"]
            ]
            artifact_payload = (
                segments[0]
                if len(segments) == 1
                and segments[0]["request"] == params
                else {
                    "request": {
                        key: value
                        for key, value in params.items()
                    },
                    "segments": segments,
                }
            )
            artifact = artifact_from_payload(
                "kis_kr_daily",
                KIS_SOURCE_URL,
                artifact_payload,
            )
            chunk_rows: list[dict] = []
            for value in values:
                session = _date_text(value.get("stck_bsop_date"))
                if not session or not _identity_covers(identity, session):
                    continue
                try:
                    numeric = [
                        float(value.get(key))
                        for key in ("stck_oprc", "stck_hgpr", "stck_lwpr", "stck_clpr")
                    ]
                    volume = float(value.get("acml_vol") or 0)
                except (TypeError, ValueError):
                    continue
                if any(item <= 0 for item in numeric):
                    continue
                chunk_rows.append(
                    _price_record(
                        security_id,
                        symbol,
                        session,
                        *numeric,
                        volume,
                        source="kis_kr_daily",
                        source_url=KIS_SOURCE_URL,
                        retrieved_at=artifact.retrieved_at,
                        source_hash=artifact.source_hash,
                        observation_status=(
                            "suspended_or_no_trade" if volume == 0 else "traded"
                        ),
                        exchange=str(identity.get("exchange") or ""),
                        asset_type=str(identity.get("asset_type") or "STOCK"),
                    )
                )
            frame = pd.DataFrame(chunk_rows, columns=_price_columns())
            save_chunk(
                key,
                params,
                security_id,
                frame,
                artifact,
            )
            identity_frames.append(frame)
            identity_artifacts.append(
                KrArtifactReference(artifact.source_hash)
                if checkpoint_root is not None
                else artifact
            )
            identity_fetched_chunks += 1
            cursor_end = cursor_start - pd.Timedelta(days=1)
        consolidated_frames = (
            [
                pd.concat(identity_frames, ignore_index=True)
                .drop_duplicates(
                    ["security_id", "session"],
                    keep="last",
                )
            ]
            if identity_frames
            else []
        )
        return (
            consolidated_frames,
            identity_artifacts,
            identity_fetched_chunks,
            identity_cached_chunks,
        )

    identities = _identities_for_range(catalog.frame, start, end)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(fetch_identity, identity)
            for identity in identities
        ]
        for future in as_completed(futures):
            (
                identity_frames,
                identity_artifacts,
                identity_fetched_chunks,
                identity_cached_chunks,
            ) = future.result()
            frames.extend(identity_frames)
            artifacts.extend(identity_artifacts)
            fetched_chunk_count += identity_fetched_chunks
            cached_chunk_count += identity_cached_chunks
    prices = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["security_id", "session"], keep="last")
        if frames
        else _empty_price_frame()
    )
    return KrProviderResult(
        "kis",
        "ok" if not prices.empty else "failed",
        prices,
        tuple(artifacts),
        (
            f"fetched_chunks={fetched_chunk_count};"
            f"cached_chunks={cached_chunk_count}"
        ),
    )


def _kis_get_with_retry(
    requests_module,
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, str],
    before_request: Callable[[], None] | None = None,
):
    """Retry only transient KIS quotation failures without hiding rejections."""

    last_exception: Exception | None = None
    # KIS occasionally returns a short burst of HTTP 500 responses for an
    # otherwise valid historical window.  Four attempts covered only 3.5
    # seconds and could abort a 27k-chunk resumable audit for a momentary
    # backend incident.  Retry transient statuses for roughly half a minute;
    # authentication and other non-transient rejections still fail
    # immediately.
    for attempt in range(8):
        try:
            if before_request is not None:
                before_request()
            response = requests_module.get(
                url,
                headers=headers,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            return response
        except Exception as exc:
            last_exception = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status is None or status in {
                429,
                500,
                502,
                503,
                504,
            }
            if not retryable or attempt == 7:
                raise
            time.sleep(min(0.5 * (2**attempt), 8.0))
    assert last_exception is not None
    raise last_exception


def _kis_market_data_credentials() -> tuple[str, str, str, str]:
    mode = os.getenv("KIS_MARKET_DATA_MODE", "prod").strip().lower() or "prod"
    if mode not in {"prod", "paper"}:
        raise ValueError("KIS_MARKET_DATA_MODE must be prod or paper.")
    prefix = f"KIS_{mode.upper()}"
    app_key = os.getenv(f"{prefix}_APP_KEY", "").strip()
    app_secret = os.getenv(f"{prefix}_APP_SECRET", "").strip()
    if mode == "prod":
        app_key = app_key or os.getenv("KIS_APP_KEY", "").strip()
        app_secret = app_secret or os.getenv("KIS_APP_SECRET", "").strip()
    default_base_url = KIS_PROD_BASE_URL if mode == "prod" else KIS_PAPER_BASE_URL
    base_url = (
        os.getenv(f"{prefix}_BASE_URL", "").strip()
        or (os.getenv("KIS_BASE_URL", "").strip() if mode == "prod" else "")
        or default_base_url
    ).rstrip("/")
    return mode, app_key, app_secret, base_url


def _kis_access_token(
    requests_module,
    *,
    mode: str,
    app_key: str,
    app_secret: str,
    base_url: str,
) -> str:
    """Reuse the 24-hour KIS token instead of repeatedly issuing one.

    KIS documents token issuance as once per day in normal operation and rate
    limits reissuance.  The cache is credential-bound and stored with mode 600.
    """

    cache_path = Path(
        os.getenv(
            "KIS_TOKEN_CACHE_PATH",
            f"data/cache/private/kis_{mode}_access_token.json",
        )
    ).expanduser()
    credential_fingerprint = sha256_bytes(
        f"{mode}\0{base_url}\0{app_key}".encode()
    )
    with _KIS_TOKEN_LOCK:
        cached = _read_kis_token_cache(
            cache_path,
            credential_fingerprint=credential_fingerprint,
        )
        if cached:
            return cached
        token_response = requests_module.post(
            f"{base_url}/oauth2/tokenP",
            json={
                "grant_type": "client_credentials",
                "appkey": app_key,
                "appsecret": app_secret,
            },
            timeout=30,
        )
        token_response.raise_for_status()
        payload = token_response.json()
        token = str(payload.get("access_token") or "").strip()
        if not token:
            raise RuntimeError("KIS token response did not contain access_token.")
        now = datetime.now(UTC)
        try:
            expires_in = int(payload.get("expires_in") or 86_400)
        except (TypeError, ValueError):
            expires_in = 86_400
        # Never trust an unexpectedly long lifetime and refresh a few minutes
        # before the official expiry.
        expires_in = max(60, min(expires_in, 86_400))
        expires_at = now + timedelta(seconds=expires_in)
        write_atomic(
            cache_path,
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "mode": mode,
                        "credential_fingerprint": credential_fingerprint,
                        "created_at": now.isoformat(),
                        "expires_at": expires_at.isoformat(),
                        "access_token": token,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode(),
        )
        os.chmod(cache_path, 0o600)
        return token


def _read_kis_token_cache(
    path: Path,
    *,
    credential_fingerprint: str,
) -> str:
    if not path.is_file():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("credential_fingerprint") != credential_fingerprint:
            return ""
        token = str(payload.get("access_token") or "").strip()
        expires_at = pd.Timestamp(payload.get("expires_at"))
        if expires_at.tzinfo is None:
            expires_at = expires_at.tz_localize("UTC")
        now = pd.Timestamp.now(tz="UTC")
        if not token or expires_at <= now + pd.Timedelta(minutes=5):
            return ""
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return ""
    return token


def _fetch_dart_corp_codes() -> tuple[dict[str, str], SourceArtifact | None, str]:
    load_env()
    api_key = os.getenv("DART_API_KEY", "").strip()
    if not api_key:
        return {}, None, "skipped_missing_credentials"
    import requests

    try:
        response = requests.get(DART_CORP_CODE_URL, params={"crtfc_key": api_key}, timeout=60)
        response.raise_for_status()
        content = response.content
    except Exception as exc:
        return {}, None, f"failed:{type(exc).__name__}"
    artifact = SourceArtifact(
        source="opendart_corp_codes",
        source_url=DART_CORP_CODE_URL,
        retrieved_at=utc_now_iso(),
        content=content,
        content_type="application/zip",
    )
    mapping = _dart_mapping_from_corp_code_zip(content)
    if not mapping:
        return {}, artifact, "failed_empty"
    return mapping, artifact, "ok"


def _dart_mapping_from_corp_code_zip(content: bytes) -> dict[str, str]:
    """Return unambiguous six-character stock-code to DART corp-code links."""

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            xml = archive.read("CORPCODE.xml")
        root = pd.read_xml(io.BytesIO(xml), xpath=".//list")
    except (KeyError, OSError, ValueError, zipfile.BadZipFile):
        return {}
    if root is None or root.empty:
        return {}
    candidates: dict[str, set[str]] = {}
    for row in root.itertuples(index=False):
        raw_symbol = str(getattr(row, "stock_code", "") or "").strip()
        corp_code = str(getattr(row, "corp_code", "") or "").strip()
        symbol = normalize_kr_symbol(raw_symbol)
        if not is_valid_kr_symbol(symbol) or not corp_code:
            continue
        candidates.setdefault(symbol, set()).add(corp_code.zfill(8))
    return {
        symbol: next(iter(corp_codes))
        for symbol, corp_codes in candidates.items()
        if len(corp_codes) == 1
    }


def attach_dart_codes(catalog: pd.DataFrame, symbol_to_corp: dict[str, str]) -> pd.DataFrame:
    output = catalog.copy()
    output["dart_corp_code"] = output["primary_symbol"].map(symbol_to_corp).fillna("")
    return output


def _attach_unambiguous_dart_codes(
    catalog: pd.DataFrame,
    symbol_to_corp: dict[str, str],
) -> pd.DataFrame:
    """Attach ticker-only DART mappings only to one unambiguous issuer.

    OpenDART's corp-code inventory does not carry ISIN. A historical ticker can
    therefore point at a later issuer after reuse.  A symbol that maps to one
    stable KRX identity across all of its intervals is safe, including a
    delisted identity.  If KRX shows multiple identities, attach only when one
    currently active identity is unambiguous and leave historical rows blank.
    """

    output = catalog.copy()
    output["dart_corp_code"] = ""
    active_to = output["active_to"].fillna("").astype(str).str.strip()
    for symbol, corp_code in symbol_to_corp.items():
        symbol_matches = output.loc[
            output["primary_symbol"].astype(str).eq(normalize_kr_symbol(symbol))
            & output["identity_mapped"].eq(True)  # noqa: E712
        ]
        security_ids = symbol_matches["security_id"].astype(str).drop_duplicates()
        if len(security_ids) == 1:
            output.loc[
                output["security_id"].astype(str).eq(str(security_ids.iloc[0])),
                "dart_corp_code",
            ] = str(corp_code)
            continue
        current = symbol_matches.loc[active_to.loc[symbol_matches.index].eq("")]
        current_ids = current["security_id"].astype(str).drop_duplicates()
        if len(current_ids) == 1:
            output.loc[
                output.index.isin(current.index)
                & output["security_id"].astype(str).eq(str(current_ids.iloc[0])),
                "dart_corp_code",
            ] = str(corp_code)
    return output


def _find_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    required: bool = True,
) -> str | None:
    normalized = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return str(candidate)
        found = normalized.get(candidate.strip().lower())
        if found is not None:
            return found
    if required:
        raise ValueError(f"Missing expected column; candidates={', '.join(candidates)}")
    return None


def _normalize_exchange(value) -> str:
    text = str(value or "").strip().upper()
    if "KOSDAQ" in text or text in {"KQ", "KQ11"}:
        return "KOSDAQ"
    if "KOSPI" in text or text in {"KS", "KS11", "STK"}:
        return "KOSPI"
    return text


def _normalize_asset_type(value) -> str:
    text = str(value or "").strip().upper()
    if "ETF" in text:
        return "ETF"
    if "ETN" in text:
        return "ETN"
    return "STOCK"


def _normalize_index_name(value: str) -> str:
    return "".join(str(value).lower().split()).replace("kospi", "코스피").replace("kosdaq", "코스닥")


def _date_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    return "" if pd.isna(parsed) else pd.Timestamp(parsed).date().isoformat()


def _optional_float(value) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if pd.notna(output) else None


def _krx_number(value) -> float | None:
    text = str(value or "").strip().replace(",", "")
    if not text or text in {"-", "+"}:
        return None
    return _optional_float(text)


def _split_ratio(value) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            return float(numerator) / float(denominator)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return _optional_float(text)


def _frame_records(frame: pd.DataFrame) -> list[dict]:
    if frame is None or frame.empty:
        return []
    safe = frame.copy()
    safe.columns = [str(value) for value in safe.columns]
    safe = safe.where(pd.notna(safe), None)
    records = []
    for record in safe.to_dict("records"):
        records.append(
            {
                str(key): (
                    value.isoformat()
                    if isinstance(value, (pd.Timestamp, datetime))
                    else value.item()
                    if hasattr(value, "item")
                    else value
                )
                for key, value in record.items()
            }
        )
    return records


def _identities_for_range(frame: pd.DataFrame, start: str, end: str) -> tuple[dict, ...]:
    starts = pd.to_datetime(frame["active_from"], errors="coerce")
    ends = pd.to_datetime(frame["active_to"], errors="coerce")
    selected = frame.loc[
        (starts.isna() | starts.le(pd.Timestamp(end)))
        & (ends.isna() | ends.ge(pd.Timestamp(start)))
    ]
    return tuple(selected.to_dict("records"))


def _identity_covers(identity: dict, session: str) -> bool:
    when = pd.Timestamp(session)
    start = pd.to_datetime(identity.get("active_from"), errors="coerce")
    end = pd.to_datetime(identity.get("active_to"), errors="coerce")
    return bool((pd.isna(start) or start <= when) and (pd.isna(end) or end >= when))


def _extract_yahoo_ticker(raw: pd.DataFrame, ticker: str, batch_count: int) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(0):
            return raw[ticker].copy()
        if ticker in raw.columns.get_level_values(1):
            return raw.xs(ticker, axis=1, level=1).copy()
        return pd.DataFrame()
    return raw.copy() if batch_count == 1 else pd.DataFrame()


def _records_from_standard_frame(
    frame: pd.DataFrame,
    identity: dict,
    artifact: SourceArtifact,
    source: str,
) -> list[dict]:
    if frame is None or frame.empty:
        return []
    working = frame.copy()
    working.columns = [str(value).title() for value in working.columns]
    if not {"Open", "High", "Low", "Close"}.issubset(working.columns):
        return []
    rows = []
    for index, value in working.iterrows():
        session = pd.Timestamp(index).date().isoformat()
        if not _identity_covers(identity, session):
            continue
        try:
            numeric = [float(value[column]) for column in ("Open", "High", "Low", "Close")]
            volume = float(value.get("Volume", 0) or 0)
        except (TypeError, ValueError):
            continue
        if any(not pd.notna(item) or item <= 0 for item in numeric):
            continue
        rows.append(
            _price_record(
                str(identity["security_id"]),
                str(identity["primary_symbol"]),
                session,
                *numeric,
                volume,
                source=source,
                source_url=artifact.source_url,
                retrieved_at=artifact.retrieved_at,
                source_hash=artifact.source_hash,
                observation_status=(
                    "suspended_or_no_trade" if volume == 0 else "traded"
                ),
                exchange=str(identity.get("exchange") or ""),
                asset_type=str(identity.get("asset_type") or "STOCK"),
            )
        )
    return rows


def _price_record(
    security_id: str,
    symbol: str,
    session: str,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    *,
    source: str,
    source_url: str,
    retrieved_at: str,
    source_hash: str,
    observation_status: str = "traded",
    exchange: str = "",
    asset_type: str = "STOCK",
    security_name: str = "",
    official_reference_price: float | None = None,
    official_fluctuation_rate: float | None = None,
) -> dict:
    return {
        "security_id": security_id,
        "symbol": symbol,
        "session": pd.Timestamp(session).date().isoformat(),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "currency": "KRW",
        "source": source,
        "source_url": source_url,
        "retrieved_at": retrieved_at,
        "source_hash": source_hash,
        "observation_status": observation_status,
        "exchange": exchange,
        "asset_type": asset_type,
        "security_name": security_name,
        "official_reference_price": official_reference_price,
        "official_fluctuation_rate": official_fluctuation_rate,
    }


def _price_columns() -> list[str]:
    return [
        "security_id",
        "symbol",
        "session",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "currency",
        "source",
        "source_url",
        "retrieved_at",
        "source_hash",
        "observation_status",
        "exchange",
        "asset_type",
        "security_name",
        "official_reference_price",
        "official_fluctuation_rate",
    ]


def _empty_price_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_price_columns())


def _batched(values: tuple[str, ...], size: int):
    for start in range(0, len(values), max(1, size)):
        yield values[start : start + max(1, size)]


__all__ = [
    "KR_BENCHMARK_SECURITIES",
    "KR_INDEX_DEFINITIONS",
    "KrOfficialDataUnavailable",
    "KrIdentityCatalog",
    "KrProviderResult",
    "attach_dart_codes",
    "fetch_eodhd_prices",
    "fetch_eodhd_actions",
    "fetch_kis_prices",
    "fetch_kr_identity_catalog",
    "fetch_krx_membership",
    "fetch_krx_session_prices",
    "fetch_naver_prices",
    "fetch_yahoo_prices",
    "validate_krx_official_configuration",
]
