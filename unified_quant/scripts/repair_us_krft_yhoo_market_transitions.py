#!/usr/bin/env python3
"""Plan or apply exact KRFT->KHC and YHOO->AABA transition repairs offline.

The two reviewed filings distinguish legal completion/name-change dates from
the first session on which the successor security traded.  This planner moves
the lifecycle action to that market session, repairs the linked identity and
index references, and prepares release-exact adjustment-factor lineage without
changing factor economics.

Plan is the default and is strictly read-only.  ``--apply`` has no network,
EODHD, or R2 code path: it rechecks every pinned input under the repository-wide
writer lock, writes immutable versions with pointer CAS, and advances the
release only after all datasets succeed.  A transaction journal restores the
old release and pointers on any failure.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import html
import json
import math
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from supertrend_quant.market_store.adjustments import (
    CASH_DISTRIBUTION_ACTIONS,
    RATIO_ACTIONS,
)
from supertrend_quant.market_store.cross_validation import (
    TRUSTED_REVIEWED_TERMINAL_MARKET_DATE_CORRECTION_EVENT_IDS,
    TRUSTED_REVIEWED_TERMINAL_MARKET_DATE_CORRECTIONS_SHA256,
)
from supertrend_quant.market_store.lifecycle import canonical_lifecycle_event_id
from supertrend_quant.market_store.lifecycle_coverage import lifecycle_candidate_id
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.validation import (
    validate_dataset,
    validate_repository_snapshot,
)


DEFAULT_CACHE_ROOT = Path("data/cache")
OPERATION = "repair_us_krft_yhoo_market_transitions"
POLICY = "us_market_transition_dates/v1"
REVIEWED_AT = "2026-07-18T14:30:00Z"
TRANSACTION_DIR = "transactions/us-krft-yhoo-market-transitions"
RECOVERY_DIR = "recovery/us-krft-yhoo-market-transitions"

WRITE_DATASETS = (
    "corporate_actions",
    "lifecycle_resolutions",
    "security_master",
    "symbol_history",
    "index_constituent_anchors",
    "index_membership_events",
    "source_archive",
    "adjustment_factors",
)
REQUIRED_DATASETS = (*WRITE_DATASETS, "daily_price_raw")

KRFT_ID = "US:EODHD:e3b28684-f48e-582a-8836-b2f5579f2dd2"
KHC_ID = "US:EODHD:a060087d-109e-58e9-91f9-b4c398a3d415"
KRFT_SYMBOL = "KRFT"
KHC_SYMBOL = "KHC"
KRFT_LAST_SESSION = "2015-07-02"
KRFT_LEGAL_COMPLETION = "2015-07-02"
KHC_FIRST_SESSION = "2015-07-06"
KRFT_OLD_EVENT_ID = (
    "c3209167d547d7e8379cb316ec4910ea62d1c1d28679f137a80364fe876e9b7c"
)
KRFT_NEW_EVENT_ID = (
    "39caca00fbedb7b23e8b5294cdf3bdc5aba9014f98bf72b6a6da341602651192"
)
KRFT_CANDIDATE_ID = (
    "8afeca7c8f24b790e6b2f234254597ae076e1d6e3d35dcd0f41299e56642e9f6"
)
KRFT_SPECIAL_DIVIDEND_EVENT_ID = (
    "ec8bdabd737c2e25ebeb1ca9c296702277ff382b3229d76059143a81b6cd0b1f"
)
KRFT_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/1637459/"
    "000119312515244356/0001193125-15-244356.txt"
)
KRFT_SOURCE_HASH = (
    "ba6d15da235a1f5f6a186fb7b9e52b7b438c25cf76d4f28d9fac81953b7fc299"
)
KRFT_SOURCE_BYTES = 1_440_758
KRFT_ACTION_RETRIEVED_AT = "2026-07-18T10:30:21.019780Z"
KRFT_HISTORY_RETRIEVED_AT = "2026-07-18T08:11:54.578547Z"

YHOO_ID = "US:EODHD:1dfb343a-9d83-5f7b-b683-bdf884111979"
AABA_ID = "US:EODHD:9b1bbdaa-839c-5d59-8bda-99b2087022e6"
YHOO_SYMBOL = "YHOO"
AABA_SYMBOL = "AABA"
YHOO_LAST_SESSION = "2017-06-16"
YHOO_BUSINESS_SALE_COMPLETION = "2017-06-13"
YHOO_LEGAL_NAME_CHANGE = "2017-06-16"
AABA_FIRST_SESSION = "2017-06-19"
AABA_LAST_SESSION = "2019-10-02"
YHOO_WRONG_PARSED_DATE = "2017-06-13"
YHOO_OLD_EVENT_ID = (
    "b31525699f142a6ed8995b71d86b8480e19e54446ccb0694960296f753ad2be6"
)
YHOO_NEW_EVENT_ID = (
    "ea791beeaa569aaf19b1df04e59df3710d45382cd70e5d82078d7247286272c6"
)
YHOO_CANDIDATE_ID = (
    "85ae0eb05557268955665853205bf4e684f1cc7317d4d6921e739aacebcfdfd2"
)
YHOO_OLD_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/1011006/"
    "000119312517241304/0001193125-17-241304.txt"
)
YHOO_OLD_SOURCE_HASH = (
    "58dff3a89b875bcc45982e5df928c3f981eb02cd42809880a69be3da531b6726"
)
YHOO_OLD_RETRIEVED_AT = "2026-07-18T10:30:27.561287Z"
YHOO_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/1011006/"
    "000119312517206955/0001193125-17-206955.txt"
)
YHOO_SOURCE_HASH = (
    "274d98161c66f9480a4825eae722974405fffe2e3a0b73091d22f1a9acb07ca9"
)
YHOO_SOURCE_BYTES = 17_192
YHOO_RETRIEVED_AT = "2026-07-18T10:30:27.515561Z"
YHOO_STATE_CACHE_PATH = (
    "state/sec_lifecycle/"
    "d2c5a85569c781ea06c9dc6e4dc405a46fd314e29e316a9bdf6c9e0813a5a81f.bin"
)

ACTION_SOURCE = "sec_edgar+stored_price_crosscheck"
ACTION_SOURCE_KIND = "official_crosscheck"
RESOLUTION_SOURCE = "lifecycle_finalizer"
RESOLUTION_REVIEWED_BY = "us_lifecycle_finalizer_v1"
RESOLUTION_REVIEWED_AT = "2026-07-18T00:00:00Z"
ARCHIVE_DATASET = "sec_edgar_filing"
ARCHIVE_CONTENT_TYPE = "text/plain"
REPAIRED_IDENTITY_SOURCE = "official_market_transition_repair"
CONFIRMED_IDENTITY_SOURCE = "official_confirmed_identity_history_repair"

EOD_SYMBOL_SOURCE = "eodhd_exchange_symbols"
EOD_SYMBOL_URL = "https://eodhd.com/api/exchange-symbol-list/US?delisted=0"
EOD_SYMBOL_RETRIEVED_AT = "2026-07-16T15:56:01.033938Z"
EOD_SYMBOL_HASH = (
    "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99"
)

SP_SOURCE = "community_sp500_history"
SP_SOURCE_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20%28Updated%29.csv"
)
SP_SOURCE_HASH = (
    "39a9202c9ef69a74c0ff07e2113ad41fb6da7c8c5b6cd9541f0185fb4391e717"
)
SP_ANCHOR_RETRIEVED_AT = "2026-07-16T15:56:14.437763Z"
SP_EVENT_RETRIEVED_AT = "2026-07-16T15:56:14.474469Z"

NASDAQ_SOURCE = "community_nasdaq100_history"
NASDAQ_SOURCE_URL = ",".join(
    "https://raw.githubusercontent.com/jmccarrell/n100tickers/main/src/"
    f"nasdaq_100_ticker_history/n100-ticker-changes-{year}.yaml"
    for year in range(2015, 2027)
)
NASDAQ_SOURCE_HASH = (
    "83465af4e2f80f45ea239068ee41ba2069db990720896380c6ef8df4c1c9cb97"
)
NASDAQ_ANCHOR_RETRIEVED_AT = "2026-07-16T15:56:14.546419Z"
NASDAQ_EVENT_RETRIEVED_AT = "2026-07-16T15:56:14.590300Z"

KRFT_NASDAQ_ADD_OLD_ID = (
    "3760a91253550cf576b7706fbb6945fdd7aad35895cabfd8096ac5aedf67e4bd"
)
KRFT_NASDAQ_ADD_NEW_ID = (
    "0633b696b08e09562f6823bda353ed0300f4c4f9e6bb0f3b1c1371f1c216984d"
)
KRFT_NASDAQ_REMOVE_OLD_ID = (
    "e1240d38736a8793a99093c52f831b30cbcaed18cb23dffaa4dfd116f4a7d63d"
)
KRFT_NASDAQ_REMOVE_NEW_ID = (
    "b4bb017f0fbfbc175111eb0665896475cf1af096290333f62740ffaa27d6fb98"
)
KRFT_SP_ADD_ID = (
    "110429be91b814d0ab28c24649743302d8656e29ee8aa937fcb76f2a6ae3ab67"
)
KRFT_SP_REMOVE_ID = (
    "7c417d0021393bb309ae5b64214fb9b781145e3a998e0ee5397cc19d2c145632"
)
YHOO_NASDAQ_REMOVE_ID = (
    "2df2ecb5494992e34a95713d1ed3779b7feda7fa29c7a7911265bf3dc8bc4c1f"
)
YHOO_SP_REMOVE_OLD_ID = (
    "8ef4e17959a2163350801f180a2ef9dca709767b44015f057a637a57c88b3563"
)
YHOO_SP_REMOVE_NEW_ID = (
    "3b6dec593e3024431dbda4f9b3b02155d507775cc09cb672316fe22a941e29e7"
)

SPECIAL_DIVIDEND_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/1637459/"
    "000163745915000021/khc10q62815.htm"
)
SPECIAL_DIVIDEND_SOURCE_HASH = (
    "f138d8464a92839720f8a4441a55b5fc96852a89c917f2b3965d1121075f7875"
)
SPECIAL_DIVIDEND_RETRIEVED_AT = "2026-07-18T09:34:38.650933Z"
SPECIAL_DIVIDEND_DECLARATION_HASH = (
    "c9d78b9704c3b2b95c018dca5d7e7123a8a1e5d7bae8e1b30152b3037fc26849"
)
SPECIAL_DIVIDEND_DECLARATION_URL = (
    "https://www.sec.gov/Archives/edgar/data/1545158/"
    "000119312515230632/d947291d425.htm"
)

EXPECTED_PRICE_BOUNDARIES = {
    KRFT_ID: (126, "2015-01-02", KRFT_LAST_SESSION, 88.0, 88.19),
    KHC_ID: (2_773, KHC_FIRST_SESSION, "2026-07-15", 71.0, 72.96),
    YHOO_ID: (619, "2015-01-02", YHOO_LAST_SESSION, 52.79, 52.58),
    AABA_ID: (1_196, "2015-01-02", AABA_LAST_SESSION, 19.56, 19.63),
}
AABA_FIRST_BAR = (54.0, 54.46)


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: Mapping[str, str | None]
    planned_versions: Mapping[str, str]
    frames: Mapping[str, pd.DataFrame]
    evidence_payloads: Mapping[str, bytes]
    summary: Mapping[str, Any]


FailureInjector = Callable[[str], None]


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _date(value: Any) -> str:
    value = _text(value)
    if not value:
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _number(value: Any) -> float | None:
    if not _text(value):
        return None
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else float(parsed)


def _bool(value: Any, expected: bool) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value) is expected


def _same_cell(left: Any, right: Any) -> bool:
    try:
        if bool(pd.isna(left)) and bool(pd.isna(right)):
            return True
    except (TypeError, ValueError):
        pass
    return left == right


def _changed_cells(before: pd.DataFrame, after: pd.DataFrame) -> set[tuple[Any, str]]:
    if not before.index.equals(after.index) or list(before.columns) != list(after.columns):
        raise AssertionError("Repair changed frame shape/index while rewriting exact rows.")
    return {
        (index, column)
        for index in before.index
        for column in before.columns
        if not _same_cell(before.at[index, column], after.at[index, column])
    }


def _one_row(frame: pd.DataFrame, mask: pd.Series, label: str) -> tuple[Any, pd.Series]:
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one {label}; observed {len(rows)}.")
    return rows.index[0], rows.iloc[0]


def _canonical_metadata(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("Lifecycle action metadata is not valid JSON.") from exc
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _membership_event_id(
    source: str,
    index_id: str,
    effective_date: str,
    operation: str,
    security_id: str,
) -> str:
    key = "|".join((source, index_id, effective_date, operation.upper(), security_id))
    return hashlib.sha256(key.encode()).hexdigest()


def _static_contract() -> None:
    if {"stock_merger", "ticker_change"} & (
        set(RATIO_ACTIONS) | set(CASH_DISTRIBUTION_ACTIONS)
    ):
        raise RuntimeError(
            "Adjustment policy unexpectedly treats lifecycle transitions as economic factors."
        )
    if not {KRFT_NEW_EVENT_ID, YHOO_NEW_EVENT_ID}.issubset(
        TRUSTED_REVIEWED_TERMINAL_MARKET_DATE_CORRECTION_EVENT_IDS
    ):
        raise RuntimeError(
            "KRFT/YHOO transitions are not code-pinned in cross-validation."
        )
    checks = (
        (
            canonical_lifecycle_event_id(KRFT_ID, "stock_merger", KRFT_LEGAL_COMPLETION),
            KRFT_OLD_EVENT_ID,
            "KRFT old action",
        ),
        (
            canonical_lifecycle_event_id(KRFT_ID, "stock_merger", KHC_FIRST_SESSION),
            KRFT_NEW_EVENT_ID,
            "KRFT repaired action",
        ),
        (
            lifecycle_candidate_id(KRFT_ID, KRFT_LAST_SESSION),
            KRFT_CANDIDATE_ID,
            "KRFT candidate",
        ),
        (
            canonical_lifecycle_event_id(YHOO_ID, "ticker_change", YHOO_WRONG_PARSED_DATE),
            YHOO_OLD_EVENT_ID,
            "YHOO old action",
        ),
        (
            canonical_lifecycle_event_id(YHOO_ID, "ticker_change", AABA_FIRST_SESSION),
            YHOO_NEW_EVENT_ID,
            "YHOO repaired action",
        ),
        (
            lifecycle_candidate_id(YHOO_ID, YHOO_LAST_SESSION),
            YHOO_CANDIDATE_ID,
            "YHOO candidate",
        ),
        (
            canonical_lifecycle_event_id(
                KRFT_ID, "special_dividend", KRFT_LEGAL_COMPLETION
            ),
            KRFT_SPECIAL_DIVIDEND_EVENT_ID,
            "KRFT special dividend",
        ),
        (
            _membership_event_id(
                NASDAQ_SOURCE, "nasdaq100", KRFT_LEGAL_COMPLETION, "ADD", KHC_ID
            ),
            KRFT_NASDAQ_ADD_OLD_ID,
            "KRFT Nasdaq old ADD",
        ),
        (
            _membership_event_id(
                NASDAQ_SOURCE, "nasdaq100", KHC_FIRST_SESSION, "ADD", KHC_ID
            ),
            KRFT_NASDAQ_ADD_NEW_ID,
            "KRFT Nasdaq repaired ADD",
        ),
        (
            _membership_event_id(
                NASDAQ_SOURCE,
                "nasdaq100",
                KRFT_LEGAL_COMPLETION,
                "REMOVE",
                KRFT_ID,
            ),
            KRFT_NASDAQ_REMOVE_OLD_ID,
            "KRFT Nasdaq old REMOVE",
        ),
        (
            _membership_event_id(
                NASDAQ_SOURCE, "nasdaq100", KHC_FIRST_SESSION, "REMOVE", KRFT_ID
            ),
            KRFT_NASDAQ_REMOVE_NEW_ID,
            "KRFT Nasdaq repaired REMOVE",
        ),
        (
            _membership_event_id(
                SP_SOURCE, "sp500", AABA_FIRST_SESSION, "REMOVE", AABA_ID
            ),
            YHOO_SP_REMOVE_OLD_ID,
            "YHOO S&P old REMOVE",
        ),
        (
            _membership_event_id(
                SP_SOURCE, "sp500", AABA_FIRST_SESSION, "REMOVE", YHOO_ID
            ),
            YHOO_SP_REMOVE_NEW_ID,
            "YHOO S&P repaired REMOVE",
        ),
    )
    for observed, expected, label in checks:
        if observed != expected:
            raise RuntimeError(f"Pinned {label} identifier is not canonical.")


def _safe_path(root: Path, relative: str) -> Path:
    base = root.resolve()
    target = (base / relative).resolve()
    if target == base or base not in target.parents:
        raise ValueError(f"Evidence path escapes repository: {relative}.")
    return target


def _archive_path(completed_session: str, source_hash: str) -> str:
    return f"archives/{completed_session}/{source_hash}.txt.gz"


def _normalized_document(payload: bytes) -> str:
    decoded = payload.decode("utf-8", errors="replace")
    decoded = re.sub(r"<[^>]+>", " ", decoded)
    return re.sub(r"\s+", " ", html.unescape(decoded)).strip()


def _verify_payload(
    payload: bytes,
    *,
    source_hash: str,
    expected_bytes: int,
    patterns: tuple[str, ...],
    label: str,
) -> None:
    observed = hashlib.sha256(payload).hexdigest()
    if observed != source_hash or len(payload) != expected_bytes:
        raise ValueError(
            f"{label} evidence hash/size changed: hash={observed}; bytes={len(payload)}."
        )
    text = _normalized_document(payload)
    missing = [pattern for pattern in patterns if not re.search(pattern, text, re.I)]
    if missing:
        raise ValueError(f"{label} evidence no longer proves: " + ", ".join(missing))


def _archive_row_expected(
    *,
    completed_session: str,
    source_url: str,
    source_hash: str,
    retrieved_at: str,
) -> dict[str, Any]:
    return {
        "archive_id": source_hash,
        "dataset": ARCHIVE_DATASET,
        "object_path": _archive_path(completed_session, source_hash),
        "content_type": ARCHIVE_CONTENT_TYPE,
        "effective_date": completed_session,
        "source": ARCHIVE_DATASET,
        "retrieved_at": retrieved_at,
        "source_hash": source_hash,
        "source_url": source_url,
    }


def _archive_row_is_exact(row: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(
        (_date(row.get(key)) if key == "effective_date" else _text(row.get(key)))
        == value
        for key, value in expected.items()
    )


def _read_gzip(path: Path, label: str) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"{label} evidence payload is missing: {path}.")
    try:
        return gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise ValueError(f"{label} evidence payload is invalid gzip.") from exc


def _verify_krft_evidence(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    completed_session: str,
) -> bytes:
    expected = _archive_row_expected(
        completed_session=completed_session,
        source_url=KRFT_SOURCE_URL,
        source_hash=KRFT_SOURCE_HASH,
        retrieved_at=KRFT_ACTION_RETRIEVED_AT,
    )
    related = archive["archive_id"].astype(str).eq(KRFT_SOURCE_HASH)
    _, row = _one_row(archive, related, "KRFT official source_archive row")
    if not _archive_row_is_exact(row, expected):
        raise ValueError("KRFT official source_archive row changed.")
    payload = _read_gzip(
        _safe_path(repository.root, str(expected["object_path"])), "KRFT"
    )
    _verify_payload(
        payload,
        source_hash=KRFT_SOURCE_HASH,
        expected_bytes=KRFT_SOURCE_BYTES,
        patterns=(
            r"On July 2, 2015.{0,180}?became a wholly owned subsidiary",
            r"Kraft Common Stock.{0,180}?ceased trading on.{0,120}?delisted from.{0,80}?NASDAQ",
            r"KHC.{0,100}?begin trading on July 6, 2015",
            r"special cash dividend in the amount of \$16\.50 per share",
        ),
        label="KRFT",
    )
    return payload


def _verify_yhoo_payload(payload: bytes) -> None:
    _verify_payload(
        payload,
        source_hash=YHOO_SOURCE_HASH,
        expected_bytes=YHOO_SOURCE_BYTES,
        patterns=(
            r"On June 13, 2017.{0,100}?completed the sale of its operating business",
            r"On June 16, 2017.{0,80}?changed its name to.{0,40}?Altaba Inc",
            r"AABA.{0,100}?as of the open of trading on June 19, 2017",
            r"Previously, through June 16, 2017.{0,120}?YHOO",
            r"No action is required to be taken by stockholders",
        ),
        label="YHOO",
    )


def _load_yhoo_evidence(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    completed_session: str,
) -> tuple[bytes, bool]:
    expected = _archive_row_expected(
        completed_session=completed_session,
        source_url=YHOO_SOURCE_URL,
        source_hash=YHOO_SOURCE_HASH,
        retrieved_at=YHOO_RETRIEVED_AT,
    )
    related = archive["archive_id"].astype(str).eq(YHOO_SOURCE_HASH)
    if related.any():
        _, row = _one_row(archive, related, "YHOO repaired source_archive row")
        if not _archive_row_is_exact(row, expected):
            raise ValueError("YHOO repaired source_archive row is not exact.")
        payload = _read_gzip(
            _safe_path(repository.root, str(expected["object_path"])), "YHOO"
        )
        archived = True
    else:
        path = _safe_path(repository.root, YHOO_STATE_CACHE_PATH)
        if not path.is_file():
            raise FileNotFoundError(f"Pinned YHOO SEC cache payload is missing: {path}.")
        payload = path.read_bytes()
        archived = False
    _verify_yhoo_payload(payload)
    return payload, archived


def _transition_metadata(kind: str) -> str:
    if kind == "krft":
        value = {
            "policy": POLICY,
            "legal_completion_date": KRFT_LEGAL_COMPLETION,
            "source_last_trade_date": KRFT_LAST_SESSION,
            "market_effective_date": KHC_FIRST_SESSION,
            "market_date_policy": "first_successor_trading_session",
            "evidence": [
                {"source_url": KRFT_SOURCE_URL, "source_hash": KRFT_SOURCE_HASH}
            ],
        }
    elif kind == "yhoo":
        value = {
            "policy": POLICY,
            "operating_business_sale_completion_date": YHOO_BUSINESS_SALE_COMPLETION,
            "legal_name_change_date": YHOO_LEGAL_NAME_CHANGE,
            "source_last_trade_date": YHOO_LAST_SESSION,
            "market_effective_date": AABA_FIRST_SESSION,
            "market_date_policy": "first_successor_trading_session",
            "holder_action_required": False,
            "evidence": [
                {"source_url": YHOO_SOURCE_URL, "source_hash": YHOO_SOURCE_HASH}
            ],
        }
    else:
        raise ValueError(f"Unknown transition metadata kind: {kind}.")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _special_dividend_metadata() -> str:
    value = {
        "economic_sequence": [
            "KRFT special-dividend entitlement and payment",
            "1:1 KRFT-to-KHC stock merger",
        ],
        "evidence": [
            {
                "label": "kraft_special_dividend_declaration",
                "source_hash": SPECIAL_DIVIDEND_DECLARATION_HASH,
                "source_url": SPECIAL_DIVIDEND_DECLARATION_URL,
            },
            {
                "label": "kraft_special_dividend_completion_payment",
                "source_hash": SPECIAL_DIVIDEND_SOURCE_HASH,
                "source_url": SPECIAL_DIVIDEND_SOURCE_URL,
            },
            {
                "label": "kraft_heinz_merger_completion",
                "source_hash": KRFT_SOURCE_HASH,
                "source_url": KRFT_SOURCE_URL,
            },
        ],
        "payment_basis": (
            "KHC 2015 Form 10-Q states record holders received the cash "
            "dividend upon completion of the merger"
        ),
        "policy": "kraft_2015_special_dividend/v1",
        "record_time": "immediately_prior_to_2015_merger_effective_time",
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _special_dividend_is_exact(row: Mapping[str, Any]) -> bool:
    return bool(
        _text(row.get("event_id")) == KRFT_SPECIAL_DIVIDEND_EVENT_ID
        and _text(row.get("security_id")) == KRFT_ID
        and _text(row.get("action_type")) == "special_dividend"
        and _date(row.get("effective_date")) == KRFT_LEGAL_COMPLETION
        and _date(row.get("ex_date")) == KRFT_LEGAL_COMPLETION
        and _date(row.get("announcement_date")) == "2015-06-22"
        and _date(row.get("record_date")) == KRFT_LEGAL_COMPLETION
        and _date(row.get("payment_date")) == KRFT_LEGAL_COMPLETION
        and _number(row.get("cash_amount")) == 16.5
        and _number(row.get("ratio")) is None
        and _text(row.get("currency")) == "USD"
        and _text(row.get("new_security_id")) == ""
        and _text(row.get("new_symbol")) == ""
        and _bool(row.get("official"), True)
        and _text(row.get("source_url")) == SPECIAL_DIVIDEND_SOURCE_URL
        and _text(row.get("source_kind")) == ACTION_SOURCE_KIND
        and _text(row.get("source")) == "sec_edgar+reviewed_special_dividend"
        and _text(row.get("retrieved_at")) == SPECIAL_DIVIDEND_RETRIEVED_AT
        and _text(row.get("source_hash")) == SPECIAL_DIVIDEND_SOURCE_HASH
        and _canonical_metadata(row.get("metadata")) == _special_dividend_metadata()
    )


def _action_state(row: Mapping[str, Any], kind: str, repaired: bool) -> bool:
    if kind == "krft":
        common = bool(
            _text(row.get("security_id")) == KRFT_ID
            and _text(row.get("action_type")) == "stock_merger"
            and _date(row.get("announcement_date")) == KRFT_LEGAL_COMPLETION
            and _date(row.get("record_date")) == ""
            and _date(row.get("payment_date")) == ""
            and _number(row.get("cash_amount")) is None
            and _number(row.get("ratio")) == 1.0
            and _text(row.get("currency")) == "USD"
            and _text(row.get("new_security_id")) == KHC_ID
            and _text(row.get("new_symbol")) == KHC_SYMBOL
            and _bool(row.get("official"), True)
            and _text(row.get("source_url")) == KRFT_SOURCE_URL
            and _text(row.get("source_kind")) == ACTION_SOURCE_KIND
            and _text(row.get("source")) == ACTION_SOURCE
            and _text(row.get("retrieved_at")) == KRFT_ACTION_RETRIEVED_AT
            and _text(row.get("source_hash")) == KRFT_SOURCE_HASH
        )
        if repaired:
            return bool(
                common
                and _text(row.get("event_id")) == KRFT_NEW_EVENT_ID
                and _date(row.get("effective_date")) == KHC_FIRST_SESSION
                and _date(row.get("ex_date")) == KHC_FIRST_SESSION
                and _canonical_metadata(row.get("metadata"))
                == _transition_metadata("krft")
            )
        return bool(
            common
            and _text(row.get("event_id")) == KRFT_OLD_EVENT_ID
            and _date(row.get("effective_date")) == KRFT_LEGAL_COMPLETION
            and _date(row.get("ex_date")) == KRFT_LEGAL_COMPLETION
            and _text(row.get("metadata")) == ""
        )
    if kind == "yhoo":
        common = bool(
            _text(row.get("security_id")) == YHOO_ID
            and _text(row.get("action_type")) == "ticker_change"
            and _date(row.get("record_date")) == ""
            and _date(row.get("payment_date")) == ""
            and _number(row.get("cash_amount")) is None
            and _number(row.get("ratio")) is None
            and _text(row.get("currency")) == "USD"
            and _text(row.get("new_security_id")) == AABA_ID
            and _text(row.get("new_symbol")) == AABA_SYMBOL
            and _bool(row.get("official"), True)
            and _text(row.get("source_kind")) == ACTION_SOURCE_KIND
            and _text(row.get("source")) == ACTION_SOURCE
        )
        if repaired:
            return bool(
                common
                and _text(row.get("event_id")) == YHOO_NEW_EVENT_ID
                and _date(row.get("effective_date")) == AABA_FIRST_SESSION
                and _date(row.get("ex_date")) == AABA_FIRST_SESSION
                and _date(row.get("announcement_date")) == AABA_FIRST_SESSION
                and _text(row.get("source_url")) == YHOO_SOURCE_URL
                and _text(row.get("retrieved_at")) == YHOO_RETRIEVED_AT
                and _text(row.get("source_hash")) == YHOO_SOURCE_HASH
                and _canonical_metadata(row.get("metadata"))
                == _transition_metadata("yhoo")
            )
        return bool(
            common
            and _text(row.get("event_id")) == YHOO_OLD_EVENT_ID
            and _date(row.get("effective_date")) == YHOO_WRONG_PARSED_DATE
            and _date(row.get("ex_date")) == YHOO_WRONG_PARSED_DATE
            and _date(row.get("announcement_date")) == "2017-07-31"
            and _text(row.get("source_url")) == YHOO_OLD_SOURCE_URL
            and _text(row.get("retrieved_at")) == YHOO_OLD_RETRIEVED_AT
            and _text(row.get("source_hash")) == YHOO_OLD_SOURCE_HASH
            and _text(row.get("metadata")) == ""
        )
    raise ValueError(f"Unknown action kind: {kind}.")


def _rewrite_actions(actions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    special_index, special = _one_row(
        actions,
        actions["event_id"].astype(str).eq(KRFT_SPECIAL_DIVIDEND_EVENT_ID),
        "KRFT special-dividend action",
    )
    if not _special_dividend_is_exact(special):
        raise ValueError("KRFT special-dividend action changed before transition repair.")
    special_before = actions.loc[special_index].copy(deep=True)
    output = actions.copy(deep=True)
    changed: dict[str, bool] = {}
    permitted: set[tuple[Any, str]] = set()
    for kind, security_id, action_type, old_id, new_id, updates in (
        (
            "krft",
            KRFT_ID,
            "stock_merger",
            KRFT_OLD_EVENT_ID,
            KRFT_NEW_EVENT_ID,
            {
                "event_id": KRFT_NEW_EVENT_ID,
                "effective_date": KHC_FIRST_SESSION,
                "ex_date": KHC_FIRST_SESSION,
                "metadata": _transition_metadata("krft"),
            },
        ),
        (
            "yhoo",
            YHOO_ID,
            "ticker_change",
            YHOO_OLD_EVENT_ID,
            YHOO_NEW_EVENT_ID,
            {
                "event_id": YHOO_NEW_EVENT_ID,
                "effective_date": AABA_FIRST_SESSION,
                "ex_date": AABA_FIRST_SESSION,
                "announcement_date": AABA_FIRST_SESSION,
                "source_url": YHOO_SOURCE_URL,
                "retrieved_at": YHOO_RETRIEVED_AT,
                "source_hash": YHOO_SOURCE_HASH,
                "metadata": _transition_metadata("yhoo"),
            },
        ),
    ):
        mask = (
            output["security_id"].astype(str).eq(security_id)
            & output["action_type"].astype(str).eq(action_type)
            & output["event_id"].astype(str).isin({old_id, new_id})
        )
        index, row = _one_row(output, mask, f"{kind.upper()} transition action")
        old = _action_state(row, kind, False)
        repaired = _action_state(row, kind, True)
        if old == repaired:
            raise ValueError(
                f"{kind.upper()} action is neither exact old nor exact repaired state."
            )
        changed[kind] = old
        if old:
            for column, value in updates.items():
                output.at[index, column] = value
                permitted.add((index, column))
    observed = _changed_cells(actions, output)
    if observed != permitted:
        raise AssertionError(f"Action repair changed unexpected cells: {observed}.")
    for column in actions.columns:
        if not _same_cell(special_before[column], output.at[special_index, column]):
            raise AssertionError(f"KRFT special dividend changed in column {column}.")
    if not _special_dividend_is_exact(output.loc[special_index]):
        raise AssertionError("KRFT special dividend is not exact after action repair.")
    return output, changed


def _resolution_state(row: Mapping[str, Any], kind: str, repaired: bool) -> bool:
    if kind == "krft":
        return bool(
            _text(row.get("candidate_id")) == KRFT_CANDIDATE_ID
            and _text(row.get("security_id")) == KRFT_ID
            and _text(row.get("symbol")) == KRFT_SYMBOL
            and _date(row.get("last_price_date")) == KRFT_LAST_SESSION
            and _text(row.get("resolution")) == "applied"
            and _text(row.get("event_id"))
            == (KRFT_NEW_EVENT_ID if repaired else KRFT_OLD_EVENT_ID)
            and _text(row.get("exception_code")) == ""
            and _text(row.get("exception_reason")) == ""
            and _text(row.get("reviewed_by")) == RESOLUTION_REVIEWED_BY
            and _text(row.get("reviewed_at")) == RESOLUTION_REVIEWED_AT
            and _text(row.get("recheck_after")) == ""
            and _text(row.get("successor_security_id")) == KHC_ID
            and _text(row.get("successor_symbol")) == KHC_SYMBOL
            and _text(row.get("source_url")) == KRFT_SOURCE_URL
            and _text(row.get("source")) == RESOLUTION_SOURCE
            and _text(row.get("retrieved_at")) == KRFT_ACTION_RETRIEVED_AT
            and _text(row.get("source_hash")) == KRFT_SOURCE_HASH
        )
    if kind == "yhoo":
        expected_url = YHOO_SOURCE_URL if repaired else YHOO_OLD_SOURCE_URL
        expected_hash = YHOO_SOURCE_HASH if repaired else YHOO_OLD_SOURCE_HASH
        expected_retrieved = YHOO_RETRIEVED_AT if repaired else YHOO_OLD_RETRIEVED_AT
        return bool(
            _text(row.get("candidate_id")) == YHOO_CANDIDATE_ID
            and _text(row.get("security_id")) == YHOO_ID
            and _text(row.get("symbol")) == YHOO_SYMBOL
            and _date(row.get("last_price_date")) == YHOO_LAST_SESSION
            and _text(row.get("resolution")) == "applied"
            and _text(row.get("event_id"))
            == (YHOO_NEW_EVENT_ID if repaired else YHOO_OLD_EVENT_ID)
            and _text(row.get("exception_code")) == ""
            and _text(row.get("exception_reason")) == ""
            and _text(row.get("reviewed_by")) == RESOLUTION_REVIEWED_BY
            and _text(row.get("reviewed_at")) == RESOLUTION_REVIEWED_AT
            and _text(row.get("recheck_after")) == ""
            and _text(row.get("successor_security_id")) == AABA_ID
            and _text(row.get("successor_symbol")) == AABA_SYMBOL
            and _text(row.get("source_url")) == expected_url
            and _text(row.get("source")) == RESOLUTION_SOURCE
            and _text(row.get("retrieved_at")) == expected_retrieved
            and _text(row.get("source_hash")) == expected_hash
        )
    raise ValueError(f"Unknown resolution kind: {kind}.")


def _rewrite_resolutions(
    resolutions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    output = resolutions.copy(deep=True)
    changed: dict[str, bool] = {}
    permitted: set[tuple[Any, str]] = set()
    for kind, candidate_id, updates in (
        ("krft", KRFT_CANDIDATE_ID, {"event_id": KRFT_NEW_EVENT_ID}),
        (
            "yhoo",
            YHOO_CANDIDATE_ID,
            {
                "event_id": YHOO_NEW_EVENT_ID,
                "source_url": YHOO_SOURCE_URL,
                "retrieved_at": YHOO_RETRIEVED_AT,
                "source_hash": YHOO_SOURCE_HASH,
            },
        ),
    ):
        index, row = _one_row(
            output,
            output["candidate_id"].astype(str).eq(candidate_id),
            f"{kind.upper()} lifecycle resolution",
        )
        old = _resolution_state(row, kind, False)
        repaired = _resolution_state(row, kind, True)
        if old == repaired:
            raise ValueError(
                f"{kind.upper()} resolution is neither exact old nor repaired state."
            )
        changed[kind] = old
        if old:
            for column, value in updates.items():
                output.at[index, column] = value
                permitted.add((index, column))
    observed = _changed_cells(resolutions, output)
    if observed != permitted:
        raise AssertionError(f"Resolution repair changed unexpected cells: {observed}.")
    return output, changed


def _master_core(row: Mapping[str, Any], kind: str) -> bool:
    expected = {
        "krft": (KRFT_ID, KRFT_SYMBOL, "Kraft Foods Group Inc", "KRFT.US"),
        "khc": (KHC_ID, KHC_SYMBOL, "Kraft Heinz Co", "KHC.US"),
        "yhoo": (YHOO_ID, YHOO_SYMBOL, "Yahoo! Inc", "YHOO.US"),
        "aaba": (AABA_ID, AABA_SYMBOL, "Altaba Inc", "AABA.US"),
    }[kind]
    security_id, symbol, name, provider_symbol = expected
    return bool(
        _text(row.get("security_id")) == security_id
        and _text(row.get("primary_symbol")) == symbol
        and _text(row.get("name")) == name
        and _text(row.get("exchange")) == "NASDAQ"
        and _text(row.get("asset_type")) == "STOCK"
        and _text(row.get("currency")) == "USD"
        and _text(row.get("country")) == "US"
        and _text(row.get("provider_symbol")) == provider_symbol
        and _text(row.get("action_provider_symbol")) == provider_symbol
    )


def _aaba_master_state(row: Mapping[str, Any], repaired: bool) -> bool:
    if not _master_core(row, "aaba") or _date(row.get("active_to")) != AABA_LAST_SESSION:
        return False
    if repaired:
        return bool(
            _date(row.get("active_from")) == AABA_FIRST_SESSION
            and _text(row.get("source")) == REPAIRED_IDENTITY_SOURCE
            and _text(row.get("source_url")) == YHOO_SOURCE_URL
            and _text(row.get("retrieved_at")) == YHOO_RETRIEVED_AT
            and _text(row.get("source_hash")) == YHOO_SOURCE_HASH
        )
    return bool(
        _date(row.get("active_from")) == "2015-01-02"
        and _text(row.get("source")) == EOD_SYMBOL_SOURCE
        and _text(row.get("source_url")) == EOD_SYMBOL_URL
        and _text(row.get("retrieved_at")) == EOD_SYMBOL_RETRIEVED_AT
        and _text(row.get("source_hash")) == EOD_SYMBOL_HASH
    )


def _rewrite_master(master: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    output = master.copy(deep=True)
    for kind, security_id, active_from, active_to, source, url, retrieved, digest in (
        (
            "krft",
            KRFT_ID,
            "2015-01-02",
            KRFT_LAST_SESSION,
            CONFIRMED_IDENTITY_SOURCE,
            KRFT_SOURCE_URL,
            KRFT_HISTORY_RETRIEVED_AT,
            KRFT_SOURCE_HASH,
        ),
        (
            "khc",
            KHC_ID,
            KHC_FIRST_SESSION,
            "",
            CONFIRMED_IDENTITY_SOURCE,
            KRFT_SOURCE_URL,
            KRFT_HISTORY_RETRIEVED_AT,
            KRFT_SOURCE_HASH,
        ),
        (
            "yhoo",
            YHOO_ID,
            "2015-01-02",
            YHOO_LAST_SESSION,
            EOD_SYMBOL_SOURCE,
            EOD_SYMBOL_URL,
            EOD_SYMBOL_RETRIEVED_AT,
            EOD_SYMBOL_HASH,
        ),
    ):
        _, row = _one_row(
            master,
            master["security_id"].astype(str).eq(security_id),
            f"{kind.upper()} security_master row",
        )
        if not (
            _master_core(row, kind)
            and _date(row.get("active_from")) == active_from
            and _date(row.get("active_to")) == active_to
            and _text(row.get("source")) == source
            and _text(row.get("source_url")) == url
            and _text(row.get("retrieved_at")) == retrieved
            and _text(row.get("source_hash")) == digest
        ):
            raise ValueError(f"{kind.upper()} security_master boundary changed.")
    index, row = _one_row(
        output,
        output["security_id"].astype(str).eq(AABA_ID),
        "AABA security_master row",
    )
    old = _aaba_master_state(row, False)
    repaired = _aaba_master_state(row, True)
    if old == repaired:
        raise ValueError("AABA security_master is neither exact old nor repaired state.")
    permitted: set[tuple[Any, str]] = set()
    if old:
        for column, value in {
            "active_from": AABA_FIRST_SESSION,
            "source": REPAIRED_IDENTITY_SOURCE,
            "source_url": YHOO_SOURCE_URL,
            "retrieved_at": YHOO_RETRIEVED_AT,
            "source_hash": YHOO_SOURCE_HASH,
        }.items():
            output.at[index, column] = value
            permitted.add((index, column))
    if _changed_cells(master, output) != permitted:
        raise AssertionError("security_master repair changed unexpected cells.")
    return output, {"yhoo": old}


def _history_state(row: Mapping[str, Any], kind: str, repaired: bool) -> bool:
    if kind == "khc":
        return bool(
            _text(row.get("security_id")) == KHC_ID
            and _text(row.get("symbol")) == KHC_SYMBOL
            and _text(row.get("exchange")) == "NASDAQ"
            and _date(row.get("effective_from"))
            == (KHC_FIRST_SESSION if repaired else KRFT_LEGAL_COMPLETION)
            and _date(row.get("effective_to")) == ""
            and _text(row.get("source")) == CONFIRMED_IDENTITY_SOURCE
            and _text(row.get("source_url")) == KRFT_SOURCE_URL
            and _text(row.get("retrieved_at")) == KRFT_HISTORY_RETRIEVED_AT
            and _text(row.get("source_hash")) == KRFT_SOURCE_HASH
        )
    if kind in {"yhoo", "aaba"}:
        security_id = YHOO_ID if kind == "yhoo" else AABA_ID
        symbol = YHOO_SYMBOL if kind == "yhoo" else AABA_SYMBOL
        old_start = "2015-01-01"
        repaired_start = "2015-01-01" if kind == "yhoo" else AABA_FIRST_SESSION
        repaired_end = YHOO_LAST_SESSION if kind == "yhoo" else AABA_LAST_SESSION
        common = bool(
            _text(row.get("security_id")) == security_id
            and _text(row.get("symbol")) == symbol
            and _text(row.get("exchange")) == "NASDAQ"
        )
        if repaired:
            return bool(
                common
                and _date(row.get("effective_from")) == repaired_start
                and _date(row.get("effective_to")) == repaired_end
                and _text(row.get("source")) == REPAIRED_IDENTITY_SOURCE
                and _text(row.get("source_url")) == YHOO_SOURCE_URL
                and _text(row.get("retrieved_at")) == YHOO_RETRIEVED_AT
                and _text(row.get("source_hash")) == YHOO_SOURCE_HASH
            )
        return bool(
            common
            and _date(row.get("effective_from")) == old_start
            and _date(row.get("effective_to")) == ""
            and _text(row.get("source")) == EOD_SYMBOL_SOURCE
            and _text(row.get("source_url")) == EOD_SYMBOL_URL
            and _text(row.get("retrieved_at")) == EOD_SYMBOL_RETRIEVED_AT
            and _text(row.get("source_hash")) == EOD_SYMBOL_HASH
        )
    raise ValueError(f"Unknown history kind: {kind}.")


def _rewrite_history(history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    krft_index, krft = _one_row(
        history,
        history["security_id"].astype(str).eq(KRFT_ID)
        & history["symbol"].astype(str).eq(KRFT_SYMBOL),
        "KRFT symbol-history row",
    )
    if not (
        _text(krft.get("exchange")) == "NASDAQ"
        and _date(krft.get("effective_from")) == "2015-01-01"
        and _date(krft.get("effective_to")) == KRFT_LAST_SESSION
        and _text(krft.get("source")) == CONFIRMED_IDENTITY_SOURCE
        and _text(krft.get("source_url")) == KRFT_SOURCE_URL
        and _text(krft.get("retrieved_at")) == KRFT_HISTORY_RETRIEVED_AT
        and _text(krft.get("source_hash")) == KRFT_SOURCE_HASH
    ):
        raise ValueError("KRFT symbol-history boundary changed.")
    krft_before = history.loc[krft_index].copy(deep=True)
    output = history.copy(deep=True)
    changed: dict[str, bool] = {}
    permitted: set[tuple[Any, str]] = set()
    for kind, security_id, updates in (
        ("khc", KHC_ID, {"effective_from": KHC_FIRST_SESSION}),
        (
            "yhoo",
            YHOO_ID,
            {
                "effective_to": YHOO_LAST_SESSION,
                "source": REPAIRED_IDENTITY_SOURCE,
                "source_url": YHOO_SOURCE_URL,
                "retrieved_at": YHOO_RETRIEVED_AT,
                "source_hash": YHOO_SOURCE_HASH,
            },
        ),
        (
            "aaba",
            AABA_ID,
            {
                "effective_from": AABA_FIRST_SESSION,
                "effective_to": AABA_LAST_SESSION,
                "source": REPAIRED_IDENTITY_SOURCE,
                "source_url": YHOO_SOURCE_URL,
                "retrieved_at": YHOO_RETRIEVED_AT,
                "source_hash": YHOO_SOURCE_HASH,
            },
        ),
    ):
        index, row = _one_row(
            output,
            output["security_id"].astype(str).eq(security_id)
            & output["symbol"].astype(str).eq(
                {"khc": KHC_SYMBOL, "yhoo": YHOO_SYMBOL, "aaba": AABA_SYMBOL}[kind]
            ),
            f"{kind.upper()} symbol-history row",
        )
        old = _history_state(row, kind, False)
        repaired = _history_state(row, kind, True)
        if old == repaired:
            raise ValueError(
                f"{kind.upper()} history is neither exact old nor repaired state."
            )
        changed[kind] = old
        if old:
            for column, value in updates.items():
                output.at[index, column] = value
                permitted.add((index, column))
    if _changed_cells(history, output) != permitted:
        raise AssertionError("symbol_history repair changed unexpected cells.")
    for column in history.columns:
        if not _same_cell(krft_before[column], output.at[krft_index, column]):
            raise AssertionError(f"KRFT symbol history changed in {column}.")
    return output, changed


def _anchor_provenance(row: Mapping[str, Any], source: str) -> bool:
    if source == SP_SOURCE:
        return bool(
            _bool(row.get("official"), False)
            and _text(row.get("source")) == SP_SOURCE
            and _text(row.get("source_url")) == SP_SOURCE_URL
            and _text(row.get("source_kind")) == "community"
            and _text(row.get("retrieved_at")) == SP_ANCHOR_RETRIEVED_AT
            and _text(row.get("source_hash")) == SP_SOURCE_HASH
        )
    return bool(
        _bool(row.get("official"), False)
        and _text(row.get("source")) == NASDAQ_SOURCE
        and _text(row.get("source_url")) == NASDAQ_SOURCE_URL
        and _text(row.get("source_kind")) == "community"
        and _text(row.get("retrieved_at")) == NASDAQ_ANCHOR_RETRIEVED_AT
        and _text(row.get("source_hash")) == NASDAQ_SOURCE_HASH
    )


def _rewrite_anchors(anchors: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    output = anchors.copy(deep=True)
    mask = (
        output["index_id"].astype(str).str.lower().eq("sp500")
        & output["anchor_date"].map(_date).eq("2015-01-07")
        & output["security_id"].astype(str).isin({AABA_ID, YHOO_ID})
    )
    index, row = _one_row(output, mask, "YHOO/AABA S&P anchor")
    if not _anchor_provenance(row, SP_SOURCE):
        raise ValueError("YHOO/AABA S&P anchor provenance changed.")
    old = _text(row.get("security_id")) == AABA_ID
    repaired = _text(row.get("security_id")) == YHOO_ID
    if old == repaired:
        raise ValueError("YHOO/AABA S&P anchor identity is invalid.")
    permitted: set[tuple[Any, str]] = set()
    if old:
        output.at[index, "security_id"] = YHOO_ID
        permitted.add((index, "security_id"))
    for index_id, anchor_date, security_id, source in (
        ("nasdaq100", "2015-01-01", YHOO_ID, NASDAQ_SOURCE),
        ("nasdaq100", "2015-01-01", KRFT_ID, NASDAQ_SOURCE),
        ("sp500", "2015-01-07", KRFT_ID, SP_SOURCE),
    ):
        _, invariant = _one_row(
            output,
            output["index_id"].astype(str).str.lower().eq(index_id)
            & output["anchor_date"].map(_date).eq(anchor_date)
            & output["security_id"].astype(str).eq(security_id),
            f"invariant {index_id} anchor for {security_id}",
        )
        if not _anchor_provenance(invariant, source):
            raise ValueError("Invariant index-anchor provenance changed.")
    if output.duplicated(list(dataset_spec("index_constituent_anchors").primary_key)).any():
        raise ValueError("YHOO S&P anchor rebind collides with an existing row.")
    if _changed_cells(anchors, output) != permitted:
        raise AssertionError("Anchor repair changed unexpected cells.")
    return output, {"yhoo": old}


def _event_provenance(row: Mapping[str, Any], source: str) -> bool:
    if source == SP_SOURCE:
        expected = (SP_SOURCE_URL, SP_EVENT_RETRIEVED_AT, SP_SOURCE_HASH)
    else:
        expected = (NASDAQ_SOURCE_URL, NASDAQ_EVENT_RETRIEVED_AT, NASDAQ_SOURCE_HASH)
    url, retrieved, digest = expected
    return bool(
        _date(row.get("announcement_date")) == ""
        and _bool(row.get("official"), False)
        and _text(row.get("source")) == source
        and _text(row.get("source_url")) == url
        and _text(row.get("source_kind")) == "community"
        and _text(row.get("retrieved_at")) == retrieved
        and _text(row.get("source_hash")) == digest
    )


def _event_state(
    row: Mapping[str, Any],
    *,
    index_id: str,
    date: str,
    operation: str,
    security_id: str,
    event_id: str,
    source: str,
) -> bool:
    return bool(
        _text(row.get("event_id")) == event_id
        and _text(row.get("index_id")).lower() == index_id
        and _date(row.get("effective_date")) == date
        and _text(row.get("operation")).upper() == operation
        and _text(row.get("security_id")) == security_id
        and _event_provenance(row, source)
    )


def _rewrite_index_events(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    output = events.copy(deep=True)
    changed: dict[str, bool] = {}
    permitted: set[tuple[Any, str]] = set()
    specs = (
        (
            "krft_add",
            {KRFT_NASDAQ_ADD_OLD_ID, KRFT_NASDAQ_ADD_NEW_ID},
            dict(
                index_id="nasdaq100",
                operation="ADD",
                security_id=KHC_ID,
                source=NASDAQ_SOURCE,
            ),
            KRFT_LEGAL_COMPLETION,
            KRFT_NASDAQ_ADD_OLD_ID,
            KHC_FIRST_SESSION,
            KRFT_NASDAQ_ADD_NEW_ID,
        ),
        (
            "krft_remove",
            {KRFT_NASDAQ_REMOVE_OLD_ID, KRFT_NASDAQ_REMOVE_NEW_ID},
            dict(
                index_id="nasdaq100",
                operation="REMOVE",
                security_id=KRFT_ID,
                source=NASDAQ_SOURCE,
            ),
            KRFT_LEGAL_COMPLETION,
            KRFT_NASDAQ_REMOVE_OLD_ID,
            KHC_FIRST_SESSION,
            KRFT_NASDAQ_REMOVE_NEW_ID,
        ),
    )
    for label, ids, common, old_date, old_id, new_date, new_id in specs:
        index, row = _one_row(
            output,
            output["event_id"].astype(str).isin(ids),
            f"{label} membership event",
        )
        old = _event_state(row, date=old_date, event_id=old_id, **common)
        repaired = _event_state(row, date=new_date, event_id=new_id, **common)
        if old == repaired:
            raise ValueError(f"{label} is neither exact old nor repaired state.")
        changed[label] = old
        if old:
            output.at[index, "event_id"] = new_id
            output.at[index, "effective_date"] = new_date
            permitted.update({(index, "event_id"), (index, "effective_date")})

    yhoo_mask = output["event_id"].astype(str).isin(
        {YHOO_SP_REMOVE_OLD_ID, YHOO_SP_REMOVE_NEW_ID}
    )
    index, row = _one_row(output, yhoo_mask, "YHOO/AABA S&P removal")
    yhoo_common = dict(
        index_id="sp500",
        date=AABA_FIRST_SESSION,
        operation="REMOVE",
        source=SP_SOURCE,
    )
    old = _event_state(
        row,
        security_id=AABA_ID,
        event_id=YHOO_SP_REMOVE_OLD_ID,
        **yhoo_common,
    )
    repaired = _event_state(
        row,
        security_id=YHOO_ID,
        event_id=YHOO_SP_REMOVE_NEW_ID,
        **yhoo_common,
    )
    if old == repaired:
        raise ValueError("YHOO/AABA S&P removal is neither exact old nor repaired state.")
    changed["yhoo"] = old
    if old:
        output.at[index, "event_id"] = YHOO_SP_REMOVE_NEW_ID
        output.at[index, "security_id"] = YHOO_ID
        permitted.update({(index, "event_id"), (index, "security_id")})

    for event_id, common in (
        (
            KRFT_SP_ADD_ID,
            dict(
                index_id="sp500",
                date=KHC_FIRST_SESSION,
                operation="ADD",
                security_id=KHC_ID,
                source=SP_SOURCE,
            ),
        ),
        (
            KRFT_SP_REMOVE_ID,
            dict(
                index_id="sp500",
                date=KHC_FIRST_SESSION,
                operation="REMOVE",
                security_id=KRFT_ID,
                source=SP_SOURCE,
            ),
        ),
        (
            YHOO_NASDAQ_REMOVE_ID,
            dict(
                index_id="nasdaq100",
                date=AABA_FIRST_SESSION,
                operation="REMOVE",
                security_id=YHOO_ID,
                source=NASDAQ_SOURCE,
            ),
        ),
    ):
        _, invariant = _one_row(
            output,
            output["event_id"].astype(str).eq(event_id),
            f"invariant membership event {event_id}",
        )
        if not _event_state(invariant, event_id=event_id, **common):
            raise ValueError(f"Invariant membership event {event_id} changed.")
    if output.duplicated(list(dataset_spec("index_membership_events").primary_key)).any():
        raise ValueError("Membership-event rekey collides with an existing row.")
    if _changed_cells(events, output) != permitted:
        raise AssertionError("Membership repair changed unexpected cells.")
    return output, changed


def _rewrite_archive(
    archive: pd.DataFrame, *, completed_session: str
) -> tuple[pd.DataFrame, dict[str, bool]]:
    output = archive.copy(deep=True)
    expected = _archive_row_expected(
        completed_session=completed_session,
        source_url=YHOO_SOURCE_URL,
        source_hash=YHOO_SOURCE_HASH,
        retrieved_at=YHOO_RETRIEVED_AT,
    )
    matches = output["archive_id"].astype(str).eq(YHOO_SOURCE_HASH)
    if matches.any():
        _, row = _one_row(output, matches, "YHOO repaired source_archive row")
        if not _archive_row_is_exact(row, expected):
            raise ValueError("Existing YHOO repaired source_archive row is not exact.")
        return output, {"yhoo": False}
    row = {column: expected.get(column, None) for column in output.columns}
    output = pd.concat(
        [output, pd.DataFrame([row], columns=output.columns)],
        ignore_index=True,
        sort=False,
    )
    if output.duplicated(list(dataset_spec("source_archive").primary_key)).any():
        raise ValueError("YHOO source archive addition collides with an existing row.")
    return output, {"yhoo": True}


def _verify_prices(prices: pd.DataFrame) -> None:
    security_ids = prices["security_id"].astype(str)
    sessions = pd.to_datetime(prices["session"], errors="coerce")
    for security_id, (
        count,
        first,
        last,
        edge_open,
        edge_close,
    ) in EXPECTED_PRICE_BOUNDARIES.items():
        rows = prices.loc[security_ids.eq(security_id)].copy()
        rows["_session"] = sessions.loc[rows.index]
        rows.sort_values("_session", inplace=True)
        if (
            len(rows) != count
            or rows["_session"].isna().any()
            or _date(rows.iloc[0]["session"]) != first
            or _date(rows.iloc[-1]["session"]) != last
        ):
            raise ValueError(f"Reviewed price inventory changed for {security_id}.")
        edge = rows.iloc[-1] if security_id != KHC_ID else rows.iloc[0]
        if not (
            math.isclose(float(edge["open"]), edge_open, rel_tol=0, abs_tol=1e-12)
            and math.isclose(float(edge["close"]), edge_close, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError(f"Reviewed market boundary price changed for {security_id}.")
    aaba_market = prices.loc[
        security_ids.eq(AABA_ID)
        & sessions.eq(pd.Timestamp(AABA_FIRST_SESSION))
    ]
    if len(aaba_market) != 1 or not (
        math.isclose(
            float(aaba_market.iloc[0]["open"]),
            AABA_FIRST_BAR[0],
            rel_tol=0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(aaba_market.iloc[0]["close"]),
            AABA_FIRST_BAR[1],
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("AABA first market-session bar changed.")


def _planned_versions(release: DataRelease, *, plan_id: str) -> dict[str, str]:
    if not re.fullmatch(r"[0-9a-f]{32}", plan_id):
        raise ValueError("Transition repair plan_id must be a 32-character hex UUID.")
    seed = json.dumps(
        {
            "operation": OPERATION,
            "base_release": release.version,
            "krft_event": KRFT_NEW_EVENT_ID,
            "yhoo_event": YHOO_NEW_EVENT_ID,
            "yhoo_source": YHOO_SOURCE_HASH,
            "plan_id": plan_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    token = hashlib.sha256(seed.encode()).hexdigest()[:16]
    session = release.completed_session.replace("-", "")
    return {
        dataset: f"market-transitions-{session}-{token}-{dataset}"
        for dataset in WRITE_DATASETS
    }


def _factor_source_version(price_version: str, action_version: str) -> str:
    if not price_version or not action_version:
        raise RuntimeError("Factor lineage requires exact price/action versions.")
    return f"{price_version}+{action_version}"


def _rebind_factors(
    current: pd.DataFrame,
    prices: pd.DataFrame,
    actions_before: pd.DataFrame,
    actions_after: pd.DataFrame,
    *,
    source_version: str,
) -> tuple[pd.DataFrame, bool]:
    if len(current) != len(prices):
        raise ValueError("Adjustment-factor and raw-price row counts diverged.")
    required = {
        "security_id",
        "session",
        "split_factor",
        "total_return_factor",
        "source_version",
        "calculated_at",
        "source",
        "retrieved_at",
        "source_hash",
    }
    if missing := required - set(current.columns):
        raise ValueError("Adjustment factors lack lineage columns: " + ", ".join(sorted(missing)))
    # The action rewrite is already cell-scoped.  Prove here that every changed
    # action is a lifecycle-only type ignored by the factor engine; therefore a
    # full 2.1M-row rebuild would be redundant and could only reproduce the two
    # existing economic columns.
    before_by_id = actions_before.set_index("event_id", drop=False)
    after_by_id = actions_after.set_index("event_id", drop=False)
    transition_ids = {
        KRFT_OLD_EVENT_ID,
        KRFT_NEW_EVENT_ID,
        YHOO_OLD_EVENT_ID,
        YHOO_NEW_EVENT_ID,
    }
    unchanged_old_ids = set(before_by_id.index) - transition_ids
    unchanged_new_ids = set(after_by_id.index) - transition_ids
    if unchanged_old_ids != unchanged_new_ids:
        raise ValueError("Transition repair changed non-transition action inventory.")
    for event_id in unchanged_old_ids:
        left = before_by_id.loc[event_id]
        right = after_by_id.loc[event_id]
        if isinstance(left, pd.DataFrame) or isinstance(right, pd.DataFrame):
            raise ValueError("Corporate-action event_id inventory is not unique.")
        for column in actions_before.columns:
            if not _same_cell(left[column], right[column]):
                raise ValueError(
                    f"Transition repair changed unrelated action {event_id}.{column}."
                )
    changed_types = {
        _text(row.get("action_type"))
        for row in actions_after.loc[
            actions_after["event_id"].astype(str).isin(
                {KRFT_NEW_EVENT_ID, YHOO_NEW_EVENT_ID}
            )
        ].to_dict("records")
    }
    if changed_types != {"stock_merger", "ticker_change"} or changed_types & (
        set(RATIO_ACTIONS) | set(CASH_DISTRIBUTION_ACTIONS)
    ):
        raise ValueError("Transition repair no longer has zero factor-economic impact.")

    output = current.copy(deep=True)
    economic_columns = ("security_id", "session", "split_factor", "total_return_factor")
    economic_before = output.loc[:, list(economic_columns)].copy(deep=True)
    exact = bool(
        set(output["source_version"].astype(str)) == {source_version}
        and set(output["source_hash"].astype(str)) == {source_version}
        and set(output["source"].astype(str)) == {"derived"}
        and set(output["calculated_at"].astype(str)) == {REVIEWED_AT}
        and set(output["retrieved_at"].astype(str)) == {REVIEWED_AT}
    )
    output["source_version"] = source_version
    output["calculated_at"] = REVIEWED_AT
    output["source"] = "derived"
    output["retrieved_at"] = REVIEWED_AT
    output["source_hash"] = source_version
    if not economic_before.equals(output.loc[:, list(economic_columns)]):
        raise AssertionError("Factor lineage rebind changed keys or economics.")
    return output.reset_index(drop=True), not exact


class _CandidateRepository:
    def __init__(
        self,
        base: LocalDatasetRepository,
        versions: Mapping[str, str],
        overrides: Mapping[str, pd.DataFrame],
    ):
        self.base = base
        self.versions = dict(versions)
        self.overrides = dict(overrides)

    def current_manifest(self, dataset: str):
        version = self.versions.get(dataset)
        return self.base.manifest_for_version(dataset, version) if version else None

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        if dataset in self.overrides:
            return self.overrides[dataset].copy()
        version = self.versions.get(dataset)
        return self.base.read_frame(dataset, version) if version else pd.DataFrame()


def _assert_coherent(label: str, values: Mapping[str, bool]) -> bool:
    states = set(values.values())
    if len(states) != 1:
        detail = ", ".join(f"{key}={value}" for key, value in sorted(values.items()))
        raise RuntimeError(f"{label} transition repair is partially applied: {detail}.")
    return next(iter(states))


def _assert_nonfactor_replay(
    frames: Mapping[str, pd.DataFrame], *, completed_session: str
) -> None:
    replay_actions, action_flags = _rewrite_actions(frames["corporate_actions"])
    replay_resolutions, resolution_flags = _rewrite_resolutions(
        frames["lifecycle_resolutions"]
    )
    replay_master, master_flags = _rewrite_master(frames["security_master"])
    replay_history, history_flags = _rewrite_history(frames["symbol_history"])
    replay_anchors, anchor_flags = _rewrite_anchors(
        frames["index_constituent_anchors"]
    )
    replay_events, event_flags = _rewrite_index_events(
        frames["index_membership_events"]
    )
    replay_archive, archive_flags = _rewrite_archive(
        frames["source_archive"], completed_session=completed_session
    )
    flags = {
        **{f"action_{key}": value for key, value in action_flags.items()},
        **{f"resolution_{key}": value for key, value in resolution_flags.items()},
        **{f"master_{key}": value for key, value in master_flags.items()},
        **{f"history_{key}": value for key, value in history_flags.items()},
        **{f"anchor_{key}": value for key, value in anchor_flags.items()},
        **{f"event_{key}": value for key, value in event_flags.items()},
        **{f"archive_{key}": value for key, value in archive_flags.items()},
    }
    if any(flags.values()):
        raise AssertionError(f"Transition transforms are not idempotent: {flags}.")
    for dataset, replay in {
        "corporate_actions": replay_actions,
        "lifecycle_resolutions": replay_resolutions,
        "security_master": replay_master,
        "symbol_history": replay_history,
        "index_constituent_anchors": replay_anchors,
        "index_membership_events": replay_events,
        "source_archive": replay_archive,
    }.items():
        pd.testing.assert_frame_equal(
            frames[dataset].reset_index(drop=True),
            replay.reset_index(drop=True),
            check_dtype=False,
        )


def _targeted_snapshot_overrides(
    overrides: Mapping[str, pd.DataFrame], prices: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Limit expensive cross-dataset replay to the four reviewed identities.

    Full-frame schema/PK/provenance validation still runs above.  The repository
    snapshot validator's Python sets over every 2.1M price/factor key dominate
    runtime but add no coverage for rows this repair cannot change.  This exact
    slice retains every affected action, identity, anchor/event, price and
    factor row, including successor dividends through the completed session.
    """

    identities = {KRFT_ID, KHC_ID, YHOO_ID, AABA_ID}

    def related(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[frame["security_id"].astype(str).isin(identities)].copy()

    return {
        "security_master": related(overrides["security_master"]),
        "daily_price_raw": related(prices),
        "adjustment_factors": related(overrides["adjustment_factors"]),
        "corporate_actions": related(overrides["corporate_actions"]),
        "symbol_history": related(overrides["symbol_history"]),
        "index_constituent_anchors": related(
            overrides["index_constituent_anchors"]
        ),
        "index_membership_events": related(overrides["index_membership_events"]),
    }


def _pointer_etags(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, str | None]:
    """Pin every write and read dependency to the release being planned."""

    output: dict[str, str | None] = {}
    for dataset in REQUIRED_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != release.dataset_versions.get(dataset):
            raise RuntimeError(f"Release/current pointer mismatch: {dataset}.")
        output[dataset] = etag
    return output


def prepare_repair(repository: LocalDatasetRepository) -> PreparedRepair:
    _static_contract()
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    missing = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError(
            "Current release lacks transition repair datasets: " + ", ".join(missing)
        )
    pointer_etags = _pointer_etags(repository, release)
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in REQUIRED_DATASETS
    }
    krft_payload = _verify_krft_evidence(
        repository,
        frames["source_archive"],
        completed_session=release.completed_session,
    )
    yhoo_payload, yhoo_already_archived = _load_yhoo_evidence(
        repository,
        frames["source_archive"],
        completed_session=release.completed_session,
    )
    _verify_prices(frames["daily_price_raw"])

    actions, action_flags = _rewrite_actions(frames["corporate_actions"])
    resolutions, resolution_flags = _rewrite_resolutions(
        frames["lifecycle_resolutions"]
    )
    master, master_flags = _rewrite_master(frames["security_master"])
    history, history_flags = _rewrite_history(frames["symbol_history"])
    anchors, anchor_flags = _rewrite_anchors(frames["index_constituent_anchors"])
    events, event_flags = _rewrite_index_events(frames["index_membership_events"])
    archive, archive_flags = _rewrite_archive(
        frames["source_archive"], completed_session=release.completed_session
    )

    krft_changed = _assert_coherent(
        "KRFT->KHC",
        {
            "action": action_flags["krft"],
            "resolution": resolution_flags["krft"],
            "khc_history": history_flags["khc"],
            "nasdaq_add": event_flags["krft_add"],
            "nasdaq_remove": event_flags["krft_remove"],
        },
    )
    yhoo_changed = _assert_coherent(
        "YHOO->AABA",
        {
            "action": action_flags["yhoo"],
            "resolution": resolution_flags["yhoo"],
            "aaba_master": master_flags["yhoo"],
            "yhoo_history": history_flags["yhoo"],
            "aaba_history": history_flags["aaba"],
            "sp_anchor": anchor_flags["yhoo"],
            "sp_remove": event_flags["yhoo"],
            "source_archive": archive_flags["yhoo"],
        },
    )
    if yhoo_already_archived == archive_flags["yhoo"]:
        raise RuntimeError("YHOO archive payload/state agreement is inconsistent.")
    changed = krft_changed or yhoo_changed
    planned_versions = (
        _planned_versions(release, plan_id=uuid.uuid4().hex) if changed else {}
    )
    action_version = (
        planned_versions["corporate_actions"]
        if changed
        else release.dataset_versions["corporate_actions"]
    )
    factor_lineage = _factor_source_version(
        release.dataset_versions["daily_price_raw"], action_version
    )
    factors, factor_changed = _rebind_factors(
        frames["adjustment_factors"],
        frames["daily_price_raw"],
        frames["corporate_actions"],
        actions,
        source_version=factor_lineage,
    )
    if changed != factor_changed:
        raise RuntimeError(
            "Transition action state and adjustment-factor lineage are inconsistent."
        )

    overrides = {
        "corporate_actions": actions,
        "lifecycle_resolutions": resolutions,
        "security_master": master,
        "symbol_history": history,
        "index_constituent_anchors": anchors,
        "index_membership_events": events,
        "source_archive": archive,
        "adjustment_factors": factors,
    }
    for dataset, frame in overrides.items():
        validate_dataset(
            dataset,
            frame,
            completed_session=release.completed_session,
            incomplete_action_policy="block",
        ).raise_for_errors()
    targeted = _targeted_snapshot_overrides(
        overrides, frames["daily_price_raw"]
    )
    validate_repository_snapshot(
        _CandidateRepository(repository, release.dataset_versions, targeted)
    ).raise_for_errors()
    _assert_nonfactor_replay(overrides, completed_session=release.completed_session)
    expected_lineage = {factor_lineage}
    if (
        set(factors["source_version"].astype(str)) != expected_lineage
        or set(factors["source_hash"].astype(str)) != expected_lineage
        or set(factors["source"].astype(str)) != {"derived"}
        or set(factors["calculated_at"].astype(str)) != {REVIEWED_AT}
        or set(factors["retrieved_at"].astype(str)) != {REVIEWED_AT}
    ):
        raise RuntimeError("Planned adjustment factors do not have exact lineage.")

    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions=planned_versions,
        frames=overrides if changed else {},
        evidence_payloads={"krft": krft_payload, "yhoo": yhoo_payload},
        summary={
            "status": "validated_offline_plan" if changed else "already_repaired",
            "mode": "plan",
            "writes_performed": False,
            "base_release_version": release.version,
            "completed_session": release.completed_session,
            "krft_market_transition_planned": krft_changed,
            "krft_legal_completion_date": KRFT_LEGAL_COMPLETION,
            "krft_last_price_date": KRFT_LAST_SESSION,
            "khc_first_trading_date": KHC_FIRST_SESSION,
            "krft_old_event_id": KRFT_OLD_EVENT_ID,
            "krft_new_event_id": KRFT_NEW_EVENT_ID,
            "yhoo_market_transition_planned": yhoo_changed,
            "yhoo_operating_business_sale_completion_date": YHOO_BUSINESS_SALE_COMPLETION,
            "yhoo_legal_name_change_date": YHOO_LEGAL_NAME_CHANGE,
            "yhoo_last_price_date": YHOO_LAST_SESSION,
            "aaba_first_trading_date": AABA_FIRST_SESSION,
            "yhoo_old_event_id": YHOO_OLD_EVENT_ID,
            "yhoo_new_event_id": YHOO_NEW_EVENT_ID,
            "corporate_action_rows_rekeyed": int(krft_changed) + int(yhoo_changed),
            "lifecycle_resolution_rows_relinked": int(krft_changed)
            + int(yhoo_changed),
            "security_master_rows_rebounded": int(yhoo_changed),
            "symbol_history_rows_rebounded": int(krft_changed)
            + 2 * int(yhoo_changed),
            "index_anchor_rows_rebound": int(yhoo_changed),
            "index_membership_rows_rekeyed": 2 * int(krft_changed)
            + int(yhoo_changed),
            "source_archive_rows_planned": int(archive_flags["yhoo"]),
            "krft_special_dividend_preserved_exactly": True,
            "raw_price_rows_changed": 0,
            "adjustment_factor_rows": len(factors),
            "adjustment_factor_economic_rows_changed": 0,
            "adjustment_factor_provenance_rows_rebound": len(factors) if changed else 0,
            "factor_source_version": factor_lineage,
            "yhoo_archive_object_path": _archive_path(
                release.completed_session, YHOO_SOURCE_HASH
            ),
            "evidence_payloads_hash_verified": True,
            "targeted_cross_dataset_snapshot_validated": True,
            "idempotency_replay_validated": True,
            "cross_validation_market_date_inventory_sha256": (
                TRUSTED_REVIEWED_TERMINAL_MARKET_DATE_CORRECTIONS_SHA256
            ),
            "planned_versions": dict(planned_versions),
            "write_datasets": list(WRITE_DATASETS),
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        },
    )


@contextmanager
def _exclusive_repository_lock(repository: LocalDatasetRepository):
    lock_path = repository.root / ".locks/market-store-write.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery = repository.root / RECOVERY_DIR
        if recovery.exists() and tuple(recovery.glob("*.json")):
            raise RuntimeError(
                "Unresolved KRFT/YHOO transition recovery marker blocks writes."
            )
        transactions = repository.root / TRANSACTION_DIR
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = _text(json.loads(journal.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        "Interrupted KRFT/YHOO transition transaction blocks writes: "
                        f"{journal}."
                    )
        yield


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(
        path,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )


def _assert_inputs_unchanged(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> None:
    release, release_etag = repository.current_release()
    if (
        release is None
        or release.to_bytes() != prepared.release.to_bytes()
        or release_etag != prepared.release_etag
    ):
        raise RuntimeError("Current release changed after transition planning.")
    for dataset in REQUIRED_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if (
            pointer is None
            or pointer.version != prepared.release.dataset_versions[dataset]
            or etag != prepared.pointer_etags[dataset]
        ):
            raise RuntimeError(
                f"{dataset} pointer changed after transition planning."
            )


def _persist_yhoo_evidence(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> None:
    payload = prepared.evidence_payloads["yhoo"]
    _verify_yhoo_payload(payload)
    path = _safe_path(
        repository.root,
        _archive_path(prepared.release.completed_session, YHOO_SOURCE_HASH),
    )
    if path.is_file():
        existing = _read_gzip(path, "YHOO")
        if existing != payload:
            raise ValueError("Persisted YHOO evidence bytes conflict.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, gzip.compress(payload, mtime=0))
    if _read_gzip(path, "YHOO") != payload:
        raise RuntimeError("YHOO evidence post-write verification failed.")


def _metadata_for_write(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    dataset: str,
) -> dict[str, Any]:
    current = repository.manifest_for_version(
        dataset,
        prepared.release.dataset_versions[dataset],
    )
    metadata = dict(current.metadata)
    input_versions = dict(prepared.release.dataset_versions)
    output_versions = dict(input_versions)
    output_versions.update(prepared.planned_versions)
    metadata.update(
        {
            "operation": OPERATION,
            "policy": POLICY,
            "input_release_version": prepared.release.version,
            "input_versions": input_versions,
            "output_versions": output_versions,
            "krft_transition_event_id": KRFT_NEW_EVENT_ID,
            "krft_source_hash": KRFT_SOURCE_HASH,
            "yhoo_transition_event_id": YHOO_NEW_EVENT_ID,
            "yhoo_source_hash": YHOO_SOURCE_HASH,
            "cross_validation_market_date_inventory_sha256": (
                TRUSTED_REVIEWED_TERMINAL_MARKET_DATE_CORRECTIONS_SHA256
            ),
            "raw_price_rows_changed": 0,
            "adjustment_factor_economic_rows_changed": 0,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        }
    )
    if dataset == "source_archive":
        metadata.update(
            {
                "source_archive_rows_added": 1,
                "yhoo_archive_object_path": _archive_path(
                    prepared.release.completed_session, YHOO_SOURCE_HASH
                ),
            }
        )
    elif dataset == "adjustment_factors":
        lineage = _factor_source_version(
            prepared.release.dataset_versions["daily_price_raw"],
            prepared.planned_versions["corporate_actions"],
        )
        factors = prepared.frames[dataset]
        if (
            set(factors["source_version"].astype(str)) != {lineage}
            or set(factors["source_hash"].astype(str)) != {lineage}
            or set(factors["source"].astype(str)) != {"derived"}
            or set(factors["calculated_at"].astype(str)) != {REVIEWED_AT}
            or set(factors["retrieved_at"].astype(str)) != {REVIEWED_AT}
        ):
            raise RuntimeError("Prepared transition factors have stale lineage.")
        metadata.update(
            {
                "source_version": lineage,
                "source_daily_price_version": prepared.release.dataset_versions[
                    "daily_price_raw"
                ],
                "source_corporate_actions_version": prepared.planned_versions[
                    "corporate_actions"
                ],
                "expected_economic_rows_changed": 0,
            }
        )
    return metadata


def _restore_transaction(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: Mapping[str, bytes],
    planned_versions: Mapping[str, str],
    committed_release_version: str,
    old_versions: Mapping[str, str],
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        current = repository.objects.get("releases/current.json")
        if current.data != old_release_bytes:
            observed = DataRelease.from_bytes(current.data)
            expected_versions = {**dict(old_versions), **dict(planned_versions)}
            belongs = (
                bool(committed_release_version)
                and observed.version == committed_release_version
            ) or observed.dataset_versions == expected_versions
            if not belongs:
                raise RuntimeError(
                    "unexpected release during KRFT/YHOO rollback: "
                    f"{observed.version}"
                )
            repository.objects.put(
                "releases/current.json",
                old_release_bytes,
                if_match=current.etag,
            )
    except Exception as exc:
        errors.append(f"releases/current.json: {type(exc).__name__}: {exc}")
    for dataset in reversed(WRITE_DATASETS):
        key = repository.current_key(dataset)
        try:
            current = repository.objects.get(key)
            old = old_pointer_bytes[dataset]
            if current.data != old:
                pointer = CurrentPointer.from_bytes(current.data)
                if pointer.version != planned_versions[dataset]:
                    raise RuntimeError(
                        f"unexpected {dataset} pointer during KRFT/YHOO rollback: "
                        f"{pointer.version}"
                    )
                repository.objects.put(key, old, if_match=current.etag)
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _assert_applied_release(
    repository: LocalDatasetRepository,
    committed: DataRelease,
    prepared: PreparedRepair,
) -> None:
    current, _ = repository.current_release()
    if current is None or current.to_bytes() != committed.to_bytes():
        raise RuntimeError("Committed KRFT/YHOO transition release is not current.")
    for dataset, version in committed.dataset_versions.items():
        pointer, _ = repository.current_pointer(dataset)
        if pointer is None or pointer.version != version:
            raise RuntimeError(
                f"Applied transition release pointer mismatch: {dataset}."
            )
    for dataset, version in prepared.release.dataset_versions.items():
        if dataset not in WRITE_DATASETS and committed.dataset_versions[dataset] != version:
            raise RuntimeError(
                f"Transition repair changed out-of-scope dataset {dataset}."
            )
    lineage = _factor_source_version(
        committed.dataset_versions["daily_price_raw"],
        committed.dataset_versions["corporate_actions"],
    )
    factor_manifest = repository.manifest_for_version(
        "adjustment_factors", committed.dataset_versions["adjustment_factors"]
    )
    if any(
        _text(factor_manifest.metadata.get(key)) != value
        for key, value in {
            "source_version": lineage,
            "source_daily_price_version": committed.dataset_versions[
                "daily_price_raw"
            ],
            "source_corporate_actions_version": committed.dataset_versions[
                "corporate_actions"
            ],
        }.items()
    ):
        raise RuntimeError("Transition factor manifest lineage is not release-exact.")
    yhoo_path = _safe_path(
        repository.root,
        _archive_path(committed.completed_session, YHOO_SOURCE_HASH),
    )
    _verify_yhoo_payload(_read_gzip(yhoo_path, "YHOO"))
    replay = prepare_repair(repository)
    if replay.summary["status"] != "already_repaired":
        raise RuntimeError("KRFT/YHOO transition repair is not idempotent.")
    if replay.summary["raw_price_rows_changed"] != 0:
        raise RuntimeError("Applied transition repair reports a price mutation.")


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    inject_failure: FailureInjector | None = None,
) -> dict[str, Any]:
    inject = inject_failure or (lambda _stage: None)
    with _exclusive_repository_lock(repository):
        _assert_inputs_unchanged(repository, prepared)
        # Rebuild the complete plan under the exclusive lock.  This rejects a
        # caller-mutated PreparedRepair and closes the check/write interval.
        current_plan = prepare_repair(repository)
        if current_plan.summary["status"] == "already_repaired":
            return {
                **current_plan.summary,
                "mode": "apply",
                "writes_performed": False,
            }
        archive = repository.read_frame(
            "source_archive",
            current_plan.release.dataset_versions["source_archive"],
        )
        krft = _verify_krft_evidence(
            repository,
            archive,
            completed_session=current_plan.release.completed_session,
        )
        yhoo, already_archived = _load_yhoo_evidence(
            repository,
            archive,
            completed_session=current_plan.release.completed_session,
        )
        if already_archived:
            raise RuntimeError("Old transition state unexpectedly has YHOO archived.")
        if (
            krft != current_plan.evidence_payloads["krft"]
            or yhoo != current_plan.evidence_payloads["yhoo"]
        ):
            raise RuntimeError("Transition evidence changed during locked replanning.")
        prices = repository.read_frame(
            "daily_price_raw",
            current_plan.release.dataset_versions["daily_price_raw"],
        )
        _verify_prices(prices)
        planned = dict(current_plan.planned_versions)
        if set(planned) != set(WRITE_DATASETS) or len(set(planned.values())) != len(
            WRITE_DATASETS
        ):
            raise RuntimeError("Prepared transition repair has invalid versions.")
        old_release = repository.objects.get("releases/current.json")
        if old_release.etag != current_plan.release_etag:
            raise RuntimeError("Release CAS changed during locked transition planning.")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version
                != current_plan.release.dataset_versions[dataset]
                or value.etag != current_plan.pointer_etags[dataset]
            ):
                raise RuntimeError(
                    f"{dataset} pointer changed before transition apply."
                )
            old_pointers[dataset] = value.data
        transaction_id = uuid.uuid4().hex
        journal_path = repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        journal: dict[str, Any] = {
            "schema": "us_krft_yhoo_market_transitions_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": current_plan.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                key: base64.b64encode(value).decode("ascii")
                for key, value in old_pointers.items()
            },
            "planned_versions": planned,
            "krft_source_hash": KRFT_SOURCE_HASH,
            "yhoo_source_hash": YHOO_SOURCE_HASH,
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        try:
            inject("after_journal")
            _persist_yhoo_evidence(repository, current_plan)
            inject("after_evidence_write")
            versions = dict(current_plan.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    current_plan.frames[dataset],
                    completed_session=current_plan.release.completed_session,
                    incomplete_action_policy="block",
                    metadata=_metadata_for_write(
                        repository, current_plan, dataset
                    ),
                    expected_pointer_etag=current_plan.pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"{dataset} write conflicted: {result.conflict_path}."
                    )
                if result.manifest.version != planned[dataset]:
                    raise RuntimeError(f"Unexpected {dataset} version was written.")
                versions[dataset] = result.manifest.version
                inject(f"after_write:{dataset}")
            committed = repository.commit_release(
                current_plan.release.completed_session,
                versions,
                quality=current_plan.release.quality,
                warnings=current_plan.release.warnings,
                expected_etag=current_plan.release_etag,
            )
            inject("after_release_commit")
            _assert_applied_release(repository, committed, current_plan)
            journal.update(
                {
                    "status": "committed",
                    "committed_release_version": committed.version,
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            return {
                **current_plan.summary,
                "status": "applied",
                "mode": "apply",
                "writes_performed": True,
                "new_release_version": committed.version,
                "transaction_id": transaction_id,
                "quality": str(committed.quality),
                "warnings": list(committed.warnings),
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_versions=planned,
                committed_release_version=committed.version if committed else "",
                old_versions=current_plan.release.dataset_versions,
            )
            journal.update(
                {
                    "status": "rollback_failed" if rollback_errors else "rolled_back",
                    "original_error": f"{type(original).__name__}: {original}",
                    "rollback_errors": list(rollback_errors),
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            if rollback_errors:
                recovery = repository.root / RECOVERY_DIR / f"{transaction_id}.json"
                _write_journal(recovery, journal)
                raise RuntimeError(
                    "KRFT/YHOO rollback was incomplete; recovery marker blocks "
                    f"writes: {recovery}; errors={rollback_errors}."
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or apply KRFT/KHC and YHOO/AABA transitions offline."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly select the default read-only plan mode.",
    )
    mode.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repository = LocalDatasetRepository(args.cache_root)
    prepared = prepare_repair(repository)
    result = apply_repair(repository, prepared) if args.apply else prepared.summary
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
