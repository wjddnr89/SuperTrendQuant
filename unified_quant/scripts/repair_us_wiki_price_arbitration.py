#!/usr/bin/env python3
"""Archive a reproducible frozen-WIKI price arbitration for BBBY and BBT.

This operation is intentionally price-only.  It proves that the current
identity-bound BBBY and BBT raw histories agree with the frozen Quandl WIKI
histories closely enough to replace unsafe Yahoo symbol-only comparisons.
It never rewrites prices, corporate actions, adjustment factors, identities,
or index membership.

Legacy DuPont (DD) is audited in the same plan but remains fail-closed.  The
WIKI ``3.2`` distribution is a rounded proxy for the 2015 Chemours spin-off,
not a cash dividend.  The local store has neither hash-pinned official 2015
spin-off bytes nor a complete identity-bound Chemours price path, so this
script must not create that action or change DD factors.

Plan mode is read-only and is the default.  Apply writes only two immutable
evidence objects plus a new ``source_archive`` version and release pointer.
There is no network, EODHD, R2, or deletion code path.  Kaggle reports the
formal license as ``Unknown``; apply therefore requires an explicit local
private/internal-only acknowledgement.  R2 publication requires a separate
publisher acknowledgement and is outside this script.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import io
import json
import math
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from supertrend_quant.indicators import add_triple_supertrend
from supertrend_quant.market_store.adjustments import apply_adjustment_factors
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.validation import (
    validate_dataset,
    validate_repository_snapshot,
)


OPERATION = "repair_us_wiki_price_arbitration"
DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_WIKI_ZIP = Path("/tmp/marketneutral-quandl-wiki-prices.zip")
DEFAULT_KAGGLE_METADATA = Path("/tmp/kaggle_wiki_metadata.json")
DATASET = "source_archive"
REQUIRED_DATASETS = (
    "daily_price_raw",
    "corporate_actions",
    "adjustment_factors",
    "security_master",
    "symbol_history",
    DATASET,
)
TRANSACTION_DIR = "transactions/us-wiki-price-arbitration"
RECOVERY_DIR = "recovery/us-wiki-price-arbitration"

WIKI_MEMBER = "WIKI_PRICES.csv"
WIKI_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "marketneutral/quandl-wiki-prices-us-equites/WIKI_PRICES.csv"
)
KAGGLE_METADATA_URL = (
    "https://www.kaggle.com/api/v1/datasets/view/"
    "marketneutral/quandl-wiki-prices-us-equites"
)
WIKI_RETRIEVED_AT = "2026-07-18T03:58:26.808706Z"
REVIEWED_AT = "2026-07-19T04:30:00Z"
WIKI_LICENSE_WARNING = (
    "Kaggle Quandl WIKI licenseName=Unknown; private/internal-only; "
    "redistribution/public publication blocked."
)

BBBY_ID = "US:EODHD:dbd287da-b35f-5aaa-873c-76f941b4d93b"
BBT_ID = "US:EODHD:aadcce22-62c7-522f-bbeb-861933af1d99"
TFC_ID = "US:EODHD:e9a02afb-49bb-545e-8b5a-824d630a1332"
LEGACY_DD_ID = "US:SEC:7992d65b-3a26-5cae-b96d-bd01f695d1c1"
DWDP_ID = "US:EODHD:f174ac51-b4a4-5f29-84b0-12b8b73f0bc3"

DD_SEGMENT_HASH = (
    "36bc4a610a64882f576cecff4f73fb9022e19083664e8c3383d75b25e40bc77a"
)
DD_MERGER_EVENT_ID = (
    "7ad3b0a7ccdec1034fb7bd56914e8ad20b30d13045831ec776236626db8342c2"
)
DD_MERGER_SOURCE_HASH = (
    "098828aa2714df3fdd52a18b1fffb91d6a72865ff8dd4e94e84f7bc079cf0e64"
)
BBT_TICKER_EVENT_ID = (
    "bba9b3139a40f93f1b90790cdbded3fe0db106526ca9890f1c611c67ee267131"
)
BBT_TICKER_SOURCE_HASH = (
    "094d8ee3bb3b2cdf33ff4492d6cc93a3738098e4d8bfbeb0c7962d8d9bc3208d"
)


@dataclass(frozen=True)
class PriceTarget:
    symbol: str
    security_id: str
    end: str
    expected_rows: int
    full_rows: int
    full_lines_sha256: str
    price_source_hash: str
    relation_sha256: str
    signal_sha256: str
    expected_exact: Mapping[str, int]
    expected_max_abs: Mapping[str, float]
    expected_max_relative: Mapping[str, float]
    expected_volume_exact: int
    expected_volume_median_abs: float
    expected_volume_max_abs: float
    expected_return_correlation: float
    identity_source: str
    identity_source_hash: str
    provider_symbol: str


TARGETS = (
    PriceTarget(
        symbol="BBBY",
        security_id=BBBY_ID,
        end="2018-03-07",
        expected_rows=650,
        full_rows=6_337,
        full_lines_sha256=(
            "ee5236d76488c012a44c701d3c49c3d8b5dc290a9932b08f6d868e5c60d52b51"
        ),
        price_source_hash=(
            "84083e814e2fafef9d71866315ef17eae16f26e4f3ced9d799637eef6b362825"
        ),
        relation_sha256=(
            "fecd59e84360bd2173ab0e0d30d190731cfb9f97d1a3ef3dae81292783e0ab1a"
        ),
        signal_sha256=(
            "2d964d020e6707aff59cd96ee362b131f99a68ef4ccdf2758f5e369b0f744ec2"
        ),
        expected_exact={"open": 648, "high": 575, "low": 579, "close": 637},
        expected_max_abs={"open": 0.05, "high": 0.005, "low": 0.005, "close": 0.01},
        expected_max_relative={
            "open": 0.0012553351744916963,
            "high": 0.00022815423226112518,
            "low": 0.00016580998176086902,
            "close": 0.00034223134839144457,
        },
        expected_volume_exact=1,
        expected_volume_median_abs=37.0,
        expected_volume_max_abs=7_288_748.0,
        expected_return_correlation=0.9999990453003861,
        identity_source="bbby_identity_repair",
        identity_source_hash=(
            "7c920e8e63a7ffacfa1b4896b08d2b93d2429c9c9c56bb2afa92bbb4d00a35b6"
        ),
        provider_symbol="BBBY_old.US",
    ),
    PriceTarget(
        symbol="BBT",
        security_id=BBT_ID,
        end="2018-03-27",
        expected_rows=813,
        full_rows=7_056,
        full_lines_sha256=(
            "16582afee722288d3227dd2d1c8a14884473366cb4ddefd71724d8c722625ec3"
        ),
        price_source_hash=(
            "4e6c9fe8f35b4e333e289791d8d89ac85b27f17f0d227b69d397249fb74d57cd"
        ),
        relation_sha256=(
            "ed5375f9a9e4e5e83db239bc4c5e7d70af86cfd75ed703331d55efabb3c5770c"
        ),
        signal_sha256=(
            "d55ecc680990a769157863997388ba4f88b71a297157f5070ae23e1241301853"
        ),
        expected_exact={"open": 756, "high": 805, "low": 805, "close": 811},
        expected_max_abs={"open": 0.32, "high": 0.0005, "low": 0.0005, "close": 0.015},
        expected_max_relative={
            "open": 0.005844748858447494,
            "high": 0.000013640145676627118,
            "low": 0.000012381600941060792,
            "close": 0.00031403747513871177,
        },
        expected_volume_exact=252,
        expected_volume_median_abs=211_365.0,
        expected_volume_max_abs=3_536_390.0,
        expected_return_correlation=0.9999989350223387,
        identity_source="eodhd_exchange_symbols",
        identity_source_hash=(
            "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99"
        ),
        provider_symbol="BBT_old.US",
    ),
)


@dataclass(frozen=True)
class EvidencePins:
    zip_sha256: str
    zip_size: int
    member_sha256: str
    member_size: int
    member_crc32: int
    combined_extract_sha256: str
    combined_extract_size: int
    combined_extract_lines: int
    metadata_sha256: str
    metadata_size: int
    metadata_id: int
    metadata_ref: str
    metadata_version: int
    metadata_last_updated: str
    metadata_total_bytes: int
    dd_segment_sha256: str = DD_SEGMENT_HASH
    dd_segment_rows: int = 672
    enforce_reviewed_profile: bool = True


DEFAULT_PINS = EvidencePins(
    zip_sha256=(
        "36c667bbecf42c43e5b9e8e4e5d9a1268522705bc40a4bac9671d62c1a20cbae"
    ),
    zip_size=463_184_323,
    member_sha256=(
        "ca7fb174c7948db85638917d25ff65d438e27d5cb23675da784c54db01e3d003"
    ),
    member_size=1_797_003_576,
    member_crc32=0x946874CE,
    combined_extract_sha256=(
        "a6a6f651265825ed9ed95a1dfb9889f70586a728aa53eeae8585b8c00e4af52f"
    ),
    combined_extract_size=186_580,
    combined_extract_lines=1_464,
    metadata_sha256=(
        "e83992cf9a4051e35f91e717616b5005c04deb4f290d366679e67b235cd9401b"
    ),
    metadata_size=4_683,
    metadata_id=1_907_403,
    metadata_ref="marketneutral/quandl-wiki-prices-us-equites",
    metadata_version=1,
    metadata_last_updated="2022-02-02T15:00:57.923Z",
    metadata_total_bytes=1_797_003_576,
)


@dataclass(frozen=True)
class ArchiveArtifact:
    dataset: str
    source: str
    source_url: str
    content_type: str
    extension: str
    payload: bytes
    retrieved_at: str

    @property
    def source_hash(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    def object_path(self, completed_session: str) -> str:
        return f"archives/{completed_session}/{self.source_hash}.{self.extension}.gz"


@dataclass(frozen=True)
class EvidenceBundle:
    extract: ArchiveArtifact
    rows: Mapping[str, pd.DataFrame]
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etag: str | None
    frame: pd.DataFrame
    artifacts: tuple[ArchiveArtifact, ...]
    pins: EvidencePins
    wiki_zip_path: Path
    kaggle_metadata_path: Path
    targets: tuple[PriceTarget, ...]
    allowed_index_identity_gap_fingerprints: tuple[str, ...]
    summary: Mapping[str, Any]


FailureInjector = Callable[[str], None]


def _noop_injector(_stage: str) -> None:
    return None


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _date(value: Any) -> str:
    parsed = pd.to_datetime(_text(value), errors="coerce")
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _one(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"{label} inventory changed: expected=1; observed={len(rows)}.")
    return rows.iloc[0]


def _safe_path(root: Path, object_path: str) -> Path:
    base = root.resolve()
    path = (base / object_path).resolve()
    if path == base or base not in path.parents:
        raise ValueError(f"Archive path escapes repository: {object_path}.")
    return path


def _read_archived_payload(repository: LocalDatasetRepository, row: Mapping[str, Any]) -> bytes:
    path = _safe_path(repository.root, _text(row.get("object_path")))
    if not path.is_file():
        raise FileNotFoundError(f"Archived payload is missing: {path}.")
    try:
        payload = gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise ValueError(f"Archived payload is not valid gzip: {path}.") from exc
    observed = hashlib.sha256(payload).hexdigest()
    expected = _text(row.get("source_hash"))
    if observed != expected:
        raise ValueError(
            f"Archived payload hash changed: expected={expected}; observed={observed}."
        )
    return payload


def load_evidence_bundle(
    wiki_zip_path: Path,
    kaggle_metadata_path: Path,
    *,
    pins: EvidencePins = DEFAULT_PINS,
    targets: Sequence[PriceTarget] = TARGETS,
) -> EvidenceBundle:
    """Verify frozen bytes and return the minimal header+BBBY+BBT extract."""

    if not wiki_zip_path.is_file():
        raise FileNotFoundError(f"Frozen WIKI ZIP is missing: {wiki_zip_path}.")
    if wiki_zip_path.stat().st_size != pins.zip_size:
        raise ValueError("Frozen WIKI ZIP size changed.")
    if _sha256_file(wiki_zip_path) != pins.zip_sha256:
        raise ValueError("Frozen WIKI ZIP hash changed.")

    target_by_symbol = {target.symbol: target for target in targets}
    full_digests = {symbol: hashlib.sha256() for symbol in target_by_symbol}
    full_counts = {symbol: 0 for symbol in target_by_symbol}
    extract_lines: list[bytes] = []
    member_digest = hashlib.sha256()
    with zipfile.ZipFile(wiki_zip_path) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != WIKI_MEMBER:
            raise ValueError("Frozen WIKI ZIP member inventory changed.")
        info = infos[0]
        if info.file_size != pins.member_size or info.CRC != pins.member_crc32:
            raise ValueError("Frozen WIKI member size/CRC changed.")
        with archive.open(info, "r") as member:
            for line_number, line in enumerate(member, start=1):
                member_digest.update(line)
                if line_number == 1:
                    if not line.startswith(b"ticker,date,open,high,low,close,volume,"):
                        raise ValueError("Frozen WIKI header changed.")
                    extract_lines.append(line)
                    continue
                fields = line.split(b",", 2)
                if len(fields) < 3:
                    continue
                symbol = fields[0].decode("ascii")
                target = target_by_symbol.get(symbol)
                if target is None:
                    continue
                full_counts[symbol] += 1
                full_digests[symbol].update(line)
                session = fields[1].decode("ascii")
                if "2015-01-02" <= session <= target.end:
                    extract_lines.append(line)

    if member_digest.hexdigest() != pins.member_sha256:
        raise ValueError("Frozen WIKI CSV member hash changed.")
    if pins.enforce_reviewed_profile:
        for target in targets:
            if (
                full_counts[target.symbol] != target.full_rows
                or full_digests[target.symbol].hexdigest() != target.full_lines_sha256
            ):
                raise ValueError(f"Frozen WIKI full {target.symbol} inventory changed.")

    extract = b"".join(extract_lines)
    if (
        hashlib.sha256(extract).hexdigest() != pins.combined_extract_sha256
        or len(extract) != pins.combined_extract_size
        or len(extract_lines) != pins.combined_extract_lines
    ):
        raise ValueError("Frozen WIKI BBBY/BBT extract pin changed.")
    parsed = pd.read_csv(io.BytesIO(extract))
    rows: dict[str, pd.DataFrame] = {}
    for target in targets:
        selected = parsed.loc[parsed["ticker"].astype(str).eq(target.symbol)].copy()
        selected["date"] = selected["date"].astype(str)
        if selected["date"].duplicated().any():
            raise ValueError(f"Frozen WIKI {target.symbol} dates are duplicated.")
        if pins.enforce_reviewed_profile and (
            len(selected) != target.expected_rows
            or selected["date"].min() != "2015-01-02"
            or selected["date"].max() != target.end
        ):
            raise ValueError(f"Frozen WIKI {target.symbol} overlap topology changed.")
        rows[target.symbol] = selected.reset_index(drop=True)

    metadata_bytes = kaggle_metadata_path.read_bytes()
    if (
        len(metadata_bytes) != pins.metadata_size
        or hashlib.sha256(metadata_bytes).hexdigest() != pins.metadata_sha256
    ):
        raise ValueError("Frozen Kaggle metadata bytes changed.")
    try:
        metadata = json.loads(metadata_bytes)
    except ValueError as exc:
        raise ValueError("Frozen Kaggle metadata is invalid JSON.") from exc
    versions = metadata.get("versions") or []
    observed_version = int(versions[0].get("versionNumber")) if len(versions) == 1 else -1
    expected = {
        "id": pins.metadata_id,
        "ref": pins.metadata_ref,
        "licenseName": "Unknown",
        "lastUpdated": pins.metadata_last_updated,
        "totalBytes": pins.metadata_total_bytes,
    }
    changed = [key for key, value in expected.items() if metadata.get(key) != value]
    if changed or observed_version != pins.metadata_version:
        raise ValueError("Frozen Kaggle identity/license metadata changed.")

    artifact = ArchiveArtifact(
        dataset="kaggle_quandl_wiki_bbby_bbt_price_extract",
        source="kaggle_quandl_wiki_bbby_bbt_price_extract",
        source_url=WIKI_DOWNLOAD_URL,
        content_type="text/csv",
        extension="csv",
        payload=extract,
        retrieved_at=WIKI_RETRIEVED_AT,
    )
    return EvidenceBundle(
        extract=artifact,
        rows=rows,
        audit={
            "zip_sha256": pins.zip_sha256,
            "zip_size": pins.zip_size,
            "member_name": WIKI_MEMBER,
            "member_sha256": pins.member_sha256,
            "member_size": pins.member_size,
            "member_crc32": f"{pins.member_crc32:08x}",
            "extract_sha256": artifact.source_hash,
            "extract_size": len(extract),
            "extract_line_count": len(extract_lines),
            "metadata_sha256": pins.metadata_sha256,
            "metadata_license_name": "Unknown",
        },
    )


def _relation_fingerprint(joined: pd.DataFrame) -> str:
    records: list[list[str]] = []
    for row in joined.itertuples(index=False):
        records.append(
            [str(row.date)]
            + [
                format(float(getattr(row, f"{column}_eod")), ".17g")
                for column in ("open", "high", "low", "close", "volume")
            ]
            + [
                format(float(getattr(row, f"{column}_wiki")), ".17g")
                for column in ("open", "high", "low", "close", "volume")
            ]
        )
    payload = (json.dumps(records, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _triple_signal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = pd.DataFrame(
        {
            "Date": pd.to_datetime(frame["session"], errors="raise"),
            "Open": pd.to_numeric(frame["open"], errors="raise"),
            "High": pd.to_numeric(frame["high"], errors="raise"),
            "Low": pd.to_numeric(frame["low"], errors="raise"),
            "Close": pd.to_numeric(frame["close"], errors="raise"),
            "Volume": pd.to_numeric(frame["volume"], errors="raise"),
        }
    )
    return add_triple_supertrend(
        prepared,
        settings=((10, 1.0), (11, 2.0), (12, 3.0)),
        atr_method="wilder",
        exit_down_count=2,
    )


SIGNAL_COLUMNS = (
    "TripleST1_Trend",
    "TripleST2_Trend",
    "TripleST3_Trend",
    "TripleAllUp",
    "TripleDownCount",
    "TripleBuySignal",
    "TripleSellSignal",
)


def _signal_hash(frame: pd.DataFrame) -> str:
    records: list[list[Any]] = []
    for session, values in zip(
        frame["Date"].dt.date.astype(str),
        frame[list(SIGNAL_COLUMNS)].itertuples(index=False, name=None),
        strict=True,
    ):
        normalized: list[Any] = []
        for value in values:
            if isinstance(value, (bool, np.bool_)):
                normalized.append(bool(value))
            else:
                normalized.append(int(value))
        records.append([session, *normalized])
    payload = (json.dumps(records, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _identity_audit(
    target: PriceTarget,
    master: pd.DataFrame,
    history: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    enforce_reviewed_profile: bool,
) -> Mapping[str, Any]:
    master_row = _one(
        master,
        master["security_id"].astype(str).eq(target.security_id),
        f"{target.symbol} security_master",
    )
    history_row = _one(
        history,
        history["security_id"].astype(str).eq(target.security_id)
        & history["symbol"].astype(str).str.upper().eq(target.symbol),
        f"{target.symbol} symbol_history",
    )
    if enforce_reviewed_profile:
        expected_end = "2023-05-02" if target.symbol == "BBBY" else "2019-12-06"
        expected = {
            "primary_symbol": target.symbol,
            "provider_symbol": target.provider_symbol,
            "active_from": "2015-01-02",
            "active_to": expected_end,
            "source": target.identity_source,
            "source_hash": target.identity_source_hash,
        }
        changed = [key for key, value in expected.items() if _text(master_row.get(key)) != value]
        if changed:
            raise ValueError(
                f"{target.symbol} identity pin changed: {', '.join(changed)}."
            )
        if (
            _date(history_row.get("effective_from")) != "2015-01-01"
            or _text(history_row.get("source")) != target.identity_source
            or _text(history_row.get("source_hash")) != target.identity_source_hash
        ):
            raise ValueError(f"{target.symbol} symbol-history identity pin changed.")

    result: dict[str, Any] = {
        "security_id": target.security_id,
        "symbol": target.symbol,
        "master_source": _text(master_row.get("source")),
        "master_source_hash": _text(master_row.get("source_hash")),
        "provider_symbol": _text(master_row.get("provider_symbol")),
        "identity_mutated": False,
        "yahoo_symbol_only_identity_reuse_allowed": False,
    }
    if target.symbol == "BBT":
        ticker = _one(
            actions,
            actions["event_id"].astype(str).eq(BBT_TICKER_EVENT_ID),
            "BBT official ticker-change action",
        )
        if enforce_reviewed_profile and not (
            _text(ticker.get("security_id")) == BBT_ID
            and _text(ticker.get("action_type")) == "ticker_change"
            and _date(ticker.get("effective_date")) == "2019-12-06"
            and _text(ticker.get("new_security_id")) == TFC_ID
            and _text(ticker.get("new_symbol")) == "TFC"
            and bool(ticker.get("official"))
            and _text(ticker.get("source_hash")) == BBT_TICKER_SOURCE_HASH
        ):
            raise ValueError("BBT -> TFC official identity boundary changed.")
        result["official_terminal_identity_event_id"] = _text(ticker.get("event_id"))
        result["official_terminal_identity_source_hash"] = _text(ticker.get("source_hash"))
    return result


def _action_coverage_audit(
    target: PriceTarget,
    wiki: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    enforce_reviewed_profile: bool,
) -> Mapping[str, Any]:
    wiki_amounts = pd.to_numeric(wiki["ex-dividend"], errors="raise")
    wiki_events = sorted(
        (str(row.date), float(row.amount))
        for row in pd.DataFrame(
            {"date": wiki["date"].astype(str), "amount": wiki_amounts}
        ).loc[wiki_amounts.gt(0)].itertuples(index=False)
    )
    current = actions.loc[
        actions["security_id"].astype(str).eq(target.security_id)
        & actions["action_type"].astype(str).eq("cash_dividend")
        & actions["effective_date"].astype(str).between("2015-01-02", target.end)
    ].copy()
    current_events = sorted(
        (str(row.effective_date), float(row.cash_amount))
        for row in current.itertuples(index=False)
    )
    missing = sorted(set(wiki_events) - set(current_events))
    extra = sorted(set(current_events) - set(wiki_events))
    if enforce_reviewed_profile:
        expected_missing = (
            []
            if target.symbol == "BBBY"
            else [
                ("2015-05-13", 0.27),
                ("2015-08-12", 0.27),
                ("2015-11-10", 0.27),
                ("2016-02-10", 0.27),
            ]
        )
        expected_extra = (
            [("2017-09-14", 0.15), ("2017-12-14", 0.15)]
            if target.symbol == "BBBY"
            else [("2018-02-08", 0.33), ("2018-03-05", 0.045)]
        )
        if missing != expected_missing or extra != expected_extra:
            raise ValueError(f"{target.symbol} action-coverage relation changed.")
    return {
        "status": "incomplete_not_rewritten",
        "wiki_dividend_event_count": len(wiki_events),
        "current_dividend_event_count_in_overlap": len(current_events),
        "wiki_events_missing_from_current": [list(value) for value in missing],
        "current_events_missing_from_wiki": [list(value) for value in extra],
        "actions_rewritten": False,
        "factors_rewritten": False,
        "price_only_pass_must_not_imply_action_factor_pass": True,
    }


def _audit_price_target(
    target: PriceTarget,
    wiki: pd.DataFrame,
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    master: pd.DataFrame,
    history: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    enforce_reviewed_profile: bool,
) -> Mapping[str, Any]:
    target_prices = prices.loc[
        prices["security_id"].astype(str).eq(target.security_id)
    ].copy()
    target_prices["date"] = pd.to_datetime(
        target_prices["session"], errors="raise"
    ).dt.date.astype(str)
    overlap = target_prices.loc[
        target_prices["date"].between("2015-01-02", target.end)
    ].copy()
    if (
        overlap["date"].duplicated().any()
        or not set(wiki["date"]).issubset(set(overlap["date"]))
    ):
        raise ValueError(f"{target.symbol} WIKI/current session inventory changed.")
    if enforce_reviewed_profile and (
        set(overlap["source"].astype(str)) != {"eodhd_eod"}
        or set(overlap["source_hash"].astype(str)) != {target.price_source_hash}
    ):
        raise ValueError(f"{target.symbol} current raw price pin changed.")
    joined = overlap.merge(
        wiki,
        on="date",
        suffixes=("_eod", "_wiki"),
        validate="one_to_one",
    ).sort_values("date", ignore_index=True)
    if enforce_reviewed_profile and len(joined) != target.expected_rows:
        raise ValueError(f"{target.symbol} WIKI overlap row count changed.")
    relation_hash = _relation_fingerprint(joined)
    if enforce_reviewed_profile and relation_hash != target.relation_sha256:
        raise ValueError(f"{target.symbol} exact price relation fingerprint changed.")

    relation: dict[str, Any] = {
        "row_count": len(joined),
        "start": str(joined["date"].min()),
        "end": str(joined["date"].max()),
        "relation_sha256": relation_hash,
    }
    for column in ("open", "high", "low", "close"):
        left = pd.to_numeric(joined[f"{column}_eod"], errors="raise")
        right = pd.to_numeric(joined[f"{column}_wiki"], errors="raise")
        delta = (left - right).abs()
        relative = delta / right.abs()
        exact = int(np.isclose(delta, 0.0, rtol=0.0, atol=1e-12).sum())
        maximum = float(delta.max())
        maximum_relative = float(relative.max())
        relation[f"{column}_exact"] = exact
        relation[f"{column}_max_abs_difference"] = maximum
        relation[f"{column}_max_relative_difference"] = maximum_relative
        if enforce_reviewed_profile and (
            exact != target.expected_exact[column]
            or not math.isclose(
                maximum,
                target.expected_max_abs[column],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                maximum_relative,
                target.expected_max_relative[column],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError(f"{target.symbol} {column} relation changed.")
    eod_close = pd.to_numeric(joined["close_eod"], errors="raise")
    wiki_close = pd.to_numeric(joined["close_wiki"], errors="raise")
    return_correlation = float(eod_close.pct_change().corr(wiki_close.pct_change()))
    eod_volume = pd.to_numeric(joined["volume_eod"], errors="raise")
    wiki_volume = pd.to_numeric(joined["volume_wiki"], errors="raise")
    volume_delta = (eod_volume - wiki_volume).abs()
    relation.update(
        {
            "close_return_correlation": return_correlation,
            "volume_exact": int(np.isclose(volume_delta, 0.0, rtol=0.0, atol=0.0).sum()),
            "volume_median_abs_difference": float(volume_delta.median()),
            "volume_max_abs_difference": float(volume_delta.max()),
            "raw_volume_used_for_price_pass": False,
        }
    )
    if enforce_reviewed_profile and (
        not math.isclose(
            return_correlation,
            target.expected_return_correlation,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or relation["volume_exact"] != target.expected_volume_exact
        or relation["volume_median_abs_difference"] != target.expected_volume_median_abs
        or relation["volume_max_abs_difference"] != target.expected_volume_max_abs
    ):
        raise ValueError(f"{target.symbol} close-return/volume relation changed.")

    target_factors = factors.loc[
        factors["security_id"].astype(str).eq(target.security_id)
    ].copy()
    if set(pd.to_numeric(target_factors["split_factor"], errors="raise")) != {1.0}:
        raise ValueError(f"{target.symbol} split-factor economics changed.")
    candidate = target_prices.drop(columns="date").copy()
    candidate["_date"] = pd.to_datetime(candidate["session"]).dt.date.astype(str)
    by_date = wiki.set_index("date")
    replace_mask = candidate["_date"].isin(by_date.index)
    for column in ("open", "high", "low", "close", "volume"):
        candidate.loc[replace_mask, column] = candidate.loc[replace_mask, "_date"].map(
            pd.to_numeric(by_date[column], errors="raise")
        )
    candidate = candidate.drop(columns="_date")
    current_adjusted = apply_adjustment_factors(
        target_prices.drop(columns="date"), target_factors, mode="total_return_adjusted"
    ).sort_values("session", ignore_index=True)
    substituted_adjusted = apply_adjustment_factors(
        candidate, target_factors, mode="total_return_adjusted"
    ).sort_values("session", ignore_index=True)
    adjusted_sensitivity: dict[str, Any] = {}
    for column in ("open", "high", "low", "close", "volume"):
        delta = (
            pd.to_numeric(current_adjusted[column], errors="raise")
            - pd.to_numeric(substituted_adjusted[column], errors="raise")
        ).abs()
        adjusted_sensitivity[f"{column}_changed_rows"] = int(delta.gt(1e-12).sum())
        adjusted_sensitivity[f"{column}_max_abs_difference"] = float(delta.max())
    current_signals = _triple_signal_frame(current_adjusted)
    substituted_signals = _triple_signal_frame(substituted_adjusted)
    signal_differences = {
        column: int((~current_signals[column].eq(substituted_signals[column])).sum())
        for column in SIGNAL_COLUMNS
    }
    current_signal_hash = _signal_hash(current_signals)
    substituted_signal_hash = _signal_hash(substituted_signals)
    if any(signal_differences.values()) or current_signal_hash != substituted_signal_hash:
        raise ValueError(f"{target.symbol} WIKI substitution changed Triple Supertrend.")
    if enforce_reviewed_profile and current_signal_hash != target.signal_sha256:
        raise ValueError(f"{target.symbol} Triple Supertrend fingerprint changed.")

    identity = _identity_audit(
        target,
        master,
        history,
        actions,
        enforce_reviewed_profile=enforce_reviewed_profile,
    )
    action_coverage = _action_coverage_audit(
        target,
        wiki,
        actions,
        enforce_reviewed_profile=enforce_reviewed_profile,
    )
    return {
        "status": "passed_price_only_arbitration",
        "symbol": target.symbol,
        "security_id": target.security_id,
        "raw_price_relation": relation,
        "identity": identity,
        "action_factor_coverage": action_coverage,
        "wiki_raw_substitution_sensitivity": {
            "adjusted_ohlcv": adjusted_sensitivity,
            "triple_supertrend_field_differences": signal_differences,
            "current_signal_sha256": current_signal_hash,
            "substituted_signal_sha256": substituted_signal_hash,
        },
        "raw_price_rewritten": False,
        "corporate_actions_rewritten": False,
        "adjustment_factors_rewritten": False,
        "adjusted_ohlc_changed_by_archive_operation": False,
        "triple_supertrend_changed_by_archive_operation": False,
    }


WIKI_COLUMNS = (
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ex-dividend",
    "split_ratio",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "adj_volume",
)


def _audit_dd_blocked(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    master: pd.DataFrame,
    history: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    pins: EvidencePins,
) -> Mapping[str, Any]:
    segment_row = _one(
        archive,
        archive["archive_id"].astype(str).eq(pins.dd_segment_sha256),
        "legacy DD frozen-WIKI segment",
    )
    segment_payload = _read_archived_payload(repository, segment_row)
    if hashlib.sha256(segment_payload).hexdigest() != pins.dd_segment_sha256:
        raise ValueError("Legacy DD segment hash changed.")
    dd_wiki = pd.read_csv(
        io.BytesIO(segment_payload), header=None, names=list(WIKI_COLUMNS)
    )
    dd_wiki["date"] = dd_wiki["date"].astype(str)
    if (
        dd_wiki["date"].duplicated().any()
        or set(dd_wiki["ticker"].astype(str)) != {"DD"}
        or (pins.enforce_reviewed_profile and len(dd_wiki) != pins.dd_segment_rows)
    ):
        raise ValueError("Legacy DD frozen-WIKI segment topology changed.")

    distributions = dd_wiki.loc[
        pd.to_numeric(dd_wiki["ex-dividend"], errors="raise").gt(0),
        ["date", "ex-dividend"],
    ].copy()
    distribution_events = sorted(
        (str(row.date), float(row.amount))
        for row in distributions.rename(columns={"ex-dividend": "amount"}).itertuples(
            index=False
        )
    )
    spin_proxy = [value for value in distribution_events if value[0] == "2015-07-01"]
    if pins.enforce_reviewed_profile and not (
        len(distribution_events) == 12 and spin_proxy == [("2015-07-01", 3.2)]
    ):
        raise ValueError("Legacy DD WIKI distribution inventory changed.")

    master_row = _one(
        master,
        master["security_id"].astype(str).eq(LEGACY_DD_ID),
        "legacy DD security_master",
    )
    history_row = _one(
        history,
        history["security_id"].astype(str).eq(LEGACY_DD_ID)
        & history["symbol"].astype(str).str.upper().eq("DD"),
        "legacy DD symbol_history",
    )
    legacy_actions = actions.loc[
        actions["security_id"].astype(str).eq(LEGACY_DD_ID)
    ].copy()
    merger = _one(
        legacy_actions,
        legacy_actions["event_id"].astype(str).eq(DD_MERGER_EVENT_ID),
        "legacy DD official merger",
    )
    if pins.enforce_reviewed_profile and not (
        len(legacy_actions) == 1
        and _text(master_row.get("primary_symbol")) == "DD"
        and _text(master_row.get("provider_symbol")) == "DD_old.US"
        and _date(master_row.get("active_from")) == "2015-01-02"
        and _date(master_row.get("active_to")) == "2017-08-31"
        and _text(history_row.get("source_hash")) == DD_MERGER_SOURCE_HASH
        and _text(merger.get("action_type")) == "stock_merger"
        and _date(merger.get("effective_date")) == "2017-09-01"
        and math.isclose(float(merger.get("ratio")), 1.282, abs_tol=1e-15)
        and _text(merger.get("new_security_id")) == DWDP_ID
        and _text(merger.get("source_hash")) == DD_MERGER_SOURCE_HASH
    ):
        raise ValueError("Legacy DD official identity/action pin changed.")

    master_text = master.astype(str)
    chemours_master = master_text.apply(
        lambda column: column.str.contains("chemours", case=False, regex=False).any(),
        axis=1,
    )
    if "primary_symbol" in master:
        chemours_master |= master["primary_symbol"].astype(str).str.upper().eq("CC")
    history_cc = history["symbol"].astype(str).str.upper().eq("CC")
    archive_cc = archive.astype(str).apply(
        lambda column: column.str.contains("chemours|/CC.US|CC_old.US", case=False, regex=True).any(),
        axis=1,
    )
    if chemours_master.any() or history_cc.any() or archive_cc.any():
        raise ValueError("An unreviewed local Chemours identity/evidence path appeared.")

    dd_prices = prices.loc[
        prices["security_id"].astype(str).eq(LEGACY_DD_ID)
    ].copy().sort_values("session", ignore_index=True)
    dd_factors = factors.loc[
        factors["security_id"].astype(str).eq(LEGACY_DD_ID)
    ].copy().sort_values("session", ignore_index=True)
    if (
        len(dd_prices) != len(dd_wiki)
        or set(pd.to_numeric(dd_factors["split_factor"], errors="raise")) != {1.0}
        or set(pd.to_numeric(dd_factors["total_return_factor"], errors="raise")) != {1.0}
    ):
        raise ValueError("Legacy DD current factor topology changed.")
    current = apply_adjustment_factors(
        dd_prices, dd_factors, mode="total_return_adjusted"
    ).sort_values("session", ignore_index=True)
    proxy_factors = dd_factors.copy()
    exact_wiki_factor = (
        pd.to_numeric(dd_wiki["adj_close"], errors="raise")
        / pd.to_numeric(dd_wiki["close"], errors="raise")
    )
    by_date = pd.Series(exact_wiki_factor.to_numpy(), index=dd_wiki["date"])
    proxy_dates = pd.to_datetime(proxy_factors["session"]).dt.date.astype(str)
    if not proxy_dates.isin(by_date.index).all():
        raise ValueError("Legacy DD proxy-factor date mapping changed.")
    proxy_factors["total_return_factor"] = proxy_dates.map(by_date)
    proxy = apply_adjustment_factors(
        dd_prices, proxy_factors, mode="total_return_adjusted"
    ).sort_values("session", ignore_index=True)
    maximum_wiki_residual = 0.0
    for column in ("open", "high", "low", "close"):
        residual = (
            pd.to_numeric(proxy[column], errors="raise")
            - pd.to_numeric(dd_wiki[f"adj_{column}"], errors="raise")
        ).abs()
        maximum_wiki_residual = max(maximum_wiki_residual, float(residual.max()))
    current_signals = _triple_signal_frame(current)
    proxy_signals = _triple_signal_frame(proxy)
    differences = {
        column: int((~current_signals[column].eq(proxy_signals[column])).sum())
        for column in SIGNAL_COLUMNS
    }
    current_hash = _signal_hash(current_signals)
    proxy_hash = _signal_hash(proxy_signals)
    if pins.enforce_reviewed_profile:
        expected_differences = {
            "TripleST1_Trend": 4,
            "TripleST2_Trend": 0,
            "TripleST3_Trend": 0,
            "TripleAllUp": 0,
            "TripleDownCount": 4,
            "TripleBuySignal": 0,
            "TripleSellSignal": 2,
        }
        if (
            differences != expected_differences
            or current_hash
            != "38c920a8e9efbd9efce4ee4600c30df4316c64f8dfd9f4eb8f96c3000fef63be"
            or proxy_hash
            != "d0276d5bbb381e809a67c9174e985ae902581f02b0d2d77714d0fa026f80fc78"
            or maximum_wiki_residual > 1e-10
        ):
            raise ValueError("Legacy DD proxy-only impact fingerprint changed.")

    return {
        "status": "blocked_fail_closed",
        "security_id": LEGACY_DD_ID,
        "identity_source_hash": _text(master_row.get("source_hash")),
        "current_action_count": len(legacy_actions),
        "current_cash_distribution_count": int(
            legacy_actions["action_type"].astype(str).isin(
                {"cash_dividend", "special_dividend"}
            ).sum()
        ),
        "wiki_distribution_events": [list(value) for value in distribution_events],
        "wiki_2015_07_01_value": 3.2,
        "wiki_2015_07_01_is_cash_dividend": False,
        "known_official_terms_discovery_only_not_hash_pinned": {
            "distribution_date": "2015-07-01",
            "ratio": "1 CC share per 5 legacy DD shares",
            "suggested_child_value_per_dd_share": 3.242,
            "suggested_parent_basis_fraction": 0.94915,
            "suggested_child_basis_fraction": 0.05085,
            "sec_completion_url": (
                "https://www.sec.gov/Archives/edgar/data/30554/"
                "000003055415000065/exhibit991pressrelease.htm"
            ),
            "issuer_form_8937_url": (
                "https://s23.q4cdn.com/116192123/files/doc_downloads/"
                "Tax-Cost-Basis-Allocation.pdf"
            ),
        },
        "blocking_reasons": [
            "official_2015_spinoff_bytes_not_local_hash_pinned",
            "chemours_child_security_id_absent",
            "complete_identity_bound_chemours_price_path_absent",
            "rounded_wiki_3_2_cannot_be_booked_as_cash_dividend",
            "ordinary_dividend_exact_ex_dates_not_officially_hash_pinned",
        ],
        "proxy_only_sensitivity_not_applied": {
            "method": "frozen_wiki_adj_close_divided_by_close",
            "adjusted_ohlc_vs_wiki_max_abs_difference": maximum_wiki_residual,
            "triple_supertrend_field_differences": differences,
            "current_signal_sha256": current_hash,
            "proxy_signal_sha256": proxy_hash,
        },
        "raw_price_rewritten": False,
        "corporate_actions_rewritten": False,
        "adjustment_factors_rewritten": False,
        "identity_rewritten": False,
        "apply_allowed": False,
    }


def _verify_existing_metadata(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    pins: EvidencePins,
) -> None:
    row = _one(
        archive,
        archive["archive_id"].astype(str).eq(pins.metadata_sha256),
        "Kaggle raw metadata archive",
    )
    expected = {
        "dataset": "kaggle_dataset_metadata",
        "source": "kaggle_dataset_metadata",
        "source_hash": pins.metadata_sha256,
        "content_type": "application/json",
    }
    if any(_text(row.get(key)) != value for key, value in expected.items()):
        raise ValueError("Kaggle metadata archive row changed.")
    payload = _read_archived_payload(repository, row)
    if len(payload) != pins.metadata_size:
        raise ValueError("Archived Kaggle metadata size changed.")


def _provenance_artifact(
    evidence: EvidenceBundle,
    price_audits: Sequence[Mapping[str, Any]],
    dd_audit: Mapping[str, Any],
) -> ArchiveArtifact:
    payload = _canonical_json(
        {
            "schema": "us_wiki_price_arbitration/v1",
            "reviewed_at": REVIEWED_AT,
            "scope": {
                "passed_price_only": [BBBY_ID, BBT_ID],
                "blocked": [LEGACY_DD_ID],
                "write_dataset": DATASET,
                "non_write_datasets": [
                    "daily_price_raw",
                    "corporate_actions",
                    "adjustment_factors",
                    "security_master",
                    "symbol_history",
                    "index_constituent_anchors",
                    "index_membership_events",
                ],
            },
            "frozen_evidence": dict(evidence.audit),
            "price_arbitrations": list(price_audits),
            "legacy_dd": dict(dd_audit),
            "license_policy": {
                "formal_license_name": "Unknown",
                "allowed_scope": "private_internal_only",
                "redistribution_allowed": False,
                "public_publication_allowed": False,
                "local_apply_ack_required": True,
                "private_r2_publisher_ack_required_separately": True,
                "fail_closed": True,
            },
        }
    )
    return ArchiveArtifact(
        dataset="reviewed_us_wiki_price_arbitration",
        source="reviewed_us_wiki_price_arbitration",
        source_url=WIKI_DOWNLOAD_URL,
        content_type="application/json",
        extension="json",
        payload=payload,
        retrieved_at=REVIEWED_AT,
    )


def _artifact_row(
    artifact: ArchiveArtifact,
    *,
    completed_session: str,
    columns: Sequence[str],
) -> dict[str, Any]:
    values = {
        "archive_id": artifact.source_hash,
        "dataset": artifact.dataset,
        "object_path": artifact.object_path(completed_session),
        "content_type": artifact.content_type,
        "effective_date": completed_session,
        "source": artifact.source,
        "retrieved_at": artifact.retrieved_at,
        "source_hash": artifact.source_hash,
        "source_url": artifact.source_url,
    }
    return {column: values.get(column, np.nan) for column in columns}


def _verify_artifact_row(
    repository: LocalDatasetRepository,
    row: Mapping[str, Any],
    artifact: ArchiveArtifact,
    *,
    completed_session: str,
) -> None:
    expected = {
        "archive_id": artifact.source_hash,
        "dataset": artifact.dataset,
        "object_path": artifact.object_path(completed_session),
        "content_type": artifact.content_type,
        "effective_date": completed_session,
        "source": artifact.source,
        "retrieved_at": artifact.retrieved_at,
        "source_hash": artifact.source_hash,
        "source_url": artifact.source_url,
    }
    changed = [
        key
        for key, value in expected.items()
        if (_date(row.get(key)) if key == "effective_date" else _text(row.get(key)))
        != value
    ]
    if changed:
        raise ValueError(
            f"Existing arbitration artifact row changed: {', '.join(changed)}."
        )
    if _read_archived_payload(repository, row) != artifact.payload:
        raise ValueError("Existing arbitration artifact payload bytes changed.")


def _append_or_verify_artifacts(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    artifacts: Sequence[ArchiveArtifact],
    *,
    completed_session: str,
) -> tuple[pd.DataFrame, bool]:
    existence: list[bool] = []
    for artifact in artifacts:
        rows = archive.loc[
            archive["archive_id"].astype(str).eq(artifact.source_hash)
        ]
        if len(rows) > 1:
            raise ValueError("Arbitration evidence archive_id is duplicated.")
        existence.append(len(rows) == 1)
        if len(rows) == 1:
            _verify_artifact_row(
                repository,
                rows.iloc[0],
                artifact,
                completed_session=completed_session,
            )
    if any(existence) and not all(existence):
        raise ValueError("Arbitration evidence is only partially archived.")
    if all(existence):
        return archive.copy(), False
    additions = pd.DataFrame(
        [
            _artifact_row(
                artifact,
                completed_session=completed_session,
                columns=archive.columns,
            )
            for artifact in artifacts
        ],
        columns=archive.columns,
    )
    output = pd.concat([archive, additions], ignore_index=True)
    if output["archive_id"].astype(str).duplicated().any():
        raise ValueError("Arbitration source_archive candidate has duplicate IDs.")
    return output, True


class _CandidateRepository:
    def __init__(
        self,
        base: LocalDatasetRepository,
        versions: Mapping[str, str],
        source_archive: pd.DataFrame,
    ):
        self.base = base
        self.versions = dict(versions)
        self.source_archive = source_archive.copy()

    def current_manifest(self, dataset: str):
        version = self.versions.get(dataset)
        return self.base.manifest_for_version(dataset, version) if version else None

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        if dataset == DATASET:
            return self.source_archive.copy()
        version = self.versions.get(dataset)
        if not version:
            return pd.DataFrame()
        return self.base.read_frame(dataset, version)


def _load_release_frames(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> tuple[dict[str, pd.DataFrame], dict[str, str | None]]:
    frames: dict[str, pd.DataFrame] = {}
    etags: dict[str, str | None] = {}
    for dataset in REQUIRED_DATASETS:
        version = release.dataset_versions.get(dataset)
        if not version:
            raise RuntimeError(f"Current release lacks required dataset: {dataset}.")
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != version:
            raise RuntimeError(f"{dataset} release/current pointer mismatch.")
        frames[dataset] = repository.read_frame(dataset, version)
        etags[dataset] = etag
    return frames, etags


def _reviewed_inherited_identity_gaps(
    repository: LocalDatasetRepository,
) -> tuple[str, ...]:
    base = validate_repository_snapshot(repository)
    allowed = tuple(
        sorted(
            {
                fingerprint
                for issue in base.issues
                if issue.code == "index_member_missing_active_symbol"
                for fingerprint in issue.fingerprints
            }
        )
    )
    unexpected_errors = [
        issue.message
        for issue in base.issues
        if issue.severity == "error"
        and issue.code != "index_member_missing_active_symbol"
    ]
    if unexpected_errors:
        raise ValueError("; ".join(unexpected_errors))
    return allowed


def prepare_repair(
    repository: LocalDatasetRepository,
    *,
    wiki_zip_path: Path = DEFAULT_WIKI_ZIP,
    kaggle_metadata_path: Path = DEFAULT_KAGGLE_METADATA,
    pins: EvidencePins = DEFAULT_PINS,
    targets: Sequence[PriceTarget] = TARGETS,
) -> PreparedRepair:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    frames, pointer_etags = _load_release_frames(repository, release)
    allowed_identity_gaps = _reviewed_inherited_identity_gaps(repository)
    evidence = load_evidence_bundle(
        wiki_zip_path,
        kaggle_metadata_path,
        pins=pins,
        targets=targets,
    )
    archive = frames[DATASET]
    _verify_existing_metadata(repository, archive, pins=pins)
    audits = [
        _audit_price_target(
            target,
            evidence.rows[target.symbol],
            frames["daily_price_raw"],
            frames["adjustment_factors"],
            frames["security_master"],
            frames["symbol_history"],
            frames["corporate_actions"],
            enforce_reviewed_profile=pins.enforce_reviewed_profile,
        )
        for target in targets
    ]
    dd_audit = _audit_dd_blocked(
        repository,
        archive,
        frames["daily_price_raw"],
        frames["adjustment_factors"],
        frames["security_master"],
        frames["symbol_history"],
        frames["corporate_actions"],
        pins=pins,
    )
    provenance = _provenance_artifact(evidence, audits, dd_audit)
    artifacts = (evidence.extract, provenance)
    candidate, changed = _append_or_verify_artifacts(
        repository,
        archive,
        artifacts,
        completed_session=release.completed_session,
    )
    warning_present = WIKI_LICENSE_WARNING in release.warnings
    if not changed and not warning_present:
        raise RuntimeError(
            "Arbitration artifacts exist without the required Unknown-license warning."
        )
    validate_dataset(
        DATASET, candidate, completed_session=release.completed_session
    ).raise_for_errors()
    validate_repository_snapshot(
        _CandidateRepository(repository, release.dataset_versions, candidate),
        allowed_index_identity_gap_fingerprints=allowed_identity_gaps,
    ).raise_for_errors()
    status = "validated_offline_plan" if changed else "already_applied"
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etag=pointer_etags[DATASET],
        frame=candidate,
        artifacts=artifacts,
        pins=pins,
        wiki_zip_path=wiki_zip_path,
        kaggle_metadata_path=kaggle_metadata_path,
        targets=tuple(targets),
        allowed_index_identity_gap_fingerprints=allowed_identity_gaps,
        summary={
            "status": status,
            "base_release_version": release.version,
            "source_archive_base_version": release.dataset_versions[DATASET],
            "passed_price_only_security_ids": [audit["security_id"] for audit in audits],
            "price_arbitrations": audits,
            "legacy_dd": dd_audit,
            "source_archive_rows_added": 2 if changed else 0,
            "source_archive_only": True,
            "daily_price_raw_rows_changed": 0,
            "corporate_action_rows_changed": 0,
            "adjustment_factor_rows_changed": 0,
            "identity_rows_changed": 0,
            "index_rows_changed": 0,
            "license_name": "Unknown",
            "private_internal_only": True,
            "redistribution_allowed": False,
            "public_publication_allowed": False,
            "local_apply_ack_required": True,
            "private_r2_publisher_ack_required_separately": True,
            "release_warning_present": warning_present,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
            "inherited_index_identity_gap_fingerprints": list(
                allowed_identity_gaps
            ),
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
            raise RuntimeError("Unresolved WIKI arbitration recovery marker blocks writes.")
        transactions = repository.root / TRANSACTION_DIR
        if transactions.exists():
            for path in transactions.glob("*.json"):
                try:
                    status = _text(json.loads(path.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        f"Interrupted WIKI arbitration transaction blocks writes: {path}."
                    )
        yield


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(
        path,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )


def _assert_base_unchanged(
    repository: LocalDatasetRepository, prepared: PreparedRepair
) -> None:
    release, release_etag = repository.current_release()
    if (
        release is None
        or release.version != prepared.release.version
        or release_etag != prepared.release_etag
    ):
        raise RuntimeError("Current release changed after WIKI arbitration planning.")
    pointer, pointer_etag = repository.current_pointer(DATASET)
    if (
        pointer is None
        or pointer.version != prepared.release.dataset_versions[DATASET]
        or pointer_etag != prepared.pointer_etag
    ):
        raise RuntimeError("source_archive pointer changed after WIKI arbitration planning.")


def _write_artifact(repository: LocalDatasetRepository, artifact: ArchiveArtifact, session: str) -> None:
    path = _safe_path(repository.root, artifact.object_path(session))
    if path.exists():
        try:
            existing = gzip.decompress(path.read_bytes())
        except (OSError, EOFError) as exc:
            raise ValueError(f"Existing arbitration artifact is invalid gzip: {path}.") from exc
        if existing != artifact.payload:
            raise ValueError(f"Existing arbitration artifact bytes conflict: {path}.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, gzip.compress(artifact.payload, mtime=0))


def _restore_transaction(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: bytes,
    planned_version: str,
    committed_release_version: str,
    old_versions: Mapping[str, str],
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        current = repository.objects.get("releases/current.json")
        if current.data != old_release_bytes:
            observed = DataRelease.from_bytes(current.data)
            expected = {**dict(old_versions), DATASET: planned_version}
            belongs = (
                bool(committed_release_version)
                and observed.version == committed_release_version
            ) or observed.dataset_versions == expected
            if not belongs:
                raise RuntimeError(
                    f"unexpected release during rollback: {observed.version}"
                )
            repository.objects.put(
                "releases/current.json", old_release_bytes, if_match=current.etag
            )
    except Exception as exc:
        errors.append(f"releases/current.json: {type(exc).__name__}: {exc}")
    try:
        key = repository.current_key(DATASET)
        current = repository.objects.get(key)
        if current.data != old_pointer_bytes:
            observed = CurrentPointer.from_bytes(current.data)
            if observed.version != planned_version:
                raise RuntimeError(
                    f"unexpected source_archive pointer during rollback: {observed.version}"
                )
            repository.objects.put(key, old_pointer_bytes, if_match=current.etag)
    except Exception as exc:
        errors.append(f"{repository.current_key(DATASET)}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    ack_private_internal_only_local_repair: bool = False,
    inject_failure: FailureInjector = _noop_injector,
) -> dict[str, Any]:
    if prepared.summary["status"] == "already_applied":
        return {**prepared.summary, "mode": "apply", "writes_performed": False}
    if not ack_private_internal_only_local_repair:
        raise PermissionError(
            "Unknown-license WIKI evidence requires "
            "ack_private_internal_only_local_repair=True."
        )
    with _exclusive_repository_lock(repository):
        _assert_base_unchanged(repository, prepared)
        old_release = repository.objects.get("releases/current.json")
        old_pointer = repository.objects.get(repository.current_key(DATASET))
        transaction_id = uuid.uuid4().hex
        planned_version = (
            f"wiki-price-arbitration-{prepared.release.completed_session.replace('-', '')}-"
            f"{transaction_id}-{DATASET}"
        )
        journal_path = repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        journal: dict[str, Any] = {
            "schema": "us_wiki_price_arbitration_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": base64.b64encode(old_pointer.data).decode("ascii"),
            "planned_source_archive_version": planned_version,
            "local_private_internal_only_ack": True,
            "private_r2_publisher_ack": False,
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        try:
            inject_failure("after_journal")
            for artifact in prepared.artifacts:
                _write_artifact(
                    repository, artifact, prepared.release.completed_session
                )
            inject_failure("after_artifacts")
            result = repository.write_frame(
                DATASET,
                prepared.frame,
                completed_session=prepared.release.completed_session,
                metadata={
                    "operation": OPERATION,
                    "passed_price_only_security_ids": [BBBY_ID, BBT_ID],
                    "blocked_security_ids": [LEGACY_DD_ID],
                    "source_archive_rows_added": 2,
                    "license_name": "Unknown",
                    "private_internal_only": True,
                    "network_accessed": False,
                    "eodhd_calls": 0,
                    "r2_accessed": False,
                },
                expected_pointer_etag=prepared.pointer_etag,
                version=planned_version,
            )
            if result.conflict:
                raise RuntimeError(
                    f"source_archive write conflicted: {result.conflict_path}."
                )
            inject_failure("after_source_archive_write")
            versions = dict(prepared.release.dataset_versions)
            versions[DATASET] = result.manifest.version
            warnings = tuple(
                dict.fromkeys((*prepared.release.warnings, WIKI_LICENSE_WARNING))
            )
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=prepared.release.quality,
                warnings=warnings,
                expected_etag=prepared.release_etag,
            )
            for dataset, version in prepared.release.dataset_versions.items():
                if dataset != DATASET and committed.dataset_versions.get(dataset) != version:
                    raise RuntimeError(f"Non-source_archive dataset changed: {dataset}.")
            inject_failure("after_release_commit")
            validate_repository_snapshot(
                repository,
                allowed_index_identity_gap_fingerprints=(
                    prepared.allowed_index_identity_gap_fingerprints
                ),
            ).raise_for_errors()
            replay = prepare_repair(
                repository,
                wiki_zip_path=prepared.wiki_zip_path,
                kaggle_metadata_path=prepared.kaggle_metadata_path,
                pins=prepared.pins,
                targets=prepared.targets,
            )
            if replay.summary["status"] != "already_applied":
                raise RuntimeError("WIKI arbitration apply is not idempotent.")
            journal.update(
                {
                    "status": "committed",
                    "committed_release_version": committed.version,
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            return {
                **prepared.summary,
                "status": "applied",
                "mode": "apply",
                "writes_performed": True,
                "new_release_version": committed.version,
                "new_source_archive_version": result.manifest.version,
                "transaction_id": transaction_id,
                "warnings": list(committed.warnings),
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointer.data,
                planned_version=planned_version,
                committed_release_version=committed.version if committed else "",
                old_versions=prepared.release.dataset_versions,
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
                    f"WIKI arbitration rollback incomplete: {recovery}; "
                    f"errors={rollback_errors}."
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive exact frozen-WIKI BBBY/BBT price-only arbitration."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--wiki-zip", type=Path, default=DEFAULT_WIKI_ZIP)
    parser.add_argument(
        "--kaggle-metadata", type=Path, default=DEFAULT_KAGGLE_METADATA
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--ack-private-internal-only-local-repair", action="store_true"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repository = LocalDatasetRepository(args.cache_root)
    prepared = prepare_repair(
        repository,
        wiki_zip_path=args.wiki_zip,
        kaggle_metadata_path=args.kaggle_metadata,
    )
    result = (
        apply_repair(
            repository,
            prepared,
            ack_private_internal_only_local_repair=(
                args.ack_private_internal_only_local_repair
            ),
        )
        if args.apply
        else {**prepared.summary, "mode": "plan", "writes_performed": False}
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
