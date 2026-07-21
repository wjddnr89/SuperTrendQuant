#!/usr/bin/env python3
"""Plan or atomically repair the historical COL decimal-scale defect offline.

The frozen EODHD COL history is one decimal place too small before
2016-06-01.  The same response also contains a synthetic 0.1 split on that
date and six dividends that are one decimal place too small.  A frozen Quandl
WIKI mirror shows an economically continuous series through the boundary.

The repair is deliberately indivisible:

* multiply pre-2016-06-01 raw OHLC (never volume) by ten;
* multiply the six reviewed dividends by ten;
* remove the exact synthetic 0.1 split;
* rebuild the complete adjustment-factor inventory against planned price and
  action versions;
* replace the full WIKI ZIP row in ``source_archive`` with a content-addressed
  header+COL extract, raw Kaggle metadata, and a canonical provenance record.

Plan mode is read-only and is the default.  Apply has no network, EODHD, R2,
or deletion code path.  Kaggle reports the formal license as ``Unknown``;
therefore apply fails closed unless the caller explicitly acknowledges
private/internal-only use.  The old full-ZIP object is left unreferenced on
disk for rollback/audit safety and is never published by the candidate
``source_archive`` version.
"""

from __future__ import annotations

import argparse
import base64
import csv
import fcntl
import gzip
import hashlib
import io
import json
import math
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from supertrend_quant.market_store.adjustments import (
    apply_adjustment_factors,
    build_adjustment_factors,
)
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


OPERATION = "repair_us_col_scaling"
DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_WIKI_ZIP = Path("/tmp/marketneutral-quandl-wiki-prices.zip")
DEFAULT_KAGGLE_METADATA = Path("/tmp/kaggle_wiki_metadata.json")
WRITE_DATASETS = (
    "daily_price_raw",
    "corporate_actions",
    "adjustment_factors",
    "source_archive",
)
REQUIRED_DATASETS = WRITE_DATASETS
TRANSACTION_DIR = "transactions/us-col-scaling-repair"
RECOVERY_DIR = "recovery/us-col-scaling-repair"

COL_SECURITY_ID = "US:EODHD:4ddb0638-fe2a-5f9c-97c8-691e9c42d5f3"
SYMBOL = "COL"
PRE_START = "2015-01-02"
PRE_END = "2016-05-31"
BOUNDARY = "2016-06-01"
SCALE = 10.0
EXPECTED_PRE_SESSIONS = 355
REPAIR_REVIEWED_AT = "2026-07-19T02:30:00Z"

ORIGINAL_PRICE_SOURCE = "eodhd_eod"
ORIGINAL_PRICE_HASH = (
    "2634cc111dc81eba972d563bbf05c9d6b2f79ba9812cb3b0c8540d4ba9dc5b14"
)
ORIGINAL_DIVIDEND_SOURCE = "eodhd_div"
ORIGINAL_DIVIDEND_HASH = (
    "733871586ba414a4d3327929d2fc29538189942118b568b53d9bf93152a4cf43"
)
ORIGINAL_SPLIT_SOURCE = "eodhd_splits"
ORIGINAL_SPLIT_HASH = (
    "affe85f8c0d4899f37c8c7511808d80a351a482066135710c0ff457879f73cd7"
)
SPLIT_EVENT_ID = (
    "57af7cae14c8c6baa5371dafe1186ed3040700feff707fb9252e3149bd52ccfe"
)
SPLIT_URL = (
    "https://eodhd.com/api/splits/COL.US?from=2015-01-01&to=2026-07-15"
)
DIVIDEND_URL = (
    "https://eodhd.com/api/div/COL.US?from=2015-01-01&to=2026-07-15"
)

# Stable provider event IDs are retained so no downstream event reference is
# silently invalidated.  Their corrected row-level provenance and metadata
# make the reviewed transformation explicit.
DIVIDENDS: Mapping[str, tuple[str, float, float]] = {
    "19bf88297b07420e01483e0f1f92424ddd2f81af0435e61c904b9e84ceaae2fb": (
        "2015-02-12",
        0.030,
        0.300,
    ),
    "af8fb42cc42f48d8909c2ef517eee358e4f9ccc9bde420cd65426008d88e30aa": (
        "2015-05-15",
        0.033,
        0.330,
    ),
    "75c5cd6da962afb9804a2d392cc044a536f6f248e67383c68737d148d1a987e9": (
        "2015-08-13",
        0.033,
        0.330,
    ),
    "a9e64d48261fbb8519b717f17db401f72a6a7967fed01213808a0a07d4382c91": (
        "2015-11-12",
        0.033,
        0.330,
    ),
    "1d298914616dc9495d4fc29f0e42389731662eac4f4a70b93d4500cc533e7697": (
        "2016-02-11",
        0.033,
        0.330,
    ),
    "74eac9e043fff5ecdbaa831b94c6b13d475a07700bb7f599c87cbc9bd598c45b": (
        "2016-05-12",
        0.033,
        0.330,
    ),
}

REPAIR_SOURCE = "reviewed_col_scaling_repair"
REPAIR_SOURCE_KIND = "reviewed_crosscheck"
WIKI_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "marketneutral/quandl-wiki-prices-us-equites/WIKI_PRICES.csv"
)
KAGGLE_METADATA_URL = (
    "https://www.kaggle.com/api/v1/datasets/view/"
    "marketneutral/quandl-wiki-prices-us-equites"
)
WIKI_RETRIEVED_AT = "2026-07-18T03:58:26.808706Z"
WIKI_MEMBER = "WIKI_PRICES.csv"


@dataclass(frozen=True)
class EvidencePins:
    zip_sha256: str
    zip_size: int
    member_sha256: str
    member_size: int
    member_crc32: int
    full_col_lines_sha256: str
    full_col_row_count: int
    extract_sha256: str
    extract_size: int
    extract_line_count: int
    metadata_sha256: str
    metadata_size: int
    metadata_id: int
    metadata_ref: str
    metadata_version: int
    metadata_last_updated: str
    metadata_total_bytes: int
    metadata_license_name: str = "Unknown"
    original_price_hash: str = ORIGINAL_PRICE_HASH
    original_dividend_hash: str = ORIGINAL_DIVIDEND_HASH
    original_split_hash: str = ORIGINAL_SPLIT_HASH
    enforce_reviewed_relation_profile: bool = True


DEFAULT_EVIDENCE_PINS = EvidencePins(
    zip_sha256=(
        "36c667bbecf42c43e5b9e8e4e5d9a1268522705bc40a4bac9671d62c1a20cbae"
    ),
    zip_size=463_184_323,
    member_sha256=(
        "ca7fb174c7948db85638917d25ff65d438e27d5cb23675da784c54db01e3d003"
    ),
    member_size=1_797_003_576,
    member_crc32=0x946874CE,
    full_col_lines_sha256=(
        "bef8afd45e986a70c32d43aaed2b43593e5e152bf60d509f9ec224e019d11ed0"
    ),
    full_col_row_count=4_220,
    extract_sha256=(
        "7b8833e6a05fecc7a2830d86a61b29736ac455f5070bd96915ce75a080bb0327"
    ),
    extract_size=45_958,
    extract_line_count=357,
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
        return (
            f"archives/{completed_session}/{self.source_hash}."
            f"{self.extension}.gz"
        )


@dataclass(frozen=True)
class EvidenceBundle:
    extract: ArchiveArtifact
    metadata: ArchiveArtifact
    wiki_rows: pd.DataFrame
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: Mapping[str, str | None]
    planned_versions: Mapping[str, str]
    frames: Mapping[str, pd.DataFrame]
    artifacts: tuple[ArchiveArtifact, ...]
    allowed_index_identity_gap_fingerprints: tuple[str, ...]
    summary: Mapping[str, Any]
    wiki_zip_path: Path
    kaggle_metadata_path: Path
    pins: EvidencePins


FailureInjector = Callable[[str], None]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _date(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _float(value: Any, *, field: str) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        raise ValueError(f"COL {field} is not numeric.")
    return float(parsed)


def _require_hash(observed: str, expected: str, *, label: str) -> None:
    if observed != expected:
        raise ValueError(
            f"Frozen COL evidence {label} hash changed: "
            f"expected={expected}; observed={observed}."
        )


def load_evidence_bundle(
    wiki_zip_path: Path,
    kaggle_metadata_path: Path,
    *,
    pins: EvidencePins = DEFAULT_EVIDENCE_PINS,
) -> EvidenceBundle:
    """Verify the frozen files and extract only the reviewed COL evidence."""

    if not wiki_zip_path.is_file():
        raise FileNotFoundError(f"Frozen WIKI ZIP is missing: {wiki_zip_path}.")
    if wiki_zip_path.stat().st_size != pins.zip_size:
        raise ValueError("Frozen WIKI ZIP size changed.")
    _require_hash(_sha256_file(wiki_zip_path), pins.zip_sha256, label="ZIP")

    member_digest = hashlib.sha256()
    full_col_digest = hashlib.sha256()
    full_col_rows = 0
    extract_lines: list[bytes] = []
    with zipfile.ZipFile(wiki_zip_path) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != WIKI_MEMBER:
            raise ValueError("Frozen WIKI ZIP member inventory changed.")
        info = infos[0]
        if info.file_size != pins.member_size or info.CRC != pins.member_crc32:
            raise ValueError("Frozen WIKI CSV member size/CRC changed.")
        with archive.open(info, "r") as member:
            for line_number, line in enumerate(member, start=1):
                member_digest.update(line)
                if line_number == 1:
                    if not line.startswith(b"ticker,date,open,high,low,close,volume,"):
                        raise ValueError("Frozen WIKI CSV header changed.")
                    extract_lines.append(line)
                    continue
                fields = line.split(b",", 2)
                if len(fields) < 3 or fields[0] != b"COL":
                    continue
                full_col_rows += 1
                full_col_digest.update(line)
                date = fields[1].decode("ascii")
                if PRE_START <= date <= BOUNDARY:
                    extract_lines.append(line)
    _require_hash(member_digest.hexdigest(), pins.member_sha256, label="CSV member")
    _require_hash(
        full_col_digest.hexdigest(),
        pins.full_col_lines_sha256,
        label="full COL line inventory",
    )
    if full_col_rows != pins.full_col_row_count:
        raise ValueError("Frozen WIKI full COL row count changed.")

    extract = b"".join(extract_lines)
    _require_hash(hashlib.sha256(extract).hexdigest(), pins.extract_sha256, label="COL extract")
    if len(extract) != pins.extract_size or len(extract_lines) != pins.extract_line_count:
        raise ValueError("Frozen WIKI COL extract size/line inventory changed.")
    wiki_rows = pd.read_csv(io.BytesIO(extract))
    if (
        len(wiki_rows) != pins.extract_line_count - 1
        or set(wiki_rows["ticker"].astype(str)) != {SYMBOL}
        or wiki_rows["date"].astype(str).duplicated().any()
        or str(wiki_rows.iloc[0]["date"]) != PRE_START
        or str(wiki_rows.iloc[-1]["date"]) != BOUNDARY
    ):
        raise ValueError("Frozen WIKI COL extract topology changed.")
    boundary = wiki_rows.loc[wiki_rows["date"].astype(str).eq(BOUNDARY)]
    if not (
        len(boundary) == 1
        and _float(boundary.iloc[0]["close"], field="WIKI boundary close") == 88.61
        and _float(boundary.iloc[0]["split_ratio"], field="WIKI boundary split") == 1.0
    ):
        raise ValueError("Frozen WIKI COL boundary evidence changed.")

    if not kaggle_metadata_path.is_file():
        raise FileNotFoundError(
            f"Frozen Kaggle metadata is missing: {kaggle_metadata_path}."
        )
    metadata_bytes = kaggle_metadata_path.read_bytes()
    if len(metadata_bytes) != pins.metadata_size:
        raise ValueError("Frozen Kaggle metadata size changed.")
    _require_hash(
        hashlib.sha256(metadata_bytes).hexdigest(),
        pins.metadata_sha256,
        label="Kaggle metadata",
    )
    try:
        metadata = json.loads(metadata_bytes)
    except (TypeError, ValueError) as exc:
        raise ValueError("Frozen Kaggle metadata is not valid JSON.") from exc
    version_values = metadata.get("versions") or []
    observed_version = (
        int(version_values[0].get("versionNumber"))
        if len(version_values) == 1
        else -1
    )
    expected_metadata = {
        "id": pins.metadata_id,
        "ref": pins.metadata_ref,
        "licenseName": pins.metadata_license_name,
        "lastUpdated": pins.metadata_last_updated,
        "totalBytes": pins.metadata_total_bytes,
    }
    changed = [
        key for key, expected in expected_metadata.items() if metadata.get(key) != expected
    ]
    if changed or observed_version != pins.metadata_version:
        raise ValueError(
            "Frozen Kaggle metadata identity/license changed: "
            + ", ".join(changed or ["versionNumber"])
            + "."
        )
    if pins.metadata_license_name != "Unknown":
        raise ValueError("COL repair license gate must remain fail-closed on Unknown.")

    extract_artifact = ArchiveArtifact(
        dataset="kaggle_quandl_wiki_col_extract",
        source="kaggle_quandl_wiki_col_extract",
        source_url=WIKI_DOWNLOAD_URL,
        content_type="text/csv",
        extension="csv",
        payload=extract,
        retrieved_at=WIKI_RETRIEVED_AT,
    )
    metadata_artifact = ArchiveArtifact(
        dataset="kaggle_dataset_metadata",
        source="kaggle_dataset_metadata",
        source_url=KAGGLE_METADATA_URL,
        content_type="application/json",
        extension="json",
        payload=metadata_bytes,
        retrieved_at=REPAIR_REVIEWED_AT,
    )
    return EvidenceBundle(
        extract=extract_artifact,
        metadata=metadata_artifact,
        wiki_rows=wiki_rows,
        audit={
            "zip_sha256": pins.zip_sha256,
            "zip_size": pins.zip_size,
            "member_name": WIKI_MEMBER,
            "member_sha256": pins.member_sha256,
            "member_size": pins.member_size,
            "member_crc32": f"{pins.member_crc32:08x}",
            "full_col_lines_sha256": pins.full_col_lines_sha256,
            "full_col_row_count": pins.full_col_row_count,
            "extract_sha256": extract_artifact.source_hash,
            "extract_size": len(extract),
            "extract_line_count": len(extract_lines),
            "metadata_sha256": metadata_artifact.source_hash,
            "metadata_size": len(metadata_bytes),
            "metadata_ref": pins.metadata_ref,
            "metadata_version": pins.metadata_version,
            "metadata_license_name": pins.metadata_license_name,
        },
    )


def _target_sessions(frame: pd.DataFrame) -> pd.Series:
    sessions = pd.to_datetime(frame["session"], errors="coerce")
    if sessions.isna().any():
        raise ValueError("COL price/factor session inventory contains invalid dates.")
    return sessions.dt.date.astype(str)


def _audit_raw_price_relation(
    prices: pd.DataFrame,
    evidence: EvidenceBundle,
    *,
    pins: EvidencePins,
) -> tuple[pd.Series, Mapping[str, Any]]:
    sessions = _target_sessions(prices)
    target = prices["security_id"].astype(str).eq(COL_SECURITY_ID)
    pre = target & sessions.between(PRE_START, PRE_END)
    current = prices.loc[pre].copy()
    current["date"] = sessions.loc[pre].to_numpy()
    wiki = evidence.wiki_rows.loc[
        evidence.wiki_rows["date"].astype(str).between(PRE_START, PRE_END)
    ].copy()
    wiki["date"] = wiki["date"].astype(str)
    if len(current) != EXPECTED_PRE_SESSIONS or len(wiki) != EXPECTED_PRE_SESSIONS:
        raise ValueError("COL pre-boundary session count changed.")
    if current["date"].duplicated().any() or set(current["date"]) != set(wiki["date"]):
        raise ValueError("COL pre-boundary XNYS/WIKI session inventory changed.")
    if set(current["source"].astype(str)) != {ORIGINAL_PRICE_SOURCE}:
        raise ValueError("COL pre-boundary provider source changed.")
    if set(current["source_hash"].astype(str)) != {pins.original_price_hash}:
        raise ValueError("COL pre-boundary provider hash changed.")

    joined = current.merge(wiki, on="date", suffixes=("_eod", "_wiki"), validate="one_to_one")
    audit: dict[str, Any] = {
        "session_count": len(joined),
        "start": joined["date"].min(),
        "end": joined["date"].max(),
    }
    expected_exact = {"open": 347, "high": 347, "low": 346, "close": 355}
    expected_max = {"open": 0.26, "high": 0.0005, "low": 0.0005, "close": 0.0}
    for column in ("open", "high", "low", "close"):
        corrected = pd.to_numeric(joined[f"{column}_eod"], errors="raise") * SCALE
        independent = pd.to_numeric(joined[f"{column}_wiki"], errors="raise")
        residual = (corrected - independent).abs()
        exact = int(np.isclose(residual, 0.0, rtol=0.0, atol=1e-12).sum())
        maximum = float(residual.max())
        audit[f"{column}_exact_after_x10"] = exact
        audit[f"{column}_max_abs_residual_after_x10"] = maximum
        if pins.enforce_reviewed_relation_profile and (
            exact != expected_exact[column]
            or not math.isclose(maximum, expected_max[column], rel_tol=0.0, abs_tol=1e-10)
        ):
            raise ValueError(f"Reviewed COL {column} x10 relation changed.")
    eod_volume = pd.to_numeric(joined["volume_eod"], errors="raise")
    wiki_volume = pd.to_numeric(joined["volume_wiki"], errors="raise")
    volume_diff = (eod_volume - wiki_volume).abs()
    audit.update(
        {
            "volume_equal_sessions": int(np.isclose(volume_diff, 0.0, rtol=0.0, atol=0.0).sum()),
            "volume_median_abs_difference": float(volume_diff.median()),
            "volume_max_abs_difference": float(volume_diff.max()),
            "volume_scaled": False,
        }
    )
    if pins.enforce_reviewed_relation_profile and (
        audit["volume_equal_sessions"] != 252
        or audit["volume_median_abs_difference"] != 0.0
        or audit["volume_max_abs_difference"] != 877_870.0
    ):
        raise ValueError("Reviewed COL independent volume relation changed.")

    boundary = prices.loc[target & sessions.eq(BOUNDARY)]
    if len(boundary) != 1 or _float(boundary.iloc[0]["close"], field="boundary close") != 88.61:
        raise ValueError("COL EODHD boundary close changed.")
    audit["boundary_close"] = 88.61
    audit["boundary_wiki_split_ratio"] = 1.0
    return pre, audit


def _provenance_artifact(
    evidence: EvidenceBundle,
    price_audit: Mapping[str, Any],
    *,
    pins: EvidencePins,
) -> ArchiveArtifact:
    payload = _canonical_json(
        {
            "schema": "us_col_scaling_repair_evidence/v1",
            "reviewed_at": REPAIR_REVIEWED_AT,
            "security_id": COL_SECURITY_ID,
            "symbol": SYMBOL,
            "repair": {
                "pre_start": PRE_START,
                "pre_end": PRE_END,
                "boundary": BOUNDARY,
                "ohlc_multiplier": SCALE,
                "raw_volume_multiplier": 1.0,
                "dividend_event_ids": sorted(DIVIDENDS),
                "dividend_multiplier": SCALE,
                "removed_split_event_id": SPLIT_EVENT_ID,
                "removed_split_ratio": 0.1,
                "original_price_source_hash": pins.original_price_hash,
                "original_dividend_source_hash": pins.original_dividend_hash,
                "original_split_source_hash": pins.original_split_hash,
            },
            "frozen_evidence": dict(evidence.audit),
            "price_relation": dict(price_audit),
            "license_policy": {
                "formal_license_name": "Unknown",
                "uploader_description_claim_is_not_formal_license": True,
                "allowed_scope": "private_internal_only",
                "redistribution_allowed": False,
                "publication_allowed": False,
                "fail_closed": True,
            },
        }
    )
    return ArchiveArtifact(
        dataset="reviewed_col_scaling_provenance",
        source=REPAIR_SOURCE,
        source_url=WIKI_DOWNLOAD_URL,
        content_type="application/json",
        extension="json",
        payload=payload,
        retrieved_at=REPAIR_REVIEWED_AT,
    )


def _repair_prices(
    prices: pd.DataFrame,
    pre_mask: pd.Series,
    provenance: ArchiveArtifact,
) -> pd.DataFrame:
    output = prices.copy(deep=True)
    original_volume = output["volume"].copy(deep=True)
    for column in ("open", "high", "low", "close"):
        output.loc[pre_mask, column] = (
            pd.to_numeric(output.loc[pre_mask, column], errors="raise") * SCALE
        )
    output.loc[pre_mask, "source"] = REPAIR_SOURCE
    output.loc[pre_mask, "retrieved_at"] = REPAIR_REVIEWED_AT
    output.loc[pre_mask, "source_hash"] = provenance.source_hash
    output.loc[pre_mask, "source_url"] = WIKI_DOWNLOAD_URL
    if not original_volume.equals(output["volume"]):
        raise AssertionError("COL scale repair changed raw volume.")
    changed_non_target = ~pre_mask
    for column in prices.columns:
        if not prices.loc[changed_non_target, column].equals(output.loc[changed_non_target, column]):
            raise AssertionError(f"COL scale repair changed out-of-scope price field: {column}.")
    return output


def _corrected_dividend_metadata(
    *,
    event_id: str,
    old_amount: float,
    provenance_hash: str,
    pins: EvidencePins,
) -> str:
    return json.dumps(
        {
            "correction": "provider_decimal_scale_x10",
            "evidence_sha256": provenance_hash,
            "license_scope": "private_internal_only",
            "original_cash_amount": old_amount,
            "original_event_id_retained": event_id,
            "original_source": ORIGINAL_DIVIDEND_SOURCE,
            "original_source_hash": pins.original_dividend_hash,
            "original_source_url": DIVIDEND_URL,
            "scale": SCALE,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _repair_actions(
    actions: pd.DataFrame,
    evidence: EvidenceBundle,
    provenance: ArchiveArtifact,
    *,
    pins: EvidencePins,
) -> pd.DataFrame:
    split = actions["event_id"].astype(str).eq(SPLIT_EVENT_ID)
    if int(split.sum()) != 1:
        raise ValueError("COL synthetic split inventory changed.")
    split_row = actions.loc[split].iloc[0]
    if not (
        _text(split_row.get("security_id")) == COL_SECURITY_ID
        and _text(split_row.get("action_type")) == "split"
        and _date(split_row.get("effective_date")) == BOUNDARY
        and _date(split_row.get("ex_date")) == BOUNDARY
        and _float(split_row.get("ratio"), field="synthetic split ratio") == 0.1
        and _text(split_row.get("source")) == ORIGINAL_SPLIT_SOURCE
        and _text(split_row.get("source_hash")) == pins.original_split_hash
        and _text(split_row.get("source_url")) == SPLIT_URL
    ):
        raise ValueError("COL synthetic split terms/provenance changed.")

    output = actions.loc[~split].copy(deep=True)
    # A narrow fixture or future compacted release can contain only null
    # metadata, which parquet round-trips as float64.  Promote before writing
    # the reviewed JSON strings instead of relying on pandas' implicit upcast.
    output["metadata"] = output["metadata"].astype("object")
    wiki = evidence.wiki_rows.copy()
    wiki["date"] = wiki["date"].astype(str)
    for event_id, (date, old_amount, new_amount) in DIVIDENDS.items():
        matches = output["event_id"].astype(str).eq(event_id)
        if int(matches.sum()) != 1:
            raise ValueError(f"COL dividend inventory changed: {event_id}.")
        index = output.index[matches][0]
        row = output.loc[index]
        wiki_row = wiki.loc[wiki["date"].eq(date)]
        if len(wiki_row) != 1 or not math.isclose(
            _float(wiki_row.iloc[0]["ex-dividend"], field=f"WIKI dividend {date}"),
            new_amount,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Frozen WIKI COL dividend evidence changed: {date}.")
        if not (
            _text(row.get("security_id")) == COL_SECURITY_ID
            and _text(row.get("action_type")) == "cash_dividend"
            and _date(row.get("effective_date")) == date
            and _date(row.get("ex_date")) == date
            and math.isclose(
                _float(row.get("cash_amount"), field=f"dividend {date}"),
                old_amount,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and _text(row.get("source")) == ORIGINAL_DIVIDEND_SOURCE
            and _text(row.get("source_hash")) == pins.original_dividend_hash
            and _text(row.get("source_url")) == DIVIDEND_URL
        ):
            raise ValueError(f"COL dividend terms/provenance changed: {event_id}.")
        output.at[index, "cash_amount"] = new_amount
        output.at[index, "source"] = REPAIR_SOURCE
        output.at[index, "source_kind"] = REPAIR_SOURCE_KIND
        output.at[index, "source_url"] = WIKI_DOWNLOAD_URL
        output.at[index, "source_hash"] = provenance.source_hash
        output.at[index, "retrieved_at"] = REPAIR_REVIEWED_AT
        output.at[index, "metadata"] = _corrected_dividend_metadata(
            event_id=event_id,
            old_amount=old_amount,
            provenance_hash=provenance.source_hash,
            pins=pins,
        )
    output.reset_index(drop=True, inplace=True)
    return output


def _factor_lineage(price_version: str, action_version: str) -> str:
    if not price_version or not action_version:
        raise ValueError("COL factor lineage requires exact planned versions.")
    return f"{price_version}+{action_version}"


def _build_factors(
    current: pd.DataFrame,
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    source_version: str,
) -> pd.DataFrame:
    output = build_adjustment_factors(
        prices,
        actions,
        source_version=source_version,
    ).reindex(columns=current.columns)
    output["source_version"] = source_version
    output["calculated_at"] = REPAIR_REVIEWED_AT
    output["source"] = "derived"
    output["retrieved_at"] = REPAIR_REVIEWED_AT
    output["source_hash"] = source_version
    return output


def _sort_factors(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["security_id"] = output["security_id"].astype(str)
    output["session"] = pd.to_datetime(output["session"], errors="raise").dt.normalize()
    return output.sort_values(["security_id", "session"], ignore_index=True)


def _audit_factor_impact(
    current_prices: pd.DataFrame,
    repaired_prices: pd.DataFrame,
    current_factors: pd.DataFrame,
    repaired_factors: pd.DataFrame,
) -> Mapping[str, Any]:
    before = _sort_factors(current_factors)
    after = _sort_factors(repaired_factors)
    keys = ["security_id", "session"]
    if len(before) != len(after) or not before[keys].equals(after[keys]):
        raise ValueError("COL repair changed adjustment-factor key inventory.")
    outside = ~before["security_id"].eq(COL_SECURITY_ID)
    for column in ("split_factor", "total_return_factor"):
        left = pd.to_numeric(before.loc[outside, column], errors="raise").to_numpy(float)
        right = pd.to_numeric(after.loc[outside, column], errors="raise").to_numpy(float)
        if not np.array_equal(left, right, equal_nan=True):
            raise ValueError("COL repair changed another security's factor economics.")

    current_target_prices = current_prices.loc[
        current_prices["security_id"].astype(str).eq(COL_SECURITY_ID)
    ].copy()
    repaired_target_prices = repaired_prices.loc[
        repaired_prices["security_id"].astype(str).eq(COL_SECURITY_ID)
    ].copy()
    current_target_factors = current_factors.loc[
        current_factors["security_id"].astype(str).eq(COL_SECURITY_ID)
    ].copy()
    repaired_target_factors = repaired_factors.loc[
        repaired_factors["security_id"].astype(str).eq(COL_SECURITY_ID)
    ].copy()
    current_adjusted = apply_adjustment_factors(
        current_target_prices,
        current_target_factors,
        mode="total_return_adjusted",
    ).sort_values("session", ignore_index=True)
    repaired_adjusted = apply_adjustment_factors(
        repaired_target_prices,
        repaired_target_factors,
        mode="total_return_adjusted",
    ).sort_values("session", ignore_index=True)
    maximum = 0.0
    for column in ("open", "high", "low", "close"):
        delta = (
            pd.to_numeric(current_adjusted[column], errors="raise")
            - pd.to_numeric(repaired_adjusted[column], errors="raise")
        ).abs()
        maximum = max(maximum, float(delta.max()))
        if not np.allclose(
            current_adjusted[column],
            repaired_adjusted[column],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("COL atomic repair changed adjusted strategy OHLC.")

    sessions = pd.to_datetime(current_adjusted["session"]).dt.date.astype(str)
    pre = sessions.between(PRE_START, PRE_END)
    current_volume = pd.to_numeric(current_adjusted.loc[pre, "volume"], errors="raise")
    repaired_volume = pd.to_numeric(repaired_adjusted.loc[pre, "volume"], errors="raise")
    if not np.allclose(repaired_volume, current_volume * SCALE, rtol=0.0, atol=1e-9):
        raise ValueError("COL adjusted-volume impact changed from the reviewed x10 relation.")

    current_target = _sort_factors(current_target_factors)
    repaired_target = _sort_factors(repaired_target_factors)
    target_sessions = current_target["session"].dt.date.astype(str)
    target_pre = target_sessions.between(PRE_START, PRE_END)
    if set(pd.to_numeric(current_target.loc[target_pre, "split_factor"])) != {10.0}:
        raise ValueError("Current COL pre-boundary split factor is no longer 10.")
    if set(pd.to_numeric(repaired_target.loc[target_pre, "split_factor"])) != {1.0}:
        raise ValueError("Repaired COL pre-boundary split factor is not 1.")
    current_total = pd.to_numeric(
        current_target.loc[target_pre, "total_return_factor"], errors="raise"
    ).to_numpy(float)
    repaired_total = pd.to_numeric(
        repaired_target.loc[target_pre, "total_return_factor"], errors="raise"
    ).to_numpy(float)
    factor_delta = float(np.max(np.abs(repaired_total - current_total / SCALE)))
    if factor_delta > 1e-12:
        raise ValueError("COL repaired total-return factors lost the reviewed /10 relation.")
    return {
        "current_pre_split_factor": 10.0,
        "repaired_pre_split_factor": 1.0,
        "pre_total_return_factor_vs_current_divided_by_10_max_abs_diff": factor_delta,
        "adjusted_ohlc_max_abs_diff": maximum,
        "adjusted_volume_pre_boundary_multiplier": SCALE,
        "raw_volume_changed": False,
        "strategy_uses_volume": False,
        "strategy_adjusted_ohlc_equivalent": True,
        "expected_triple_supertrend_signal_change": 0,
    }


def _artifact_rows(
    artifacts: tuple[ArchiveArtifact, ...],
    *,
    completed_session: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
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
            for artifact in artifacts
        ],
        columns=dataset_spec("source_archive").required_columns,
    )


def _repair_archive(
    current: pd.DataFrame,
    artifacts: tuple[ArchiveArtifact, ...],
    *,
    completed_session: str,
    pins: EvidencePins,
) -> pd.DataFrame:
    full_zip = current["archive_id"].astype(str).eq(pins.zip_sha256)
    if int(full_zip.sum()) != 1:
        raise ValueError("Current source_archive full WIKI ZIP row inventory changed.")
    row = current.loc[full_zip].iloc[0]
    expected = {
        "archive_id": pins.zip_sha256,
        "dataset": "kaggle_frozen_quandl_wiki_mirror",
        "object_path": f"archives/{completed_session}/{pins.zip_sha256}.zip.gz",
        "content_type": "application/zip",
        "effective_date": completed_session,
        "source": "kaggle_frozen_quandl_wiki_mirror",
        "retrieved_at": WIKI_RETRIEVED_AT,
        "source_hash": pins.zip_sha256,
        "source_url": WIKI_DOWNLOAD_URL,
    }
    changed = [
        key
        for key, value in expected.items()
        if (_date(row.get(key)) if key == "effective_date" else _text(row.get(key)))
        != value
    ]
    if changed:
        raise ValueError("Current full WIKI ZIP archive row changed: " + ", ".join(changed))
    additions = _artifact_rows(artifacts, completed_session=completed_session)
    collision = current["archive_id"].astype(str).isin(additions["archive_id"].astype(str))
    if collision.any():
        raise ValueError("COL minimal evidence artifacts already partially exist.")
    output = pd.concat([current.loc[~full_zip], additions], ignore_index=True)
    if output["archive_id"].astype(str).duplicated().any():
        raise ValueError("COL source_archive candidate contains duplicate content IDs.")
    return output


def _new_planned_versions(release: DataRelease) -> dict[str, str]:
    token = uuid.uuid4().hex
    session = release.completed_session.replace("-", "")
    return {
        dataset: f"col-scale-repair-{session}-{token}-{dataset}"
        for dataset in WRITE_DATASETS
    }


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
        if dataset in self.overrides:
            return self.base.manifest_for_version(dataset, self.versions[dataset])
        version = self.versions.get(dataset)
        return self.base.manifest_for_version(dataset, version) if version else None

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        if dataset in self.overrides:
            return self.overrides[dataset].copy()
        version = self.versions.get(dataset)
        return self.base.read_frame(dataset, version) if version else pd.DataFrame()


def _reviewed_inherited_identity_gaps(
    repository: LocalDatasetRepository,
) -> tuple[str, ...]:
    """Carry forward only the base release's exact reviewed gap fingerprints."""

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
    validate_repository_snapshot(
        repository,
        allowed_index_identity_gap_fingerprints=allowed,
    ).raise_for_errors()
    return allowed


def _capture_pointer_etags(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for dataset in REQUIRED_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != release.dataset_versions.get(dataset):
            raise RuntimeError(f"COL release/current pointer mismatch: {dataset}.")
        output[dataset] = etag
    return output


def prepare_repair(
    repository: LocalDatasetRepository,
    *,
    wiki_zip_path: Path = DEFAULT_WIKI_ZIP,
    kaggle_metadata_path: Path = DEFAULT_KAGGLE_METADATA,
    pins: EvidencePins = DEFAULT_EVIDENCE_PINS,
) -> PreparedRepair:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required for COL repair.")
    missing = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks COL repair datasets: " + ", ".join(missing))
    pointer_etags = _capture_pointer_etags(repository, release)
    allowed_identity_gaps = _reviewed_inherited_identity_gaps(repository)
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in REQUIRED_DATASETS
    }
    evidence = load_evidence_bundle(
        wiki_zip_path,
        kaggle_metadata_path,
        pins=pins,
    )
    pre_mask, price_audit = _audit_raw_price_relation(
        frames["daily_price_raw"], evidence, pins=pins
    )
    provenance = _provenance_artifact(evidence, price_audit, pins=pins)
    artifacts = (evidence.extract, evidence.metadata, provenance)
    planned_versions = _new_planned_versions(release)
    prices = _repair_prices(frames["daily_price_raw"], pre_mask, provenance)
    actions = _repair_actions(
        frames["corporate_actions"], evidence, provenance, pins=pins
    )
    factor_source_version = _factor_lineage(
        planned_versions["daily_price_raw"],
        planned_versions["corporate_actions"],
    )
    factors = _build_factors(
        frames["adjustment_factors"],
        prices,
        actions,
        source_version=factor_source_version,
    )
    factor_audit = _audit_factor_impact(
        frames["daily_price_raw"],
        prices,
        frames["adjustment_factors"],
        factors,
    )
    source_archive = _repair_archive(
        frames["source_archive"],
        artifacts,
        completed_session=release.completed_session,
        pins=pins,
    )
    overrides = {
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "adjustment_factors": factors,
        "source_archive": source_archive,
    }
    for dataset, frame in overrides.items():
        validate_dataset(
            dataset,
            frame,
            completed_session=release.completed_session,
            incomplete_action_policy="block",
        ).raise_for_errors()
    validate_repository_snapshot(
        _CandidateRepository(repository, release.dataset_versions, overrides),
        allowed_index_identity_gap_fingerprints=allowed_identity_gaps,
    ).raise_for_errors()
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions=planned_versions,
        frames=overrides,
        artifacts=artifacts,
        allowed_index_identity_gap_fingerprints=allowed_identity_gaps,
        wiki_zip_path=wiki_zip_path,
        kaggle_metadata_path=kaggle_metadata_path,
        pins=pins,
        summary={
            "status": "validated_offline_plan",
            "base_release_version": release.version,
            "security_id": COL_SECURITY_ID,
            "symbol": SYMBOL,
            "pre_boundary_price_rows_scaled": int(pre_mask.sum()),
            "raw_volume_rows_changed": 0,
            "dividend_rows_scaled": len(DIVIDENDS),
            "false_split_rows_removed": 1,
            "false_split_event_id": SPLIT_EVENT_ID,
            "factor_rows_rebuilt": len(factors),
            "factor_source_version": factor_source_version,
            "price_relation": dict(price_audit),
            "factor_impact": dict(factor_audit),
            "evidence": {
                **dict(evidence.audit),
                "provenance_sha256": provenance.source_hash,
                "provenance_size": len(provenance.payload),
                "archive_artifact_sha256": [item.source_hash for item in artifacts],
            },
            "source_archive_full_zip_rows_removed": 1,
            "source_archive_minimal_rows_added": len(artifacts),
            "full_zip_payload_deleted": False,
            "full_zip_payload_unreferenced_after_apply": True,
            "license_name": "Unknown",
            "license_status": "unverified",
            "license_scope": "private_internal_only",
            "redistribution_allowed": False,
            "publication_allowed": False,
            "apply_requires_private_internal_ack": True,
            "planned_versions": dict(planned_versions),
            "write_datasets": list(WRITE_DATASETS),
            "inherited_index_identity_gap_fingerprints": list(
                allowed_identity_gaps
            ),
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        },
    )


@contextmanager
def _exclusive_repository_lock(repository: LocalDatasetRepository):
    path = repository.root / ".locks/market-store-write.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery = repository.root / RECOVERY_DIR
        if recovery.exists() and tuple(recovery.glob("*.json")):
            raise RuntimeError("Unresolved COL repair recovery marker blocks writes.")
        yield


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(
        path,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )


def _safe_path(root: Path, object_path: str) -> Path:
    base = root.resolve()
    target = (base / object_path).resolve()
    if target == base or base not in target.parents:
        raise ValueError(f"COL archive path escapes repository: {object_path}.")
    return target


def _persist_artifacts(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> None:
    for artifact in prepared.artifacts:
        path = _safe_path(
            repository.root,
            artifact.object_path(prepared.release.completed_session),
        )
        if path.exists():
            try:
                observed = gzip.decompress(path.read_bytes())
            except (OSError, EOFError) as exc:
                raise ValueError(f"Existing COL archive artifact is invalid gzip: {path}.") from exc
            if hashlib.sha256(observed).hexdigest() != artifact.source_hash:
                raise ValueError(f"Existing COL archive artifact hash changed: {path}.")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(path, gzip.compress(artifact.payload, mtime=0))


def _assert_inputs_unchanged(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> None:
    release, etag = repository.current_release()
    if (
        release is None
        or release.version != prepared.release.version
        or etag != prepared.release_etag
    ):
        raise RuntimeError("Current release changed after COL repair planning.")
    for dataset in REQUIRED_DATASETS:
        pointer, pointer_etag = repository.current_pointer(dataset)
        if (
            pointer is None
            or pointer.version != prepared.release.dataset_versions[dataset]
            or pointer_etag != prepared.pointer_etags[dataset]
        ):
            raise RuntimeError(f"{dataset} pointer changed after COL repair planning.")


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
    metadata.update(
        {
            "operation": OPERATION,
            "input_release_version": prepared.release.version,
            "col_security_id": COL_SECURITY_ID,
            "col_scale_boundary": BOUNDARY,
            "col_provenance_sha256": prepared.artifacts[-1].source_hash,
            "license_name": "Unknown",
            "license_scope": "private_internal_only",
            "redistribution_allowed": False,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        }
    )
    if dataset == "adjustment_factors":
        lineage = _factor_lineage(
            prepared.planned_versions["daily_price_raw"],
            prepared.planned_versions["corporate_actions"],
        )
        metadata.update(
            {
                "source_version": lineage,
                "source_daily_price_version": prepared.planned_versions[
                    "daily_price_raw"
                ],
                "source_corporate_actions_version": prepared.planned_versions[
                    "corporate_actions"
                ],
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
                    f"unexpected release during COL rollback: {observed.version}"
                )
            repository.objects.put(
                "releases/current.json",
                old_release_bytes,
                if_match=current.etag,
            )
    except Exception as exc:  # pragma: no cover - exercised by failure injection
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
                        f"unexpected {dataset} pointer during COL rollback: {pointer.version}"
                    )
                repository.objects.put(key, old, if_match=current.etag)
        except Exception as exc:  # pragma: no cover - exercised by failure injection
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    private_internal_only_ack: bool = False,
    inject_failure: FailureInjector | None = None,
) -> dict[str, Any]:
    if not private_internal_only_ack:
        raise RuntimeError(
            "Kaggle WIKI licenseName is Unknown; COL apply requires explicit "
            "private/internal-only acknowledgement."
        )
    inject = inject_failure or (lambda _stage: None)
    with _exclusive_repository_lock(repository):
        _assert_inputs_unchanged(repository, prepared)
        # Re-read and re-hash the frozen evidence under the writer lock.
        load_evidence_bundle(
            prepared.wiki_zip_path,
            prepared.kaggle_metadata_path,
            pins=prepared.pins,
        )
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != prepared.release.dataset_versions[dataset]
                or value.etag != prepared.pointer_etags[dataset]
            ):
                raise RuntimeError(f"{dataset} pointer changed before COL apply.")
            old_pointers[dataset] = value.data
        transaction_id = uuid.uuid4().hex
        journal_path = repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        journal: dict[str, Any] = {
            "schema": "us_col_scaling_repair_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                key: base64.b64encode(value).decode("ascii")
                for key, value in old_pointers.items()
            },
            "planned_versions": dict(prepared.planned_versions),
            "license_scope": "private_internal_only",
            "created_at": utc_now_iso(),
        }
        _write_json(journal_path, journal)
        committed: DataRelease | None = None
        try:
            _persist_artifacts(repository, prepared)
            inject("after_artifacts")
            versions = dict(prepared.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    prepared.frames[dataset],
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy="block",
                    metadata=_metadata_for_write(repository, prepared, dataset),
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=prepared.planned_versions[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"{dataset} write conflicted: {result.conflict_path}."
                    )
                versions[dataset] = result.manifest.version
                inject(f"after_write:{dataset}")
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=prepared.release.quality,
                warnings=prepared.release.warnings,
                expected_etag=prepared.release_etag,
            )
            inject("after_release_commit")
            validate_repository_snapshot(
                repository,
                allowed_index_identity_gap_fingerprints=(
                    prepared.allowed_index_identity_gap_fingerprints
                ),
            ).raise_for_errors()
            journal.update(
                {
                    "status": "committed",
                    "committed_release_version": committed.version,
                    "completed_at": utc_now_iso(),
                }
            )
            _write_json(journal_path, journal)
            return {
                **prepared.summary,
                "status": "applied",
                "mode": "apply",
                "writes_performed": True,
                "new_release_version": committed.version,
                "transaction_id": transaction_id,
                "license_scope": "private_internal_only",
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_versions=prepared.planned_versions,
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
            _write_json(journal_path, journal)
            if rollback_errors:
                recovery = repository.root / RECOVERY_DIR / f"{transaction_id}.json"
                _write_json(recovery, journal)
                raise RuntimeError(
                    "COL rollback was incomplete; recovery marker blocks writes: "
                    f"{recovery}; errors={rollback_errors}."
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan the atomic offline COL scale/action repair."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--wiki-zip", type=Path, default=DEFAULT_WIKI_ZIP)
    parser.add_argument(
        "--kaggle-metadata",
        type=Path,
        default=DEFAULT_KAGGLE_METADATA,
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--ack-private-internal-only", action="store_true")
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
            private_internal_only_ack=args.ack_private_internal_only,
        )
        if args.apply
        else {**prepared.summary, "mode": "plan", "writes_performed": False}
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
