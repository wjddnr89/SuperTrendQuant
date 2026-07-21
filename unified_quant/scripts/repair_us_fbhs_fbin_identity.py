#!/usr/bin/env python3
"""Repair the duplicated FBHS -> FBIN same-security lineage, offline.

The reviewed SEC filing proves that Fortune Brands Home & Security, Inc. kept
the same legal issuer and common stock while changing its name and NYSE ticker
to Fortune Brands Innovations, Inc. / FBIN at the open on 2022-12-15.  The
bootstrap snapshot nevertheless contains two security IDs and two overlapping
provider histories.  The active FBIN endpoint is the canonical uninterrupted
price/action source; no FBHS provider row is copied into it.

Plan and apply are cache-only.  Exact SEC and EODHD successor-bundle bytes are
hash/size pinned.  Independent review classified EODHD's transition-date
``117/100`` row as a synthetic spinoff adjustment rather than a legal split,
so it is explicitly removed.  The official MBC 1:1 spinoff remains a named
fail-closed publication follow-up.  Apply uses the repository writer lock,
release/pointer CAS, a rollback journal, persisted archive verification, and
an idempotence check.
This tool never calls EODHD, any other network endpoint, or R2.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import html
import json
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.ingest import SourceArtifact
from supertrend_quant.market_store.lifecycle import canonical_lifecycle_event_id
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    sha256_bytes,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.validation import (
    validate_dataset,
    validate_repository_snapshot,
)


DEFAULT_CACHE_ROOT = Path("data/cache")

OLD_SECURITY_ID = "US:EODHD:724457bc-0eaf-5959-8c93-f0c2a03c80de"
CANONICAL_SECURITY_ID = "US:EODHD:89fe6d28-737c-5b16-82e6-c1207561311c"
OLD_SYMBOL = "FBHS"
NEW_SYMBOL = "FBIN"
HISTORY_START = "2015-01-01"
PRICE_START = "2015-01-02"
OLD_LAST_SESSION = "2022-12-14"
TRANSITION_DATE = "2022-12-15"
SOURCE_OLD_LAST_SESSION = "2023-01-20"
SOURCE_NEW_FIRST_SESSION = "2023-01-23"
PRICE_END = "2026-07-15"

SEC_URL = (
    "https://www.sec.gov/Archives/edgar/data/1519751/"
    "000119312522306146/0001193125-22-306146.txt"
)
SEC_SHA256 = "2c2703ed8949f1d72ceea49e655005cd39165a8020b1750c71894d185d987135"
SEC_EXACT_BYTES = 1_430_923
SEC_RETRIEVED_AT = "2026-07-17T20:04:15.373205Z"
SEC_CACHE_KEY = sha256_bytes(f"{SEC_URL}?".encode())
SEC_REQUIRED_TEXT_GROUPS = (
    ("accession number: 0001193125-22-306146",),
    ("conformed submission type: 8-k",),
    ("central index key: 0001519751",),
    (
        "fortune brands home & security, inc., now known as fortune brands innovations, inc.",
    ),
    (
        "each company stockholder received one share of masterbrand common stock for every one share of company common stock",
    ),
    (
        "change its corporate name from \"fortune brands home & security, inc.\" to \"fortune brands innovations, inc.,\" effective as of 12:01 a.m. eastern time on december 15, 2022",
    ),
    (
        "effective at the open of business on december 15, 2022, the company's shares of common stock, par value $0.01 per share, began trading on the new york stock exchange under the new ticker symbol \"fbin\"",
    ),
)

SUCCESSOR_BUNDLE_PATH = Path(
    "state/eodhd_lifecycle_successors/"
    "e4fcf8fee787073f9b583ff693f5f955d64985bfb83b904e0158690efcc34bc2.json.gz"
)
SUCCESSOR_BUNDLE_SHA256 = (
    "b9c84c977b9dda0f9f61a18936a1ec2bfcb27d086183b2f3148b57293f53f84d"
)
SUCCESSOR_BUNDLE_EXACT_BYTES = 2_114_325
SUCCESSOR_PAYLOAD_SHA256 = (
    "9f99fb127e75c09f0220b3cf75147654423374b24374553cd832ac3e91d7b6ab"
)
SUCCESSOR_PAYLOAD_EXACT_BYTES = 15_453_258
SUCCESSOR_ARTIFACT_COUNT = 39

FBIN_EOD_URL = "https://eodhd.com/api/eod/FBIN.US?from=2015-01-01&to=2026-07-15"
FBIN_EOD_SHA256 = "acf3bacc23ba053d26669fed7b37aff54b6c7d02d4e631646bf288ec4d783246"
FBIN_EOD_EXACT_BYTES = 336_332
FBIN_DIV_URL = "https://eodhd.com/api/div/FBIN.US?from=2015-01-01&to=2026-07-15"
FBIN_DIV_SHA256 = "05e9283a144cd4da6ddeb04b14486fc39a41ad9dcf925b403d2223866ef05b54"
FBIN_DIV_EXACT_BYTES = 8_373
FBIN_SPLITS_URL = (
    "https://eodhd.com/api/splits/FBIN.US?from=2015-01-01&to=2026-07-15"
)
FBIN_SPLITS_SHA256 = (
    "948db5813a8bcc3f51963ea162309f2253ddb01a9297cd59c477809d6f32fdc3"
)
FBIN_SPLITS_EXACT_BYTES = 55
FBIN_RETRIEVED_AT = "2026-07-17T20:37:19.646614Z"
FBHS_EOD_URL = "https://eodhd.com/api/eod/FBHS.US?from=2015-01-01&to=2026-07-15"
FBHS_EOD_SHA256 = "3e7ac22210fd5c995b024c082817ce7398a01c87aa49569a904b632e2b7fbbef"
FBHS_DIV_URL = "https://eodhd.com/api/div/FBHS.US?from=2015-01-01&to=2026-07-15"
FBHS_DIV_SHA256 = "9b7ec95b562c6ef0931a72d89fdfad78b70ebf8f824ce3e590fbcefe42f8ddc1"

EXPECTED_OLD_PRICE_ROWS = 2_027
EXPECTED_CANONICAL_PRICE_ROWS = 2_899
EXPECTED_OVERLAP_ROWS = 2_027
EXPECTED_CLOSE_EXACT_OVERLAP_ROWS = 1_731
EXPECTED_OLD_DIVIDENDS = 32
EXPECTED_CANONICAL_DIVIDENDS = 46
PROVIDER_SPLIT_EVENT_ID = (
    "42bbc1a172d3a03bff59d361361c715e38ec85b5e51b0115a6b9cf4e6b7ad176"
)
PROVIDER_SPLIT_RATIO = 1.17

# Independent review found that EODHD's 117/100 row is a synthetic adjustment
# for the MasterBrand distribution, not a legal share split.  This repair
# removes it.  The official 1:1 MBC spinoff and MBC price lineage belong to the
# dedicated follow-up transaction and must block final validation/publication
# until that transaction completes.
PROVIDER_SPLIT_DISPOSITION = "remove_pseudo_split"
ALLOWED_PROVIDER_SPLIT_DISPOSITIONS = frozenset(
    {"remove_pseudo_split"}
)
PENDING_FOLLOWUP = (
    "MBC 1:1 spinoff collection required before final validation/publication"
)

SP500_SOURCE = "community_sp500_history"
SP500_SOURCE_SHA256 = "39a9202c9ef69a74c0ff07e2113ad41fb6da7c8c5b6cd9541f0185fb4391e717"
SP500_SOURCE_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20%28Updated%29.csv"
)

OFFICIAL_SOURCE = "official_fbhs_fbin_identity_repair"
OFFICIAL_SOURCE_KIND = "official_filing"

WRITE_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "corporate_actions",
    "adjustment_factors",
    "index_constituent_anchors",
    "index_membership_events",
    "source_archive",
)


@dataclass(frozen=True)
class EvidenceBundle:
    sec: SourceArtifact
    eod: SourceArtifact
    dividends: SourceArtifact
    splits: SourceArtifact

    @property
    def archive_artifacts(self) -> tuple[SourceArtifact, ...]:
        return (self.sec, self.eod, self.dividends, self.splits)


@dataclass
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: dict[str, str | None]
    frames: dict[str, pd.DataFrame]
    evidence: EvidenceBundle
    warnings: tuple[str, ...]
    summary: dict[str, Any]


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


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
    if not _text(value):
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid FBHS/FBIN date: {value!r}")
    return pd.Timestamp(parsed).date().isoformat()


def _normalized_document_text(content: bytes) -> str:
    decoded = content.decode("utf-8", errors="replace")
    decoded = html.unescape(decoded).replace("\xa0", " ")
    decoded = re.sub(r"<[^>]+>", " ", decoded)
    decoded = re.sub(r"\s+", " ", decoded).strip().casefold()
    return decoded.replace("’", "'").replace("“", '"').replace("”", '"')


def _one_row(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"Expected one {label}; observed={len(rows)}")
    return rows.iloc[0]


def _assert_split_disposition() -> str:
    if PROVIDER_SPLIT_DISPOSITION not in ALLOWED_PROVIDER_SPLIT_DISPOSITIONS:
        raise RuntimeError(
            "FBIN provider 117/100 split disposition differs from the reviewed "
            "remove_pseudo_split outcome."
        )
    return PROVIDER_SPLIT_DISPOSITION


def _load_sec_artifact(cache_root: Path) -> SourceArtifact:
    path = cache_root / "state/sec_lifecycle" / f"{SEC_CACHE_KEY}.bin"
    if not path.is_file():
        raise FileNotFoundError(f"Pinned FBHS/FBIN SEC cache is absent: {path}")
    content = path.read_bytes()
    if len(content) != SEC_EXACT_BYTES or sha256_bytes(content) != SEC_SHA256:
        raise ValueError("Pinned FBHS/FBIN SEC filing hash/size changed.")
    normalized = _normalized_document_text(content)
    missing = [
        group
        for group in SEC_REQUIRED_TEXT_GROUPS
        if not any(phrase.casefold() in normalized for phrase in group)
    ]
    if missing:
        raise ValueError(
            "Pinned FBHS/FBIN SEC filing no longer proves reviewed claims: "
            + repr(missing)
        )
    return SourceArtifact(
        source="sec_edgar_filing",
        source_url=SEC_URL,
        retrieved_at=SEC_RETRIEVED_AT,
        content=content,
        content_type="text/plain",
    )


def _decode_successor_bundle(cache_root: Path) -> Mapping[str, Any]:
    path = cache_root / SUCCESSOR_BUNDLE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Pinned EODHD successor bundle is absent: {path}")
    encoded = path.read_bytes()
    if (
        len(encoded) != SUCCESSOR_BUNDLE_EXACT_BYTES
        or sha256_bytes(encoded) != SUCCESSOR_BUNDLE_SHA256
    ):
        raise ValueError("Pinned EODHD successor bundle hash/size changed.")
    try:
        wrapper = json.loads(gzip.decompress(encoded))
        payload = base64.b64decode(wrapper["payload_base64"], validate=True)
    except Exception as exc:
        raise ValueError("Pinned EODHD successor bundle is unreadable.") from exc
    if (
        _text(wrapper.get("payload_sha256")) != SUCCESSOR_PAYLOAD_SHA256
        or len(payload) != SUCCESSOR_PAYLOAD_EXACT_BYTES
        or sha256_bytes(payload) != SUCCESSOR_PAYLOAD_SHA256
    ):
        raise ValueError("Pinned EODHD successor payload hash/size changed.")
    decoded = json.loads(payload)
    if len(decoded.get("artifacts", ())) != SUCCESSOR_ARTIFACT_COUNT:
        raise ValueError("Pinned EODHD successor artifact inventory changed.")
    return decoded


def _bundle_artifact(
    payload: Mapping[str, Any],
    *,
    source: str,
    source_url: str,
    source_hash: str,
    exact_bytes: int,
) -> SourceArtifact:
    matches = [
        item
        for item in payload.get("artifacts", ())
        if _text(item.get("source")) == source
        and _text(item.get("source_url")) == source_url
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one exact {source} FBIN artifact; observed={len(matches)}"
        )
    row = matches[0]
    try:
        content = base64.b64decode(row["content_base64"], validate=True)
    except Exception as exc:
        raise ValueError(f"Pinned {source} FBIN artifact is unreadable.") from exc
    if len(content) != exact_bytes or sha256_bytes(content) != source_hash:
        raise ValueError(f"Pinned {source} FBIN artifact hash/size changed.")
    if (
        _text(row.get("retrieved_at")) != FBIN_RETRIEVED_AT
        or _text(row.get("content_type")) != "application/json"
    ):
        raise ValueError(f"Pinned {source} FBIN artifact metadata changed.")
    return SourceArtifact(
        source=source,
        source_url=source_url,
        retrieved_at=FBIN_RETRIEVED_AT,
        content=content,
        content_type="application/json",
    )


def load_evidence(cache_root: Path) -> EvidenceBundle:
    payload = _decode_successor_bundle(cache_root)
    evidence = EvidenceBundle(
        sec=_load_sec_artifact(cache_root),
        eod=_bundle_artifact(
            payload,
            source="eodhd_eod",
            source_url=FBIN_EOD_URL,
            source_hash=FBIN_EOD_SHA256,
            exact_bytes=FBIN_EOD_EXACT_BYTES,
        ),
        dividends=_bundle_artifact(
            payload,
            source="eodhd_div",
            source_url=FBIN_DIV_URL,
            source_hash=FBIN_DIV_SHA256,
            exact_bytes=FBIN_DIV_EXACT_BYTES,
        ),
        splits=_bundle_artifact(
            payload,
            source="eodhd_splits",
            source_url=FBIN_SPLITS_URL,
            source_hash=FBIN_SPLITS_SHA256,
            exact_bytes=FBIN_SPLITS_EXACT_BYTES,
        ),
    )
    split_payload = json.loads(evidence.splits.content)
    if split_payload != [
        {"date": TRANSITION_DATE, "split": "117.000000/100.000000"}
    ]:
        raise ValueError("Pinned FBIN provider split payload changed.")
    return evidence


def _official_ticker_action(evidence: SourceArtifact) -> dict[str, Any]:
    return {
        "event_id": canonical_lifecycle_event_id(
            CANONICAL_SECURITY_ID, "ticker_change", TRANSITION_DATE
        ),
        "security_id": CANONICAL_SECURITY_ID,
        "action_type": "ticker_change",
        "effective_date": TRANSITION_DATE,
        "ex_date": TRANSITION_DATE,
        "announcement_date": "2022-12-16",
        "record_date": "",
        "payment_date": "",
        "cash_amount": None,
        "ratio": None,
        "currency": "USD",
        "new_security_id": CANONICAL_SECURITY_ID,
        "new_symbol": NEW_SYMBOL,
        "official": True,
        "source_url": evidence.source_url,
        "source_kind": OFFICIAL_SOURCE_KIND,
        "source": OFFICIAL_SOURCE,
        "retrieved_at": evidence.retrieved_at,
        "source_hash": evidence.source_hash,
    }


REVIEWED_NONTERMINAL_EXTRACTION = {
    key: _official_ticker_action(
        SourceArtifact(
            source="sec_edgar_filing",
            source_url=SEC_URL,
            retrieved_at=SEC_RETRIEVED_AT,
            content=b"",
            content_type="text/plain",
        )
    )[key]
    for key in (
        "event_id",
        "security_id",
        "action_type",
        "effective_date",
        "new_security_id",
        "new_symbol",
        "ratio",
        "cash_amount",
        "currency",
        "source_kind",
        "source_url",
    )
}
REVIEWED_NONTERMINAL_EXTRACTION["source_hash"] = SEC_SHA256


def _one_source_archive_row(
    source_archive: pd.DataFrame,
    *,
    source_url: str,
    source_hash: str,
) -> pd.Series:
    matches = source_archive.loc[
        source_archive["source_url"].astype(str).eq(source_url)
        & source_archive["source_hash"].astype(str).eq(source_hash)
        & source_archive["archive_id"].astype(str).eq(source_hash)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one exact source_archive row for "
            f"{source_url}/{source_hash}; observed={len(matches)}"
        )
    return matches.iloc[0]


def _safe_archive_path(
    repository: LocalDatasetRepository,
    row: pd.Series,
) -> Path:
    relative = Path(_text(row.get("object_path")))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".gz":
        raise ValueError(f"Unsafe FBHS/FBIN archive object path: {relative}")
    root = repository.root.resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents or not path.is_file():
        raise ValueError(f"Missing/escaping FBHS/FBIN archive object: {relative}")
    return path


def _hash_gzip_payload(
    path: Path,
    *,
    expected_content: bytes | None = None,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with gzip.open(path, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                if expected_content is not None:
                    if chunk != expected_content[size : size + len(chunk)]:
                        raise ValueError(
                            f"FBHS/FBIN archive payload differs from evidence: {path}"
                        )
                digest.update(chunk)
                size += len(chunk)
    except (EOFError, OSError) as exc:
        raise ValueError(f"FBHS/FBIN archive is unreadable: {path}") from exc
    if expected_content is not None and size != len(expected_content):
        raise ValueError(f"FBHS/FBIN archive payload length changed: {path}")
    return digest.hexdigest(), size


def _verify_archive_pair(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    source_url: str,
    source_hash: str,
    expected_content: bytes | None = None,
) -> dict[str, Any]:
    row = _one_source_archive_row(
        archive, source_url=source_url, source_hash=source_hash
    )
    path = _safe_archive_path(repository, row)
    if not path.name.startswith(f"{source_hash}."):
        raise ValueError("FBHS/FBIN archive filename is not content-addressed.")
    observed_hash, observed_size = _hash_gzip_payload(
        path, expected_content=expected_content
    )
    if observed_hash != source_hash:
        raise ValueError(
            "FBHS/FBIN archived source hash changed: "
            f"expected={source_hash}, observed={observed_hash}"
        )
    return {"source_hash": observed_hash, "bytes": observed_size}


def _source_price_frame(evidence: EvidenceBundle) -> pd.DataFrame:
    payload = json.loads(evidence.eod.content)
    if not isinstance(payload, list) or len(payload) != EXPECTED_CANONICAL_PRICE_ROWS:
        raise ValueError("Pinned FBIN EOD row inventory changed.")
    frame = pd.DataFrame(payload).rename(columns={"date": "session"})
    expected_columns = {
        "session",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adjusted_close",
    }
    if set(frame) != expected_columns:
        raise ValueError("Pinned FBIN EOD schema changed.")
    sessions = pd.to_datetime(frame["session"], errors="coerce")
    if (
        sessions.isna().any()
        or sessions.duplicated().any()
        or sessions.min().date().isoformat() != PRICE_START
        or sessions.max().date().isoformat() != PRICE_END
    ):
        raise ValueError("Pinned FBIN EOD session inventory changed.")
    return frame.sort_values("session").reset_index(drop=True)


def _assert_prices_match_evidence(
    canonical_prices: pd.DataFrame,
    evidence: EvidenceBundle,
) -> None:
    source = _source_price_frame(evidence)
    stored = canonical_prices.copy()
    stored["session"] = pd.to_datetime(stored["session"]).dt.date.astype(str)
    stored = stored.sort_values("session").reset_index(drop=True)
    if tuple(stored["session"]) != tuple(source["session"].astype(str)):
        raise ValueError("Stored FBIN sessions differ from exact EODHD evidence.")
    for column in ("open", "high", "low", "close", "volume"):
        left = pd.to_numeric(stored[column], errors="coerce")
        right = pd.to_numeric(source[column], errors="coerce")
        if (
            left.isna().any()
            or right.isna().any()
            or not bool((left.to_numpy() == right.to_numpy()).all())
        ):
            raise ValueError(
                f"Stored FBIN {column} differs from exact EODHD evidence."
            )


def _identity_preflight(
    frames: Mapping[str, pd.DataFrame],
    evidence: EvidenceBundle,
) -> dict[str, Any]:
    disposition = _assert_split_disposition()
    master = frames["security_master"]
    history = frames["symbol_history"]
    prices = frames["daily_price_raw"]
    actions = frames["corporate_actions"]
    factors = frames["adjustment_factors"]

    old_master = _one_row(
        master,
        master["security_id"].astype(str).eq(OLD_SECURITY_ID),
        "legacy FBHS master row",
    )
    canonical_master = _one_row(
        master,
        master["security_id"].astype(str).eq(CANONICAL_SECURITY_ID),
        "canonical FBIN master row",
    )
    if (
        _text(old_master.get("primary_symbol")).upper() != OLD_SYMBOL
        or _text(canonical_master.get("primary_symbol")).upper() != NEW_SYMBOL
        or _date(old_master.get("active_to")) != SOURCE_OLD_LAST_SESSION
        or _date(canonical_master.get("active_to"))
        or _date(old_master.get("active_from")) != PRICE_START
        or _date(canonical_master.get("active_from")) != PRICE_START
    ):
        raise ValueError("FBHS/FBIN source master boundaries changed.")
    old_history = _one_row(
        history,
        history["security_id"].astype(str).eq(OLD_SECURITY_ID)
        & history["symbol"].astype(str).str.upper().eq(OLD_SYMBOL),
        "legacy FBHS symbol history",
    )
    canonical_history = _one_row(
        history,
        history["security_id"].astype(str).eq(CANONICAL_SECURITY_ID)
        & history["symbol"].astype(str).str.upper().eq(NEW_SYMBOL),
        "canonical FBIN symbol history",
    )
    if (
        _date(old_history.get("effective_from")) != HISTORY_START
        or _date(old_history.get("effective_to")) != SOURCE_OLD_LAST_SESSION
        or _date(canonical_history.get("effective_from"))
        != SOURCE_NEW_FIRST_SESSION
        or _date(canonical_history.get("effective_to"))
    ):
        raise ValueError("FBHS/FBIN source symbol-history boundaries changed.")

    old_prices = prices.loc[
        prices["security_id"].astype(str).eq(OLD_SECURITY_ID)
    ].copy()
    canonical_prices = prices.loc[
        prices["security_id"].astype(str).eq(CANONICAL_SECURITY_ID)
    ].copy()
    if len(old_prices) != EXPECTED_OLD_PRICE_ROWS or len(
        canonical_prices
    ) != EXPECTED_CANONICAL_PRICE_ROWS:
        raise ValueError("FBHS/FBIN source price row counts changed.")
    if set(old_prices["source_hash"].astype(str)) != {FBHS_EOD_SHA256}:
        raise ValueError("Legacy FBHS EOD source hash changed.")
    if (
        set(canonical_prices["source_hash"].astype(str)) != {FBIN_EOD_SHA256}
        or set(canonical_prices["source"].astype(str)) != {"eodhd_eod"}
    ):
        raise ValueError("Canonical FBIN EOD provenance changed.")
    _assert_prices_match_evidence(canonical_prices, evidence)
    overlap = old_prices.merge(
        canonical_prices,
        on="session",
        suffixes=("_old", "_canonical"),
        validate="one_to_one",
    )
    close_exact = int(
        pd.to_numeric(overlap["close_old"], errors="coerce").eq(
            pd.to_numeric(overlap["close_canonical"], errors="coerce")
        ).sum()
    )
    if (
        len(overlap) != EXPECTED_OVERLAP_ROWS
        or close_exact != EXPECTED_CLOSE_EXACT_OVERLAP_ROWS
    ):
        raise ValueError("FBHS/FBIN duplicate price-overlap signature changed.")

    old_actions = actions.loc[
        actions["security_id"].astype(str).eq(OLD_SECURITY_ID)
    ].copy()
    canonical_actions = actions.loc[
        actions["security_id"].astype(str).eq(CANONICAL_SECURITY_ID)
    ].copy()
    old_dividends = old_actions.loc[
        old_actions["action_type"].astype(str).eq("cash_dividend")
    ]
    canonical_dividends = canonical_actions.loc[
        canonical_actions["action_type"].astype(str).eq("cash_dividend")
    ]
    split = canonical_actions.loc[
        canonical_actions["event_id"].astype(str).eq(PROVIDER_SPLIT_EVENT_ID)
    ]
    if (
        len(old_actions) != EXPECTED_OLD_DIVIDENDS
        or len(old_dividends) != EXPECTED_OLD_DIVIDENDS
        or len(canonical_dividends) != EXPECTED_CANONICAL_DIVIDENDS
        or len(canonical_actions) != EXPECTED_CANONICAL_DIVIDENDS + 1
        or len(split) != 1
    ):
        raise ValueError("FBHS/FBIN source action inventory changed.")
    split_row = split.iloc[0]
    if not (
        _text(split_row.get("action_type")) == "split"
        and _date(split_row.get("effective_date")) == TRANSITION_DATE
        and float(split_row.get("ratio")) == PROVIDER_SPLIT_RATIO
        and _text(split_row.get("source_hash")) == FBIN_SPLITS_SHA256
        and _text(split_row.get("source_url")) == FBIN_SPLITS_URL
        and not bool(split_row.get("official"))
    ):
        raise ValueError("FBIN provider 117/100 split row changed.")
    if set(old_dividends["source_hash"].astype(str)) != {FBHS_DIV_SHA256}:
        raise ValueError("Legacy FBHS dividend source hash changed.")
    if set(canonical_dividends["source_hash"].astype(str)) != {FBIN_DIV_SHA256}:
        raise ValueError("Canonical FBIN dividend source hash changed.")
    duplicate_dividends = old_dividends.merge(
        canonical_dividends,
        on="effective_date",
        suffixes=("_old", "_canonical"),
        validate="one_to_one",
    )
    if len(duplicate_dividends) != EXPECTED_OLD_DIVIDENDS:
        raise ValueError("FBHS dividends are no longer a subset of FBIN dividends.")
    for column in (
        "ex_date",
        "record_date",
        "payment_date",
        "cash_amount",
        "currency",
    ):
        left = duplicate_dividends[f"{column}_old"].fillna("").astype(str)
        right = duplicate_dividends[f"{column}_canonical"].fillna("").astype(str)
        if not left.equals(right):
            raise ValueError(f"FBHS/FBIN duplicate dividend {column} changed.")
    dividend_payload = json.loads(evidence.dividends.content)
    if (
        len(dividend_payload) != EXPECTED_CANONICAL_DIVIDENDS
        or {item["date"] for item in dividend_payload}
        != set(canonical_dividends["effective_date"].astype(str))
    ):
        raise ValueError("Stored FBIN dividends differ from exact provider evidence.")

    old_factors = factors["security_id"].astype(str).eq(OLD_SECURITY_ID)
    canonical_factors = factors["security_id"].astype(str).eq(
        CANONICAL_SECURITY_ID
    )
    if (
        int(old_factors.sum()) != EXPECTED_OLD_PRICE_ROWS
        or int(canonical_factors.sum()) != EXPECTED_CANONICAL_PRICE_ROWS
    ):
        raise ValueError("FBHS/FBIN source adjustment-factor inventory changed.")

    anchors = frames["index_constituent_anchors"]
    events = frames["index_membership_events"]
    affected_ids = {OLD_SECURITY_ID, CANONICAL_SECURITY_ID}
    target_anchors = anchors.loc[
        anchors["security_id"].astype(str).isin(affected_ids)
    ]
    target_events = events.loc[
        events["security_id"].astype(str).isin(affected_ids)
    ].sort_values("effective_date")
    event_signature = tuple(
        zip(
            target_events["index_id"].astype(str),
            target_events["effective_date"].astype(str),
            target_events["operation"].astype(str),
            target_events["security_id"].astype(str),
        )
    )
    if not target_anchors.empty or event_signature != (
        ("sp500", "2016-06-24", "ADD", OLD_SECURITY_ID),
        ("sp500", "2022-12-19", "REMOVE", OLD_SECURITY_ID),
    ):
        raise ValueError("FBHS/FBIN S&P 500 reference inventory changed.")
    if (
        set(target_events["source"].astype(str)) != {SP500_SOURCE}
        or set(target_events["source_url"].astype(str)) != {SP500_SOURCE_URL}
        or set(target_events["source_hash"].astype(str)) != {SP500_SOURCE_SHA256}
    ):
        raise ValueError("FBHS/FBIN S&P 500 provenance changed.")
    return {
        "provider_split_disposition": disposition,
        "old_master_rows_removed": 1,
        "old_symbol_history_rows_removed": 1,
        "canonical_symbol_history_rows_replaced": 1,
        "old_price_rows_removed": len(old_prices),
        "canonical_price_rows_preserved": len(canonical_prices),
        "overlap_price_rows_removed": len(overlap),
        "close_exact_overlap_rows": close_exact,
        "old_action_rows_removed": len(old_actions),
        "duplicate_dividend_rows_removed": len(duplicate_dividends),
        "old_factor_rows_removed": int(old_factors.sum()),
        "canonical_factor_rows_rebuilt": int(canonical_factors.sum()),
    }


def _rewrite_master_history(
    frames: Mapping[str, pd.DataFrame],
    evidence: SourceArtifact,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = frames["security_master"].copy()
    history = frames["symbol_history"].copy()
    master = master.loc[
        ~master["security_id"].astype(str).eq(OLD_SECURITY_ID)
    ].copy()
    canonical = master["security_id"].astype(str).eq(CANONICAL_SECURITY_ID)
    if int(canonical.sum()) != 1:
        raise ValueError("Canonical FBIN master row disappeared during rewrite.")
    updates = {
        "primary_symbol": NEW_SYMBOL,
        "name": "Fortune Brands Innovations, Inc.",
        "active_from": PRICE_START,
        "active_to": "",
        "source": OFFICIAL_SOURCE,
        "source_url": evidence.source_url,
        "retrieved_at": evidence.retrieved_at,
        "source_hash": evidence.source_hash,
    }
    if "provider_symbol" in master:
        updates["provider_symbol"] = "FBIN.US"
    if "action_provider_symbol" in master:
        updates["action_provider_symbol"] = "FBIN.US"
    for column, value in updates.items():
        master.loc[canonical, column] = value

    old_template = _one_row(
        history,
        history["security_id"].astype(str).eq(OLD_SECURITY_ID),
        "FBHS symbol-history template",
    ).copy()
    new_template = _one_row(
        history,
        history["security_id"].astype(str).eq(CANONICAL_SECURITY_ID),
        "FBIN symbol-history template",
    ).copy()
    rows: list[pd.Series] = []
    for template, symbol, start, end in (
        (old_template, OLD_SYMBOL, HISTORY_START, OLD_LAST_SESSION),
        (new_template, NEW_SYMBOL, TRANSITION_DATE, ""),
    ):
        row = template.copy()
        row["security_id"] = CANONICAL_SECURITY_ID
        row["symbol"] = symbol
        row["effective_from"] = start
        row["effective_to"] = end
        row["source"] = OFFICIAL_SOURCE
        row["source_url"] = evidence.source_url
        row["retrieved_at"] = evidence.retrieved_at
        row["source_hash"] = evidence.source_hash
        rows.append(row)
    affected = history["security_id"].astype(str).isin(
        {OLD_SECURITY_ID, CANONICAL_SECURITY_ID}
    )
    additions = pd.DataFrame(rows)
    history = pd.concat(
        [history.loc[~affected], additions.loc[:, history.columns]],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(list(dataset_spec("symbol_history").primary_key), keep="last")
    return master.reset_index(drop=True), history.reset_index(drop=True)


def _rewrite_prices_actions_factors(
    frames: Mapping[str, pd.DataFrame],
    evidence: SourceArtifact,
    *,
    source_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    disposition = _assert_split_disposition()
    prices = frames["daily_price_raw"].loc[
        ~frames["daily_price_raw"]["security_id"].astype(str).eq(OLD_SECURITY_ID)
    ].copy()
    canonical_price_mask = prices["security_id"].astype(str).eq(
        CANONICAL_SECURITY_ID
    )
    if "source_url" in prices:
        prices.loc[canonical_price_mask, "source_url"] = FBIN_EOD_URL
    prices = prices.drop_duplicates(
        list(dataset_spec("daily_price_raw").primary_key), keep="last"
    )
    actions = frames["corporate_actions"].loc[
        ~frames["corporate_actions"]["security_id"].astype(str).eq(
            OLD_SECURITY_ID
        )
    ].copy()
    if disposition == "remove_pseudo_split":
        actions = actions.loc[
            ~actions["event_id"].astype(str).eq(PROVIDER_SPLIT_EVENT_ID)
        ].copy()
    elif disposition != "retain_provider_adjustment":
        raise AssertionError("Unreachable provider split disposition.")
    actions = pd.concat(
        [actions, pd.DataFrame([_official_ticker_action(evidence)])],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(list(dataset_spec("corporate_actions").primary_key), keep="last")

    factors = frames["adjustment_factors"].loc[
        ~frames["adjustment_factors"]["security_id"].astype(str).isin(
            {OLD_SECURITY_ID, CANONICAL_SECURITY_ID}
        )
    ].copy()
    canonical_prices = prices.loc[
        prices["security_id"].astype(str).eq(CANONICAL_SECURITY_ID)
    ].copy()
    canonical_actions = actions.loc[
        actions["security_id"].astype(str).eq(CANONICAL_SECURITY_ID)
    ].copy()
    rebuilt = build_adjustment_factors(
        canonical_prices,
        canonical_actions,
        source_version=source_version,
    )
    factors = pd.concat([factors, rebuilt], ignore_index=True, sort=False).drop_duplicates(
        list(dataset_spec("adjustment_factors").primary_key), keep="last"
    )
    return (
        prices.sort_values(["security_id", "session"]).reset_index(drop=True),
        actions.sort_values(
            ["security_id", "effective_date", "event_id"]
        ).reset_index(drop=True),
        factors.sort_values(["security_id", "session"]).reset_index(drop=True),
    )


def _remapped_index_event_id(row: Mapping[str, Any]) -> str:
    return sha256_bytes(
        _canonical_json_bytes(
            {
                "operation": "fbhs_fbin_index_identity_remap/v1",
                "prior_event_id": _text(row.get("event_id")),
                "index_id": _text(row.get("index_id")),
                "effective_date": _date(row.get("effective_date")),
                "membership_operation": _text(row.get("operation")).upper(),
                "security_id": CANONICAL_SECURITY_ID,
            }
        )
    )


def _rewrite_index_references(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    anchors = frames["index_constituent_anchors"].copy()
    events = frames["index_membership_events"].copy()
    anchor_mask = anchors["security_id"].astype(str).eq(OLD_SECURITY_ID)
    event_mask = events["security_id"].astype(str).eq(OLD_SECURITY_ID)
    provenance_columns = tuple(
        column
        for column in (
            "official",
            "source",
            "source_url",
            "source_kind",
            "retrieved_at",
            "source_hash",
        )
        if column in events.columns
    )
    prior_event_provenance = events.loc[event_mask, provenance_columns].copy()
    anchors.loc[anchor_mask, "security_id"] = CANONICAL_SECURITY_ID
    for index in events.index[event_mask]:
        prior = events.loc[index].to_dict()
        events.loc[index, "security_id"] = CANONICAL_SECURITY_ID
        events.loc[index, "event_id"] = _remapped_index_event_id(prior)
    if not events.loc[event_mask, provenance_columns].equals(prior_event_provenance):
        raise ValueError("FBHS/FBIN index provenance changed during rekey.")
    for frame, dataset in (
        (anchors, "index_constituent_anchors"),
        (events, "index_membership_events"),
    ):
        if frame.duplicated(list(dataset_spec(dataset).primary_key), keep=False).any():
            raise ValueError(f"FBHS/FBIN {dataset} rekey collides with an existing row.")
    return anchors.reset_index(drop=True), events.reset_index(drop=True), {
        "index_anchor_rows_rekeyed": int(anchor_mask.sum()),
        "index_event_rows_rekeyed": int(event_mask.sum()),
    }


def _archive_extension(artifact: SourceArtifact) -> str:
    content_type = artifact.content_type.lower().split(";", 1)[0].strip()
    if content_type == "application/json":
        return "json"
    if content_type == "text/plain":
        return "txt"
    return "bin"


def _append_source_archive(
    source_archive: pd.DataFrame,
    artifacts: Iterable[SourceArtifact],
    *,
    completed_session: str,
) -> pd.DataFrame:
    rows = [
        {
            "archive_id": artifact.source_hash,
            "dataset": artifact.source,
            "object_path": (
                f"archives/{completed_session}/{artifact.source_hash}."
                f"{_archive_extension(artifact)}.gz"
            ),
            "content_type": artifact.content_type,
            "effective_date": completed_session,
            "source": artifact.source,
            "source_url": artifact.source_url,
            "retrieved_at": artifact.retrieved_at,
            "source_hash": artifact.source_hash,
        }
        for artifact in artifacts
    ]
    output = pd.concat(
        [source_archive, pd.DataFrame(rows)], ignore_index=True, sort=False
    )
    return output.drop_duplicates(
        list(dataset_spec("source_archive").primary_key), keep="last"
    ).reset_index(drop=True)


def _archive_pair_exists(
    archive: pd.DataFrame,
    source_url: str,
    source_hash: str,
) -> bool:
    return bool(
        (
            archive["source_url"].astype(str).eq(source_url)
            & archive["source_hash"].astype(str).eq(source_hash)
            & archive["archive_id"].astype(str).eq(source_hash)
        ).any()
    )


def _validate_preexisting_archives(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
) -> dict[str, Any]:
    observed = {
        "fbhs_eod": _verify_archive_pair(
            repository,
            archive,
            source_url=FBHS_EOD_URL,
            source_hash=FBHS_EOD_SHA256,
        ),
        "fbhs_dividends": _verify_archive_pair(
            repository,
            archive,
            source_url=FBHS_DIV_URL,
            source_hash=FBHS_DIV_SHA256,
        ),
        "sp500_membership": _verify_archive_pair(
            repository,
            archive,
            source_url=SP500_SOURCE_URL,
            source_hash=SP500_SOURCE_SHA256,
        ),
    }
    return {
        "preexisting_archives_verified": True,
        "preexisting_archive_bytes": {
            key: int(value["bytes"]) for key, value in observed.items()
        },
    }


def _target_history_signature(history: pd.DataFrame) -> set[tuple[str, str, str]]:
    target = history.loc[
        history["security_id"].astype(str).eq(CANONICAL_SECURITY_ID)
    ]
    return {
        (
            _text(row.symbol).upper(),
            _date(row.effective_from),
            _date(row.effective_to),
        )
        for row in target.itertuples(index=False)
    }


def validate_repaired_frames(
    frames: Mapping[str, pd.DataFrame],
    evidence: EvidenceBundle,
    *,
    completed_session: str,
    repository: LocalDatasetRepository | None = None,
    require_persisted_archives: bool = False,
) -> dict[str, Any]:
    disposition = _assert_split_disposition()
    missing = sorted(set(WRITE_DATASETS) - set(frames))
    if missing:
        raise ValueError("FBHS/FBIN validation lacks datasets: " + ", ".join(missing))
    for dataset in WRITE_DATASETS:
        validate_dataset(
            dataset,
            frames[dataset],
            incomplete_action_policy="warn",
            completed_session=completed_session,
        ).raise_for_errors()

    for dataset in (
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "corporate_actions",
        "adjustment_factors",
        "index_constituent_anchors",
        "index_membership_events",
    ):
        if frames[dataset]["security_id"].astype(str).eq(OLD_SECURITY_ID).any():
            raise ValueError(f"Retired FBHS ID remains in {dataset}.")

    master = frames["security_master"]
    canonical_master = _one_row(
        master,
        master["security_id"].astype(str).eq(CANONICAL_SECURITY_ID),
        "repaired FBIN master row",
    )
    if not (
        _text(canonical_master.get("primary_symbol")).upper() == NEW_SYMBOL
        and _text(canonical_master.get("provider_symbol")).upper() == "FBIN.US"
        and _text(canonical_master.get("action_provider_symbol")).upper()
        == "FBIN.US"
        and _date(canonical_master.get("active_from")) == PRICE_START
        and not _date(canonical_master.get("active_to"))
        and _text(canonical_master.get("source")) == OFFICIAL_SOURCE
        and _text(canonical_master.get("source_url")) == SEC_URL
        and _text(canonical_master.get("source_hash")) == SEC_SHA256
    ):
        raise ValueError("Canonical FBIN master row is not exact.")

    history = frames["symbol_history"]
    expected_history = {
        (OLD_SYMBOL, HISTORY_START, OLD_LAST_SESSION),
        (NEW_SYMBOL, TRANSITION_DATE, ""),
    }
    if _target_history_signature(history) != expected_history:
        raise ValueError("Canonical FBHS/FBIN symbol-history intervals are not exact.")
    target_history = history.loc[
        history["security_id"].astype(str).eq(CANONICAL_SECURITY_ID)
    ]
    if (
        set(target_history["source"].astype(str)) != {OFFICIAL_SOURCE}
        or set(target_history["source_url"].astype(str)) != {SEC_URL}
        or set(target_history["source_hash"].astype(str)) != {SEC_SHA256}
    ):
        raise ValueError("Canonical FBHS/FBIN history lacks exact SEC provenance.")

    prices = frames["daily_price_raw"]
    canonical_prices = prices.loc[
        prices["security_id"].astype(str).eq(CANONICAL_SECURITY_ID)
    ].copy()
    if len(canonical_prices) != EXPECTED_CANONICAL_PRICE_ROWS:
        raise ValueError("Canonical FBIN price inventory changed after repair.")
    _assert_prices_match_evidence(canonical_prices, evidence)
    if (
        set(canonical_prices["source_hash"].astype(str)) != {FBIN_EOD_SHA256}
        or set(canonical_prices["source_url"].astype(str)) != {FBIN_EOD_URL}
    ):
        raise ValueError("Canonical FBIN prices did not preserve one source basis.")

    actions = frames["corporate_actions"]
    canonical_actions = actions.loc[
        actions["security_id"].astype(str).eq(CANONICAL_SECURITY_ID)
    ]
    dividends = canonical_actions.loc[
        canonical_actions["action_type"].astype(str).eq("cash_dividend")
    ]
    if len(dividends) != EXPECTED_CANONICAL_DIVIDENDS:
        raise ValueError("Canonical FBIN dividend inventory changed after repair.")
    split = canonical_actions.loc[
        canonical_actions["event_id"].astype(str).eq(PROVIDER_SPLIT_EVENT_ID)
    ]
    expected_split_rows = int(disposition == "retain_provider_adjustment")
    if len(split) != expected_split_rows:
        raise ValueError(
            "FBIN provider split outcome differs from reviewed disposition."
        )
    official_id = canonical_lifecycle_event_id(
        CANONICAL_SECURITY_ID, "ticker_change", TRANSITION_DATE
    )
    ticker = _one_row(
        actions,
        actions["event_id"].astype(str).eq(official_id),
        "official FBHS/FBIN ticker change",
    )
    if not (
        _text(ticker.get("security_id")) == CANONICAL_SECURITY_ID
        and _text(ticker.get("action_type")) == "ticker_change"
        and _date(ticker.get("effective_date")) == TRANSITION_DATE
        and _text(ticker.get("new_security_id")) == CANONICAL_SECURITY_ID
        and _text(ticker.get("new_symbol")).upper() == NEW_SYMBOL
        and not _text(ticker.get("ratio"))
        and bool(ticker.get("official"))
        and _text(ticker.get("source_url")) == SEC_URL
        and _text(ticker.get("source_hash")) == SEC_SHA256
    ):
        raise ValueError("Official FBHS/FBIN ticker-change row is not exact.")
    old_event_id = canonical_lifecycle_event_id(
        OLD_SECURITY_ID, "ticker_change", TRANSITION_DATE
    )
    if actions["event_id"].astype(str).eq(old_event_id).any():
        raise ValueError("Retired old-ID FBHS ticker event remains after canonical merge.")

    factors = frames["adjustment_factors"].loc[
        frames["adjustment_factors"]["security_id"].astype(str).eq(
            CANONICAL_SECURITY_ID
        )
    ].copy()
    price_sessions = set(pd.to_datetime(canonical_prices["session"]).dt.date.astype(str))
    factor_sessions = set(pd.to_datetime(factors["session"]).dt.date.astype(str))
    if len(factors) != EXPECTED_CANONICAL_PRICE_ROWS or factor_sessions != price_sessions:
        raise ValueError("Canonical FBIN factors do not exactly cover canonical prices.")
    split_factor = pd.to_numeric(factors["split_factor"], errors="coerce")
    if split_factor.isna().any():
        raise ValueError("Canonical FBIN split factors contain non-numeric values.")
    if disposition == "remove_pseudo_split" and not split_factor.eq(1.0).all():
        raise ValueError("Removed pseudo-split still affects canonical split factors.")
    if disposition == "retain_provider_adjustment" and split_factor.eq(1.0).all():
        raise ValueError("Retained provider adjustment is absent from split factors.")

    anchors = frames["index_constituent_anchors"]
    events = frames["index_membership_events"]
    if anchors["security_id"].astype(str).eq(CANONICAL_SECURITY_ID).any():
        raise ValueError("Unexpected FBIN index anchor appeared during repair.")
    target_events = events.loc[
        events["security_id"].astype(str).eq(CANONICAL_SECURITY_ID)
    ].sort_values("effective_date")
    signature = tuple(
        zip(
            target_events["index_id"].astype(str),
            target_events["effective_date"].astype(str),
            target_events["operation"].astype(str),
        )
    )
    if signature != (
        ("sp500", "2016-06-24", "ADD"),
        ("sp500", "2022-12-19", "REMOVE"),
    ):
        raise ValueError("Canonical FBIN S&P 500 membership continuity changed.")
    if (
        set(target_events["source"].astype(str)) != {SP500_SOURCE}
        or set(target_events["source_url"].astype(str)) != {SP500_SOURCE_URL}
        or set(target_events["source_hash"].astype(str)) != {SP500_SOURCE_SHA256}
    ):
        raise ValueError("Canonical FBIN S&P 500 provenance changed.")

    archive = frames["source_archive"]
    for artifact in evidence.archive_artifacts:
        if not _archive_pair_exists(
            archive, artifact.source_url, artifact.source_hash
        ):
            raise ValueError(
                "FBHS/FBIN exact evidence is not archived: "
                f"{artifact.source_url}/{artifact.source_hash}"
            )
    if require_persisted_archives:
        if repository is None:
            raise ValueError("Persisted archive validation requires a repository.")
        for artifact in evidence.archive_artifacts:
            _verify_archive_pair(
                repository,
                archive,
                source_url=artifact.source_url,
                source_hash=artifact.source_hash,
                expected_content=artifact.content,
            )
        _verify_archive_pair(
            repository,
            archive,
            source_url=SP500_SOURCE_URL,
            source_hash=SP500_SOURCE_SHA256,
        )
    return {
        "canonical_security_id": CANONICAL_SECURITY_ID,
        "retired_security_id": OLD_SECURITY_ID,
        "canonical_price_rows": len(canonical_prices),
        "canonical_dividend_rows": len(dividends),
        "provider_split_rows": len(split),
        "provider_split_disposition": disposition,
        "canonical_factor_rows": len(factors),
        "official_ticker_change_rows": 1,
        "index_event_rows": len(target_events),
        "sec_evidence_sha256": evidence.sec.source_hash,
        "eodhd_eod_evidence_sha256": evidence.eod.source_hash,
        "eodhd_div_evidence_sha256": evidence.dividends.source_hash,
        "eodhd_splits_evidence_sha256": evidence.splits.source_hash,
        "persisted_archives_verified": bool(require_persisted_archives),
        "network_accessed": False,
        "r2_accessed": False,
        "pending_followup": PENDING_FOLLOWUP,
        "final_validation_ready": False,
        "publication_ready": False,
    }


def prepare_repair_frames(
    frames: Mapping[str, pd.DataFrame],
    evidence: EvidenceBundle,
    *,
    completed_session: str,
    source_version: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    missing = sorted(set(WRITE_DATASETS) - set(frames))
    if missing:
        raise ValueError("FBHS/FBIN repair lacks datasets: " + ", ".join(missing))
    preflight = _identity_preflight(frames, evidence)
    master, history = _rewrite_master_history(frames, evidence.sec)
    prices, actions, factors = _rewrite_prices_actions_factors(
        frames, evidence.sec, source_version=source_version
    )
    anchors, events, index_summary = _rewrite_index_references(frames)
    archive = _append_source_archive(
        frames["source_archive"],
        evidence.archive_artifacts,
        completed_session=completed_session,
    )
    rewritten = {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "adjustment_factors": factors,
        "index_constituent_anchors": anchors,
        "index_membership_events": events,
        "source_archive": archive,
    }
    summary = validate_repaired_frames(
        rewritten, evidence, completed_session=completed_session
    )
    return rewritten, {
        **summary,
        **preflight,
        **index_summary,
        "status": "validated_offline_plan",
    }


def _looks_repaired(frames: Mapping[str, pd.DataFrame]) -> bool:
    event_id = canonical_lifecycle_event_id(
        CANONICAL_SECURITY_ID, "ticker_change", TRANSITION_DATE
    )
    return bool(
        not frames["security_master"]["security_id"]
        .astype(str)
        .eq(OLD_SECURITY_ID)
        .any()
        and frames["corporate_actions"]["event_id"].astype(str).eq(event_id).any()
    )


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
        if not version:
            return pd.DataFrame(columns=dataset_spec(dataset).required_columns)
        return self.base.read_frame(dataset, version)


def _capture_pointer_etags(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for dataset in WRITE_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        expected = release.dataset_versions.get(dataset)
        if pointer is None or pointer.version != expected:
            raise RuntimeError(f"FBHS/FBIN release/pointer mismatch: {dataset}")
        output[dataset] = etag
    return output


def prepare_run(repository: LocalDatasetRepository) -> PreparedRepair:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required for FBHS/FBIN repair.")
    missing = sorted(set(WRITE_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks datasets: " + ", ".join(missing))
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in WRITE_DATASETS
    }
    pointer_etags = _capture_pointer_etags(repository, release)
    evidence = load_evidence(repository.root)
    archive_summary = _validate_preexisting_archives(
        repository, frames["source_archive"]
    )
    if _looks_repaired(frames):
        summary = validate_repaired_frames(
            frames,
            evidence,
            completed_session=release.completed_session,
            repository=repository,
            require_persisted_archives=True,
        )
        validate_repository_snapshot(repository).raise_for_errors()
        return PreparedRepair(
            release=release,
            release_etag=release_etag,
            pointer_etags=pointer_etags,
            frames={dataset: frames[dataset].copy() for dataset in WRITE_DATASETS},
            evidence=evidence,
            warnings=tuple(dict.fromkeys((*release.warnings, PENDING_FOLLOWUP))),
            summary={
                **summary,
                **archive_summary,
                "status": "already_repaired",
                "release_version": release.version,
            },
        )
    rewritten, summary = prepare_repair_frames(
        frames,
        evidence,
        completed_session=release.completed_session,
        source_version=f"fbhs-fbin-identity-repair/{release.version}",
    )
    candidate = _CandidateRepository(repository, release.dataset_versions, rewritten)
    validate_repository_snapshot(candidate).raise_for_errors()
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        frames=rewritten,
        evidence=evidence,
        warnings=tuple(dict.fromkeys((*release.warnings, PENDING_FOLLOWUP))),
        summary={
            **summary,
            **archive_summary,
            "release_version": release.version,
        },
    )


def _persist_archive_payloads(
    repository: LocalDatasetRepository,
    artifacts: Iterable[SourceArtifact],
    *,
    completed_session: str,
) -> None:
    for artifact in artifacts:
        path = (
            repository.root
            / "archives"
            / completed_session
            / f"{artifact.source_hash}.{_archive_extension(artifact)}.gz"
        )
        encoded = gzip.compress(artifact.content, mtime=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            try:
                existing = gzip.decompress(path.read_bytes())
            except Exception as exc:
                raise ValueError(
                    f"Existing FBHS/FBIN archive is unreadable: {path}"
                ) from exc
            if existing != artifact.content:
                raise RuntimeError(f"Immutable FBHS/FBIN archive changed: {path}")
        else:
            write_atomic(path, encoded)
        if gzip.decompress(path.read_bytes()) != artifact.content:
            raise RuntimeError(f"FBHS/FBIN archive verification failed: {path}")


@contextmanager
def _exclusive_repository_lock(repository: LocalDatasetRepository):
    path = repository.root / ".locks/market-store-write.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery = repository.root / "recovery"
        pending = tuple(recovery.rglob("*.json")) if recovery.exists() else ()
        if pending:
            raise RuntimeError(
                "A recovery marker blocks FBHS/FBIN writes: "
                + ", ".join(str(item) for item in pending)
            )
        transactions = repository.root / "transactions"
        interrupted: list[Path] = []
        if transactions.exists():
            for item in transactions.rglob("*.json"):
                try:
                    status = _text(json.loads(item.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    interrupted.append(item)
        if interrupted:
            raise RuntimeError(
                "An interrupted transaction blocks FBHS/FBIN writes: "
                + ", ".join(str(item) for item in interrupted)
            )
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, _canonical_json_bytes(dict(value)))


def _restore_transaction(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: Mapping[str, bytes],
    planned_versions: Mapping[str, str],
    committed_release_version: str,
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        current = repository.objects.get("releases/current.json")
        if current.data != old_release_bytes:
            observed = DataRelease.from_bytes(current.data)
            ours = observed.version == committed_release_version or all(
                observed.dataset_versions.get(dataset) == version
                for dataset, version in planned_versions.items()
            )
            if not ours:
                raise RuntimeError(
                    "Unexpected release during FBHS/FBIN rollback: "
                    f"{observed.version}"
                )
            repository.objects.put(
                "releases/current.json", old_release_bytes, if_match=current.etag
            )
    except Exception as exc:
        errors.append(f"releases/current.json: {type(exc).__name__}: {exc}")
    for dataset in reversed(WRITE_DATASETS):
        key = repository.current_key(dataset)
        try:
            current = repository.objects.get(key)
            if current.data != old_pointer_bytes[dataset]:
                pointer = CurrentPointer.from_bytes(current.data)
                if pointer.version != planned_versions[dataset]:
                    raise RuntimeError(
                        "Unexpected FBHS/FBIN pointer during rollback: "
                        f"{pointer.version}"
                    )
                repository.objects.put(
                    key, old_pointer_bytes[dataset], if_match=current.etag
                )
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> dict[str, Any]:
    if prepared.summary["status"] == "already_repaired":
        return dict(prepared.summary)
    if (
        prepared.summary.get("provider_split_disposition")
        != _assert_split_disposition()
    ):
        raise RuntimeError("FBHS/FBIN provider split decision changed after plan.")
    with _exclusive_repository_lock(repository):
        current, current_etag = repository.current_release()
        if (
            current is None
            or current.version != prepared.release.version
            or current_etag != prepared.release_etag
        ):
            raise RuntimeError("Current release changed after FBHS/FBIN preflight.")
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != prepared.release.dataset_versions[dataset]
                or value.etag != prepared.pointer_etags[dataset]
            ):
                raise RuntimeError(
                    f"FBHS/FBIN pointer changed before apply: {dataset}"
                )
            old_pointers[dataset] = value.data

        transaction_id = uuid.uuid4().hex
        session = prepared.release.completed_session.replace("-", "")
        planned = {
            dataset: f"fbhs-fbin-identity-{session}-{transaction_id}-{dataset}"
            for dataset in WRITE_DATASETS
        }
        journal_path = (
            repository.root
            / "transactions/fbhs-fbin-identity-repair"
            / f"{transaction_id}.json"
        )
        journal: dict[str, Any] = {
            "schema": "fbhs_fbin_identity_repair_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                dataset: base64.b64encode(value).decode("ascii")
                for dataset, value in old_pointers.items()
            },
            "planned_versions": planned,
            "provider_split_disposition": _assert_split_disposition(),
            "pending_followup": PENDING_FOLLOWUP,
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        try:
            _persist_archive_payloads(
                repository,
                prepared.evidence.archive_artifacts,
                completed_session=prepared.release.completed_session,
            )
            versions = dict(prepared.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    prepared.frames[dataset],
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy="warn",
                    metadata={
                        "operation": "repair_us_fbhs_fbin_identity",
                        "canonical_security_id": CANONICAL_SECURITY_ID,
                        "retired_security_id": OLD_SECURITY_ID,
                        "transition_date": TRANSITION_DATE,
                        "sec_evidence_sha256": SEC_SHA256,
                        "eodhd_eod_evidence_sha256": FBIN_EOD_SHA256,
                        "eodhd_div_evidence_sha256": FBIN_DIV_SHA256,
                        "eodhd_splits_evidence_sha256": FBIN_SPLITS_SHA256,
                        "provider_split_disposition": _assert_split_disposition(),
                        "pending_followup": PENDING_FOLLOWUP,
                        "network_accessed": False,
                        "r2_accessed": False,
                    },
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"FBHS/FBIN write conflicted: {dataset}/{result.conflict_path}"
                    )
                versions[dataset] = result.manifest.version
            written = {
                dataset: repository.read_frame(dataset, planned[dataset])
                for dataset in WRITE_DATASETS
            }
            validate_repaired_frames(
                written,
                prepared.evidence,
                completed_session=prepared.release.completed_session,
                repository=repository,
                require_persisted_archives=True,
            )
            candidate = _CandidateRepository(repository, versions, written)
            validate_repository_snapshot(candidate).raise_for_errors()
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=DataQuality.DEGRADED,
                warnings=prepared.warnings,
                expected_etag=prepared.release_etag,
            )
            latest, _ = repository.current_release()
            if latest is None or latest.to_bytes() != committed.to_bytes():
                raise RuntimeError("Committed FBHS/FBIN release is not current.")
            replay = prepare_run(repository)
            if replay.summary.get("status") != "already_repaired":
                raise RuntimeError("FBHS/FBIN post-commit idempotence check failed.")
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
                "status": "applied_pending_mbc_followup",
                "transaction_id": transaction_id,
                "new_release_version": committed.version,
                "quality": committed.quality,
                "warnings": list(committed.warnings),
                "idempotence_status": replay.summary["status"],
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_versions=planned,
                committed_release_version=(committed.version if committed else ""),
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
                recovery = (
                    repository.root
                    / "recovery/fbhs-fbin-identity-repair"
                    / f"{transaction_id}.json"
                )
                _write_journal(recovery, journal)
                raise RuntimeError(
                    "FBHS/FBIN rollback failed; recovery marker blocks writes: "
                    f"{recovery}; errors={rollback_errors}"
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair the FBHS -> FBIN continuous NYSE common-stock identity."
    )
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline-plan", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def run(
    args: argparse.Namespace,
    *,
    repository_factory: Callable[[str | Path], LocalDatasetRepository] = (
        LocalDatasetRepository
    ),
) -> dict[str, Any]:
    repository = repository_factory(args.cache_root)
    prepared = prepare_run(repository)
    if not bool(getattr(args, "apply", False)):
        return prepared.summary
    return apply_repair(repository, prepared)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
