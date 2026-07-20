#!/usr/bin/env python3
"""Collect and repair the LMCA/LMCK -> FWONA/FWONK identity transition.

Liberty Media's Series A and Series C tracking shares did not become new
securities when the legal name change took effect on 2017-01-24.  The retired
Nasdaq tickers remained the market-data identity for that session; the FWON
tickers began on 2017-01-25.  This tool therefore keeps the two existing
persistent ``security_id`` values and:

* preserves every LMCA/LMCK row through 2017-01-24;
* closes LMCA/LMCK symbol history on 2017-01-24;
* opens FWONA/FWONK symbol history on 2017-01-25;
* appends FWONA/FWONK history beginning exactly on 2017-01-25;
* adds ticker-change actions dated 2017-01-25 with no adjustment ratio;
* rebuilds adjustment factors; and
* archives all six EODHD endpoint responses plus both reviewed SEC filings.

Provider HTTP is opt-in and capped at exactly six one-shot attempts: EOD,
dividends and splits for each of FWONA and FWONK.  Cache-only plan, replay and
apply construct no HTTP client.  Apply is protected by a repository-wide lock,
compare-and-swap pointers, a rollback journal and an idempotent repaired-state
check.  The script never uploads to R2.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import html
import json
import os
import re
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.env import load_env
from supertrend_quant.market_store.ingest import (
    EodhdClient,
    SourceArtifact,
)
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


LEGAL_EFFECTIVE_DATE = "2017-01-24"
TRADING_TRANSITION_DATE = "2017-01-25"
# Public compatibility name used by the extraction handoff and tests.  It is
# deliberately the trading/symbol-history boundary, not the legal charter date.
TRANSITION_DATE = TRADING_TRANSITION_DATE
OLD_PREVIOUS_SESSION = "2017-01-23"
OLD_LAST_SESSION = LEGAL_EFFECTIVE_DATE
FETCH_START = TRADING_TRANSITION_DATE
EODHD_ENDPOINTS = ("eod", "div", "splits")
MAX_EODHD_HTTP_ATTEMPTS = 6
MAX_OFFICIAL_HTTP_ATTEMPTS = 1
MINIMUM_SESSION_COVERAGE = 0.98
MAXIMUM_TRANSITION_RETURN = 0.20

ACTIVE_CATALOG_URL = "https://eodhd.com/api/exchange-symbol-list/US?delisted=0"
ACTIVE_CATALOG_SHA256 = (
    "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99"
)

# The Jan. 24 Rule 424(b)(3) filing is the exact market-boundary source.  It
# says the current charter was filed that day and gave effect to the one-for-one
# name change, while FWONK remained listed under LMCK and was expected to begin
# trading as FWONK on Jan. 25.  Its raw SHA must be observed once, reviewed and
# code-pinned before cache-only plan/apply may proceed.
MARKET_BOUNDARY_URL = (
    "https://www.sec.gov/Archives/edgar/data/1560385/"
    "000104746917000332/a2230745z424b3.htm"
)
MARKET_BOUNDARY_SHA256 = (
    "6a4fe3ee6fea801819f375c2c4426cfb3b619e659dbe93ae5cfdcfe6d4cc45ce"
)

# This already-cached Jan. 17 Form 8-K directly records shareholder approval
# of the exact one-share-for-one-corresponding-series reclassification solely
# to effect the group/share name change.  It is legal-terms evidence, not the
# Jan. 25 trading-boundary source.
LEGAL_TERMS_URL = (
    "https://www.sec.gov/Archives/edgar/data/1560385/"
    "000156038517000002/0001560385-17-000002.txt"
)
LEGAL_TERMS_SHA256 = (
    "1d6f1e8ec946b87af377cc6e5213f0eab5764e62f0c9afae4bafdd4797582b27"
)

OFFICIAL_SOURCE = "formula_one_official_identity_repair"
OFFICIAL_SOURCE_KIND = "official_filing"
EODHD_CACHE_SCHEMA = "formula_one_eodhd_raw/v1"
OFFICIAL_CACHE_SCHEMA = "formula_one_official_evidence_raw/v1"


@dataclass(frozen=True)
class FormulaOneLineage:
    old_symbol: str
    new_symbol: str
    security_id: str
    isin: str
    old_eod_url: str
    old_eod_sha256: str
    name_tokens: tuple[str, ...]

    @property
    def provider_symbol(self) -> str:
        return f"{self.new_symbol}.US"

    @property
    def forbidden_new_security_id(self) -> str:
        import uuid as _uuid

        value = f"eodhd:US:{self.new_symbol}:symbol:{self.new_symbol}"
        return f"US:EODHD:{_uuid.uuid5(_uuid.NAMESPACE_URL, value)}"


LINEAGES: tuple[FormulaOneLineage, ...] = (
    FormulaOneLineage(
        old_symbol="LMCA",
        new_symbol="FWONA",
        security_id="US:EODHD:6c98b8f3-f222-5def-92e5-a0633c3f0775",
        isin="US5312297717",
        old_eod_url=(
            "https://eodhd.com/api/eod/LMCA.US?from=2015-01-01&to=2026-07-15"
        ),
        old_eod_sha256=(
            "a24b16dbdab994f25eb215c2214d242d487680633e9c08e7e9ad770b1d1edaf4"
        ),
        name_tokens=("liberty", "formula one", "series a"),
    ),
    FormulaOneLineage(
        old_symbol="LMCK",
        new_symbol="FWONK",
        security_id="US:EODHD:8e7e0713-31d7-55a7-8878-74ba653d9090",
        isin="US5312297550",
        old_eod_url=(
            "https://eodhd.com/api/eod/LMCK.US?from=2015-01-01&to=2026-07-15"
        ),
        old_eod_sha256=(
            "b4340db7ba9299755d99a7fbe24f8de1b1f22d7505ff4b90c2eb1a7fe9815c68"
        ),
        name_tokens=("liberty", "formula one", "series c"),
    ),
)

if len(LINEAGES) * len(EODHD_ENDPOINTS) != MAX_EODHD_HTTP_ATTEMPTS:
    raise RuntimeError("Formula One EODHD inventory changed without a cap audit.")


WRITE_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "corporate_actions",
    "adjustment_factors",
    "source_archive",
)


def _reviewed_extraction(lineage: FormulaOneLineage) -> dict[str, Any]:
    return {
        "event_id": canonical_lifecycle_event_id(
            lineage.security_id, "ticker_change", TRANSITION_DATE
        ),
        "security_id": lineage.security_id,
        "action_type": "ticker_change",
        "effective_date": TRANSITION_DATE,
        "new_security_id": lineage.security_id,
        "new_symbol": lineage.new_symbol,
        # ticker_change rows intentionally carry no economic ratio.  The exact
        # one-for-one reclassification remains a separately asserted official
        # evidence claim and transition bridge, not an adjustment instruction.
        "ratio": None,
        "cash_amount": None,
        "currency": "USD",
        "source_kind": OFFICIAL_SOURCE_KIND,
        "source_url": MARKET_BOUNDARY_URL,
        "source_hash": MARKET_BOUNDARY_SHA256,
    }


# Kept here as a code-side handoff to the independent R2 cross-validation
# policy.  Publication must contain these exact twelve-field entries.
REVIEWED_NONTERMINAL_EXTRACTIONS = tuple(
    _reviewed_extraction(lineage) for lineage in LINEAGES
)


@dataclass(frozen=True)
class CatalogProof:
    rows: dict[str, dict[str, Any]]
    artifact: SourceArtifact


@dataclass(frozen=True)
class OfficialEvidence:
    legal_terms: SourceArtifact
    market_boundary: SourceArtifact
    http_attempts: int = 0
    missing_reviewed_claims: tuple[str, ...] = ()

    @property
    def artifacts(self) -> tuple[SourceArtifact, ...]:
        return (self.legal_terms, self.market_boundary)


@dataclass(frozen=True)
class ProviderBundle:
    prices: pd.DataFrame
    corporate_actions: pd.DataFrame
    artifacts: tuple[SourceArtifact, ...]
    http_attempts: int
    budget_claims: tuple[int, ...] = ()


@dataclass
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: dict[str, str | None]
    frames: dict[str, pd.DataFrame]
    archive_artifacts: tuple[SourceArtifact, ...]
    warnings: tuple[str, ...]
    summary: dict[str, Any]


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _date(value: Any) -> str:
    if not _text(value):
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid Formula One date: {value!r}")
    return pd.Timestamp(parsed).date().isoformat()


def _normalized_document_text(content: bytes) -> str:
    decoded = content.decode("utf-8", errors="replace")
    decoded = html.unescape(decoded)
    decoded = re.sub(r"<[^>]+>", " ", decoded)
    return re.sub(r"\s+", " ", decoded).strip().lower()


def _near(value: str, left: str, right: str, *, distance: int) -> bool:
    patterns = (
        rf"{left}.{{0,{distance}}}{right}",
        rf"{right}.{{0,{distance}}}{left}",
    )
    return any(re.search(pattern, value) is not None for pattern in patterns)


def _one_for_one_corresponding_series(value: str) -> bool:
    return bool(
        re.search(
            r"\beach share\b.{0,800}\bone share\b.{0,800}"
            r"\bcorresponding series\b",
            value,
        )
        or re.search(
            r"\bcorresponding series\b.{0,800}\bone share\b.{0,800}"
            r"\beach share\b",
            value,
        )
    )


def _official_claim_gaps(
    *,
    legal_terms: bytes,
    market_boundary: bytes,
) -> tuple[str, ...]:
    """Return missing claims from the reviewed two-filing SEC composite."""

    legal = _normalized_document_text(legal_terms)
    boundary = _normalized_document_text(market_boundary)
    checks = {
        "legal_approval_2017-01-17": _near(
            legal, r"\bapproved\b", r"\bjanuary 17, 2017\b", distance=500
        ),
        "legal_one_for_one_corresponding_series": (
            _one_for_one_corresponding_series(legal)
        ),
        "legal_solely_name_change": _near(
            legal, r"\bsolely\b", r"\bname change\b", distance=500
        ),
        "market_charter_effective_2017-01-24": bool(
            re.search(
                r"\bcurrent charter\b.{0,500}\bfiled\b.{0,500}"
                r"\bjanuary 24, 2017\b.{0,300}\bgave effect\b",
                boundary,
            )
        ),
        "market_one_for_one_corresponding_series": (
            _one_for_one_corresponding_series(boundary)
        ),
        "market_solely_name_change": _near(
            boundary, r"\bsolely\b", r"\bname change\b", distance=500
        ),
        "market_lmck_to_fwonk_2017-01-25": bool(
            re.search(
                r"\bshares of fwonk\b.{0,700}\bsymbol\b.{0,100}\blmck\b"
                r".{0,700}\btrade\b.{0,200}\bsymbol\b.{0,100}\bfwonk\b"
                r".{0,300}\bjanuary 25, 2017\b",
                boundary,
            )
        ),
    }
    return tuple(claim for claim, passed in checks.items() if not passed)


def _pinned_official_hash() -> str:
    value = _text(MARKET_BOUNDARY_SHA256).lower()
    if not (
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(
            "Formula One Jan. 24 424(b)(3) SHA-256 is not code-pinned. Run only "
            "the explicit --fetch-official-evidence acquisition, review its "
            "observed SHA/claims, and apply_patch MARKET_BOUNDARY_SHA256 before "
            "plan or apply."
        )
    return value


class FormulaOneOfficialEvidenceSource:
    """Two-file SEC evidence with one explicit no-retry network request."""

    def __init__(
        self,
        root: Path,
        *,
        allow_http: bool,
        legal_cache_root: Path | None = None,
        opener: Callable[..., Any] = urlopen,
    ):
        load_env()
        self.root = Path(root)
        self.legal_cache_root = (
            Path(legal_cache_root)
            if legal_cache_root is not None
            else self.root.parent / "sec_lifecycle"
        )
        self.allow_http = bool(allow_http)
        self.opener = opener
        self.http_attempts = 0

    @property
    def path(self) -> Path:
        return self.root / f"{sha256_bytes(MARKET_BOUNDARY_URL.encode())}.json.gz"

    @property
    def legal_terms_path(self) -> Path:
        key = sha256_bytes((LEGAL_TERMS_URL + "?").encode())
        return self.legal_cache_root / f"{key}.bin"

    def _load_legal_terms(self) -> SourceArtifact:
        path = self.legal_terms_path
        if not path.is_file():
            raise FileNotFoundError(
                "Pinned cached Formula One legal-terms filing is missing: "
                f"{path}"
            )
        content = path.read_bytes()
        observed = sha256_bytes(content)
        if observed != LEGAL_TERMS_SHA256:
            raise ValueError(
                "Pinned cached Formula One legal-terms filing hash changed: "
                f"expected={LEGAL_TERMS_SHA256}, observed={observed}"
            )
        retrieved_at = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
        return SourceArtifact(
            source="sec_edgar_filing",
            source_url=LEGAL_TERMS_URL,
            retrieved_at=retrieved_at,
            content=content,
            content_type="text/plain",
        )

    def _decode(self, payload: bytes) -> SourceArtifact:
        try:
            envelope = json.loads(gzip.decompress(payload))
            content = base64.b64decode(envelope["content_base64"], validate=True)
        except Exception as exc:
            raise ValueError(
                f"Formula One market-boundary cache is unreadable: {self.path}"
            ) from exc
        if (
            envelope.get("schema") != OFFICIAL_CACHE_SCHEMA
            or envelope.get("source_url") != MARKET_BOUNDARY_URL
            or envelope.get("source_hash") != sha256_bytes(content)
        ):
            raise ValueError("Formula One market-boundary cache identity/hash mismatch.")
        return SourceArtifact(
            source="sec_edgar_filing",
            source_url=MARKET_BOUNDARY_URL,
            retrieved_at=_text(envelope.get("retrieved_at")),
            content=content,
            content_type=_text(envelope.get("content_type")) or "text/html",
        )

    def get(self) -> SourceArtifact | None:
        return self._decode(self.path.read_bytes()) if self.path.is_file() else None

    def _fetch_once(self) -> SourceArtifact:
        if not self.allow_http:
            raise FileNotFoundError(
                "Formula One exact Jan. 24 424(b)(3) cache is missing; only the "
                "explicit --fetch-official-evidence acquisition may request it."
            )
        if self.http_attempts >= MAX_OFFICIAL_HTTP_ATTEMPTS:
            raise RuntimeError("Formula One official evidence request cap reached.")
        user_agent = os.getenv("SEC_USER_AGENT", "").strip()
        if not user_agent:
            raise RuntimeError(
                "SEC_USER_AGENT is required for the one-shot Formula One filing request."
            )
        request = Request(
            MARKET_BOUNDARY_URL,
            headers={
                "User-Agent": user_agent,
                "Accept-Encoding": "identity",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        self.http_attempts += 1
        try:
            with self.opener(request, timeout=60) as response:
                status = int(getattr(response, "status", 200))
                content_type = _text(
                    getattr(response, "headers", {}).get("Content-Type")
                ) or "text/html"
                content = response.read(50 * 1024 * 1024 + 1)
        except HTTPError as exc:
            raise RuntimeError(
                f"Formula One official evidence single request failed: HTTP {exc.code}"
            ) from None
        except URLError as exc:
            raise RuntimeError(
                f"Formula One official evidence single request failed: {exc.reason}"
            ) from None
        except Exception as exc:
            raise RuntimeError(
                "Formula One official evidence single request failed: "
                f"{type(exc).__name__}"
            ) from None
        if status != 200 or not content or len(content) > 50 * 1024 * 1024:
            raise RuntimeError(
                "Formula One official evidence response rejected: "
                f"status={status}, bytes={len(content)}"
            )
        artifact = SourceArtifact(
            source="sec_edgar_filing",
            source_url=MARKET_BOUNDARY_URL,
            retrieved_at=utc_now_iso(),
            content=content,
            content_type=content_type,
        )
        envelope = {
            "schema": OFFICIAL_CACHE_SCHEMA,
            "source_url": MARKET_BOUNDARY_URL,
            "retrieved_at": artifact.retrieved_at,
            "content_type": artifact.content_type,
            "source_hash": artifact.source_hash,
            "content_base64": base64.b64encode(content).decode("ascii"),
        }
        encoded = gzip.compress(_canonical_json_bytes(envelope), mtime=0)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            existing = self._decode(self.path.read_bytes())
            if existing.content != content:
                raise RuntimeError("Immutable Formula One official cache changed.")
            return existing
        write_atomic(self.path, encoded)
        return self._decode(self.path.read_bytes())

    def acquire(self, *, require_pinned: bool) -> OfficialEvidence:
        # Verify the already-cached legal filing before spending the single
        # permitted SEC request on the missing market-boundary filing.
        legal_terms = self._load_legal_terms()
        market_boundary = self.get()
        if market_boundary is None:
            market_boundary = self._fetch_once()
        gaps = _official_claim_gaps(
            legal_terms=legal_terms.content,
            market_boundary=market_boundary.content,
        )
        if require_pinned:
            expected = _pinned_official_hash()
            if market_boundary.source_hash != expected:
                raise ValueError(
                    "Formula One Jan. 24 424(b)(3) exact bytes do not match the "
                    "code-pinned SHA-256."
                )
            if gaps:
                raise ValueError(
                    "Formula One SEC evidence composite lacks directly reviewed "
                    "claims: " + ", ".join(gaps)
                )
        return OfficialEvidence(
            legal_terms=legal_terms,
            market_boundary=market_boundary,
            http_attempts=self.http_attempts,
            missing_reviewed_claims=gaps,
        )


def load_official_evidence(cache_root: Path) -> OfficialEvidence:
    # This path never enables HTTP.  A blank/unreviewed SHA blocks before any
    # EODHD source is constructed in the normal plan/apply workflow.
    _pinned_official_hash()
    source = FormulaOneOfficialEvidenceSource(
        cache_root / "state/formula-one-official-evidence",
        legal_cache_root=cache_root / "state/sec_lifecycle",
        allow_http=False,
    )
    return source.acquire(require_pinned=True)


def _safe_archive_payload(
    repository: LocalDatasetRepository,
    row: Mapping[str, Any],
) -> bytes:
    root = repository.root.resolve()
    path = (root / _text(row.get("object_path"))).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Formula One archive path escapes repository: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Formula One archive payload is missing: {path}")
    encoded = path.read_bytes()
    try:
        content = gzip.decompress(encoded) if path.suffix == ".gz" else encoded
    except Exception as exc:
        raise ValueError(f"Formula One archive payload is unreadable: {path}") from exc
    digest = sha256_bytes(content)
    if (
        digest != _text(row.get("source_hash"))
        or digest != _text(row.get("archive_id"))
    ):
        raise ValueError(f"Formula One archive hash mismatch: {path}")
    return content


def load_catalog_proof(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
) -> CatalogProof:
    matches = archive.loc[
        archive["source_url"].astype(str).eq(ACTIVE_CATALOG_URL)
        & archive["source_hash"].astype(str).eq(ACTIVE_CATALOG_SHA256)
    ]
    if len(matches) != 1:
        raise ValueError("Exact frozen active EODHD catalog archive is missing.")
    row = matches.iloc[0]
    try:
        values = json.loads(_safe_archive_payload(repository, row))
    except json.JSONDecodeError as exc:
        raise ValueError("Frozen active EODHD catalog is invalid JSON.") from exc
    if not isinstance(values, list):
        raise ValueError("Frozen active EODHD catalog has the wrong shape.")
    selected: dict[str, dict[str, Any]] = {}
    for lineage in LINEAGES:
        candidates = [
            value
            for value in values
            if isinstance(value, dict)
            and _text(value.get("Code")).upper() == lineage.new_symbol
            and _text(value.get("Isin")).upper() == lineage.isin
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Frozen catalog identity is not unique: {lineage.new_symbol}/{lineage.isin}"
            )
        value = dict(candidates[0])
        name = _text(value.get("Name")).lower()
        if (
            _text(value.get("Exchange")).upper() != "NASDAQ"
            or _text(value.get("Currency")).upper() != "USD"
            or "stock" not in _text(value.get("Type")).lower()
            or not all(token in name for token in lineage.name_tokens)
        ):
            raise ValueError(f"Frozen catalog row changed: {lineage.new_symbol}")
        selected[lineage.new_symbol] = value
    artifact = SourceArtifact(
        source=_text(row.get("source")) or "eodhd_exchange_symbols",
        source_url=ACTIVE_CATALOG_URL,
        retrieved_at=_text(row.get("retrieved_at")),
        content=_safe_archive_payload(repository, row),
        content_type=_text(row.get("content_type")) or "application/json",
    )
    return CatalogProof(rows=selected, artifact=artifact)


def _validate_old_raw_archives(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    for lineage in LINEAGES:
        matches = archive.loc[
            archive["source_url"].astype(str).eq(lineage.old_eod_url)
            & archive["source_hash"].astype(str).eq(lineage.old_eod_sha256)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Exact archived retired-code EODHD payload is missing: {lineage.old_symbol}"
            )
        raw = json.loads(_safe_archive_payload(repository, matches.iloc[0]))
        if not isinstance(raw, list):
            raise ValueError(f"Retired-code EODHD payload is not a list: {lineage.old_symbol}")
        by_date = {
            _text(value.get("date")): value
            for value in raw
            if isinstance(value, dict) and _text(value.get("date"))
        }
        for session in (OLD_PREVIOUS_SESSION, OLD_LAST_SESSION):
            raw_row = by_date.get(session)
            frame_row = prices.loc[
                prices["security_id"].astype(str).eq(lineage.security_id)
                & pd.to_datetime(prices["session"], errors="coerce").eq(
                    pd.Timestamp(session)
                )
            ]
            if raw_row is None or len(frame_row) != 1:
                raise ValueError(
                    f"Retired-code boundary inventory changed: {lineage.old_symbol}/{session}"
                )
            stored = frame_row.iloc[0]
            if _text(stored.get("source_hash")) != lineage.old_eod_sha256:
                raise ValueError(f"Retired-code row provenance changed: {lineage.old_symbol}")
            for column in ("open", "high", "low", "close", "volume"):
                if not np.isclose(
                    float(stored[column]),
                    float(raw_row[column]),
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"Retired-code archived row mismatch: {lineage.old_symbol}/{session}/{column}"
                    )


class CappedSingleAttemptEodhdClient(EodhdClient):
    """One budget claim and one HTTP attempt per endpoint, with no retry."""

    def __init__(
        self,
        *args,
        max_attempts: int = MAX_EODHD_HTTP_ATTEMPTS,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_attempts = int(max_attempts)
        self._attempt_count = 0
        self._lock = threading.Lock()
        self._budget_claims: list[int] = []

    @property
    def attempt_count(self) -> int:
        with self._lock:
            return self._attempt_count

    @property
    def budget_claims(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._budget_claims)

    def get_json(
        self,
        endpoint: str,
        *,
        params: dict[str, object] | None = None,
    ):
        safe_endpoint = "/" + endpoint.strip("/")
        query = {**(params or {}), "api_token": self.token, "fmt": "json"}
        with self._lock:
            if self._attempt_count >= self.max_attempts:
                raise RuntimeError("Formula One EODHD request cap reached before HTTP.")
            claimed = int(self.budget.claim())
            self._attempt_count += 1
            self._budget_claims.append(claimed)
        try:
            response = self.session.get(
                self.base_url + safe_endpoint,
                params=query,
                timeout=120,
            )
            response.raise_for_status()
            value = response.json()
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            detail = f"HTTP {status}" if status else type(exc).__name__
            raise RuntimeError(
                f"Formula One EODHD single attempt failed for {safe_endpoint}: {detail}"
            ) from None
        if not isinstance(value, list):
            raise RuntimeError(
                f"Formula One EODHD endpoint returned a non-list: {safe_endpoint}"
            )
        return value


class FormulaOneEodhdSource:
    """Six immutable endpoint caches; HTTP is constructed only when opted in."""

    def __init__(
        self,
        root: Path,
        *,
        completed_session: str,
        allow_http: bool,
        client_factory: Callable[[], Any] = CappedSingleAttemptEodhdClient,
    ):
        self.root = Path(root)
        self.completed_session = _date(completed_session)
        self.allow_http = bool(allow_http)
        self.client_factory = client_factory

    @staticmethod
    def _params(completed_session: str) -> dict[str, str]:
        return {"from": FETCH_START, "to": completed_session}

    def public_url(self, endpoint: str, lineage: FormulaOneLineage) -> str:
        return (
            f"https://eodhd.com/api/{endpoint}/{lineage.provider_symbol}?"
            + urlencode(self._params(self.completed_session))
        )

    def path(self, endpoint: str, lineage: FormulaOneLineage) -> Path:
        return self.root / f"{sha256_bytes(self.public_url(endpoint, lineage).encode())}.json.gz"

    def _decode(
        self,
        endpoint: str,
        lineage: FormulaOneLineage,
        payload: bytes,
    ) -> SourceArtifact:
        path = self.path(endpoint, lineage)
        try:
            envelope = json.loads(gzip.decompress(payload))
            content = base64.b64decode(envelope["content_base64"], validate=True)
        except Exception as exc:
            raise ValueError(f"Unreadable Formula One EODHD cache: {path}") from exc
        expected_url = self.public_url(endpoint, lineage)
        if (
            envelope.get("schema") != EODHD_CACHE_SCHEMA
            or envelope.get("endpoint") != endpoint
            or envelope.get("provider_symbol") != lineage.provider_symbol
            or envelope.get("source_url") != expected_url
            or envelope.get("source_hash") != sha256_bytes(content)
        ):
            raise ValueError(f"Formula One EODHD cache identity/hash mismatch: {path}")
        try:
            value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Formula One EODHD cache JSON is invalid: {path}") from exc
        if not isinstance(value, list):
            raise ValueError(f"Formula One EODHD cache payload is not a list: {path}")
        return SourceArtifact(
            source=f"eodhd_{endpoint}",
            source_url=expected_url,
            retrieved_at=_text(envelope.get("retrieved_at")),
            content=content,
            content_type="application/json",
        )

    def get(
        self, endpoint: str, lineage: FormulaOneLineage
    ) -> SourceArtifact | None:
        path = self.path(endpoint, lineage)
        return self._decode(endpoint, lineage, path.read_bytes()) if path.is_file() else None

    def _store(
        self,
        endpoint: str,
        lineage: FormulaOneLineage,
        artifact: SourceArtifact,
    ) -> SourceArtifact:
        path = self.path(endpoint, lineage)
        envelope = {
            "schema": EODHD_CACHE_SCHEMA,
            "endpoint": endpoint,
            "provider_symbol": lineage.provider_symbol,
            "source_url": self.public_url(endpoint, lineage),
            "retrieved_at": artifact.retrieved_at,
            "source_hash": artifact.source_hash,
            "content_base64": base64.b64encode(artifact.content).decode("ascii"),
        }
        encoded = gzip.compress(_canonical_json_bytes(envelope), mtime=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            existing = self._decode(endpoint, lineage, path.read_bytes())
            if existing.content != artifact.content:
                raise RuntimeError(f"Immutable Formula One EODHD cache changed: {path}")
            return existing
        write_atomic(path, encoded)
        return self._decode(endpoint, lineage, path.read_bytes())

    def fetch(self) -> ProviderBundle:
        cached: dict[tuple[str, str], SourceArtifact | None] = {
            (lineage.new_symbol, endpoint): self.get(endpoint, lineage)
            for lineage in LINEAGES
            for endpoint in EODHD_ENDPOINTS
        }
        missing = [key for key, artifact in cached.items() if artifact is None]
        if missing and not self.allow_http:
            raise FileNotFoundError(
                "Formula One EODHD cache is incomplete; explicitly use "
                "--fetch-missing-eodhd for reviewed acquisition: "
                + ", ".join(f"{symbol}/{endpoint}" for symbol, endpoint in missing)
            )
        if len(missing) > MAX_EODHD_HTTP_ATTEMPTS:
            raise RuntimeError("Formula One missing endpoint inventory exceeds the cap.")
        client = self.client_factory() if missing else None
        for lineage in LINEAGES:
            for endpoint in EODHD_ENDPOINTS:
                key = (lineage.new_symbol, endpoint)
                if cached[key] is not None:
                    continue
                rows = client.get_json(
                    f"{endpoint}/{lineage.provider_symbol}",
                    params=self._params(self.completed_session),
                )
                content = _canonical_json_bytes(rows)
                artifact = SourceArtifact(
                    source=f"eodhd_{endpoint}",
                    source_url=self.public_url(endpoint, lineage),
                    retrieved_at=utc_now_iso(),
                    content=content,
                    content_type="application/json",
                )
                cached[key] = self._store(endpoint, lineage, artifact)
        artifacts = tuple(
            cached[(lineage.new_symbol, endpoint)]
            for lineage in LINEAGES
            for endpoint in EODHD_ENDPOINTS
        )
        if any(artifact is None for artifact in artifacts):
            raise RuntimeError("Formula One EODHD cache did not fill completely.")
        attempts = int(getattr(client, "attempt_count", 0)) if client else 0
        claims = tuple(getattr(client, "budget_claims", ())) if client else ()
        if attempts != len(missing) or len(claims) not in {0, attempts}:
            raise RuntimeError("Formula One EODHD attempt/budget accounting changed.")
        return _parse_provider_artifacts(
            tuple(artifact for artifact in artifacts if artifact is not None),
            http_attempts=attempts,
            budget_claims=claims,
        )


def _provider_event(
    lineage: FormulaOneLineage,
    *,
    action_type: str,
    effective_date: str,
    cash_amount: Any,
    ratio: Any,
    artifact: SourceArtifact,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    event_key = (
        f"{artifact.source}|{lineage.security_id}|{action_type}|{effective_date}".encode()
    )
    return {
        "event_id": hashlib.sha256(event_key).hexdigest(),
        "security_id": lineage.security_id,
        "action_type": action_type,
        "effective_date": effective_date,
        "ex_date": effective_date,
        "announcement_date": _text(row.get("declarationDate")),
        "record_date": _text(row.get("recordDate")),
        "payment_date": _text(row.get("paymentDate")),
        "cash_amount": cash_amount,
        "ratio": ratio,
        "currency": _text(row.get("currency")) or "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": False,
        "source_url": artifact.source_url,
        "source_kind": "provider",
        "source": artifact.source,
        "retrieved_at": artifact.retrieved_at,
        "source_hash": artifact.source_hash,
    }


def _split_ratio(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            parsed = float(numerator) / float(denominator)
        else:
            parsed = float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return parsed if parsed > 0 else None


def _parse_provider_artifacts(
    artifacts: tuple[SourceArtifact, ...],
    *,
    http_attempts: int,
    budget_claims: tuple[int, ...],
) -> ProviderBundle:
    by_key: dict[tuple[str, str], SourceArtifact] = {}
    for artifact in artifacts:
        matched = [
            lineage
            for lineage in LINEAGES
            if f"/{lineage.provider_symbol}?" in artifact.source_url
        ]
        endpoint = artifact.source.removeprefix("eodhd_")
        if len(matched) != 1 or endpoint not in EODHD_ENDPOINTS:
            raise ValueError("Formula One EODHD artifact identity is unexpected.")
        key = (matched[0].new_symbol, endpoint)
        if key in by_key:
            raise ValueError("Duplicate Formula One EODHD endpoint artifact.")
        by_key[key] = artifact
    expected = {
        (lineage.new_symbol, endpoint)
        for lineage in LINEAGES
        for endpoint in EODHD_ENDPOINTS
    }
    if set(by_key) != expected:
        raise ValueError("Formula One EODHD artifact inventory is incomplete.")

    prices: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for lineage in LINEAGES:
        decoded: dict[str, list[dict[str, Any]]] = {}
        for endpoint in EODHD_ENDPOINTS:
            artifact = by_key[(lineage.new_symbol, endpoint)]
            value = json.loads(artifact.content)
            if not isinstance(value, list) or not all(
                isinstance(row, dict) for row in value
            ):
                raise ValueError(
                    f"Formula One EODHD {lineage.new_symbol}/{endpoint} payload is invalid."
                )
            decoded[endpoint] = value
        eod_artifact = by_key[(lineage.new_symbol, "eod")]
        for row in decoded["eod"]:
            session = _text(row.get("date"))
            if not session or row.get("close") is None:
                continue
            prices.append(
                {
                    "security_id": lineage.security_id,
                    "session": session,
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": row.get("close"),
                    "volume": row.get("volume", 0),
                    "currency": "USD",
                    "source": eod_artifact.source,
                    "source_url": eod_artifact.source_url,
                    "retrieved_at": eod_artifact.retrieved_at,
                    "source_hash": eod_artifact.source_hash,
                }
            )
        div_artifact = by_key[(lineage.new_symbol, "div")]
        for row in decoded["div"]:
            effective = _text(row.get("date"))
            if not effective:
                continue
            actions.append(
                _provider_event(
                    lineage,
                    action_type="cash_dividend",
                    effective_date=effective,
                    cash_amount=row.get("unadjustedValue", row.get("value")),
                    ratio=None,
                    artifact=div_artifact,
                    row=row,
                )
            )
        split_artifact = by_key[(lineage.new_symbol, "splits")]
        for row in decoded["splits"]:
            effective = _text(row.get("date"))
            ratio = _split_ratio(row.get("split"))
            if not effective or ratio is None:
                continue
            actions.append(
                _provider_event(
                    lineage,
                    action_type="split",
                    effective_date=effective,
                    cash_amount=None,
                    ratio=ratio,
                    artifact=split_artifact,
                    row=row,
                )
            )
    # ``source_url`` is intentionally retained even though it is an optional
    # schema column: the transition gate binds every replacement row to the
    # exact token-free FWONA/FWONK endpoint URL.
    price_columns = tuple(
        dict.fromkeys(
            (*dataset_spec("daily_price_raw").required_columns, "source_url")
        )
    )
    price_frame = pd.DataFrame(prices, columns=price_columns)
    action_frame = pd.DataFrame(
        actions, columns=dataset_spec("corporate_actions").required_columns
    )
    return ProviderBundle(
        prices=price_frame,
        corporate_actions=action_frame,
        artifacts=artifacts,
        http_attempts=int(http_attempts),
        budget_claims=tuple(int(value) for value in budget_claims),
    )


def validate_provider_bundle(
    bundle: ProviderBundle,
    *,
    completed_session: str,
) -> None:
    if len(bundle.artifacts) != MAX_EODHD_HTTP_ATTEMPTS:
        raise ValueError("Formula One provider artifact count is not exactly six.")
    if bundle.http_attempts < 0 or bundle.http_attempts > MAX_EODHD_HTTP_ATTEMPTS:
        raise ValueError("Formula One EODHD attempt count is outside the cap.")
    if bundle.budget_claims and len(bundle.budget_claims) != bundle.http_attempts:
        raise ValueError("Formula One EODHD budget claims do not equal HTTP attempts.")
    validate_dataset(
        "daily_price_raw", bundle.prices, completed_session=completed_session
    ).raise_for_errors()
    if not bundle.corporate_actions.empty:
        validate_dataset(
            "corporate_actions",
            bundle.corporate_actions,
            completed_session=completed_session,
            incomplete_action_policy="block",
        ).raise_for_errors()
    try:
        import exchange_calendars as xcals

        expected_sessions = {
            pd.Timestamp(value).date().isoformat()
            for value in xcals.get_calendar("XNYS").sessions_in_range(
                FETCH_START, completed_session
            )
        }
    except Exception as exc:  # pragma: no cover - runtime dependency is installed
        raise RuntimeError("XNYS calendar is required for Formula One repair.") from exc
    for lineage in LINEAGES:
        rows = bundle.prices.loc[
            bundle.prices["security_id"].astype(str).eq(lineage.security_id)
        ].copy()
        if rows.empty:
            raise ValueError(f"No fetched history for {lineage.new_symbol}.")
        sessions = pd.to_datetime(rows["session"], errors="coerce")
        if sessions.isna().any() or sessions.duplicated().any():
            raise ValueError(f"Invalid/duplicate fetched sessions: {lineage.new_symbol}")
        actual = set(sessions.dt.date.astype(str))
        if min(actual) != FETCH_START or max(actual) != _date(completed_session):
            raise ValueError(f"Fetched boundary is incomplete: {lineage.new_symbol}")
        coverage = len(actual & expected_sessions) / max(1, len(expected_sessions))
        if coverage < MINIMUM_SESSION_COVERAGE:
            raise ValueError(
                f"Fetched session coverage is too low: {lineage.new_symbol}/{coverage:.6f}"
            )
        if not actual.issubset(expected_sessions):
            raise ValueError(f"Fetched non-XNYS sessions: {lineage.new_symbol}")
        if set(rows["source_url"].astype(str)) != {
            next(
                artifact.source_url
                for artifact in bundle.artifacts
                if artifact.source == "eodhd_eod"
                and f"/{lineage.provider_symbol}?" in artifact.source_url
            )
        }:
            raise ValueError(f"Fetched price provenance changed: {lineage.new_symbol}")
    allowed_ids = {lineage.security_id for lineage in LINEAGES}
    if not set(bundle.corporate_actions.get("security_id", pd.Series(dtype=str)).astype(str)).issubset(
        allowed_ids
    ):
        raise ValueError("Formula One provider actions reference an unexpected identity.")


def _official_action(
    lineage: FormulaOneLineage,
    evidence: OfficialEvidence,
) -> dict[str, Any]:
    return {
        "event_id": canonical_lifecycle_event_id(
            lineage.security_id, "ticker_change", TRANSITION_DATE
        ),
        "security_id": lineage.security_id,
        "action_type": "ticker_change",
        "effective_date": TRANSITION_DATE,
        "ex_date": TRANSITION_DATE,
        "announcement_date": "2017-01-17",
        "record_date": "",
        "payment_date": "",
        "cash_amount": None,
        "ratio": None,
        "currency": "USD",
        "new_security_id": lineage.security_id,
        "new_symbol": lineage.new_symbol,
        "official": True,
        "source_url": evidence.market_boundary.source_url,
        "source_kind": OFFICIAL_SOURCE_KIND,
        "source": OFFICIAL_SOURCE,
        "retrieved_at": evidence.market_boundary.retrieved_at,
        "source_hash": evidence.market_boundary.source_hash,
    }


def _concat_unique(
    frames: Iterable[pd.DataFrame],
    *,
    dataset: str,
) -> pd.DataFrame:
    values = [frame for frame in frames if frame is not None and not frame.empty]
    if not values:
        return pd.DataFrame(columns=dataset_spec(dataset).required_columns)
    output = pd.concat(values, ignore_index=True, sort=False)
    return output.drop_duplicates(
        list(dataset_spec(dataset).primary_key), keep="last"
    ).reset_index(drop=True)


def _archive_extension(artifact: SourceArtifact) -> str:
    content_type = artifact.content_type.lower()
    if "json" in content_type:
        return "json"
    if "html" in content_type:
        return "html"
    if "csv" in content_type:
        return "csv"
    if "pdf" in content_type:
        return "pdf"
    return "txt"


def _archive_rows(
    artifacts: Iterable[SourceArtifact],
    *,
    completed_session: str,
) -> pd.DataFrame:
    rows = []
    for artifact in artifacts:
        extension = _archive_extension(artifact)
        rows.append(
            {
                "archive_id": artifact.source_hash,
                "dataset": artifact.source,
                "object_path": (
                    f"archives/{completed_session}/{artifact.source_hash}.{extension}.gz"
                ),
                "content_type": artifact.content_type,
                "effective_date": completed_session,
                "source": artifact.source,
                "source_url": artifact.source_url,
                "retrieved_at": artifact.retrieved_at,
                "source_hash": artifact.source_hash,
            }
        )
    # ``source_url`` is optional in the shared schema but is mandatory for
    # hash-pinned official provenance and cross-source publication gates.
    columns = tuple(
        dict.fromkeys(
            (*dataset_spec("source_archive").required_columns, "source_url")
        )
    )
    return pd.DataFrame(rows, columns=columns)


def _identity_is_repaired(frames: Mapping[str, pd.DataFrame]) -> bool:
    master = frames["security_master"]
    history = frames["symbol_history"]
    prices = frames["daily_price_raw"]
    actions = frames["corporate_actions"]
    archive = frames["source_archive"]
    for lineage in LINEAGES:
        master_rows = master.loc[
            master["security_id"].astype(str).eq(lineage.security_id)
        ]
        if len(master_rows) != 1:
            return False
        row = master_rows.iloc[0]
        if not (
            _text(row.get("primary_symbol")) == lineage.new_symbol
            and _text(row.get("provider_symbol")) == lineage.provider_symbol
            and _date(row.get("active_to")) == ""
        ):
            return False
        intervals = history.loc[
            history["security_id"].astype(str).eq(lineage.security_id)
        ].copy()
        actual = {
            (
                _text(value.symbol),
                _date(value.effective_from),
                _date(value.effective_to),
            )
            for value in intervals.itertuples(index=False)
            if _text(value.symbol) in {lineage.old_symbol, lineage.new_symbol}
        }
        old_start = min(
            (
                _date(value.effective_from)
                for value in intervals.itertuples(index=False)
                if _text(value.symbol) == lineage.old_symbol
            ),
            default="",
        )
        if actual != {
            (lineage.old_symbol, old_start, OLD_LAST_SESSION),
            (lineage.new_symbol, TRANSITION_DATE, ""),
        }:
            return False
        first = prices.loc[
            prices["security_id"].astype(str).eq(lineage.security_id)
            & pd.to_datetime(prices["session"], errors="coerce").eq(
                pd.Timestamp(TRANSITION_DATE)
            )
        ]
        if len(first) != 1 or f"/{lineage.provider_symbol}?" not in _text(
            first.iloc[0].get("source_url")
        ):
            return False
        event_id = canonical_lifecycle_event_id(
            lineage.security_id, "ticker_change", TRANSITION_DATE
        )
        action = actions.loc[actions["event_id"].astype(str).eq(event_id)]
        if len(action) != 1:
            return False
        action_row = action.iloc[0]
        if not (
            _text(action_row.get("new_security_id")) == lineage.security_id
            and _text(action_row.get("new_symbol")) == lineage.new_symbol
            and _text(action_row.get("ratio")) == ""
            and _text(action_row.get("source_url")) == MARKET_BOUNDARY_URL
            and _text(action_row.get("source_hash")) == MARKET_BOUNDARY_SHA256
        ):
            return False
    required_hashes = {MARKET_BOUNDARY_SHA256, LEGAL_TERMS_SHA256}
    return required_hashes.issubset(set(archive["source_hash"].astype(str)))


def prepare_formula_one_repair(
    existing: Mapping[str, pd.DataFrame],
    *,
    catalog: CatalogProof,
    evidence: OfficialEvidence,
    provider: ProviderBundle,
    completed_session: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], tuple[SourceArtifact, ...]]:
    missing = sorted(set(WRITE_DATASETS) - set(existing))
    if missing:
        raise ValueError("Formula One repair is missing frames: " + ", ".join(missing))
    master = existing["security_master"].copy()
    history = existing["symbol_history"].copy()
    prices = existing["daily_price_raw"].copy()
    actions = existing["corporate_actions"].copy()
    factors = existing["adjustment_factors"].copy()
    archive = existing["source_archive"].copy()

    forbidden_ids = {lineage.forbidden_new_security_id for lineage in LINEAGES}
    if set(master["security_id"].astype(str)) & forbidden_ids:
        raise ValueError("FWONA/FWONK were incorrectly assigned new security_ids.")
    validate_provider_bundle(provider, completed_session=completed_session)

    transition_crosschecks: dict[str, dict[str, Any]] = {}
    remove_prices = pd.Series(False, index=prices.index)
    remove_factors = pd.Series(False, index=factors.index)
    new_history_rows: list[dict[str, Any]] = []
    for lineage in LINEAGES:
        master_index = master.index[
            master["security_id"].astype(str).eq(lineage.security_id)
        ]
        old_history_index = history.index[
            history["security_id"].astype(str).eq(lineage.security_id)
            & history["symbol"].astype(str).eq(lineage.old_symbol)
        ]
        if len(master_index) != 1 or len(old_history_index) != 1:
            raise ValueError(f"Formula One old identity is not exact: {lineage.old_symbol}")
        master_row = master.loc[master_index[0]]
        old_history = history.loc[old_history_index[0]]
        if not (
            _text(master_row.get("primary_symbol")) == lineage.old_symbol
            and _text(master_row.get("provider_symbol")) == f"{lineage.old_symbol}.US"
            and _date(master_row.get("active_to")) == LEGAL_EFFECTIVE_DATE
            and _date(old_history.get("effective_to"))
            in {"", LEGAL_EFFECTIVE_DATE}
        ):
            raise ValueError(f"Formula One original identity changed: {lineage.old_symbol}")

        target_prices = prices["security_id"].astype(str).eq(lineage.security_id)
        sessions = pd.to_datetime(prices["session"], errors="coerce")
        if sessions.loc[target_prices].isna().any():
            raise ValueError(f"Invalid old sessions: {lineage.old_symbol}")
        terminal_rows = prices.loc[
            target_prices & sessions.eq(pd.Timestamp(OLD_LAST_SESSION))
        ]
        overrun = target_prices & sessions.ge(pd.Timestamp(TRANSITION_DATE))
        if len(terminal_rows) != 1 or int(overrun.sum()) != 0:
            raise ValueError(f"Formula One old price boundary changed: {lineage.old_symbol}")

        fetched_rows = provider.prices.loc[
            provider.prices["security_id"].astype(str).eq(lineage.security_id)
        ].copy()
        fetched_sessions = pd.to_datetime(fetched_rows["session"], errors="coerce")
        first = fetched_rows.loc[fetched_sessions.eq(pd.Timestamp(TRANSITION_DATE))]
        if len(first) != 1:
            raise ValueError(f"Formula One first successor row is not exact: {lineage.new_symbol}")
        successor = first.iloc[0]
        old_close = float(terminal_rows.iloc[0]["close"])
        new_close = float(successor["close"])
        transition_return = new_close / old_close - 1.0
        if abs(transition_return) > MAXIMUM_TRANSITION_RETURN:
            raise ValueError(
                f"Formula One 1:1 economic transition return is implausible: "
                f"{lineage.new_symbol}/{transition_return:.8f}"
            )
        transition_crosschecks[lineage.new_symbol] = {
            "security_id_preserved": lineage.security_id,
            "retired_symbol_last_session": OLD_LAST_SESSION,
            "successor_symbol_first_session": TRANSITION_DATE,
            "bridge_kind": "adjacent_market_sessions",
            "legal_effective_date": LEGAL_EFFECTIVE_DATE,
            "old_close": old_close,
            "new_close": new_close,
            "one_for_one_transition_return": transition_return,
        }
        target_factors = factors["security_id"].astype(str).eq(lineage.security_id)
        factor_sessions = pd.to_datetime(factors["session"], errors="coerce")
        factor_overrun = target_factors & factor_sessions.ge(pd.Timestamp(TRANSITION_DATE))
        if int(factor_overrun.sum()) != 0:
            raise ValueError(f"Formula One old factor boundary changed: {lineage.old_symbol}")

        catalog_row = catalog.rows[lineage.new_symbol]
        official = evidence.market_boundary
        for column, value in {
            "primary_symbol": lineage.new_symbol,
            "provider_symbol": lineage.provider_symbol,
            "action_provider_symbol": lineage.provider_symbol,
            "name": _text(catalog_row.get("Name")),
            "exchange": _text(catalog_row.get("Exchange")),
            "asset_type": "STOCK",
            "currency": "USD",
            "country": "US",
            "active_to": "",
            "isin": lineage.isin,
            "source": OFFICIAL_SOURCE,
            "source_url": official.source_url,
            "retrieved_at": official.retrieved_at,
            "source_hash": official.source_hash,
        }.items():
            master.loc[master_index, column] = value

        for column, value in {
            "effective_to": OLD_LAST_SESSION,
            "source": OFFICIAL_SOURCE,
            "source_url": evidence.market_boundary.source_url,
            "retrieved_at": evidence.market_boundary.retrieved_at,
            "source_hash": evidence.market_boundary.source_hash,
        }.items():
            history.loc[old_history_index, column] = value
        new_history_rows.append(
            {
                **old_history.to_dict(),
                "symbol": lineage.new_symbol,
                "exchange": _text(catalog_row.get("Exchange")),
                "effective_from": TRANSITION_DATE,
                "effective_to": "",
                "source": OFFICIAL_SOURCE,
                "source_url": official.source_url,
                "retrieved_at": official.retrieved_at,
                "source_hash": official.source_hash,
            }
        )

    if int(remove_prices.sum()) != 0 or int(remove_factors.sum()) != 0:
        raise ValueError("Formula One Jan. 24 rows must be preserved without removal.")

    history = _concat_unique(
        (history, pd.DataFrame(new_history_rows)), dataset="symbol_history"
    )
    prices = _concat_unique(
        (prices.loc[~remove_prices], provider.prices), dataset="daily_price_raw"
    )
    official_actions = pd.DataFrame(
        [_official_action(lineage, evidence) for lineage in LINEAGES]
    )
    official_event_ids = set(official_actions["event_id"].astype(str))
    actions = _concat_unique(
        (
            actions.loc[~actions["event_id"].astype(str).isin(official_event_ids)],
            provider.corporate_actions,
            official_actions,
        ),
        dataset="corporate_actions",
    )
    factor_source = "formula-one-identity:" + sha256_bytes(
        _canonical_json_bytes(
            {
                "completed_session": completed_session,
                "provider_hashes": sorted(
                    artifact.source_hash for artifact in provider.artifacts
                ),
                "official_hashes": sorted(
                    (MARKET_BOUNDARY_SHA256, LEGAL_TERMS_SHA256)
                ),
            }
        )
    )
    factors = build_adjustment_factors(
        prices,
        actions,
        source_version=factor_source,
    )
    archive_artifacts = tuple((*provider.artifacts, *evidence.artifacts))
    archive = _concat_unique(
        (
            archive,
            _archive_rows(
                archive_artifacts, completed_session=completed_session
            ),
        ),
        dataset="source_archive",
    )
    frames = {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "adjustment_factors": factors,
        "source_archive": archive,
    }
    for dataset, frame in frames.items():
        validate_dataset(
            dataset,
            frame,
            completed_session=completed_session,
            incomplete_action_policy="warn",
        ).raise_for_errors()
    if not _identity_is_repaired(frames):
        raise RuntimeError("Formula One prepared snapshot failed its identity invariant.")
    summary = {
        "status": "validated_dry_run",
        "network_accessed": bool(provider.http_attempts),
        "eodhd_http_attempts": provider.http_attempts,
        "eodhd_budget_claim_count": len(provider.budget_claims),
        "eodhd_max_http_attempts": MAX_EODHD_HTTP_ATTEMPTS,
        "security_ids_preserved": {
            lineage.new_symbol: lineage.security_id for lineage in LINEAGES
        },
        "forbidden_new_security_ids_absent": sorted(forbidden_ids),
        "price_rows_removed": int(remove_prices.sum()),
        "successor_price_rows": len(provider.prices),
        "successor_provider_action_rows": len(provider.corporate_actions),
        "official_ticker_change_rows": len(official_actions),
        "source_artifacts_archived": len(archive_artifacts),
        "transition_crosschecks": transition_crosschecks,
        "official_terms": {
            "market_boundary_url": MARKET_BOUNDARY_URL,
            "market_boundary_sha256": MARKET_BOUNDARY_SHA256,
            "legal_terms_url": LEGAL_TERMS_URL,
            "legal_terms_sha256": LEGAL_TERMS_SHA256,
            "one_for_one_evidence_ratio": 1.0,
            "action_ratio": None,
            "legal_effective_date": LEGAL_EFFECTIVE_DATE,
            "trading_effective_date": TRANSITION_DATE,
        },
        "reviewed_nonterminal_extractions": list(
            REVIEWED_NONTERMINAL_EXTRACTIONS
        ),
    }
    return frames, summary, archive_artifacts


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
            raise RuntimeError(f"Formula One release/pointer mismatch: {dataset}")
        output[dataset] = etag
    return output


def prepare_run(
    repository: LocalDatasetRepository,
    *,
    allow_fetch: bool,
    source_factory: Callable[..., FormulaOneEodhdSource] = FormulaOneEodhdSource,
) -> PreparedRepair:
    # Fail before any provider source can be constructed when the reviewed
    # official bytes have not yet been code-pinned.
    _pinned_official_hash()
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required for Formula One repair.")
    missing = sorted(set(WRITE_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError(
            "Current release lacks Formula One repair datasets: " + ", ".join(missing)
        )
    existing = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in WRITE_DATASETS
    }
    pointer_etags = _capture_pointer_etags(repository, release)
    if _identity_is_repaired(existing):
        validate_repository_snapshot(repository).raise_for_errors()
        return PreparedRepair(
            release=release,
            release_etag=release_etag,
            pointer_etags=pointer_etags,
            frames={dataset: existing[dataset].copy() for dataset in WRITE_DATASETS},
            archive_artifacts=(),
            warnings=release.warnings,
            summary={
                "status": "already_repaired",
                "release_version": release.version,
                "network_accessed": False,
                "eodhd_http_attempts": 0,
                "eodhd_budget_claim_count": 0,
                "security_ids_preserved": {
                    lineage.new_symbol: lineage.security_id for lineage in LINEAGES
                },
            },
        )

    catalog = load_catalog_proof(repository, existing["source_archive"])
    _validate_old_raw_archives(
        repository, existing["source_archive"], existing["daily_price_raw"]
    )
    evidence = load_official_evidence(repository.root)
    source = source_factory(
        repository.root / "state/eodhd-formula-one-successors",
        completed_session=release.completed_session,
        allow_http=allow_fetch,
    )
    provider = source.fetch()
    frames, summary, artifacts = prepare_formula_one_repair(
        existing,
        catalog=catalog,
        evidence=evidence,
        provider=provider,
        completed_session=release.completed_session,
    )
    candidate = _CandidateRepository(repository, release.dataset_versions, frames)
    validate_repository_snapshot(candidate).raise_for_errors()
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        frames=frames,
        archive_artifacts=artifacts,
        warnings=release.warnings,
        summary={**summary, "release_version": release.version},
    )


def _persist_archive_payloads(
    repository: LocalDatasetRepository,
    artifacts: Iterable[SourceArtifact],
    completed_session: str,
) -> None:
    for artifact in artifacts:
        extension = _archive_extension(artifact)
        path = (
            repository.root
            / f"archives/{completed_session}/{artifact.source_hash}.{extension}.gz"
        )
        encoded = gzip.compress(artifact.content, mtime=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            try:
                existing = gzip.decompress(path.read_bytes())
            except Exception as exc:
                raise ValueError(f"Existing Formula One archive is unreadable: {path}") from exc
            if existing != artifact.content:
                raise RuntimeError(f"Immutable Formula One archive changed: {path}")
        else:
            write_atomic(path, encoded)
        if gzip.decompress(path.read_bytes()) != artifact.content:
            raise RuntimeError(f"Formula One archive verification failed: {path}")


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
                "A market-store recovery marker blocks Formula One writes: "
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
                "An interrupted market-store transaction blocks Formula One writes: "
                + ", ".join(str(item) for item in interrupted)
            )
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
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
                    f"Unexpected release during Formula One rollback: {observed.version}"
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
                        f"Unexpected Formula One pointer during rollback: {pointer.version}"
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
    with _exclusive_repository_lock(repository):
        current, current_etag = repository.current_release()
        if (
            current is None
            or current.version != prepared.release.version
            or current_etag != prepared.release_etag
        ):
            raise RuntimeError("Current release changed after Formula One preflight.")
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != prepared.release.dataset_versions[dataset]
                or value.etag != prepared.pointer_etags[dataset]
            ):
                raise RuntimeError(f"Formula One pointer changed before apply: {dataset}")
            old_pointers[dataset] = value.data

        transaction_id = uuid.uuid4().hex
        session = prepared.release.completed_session.replace("-", "")
        planned = {
            dataset: f"formula-one-identity-{session}-{transaction_id}-{dataset}"
            for dataset in WRITE_DATASETS
        }
        journal_path = (
            repository.root
            / "transactions/formula-one-identity-repair"
            / f"{transaction_id}.json"
        )
        journal: dict[str, Any] = {
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                dataset: base64.b64encode(value).decode("ascii")
                for dataset, value in old_pointers.items()
            },
            "planned_versions": planned,
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        try:
            _persist_archive_payloads(
                repository,
                prepared.archive_artifacts,
                prepared.release.completed_session,
            )
            versions = dict(prepared.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    prepared.frames[dataset],
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy="warn",
                    metadata={
                        "operation": "repair_us_formula_one_identity",
                        "transition_date": TRANSITION_DATE,
                        "security_ids_preserved": [
                            lineage.security_id for lineage in LINEAGES
                        ],
                        "eodhd_http_attempts": prepared.summary[
                            "eodhd_http_attempts"
                        ],
                        "network_accessed": prepared.summary["network_accessed"],
                        "strict_formula_one_gate": "passed",
                    },
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"Formula One write conflicted: {dataset}/{result.conflict_path}"
                    )
                versions[dataset] = result.manifest.version
            written = {
                dataset: repository.read_frame(dataset, planned[dataset])
                for dataset in WRITE_DATASETS
            }
            if not _identity_is_repaired(written):
                raise RuntimeError("Written Formula One snapshot failed its invariant.")
            candidate = _CandidateRepository(repository, versions, written)
            validate_repository_snapshot(candidate).raise_for_errors()
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=(
                    DataQuality.DEGRADED
                    if prepared.warnings
                    else DataQuality.VALID
                ),
                warnings=prepared.warnings,
                expected_etag=prepared.release_etag,
            )
            current, _ = repository.current_release()
            if current is None or current.to_bytes() != committed.to_bytes():
                raise RuntimeError("Committed Formula One release is not current.")
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
                "transaction_id": transaction_id,
                "new_release_version": committed.version,
                "quality": committed.quality,
                "warnings": list(committed.warnings),
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
                    / "recovery/formula-one-identity-repair"
                    / f"{transaction_id}.json"
                )
                _write_journal(recovery, journal)
                raise RuntimeError(
                    f"Formula One rollback failed; recovery marker blocks writes: "
                    f"{recovery}; errors={rollback_errors}"
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect/repair LMCA/LMCK -> FWONA/FWONK one-for-one history."
    )
    parser.add_argument("--cache-root", default="data/cache")
    parser.add_argument(
        "--fetch-missing-eodhd",
        action="store_true",
        help="Allow at most six one-shot FWONA/FWONK EODHD requests.",
    )
    parser.add_argument(
        "--fetch-official-evidence",
        action="store_true",
        help=(
            "Acquisition-only: allow one no-retry Jan. 24 SEC 424(b)(3) request, "
            "pair it with the pinned cached Jan. 17 8-K, and report exact hashes/"
            "claim gaps. Never writes a data release."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--offline-plan", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def acquire_official_evidence_only(
    repository: LocalDatasetRepository,
    *,
    source_factory: Callable[..., FormulaOneOfficialEvidenceSource] = (
        FormulaOneOfficialEvidenceSource
    ),
) -> dict[str, Any]:
    before, before_etag = repository.current_release()
    source = source_factory(
        repository.root / "state/formula-one-official-evidence",
        legal_cache_root=repository.root / "state/sec_lifecycle",
        allow_http=True,
    )
    evidence = source.acquire(require_pinned=False)
    after, after_etag = repository.current_release()
    before_bytes = before.to_bytes() if before is not None else None
    after_bytes = after.to_bytes() if after is not None else None
    if before_bytes != after_bytes or before_etag != after_etag:
        raise RuntimeError(
            "Data release changed during acquisition-only official evidence fetch."
        )
    observed = evidence.market_boundary.source_hash
    pinned = _text(MARKET_BOUNDARY_SHA256).lower()
    pinned_match = bool(pinned) and observed == pinned
    claims_passed = not evidence.missing_reviewed_claims
    return {
        "status": (
            "official_evidence_ready"
            if pinned_match and claims_passed
            else "official_evidence_observed_unpinned"
        ),
        "source_url": MARKET_BOUNDARY_URL,
        "observed_sha256": observed,
        "configured_sha256": pinned,
        "configured_sha256_matches": pinned_match,
        "reviewed_phrase_gate_passed": claims_passed,
        "missing_reviewed_claims": list(evidence.missing_reviewed_claims),
        "official_http_attempts": evidence.http_attempts,
        "official_max_http_attempts": MAX_OFFICIAL_HTTP_ATTEMPTS,
        "eodhd_http_attempts": 0,
        "cache_path": str(source.path),
        "exact_byte_count": len(evidence.market_boundary.content),
        "legal_terms_url": LEGAL_TERMS_URL,
        "legal_terms_sha256": evidence.legal_terms.source_hash,
        "legal_terms_exact_byte_count": len(evidence.legal_terms.content),
        "release_mutated": False,
        "apply_allowed": bool(pinned_match and claims_passed),
    }


def run(
    args: argparse.Namespace,
    *,
    repository_factory: Callable[[str | Path], LocalDatasetRepository] = (
        LocalDatasetRepository
    ),
    source_factory: Callable[..., FormulaOneEodhdSource] = FormulaOneEodhdSource,
) -> dict[str, Any]:
    fetch_eodhd = bool(getattr(args, "fetch_missing_eodhd", False))
    fetch_official = bool(getattr(args, "fetch_official_evidence", False))
    offline_plan = bool(getattr(args, "offline_plan", False))
    apply = bool(getattr(args, "apply", False))
    if offline_plan and fetch_eodhd:
        raise ValueError("--offline-plan cannot enable EODHD HTTP.")
    repository = repository_factory(args.cache_root)
    if fetch_official:
        if fetch_eodhd or offline_plan or apply:
            raise ValueError(
                "--fetch-official-evidence is acquisition-only and cannot be "
                "combined with EODHD fetch, offline plan, or apply."
            )
        return acquire_official_evidence_only(repository)
    prepared = prepare_run(
        repository,
        allow_fetch=fetch_eodhd,
        source_factory=source_factory,
    )
    if not apply:
        return prepared.summary
    return apply_repair(repository, prepared)


def main() -> int:
    args = _parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
