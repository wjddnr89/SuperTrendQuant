#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

import pandas as pd
import yaml

from supertrend_quant.env import load_env
from supertrend_quant.market_store.lifecycle import (
    LifecycleEvidence,
    ParsedLifecycleEvent,
    SecEdgarLifecycleSource,
    SecFiling,
    build_lifecycle_candidates,
    resolve_new_security_id,
)
from supertrend_quant.market_store.lifecycle_report_provenance import (
    DEFAULT_SEC_MAX_HTTP_ATTEMPTS,
    DEFAULT_SEC_MAX_HTTP_ATTEMPTS_PER_CANDIDATE,
    SEC_FETCH_POLICY_CACHE_ONLY,
    SEC_FETCH_POLICY_FETCH_MISSING,
    SEC_MAX_HTTP_ATTEMPTS_PER_REQUEST,
    build_lifecycle_report_binding,
    update_lifecycle_report_http_attempts,
    validate_lifecycle_report_binding,
)
from supertrend_quant.market_store.ingest import SourceArtifact
from supertrend_quant.market_store.lifecycle_coverage import lifecycle_candidate_id
from supertrend_quant.market_store.manifest import write_atomic
from supertrend_quant.market_store.official_lifecycle_evidence import (
    OfficialLifecycleExceptionEvidenceSource,
    OfficialLifecycleExceptionEvidenceSpec,
    include_bound_official_applied_event_candidates,
    load_official_lifecycle_exception_evidence,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.source_archive import validate_source_archive_id


DEFAULT_HINTS = Path("unified_quant/configs/us_lifecycle_hints.yaml")
DEFAULT_OUTPUT = Path("results/data_quality/us_lifecycle/sec_collection.json")

_NTCO_SECURITY_ID = "US:EODHD:7e3cea59-409f-5cf0-8429-9d4245013622"
_NTCO_RELEASE_EVENT_KEY = f"{_NTCO_SECURITY_ID}|2024-08-07"
_NTCO_RELEASE_EVENT_CONTRACT = "ntco_ntcoy_price_identity_terminal_only/v1"
_NTCO_TICKER_EVENT_ID = (
    "0bd231d216016d33c8d6eb1ec9bb85450a21b9298c4abda34a3399727e9b2c00"
)
_NTCO_TERMINAL_EVENT_ID = (
    "d1eeadf8f4779212ff4b3162af1731bf7bfc804c10756cc4bb8074c307e1f746"
)
_NTCO_TERMINAL_SOURCE_HASH = (
    "3be830009ee6942d4ca604aafdc19730a693f13e62b046641a638e9a1ee112c1"
)
_NTCO_IDENTITY_SOURCE_HASH = (
    "8c9312d2079c238a4fa47b701d24b8e707c040080cb8a5ce0d62f6bd82fd54cb"
)
_NTCO_DECISION_SOURCE_HASH = (
    "fa349557d3f433371a2b08015cd7d672801106a2e442b3be4e51478e69587557"
)
_NTCO_RETRIEVED_AT = "2026-07-18T18:47:16.808110Z"
_NTCO_REVIEWED_AT = "2026-07-19T00:00:00Z"
_NTCO_DIVIDEND_EVENT_AMOUNTS = {
    "658cb5351b78504a2c20ca3ae75d4d5a2660ea884fc1e2650b1c9a0370551cc0": (
        "2024-03-21",
        0.28427,
    ),
    "ebbf2e8b20dfeb94521486d8ed81342ae1fb631c01796857e53795fcafbd163c": (
        "2024-04-09",
        0.01099,
    ),
}

# Every post-apply collector replay is bound to these exact raw/derived rows.
# The sole non-content archive id is the centrally registered empty-splits
# provenance tuple; no arbitrary composite ids are accepted here.
_NTCO_ARCHIVE_CONTRACT: tuple[Mapping[str, str], ...] = (
    {
        "source": "official_cboe",
        "source_url": "https://cdn.cboe.com/resources/product_restriction/2024/Cboe-Options-Exchanges-Restrictions-on-Transactions-in-Options-on-Natura-Co-Holding-S-A.pdf",
        "source_hash": "e67d8046a4c532daa11dac1faacb9e573238812c90d15be9792b5d6a46e67928",
        "archive_id": "e67d8046a4c532daa11dac1faacb9e573238812c90d15be9792b5d6a46e67928",
        "content_type": "application/pdf",
        "retrieved_at": "2026-07-18T17:41:15.684878Z",
        "suffix": "bin",
    },
    {
        "source": "official_occ",
        "source_url": "https://infomemo.theocc.com/infomemos?number=54105",
        "source_hash": "5eb3c0058195d6d6c22361e362f1623c4eae7dd7262e9497ef7b53f8a564e913",
        "archive_id": "5eb3c0058195d6d6c22361e362f1623c4eae7dd7262e9497ef7b53f8a564e913",
        "content_type": "text/html",
        "retrieved_at": "2026-07-18T17:41:32.027461Z",
        "suffix": "bin",
    },
    {
        "source": "official_bny",
        "source_url": "https://www.adrbny.com/content/dam/adr/documents/corporate-actions-dr/files/ad1145447.pdf",
        "source_hash": "057a95798bd3b9ff4f0f0be6fd34d7ed42d6ad1d27a12ad1c41e30700c8da28b",
        "archive_id": "057a95798bd3b9ff4f0f0be6fd34d7ed42d6ad1d27a12ad1c41e30700c8da28b",
        "content_type": "application/pdf",
        "retrieved_at": "2026-07-18T17:42:06.789266Z",
        "suffix": "bin",
    },
    {
        "source": "eodhd_eod",
        "source_url": "https://eodhd.com/api/eod/NTCOY.US?from=2024-02-12&to=2024-09-03",
        "source_hash": "3ef3a1f03ec97252ac4db079298cdb90ddc32bdeb41fd64a71aaf6d667153e54",
        "archive_id": "3ef3a1f03ec97252ac4db079298cdb90ddc32bdeb41fd64a71aaf6d667153e54",
        "content_type": "application/json",
        "retrieved_at": "2026-07-18T18:28:42.473931Z",
        "suffix": "json",
    },
    {
        "source": "eodhd_div",
        "source_url": "https://eodhd.com/api/div/NTCOY.US?from=2024-02-12&to=2024-09-03",
        "source_hash": "6adc67e2b64dd8dcf0acfc0a3bf20bb0d275844f2305b66c1ff4d2a3789d8175",
        "archive_id": "6adc67e2b64dd8dcf0acfc0a3bf20bb0d275844f2305b66c1ff4d2a3789d8175",
        "content_type": "application/json",
        "retrieved_at": "2026-07-18T18:28:42.473931Z",
        "suffix": "json",
    },
    {
        "source": "eodhd_splits",
        "source_url": "https://eodhd.com/api/splits/NTCOY.US?from=2024-02-12&to=2024-09-03",
        "source_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        "archive_id": "6a09ccaafcdf8ad57177fd1be2146ce912c84c4269cdc11ce736c7b4faad4461",
        "content_type": "application/json",
        "retrieved_at": "2026-07-18T18:28:42.473931Z",
        "suffix": "json",
    },
    {
        "source": "official_bny_termination",
        "source_url": "https://www.adrbny.com/content/dam/adr/documents/corporate-actions-dr/files/ad1140774.pdf",
        "source_hash": "abc23f86378f82f5efb475fc23e0a82e398db4201dc7e52fd4064d088eb47b83",
        "archive_id": "abc23f86378f82f5efb475fc23e0a82e398db4201dc7e52fd4064d088eb47b83",
        "content_type": "application/pdf",
        "retrieved_at": "2026-07-18T18:42:57.215876Z",
        "suffix": "bin",
    },
    {
        "source": "official_bny_books_closed",
        "source_url": "https://www.adrbny.com/content/dam/adr/documents/books-closed/files/bc1141635.pdf",
        "source_hash": "b5d449fcad19cd38fb461c8f9bae4c1856c0fb7ccd4903e062f87840315ba675",
        "archive_id": "b5d449fcad19cd38fb461c8f9bae4c1856c0fb7ccd4903e062f87840315ba675",
        "content_type": "application/pdf",
        "retrieved_at": "2026-07-18T18:47:16.808110Z",
        "suffix": "bin",
    },
    {
        "source": "official_ntco_ntcoy_identity",
        "source_url": "https://infomemo.theocc.com/infomemos?number=54105",
        "source_hash": _NTCO_IDENTITY_SOURCE_HASH,
        "archive_id": _NTCO_IDENTITY_SOURCE_HASH,
        "content_type": "application/json",
        "retrieved_at": _NTCO_RETRIEVED_AT,
        "suffix": "json",
    },
    {
        "source": "official_ntcoy_cash_termination",
        "source_url": "https://www.adrbny.com/content/dam/adr/documents/corporate-actions-dr/files/ad1145447.pdf",
        "source_hash": _NTCO_TERMINAL_SOURCE_HASH,
        "archive_id": _NTCO_TERMINAL_SOURCE_HASH,
        "content_type": "application/json",
        "retrieved_at": _NTCO_RETRIEVED_AT,
        "suffix": "json",
    },
    {
        "source": "reviewed_ntco_ntcoy_transition_decision",
        "source_url": "https://eodhd.com/api/div/NTCOY.US?from=2024-02-12&to=2024-09-03",
        "source_hash": _NTCO_DECISION_SOURCE_HASH,
        "archive_id": _NTCO_DECISION_SOURCE_HASH,
        "content_type": "application/json",
        "retrieved_at": _NTCO_RETRIEVED_AT,
        "suffix": "json",
    },
)

_DIRECT_SEC_ARCHIVE_SOURCES = frozenset(
    {
        "official_identity_evidence_raw",
        "sec_bbby_identity_evidence",
        "sec_edgar_filing",
        "sec_rule_provision_notice",
    }
)
_DIRECT_SEC_ARCHIVE_CONTENT_TYPES = frozenset(
    {"text/html", "text/plain", "application/xhtml+xml"}
)


class _CurrentReleaseSecArchiveReplay:
    """Read exact reviewed SEC bytes from the bound release without mutation."""

    def __init__(
        self,
        repository: LocalDatasetRepository,
        release,
        candidate_urls: Mapping[Any, Iterable[str]],
    ) -> None:
        archive_version = str(release.dataset_versions.get("source_archive") or "")
        if not archive_version:
            raise RuntimeError(
                "Current release has no source_archive for SEC offline replay."
            )
        self.repository = repository
        self.release_version = str(release.version)
        self.completed_session = str(release.completed_session)
        self.dataset_versions = {
            str(key): str(value)
            for key, value in sorted(release.dataset_versions.items())
        }
        self.archive_version = archive_version
        self.archive = repository.read_frame("source_archive", archive_version)
        required = {
            "archive_id",
            "dataset",
            "object_path",
            "content_type",
            "source",
            "source_hash",
            "source_url",
        }
        missing = sorted(required - set(self.archive.columns))
        if missing:
            raise RuntimeError(
                "Bound source_archive lacks SEC replay provenance columns: "
                + ", ".join(missing)
            )

        self.candidate_urls: dict[Any, frozenset[str]] = {}
        self.url_owners: dict[str, set[Any]] = {}
        for candidate, values in candidate_urls.items():
            urls = frozenset(
                str(value).strip()
                for value in values
                if _is_direct_sec_archive_url(value)
            )
            if not urls:
                continue
            self.candidate_urls[candidate] = urls
            for url in urls:
                self.url_owners.setdefault(url, set()).add(candidate)

    def __call__(self, request_url: str, candidate) -> bytes | None:
        owners = self.url_owners.get(str(request_url))
        if not owners:
            return None
        if candidate is None or candidate not in owners:
            raise RuntimeError(
                "Archived SEC URL is not bound to the active current-release "
                f"candidate: {request_url}"
            )
        self._validate_current_release()

        matches = self.archive.loc[
            self.archive["source_url"].map(_archive_text).eq(request_url)
        ]
        if matches.empty:
            raise RuntimeError(
                "Archived SEC URL has no current-release source_archive row: "
                f"{request_url}"
            )
        if len(matches) > 1:
            # A finalizer may archive the same immutable SEC object under both
            # its reviewed role (for example sec_rule_provision_notice) and the
            # generic sec_edgar_filing role discovered by the lifecycle report.
            # This is not byte ambiguity when every row binds the same hash,
            # object, and content type under a distinct valid archive ID.
            archive_ids = matches["archive_id"].map(_archive_text).str.lower()
            immutable_keys = {
                (
                    _archive_text(row.get("source_hash")).lower(),
                    _archive_text(row.get("object_path")),
                    _archive_text(row.get("content_type")).lower(),
                )
                for _, row in matches.iterrows()
            }
            if archive_ids.duplicated().any() or len(immutable_keys) != 1:
                raise RuntimeError(
                    "Archived SEC URL resolves to ambiguous current-release "
                    "source_archive rows: "
                    f"{request_url}; matches={len(matches)}"
                )
            for _, duplicate_row in matches.iterrows():
                duplicate_source = _archive_text(duplicate_row.get("source"))
                duplicate_dataset = _archive_text(duplicate_row.get("dataset"))
                duplicate_content_type = _archive_text(
                    duplicate_row.get("content_type")
                ).lower()
                duplicate_hash = _archive_text(
                    duplicate_row.get("source_hash")
                ).lower()
                if (
                    duplicate_source != duplicate_dataset
                    or duplicate_source not in _DIRECT_SEC_ARCHIVE_SOURCES
                    or duplicate_content_type
                    not in _DIRECT_SEC_ARCHIVE_CONTENT_TYPES
                ):
                    raise RuntimeError(
                        "Archived SEC URL duplicate is not a direct raw official "
                        f"filing artifact: {request_url}"
                    )
                try:
                    validate_source_archive_id(
                        _archive_text(duplicate_row.get("archive_id")).lower(),
                        source=duplicate_source,
                        source_url=request_url,
                        source_hash=duplicate_hash,
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        "Archived SEC URL duplicate has an invalid "
                        "archive_id/source_hash binding: "
                        f"{request_url}"
                    ) from exc
            matches = matches.assign(
                _generic_source=matches["source"].map(_archive_text).ne(
                    "sec_edgar_filing"
                ),
                _archive_id=archive_ids,
            ).sort_values(["_generic_source", "_archive_id"])
        row = matches.iloc[0]
        source = _archive_text(row.get("source"))
        dataset = _archive_text(row.get("dataset"))
        content_type = _archive_text(row.get("content_type")).lower()
        if (
            source != dataset
            or source not in _DIRECT_SEC_ARCHIVE_SOURCES
            or content_type not in _DIRECT_SEC_ARCHIVE_CONTENT_TYPES
        ):
            raise RuntimeError(
                "Archived SEC URL is not a direct raw official filing artifact: "
                f"{request_url}; dataset={dataset!r}, source={source!r}, "
                f"content_type={content_type!r}"
            )

        archive_id = _archive_text(row.get("archive_id")).lower()
        source_hash = _archive_text(row.get("source_hash")).lower()
        try:
            validate_source_archive_id(
                archive_id,
                source=source,
                source_url=request_url,
                source_hash=source_hash,
            )
        except ValueError as exc:
            raise RuntimeError(
                "Archived SEC URL has an invalid archive_id/source_hash binding: "
                f"{request_url}"
            ) from exc

        object_path = _archive_text(row.get("object_path"))
        relative = Path(object_path)
        if (
            not object_path
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix.lower() != ".gz"
            or not relative.name.startswith(f"{source_hash}.")
        ):
            raise RuntimeError(
                "Archived SEC object path is not an exact hash-named gzip object: "
                f"{object_path!r}"
            )
        root = self.repository.root.resolve()
        path = (root / relative).resolve()
        if path == root or root not in path.parents:
            raise RuntimeError(
                f"Archived SEC object path escapes repository root: {object_path}"
            )
        if not path.is_file():
            raise RuntimeError(f"Archived SEC object is missing: {object_path}")
        try:
            payload = gzip.decompress(path.read_bytes())
        except Exception as exc:
            raise RuntimeError(
                f"Archived SEC gzip object is unreadable: {object_path}"
            ) from exc
        observed_hash = hashlib.sha256(payload).hexdigest()
        if observed_hash != source_hash:
            raise RuntimeError(
                "Archived SEC payload hash does not match source_hash: "
                f"{request_url}; expected={source_hash}, observed={observed_hash}"
            )
        return payload

    def _validate_current_release(self) -> None:
        current, _ = self.repository.current_release()
        if current is None:
            raise RuntimeError("Current release disappeared during SEC archive replay.")
        observed_versions = {
            str(key): str(value)
            for key, value in sorted(current.dataset_versions.items())
        }
        if (
            str(current.version) != self.release_version
            or str(current.completed_session) != self.completed_session
            or observed_versions != self.dataset_versions
            or observed_versions.get("source_archive") != self.archive_version
        ):
            raise RuntimeError(
                "Current release changed during bound SEC archive replay; refusing "
                "to reuse archived bytes."
            )


def _archive_text(value: Any) -> str:
    if value is None or bool(pd.isna(value)):
        return ""
    return str(value).strip()


def _ntco_exact_archive_artifacts(
    repository: LocalDatasetRepository,
    release,
) -> tuple[SourceArtifact, ...]:
    version = str(release.dataset_versions.get("source_archive") or "")
    if not version:
        raise RuntimeError("NTCO release-event replay requires source_archive.")
    archive = repository.read_frame("source_archive", version)
    artifacts: list[SourceArtifact] = []
    payloads: dict[str, bytes] = {}
    for expected in _NTCO_ARCHIVE_CONTRACT:
        rows = archive.loc[
            archive["archive_id"].astype(str).eq(expected["archive_id"])
        ]
        if len(rows) != 1:
            raise RuntimeError(
                "NTCO release-event archive row is missing or ambiguous: "
                f"{expected['source']}/{expected['archive_id']}"
            )
        row = rows.iloc[0]
        wanted = {
            "dataset": expected["source"],
            "source": expected["source"],
            "source_url": expected["source_url"],
            "source_hash": expected["source_hash"],
            "content_type": expected["content_type"],
            "retrieved_at": expected["retrieved_at"],
            "effective_date": str(release.completed_session),
            "object_path": (
                f"archives/{release.completed_session}/"
                f"{expected['source_hash']}.{expected['suffix']}.gz"
            ),
        }
        if any(_archive_text(row.get(field)) != value for field, value in wanted.items()):
            raise RuntimeError(
                "NTCO release-event archive provenance changed: "
                f"{expected['source']}"
            )
        validate_source_archive_id(
            expected["archive_id"],
            source=expected["source"],
            source_url=expected["source_url"],
            source_hash=expected["source_hash"],
        )
        relative = Path(wanted["object_path"])
        root = repository.root.resolve()
        path = (root / relative).resolve()
        if path == root or root not in path.parents or not path.is_file():
            raise RuntimeError(
                f"NTCO release-event archive object is missing: {relative}"
            )
        try:
            content = gzip.decompress(path.read_bytes())
        except Exception as exc:
            raise RuntimeError(
                f"NTCO release-event archive object is unreadable: {relative}"
            ) from exc
        if hashlib.sha256(content).hexdigest() != expected["source_hash"]:
            raise RuntimeError(
                f"NTCO release-event archive payload hash changed: {relative}"
            )
        payloads[expected["source"]] = content
        artifacts.append(
            SourceArtifact(
                source=expected["source"],
                source_url=expected["source_url"],
                retrieved_at=expected["retrieved_at"],
                content=content,
                content_type=expected["content_type"],
            )
        )

    try:
        decision = json.loads(
            payloads["reviewed_ntco_ntcoy_transition_decision"]
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("NTCO transition decision audit is invalid JSON.") from exc
    if (
        decision.get("schema") != "reviewed_ntco_ntcoy_transition_decision/v1"
        or decision.get("security_id") != _NTCO_SECURITY_ID
        or decision.get("decision_mode") != "price_identity_terminal_only"
        or decision.get("provider_dividend_economics_accepted") is not False
        or decision.get("provider_dividend_raw_decision")
        != "archive_exact_ntcoy_raw_reject_economics_preserve_ntco_actions"
        or decision.get("maximum_absolute_sensitivity_usd_per_ads") != "0.01585"
        or (decision.get("release_scope_audit") or {}).get("target_row_count") != 0
        or (decision.get("release_scope_audit") or {}).get("absence_proven") is not True
    ):
        raise RuntimeError("NTCO transition decision audit policy changed.")
    return tuple(artifacts)


def _collect_existing_release_event(
    repository: LocalDatasetRepository,
    release,
    candidate,
    value: Mapping[str, Any],
) -> tuple[LifecycleEvidence, tuple[SourceArtifact, ...]]:
    """Replay the one exact already-applied NTCO terminal event with zero HTTP."""

    if (
        _candidate_hint_key(candidate) != _NTCO_RELEASE_EVENT_KEY
        or str(candidate.symbol).strip().upper() != "NTCOY"
        or dict(value) != {"contract": _NTCO_RELEASE_EVENT_CONTRACT}
    ):
        raise RuntimeError("Existing-release lifecycle replay is not exact NTCOY.")
    current, _ = repository.current_release()
    if (
        current is None
        or str(current.version) != str(release.version)
        or dict(current.dataset_versions) != dict(release.dataset_versions)
    ):
        raise RuntimeError("Current release changed before NTCOY event replay.")

    actions = repository.read_frame(
        "corporate_actions", release.dataset_versions["corporate_actions"]
    )
    own = actions.loc[actions["security_id"].astype(str).eq(_NTCO_SECURITY_ID)].copy()
    effective = pd.to_datetime(own["effective_date"], errors="coerce").dt.date.astype(str)
    tail = own.loc[effective.ge("2024-02-12")]
    expected_ids = {
        _NTCO_TICKER_EVENT_ID,
        _NTCO_TERMINAL_EVENT_ID,
        *_NTCO_DIVIDEND_EVENT_AMOUNTS,
    }
    if set(tail["event_id"].astype(str)) != expected_ids or len(tail) != 4:
        raise RuntimeError("NTCOY applied action inventory changed.")

    def exact_action(event_id: str) -> Mapping[str, Any]:
        rows = tail.loc[tail["event_id"].astype(str).eq(event_id)]
        if len(rows) != 1:
            raise RuntimeError(f"NTCOY exact action is missing: {event_id}")
        return rows.iloc[0]

    ticker = exact_action(_NTCO_TICKER_EVENT_ID)
    if (
        _archive_text(ticker.get("action_type")) != "ticker_change"
        or _archive_text(ticker.get("effective_date")) != "2024-02-12"
        or _archive_text(ticker.get("new_security_id")) != _NTCO_SECURITY_ID
        or _archive_text(ticker.get("new_symbol")) != "NTCOY"
        or _archive_text(ticker.get("source")) != "official_ntco_ntcoy_identity"
        or _archive_text(ticker.get("source_hash")) != _NTCO_IDENTITY_SOURCE_HASH
        or _archive_text(ticker.get("retrieved_at")) != _NTCO_RETRIEVED_AT
    ):
        raise RuntimeError("NTCOY exact ticker-change action changed.")
    terminal = exact_action(_NTCO_TERMINAL_EVENT_ID)
    if (
        _archive_text(terminal.get("action_type")) != "delisting"
        or _archive_text(terminal.get("effective_date")) != "2024-09-04"
        or float(terminal.get("cash_amount")) != 5.043659
        or _archive_text(terminal.get("currency")) != "USD"
        or _archive_text(terminal.get("source"))
        != "official_ntcoy_cash_termination"
        or _archive_text(terminal.get("source_hash")) != _NTCO_TERMINAL_SOURCE_HASH
        or _archive_text(terminal.get("retrieved_at")) != _NTCO_RETRIEVED_AT
    ):
        raise RuntimeError("NTCOY exact terminal action changed.")
    for event_id, (event_date, cash_amount) in _NTCO_DIVIDEND_EVENT_AMOUNTS.items():
        row = exact_action(event_id)
        if (
            _archive_text(row.get("action_type")) != "cash_dividend"
            or _archive_text(row.get("effective_date")) != event_date
            or float(row.get("cash_amount")) != cash_amount
            or _archive_text(row.get("source")) != "eodhd_div"
            or _archive_text(row.get("source_hash"))
            != "b2a5b7c6a26165cf4f92618e4a76c06b0cd7de55673fd5cc7162073374469fa0"
            or _archive_text(row.get("retrieved_at"))
            != "2026-07-17T20:37:19.646249Z"
        ):
            raise RuntimeError("A preserved NTCO dividend action changed.")

    resolutions = repository.read_frame(
        "lifecycle_resolutions",
        release.dataset_versions["lifecycle_resolutions"],
    )
    candidate_id = lifecycle_candidate_id(_NTCO_SECURITY_ID, "2024-08-07")
    rows = resolutions.loc[
        resolutions["candidate_id"].astype(str).eq(candidate_id)
    ]
    if len(rows) != 1:
        raise RuntimeError("NTCOY exact lifecycle resolution is missing.")
    resolution = rows.iloc[0]
    if (
        _archive_text(resolution.get("security_id")) != _NTCO_SECURITY_ID
        or _archive_text(resolution.get("symbol")) != "NTCOY"
        or _archive_text(resolution.get("last_price_date")) != "2024-08-07"
        or _archive_text(resolution.get("resolution")) != "applied"
        or _archive_text(resolution.get("event_id")) != _NTCO_TERMINAL_EVENT_ID
        or _archive_text(resolution.get("exception_code"))
        or _archive_text(resolution.get("reviewed_by"))
        != "us_ntco_ntcoy_transition_repair_v1"
        or _archive_text(resolution.get("reviewed_at")) != _NTCO_REVIEWED_AT
        or _archive_text(resolution.get("source"))
        != "official_ntcoy_cash_termination"
        or _archive_text(resolution.get("source_hash"))
        != _NTCO_TERMINAL_SOURCE_HASH
    ):
        raise RuntimeError("NTCOY exact lifecycle resolution changed.")

    artifacts = _ntco_exact_archive_artifacts(repository, release)
    filing = SecFiling(
        cik="",
        accession_number="bny-ad1145447",
        filing_date="2024-08-26",
        form="depositary-notice",
        items=(),
        display_name="Natura &Co Holding S.A. ADS mandatory cash exchange",
        score=100.0,
    )
    parsed = ParsedLifecycleEvent(
        action_type="delisting",
        effective_date="2024-09-04",
        cash_amount=5.043659,
        ratio=None,
        new_symbol="",
        confidence="high",
        reason=(
            "Exact current-release BNY evidence and applied action bind the "
            "NTCOY ADS mandatory cash exchange at USD 5.043659 per ADS."
        ),
    )
    return (
        LifecycleEvidence(
            candidate=candidate,
            filing=filing,
            parsed=parsed,
            source_url=(
                "https://www.adrbny.com/content/dam/adr/documents/"
                "corporate-actions-dr/files/ad1145447.pdf"
            ),
            source_hash=_NTCO_TERMINAL_SOURCE_HASH,
        ),
        artifacts,
    )


def _is_direct_sec_archive_url(value: Any) -> bool:
    try:
        parsed = urlsplit(str(value).strip())
        port = parsed.port
    except (TypeError, ValueError):
        return False
    return bool(
        parsed.scheme == "https"
        and (parsed.hostname or "").lower() == "www.sec.gov"
        and parsed.username is None
        and parsed.password is None
        and port in {None, 443}
        and not parsed.query
        and not parsed.fragment
        and parsed.path.startswith("/Archives/edgar/data/")
        and ".." not in Path(parsed.path).parts
    )


def _build_current_release_sec_archive_replay(
    repository: LocalDatasetRepository,
    release,
    candidates: Iterable[Any],
    hints: Mapping[str, Mapping[str, Any]],
    identity_bound_hints: Mapping[str, Mapping[str, Any]],
) -> _CurrentReleaseSecArchiveReplay:
    candidate_urls: dict[Any, tuple[str, ...]] = {}
    for candidate in candidates:
        hint = _hint_for_candidate(candidate, hints, identity_bound_hints)
        verified = hint.get("verified_event")
        if not isinstance(verified, dict):
            continue
        urls = tuple(
            str(value).strip()
            for value in verified.get("source_urls", ())
            if _is_direct_sec_archive_url(value)
        )
        if urls:
            candidate_urls[candidate] = urls
    return _CurrentReleaseSecArchiveReplay(repository, release, candidate_urls)


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(
        description=(
            "Collect official SEC lifecycle evidence for terminal US index constituents "
            "and cross-check it against stored EODHD prices and index removals."
        )
    )
    parser.add_argument("--cache-root", default="data/cache")
    parser.add_argument("--hints", default=str(DEFAULT_HINTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--symbols", default="", help="Comma-separated subset.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-unresolved", action="store_true")
    parser.add_argument(
        "--fetch-missing-sec",
        action="store_true",
        help=(
            "Opt in to SEC HTTP only for exact cache misses. The default is a "
            "strict offline/cache-only replay."
        ),
    )
    parser.add_argument(
        "--sec-max-http-attempts",
        type=int,
        default=DEFAULT_SEC_MAX_HTTP_ATTEMPTS,
        help="Hard global SEC HTTP-attempt cap for this report, including retries.",
    )
    parser.add_argument(
        "--sec-max-http-attempts-per-candidate",
        type=int,
        default=DEFAULT_SEC_MAX_HTTP_ATTEMPTS_PER_CANDIDATE,
        help="Hard per-candidate SEC HTTP-attempt cap, including retries.",
    )
    official_mode = parser.add_mutually_exclusive_group()
    official_mode.add_argument(
        "--fetch-official-exception-evidence",
        action="store_true",
        help=(
            "Stage 1: allow one request per missing exact SEC/FDIC allow-list URL, "
            "record the observed SHA-256, but do not approve an unpinned artifact."
        ),
    )
    official_mode.add_argument(
        "--verify-official-exception-evidence",
        action="store_true",
        help=(
            "Stage 2: perform no HTTP and require every selected official artifact "
            "to match its reviewer-pinned SHA-256 and exact candidate binding."
        ),
    )
    parser.add_argument(
        "--official-exception-only",
        action="store_true",
        help=(
            "Update official exception artifacts in an existing full --resume report "
            "without running general SEC searches."
        ),
    )
    parser.add_argument(
        "--official-evidence-cache",
        default="",
        help="Immutable official evidence cache; defaults to CACHE_ROOT/state/sec_lifecycle.",
    )
    args = parser.parse_args()

    official_mode_enabled = bool(
        args.fetch_official_exception_evidence
        or args.verify_official_exception_evidence
    )
    if args.official_exception_only and not official_mode_enabled:
        parser.error(
            "--official-exception-only requires a fetch or verify official-evidence mode."
        )
    if args.official_exception_only and not args.resume:
        parser.error("--official-exception-only requires --resume.")
    if args.official_exception_only and args.limit:
        parser.error("--official-exception-only cannot be combined with --limit.")
    if args.sec_max_http_attempts < 1:
        parser.error("--sec-max-http-attempts must be a positive integer.")
    if args.sec_max_http_attempts_per_candidate < 1:
        parser.error(
            "--sec-max-http-attempts-per-candidate must be a positive integer."
        )

    repository = LocalDatasetRepository(args.cache_root)
    release, _ = repository.current_release()
    if release is None:
        raise RuntimeError("No local data release is available.")
    master = repository.read_frame(
        "security_master", release.dataset_versions["security_master"]
    )
    history = repository.read_frame(
        "symbol_history", release.dataset_versions["symbol_history"]
    )
    prices = repository.read_frame(
        "daily_price_raw", release.dataset_versions["daily_price_raw"]
    )
    price_histories = _build_price_histories(prices)
    official_specs = load_official_lifecycle_exception_evidence(Path(args.hints))
    all_candidates = list(
        include_bound_official_applied_event_candidates(
            build_lifecycle_candidates(repository, release=release),
            repository,
            release,
            official_specs,
        )
    )
    requested = {
        value.strip().upper() for value in args.symbols.split(",") if value.strip()
    }
    candidates = list(all_candidates)
    if not args.official_exception_only:
        if requested:
            candidates = [item for item in candidates if item.symbol.upper() in requested]
        if args.limit > 0:
            candidates = candidates[: args.limit]
    else:
        candidates = []

    hints = _load_hints(Path(args.hints))
    identity_bound_hints = _load_identity_bound_hints(Path(args.hints))
    archive_replay = _build_current_release_sec_archive_replay(
        repository,
        release,
        all_candidates,
        hints,
        identity_bound_hints,
    )
    hints_path = Path(args.hints)
    output_path = Path(args.output)
    prior_report = (
        _load_report(output_path)
        if bool(args.resume) and output_path.is_file()
        else {}
    )
    sec_fetch_policy = (
        SEC_FETCH_POLICY_FETCH_MISSING
        if bool(args.fetch_missing_sec)
        else SEC_FETCH_POLICY_CACHE_ONLY
    )
    report_binding = build_lifecycle_report_binding(
        release_version=release.version,
        completed_session=release.completed_session,
        dataset_versions=release.dataset_versions,
        candidates=all_candidates,
        hints_path=hints_path,
        sec_fetch_policy=sec_fetch_policy,
        sec_max_http_attempts=args.sec_max_http_attempts,
        sec_max_http_attempts_per_candidate=(
            args.sec_max_http_attempts_per_candidate
        ),
        sec_max_http_attempts_per_request=SEC_MAX_HTTP_ATTEMPTS_PER_REQUEST,
        sec_http_attempts=prior_report.get("sec_http_attempts", 0),
        sec_http_attempts_by_candidate=prior_report.get(
            "sec_http_attempts_by_candidate", {}
        ),
    )
    report = _load_or_initialize_report(
        output_path,
        resume=bool(args.resume),
        expected_binding=report_binding,
    )
    if args.official_exception_only:
        _validate_full_resume_report(report, report_binding, all_candidates)
    records: dict[str, dict[str, Any]] = report.setdefault("records", {})
    known_symbols = set(master["primary_symbol"].astype(str).str.upper())
    source = None
    if candidates:
        source = SecEdgarLifecycleSource(
            cache_dir=repository.root / "state/sec_lifecycle",
            allow_http=bool(args.fetch_missing_sec),
            max_http_attempts=args.sec_max_http_attempts,
            max_http_attempts_per_candidate=(
                args.sec_max_http_attempts_per_candidate
            ),
            max_http_attempts_per_request=SEC_MAX_HTTP_ATTEMPTS_PER_REQUEST,
            initial_http_attempts=int(report["sec_http_attempts"]),
            initial_http_attempts_by_candidate=dict(
                report["sec_http_attempts_by_candidate"]
            ),
            archive_replay=archive_replay,
        )

    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        prior = records.get(candidate.security_id)
        if prior and not args.retry_unresolved:
            print(f"[{index}/{total}] {candidate.symbol}: cached")
            continue
        if prior and args.retry_unresolved and prior.get("eligible_for_apply"):
            print(f"[{index}/{total}] {candidate.symbol}: already verified")
            continue
        hint = _hint_for_candidate(candidate, hints, identity_bound_hints)
        if source is None:  # pragma: no cover - candidates imply a source
            raise RuntimeError("SEC source was not initialized.")
        try:
            if hint.get("existing_release_event"):
                evidence, artifacts = _collect_existing_release_event(
                    repository,
                    release,
                    candidate,
                    dict(hint["existing_release_event"]),
                )
            else:
                with source.candidate_http_scope(candidate):
                    if hint.get("verified_event"):
                        evidence, artifacts = _collect_verified_event(
                            source,
                            candidate,
                            dict(hint["verified_event"]),
                        )
                    else:
                        evidence, artifacts = source.collect(
                            candidate,
                            known_symbols=known_symbols,
                            related_symbols=hint.get("related_symbols", ()),
                            related_names=hint.get("related_names", ()),
                            preferred_symbols=hint.get(
                                "preferred_symbols", hint.get("related_symbols", ())
                            ),
                            expected_action=str(hint.get("expected_action") or ""),
                            anchor_dates=hint.get("anchor_dates", ()),
                        )
        except Exception:
            report_binding = _sync_sec_http_attempts(report, report_binding, source)
            _finalize_report(report, report_binding, len(all_candidates))
            _write_report(output_path, report)
            raise
        successor_id = ""
        if evidence.parsed and evidence.parsed.new_symbol:
            successor_id = resolve_new_security_id(
                master,
                new_symbol=evidence.parsed.new_symbol,
                effective_date=evidence.parsed.effective_date,
                symbol_history=history,
            )
        crosscheck = _crosscheck(
            evidence.to_dict(),
            successor_security_id=successor_id,
            price_histories=price_histories,
        )
        eligible = bool(
            evidence.parsed
            and evidence.parsed.confidence == "high"
            and crosscheck["passed"]
            and not bool(hint.get("manual_review"))
            and (
                evidence.parsed.action_type not in {"stock_merger", "ticker_change"}
                or successor_id
            )
            and (
                evidence.parsed.action_type != "delisting"
                or evidence.parsed.cash_amount is not None
            )
        )
        value = evidence.to_dict()
        value.update(
            {
                "successor_security_id": successor_id,
                "crosscheck": crosscheck,
                "eligible_for_apply": eligible,
                "manual_review_reason": str(hint.get("manual_review_reason") or ""),
                "artifacts": [
                    {
                        "source": artifact.source,
                        "source_url": artifact.source_url,
                        "retrieved_at": artifact.retrieved_at,
                        "content_type": artifact.content_type,
                        "source_hash": artifact.source_hash,
                    }
                    for artifact in artifacts
                ],
            }
        )
        records[candidate.security_id] = value
        report_binding = _sync_sec_http_attempts(report, report_binding, source)
        _finalize_report(report, report_binding, len(all_candidates))
        _write_report(output_path, report)
        parsed = evidence.parsed.action_type if evidence.parsed else "unresolved"
        confidence = evidence.parsed.confidence if evidence.parsed else ""
        print(
            f"[{index}/{total}] {candidate.symbol}: {parsed} {confidence} "
            f"crosscheck={crosscheck['passed']} eligible={eligible}",
            flush=True,
        )

    if official_mode_enabled:
        evidence_cache = (
            Path(args.official_evidence_cache)
            if str(args.official_evidence_cache).strip()
            else repository.root / "state/sec_lifecycle"
        )
        official_source = OfficialLifecycleExceptionEvidenceSource(
            evidence_cache,
            allow_http=bool(args.fetch_official_exception_evidence),
            user_agent=os.getenv("SEC_USER_AGENT", ""),
        )
        _collect_official_exception_evidence(
            report,
            candidates=all_candidates,
            specs=official_specs,
            source=official_source,
            requested_symbols=requested,
            require_pinned=bool(args.verify_official_exception_evidence),
        )

    if source is not None:
        report_binding = _sync_sec_http_attempts(report, report_binding, source)

    _finalize_report(
        report,
        report_binding,
        len(all_candidates),
    )
    _write_report(output_path, report)
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    return 0


def _collect_verified_event(
    source: SecEdgarLifecycleSource,
    candidate,
    value: dict[str, Any],
):
    source_urls = tuple(str(item) for item in value.get("source_urls", ()) if item)
    if not source_urls:
        raise ValueError(f"verified_event for {candidate.symbol} requires source_urls")
    artifacts = tuple(source.fetch_url(url)[1] for url in source_urls)
    filing = SecFiling(
        cik=str(value["cik"]),
        accession_number=str(value["accession_number"]),
        filing_date=str(value["filing_date"]),
        form=str(value.get("form") or "8-K"),
        items=tuple(str(item) for item in value.get("items", ("2.01", "3.01"))),
        display_name=str(value.get("display_name") or candidate.name),
        score=100.0,
    )
    parsed = ParsedLifecycleEvent(
        action_type=str(value["action_type"]),
        effective_date=str(value["effective_date"]),
        cash_amount=(
            float(value["cash_amount"])
            if value.get("cash_amount") is not None
            else None
        ),
        ratio=float(value["ratio"]) if value.get("ratio") is not None else None,
        new_symbol=str(value.get("new_symbol") or ""),
        confidence="high",
        reason=str(value["reason"]),
    )
    return (
        LifecycleEvidence(
            candidate=candidate,
            filing=filing,
            parsed=parsed,
            source_url=source_urls[0],
            source_hash=artifacts[0].source_hash,
        ),
        artifacts,
    )


def _build_price_histories(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    value = prices.loc[:, ["security_id", "session", "close"]].copy()
    value["session"] = pd.to_datetime(value["session"], errors="coerce")
    value["close"] = pd.to_numeric(value["close"], errors="coerce")
    value = value.dropna(subset=["session", "close"])
    return {
        str(security_id): group.loc[:, ["session", "close"]]
        .sort_values("session")
        .reset_index(drop=True)
        for security_id, group in value.groupby(value["security_id"].astype(str))
    }


def _nearest_close(
    histories: dict[str, pd.DataFrame],
    security_id: str,
    effective: pd.Timestamp,
    *,
    direction: str,
    max_gap_days: int = 10,
) -> tuple[str, float] | None:
    frame = histories.get(str(security_id))
    if frame is None or frame.empty:
        return None
    if direction == "on_or_before":
        eligible = frame.loc[frame["session"] <= effective]
        distances = (effective - eligible["session"]).dt.days
    elif direction == "on_or_after":
        eligible = frame.loc[frame["session"] >= effective]
        distances = (eligible["session"] - effective).dt.days
    else:
        raise ValueError(f"Unsupported close direction: {direction}")
    if eligible.empty:
        return None
    index = distances.idxmin()
    gap = int(distances.loc[index])
    if gap > max_gap_days:
        return None
    row = eligible.loc[index]
    return pd.Timestamp(row["session"]).date().isoformat(), float(row["close"])


def _economic_crosscheck(
    evidence: dict[str, Any],
    *,
    successor_security_id: str,
    price_histories: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    parsed = evidence["parsed"]
    candidate = evidence["candidate"]
    action_type = str(parsed["action_type"])
    effective = pd.Timestamp(parsed["effective_date"])
    old_price = _nearest_close(
        price_histories,
        str(candidate["security_id"]),
        effective,
        direction="on_or_before",
    )
    result: dict[str, Any] = {
        "economic_terms_passed": False,
        "old_price_session": old_price[0] if old_price else None,
        "old_close": old_price[1] if old_price else None,
        "successor_price_session": None,
        "successor_close": None,
        "implied_consideration": None,
        "relative_deviation": None,
    }
    if action_type == "delisting" and parsed.get("cash_amount") is not None:
        result["economic_terms_passed"] = True
        return result
    if old_price is None:
        return result

    old_close = old_price[1]
    if action_type == "cash_merger":
        implied = float(parsed["cash_amount"])
    elif action_type in {"stock_merger", "ticker_change"}:
        if not successor_security_id:
            return result
        successor_price = _nearest_close(
            price_histories,
            successor_security_id,
            effective,
            direction="on_or_after",
        )
        if successor_price is None:
            return result
        result["successor_price_session"] = successor_price[0]
        result["successor_close"] = successor_price[1]
        if action_type == "ticker_change":
            implied = successor_price[1]
        else:
            implied = (
                float(parsed["ratio"]) * successor_price[1]
                + float(parsed.get("cash_amount") or 0.0)
            )
    else:
        return result

    denominator = max(abs(old_close), abs(implied), 1e-12)
    deviation = abs(old_close - implied) / denominator
    result["implied_consideration"] = implied
    result["relative_deviation"] = deviation
    result["economic_terms_passed"] = deviation <= 0.20
    return result


def _crosscheck(
    evidence: dict[str, Any],
    *,
    successor_security_id: str,
    price_histories: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    parsed = evidence.get("parsed")
    candidate = evidence["candidate"]
    if not parsed:
        return {
            "passed": False,
            "date_passed": False,
            "economic_terms_passed": False,
            "terminal_gap_days": None,
            "nearest_index_remove_gap_days": None,
            "basis": "no_parsed_event",
        }
    effective = pd.Timestamp(parsed["effective_date"])
    terminal_gap = abs((effective - pd.Timestamp(candidate["last_price_date"])).days)
    remove_gaps = [
        abs((effective - pd.Timestamp(value)).days)
        for value in candidate.get("index_remove_dates", ())
    ]
    remove_gap = min(remove_gaps) if remove_gaps else None
    if (
        parsed.get("action_type") == "delisting"
        and parsed.get("confidence") == "high"
        and parsed.get("cash_amount") is not None
        and effective >= pd.Timestamp(candidate["last_price_date"])
    ):
        economic = _economic_crosscheck(
            evidence,
            successor_security_id=successor_security_id,
            price_histories=price_histories,
        )
        return {
            "passed": True,
            "date_passed": True,
            "terminal_gap_days": terminal_gap,
            "nearest_index_remove_gap_days": remove_gap,
            "basis": "sec_cancellation_after_eodhd_terminal",
            **economic,
        }
    passed = terminal_gap <= 7 or (remove_gap is not None and remove_gap <= 7)
    basis = (
        "eodhd_terminal_price+index_remove"
        if terminal_gap <= 7 and remove_gap is not None and remove_gap <= 7
        else "eodhd_terminal_price"
        if terminal_gap <= 7
        else "index_remove"
        if remove_gap is not None and remove_gap <= 7
        else "date_mismatch"
    )
    economic = _economic_crosscheck(
        evidence,
        successor_security_id=successor_security_id,
        price_histories=price_histories,
    )
    return {
        "passed": passed and bool(economic["economic_terms_passed"]),
        "date_passed": passed,
        "terminal_gap_days": terminal_gap,
        "nearest_index_remove_gap_days": remove_gap,
        "basis": basis,
        **economic,
    }


def _validate_full_resume_report(
    report: Mapping[str, Any],
    expected_binding: Mapping[str, Any],
    candidates: Iterable[Any],
) -> None:
    validate_lifecycle_report_binding(
        report,
        expected_binding,
        purpose="official-evidence-only resume",
    )
    records = report.get("records")
    if not isinstance(records, dict):
        raise RuntimeError("Official-evidence-only mode requires a full records object.")
    expected = {str(item.security_id): item for item in candidates}
    if set(records) != set(expected):
        raise RuntimeError(
            "Official-evidence-only mode refuses a partial candidate report: "
            f"missing={sorted(set(expected) - set(records))}, "
            f"extra={sorted(set(records) - set(expected))}"
        )
    for security_id, candidate in expected.items():
        identity = records[security_id].get("candidate") or {}
        actual = (
            str(identity.get("security_id") or ""),
            str(identity.get("symbol") or "").upper(),
            str(identity.get("last_price_date") or ""),
        )
        wanted = (
            security_id,
            str(candidate.symbol).upper(),
            str(candidate.last_price_date),
        )
        if actual != wanted:
            raise RuntimeError(
                "Official-evidence-only candidate identity mismatch: "
                f"expected={wanted!r}, found={actual!r}"
            )


def _official_artifact_metadata(
    spec: OfficialLifecycleExceptionEvidenceSpec,
    artifact,
    *,
    matched_phrases: Iterable[str],
) -> dict[str, Any]:
    return {
        "source": artifact.source,
        "source_url": artifact.source_url,
        "retrieved_at": artifact.retrieved_at,
        "content_type": artifact.content_type,
        "source_hash": artifact.source_hash,
        "evidence_id": spec.evidence_id,
        "resolution_kind": spec.resolution_kind,
        "exception_code": spec.exception_code,
        "claim": spec.claim,
        "effective_date": spec.effective_date,
        "pin_status": "verified_pinned" if spec.pinned else "observed_unpinned",
        "matched_phrases": list(matched_phrases),
    }


def _collect_official_exception_evidence(
    report: dict[str, Any],
    *,
    candidates: Iterable[Any],
    specs: Mapping[str, OfficialLifecycleExceptionEvidenceSpec],
    source: OfficialLifecycleExceptionEvidenceSource,
    requested_symbols: set[str] | frozenset[str] = frozenset(),
    require_pinned: bool,
) -> None:
    selected = tuple(
        spec
        for spec in specs.values()
        if not requested_symbols
        or any(spec.targets_symbol(symbol) for symbol in requested_symbols)
    )
    if not selected:
        raise ValueError(
            "No official exception evidence target matches the requested symbols."
        )
    candidate_values = tuple(candidates)
    records = report.get("records")
    if not isinstance(records, dict):
        raise RuntimeError("Official exception evidence requires lifecycle report records.")
    evidence_report: dict[str, dict[str, Any]] = report.setdefault(
        "official_exception_evidence", {}
    )
    for retired in sorted(set(evidence_report) - set(specs)):
        evidence_report.pop(retired)
    selected_ids = {spec.evidence_id for spec in selected}
    for stale in sorted(set(evidence_report) & selected_ids):
        evidence_report.pop(stale)

    for spec in selected:
        if require_pinned and not spec.binding_complete:
            raise RuntimeError(
                "Official lifecycle exception evidence candidate binding is pending: "
                f"{spec.evidence_id}"
            )
        artifact, matched_phrases = source.load(
            spec,
            require_pinned=require_pinned,
        )
        matches = tuple(
            candidate
            for candidate in candidate_values
            if spec.matches_candidate(candidate)
        )
        if len(matches) > 1:
            raise RuntimeError(
                "Official lifecycle exception evidence matched multiple candidates: "
                f"{spec.evidence_id}/{[item.security_id for item in matches]}"
            )
        if require_pinned and len(matches) != 1:
            raise RuntimeError(
                "Pinned official lifecycle exception evidence did not match exactly one "
                f"current candidate: {spec.evidence_id}"
            )
        candidate = matches[0] if matches else None
        metadata = _official_artifact_metadata(
            spec,
            artifact,
            matched_phrases=matched_phrases,
        )
        if candidate is not None:
            record = records.get(str(candidate.security_id))
            if not isinstance(record, dict):
                raise RuntimeError(
                    "Official exception evidence candidate is absent from the report: "
                    f"{candidate.security_id}"
                )
            artifacts = [
                dict(item)
                for item in (record.get("artifacts") or ())
                if not (
                    isinstance(item, dict)
                    and str(item.get("evidence_id") or "") == spec.evidence_id
                )
            ]
            artifacts.append(metadata)
            record["artifacts"] = artifacts
            if require_pinned and spec.resolution_kind == "applied_event":
                verified_event = {
                    "action_type": spec.action_type,
                    "effective_date": spec.effective_date,
                    "cash_amount": spec.cash_amount,
                    "ratio": None,
                    "new_symbol": "",
                    "confidence": "high",
                    "filing_date": spec.filing_date,
                    "source_url": artifact.source_url,
                    "source_hash": artifact.source_hash,
                    "retrieved_at": artifact.retrieved_at,
                    "content_type": artifact.content_type,
                    "reason": spec.claim,
                }
                prior = record.get("verified_event")
                prior_evidence_id = str(
                    record.get("verified_event_evidence_id") or ""
                )
                if prior is not None and (
                    prior_evidence_id != spec.evidence_id
                    or not isinstance(prior, dict)
                    or any(
                        prior.get(field) != verified_event[field]
                        for field in (
                            "action_type",
                            "effective_date",
                            "cash_amount",
                            "source_url",
                            "source_hash",
                        )
                    )
                ):
                    raise RuntimeError(
                        "Official lifecycle verified-event promotion conflicts with an "
                        f"existing override: {spec.evidence_id}"
                    )
                record["verified_event"] = verified_event
                record["verified_event_evidence_id"] = spec.evidence_id
                record["manual_review"] = False
                record["manual_review_reason"] = ""
                record["eligible_for_apply"] = False

        evidence_report[spec.evidence_id] = {
            "status": (
                "verified_pinned_promoted"
                if spec.pinned
                and candidate is not None
                and spec.resolution_kind == "applied_event"
                else "verified_pinned_attached"
                if spec.pinned and candidate is not None
                else "verified_pinned_unbound"
                if spec.pinned
                else "observed_unpinned_attached"
                if candidate is not None
                else "observed_unpinned_unbound"
            ),
            "candidate_binding_status": spec.binding_status,
            "candidate_security_id": (
                str(candidate.security_id) if candidate is not None else ""
            ),
            "candidate_symbol": (
                str(candidate.symbol) if candidate is not None else ""
            ),
            "candidate_last_price_date": (
                str(candidate.last_price_date) if candidate is not None else ""
            ),
            "resolution_kind": spec.resolution_kind,
            "exception_code": spec.exception_code,
            "action_type": spec.action_type,
            "cash_amount": spec.cash_amount,
            "claim": spec.claim,
            "effective_date": spec.effective_date,
            "source_url": artifact.source_url,
            "observed_sha256": artifact.source_hash,
            "pinned_sha256": spec.source_sha256,
            "retrieved_at": artifact.retrieved_at,
            "content_type": artifact.content_type,
            "content_bytes": len(artifact.content),
            "matched_phrases": list(matched_phrases),
        }
    report["official_exception_evidence_http_attempts"] = source.http_attempts


def _load_hints(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        str(symbol).upper(): dict(value or {})
        for symbol, value in (raw.get("symbols") or {}).items()
    }


def _candidate_hint_key(candidate: Any) -> str:
    security_id = str(getattr(candidate, "security_id", "")).strip()
    parsed = pd.to_datetime(getattr(candidate, "last_price_date", ""), errors="coerce")
    if not security_id or pd.isna(parsed):
        raise ValueError("Lifecycle candidate requires an exact security_id and terminal date.")
    return f"{security_id}|{pd.Timestamp(parsed).date().isoformat()}"


def _load_identity_bound_hints(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    values = raw.get("identity_bound_hints") or {}
    if not isinstance(values, dict):
        raise ValueError("identity_bound_hints must be a YAML mapping.")
    output: dict[str, dict[str, Any]] = {}
    for raw_key, raw_value in values.items():
        key = str(raw_key).strip()
        if "|" not in key:
            raise ValueError(
                "Identity-bound lifecycle hint keys must be security_id|last_price_date."
            )
        security_id, terminal = key.rsplit("|", 1)
        parsed = pd.to_datetime(terminal, errors="coerce")
        if not security_id.strip() or pd.isna(parsed):
            raise ValueError(f"Invalid identity-bound lifecycle hint key: {key!r}")
        canonical = f"{security_id.strip()}|{pd.Timestamp(parsed).date().isoformat()}"
        if canonical != key:
            raise ValueError(
                "Identity-bound lifecycle hint keys must use canonical ISO dates: "
                f"{key!r}"
            )
        if not isinstance(raw_value, dict):
            raise ValueError(f"Identity-bound lifecycle hint must be an object: {key}")
        value = dict(raw_value)
        symbol = str(value.get("candidate_symbol") or "").strip().upper()
        if not symbol:
            raise ValueError(f"Identity-bound lifecycle hint has no candidate_symbol: {key}")
        verified = isinstance(value.get("verified_event"), dict)
        existing_release = isinstance(value.get("existing_release_event"), dict)
        if int(verified) + int(existing_release) != 1:
            raise ValueError(
                "Identity-bound lifecycle hint requires exactly one verified_event "
                f"or existing_release_event: {key}"
            )
        if existing_release and (
            canonical != _NTCO_RELEASE_EVENT_KEY
            or symbol != "NTCOY"
            or value["existing_release_event"]
            != {"contract": _NTCO_RELEASE_EVENT_CONTRACT}
        ):
            raise ValueError(
                "existing_release_event is restricted to the exact NTCOY contract."
            )
        value["candidate_symbol"] = symbol
        output[canonical] = value
    return output


def _hint_for_candidate(
    candidate: Any,
    symbol_hints: Mapping[str, Mapping[str, Any]],
    identity_bound_hints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    symbol = str(getattr(candidate, "symbol", "")).strip().upper()
    base = dict(symbol_hints.get(symbol) or {})
    key = _candidate_hint_key(candidate)
    exact = identity_bound_hints.get(key)
    exact_for_symbol = tuple(
        bound_key
        for bound_key, value in identity_bound_hints.items()
        if str(value.get("candidate_symbol") or "").strip().upper() == symbol
    )
    if exact is not None:
        exact_value = dict(exact)
        exact_symbol = str(exact_value.pop("candidate_symbol", "")).strip().upper()
        if exact_symbol != symbol:
            raise RuntimeError(
                "Identity-bound lifecycle hint symbol disagrees with the candidate: "
                f"{key}/{symbol}/{exact_symbol}"
            )
        if isinstance(base.get("verified_event"), dict):
            raise RuntimeError(
                "Symbol verified_event fallback collides with an exact identity-bound "
                f"lifecycle hint: {key}"
            )
        base.update(exact_value)
        return base
    if isinstance(base.get("verified_event"), dict) and exact_for_symbol:
        raise RuntimeError(
            "Symbol verified_event fallback is unsafe for an identity-bound ticker: "
            f"{key}; reviewed_bindings={sorted(exact_for_symbol)}"
        )
    return base


def _empty_report(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(binding), "records": {}, "summary": {}}


def _load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"records": {}, "summary": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _sync_sec_http_attempts(
    report: dict[str, Any],
    binding: Mapping[str, Any],
    source: SecEdgarLifecycleSource,
) -> dict[str, Any]:
    updated = update_lifecycle_report_http_attempts(
        binding,
        sec_http_attempts=source.http_attempts,
        sec_http_attempts_by_candidate=source.http_attempts_by_candidate,
    )
    for field, value in updated.items():
        if field in binding:
            report[field] = value
    return updated


def _load_or_initialize_report(
    path: Path,
    *,
    resume: bool,
    expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if not resume or not path.is_file():
        return _empty_report(expected_binding)
    report = _load_report(path)
    validate_lifecycle_report_binding(
        report,
        expected_binding,
        purpose="collector resume",
    )
    return report


def _finalize_report(
    report: dict[str, Any],
    expected_binding: Mapping[str, Any],
    candidate_count: int,
) -> None:
    validate_lifecycle_report_binding(
        report,
        expected_binding,
        purpose="report finalization",
    )
    records = list(report.get("records", {}).values())
    counts: dict[str, int] = {}
    for record in records:
        parsed = record.get("parsed") or {}
        key = str(parsed.get("action_type") or "unresolved")
        counts[key] = counts.get(key, 0) + 1
    report["summary"] = {
        "candidate_count": candidate_count,
        "collected_count": len(records),
        "eligible_count": sum(bool(item.get("eligible_for_apply")) for item in records),
        "unresolved_count": sum(not bool(item.get("eligible_for_apply")) for item in records),
        "action_type_counts": dict(sorted(counts.items())),
        "sec_fetch_policy": report["sec_fetch_policy"],
        "sec_max_http_attempts": int(report["sec_max_http_attempts"]),
        "sec_max_http_attempts_per_candidate": int(
            report["sec_max_http_attempts_per_candidate"]
        ),
        "sec_max_http_attempts_per_request": int(
            report["sec_max_http_attempts_per_request"]
        ),
        "sec_http_attempts": int(report["sec_http_attempts"]),
        "sec_http_attempts_remaining": int(report["sec_max_http_attempts"])
        - int(report["sec_http_attempts"]),
        "sec_http_attempts_by_candidate": dict(
            report["sec_http_attempts_by_candidate"]
        ),
    }
    official = report.get("official_exception_evidence") or {}
    if official:
        statuses: dict[str, int] = {}
        for value in official.values():
            status = str((value or {}).get("status") or "invalid")
            statuses[status] = statuses.get(status, 0) + 1
        report["summary"]["official_exception_evidence"] = {
            "evidence_count": len(official),
            "status_counts": dict(sorted(statuses.items())),
            "http_attempts": int(
                report.get("official_exception_evidence_http_attempts", 0)
            ),
        }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(
        path,
        (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
