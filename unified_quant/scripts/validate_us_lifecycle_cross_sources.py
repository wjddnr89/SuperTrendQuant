#!/usr/bin/env python3
"""Cross-validate US lifecycle events and stored OHLC before R2 publication.

Default execution is offline and read-only.  ``--fetch-missing`` performs at
most the policy's one-attempt-per-symbol Yahoo chart requests and only fills an
immutable local cache.  ``--apply`` is deliberately a separate, cache-only
step which archives all raw bytes and commits a two-dataset release.

Yahoo-primary OLD LILA/LILAK are never compared with Yahoo again.  Their
already-pinned CC0/Boris overlap bytes are reloaded and recomputed locally,
with the undisclosed upstream and 33-session tail retained in the report.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import io
import json
import math
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlparse

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import yaml

from supertrend_quant.config import DEFAULT_DATA_CONFIG_PATH, load_data_store_config
from supertrend_quant.market_store.cross_validation import (
    BLOCKED_NO_DATA_SUCCESSOR_CHAIN_TARGET_IDS,
    CROSS_VALIDATION_DATASET,
    CROSS_VALIDATION_SCHEMA,
    NONTERMINAL_EVENT_VALIDATION,
    PERMANENT_EXCEPTION_NO_DATA_CODE,
    PINNED_EXTERNAL_OVERLAP_VALIDATION,
    REVIEWED_NO_DATA_UNSUPPORTED_PATH_BASIS,
    REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_BASIS,
    REVIEWED_PERMANENT_EXCEPTION_NO_DATA_BASIS,
    REVIEWED_NO_DATA_SUCCESSOR_CHAIN_BASIS,
    TERMINAL_EVENT_VALIDATION,
    TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256,
    TRUSTED_REVIEWED_NO_DATA_SUCCESSOR_CHAIN_ROOT_TARGET_IDS,
    TRUSTED_REVIEWED_NO_DATA_SUCCESSOR_CHAINS_SHA256,
    TRUSTED_REVIEWED_NO_DATA_UNSUPPORTED_PATH_SHA256,
    TRUSTED_REVIEWED_NO_DATA_UNSUPPORTED_PATH_TARGET_IDS,
    TRUSTED_REVIEWED_TERMINAL_MARKET_DATE_CORRECTION_EVENT_IDS,
    TRUSTED_REVIEWED_TERMINAL_MARKET_DATE_CORRECTIONS_SHA256,
    TRUSTED_REVIEWED_TERMINAL_EVENT_GATE_EVENT_IDS,
    TRUSTED_REVIEWED_TERMINAL_EVENT_GATES_SHA256,
    TRUSTED_REVIEWED_TERMINAL_OVERRIDE_EVENT_IDS,
    TRUSTED_REVIEWED_TERMINAL_OVERRIDES_SHA256,
    TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTION_EVENT_IDS,
    TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256,
    TRUSTED_FRC_EVIDENCE_BINDING_EVENT_IDS,
    TRUSTED_FRC_EVIDENCE_BINDINGS_SHA256,
    TRUSTED_NTCO_EVIDENCE_BINDING_EVENT_IDS,
    TRUSTED_NTCO_EVIDENCE_BINDINGS_SHA256,
    TRUSTED_SIVB_EVIDENCE_BINDING_EVENT_IDS,
    TRUSTED_SIVB_EVIDENCE_BINDINGS_SHA256,
    VALIDATED_DATASETS,
    YAHOO_NO_DATA_TERMINAL_ACTION_TYPES,
    YAHOO_NO_DATA_TERMINAL_DATE_RELATIONS,
    YAHOO_NO_DATA_SUCCESSOR_VALIDATION_BASIS,
    canonical_json_bytes,
    canonical_json_sha256,
    dataframe_sha256,
    independent_provider_source_mask,
    _terminal_event_date_binding,
    pinned_external_overlap_spec_is_trusted,
    permanent_exception_spec_for_resolution,
    permanent_exception_no_data_binding,
    provider_affected_identity_ids,
    reviewed_nonterminal_extraction_mismatches,
    reviewed_nonterminal_extraction_sha256,
    reviewed_nonterminal_extractions,
    reviewed_nonterminal_inventory_sha256,
    reviewed_nonterminal_same_sid_no_data_binding,
    reviewed_no_data_successor_chain_inventory_sha256,
    reviewed_no_data_successor_chains,
    reviewed_no_data_unsupported_path_inventory_sha256,
    reviewed_no_data_unsupported_paths,
    reviewed_terminal_override_inventory_sha256,
    reviewed_terminal_override_mismatches,
    reviewed_terminal_override_sha256,
    reviewed_terminal_overrides,
    reviewed_terminal_market_date_action_mismatches,
    reviewed_terminal_market_date_correction_inventory_sha256,
    reviewed_terminal_market_date_correction_sha256,
    reviewed_terminal_market_date_corrections,
    reviewed_terminal_market_date_report_mismatches,
    reviewed_terminal_event_gate_inventory_sha256,
    reviewed_terminal_event_gate_mismatches,
    reviewed_terminal_event_gate_sha256,
    reviewed_terminal_event_gates,
    reviewed_terminal_report_mismatches,
    reviewed_terminal_price_tail_action_mismatches,
    reviewed_terminal_price_tail_correction_inventory_sha256,
    reviewed_terminal_price_tail_correction_sha256,
    reviewed_terminal_price_tail_corrections,
    reviewed_terminal_price_tail_report_mismatches,
    source_archive_binding_matches,
    successor_price_check_binding,
    unsupported_path_no_data_binding,
    trusted_frc_evidence_binding_diagnostic,
    trusted_frc_evidence_binding_inventory_sha256,
    trusted_frc_evidence_bindings,
    trusted_frc_report_diagnostic_passed,
    trusted_ntco_evidence_binding_diagnostic,
    trusted_ntco_evidence_binding_inventory_sha256,
    trusted_ntco_evidence_bindings,
    trusted_ntco_report_diagnostic_passed,
    trusted_sivb_evidence_binding_diagnostic,
    trusted_sivb_evidence_binding_inventory_sha256,
    trusted_sivb_evidence_bindings,
    trusted_sivb_report_diagnostic_passed,
    trusted_permanent_exception_specs,
)
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    sha256_bytes,
    write_atomic,
)
from supertrend_quant.market_store.lifecycle_coverage import lifecycle_candidate_id
from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.reviewed_price_evidence import (
    REVIEWED_PRICE_EVIDENCE_BASIS,
    TRUSTED_REVIEWED_PRICE_EVIDENCE_SHA256,
    TRUSTED_REVIEWED_PRICE_EVIDENCE_TARGET_IDS,
    build_reviewed_price_projection,
    reviewed_price_evidence_inventory_sha256,
    reviewed_price_evidence_registry,
    reviewed_price_evidence_sha256,
    verify_reviewed_price_projection,
)
from supertrend_quant.market_store.reviewed_remaining_price_exceptions import (
    TRUSTED_REVIEWED_REMAINING_PRICE_EXCEPTION_INVENTORY_SHA256,
    apply_reviewed_remaining_price_exceptions,
)
from supertrend_quant.market_store.source_archive_price_evidence import (
    REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_BASIS,
    TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_SHA256,
    TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_TARGET_IDS,
    WIKI_DOWNLOAD_URL,
    WIKI_EXTRACT_RETRIEVED_AT,
    WIKI_EXTRACT_SHA256,
    WIKI_PROVENANCE_SHA256,
    source_archive_price_only_inventory_sha256,
    source_archive_price_only_registry,
    verify_source_archive_price_only_evidence,
)
from supertrend_quant.market_store.wiki14_price_evidence import (
    REVIEWED_WIKI14_PRICE_ONLY_BASIS,
    TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_SHA256,
    TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_TARGET_IDS,
    WIKI14_DOWNLOAD_URL,
    WIKI14_EXTRACT_RETRIEVED_AT,
    WIKI14_PROVENANCE_SHA256,
    verify_wiki14_price_only_evidence,
    wiki14_price_only_inventory_sha256,
    wiki14_price_only_registry,
)
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.storage import ObjectNotFound
from supertrend_quant.market_store.terminal_policy_exceptions import (
    TRUSTED_REVIEWED_TERMINAL_POLICY_EXCEPTION_EVENT_IDS,
    TRUSTED_REVIEWED_TERMINAL_POLICY_EXCEPTIONS_SHA256,
    reviewed_terminal_policy_action_mismatches,
    reviewed_terminal_policy_exception_inventory_sha256,
    reviewed_terminal_policy_exception_sha256,
    reviewed_terminal_policy_exceptions,
    reviewed_terminal_policy_report_mismatches,
)
from supertrend_quant.market_store.yahoo_chart import (
    ALLOWED_US_EXCHANGE_NAMES,
    US_EXCHANGE_TIMEZONE,
    YahooChartCache as RawYahooChartCache,
    YahooChartCachedResponse as CachedResponse,
    normalize_yahoo_symbol,
    parse_yahoo_chart_json,
    parse_yahoo_chart_no_data_evidence,
)
from supertrend_quant.market_store.validation import validate_dataset


DEFAULT_POLICY = Path(__file__).parents[1] / "configs/us_cross_validation.yaml"
DEFAULT_CACHE = Path("data/cache/state/us_cross_validation/yahoo_chart")
DEFAULT_OUTPUT_ROOT = Path("results/data_quality/us_cross_validation")
LIFECYCLE_ACTION_TYPES = frozenset(
    {"cash_merger", "stock_merger", "spinoff", "ticker_change", "delisting"}
)
SPLIT_ACTION_TYPES = frozenset({"split", "capital_reduction", "stock_dividend"})
REPORT_SOURCE = "us_lifecycle_cross_validation"
YAHOO_SOURCE = "yahoo_chart_json"
YAHOO_ENVELOPE_SOURCE = "yahoo_chart_cache_envelope"
PERMANENT_EXCEPTION_CODES = frozenset(
    {"unsupported_consideration", "recovery_uncertain"}
)
PERMANENT_EXCEPTION_VALIDATION = (
    "permanent_lifecycle_exception_official_provenance"
)


@dataclass(frozen=True)
class Policy:
    value: dict[str, Any]

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.value)

    @property
    def provider(self) -> Mapping[str, Any]:
        return self.value["provider"]

    @property
    def events(self) -> Mapping[str, Any]:
        return self.value["events"]

    @property
    def prices(self) -> Mapping[str, Any]:
        return self.value["prices"]


@dataclass(frozen=True)
class PriceTarget:
    security_id: str
    symbol: str
    origins: tuple[str, ...]
    active_from: str = ""
    active_to: str = ""
    terminal_event_id: str = ""
    successor_security_id: str = ""
    request_start: str = ""
    request_end: str = ""

    @property
    def provider_symbol(self) -> str:
        return normalize_yahoo_symbol(self.symbol)

    @property
    def target_id(self) -> str:
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "provider": "yahoo_chart",
                    "security_id": self.security_id,
                    "provider_symbol": self.provider_symbol,
                    "active_from": self.active_from,
                    "active_to": self.active_to,
                }
            )
        )


@dataclass(frozen=True)
class ArchiveArtifact:
    source: str
    source_url: str
    retrieved_at: str
    content: bytes
    content_type: str
    object_path: str

    @property
    def source_hash(self) -> str:
        return sha256_bytes(self.content)


@dataclass(frozen=True)
class PinnedOverlapEvidence:
    spec: dict[str, Any]
    primary_prices: pd.DataFrame
    external_prices: pd.DataFrame
    primary_source_url: str
    primary_source_hash: str
    external_source_url: str
    external_source_hash: str
    retrieved_at: str


@dataclass(frozen=True)
class PreparedCrossValidation:
    release: DataRelease
    release_etag: str | None
    pointer_etags: dict[str, str | None]
    planned_versions: dict[str, str]
    report: dict[str, Any]
    report_bytes: bytes
    report_hash: str
    frames: dict[str, pd.DataFrame]
    artifacts: tuple[ArchiveArtifact, ...]
    summary: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else pd.Timestamp(parsed).date().isoformat()


def _bounded_yahoo_request(target: PriceTarget) -> tuple[str, str, int, int]:
    """Return an exact inclusive date window and Yahoo's exclusive period2."""

    start = _date(target.request_start or target.active_from)
    end = _date(target.request_end or target.active_to)
    _require(
        bool(start and end),
        f"Yahoo request bounds are incomplete for {target.security_id}/{target.symbol}.",
    )
    start_day = pd.Timestamp(start, tz="UTC")
    end_day = pd.Timestamp(end, tz="UTC")
    _require(
        end_day >= start_day,
        f"Yahoo request bounds are reversed for {target.security_id}/{target.symbol}.",
    )
    period1 = int(start_day.timestamp())
    period2 = int((end_day + pd.Timedelta(days=1)).timestamp())
    return start, end, period1, period2


def _number(value: Any) -> float | None:
    parsed = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(parsed) else float(parsed)


def _same_number(left: Any, right: Any) -> bool:
    a, b = _number(left), _number(right)
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def require_no_temporary_lifecycle_exceptions(
    resolutions: pd.DataFrame,
) -> None:
    """Fail before any provider access while a lifecycle recheck is pending."""

    _require(
        {"resolution", "recheck_after"}.issubset(resolutions.columns),
        "Lifecycle resolutions lack temporary-exception fields.",
    )
    temporary = resolutions.loc[
        resolutions["resolution"].astype(str).eq("exception")
        & resolutions["recheck_after"].fillna("").astype(str).str.strip().ne("")
    ]
    _require(
        temporary.empty,
        "Lifecycle temporary exceptions must be zero before cross-validation: "
        f"found {len(temporary)}.",
    )


def _official_host(url: str, allowed: Iterable[str]) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and any(
        host == domain.lower() or host.endswith("." + domain.lower())
        for domain in allowed
    )


def _official_exception_url(url: str) -> bool:
    parsed = urlparse(_text(url))
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.query
        or parsed.fragment
    ):
        return False
    if host == "www.sec.gov":
        return parsed.path.startswith("/Archives/edgar/data/")
    if host == "www.fdic.gov":
        return parsed.path.startswith("/resources/resolutions/bank-failures/")
    return False


def load_policy(path: Path) -> Policy:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), "Cross-validation policy must be an object.")
    _require(raw.get("schema_version") == 6, "Unsupported cross-validation policy.")
    provider = raw.get("provider")
    events = raw.get("events")
    prices = raw.get("prices")
    _require(isinstance(provider, dict), "Policy provider section is missing.")
    _require(isinstance(events, dict), "Policy events section is missing.")
    _require(isinstance(prices, dict), "Policy prices section is missing.")
    _require(
        provider.get("name") == "yahoo_chart",
        "Only Yahoo chart is accepted as provider.",
    )
    template = str(provider.get("endpoint_template", ""))
    _require(
        template == "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        "Yahoo chart endpoint template is invalid.",
    )
    _require(
        int(provider.get("max_http_attempts", 0)) == 400,
        "Yahoo chart HTTP attempt cap must be exactly 400.",
    )
    _require(
        int(provider.get("max_attempts_per_target", -1)) == 1
        and int(provider.get("retry_count", -1)) == 0,
        "Yahoo chart policy must allow one attempt per target and no retries.",
    )
    _require(
        provider.get("repository_visibility") == "private"
        and provider.get("r2_visibility") == "private"
        and provider.get("redistribution_allowed") is False
        and bool(_text(provider.get("use_restriction"))),
        "Yahoo chart use is restricted to personal use in private repository/R2 storage.",
    )
    _require(
        set(events.get("action_types", ())) == LIFECYCLE_ACTION_TYPES,
        "Policy must cover all five lifecycle action types.",
    )
    terminal_hosts = tuple(_text(item).lower() for item in events.get("official_hosts", ()))
    provenance_hosts = tuple(
        _text(item).lower()
        for item in events.get("official_provenance_hosts", ())
    )
    provenance_kinds = {
        _text(item) for item in events.get("official_provenance_source_kinds", ())
    }
    _require(
        set(terminal_hosts) == {"sec.gov"}
        and set(provenance_hosts)
        == {
            "sec.gov",
            "spglobal.com",
            "investor.ovintiv.com",
            "investors.qvcgrp.com",
        },
        "Official provenance hosts must remain the code-pinned reviewed set.",
    )
    _require(
        provenance_kinds
        == {
            "official_crosscheck",
            "official_filing",
            "official_filing_exit_mark",
            "official_issuer_plus_sec_crosscheck",
            "official_issuer_market_transition",
        },
        "Official provenance kinds must remain the code-pinned reviewed set.",
    )
    reviewed = reviewed_nonterminal_extractions(events)
    _require(
        bool(reviewed),
        "Policy must contain reviewed nonterminal extraction inventory.",
    )
    _require(
        reviewed_nonterminal_inventory_sha256(events)
        == TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256,
        "Reviewed nonterminal extraction inventory is not code-pinned.",
    )
    for event_id, extraction in reviewed.items():
        _require(
            _official_host(extraction["source_url"], provenance_hosts)
            and extraction["source_kind"] in provenance_kinds,
            "Reviewed nonterminal extraction has unapproved provenance: "
            + event_id,
        )
    event_gates = reviewed_terminal_event_gates(events)
    _require(
        set(event_gates) == set(TRUSTED_REVIEWED_TERMINAL_EVENT_GATE_EVENT_IDS)
        and reviewed_terminal_event_gate_inventory_sha256(events)
        == TRUSTED_REVIEWED_TERMINAL_EVENT_GATES_SHA256,
        "Reviewed terminal event-gate inventory is not code-pinned.",
    )
    terminal_overrides = reviewed_terminal_overrides(events)
    _require(
        set(terminal_overrides)
        == set(TRUSTED_REVIEWED_TERMINAL_OVERRIDE_EVENT_IDS)
        and reviewed_terminal_override_inventory_sha256(events)
        == TRUSTED_REVIEWED_TERMINAL_OVERRIDES_SHA256,
        "Reviewed terminal override inventory is not code-pinned.",
    )
    for event_id, override in terminal_overrides.items():
        _require(
            _official_host(override["source_url"], terminal_hosts)
            and override["source_kind"] == "official_crosscheck",
            "Reviewed terminal override has unapproved provenance: "
            + event_id,
        )
    market_date_corrections = reviewed_terminal_market_date_corrections(events)
    _require(
        set(market_date_corrections)
        == set(TRUSTED_REVIEWED_TERMINAL_MARKET_DATE_CORRECTION_EVENT_IDS)
        and reviewed_terminal_market_date_correction_inventory_sha256(events)
        == TRUSTED_REVIEWED_TERMINAL_MARKET_DATE_CORRECTIONS_SHA256,
        "Reviewed terminal market-date correction inventory is not code-pinned.",
    )
    _require(
        not (set(market_date_corrections) & set(terminal_overrides))
        and not (set(market_date_corrections) & set(reviewed)),
        "Terminal market-date corrections must remain separate from other "
        "reviewed exception inventories.",
    )
    for event_id, correction in market_date_corrections.items():
        _require(
            _official_host(correction["source_url"], terminal_hosts)
            and _official_host(correction["report_source_url"], terminal_hosts)
            and correction["source_kind"] == "official_crosscheck",
            "Reviewed terminal market-date correction has unapproved provenance: "
            + event_id,
        )
    policy_exceptions = reviewed_terminal_policy_exceptions(events)
    _require(
        set(policy_exceptions)
        == set(TRUSTED_REVIEWED_TERMINAL_POLICY_EXCEPTION_EVENT_IDS)
        and reviewed_terminal_policy_exception_inventory_sha256(events)
        == TRUSTED_REVIEWED_TERMINAL_POLICY_EXCEPTIONS_SHA256,
        "Reviewed terminal policy exception inventory is not code-pinned.",
    )
    _require(
        not (set(policy_exceptions) & set(market_date_corrections))
        and not (set(policy_exceptions) & set(terminal_overrides))
        and not (set(policy_exceptions) & set(reviewed)),
        "Terminal policy exceptions must remain separate from every other "
        "reviewed exception inventory.",
    )
    for event_id, exception in policy_exceptions.items():
        _require(
            _official_host(exception["source_url"], terminal_hosts)
            and _official_host(exception["report_source_url"], terminal_hosts),
            "Reviewed terminal policy exception has unapproved SEC provenance: "
            + event_id,
        )
    tail_corrections = reviewed_terminal_price_tail_corrections(events)
    _require(
        set(tail_corrections)
        == set(TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTION_EVENT_IDS)
        and reviewed_terminal_price_tail_correction_inventory_sha256(events)
        == TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256,
        "Reviewed terminal price-tail correction inventory is not code-pinned.",
    )
    _require(
        not (set(tail_corrections) & set(policy_exceptions))
        and not (set(tail_corrections) & set(market_date_corrections))
        and not (set(tail_corrections) & set(terminal_overrides))
        and not (set(tail_corrections) & set(reviewed)),
        "Terminal price-tail corrections must remain separate from other "
        "reviewed exception inventories.",
    )
    for event_id, correction in tail_corrections.items():
        _require(
            _official_host(correction["official_source_url"], terminal_hosts),
            "Reviewed terminal price-tail correction has unapproved provenance: "
            + event_id,
        )
    sivb_bindings = trusted_sivb_evidence_bindings()
    _require(
        set(sivb_bindings) == set(TRUSTED_SIVB_EVIDENCE_BINDING_EVENT_IDS)
        and trusted_sivb_evidence_binding_inventory_sha256()
        == TRUSTED_SIVB_EVIDENCE_BINDINGS_SHA256
        and not (set(sivb_bindings) & set(reviewed))
        and not (set(sivb_bindings) & set(terminal_overrides))
        and not (set(sivb_bindings) & set(policy_exceptions))
        and not (set(sivb_bindings) & set(tail_corrections))
        and not (set(sivb_bindings) & set(market_date_corrections)),
        "Trusted SIVB evidence bindings overlap an unreviewed policy path.",
    )
    frc_bindings = trusted_frc_evidence_bindings()
    _require(
        set(frc_bindings) == set(TRUSTED_FRC_EVIDENCE_BINDING_EVENT_IDS)
        and trusted_frc_evidence_binding_inventory_sha256()
        == TRUSTED_FRC_EVIDENCE_BINDINGS_SHA256
        and not (set(frc_bindings) & set(reviewed))
        and not (set(frc_bindings) & set(terminal_overrides))
        and not (set(frc_bindings) & set(policy_exceptions))
        and not (set(frc_bindings) & set(tail_corrections))
        and not (set(frc_bindings) & set(market_date_corrections)),
        "Trusted FRC evidence bindings overlap another reviewed policy path.",
    )
    ntco_bindings = trusted_ntco_evidence_bindings()
    configured_ntco_ids = {
        _text(value)
        for value in events.get("reviewed_ntco_transition_event_ids", ())
    }
    _require(
        configured_ntco_ids == set(TRUSTED_NTCO_EVIDENCE_BINDING_EVENT_IDS)
        and set(ntco_bindings) == set(TRUSTED_NTCO_EVIDENCE_BINDING_EVENT_IDS)
        and trusted_ntco_evidence_binding_inventory_sha256()
        == TRUSTED_NTCO_EVIDENCE_BINDINGS_SHA256
        and not (set(ntco_bindings) & set(reviewed))
        and not (set(ntco_bindings) & set(terminal_overrides))
        and not (set(ntco_bindings) & set(policy_exceptions))
        and not (set(ntco_bindings) & set(tail_corrections))
        and not (set(ntco_bindings) & set(market_date_corrections))
        and not (set(ntco_bindings) & set(sivb_bindings))
        and not (set(ntco_bindings) & set(frc_bindings)),
        "Trusted NTCO evidence bindings are not the exact isolated policy set.",
    )
    reviewed_prices = reviewed_price_evidence_registry(prices)
    _require(
        set(reviewed_prices) == set(TRUSTED_REVIEWED_PRICE_EVIDENCE_TARGET_IDS)
        and reviewed_price_evidence_inventory_sha256(prices)
        == TRUSTED_REVIEWED_PRICE_EVIDENCE_SHA256,
        "Reviewed price-evidence inventory is not code-pinned.",
    )
    source_archive_price_only = source_archive_price_only_registry(prices)
    _require(
        set(source_archive_price_only)
        == set(TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_TARGET_IDS)
        and source_archive_price_only_inventory_sha256(prices)
        == TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_SHA256
        and not (set(source_archive_price_only) & set(reviewed_prices)),
        "Frozen WIKI price-only evidence inventory is not the exact isolated "
        "code-pinned pair.",
    )
    wiki14_price_only = wiki14_price_only_registry(prices)
    _require(
        set(wiki14_price_only)
        == set(TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_TARGET_IDS)
        and wiki14_price_only_inventory_sha256(prices)
        == TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_SHA256
        and not (set(wiki14_price_only) & set(reviewed_prices))
        and not (set(wiki14_price_only) & set(source_archive_price_only)),
        "Frozen WIKI14 price-only evidence inventory is not the exact "
        "isolated code-pinned set.",
    )
    unsupported_paths = reviewed_no_data_unsupported_paths(prices)
    _require(
        set(unsupported_paths)
        == set(TRUSTED_REVIEWED_NO_DATA_UNSUPPORTED_PATH_TARGET_IDS)
        and reviewed_no_data_unsupported_path_inventory_sha256(prices)
        == TRUSTED_REVIEWED_NO_DATA_UNSUPPORTED_PATH_SHA256
        and not (set(unsupported_paths) & set(reviewed_prices))
        and not (set(unsupported_paths) & set(source_archive_price_only))
        and not (set(unsupported_paths) & set(wiki14_price_only)),
        "Reviewed no-data unsupported-path inventory is not the exact isolated "
        "code-pinned set.",
    )
    for target_id, spec in unsupported_paths.items():
        _require(
            _official_host(spec["official_source_url"], terminal_hosts),
            "Reviewed no-data unsupported-path evidence has unapproved "
            "provenance: "
            + target_id,
        )
    reviewed_successor_chains = reviewed_no_data_successor_chains(prices)
    _require(
        set(reviewed_successor_chains)
        == set(TRUSTED_REVIEWED_NO_DATA_SUCCESSOR_CHAIN_ROOT_TARGET_IDS)
        and reviewed_no_data_successor_chain_inventory_sha256(prices)
        == TRUSTED_REVIEWED_NO_DATA_SUCCESSOR_CHAINS_SHA256,
        "Reviewed no-data successor-chain inventory is not code-pinned.",
    )
    _require(
        not (set(reviewed_successor_chains) & set(unsupported_paths)),
        "No-data finite chains and unsupported-path exceptions must be disjoint.",
    )
    reviewed_chain_target_ids = {
        target_id
        for chain in reviewed_successor_chains.values()
        for target_id in (
            [node["target_id"] for node in chain["nodes"]]
            + [chain["final"]["target_id"]]
        )
    }
    _require(
        not (
            reviewed_chain_target_ids
            & set(BLOCKED_NO_DATA_SUCCESSOR_CHAIN_TARGET_IDS)
        ),
        "Reviewed no-data successor chains include a blocked cycle or ticker reuse.",
    )
    for key in (
        "minimum_overlap_sessions",
        "terminal_calendar_window_sessions",
        "minimum_split_regime_sessions",
    ):
        _require(int(prices.get(key, 0)) > 0, f"Policy {key} must be positive.")
    for key in (
        "minimum_session_coverage_ratio",
        "close_relative_tolerance",
        "ohl_relative_tolerance",
        "scale_stability_relative_tolerance",
    ):
        value = float(prices.get(key, -1))
        _require(0 <= value <= 1, f"Policy {key} is out of range.")
    no_data_action_types = prices.get("no_data_terminal_action_types")
    _require(
        isinstance(no_data_action_types, list)
        and len(no_data_action_types) == len(YAHOO_NO_DATA_TERMINAL_ACTION_TYPES)
        and {_text(value).lower() for value in no_data_action_types}
        == set(YAHOO_NO_DATA_TERMINAL_ACTION_TYPES),
        "Yahoo no-data terminal action allowlist must be exact.",
    )
    no_data_date_relations = prices.get("no_data_terminal_date_relations")
    _require(
        isinstance(no_data_date_relations, list)
        and len(no_data_date_relations)
        == len(YAHOO_NO_DATA_TERMINAL_DATE_RELATIONS)
        and {_text(value) for value in no_data_date_relations}
        == set(YAHOO_NO_DATA_TERMINAL_DATE_RELATIONS)
        and prices.get("no_data_successor_validation_basis")
        == YAHOO_NO_DATA_SUCCESSOR_VALIDATION_BASIS,
        "Yahoo no-data date/successor policy must be exact.",
    )
    _require(
        prices.get("currency") == "USD"
        and prices.get("instrument_type") == "EQUITY"
        and set(prices.get("allowed_exchange_names") or ())
        == set(ALLOWED_US_EXCHANGE_NAMES)
        and prices.get("exchange_timezone") == US_EXCHANGE_TIMEZONE
        and "indicators.quote" in str(prices.get("adjustment_basis", ""))
        and "adjclose is never" in str(prices.get("adjustment_basis", "")),
        "Yahoo price policy must use USD US-equity raw quote OHLCV, never adjusted close.",
    )
    overlap_specs = prices.get("pinned_external_overlaps")
    _require(
        isinstance(overlap_specs, list)
        and {_text(item.get("symbol")).upper() for item in overlap_specs if isinstance(item, dict)}
        == {"LILA", "LILAK"},
        "Pinned external overlap policy must cover exactly OLD LILA and LILAK.",
    )
    for spec in overlap_specs:
        _require(isinstance(spec, dict), "Pinned external overlap entry is invalid.")
        external_url = urlparse(_text(spec.get("external_source_url")))
        primary_url = urlparse(_text(spec.get("primary_source_url")))
        _require(
            pinned_external_overlap_spec_is_trusted(spec)
            and spec.get("primary_source") == "yahoo_chart_adjusted_basis_primary"
            and spec.get("external_source") == "boris_kaggle_cc0_v3"
            and primary_url.scheme == "https"
            and (primary_url.hostname or "").lower() == "query1.finance.yahoo.com"
            and external_url.scheme == "https"
            and (external_url.hostname or "").lower() == "www.kaggle.com"
            and len(_text(spec.get("external_source_sha256"))) == 64
            and int(spec.get("overlap_sessions", 0)) == 597
            and int(spec.get("primary_sessions", 0)) == 630
            and int(spec.get("uncrosschecked_tail_sessions", 0)) == 33
            and spec.get("upstream_provider_disclosed") is False
            and spec.get("independent_provider_claimed") is False,
            "Pinned external overlap controls changed.",
        )
    return Policy(dict(raw))


def _safe_archive_path(root: Path, object_path: str) -> Path:
    base = root.resolve()
    target = (base / object_path).resolve()
    _require(target != base and base in target.parents, f"Unsafe archive path: {object_path}")
    return target


def _archive_content(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    archive_id: str,
) -> bytes:
    matches = archive.loc[archive["archive_id"].astype(str).eq(archive_id)]
    _require(len(matches) == 1, f"Expected one source_archive row for {archive_id}.")
    path = _safe_archive_path(repository.root, str(matches.iloc[0]["object_path"]))
    _require(path.is_file(), f"Archived source payload is missing: {path}")
    encoded = path.read_bytes()
    payload = gzip.decompress(encoded) if path.suffix == ".gz" else encoded
    _require(sha256_bytes(payload) == archive_id, f"Archived source hash mismatch: {archive_id}")
    return payload


def build_permanent_exception_checks(
    repository: LocalDatasetRepository,
    resolutions: pd.DataFrame,
    archive: pd.DataFrame,
    official_evidence_specs: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Prove permanent exceptions from official bytes, never a generated report."""

    selector_fields = {"resolution", "exception_code", "recheck_after"}
    _require(
        selector_fields.issubset(resolutions.columns),
        "Lifecycle resolutions lack permanent-exception selector fields.",
    )
    permanent = resolutions.loc[
        resolutions["resolution"].astype(str).eq("exception")
        & resolutions["exception_code"].astype(str).isin(PERMANENT_EXCEPTION_CODES)
        & resolutions["recheck_after"].fillna("").astype(str).str.strip().eq("")
    ]
    if permanent.empty:
        return []
    trusted_specs = (
        dict(official_evidence_specs)
        if official_evidence_specs is not None
        else trusted_permanent_exception_specs()
    )
    required = {
        "candidate_id",
        "security_id",
        "symbol",
        "last_price_date",
        "resolution",
        "exception_code",
        "exception_reason",
        "recheck_after",
        "source_url",
        "source_hash",
    }
    _require(
        required.issubset(resolutions.columns),
        "Lifecycle resolutions lack permanent-exception provenance fields.",
    )
    checks: list[dict[str, Any]] = []
    for row in permanent.sort_values("candidate_id", kind="stable").to_dict(
        orient="records"
    ):
        reasons: list[str] = []
        security_id = _text(row.get("security_id"))
        last_price_date = _date(row.get("last_price_date"))
        candidate_id = _text(row.get("candidate_id"))
        expected_candidate_id = (
            lifecycle_candidate_id(security_id, last_price_date)
            if security_id and last_price_date
            else ""
        )
        identity_date_bound = bool(
            candidate_id
            and candidate_id == expected_candidate_id
            and _text(row.get("symbol"))
            and _text(row.get("exception_reason"))
        )
        if not identity_date_bound:
            reasons.append("candidate identity/date binding")

        official_spec = permanent_exception_spec_for_resolution(
            row, trusted_specs
        )
        registry_binding_passed = official_spec is not None
        if not registry_binding_passed:
            reasons.append("exact code-pinned official evidence registry binding")

        source_url = _text(row.get("source_url"))
        evidence_hash = _text(row.get("source_hash")).lower()
        official_original = _official_exception_url(source_url)
        if not official_original:
            reasons.append("official SEC/FDIC URL")
        valid_hash = len(evidence_hash) == 64 and all(
            value in "0123456789abcdef" for value in evidence_hash
        )
        if not valid_hash:
            reasons.append("exact SHA-256")
        reviewer_pin_passed = bool(
            official_spec is not None
            and official_spec.pinned
            and _text(official_spec.exception_code)
            == _text(row.get("exception_code"))
            and _text(official_spec.claim) == _text(row.get("exception_reason"))
            and official_spec.source_url == source_url
            and official_spec.source_sha256 == evidence_hash
        )
        if not reviewer_pin_passed:
            reasons.append("exact reviewer-pinned code/claim/URL/SHA")
        matches = archive.loc[
            archive.get("archive_id", pd.Series(index=archive.index, dtype="object"))
            .astype(str)
            .eq(evidence_hash)
        ]
        exact_archive_pair = bool(
            valid_hash
            and len(matches) == 1
            and _text(matches.iloc[0].get("source_hash")).lower() == evidence_hash
            and _text(matches.iloc[0].get("source_url")) == source_url
        )
        if not exact_archive_pair:
            reasons.append("exact archived URL/hash pair")
        archive_payload_verified = False
        if exact_archive_pair:
            try:
                archive_payload_verified = (
                    sha256_bytes(
                        _archive_content(repository, archive, evidence_hash)
                    )
                    == evidence_hash
                )
            except (OSError, RuntimeError, ValueError):
                archive_payload_verified = False
        if not archive_payload_verified:
            reasons.append("archived official payload bytes")
        checks.append(
            {
                "validation_kind": PERMANENT_EXCEPTION_VALIDATION,
                "evidence_id": (
                    official_spec.evidence_id if official_spec is not None else ""
                ),
                "candidate_id": candidate_id,
                "security_id": security_id,
                "symbol": _text(row.get("symbol")).upper(),
                "last_price_date": last_price_date,
                "exception_code": _text(row.get("exception_code")),
                "exception_reason": _text(row.get("exception_reason")),
                "status": "passed" if not reasons else "mismatch",
                "identity_date_bound": identity_date_bound,
                "registry_binding_passed": registry_binding_passed,
                "reviewer_pin_passed": reviewer_pin_passed,
                "official_original": official_original,
                "exact_archive_pair": exact_archive_pair,
                "archive_payload_verified": archive_payload_verified,
                "source_url": source_url,
                "evidence_sha256": evidence_hash,
                "reasons": reasons,
            }
        )
    return checks


def _lifecycle_evidence_report(
    repository: LocalDatasetRepository,
    release: DataRelease,
    archive: pd.DataFrame,
) -> tuple[dict[str, Any], str]:
    version = release.dataset_versions.get("lifecycle_resolutions", "")
    _require(version, "Release has no lifecycle_resolutions.")
    manifest = repository.manifest_for_version("lifecycle_resolutions", version)
    report_hash = _text(manifest.metadata.get("evidence_report_sha256")).lower()
    _require(len(report_hash) == 64, "Lifecycle evidence report hash is missing.")
    try:
        value = json.loads(_archive_content(repository, archive, report_hash))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Lifecycle evidence report is invalid JSON.") from exc
    _require(isinstance(value, dict), "Lifecycle evidence report must be an object.")
    _require(isinstance(value.get("records"), dict), "Lifecycle evidence report has no records.")
    return value, report_hash


def _event_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    event = record.get("verified_event")
    if isinstance(event, dict):
        return dict(event)
    parsed = record.get("parsed")
    if not isinstance(parsed, dict):
        return {}
    value = dict(parsed)
    value["source_url"] = record.get("source_url")
    value["source_hash"] = record.get("source_hash")
    return value


def build_event_checks(
    actions: pd.DataFrame,
    resolutions: pd.DataFrame,
    lifecycle_report: Mapping[str, Any],
    archive: pd.DataFrame,
    policy: Policy,
    *,
    lifecycle_report_sha256: str = "",
) -> list[dict[str, Any]]:
    """Validate terminal resolutions and nonterminal official actions separately.

    A terminal action is defined by an ``applied`` lifecycle resolution that
    references its event_id.  Only those actions are required to bind one-to-one
    to the lifecycle evidence report.  Other stored lifecycle actions (most
    notably intermediate ticker transitions) are still fail-closed, but against
    their own exact official URL/hash archive provenance instead of a terminal
    candidate report that was never meant to describe them.
    """

    action_values = [
        row
        for row in actions.to_dict(orient="records")
        if _text(row.get("action_type")).lower() in LIFECYCLE_ACTION_TYPES
    ]
    action_ids = [_text(row.get("event_id")) for row in action_values]
    _require(
        all(action_ids) and len(action_ids) == len(set(action_ids)),
        "In-scope lifecycle corporate actions require unique non-empty event_id values.",
    )
    records = lifecycle_report["records"]
    terminal_hosts = tuple(str(item) for item in policy.events["official_hosts"])
    nonterminal_hosts = tuple(
        str(item)
        for item in policy.events.get(
            "official_provenance_hosts", policy.events["official_hosts"]
        )
    )
    nonterminal_source_kinds = {
        _text(item)
        for item in policy.events.get(
            "official_provenance_source_kinds",
            ("official_crosscheck", "official_filing"),
        )
        if _text(item)
    }
    terminal_source_kinds = {
        _text(item)
        for item in policy.events.get(
            "terminal_official_source_kinds",
            ("official_crosscheck",),
        )
        if _text(item)
    }
    reviewed_extractions = reviewed_nonterminal_extractions(policy.events)
    terminal_overrides = reviewed_terminal_overrides(policy.events)
    event_gates = reviewed_terminal_event_gates(policy.events)
    market_date_corrections = reviewed_terminal_market_date_corrections(
        policy.events
    )
    policy_exceptions = reviewed_terminal_policy_exceptions(policy.events)
    tail_corrections = reviewed_terminal_price_tail_corrections(policy.events)
    applied = resolutions.loc[resolutions["resolution"].astype(str).eq("applied")]
    applied_event_ids = [
        _text(value) for value in applied.get("event_id", pd.Series(dtype="object"))
    ]
    _require(
        all(applied_event_ids)
        and len(applied_event_ids) == len(set(applied_event_ids)),
        "Applied terminal resolutions require unique non-empty event_id values.",
    )
    resolution_groups = {
        event_id: group.to_dict(orient="records")
        for event_id, group in applied.assign(
            _event_id=applied["event_id"].fillna("").astype(str).str.strip()
        ).groupby("_event_id", sort=False)
        if event_id
    }
    orphan_resolutions = sorted(set(resolution_groups) - set(action_ids))
    _require(
        not orphan_resolutions,
        "Applied lifecycle resolutions have no in-scope corporate action: "
        + ", ".join(orphan_resolutions),
    )
    checks: list[dict[str, Any]] = []
    used_report_records: set[str] = set()
    for action in sorted(action_values, key=lambda row: _text(row.get("event_id"))):
        event_id = _text(action.get("event_id"))
        linked = resolution_groups.get(event_id, [])
        terminal = bool(linked)
        validation_kind = (
            TERMINAL_EVENT_VALIDATION if terminal else NONTERMINAL_EVENT_VALIDATION
        )
        resolution = linked[0] if len(linked) == 1 else {}
        security_id = _text(action.get("security_id"))
        reasons: list[str] = []
        source_url = _text(action.get("source_url"))
        evidence_hash = _text(action.get("source_hash")).lower()
        archive_mask = (
            archive["source_hash"].map(_text).str.lower().eq(evidence_hash)
            & archive["source_url"].map(_text).eq(source_url)
        )
        archive_matches = archive.loc[archive_mask]
        raw_archive_source = (
            _text(archive_matches.iloc[0].get("source"))
            if len(archive_matches) == 1
            else ""
        )
        exact_archive_pair = bool(
            len(archive_matches) == 1
            and raw_archive_source
            and _text(archive_matches.iloc[0].get("dataset"))
            == raw_archive_source
            and source_archive_binding_matches(
                archive_matches.iloc[0].to_dict(),
                source=raw_archive_source,
                source_url=source_url,
                source_hash=evidence_hash,
            )
        )
        source_kind = _text(action.get("source_kind"))
        provenance_hosts = terminal_hosts if terminal else nonterminal_hosts
        source_kind_passed = (
            source_kind in terminal_source_kinds
            if terminal
            else source_kind in nonterminal_source_kinds
        )
        official_original = (
            _text(action.get("official")).lower() == "true"
            and source_kind_passed
            and _official_host(source_url, provenance_hosts)
            and len(evidence_hash) == 64
            and exact_archive_pair
        )
        action_type = _text(action.get("action_type")).lower()
        effective = _date(action.get("effective_date"))
        reviewed_extraction_hash = ""
        reviewed_extraction_match = False
        reviewed_terminal_override_hash = ""
        reviewed_terminal_override_match = False
        reviewed_terminal_override_applied = False
        reviewed_terminal_event_gate_hash = ""
        reviewed_terminal_event_gate_match = False
        reviewed_terminal_event_gate_applied = False
        reviewed_market_date_correction_hash = ""
        reviewed_market_date_correction_match = False
        reviewed_market_date_correction_applied = False
        reviewed_terminal_policy_exception_hash = ""
        reviewed_terminal_policy_exception_match = False
        reviewed_terminal_policy_exception_applied = False
        reviewed_terminal_policy_exception_code = ""
        reviewed_price_tail_correction_hash = ""
        reviewed_price_tail_correction_match = False
        reviewed_price_tail_correction_applied = False
        report_effective_date = ""
        official_completion_date = ""
        market_date_relation = ""
        lifecycle_report_collector_approved = False
        sivb_evidence_binding = (
            trusted_sivb_evidence_binding_diagnostic(action, archive)
        )
        sivb_evidence_binding_applied = bool(
            sivb_evidence_binding is not None
            and sivb_evidence_binding.get("status") == "trusted"
        )
        if sivb_evidence_binding is not None and not sivb_evidence_binding_applied:
            if not sivb_evidence_binding["action_binding_exact"]:
                reasons.append("trusted SIVB action binding differs")
            missing_support = sorted(
                role
                for role, passed in sivb_evidence_binding[
                    "evidence_archive_bindings"
                ].items()
                if not passed
            )
            if missing_support:
                reasons.append(
                    "trusted SIVB evidence archive binding differs: "
                    + ", ".join(missing_support)
                )
        if sivb_evidence_binding_applied:
            official_original = True
        frc_evidence_binding = trusted_frc_evidence_binding_diagnostic(
            action, archive
        )
        frc_evidence_binding_applied = bool(
            frc_evidence_binding is not None
            and frc_evidence_binding.get("status") == "trusted"
        )
        if frc_evidence_binding is not None and not frc_evidence_binding_applied:
            if not frc_evidence_binding["action_binding_exact"]:
                reasons.append("trusted FRC action binding differs")
            missing_support = sorted(
                role
                for role, passed in frc_evidence_binding[
                    "evidence_archive_bindings"
                ].items()
                if not passed
            )
            if missing_support:
                reasons.append(
                    "trusted FRC evidence archive binding differs: "
                    + ", ".join(missing_support)
                )
        if frc_evidence_binding_applied:
            official_original = True
        ntco_evidence_binding = trusted_ntco_evidence_binding_diagnostic(
            action, archive
        )
        ntco_evidence_binding_applied = bool(
            ntco_evidence_binding is not None
            and ntco_evidence_binding.get("status") == "trusted"
            and ntco_evidence_binding.get("expected_terminal") is terminal
        )
        if ntco_evidence_binding is not None and not ntco_evidence_binding_applied:
            if not ntco_evidence_binding["action_binding_exact"]:
                reasons.append("trusted NTCO action binding differs")
            if ntco_evidence_binding.get("expected_terminal") is not terminal:
                reasons.append("trusted NTCO terminal classification differs")
            missing_support = sorted(
                role
                for role, passed in ntco_evidence_binding[
                    "evidence_archive_bindings"
                ].items()
                if not passed
            )
            if missing_support:
                reasons.append(
                    "trusted NTCO evidence archive binding differs: "
                    + ", ".join(missing_support)
                )
        if ntco_evidence_binding_applied:
            official_original = True
        if terminal:
            record_key = _text(resolution.get("security_id")) or security_id
            record = records.get(record_key)
            event = _event_from_record(record or {})
            if len(linked) != 1:
                reasons.append("exactly one applied lifecycle resolution")
            elif (
                _text(resolution.get("event_id")) != event_id
                or _text(resolution.get("security_id")) != security_id
            ):
                reasons.append(
                    "resolution identity/event does not equal the corporate action"
                )
            if not isinstance(record, dict):
                reasons.append("lifecycle evidence record is missing")
            elif record_key in used_report_records:
                reasons.append("lifecycle evidence report record is not one-to-one")
            else:
                used_report_records.add(record_key)
            record_source_url = _text(
                event.get("source_url") or (record or {}).get("source_url")
            )
            record_source_hash = _text(
                event.get("source_hash") or (record or {}).get("source_hash")
            ).lower()
            market_date_correction = market_date_corrections.get(event_id)
            reviewed_source_replacement = bool(
                market_date_correction is not None
                and source_url == market_date_correction["source_url"]
                and evidence_hash == market_date_correction["source_hash"]
                and record_source_url
                == market_date_correction["report_source_url"]
                and record_source_hash
                == market_date_correction["report_source_hash"]
            )
            direct_evidence_match = bool(
                (source_url == record_source_url and evidence_hash == record_source_hash)
                or reviewed_source_replacement
            )
            parsed_type = _text(event.get("action_type")).lower()
            parsed_effective = _date(event.get("effective_date"))
            report_effective_date = parsed_effective
            direct_date_match = bool(effective and effective == parsed_effective)
            direct_terms_match = (
                action_type == parsed_type
                and _same_number(action.get("cash_amount"), event.get("cash_amount"))
                and _same_number(action.get("ratio"), event.get("ratio"))
                and _text(action.get("new_symbol")).upper()
                == _text(event.get("new_symbol")).upper()
                and _text(action.get("currency") or "USD").upper() == "USD"
                and _text(action.get("new_security_id"))
                == _text(resolution.get("successor_security_id"))
            )
            verified_override = isinstance((record or {}).get("verified_event"), dict)
            cross = (record or {}).get("crosscheck") or {}
            manual_review = bool((record or {}).get("manual_review")) or bool(
                _text((record or {}).get("manual_review_reason"))
            )
            lifecycle_report_collector_approved = (
                verified_override and not manual_review
            ) or (
                bool((record or {}).get("eligible_for_apply"))
                and all(
                    bool(cross.get(key))
                    for key in ("passed", "date_passed", "economic_terms_passed")
                )
                and not manual_review
            )
            event_gate = event_gates.get(event_id)
            if event_gate is not None:
                reviewed_terminal_event_gate_hash = (
                    reviewed_terminal_event_gate_sha256(event_gate)
                )
                event_gate_mismatches = reviewed_terminal_event_gate_mismatches(
                    action,
                    resolution,
                    record,
                    archive,
                    event_gate,
                    lifecycle_report_sha256,
                )
                if (
                    _text(event_gate.get("policy_code"))
                    == "sivbq_verified_legal_cancellation/v1"
                    and not sivb_evidence_binding_applied
                ):
                    event_gate_mismatches = (
                        *event_gate_mismatches,
                        "trusted_sivb_evidence_binding",
                    )
                reviewed_terminal_event_gate_match = not event_gate_mismatches
                if event_gate_mismatches:
                    reasons.append(
                        "reviewed terminal event gate differs: "
                        + ", ".join(event_gate_mismatches)
                    )
                reviewed_terminal_event_gate_applied = (
                    reviewed_terminal_event_gate_match
                )
                official_original = bool(
                    official_original or reviewed_terminal_event_gate_applied
                )
            policy_exception = policy_exceptions.get(event_id)
            if policy_exception is not None:
                reviewed_terminal_policy_exception_hash = (
                    reviewed_terminal_policy_exception_sha256(policy_exception)
                )
                reviewed_terminal_policy_exception_code = _text(
                    policy_exception.get("policy_code")
                )
                policy_action_mismatches = (
                    reviewed_terminal_policy_action_mismatches(
                        action, policy_exception
                    )
                )
                policy_report_mismatches = (
                    reviewed_terminal_policy_report_mismatches(
                        action,
                        resolution,
                        record,
                        policy_exception,
                        lifecycle_report_sha256,
                    )
                )
                reviewed_terminal_policy_exception_match = not (
                    policy_action_mismatches or policy_report_mismatches
                )
                if policy_action_mismatches:
                    reasons.append(
                        "reviewed terminal policy action differs: "
                        + ", ".join(policy_action_mismatches)
                    )
                if policy_report_mismatches:
                    reasons.append(
                        "reviewed terminal policy report differs: "
                        + ", ".join(policy_report_mismatches)
                    )
                reviewed_terminal_policy_exception_applied = bool(
                    reviewed_terminal_policy_exception_match
                    and exact_archive_pair
                    and _text(action.get("official")).lower() == "true"
                    and _official_host(source_url, terminal_hosts)
                    and len(evidence_hash) == 64
                )
                official_original = bool(
                    official_original
                    or reviewed_terminal_policy_exception_applied
                )
            terminal_override = terminal_overrides.get(event_id)
            if terminal_override is not None:
                reviewed_terminal_override_hash = (
                    reviewed_terminal_override_sha256(terminal_override)
                )
                terminal_override_mismatches = (
                    reviewed_terminal_override_mismatches(
                        action, terminal_override
                    )
                )
                reviewed_terminal_override_match = not terminal_override_mismatches
                if terminal_override_mismatches:
                    reasons.append(
                        "reviewed terminal override differs: "
                        + ", ".join(terminal_override_mismatches)
                    )
                terminal_report_mismatches = (
                    reviewed_terminal_report_mismatches(
                        action, resolution, record
                    )
                )
                if terminal_report_mismatches:
                    reasons.append(
                        "reviewed terminal override report differs: "
                        + ", ".join(terminal_report_mismatches)
                    )
                reviewed_terminal_override_applied = (
                    official_original
                    and reviewed_terminal_override_match
                    and not terminal_report_mismatches
                )
            if market_date_correction is not None:
                reviewed_market_date_correction_hash = (
                    reviewed_terminal_market_date_correction_sha256(
                        market_date_correction
                    )
                )
                action_mismatches = (
                    reviewed_terminal_market_date_action_mismatches(
                        action, market_date_correction
                    )
                )
                report_mismatches = (
                    reviewed_terminal_market_date_report_mismatches(
                        action,
                        resolution,
                        record,
                        market_date_correction,
                        lifecycle_report_sha256,
                    )
                )
                reviewed_market_date_correction_match = not (
                    action_mismatches or report_mismatches
                )
                if action_mismatches:
                    reasons.append(
                        "reviewed terminal market-date action differs: "
                        + ", ".join(action_mismatches)
                    )
                if report_mismatches:
                    reasons.append(
                        "reviewed terminal market-date report differs: "
                        + ", ".join(report_mismatches)
                    )
                reviewed_market_date_correction_applied = (
                    official_original and reviewed_market_date_correction_match
                )
                official_completion_date = _date(
                    market_date_correction.get("official_completion_date")
                )
                market_date_relation = _text(
                    market_date_correction.get("date_relation")
                )
            tail_correction = tail_corrections.get(event_id)
            if tail_correction is not None:
                reviewed_price_tail_correction_hash = (
                    reviewed_terminal_price_tail_correction_sha256(
                        tail_correction
                    )
                )
                tail_action_mismatches = (
                    reviewed_terminal_price_tail_action_mismatches(
                        action, tail_correction
                    )
                )
                tail_report_mismatches = (
                    reviewed_terminal_price_tail_report_mismatches(
                        action,
                        resolution,
                        record,
                        archive,
                        tail_correction,
                        lifecycle_report_sha256,
                    )
                )
                if reviewed_terminal_event_gate_applied:
                    # The immutable tail-repair registry remains byte-for-byte
                    # bound to the repair manifests.  Its pre-repair report
                    # candidate fields are superseded only by the separately
                    # code-pinned current event gate, which hashes the full
                    # current report semantics.  All action, resolution,
                    # archive, filing, terms and transition checks remain.
                    superseded_report_projection = {
                        "lifecycle_evidence_report_sha256",
                        "candidate_last_price_date",
                        "candidate_active_to",
                        "old_candidate_id",
                        "report_crosscheck_old_price_session",
                    }
                    tail_report_mismatches = tuple(
                        field
                        for field in tail_report_mismatches
                        if field not in superseded_report_projection
                    )
                reviewed_price_tail_correction_match = not (
                    tail_action_mismatches or tail_report_mismatches
                )
                if tail_action_mismatches:
                    reasons.append(
                        "reviewed terminal price-tail action differs: "
                        + ", ".join(tail_action_mismatches)
                    )
                if tail_report_mismatches:
                    reasons.append(
                        "reviewed terminal price-tail report differs: "
                        + ", ".join(tail_report_mismatches)
                    )
                reviewed_price_tail_correction_applied = (
                    official_original and reviewed_price_tail_correction_match
                )
                official_completion_date = _date(
                    tail_correction.get("official_completion_date")
                )
                market_date_relation = _text(
                    tail_correction.get("date_relation")
                )
            date_match = bool(
                direct_date_match
                or reviewed_terminal_event_gate_applied
                or reviewed_market_date_correction_applied
                or reviewed_terminal_policy_exception_applied
                or reviewed_price_tail_correction_applied
            )
            terms_match = bool(
                direct_terms_match
                or reviewed_terminal_event_gate_applied
                or reviewed_terminal_policy_exception_applied
            )
            if not official_original:
                reasons.append("exact archived URL/hash pair")
            if (
                not direct_evidence_match
                and not reviewed_terminal_event_gate_applied
                and not reviewed_terminal_policy_exception_applied
            ):
                reasons.append("action evidence does not equal parsed report evidence")
            if not terms_match:
                reasons.append("economic terms differ from official extraction")
            if not date_match:
                reasons.append("effective date differs from official extraction")
            extraction_approved = (
                lifecycle_report_collector_approved
                or reviewed_terminal_event_gate_applied
                or reviewed_terminal_override_applied
                or reviewed_market_date_correction_applied
                or reviewed_terminal_policy_exception_applied
                or reviewed_price_tail_correction_applied
            )
            if not extraction_approved:
                reasons.append("official extraction was not approved for apply")
            provenance_passed = official_original
        else:
            if not official_original:
                reasons.append("exact archived URL/hash pair")
            date_match = bool(effective)
            currency_passed = _text(action.get("currency") or "USD").upper() == "USD"
            successor_passed = True
            economic_terms_passed = True
            if action_type in {"stock_merger", "ticker_change"}:
                successor_passed = bool(
                    _text(action.get("new_security_id"))
                    and _text(action.get("new_symbol"))
                )
            if action_type == "stock_merger":
                ratio = _number(action.get("ratio"))
                economic_terms_passed = ratio is not None and ratio > 0
            elif action_type == "cash_merger":
                cash = _number(action.get("cash_amount"))
                economic_terms_passed = cash is not None and cash > 0
            elif action_type == "delisting":
                cash = _number(action.get("cash_amount"))
                economic_terms_passed = cash is not None and cash >= 0
            reviewed = reviewed_extractions.get(event_id)
            reviewed_mismatches: tuple[str, ...] = ()
            if (
                reviewed is None
                and not sivb_evidence_binding_applied
                and not frc_evidence_binding_applied
                and not ntco_evidence_binding_applied
            ):
                reasons.append("reviewed nonterminal extraction is missing")
            elif reviewed is not None:
                reviewed_extraction_hash = reviewed_nonterminal_extraction_sha256(
                    reviewed
                )
                reviewed_mismatches = reviewed_nonterminal_extraction_mismatches(
                    action, reviewed
                )
                reviewed_extraction_match = not reviewed_mismatches
                if reviewed_mismatches:
                    reasons.append(
                        "reviewed nonterminal extraction differs: "
                        + ", ".join(reviewed_mismatches)
                    )
            terms_match = (
                currency_passed
                and successor_passed
                and economic_terms_passed
                and (
                    reviewed_extraction_match
                    or sivb_evidence_binding_applied
                    or frc_evidence_binding_applied
                    or ntco_evidence_binding_applied
                )
            )
            extraction_approved = False
            provenance_passed = official_original and date_match and terms_match
            if not date_match:
                reasons.append("nonterminal action has no effective date")
            if not terms_match:
                reasons.append(
                    "nonterminal action does not match reviewed exact terms"
                )
        status = "passed" if not reasons else "mismatch"
        checks.append(
            {
                "validation_kind": validation_kind,
                "candidate_id": _text(resolution.get("candidate_id")),
                "event_id": event_id,
                "security_id": security_id,
                "symbol": _text(resolution.get("symbol")),
                "action_type": action_type,
                "effective_date": effective,
                "cash_amount": _number(action.get("cash_amount")),
                "ratio": _number(action.get("ratio")),
                "new_security_id": _text(action.get("new_security_id")),
                "new_symbol": _text(action.get("new_symbol")).upper(),
                "currency": _text(action.get("currency")).upper(),
                "status": status,
                "date_match": date_match,
                "terms_match": terms_match,
                "official_original": official_original,
                "official_provenance_passed": provenance_passed,
                "source_kind": source_kind,
                "source_url": source_url,
                "evidence_sha256": evidence_hash,
                "lifecycle_report_extraction_approved": extraction_approved,
                "lifecycle_report_collector_approved": (
                    lifecycle_report_collector_approved
                ),
                "reviewed_terminal_override_applied": (
                    reviewed_terminal_override_applied
                ),
                "reviewed_terminal_override_match": (
                    reviewed_terminal_override_match
                ),
                "reviewed_terminal_override_sha256": (
                    reviewed_terminal_override_hash
                ),
                "reviewed_terminal_event_gate_applied": (
                    reviewed_terminal_event_gate_applied
                ),
                "reviewed_terminal_event_gate_match": (
                    reviewed_terminal_event_gate_match
                ),
                "reviewed_terminal_event_gate_sha256": (
                    reviewed_terminal_event_gate_hash
                ),
                "reviewed_terminal_market_date_correction_applied": (
                    reviewed_market_date_correction_applied
                ),
                "reviewed_terminal_market_date_correction_match": (
                    reviewed_market_date_correction_match
                ),
                "reviewed_terminal_market_date_correction_sha256": (
                    reviewed_market_date_correction_hash
                ),
                "reviewed_terminal_policy_exception_applied": (
                    reviewed_terminal_policy_exception_applied
                ),
                "reviewed_terminal_policy_exception_match": (
                    reviewed_terminal_policy_exception_match
                ),
                "reviewed_terminal_policy_exception_sha256": (
                    reviewed_terminal_policy_exception_hash
                ),
                "reviewed_terminal_policy_exception_code": (
                    reviewed_terminal_policy_exception_code
                ),
                "reviewed_terminal_price_tail_correction_applied": (
                    reviewed_price_tail_correction_applied
                ),
                "reviewed_terminal_price_tail_correction_match": (
                    reviewed_price_tail_correction_match
                ),
                "reviewed_terminal_price_tail_correction_sha256": (
                    reviewed_price_tail_correction_hash
                ),
                "lifecycle_report_effective_date": report_effective_date,
                "official_completion_date": official_completion_date,
                "terminal_market_date_relation": market_date_relation,
                "reviewed_extraction_match": reviewed_extraction_match,
                "reviewed_extraction_sha256": reviewed_extraction_hash,
                "trusted_frc_evidence_binding": frc_evidence_binding,
                "trusted_ntco_evidence_binding": ntco_evidence_binding,
                "trusted_sivb_evidence_binding": sivb_evidence_binding,
                "reasons": reasons,
            }
        )
    return checks


def _symbol_on(history: pd.DataFrame, security_id: str, effective: str) -> str:
    rows = history.loc[history["security_id"].astype(str).eq(security_id)].copy()
    if rows.empty:
        return ""
    when = pd.Timestamp(effective)
    start = pd.to_datetime(rows["effective_from"], errors="coerce")
    end = pd.to_datetime(rows["effective_to"], errors="coerce")
    active = rows.loc[start.le(when) & (end.isna() | end.ge(when))]
    if active.empty:
        active = rows.assign(_start=start).sort_values("_start").tail(1)
    return _text(active.iloc[-1]["symbol"]).upper()


def build_price_targets(
    master: pd.DataFrame,
    history: pd.DataFrame,
    actions: pd.DataFrame,
    resolutions: pd.DataFrame,
    prices: pd.DataFrame | None = None,
) -> list[PriceTarget]:
    """Build the complete lifecycle and provider-affected identity target set."""

    request_release_end = ""
    price_ranges: dict[str, tuple[str, str]] = {}
    if prices is not None and not prices.empty:
        parsed_sessions = pd.to_datetime(prices["session"], errors="coerce")
        _require(
            not bool(parsed_sessions.isna().any()),
            "daily_price_raw has invalid sessions while building Yahoo requests.",
        )
        request_release_end = parsed_sessions.max().date().isoformat()
        ranged = prices.assign(_session=parsed_sessions).groupby(
            prices["security_id"].astype(str), sort=False
        )["_session"]
        price_ranges = {
            str(security_id): (
                group.min().date().isoformat(),
                group.max().date().isoformat(),
            )
            for security_id, group in ranged
        }

    master_symbols = {
        _text(row["security_id"]): _text(row["primary_symbol"]).upper()
        for row in master.to_dict(orient="records")
    }
    master_ranges = {
        _text(row["security_id"]): (
            _date(row.get("active_from")),
            _date(row.get("active_to")),
        )
        for row in master.to_dict(orient="records")
    }
    values: dict[str, dict[str, Any]] = {}

    def add(
        security_id: str,
        symbol: str,
        origin: str,
        priority: int,
        *,
        terminal_event_id: str = "",
        successor_security_id: str = "",
    ) -> None:
        security_id = _text(security_id)
        symbol = _text(symbol).upper() or master_symbols.get(security_id, "")
        if not security_id or not symbol:
            return
        current = values.setdefault(
            security_id,
            {
                "symbol": symbol,
                "priority": priority,
                "origins": set(),
                "terminal_event_id": "",
                "successor_security_id": "",
            },
        )
        if priority > current["priority"]:
            current["symbol"] = symbol
            current["priority"] = priority
        current["origins"].add(origin)
        if terminal_event_id:
            current["terminal_event_id"] = terminal_event_id
            current["successor_security_id"] = successor_security_id

    applied = resolutions.loc[resolutions["resolution"].astype(str).eq("applied")]
    for row in applied.to_dict(orient="records"):
        add(
            row.get("security_id"),
            row.get("symbol"),
            "applied_resolution_source",
            70,
            terminal_event_id=_text(row.get("event_id")),
            successor_security_id=_text(row.get("successor_security_id")),
        )
        add(
            row.get("successor_security_id"),
            row.get("successor_symbol"),
            "applied_resolution_successor",
            100,
        )
    lifecycle_actions = actions.loc[
        actions["action_type"].astype(str).str.lower().isin(LIFECYCLE_ACTION_TYPES)
    ]
    ticker_actions_by_security: dict[str, list[dict[str, Any]]] = {}
    for row in lifecycle_actions.to_dict(orient="records"):
        effective = _date(row.get("effective_date"))
        security_id = _text(row.get("security_id"))
        add(
            security_id,
            _symbol_on(history, security_id, effective),
            "lifecycle_action_source",
            50,
        )
        add(
            row.get("new_security_id"),
            row.get("new_symbol"),
            "lifecycle_action_successor",
            90,
        )
        if _text(row.get("action_type")).lower() == "ticker_change":
            ticker_actions_by_security.setdefault(security_id, []).append(row)

    def intermediate_ticker_transition(
        security_id: str,
        interval: Mapping[str, Any],
        next_interval: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """Bind one closed symbol segment to its exact next ticker action."""

        active_to = _date(interval.get("_to"))
        next_from = _date(next_interval.get("_from"))
        next_symbol = _text(next_interval.get("_symbol")).upper()
        if not active_to or not next_from or not next_symbol or active_to >= next_from:
            return None
        calendar = xcals.get_calendar("XNYS")
        sessions = calendar.sessions_in_range(
            pd.Timestamp(active_to) + pd.Timedelta(days=1),
            pd.Timestamp(next_from),
        )
        normalized = [
            pd.Timestamp(value).tz_localize(None).date().isoformat()
            for value in sessions
        ]
        # The next identity begins either on the first exchange session after
        # the old boundary or during the sessionless legal-completion gap.
        if normalized not in ([], [next_from]):
            return None
        matches = [
            row
            for row in ticker_actions_by_security.get(security_id, ())
            if _date(row.get("effective_date")) == next_from
            and _text(row.get("new_security_id")) == security_id
            and _text(row.get("new_symbol")).upper() == next_symbol
            and _text(row.get("event_id"))
        ]
        return matches[0] if len(matches) == 1 else None
    if prices is not None:
        direct_provider_ids = {
            _text(value)
            for value in prices.loc[
                independent_provider_source_mask(prices), "security_id"
            ]
            if _text(value)
        }
        for security_id in provider_affected_identity_ids(master, prices):
            add(
                security_id,
                master_symbols.get(security_id, ""),
                (
                    "independent_provider_internal_source"
                    if security_id in direct_provider_ids
                    else "independent_provider_reused_symbol_peer"
                ),
                80,
            )
    targets: list[PriceTarget] = []
    for security_id, value in sorted(values.items()):
        intervals = history.loc[
            history["security_id"].astype(str).eq(security_id)
        ].copy()
        if not intervals.empty:
            intervals["_from"] = intervals["effective_from"].map(_date)
            intervals["_to"] = intervals["effective_to"].map(_date)
            intervals["_symbol"] = (
                intervals["symbol"].fillna("").astype(str).str.strip().str.upper()
            )
            intervals = intervals.loc[
                intervals["_from"].ne("") & intervals["_symbol"].ne("")
            ].sort_values(["_from", "_to", "_symbol"], kind="stable")
        interval_rows = intervals.to_dict(orient="records")
        if not interval_rows:
            active_from, active_to = master_ranges.get(security_id, ("", ""))
            interval_rows = [
                {
                    "_symbol": value["symbol"],
                    "_from": active_from,
                    "_to": active_to,
                }
            ]
        for index, interval in enumerate(interval_rows):
            final_interval = index == len(interval_rows) - 1
            active_from = _text(interval.get("_from"))
            active_to = _text(interval.get("_to"))
            price_start, price_end = price_ranges.get(security_id, ("", ""))
            intermediate_transition = (
                intermediate_ticker_transition(
                    security_id,
                    interval,
                    interval_rows[index + 1],
                )
                if not final_interval
                else None
            )
            terminal_event_id = (
                _text(intermediate_transition.get("event_id"))
                if intermediate_transition is not None
                else value["terminal_event_id"] if final_interval else ""
            )
            successor_security_id = (
                _text(intermediate_transition.get("new_security_id"))
                if intermediate_transition is not None
                else value["successor_security_id"] if final_interval else ""
            )
            targets.append(
                PriceTarget(
                    security_id=security_id,
                    symbol=_text(interval.get("_symbol")).upper(),
                    origins=tuple(
                        sorted({*value["origins"], "symbol_history_interval"})
                    ),
                    active_from=active_from,
                    active_to=active_to,
                    request_start=active_from or price_start,
                    request_end=active_to or request_release_end or price_end,
                    terminal_event_id=terminal_event_id,
                    successor_security_id=successor_security_id,
                )
            )
    return sorted(
        targets,
        key=lambda item: (
            item.security_id,
            item.active_from,
            item.active_to,
            item.provider_symbol,
        ),
    )


def resolve_identity_boundary_evidence(
    repository: LocalDatasetRepository,
    targets: Iterable[PriceTarget],
    archive: pd.DataFrame,
    policy: Policy,
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Resolve configured boundary URLs to exact, locally archived payload hashes."""

    specs = policy.value.get("identity_boundaries") or []
    _require(isinstance(specs, list), "identity_boundaries policy must be a list.")
    output: dict[str, tuple[dict[str, Any], ...]] = {}
    for target in targets:
        accepted: list[dict[str, Any]] = []
        for spec in specs:
            if not isinstance(spec, dict) or _text(spec.get("symbol")).upper() != target.symbol:
                continue
            boundary = _text(spec.get("boundary"))
            expected_date = target.active_from if boundary == "active_from" else target.active_to if boundary == "active_to" else ""
            if not expected_date or _date(spec.get("date")) != expected_date:
                continue
            source_url = _text(spec.get("source_url"))
            rows = archive.loc[
                archive.get("source_url", pd.Series(index=archive.index, dtype="object"))
                .astype(str)
                .eq(source_url)
            ]
            hashes = sorted(
                {
                    _text(value).lower()
                    for value in rows.get("archive_id", pd.Series(dtype="object"))
                    if len(_text(value)) == 64
                }
            )
            if len(hashes) != 1:
                continue
            evidence_hash = hashes[0]
            _archive_content(repository, archive, evidence_hash)
            accepted.append(
                {
                    "boundary": boundary,
                    "date": expected_date,
                    "source_url": source_url,
                    "source_kind": _text(spec.get("source_kind")),
                    "evidence_sha256": evidence_hash,
                    "official_original": _official_host(
                        source_url, ("nasdaqtrader.com", "sec.gov", "nyse.com")
                    ),
                }
            )
        output[target.target_id] = tuple(accepted)
    return output


def _xnys_sessions(start: str, end: str) -> tuple[str, ...]:
    values = xcals.get_calendar("XNYS").sessions_in_range(start, end)
    return tuple(pd.Timestamp(value).tz_localize(None).date().isoformat() for value in values)


def _target_price_rows(prices: pd.DataFrame, target: PriceTarget) -> pd.DataFrame:
    rows = prices.loc[prices["security_id"].astype(str).eq(target.security_id)].copy()
    sessions = pd.to_datetime(rows["session"], errors="coerce")
    _require(not bool(sessions.isna().any()), f"Invalid sessions for {target.security_id}.")
    if target.active_from:
        rows = rows.loc[sessions.ge(pd.Timestamp(target.active_from))].copy()
        sessions = sessions.loc[rows.index]
    if target.active_to:
        rows = rows.loc[sessions.le(pd.Timestamp(target.active_to))].copy()
    return rows.sort_values("session", kind="stable").reset_index(drop=True)


def _parse_pinned_external_prices(payload: bytes, spec: Mapping[str, Any]) -> pd.DataFrame:
    symbol = _text(spec.get("symbol")).upper()
    _require(not payload.lstrip().startswith((b"<", b"<!")), f"Pinned {symbol} payload is HTML.")
    try:
        raw = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:
        raise RuntimeError(f"Pinned {symbol} CSV is unreadable.") from exc
    expected_columns = ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"]
    _require(list(raw.columns) == expected_columns, f"Pinned {symbol} CSV schema changed.")
    _require(len(raw) == int(spec["raw_rows"]), f"Pinned {symbol} raw row count changed.")
    sessions = pd.to_datetime(raw["Date"], format="%Y-%m-%d", errors="coerce")
    _require(not bool(sessions.isna().any()), f"Pinned {symbol} dates are invalid.")
    raw["session"] = sessions.dt.date.astype(str)
    _require(
        not bool(raw["session"].duplicated().any())
        and bool(raw["session"].is_monotonic_increasing),
        f"Pinned {symbol} sessions are not unique and sorted.",
    )
    for column in ("Open", "High", "Low", "Close", "Volume", "OpenInt"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    numeric = raw[["Open", "High", "Low", "Close", "Volume", "OpenInt"]]
    finite = numeric.apply(lambda values: values.map(math.isfinite)).all().all()
    coherent = (
        numeric[["Open", "High", "Low", "Close"]].gt(0).all(axis=1)
        & numeric["Volume"].ge(0)
        & numeric["OpenInt"].ge(0)
        & numeric["High"].ge(numeric[["Open", "Low", "Close"]].max(axis=1))
        & numeric["Low"].le(numeric[["Open", "High", "Close"]].min(axis=1))
    )
    _require(bool(finite) and bool(coherent.all()), f"Pinned {symbol} OHLCV is invalid.")
    segment = raw.loc[
        raw["session"].ge(_text(spec.get("overlap_start")))
        & raw["session"].le(_text(spec.get("overlap_end")))
    ].copy()
    expected = _xnys_sessions(
        _text(spec.get("overlap_start")), _text(spec.get("overlap_end"))
    )
    _require(
        len(expected) == int(spec["overlap_sessions"])
        and tuple(segment["session"].astype(str)) == expected,
        f"Pinned {symbol} overlap coverage changed.",
    )
    return segment.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    ).loc[:, ["session", "open", "high", "low", "close", "volume"]]


def _verify_pinned_primary_payload(
    payload: bytes,
    target: PriceTarget,
    primary: pd.DataFrame,
) -> None:
    """Prove stored Yahoo-primary rows equal their exact archived chart bytes."""

    try:
        parsed = parse_yahoo_chart_json(payload, target.provider_symbol)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"Archived Yahoo primary is invalid for {target.symbol}: {exc}"
        ) from exc
    _require(parsed.currency == "USD", f"Archived Yahoo primary is not USD for {target.symbol}.")
    bars = parsed.bars.copy()
    bars["session"] = pd.to_datetime(bars["session"], errors="coerce").dt.date.astype(str)
    bars = bars.loc[
        bars["session"].ge(target.active_from)
        & bars["session"].le(target.active_to)
    ].sort_values("session", kind="stable")
    _require(
        tuple(bars["session"].astype(str)) == tuple(primary["session"].astype(str)),
        f"Archived Yahoo primary sessions differ for {target.symbol}.",
    )
    for column in ("open", "high", "low", "close", "volume"):
        actual = pd.to_numeric(primary[column], errors="coerce").to_numpy(dtype="float64")
        expected = pd.to_numeric(bars[column], errors="coerce").to_numpy(dtype="float64")
        _require(
            bool(np.isfinite(actual).all())
            and np.array_equal(actual, expected),
            f"Stored Yahoo primary {column} differs from archived bytes for {target.symbol}.",
        )


def resolve_pinned_overlap_evidence(
    repository: LocalDatasetRepository,
    targets: Iterable[PriceTarget],
    prices: pd.DataFrame,
    archive: pd.DataFrame,
    policy: Policy,
) -> dict[str, PinnedOverlapEvidence]:
    """Load the exact archived Boris overlap for Yahoo-primary old LILA/K."""

    specs = policy.prices.get("pinned_external_overlaps") or []
    by_key = {
        (
            _text(spec.get("symbol")).upper(),
            _date(spec.get("active_from")),
            _date(spec.get("active_to")),
        ): dict(spec)
        for spec in specs
    }
    output: dict[str, PinnedOverlapEvidence] = {}
    for target in targets:
        spec = by_key.get((target.symbol, target.active_from, target.active_to))
        if spec is None:
            continue
        primary = _target_price_rows(prices, target)
        expected_primary = _xnys_sessions(target.active_from, target.active_to)
        _require(
            len(expected_primary) == int(spec["primary_sessions"])
            and tuple(primary["session"].astype(str)) == expected_primary,
            f"Pinned overlap primary coverage changed for {target.symbol}.",
        )
        primary_source = _text(spec.get("primary_source"))
        primary_url = _text(spec.get("primary_source_url"))
        primary_hashes = {
            _text(value).lower() for value in primary["source_hash"] if _text(value)
        }
        _require(
            set(primary["source"].astype(str)) == {primary_source}
            and set(primary["source_url"].astype(str)) == {primary_url}
            and len(primary_hashes) == 1,
            f"Pinned overlap primary provenance changed for {target.symbol}.",
        )
        primary_hash = next(iter(primary_hashes))
        primary_archive = archive.loc[
            archive["archive_id"].astype(str).eq(primary_hash)
            & archive["source_hash"].astype(str).eq(primary_hash)
            & archive["source_url"].astype(str).eq(primary_url)
        ]
        _require(
            len(primary_archive) == 1,
            f"Pinned overlap primary archive is missing for {target.symbol}.",
        )
        primary_payload = _archive_content(repository, archive, primary_hash)
        _verify_pinned_primary_payload(primary_payload, target, primary)

        external_hash = _text(spec.get("external_source_sha256")).lower()
        external_url = _text(spec.get("external_source_url"))
        external_archive = archive.loc[
            archive["archive_id"].astype(str).eq(external_hash)
            & archive["source_hash"].astype(str).eq(external_hash)
            & archive["source_url"].astype(str).eq(external_url)
        ]
        _require(
            len(external_archive) == 1,
            f"Pinned external overlap archive is missing for {target.symbol}.",
        )
        payload = _archive_content(repository, archive, external_hash)
        external = _parse_pinned_external_prices(payload, spec)
        output[target.target_id] = PinnedOverlapEvidence(
            spec=spec,
            primary_prices=primary,
            external_prices=external,
            primary_source_url=primary_url,
            primary_source_hash=primary_hash,
            external_source_url=external_url,
            external_source_hash=external_hash,
            retrieved_at=_text(external_archive.iloc[0].get("retrieved_at")),
        )
    return output


def compare_pinned_overlap(
    target: PriceTarget,
    evidence: PinnedOverlapEvidence,
) -> dict[str, Any]:
    """Recompute the frozen 597-of-630 scale-normalized close overlap."""

    primary = evidence.primary_prices.loc[:, ["session", "close"]].copy()
    external = evidence.external_prices.loc[:, ["session", "close"]].copy()
    primary["close"] = pd.to_numeric(primary["close"], errors="coerce")
    external["close"] = pd.to_numeric(external["close"], errors="coerce")
    joined = primary.merge(
        external,
        on="session",
        suffixes=("_primary", "_external"),
        validate="one_to_one",
    ).sort_values("session", kind="stable")
    spec = evidence.spec
    expected_overlap = _xnys_sessions(
        _text(spec.get("overlap_start")), _text(spec.get("overlap_end"))
    )
    _require(
        tuple(joined["session"].astype(str)) == expected_overlap,
        f"Pinned overlap sessions changed for {target.symbol}.",
    )
    primary_close = joined["close_primary"]
    external_close = joined["close_external"]
    ratio = primary_close / external_close
    scale = float(ratio.median())
    normalized_error = (ratio / scale - 1.0).abs()
    return_correlation = float(primary_close.pct_change().corr(external_close.pct_change()))
    p99 = float(normalized_error.quantile(0.99))
    tail = tuple(
        value
        for value in evidence.primary_prices["session"].astype(str)
        if value > _text(spec.get("overlap_end"))
    )
    metrics_passed = (
        math.isfinite(scale)
        and scale > 0
        and math.isfinite(return_correlation)
        and return_correlation >= float(spec["minimum_return_correlation"])
        and math.isfinite(p99)
        and p99 <= float(spec["maximum_p99_scaled_close_error"])
        and len(joined) == int(spec["overlap_sessions"])
        and len(evidence.primary_prices) == int(spec["primary_sessions"])
        and len(tail) == int(spec["uncrosschecked_tail_sessions"])
    )
    _require(metrics_passed, f"Pinned external overlap failed for {target.symbol}.")
    return {
        "target_id": target.target_id,
        "security_id": target.security_id,
        "symbol": target.symbol,
        "provider_symbol": target.provider_symbol,
        "origins": list(target.origins),
        "validation_basis": PINNED_EXTERNAL_OVERLAP_VALIDATION,
        "status": "passed",
        "all_overlap_sessions_compared": True,
        "overlap_session_count": len(joined),
        "internal_history_session_count": len(evidence.primary_prices),
        "internal_history_start": evidence.primary_prices["session"].iloc[0],
        "internal_history_end": evidence.primary_prices["session"].iloc[-1],
        "external_overlap_start": joined["session"].iloc[0],
        "external_overlap_end": joined["session"].iloc[-1],
        "external_overlap_ratio": len(joined) / len(evidence.primary_prices),
        "uncrosschecked_tail_sessions": len(tail),
        "uncrosschecked_tail_start": tail[0],
        "uncrosschecked_tail_end": tail[-1],
        "median_primary_to_external_close_scale": scale,
        "return_correlation": return_correlation,
        "p99_scaled_close_error": p99,
        "minimum_return_correlation": float(spec["minimum_return_correlation"]),
        "maximum_p99_scaled_close_error": float(
            spec["maximum_p99_scaled_close_error"]
        ),
        "session_coverage_passed": True,
        "scale_stability_passed": True,
        "price_tolerance_passed": True,
        "currency_passed": True,
        "identity_boundary_passed": True,
        "provider_currency": "USD",
        "provider_adjustment_basis": "scale_normalized_close_overlap",
        "adjusted_close_used": False,
        "volume_compared": False,
        "primary_source": _text(spec.get("primary_source")),
        "primary_source_url": evidence.primary_source_url,
        "primary_source_sha256": evidence.primary_source_hash,
        "external_source": _text(spec.get("external_source")),
        "source_url": evidence.external_source_url,
        "source_sha256": evidence.external_source_hash,
        "retrieved_at": evidence.retrieved_at,
        "upstream_provider_disclosed": spec.get("upstream_provider_disclosed"),
        "independent_provider_claimed": spec.get("independent_provider_claimed"),
        "license": _text(spec.get("license")),
        "license_url": _text(spec.get("license_url")),
        "independent_internal_price_rows": 0,
        "internal_price_rows": len(evidence.primary_prices),
        "self_source_rows_excluded": 0,
        "identity_active_from": target.active_from,
        "identity_active_to": target.active_to,
    }


class YahooChartCache:
    """Target-aware adapter over the immutable Yahoo chart response cache."""

    def __init__(self, root: Path, policy: Policy):
        self.backend = RawYahooChartCache(
            Path(root),
            endpoint_template=str(policy.provider["endpoint_template"]),
            max_http_attempts=int(policy.provider["max_http_attempts"]),
            timeout_seconds=float(policy.provider["timeout_seconds"]),
            max_response_bytes=int(policy.provider["max_response_bytes"]),
        )

    @property
    def http_attempts(self) -> int:
        return self.backend.http_attempts

    def get(self, target: PriceTarget) -> CachedResponse | None:
        _, _, period1, period2 = _bounded_yahoo_request(target)
        return self.backend.get(
            target.provider_symbol,
            period1=period1,
            period2=period2,
        )

    def provenance_payload(self, target: PriceTarget) -> bytes | None:
        _, _, period1, period2 = _bounded_yahoo_request(target)
        return self.backend.provenance_payload(
            target.provider_symbol,
            period1=period1,
            period2=period2,
        )

    def fill_missing(
        self, targets: Iterable[PriceTarget]
    ) -> dict[str, CachedResponse | None]:
        targets = tuple(targets)
        requests = {
            target.target_id: (
                target.provider_symbol,
                *_bounded_yahoo_request(target)[2:],
            )
            for target in targets
        }
        fetched = self.backend.fill_missing(requests.values())
        return {
            target.target_id: fetched[requests[target.target_id]]
            for target in targets
        }


def _relative_error(actual: pd.Series, expected: pd.Series) -> pd.Series:
    denominator = expected.abs().clip(lower=1e-12)
    return (actual - expected).abs() / denominator


def compare_price_history(
    target: PriceTarget,
    eodhd: pd.DataFrame,
    provider_prices: pd.DataFrame,
    split_dates: Iterable[str],
    policy: Policy,
    provider_currency: str,
    identity_boundary_evidence: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Compare every common session after proving split-regime scale stability."""

    expected_currency = str(policy.prices["currency"]).upper()
    currencies = sorted(
        set(eodhd["currency"].dropna().astype(str).str.strip().str.upper())
    )
    eod = eodhd.copy()
    eod["session"] = pd.to_datetime(eod["session"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close"):
        eod[column] = pd.to_numeric(eod[column], errors="coerce")
    eod = eod.dropna(subset=["session", "open", "high", "low", "close"])
    provider_all = provider_prices.copy()
    provider_all["session"] = pd.to_datetime(
        provider_all["session"], errors="coerce"
    ).dt.normalize()
    for column in ("open", "high", "low", "close"):
        provider_all[column] = pd.to_numeric(provider_all[column], errors="coerce")
    provider_all = provider_all.dropna(
        subset=["session", "open", "high", "low", "close"]
    )
    active_from = pd.Timestamp(target.active_from) if target.active_from else None
    active_to = pd.Timestamp(target.active_to) if target.active_to else None
    outside_before = (
        int(provider_all["session"].lt(active_from).sum())
        if active_from is not None
        else 0
    )
    outside_after = (
        int(provider_all["session"].gt(active_to).sum())
        if active_to is not None
        else 0
    )
    if active_from is not None:
        eod = eod.loc[eod["session"].ge(active_from)].copy()
        provider_segment = provider_all.loc[
            provider_all["session"].ge(active_from)
        ].copy()
    else:
        provider_segment = provider_all.copy()
    if active_to is not None:
        eod = eod.loc[eod["session"].le(active_to)].copy()
        provider_segment = provider_segment.loc[
            provider_segment["session"].le(active_to)
        ].copy()
    provider_history_start = (
        provider_all["session"].min().date().isoformat()
        if not provider_all.empty
        else ""
    )
    provider_history_end = (
        provider_all["session"].max().date().isoformat()
        if not provider_all.empty
        else ""
    )
    provider_segment_start = (
        provider_segment["session"].min().date().isoformat()
        if not provider_segment.empty
        else ""
    )
    provider_segment_end = (
        provider_segment["session"].max().date().isoformat()
        if not provider_segment.empty
        else ""
    )
    if eod.empty or provider_segment.empty:
        return {
            "target_id": target.target_id,
            "security_id": target.security_id,
            "symbol": target.symbol,
            "provider_symbol": target.provider_symbol,
            "origins": list(target.origins),
            "status": "mismatch",
            "reason": "EODHD or Yahoo has no comparable OHLC sessions",
            "all_overlap_sessions_compared": True,
            "overlap_session_count": 0,
            "session_coverage_passed": False,
            "currency_passed": False,
            "identity_boundary_passed": False,
            "scale_stability_passed": False,
            "price_tolerance_passed": False,
            "identity_active_from": target.active_from,
            "identity_active_to": target.active_to,
            "provider_sessions_before_identity": outside_before,
            "provider_sessions_after_identity": outside_after,
            "eodhd_history_start": (
                eod["session"].min().date().isoformat() if not eod.empty else ""
            ),
            "eodhd_history_end": (
                eod["session"].max().date().isoformat() if not eod.empty else ""
            ),
            "eodhd_history_session_count": len(eod),
            "provider_history_start": provider_history_start,
            "provider_history_end": provider_history_end,
            "provider_history_session_count": len(provider_all),
            "provider_identity_segment_start": provider_segment_start,
            "provider_identity_segment_end": provider_segment_end,
            "provider_identity_segment_session_count": len(provider_segment),
            "eodhd_full_history_overlap_ratio": 0.0,
            "session_coverage_ratio": 0.0,
        }
    eod_unique = not bool(eod["session"].duplicated().any())
    provider_unique = not bool(provider_segment["session"].duplicated().any())
    merged = eod.merge(
        provider_segment, on="session", suffixes=("_eodhd", "_provider")
    )
    merged = merged.sort_values("session").reset_index(drop=True)
    overlap_count = len(merged)
    expected_overlap_sessions = set(eod["session"]) & set(provider_segment["session"])
    all_overlap_sessions_compared = (
        eod_unique
        and provider_unique
        and overlap_count == len(expected_overlap_sessions)
        and not bool(merged["session"].duplicated().any())
    )
    common_start = merged["session"].min() if overlap_count else None
    common_end = merged["session"].max() if overlap_count else None
    eod_span_count = (
        len(eod.loc[eod["session"].between(common_start, common_end)])
        if overlap_count
        else 0
    )
    coverage_ratio = overlap_count / len(eod) if len(eod) else 0.0
    minimum_overlap = int(policy.prices["minimum_overlap_sessions"])
    minimum_coverage = float(policy.prices["minimum_session_coverage_ratio"])
    boundary_evidence = [dict(item) for item in identity_boundary_evidence]
    evidenced_boundaries = {
        _text(item.get("boundary"))
        for item in boundary_evidence
        if item.get("official_original") is True
        and len(_text(item.get("evidence_sha256"))) == 64
    }
    identity_boundary_passed = (
        (outside_before == 0 or "active_from" in evidenced_boundaries)
        and (outside_after == 0 or "active_to" in evidenced_boundaries)
    )

    boundaries = sorted(
        pd.Timestamp(value).normalize()
        for value in split_dates
        if overlap_count
        and _date(value)
        and common_start <= pd.Timestamp(value).normalize() <= common_end
    )
    if overlap_count:
        merged["regime"] = merged["session"].map(
            lambda value: sum(boundary <= value for boundary in boundaries)
        )
    regime_reports: list[dict[str, Any]] = []
    mismatch_counts = {"open": 0, "high": 0, "low": 0, "close": 0}
    max_relative_errors = {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0}
    scale_stability_passed = True
    configured_min_regime = int(policy.prices["minimum_split_regime_sessions"])
    min_regime = 1 if len(eod) < configured_min_regime else configured_min_regime
    scale_tolerance = float(policy.prices["scale_stability_relative_tolerance"])
    close_tolerance = float(policy.prices["close_relative_tolerance"])
    ohl_tolerance = float(policy.prices["ohl_relative_tolerance"])
    absolute_tolerance = float(policy.prices["absolute_price_tolerance_usd"])

    for regime_id, group in merged.groupby("regime", sort=True):
        ratios = group["close_provider"] / group["close_eodhd"]
        median_scale = float(ratios.median())
        scale_deviation = ((ratios / median_scale) - 1.0).abs()
        maximum_scale_deviation = float(scale_deviation.max())
        stable = (
            len(group) >= min_regime
            and math.isfinite(median_scale)
            and median_scale > 0
            and maximum_scale_deviation <= scale_tolerance
        )
        scale_stability_passed = scale_stability_passed and stable
        regime_report = {
            "regime": int(regime_id),
            "start": group["session"].min().date().isoformat(),
            "end": group["session"].max().date().isoformat(),
            "session_count": len(group),
            "median_provider_to_eodhd_close_scale": median_scale,
            "max_scale_relative_deviation": maximum_scale_deviation,
            "scale_stable": stable,
        }
        regime_reports.append(regime_report)
        if not stable:
            continue
        for column in ("open", "high", "low", "close"):
            normalized = group[f"{column}_provider"] / median_scale
            expected = group[f"{column}_eodhd"]
            relative = _relative_error(normalized, expected)
            absolute = (normalized - expected).abs()
            tolerance = close_tolerance if column == "close" else ohl_tolerance
            passed = absolute.le(np.maximum(absolute_tolerance, expected.abs() * tolerance))
            mismatch_counts[column] += int((~passed).sum())
            max_relative_errors[column] = max(
                max_relative_errors[column], float(relative.max())
            )

    price_mismatches = sum(mismatch_counts.values())
    required_overlap = min(minimum_overlap, len(eod))
    session_coverage_passed = (
        all_overlap_sessions_compared
        and overlap_count >= required_overlap
        and coverage_ratio >= minimum_coverage
    )
    normalized_provider_currency = str(provider_currency).strip().upper()
    currency_passed = (
        currencies == [expected_currency]
        and normalized_provider_currency == expected_currency
    )
    price_tolerance_passed = (
        bool(regime_reports)
        and scale_stability_passed
        and price_mismatches == 0
        and currency_passed
        and identity_boundary_passed
    )
    status = (
        "passed"
        if session_coverage_passed and price_tolerance_passed
        else "mismatch"
    )
    return {
        "target_id": target.target_id,
        "security_id": target.security_id,
        "symbol": target.symbol,
        "provider_symbol": target.provider_symbol,
        "origins": list(target.origins),
        "status": status,
        "all_overlap_sessions_compared": all_overlap_sessions_compared,
        "overlap_session_count": overlap_count,
        "eodhd_history_start": eod["session"].min().date().isoformat(),
        "eodhd_history_end": eod["session"].max().date().isoformat(),
        "eodhd_history_session_count": len(eod),
        "provider_history_start": provider_history_start,
        "provider_history_end": provider_history_end,
        "provider_history_session_count": len(provider_all),
        "provider_identity_segment_start": provider_segment_start,
        "provider_identity_segment_end": provider_segment_end,
        "provider_identity_segment_session_count": len(provider_segment),
        "common_span_start": common_start.date().isoformat() if overlap_count else "",
        "common_span_end": common_end.date().isoformat() if overlap_count else "",
        "eodhd_full_history_overlap_ratio": overlap_count / len(eod),
        "eodhd_session_count_in_common_span": eod_span_count,
        "session_coverage_ratio": coverage_ratio,
        "session_coverage_passed": session_coverage_passed,
        "identity_active_from": target.active_from,
        "identity_active_to": target.active_to,
        "provider_sessions_before_identity": outside_before,
        "provider_sessions_after_identity": outside_after,
        "identity_boundary_passed": identity_boundary_passed,
        "identity_boundary_evidence": boundary_evidence,
        "currency_expected": expected_currency,
        "eodhd_currencies": currencies,
        "provider_currency": normalized_provider_currency,
        "provider_currency_field_present": True,
        "currency_basis": str(policy.prices["currency_basis"]),
        "currency_passed": currency_passed,
        "adjustment_basis": str(policy.prices["adjustment_basis"]),
        "provider_adjustment_basis": "raw_quote_ohlcv",
        "adjusted_close_used": False,
        "split_boundaries": [item.date().isoformat() for item in boundaries],
        "regimes": regime_reports,
        "scale_stability_passed": scale_stability_passed and bool(regime_reports),
        "mismatch_counts": mismatch_counts,
        "max_relative_errors": max_relative_errors,
        "price_tolerance_passed": price_tolerance_passed,
        "close_relative_tolerance": close_tolerance,
        "ohl_relative_tolerance": ohl_tolerance,
        "absolute_price_tolerance_usd": absolute_tolerance,
        "volume_compared": False,
        "volume_policy": str(policy.prices["volume_policy"]),
    }


def _terminal_calendar_complete(
    prices: pd.DataFrame,
    security_id: str,
    count: int,
) -> tuple[bool, dict[str, Any]]:
    rows = prices.loc[prices["security_id"].astype(str).eq(security_id)].copy()
    if rows.empty:
        return False, {"expected_sessions": count, "present_sessions": 0, "missing": []}
    sessions = set(pd.to_datetime(rows["session"], errors="coerce").dropna().dt.normalize())
    terminal = max(sessions)
    calendar = xcals.get_calendar("XNYS")
    available = calendar.sessions_in_range(terminal - pd.Timedelta(days=count * 3), terminal)
    expected = tuple(pd.Timestamp(value).tz_localize(None).normalize() for value in available[-count:])
    missing = [value.date().isoformat() for value in expected if value not in sessions]
    return len(expected) == count and not missing, {
        "terminal_session": terminal.date().isoformat(),
        "expected_sessions": count,
        "present_sessions": count - len(missing),
        "missing": missing,
    }


def _terminal_identity_date_binding(
    target: PriceTarget,
    event: Mapping[str, Any],
    terminal_detail: Mapping[str, Any],
    *,
    terminal_calendar_complete: bool,
) -> tuple[bool, str, str]:
    """Bind no-data to a stored or strictly derived terminal identity boundary."""
    return _terminal_event_date_binding(
        target.active_to,
        _date(terminal_detail.get("terminal_session")),
        _date(event.get("effective_date")),
        terminal_calendar_complete=terminal_calendar_complete,
    )


def _expected_yahoo_source_url(target: PriceTarget, policy: Policy) -> str:
    _, _, period1, period2 = _bounded_yahoo_request(target)
    cache = RawYahooChartCache(
        Path("unused"),
        endpoint_template=str(policy.provider["endpoint_template"]),
        max_http_attempts=1,
    )
    return cache.url(
        target.provider_symbol,
        period1=period1,
        period2=period2,
    )


def _yahoo_xnys_inventory(
    target: PriceTarget,
    provider_prices: pd.DataFrame,
    internal_prices: pd.DataFrame,
    policy: Policy,
) -> dict[str, Any]:
    """Prove a bounded daily response is an adequately complete XNYS series."""

    request_start, request_end, period1, period2 = _bounded_yahoo_request(target)
    expected = pd.DatetimeIndex(
        xcals.get_calendar("XNYS").sessions_in_range(request_start, request_end)
    ).tz_localize(None).normalize()
    provider = pd.DatetimeIndex(
        pd.to_datetime(provider_prices["session"], errors="coerce")
    ).normalize()
    internal = pd.DatetimeIndex(
        pd.to_datetime(internal_prices["session"], errors="coerce")
    ).normalize()
    invalid_provider_dates = int(provider.isna().sum())
    invalid_internal_dates = int(internal.isna().sum())
    provider = provider.dropna()
    internal = internal.dropna()
    provider_set = set(provider)
    expected_set = set(expected)
    request_start_day = pd.Timestamp(request_start)
    request_end_day = pd.Timestamp(request_end)
    internal_set = {
        value
        for value in internal
        if request_start_day <= value <= request_end_day
    }
    unexpected = sorted(provider_set - expected_set)
    missing_xnys = sorted(expected_set - provider_set)
    outside_request = sorted(
        value
        for value in provider_set
        if value < request_start_day or value > request_end_day
    )
    covered_xnys = provider_set & expected_set
    covered_internal = provider_set & internal_set
    request_coverage = (
        len(covered_xnys) / len(expected_set) if expected_set else 0.0
    )
    internal_coverage = (
        len(covered_internal) / len(internal_set) if internal_set else 0.0
    )
    minimum_coverage = float(policy.prices["minimum_session_coverage_ratio"])
    minimum_sessions = min(
        int(policy.prices["minimum_overlap_sessions"]), len(expected_set)
    )
    passed = (
        invalid_provider_dates == 0
        and invalid_internal_dates == 0
        and len(provider) == len(provider_set)
        and not unexpected
        and not outside_request
        and len(covered_xnys) >= minimum_sessions
        and request_coverage >= minimum_coverage
        and bool(internal_set)
        and internal_coverage >= minimum_coverage
    )
    return {
        "request_start_date": request_start,
        "request_end_date": request_end,
        "request_period1": period1,
        "request_period2": period2,
        "request_period2_is_exclusive": True,
        "request_xnys_session_count": len(expected_set),
        "provider_xnys_session_count": len(covered_xnys),
        "provider_unexpected_session_count": len(unexpected),
        "provider_unexpected_sessions": [
            value.date().isoformat() for value in unexpected
        ],
        "provider_missing_xnys_session_count": len(missing_xnys),
        "provider_missing_xnys_sessions": [
            value.date().isoformat() for value in missing_xnys
        ],
        "provider_outside_request_session_count": len(outside_request),
        "provider_outside_request_sessions": [
            value.date().isoformat() for value in outside_request
        ],
        "provider_request_xnys_coverage_ratio": request_coverage,
        "provider_internal_session_coverage_ratio": internal_coverage,
        "provider_request_inventory_passed": passed,
    }


def build_price_checks(
    targets: Iterable[PriceTarget],
    responses: Mapping[str, CachedResponse | None],
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    event_checks: Iterable[Mapping[str, Any]],
    policy: Policy,
    identity_boundary_evidence: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    pinned_overlap_evidence: Mapping[str, PinnedOverlapEvidence] | None = None,
    source_archive_price_only_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    wiki14_price_only_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    permanent_exception_checks: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    event_check_list = list(event_checks)
    permanent_exception_check_list = list(permanent_exception_checks)
    event_by_id = {_text(item.get("event_id")): item for item in event_check_list}
    reviewed_extractions = reviewed_nonterminal_extractions(policy.events)
    reviewed_price_registry = reviewed_price_evidence_registry(policy.prices)
    reviewed_successor_chains = reviewed_no_data_successor_chains(policy.prices)
    configured_no_data_types = policy.prices.get("no_data_terminal_action_types")
    no_data_terminal_action_types = {
        _text(value).lower() for value in configured_no_data_types or ()
    }
    _require(
        isinstance(configured_no_data_types, list)
        and len(configured_no_data_types)
        == len(YAHOO_NO_DATA_TERMINAL_ACTION_TYPES)
        and no_data_terminal_action_types
        == set(YAHOO_NO_DATA_TERMINAL_ACTION_TYPES),
        "Yahoo no-data terminal action allowlist must be exact.",
    )
    split_by_security: dict[str, list[str]] = {}
    for row in actions.to_dict(orient="records"):
        if _text(row.get("action_type")).lower() in SPLIT_ACTION_TYPES:
            split_by_security.setdefault(_text(row.get("security_id")), []).append(
                _date(row.get("effective_date"))
            )
    output: list[dict[str, Any]] = []
    for target in targets:
        response = responses.get(target.target_id)
        request_start, request_end, period1, period2 = _bounded_yahoo_request(target)
        expected_source_url = _expected_yahoo_source_url(target, policy)
        base = {
            "target_id": target.target_id,
            "security_id": target.security_id,
            "symbol": target.symbol,
            "provider_symbol": target.provider_symbol,
            "origins": list(target.origins),
            "identity_active_from": target.active_from,
            "identity_active_to": target.active_to,
            "terminal_event_id": target.terminal_event_id,
            "successor_security_id": target.successor_security_id,
            "request_start_date": request_start,
            "request_end_date": request_end,
            "request_period1": period1,
            "request_period2": period2,
            "request_period2_is_exclusive": True,
            "expected_source_url": expected_source_url,
            "source_url": response.source_url if response else "",
            "source_sha256": response.source_hash if response else "",
            "cache_wrapper_sha256": response.wrapper_hash if response else "",
            "retrieved_at": response.retrieved_at if response else "",
            "http_status": response.http_status if response else None,
        }
        pinned = (pinned_overlap_evidence or {}).get(target.target_id)
        if pinned is not None:
            output.append({**base, **compare_pinned_overlap(target, pinned)})
            continue
        frozen_wiki = (source_archive_price_only_evidence or {}).get(target.target_id)
        if frozen_wiki is not None:
            own_prices = prices.loc[
                prices["security_id"].astype(str).eq(target.security_id)
            ]
            output.append(
                {
                    **base,
                    "status": "explicit_exception",
                    "reason": frozen_wiki["limitation"],
                    "validation_basis": REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_BASIS,
                    "reviewed_source_archive_price_only_evidence_applied": True,
                    "reviewed_source_archive_price_only_evidence": dict(frozen_wiki),
                    "reviewed_source_archive_price_only_policy_sha256": frozen_wiki[
                        "policy_spec_sha256"
                    ],
                    "reviewed_source_archive_price_only_registry_sha256": (
                        TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_SHA256
                    ),
                    "reviewed_source_archive_price_only_projection_sha256": (
                        frozen_wiki["projection_sha256"]
                    ),
                    "source_url": WIKI_DOWNLOAD_URL,
                    "expected_source_url": WIKI_DOWNLOAD_URL,
                    "source_sha256": WIKI_EXTRACT_SHA256,
                    "provenance_sha256": WIKI_PROVENANCE_SHA256,
                    "cache_wrapper_sha256": "",
                    "retrieved_at": WIKI_EXTRACT_RETRIEVED_AT,
                    "http_status": None,
                    "response_identity_match": False,
                    "provider_support": "reviewed_frozen_archive_price_only",
                    "provider_currency": "not_attested_price_only",
                    "provider_adjustment_basis": (
                        "frozen_wiki_raw_unadjusted_ohlcv_price_only"
                    ),
                    "adjusted_close_used": False,
                    "overlap_session_count": frozen_wiki[
                        "overlap_session_count"
                    ],
                    "independent_internal_price_rows": len(own_prices),
                    "self_source_rows_excluded": 0,
                    "all_overlap_sessions_compared": True,
                    "price_only_arbitration_passed": True,
                    "price_tolerance_passed": False,
                    "session_coverage_passed": False,
                    "currency_passed": False,
                    "identity_boundary_passed": True,
                    "action_factor_status": "incomplete_not_rewritten",
                    "corporate_actions_validated": False,
                    "adjustment_factors_validated": False,
                    "generic_ticker_reuse_allowed": False,
                    "yahoo_symbol_only_identity_reuse_allowed": False,
                    "exception": {
                        "code": "reviewed_frozen_wiki_price_only",
                        "price_only_arbitration_passed": True,
                        "action_factor_status": "incomplete_not_rewritten",
                        "price_only_pass_must_not_imply_action_factor_pass": True,
                        "generic_ticker_reuse_allowed": False,
                        "limitation": frozen_wiki["limitation"],
                    },
                }
            )
            continue
        frozen_wiki14 = (wiki14_price_only_evidence or {}).get(target.target_id)
        if frozen_wiki14 is not None:
            own_prices = prices.loc[
                prices["security_id"].astype(str).eq(target.security_id)
            ]
            output.append(
                {
                    **base,
                    "status": "explicit_exception",
                    "reason": frozen_wiki14["limitation"],
                    "validation_basis": REVIEWED_WIKI14_PRICE_ONLY_BASIS,
                    "reviewed_wiki14_price_only_evidence_applied": True,
                    "reviewed_wiki14_price_only_evidence": dict(frozen_wiki14),
                    "reviewed_wiki14_price_only_policy_sha256": frozen_wiki14[
                        "policy_spec_sha256"
                    ],
                    "reviewed_wiki14_price_only_registry_sha256": (
                        TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_SHA256
                    ),
                    "reviewed_wiki14_price_only_projection_sha256": (
                        frozen_wiki14["projection_sha256"]
                    ),
                    "source_url": WIKI14_DOWNLOAD_URL,
                    "expected_source_url": WIKI14_DOWNLOAD_URL,
                    "source_sha256": frozen_wiki14["extract_sha256"],
                    "provenance_sha256": WIKI14_PROVENANCE_SHA256,
                    "cache_wrapper_sha256": "",
                    "retrieved_at": WIKI14_EXTRACT_RETRIEVED_AT,
                    "http_status": None,
                    "response_identity_match": False,
                    "provider_support": "reviewed_frozen_wiki14_archive_price_only",
                    "provider_currency": "not_attested_price_only",
                    "provider_adjustment_basis": (
                        "frozen_wiki_raw_unadjusted_ohlcv_price_only"
                    ),
                    "adjusted_close_used": False,
                    "overlap_session_count": frozen_wiki14[
                        "overlap_session_count"
                    ],
                    "independent_internal_price_rows": len(own_prices),
                    "self_source_rows_excluded": 0,
                    "all_overlap_sessions_compared": True,
                    "price_only_arbitration_passed": True,
                    "price_tolerance_passed": False,
                    "session_coverage_passed": False,
                    "currency_passed": False,
                    "identity_boundary_passed": True,
                    "action_factor_status": "incomplete_not_rewritten",
                    "corporate_actions_validated": False,
                    "adjustment_factors_validated": False,
                    "generic_ticker_reuse_allowed": False,
                    "yahoo_symbol_only_identity_reuse_allowed": False,
                    "private_internal_only": True,
                    "redistribution_allowed": False,
                    "public_publication_allowed": False,
                    "exception": {
                        "code": "reviewed_frozen_wiki14_price_only",
                        "price_only_arbitration_passed": True,
                        "action_factor_status": "incomplete_not_rewritten",
                        "price_only_pass_must_not_imply_action_factor_pass": True,
                        "generic_ticker_reuse_allowed": False,
                        "private_internal_only": True,
                        "redistribution_allowed": False,
                        "public_publication_allowed": False,
                        "limitation": frozen_wiki14["limitation"],
                    },
                }
            )
            continue
        if response is None:
            output.append(
                {
                    **base,
                    "status": "unresolved",
                    "reason": "immutable Yahoo chart response cache is missing",
                    "overlap_session_count": 0,
                }
            )
            continue
        try:
            response_symbol = normalize_yahoo_symbol(response.symbol)
        except ValueError:
            response_symbol = ""
        response_identity_match = (
            response_symbol == target.provider_symbol
            and response.source_url == expected_source_url
            and response.request_period1 == period1
            and response.request_period2 == period2
        )
        content_type = response.content_type.lower().split(";", 1)[0].strip()
        if not response_identity_match or content_type != "application/json":
            output.append(
                {
                    **base,
                    "status": "mismatch",
                    "reason": "Yahoo response target symbol/URL/content type mismatch",
                    "response_identity_match": response_identity_match,
                    "overlap_session_count": 0,
                }
            )
            continue
        own_prices = prices.loc[
            prices["security_id"].astype(str).eq(target.security_id)
        ].copy()
        self_source = independent_provider_source_mask(own_prices)
        self_source_rows = int(self_source.sum())
        own_prices = own_prices.loc[~self_source].copy()

        reviewed_price_spec = reviewed_price_registry.get(target.target_id)
        if reviewed_price_spec is not None:
            try:
                _require(
                    response.http_status == 200
                    and response.source_hash
                    == reviewed_price_spec["source_sha256"]
                    and response.wrapper_hash
                    == reviewed_price_spec["cache_wrapper_sha256"],
                    "Reviewed Yahoo response/cache identity changed.",
                )
                official_event_id = reviewed_price_spec["official_event_id"]
                official_event = event_by_id.get(official_event_id, {})
                official_event_binding_passed = not official_event_id or (
                    bool(official_event)
                    and official_event.get("status") == "passed"
                    and _text(official_event.get("event_id"))
                    == official_event_id
                    and _text(official_event.get("security_id"))
                    == reviewed_price_spec["security_id"]
                    and _date(official_event.get("effective_date"))
                    == reviewed_price_spec["official_effective_date"]
                    and _text(official_event.get("evidence_sha256")).lower()
                    == reviewed_price_spec["official_evidence_sha256"]
                )
                _require(
                    official_event_binding_passed,
                    "Reviewed Yahoo official-event binding changed.",
                )
                provider_rows, projection = build_reviewed_price_projection(
                    content=response.content,
                    spec=reviewed_price_spec,
                    target={
                        "target_id": target.target_id,
                        "security_id": target.security_id,
                        "symbol": target.symbol,
                        "active_from": target.active_from,
                        "active_to": target.active_to,
                    },
                    internal_prices=own_prices,
                    split_dates=split_by_security.get(target.security_id, ()),
                    policy_prices=policy.prices,
                )
                projection_sha256 = verify_reviewed_price_projection(
                    projection, reviewed_price_spec
                )
                provider_start = (
                    provider_rows["session"].min().date().isoformat()
                    if not provider_rows.empty
                    else ""
                )
                provider_end = (
                    provider_rows["session"].max().date().isoformat()
                    if not provider_rows.empty
                    else ""
                )
                output.append(
                    {
                        **base,
                        "status": "passed",
                        "reason": "",
                        "response_identity_match": response_identity_match,
                        "validation_basis": REVIEWED_PRICE_EVIDENCE_BASIS,
                        "reviewed_price_evidence_applied": True,
                        "reviewed_price_evidence_case_code": reviewed_price_spec[
                            "case_code"
                        ],
                        "reviewed_price_evidence_sha256": (
                            reviewed_price_evidence_sha256(reviewed_price_spec)
                        ),
                        "reviewed_price_evidence_registry_sha256": (
                            TRUSTED_REVIEWED_PRICE_EVIDENCE_SHA256
                        ),
                        "reviewed_price_projection_sha256": projection_sha256,
                        "reviewed_price_limitation": reviewed_price_spec[
                            "limitation"
                        ],
                        "reviewed_price_mismatch_rows": projection[
                            "mismatch_rows"
                        ],
                        "reviewed_triple_supertrend_signal": projection["signal"],
                        "reviewed_provider_metadata": projection["metadata"],
                        "reviewed_internal_ohlcv_sha256": projection[
                            "internal_ohlcv_sha256"
                        ],
                        "reviewed_provider_ohlcv_sha256": projection[
                            "provider_ohlcv_sha256"
                        ],
                        "reviewed_overlap_ohlcv_sha256": projection[
                            "overlap_ohlcv_sha256"
                        ],
                        "reviewed_all_null_row_count": projection[
                            "all_null_row_count"
                        ],
                        "reviewed_all_null_sessions_sha256": projection[
                            "all_null_sessions_sha256"
                        ],
                        "reviewed_official_event_binding_passed": (
                            official_event_binding_passed
                        ),
                        "reviewed_official_event_id": official_event_id,
                        "reviewed_official_evidence_sha256": reviewed_price_spec[
                            "official_evidence_sha256"
                        ],
                        "reviewed_official_effective_date": reviewed_price_spec[
                            "official_effective_date"
                        ],
                        "overlap_session_count": projection["overlap_row_count"],
                        "independent_internal_price_rows": len(own_prices),
                        "self_source_rows_excluded": self_source_rows,
                        "all_overlap_sessions_compared": True,
                        "scale_stability_passed": True,
                        "price_tolerance_passed": True,
                        "session_coverage_passed": True,
                        "currency_passed": True,
                        "identity_boundary_passed": True,
                        "identity_boundary_evidence": [],
                        "provider_currency": projection["metadata"]["currency"],
                        "provider_adjustment_basis": (
                            "reviewed_exact_raw_quote_ohlcv"
                        ),
                        "adjusted_close_used": False,
                        "provider_history_session_count": len(provider_rows),
                        "provider_history_start": provider_start,
                        "provider_history_end": provider_end,
                        "provider_internal_session_coverage_ratio": float(
                            projection["coverage_ratio"]
                        ),
                        "session_coverage_ratio": float(
                            projection["coverage_ratio"]
                        ),
                    }
                )
            except (RuntimeError, ValueError) as exc:
                output.append(
                    {
                        **base,
                        "status": "mismatch",
                        "reason": "Reviewed Yahoo evidence failed: " + str(exc),
                        "response_identity_match": response_identity_match,
                        "validation_basis": REVIEWED_PRICE_EVIDENCE_BASIS,
                        "reviewed_price_evidence_applied": False,
                        "overlap_session_count": 0,
                        "independent_internal_price_rows": len(own_prices),
                        "self_source_rows_excluded": self_source_rows,
                    }
                )
            continue

        provider_data = None
        no_data_evidence = None
        parse_reason = ""
        try:
            if response.http_status == 200:
                try:
                    provider_data = parse_yahoo_chart_json(
                        response.content, target.provider_symbol
                    )
                except (RuntimeError, ValueError) as price_exc:
                    try:
                        no_data_evidence = parse_yahoo_chart_no_data_evidence(
                            response.content,
                            target.provider_symbol,
                            http_status=response.http_status,
                            request_period1=period1,
                            request_period2=period2,
                        )
                    except (RuntimeError, ValueError):
                        raise price_exc
                if provider_data is not None and provider_data.bars.empty:
                    no_data_evidence = parse_yahoo_chart_no_data_evidence(
                        response.content,
                        target.provider_symbol,
                        http_status=response.http_status,
                        request_period1=period1,
                        request_period2=period2,
                    )
            else:
                no_data_evidence = parse_yahoo_chart_no_data_evidence(
                    response.content,
                    target.provider_symbol,
                    http_status=response.http_status,
                    request_period1=period1,
                    request_period2=period2,
                )
        except (RuntimeError, ValueError) as exc:
            parse_reason = str(exc)
        if parse_reason:
            output.append(
                {
                    **base,
                    "status": "mismatch",
                    "reason": (
                        f"Yahoo chart returned HTTP {response.http_status}: {parse_reason}"
                    ),
                    "response_identity_match": response_identity_match,
                    "overlap_session_count": 0,
                    "independent_internal_price_rows": len(own_prices),
                    "self_source_rows_excluded": self_source_rows,
                }
            )
            continue
        if provider_data is not None and not provider_data.bars.empty:
            if response.http_status != 200:
                output.append(
                    {
                        **base,
                        "status": "mismatch",
                        "reason": f"Yahoo chart returned HTTP {response.http_status}",
                        "response_identity_match": response_identity_match,
                        "overlap_session_count": 0,
                        "independent_internal_price_rows": len(own_prices),
                        "self_source_rows_excluded": self_source_rows,
                    }
                )
                continue
            inventory = _yahoo_xnys_inventory(
                target,
                provider_data.bars,
                own_prices,
                policy,
            )
            if not inventory["provider_request_inventory_passed"]:
                output.append(
                    {
                        **base,
                        **inventory,
                        "status": "mismatch",
                        "reason": (
                            "Yahoo bounded daily response failed exact XNYS "
                            "session inventory/coverage"
                        ),
                        "response_identity_match": response_identity_match,
                        "overlap_session_count": 0,
                        "independent_internal_price_rows": len(own_prices),
                        "self_source_rows_excluded": self_source_rows,
                    }
                )
                continue
            compared = compare_price_history(
                target,
                own_prices,
                provider_data.bars,
                split_by_security.get(target.security_id, ()),
                policy,
                provider_data.currency,
                (identity_boundary_evidence or {}).get(target.target_id, ()),
            )
            output.append(
                {
                    **compared,
                    **base,
                    **inventory,
                    "response_identity_match": response_identity_match,
                    "independent_internal_price_rows": len(own_prices),
                    "self_source_rows_excluded": self_source_rows,
                }
            )
            continue

        event = event_by_id.get(target.terminal_event_id, {})
        terminal_prices = own_prices.copy()
        if target.active_from and "session" in terminal_prices:
            terminal_prices = terminal_prices.loc[
                pd.to_datetime(terminal_prices["session"], errors="coerce").ge(
                    pd.Timestamp(target.active_from)
                )
            ]
        if target.active_to and "session" in terminal_prices:
            terminal_prices = terminal_prices.loc[
                pd.to_datetime(terminal_prices["session"], errors="coerce").le(
                    pd.Timestamp(target.active_to)
                )
            ]
        terminal_complete, terminal_detail = _terminal_calendar_complete(
            terminal_prices,
            target.security_id,
            int(policy.prices["terminal_calendar_window_sessions"]),
        )
        target_projection = {
            "target_id": target.target_id,
            "security_id": target.security_id,
            "symbol": target.symbol,
            "provider_symbol": target.provider_symbol,
            "active_from": target.active_from,
            "active_to": target.active_to,
            "terminal_event_id": target.terminal_event_id,
            "successor_security_id": target.successor_security_id,
        }
        reviewed_nonterminal_binding = (
            reviewed_nonterminal_same_sid_no_data_binding(
                target_projection,
                event,
                reviewed_extractions,
            )
        )
        permanent_binding = permanent_exception_no_data_binding(
            target_projection,
            _date(terminal_detail.get("terminal_session")),
            permanent_exception_check_list,
        )
        unsupported_binding = unsupported_path_no_data_binding(
            target_projection,
            _date(terminal_detail.get("terminal_session")),
            event,
            terminal_prices,
            policy.prices,
            source_sha256=response.source_hash,
            cache_wrapper_sha256=response.wrapper_hash,
        )
        reviewed_no_data_binding = unsupported_binding or permanent_binding
        if reviewed_no_data_binding is not None:
            reviewed_no_data_binding.update(
                {
                    "official_event_verified": bool(event)
                    and event.get("status") == "passed",
                    "identity_event_match": bool(event)
                    and _text(event.get("event_id"))
                    == target.terminal_event_id,
                    "terminal_calendar_complete": terminal_complete,
                    "terminal_calendar": terminal_detail,
                    "successor_security_id": target.successor_security_id,
                    "successor_requirement_passed": not bool(
                        target.successor_security_id
                    ),
                    "response_identity_match": response_identity_match,
                    "no_data_evidence_validated": no_data_evidence is not None,
                }
            )
            binding_valid = bool(
                response_identity_match
                and no_data_evidence is not None
                and not target.successor_security_id
            )
            output.append(
                {
                    **base,
                    "status": (
                        "explicit_exception" if binding_valid else "mismatch"
                    ),
                    "reason": _text(
                        reviewed_no_data_binding.get("limitation")
                        or reviewed_no_data_binding.get("exception_reason")
                    ),
                    "validation_basis": reviewed_no_data_binding[
                        "validation_basis"
                    ],
                    "reviewed_permanent_exception_no_data_applied": (
                        reviewed_no_data_binding.get("code")
                        == PERMANENT_EXCEPTION_NO_DATA_CODE
                    ),
                    "reviewed_unsupported_path_no_data_applied": (
                        reviewed_no_data_binding.get("validation_basis")
                        == REVIEWED_NO_DATA_UNSUPPORTED_PATH_BASIS
                    ),
                    "provider_support": "no_data",
                    "provider_currency": "unavailable_no_price_payload",
                    "provider_adjustment_basis": "no_price_payload",
                    "adjusted_close_used": False,
                    "response_identity_match": response_identity_match,
                    "no_data_evidence_kind": (
                        no_data_evidence.kind if no_data_evidence is not None else ""
                    ),
                    "no_data_error_code": (
                        no_data_evidence.error_code
                        if no_data_evidence is not None
                        else ""
                    ),
                    "no_data_error_description": (
                        no_data_evidence.error_description
                        if no_data_evidence is not None
                        else ""
                    ),
                    "exception": reviewed_no_data_binding,
                    "overlap_session_count": 0,
                    "independent_internal_price_rows": len(own_prices),
                    "self_source_rows_excluded": self_source_rows,
                }
            )
            continue
        official_verified = bool(event) and event.get("status") == "passed"
        official_action_type = _text(event.get("action_type")).lower()
        identity_event_match = (
            official_verified
            and _text(event.get("event_id")) == target.terminal_event_id
            and _text(event.get("security_id")) == target.security_id
            and official_action_type in no_data_terminal_action_types
        )
        identity_date_match = False
        identity_date_basis = ""
        derived_identity_active_to = ""
        if identity_event_match:
            (
                identity_date_match,
                identity_date_basis,
                derived_identity_active_to,
            ) = _terminal_identity_date_binding(
                target,
                event,
                terminal_detail,
                terminal_calendar_complete=terminal_complete,
            )
        exception = {
            "code": str(policy.prices["delisted_exception_code"]),
            "official_event_verified": official_verified,
            "official_event_id": target.terminal_event_id,
            "official_action_type": official_action_type,
            "official_evidence_sha256": _text(event.get("evidence_sha256")),
            "identity_event_match": identity_event_match,
            "identity_date_match": identity_date_match,
            "identity_date_basis": identity_date_basis,
            "derived_identity_active_to": derived_identity_active_to,
            "terminal_calendar_complete": terminal_complete,
            "terminal_calendar": terminal_detail,
            "successor_security_id": target.successor_security_id,
            "successor_requirement_passed": not bool(target.successor_security_id),
            "response_identity_match": response_identity_match,
            "no_data_evidence_validated": no_data_evidence is not None,
        }
        if reviewed_nonterminal_binding is not None:
            exception["reviewed_nonterminal_same_sid_binding"] = (
                reviewed_nonterminal_binding
            )
        output.append(
            {
                **base,
                "status": "pending_exception",
                **(
                    {
                        "validation_basis": (
                            REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_BASIS
                        ),
                        "reviewed_nonterminal_same_sid_no_data_applied": True,
                    }
                    if reviewed_nonterminal_binding is not None
                    else {}
                ),
                "provider_support": "no_data",
                "provider_currency": "unavailable_no_price_payload",
                "provider_adjustment_basis": "no_price_payload",
                "adjusted_close_used": False,
                "response_identity_match": response_identity_match,
                "no_data_evidence_kind": (
                    no_data_evidence.kind if no_data_evidence is not None else ""
                ),
                "no_data_error_code": (
                    no_data_evidence.error_code if no_data_evidence is not None else ""
                ),
                "no_data_error_description": (
                    no_data_evidence.error_description
                    if no_data_evidence is not None
                    else ""
                ),
                "exception": exception,
                "overlap_session_count": 0,
                "independent_internal_price_rows": len(own_prices),
                "self_source_rows_excluded": self_source_rows,
            }
        )

    for item in output:
        if item.get("status") != "pending_exception":
            continue
        exception = item["exception"]
        event = event_by_id.get(_text(exception.get("official_event_id")), {})
        reviewed_nonterminal_binding = (
            reviewed_nonterminal_same_sid_no_data_binding(
                item,
                event,
                reviewed_extractions,
            )
        )
        nonterminal_event = (
            _text(event.get("validation_kind"))
            == NONTERMINAL_EVENT_VALIDATION
        )
        successor_binding = successor_price_check_binding(
            output,
            event,
            source_target_id=_text(item.get("target_id")),
            expected_successor_security_id=_text(
                exception.get("successor_security_id")
            ),
            reviewed_successor_chains=reviewed_successor_chains,
            event_checks=event_check_list,
        )
        successor_passed = successor_binding["passed"] is True
        exception["successor_validation"] = successor_binding
        exception["successor_requirement_passed"] = successor_passed
        if (
            exception["official_event_verified"]
            and exception["identity_event_match"]
            and exception["identity_date_match"]
            and exception["terminal_calendar_complete"]
            and successor_passed
            and exception["response_identity_match"]
            and exception["no_data_evidence_validated"]
            and (
                not nonterminal_event
                or (
                    reviewed_nonterminal_binding is not None
                    and exception.get(
                        "reviewed_nonterminal_same_sid_binding"
                    )
                    == reviewed_nonterminal_binding
                    and _text(item.get("validation_basis"))
                    == REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_BASIS
                    and item.get(
                        "reviewed_nonterminal_same_sid_no_data_applied"
                    )
                    is True
                )
            )
        ):
            item["status"] = "explicit_exception"
        else:
            item["status"] = "mismatch"
            item["reason"] = "terminal no-data exception requirements did not pass"
            if successor_binding["required"] and not successor_passed:
                item["successor_failure"] = {
                    "target_id": successor_binding["target_id"],
                    "provider_symbol": successor_binding["provider_symbol"],
                    "status": successor_binding["status"],
                    "reason": successor_binding["reason"],
                    "candidate_count": successor_binding["candidate_count"],
                }
    # Recompute diagnostics only after every no-data target has reached its
    # final status.  This cannot change pass/fail (a no-data target is never an
    # independent successor pass), but it prevents a predecessor from reporting
    # the implementation-only ``pending_exception`` state instead of the real
    # successor mismatch/exception/unresolved status and target ID.
    for item in output:
        if item.get("provider_support") != "no_data":
            continue
        exception = item["exception"]
        event = event_by_id.get(_text(exception.get("official_event_id")), {})
        successor_binding = successor_price_check_binding(
            output,
            event,
            source_target_id=_text(item.get("target_id")),
            expected_successor_security_id=_text(
                exception.get("successor_security_id")
            ),
            reviewed_successor_chains=reviewed_successor_chains,
            event_checks=event_check_list,
        )
        exception["successor_validation"] = successor_binding
        exception["successor_requirement_passed"] = (
            successor_binding["passed"] is True
        )
        if successor_binding["required"] and not successor_binding["passed"]:
            item["successor_failure"] = {
                "target_id": successor_binding["target_id"],
                "provider_symbol": successor_binding["provider_symbol"],
                "status": successor_binding["status"],
                "reason": successor_binding["reason"],
                "candidate_count": successor_binding["candidate_count"],
            }
        else:
            item.pop("successor_failure", None)
    return output


def _summary(
    events: list[dict[str, Any]],
    prices: list[dict[str, Any]],
    permanent_exceptions: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> dict[str, int]:
    return {
        "event_count": len(events),
        "event_mismatch_count": sum(item["status"] != "passed" for item in events),
        "nonterminal_event_count": sum(
            item.get("validation_kind") == NONTERMINAL_EVENT_VALIDATION
            for item in events
        ),
        "reviewed_nonterminal_event_count": sum(
            item.get("validation_kind") == NONTERMINAL_EVENT_VALIDATION
            and (
                (
                    item.get("reviewed_extraction_match") is True
                    and len(_text(item.get("reviewed_extraction_sha256"))) == 64
                )
                or trusted_sivb_report_diagnostic_passed(item)
                or trusted_frc_report_diagnostic_passed(item)
                or trusted_ntco_report_diagnostic_passed(item)
            )
            for item in events
        ),
        "permanent_exception_count": len(permanent_exceptions),
        "permanent_exception_mismatch_count": sum(
            item.get("status") != "passed" for item in permanent_exceptions
        ),
        "price_target_count": len(prices),
        "price_pass_count": sum(item["status"] == "passed" for item in prices),
        "price_exception_count": sum(
            item["status"] == "explicit_exception" for item in prices
        ),
        "price_unresolved_count": sum(item["status"] == "unresolved" for item in prices),
        "price_mismatch_count": sum(item["status"] == "mismatch" for item in prices),
        "overlap_session_count": sum(
            int(item.get("overlap_session_count", 0)) for item in prices
        ),
    }


def _archive_object_path(completed_session: str, artifact: ArchiveArtifact) -> str:
    content_type = artifact.content_type.lower()
    extension = "csv" if "csv" in content_type else "json" if "json" in content_type else "bin"
    return f"archives/{completed_session}/{artifact.source_hash}.{extension}.gz"


def _archive_rows(artifacts: Iterable[ArchiveArtifact], completed_session: str) -> pd.DataFrame:
    columns = tuple(dict.fromkeys((*dataset_spec("source_archive").required_columns, "source_url")))
    return pd.DataFrame(
        [
            {
                "archive_id": item.source_hash,
                "dataset": item.source,
                "object_path": item.object_path,
                "content_type": item.content_type,
                "effective_date": completed_session,
                "source": item.source,
                "source_url": item.source_url,
                "retrieved_at": item.retrieved_at,
                "source_hash": item.source_hash,
            }
            for item in artifacts
        ],
        columns=columns,
    )


def prepare_cross_validation(
    repository: LocalDatasetRepository,
    policy: Policy,
    cache: YahooChartCache,
    *,
    fetch_missing: bool = False,
) -> PreparedCrossValidation:
    release, release_etag = repository.current_release()
    _require(release is not None, "Current release is required.")
    missing = [name for name in (*VALIDATED_DATASETS, "source_archive") if not release.dataset_versions.get(name)]
    _require(not missing, "Release lacks cross-validation inputs: " + ", ".join(missing))
    frames = {
        name: repository.read_frame(name, release.dataset_versions[name])
        for name in (*VALIDATED_DATASETS, "source_archive")
    }
    require_no_temporary_lifecycle_exceptions(frames["lifecycle_resolutions"])
    permanent_exception_checks = build_permanent_exception_checks(
        repository,
        frames["lifecycle_resolutions"],
        frames["source_archive"],
    )
    _require(
        all(item["status"] == "passed" for item in permanent_exception_checks),
        "Permanent lifecycle exception official provenance is incomplete.",
    )
    lifecycle_report, lifecycle_report_hash = _lifecycle_evidence_report(
        repository, release, frames["source_archive"]
    )
    events = build_event_checks(
        frames["corporate_actions"],
        frames["lifecycle_resolutions"],
        lifecycle_report,
        frames["source_archive"],
        policy,
        lifecycle_report_sha256=lifecycle_report_hash,
    )
    applied_event_gates = {
        _text(item.get("event_id"))
        for item in events
        if item.get("reviewed_terminal_event_gate_applied") is True
    }
    _require(
        applied_event_gates
        == (
            set(reviewed_terminal_event_gates(policy.events))
            & {_text(item.get("event_id")) for item in events}
        )
        and applied_event_gates
        == set(TRUSTED_REVIEWED_TERMINAL_EVENT_GATE_EVENT_IDS),
        "Cross-validation did not bind all "
        + str(len(TRUSTED_REVIEWED_TERMINAL_EVENT_GATE_EVENT_IDS))
        + " reviewed terminal event gates exactly once.",
    )
    applied_market_date_corrections = {
        _text(item.get("event_id"))
        for item in events
        if item.get("reviewed_terminal_market_date_correction_applied") is True
    }
    _require(
        applied_market_date_corrections
        == (
            set(reviewed_terminal_market_date_corrections(policy.events))
            & {_text(item.get("event_id")) for item in events}
        ),
        "Cross-validation did not bind every reviewed terminal market-date "
        "correction present in the input exactly once.",
    )
    applied_policy_exceptions = {
        _text(item.get("event_id"))
        for item in events
        if item.get("reviewed_terminal_policy_exception_applied") is True
    }
    _require(
        applied_policy_exceptions
        == (
            set(reviewed_terminal_policy_exceptions(policy.events))
            & {_text(item.get("event_id")) for item in events}
        ),
        "Cross-validation did not bind every reviewed terminal policy "
        "exception present in the input exactly once.",
    )
    applied_tail_corrections = {
        _text(item.get("event_id"))
        for item in events
        if item.get("reviewed_terminal_price_tail_correction_applied") is True
    }
    _require(
        applied_tail_corrections
        == (
            set(reviewed_terminal_price_tail_corrections(policy.events))
            & {_text(item.get("event_id")) for item in events}
        ),
        "Cross-validation did not bind every reviewed terminal price-tail "
        "correction present in the input exactly once.",
    )
    targets = build_price_targets(
        frames["security_master"],
        frames["symbol_history"],
        frames["corporate_actions"],
        frames["lifecycle_resolutions"],
        frames["daily_price_raw"],
    )
    source_archive_price_only_targets = {
        target.target_id: {
            "target_id": target.target_id,
            "security_id": target.security_id,
            "symbol": target.symbol,
            "provider_symbol": target.provider_symbol,
            "active_from": target.active_from,
            "active_to": target.active_to,
            "terminal_event_id": target.terminal_event_id,
        }
        for target in targets
        if target.target_id
        in TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_TARGET_IDS
    }
    source_archive_price_only = (
        verify_source_archive_price_only_evidence(
            repository,
            frames["source_archive"],
            prices=frames["daily_price_raw"],
            factors=frames["adjustment_factors"],
            master=frames["security_master"],
            history=frames["symbol_history"],
            actions=frames["corporate_actions"],
            targets=source_archive_price_only_targets,
            prices_policy=policy.prices,
            release_warnings=getattr(release, "warnings", ()),
        )
        if source_archive_price_only_targets
        else {}
    )
    wiki14_price_only_targets = {
        target.target_id: {
            "target_id": target.target_id,
            "security_id": target.security_id,
            "symbol": target.symbol,
            "provider_symbol": target.provider_symbol,
            "active_from": target.active_from,
            "active_to": target.active_to,
            "terminal_event_id": target.terminal_event_id,
        }
        for target in targets
        if target.target_id in TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_TARGET_IDS
    }
    wiki14_price_only = (
        verify_wiki14_price_only_evidence(
            repository,
            frames["source_archive"],
            prices=frames["daily_price_raw"],
            factors=frames["adjustment_factors"],
            master=frames["security_master"],
            history=frames["symbol_history"],
            actions=frames["corporate_actions"],
            targets=wiki14_price_only_targets,
            prices_policy=policy.prices,
            release_warnings=getattr(release, "warnings", ()),
        )
        if wiki14_price_only_targets
        else {}
    )
    pinned_overlaps = resolve_pinned_overlap_evidence(
        repository,
        targets,
        frames["daily_price_raw"],
        frames["source_archive"],
        policy,
    )
    yahoo_targets = [
        target
        for target in targets
        if target.target_id not in pinned_overlaps
        and target.target_id not in source_archive_price_only
        and target.target_id not in wiki14_price_only
    ]
    if fetch_missing:
        responses = cache.fill_missing(yahoo_targets)
    else:
        responses = {
            target.target_id: cache.get(target) for target in yahoo_targets
        }
    price_checks = build_price_checks(
        targets,
        responses,
        frames["daily_price_raw"],
        frames["corporate_actions"],
        events,
        policy,
        resolve_identity_boundary_evidence(
            repository,
            targets,
            frames["source_archive"],
            policy,
        ),
        pinned_overlaps,
        source_archive_price_only,
        wiki14_price_only,
        permanent_exception_checks,
    )
    applied_reviewed_prices = {
        _text(item.get("target_id"))
        for item in price_checks
        if item.get("reviewed_price_evidence_applied") is True
    }
    _require(
        applied_reviewed_prices
        == set(TRUSTED_REVIEWED_PRICE_EVIDENCE_TARGET_IDS),
        "Cross-validation did not bind every code-pinned reviewed price "
        "target exactly once.",
    )
    applied_source_archive_price_only = {
        _text(item.get("target_id"))
        for item in price_checks
        if item.get("reviewed_source_archive_price_only_evidence_applied") is True
    }
    _require(
        applied_source_archive_price_only
        == set(TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_TARGET_IDS),
        "Cross-validation did not bind the exact BBBY/BBT frozen WIKI "
        "price-only pair.",
    )
    applied_wiki14_price_only = {
        _text(item.get("target_id"))
        for item in price_checks
        if item.get("reviewed_wiki14_price_only_evidence_applied") is True
    }
    _require(
        applied_wiki14_price_only
        == set(TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_TARGET_IDS),
        "Cross-validation did not bind the exact 14-target frozen WIKI "
        "price-only inventory.",
    )
    price_checks = apply_reviewed_remaining_price_exceptions(price_checks)
    summary = _summary(events, price_checks, permanent_exception_checks)
    passed = (
        summary["event_count"]
        == int(
            frames["corporate_actions"]["action_type"]
            .astype(str)
            .str.lower()
            .isin(LIFECYCLE_ACTION_TYPES)
            .sum()
        )
        and summary["event_count"] > 0
        and summary["event_mismatch_count"] == 0
        and summary["permanent_exception_mismatch_count"] == 0
        and summary["price_target_count"] > 0
        and summary["price_unresolved_count"] == 0
        and summary["price_mismatch_count"] == 0
    )
    timestamps = [
        response.retrieved_at for response in responses.values() if response is not None
    ]
    timestamps.extend(
        evidence.retrieved_at
        for evidence in pinned_overlaps.values()
        if evidence.retrieved_at
    )
    validated_at = max(timestamps, default=release.created_at)
    validated_versions = {
        name: release.dataset_versions[name] for name in VALIDATED_DATASETS
    }
    lifecycle_manifest = repository.manifest_for_version(
        "lifecycle_resolutions", validated_versions["lifecycle_resolutions"]
    )
    candidate_set_sha256 = _text(
        lifecycle_manifest.metadata.get("candidate_set_sha256")
    )
    _require(
        len(candidate_set_sha256) == 64,
        "Lifecycle manifest candidate_set_sha256 is required for cross-validation.",
    )
    input_hashes = {
        "candidate_set_sha256": candidate_set_sha256,
        "lifecycle_resolutions_sha256": dataframe_sha256(
            frames["lifecycle_resolutions"],
            dataset_spec("lifecycle_resolutions").primary_key,
        ),
        "lifecycle_evidence_report_sha256": lifecycle_report_hash,
    }
    report = {
        "schema": CROSS_VALIDATION_SCHEMA,
        "status": "passed" if passed else "incomplete",
        "base_release_version": release.version,
        "validated_at": validated_at,
        "validated_versions": validated_versions,
        "input_hashes": input_hashes,
        "provider": {
            "name": "yahoo_chart",
            "access_class": policy.provider["access_class"],
            "stability_note": policy.provider["stability_note"],
            "http_attempts_this_run": cache.http_attempts,
            "request_cap": int(policy.provider["max_http_attempts"]),
            "attempts_per_target_cap": int(
                policy.provider["max_attempts_per_target"]
            ),
            "retry_count": int(policy.provider["retry_count"]),
            "raw_response_cache_required": True,
            "exact_response_bytes_archived": True,
            "request_mode": "bounded_period1_period2_daily",
            "range_max_allowed": False,
            "period2_semantics": "exclusive_next_utc_midnight",
            "data_granularity_required": "1d",
            "xnys_inventory_recomputed": True,
            "adjustment_basis": "raw_quote_ohlcv",
            "personal_use_only": True,
            "private_repository_required": True,
            "private_r2_required": True,
            "redistribution_allowed": False,
            "use_restriction": policy.provider["use_restriction"],
            "pinned_external_overlap_targets": len(pinned_overlaps),
            "reviewed_exact_price_evidence_targets": sum(
                item.get("validation_basis") == REVIEWED_PRICE_EVIDENCE_BASIS
                and item.get("status") == "passed"
                for item in price_checks
            ),
            "reviewed_exact_price_evidence_registry_sha256": (
                TRUSTED_REVIEWED_PRICE_EVIDENCE_SHA256
            ),
            "reviewed_source_archive_price_only_targets": sum(
                item.get("reviewed_source_archive_price_only_evidence_applied")
                is True
                for item in price_checks
            ),
            "reviewed_source_archive_price_only_registry_sha256": (
                TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_SHA256
            ),
            "reviewed_wiki14_price_only_targets": sum(
                item.get("reviewed_wiki14_price_only_evidence_applied") is True
                for item in price_checks
            ),
            "reviewed_wiki14_price_only_registry_sha256": (
                TRUSTED_REVIEWED_WIKI14_PRICE_ONLY_SHA256
            ),
            "reviewed_remaining_price_exception_targets": sum(
                item.get("reviewed_remaining_price_exception_applied") is True
                for item in price_checks
            ),
            "reviewed_remaining_price_exception_inventory_sha256": (
                TRUSTED_REVIEWED_REMAINING_PRICE_EXCEPTION_INVENTORY_SHA256
            ),
        },
        "policy": policy.value,
        "lifecycle_evidence_report_sha256": lifecycle_report_hash,
        "events": events,
        "permanent_exceptions": permanent_exception_checks,
        "prices": price_checks,
        "summary": summary,
    }
    report_bytes = canonical_json_bytes(report)
    report_hash = sha256_bytes(report_bytes)

    artifacts_by_hash: dict[str, ArchiveArtifact] = {}
    targets_by_id = {target.target_id: target for target in yahoo_targets}
    for target_id, response in responses.items():
        if response is None:
            continue
        artifact = ArchiveArtifact(
            source=YAHOO_SOURCE,
            source_url=response.source_url,
            retrieved_at=response.retrieved_at,
            content=response.content,
            content_type=response.content_type,
            object_path="",
        )
        artifact = ArchiveArtifact(
            **{
                **artifact.__dict__,
                "object_path": _archive_object_path(release.completed_session, artifact),
            }
        )
        artifacts_by_hash[artifact.source_hash] = artifact
        provenance_payload = cache.provenance_payload(targets_by_id[target_id])
        _require(
            provenance_payload is not None
            and sha256_bytes(provenance_payload) == response.wrapper_hash,
            "Yahoo cache request provenance is not reproducible.",
        )
        provenance_artifact = ArchiveArtifact(
            source=YAHOO_ENVELOPE_SOURCE,
            source_url=response.source_url,
            retrieved_at=response.retrieved_at,
            content=provenance_payload,
            content_type="application/json",
            object_path="",
        )
        provenance_artifact = ArchiveArtifact(
            **{
                **provenance_artifact.__dict__,
                "object_path": _archive_object_path(
                    release.completed_session, provenance_artifact
                ),
            }
        )
        artifacts_by_hash[provenance_artifact.source_hash] = provenance_artifact
    report_artifact = ArchiveArtifact(
        source=REPORT_SOURCE,
        source_url=f"archive://cross_validation/{report_hash}",
        retrieved_at=validated_at,
        content=report_bytes,
        content_type="application/json",
        object_path="",
    )
    report_artifact = ArchiveArtifact(
        **{
            **report_artifact.__dict__,
            "object_path": _archive_object_path(release.completed_session, report_artifact),
        }
    )
    artifacts_by_hash[report_hash] = report_artifact
    artifacts = tuple(artifacts_by_hash[key] for key in sorted(artifacts_by_hash))
    archive_delta = _archive_rows(artifacts, release.completed_session)
    new_archive = pd.concat(
        [frames["source_archive"], archive_delta], ignore_index=True, sort=False
    ).drop_duplicates("archive_id", keep="last")
    row = {
        "report_id": report_hash,
        "base_release_version": release.version,
        "validated_at": validated_at,
        "status": report["status"],
        "provider": "yahoo_chart",
        "policy_sha256": policy.sha256,
        "lifecycle_evidence_report_sha256": lifecycle_report_hash,
        "validated_versions_json": json.dumps(
            validated_versions, sort_keys=True, separators=(",", ":")
        ),
        **summary,
        "report_archive_id": report_hash,
        "source": REPORT_SOURCE,
        "retrieved_at": validated_at,
        "source_hash": report_hash,
    }
    report_frame = pd.DataFrame(
        [row], columns=dataset_spec(CROSS_VALIDATION_DATASET).required_columns
    )
    validate_dataset(CROSS_VALIDATION_DATASET, report_frame).raise_for_errors()
    validate_dataset("source_archive", new_archive).raise_for_errors()

    token = uuid.uuid4().hex
    planned_versions = {
        "source_archive": f"cross-validation-{release.completed_session.replace('-', '')}-{token}-source_archive",
        CROSS_VALIDATION_DATASET: f"cross-validation-{release.completed_session.replace('-', '')}-{token}-reports",
    }
    pointer_etags = {
        name: repository.current_pointer(name)[1]
        for name in ("source_archive", CROSS_VALIDATION_DATASET)
    }
    result_summary = {
        "status": "validated_plan" if passed else "incomplete_plan",
        "mode": "plan",
        "network_accessed": cache.http_attempts > 0,
        "yahoo_chart_http_attempts": cache.http_attempts,
        "writes_performed": False,
        "base_release_version": release.version,
        "report_sha256": report_hash,
        "policy_sha256": policy.sha256,
        "lifecycle_evidence_report_sha256": lifecycle_report_hash,
        "validated_versions": validated_versions,
        "summary": summary,
        "artifact_count": len(artifacts),
        "target_symbols": [target.provider_symbol for target in targets],
        "target_requests": [
            {
                "target_id": target.target_id,
                "provider_symbol": target.provider_symbol,
                "request_start_date": _bounded_yahoo_request(target)[0],
                "request_end_date": _bounded_yahoo_request(target)[1],
                "request_period1": _bounded_yahoo_request(target)[2],
                "request_period2": _bounded_yahoo_request(target)[3],
            }
            for target in yahoo_targets
        ],
    }
    return PreparedCrossValidation(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions=planned_versions,
        report=report,
        report_bytes=report_bytes,
        report_hash=report_hash,
        frames={
            "source_archive": new_archive.reset_index(drop=True),
            CROSS_VALIDATION_DATASET: report_frame,
        },
        artifacts=artifacts,
        summary=result_summary,
    )


def _assert_release_unchanged(
    repository: LocalDatasetRepository,
    release: DataRelease,
    release_etag: str | None,
) -> None:
    current, etag = repository.current_release()
    _require(
        current is not None and current.to_bytes() == release.to_bytes() and etag == release_etag,
        "Current release changed after cross-validation began.",
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
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _persist_artifacts(
    repository: LocalDatasetRepository,
    artifacts: Iterable[ArchiveArtifact],
) -> None:
    for artifact in artifacts:
        destination = _safe_archive_path(repository.root, artifact.object_path)
        if destination.is_file():
            existing = gzip.decompress(destination.read_bytes())
            _require(existing == artifact.content, f"Archive payload conflict: {destination}")
            continue
        write_atomic(destination, gzip.compress(artifact.content, mtime=0))
        _require(
            gzip.decompress(destination.read_bytes()) == artifact.content,
            f"Archive payload verification failed: {destination}",
        )


def _delete_pointer(repository: LocalDatasetRepository, dataset: str, version: str) -> None:
    key = repository.current_key(dataset)
    current = repository.objects.get(key)
    _require(CurrentPointer.from_bytes(current.data).version == version, "Unexpected rollback pointer.")
    path = _safe_archive_path(repository.root, key)
    path.unlink()


def _rollback(
    repository: LocalDatasetRepository,
    old_release: bytes,
    old_pointers: Mapping[str, bytes | None],
    planned_versions: Mapping[str, str],
    committed_release_version: str,
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        current = repository.objects.get("releases/current.json")
        if current.data != old_release:
            observed = DataRelease.from_bytes(current.data)
            _require(
                observed.version == committed_release_version
                or all(
                    observed.dataset_versions.get(name) == version
                    for name, version in planned_versions.items()
                ),
                "Unexpected release during rollback.",
            )
            repository.objects.put("releases/current.json", old_release, if_match=current.etag)
    except Exception as exc:
        errors.append(f"release: {type(exc).__name__}: {exc}")
    for dataset in reversed(("source_archive", CROSS_VALIDATION_DATASET)):
        try:
            previous = old_pointers[dataset]
            try:
                current = repository.objects.get(repository.current_key(dataset))
            except ObjectNotFound:
                current = None
            if previous is None:
                if current is not None:
                    _delete_pointer(repository, dataset, planned_versions[dataset])
            elif current is None:
                raise RuntimeError("Pointer disappeared during rollback.")
            elif current.data != previous:
                _require(
                    CurrentPointer.from_bytes(current.data).version == planned_versions[dataset],
                    "Unexpected pointer version during rollback.",
                )
                repository.objects.put(
                    repository.current_key(dataset), previous, if_match=current.etag
                )
        except Exception as exc:
            errors.append(f"{dataset}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def apply_cross_validation(
    repository: LocalDatasetRepository,
    prepared: PreparedCrossValidation,
    *,
    failure_injector: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    _require(prepared.report.get("status") == "passed", "Incomplete cross-validation cannot apply.")
    inject = failure_injector or (lambda _stage: None)
    with _exclusive_repository_lock(repository):
        _assert_release_unchanged(repository, prepared.release, prepared.release_etag)
        old_release = repository.objects.get("releases/current.json").data
        old_pointers: dict[str, bytes | None] = {}
        for dataset in ("source_archive", CROSS_VALIDATION_DATASET):
            pointer, etag = repository.current_pointer(dataset)
            _require(etag == prepared.pointer_etags[dataset], f"{dataset} pointer changed.")
            if pointer is None:
                old_pointers[dataset] = None
            else:
                _require(
                    pointer.version == prepared.release.dataset_versions.get(dataset),
                    f"{dataset} pointer is outside frozen release.",
                )
                old_pointers[dataset] = repository.objects.get(
                    repository.current_key(dataset)
                ).data
        committed: DataRelease | None = None
        try:
            _persist_artifacts(repository, prepared.artifacts)
            inject("after_artifacts")
            versions = dict(prepared.release.dataset_versions)
            metadata = {
                "operation": "validate_us_lifecycle_cross_sources",
                "report_id": prepared.report_hash,
                "status": "passed",
                "provider": "yahoo_chart",
                "policy_sha256": canonical_json_sha256(prepared.report["policy"]),
                "lifecycle_evidence_report_sha256": prepared.report[
                    "lifecycle_evidence_report_sha256"
                ],
                "validated_versions": prepared.report["validated_versions"],
                "input_hashes": prepared.report["input_hashes"],
                **prepared.report["summary"],
            }
            for dataset in ("source_archive", CROSS_VALIDATION_DATASET):
                result = repository.write_frame(
                    dataset,
                    prepared.frames[dataset],
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy="block",
                    metadata=metadata,
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=prepared.planned_versions[dataset],
                )
                _require(not result.conflict, f"{dataset} write conflicted.")
                versions[dataset] = result.manifest.version
                inject(f"after_{dataset}")
            inherited_warnings = tuple(
                warning
                for warning in prepared.release.warnings
                if "cross-validation" not in warning.lower()
            )
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=(DataQuality.DEGRADED if inherited_warnings else DataQuality.VALID),
                warnings=inherited_warnings,
                expected_etag=prepared.release_etag,
            )
            inject("after_release")
            return {
                **prepared.summary,
                "status": "applied",
                "mode": "apply",
                "writes_performed": True,
                "new_release_version": committed.version,
                "new_dataset_versions": versions,
            }
        except BaseException as original:
            errors = _rollback(
                repository,
                old_release,
                old_pointers,
                prepared.planned_versions,
                committed.version if committed else "",
            )
            if errors:
                marker = repository.root / "recovery/cross-validation" / f"{uuid.uuid4().hex}.json"
                write_atomic(
                    marker,
                    canonical_json_bytes(
                        {
                            "error": f"{type(original).__name__}: {original}",
                            "rollback_errors": list(errors),
                        }
                    ),
                )
                raise RuntimeError(
                    "Cross-validation apply failed and rollback was incomplete: "
                    + "; ".join(errors)
                ) from original
            raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG_PATH))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--fetch-missing",
        action="store_true",
        help="Fill only missing immutable Yahoo chart caches under the hard call cap.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Cache-only atomic release apply; cannot be combined with --fetch-missing.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    _require(
        not (args.fetch_missing and args.apply),
        "Use separate fetch/review and offline apply runs.",
    )
    config = load_data_store_config(args.data_config)
    repository = LocalDatasetRepository(config.local_cache_dir)
    policy = load_policy(Path(args.policy))
    cache = YahooChartCache(Path(args.cache_dir), policy)
    prepared = prepare_cross_validation(
        repository,
        policy,
        cache,
        fetch_missing=bool(args.fetch_missing),
    )
    output = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_ROOT / f"{prepared.release.version}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(output, prepared.report_bytes)
    summary = (
        apply_cross_validation(repository, prepared) if args.apply else prepared.summary
    )
    summary["report_path"] = str(output)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
