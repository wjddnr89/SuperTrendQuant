from __future__ import annotations

import copy
import base64
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import yaml

import supertrend_quant.market_store.cross_validation as cross_validation
from supertrend_quant.market_store.cross_validation import (
    CROSS_VALIDATION_SCHEMA,
    VALIDATED_DATASETS,
    canonical_json_bytes,
    canonical_json_sha256,
    dataframe_sha256,
    validate_cross_validation_gate,
)
from supertrend_quant.market_store.manifest import DataRelease, sha256_bytes


POLICY_PATH = Path(__file__).parents[1] / "configs/us_cross_validation.yaml"


def _policy() -> dict:
    return yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))


def _request_periods(start: str, end: str) -> tuple[int, int]:
    start_day = pd.Timestamp(start, tz="UTC")
    end_day = pd.Timestamp(end, tz="UTC")
    return (
        int(start_day.timestamp()),
        int((end_day + pd.Timedelta(days=1)).timestamp()),
    )


def _source_url(symbol: str, start: str, end: str) -> str:
    period1, period2 = _request_periods(start, end)
    return (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}"
        "&events=history&includeAdjustedClose=true&interval=1d"
    )


def _request_report_fields(symbol: str, start: str, end: str, count: int) -> dict:
    period1, period2 = _request_periods(start, end)
    source_url = _source_url(symbol, start, end)
    return {
        "request_start_date": start,
        "request_end_date": end,
        "request_period1": period1,
        "request_period2": period2,
        "request_period2_is_exclusive": True,
        "expected_source_url": source_url,
        "request_xnys_session_count": count,
        "provider_xnys_session_count": count,
        "provider_unexpected_session_count": 0,
        "provider_unexpected_sessions": [],
        "provider_missing_xnys_session_count": 0,
        "provider_missing_xnys_sessions": [],
        "provider_outside_request_session_count": 0,
        "provider_outside_request_sessions": [],
        "provider_request_xnys_coverage_ratio": 1.0,
        "provider_internal_session_coverage_ratio": 1.0,
        "provider_request_inventory_passed": True,
    }


def _target_id(
    security_id: str,
    symbol: str,
    active_from: str,
    active_to: str,
) -> str:
    return canonical_json_sha256(
        {
            "provider": "yahoo_chart",
            "security_id": security_id,
            "provider_symbol": symbol,
            "active_from": active_from,
            "active_to": active_to,
        }
    )


def _internal_prices(security_id: str, count: int) -> list[dict]:
    sessions = pd.DatetimeIndex(
        cross_validation.xcals.get_calendar("XNYS").sessions_in_range(
            "2024-01-02", "2024-03-01"
        )[:count]
    ).tz_localize(None)
    rows = []
    for index, session in enumerate(sessions):
        close = float(100 + index)
        rows.append(
            {
                "security_id": security_id,
                "session": session.date().isoformat(),
                "open": close - 1.0,
                "high": close + 1.0,
                "low": close - 2.0,
                "close": close,
                "volume": 1000.0,
                "currency": "USD",
                "source": "eodhd",
                "source_url": "",
            }
        )
    return rows


def _internal_prices_for_sessions(
    security_id: str, sessions: pd.DatetimeIndex
) -> list[dict]:
    rows = []
    for index, session in enumerate(sessions):
        close = float(100 + index)
        rows.append(
            {
                "security_id": security_id,
                "session": pd.Timestamp(session).date().isoformat(),
                "open": close - 1.0,
                "high": close + 1.0,
                "low": close - 2.0,
                "close": close,
                "volume": 1000.0,
                "currency": "USD",
                "source": "eodhd",
                "source_url": "",
            }
        )
    return rows


def _chart_response(
    symbol: str,
    count: int,
    *,
    sessions=None,
    close_values: list[float] | None = None,
) -> bytes:
    sessions = (
        pd.DatetimeIndex(
            cross_validation.xcals.get_calendar("XNYS").sessions_in_range(
                "2024-01-02", "2024-03-01"
            )[:count]
        ).tz_localize(None)
        if sessions is None
        else pd.DatetimeIndex(pd.to_datetime(sessions))
    )
    timestamps = [
        int((pd.Timestamp(value).tz_localize("UTC") + pd.Timedelta(hours=16)).timestamp())
        for value in sessions
    ]
    close = (
        [float(100 + index) for index in range(count)]
        if close_values is None
        else [float(value) for value in close_values]
    )
    quote = {
        "open": [value - 1.0 for value in close],
        "high": [value + 1.0 for value in close],
        "low": [value - 2.0 for value in close],
        "close": close,
        "volume": [1000.0] * count,
    }
    return json.dumps(
        {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "symbol": symbol,
                            "currency": "USD",
                            "instrumentType": "EQUITY",
                            "exchangeName": "NMS",
                            "exchangeTimezoneName": "America/New_York",
                            "dataGranularity": "1d",
                        },
                        "timestamp": timestamps,
                        "indicators": {
                            "quote": [quote],
                            "adjclose": [{"adjclose": [1.0] * count}],
                        },
                    }
                ],
                "error": None,
            }
        },
        separators=(",", ":"),
    ).encode()


def _provider_report() -> dict:
    provider = _policy()["provider"]
    return {
        "name": "yahoo_chart",
        "access_class": provider["access_class"],
        "stability_note": provider["stability_note"],
        "http_attempts_this_run": 0,
        "request_cap": 400,
        "attempts_per_target_cap": 1,
        "retry_count": 0,
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
        "use_restriction": provider["use_restriction"],
    }


def _yahoo_envelope_payload(price_item: dict, raw_payload: bytes) -> bytes:
    return canonical_json_bytes(
        {
            "schema": "yahoo_chart_raw_response/v2",
            "symbol": str(price_item["provider_symbol"]),
            "request_period1": int(price_item["request_period1"]),
            "request_period2": int(price_item["request_period2"]),
            "source_url": str(price_item["source_url"]),
            "retrieved_at": "2026-07-18T00:00:00Z",
            "http_status": int(price_item["http_status"]),
            "content_type": "application/json",
            "content_sha256": sha256_bytes(raw_payload),
            "content_base64": base64.b64encode(raw_payload).decode("ascii"),
        }
    )


def _install_yahoo_envelopes(
    root: Path,
    repository: "FixtureRepository | None",
    price_rows: list[dict],
    payloads: dict[str, bytes] | None = None,
    payload_urls: dict[str, str] | None = None,
    payload_sources: dict[str, str] | None = None,
) -> None:
    for item in price_rows:
        source_url = str(item.get("source_url", ""))
        source_hash = str(item.get("source_sha256", ""))
        if not source_url.startswith("https://query1.finance.yahoo.com/") or not item.get(
            "cache_wrapper_sha256"
        ):
            continue
        if payloads is not None and source_hash in payloads:
            raw_payload = payloads[source_hash]
        else:
            assert repository is not None
            raw_row = repository.frames["source_archive"].loc[
                repository.frames["source_archive"]["source_hash"]
                .astype(str)
                .eq(source_hash)
            ].iloc[0]
            raw_payload = gzip.decompress(
                (root / str(raw_row["object_path"])).read_bytes()
            )
        envelope = _yahoo_envelope_payload(item, raw_payload)
        wrapper_hash = sha256_bytes(envelope)
        item["cache_wrapper_sha256"] = wrapper_hash
        if payloads is not None:
            payloads[wrapper_hash] = envelope
            assert payload_urls is not None and payload_sources is not None
            payload_urls[wrapper_hash] = source_url
            payload_sources[wrapper_hash] = "yahoo_chart_cache_envelope"
            continue
        assert repository is not None
        object_path = f"archives/2026-07-17/{wrapper_hash}.json.gz"
        (root / object_path).write_bytes(gzip.compress(envelope, mtime=0))
        row = {
            "archive_id": wrapper_hash,
            "object_path": object_path,
            "dataset": "yahoo_chart_cache_envelope",
            "source": "yahoo_chart_cache_envelope",
            "source_hash": wrapper_hash,
            "source_url": source_url,
            "content_type": "application/json",
        }
        existing = repository.frames["source_archive"]["archive_id"].astype(str).eq(
            wrapper_hash
        )
        if existing.any():
            for key, value in row.items():
                repository.frames["source_archive"].loc[existing, key] = value
        else:
            repository.frames["source_archive"] = pd.concat(
                [repository.frames["source_archive"], pd.DataFrame([row])],
                ignore_index=True,
            )


class FixtureRepository:
    def __init__(self, root: Path, frames, manifests):
        self.root = root
        self.frames = frames
        self.manifests = manifests

    def read_frame(self, dataset: str, _version: str):
        return self.frames[dataset].copy()

    def manifest_for_version(self, dataset: str, _version: str):
        return self.manifests[dataset]


def _fixture(
    root: Path,
    *,
    missing_price_identity: bool = False,
    unresolved: bool = False,
    with_boundary: bool = False,
    with_nonterminal: bool = False,
):
    versions = {
        "security_master": "master-v1",
        "symbol_history": "history-v1",
        "daily_price_raw": "price-v1",
        "corporate_actions": "actions-v1",
        "lifecycle_resolutions": "resolutions-v1",
        "adjustment_factors": "factors-v1",
    }
    release_versions = {
        **versions,
        "source_archive": "archive-v2",
        "cross_validation_reports": "cross-v1",
    }
    release = DataRelease(
        version="release-after-cross-validation",
        created_at="2026-07-18T00:00:00Z",
        completed_session="2026-07-17",
        dataset_versions=release_versions,
    )
    resolutions = pd.DataFrame(
        [
            {
                "candidate_id": "candidate-1",
                "security_id": "OLD",
                "symbol": "OLD",
                "last_price_date": "2024-01-05",
                "resolution": "applied",
                "event_id": "event-1",
                "exception_code": "",
                "exception_reason": "",
                "reviewed_by": "reviewer",
                "reviewed_at": "2026-07-18",
                "recheck_after": "",
                "successor_security_id": "NEW",
                "successor_symbol": "NEW",
                "source_url": "https://www.sec.gov/Archives/event.txt",
                "source": "lifecycle_finalizer",
                "retrieved_at": "2026-07-18T00:00:00Z",
                "source_hash": "a" * 64,
            }
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "event_id": "event-1",
                "security_id": "OLD",
                "action_type": "stock_merger",
                "effective_date": "2024-01-17",
                "new_security_id": "NEW",
                "new_symbol": "NEW",
                "cash_amount": None,
                "ratio": 1.0,
                "currency": "USD",
                "source_url": "https://www.sec.gov/Archives/event.txt",
                "source_hash": "",
                "source": "sec_edgar_filing",
                "source_kind": "official_crosscheck",
                "official": True,
            }
        ]
    )
    candidate_hash = "c" * 64
    lifecycle_evidence_report = canonical_json_bytes({"records": {"OLD": {}}})
    lifecycle_evidence_report_hash = sha256_bytes(lifecycle_evidence_report)
    input_hashes = {
        "candidate_set_sha256": candidate_hash,
        "lifecycle_resolutions_sha256": dataframe_sha256(
            resolutions, ("candidate_id",)
        ),
        "lifecycle_evidence_report_sha256": lifecycle_evidence_report_hash,
    }
    official = b"official filing bytes"
    old_response = _chart_response("OLD", 10)
    new_response = _chart_response("NEW", 12)
    official_hash = sha256_bytes(official)
    actions.loc[0, "source_hash"] = official_hash
    nonterminal_event = None
    nonterminal_extraction = None
    if with_nonterminal:
        actions = pd.concat(
            [
                actions,
                pd.DataFrame(
                    [
                        {
                            "event_id": "event-ticker-intermediate",
                            "security_id": "OLD",
                            "action_type": "ticker_change",
                            "effective_date": "2024-01-08",
                            "new_security_id": "OLD",
                            "new_symbol": "OLD2",
                            "cash_amount": None,
                            "ratio": None,
                            "currency": "USD",
                            "source_url": "https://www.sec.gov/Archives/event.txt",
                            "source_hash": official_hash,
                            "source": "sec_edgar_filing",
                            "source_kind": "official_filing",
                            "official": True,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        nonterminal_extraction = {
            "event_id": "event-ticker-intermediate",
            "security_id": "OLD",
            "action_type": "ticker_change",
            "effective_date": "2024-01-08",
            "new_security_id": "OLD",
            "new_symbol": "OLD2",
            "cash_amount": None,
            "ratio": None,
            "currency": "USD",
            "source_kind": "official_filing",
            "source_url": "https://www.sec.gov/Archives/event.txt",
            "source_hash": official_hash,
        }
        nonterminal_event = {
            "validation_kind": "nonterminal_official_provenance",
            "candidate_id": "",
            "event_id": "event-ticker-intermediate",
            "security_id": "OLD",
            "action_type": "ticker_change",
            "effective_date": "2024-01-08",
            "new_security_id": "OLD",
            "new_symbol": "OLD2",
            "cash_amount": None,
            "ratio": None,
            "currency": "USD",
            "status": "passed",
            "terms_match": True,
            "date_match": True,
            "official_original": True,
            "official_provenance_passed": True,
            "source_kind": "official_filing",
            "source_url": "https://www.sec.gov/Archives/event.txt",
            "evidence_sha256": official_hash,
            "lifecycle_report_extraction_approved": False,
            "reviewed_extraction_match": True,
            "reviewed_extraction_sha256": (
                cross_validation.reviewed_nonterminal_extraction_sha256(
                    nonterminal_extraction
                )
            ),
        }
    old_hash = sha256_bytes(old_response)
    new_hash = sha256_bytes(new_response)
    boundary_payload = b"official identity boundary bytes"
    boundary_hash = sha256_bytes(boundary_payload)
    boundary_url = "https://www.sec.gov/Archives/identity-boundary.txt"
    event = {
        "validation_kind": "terminal_resolution_report",
        "candidate_id": "candidate-1",
        "event_id": "event-1",
        "security_id": "OLD",
        "action_type": "stock_merger",
        "effective_date": "2024-01-17",
        "new_security_id": "NEW",
        "new_symbol": "NEW",
        "cash_amount": None,
        "ratio": 1.0,
        "currency": "USD",
        "status": "passed",
        "terms_match": True,
        "date_match": True,
        "official_original": True,
        "official_provenance_passed": True,
        "source_kind": "official_crosscheck",
        "source_url": "https://www.sec.gov/Archives/event.txt",
        "evidence_sha256": official_hash,
        "lifecycle_report_extraction_approved": True,
    }
    price_status = "unresolved" if unresolved else "passed"
    price_rows = [
        {
            "target_id": _target_id("OLD", "OLD", "2024-01-02", "2024-01-16"),
            "security_id": "OLD",
            "symbol": "OLD",
            "provider_symbol": "OLD",
            "identity_active_from": "2024-01-02",
            "identity_active_to": "2024-01-16",
            "status": price_status,
            "all_overlap_sessions_compared": not unresolved,
            "scale_stability_passed": not unresolved,
            "price_tolerance_passed": not unresolved,
            "session_coverage_passed": not unresolved,
            "currency_passed": not unresolved,
            "identity_boundary_passed": not unresolved,
            "overlap_session_count": 10 if not unresolved else 0,
            "independent_internal_price_rows": 10 if not unresolved else 0,
            "self_source_rows_excluded": 0,
            "source_sha256": old_hash,
            "cache_wrapper_sha256": "1" * 64,
            "source_url": _source_url("OLD", "2024-01-02", "2024-01-16"),
            "provider_symbol": "OLD",
            "http_status": 200,
            "provider_currency": "USD",
            "provider_adjustment_basis": "raw_quote_ohlcv",
            "adjusted_close_used": False,
            "provider_sessions_before_identity": 0,
            "provider_sessions_after_identity": 0,
            "provider_history_start": "2024-01-02",
            "provider_history_end": "2024-01-16",
            "provider_history_session_count": 10,
            "eodhd_history_start": "2024-01-02",
            "eodhd_history_end": "2024-01-16",
            "eodhd_history_session_count": 10,
            "eodhd_full_history_overlap_ratio": 1.0,
            "session_coverage_ratio": 1.0,
            **_request_report_fields("OLD", "2024-01-02", "2024-01-16", 10),
        }
    ]
    if not missing_price_identity:
        price_rows.append(
            {
                "target_id": _target_id("NEW", "NEW", "2024-01-02", ""),
                "security_id": "NEW",
                "symbol": "NEW",
                "provider_symbol": "NEW",
                "identity_active_from": "2024-01-02",
                "identity_active_to": "",
                "status": "passed",
                "all_overlap_sessions_compared": True,
                "scale_stability_passed": True,
                "price_tolerance_passed": True,
                "session_coverage_passed": True,
                "currency_passed": True,
                "identity_boundary_passed": True,
                "overlap_session_count": 12,
                "independent_internal_price_rows": 12,
                "self_source_rows_excluded": 0,
                "source_sha256": new_hash,
                "cache_wrapper_sha256": "2" * 64,
                "source_url": _source_url("NEW", "2024-01-02", "2024-01-18"),
                "provider_symbol": "NEW",
                "http_status": 200,
                "provider_currency": "USD",
                "provider_adjustment_basis": "raw_quote_ohlcv",
                "adjusted_close_used": False,
                "provider_sessions_before_identity": 0,
                "provider_sessions_after_identity": 0,
                "provider_history_start": "2024-01-02",
                "provider_history_end": "2024-01-18",
                "provider_history_session_count": 12,
                "eodhd_history_start": "2024-01-02",
                "eodhd_history_end": "2024-01-18",
                "eodhd_history_session_count": 12,
                "eodhd_full_history_overlap_ratio": 1.0,
                "session_coverage_ratio": 1.0,
                **_request_report_fields("NEW", "2024-01-02", "2024-01-18", 12),
            }
        )
    if with_boundary:
        price_rows[0].update(
            {
                "identity_active_from": "2024-01-02",
                "identity_active_to": "2024-01-16",
                "provider_sessions_before_identity": 1,
                "provider_sessions_after_identity": 0,
                "identity_boundary_evidence": [
                    {
                        "boundary": "active_from",
                        "date": "2024-01-02",
                        "source_url": boundary_url,
                        "source_kind": "sec_filing",
                        "evidence_sha256": boundary_hash,
                        "official_original": True,
                    }
                ],
            }
        )
    report_events = [event] + ([nonterminal_event] if nonterminal_event else [])
    summary = {
        "event_count": len(report_events),
        "event_mismatch_count": 0,
        "nonterminal_event_count": int(with_nonterminal),
        "reviewed_nonterminal_event_count": int(with_nonterminal),
        "permanent_exception_count": 0,
        "permanent_exception_mismatch_count": 0,
        "price_target_count": len(price_rows),
        "price_pass_count": sum(item["status"] == "passed" for item in price_rows),
        "price_exception_count": 0,
        "price_unresolved_count": sum(item["status"] == "unresolved" for item in price_rows),
        "price_mismatch_count": 0,
        "overlap_session_count": sum(item["overlap_session_count"] for item in price_rows),
    }
    policy = _policy()
    if nonterminal_extraction is not None:
        policy["events"]["reviewed_nonterminal_extractions"] = [
            *policy["events"]["reviewed_nonterminal_extractions"],
            nonterminal_extraction,
        ]
    if with_boundary:
        policy["identity_boundaries"] = [
            {
                "symbol": "OLD",
                "boundary": "active_from",
                "date": "2024-01-02",
                "source_url": boundary_url,
                "source_kind": "sec_filing",
            }
        ]
    report = {
        "schema": CROSS_VALIDATION_SCHEMA,
        "status": "passed",
        "provider": _provider_report(),
        "validated_versions": versions,
        "input_hashes": input_hashes,
        "lifecycle_evidence_report_sha256": lifecycle_evidence_report_hash,
        "policy": policy,
        "events": report_events,
        "permanent_exceptions": [],
        "prices": price_rows,
        "summary": summary,
    }
    payloads = {
        lifecycle_evidence_report_hash: lifecycle_evidence_report,
        official_hash: official,
        old_hash: old_response,
        new_hash: new_response,
    }
    payload_urls = {
        official_hash: "https://www.sec.gov/Archives/event.txt",
        lifecycle_evidence_report_hash: "archive://lifecycle/evidence-report",
        old_hash: _source_url("OLD", "2024-01-02", "2024-01-16"),
        new_hash: _source_url("NEW", "2024-01-02", "2024-01-18"),
    }
    payload_sources = {
        lifecycle_evidence_report_hash: "us_lifecycle_evidence_report",
        official_hash: "sec_edgar_filing",
        old_hash: "yahoo_chart_json",
        new_hash: "yahoo_chart_json",
    }
    _install_yahoo_envelopes(
        root,
        None,
        price_rows,
        payloads,
        payload_urls,
        payload_sources,
    )
    report_bytes = canonical_json_bytes(report)
    report_hash = sha256_bytes(report_bytes)
    payloads[report_hash] = report_bytes
    payload_sources[report_hash] = "us_lifecycle_cross_validation"
    if with_boundary:
        payloads[boundary_hash] = boundary_payload
        payload_urls[boundary_hash] = boundary_url
        payload_sources[boundary_hash] = "sec_edgar_filing"
    archive_rows = []
    for digest, payload in payloads.items():
        object_path = f"archives/2026-07-17/{digest}.bin.gz"
        path = root / object_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(payload, mtime=0))
        archive_rows.append(
            {
                "archive_id": digest,
                "object_path": object_path,
                "dataset": payload_sources[digest],
                "source": payload_sources[digest],
                "source_hash": digest,
                "source_url": payload_urls.get(digest, ""),
                "content_type": (
                    "application/json"
                    if payload_urls.get(digest, "").startswith(
                        "https://query1.finance.yahoo.com/"
                    )
                    else "application/octet-stream"
                ),
            }
        )
    cross_row = {
        "report_id": report_hash,
        "report_archive_id": report_hash,
        "source_hash": report_hash,
        "status": "passed",
        "provider": "yahoo_chart",
        "policy_sha256": canonical_json_sha256(policy),
        "lifecycle_evidence_report_sha256": lifecycle_evidence_report_hash,
        "validated_versions_json": json.dumps(
            versions, sort_keys=True, separators=(",", ":")
        ),
        **summary,
    }
    metadata = {
        "report_id": report_hash,
        "status": "passed",
        "provider": "yahoo_chart",
        "policy_sha256": canonical_json_sha256(policy),
        "lifecycle_evidence_report_sha256": lifecycle_evidence_report_hash,
        "validated_versions": versions,
        "input_hashes": input_hashes,
        **summary,
    }
    frames = {
        "cross_validation_reports": pd.DataFrame([cross_row]),
        "source_archive": pd.DataFrame(archive_rows),
        "security_master": pd.DataFrame(
            [
                {
                    "security_id": "OLD",
                    "primary_symbol": "OLD",
                    "active_from": "2024-01-02",
                    "active_to": "2024-01-16",
                },
                {
                    "security_id": "NEW",
                    "primary_symbol": "NEW",
                    "active_from": "2024-01-02",
                    "active_to": "",
                },
            ]
        ),
        "symbol_history": pd.DataFrame(
            [
                {
                    "security_id": "OLD",
                    "symbol": "OLD",
                    "effective_from": "2024-01-02",
                    "effective_to": "2024-01-16",
                },
                {
                    "security_id": "NEW",
                    "symbol": "NEW",
                    "effective_from": "2024-01-02",
                    "effective_to": "",
                },
            ]
        ),
        "daily_price_raw": pd.DataFrame(
            [*_internal_prices("OLD", 10), *_internal_prices("NEW", 12)]
        ),
        "lifecycle_resolutions": resolutions,
        "corporate_actions": actions,
        "adjustment_factors": pd.DataFrame(),
    }
    manifests = {
        "cross_validation_reports": SimpleNamespace(metadata=metadata),
        "lifecycle_resolutions": SimpleNamespace(
            metadata={
                "candidate_set_sha256": candidate_hash,
                "evidence_report_sha256": lifecycle_evidence_report_hash,
            }
        ),
    }
    return FixtureRepository(root, frames, manifests), release


def _rewrite_report(root: Path, repository: FixtureRepository, mutate) -> None:
    row = repository.frames["cross_validation_reports"].iloc[0]
    old_id = str(row["report_id"])
    archive_mask = repository.frames["source_archive"]["archive_id"].astype(str).eq(
        old_id
    )
    archive_row = repository.frames["source_archive"].loc[archive_mask].iloc[0]
    report = json.loads(
        gzip.decompress((root / str(archive_row["object_path"])).read_bytes())
    )
    mutate(report)
    _install_yahoo_envelopes(root, repository, report["prices"])
    archive_mask = repository.frames["source_archive"]["archive_id"].astype(str).eq(
        old_id
    )
    payload = canonical_json_bytes(report)
    new_id = sha256_bytes(payload)
    object_path = f"archives/2026-07-17/{new_id}.json.gz"
    destination = root / object_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(gzip.compress(payload, mtime=0))
    repository.frames["source_archive"].loc[archive_mask, "archive_id"] = new_id
    repository.frames["source_archive"].loc[archive_mask, "source_hash"] = new_id
    repository.frames["source_archive"].loc[archive_mask, "object_path"] = object_path
    for key in ("report_id", "report_archive_id", "source_hash"):
        repository.frames["cross_validation_reports"].loc[0, key] = new_id
    repository.manifests["cross_validation_reports"].metadata["report_id"] = new_id


def _install_permanent_exception(
    root: Path,
    repository: FixtureRepository,
    *,
    self_authored: bool = False,
) -> tuple[str, SimpleNamespace]:
    lifecycle_metadata = repository.manifests["lifecycle_resolutions"].metadata
    official_payload = b"official permanent lifecycle exception filing"
    official_hash = sha256_bytes(official_payload)
    official_url = (
        "https://www.sec.gov/Archives/edgar/data/1/"
        "000000000124000001/permanent-exception.txt"
    )
    official_object_path = f"archives/2026-07-17/{official_hash}.bin.gz"
    (root / official_object_path).write_bytes(
        gzip.compress(official_payload, mtime=0)
    )
    repository.frames["source_archive"] = pd.concat(
        [
            repository.frames["source_archive"],
            pd.DataFrame(
                [
                    {
                        "archive_id": official_hash,
                        "object_path": official_object_path,
                        "source_hash": official_hash,
                        "source_url": official_url,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    if self_authored:
        source_hash = str(lifecycle_metadata["evidence_report_sha256"])
        source_url = "archive://lifecycle/evidence-report"
    else:
        source_hash = official_hash
        source_url = official_url
    last_price_date = "2024-01-05"
    candidate_id = cross_validation.lifecycle_candidate_id(
        "OLD", last_price_date
    )
    permanent_resolution = {
        "candidate_id": candidate_id,
        "security_id": "OLD",
        "symbol": "OLD",
        "last_price_date": last_price_date,
        "resolution": "exception",
        "event_id": "",
        "exception_code": "unsupported_consideration",
        "exception_reason": "CVR terms are not representable.",
        "reviewed_by": "reviewer",
        "reviewed_at": "2026-07-18",
        "recheck_after": "",
        "successor_security_id": "",
        "successor_symbol": "",
        "source_url": source_url,
        "source": "lifecycle_finalizer",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": source_hash,
    }
    repository.frames["lifecycle_resolutions"] = pd.concat(
        [
            repository.frames["lifecycle_resolutions"],
            pd.DataFrame([permanent_resolution]),
        ],
        ignore_index=True,
    )
    resolution_hash = dataframe_sha256(
        repository.frames["lifecycle_resolutions"], ("candidate_id",)
    )
    permanent_report = {
        "validation_kind": (
            "permanent_lifecycle_exception_official_provenance"
        ),
        "evidence_id": "fixture_permanent_exception",
        "candidate_id": candidate_id,
        "security_id": "OLD",
        "symbol": "OLD",
        "last_price_date": last_price_date,
        "exception_code": "unsupported_consideration",
        "exception_reason": "CVR terms are not representable.",
        "status": "passed",
        "identity_date_bound": True,
        "registry_binding_passed": True,
        "reviewer_pin_passed": True,
        "official_original": True,
        "exact_archive_pair": True,
        "archive_payload_verified": True,
        "source_url": source_url,
        "evidence_sha256": source_hash,
        "reasons": [],
    }

    def mutate(report):
        report["input_hashes"]["lifecycle_resolutions_sha256"] = resolution_hash
        report["permanent_exceptions"] = [permanent_report]
        report["summary"]["permanent_exception_count"] = 1
        report["summary"]["permanent_exception_mismatch_count"] = 0

    _rewrite_report(root, repository, mutate)
    cross_row = repository.frames["cross_validation_reports"]
    cross_row.loc[0, "permanent_exception_count"] = 1
    cross_row.loc[0, "permanent_exception_mismatch_count"] = 0
    metadata = repository.manifests["cross_validation_reports"].metadata
    metadata["input_hashes"]["lifecycle_resolutions_sha256"] = resolution_hash
    metadata["permanent_exception_count"] = 1
    metadata["permanent_exception_mismatch_count"] = 0
    official_spec = SimpleNamespace(
        evidence_id="fixture_permanent_exception",
        resolution_kind="exception",
        candidate_security_ids=("OLD",),
        candidate_symbols=("OLD",),
        candidate_last_price_dates=(last_price_date,),
        pinned=True,
        exception_code="unsupported_consideration",
        claim="CVR terms are not representable.",
        source_url=official_url,
        source_sha256=official_hash,
    )
    return candidate_id, official_spec


def _embedded_reviewed_inventory_sha256(
    root: Path, repository: FixtureRepository
) -> str:
    report_id = str(
        repository.frames["cross_validation_reports"].iloc[0]["report_id"]
    )
    archive_row = repository.frames["source_archive"].loc[
        repository.frames["source_archive"]["archive_id"].astype(str).eq(
            report_id
        )
    ].iloc[0]
    report = json.loads(
        gzip.decompress((root / str(archive_row["object_path"])).read_bytes())
    )
    return cross_validation.reviewed_nonterminal_inventory_sha256(
        report["policy"]["events"]
    )


def _rewrite_price_response(
    root: Path,
    repository: FixtureRepository,
    *,
    price_index: int,
    payload: bytes,
) -> None:
    report_id = str(repository.frames["cross_validation_reports"].iloc[0]["report_id"])
    report_row = repository.frames["source_archive"].loc[
        repository.frames["source_archive"]["archive_id"].astype(str).eq(report_id)
    ].iloc[0]
    report = json.loads(
        gzip.decompress((root / str(report_row["object_path"])).read_bytes())
    )
    old_hash = str(report["prices"][price_index]["source_sha256"])
    new_hash = sha256_bytes(payload)
    object_path = f"archives/2026-07-17/{new_hash}.json.gz"
    (root / object_path).write_bytes(gzip.compress(payload, mtime=0))
    source_mask = repository.frames["source_archive"]["archive_id"].astype(str).eq(
        old_hash
    )
    repository.frames["source_archive"].loc[source_mask, "archive_id"] = new_hash
    repository.frames["source_archive"].loc[source_mask, "source_hash"] = new_hash
    repository.frames["source_archive"].loc[source_mask, "object_path"] = object_path
    _rewrite_report(
        root,
        repository,
        lambda value: value["prices"][price_index].update(
            {"source_sha256": new_hash}
        ),
    )


def _install_terminal_no_data_exception(
    root: Path,
    repository: FixtureRepository,
    *,
    derived_boundary: bool = False,
) -> None:
    report_id = str(
        repository.frames["cross_validation_reports"].iloc[0]["report_id"]
    )
    report_row = repository.frames["source_archive"].loc[
        repository.frames["source_archive"]["archive_id"].astype(str).eq(
            report_id
        )
    ].iloc[0]
    report = json.loads(
        gzip.decompress((root / str(report_row["object_path"])).read_bytes())
    )
    old_source_hash = str(report["prices"][0]["source_sha256"])

    sessions = pd.DatetimeIndex(
        cross_validation.xcals.get_calendar("XNYS").sessions_in_range(
            "2023-01-01", "2024-01-16"
        )[-60:]
    ).tz_localize(None)
    active_from = sessions[0].date().isoformat()
    active_to = sessions[-1].date().isoformat()
    request_end = (
        pd.to_datetime(
            repository.frames["daily_price_raw"]["session"], errors="coerce"
        )
        .max()
        .date()
        .isoformat()
        if derived_boundary
        else active_to
    )
    effective_date = pd.Timestamp(
        cross_validation.xcals.get_calendar("XNYS").next_session(
            pd.Timestamp(active_to)
        )
    ).tz_localize(None).date().isoformat()
    source_url = _source_url("OLD", active_from, request_end)
    no_data_payload = json.dumps(
        {
            "chart": {
                "result": None,
                "error": {
                    "code": "Not Found",
                    "description": "No data found, symbol may be delisted",
                },
            }
        },
        separators=(",", ":"),
    ).encode()
    source_hash = sha256_bytes(no_data_payload)
    object_path = f"archives/2026-07-17/{source_hash}.json.gz"
    (root / object_path).write_bytes(gzip.compress(no_data_payload, mtime=0))
    source_mask = repository.frames["source_archive"]["archive_id"].astype(
        str
    ).eq(old_source_hash)
    repository.frames["source_archive"].loc[source_mask, "archive_id"] = source_hash
    repository.frames["source_archive"].loc[source_mask, "source_hash"] = source_hash
    repository.frames["source_archive"].loc[source_mask, "object_path"] = object_path
    repository.frames["source_archive"].loc[source_mask, "source_url"] = source_url

    old_prices = pd.DataFrame(_internal_prices_for_sessions("OLD", sessions))
    repository.frames["daily_price_raw"] = pd.concat(
        [
            old_prices,
            repository.frames["daily_price_raw"].loc[
                repository.frames["daily_price_raw"]["security_id"].eq("NEW")
            ],
        ],
        ignore_index=True,
    )
    repository.frames["security_master"].loc[
        repository.frames["security_master"]["security_id"].eq("OLD"),
        ["active_from", "active_to"],
    ] = [active_from, "" if derived_boundary else active_to]
    repository.frames["symbol_history"].loc[
        repository.frames["symbol_history"]["security_id"].eq("OLD"),
        ["effective_from", "effective_to"],
    ] = [active_from, "" if derived_boundary else active_to]
    repository.frames["corporate_actions"].loc[
        repository.frames["corporate_actions"]["event_id"].eq("event-1"),
        "effective_date",
    ] = effective_date
    repository.frames["lifecycle_resolutions"].loc[
        repository.frames["lifecycle_resolutions"]["event_id"].eq("event-1"),
        "last_price_date",
    ] = active_to
    resolution_hash = dataframe_sha256(
        repository.frames["lifecycle_resolutions"], ("candidate_id",)
    )
    terminal_calendar = {
        "terminal_session": active_to,
        "expected_sessions": 60,
        "present_sessions": 60,
        "missing": [],
    }
    request_fields = _request_report_fields("OLD", active_from, request_end, 60)

    def mutate(value):
        event = value["events"][0]
        event["effective_date"] = effective_date
        price = value["prices"][0]
        price.update(
            {
                "target_id": _target_id(
                    "OLD",
                    "OLD",
                    active_from,
                    "" if derived_boundary else active_to,
                ),
                "identity_active_from": active_from,
                "identity_active_to": "" if derived_boundary else active_to,
                "terminal_event_id": "event-1",
                "successor_security_id": "NEW",
                "status": "explicit_exception",
                "provider_support": "no_data",
                "provider_currency": "unavailable_no_price_payload",
                "provider_adjustment_basis": "no_price_payload",
                "adjusted_close_used": False,
                "response_identity_match": True,
                "no_data_evidence_kind": "chart_not_found",
                "no_data_error_code": "Not Found",
                "no_data_error_description": (
                    "No data found, symbol may be delisted"
                ),
                "source_url": source_url,
                "source_sha256": source_hash,
                "cache_wrapper_sha256": "f" * 64,
                "http_status": 404,
                "overlap_session_count": 0,
                "independent_internal_price_rows": 60,
                "self_source_rows_excluded": 0,
                "exception": {
                    "code": "delisted_provider_unsupported",
                    "official_event_verified": True,
                    "official_event_id": "event-1",
                    "official_action_type": "stock_merger",
                    "official_evidence_sha256": event["evidence_sha256"],
                    "identity_event_match": True,
                    "identity_date_match": True,
                    "identity_date_basis": (
                        "derived_local_terminal_session"
                        if derived_boundary
                        else "stored_identity_active_to"
                    ),
                    "derived_identity_active_to": (
                        active_to if derived_boundary else ""
                    ),
                    "terminal_calendar_complete": True,
                    "terminal_calendar": terminal_calendar,
                    "successor_security_id": "NEW",
                    "successor_requirement_passed": True,
                    "successor_validation": {
                        "required": True,
                        "passed": True,
                        "target_id": _target_id("NEW", "NEW", "2024-01-02", ""),
                        "provider_symbol": "NEW",
                        "status": "passed",
                        "reason": "",
                        "candidate_count": 1,
                    },
                    "response_identity_match": True,
                    "no_data_evidence_validated": True,
                },
                **request_fields,
            }
        )
        value["input_hashes"]["lifecycle_resolutions_sha256"] = resolution_hash
        value["summary"].update(
            {
                "price_pass_count": 1,
                "price_exception_count": 1,
                "price_mismatch_count": 0,
                "price_unresolved_count": 0,
                "overlap_session_count": 12,
            }
        )

    _rewrite_report(root, repository, mutate)
    cross_row = repository.frames["cross_validation_reports"]
    metadata = repository.manifests["cross_validation_reports"].metadata
    for key, value in {
        "price_pass_count": 1,
        "price_exception_count": 1,
        "price_mismatch_count": 0,
        "price_unresolved_count": 0,
        "overlap_session_count": 12,
    }.items():
        cross_row.loc[0, key] = value
        metadata[key] = value
    metadata["input_hashes"]["lifecycle_resolutions_sha256"] = resolution_hash


class CrossValidationPublicationGateTest(unittest.TestCase):
    def test_publication_policy_rejects_successor_chain_tampering_and_cycles(self):
        exact = _policy()
        cross_validation._validate_policy_contract(exact)

        changed_hash = copy.deepcopy(exact)
        changed_hash["prices"]["reviewed_no_data_successor_chains"][0]["final"][
            "source_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(
            RuntimeError, "successor-chain inventory is not code-pinned"
        ):
            cross_validation._validate_policy_contract(changed_hash)

        cyclic = copy.deepcopy(exact)
        chain = cyclic["prices"]["reviewed_no_data_successor_chains"][0]
        chain["final"]["target_id"] = chain["root_target_id"]
        with self.assertRaisesRegex(RuntimeError, "cyclic or repeats a target"):
            cross_validation._validate_policy_contract(cyclic)

    def test_publication_policy_rejects_unsupported_path_tampering(self):
        exact = _policy()
        cross_validation._validate_policy_contract(exact)

        changed = copy.deepcopy(exact)
        changed["prices"]["reviewed_no_data_unsupported_paths"][0][
            "cache_wrapper_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(
            RuntimeError,
            "unsupported-path inventory is not the exact isolated code-pinned set",
        ):
            cross_validation._validate_policy_contract(changed)

    def test_publication_unsupported_path_binding_is_exactly_pinned(self):
        prices_policy = _policy()["prices"]
        spec = prices_policy["reviewed_no_data_unsupported_paths"][0]
        target = {
            "target_id": spec["target_id"],
            "security_id": spec["security_id"],
            "provider_symbol": spec["provider_symbol"],
            "active_from": spec["identity_active_from"],
            "active_to": spec["identity_active_to"],
        }
        event = {
            "event_id": spec["event_id"],
            "candidate_id": spec["candidate_id"],
            "action_type": spec["action_type"],
            "effective_date": spec["event_effective_date"],
            "evidence_sha256": spec["official_evidence_sha256"],
            "status": "passed",
        }
        prices = pd.DataFrame(
            [
                {
                    "security_id": spec["security_id"],
                    "session": spec["last_price_date"],
                    "open": 2.3,
                    "high": 2.3,
                    "low": 2.3,
                    "close": 2.3,
                    "volume": 0,
                    "currency": "USD",
                    "source_hash": spec["internal_price_source_hash"],
                }
            ]
        )

        binding = cross_validation.unsupported_path_no_data_binding(
            target,
            spec["last_price_date"],
            event,
            prices,
            prices_policy,
            source_sha256=spec["source_sha256"],
            cache_wrapper_sha256=spec["cache_wrapper_sha256"],
        )
        self.assertIsNotNone(binding)
        self.assertEqual(
            binding["validation_basis"],
            cross_validation.REVIEWED_NO_DATA_UNSUPPORTED_PATH_BASIS,
        )
        self.assertFalse(binding["generic_date_tolerance"])

        changed_event = copy.deepcopy(event)
        changed_event["evidence_sha256"] = "0" * 64
        changed_prices = prices.copy()
        changed_prices.loc[0, "source_hash"] = "0" * 64
        for changed_event_value, changed_prices_value, wrapper in (
            (changed_event, prices, spec["cache_wrapper_sha256"]),
            (event, changed_prices, spec["cache_wrapper_sha256"]),
            (event, prices, "0" * 64),
        ):
            with self.subTest(
                event_hash=changed_event_value["evidence_sha256"],
                price_hash=changed_prices_value.loc[0, "source_hash"],
                wrapper=wrapper,
            ):
                self.assertIsNone(
                    cross_validation.unsupported_path_no_data_binding(
                        target,
                        spec["last_price_date"],
                        changed_event_value,
                        changed_prices_value,
                        prices_policy,
                        source_sha256=spec["source_sha256"],
                        cache_wrapper_sha256=wrapper,
                    )
                )

    def test_ntco_dedicated_event_policy_is_code_pinned(self):
        policy = _policy()
        cross_validation._validate_policy_contract(policy)
        policy["events"]["reviewed_ntco_transition_event_ids"] = [
            "0bd231d216016d33c8d6eb1ec9bb85450a21b9298c4abda34a3399727e9b2c00"
        ]
        with self.assertRaisesRegex(RuntimeError, "exact isolated policy set"):
            cross_validation._validate_policy_contract(policy)

    def test_code_pinned_registry_includes_exact_bankruptcy_warrant_bindings(self):
        specs = cross_validation.trusted_permanent_exception_specs()
        expected = {
            "legacy_do_2021_warrant_consideration": (
                "US:EODHD:2826c370-0467-5e82-9617-dcece5be407f",
                "DO",
                "2020-04-24",
                "2021-04-23",
                "048886a54f9b70198d2e70805aedc56a4eb11c4e84f970f89bc931006594d210",
            ),
            "legacy_dnr_2020_warrant_consideration": (
                "US:EODHD:6d9d4638-4922-5f6c-89fd-6b79db60c1c3",
                "DNR",
                "2020-07-28",
                "2020-09-18",
                "79295f28796f0e0c5de91e88a6accde76517e25267a189eccbf852674cb78229",
            ),
            "legacy_ne_2021_warrant_consideration": (
                "US:EODHD:81b3ca1f-cf1b-5234-bc24-4399b8ecf149",
                "NE",
                "2020-10-22",
                "2021-02-05",
                "9bdaadb02c741ee474a15aa6a95f0674a6fa41d4acd3f99384cc10d9dc60e65f",
            ),
        }
        for evidence_id, values in expected.items():
            with self.subTest(evidence_id=evidence_id):
                security_id, symbol, terminal, effective, source_sha256 = values
                spec = specs[evidence_id]
                self.assertEqual(spec.candidate_security_ids, (security_id,))
                self.assertEqual(spec.candidate_symbols, (symbol,))
                self.assertEqual(spec.candidate_last_price_dates, (terminal,))
                self.assertEqual(spec.effective_date, effective)
                self.assertEqual(spec.exception_code, "unsupported_consideration")
                self.assertEqual(spec.source_sha256, source_sha256)

    def test_valid_exact_report_covers_applied_event_and_both_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, release = _fixture(Path(directory))
            result = validate_cross_validation_gate(repository, release)
            self.assertEqual(result["event_count"], 1)
            self.assertEqual(result["price_target_count"], 2)
            self.assertEqual(result["price_unresolved_count"], 0)

    def test_terminal_no_data_exception_rejects_forged_allowed_action_type(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(root)
            _install_terminal_no_data_exception(root, repository)

            result = validate_cross_validation_gate(repository, release)
            self.assertEqual(result["price_exception_count"], 1)

            _rewrite_report(
                root,
                repository,
                lambda report: report["prices"][0]["exception"].update(
                    {"official_action_type": "ticker_change"}
                ),
            )
            with self.assertRaisesRegex(
                RuntimeError, "exact official terminal identity/date"
            ):
                validate_cross_validation_gate(repository, release)

    def test_derived_terminal_boundary_is_recomputed_and_cannot_be_forged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(root)
            _install_terminal_no_data_exception(
                root,
                repository,
                derived_boundary=True,
            )

            result = validate_cross_validation_gate(repository, release)
            self.assertEqual(result["price_exception_count"], 1)

            for field, value in (
                ("identity_date_basis", "stored_identity_active_to"),
                ("derived_identity_active_to", "2024-01-12"),
            ):
                with self.subTest(field=field):
                    changed_repository, changed_release = _fixture(root)
                    _install_terminal_no_data_exception(
                        root,
                        changed_repository,
                        derived_boundary=True,
                    )
                    _rewrite_report(
                        root,
                        changed_repository,
                        lambda report, key=field, item=value: report["prices"][0][
                            "exception"
                        ].update({key: item}),
                    )
                    with self.assertRaisesRegex(
                        RuntimeError, "exact official terminal identity/date"
                    ):
                        validate_cross_validation_gate(
                            changed_repository, changed_release
                        )

    def test_http_400_exception_requires_exact_report_request_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(root)
            _install_terminal_no_data_exception(root, repository)
            report_id = str(
                repository.frames["cross_validation_reports"].iloc[0][
                    "report_id"
                ]
            )
            report_archive = repository.frames["source_archive"].loc[
                repository.frames["source_archive"]["archive_id"]
                .astype(str)
                .eq(report_id)
            ].iloc[0]
            report = json.loads(
                gzip.decompress(
                    (root / str(report_archive["object_path"])).read_bytes()
                )
            )
            period1 = int(report["prices"][0]["request_period1"])
            period2 = int(report["prices"][0]["request_period2"])

            def payload(echoed_start: int, echoed_end: int) -> bytes:
                return json.dumps(
                    {
                        "chart": {
                            "result": None,
                            "error": {
                                "code": "Bad Request",
                                "description": (
                                    "Data doesn't exist for startDate = "
                                    f"{echoed_start}, endDate = {echoed_end}"
                                ),
                            },
                        }
                    },
                    separators=(",", ":"),
                ).encode()

            exact = payload(period1, period2)
            _rewrite_price_response(
                root,
                repository,
                price_index=0,
                payload=exact,
            )
            _rewrite_report(
                root,
                repository,
                lambda report: report["prices"][0].update(
                    {
                        "http_status": 400,
                        "no_data_evidence_kind": (
                            "http_400_bounded_history_not_found"
                        ),
                        "no_data_error_code": "Bad Request",
                        "no_data_error_description": (
                            "Data doesn't exist for startDate = "
                            f"{period1}, endDate = {period2}"
                        ),
                    }
                ),
            )
            result = validate_cross_validation_gate(repository, release)
            self.assertEqual(result["price_exception_count"], 1)

            _rewrite_price_response(
                root,
                repository,
                price_index=0,
                payload=payload(period1 + 1, period2),
            )
            with self.assertRaisesRegex(
                RuntimeError, "failed strict validation"
            ):
                validate_cross_validation_gate(repository, release)

    def test_retired_yhd_exception_requires_an_exact_empty_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(root)
            _install_terminal_no_data_exception(root, repository)

            def payload(*, exchange: str = "YHD", quote=None) -> bytes:
                return json.dumps(
                    {
                        "chart": {
                            "result": [
                                {
                                    "meta": {
                                        "symbol": "OLD",
                                        "currency": None,
                                        "instrumentType": "MUTUALFUND",
                                        "exchangeName": exchange,
                                        "fullExchangeName": "YHD",
                                        "exchangeTimezoneName": (
                                            "America/New_York"
                                        ),
                                        "dataGranularity": "1d",
                                        "range": "",
                                    },
                                    "indicators": {
                                        "quote": [{}] if quote is None else quote,
                                        "adjclose": [{}],
                                    },
                                }
                            ],
                            "error": None,
                        }
                    },
                    separators=(",", ":"),
                ).encode()

            exact = payload()
            _rewrite_price_response(
                root,
                repository,
                price_index=0,
                payload=exact,
            )
            _rewrite_report(
                root,
                repository,
                lambda report: report["prices"][0].update(
                    {
                        "http_status": 200,
                        "no_data_evidence_kind": (
                            "http_200_empty_retired_yhd_placeholder"
                        ),
                        "no_data_error_code": "",
                        "no_data_error_description": "",
                    }
                ),
            )
            result = validate_cross_validation_gate(repository, release)
            self.assertEqual(result["price_exception_count"], 1)

            _rewrite_price_response(
                root,
                repository,
                price_index=0,
                payload=payload(exchange="NMS"),
            )
            with self.assertRaisesRegex(
                RuntimeError, "failed strict validation"
            ):
                validate_cross_validation_gate(repository, release)

    def test_permanent_exception_requires_exact_official_archive_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(root)
            _, official_spec = _install_permanent_exception(root, repository)

            with patch.object(
                cross_validation,
                "trusted_permanent_exception_specs",
                return_value={official_spec.evidence_id: official_spec},
            ):
                result = validate_cross_validation_gate(repository, release)

        self.assertEqual(result["permanent_exception_count"], 1)
        self.assertEqual(result["permanent_exception_mismatch_count"], 0)

    def test_self_authored_report_cannot_authorize_permanent_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(root)
            candidate_id, official_spec = _install_permanent_exception(
                root, repository, self_authored=True
            )

            with patch.object(
                cross_validation,
                "trusted_permanent_exception_specs",
                return_value={official_spec.evidence_id: official_spec},
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Permanent lifecycle exception identity/date/provenance is invalid",
                ):
                    validate_cross_validation_gate(repository, release)

        self.assertTrue(candidate_id)

    def test_nonterminal_official_action_does_not_require_applied_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(
                root, with_nonterminal=True
            )
            with patch.object(
                cross_validation,
                "TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256",
                _embedded_reviewed_inventory_sha256(root, repository),
            ):
                result = validate_cross_validation_gate(repository, release)
            self.assertEqual(result["event_count"], 2)
            self.assertEqual(result["event_mismatch_count"], 0)

    def test_nonterminal_action_still_requires_reviewed_official_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(
                root, with_nonterminal=True
            )
            action = repository.frames["corporate_actions"]["event_id"].eq(
                "event-ticker-intermediate"
            )
            repository.frames["corporate_actions"].loc[
                action, "source_kind"
            ] = "provider"

            with patch.object(
                cross_validation,
                "TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256",
                _embedded_reviewed_inventory_sha256(root, repository),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "Nonterminal lifecycle event official provenance"
                ):
                    validate_cross_validation_gate(repository, release)

    def test_embedded_nonterminal_manifest_cannot_self_authorize(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, release = _fixture(
                Path(directory), with_nonterminal=True
            )

            with self.assertRaisesRegex(
                RuntimeError, "not the code-pinned manifest"
            ):
                validate_cross_validation_gate(repository, release)

    def test_nonterminal_action_and_report_cannot_jointly_forge_reviewed_terms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(root, with_nonterminal=True)
            target = repository.frames["corporate_actions"]["event_id"].eq(
                "event-ticker-intermediate"
            )
            repository.frames["corporate_actions"].loc[
                target, "new_security_id"
            ] = "FORGED-ID"
            _rewrite_report(
                root,
                repository,
                lambda report: report["events"][1].update(
                    {"new_security_id": "FORGED-ID"}
                ),
            )

            with patch.object(
                cross_validation,
                "TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256",
                _embedded_reviewed_inventory_sha256(root, repository),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "Nonterminal lifecycle event official provenance"
                ):
                    validate_cross_validation_gate(repository, release)

    def test_lifecycle_evidence_report_hash_is_rederived_from_release_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, release = _fixture(Path(directory))
            repository.manifests["lifecycle_resolutions"].metadata[
                "evidence_report_sha256"
            ] = "d" * 64

            with self.assertRaisesRegex(
                RuntimeError, "Cross-validation evidence is absent"
            ):
                validate_cross_validation_gate(repository, release)

    def test_lifecycle_evidence_report_hash_is_bound_in_report_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(root)
            _rewrite_report(
                root,
                repository,
                lambda report: report["input_hashes"].update(
                    {"lifecycle_evidence_report_sha256": "d" * 64}
                ),
            )

            with self.assertRaisesRegex(
                RuntimeError, "candidate/resolution hashes"
            ):
                validate_cross_validation_gate(repository, release)

    def test_lifecycle_evidence_report_hash_is_bound_in_dataset_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, release = _fixture(Path(directory))
            repository.manifests["cross_validation_reports"].metadata[
                "lifecycle_evidence_report_sha256"
            ] = "d" * 64

            with self.assertRaisesRegex(
                RuntimeError,
                "manifest metadata mismatch for lifecycle_evidence_report_sha256",
            ):
                validate_cross_validation_gate(repository, release)

    def test_pinned_old_lila_overlap_is_recomputed_without_yahoo_self_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(root)
            policy = _policy()
            spec = policy["prices"]["pinned_external_overlaps"][0]
            primary_sessions = cross_validation._xnys_session_strings(
                spec["active_from"], spec["active_to"]
            )
            overlap_sessions = cross_validation._xnys_session_strings(
                spec["overlap_start"], spec["overlap_end"]
            )
            primary_close = {
                session: float(100 + index)
                for index, session in enumerate(primary_sessions)
            }
            external_rows = []
            for index, session in enumerate(
                ["2015-06-22", "2015-06-23", *overlap_sessions]
            ):
                close = (
                    primary_close[session] / 2.0
                    if session in primary_close
                    else float(40 + index)
                )
                external_rows.append(
                    {
                        "Date": session,
                        "Open": close * 0.99,
                        "High": close * 1.01,
                        "Low": close * 0.98,
                        "Close": close,
                        "Volume": 1000,
                        "OpenInt": 0,
                    }
                )
            external_payload = pd.DataFrame(external_rows).to_csv(index=False).encode()
            external_hash = sha256_bytes(external_payload)
            spec["external_source_sha256"] = external_hash
            primary_payload = _chart_response(
                "LILA",
                len(primary_sessions),
                sessions=primary_sessions,
                close_values=[primary_close[value] for value in primary_sessions],
            )
            primary_hash = sha256_bytes(primary_payload)

            for digest, payload, url in (
                (primary_hash, primary_payload, spec["primary_source_url"]),
                (external_hash, external_payload, spec["external_source_url"]),
            ):
                object_path = f"archives/2026-07-17/{digest}.bin.gz"
                (root / object_path).write_bytes(gzip.compress(payload, mtime=0))
                repository.frames["source_archive"] = pd.concat(
                    [
                        repository.frames["source_archive"],
                        pd.DataFrame(
                            [
                                {
                                    "archive_id": digest,
                                    "object_path": object_path,
                                    "source_hash": digest,
                                    "source_url": url,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

            repository.frames["security_master"] = pd.concat(
                [
                    repository.frames["security_master"],
                    pd.DataFrame(
                        [
                            {
                                "security_id": "OLD-LILA",
                                "primary_symbol": "LILA",
                                "active_from": spec["active_from"],
                                "active_to": spec["active_to"],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            repository.frames["symbol_history"] = pd.concat(
                [
                    repository.frames["symbol_history"],
                    pd.DataFrame(
                        [
                            {
                                "security_id": "OLD-LILA",
                                "symbol": "LILA",
                                "effective_from": spec["active_from"],
                                "effective_to": spec["active_to"],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            primary_rows = []
            for session in primary_sessions:
                close = primary_close[session]
                primary_rows.append(
                    {
                        "security_id": "OLD-LILA",
                        "session": session,
                        "open": close - 1.0,
                        "high": close + 1.0,
                        "low": close - 2.0,
                        "close": close,
                        "volume": 1000.0,
                        "currency": "USD",
                        "source": spec["primary_source"],
                        "source_url": spec["primary_source_url"],
                        "source_hash": primary_hash,
                    }
                )
            repository.frames["daily_price_raw"] = pd.concat(
                [
                    repository.frames["daily_price_raw"],
                    pd.DataFrame(primary_rows),
                ],
                ignore_index=True,
            )
            target = {
                "security_id": "OLD-LILA",
                "symbol": "LILA",
                "active_from": spec["active_from"],
                "active_to": spec["active_to"],
            }
            internal = cross_validation._all_internal_target_rows(
                repository.frames["daily_price_raw"], target
            )
            external = cross_validation._parse_pinned_external_payload(
                external_payload, spec
            )
            metrics = cross_validation._recompute_pinned_overlap(
                internal, external, spec
            )
            price_item = {
                "target_id": _target_id(
                    "OLD-LILA", "LILA", spec["active_from"], spec["active_to"]
                ),
                "security_id": "OLD-LILA",
                "symbol": "LILA",
                "provider_symbol": "LILA",
                "identity_active_from": spec["active_from"],
                "identity_active_to": spec["active_to"],
                "validation_basis": "pinned_external_overlap",
                "status": "passed",
                "all_overlap_sessions_compared": True,
                **metrics,
                "external_overlap_ratio": len(overlap_sessions)
                / len(primary_sessions),
                "minimum_return_correlation": spec["minimum_return_correlation"],
                "maximum_p99_scaled_close_error": spec[
                    "maximum_p99_scaled_close_error"
                ],
                "session_coverage_passed": True,
                "scale_stability_passed": True,
                "price_tolerance_passed": True,
                "currency_passed": True,
                "identity_boundary_passed": True,
                "provider_currency": "USD",
                "provider_adjustment_basis": "scale_normalized_close_overlap",
                "adjusted_close_used": False,
                "primary_source": spec["primary_source"],
                "primary_source_url": spec["primary_source_url"],
                "primary_source_sha256": primary_hash,
                "external_source": spec["external_source"],
                "source_url": spec["external_source_url"],
                "source_sha256": external_hash,
                "upstream_provider_disclosed": False,
                "independent_provider_claimed": False,
                "license": spec["license"],
                "license_url": spec["license_url"],
                "independent_internal_price_rows": 0,
                "internal_price_rows": len(primary_sessions),
                "self_source_rows_excluded": 0,
            }

            def add_overlap(report):
                report["policy"] = policy
                report["provider"]["pinned_external_overlap_targets"] = 1
                report["prices"].append(price_item)
                report["summary"]["price_target_count"] += 1
                report["summary"]["price_pass_count"] += 1
                report["summary"]["overlap_session_count"] += len(overlap_sessions)

            _rewrite_report(root, repository, add_overlap)
            policy_hash = canonical_json_sha256(policy)
            summary_updates = {
                "price_target_count": 3,
                "price_pass_count": 3,
                "overlap_session_count": 10 + 12 + len(overlap_sessions),
            }
            repository.frames["cross_validation_reports"].loc[
                0, "policy_sha256"
            ] = policy_hash
            for key, value in summary_updates.items():
                repository.frames["cross_validation_reports"].loc[0, key] = value
                repository.manifests["cross_validation_reports"].metadata[key] = value
            repository.manifests["cross_validation_reports"].metadata[
                "policy_sha256"
            ] = policy_hash

            with patch.dict(
                cross_validation.TRUSTED_PINNED_EXTERNAL_OVERLAPS["LILA"],
                {"external_source_sha256": external_hash},
            ):
                result = validate_cross_validation_gate(repository, release)
            self.assertEqual(result["price_target_count"], 3)
            self.assertEqual(result["price_mismatch_count"], 0)

    def test_unchecked_in_scope_action_cannot_hide_behind_an_applied_event(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, release = _fixture(Path(directory))
            extra = repository.frames["corporate_actions"].iloc[0].copy()
            extra["event_id"] = "unchecked-ticker-change"
            extra["action_type"] = "ticker_change"
            extra["new_security_id"] = "OLD"
            repository.frames["corporate_actions"] = pd.concat(
                [
                    repository.frames["corporate_actions"],
                    pd.DataFrame([extra]),
                ],
                ignore_index=True,
            )

            with self.assertRaisesRegex(
                RuntimeError, "every in-scope lifecycle corporate action"
            ):
                validate_cross_validation_gate(repository, release)

    def test_event_claimed_url_must_equal_archived_url_for_same_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, release = _fixture(Path(directory))
            report_id = str(
                repository.frames["cross_validation_reports"].iloc[0]["report_id"]
            )
            report_row = repository.frames["source_archive"].loc[
                repository.frames["source_archive"]["archive_id"].astype(str).eq(
                    report_id
                )
            ].iloc[0]
            report = json.loads(
                gzip.decompress(
                    (Path(directory) / str(report_row["object_path"])).read_bytes()
                )
            )
            event_hash = str(report["events"][0]["evidence_sha256"])
            event_archive = repository.frames["source_archive"]["archive_id"].astype(
                str
            ).eq(event_hash)
            repository.frames["source_archive"].loc[
                event_archive, "source_url"
            ] = "https://www.sec.gov/Archives/different.txt"

            with self.assertRaisesRegex(RuntimeError, "event URL/hash provenance"):
                validate_cross_validation_gate(repository, release)

    def test_exact_identity_boundary_date_url_and_payload_hash_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, release = _fixture(Path(directory), with_boundary=True)
            result = validate_cross_validation_gate(repository, release)
            self.assertEqual(result["price_target_count"], 2)
            self.assertEqual(result["price_mismatch_count"], 0)

    def test_missing_lifecycle_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, release = _fixture(
                Path(directory), missing_price_identity=True
            )
            with self.assertRaisesRegex(RuntimeError, "every lifecycle identity"):
                validate_cross_validation_gate(repository, release)

    def test_unresolved_price_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, release = _fixture(Path(directory), unresolved=True)
            with self.assertRaisesRegex(RuntimeError, "unresolved or mismatched"):
                validate_cross_validation_gate(repository, release)

    def test_old_stooq_schema_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(root)
            _rewrite_report(
                root,
                repository,
                lambda report: report.update(
                    {
                        "schema": "us_lifecycle_cross_validation/v1",
                        "provider": {"name": "stooq"},
                    }
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "report schema is invalid"):
                validate_cross_validation_gate(repository, release)

    def test_legacy_stooq_boundary_key_cannot_bypass_yahoo_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(root)

            def legacy_alias(report):
                report["prices"][0].pop("provider_sessions_before_identity")
                report["prices"][0]["stooq_sessions_before_identity"] = 100

            _rewrite_report(root, repository, legacy_alias)
            with self.assertRaisesRegex(
                RuntimeError, "provider_sessions_before_identity is not an integer"
            ):
                validate_cross_validation_gate(repository, release)

    def test_unsafe_yahoo_url_with_crumb_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(root)
            _rewrite_report(
                root,
                repository,
                lambda report: report["prices"][0].update(
                    {
                        "source_url": _source_url(
                            "OLD", "2024-01-02", "2024-01-16"
                        )
                        + "&crumb=secret"
                    }
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "unsafe or unpinned"):
                validate_cross_validation_gate(repository, release)

    def test_legacy_range_max_yahoo_url_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(root)

            def use_range_max(report):
                url = (
                    "https://query1.finance.yahoo.com/v8/finance/chart/OLD"
                    "?events=history&includeAdjustedClose=true&interval=1d&range=max"
                )
                report["prices"][0]["source_url"] = url
                report["prices"][0]["expected_source_url"] = url

            _rewrite_report(root, repository, use_range_max)
            with self.assertRaisesRegex(RuntimeError, "unsafe or unpinned"):
                validate_cross_validation_gate(repository, release)

    def test_safe_but_wrong_bounded_period_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(root)

            def shift_start(report):
                url = _source_url("OLD", "2024-01-03", "2024-01-16")
                report["prices"][0]["source_url"] = url
                report["prices"][0]["expected_source_url"] = url
                report["prices"][0]["request_start_date"] = "2024-01-03"
                period1, period2 = _request_periods(
                    "2024-01-03", "2024-01-16"
                )
                report["prices"][0]["request_period1"] = period1
                report["prices"][0]["request_period2"] = period2

            _rewrite_report(root, repository, shift_start)
            with self.assertRaisesRegex(RuntimeError, "exact identity interval"):
                validate_cross_validation_gate(repository, release)

    def test_archived_wrong_yahoo_symbol_is_rejected_by_publication_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(root)
            _rewrite_price_response(
                root,
                repository,
                price_index=0,
                payload=_chart_response("OTHER", 10),
            )
            with self.assertRaisesRegex(RuntimeError, "metadata symbol mismatch"):
                validate_cross_validation_gate(repository, release)

    def test_archived_non_usd_yahoo_response_is_rejected_by_publication_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(root)
            payload = json.loads(_chart_response("OLD", 10))
            payload["chart"]["result"][0]["meta"]["currency"] = "EUR"
            _rewrite_price_response(
                root,
                repository,
                price_index=0,
                payload=json.dumps(payload, separators=(",", ":")).encode(),
            )
            with self.assertRaisesRegex(RuntimeError, "currency must be USD"):
                validate_cross_validation_gate(repository, release)

    def test_archived_non_daily_granularity_is_rejected_by_publication_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(root)
            payload = json.loads(_chart_response("OLD", 10))
            payload["chart"]["result"][0]["meta"]["dataGranularity"] = "3mo"
            _rewrite_price_response(
                root,
                repository,
                price_index=0,
                payload=json.dumps(payload, separators=(",", ":")).encode(),
            )
            with self.assertRaisesRegex(RuntimeError, "dataGranularity"):
                validate_cross_validation_gate(repository, release)

    def test_archived_sparse_daily_inventory_is_independently_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(root)
            sessions = pd.DatetimeIndex(
                cross_validation.xcals.get_calendar("XNYS").sessions_in_range(
                    "2024-01-02", "2024-01-16"
                )
            ).tz_localize(None)[::2]
            payload = _chart_response("OLD", len(sessions), sessions=sessions)
            _rewrite_price_response(
                root,
                repository,
                price_index=0,
                payload=payload,
            )

            def claim_sparse_inventory(report):
                item = report["prices"][0]
                item["provider_history_start"] = sessions[0].date().isoformat()
                item["provider_history_end"] = sessions[-1].date().isoformat()
                item["provider_history_session_count"] = len(sessions)
                item["provider_xnys_session_count"] = len(sessions)
                item["provider_request_xnys_coverage_ratio"] = len(sessions) / 10

            _rewrite_report(root, repository, claim_sparse_inventory)
            with self.assertRaisesRegex(RuntimeError, "exact XNYS inventory"):
                validate_cross_validation_gate(repository, release)

    def test_yahoo_supplement_and_reused_ticker_peer_cannot_be_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, release = _fixture(Path(directory))
            repository.frames["security_master"] = pd.concat(
                [
                    repository.frames["security_master"],
                    pd.DataFrame(
                        [
                            {"security_id": "OLD-LILA", "primary_symbol": "LILA"},
                            {
                                "security_id": "CURRENT-LILA",
                                "primary_symbol": "LILA",
                            },
                        ]
                    ),
                ],
                ignore_index=True,
            )
            repository.frames["daily_price_raw"] = pd.concat(
                [
                    repository.frames["daily_price_raw"],
                    pd.DataFrame(
                        [
                            {
                                "security_id": "OLD-LILA",
                                "source": "identity_repair_supplement",
                                "source_url": _source_url(
                                    "LILA", "2024-01-02", "2024-01-18"
                                ),
                            },
                            {
                                "security_id": "CURRENT-LILA",
                                "source": "eodhd",
                                "source_url": "",
                            },
                        ]
                    ),
                ],
                ignore_index=True,
            )

            with self.assertRaisesRegex(
                RuntimeError, "independent-provider-affected identity"
            ):
                validate_cross_validation_gate(repository, release)

    def test_passed_target_requires_provider_independent_internal_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, release = _fixture(Path(directory))
            old = repository.frames["daily_price_raw"]["security_id"].eq("OLD")
            repository.frames["daily_price_raw"].loc[old, "source"] = (
                "yahoo_chart_json"
            )

            with self.assertRaisesRegex(
                RuntimeError, "no provider-independent internal source"
            ):
                validate_cross_validation_gate(repository, release)

    def test_every_evidence_payload_is_rehashed_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _fixture(root)
            report_id = str(
                repository.frames["cross_validation_reports"].iloc[0]["report_id"]
            )
            evidence_row = repository.frames["source_archive"].loc[
                ~repository.frames["source_archive"]["archive_id"].astype(str).eq(
                    report_id
                )
            ].iloc[0]
            (root / str(evidence_row["object_path"])).write_bytes(
                gzip.compress(b"corrupt evidence", mtime=0)
            )

            with self.assertRaisesRegex(RuntimeError, "payload hash mismatch"):
                validate_cross_validation_gate(repository, release)


if __name__ == "__main__":
    unittest.main()
