from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from supertrend_quant.market_store.lifecycle import LifecycleCandidate
from supertrend_quant.market_store.lifecycle_coverage import lifecycle_candidate_id
from supertrend_quant.market_store.manifest import sha256_bytes
from supertrend_quant.market_store.repository import LocalDatasetRepository


def _load(name: str, filename: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


finalizer = _load(
    "finalize_us_lifecycle_coverage_exact_repair_test",
    "finalize_us_lifecycle_coverage.py",
)
celg = _load("repair_us_celg_bmy_cvr_preservation_test", "repair_us_celg_bmy_cvr.py")
abmd = _load(
    "repair_us_abmd_cvr_lower_bound_preservation_test",
    "repair_us_abmd_cvr_lower_bound.py",
)
frc_para = _load(
    "repair_us_frc_para_lifecycle_preservation_test",
    "repair_us_frc_para_lifecycle.py",
)


def _archive_rows(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "archive_id": row["source_hash"],
                "dataset": row["source"],
                "object_path": f"archives/{row['source_hash']}.gz",
                "content_type": row.get("content_type", "application/json"),
                "effective_date": "2026-07-15",
                "source": row["source"],
                "source_url": row["source_url"],
                "retrieved_at": row["retrieved_at"],
                "source_hash": row["source_hash"],
            }
            for row in rows
        ]
    )


def _celg_fixture():
    retrieved = "2026-07-18T09:00:00Z"
    provider_code = "BMYRT"
    cvr_id = (
        "US:EODHD:"
        + str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"eodhd:US:{provider_code}:symbol:{celg.CVR_SYMBOL}",
            )
        )
    )
    terms = SimpleNamespace(
        source="sec_edgar_filing",
        source_url=finalizer.CELG_EXACT_TERMS_URL,
        retrieved_at="2026-07-18T07:25:37.414638Z",
        source_hash=finalizer.CELG_EXACT_TERMS_SHA256,
    )
    termination = SimpleNamespace(
        source="sec_bmy_2020_10k",
        source_url=finalizer.CELG_EXACT_TERMINATION_URL,
        retrieved_at="2026-07-18T08:39:00Z",
        source_hash=finalizer.CELG_EXACT_TERMINATION_SHA256,
    )
    bundle = SimpleNamespace(security_id=cvr_id)
    actions = celg._official_actions(bundle, terms, termination)
    sessions = finalizer._xnys_sessions(
        finalizer.CELG_EXACT_EFFECTIVE_DATE,
        finalizer.CELG_EXACT_CVR_LAST_SESSION,
    )
    eod_rows = [
        {
            "date": session,
            "open": 2.30,
            "high": 2.40,
            "low": 2.20,
            "close": 2.30,
            "volume": 1_000 + index,
        }
        for index, session in enumerate(sessions)
    ]
    eod_content = json.dumps(
        eod_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    eod_hash = sha256_bytes(eod_content)
    eod_url = (
        f"https://eodhd.com/api/eod/{provider_code}.US?"
        f"from={finalizer.CELG_EXACT_EFFECTIVE_DATE}"
        f"&to={finalizer.CELG_EXACT_CVR_LAST_SESSION}"
    )
    prices = pd.DataFrame(
        [
            {
                "security_id": cvr_id,
                "session": row["date"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "currency": "USD",
                "source": "eodhd_eod",
                "source_url": eod_url,
                "retrieved_at": retrieved,
                "source_hash": eod_hash,
            }
            for row in eod_rows
        ]
    )
    search_url = "https://eodhd.com/api/search/BMYRT?limit=10"
    search_rows = [
        {
            "Code": provider_code,
            "Name": "Bristol-Myers Squibb Contingent Value Rights",
            "Exchange": "US",
            "Country": "USA",
        }
    ]
    search_content = json.dumps(
        search_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    identity = {
        "schema": "celg_bmyrt_identity_resolution/v1",
        "security_id": cvr_id,
        "symbol": "BMYRT",
        "provider_code": provider_code,
        "provider_symbol": f"{provider_code}.US",
        "exchange": "NYSE",
        "active_from": finalizer.CELG_EXACT_EFFECTIVE_DATE,
        "active_to": finalizer.CELG_EXACT_CVR_LAST_SESSION,
        "eodhd_search_url": search_url,
        "eodhd_search_sha256": sha256_bytes(search_content),
        "official_merger_url": finalizer.CELG_EXACT_TERMS_URL,
        "official_merger_sha256": finalizer.CELG_EXACT_TERMS_SHA256,
        "official_termination_url": finalizer.CELG_EXACT_TERMINATION_URL,
        "official_termination_sha256": finalizer.CELG_EXACT_TERMINATION_SHA256,
    }
    identity_content = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    identity_hash = sha256_bytes(identity_content)
    master = pd.DataFrame(
        [
            {
                "security_id": cvr_id,
                "primary_symbol": "BMYRT",
                "provider_symbol": f"{provider_code}.US",
                "name": search_rows[0]["Name"],
                "exchange": "NYSE",
                "active_from": finalizer.CELG_EXACT_EFFECTIVE_DATE,
                "active_to": finalizer.CELG_EXACT_CVR_LAST_SESSION,
                "source": "celg_bmyrt_identity_resolution",
                "source_url": finalizer.CELG_EXACT_TERMS_URL,
                "retrieved_at": retrieved,
                "source_hash": identity_hash,
            }
        ]
    )
    history = pd.DataFrame(
        [
            {
                "security_id": cvr_id,
                "symbol": "BMYRT",
                "exchange": "NYSE",
                "effective_from": finalizer.CELG_EXACT_EFFECTIVE_DATE,
                "effective_to": finalizer.CELG_EXACT_CVR_LAST_SESSION,
                "source": "celg_bmyrt_identity_resolution",
                "source_url": finalizer.CELG_EXACT_TERMS_URL,
                "retrieved_at": retrieved,
                "source_hash": identity_hash,
            }
        ]
    )
    resolution = pd.DataFrame(
        [
            {
                "candidate_id": lifecycle_candidate_id(
                    finalizer.CELG_EXACT_SECURITY_ID,
                    finalizer.CELG_EXACT_LAST_SESSION,
                ),
                "security_id": finalizer.CELG_EXACT_SECURITY_ID,
                "symbol": "CELG",
                "last_price_date": finalizer.CELG_EXACT_LAST_SESSION,
                "resolution": "applied",
                "event_id": finalizer.CELG_EXACT_MERGER_EVENT_ID,
                "exception_code": "",
                "exception_reason": "",
                "reviewed_by": "celg_bmy_cvr_exact_model/v1",
                "reviewed_at": "2026-07-18T08:39:00Z",
                "recheck_after": "",
                "successor_security_id": finalizer.CELG_EXACT_BMY_SECURITY_ID,
                "successor_symbol": "BMY",
                "source_url": finalizer.CELG_EXACT_TERMS_URL,
                "source": "celg_bmy_cvr_exact_repair",
                "retrieved_at": terms.retrieved_at,
                "source_hash": finalizer.CELG_EXACT_TERMS_SHA256,
            }
        ]
    )
    archive = _archive_rows(
        [
            {
                "source": "sec_edgar_filing",
                "source_url": finalizer.CELG_EXACT_TERMS_URL,
                "source_hash": finalizer.CELG_EXACT_TERMS_SHA256,
                "retrieved_at": terms.retrieved_at,
            },
            {
                "source": "sec_bmy_2020_10k",
                "source_url": finalizer.CELG_EXACT_TERMINATION_URL,
                "source_hash": finalizer.CELG_EXACT_TERMINATION_SHA256,
                "retrieved_at": termination.retrieved_at,
            },
        ]
    )
    frames = {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "lifecycle_resolutions": resolution,
        "source_archive": archive,
    }
    payloads = {
        "celg_bmyrt_identity_resolution": (identity_content, {"retrieved_at": retrieved}),
        "eodhd_search": (search_content, {"retrieved_at": retrieved}),
        "eodhd_eod": (eod_content, {"retrieved_at": retrieved}),
        "eodhd_div": (b"[]", {"retrieved_at": retrieved}),
        "eodhd_splits": (b"[]", {"retrieved_at": retrieved}),
    }
    candidate = LifecycleCandidate(
        finalizer.CELG_EXACT_SECURITY_ID,
        "CELG",
        "Celgene Corporation",
        "NASDAQ",
        finalizer.CELG_EXACT_LAST_SESSION,
        finalizer.CELG_EXACT_LAST_SESSION,
    )
    return candidate, frames, payloads


def _payload_side_effect(payloads):
    def read(_repository, _archive, *, source_url, source_hash, source):
        key = (source, source_url)
        return payloads[key] if key in payloads else payloads[source]

    return read


def test_finalizer_preserves_only_the_exact_celg_three_leg_resolution():
    candidate, frames, payloads = _celg_fixture()
    repository = SimpleNamespace(root=Path("/unused"))
    release = SimpleNamespace(warnings=())
    with patch.object(finalizer, "_release_archive_content", return_value=b"official"), patch.object(
        finalizer,
        "_archive_pair_content",
        side_effect=_payload_side_effect(payloads),
    ):
        preserved = finalizer._preserved_exact_repair_resolution(
            candidate, repository, release, frames
        )
        assert preserved["resolution"] == "applied"
        assert preserved["event_id"] == finalizer.CELG_EXACT_MERGER_EVENT_ID
        tampered = {name: frame.copy() for name, frame in frames.items()}
        terminal = tampered["corporate_actions"]["action_type"].eq("delisting")
        tampered["corporate_actions"].loc[terminal, "cash_amount"] = 0.01
        with pytest.raises(RuntimeError, match="economics changed"):
            finalizer._preserved_exact_repair_resolution(
                candidate, repository, release, tampered
            )
        partial = {name: frame.copy() for name, frame in frames.items()}
        partial["daily_price_raw"] = partial["daily_price_raw"].iloc[:-1]
        with pytest.raises(RuntimeError, match="280-session"):
            finalizer._preserved_exact_repair_resolution(
                candidate, repository, release, partial
            )


def _celg_official_exit_fixture():
    terms = SimpleNamespace(
        source="sec_edgar_filing",
        source_url=celg.MERGER_TERMS_URL,
        retrieved_at="2026-07-18T08:12:13.860411Z",
        source_hash=celg.MERGER_TERMS_SHA256,
    )
    termination = SimpleNamespace(
        source="sec_bmy_2020_10k",
        source_url=celg.TERMINATION_URL,
        retrieved_at=celg.TERMINATION_RETRIEVED_AT,
        source_hash=celg.TERMINATION_SHA256,
    )
    secondary = tuple(
        {
            "role": "secondary_ambiguous",
            "reason": "Provider alias lacks the SEC submitter ticker and ISIN binding.",
            "source_url": celg.DELISTED_CATALOG_URL,
            "source_hash": celg.DELISTED_CATALOG_SHA256,
            "row": {
                "Code": code,
                "Country": "USA",
                "Currency": "USD",
                "Exchange": "NYSE",
                "Isin": None,
                "Name": code,
                "Type": "Common Stock",
            },
        }
        for code in celg.SECONDARY_CATALOG_CODES
    )
    selection = celg.FrozenCatalogSelection(
        provider_code=celg.CVR_PROVIDER_CODE,
        source_url=celg.ACTIVE_CATALOG_URL,
        source_hash=celg.ACTIVE_CATALOG_SHA256,
        row=dict(celg.EXACT_CATALOG_ROW),
        secondary_evidence=secondary,
    )
    model = celg._official_exit_model(selection)
    policy = celg._official_exit_policy_artifact(model, termination)
    identity = celg._official_exit_identity_artifact(
        model, terms, termination, policy
    )
    master = pd.DataFrame(
        [
            {
                "security_id": celg.CELG_SECURITY_ID,
                "primary_symbol": "CELG",
                "provider_symbol": "CELG.US",
                "exchange": "NASDAQ",
                "active_from": "2015-01-02",
                "active_to": celg.CELG_LAST_SESSION,
                "source": "sec_edgar_filing",
                "source_url": terms.source_url,
                "retrieved_at": terms.retrieved_at,
                "source_hash": terms.source_hash,
            },
            celg._expected_cvr_master(model, identity),
        ]
    )
    history = pd.DataFrame(
        [
            {
                "security_id": celg.CELG_SECURITY_ID,
                "symbol": "CELG",
                "exchange": "NASDAQ",
                "effective_from": "2015-01-01",
                "effective_to": celg.CELG_LAST_SESSION,
                "source": "sec_edgar_filing",
                "source_url": terms.source_url,
                "retrieved_at": terms.retrieved_at,
                "source_hash": terms.source_hash,
            },
            celg._expected_cvr_history(model, identity),
        ]
    )
    resolution = pd.DataFrame(
        [
            {
                "candidate_id": lifecycle_candidate_id(
                    celg.CELG_SECURITY_ID, celg.CELG_LAST_SESSION
                ),
                "security_id": celg.CELG_SECURITY_ID,
                "symbol": "CELG",
                "last_price_date": celg.CELG_LAST_SESSION,
                "resolution": "applied",
                "event_id": celg.CELG_STOCK_MERGER_EVENT_ID,
                "exception_code": "",
                "exception_reason": "",
                "reviewed_by": celg.OFFICIAL_EXIT_REVIEWED_BY,
                "reviewed_at": celg.REVIEWED_AT,
                "recheck_after": "",
                "successor_security_id": celg.BMY_SECURITY_ID,
                "successor_symbol": "BMY",
                "source_url": terms.source_url,
                "source": "celg_bmy_cvr_official_exit_mark_repair",
                "retrieved_at": terms.retrieved_at,
                "source_hash": terms.source_hash,
            }
        ]
    )
    factor_source = "official-price+official-actions"
    factors = pd.DataFrame(
        [
            {
                "security_id": model.security_id,
                "session": celg.MERGER_SESSION,
                "split_factor": 1.0,
                "total_return_factor": 1.0,
                "source_version": factor_source,
                "calculated_at": celg.REVIEWED_AT,
                "source": "derived",
                "retrieved_at": celg.REVIEWED_AT,
                "source_hash": factor_source,
            }
        ]
    )
    archive = _archive_rows(
        [
            {
                "source": "sec_edgar_filing",
                "source_url": terms.source_url,
                "source_hash": terms.source_hash,
                "retrieved_at": terms.retrieved_at,
            },
            {
                "source": "sec_bmy_2020_10k",
                "source_url": termination.source_url,
                "source_hash": termination.source_hash,
                "retrieved_at": termination.retrieved_at,
            },
            {
                "source": policy.source,
                "source_url": policy.source_url,
                "source_hash": policy.source_hash,
                "retrieved_at": policy.retrieved_at,
            },
            {
                "source": identity.source,
                "source_url": identity.source_url,
                "source_hash": identity.source_hash,
                "retrieved_at": identity.retrieved_at,
            },
            {
                "source": "eodhd_exchange_symbols",
                "source_url": celg.ACTIVE_CATALOG_URL,
                "source_hash": celg.ACTIVE_CATALOG_SHA256,
                "retrieved_at": "2026-07-16T15:56:01.033938Z",
            },
            {
                "source": "eodhd_exchange_symbols",
                "source_url": celg.DELISTED_CATALOG_URL,
                "source_hash": celg.DELISTED_CATALOG_SHA256,
                "retrieved_at": "2026-07-16T15:56:01.033938Z",
            },
        ]
    )
    frames = {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": celg._official_exit_price(model, policy),
        "corporate_actions": celg._official_exit_actions(
            model, terms, termination
        ),
        "adjustment_factors": factors,
        "lifecycle_resolutions": resolution,
        "source_archive": archive,
    }
    payloads = {
        policy.source: (policy.content, {"retrieved_at": policy.retrieved_at}),
        identity.source: (
            identity.content,
            {"retrieved_at": identity.retrieved_at},
        ),
        (
            "eodhd_exchange_symbols",
            celg.ACTIVE_CATALOG_URL,
        ): (
            json.dumps([dict(celg.EXACT_CATALOG_ROW)]).encode(),
            {"retrieved_at": "2026-07-16T15:56:01.033938Z"},
        ),
        (
            "eodhd_exchange_symbols",
            celg.DELISTED_CATALOG_URL,
        ): (
            json.dumps(
                [
                    {
                        "Code": code,
                        "Country": "USA",
                        "Currency": "USD",
                        "Exchange": "NYSE",
                        "Isin": None,
                        "Name": code,
                        "Type": "Common Stock",
                    }
                    for code in celg.SECONDARY_CATALOG_CODES
                ]
            ).encode(),
            {"retrieved_at": "2026-07-16T15:56:01.033938Z"},
        ),
    }
    candidate = LifecycleCandidate(
        celg.CELG_SECURITY_ID,
        "CELG",
        "Celgene Corporation",
        "NASDAQ",
        celg.CELG_LAST_SESSION,
        celg.CELG_LAST_SESSION,
    )
    return candidate, frames, payloads


def test_finalizer_preserves_exact_official_exit_mark_and_rejects_drift():
    candidate, frames, payloads = _celg_official_exit_fixture()
    factor_metadata = {
        "source_daily_price_version": "official-price",
        "source_corporate_actions_version": "official-actions",
        "source_version": "official-price+official-actions",
    }
    repository = SimpleNamespace(
        root=Path("/unused"),
        manifest_for_version=lambda dataset, version: SimpleNamespace(
            dataset=dataset,
            version=version,
            metadata=factor_metadata,
        ),
    )
    release = SimpleNamespace(
        warnings=(celg.OFFICIAL_EXIT_WARNING,),
        dataset_versions={
            "daily_price_raw": "official-price",
            "corporate_actions": "official-actions",
            "adjustment_factors": "official-factors",
        },
    )
    with patch.object(
        finalizer, "_release_archive_content", return_value=b"official"
    ), patch.object(
        finalizer,
        "_archive_pair_content",
        side_effect=_payload_side_effect(payloads),
    ):
        preserved = finalizer._preserved_exact_repair_resolution(
            candidate, repository, release, frames
        )
        assert preserved["reviewed_by"] == celg.OFFICIAL_EXIT_REVIEWED_BY
        no_warning = SimpleNamespace(
            warnings=(), dataset_versions=release.dataset_versions
        )
        with pytest.raises(RuntimeError, match="release warning"):
            finalizer._preserved_exact_repair_resolution(
                candidate, repository, no_warning, frames
            )
        drift = {name: frame.copy() for name, frame in frames.items()}
        drift["daily_price_raw"].loc[:, "close"] = 2.31
        with pytest.raises(RuntimeError, match="mark row changed"):
            finalizer._preserved_exact_repair_resolution(
                candidate, repository, release, drift
            )
        stale_repository = SimpleNamespace(
            root=Path("/unused"),
            manifest_for_version=lambda _dataset, _version: SimpleNamespace(
                metadata={
                    **factor_metadata,
                    "source_corporate_actions_version": "stale-actions",
                }
            ),
        )
        with pytest.raises(RuntimeError, match="manifest lineage is stale"):
            finalizer._preserved_exact_repair_resolution(
                candidate, stale_repository, release, frames
            )


def test_finalizer_closes_exact_bmyrt_official_exit_mark_candidate():
    _candidate, frames, payloads = _celg_official_exit_fixture()
    candidate = LifecycleCandidate(
        finalizer.CELG_OFFICIAL_EXIT_SECURITY_ID,
        "BMYRT",
        "Bristol-Myers Squibb Company Ce",
        "NYSE",
        finalizer.CELG_EXACT_EFFECTIVE_DATE,
        finalizer.CELG_EXACT_CVR_LAST_SESSION,
    )
    factor_metadata = {
        "source_daily_price_version": "official-price",
        "source_corporate_actions_version": "official-actions",
        "source_version": "official-price+official-actions",
    }
    repository = SimpleNamespace(
        root=Path("/unused"),
        manifest_for_version=lambda dataset, version: SimpleNamespace(
            dataset=dataset,
            version=version,
            metadata=factor_metadata,
        ),
    )
    release = SimpleNamespace(
        warnings=(celg.OFFICIAL_EXIT_WARNING,),
        dataset_versions={
            "daily_price_raw": "official-price",
            "corporate_actions": "official-actions",
            "adjustment_factors": "official-factors",
        },
    )
    with patch.object(
        finalizer, "_release_archive_content", return_value=b"official"
    ), patch.object(
        finalizer,
        "_archive_pair_content",
        side_effect=_payload_side_effect(payloads),
    ):
        preserved = finalizer._preserved_exact_repair_resolution(
            candidate, repository, release, frames
        )
        assert preserved == {
            "candidate_id": lifecycle_candidate_id(
                finalizer.CELG_OFFICIAL_EXIT_SECURITY_ID,
                finalizer.CELG_EXACT_EFFECTIVE_DATE,
            ),
            "security_id": finalizer.CELG_OFFICIAL_EXIT_SECURITY_ID,
            "symbol": "BMYRT",
            "last_price_date": finalizer.CELG_EXACT_EFFECTIVE_DATE,
            "resolution": "applied",
            "event_id": finalizer.CELG_OFFICIAL_EXIT_EVENT_ID,
            "exception_code": "",
            "exception_reason": "",
            "reviewed_by": "celg_bmy_cvr_official_exit_mark/v1",
            "reviewed_at": "2026-07-18T08:39:00Z",
            "recheck_after": "",
            "successor_security_id": "",
            "successor_symbol": "",
            "source_url": finalizer.CELG_EXACT_TERMINATION_URL,
            "source": "sec_bmy_2020_10k",
            "retrieved_at": celg.TERMINATION_RETRIEVED_AT,
            "source_hash": finalizer.CELG_EXACT_TERMINATION_SHA256,
        }
        repaired = {name: frame.copy() for name, frame in frames.items()}
        repaired["lifecycle_resolutions"] = pd.concat(
            [
                repaired["lifecycle_resolutions"],
                pd.DataFrame([preserved]),
            ],
            ignore_index=True,
        )
        assert finalizer._preserved_exact_repair_resolution(
            candidate, repository, release, repaired
        ) == preserved
        conflicting = {name: frame.copy() for name, frame in repaired.items()}
        child = conflicting["lifecycle_resolutions"]["security_id"].eq(
            finalizer.CELG_OFFICIAL_EXIT_SECURITY_ID
        )
        conflicting["lifecycle_resolutions"].loc[child, "event_id"] = "tampered"
        with pytest.raises(RuntimeError, match="resolution field changed"):
            finalizer._preserved_exact_repair_resolution(
                candidate, repository, release, conflicting
            )
        drift = {name: frame.copy() for name, frame in frames.items()}
        drift["corporate_actions"].loc[
            drift["corporate_actions"]["event_id"].eq(
                finalizer.CELG_OFFICIAL_RESIDUAL_EVENT_ID
            ),
            "cash_amount",
        ] = 0.01
        with pytest.raises(RuntimeError, match="economics changed"):
            finalizer._preserved_exact_repair_resolution(
                candidate, repository, release, drift
            )


def test_finalizer_accepts_exact_early_terminal_history_factor_lineage():
    candidate, frames, payloads = _celg_official_exit_fixture()
    price_version = "early-price"
    action_version = "early-actions"
    factor_version = "early-factors"
    lineage = (
        finalizer.EARLY_TERMINAL_HISTORY_FACTOR_PREFIX
        + finalizer.sha256_bytes(
            f"{price_version}|{action_version}".encode()
        )
    )
    frames["adjustment_factors"].loc[:, "source_version"] = lineage
    frames["adjustment_factors"].loc[:, "source_hash"] = lineage
    metadata = {
        "operation": finalizer.EARLY_TERMINAL_HISTORY_FACTOR_OPERATION,
        "source_daily_price_version": price_version,
        "source_corporate_actions_version": action_version,
        "source_version": lineage,
    }
    repository = SimpleNamespace(
        root=Path("/unused"),
        manifest_for_version=lambda dataset, version: SimpleNamespace(
            dataset=dataset,
            version=version,
            metadata=metadata,
        ),
    )
    release = SimpleNamespace(
        warnings=(celg.OFFICIAL_EXIT_WARNING,),
        dataset_versions={
            "daily_price_raw": price_version,
            "corporate_actions": action_version,
            "adjustment_factors": factor_version,
        },
    )
    with patch.object(
        finalizer, "_release_archive_content", return_value=b"official"
    ), patch.object(
        finalizer,
        "_archive_pair_content",
        side_effect=_payload_side_effect(payloads),
    ):
        preserved = finalizer._preserved_exact_repair_resolution(
            candidate, repository, release, frames
        )
        assert preserved["event_id"] == celg.CELG_STOCK_MERGER_EVENT_ID
        metadata["source_version"] = f"{lineage}-tampered"
        with pytest.raises(RuntimeError, match="manifest lineage is stale"):
            finalizer._preserved_exact_repair_resolution(
                candidate, repository, release, frames
            )


def _abmd_fixture():
    retrieved = "2026-07-18T07:26:22.110643Z"
    action = abmd._expected_action(retrieved_at=retrieved)
    master = pd.DataFrame(
        [
            {
                "security_id": finalizer.ABMD_EXACT_SECURITY_ID,
                "primary_symbol": "ABMD",
                "active_to": finalizer.ABMD_EXACT_LAST_SESSION,
            }
        ]
    )
    history = pd.DataFrame(
        [
            {
                "security_id": finalizer.ABMD_EXACT_SECURITY_ID,
                "symbol": "ABMD",
                "effective_to": finalizer.ABMD_EXACT_LAST_SESSION,
            }
        ]
    )
    prices = pd.DataFrame(
        [
            {
                "security_id": finalizer.ABMD_EXACT_SECURITY_ID,
                "session": finalizer.ABMD_EXACT_LAST_SESSION,
                "close": finalizer.ABMD_EXACT_LAST_CLOSE,
            }
        ]
    )
    resolution = pd.DataFrame(
        [
            {
                "candidate_id": lifecycle_candidate_id(
                    finalizer.ABMD_EXACT_SECURITY_ID,
                    finalizer.ABMD_EXACT_LAST_SESSION,
                ),
                "security_id": finalizer.ABMD_EXACT_SECURITY_ID,
                "symbol": "ABMD",
                "last_price_date": finalizer.ABMD_EXACT_LAST_SESSION,
                "resolution": "applied",
                "event_id": finalizer.ABMD_EXACT_EVENT_ID,
                "exception_code": "",
                "exception_reason": "",
                "reviewed_by": "abmd_cvr_lower_bound_policy_v1",
                "reviewed_at": "2026-07-18T00:00:00Z",
                "recheck_after": "",
                "successor_security_id": "",
                "successor_symbol": "",
                "source_url": finalizer.ABMD_EXACT_TERMS_URL,
                "source": "abmd_cvr_lower_bound_repair",
                "retrieved_at": retrieved,
                "source_hash": finalizer.ABMD_EXACT_TERMS_SHA256,
            }
        ]
    )
    archive = _archive_rows(
        [
            {
                "source": "sec_edgar_filing",
                "source_url": finalizer.ABMD_EXACT_TERMS_URL,
                "source_hash": finalizer.ABMD_EXACT_TERMS_SHA256,
                "retrieved_at": retrieved,
            },
            {
                "source": "jnj_2025_annual_report",
                "source_url": finalizer.ABMD_EXACT_VALUATION_URL,
                "source_hash": finalizer.ABMD_EXACT_VALUATION_SHA256,
                "retrieved_at": "2026-07-18T08:15:00Z",
            },
        ]
    )
    frames = {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": pd.DataFrame([action]),
        "lifecycle_resolutions": resolution,
        "source_archive": archive,
    }
    candidate = LifecycleCandidate(
        finalizer.ABMD_EXACT_SECURITY_ID,
        "ABMD",
        "ABIOMED",
        "NASDAQ",
        finalizer.ABMD_EXACT_LAST_SESSION,
        finalizer.ABMD_EXACT_LAST_SESSION,
    )
    return candidate, frames


def test_finalizer_preserves_abmd_only_with_exact_policy_metadata_and_warning():
    candidate, frames = _abmd_fixture()
    repository = SimpleNamespace(root=Path("/unused"))
    release = SimpleNamespace(warnings=(finalizer.ABMD_EXACT_WARNING,))
    with patch.object(finalizer, "_release_archive_content", return_value=b"official"):
        preserved = finalizer._preserved_exact_repair_resolution(
            candidate, repository, release, frames
        )
        assert preserved["resolution"] == "applied"
        assert preserved["event_id"] == finalizer.ABMD_EXACT_EVENT_ID
        missing_warning = SimpleNamespace(warnings=())
        with pytest.raises(RuntimeError, match="release warning is missing"):
            finalizer._preserved_exact_repair_resolution(
                candidate, repository, missing_warning, frames
            )
        tampered = {name: frame.copy() for name, frame in frames.items()}
        metadata = json.loads(tampered["corporate_actions"].iloc[0]["metadata"])
        metadata["cvr"]["mark_per_right"] = 1.0
        tampered["corporate_actions"].at[0, "metadata"] = json.dumps(metadata)
        with pytest.raises(RuntimeError, match="metadata hash changed"):
            finalizer._preserved_exact_repair_resolution(
                candidate, repository, release, tampered
            )
        reverted = {name: frame.copy() for name, frame in frames.items()}
        reverted["lifecycle_resolutions"].at[0, "resolution"] = "exception"
        reverted["lifecycle_resolutions"].at[0, "event_id"] = ""
        with pytest.raises(RuntimeError, match="resolution field changed"):
            finalizer._preserved_exact_repair_resolution(
                candidate, repository, release, reverted
            )


def test_old_exception_path_remains_when_no_exact_repair_structure_exists():
    candidate, frames = _abmd_fixture()
    frames["corporate_actions"] = frames["corporate_actions"].iloc[:0]
    frames["lifecycle_resolutions"].at[0, "resolution"] = "exception"
    frames["lifecycle_resolutions"].at[0, "event_id"] = ""
    assert (
        finalizer._preserved_exact_repair_resolution(
            candidate,
            SimpleNamespace(root=Path("/unused")),
            SimpleNamespace(warnings=()),
            frames,
        )
        is None
    )


def _frc_para_fixture():
    raw_retrieved_at = "2026-07-18T09:12:12.828758Z"
    sessions = finalizer._xnys_sessions(
        finalizer.FRC_EXACT_TRANSITION,
        finalizer.FRC_EXACT_PRICE_END,
    )
    assert len(sessions) == finalizer.FRC_EXACT_RAW_EOD_ROWS
    raw_rows = []
    for session in sessions:
        row = {
            "date": session,
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 1000,
        }
        if session == "2024-12-30":
            row.update(
                {
                    "open": 0.003,
                    "high": 0.006,
                    "low": 0.0,
                    "close": 0.004,
                    "volume": 629864,
                }
            )
        raw_rows.append(row)
    raw_content = json.dumps(raw_rows, separators=(",", ":")).encode()

    prices = [
        {
            "security_id": finalizer.FRC_EXACT_SECURITY_ID,
            "session": "2015-01-02",
            "open": 40.0,
            "high": 40.0,
            "low": 40.0,
            "close": 40.0,
            "volume": 1000,
            "currency": "USD",
            "source": "eodhd_eod",
            "source_url": "https://eodhd.com/api/eod/FRC.US",
            "retrieved_at": "2026-07-16T00:00:00Z",
            "source_hash": "a" * 64,
        }
    ]
    for row in raw_rows:
        prices.append(
            {
                "security_id": finalizer.FRC_EXACT_SECURITY_ID,
                "session": row["date"],
                "open": row["open"],
                "high": row["high"],
                "low": 0.003 if row["date"] == "2024-12-30" else row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "currency": "USD",
                "source": "eodhd_eod",
                "source_url": finalizer.FRC_EXACT_RAW_EOD_URL,
                "retrieved_at": raw_retrieved_at,
                "source_hash": finalizer.FRC_EXACT_RAW_EOD_SHA256,
            }
        )
    prices.extend(
        [
            {
                "security_id": finalizer.PARA_EXACT_SECURITY_ID,
                "session": finalizer.PARA_EXACT_LAST,
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 1000,
                "currency": "USD",
                "source": "eodhd_eod",
                "source_url": "",
                "retrieved_at": "2026-07-16T00:00:00Z",
                "source_hash": "b" * 64,
            },
            {
                "security_id": finalizer.PSKY_EXACT_SECURITY_ID,
                "session": finalizer.PARA_EXACT_TRANSITION,
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 1000,
                "currency": "USD",
                "source": "eodhd_eod",
                "source_url": "",
                "retrieved_at": "2026-07-16T00:00:00Z",
                "source_hash": "c" * 64,
            },
        ]
    )
    price_frame = pd.DataFrame(prices)
    factors = pd.DataFrame(
        [
            {"security_id": row["security_id"], "session": row["session"]}
            for row in prices
        ]
    )
    master = pd.DataFrame(
        [
            {
                "security_id": finalizer.FRC_EXACT_SECURITY_ID,
                "primary_symbol": "FRCB",
                "provider_symbol": "FRCB.US",
                "action_provider_symbol": "FRCB.US",
                "exchange": "PINK",
                "active_from": "2015-01-02",
                "active_to": "",
                "source": "occ_reviewed_memo_extraction",
                "source_url": finalizer.FRC_EXACT_OCC_URL,
                "retrieved_at": "2026-07-18T00:00:00Z",
                "source_hash": finalizer.FRC_EXACT_OCC_SHA256,
            },
            {
                "security_id": finalizer.PARA_EXACT_SECURITY_ID,
                "primary_symbol": "PARA",
                "provider_symbol": "PARA.US",
                "action_provider_symbol": "PARA.US",
                "exchange": "NASDAQ",
                "active_from": "2015-01-02",
                "active_to": finalizer.PARA_EXACT_TRANSITION,
                "source": "eodhd_exchange_symbols",
                "source_url": "",
                "retrieved_at": "2026-07-16T00:00:00Z",
                "source_hash": "d" * 64,
            },
            {
                "security_id": finalizer.PSKY_EXACT_SECURITY_ID,
                "primary_symbol": "PSKY",
                "provider_symbol": "PSKY.US",
                "action_provider_symbol": "PSKY.US",
                "exchange": "NASDAQ",
                "active_from": finalizer.PARA_EXACT_TRANSITION,
                "active_to": "",
                "source": "sec_edgar_filing",
                "source_url": finalizer.PARA_EXACT_SEC_URL,
                "retrieved_at": finalizer.PARA_EXACT_RETRIEVED_AT,
                "source_hash": finalizer.PARA_EXACT_SEC_SHA256,
            },
        ]
    )
    occ = frc_para._occ_artifact()
    history = frc_para._history_rows(occ)
    actions = frc_para._official_action_rows(occ)
    frc_action = actions["event_id"].astype(str).eq(finalizer.FRC_EXACT_EVENT_ID)
    actions.loc[frc_action, "source_kind"] = "official_crosscheck"
    actions.loc[frc_action, "source"] = "occ_information_memo"
    actions.loc[frc_action, "retrieved_at"] = finalizer.FRC_EXACT_OCC_PDF_RETRIEVED_AT
    actions.loc[frc_action, "source_hash"] = finalizer.FRC_EXACT_OCC_PDF_SHA256
    actions.loc[frc_action, "metadata"] = json.dumps(
        {
            "cusip": "33616C100",
            "evidence_binding_schema": "occ_information_memo_binding/v1",
            "memo_number": "52352",
            "occ_disclaimer_role": "unofficial_corporate_event_summary",
            "occ_legacy_reviewed_extraction_sha256": finalizer.FRC_EXACT_OCC_SHA256,
            "occ_official_origin_confirmed": True,
            "occ_raw_pdf_bytes": finalizer.FRC_EXACT_OCC_PDF_BYTES,
            "occ_raw_pdf_extracted_text_sha256": (
                "2ac7ffa0dc86d90b035c5f3e48ad43a8f6d88a54c14a3650bf386361c49e41af"
            ),
            "occ_raw_pdf_object_path": finalizer.FRC_EXACT_OCC_PDF_OBJECT_PATH,
            "occ_raw_pdf_page_count": 2,
            "occ_raw_pdf_reviewed_at": finalizer.FRC_EXACT_OCC_PDF_RETRIEVED_AT,
            "occ_raw_pdf_reviewed_by": "codex-independent-pdf-review",
            "occ_raw_pdf_sha256": finalizer.FRC_EXACT_OCC_PDF_SHA256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    resolutions = pd.DataFrame(
        [
            {
                "candidate_id": lifecycle_candidate_id(
                    finalizer.FRC_EXACT_SECURITY_ID,
                    finalizer.FRC_EXACT_OLD_LAST,
                ),
                "security_id": finalizer.FRC_EXACT_SECURITY_ID,
                "symbol": "FRC",
                "last_price_date": finalizer.FRC_EXACT_OLD_LAST,
                "resolution": "applied",
                "event_id": finalizer.FRC_EXACT_EVENT_ID,
                "exception_code": "",
                "exception_reason": "",
                "reviewed_by": "us_frc_para_lifecycle_repair_v1",
                "reviewed_at": "2026-07-18T00:00:00Z",
                "recheck_after": "",
                "successor_security_id": finalizer.FRC_EXACT_SECURITY_ID,
                "successor_symbol": "FRCB",
                "source_url": finalizer.FRC_EXACT_OCC_URL,
                "source": "occ_reviewed_memo_extraction",
                "retrieved_at": "2026-07-18T00:00:00Z",
                "source_hash": finalizer.FRC_EXACT_OCC_SHA256,
            },
            {
                "candidate_id": lifecycle_candidate_id(
                    finalizer.PARA_EXACT_SECURITY_ID,
                    finalizer.PARA_EXACT_LAST,
                ),
                "security_id": finalizer.PARA_EXACT_SECURITY_ID,
                "symbol": "PARA",
                "last_price_date": finalizer.PARA_EXACT_LAST,
                "resolution": "applied",
                "event_id": finalizer.PARA_EXACT_EVENT_ID,
                "exception_code": "",
                "exception_reason": "",
                "reviewed_by": "us_frc_para_lifecycle_repair_v1",
                "reviewed_at": "2026-07-18T00:00:00Z",
                "recheck_after": "",
                "successor_security_id": finalizer.PSKY_EXACT_SECURITY_ID,
                "successor_symbol": "PSKY",
                "source_url": finalizer.PARA_EXACT_SEC_URL,
                "source": "sec_edgar_filing",
                "retrieved_at": finalizer.PARA_EXACT_RETRIEVED_AT,
                "source_hash": finalizer.PARA_EXACT_SEC_SHA256,
            },
        ]
    )
    archive_specs = [
        {
            "source": "occ_reviewed_memo_extraction",
            "source_url": finalizer.FRC_EXACT_OCC_URL,
            "source_hash": finalizer.FRC_EXACT_OCC_SHA256,
            "retrieved_at": "2026-07-18T00:00:00Z",
        },
        {
            "source": "occ_information_memo",
            "source_url": finalizer.FRC_EXACT_OCC_URL,
            "source_hash": finalizer.FRC_EXACT_OCC_PDF_SHA256,
            "retrieved_at": finalizer.FRC_EXACT_OCC_PDF_RETRIEVED_AT,
        },
        {
            "source": "fdic_failed_bank_receivership",
            "source_url": finalizer.FRC_EXACT_FDIC_URL,
            "source_hash": finalizer.FRC_EXACT_FDIC_SHA256,
            "retrieved_at": "2026-07-18T00:39:18Z",
        },
        {
            "source": "eodhd_eod",
            "source_url": finalizer.FRC_EXACT_RAW_EOD_URL,
            "source_hash": finalizer.FRC_EXACT_RAW_EOD_SHA256,
            "retrieved_at": raw_retrieved_at,
        },
        {
            "source": "frcb_reviewed_ohlcv_envelope_correction",
            "source_url": finalizer.FRC_EXACT_RAW_EOD_URL,
            "source_hash": finalizer.FRC_EXACT_CORRECTION_SHA256,
            "retrieved_at": raw_retrieved_at,
        },
        {
            "source": "sec_edgar_filing",
            "source_url": finalizer.PARA_EXACT_SEC_URL,
            "source_hash": finalizer.PARA_EXACT_SEC_SHA256,
            "retrieved_at": finalizer.PARA_EXACT_RETRIEVED_AT,
        },
    ]
    frames = {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": price_frame,
        "corporate_actions": actions,
        "adjustment_factors": factors,
        "lifecycle_resolutions": resolutions,
        "source_archive": _archive_rows(archive_specs),
    }
    legacy_archive = frames["source_archive"]["source_hash"].astype(str).eq(
        finalizer.FRC_EXACT_OCC_SHA256
    )
    frames["source_archive"].loc[
        legacy_archive, "archive_id"
    ] = finalizer.FRC_EXACT_OCC_LEGACY_ARCHIVE_ID
    frames["source_archive"].loc[
        legacy_archive, "object_path"
    ] = f"archives/2026-07-15/{finalizer.FRC_EXACT_OCC_SHA256}.json.gz"
    raw_occ_archive = frames["source_archive"]["source_hash"].astype(str).eq(
        finalizer.FRC_EXACT_OCC_PDF_SHA256
    )
    frames["source_archive"].loc[raw_occ_archive, "dataset"] = "occ_information_memo"
    frames["source_archive"].loc[raw_occ_archive, "content_type"] = "application/pdf"
    frames["source_archive"].loc[
        raw_occ_archive, "object_path"
    ] = finalizer.FRC_EXACT_OCC_PDF_OBJECT_PATH
    pdf_prefix = b"%PDF-1.5\n"
    pdf_suffix = b"\n%%EOF\n"
    occ_pdf_content = (
        pdf_prefix
        + b" "
        * (finalizer.FRC_EXACT_OCC_PDF_BYTES - len(pdf_prefix) - len(pdf_suffix))
        + pdf_suffix
    )
    payloads = {
        "occ_reviewed_memo_extraction": (
            (
                json.dumps(
                    finalizer.FRC_EXACT_OCC_EXTRACTION,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode(),
            frames["source_archive"].loc[legacy_archive].iloc[0].to_dict(),
        ),
        "occ_information_memo": (
            occ_pdf_content,
            frames["source_archive"].loc[raw_occ_archive].iloc[0].to_dict(),
        ),
        "fdic_failed_bank_receivership": (
            b"official FDIC evidence",
            {"retrieved_at": "2026-07-18T00:39:18Z"},
        ),
        "eodhd_eod": (
            raw_content,
            {"retrieved_at": raw_retrieved_at},
        ),
        "frcb_reviewed_ohlcv_envelope_correction": (
            json.dumps(finalizer.FRC_EXACT_CORRECTION_METADATA).encode(),
            {"retrieved_at": raw_retrieved_at},
        ),
        "sec_edgar_filing": (
            b"official SEC evidence",
            {"retrieved_at": finalizer.PARA_EXACT_RETRIEVED_AT},
        ),
    }
    candidate = LifecycleCandidate(
        finalizer.PARA_EXACT_SECURITY_ID,
        "PARA",
        "Paramount Global Class B",
        "NASDAQ",
        finalizer.PARA_EXACT_LAST,
        finalizer.PARA_EXACT_TRANSITION,
    )
    release = SimpleNamespace(
        completed_session=finalizer.FRC_EXACT_PRICE_END,
        warnings=(finalizer.FRC_EXACT_WARNING,),
    )
    return frames, payloads, candidate, release


def test_finalizer_preserves_exact_frc_para_contract_and_drops_stale_frc_resolution():
    frames, payloads, candidate, release = _frc_para_fixture()
    repository = SimpleNamespace(root=Path("/unused"))
    with patch.object(
        finalizer,
        "_archive_pair_content",
        side_effect=_payload_side_effect(payloads),
    ):
        preserved, markers = finalizer._preserved_exact_frc_para_repairs(
            repository, release, frames, [candidate]
        )
        assert markers == ("FRC/FRCB", "PARA/PSKY")
        assert set(preserved) == {
            finalizer._key(
                finalizer.PARA_EXACT_SECURITY_ID, finalizer.PARA_EXACT_LAST
            )
        }
        assert preserved[next(iter(preserved))]["event_id"] == finalizer.PARA_EXACT_EVENT_ID

        post_finalizer = {name: frame.copy() for name, frame in frames.items()}
        post_finalizer["lifecycle_resolutions"] = post_finalizer[
            "lifecycle_resolutions"
        ].loc[
            ~post_finalizer["lifecycle_resolutions"]["security_id"]
            .astype(str)
            .eq(finalizer.FRC_EXACT_SECURITY_ID)
        ]
        preserved_again, _ = finalizer._preserved_exact_frc_para_repairs(
            repository, release, post_finalizer, [candidate]
        )
        assert len(preserved_again) == 1


def test_finalizer_rejects_any_frc_para_exact_repair_drift():
    frames, payloads, candidate, release = _frc_para_fixture()
    repository = SimpleNamespace(root=Path("/unused"))
    with patch.object(
        finalizer,
        "_archive_pair_content",
        side_effect=_payload_side_effect(payloads),
    ):
        bad_price = {name: frame.copy() for name, frame in frames.items()}
        mask = (
            bad_price["daily_price_raw"]["security_id"]
            .astype(str)
            .eq(finalizer.FRC_EXACT_SECURITY_ID)
            & bad_price["daily_price_raw"]["session"].astype(str).eq("2024-12-30")
        )
        bad_price["daily_price_raw"].loc[mask, "low"] = 0.0
        with pytest.raises(RuntimeError, match="raw plus one correction"):
            finalizer._preserved_exact_frc_para_repairs(
                repository, release, bad_price, [candidate]
            )


def _ntco_fixture():
    sessions = finalizer._xnys_sessions(
        finalizer.NTCO_EXACT_TICKER_DATE,
        finalizer.NTCO_EXACT_LAST_SESSION,
    )
    assert len(sessions) == finalizer.NTCO_EXACT_EOD_ROWS
    raw_rows = [
        {
            "date": session,
            "open": 6.0 + index / 1_000,
            "high": 6.2 + index / 1_000,
            "low": 5.8 + index / 1_000,
            "close": 6.1 + index / 1_000,
            "volume": 10_000 + index,
        }
        for index, session in enumerate(sessions)
    ]
    eod_content = json.dumps(raw_rows, separators=(",", ":")).encode()
    prices = pd.DataFrame(
        [
            {
                "security_id": finalizer.NTCO_EXACT_SECURITY_ID,
                "session": row["date"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "currency": "USD",
                "source": "eodhd_eod",
                "source_url": finalizer.NTCO_EXACT_EOD_URL,
                "retrieved_at": finalizer.NTCO_EXACT_EOD_RETRIEVED_AT,
                "source_hash": finalizer.NTCO_EXACT_EOD_RAW_SHA256,
            }
            for row in raw_rows
        ]
    )
    ticker_metadata = {
        "cboe_source_url": finalizer.NTCO_EXACT_CBOE_URL,
        "occ_source_url": finalizer.NTCO_EXACT_OCC_URL,
        "official_destination_market": "Other-OTC",
        "canonical_exchange": "OTC",
        "cusip": "63884N108",
        "deliverable": "100 American Depositary Shares",
    }
    terminal_metadata = {
        "mandatory_exchange": True,
        "gross_rate_per_ads": "5.043659",
        "cancellation_fee_per_ads": "0",
        "net_rate_per_ads": "5.043659",
        "ads_to_underlying_ratio": "1:2",
    }
    actions = pd.DataFrame(
        [
            {
                "event_id": finalizer.NTCO_EXACT_TICKER_EVENT_ID,
                "security_id": finalizer.NTCO_EXACT_SECURITY_ID,
                "action_type": "ticker_change",
                "effective_date": finalizer.NTCO_EXACT_TICKER_DATE,
                "ex_date": finalizer.NTCO_EXACT_TICKER_DATE,
                "announcement_date": "2024-02-09",
                "record_date": "",
                "payment_date": "",
                "cash_amount": None,
                "ratio": None,
                "currency": "USD",
                "new_security_id": finalizer.NTCO_EXACT_SECURITY_ID,
                "new_symbol": finalizer.NTCO_EXACT_NEW_SYMBOL,
                "official": True,
                "source_url": finalizer.NTCO_EXACT_OCC_URL,
                "source_kind": "clearing_and_exchange_notices",
                "source": "official_ntco_ntcoy_identity",
                "retrieved_at": finalizer.NTCO_EXACT_RETRIEVED_AT,
                "source_hash": finalizer.NTCO_EXACT_IDENTITY_SHA256,
                "metadata": json.dumps(ticker_metadata, sort_keys=True),
            },
            {
                "event_id": finalizer.NTCO_EXACT_TERMINAL_EVENT_ID,
                "security_id": finalizer.NTCO_EXACT_SECURITY_ID,
                "action_type": "delisting",
                "effective_date": finalizer.NTCO_EXACT_TERMINAL_DATE,
                "ex_date": finalizer.NTCO_EXACT_TERMINAL_DATE,
                "announcement_date": "2024-08-26",
                "record_date": "",
                "payment_date": finalizer.NTCO_EXACT_TERMINAL_DATE,
                "cash_amount": finalizer.NTCO_EXACT_TERMINAL_CASH,
                "ratio": None,
                "currency": "USD",
                "new_security_id": "",
                "new_symbol": "",
                "official": True,
                "source_url": finalizer.NTCO_EXACT_BNY_CASH_URL,
                "source_kind": "depositary_corporate_action_notice",
                "source": "official_ntcoy_cash_termination",
                "retrieved_at": finalizer.NTCO_EXACT_RETRIEVED_AT,
                "source_hash": finalizer.NTCO_EXACT_TERMINAL_SHA256,
                "metadata": json.dumps(terminal_metadata, sort_keys=True),
            },
            *[
                {
                    "event_id": event_id,
                    "security_id": finalizer.NTCO_EXACT_SECURITY_ID,
                    "action_type": "cash_dividend",
                    "effective_date": effective_date,
                    "ex_date": effective_date,
                    "announcement_date": "",
                    "record_date": "",
                    "payment_date": "",
                    "cash_amount": cash_amount,
                    "ratio": None,
                    "currency": "USD",
                    "new_security_id": "",
                    "new_symbol": "",
                    "official": False,
                    "source_url": finalizer.NTCO_EXACT_PRESERVED_DIVIDEND_URL,
                    "source_kind": "provider",
                    "source": "eodhd_div",
                    "retrieved_at": (
                        finalizer.NTCO_EXACT_PRESERVED_DIVIDEND_RETRIEVED_AT
                    ),
                    "source_hash": (
                        finalizer.NTCO_EXACT_PRESERVED_DIVIDEND_SHA256
                    ),
                    "metadata": "",
                }
                for event_id, (effective_date, cash_amount) in (
                    finalizer.NTCO_EXACT_PRESERVED_DIVIDENDS.items()
                )
            ],
        ]
    )
    lineage = finalizer._adjustment_source_version(
        finalizer.NTCO_EXACT_MIXED_PRICE_VERSION,
        finalizer.NTCO_EXACT_MIXED_ACTION_VERSION,
    )
    factors = finalizer.build_adjustment_factors(
        prices,
        actions,
        source_version=lineage,
    )
    factors["calculated_at"] = finalizer.NTCO_EXACT_REVIEWED_AT
    factors["retrieved_at"] = finalizer.NTCO_EXACT_REVIEWED_AT
    master = pd.DataFrame(
        [
            {
                "security_id": finalizer.NTCO_EXACT_SECURITY_ID,
                "primary_symbol": finalizer.NTCO_EXACT_NEW_SYMBOL,
                "provider_symbol": "NTCOY.US",
                "action_provider_symbol": "NTCOY.US",
                "exchange": "OTC",
                "active_from": finalizer.NTCO_EXACT_ACTIVE_FROM,
                "active_to": finalizer.NTCO_EXACT_TERMINAL_DATE,
                "source": "official_ntco_ntcoy_identity",
                "source_url": finalizer.NTCO_EXACT_OCC_URL,
                "retrieved_at": finalizer.NTCO_EXACT_RETRIEVED_AT,
                "source_hash": finalizer.NTCO_EXACT_IDENTITY_SHA256,
            }
        ]
    )
    history = pd.DataFrame(
        [
            {
                "security_id": finalizer.NTCO_EXACT_SECURITY_ID,
                "symbol": finalizer.NTCO_EXACT_OLD_SYMBOL,
                "exchange": "NYSE",
                "effective_from": finalizer.NTCO_EXACT_ACTIVE_FROM,
                "effective_to": finalizer.NTCO_EXACT_OLD_SYMBOL_END,
                "source": "official_ntco_ntcoy_identity",
                "source_url": finalizer.NTCO_EXACT_OCC_URL,
                "retrieved_at": finalizer.NTCO_EXACT_RETRIEVED_AT,
                "source_hash": finalizer.NTCO_EXACT_IDENTITY_SHA256,
            },
            {
                "security_id": finalizer.NTCO_EXACT_SECURITY_ID,
                "symbol": finalizer.NTCO_EXACT_NEW_SYMBOL,
                "exchange": "OTC",
                "effective_from": finalizer.NTCO_EXACT_TICKER_DATE,
                "effective_to": finalizer.NTCO_EXACT_LAST_SESSION,
                "source": "official_ntco_ntcoy_identity",
                "source_url": finalizer.NTCO_EXACT_OCC_URL,
                "retrieved_at": finalizer.NTCO_EXACT_RETRIEVED_AT,
                "source_hash": finalizer.NTCO_EXACT_IDENTITY_SHA256,
            },
        ]
    )
    resolution = pd.DataFrame(
        [
            {
                "candidate_id": lifecycle_candidate_id(
                    finalizer.NTCO_EXACT_SECURITY_ID,
                    finalizer.NTCO_EXACT_LAST_SESSION,
                ),
                "security_id": finalizer.NTCO_EXACT_SECURITY_ID,
                "symbol": finalizer.NTCO_EXACT_NEW_SYMBOL,
                "last_price_date": finalizer.NTCO_EXACT_LAST_SESSION,
                "resolution": "applied",
                "event_id": finalizer.NTCO_EXACT_TERMINAL_EVENT_ID,
                "exception_code": "",
                "exception_reason": "",
                "reviewed_by": finalizer.NTCO_EXACT_REVIEWED_BY,
                "reviewed_at": finalizer.NTCO_EXACT_REVIEWED_AT,
                "recheck_after": "",
                "successor_security_id": "",
                "successor_symbol": "",
                "source_url": finalizer.NTCO_EXACT_BNY_CASH_URL,
                "source": "official_ntcoy_cash_termination",
                "retrieved_at": finalizer.NTCO_EXACT_RETRIEVED_AT,
                "source_hash": finalizer.NTCO_EXACT_TERMINAL_SHA256,
            }
        ]
    )
    identity_value = {
        "schema": "official_ntco_ntcoy_identity/v1",
        "security_id": finalizer.NTCO_EXACT_SECURITY_ID,
        "effective_date": finalizer.NTCO_EXACT_TICKER_DATE,
        "old_symbol": finalizer.NTCO_EXACT_OLD_SYMBOL,
        "new_symbol": finalizer.NTCO_EXACT_NEW_SYMBOL,
        "canonical_exchange": "OTC",
        "official_destination_market": "Other-OTC",
        "cusip": "63884N108",
        "deliverable": "100 American Depositary Shares",
        "cboe_raw_sha256": finalizer.NTCO_EXACT_CBOE_RAW_SHA256,
        "occ_raw_sha256": finalizer.NTCO_EXACT_OCC_RAW_SHA256,
    }
    terminal_value = {
        "schema": "official_ntcoy_cash_termination/v1",
        "security_id": finalizer.NTCO_EXACT_SECURITY_ID,
        "action_type": "delisting",
        "effective_date": finalizer.NTCO_EXACT_TERMINAL_DATE,
        "cash_amount": "5.043659",
        "currency": "USD",
        "ads_to_underlying_ratio": "1:2",
        "fee_per_ads": "0",
        "bny_raw_sha256": finalizer.NTCO_EXACT_BNY_CASH_RAW_SHA256,
    }
    decision_value = {
        "schema": "reviewed_ntco_ntcoy_transition_decision/v1",
        "security_id": finalizer.NTCO_EXACT_SECURITY_ID,
        "decision_mode": "price_identity_terminal_only",
        "provider_price_raw_sha256": finalizer.NTCO_EXACT_EOD_RAW_SHA256,
        "provider_splits_raw_sha256": finalizer.NTCO_EXACT_SPLITS_RAW_SHA256,
        "provider_dividend_economics_accepted": False,
        "provider_dividend_raw_decision": (
            "archive_exact_ntcoy_raw_reject_economics_preserve_ntco_actions"
        ),
        "provider_dividend_raw_sha256": finalizer.NTCO_EXACT_DIV_RAW_SHA256,
        "maximum_absolute_sensitivity_usd_per_ads": "0.01585",
    }
    pdf = b"%PDF-1.5\nfixture\n%%EOF\n"
    contents = {
        ("official_ntco_ntcoy_identity", finalizer.NTCO_EXACT_OCC_URL): (
            json.dumps(identity_value).encode()
        ),
        ("official_ntcoy_cash_termination", finalizer.NTCO_EXACT_BNY_CASH_URL): (
            json.dumps(terminal_value).encode()
        ),
        ("reviewed_ntco_ntcoy_transition_decision", finalizer.NTCO_EXACT_DECISION_URL): (
            json.dumps(decision_value).encode()
        ),
        ("eodhd_eod", finalizer.NTCO_EXACT_EOD_URL): eod_content,
        ("eodhd_div", finalizer.NTCO_EXACT_DIV_URL): b"[{},{}]",
        ("eodhd_splits", finalizer.NTCO_EXACT_SPLITS_URL): b"[]",
        ("official_cboe", finalizer.NTCO_EXACT_CBOE_URL): pdf,
        ("official_occ", finalizer.NTCO_EXACT_OCC_URL): pdf,
        ("official_bny", finalizer.NTCO_EXACT_BNY_CASH_URL): pdf,
        (
            "official_bny_termination",
            finalizer.NTCO_EXACT_BNY_TERMINATION_URL,
        ): pdf,
        (
            "official_bny_books_closed",
            finalizer.NTCO_EXACT_BNY_BOOKS_CLOSED_URL,
        ): pdf,
    }
    archive_specs = (
        (
            "official_ntco_ntcoy_identity",
            finalizer.NTCO_EXACT_OCC_URL,
            finalizer.NTCO_EXACT_IDENTITY_SHA256,
            finalizer.NTCO_EXACT_RETRIEVED_AT,
            "application/json",
            "",
        ),
        (
            "official_ntcoy_cash_termination",
            finalizer.NTCO_EXACT_BNY_CASH_URL,
            finalizer.NTCO_EXACT_TERMINAL_SHA256,
            finalizer.NTCO_EXACT_RETRIEVED_AT,
            "application/json",
            "",
        ),
        (
            "reviewed_ntco_ntcoy_transition_decision",
            finalizer.NTCO_EXACT_DECISION_URL,
            finalizer.NTCO_EXACT_DECISION_SHA256,
            finalizer.NTCO_EXACT_RETRIEVED_AT,
            "application/json",
            "",
        ),
        (
            "eodhd_eod",
            finalizer.NTCO_EXACT_EOD_URL,
            finalizer.NTCO_EXACT_EOD_RAW_SHA256,
            finalizer.NTCO_EXACT_EOD_RETRIEVED_AT,
            "application/json",
            "",
        ),
        (
            "eodhd_div",
            finalizer.NTCO_EXACT_DIV_URL,
            finalizer.NTCO_EXACT_DIV_RAW_SHA256,
            finalizer.NTCO_EXACT_EOD_RETRIEVED_AT,
            "application/json",
            "",
        ),
        (
            "eodhd_splits",
            finalizer.NTCO_EXACT_SPLITS_URL,
            finalizer.NTCO_EXACT_SPLITS_RAW_SHA256,
            finalizer.NTCO_EXACT_EOD_RETRIEVED_AT,
            "application/json",
            finalizer.NTCO_EXACT_SPLITS_ARCHIVE_ID,
        ),
        (
            "official_cboe",
            finalizer.NTCO_EXACT_CBOE_URL,
            finalizer.NTCO_EXACT_CBOE_RAW_SHA256,
            finalizer.NTCO_EXACT_CBOE_RAW_RETRIEVED_AT,
            "application/pdf",
            "",
        ),
        (
            "official_occ",
            finalizer.NTCO_EXACT_OCC_URL,
            finalizer.NTCO_EXACT_OCC_RAW_SHA256,
            finalizer.NTCO_EXACT_OCC_RAW_RETRIEVED_AT,
            "text/html",
            "",
        ),
        (
            "official_bny",
            finalizer.NTCO_EXACT_BNY_CASH_URL,
            finalizer.NTCO_EXACT_BNY_CASH_RAW_SHA256,
            finalizer.NTCO_EXACT_BNY_CASH_RAW_RETRIEVED_AT,
            "application/pdf",
            "",
        ),
        (
            "official_bny_termination",
            finalizer.NTCO_EXACT_BNY_TERMINATION_URL,
            finalizer.NTCO_EXACT_BNY_TERMINATION_RAW_SHA256,
            finalizer.NTCO_EXACT_BNY_TERMINATION_RETRIEVED_AT,
            "application/pdf",
            "",
        ),
        (
            "official_bny_books_closed",
            finalizer.NTCO_EXACT_BNY_BOOKS_CLOSED_URL,
            finalizer.NTCO_EXACT_BNY_BOOKS_CLOSED_RAW_SHA256,
            finalizer.NTCO_EXACT_BNY_BOOKS_CLOSED_RETRIEVED_AT,
            "application/pdf",
            "",
        ),
    )
    archive_rows = []
    for source, source_url, source_hash, retrieved_at, content_type, archive_id in archive_specs:
        suffix = "json" if content_type == "application/json" else "bin"
        archive_rows.append(
            {
                "archive_id": archive_id or source_hash,
                "dataset": source,
                "object_path": (
                    f"archives/2026-07-15/{source_hash}.{suffix}.gz"
                ),
                "content_type": content_type,
                "effective_date": "2026-07-15",
                "source": source,
                "source_url": source_url,
                "retrieved_at": retrieved_at,
                "source_hash": source_hash,
            }
        )
    frames = {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "adjustment_factors": factors,
        "source_archive": pd.DataFrame(archive_rows),
        "lifecycle_resolutions": resolution,
    }
    candidate = LifecycleCandidate(
        finalizer.NTCO_EXACT_SECURITY_ID,
        finalizer.NTCO_EXACT_NEW_SYMBOL,
        "Natura &Co Holding S.A.",
        "OTC",
        finalizer.NTCO_EXACT_LAST_SESSION,
        finalizer.NTCO_EXACT_TERMINAL_DATE,
    )
    release = SimpleNamespace(
        completed_session="2026-07-15",
        warnings=(),
        dataset_versions={
            "daily_price_raw": finalizer.NTCO_EXACT_MIXED_PRICE_VERSION,
            "corporate_actions": finalizer.NTCO_EXACT_MIXED_ACTION_VERSION,
            "adjustment_factors": finalizer.NTCO_EXACT_MIXED_FACTOR_VERSION,
        },
    )
    return frames, contents, candidate, release


def _ntco_archive_side_effect(contents):
    def read(_repository, archive, *, source_url, source_hash, source):
        rows = archive.loc[
            archive["source"].astype(str).eq(source)
            & archive["source_url"].astype(str).eq(source_url)
            & archive["source_hash"].astype(str).eq(source_hash)
        ]
        assert len(rows) == 1
        return contents[(source, source_url)], rows.iloc[0].to_dict()

    return read


def _ntco_mixed_lineage_fixture():
    prior_action_version = finalizer.EXACT_PROVENANCE_BRIDGE_FACTOR_ACTION_VERSION
    prior_price_version = prior_action_version.replace(
        "-corporate_actions", "-daily_price_raw"
    )
    prior_factor_version = prior_action_version.replace(
        "-corporate_actions", "-adjustment_factors"
    )
    prior_source_version = finalizer._adjustment_source_version(
        prior_price_version, prior_action_version
    )
    current_source_version = finalizer._adjustment_source_version(
        finalizer.NTCO_EXACT_MIXED_PRICE_VERSION,
        finalizer.NTCO_EXACT_MIXED_ACTION_VERSION,
    )
    other_security_id = "US:EODHD:other-lineage-security"

    def factor_row(security_id, source_version):
        return {
            "security_id": security_id,
            "session": "2024-02-12",
            "split_factor": 1.0,
            "total_return_factor": 1.0,
            "source_version": source_version,
            "calculated_at": "2026-07-19T00:00:00Z",
            "source": "derived",
            "source_url": "",
            "retrieved_at": "2026-07-19T00:00:00Z",
            "source_hash": source_version,
        }

    def price_row(security_id, close):
        return {
            "security_id": security_id,
            "session": "2024-02-12",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 100.0,
            "currency": "USD",
            "source": "eodhd_eod",
            "source_url": "",
            "retrieved_at": "2026-07-18T00:00:00Z",
            "source_hash": "a" * 64,
        }

    def action_row(
        event_id,
        security_id,
        action_type,
        effective_date,
        *,
        new_security_id="",
        new_symbol="",
        cash_amount=None,
        source="old_reviewed_source",
        source_hash="1" * 64,
    ):
        return {
            "event_id": event_id,
            "security_id": security_id,
            "action_type": action_type,
            "effective_date": effective_date,
            "ex_date": effective_date,
            "announcement_date": "",
            "record_date": "",
            "payment_date": "",
            "cash_amount": cash_amount,
            "ratio": None,
            "currency": "USD",
            "new_security_id": new_security_id,
            "new_symbol": new_symbol,
            "official": True,
            "source_url": "https://example.test/evidence",
            "source_kind": "official_crosscheck",
            "source": source,
            "retrieved_at": "2026-07-18T00:00:00Z",
            "source_hash": source_hash,
            "metadata": "",
        }

    prior_actions = pd.DataFrame(
        [
            action_row(
                finalizer.SIVB_EXACT_TICKER_EVENT_ID,
                finalizer.SIVB_EXACT_SECURITY_ID,
                "ticker_change",
                finalizer.SIVB_EXACT_OTC_START,
                new_security_id=finalizer.SIVB_EXACT_SECURITY_ID,
                new_symbol="SIVBQ",
            ),
            action_row(
                finalizer.FRC_EXACT_EVENT_ID,
                finalizer.FRC_EXACT_SECURITY_ID,
                "ticker_change",
                finalizer.FRC_EXACT_TRANSITION,
                new_security_id=finalizer.FRC_EXACT_SECURITY_ID,
                new_symbol="FRCB",
            ),
        ]
    )
    current_actions = prior_actions.copy(deep=True)
    changed = current_actions["event_id"].astype(str).isin(
        {
            finalizer.SIVB_EXACT_TICKER_EVENT_ID,
            finalizer.FRC_EXACT_EVENT_ID,
        }
    )
    current_actions.loc[changed, "source"] = "raw_official_notice"
    current_actions.loc[changed, "source_hash"] = "2" * 64
    current_actions = pd.concat(
        [
            current_actions,
            pd.DataFrame(
                [
                    action_row(
                        finalizer.NTCO_EXACT_TICKER_EVENT_ID,
                        finalizer.NTCO_EXACT_SECURITY_ID,
                        "ticker_change",
                        finalizer.NTCO_EXACT_TICKER_DATE,
                        new_security_id=finalizer.NTCO_EXACT_SECURITY_ID,
                        new_symbol=finalizer.NTCO_EXACT_NEW_SYMBOL,
                        source="official_ntco_ntcoy_identity",
                        source_hash=finalizer.NTCO_EXACT_IDENTITY_SHA256,
                    ),
                    action_row(
                        finalizer.NTCO_EXACT_TERMINAL_EVENT_ID,
                        finalizer.NTCO_EXACT_SECURITY_ID,
                        "delisting",
                        finalizer.NTCO_EXACT_TERMINAL_DATE,
                        cash_amount=finalizer.NTCO_EXACT_TERMINAL_CASH,
                        source="official_ntcoy_cash_termination",
                        source_hash=finalizer.NTCO_EXACT_TERMINAL_SHA256,
                    ),
                ]
            ),
        ],
        ignore_index=True,
    )
    prior_factors = pd.DataFrame(
        [
            factor_row(other_security_id, prior_source_version),
            factor_row(finalizer.NTCO_EXACT_SECURITY_ID, prior_source_version),
        ]
    )
    current_factors = pd.DataFrame(
        [
            factor_row(other_security_id, prior_source_version),
            factor_row(finalizer.NTCO_EXACT_SECURITY_ID, current_source_version),
        ]
    )
    prior_prices = pd.DataFrame(
        [
            price_row(other_security_id, 10.0),
            price_row(finalizer.NTCO_EXACT_SECURITY_ID, 6.0),
        ]
    )
    current_prices = pd.DataFrame(
        [
            price_row(other_security_id, 10.0),
            price_row(finalizer.NTCO_EXACT_SECURITY_ID, 6.1),
        ]
    )
    frames = {
        "security_master": pd.DataFrame(),
        "symbol_history": pd.DataFrame(),
        "daily_price_raw": current_prices,
        "corporate_actions": current_actions,
        "adjustment_factors": current_factors,
        "source_archive": pd.DataFrame(),
        "lifecycle_resolutions": pd.DataFrame(),
    }
    factor_manifest = SimpleNamespace(
        version=finalizer.NTCO_EXACT_MIXED_FACTOR_VERSION,
        metadata={
            "source_daily_price_version": prior_price_version,
            "source_corporate_actions_version": prior_action_version,
            "source_version": prior_source_version,
            "daily_price_version": finalizer.NTCO_EXACT_MIXED_PRICE_VERSION,
            "corporate_action_version": finalizer.NTCO_EXACT_MIXED_ACTION_VERSION,
            "operation": "repair_us_ntco_ntcoy_transition",
        },
    )
    prior_frames = {
        ("daily_price_raw", prior_price_version): prior_prices,
        ("corporate_actions", prior_action_version): prior_actions,
        ("adjustment_factors", prior_factor_version): prior_factors,
    }
    repository = SimpleNamespace(
        manifest_for_version=lambda _dataset, _version: factor_manifest,
        read_frame=lambda dataset, version: prior_frames[(dataset, version)].copy(
            deep=True
        ),
    )
    release = SimpleNamespace(
        completed_session="2026-07-15",
        warnings=(),
        dataset_versions={
            "daily_price_raw": finalizer.NTCO_EXACT_MIXED_PRICE_VERSION,
            "corporate_actions": finalizer.NTCO_EXACT_MIXED_ACTION_VERSION,
            "adjustment_factors": finalizer.NTCO_EXACT_MIXED_FACTOR_VERSION,
        },
    )
    return (
        repository,
        release,
        frames,
        factor_manifest,
        prior_source_version,
        other_security_id,
    )


def test_finalizer_accepts_only_exact_ntco_mixed_adjustment_lineage_bridge():
    repository, release, frames, _manifest, prior_lineage, _other = (
        _ntco_mixed_lineage_fixture()
    )
    with patch.object(
        finalizer, "_require_exact_frc_occ_action"
    ) as frc_check, patch.object(
        finalizer, "_require_exact_sivb_occ_action"
    ) as sivb_check, patch.object(
        finalizer, "_preserve_exact_ntco_resolution", return_value={}
    ) as ntco_check:
        observed = finalizer._validate_input_adjustment_lineage_for_refinalization(
            repository, release, frames
        )
    assert observed == prior_lineage
    frc_check.assert_called_once_with(frames["corporate_actions"])
    sivb_check.assert_called_once_with(frames["corporate_actions"])
    ntco_check.assert_called_once()


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("non_target_factor", "non-NTCO adjustment-factor row"),
        ("non_target_price", "non-NTCO price row"),
        ("target_lineage", "factor row provenance changed"),
        ("action_delta", "corporate-action delta changed"),
        ("manifest", "manifest contract changed"),
    ],
)
def test_finalizer_rejects_ntco_mixed_adjustment_lineage_drift(drift, message):
    repository, release, frames, manifest, _prior_lineage, other = (
        _ntco_mixed_lineage_fixture()
    )
    if drift == "non_target_factor":
        row = frames["adjustment_factors"]["security_id"].astype(str).eq(other)
        frames["adjustment_factors"].loc[row, "split_factor"] = 0.5
    elif drift == "non_target_price":
        row = frames["daily_price_raw"]["security_id"].astype(str).eq(other)
        frames["daily_price_raw"].loc[row, "close"] = 9.0
    elif drift == "target_lineage":
        row = (
            frames["adjustment_factors"]["security_id"]
            .astype(str)
            .eq(finalizer.NTCO_EXACT_SECURITY_ID)
        )
        frames["adjustment_factors"].loc[row, "source_hash"] = "0" * 64
    elif drift == "action_delta":
        extra = frames["corporate_actions"].iloc[[0]].copy(deep=True)
        extra.loc[:, "event_id"] = "unreviewed-event"
        frames["corporate_actions"] = pd.concat(
            [frames["corporate_actions"], extra], ignore_index=True
        )
    elif drift == "manifest":
        manifest.metadata["daily_price_version"] = "unreviewed-price"
    with patch.object(
        finalizer, "_require_exact_frc_occ_action"
    ), patch.object(
        finalizer, "_require_exact_sivb_occ_action"
    ), patch.object(
        finalizer, "_preserve_exact_ntco_resolution", return_value={}
    ), pytest.raises(RuntimeError, match=message):
        finalizer._validate_input_adjustment_lineage_for_refinalization(
            repository, release, frames
        )


def test_finalizer_keeps_unreviewed_stale_price_lineage_fail_closed():
    repository, release, frames, manifest, _prior_lineage, _other = (
        _ntco_mixed_lineage_fixture()
    )
    release.dataset_versions["adjustment_factors"] = "unreviewed-factor-version"
    manifest.version = "unreviewed-factor-version"
    with pytest.raises(RuntimeError, match="manifest lineage is stale"):
        finalizer._validate_input_adjustment_lineage_for_refinalization(
            repository, release, frames
        )


def test_finalizer_preserves_exact_ntco_price_only_resolution_before_report_parsing():
    frames, contents, candidate, release = _ntco_fixture()
    with patch.object(
        finalizer,
        "_archive_pair_content",
        side_effect=_ntco_archive_side_effect(contents),
    ):
        preserved = finalizer._preserved_exact_repair_resolution(
            candidate,
            SimpleNamespace(root=Path("/unused")),
            release,
            frames,
        )
    assert preserved is not None
    assert preserved["event_id"] == finalizer.NTCO_EXACT_TERMINAL_EVENT_ID
    assert preserved["last_price_date"] == finalizer.NTCO_EXACT_LAST_SESSION


def test_finalizer_rejects_ntco_price_and_bny_provenance_drift():
    frames, contents, candidate, release = _ntco_fixture()
    changed_price = {name: frame.copy(deep=True) for name, frame in frames.items()}
    changed_price["daily_price_raw"].loc[0, "close"] = 0.01
    with patch.object(
        finalizer,
        "_archive_pair_content",
        side_effect=_ntco_archive_side_effect(contents),
    ), pytest.raises(RuntimeError, match="stored OHLCV changed"):
        finalizer._preserved_exact_repair_resolution(
            candidate,
            SimpleNamespace(root=Path("/unused")),
            release,
            changed_price,
        )

    changed_archive = {name: frame.copy(deep=True) for name, frame in frames.items()}
    bny = changed_archive["source_archive"]["source"].astype(str).eq("official_bny")
    changed_archive["source_archive"].loc[bny, "object_path"] = (
        "archives/2026-07-15/changed.bin.gz"
    )
    with patch.object(
        finalizer,
        "_archive_pair_content",
        side_effect=_ntco_archive_side_effect(contents),
    ), pytest.raises(RuntimeError, match="source_archive provenance changed"):
        finalizer._preserved_exact_repair_resolution(
            candidate,
            SimpleNamespace(root=Path("/unused")),
            release,
            changed_archive,
        )


def _current_market_date_transition_fixture():
    root = Path(__file__).resolve().parents[2] / "data/cache"
    repository = LocalDatasetRepository(root)
    release, _etag = repository.current_release()
    if release is None:
        pytest.skip("Current local market-store release is unavailable.")
    required = {
        "security_master",
        "symbol_history",
        "corporate_actions",
        "source_archive",
        "lifecycle_resolutions",
    }
    if not required.issubset(release.dataset_versions):
        pytest.skip("Current release lacks market-date transition inputs.")
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in required
    }
    candidates = {
        spec["candidate"]["symbol"]: LifecycleCandidate(
            **spec["candidate"]
        )
        for spec in finalizer.EXACT_REVIEWED_MARKET_DATE_TRANSITIONS.values()
    }
    return repository, release, frames, candidates






def _current_short_terminal_market_transition_fixture():
    root = Path(__file__).resolve().parents[2] / "data/cache"
    repository = LocalDatasetRepository(root)
    release, _etag = repository.current_release()
    if release is None:
        pytest.skip("Current local market-store release is unavailable.")
    required = {
        "daily_price_raw",
        "adjustment_factors",
        "corporate_actions",
        "source_archive",
        "lifecycle_resolutions",
    }
    if not required.issubset(release.dataset_versions):
        pytest.skip("Current release lacks short-terminal transition inputs.")
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in required
    }
    candidates = {}
    for key, spec in finalizer.EXACT_SHORT_TERMINAL_MARKET_TRANSITIONS.items():
        security_id, last_price_date = key.split("|", 1)
        candidate_spec = spec["candidate"]
        candidates[candidate_spec["symbol"]] = LifecycleCandidate(
            security_id=security_id,
            symbol=candidate_spec["symbol"],
            name=candidate_spec["name"],
            exchange=candidate_spec["exchange"],
            last_price_date=last_price_date,
            active_to=candidate_spec["active_to"],
            index_remove_dates=candidate_spec["index_remove_dates"],
        )
    return repository, release, frames, candidates






def test_finalizer_frame_validation_routes_through_reviewed_operational_policy():
    valid = SimpleNamespace(issues=(), raise_for_errors=lambda: None)
    with patch.object(
        finalizer,
        "validate_operational_repository_snapshot",
        return_value=valid,
    ) as operational:
        assert finalizer._validate_all_frames({}, "2026-07-15") == ()
    operational.assert_called_once()


def _reviewed_nbl_fixture():
    expected = finalizer.TRUSTED_OPERATIONAL_NBL_TERMINAL_STATE
    candidate = LifecycleCandidate(
        security_id=expected["security_id"],
        symbol=expected["symbol"],
        name="Noble Energy Inc",
        exchange=expected["identity_exchange"],
        last_price_date=expected["last_real_session"],
        active_to=expected["last_real_session"],
        index_remove_dates=(expected["next_remove_effective_date"],),
    )
    resolution = pd.DataFrame(
        [
            {
                "candidate_id": expected["candidate_id"],
                "security_id": expected["security_id"],
                "symbol": expected["symbol"],
                "last_price_date": expected["last_real_session"],
                "resolution": "applied",
                "event_id": expected["event_id"],
                "exception_code": "",
                "exception_reason": "",
                "reviewed_by": expected["resolution_reviewer"],
                "reviewed_at": expected["repair_reviewed_at"],
                "recheck_after": "",
                "successor_security_id": expected["successor_security_id"],
                "successor_symbol": expected["successor_symbol"],
                "source_url": expected["official_source_url"],
                "source": expected["resolution_source"],
                "retrieved_at": expected["repair_reviewed_at"],
                "source_hash": expected["official_source_hash"],
            }
        ]
    )
    frames = {
        "corporate_actions": pd.DataFrame(
            [{"event_id": expected["event_id"]}]
        ),
        "lifecycle_resolutions": resolution,
    }
    return expected, candidate, frames


def test_finalizer_preserves_nbl_only_under_exact_operational_attestation():
    expected, candidate, frames = _reviewed_nbl_fixture()
    with patch.object(
        finalizer,
        "reviewed_operational_index_identity_gap_fingerprints",
        return_value=(expected["fingerprint"],),
    ) as reviewed:
        preserved = finalizer._preserved_exact_repair_resolution(
            candidate,
            SimpleNamespace(root=Path("/unused")),
            SimpleNamespace(completed_session="2026-07-15"),
            frames,
        )
    assert preserved is not None
    assert preserved["event_id"] == expected["event_id"]
    reviewed.assert_called_once()


def test_finalizer_rejects_nbl_operational_or_resolution_drift():
    expected, candidate, frames = _reviewed_nbl_fixture()
    with patch.object(
        finalizer,
        "reviewed_operational_index_identity_gap_fingerprints",
        return_value=(),
    ), pytest.raises(RuntimeError, match="operational terminal state changed"):
        finalizer._preserved_exact_repair_resolution(
            candidate,
            SimpleNamespace(root=Path("/unused")),
            SimpleNamespace(completed_session="2026-07-15"),
            frames,
        )

    frames["lifecycle_resolutions"].loc[0, "source_hash"] = "0" * 64
    with patch.object(
        finalizer,
        "reviewed_operational_index_identity_gap_fingerprints",
        return_value=(expected["fingerprint"],),
    ), pytest.raises(RuntimeError, match="resolution field changed"):
        finalizer._preserved_exact_repair_resolution(
            candidate,
            SimpleNamespace(root=Path("/unused")),
            SimpleNamespace(completed_session="2026-07-15"),
            frames,
        )


def _current_avp_fixture():
    root = Path(__file__).resolve().parents[2] / "data/cache"
    repository = LocalDatasetRepository(root)
    release, _etag = repository.current_release()
    if release is None:
        pytest.skip("Current local market-store release is unavailable.")
    required = {
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "corporate_actions",
        "adjustment_factors",
        "source_archive",
        "lifecycle_resolutions",
    }
    if not required.issubset(release.dataset_versions):
        pytest.skip("Current release lacks exact AVP preservation inputs.")
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in required
    }
    candidates = [
        item
        for item in finalizer.build_lifecycle_candidates(
            repository, release=release
        )
        if item.security_id == finalizer.AVP_EXACT_SECURITY_ID
    ]
    if len(candidates) != 1:
        pytest.skip("Current release does not contain the exact AVP candidate.")
    return repository, release, frames, candidates[0]


def _with_exact_avp_temporary_resolution(frames):
    output = {name: frame.copy(deep=True) for name, frame in frames.items()}
    mask = output["lifecycle_resolutions"]["security_id"].astype(str).eq(
        finalizer.AVP_EXACT_SECURITY_ID
    )
    assert int(mask.sum()) == 1
    expected = {
        "candidate_id": lifecycle_candidate_id(
            finalizer.AVP_EXACT_SECURITY_ID,
            finalizer.AVP_EXACT_LAST_SESSION,
        ),
        "security_id": finalizer.AVP_EXACT_SECURITY_ID,
        "symbol": "AVP",
        "last_price_date": finalizer.AVP_EXACT_LAST_SESSION,
        "resolution": "exception",
        "event_id": "",
        "exception_code": "successor_unresolved",
        "exception_reason": "AVP to NTCO successor chain is not fully crosschecked.",
        "reviewed_by": finalizer.REVIEWED_BY,
        "reviewed_at": finalizer.REVIEWED_AT,
        "recheck_after": finalizer.DEFAULT_RECHECK_AFTER,
        "successor_security_id": "",
        "successor_symbol": "",
        "source_url": (
            "archive://source_archive/"
            + finalizer.AVP_EXACT_TEMPORARY_REPORT_SHA256
        ),
        "source": "lifecycle_evidence_report",
        "retrieved_at": finalizer.AVP_EXACT_TEMPORARY_REPORT_RETRIEVED_AT,
        "source_hash": finalizer.AVP_EXACT_TEMPORARY_REPORT_SHA256,
    }
    index = output["lifecycle_resolutions"].index[mask][0]
    for field, value in expected.items():
        output["lifecycle_resolutions"].at[index, field] = value
    return output






def _current_sivb_fixture():
    root = Path(__file__).resolve().parents[2] / "data/cache"
    repository = LocalDatasetRepository(root)
    release, _etag = repository.current_release()
    if release is None:
        pytest.skip("Current local market-store release is unavailable.")
    required = {
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "corporate_actions",
        "adjustment_factors",
        "source_archive",
        "lifecycle_resolutions",
    }
    if not required.issubset(release.dataset_versions):
        pytest.skip("Current release lacks exact SIVB preservation inputs.")
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in required
    }
    candidates = [
        item
        for item in finalizer.build_lifecycle_candidates(
            repository, release=release
        )
        if item.security_id == finalizer.SIVB_EXACT_SECURITY_ID
    ]
    if len(candidates) != 1:
        pytest.skip("Current release does not contain the repaired SIVBQ candidate.")
    return repository, release, frames, candidates[0]






def _current_prior_terminal_transition_fixture():
    root = Path(__file__).resolve().parents[2] / "data/cache"
    repository = LocalDatasetRepository(root)
    release, _etag = repository.current_release()
    if release is None:
        pytest.skip("Current local market-store release is unavailable.")
    required = {
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "corporate_actions",
        "source_archive",
        "lifecycle_resolutions",
    }
    if not required.issubset(release.dataset_versions):
        pytest.skip("Current release lacks prior terminal-transition inputs.")
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in required
    }
    candidates = tuple(
        finalizer.build_lifecycle_candidates(repository, release=release)
    )
    exact_keys = set(finalizer.EXACT_PRIOR_TERMINAL_TRANSITIONS)
    observed = {
        finalizer._key(candidate.security_id, candidate.last_price_date)
        for candidate in candidates
    }
    if not exact_keys.issubset(observed):
        pytest.skip("Current release lacks all four prior terminal candidates.")
    return repository, release, frames, candidates


def _with_canonical_prior_terminal_resolutions(frames, restored):
    output = {name: frame.copy(deep=True) for name, frame in frames.items()}
    for resolution in restored.values():
        mask = output["lifecycle_resolutions"]["security_id"].astype(str).eq(
            resolution["security_id"]
        )
        assert int(mask.sum()) == 1
        index = output["lifecycle_resolutions"].index[mask][0]
        for field, value in resolution.items():
            output["lifecycle_resolutions"].at[index, field] = value
    return output
