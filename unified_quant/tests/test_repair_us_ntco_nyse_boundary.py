from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import urllib.request
import urllib.response
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    DatasetManifest,
)
from supertrend_quant.market_store.official_lifecycle_evidence import (
    OFFICIAL_EXCEPTION_EVIDENCE_URL_ALLOWLIST,
    load_official_lifecycle_exception_evidence,
)
from supertrend_quant.market_store.repository import DatasetWriteResult
from supertrend_quant.market_store.storage import LocalObjectStore
from supertrend_quant.market_store.validation import ValidationReport


pytestmark = pytest.mark.skip(
    reason=(
        "Superseded audit draft: official Cboe/OCC/BNY evidence requires the "
        "NTCO->NTCOY continuation plan instead of an NYSE-only trim."
    )
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_ntco_nyse_boundary.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_ntco_nyse_boundary", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2] / "data" / "cache"
HINTS_PATH = Path(__file__).resolve().parents[1] / "configs/us_lifecycle_hints.yaml"


def _official_content() -> bytes:
    return (
        "<html><body>Natura &amp;Co Holding S.A. "
        "NTCO last day of trading was February 9, 2024. "
        "The company has not arranged for the listing or quotation of the ADSs "
        "on another exchange or in a quotation medium. "
        "The deposit agreement will terminate on August 7, 2024. "
        "Each ADS represents two common shares. Holders may surrender their ADSs "
        "and withdraw the underlying shares.</body></html>"
    ).encode()


def _tree(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_default_plan_without_cache_is_read_only_and_one_url_limited(
    tmp_path: Path,
) -> None:
    before = _tree(tmp_path)

    result = script.readiness_plan(
        SimpleNamespace(),
        hints_path=HINTS_PATH,
        evidence_dir=tmp_path,
    )

    assert result["status"] == "ready_for_authorized_one_url_fetch"
    assert result["source_url"] == script.OFFICIAL_SOURCE_URL
    assert result["max_http_attempts"] == 1
    assert result["http_attempts_this_run"] == 0
    assert result["network_accessed"] is False
    assert result["writes_performed"] is False
    assert result["eodhd_calls"] == 0
    assert result["r2_accessed"] is False
    assert result["reviewer_registry_draft"]["source_sha256"] == ""
    assert _tree(tmp_path) == before


def test_fetch_official_uses_one_exact_url_then_replays_cache(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []

    def fetcher(url: str, user_agent: str) -> bytes:
        calls.append((url, user_agent))
        return _official_content()

    first = script.fetch_official(
        tmp_path,
        user_agent="Researcher researcher@example.com",
        fetcher=fetcher,
    )
    second = script.fetch_official(
        tmp_path,
        user_agent="Researcher researcher@example.com",
        fetcher=lambda *_args: pytest.fail("cache replay attempted network"),
    )

    assert calls == [
        (
            script.OFFICIAL_SOURCE_URL,
            "Researcher researcher@example.com",
        )
    ]
    assert first["http_attempts_this_run"] == 1
    assert first["network_accessed"] is True
    assert first["status"] == "collected_pending_reviewer_pin"
    assert second["http_attempts_this_run"] == 0
    assert second["network_accessed"] is False
    assert second["status"] == "cache_verified_pending_reviewer_pin"
    staged = script.verify_staged_evidence(tmp_path)
    assert staged is not None
    assert staged.content == _official_content()
    assert staged.source_sha256 == hashlib.sha256(_official_content()).hexdigest()


def test_fetch_requires_contact_before_attempt_and_never_retries(
    tmp_path: Path,
) -> None:
    attempts = 0

    def fetcher(_url: str, _user_agent: str) -> bytes:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("one-shot failure")

    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        script.fetch_official(
            tmp_path,
            user_agent="missing-contact",
            fetcher=fetcher,
        )
    assert attempts == 0

    with pytest.raises(RuntimeError, match="one-shot failure"):
        script.fetch_official(
            tmp_path,
            user_agent="Researcher researcher@example.com",
            fetcher=fetcher,
        )
    assert attempts == 1
    assert _tree(tmp_path) == {}


def test_no_redirect_handler_rejects_3xx_before_a_followup_request() -> None:
    attempts: list[str] = []

    class RedirectingTransport(urllib.request.BaseHandler):
        handler_order = 100

        def https_open(self, request: urllib.request.Request):  # type: ignore[no-untyped-def]
            attempts.append(request.full_url)
            headers = Message()
            headers["Location"] = script.OFFICIAL_SOURCE_URL + "?redirected=1"
            response = urllib.response.addinfourl(
                io.BytesIO(b"redirect"),
                headers,
                request.full_url,
                302,
            )
            response.msg = "Found"
            return response

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        script._NoRedirectHandler(),
        RedirectingTransport(),
    )
    request = urllib.request.Request(script.OFFICIAL_SOURCE_URL)

    with pytest.raises(RuntimeError, match="automatic follow-up requests are disabled"):
        opener.open(request, timeout=1)

    assert attempts == [script.OFFICIAL_SOURCE_URL]


@pytest.mark.parametrize(
    "phrase",
    [
        "last day of trading was February 9, 2024",
        "has not arranged for the listing or quotation",
        "quotation medium",
        "August 7, 2024",
        "Each ADS represents two common shares",
        "surrender their ADSs",
    ],
)
def test_stage_one_rejects_missing_official_fact_without_writes(
    tmp_path: Path,
    phrase: str,
) -> None:
    content = _official_content().replace(phrase.encode(), b"removed")

    with pytest.raises(ValueError, match="lacks reviewed official term"):
        script.fetch_official(
            tmp_path,
            user_agent="Researcher researcher@example.com",
            fetcher=lambda *_args: content,
        )

    assert _tree(tmp_path) == {}


def test_staged_payload_and_report_tampering_fail_closed(tmp_path: Path) -> None:
    result = script.fetch_official(
        tmp_path,
        user_agent="Researcher researcher@example.com",
        fetcher=lambda *_args: _official_content(),
    )
    payload_path = Path(result["payload_path"])
    payload_path.write_bytes(payload_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash/size verification failed"):
        script.verify_staged_evidence(tmp_path)

    payload_path.write_bytes(_official_content())
    report_path = tmp_path / script.EVIDENCE_REPORT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["evidence"]["source_url"] = "https://www.sec.gov/Archives/wrong.htm"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="filing identity changed"):
        script.verify_staged_evidence(tmp_path)


def test_active_registry_is_exact_bound_but_deliberately_unpinned() -> None:
    assert (
        OFFICIAL_EXCEPTION_EVIDENCE_URL_ALLOWLIST[script.EVIDENCE_ID]
        == script.OFFICIAL_SOURCE_URL
    )
    specs = load_official_lifecycle_exception_evidence(HINTS_PATH)
    spec = specs[script.EVIDENCE_ID]
    assert spec.candidate_symbols == (script.SYMBOL,)
    assert spec.candidate_security_ids == (script.SECURITY_ID,)
    assert spec.candidate_last_price_dates == (script.LAST_NYSE_SESSION,)
    assert spec.effective_date == script.BOUNDARY_EFFECTIVE_DATE
    assert spec.resolution_kind == "exception"
    assert spec.exception_code == "unsupported_consideration"
    assert spec.claim == script.EXCEPTION_CLAIM
    assert spec.source_sha256 == ""
    assert spec.required_text_groups == script.REGISTRY_REQUIRED_TEXT_GROUPS


def test_staged_hash_cannot_self_approve_and_apply_remains_blocked(
    tmp_path: Path,
) -> None:
    staged_result = script.fetch_official(
        tmp_path,
        user_agent="Researcher researcher@example.com",
        fetcher=lambda *_args: _official_content(),
    )
    observed = staged_result["evidence"]["source_sha256"]

    result = script.readiness_plan(
        SimpleNamespace(),
        hints_path=HINTS_PATH,
        evidence_dir=tmp_path,
    )

    assert result["status"] == "blocked_pending_reviewer_pin"
    assert result["observed_source_sha256"] == observed
    assert result["reviewer_registry_draft"]["source_sha256"] == observed
    assert result["network_accessed"] is False
    assert result["writes_performed"] is False
    with pytest.raises(RuntimeError, match="not reviewer-pinned"):
        script.prepare_repair(
            SimpleNamespace(),
            hints_path=HINTS_PATH,
            evidence_dir=tmp_path,
        )


def test_fetch_and_apply_flags_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        script._parse_args(["--fetch-official", "--apply"])


@pytest.mark.skipif(
    not (REPOSITORY_ROOT / "releases/current.json").is_file(),
    reason="Actual current-release cache is unavailable.",
)
def test_actual_archived_eodhd_inventory_proves_exact_quarantine() -> None:
    repository = script.LocalDatasetRepository(REPOSITORY_ROOT)
    release, _ = repository.current_release()
    assert release is not None
    archive = repository.read_frame(
        "source_archive", release.dataset_versions["source_archive"]
    )

    prices, dividends = script._verify_eodhd_evidence(repository, archive)

    tail = [
        item
        for item in prices
        if script._date(item.get("date")) > script.LAST_NYSE_SESSION
    ]
    dividend_tail = [
        item
        for item in dividends
        if script._date(item.get("date")) > script.LAST_NYSE_SESSION
    ]
    assert len(tail) == 43
    assert tail[0]["date"] == "2024-02-12"
    assert tail[-1]["date"] == "2024-04-12"
    assert script._canonical_json_sha256(tail) == (
        script.QUARANTINED_PRICE_RECORDS_SHA256
    )
    assert len(dividend_tail) == 2
    assert script._canonical_json_sha256(dividend_tail) == (
        script.QUARANTINED_DIVIDEND_RECORDS_SHA256
    )


@pytest.mark.skipif(
    not (REPOSITORY_ROOT / "releases/current.json").is_file(),
    reason="Actual current-release cache is unavailable.",
)
def test_actual_rewrites_are_narrow_and_preserve_raw_archives() -> None:
    repository = script.LocalDatasetRepository(REPOSITORY_ROOT)
    release, _ = repository.current_release()
    assert release is not None
    prices = repository.read_frame(
        "daily_price_raw", release.dataset_versions["daily_price_raw"]
    )
    actions = repository.read_frame(
        "corporate_actions", release.dataset_versions["corporate_actions"]
    )
    archive = repository.read_frame(
        "source_archive", release.dataset_versions["source_archive"]
    )
    raw_prices, raw_dividends = script._verify_eodhd_evidence(repository, archive)

    assert script._price_state(prices, raw_prices) == "old"
    assert script._action_state(actions, raw_dividends) == "old"
    new_prices = script._rewrite_prices(prices)
    new_actions = script._rewrite_actions(actions)
    assert len(prices) - len(new_prices) == 43
    assert len(actions) - len(new_actions) == 2
    assert script._price_state(new_prices, raw_prices) == "repaired"
    assert script._action_state(new_actions, raw_dividends) == "repaired"
    assert len(archive) == len(
        repository.read_frame(
            "source_archive", release.dataset_versions["source_archive"]
        )
    )
    non_target_old = prices.loc[~prices["security_id"].astype(str).eq(script.SECURITY_ID)]
    non_target_new = new_prices.loc[
        ~new_prices["security_id"].astype(str).eq(script.SECURITY_ID)
    ]
    pd.testing.assert_frame_equal(
        non_target_old.reset_index(drop=True),
        non_target_new.reset_index(drop=True),
        check_dtype=True,
    )


@pytest.mark.skipif(
    not (REPOSITORY_ROOT / "releases/current.json").is_file(),
    reason="Actual current-release cache is unavailable.",
)
def test_actual_factor_rebuild_changes_only_ntco_total_return_economics() -> None:
    repository = script.LocalDatasetRepository(REPOSITORY_ROOT)
    release, _ = repository.current_release()
    assert release is not None
    prices = repository.read_frame(
        "daily_price_raw", release.dataset_versions["daily_price_raw"]
    )
    actions = repository.read_frame(
        "corporate_actions", release.dataset_versions["corporate_actions"]
    )
    factors = repository.read_frame(
        "adjustment_factors", release.dataset_versions["adjustment_factors"]
    )
    rebuilt, changes = script._rebuild_factors(
        factors,
        script._rewrite_prices(prices),
        script._rewrite_actions(actions),
        source_version="test-price+test-actions",
    )

    assert changes == {
        "removed_rows": 43,
        "retained_total_return_rows_changed": 1_032,
        "retained_split_rows_changed": 0,
        "non_target_economic_rows_changed": 0,
    }
    ntco = rebuilt.loc[rebuilt["security_id"].astype(str).eq(script.SECURITY_ID)]
    assert len(ntco) == 1_032
    assert pd.to_datetime(ntco["session"]).max().date().isoformat() == "2024-02-09"
    assert set(ntco["source_version"].astype(str)) == {
        "test-price+test-actions"
    }


class _TransactionRepository:
    """Small CAS repository used only to exercise release rollback."""

    def __init__(self, root: Path):
        self.root = root
        self.objects = LocalObjectStore(root)
        self.manifests: dict[tuple[str, str], DatasetManifest] = {}
        self.frames: dict[tuple[str, str], pd.DataFrame] = {}
        versions = {
            dataset: f"base-{dataset}" for dataset in script.REQUIRED_DATASETS
        }
        for dataset, version in versions.items():
            metadata = {"preserved": dataset}
            if dataset == "lifecycle_resolutions":
                metadata["evidence_report_sha256"] = "report-hash"
            manifest = DatasetManifest.create(
                dataset,
                version,
                "2026-07-15",
                (),
                metadata=metadata,
            )
            self.manifests[(dataset, version)] = manifest
            manifest_path = f"datasets/{dataset}/versions/{version}/manifest.json"
            self.objects.put(manifest_path, manifest.to_bytes(), if_none_match=True)
            pointer = CurrentPointer.create(manifest, manifest_path)
            self.objects.put(
                self.current_key(dataset), pointer.to_bytes(), if_none_match=True
            )
            self.frames[(dataset, version)] = pd.DataFrame(
                {"base_marker": [dataset]}
            )
        release = DataRelease(
            version="base-release",
            created_at="2026-07-18T00:00:00Z",
            completed_session="2026-07-15",
            dataset_versions=versions,
            quality="valid",
            warnings=(),
        )
        self.objects.put("releases/current.json", release.to_bytes(), if_none_match=True)

    @staticmethod
    def current_key(dataset: str) -> str:
        return script.LocalDatasetRepository.current_key(dataset)

    def current_release(self):
        value = self.objects.get("releases/current.json")
        return DataRelease.from_bytes(value.data), value.etag

    def current_pointer(self, dataset: str):
        value = self.objects.get(self.current_key(dataset))
        return CurrentPointer.from_bytes(value.data), value.etag

    def manifest_for_version(self, dataset: str, version: str):
        return self.manifests[(dataset, version)]

    def write_frame(
        self,
        dataset: str,
        frame: pd.DataFrame,
        *,
        completed_session: str,
        incomplete_action_policy: str,
        metadata: dict,
        expected_pointer_etag: str | None,
        version: str,
    ) -> DatasetWriteResult:
        assert incomplete_action_policy == "block"
        manifest = DatasetManifest.create(
            dataset,
            version,
            completed_session,
            (),
            metadata=metadata,
        )
        manifest_path = f"datasets/{dataset}/versions/{version}/manifest.json"
        self.objects.put(manifest_path, manifest.to_bytes(), if_none_match=True)
        self.objects.put(
            self.current_key(dataset),
            CurrentPointer.create(manifest, manifest_path).to_bytes(),
            if_match=expected_pointer_etag,
        )
        self.manifests[(dataset, version)] = manifest
        self.frames[(dataset, version)] = frame.copy(deep=True)
        return DatasetWriteResult(manifest, ValidationReport(dataset))

    def commit_release(
        self,
        completed_session: str,
        dataset_versions: dict[str, str],
        *,
        quality: str,
        warnings: tuple[str, ...],
        expected_etag: str | None,
    ) -> DataRelease:
        release = DataRelease(
            version="committed-release",
            created_at="2026-07-18T00:00:01Z",
            completed_session=completed_session,
            dataset_versions=dict(dataset_versions),
            quality=quality,
            warnings=warnings,
        )
        self.objects.put(
            "releases/current.json", release.to_bytes(), if_match=expected_etag
        )
        return release


def _transaction_plan(
    repository: _TransactionRepository,
    token: str,
) -> script.PreparedRepair:
    release, release_etag = repository.current_release()
    assert release is not None
    planned = {
        dataset: f"ntco-{token}-{dataset}" for dataset in script.WRITE_DATASETS
    }
    pointers = {
        dataset: repository.current_pointer(dataset)[1]
        for dataset in script.REQUIRED_DATASETS
    }
    frames = {
        dataset: pd.DataFrame({"locked_plan": [dataset]})
        for dataset in script.WRITE_DATASETS
    }
    evidence = script.StagedEvidence(
        source_url=script.OFFICIAL_SOURCE_URL,
        source_sha256="a" * 64,
        content_bytes=1,
        filename=f"{'a' * 64}.html",
        retrieved_at="2026-07-19T00:00:00Z",
        content=b"x",
    )
    summary = {
        "status": "validated_offline_plan",
        "factor_changes": {
            "removed_rows": 43,
            "retained_total_return_rows_changed": 1032,
            "retained_split_rows_changed": 0,
            "non_target_economic_rows_changed": 0,
        },
        "coverage_gate_version": 1,
        "selection_rule": "us_terminal_v1",
        "candidate_set_sha256": "candidate",
        "resolution_set_sha256": "resolution",
        "candidate_count": 1,
        "resolution_count": 1,
        "applied_count": 0,
        "exception_count": 1,
        "open_count": 0,
    }
    return script.PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointers,
        planned_versions=planned,
        frames=frames,
        evidence=evidence,
        summary=summary,
    )


def _control_bytes(repository: _TransactionRepository) -> dict[str, bytes]:
    keys = ["releases/current.json"] + [
        repository.current_key(dataset) for dataset in script.REQUIRED_DATASETS
    ]
    return {key: repository.objects.get(key).data for key in keys}


def test_apply_failure_after_release_commit_rolls_back_every_control_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _TransactionRepository(tmp_path / "repo")
    caller = _transaction_plan(repository, "caller")
    before = _control_bytes(repository)
    monkeypatch.setattr(
        script,
        "prepare_repair",
        lambda *_args, **_kwargs: _transaction_plan(repository, "locked"),
    )
    monkeypatch.setattr(script, "_persist_official", lambda *_args, **_kwargs: None)

    def fail(stage: str) -> None:
        if stage == "after_release_commit":
            raise RuntimeError("synthetic post-commit failure")

    with pytest.raises(RuntimeError, match="synthetic post-commit failure"):
        script.apply_repair(
            repository,
            caller,
            hints_path=HINTS_PATH,
            evidence_dir=tmp_path / "evidence",
            inject_failure=fail,
        )

    assert _control_bytes(repository) == before
    journals = tuple(
        (repository.root / script.TRANSACTION_DIR).glob("*.json")
    )
    assert len(journals) == 1
    journal = json.loads(journals[0].read_text(encoding="utf-8"))
    assert journal["status"] == "rolled_back"
    assert journal["rollback_errors"] == []
    assert not (repository.root / script.RECOVERY_DIR).exists()
