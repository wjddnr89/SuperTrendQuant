from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest

from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.repository import LocalDatasetRepository


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_fisv_confirmation_evidence.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_fisv_confirmation_evidence", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


COMPLETED_SESSION = "2026-07-15"
RETRIEVED_AT = "2026-07-18T00:00:00Z"


def _primary_content() -> bytes:
    return (
        "<html><body>trading will begin on Nasdaq at market open on or about "
        'November 11, 2025 and trade under the symbols, "FISV"</body></html>'
    ).encode()


def _confirmation_content() -> bytes:
    return (
        "<html><body>Fiserv, Inc. quarter ended March 31, 2026 "
        "Trading Symbol(s) FISV The Nasdaq Stock Market LLC</body></html>"
    ).encode()


@pytest.fixture
def evidence_constants(monkeypatch: pytest.MonkeyPatch):
    content = _primary_content()
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(script, "PRIMARY_SOURCE_HASH", digest)
    monkeypatch.setattr(script, "PRIMARY_EXACT_BYTES", len(content))
    reviewed = dict(script.REVIEWED_EXTRACTION)
    reviewed["source_hash"] = digest
    monkeypatch.setattr(script, "REVIEWED_EXTRACTION", reviewed)
    return content, digest


def _action(primary_hash: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": script.EVENT_ID,
                "security_id": script.SECURITY_ID,
                "action_type": script.ACTION_TYPE,
                "effective_date": script.EFFECTIVE_DATE,
                "ex_date": script.EFFECTIVE_DATE,
                "announcement_date": "2025-10-29",
                "record_date": "",
                "payment_date": "",
                "cash_amount": None,
                "ratio": None,
                "currency": "USD",
                "new_security_id": script.SECURITY_ID,
                "new_symbol": script.NEW_SYMBOL,
                "official": True,
                "source_url": script.PRIMARY_SOURCE_URL,
                "source_kind": "official_crosscheck",
                "source": "sec_edgar+stored_price_crosscheck",
                "retrieved_at": RETRIEVED_AT,
                "source_hash": primary_hash,
                "metadata": None,
            }
        ]
    )


def _archive_row(
    digest: str,
    url: str,
    object_path: str,
    *,
    content_type: str,
) -> dict[str, object]:
    return {
        "archive_id": digest,
        "dataset": "sec_edgar_filing",
        "object_path": object_path,
        "content_type": content_type,
        "effective_date": COMPLETED_SESSION,
        "source": "sec_edgar_filing",
        "retrieved_at": RETRIEVED_AT,
        "source_hash": digest,
        "source_url": url,
    }


def _write_confirmation_report(
    evidence_dir: Path,
    *,
    content: bytes | None = None,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, Path]:
    content = _confirmation_content() if content is None else content
    digest = hashlib.sha256(content).hexdigest()
    filename = f"{digest}.html"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / filename
    path.write_bytes(content)
    report = {
        "schema": script.EVIDENCE_SCHEMA,
        "status": "collected",
        "evidence": {
            "label": "fisv_nasdaq_post_transition_confirmation",
            "source_url": script.CONFIRMATION_SOURCE_URL,
            "source_hash": digest,
            "size": len(content),
            "filename": filename,
            "retrieved_at": RETRIEVED_AT,
            "form": "10-Q",
            "period_end": "2026-03-31",
        },
        "http_attempts_total": 1,
        "eodhd_calls": 0,
        "r2_accessed": False,
    }
    if mutate is not None:
        mutate(report)
    (evidence_dir / script.EVIDENCE_REPORT).write_text(
        json.dumps(report), encoding="utf-8"
    )
    return digest, path


def _repository(
    root: Path,
    primary_content: bytes,
    primary_hash: str,
    *,
    mutate_action: Callable[[pd.DataFrame], None] | None = None,
    mutate_archive: Callable[[dict[str, object]], None] | None = None,
) -> LocalDatasetRepository:
    repository = LocalDatasetRepository(root)
    primary_object = f"archives/{COMPLETED_SESSION}/{primary_hash}.txt.gz"
    primary_path = root / primary_object
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    primary_path.write_bytes(gzip.compress(primary_content, mtime=0))
    unrelated_content = b"unrelated"
    unrelated_hash = hashlib.sha256(unrelated_content).hexdigest()
    unrelated_object = f"archives/{COMPLETED_SESSION}/{unrelated_hash}.txt.gz"
    unrelated_path = root / unrelated_object
    unrelated_path.write_bytes(gzip.compress(unrelated_content, mtime=0))
    primary_row = _archive_row(
        primary_hash,
        script.PRIMARY_SOURCE_URL,
        primary_object,
        content_type="text/plain",
    )
    if mutate_archive is not None:
        mutate_archive(primary_row)
    archive = pd.DataFrame(
        [
            primary_row,
            _archive_row(
                unrelated_hash,
                "https://www.sec.gov/Archives/edgar/data/1/unrelated.txt",
                unrelated_object,
                content_type="text/plain",
            ),
        ]
    )
    actions = _action(primary_hash)
    if mutate_action is not None:
        mutate_action(actions)
    archive_result = repository.write_frame(
        "source_archive",
        archive,
        completed_session=COMPLETED_SESSION,
        version="base-source-archive",
    )
    action_result = repository.write_frame(
        "corporate_actions",
        actions,
        completed_session=COMPLETED_SESSION,
        version="base-corporate-actions",
        incomplete_action_policy="block",
    )
    master_result = repository.write_frame(
        "security_master",
        pd.DataFrame(
            [
                {
                    "security_id": script.SECURITY_ID,
                    "primary_symbol": script.NEW_SYMBOL,
                    "name": "Fiserv, Inc.",
                    "exchange": "NASDAQ",
                    "asset_type": "STOCK",
                    "currency": "USD",
                    "country": "US",
                    "active_from": "2015-01-02",
                    "active_to": "",
                    "source": "fixture",
                    "retrieved_at": RETRIEVED_AT,
                    "source_hash": hashlib.sha256(b"master").hexdigest(),
                }
            ]
        ),
        completed_session=COMPLETED_SESSION,
        version="base-security-master",
    )
    history_result = repository.write_frame(
        "symbol_history",
        pd.DataFrame(
            [
                {
                    "security_id": script.SECURITY_ID,
                    "symbol": "FI",
                    "exchange": "NYSE",
                    "effective_from": "2023-06-07",
                    "effective_to": "2025-11-10",
                    "source": "fixture",
                    "retrieved_at": RETRIEVED_AT,
                    "source_hash": hashlib.sha256(b"history-fi").hexdigest(),
                },
                {
                    "security_id": script.SECURITY_ID,
                    "symbol": script.NEW_SYMBOL,
                    "exchange": "NASDAQ",
                    "effective_from": script.EFFECTIVE_DATE,
                    "effective_to": "",
                    "source": "fixture",
                    "retrieved_at": RETRIEVED_AT,
                    "source_hash": hashlib.sha256(b"history-fisv").hexdigest(),
                },
            ]
        ),
        completed_session=COMPLETED_SESSION,
        version="base-symbol-history",
    )
    repository.commit_release(
        COMPLETED_SESSION,
        {
            "source_archive": archive_result.manifest.version,
            "corporate_actions": action_result.manifest.version,
            "security_master": master_result.manifest.version,
            "symbol_history": history_result.manifest.version,
        },
        quality=DataQuality.DEGRADED,
        warnings=("fixture warning preserved",),
    )
    return repository


def _tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_readiness_plan_blocks_without_fetch_and_writes_nothing(
    tmp_path: Path, evidence_constants
):
    primary_content, primary_hash = evidence_constants
    repository = _repository(tmp_path, primary_content, primary_hash)
    before = _tree(tmp_path)

    result = script.readiness_plan(
        repository, evidence_dir=tmp_path / "missing-evidence"
    )

    assert result["status"] == "blocked_pending_authorized_one_url_fetch"
    assert result["source_url"] == script.CONFIRMATION_SOURCE_URL
    assert result["network_accessed"] is False
    assert result["writes_performed"] is False
    assert result["eodhd_calls"] == 0
    assert result["r2_accessed"] is False
    assert _tree(tmp_path) == before


def test_plan_verifies_both_roles_and_only_adds_confirmation_archive_row(
    tmp_path: Path, evidence_constants
):
    primary_content, primary_hash = evidence_constants
    repository = _repository(tmp_path, primary_content, primary_hash)
    evidence_dir = tmp_path / "staged"
    confirmation_hash, _path = _write_confirmation_report(evidence_dir)
    before = _tree(tmp_path)
    release, _ = repository.current_release()
    assert release is not None
    archive_before = repository.read_frame(
        "source_archive", release.dataset_versions["source_archive"]
    )

    prepared = script.prepare_repair(repository, evidence_dir=evidence_dir)

    assert prepared.summary["status"] == "validated_offline_plan"
    assert prepared.summary["source_archive_rows_added"] == 1
    assert prepared.summary["corporate_action_changed"] is False
    assert prepared.summary["network_accessed"] is False
    assert _tree(tmp_path) == before
    assert len(prepared.frame) == len(archive_before) + 1
    row = prepared.frame.loc[
        prepared.frame["archive_id"].astype(str).eq(confirmation_hash)
    ].iloc[0]
    assert row["source_url"] == script.CONFIRMATION_SOURCE_URL
    assert row["source_hash"] == confirmation_hash
    roles = prepared.summary["evidence_roles"]
    assert roles["transition_schedule"]["source_hash"] == primary_hash
    assert (
        roles["post_transition_confirmation"]["source_hash"]
        == confirmation_hash
    )
    assert roles["primary_action_source_unchanged"] is True
    assert prepared.summary["reviewed_nonterminal_extraction"]["source_hash"] == primary_hash


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("effective_date", "2025-11-10"),
        ("new_symbol", "FI"),
        ("source_url", "https://www.sec.gov/Archives/wrong.txt"),
        ("source_kind", "official_filing"),
    ],
)
def test_plan_rejects_any_primary_action_change(
    tmp_path: Path, evidence_constants, field: str, value: object
):
    primary_content, primary_hash = evidence_constants

    def mutate(actions: pd.DataFrame) -> None:
        actions.loc[0, field] = value

    repository = _repository(
        tmp_path, primary_content, primary_hash, mutate_action=mutate
    )
    evidence_dir = tmp_path / "staged"
    _write_confirmation_report(evidence_dir)

    with pytest.raises(ValueError, match="corporate action differs"):
        script.prepare_repair(repository, evidence_dir=evidence_dir)


def test_exact_action_rejects_nonofficial_row(evidence_constants):
    _primary_content_value, primary_hash = evidence_constants
    actions = _action(primary_hash)
    actions.loc[0, "official"] = False

    with pytest.raises(ValueError, match="corporate action differs"):
        script._exact_action(actions)


def test_plan_rejects_primary_archive_binding_or_payload_tamper(
    tmp_path: Path, evidence_constants
):
    primary_content, primary_hash = evidence_constants

    def mutate(row: dict[str, object]) -> None:
        row["source_url"] = "https://www.sec.gov/Archives/wrong.txt"

    repository = _repository(
        tmp_path, primary_content, primary_hash, mutate_archive=mutate
    )
    evidence_dir = tmp_path / "staged"
    _write_confirmation_report(evidence_dir)

    with pytest.raises(ValueError, match="archive URL/hash binding changed"):
        script.prepare_repair(repository, evidence_dir=evidence_dir)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report["evidence"].__setitem__(
            "source_url", "https://www.sec.gov/Archives/wrong.htm"
        ),
        lambda report: report["evidence"].__setitem__("form", "8-K"),
        lambda report: report.__setitem__("http_attempts_total", 2),
    ],
)
def test_plan_rejects_confirmation_report_tamper(
    tmp_path: Path, evidence_constants, mutation
):
    primary_content, primary_hash = evidence_constants
    repository = _repository(tmp_path, primary_content, primary_hash)
    evidence_dir = tmp_path / "staged"
    _write_confirmation_report(evidence_dir, mutate=mutation)

    with pytest.raises(ValueError, match="FISV confirmation"):
        script.prepare_repair(repository, evidence_dir=evidence_dir)


def test_plan_rejects_confirmation_without_nasdaq_fisv_terms(
    tmp_path: Path, evidence_constants
):
    primary_content, primary_hash = evidence_constants
    repository = _repository(tmp_path, primary_content, primary_hash)
    evidence_dir = tmp_path / "staged"
    _write_confirmation_report(
        evidence_dir,
        content=b"<html>Fiserv, Inc. March 31, 2026 Trading Symbol(s) FISV</html>",
    )

    with pytest.raises(ValueError, match="lacks reviewed official term"):
        script.prepare_repair(repository, evidence_dir=evidence_dir)


def test_apply_writes_only_source_archive_and_keeps_primary_action_source(
    tmp_path: Path, evidence_constants
):
    primary_content, primary_hash = evidence_constants
    repository = _repository(tmp_path, primary_content, primary_hash)
    evidence_dir = tmp_path / "staged"
    confirmation_hash, _path = _write_confirmation_report(evidence_dir)
    before_release, _ = repository.current_release()
    assert before_release is not None
    before_action_version = before_release.dataset_versions["corporate_actions"]
    prepared = script.prepare_repair(repository, evidence_dir=evidence_dir)

    result = script.apply_repair(
        repository, prepared, evidence_dir=evidence_dir
    )

    after_release, _ = repository.current_release()
    assert after_release is not None
    assert result["status"] == "applied"
    assert result["writes_performed"] is True
    assert after_release.dataset_versions["corporate_actions"] == before_action_version
    assert (
        after_release.dataset_versions["source_archive"]
        != before_release.dataset_versions["source_archive"]
    )
    actions = repository.read_frame("corporate_actions", before_action_version)
    row = actions.loc[actions["event_id"].astype(str).eq(script.EVENT_ID)].iloc[0]
    assert row["source_url"] == script.PRIMARY_SOURCE_URL
    assert row["source_hash"] == primary_hash
    archive = repository.read_frame(
        "source_archive", after_release.dataset_versions["source_archive"]
    )
    confirmation = archive.loc[
        archive["archive_id"].astype(str).eq(confirmation_hash)
    ]
    assert len(confirmation) == 1
    replay = script.prepare_repair(repository, evidence_dir=evidence_dir)
    assert replay.summary["status"] == "already_archived"
