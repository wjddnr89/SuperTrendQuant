from __future__ import annotations

import dataclasses
import gzip
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.repository import LocalDatasetRepository


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_aa_spinoff_provenance.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_aa_spinoff_provenance", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


COMPLETED_SESSION = "2016-11-02"
PRICE_VERSION = "base-daily-price"
ACTION_VERSION = "base-corporate-actions"
FACTOR_VERSION = "base-adjustment-factors"
ARCHIVE_VERSION = "base-source-archive"
FIXTURE_WARNING = "fixture warning must survive"
PAYLOAD = b"reviewed Alcoa 2018 Form 10-K evidence"


@dataclass(frozen=True)
class Fixture:
    repository: LocalDatasetRepository
    evidence: script.EvidenceSpec


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source(label: str) -> dict[str, object]:
    return {
        "source": label,
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": _sha(label.encode()),
    }


def _evidence(payload: bytes = PAYLOAD) -> script.EvidenceSpec:
    return script.EvidenceSpec(
        old_url="https://www.sec.gov/Archives/conditional-aa-release.htm",
        old_hash=_sha(b"conditional AA release"),
        confirmatory_url=(
            "https://www.sec.gov/Archives/edgar/data/4281/"
            "000000428119000031/form10k_4q18.htm"
        ),
        confirmatory_hash=_sha(payload),
        archive_dataset="official_identity_evidence_raw",
        archive_source="official_identity_evidence_raw",
        content_type="text/html",
        retrieved_at="2026-07-18T02:19:25.834601Z",
        extension="html",
    )


def _aa_action(evidence: script.EvidenceSpec) -> dict[str, object]:
    return {
        "event_id": script.AA_EVENT_ID,
        "security_id": script.PARENT_SECURITY_ID,
        "action_type": "spinoff",
        "effective_date": script.EFFECTIVE_DATE,
        "ex_date": script.EFFECTIVE_DATE,
        "announcement_date": "",
        "record_date": script.RECORD_DATE,
        "payment_date": script.EFFECTIVE_DATE,
        "cash_amount": None,
        "ratio": script.RATIO,
        "currency": "USD",
        "new_security_id": script.AA_SECURITY_ID,
        "new_symbol": "AA",
        "official": True,
        "source_url": evidence.old_url,
        "source_kind": "official_filing",
        "source": "official_identity_repair",
        "retrieved_at": script.ACTION_RETRIEVED_AT,
        "source_hash": evidence.old_hash,
        "metadata": script.EXPECTED_METADATA,
    }


def _unrelated_action() -> dict[str, object]:
    return {
        "event_id": _sha(b"unrelated cash dividend"),
        "security_id": script.PARENT_SECURITY_ID,
        "action_type": "cash_dividend",
        "effective_date": script.EFFECTIVE_DATE,
        "ex_date": script.EFFECTIVE_DATE,
        "announcement_date": "2016-10-01",
        "record_date": "2016-11-03",
        "payment_date": "2016-11-10",
        "cash_amount": 0.5,
        "ratio": None,
        "currency": "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": True,
        "source_url": "https://www.sec.gov/Archives/unrelated-dividend.htm",
        "source_kind": "official_filing",
        "source": "fixture",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": _sha(b"unrelated dividend evidence"),
        "metadata": None,
    }


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "security_id": script.PARENT_SECURITY_ID,
                "session": session,
                "open": close - 1.0,
                "high": close + 1.0,
                "low": close - 2.0,
                "close": close,
                "volume": 1_000_000 + offset,
                "currency": "USD",
                **_source(f"price-{session}"),
            }
            for offset, (session, close) in enumerate(
                (
                    ("2016-10-31", 100.0),
                    ("2016-11-01", 101.0),
                    (COMPLETED_SESSION, 102.0),
                )
            )
        ]
    )


def _archive_row(evidence: script.EvidenceSpec) -> dict[str, object]:
    return {
        "archive_id": evidence.confirmatory_hash,
        "dataset": evidence.archive_dataset,
        "object_path": evidence.object_path(COMPLETED_SESSION),
        "content_type": evidence.content_type,
        "effective_date": COMPLETED_SESSION,
        "source": evidence.archive_source,
        "retrieved_at": evidence.retrieved_at,
        "source_hash": evidence.confirmatory_hash,
        "source_url": evidence.confirmatory_url,
    }


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _current_frame(
    repository: LocalDatasetRepository, dataset: str
) -> pd.DataFrame:
    release, _ = repository.current_release()
    assert release is not None
    return repository.read_frame(dataset, release.dataset_versions[dataset])


def _build_repository(
    root: Path,
    *,
    archived_payload: bytes = PAYLOAD,
    mutate_archive_row: Callable[[dict[str, object]], None] | None = None,
    mutate_action: Callable[[dict[str, object]], None] | None = None,
    mutate_factors: Callable[[pd.DataFrame], None] | None = None,
) -> Fixture:
    repository = LocalDatasetRepository(root)
    evidence = _evidence(PAYLOAD)
    archive_path = root / evidence.object_path(COMPLETED_SESSION)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(gzip.compress(archived_payload, mtime=0))

    action = _aa_action(evidence)
    if mutate_action is not None:
        mutate_action(action)
    actions = pd.DataFrame([action, _unrelated_action()])
    prices = _prices()
    lineage = script._adjustment_source_version(
        PRICE_VERSION, ACTION_VERSION
    )
    factors = build_adjustment_factors(
        prices, actions, source_version=lineage
    )
    if mutate_factors is not None:
        mutate_factors(factors)

    archive_row = _archive_row(evidence)
    if mutate_archive_row is not None:
        mutate_archive_row(archive_row)
    archive = pd.DataFrame([archive_row])

    price_result = repository.write_frame(
        "daily_price_raw",
        prices,
        completed_session=COMPLETED_SESSION,
        version=PRICE_VERSION,
    )
    action_result = repository.write_frame(
        "corporate_actions",
        actions,
        completed_session=COMPLETED_SESSION,
        incomplete_action_policy="block",
        version=ACTION_VERSION,
    )
    factor_result = repository.write_frame(
        "adjustment_factors",
        factors,
        completed_session=COMPLETED_SESSION,
        version=FACTOR_VERSION,
    )
    archive_result = repository.write_frame(
        "source_archive",
        archive,
        completed_session=COMPLETED_SESSION,
        version=ARCHIVE_VERSION,
    )
    repository.commit_release(
        COMPLETED_SESSION,
        {
            "daily_price_raw": price_result.manifest.version,
            "corporate_actions": action_result.manifest.version,
            "adjustment_factors": factor_result.manifest.version,
            "source_archive": archive_result.manifest.version,
        },
        quality=DataQuality.DEGRADED,
        warnings=(FIXTURE_WARNING,),
    )
    return Fixture(repository=repository, evidence=evidence)


def _economic_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame[
            [
                "security_id",
                "session",
                "split_factor",
                "total_return_factor",
            ]
        ]
        .sort_values(["security_id", "session"])
        .reset_index(drop=True)
    )


def test_exact_action_provenance_only_and_full_factor_rebind(
    tmp_path: Path,
) -> None:
    fixture = _build_repository(tmp_path)
    current_actions = _current_frame(fixture.repository, "corporate_actions")
    current_factors = _current_frame(fixture.repository, "adjustment_factors")

    prepared = script.prepare_repair(
        fixture.repository, evidence=fixture.evidence
    )

    repaired_actions = prepared.frames["corporate_actions"]
    repaired_factors = prepared.frames["adjustment_factors"]
    assert prepared.summary["status"] == "validated_offline_plan"
    assert prepared.summary["corporate_action_rows_changed"] == 1
    assert prepared.summary["adjustment_factor_economic_rows_changed"] == 0
    assert prepared.summary["adjustment_factor_provenance_rows_rebound"] == len(
        current_factors
    )
    for column in current_actions.columns:
        if column not in {"source_url", "source_hash"}:
            pd.testing.assert_series_equal(
                current_actions[column], repaired_actions[column]
            )
    changed_cells = (
        current_actions[["source_url", "source_hash"]]
        .astype(str)
        .ne(repaired_actions[["source_url", "source_hash"]].astype(str))
    )
    assert int(changed_cells.to_numpy().sum()) == 2
    aa = repaired_actions.loc[
        repaired_actions["event_id"].astype(str).eq(script.AA_EVENT_ID)
    ].iloc[0]
    assert aa["source_url"] == fixture.evidence.confirmatory_url
    assert aa["source_hash"] == fixture.evidence.confirmatory_hash

    pd.testing.assert_frame_equal(
        _economic_columns(current_factors),
        _economic_columns(repaired_factors),
        check_exact=True,
    )
    expected_lineage = script._adjustment_source_version(
        PRICE_VERSION, prepared.planned_versions["corporate_actions"]
    )
    assert set(repaired_factors["source_version"].astype(str)) == {
        expected_lineage
    }
    assert set(repaired_factors["source_hash"].astype(str)) == {
        expected_lineage
    }
    assert set(repaired_factors["source"].astype(str)) == {"derived"}
    assert set(repaired_factors["calculated_at"].astype(str)) == {
        script.REPAIR_REVIEWED_AT
    }
    assert set(repaired_factors["retrieved_at"].astype(str)) == {
        script.REPAIR_REVIEWED_AT
    }


def test_plan_is_strictly_read_only(tmp_path: Path) -> None:
    fixture = _build_repository(tmp_path)
    before = _tree_hashes(tmp_path)

    prepared = script.prepare_repair(
        fixture.repository, evidence=fixture.evidence
    )

    assert prepared.summary["status"] == "validated_offline_plan"
    assert prepared.summary["network_accessed"] is False
    assert prepared.summary["eodhd_calls"] == 0
    assert prepared.summary["r2_accessed"] is False
    assert prepared.summary["write_datasets"] == [
        "corporate_actions",
        "adjustment_factors",
    ]
    assert _tree_hashes(tmp_path) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("archive_id", "1" * 64),
        ("dataset", "wrong_dataset"),
        ("object_path", "archives/wrong/evidence.html.gz"),
        ("content_type", "application/json"),
        ("effective_date", "2016-11-01"),
        ("source", "wrong_source"),
        ("retrieved_at", "2026-07-18T02:19:25Z"),
        ("source_hash", "2" * 64),
        ("source_url", "https://www.sec.gov/Archives/wrong.htm"),
    ],
)
def test_plan_rejects_any_confirmatory_archive_row_change(
    tmp_path: Path, field: str, value: object
) -> None:
    def mutate(row: dict[str, object]) -> None:
        row[field] = value

    fixture = _build_repository(tmp_path, mutate_archive_row=mutate)

    with pytest.raises(ValueError, match="source_archive row changed"):
        script.prepare_repair(fixture.repository, evidence=fixture.evidence)


def test_plan_rejects_tampered_decompressed_archive_payload(
    tmp_path: Path,
) -> None:
    fixture = _build_repository(
        tmp_path, archived_payload=b"tampered Form 10-K bytes"
    )

    with pytest.raises(ValueError, match="payload hash changed"):
        script.prepare_repair(fixture.repository, evidence=fixture.evidence)


def test_archive_path_cannot_escape_repository(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes repository"):
        script._safe_archive_path(tmp_path, "../outside.html.gz")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_url", "https://www.sec.gov/Archives/other-aa-source.htm"),
        ("source_hash", "f" * 64),
    ],
)
def test_plan_rejects_unreviewed_conditional_action_source_pair(
    tmp_path: Path, field: str, value: str
) -> None:
    def mutate(action: dict[str, object]) -> None:
        action[field] = value

    fixture = _build_repository(tmp_path, mutate_action=mutate)

    with pytest.raises(ValueError, match="neither the reviewed conditional source"):
        script.prepare_repair(fixture.repository, evidence=fixture.evidence)


def test_plan_rejects_any_economic_factor_change(tmp_path: Path) -> None:
    def mutate(factors: pd.DataFrame) -> None:
        factors.loc[factors.index[0], "split_factor"] = 1.25

    fixture = _build_repository(tmp_path, mutate_factors=mutate)

    with pytest.raises(ValueError, match="would change adjustment economics"):
        script.prepare_repair(fixture.repository, evidence=fixture.evidence)


def test_apply_writes_only_actions_and_factors_then_replays_idempotently(
    tmp_path: Path,
) -> None:
    fixture = _build_repository(tmp_path)
    before_release, _ = fixture.repository.current_release()
    assert before_release is not None
    before_pointers = {
        dataset: fixture.repository.current_pointer(dataset)[0]
        for dataset in script.REQUIRED_DATASETS
    }
    before_factors = _current_frame(fixture.repository, "adjustment_factors")
    prepared = script.prepare_repair(
        fixture.repository, evidence=fixture.evidence
    )

    result = script.apply_repair(fixture.repository, prepared)

    after_release, _ = fixture.repository.current_release()
    assert after_release is not None
    assert result["status"] == "applied"
    assert result["writes_performed"] is True
    assert after_release.quality == before_release.quality
    assert after_release.warnings == before_release.warnings
    for dataset in ("daily_price_raw", "source_archive"):
        pointer, _ = fixture.repository.current_pointer(dataset)
        assert pointer is not None and before_pointers[dataset] is not None
        assert pointer.version == before_pointers[dataset].version
        assert (
            after_release.dataset_versions[dataset]
            == before_release.dataset_versions[dataset]
        )
    for dataset in script.WRITE_DATASETS:
        assert (
            after_release.dataset_versions[dataset]
            == prepared.planned_versions[dataset]
        )
        assert (
            after_release.dataset_versions[dataset]
            != before_release.dataset_versions[dataset]
        )

    after_actions = _current_frame(fixture.repository, "corporate_actions")
    aa = after_actions.loc[
        after_actions["event_id"].astype(str).eq(script.AA_EVENT_ID)
    ].iloc[0]
    assert aa["source_url"] == fixture.evidence.confirmatory_url
    assert aa["source_hash"] == fixture.evidence.confirmatory_hash
    after_factors = _current_frame(fixture.repository, "adjustment_factors")
    pd.testing.assert_frame_equal(
        _economic_columns(before_factors),
        _economic_columns(after_factors),
        check_exact=True,
    )
    lineage = script._adjustment_source_version(
        PRICE_VERSION, after_release.dataset_versions["corporate_actions"]
    )
    assert set(after_factors["source_version"].astype(str)) == {lineage}
    factor_manifest = fixture.repository.manifest_for_version(
        "adjustment_factors",
        after_release.dataset_versions["adjustment_factors"],
    )
    assert factor_manifest.metadata["source_version"] == lineage
    assert (
        factor_manifest.metadata["source_corporate_actions_version"]
        == after_release.dataset_versions["corporate_actions"]
    )

    replay = script.prepare_repair(
        fixture.repository, evidence=fixture.evidence
    )
    before_replay = _tree_hashes(tmp_path)
    replay_result = script.apply_repair(fixture.repository, replay)
    assert replay.summary["status"] == "already_repaired"
    assert replay.frames == {}
    assert replay_result["writes_performed"] is False
    assert _tree_hashes(tmp_path) == before_replay


def test_apply_fails_closed_on_stale_pointer_cas(tmp_path: Path) -> None:
    fixture = _build_repository(tmp_path)
    old_release = fixture.repository.objects.get("releases/current.json").data
    old_pointers = {
        dataset: fixture.repository.objects.get(
            fixture.repository.current_key(dataset)
        ).data
        for dataset in script.WRITE_DATASETS
    }
    prepared = script.prepare_repair(
        fixture.repository, evidence=fixture.evidence
    )
    stale_etags = dict(prepared.pointer_etags)
    stale_etags["corporate_actions"] = "stale-etag"
    stale = dataclasses.replace(prepared, pointer_etags=stale_etags)

    with pytest.raises(RuntimeError, match="pointer changed after AA planning"):
        script.apply_repair(fixture.repository, stale)

    assert (
        fixture.repository.objects.get("releases/current.json").data
        == old_release
    )
    for dataset, old in old_pointers.items():
        assert (
            fixture.repository.objects.get(
                fixture.repository.current_key(dataset)
            ).data
            == old
        )


@pytest.mark.parametrize(
    "failure_stage",
    [
        "after_write:corporate_actions",
        "after_write:adjustment_factors",
        "after_release_commit",
    ],
)
def test_apply_rolls_back_release_and_both_pointers(
    tmp_path: Path, failure_stage: str
) -> None:
    fixture = _build_repository(tmp_path)
    old_release = fixture.repository.objects.get("releases/current.json").data
    old_pointers = {
        dataset: fixture.repository.objects.get(
            fixture.repository.current_key(dataset)
        ).data
        for dataset in script.WRITE_DATASETS
    }
    prepared = script.prepare_repair(
        fixture.repository, evidence=fixture.evidence
    )

    def inject(stage: str) -> None:
        if stage == failure_stage:
            raise RuntimeError(f"injected:{stage}")

    with pytest.raises(RuntimeError, match=f"injected:{failure_stage}"):
        script.apply_repair(
            fixture.repository, prepared, inject_failure=inject
        )

    assert (
        fixture.repository.objects.get("releases/current.json").data
        == old_release
    )
    for dataset, old in old_pointers.items():
        assert (
            fixture.repository.objects.get(
                fixture.repository.current_key(dataset)
            ).data
            == old
        )
    journals = list((tmp_path / script.TRANSACTION_DIR).glob("*.json"))
    assert len(journals) == 1
    assert json.loads(journals[0].read_text())["status"] == "rolled_back"
    recovery = tmp_path / script.RECOVERY_DIR
    assert not recovery.exists() or not tuple(recovery.glob("*.json"))
    assert (
        script.prepare_repair(
            fixture.repository, evidence=fixture.evidence
        ).summary["status"]
        == "validated_offline_plan"
    )


def test_cli_defaults_to_read_only_plan() -> None:
    args = script._parse_args([])
    assert args.apply is False
    assert args.cache_root == script.DEFAULT_CACHE_ROOT
