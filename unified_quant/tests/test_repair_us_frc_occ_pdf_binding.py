from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.repository import LocalDatasetRepository


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_frc_occ_pdf_binding.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_frc_occ_pdf_binding", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


COMPLETED_SESSION = "2026-07-15"
RETRIEVED_AT = "2026-07-18T14:11:49.785762Z"
REVIEWED_AT = "2026-07-19T03:00:00Z"
IMPORTED_AT = "2026-07-19T03:01:00Z"

PAGE_ONE = [
    "#52352",
    "Date: May 2, 2023",
    "Subject: First Republic Bank - Symbol Change",
    "Option Symbol: FRC",
    "New Symbol: FRCB",
    "Date: 05/03/2023",
    "First Republic Bank (FRC) will change its trading symbol to FRCB",
    "effective May 3, 2023, due to the listing of the company on an OTC market.",
    "As a result, option symbol FRC will change to FRCB effective at the",
    "opening of business on May 3, 2023.",
    "Strike prices and all other option terms will not change.",
    "Clearing Member input to OCC must use the new option symbol FRCB",
    "commencing May 3, 2023.",
    "Date: May 3, 2023",
    "Option Symbol: FRC changes to FRCB",
    "Underlying Security: FRC changes to FRCB",
    "Contract Multiplier: 1",
    "Strike Divisor: 1",
    "New Multiplier: 100",
    "Deliverable Per Contract:",
    "100 First Republic Bank (FRCB) Common Shares",
    "CUSIP: 33616C100",
    "Disclaimer",
    "This Information Memo provides an unofficial summary of the terms of",
    "corporate events affecting listed options or futures.",
]

PAGE_TWO = [
    "The determination to adjust options and the nature of any adjustment",
    "is made by OCC pursuant to OCC By-Laws.",
    "ALL CLEARING MEMBERS ARE REQUESTED TO IMMEDIATELY ADVISE ALL BRANCH OFFICES",
    "AND CORRESPONDENTS ON THE ABOVE.",
    "For questions regarding this memo, please email options@theocc.com.",
]


def _pdf_bytes(pages: list[list[str]]) -> bytes:
    page_count = len(pages)
    font_id = 3 + page_count * 2
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            "<< /Type /Pages /Kids ["
            + " ".join(f"{3 + index * 2} 0 R" for index in range(page_count))
            + f"] /Count {page_count} >>"
        ).encode("ascii"),
        font_id: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for index, lines in enumerate(pages):
        page_id = 3 + index * 2
        content_id = page_id + 1
        operators = ["BT", "/F1 9 Tf", "42 770 Td", "12 TL"]
        for line in lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            operators.extend((f"({escaped}) Tj", "T*"))
        operators.append("ET")
        stream = ("\n".join(operators) + "\n").encode("ascii")
        objects[page_id] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode("ascii")
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"endstream"
        )
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0] * (font_id + 1)
    for object_id in range(1, font_id + 1):
        offsets[object_id] = len(output)
        output.extend(f"{object_id} 0 obj\n".encode("ascii"))
        output.extend(objects[object_id])
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {font_id + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for object_id in range(1, font_id + 1):
        output.extend(f"{offsets[object_id]:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {font_id + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)


def _write_pdf(
    path: Path,
    *,
    page_one: list[str] | None = None,
    page_two: list[str] | None = None,
) -> Path:
    path.write_bytes(
        _pdf_bytes(
            [
                list(PAGE_ONE if page_one is None else page_one),
                list(PAGE_TWO if page_two is None else page_two),
            ]
        )
    )
    return path


def _evidence(path: Path):
    script.REVIEWED_OCC_PDF_SHA256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return script.load_occ_pdf(
        path,
        reviewed_by="Codex PDF review",
        reviewed_at=REVIEWED_AT,
        official_origin_confirmed=True,
    )


def _legacy_action() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": script.EVENT_ID,
                "security_id": script.SECURITY_ID,
                "action_type": "ticker_change",
                "effective_date": script.EFFECTIVE_DATE,
                "ex_date": script.EFFECTIVE_DATE,
                "announcement_date": script.ANNOUNCEMENT_DATE,
                "record_date": "",
                "payment_date": "",
                "cash_amount": None,
                "ratio": None,
                "currency": "USD",
                "new_security_id": script.SECURITY_ID,
                "new_symbol": script.NEW_SYMBOL,
                "official": True,
                "source_url": script.OFFICIAL_OCC_URL,
                "source_kind": "clearing_notice_reviewed_extraction",
                "source": "occ_reviewed_memo_extraction",
                "retrieved_at": RETRIEVED_AT,
                "source_hash": script.LEGACY_REVIEWED_EXTRACTION_SHA256,
                "metadata": json.dumps(
                    {
                        "cusip": script.CUSIP,
                        "memo_number": script.MEMO_NUMBER,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        ]
    )


def _archive_row(
    digest: str,
    *,
    dataset: str,
    suffix: str,
    content_type: str,
    source_url: str,
    retrieved_at: str = RETRIEVED_AT,
) -> dict[str, object]:
    return {
        "archive_id": digest,
        "dataset": dataset,
        "object_path": f"archives/{COMPLETED_SESSION}/{digest}.{suffix}.gz",
        "content_type": content_type,
        "effective_date": COMPLETED_SESSION,
        "source": dataset,
        "retrieved_at": retrieved_at,
        "source_hash": digest,
        "source_url": source_url,
    }


def _repository(
    root: Path,
    *,
    mutate_action: Callable[[pd.DataFrame], None] | None = None,
    mutate_legacy_payload: Callable[[bytes], bytes] | None = None,
) -> LocalDatasetRepository:
    repository = LocalDatasetRepository(root)
    legacy_payload = script._legacy_payload()
    if mutate_legacy_payload is not None:
        legacy_payload = mutate_legacy_payload(legacy_payload)
    legacy_row = _archive_row(
        script.LEGACY_REVIEWED_EXTRACTION_SHA256,
        dataset="occ_reviewed_memo_extraction",
        suffix="json",
        content_type="application/json",
        source_url=script.OFFICIAL_OCC_URL,
    )
    legacy_row["archive_id"] = script.LEGACY_REVIEWED_EXTRACTION_ARCHIVE_ID
    legacy_path = root / str(legacy_row["object_path"])
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_bytes(gzip.compress(legacy_payload, mtime=0))
    archive_result = repository.write_frame(
        "source_archive",
        pd.DataFrame([legacy_row]),
        completed_session=COMPLETED_SESSION,
        version="base-source-archive",
    )
    actions = _legacy_action()
    if mutate_action is not None:
        mutate_action(actions)
    action_result = repository.write_frame(
        "corporate_actions",
        actions,
        completed_session=COMPLETED_SESSION,
        incomplete_action_policy="block",
        version="base-corporate-actions",
    )
    master_result = repository.write_frame(
        "security_master",
        pd.DataFrame(
            [
                {
                    "security_id": script.SECURITY_ID,
                    "primary_symbol": script.NEW_SYMBOL,
                    "name": "First Republic Bank",
                    "exchange": "OTC",
                    "asset_type": "STOCK",
                    "currency": "USD",
                    "country": "US",
                    "active_from": "2015-01-02",
                    "active_to": "2024-11-08",
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
                    "symbol": script.OLD_SYMBOL,
                    "exchange": "NASDAQ",
                    "effective_from": "2015-01-02",
                    "effective_to": "2023-05-02",
                    "source": "fixture",
                    "retrieved_at": RETRIEVED_AT,
                    "source_hash": hashlib.sha256(b"history-frc").hexdigest(),
                },
                {
                    "security_id": script.SECURITY_ID,
                    "symbol": script.NEW_SYMBOL,
                    "exchange": "OTC",
                    "effective_from": script.EFFECTIVE_DATE,
                    "effective_to": "2024-11-07",
                    "source": "fixture",
                    "retrieved_at": RETRIEVED_AT,
                    "source_hash": hashlib.sha256(b"history-frcb").hexdigest(),
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


def test_pdf_requires_exact_claims_and_visual_origin_attestation(tmp_path: Path):
    path = _write_pdf(tmp_path / "occ-52352.pdf")
    evidence = _evidence(path)

    assert evidence.source_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    assert evidence.page_count == 2
    assert all(evidence.claims.values())
    assert evidence.reviewed_by == "Codex PDF review"
    assert evidence.official_origin_confirmed is True

    with pytest.raises(script.EvidenceError, match="confirm download"):
        script.load_occ_pdf(
            path,
            reviewed_by="reviewer",
            reviewed_at=REVIEWED_AT,
            official_origin_confirmed=False,
        )


def test_pdf_rejects_missing_claim_and_wrong_page_count(tmp_path: Path):
    missing = [line for line in PAGE_ONE if "33616C100" not in line]
    with pytest.raises(script.EvidenceError, match="cusip_33616c100"):
        _evidence(_write_pdf(tmp_path / "missing.pdf", page_one=missing))

    one_page = tmp_path / "one-page.pdf"
    one_page.write_bytes(_pdf_bytes([PAGE_ONE + PAGE_TWO]))
    with pytest.raises(script.EvidenceError, match="must have 2 pages"):
        _evidence(one_page)


def test_pdf_hash_pin_rejects_semantically_equivalent_other_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    reviewed = _write_pdf(tmp_path / "reviewed.pdf")
    alternate = _write_pdf(
        tmp_path / "alternate.pdf",
        page_two=[*PAGE_TWO, "Semantically harmless but byte-changing line."],
    )
    monkeypatch.setattr(
        script,
        "REVIEWED_OCC_PDF_SHA256",
        hashlib.sha256(reviewed.read_bytes()).hexdigest(),
    )

    script.load_occ_pdf(
        reviewed,
        reviewed_by="reviewer",
        reviewed_at=REVIEWED_AT,
        official_origin_confirmed=True,
    )
    with pytest.raises(script.EvidenceError, match="independently reviewed raw pin"):
        script.load_occ_pdf(
            alternate,
            reviewed_by="reviewer",
            reviewed_at=REVIEWED_AT,
            official_origin_confirmed=True,
        )


def test_checked_in_raw_pin_is_the_independently_reviewed_occ_hash():
    assert (
        SCRIPT_PATH.read_text(encoding="utf-8").count(
            "0ab8f0c49076f97049dcde80078affda258108ce7c904241bf1eba46b44aec66"
        )
        == 1
    )


def test_plan_is_read_only_and_changes_only_provenance_and_archive(tmp_path: Path):
    repository = _repository(tmp_path / "store")
    evidence = _evidence(_write_pdf(tmp_path / "occ.pdf"))
    before = _tree(repository.root)
    release, _ = repository.current_release()
    assert release is not None
    actions_before = repository.read_frame(
        "corporate_actions", release.dataset_versions["corporate_actions"]
    )

    prepared = script.prepare_repair(
        repository, evidence, imported_at=IMPORTED_AT
    )

    assert prepared.summary["status"] == "validated_offline_plan"
    assert prepared.summary["source_archive_rows_added"] == 1
    assert prepared.summary["corporate_action_rows_changed"] == 1
    assert prepared.summary["network_accessed"] is False
    assert prepared.summary["eodhd_calls"] == 0
    assert prepared.summary["sec_calls"] == 0
    assert _tree(repository.root) == before
    row_before = actions_before.iloc[0]
    row_after = prepared.frames["corporate_actions"].iloc[0]
    economic_fields = (
        "event_id",
        "security_id",
        "action_type",
        "effective_date",
        "ex_date",
        "announcement_date",
        "cash_amount",
        "ratio",
        "currency",
        "new_security_id",
        "new_symbol",
        "official",
    )
    for field in economic_fields:
        before_value = row_before[field]
        after_value = row_after[field]
        if pd.isna(before_value):
            assert pd.isna(after_value)
        else:
            assert after_value == before_value
    assert row_after["source_hash"] == evidence.source_hash
    assert row_after["source"] == "occ_information_memo"
    assert row_after["source_kind"] == "official_crosscheck"
    metadata = json.loads(row_after["metadata"])
    assert metadata["occ_raw_pdf_sha256"] == evidence.source_hash
    assert (
        metadata["occ_legacy_reviewed_extraction_sha256"]
        == script.LEGACY_REVIEWED_EXTRACTION_SHA256
    )
    assert prepared.summary["legacy_extraction"]["preserved"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("effective_date", "2023-05-04"),
        ("new_symbol", "FRC"),
        ("source_hash", "b" * 64),
        ("source_kind", "official_filing"),
    ],
)
def test_plan_rejects_action_drift(
    tmp_path: Path, field: str, value: object
):
    def mutate(frame: pd.DataFrame) -> None:
        frame.loc[0, field] = value

    repository = _repository(tmp_path / "store", mutate_action=mutate)
    evidence = _evidence(_write_pdf(tmp_path / "occ.pdf"))

    with pytest.raises(script.EvidenceError):
        script.prepare_repair(repository, evidence, imported_at=IMPORTED_AT)


def test_exact_action_rejects_nonofficial_row(tmp_path: Path):
    evidence = _evidence(_write_pdf(tmp_path / "occ.pdf"))
    actions = _legacy_action()
    actions.loc[0, "official"] = False

    with pytest.raises(script.EvidenceError, match="official"):
        script._exact_action(actions, evidence)


def test_plan_rejects_legacy_archive_payload_tamper(tmp_path: Path):
    repository = _repository(
        tmp_path / "store",
        mutate_legacy_payload=lambda payload: payload + b"tamper",
    )
    evidence = _evidence(_write_pdf(tmp_path / "occ.pdf"))

    with pytest.raises(script.EvidenceError, match="payload changed"):
        script.prepare_repair(repository, evidence, imported_at=IMPORTED_AT)


def test_apply_is_transactional_and_idempotent(tmp_path: Path):
    repository = _repository(tmp_path / "store")
    evidence = _evidence(_write_pdf(tmp_path / "occ.pdf"))
    before_release, _ = repository.current_release()
    assert before_release is not None
    prepared = script.prepare_repair(repository, evidence, imported_at=IMPORTED_AT)

    result = script.apply_repair(repository, prepared)

    after_release, _ = repository.current_release()
    assert after_release is not None
    assert result["status"] == "applied"
    assert result["writes_performed"] is True
    assert after_release.quality == before_release.quality
    assert after_release.warnings == before_release.warnings
    assert after_release.dataset_versions["security_master"] == (
        before_release.dataset_versions["security_master"]
    )
    assert after_release.dataset_versions["symbol_history"] == (
        before_release.dataset_versions["symbol_history"]
    )
    actions = repository.read_frame(
        "corporate_actions", after_release.dataset_versions["corporate_actions"]
    )
    row = actions.loc[actions["event_id"].astype(str).eq(script.EVENT_ID)].iloc[0]
    assert row["source_hash"] == evidence.source_hash
    archive = repository.read_frame(
        "source_archive", after_release.dataset_versions["source_archive"]
    )
    assert len(
        archive.loc[
            archive["archive_id"].astype(str).eq(
                script.LEGACY_REVIEWED_EXTRACTION_ARCHIVE_ID
            )
        ]
    ) == 1
    raw = archive.loc[archive["archive_id"].astype(str).eq(evidence.source_hash)]
    assert len(raw) == 1
    object_path = repository.root / raw.iloc[0]["object_path"]
    assert gzip.decompress(object_path.read_bytes()) == evidence.content

    replay = script.prepare_repair(repository, evidence, imported_at=IMPORTED_AT)
    assert replay.summary["status"] == "already_bound"
    second = script.apply_repair(repository, replay)
    assert second["writes_performed"] is False
    final_release, _ = repository.current_release()
    assert final_release is not None
    assert final_release.version == after_release.version


def test_already_bound_action_rejects_raw_metadata_tamper(tmp_path: Path):
    repository = _repository(tmp_path / "store")
    evidence = _evidence(_write_pdf(tmp_path / "occ.pdf"))
    prepared = script.prepare_repair(repository, evidence, imported_at=IMPORTED_AT)
    script.apply_repair(repository, prepared)
    release, _ = repository.current_release()
    assert release is not None
    actions = repository.read_frame(
        "corporate_actions", release.dataset_versions["corporate_actions"]
    )
    row = actions.loc[actions["event_id"].astype(str).eq(script.EVENT_ID)].iloc[0]
    metadata = json.loads(row["metadata"])
    metadata["occ_raw_pdf_page_count"] = 3
    row["metadata"] = json.dumps(metadata)

    with pytest.raises(script.EvidenceError, match="neither the exact legacy"):
        script._exact_action(
            pd.DataFrame([row]),
            evidence,
            object_path=evidence.object_path(COMPLETED_SESSION),
        )


@pytest.mark.parametrize(
    "stage",
    [
        "after_pdf_write",
        "after_source_archive_write",
        "after_corporate_actions_write",
        "after_release_commit",
    ],
)
def test_failure_rolls_back_release_and_both_dataset_pointers(
    tmp_path: Path, stage: str
):
    repository = _repository(tmp_path / "store")
    evidence = _evidence(_write_pdf(tmp_path / "occ.pdf"))
    prepared = script.prepare_repair(repository, evidence, imported_at=IMPORTED_AT)
    before_release, _ = repository.current_release()
    assert before_release is not None
    before_pointers = {
        dataset: repository.current_pointer(dataset)[0].version
        for dataset in script.WRITE_DATASETS
    }

    def fail(observed: str) -> None:
        if observed == stage:
            raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        script.apply_repair(repository, prepared, inject_failure=fail)

    after_release, _ = repository.current_release()
    assert after_release is not None
    assert after_release.version == before_release.version
    for dataset, version in before_pointers.items():
        pointer, _ = repository.current_pointer(dataset)
        assert pointer is not None and pointer.version == version
    recovery = repository.root / script.RECOVERY_DIR
    assert not recovery.exists() or not tuple(recovery.glob("*.json"))
    journals = list((repository.root / script.TRANSACTION_DIR).glob("*.json"))
    assert len(journals) == 1
    assert json.loads(journals[0].read_text())["status"] == "rolled_back"


def test_stale_prepared_plan_is_rejected_before_transaction(tmp_path: Path):
    repository = _repository(tmp_path / "store")
    evidence = _evidence(_write_pdf(tmp_path / "occ.pdf"))
    prepared = script.prepare_repair(repository, evidence, imported_at=IMPORTED_AT)
    release, _ = repository.current_release()
    assert release is not None
    archive = repository.read_frame(
        "source_archive", release.dataset_versions["source_archive"]
    )
    pointer, etag = repository.current_pointer("source_archive")
    assert pointer is not None
    repository.write_frame(
        "source_archive",
        archive,
        completed_session=COMPLETED_SESSION,
        expected_pointer_etag=etag,
        version="interleaved-source-archive",
    )

    with pytest.raises(RuntimeError, match="pointer changed"):
        script.apply_repair(repository, prepared)


def test_module_contains_no_network_collector():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "urllib.request" not in source
    assert "requests." not in source
    assert "eodhd.com/api" not in source
    assert "www.sec.gov/Archives" not in source
