from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plan_us_frc_occ_pdf_binding.py"
)
SPEC = importlib.util.spec_from_file_location(
    "plan_us_frc_occ_pdf_binding", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


COMPLETED_SESSION = "2026-07-15"
IMPORTED_AT = "2026-07-18T15:00:00Z"
POLICY_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "us_cross_validation.yaml"
)


PAGE_ONE = [
    "#52352",
    "Date: May 02, 2023",
    "Subject: First Republic Bank - Symbol Change",
    "Option Symbol: FRC",
    "New Symbol: FRCB",
    "Date: 05/03/2023",
    "First Republic Bank (FRC) will change its trading symbol to FRCB",
    "effective at the opening of business on May 3, 2023.",
    "Strike prices and all other option terms will not change.",
    "Date: May 3, 2023",
    "Option Symbol: FRC changes to FRCB",
    "Underlying Security: FRC changes to FRCB",
    "Contract Multiplier: 1",
    "Strike Divisor: 1",
    "New Multiplier: 100",
    "Deliverable Per Contract:",
    "100 First Republic Bank (FRCB) Common Shares",
    "CUSIP: 33616C100",
]

PAGE_TWO = [
    "Disclaimer",
    "This Information Memo provides an unofficial summary of the terms of",
    "corporate events affecting listed options or futures.",
    "ALL CLEARING MEMBERS ARE REQUESTED TO ADVISE ALL BRANCH OFFICES.",
]


def _pdf_bytes(pages: list[list[str]]) -> bytes:
    """Build a small standards-compliant text PDF without a fixture library."""

    page_count = len(pages)
    font_id = 3 + page_count * 2
    object_count = font_id
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
        operators = ["BT", "/F1 10 Tf", "54 760 Td", "14 TL"]
        for line in lines:
            escaped = (
                line.replace("\\", "\\\\")
                .replace("(", "\\(")
                .replace(")", "\\)")
            )
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
    offsets = [0] * (object_count + 1)
    for object_id in range(1, object_count + 1):
        offsets[object_id] = len(output)
        output.extend(f"{object_id} 0 obj\n".encode("ascii"))
        output.extend(objects[object_id])
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {object_count + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for object_id in range(1, object_count + 1):
        output.extend(f"{offsets[object_id]:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {object_count + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
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


def _action() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": script.EVENT_ID,
                "security_id": script.SECURITY_ID,
                "action_type": "ticker_change",
                "effective_date": script.EFFECTIVE_DATE,
                "ex_date": script.EFFECTIVE_DATE,
                "announcement_date": "2023-05-02",
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
                "retrieved_at": "2026-07-18T00:00:00Z",
                "source_hash": script.LEGACY_REVIEWED_EXTRACTION_SHA256,
                "metadata": json.dumps(
                    {"cusip": script.FRC_CUSIP, "memo_number": script.MEMO_NUMBER}
                ),
            }
        ]
    )


def _archive() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "archive_id",
            "dataset",
            "object_path",
            "content_type",
            "effective_date",
            "source",
            "source_url",
            "retrieved_at",
            "source_hash",
        ]
    )


def _policy() -> dict:
    value = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _plan(evidence: script.OccPdfEvidence, *, actions: pd.DataFrame | None = None):
    return script.build_plan(
        evidence=evidence,
        actions=_action() if actions is None else actions,
        archive=_archive(),
        policy=_policy(),
        base_release_version="fixture-release",
        completed_session=COMPLETED_SESSION,
        dataset_versions={
            "corporate_actions": "fixture-actions",
            "source_archive": "fixture-archive",
        },
        imported_at=IMPORTED_AT,
    )


def test_valid_synthetic_pdf_builds_only_a_blocked_dry_run_plan(tmp_path: Path):
    path = _write_pdf(tmp_path / "occ-52352.pdf")
    evidence = script.load_occ_pdf(path)
    plan = _plan(evidence)

    assert evidence.source_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    assert evidence.exact_bytes == path.stat().st_size
    assert evidence.page_count == 2
    assert all(evidence.claims.values())
    assert plan["status"] == "validated_offline_plan"
    assert plan["publication_ready"] is False
    assert plan["source_archive_plan"]["rows_added"] == 1
    assert plan["source_archive_plan"]["payload_write_required"] is True
    assert plan["corporate_action_plan"]["rows_changed"] == 1
    assert (
        plan["reviewed_nonterminal_plan"]["reviewed_nonterminal_rows_added"]
        == 1
    )
    assert plan["visual_review"]["status"] == "required_before_apply"
    assert plan["safety"] == {
        "writes_performed": False,
        "network_accessed": False,
        "eodhd_calls": 0,
        "r2_accessed": False,
        "apply_mode_available": False,
        "official_pdf_hash_invented": False,
    }
    assert not hasattr(script, "apply_repair")


def test_official_url_and_local_file_provenance_are_not_conflated(tmp_path: Path):
    path = _write_pdf(tmp_path / "owner-download.pdf")
    plan = _plan(script.load_occ_pdf(path))

    official = plan["evidence"]["official_source_metadata"]
    local = plan["evidence"]["local_file_provenance"]
    assert official["url"] == script.OFFICIAL_OCC_URL
    assert official["host"] == script.OFFICIAL_OCC_HOST
    assert "input_path" not in official
    assert local["resolved_path"] == str(path.resolve())
    assert local["acquisition_method"] == "user_supplied_local_file"
    assert local["origin_authentication_status"] == "pending_manual_confirmation"
    assert local["network_retrieval_performed_by_planner"] is False
    assert local["official_download_timestamp"] is None


def test_pdf_hash_is_derived_from_bytes_and_never_predeclared(tmp_path: Path):
    first = _write_pdf(tmp_path / "first.pdf")
    second_lines = [*PAGE_TWO, "Reviewer-visible benign extra line."]
    second = _write_pdf(tmp_path / "second.pdf", page_two=second_lines)
    first_evidence = script.load_occ_pdf(first)
    second_evidence = script.load_occ_pdf(second)

    assert first_evidence.source_hash != second_evidence.source_hash
    assert _plan(first_evidence)["evidence"]["exact_pdf"][
        "expected_sha256_predeclared"
    ] is False
    assert _plan(second_evidence)["source_archive_plan"][
        "payload_raw_sha256"
    ] == second_evidence.source_hash


@pytest.mark.parametrize(
    ("old", "new", "missing_claim"),
    [
        ("#52352", "#52353", "memo_number_52352"),
        ("New Symbol: FRCB", "New Symbol: FRCX", "old_and_new_symbol_fields"),
        ("CUSIP: 33616C100", "CUSIP: 33616C101", "cusip_33616c100"),
        (
            "100 First Republic Bank (FRCB) Common Shares",
            "99 First Republic Bank (FRCB) Common Shares",
            "deliverable_100_common_shares",
        ),
        (
            "Strike prices and all other option terms will not change.",
            "Other option terms may change.",
            "all_option_terms_unchanged",
        ),
    ],
)
def test_each_material_pdf_term_is_fail_closed(
    tmp_path: Path, old: str, new: str, missing_claim: str
):
    mutated = [new if line == old else line for line in PAGE_ONE]
    path = _write_pdf(tmp_path / f"mutated-{missing_claim}.pdf", page_one=mutated)

    with pytest.raises(ValueError, match=missing_claim):
        script.load_occ_pdf(path)


def test_effective_date_mutation_is_fail_closed(tmp_path: Path):
    mutated = [
        line.replace("05/03/2023", "05/04/2023")
        .replace("May 3, 2023", "May 4, 2023")
        for line in PAGE_ONE
    ]
    path = _write_pdf(tmp_path / "mutated-effective-date.pdf", page_one=mutated)

    with pytest.raises(ValueError, match="effective_date_2023_05_03"):
        script.load_occ_pdf(path)


def test_transition_sentence_mutation_is_fail_closed(tmp_path: Path):
    mutated = [
        line.replace("will change its trading symbol to FRCB", "may use FRCB")
        for line in PAGE_ONE
    ]
    mutated = [
        line.replace("FRC changes to FRCB", "FRC is unrelated to FRCB")
        for line in mutated
    ]
    path = _write_pdf(tmp_path / "mutated-transition.pdf", page_one=mutated)

    with pytest.raises(ValueError, match="frc_to_frcb_transition"):
        script.load_occ_pdf(path)


def test_missing_or_non_pdf_input_fails_closed(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        script.load_occ_pdf(tmp_path / "missing.pdf")

    not_pdf = tmp_path / "not.pdf"
    not_pdf.write_bytes(b"x" * 200)
    with pytest.raises(ValueError, match="not a complete PDF"):
        script.load_occ_pdf(not_pdf)


def test_one_page_partial_memo_is_rejected(tmp_path: Path):
    path = tmp_path / "partial.pdf"
    path.write_bytes(_pdf_bytes([[*PAGE_ONE, *PAGE_TWO]]))

    with pytest.raises(ValueError, match="exactly 2 pages"):
        script.load_occ_pdf(path)


def test_changed_current_action_provenance_is_not_silently_overwritten(
    tmp_path: Path,
):
    path = _write_pdf(tmp_path / "occ.pdf")
    actions = _action()
    actions.loc[0, "source_hash"] = "0" * 64

    with pytest.raises(ValueError, match="neither the reviewed legacy extraction"):
        _plan(script.load_occ_pdf(path), actions=actions)


def test_cli_requires_occ_pdf_argument():
    with pytest.raises(SystemExit) as exc_info:
        script.main([])

    assert exc_info.value.code == 2
