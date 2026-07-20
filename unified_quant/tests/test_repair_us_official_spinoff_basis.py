from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from supertrend_quant.ledger import PortfolioLedger
from supertrend_quant.market_store.ingest import SourceArtifact
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.portfolio import Position


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_official_spinoff_basis.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_official_spinoff_basis", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def _actions() -> pd.DataFrame:
    rows = []
    for target in script.ACTION_SPECS:
        rows.append(
            {
                "event_id": target.event_id,
                "security_id": target.security_id,
                "action_type": "spinoff",
                "effective_date": target.effective_date,
                "ex_date": target.effective_date,
                "announcement_date": "",
                "record_date": "",
                "payment_date": "",
                "cash_amount": None,
                "ratio": 1 / 3,
                "currency": "USD",
                "new_security_id": target.new_security_id,
                "new_symbol": target.new_symbol,
                "official": True,
                "source_url": "https://www.sec.gov/example",
                "source_kind": "official_filing",
                "source": "official_test",
                "retrieved_at": "2026-07-18T00:00:00Z",
                "source_hash": target.existing_source_hash,
                "metadata": "",
            }
        )
    return pd.DataFrame(rows)


def _stored_action(target: object, *, symbol: str) -> dict[str, object]:
    return {
        "event_id": target.event_id,
        "action_type": "spinoff",
        "symbol": symbol,
        "effective_date": target.effective_date,
        "new_symbol": target.new_symbol,
        "ratio": 1 / 3,
        "metadata": script._canonical_json(target.metadata),
    }


def test_reviewed_metadata_is_exact_and_fox_remains_fail_closed():
    by_symbol = {item.new_symbol: item for item in script.ACTION_SPECS}
    assert by_symbol["DOW"].metadata["cost_basis_fraction"] == pytest.approx(
        0.3356
    )
    assert by_symbol["CTVA"].metadata["cost_basis_fraction"] == pytest.approx(
        0.2586805
    )
    for symbol in ("FOX", "FOXA"):
        metadata = by_symbol[symbol].metadata
        assert "cost_basis_fraction" not in metadata
        assert metadata["basis_status"] == "unsupported_taxable_fmv_reset"
        assert metadata["exchanged_parent_share_fraction"] == pytest.approx(
            0.263183
        )


def test_rewrite_changes_exactly_four_rows_and_is_idempotent():
    rewritten, changed = script._rewrite_actions(_actions())
    assert changed == 4
    assert rewritten["metadata"].astype(str).ne("").all()

    second, second_changed = script._rewrite_actions(rewritten)
    assert second_changed == 0
    pd.testing.assert_frame_equal(rewritten, second)


@pytest.mark.parametrize(
    ("child", "expected_parent", "expected_child"),
    [
        ("DOW", 66.44, 100.68),
        ("CTVA", 74.13195, 77.60415),
    ],
)
def test_exact_issuer_fraction_reaches_ledger(
    child: str, expected_parent: float, expected_child: float
):
    target = next(item for item in script.ACTION_SPECS if item.new_symbol == child)
    ledger = PortfolioLedger(
        cash=0.0,
        positions={"DWDP": Position("DWDP", 9.0, 100.0)},
    )

    events = ledger.apply_actions(
        [_stored_action(target, symbol="DWDP")], through=target.effective_date
    )

    assert len(events) == 1
    assert not ledger.unresolved_event_ids
    assert ledger.positions["DWDP"].avg_price == pytest.approx(expected_parent)
    assert ledger.positions[child].quantity == pytest.approx(3.0)
    assert ledger.positions[child].avg_price == pytest.approx(expected_child)


@pytest.mark.parametrize("child", ["FOX", "FOXA"])
def test_taxable_fox_fmv_basis_stays_unresolved_when_parent_is_held(child: str):
    target = next(item for item in script.ACTION_SPECS if item.new_symbol == child)
    parent = "TFCF" if child == "FOX" else "TFCFA"
    ledger = PortfolioLedger(
        cash=0.0,
        positions={parent: Position(parent, 9.0, 40.0)},
    )

    event = ledger.apply_actions(
        [_stored_action(target, symbol=parent)], through=target.effective_date
    )[0]

    assert target.event_id in ledger.unresolved_event_ids
    assert child not in ledger.positions
    assert "cost-basis" in event.message


def test_rewrite_rejects_changed_official_terms():
    actions = _actions()
    actions.loc[actions["new_symbol"].eq("DOW"), "ratio"] = 0.25

    with pytest.raises(ValueError, match="Target spin-off terms changed: DOW"):
        script._rewrite_actions(actions)


def test_archive_rows_are_hash_keyed_and_idempotent():
    artifact = SourceArtifact(
        source="official_test",
        source_url="https://issuer.example/form8937.pdf",
        retrieved_at="2026-07-18T00:00:00Z",
        content=b"reviewed evidence",
        content_type="application/pdf",
    )
    empty = pd.DataFrame(
        columns=dataset_spec("source_archive").required_columns
    )

    first, added = script._append_source_archive(
        empty, [artifact], completed_session="2026-07-15"
    )
    second, second_added = script._append_source_archive(
        first, [artifact], completed_session="2026-07-15"
    )

    assert added == 1
    assert second_added == 0
    assert len(second) == 1
    assert second.iloc[0]["archive_id"] == hashlib.sha256(
        artifact.content
    ).hexdigest()
    assert second.iloc[0]["object_path"].endswith(".pdf.gz")


def test_read_evidence_rejects_hash_or_size_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    content = b"small official attachment"
    digest = hashlib.sha256(content).hexdigest()
    spec = script.EvidenceSpec(
        key="tiny",
        url="https://issuer.example/tiny.pdf",
        sha256=digest,
        size=len(content),
        content_type="application/pdf",
        retrieved_at="2026-07-18T00:00:00Z",
    )
    monkeypatch.setattr(script, "EVIDENCE_SPECS", (spec,))
    path = tmp_path / spec.filename
    path.write_bytes(content + b"tampered")

    with pytest.raises(ValueError, match="hash/size mismatch"):
        script._read_artifacts(tmp_path)


def test_cli_defaults_to_read_only_plan():
    args = script._parse_args([])
    assert args.apply is False
    assert args.cache_root == script.DEFAULT_CACHE_ROOT
    assert args.evidence_dir == script.DEFAULT_EVIDENCE_DIR
