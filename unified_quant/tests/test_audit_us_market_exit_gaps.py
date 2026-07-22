from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

from supertrend_quant.market_store.repository import LocalDatasetRepository


ROOT = Path(__file__).parents[2]
SCRIPT_PATH = ROOT / "unified_quant/scripts/audit_us_market_exit_gaps.py"
SPEC = importlib.util.spec_from_file_location("audit_us_market_exit_gaps", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Could not load {SCRIPT_PATH}")
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _current_report():
    return audit.build_audit(LocalDatasetRepository(ROOT / "data/cache"))






def test_cached_evidence_hash_or_claim_mutation_fails_closed(tmp_path: Path):
    relative = "state/sec_lifecycle/evidence.bin"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"official claim")
    with pytest.raises(RuntimeError, match="Cached SEC bytes changed"):
        audit.verify_evidence_pin(
            tmp_path,
            audit.EvidencePin(
                cache_object=relative,
                payload_sha256="0" * 64,
                required_patterns=(r"official claim",),
                claim="test",
            ),
        )
    with pytest.raises(RuntimeError, match="Pinned SEC claim changed"):
        audit.verify_evidence_pin(
            tmp_path,
            audit.EvidencePin(
                cache_object=relative,
                payload_sha256=hashlib.sha256(b"official claim").hexdigest(),
                required_patterns=(r"different claim",),
                claim="test",
            ),
        )
