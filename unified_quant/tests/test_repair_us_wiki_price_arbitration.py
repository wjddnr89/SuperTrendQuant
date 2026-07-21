from __future__ import annotations

import dataclasses
import gzip
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from supertrend_quant.market_store.manifest import DataRelease
from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.repository import LocalDatasetRepository


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_wiki_price_arbitration.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_wiki_price_arbitration", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


HEADER = (
    "ticker,date,open,high,low,close,volume,ex-dividend,split_ratio,"
    "adj_open,adj_high,adj_low,adj_close,adj_volume\n"
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _target(symbol: str, security_id: str, dates: list[str], lines: list[bytes]):
    return script.PriceTarget(
        symbol=symbol,
        security_id=security_id,
        end=dates[-1],
        expected_rows=len(dates),
        full_rows=len(lines),
        full_lines_sha256=_sha(b"".join(lines)),
        price_source_hash="1" * 64,
        relation_sha256="2" * 64,
        signal_sha256="3" * 64,
        expected_exact={column: len(dates) for column in ("open", "high", "low", "close")},
        expected_max_abs={column: 0.0 for column in ("open", "high", "low", "close")},
        expected_max_relative={column: 0.0 for column in ("open", "high", "low", "close")},
        expected_volume_exact=len(dates),
        expected_volume_median_abs=0.0,
        expected_volume_max_abs=0.0,
        expected_return_correlation=1.0,
        identity_source="fixture_identity",
        identity_source_hash="4" * 64,
        provider_symbol=f"{symbol}_old.US",
    )


def _frozen_fixture(tmp_path: Path):
    dates = pd.bdate_range("2015-01-02", periods=24).date.astype(str).tolist()
    lines_by_symbol: dict[str, list[bytes]] = {}
    all_lines = [HEADER.encode()]
    for offset, symbol in enumerate(("AAA", "BBB")):
        lines: list[bytes] = []
        for number, date in enumerate(dates):
            close = 30.0 + offset * 10 + number / 10
            values = [
                symbol,
                date,
                close - 0.1,
                close + 0.2,
                close - 0.3,
                close,
                1_000 + number,
                0.0,
                1.0,
                close - 0.1,
                close + 0.2,
                close - 0.3,
                close,
                1_000 + number,
            ]
            line = (",".join(str(value) for value in values) + "\n").encode()
            lines.append(line)
            all_lines.append(line)
        lines_by_symbol[symbol] = lines
    member = b"".join(all_lines)
    zip_path = tmp_path / "wiki.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(script.WIKI_MEMBER, member)
    with zipfile.ZipFile(zip_path) as archive:
        info = archive.getinfo(script.WIKI_MEMBER)

    metadata = {
        "id": 7,
        "ref": "fixture/wiki",
        "licenseName": "Unknown",
        "lastUpdated": "2022-01-01T00:00:00Z",
        "totalBytes": len(member),
        "versions": [{"versionNumber": 1}],
    }
    metadata_bytes = json.dumps(metadata, sort_keys=True).encode()
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_bytes(metadata_bytes)
    targets = (
        _target("AAA", "US:TEST:AAA", dates, lines_by_symbol["AAA"]),
        _target("BBB", "US:TEST:BBB", dates, lines_by_symbol["BBB"]),
    )
    pins = script.EvidencePins(
        zip_sha256=_sha(zip_path.read_bytes()),
        zip_size=zip_path.stat().st_size,
        member_sha256=_sha(member),
        member_size=len(member),
        member_crc32=info.CRC,
        combined_extract_sha256=_sha(member),
        combined_extract_size=len(member),
        combined_extract_lines=len(all_lines),
        metadata_sha256=_sha(metadata_bytes),
        metadata_size=len(metadata_bytes),
        metadata_id=7,
        metadata_ref="fixture/wiki",
        metadata_version=1,
        metadata_last_updated="2022-01-01T00:00:00Z",
        metadata_total_bytes=len(member),
        enforce_reviewed_profile=True,
    )
    return zip_path, metadata_path, pins, targets, dates


def test_frozen_extract_is_exact_and_license_is_fail_closed(tmp_path: Path):
    zip_path, metadata_path, pins, targets, _ = _frozen_fixture(tmp_path)

    bundle = script.load_evidence_bundle(
        zip_path, metadata_path, pins=pins, targets=targets
    )

    assert bundle.extract.source_hash == pins.combined_extract_sha256
    assert bundle.audit["metadata_license_name"] == "Unknown"
    assert {key: len(value) for key, value in bundle.rows.items()} == {
        "AAA": 24,
        "BBB": 24,
    }


def test_frozen_zip_and_formal_license_tampering_are_rejected(tmp_path: Path):
    zip_path, metadata_path, pins, targets, _ = _frozen_fixture(tmp_path)
    original = zip_path.read_bytes()
    zip_path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="ZIP hash changed"):
        script.load_evidence_bundle(
            zip_path, metadata_path, pins=pins, targets=targets
        )

    zip_path.write_bytes(original)
    metadata = json.loads(metadata_path.read_bytes())
    metadata["licenseName"] = "CC0"
    tampered = json.dumps(metadata, sort_keys=True).encode()
    metadata_path.write_bytes(tampered)
    changed_pins = dataclasses.replace(
        pins,
        metadata_sha256=_sha(tampered),
        metadata_size=len(tampered),
    )
    with pytest.raises(ValueError, match="identity/license metadata changed"):
        script.load_evidence_bundle(
            zip_path, metadata_path, pins=changed_pins, targets=targets
        )


def _audit_frames(target: script.PriceTarget, dates: list[str]):
    rows = []
    wiki = []
    for number, date in enumerate(dates):
        close = 50.0 + number / 10
        values = {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.3,
            "close": close,
            "volume": 1_000.0 + number,
        }
        rows.append(
            {
                "security_id": target.security_id,
                "session": date,
                **values,
                "currency": "USD",
                "source": "eodhd_eod",
                "retrieved_at": "2026-07-19T00:00:00Z",
                "source_hash": target.price_source_hash,
                "source_url": "https://eodhd.test/eod/AAA_old.US",
            }
        )
        wiki.append(
            {
                "ticker": target.symbol,
                "date": date,
                **values,
                "ex-dividend": 0.0,
                "split_ratio": 1.0,
                "adj_open": values["open"],
                "adj_high": values["high"],
                "adj_low": values["low"],
                "adj_close": values["close"],
                "adj_volume": values["volume"],
            }
        )
    prices = pd.DataFrame(rows)
    factors = pd.DataFrame(
        [
            {
                "security_id": target.security_id,
                "session": date,
                "split_factor": 1.0,
                "total_return_factor": 1.0,
                "source_version": "fixture",
                "calculated_at": "2026-07-19T00:00:00Z",
                "source": "derived",
                "retrieved_at": "2026-07-19T00:00:00Z",
                "source_hash": "fixture",
            }
            for date in dates
        ]
    )
    master = pd.DataFrame(
        [
            {
                "security_id": target.security_id,
                "primary_symbol": target.symbol,
                "provider_symbol": target.provider_symbol,
                "source": target.identity_source,
                "source_hash": target.identity_source_hash,
            }
        ]
    )
    history = pd.DataFrame(
        [
            {
                "security_id": target.security_id,
                "symbol": target.symbol,
                "effective_from": "2015-01-01",
                "source": target.identity_source,
                "source_hash": target.identity_source_hash,
            }
        ]
    )
    actions = pd.DataFrame(
        columns=[
            "event_id",
            "security_id",
            "action_type",
            "effective_date",
            "cash_amount",
        ]
    )
    return pd.DataFrame(wiki), prices, factors, master, history, actions


def test_price_only_audit_reproduces_zero_triple_supertrend_difference(tmp_path: Path):
    _, _, _, targets, dates = _frozen_fixture(tmp_path)
    target = targets[0]
    frames = _audit_frames(target, dates)

    audit = script._audit_price_target(
        target,
        *frames,
        enforce_reviewed_profile=False,
    )

    assert audit["status"] == "passed_price_only_arbitration"
    assert not any(
        audit["wiki_raw_substitution_sensitivity"]
        ["triple_supertrend_field_differences"].values()
    )
    assert audit["raw_price_rewritten"] is False
    assert audit["corporate_actions_rewritten"] is False
    assert audit["adjustment_factors_rewritten"] is False


def _artifact(name: str, payload: bytes) -> script.ArchiveArtifact:
    return script.ArchiveArtifact(
        dataset=name,
        source=name,
        source_url="https://example.test/frozen",
        content_type="application/json",
        extension="json",
        payload=payload,
        retrieved_at="2026-07-19T00:00:00Z",
    )


def test_partial_or_tampered_archive_evidence_is_rejected(tmp_path: Path):
    repository = LocalDatasetRepository(tmp_path / "cache")
    first = _artifact("first", b"first\n")
    second = _artifact("second", b"second\n")
    path = repository.root / first.object_path("2026-07-15")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(first.payload, mtime=0))
    archive = pd.DataFrame(
        [
            script._artifact_row(
                first,
                completed_session="2026-07-15",
                columns=[
                    "archive_id",
                    "dataset",
                    "object_path",
                    "content_type",
                    "effective_date",
                    "source",
                    "retrieved_at",
                    "source_hash",
                    "source_url",
                ],
            )
        ]
    )
    with pytest.raises(ValueError, match="partially archived"):
        script._append_or_verify_artifacts(
            repository,
            archive,
            (first, second),
            completed_session="2026-07-15",
        )

    path.write_bytes(gzip.compress(b"tampered\n", mtime=0))
    with pytest.raises(ValueError, match="payload hash changed"):
        script._append_or_verify_artifacts(
            repository,
            archive,
            (first,),
            completed_session="2026-07-15",
        )


def test_apply_requires_local_private_internal_only_ack(tmp_path: Path):
    release = DataRelease.create(
        "2026-07-15",
        {"source_archive": "fixture"},
        quality=DataQuality.VALID,
    )
    prepared = script.PreparedRepair(
        release=release,
        release_etag=None,
        pointer_etag=None,
        frame=pd.DataFrame(),
        artifacts=(),
        pins=script.DEFAULT_PINS,
        wiki_zip_path=Path("missing.zip"),
        kaggle_metadata_path=Path("missing.json"),
        targets=script.TARGETS,
        allowed_index_identity_gap_fingerprints=(),
        summary={"status": "validated_offline_plan"},
    )

    with pytest.raises(PermissionError, match="private_internal_only"):
        script.apply_repair(LocalDatasetRepository(tmp_path), prepared)


def test_release_warning_blocks_redistribution_and_publication():
    assert "licenseName=Unknown" in script.WIKI_LICENSE_WARNING
    assert "private/internal-only" in script.WIKI_LICENSE_WARNING
    assert "redistribution/public publication blocked" in script.WIKI_LICENSE_WARNING
