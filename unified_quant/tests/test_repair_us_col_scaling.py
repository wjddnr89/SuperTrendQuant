from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.repository import LocalDatasetRepository


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_col_scaling.py"
)
SPEC = importlib.util.spec_from_file_location("repair_us_col_scaling", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sessions() -> list[str]:
    output = [
        value.date().isoformat()
        for value in xcals.get_calendar("XNYS").sessions_in_range(
            script.PRE_START, script.PRE_END
        )
    ]
    required = {
        script.PRE_START,
        script.PRE_END,
        *(date for date, _old, _new in script.DIVIDENDS.values()),
    }
    assert len(output) == script.EXPECTED_PRE_SESSIONS
    assert required.issubset(output)
    return output


def _evidence(tmp_path: Path) -> tuple[Path, Path, script.EvidencePins, pd.DataFrame]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    dates = [*_sessions(), script.BOUNDARY]
    header = (
        "ticker,date,open,high,low,close,volume,ex-dividend,split_ratio,"
        "adj_open,adj_high,adj_low,adj_close,adj_volume\n"
    )
    dividend_by_date = {
        date: new for date, _old, new in script.DIVIDENDS.values()
    }
    rows: list[dict[str, object]] = []
    lines = [header]
    for number, date in enumerate(dates):
        close = 88.61 if date == script.BOUNDARY else 80.0 + number / 100.0
        open_value = close - 0.2
        high = close + 0.4
        low = close - 0.5
        volume = float(1_000 + number)
        dividend = dividend_by_date.get(date, 0.0)
        split = 1.0
        values = [
            script.SYMBOL,
            date,
            open_value,
            high,
            low,
            close,
            volume,
            dividend,
            split,
            open_value,
            high,
            low,
            close,
            volume,
        ]
        lines.append(",".join(str(value) for value in values) + "\n")
        rows.append(
            {
                "date": date,
                "open": open_value,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
    member = "".join(lines).encode()
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
        "description": "Uploader says public domain; formal license remains Unknown.",
        "versions": [{"versionNumber": 1}],
    }
    metadata_bytes = json.dumps(metadata, indent=2).encode()
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_bytes(metadata_bytes)
    pins = script.EvidencePins(
        zip_sha256=_sha256(zip_path.read_bytes()),
        zip_size=zip_path.stat().st_size,
        member_sha256=_sha256(member),
        member_size=len(member),
        member_crc32=info.CRC,
        full_col_lines_sha256=_sha256(b"".join(line.encode() for line in lines[1:])),
        full_col_row_count=len(dates),
        extract_sha256=_sha256(member),
        extract_size=len(member),
        extract_line_count=len(lines),
        metadata_sha256=_sha256(metadata_bytes),
        metadata_size=len(metadata_bytes),
        metadata_id=7,
        metadata_ref="fixture/wiki",
        metadata_version=1,
        metadata_last_updated="2022-01-01T00:00:00Z",
        metadata_total_bytes=len(member),
        original_price_hash="1" * 64,
        original_dividend_hash="2" * 64,
        original_split_hash="3" * 64,
        enforce_reviewed_relation_profile=False,
    )
    return zip_path, metadata_path, pins, pd.DataFrame(rows)


def _action(
    *,
    event_id: str,
    action_type: str,
    date: str,
    cash_amount: float | None,
    ratio: float | None,
    source: str,
    source_hash: str,
    source_url: str,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "security_id": script.COL_SECURITY_ID,
        "action_type": action_type,
        "effective_date": date,
        "ex_date": date,
        "announcement_date": "",
        "record_date": "",
        "payment_date": "",
        "cash_amount": cash_amount,
        "ratio": ratio,
        "currency": "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": False,
        "source_url": source_url,
        "source_kind": "provider",
        "source": source,
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": source_hash,
        "metadata": np.nan,
    }


def _repository(
    tmp_path: Path,
    pins: script.EvidencePins,
    wiki: pd.DataFrame,
) -> LocalDatasetRepository:
    repository = LocalDatasetRepository(tmp_path / "cache")
    prices: list[dict[str, object]] = []
    for row in wiki.to_dict("records"):
        pre = row["date"] != script.BOUNDARY
        scale = script.SCALE if pre else 1.0
        prices.append(
            {
                "security_id": script.COL_SECURITY_ID,
                "session": row["date"],
                "open": float(row["open"]) / scale,
                "high": float(row["high"]) / scale,
                "low": float(row["low"]) / scale,
                "close": float(row["close"]) / scale,
                "volume": row["volume"],
                "currency": "USD",
                "source": script.ORIGINAL_PRICE_SOURCE,
                "retrieved_at": "2026-07-18T00:00:00Z",
                "source_hash": pins.original_price_hash,
                "source_url": "https://eodhd.test/COL",
            }
        )
    prices.append(
        {
            "security_id": script.COL_SECURITY_ID,
            "session": "2026-07-15",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 2_000.0,
            "currency": "USD",
            "source": script.ORIGINAL_PRICE_SOURCE,
            "retrieved_at": "2026-07-18T00:00:00Z",
            "source_hash": pins.original_price_hash,
            "source_url": "https://eodhd.test/COL",
        }
    )
    actions = [
        _action(
            event_id=event_id,
            action_type="cash_dividend",
            date=date,
            cash_amount=old,
            ratio=None,
            source=script.ORIGINAL_DIVIDEND_SOURCE,
            source_hash=pins.original_dividend_hash,
            source_url=script.DIVIDEND_URL,
        )
        for event_id, (date, old, _new) in script.DIVIDENDS.items()
    ]
    actions.append(
        _action(
            event_id=script.SPLIT_EVENT_ID,
            action_type="split",
            date=script.BOUNDARY,
            cash_amount=None,
            ratio=0.1,
            source=script.ORIGINAL_SPLIT_SOURCE,
            source_hash=pins.original_split_hash,
            source_url=script.SPLIT_URL,
        )
    )
    price_frame = pd.DataFrame(prices)
    action_frame = pd.DataFrame(actions)
    factor_frame = build_adjustment_factors(
        price_frame,
        action_frame,
        source_version="fixture-price+fixture-action",
    )
    archive_frame = pd.DataFrame(
        [
            {
                "archive_id": pins.zip_sha256,
                "dataset": "kaggle_frozen_quandl_wiki_mirror",
                "object_path": (
                    f"archives/2026-07-15/{pins.zip_sha256}.zip.gz"
                ),
                "content_type": "application/zip",
                "effective_date": "2026-07-15",
                "source": "kaggle_frozen_quandl_wiki_mirror",
                "retrieved_at": script.WIKI_RETRIEVED_AT,
                "source_hash": pins.zip_sha256,
                "source_url": script.WIKI_DOWNLOAD_URL,
            }
        ]
    )
    frames = {
        "daily_price_raw": price_frame,
        "corporate_actions": action_frame,
        "adjustment_factors": factor_frame,
        "source_archive": archive_frame,
    }
    versions: dict[str, str] = {}
    for dataset in script.WRITE_DATASETS:
        result = repository.write_frame(
            dataset,
            frames[dataset],
            completed_session="2026-07-15",
            incomplete_action_policy="block",
            version=f"fixture-{dataset}",
        )
        versions[dataset] = result.manifest.version
    repository.commit_release(
        "2026-07-15",
        versions,
        quality=DataQuality.VALID,
    )
    return repository


def _fixture(tmp_path: Path):
    zip_path, metadata_path, pins, wiki = _evidence(tmp_path)
    repository = _repository(tmp_path, pins, wiki)
    return repository, zip_path, metadata_path, pins


def test_plan_is_read_only_and_repairs_price_action_factor_archive_atomically(
    tmp_path: Path,
) -> None:
    repository, zip_path, metadata_path, pins = _fixture(tmp_path)
    release_before = repository.objects.get("releases/current.json").data
    pointers_before = {
        dataset: repository.objects.get(repository.current_key(dataset)).data
        for dataset in script.WRITE_DATASETS
    }
    raw_prices = repository.read_frame("daily_price_raw")
    prepared = script.prepare_repair(
        repository,
        wiki_zip_path=zip_path,
        kaggle_metadata_path=metadata_path,
        pins=pins,
    )

    assert repository.objects.get("releases/current.json").data == release_before
    assert {
        dataset: repository.objects.get(repository.current_key(dataset)).data
        for dataset in script.WRITE_DATASETS
    } == pointers_before
    assert prepared.summary["base_release_version"] == repository.current_release()[0].version
    assert prepared.summary["pre_boundary_price_rows_scaled"] == 355
    assert prepared.summary["dividend_rows_scaled"] == 6
    assert prepared.summary["false_split_rows_removed"] == 1
    assert prepared.summary["license_name"] == "Unknown"
    assert prepared.summary["publication_allowed"] is False
    assert prepared.summary["factor_impact"]["strategy_adjusted_ohlc_equivalent"] is True
    repaired_prices = prepared.frames["daily_price_raw"]
    assert repaired_prices["volume"].equals(raw_prices["volume"])
    pre = pd.to_datetime(repaired_prices["session"]).dt.date.astype(str).le(script.PRE_END)
    assert np.allclose(
        repaired_prices.loc[pre, "close"],
        raw_prices.loc[pre, "close"] * script.SCALE,
    )
    repaired_actions = prepared.frames["corporate_actions"]
    assert script.SPLIT_EVENT_ID not in set(repaired_actions["event_id"].astype(str))
    for event_id, (_date, _old, new) in script.DIVIDENDS.items():
        row = repaired_actions.loc[repaired_actions["event_id"].eq(event_id)].iloc[0]
        assert float(row["cash_amount"]) == new
        assert row["source"] == script.REPAIR_SOURCE
    archive = prepared.frames["source_archive"]
    assert pins.zip_sha256 not in set(archive["archive_id"].astype(str))
    assert set(item.source_hash for item in prepared.artifacts).issubset(
        set(archive["archive_id"].astype(str))
    )


def test_unknown_license_blocks_apply_without_explicit_private_ack(
    tmp_path: Path,
) -> None:
    repository, zip_path, metadata_path, pins = _fixture(tmp_path)
    prepared = script.prepare_repair(
        repository,
        wiki_zip_path=zip_path,
        kaggle_metadata_path=metadata_path,
        pins=pins,
    )
    release_before = repository.objects.get("releases/current.json").data
    with pytest.raises(RuntimeError, match="private/internal-only acknowledgement"):
        script.apply_repair(repository, prepared)
    assert repository.objects.get("releases/current.json").data == release_before


def test_apply_commits_all_four_versions_and_persists_only_minimal_evidence(
    tmp_path: Path,
) -> None:
    repository, zip_path, metadata_path, pins = _fixture(tmp_path)
    prepared = script.prepare_repair(
        repository,
        wiki_zip_path=zip_path,
        kaggle_metadata_path=metadata_path,
        pins=pins,
    )
    result = script.apply_repair(
        repository,
        prepared,
        private_internal_only_ack=True,
    )
    assert result["status"] == "applied"
    release, _ = repository.current_release()
    assert release is not None
    for dataset, version in prepared.planned_versions.items():
        assert release.dataset_versions[dataset] == version
    archive = repository.read_frame("source_archive")
    assert pins.zip_sha256 not in set(archive["archive_id"].astype(str))
    for artifact in prepared.artifacts:
        path = repository.root / artifact.object_path(release.completed_session)
        assert hashlib.sha256(__import__("gzip").decompress(path.read_bytes())).hexdigest() == artifact.source_hash
    prices = repository.read_frame("daily_price_raw")
    pre = pd.to_datetime(prices["session"]).dt.date.astype(str).le(script.PRE_END)
    assert set(prices.loc[pre, "source"].astype(str)) == {script.REPAIR_SOURCE}
    assert prices.loc[pre, "volume"].tolist() == prepared.frames["daily_price_raw"].loc[pre, "volume"].tolist()


def test_stale_release_and_mid_transaction_failure_are_rejected_or_rolled_back(
    tmp_path: Path,
) -> None:
    repository, zip_path, metadata_path, pins = _fixture(tmp_path / "stale")
    prepared = script.prepare_repair(
        repository,
        wiki_zip_path=zip_path,
        kaggle_metadata_path=metadata_path,
        pins=pins,
    )
    release, etag = repository.current_release()
    assert release is not None
    repository.commit_release(
        release.completed_session,
        dict(release.dataset_versions),
        quality=release.quality,
        warnings=release.warnings,
        expected_etag=etag,
    )
    with pytest.raises(RuntimeError, match="changed after COL repair planning"):
        script.apply_repair(
            repository,
            prepared,
            private_internal_only_ack=True,
        )

    repository, zip_path, metadata_path, pins = _fixture(tmp_path / "rollback")
    prepared = script.prepare_repair(
        repository,
        wiki_zip_path=zip_path,
        kaggle_metadata_path=metadata_path,
        pins=pins,
    )
    release_before = repository.objects.get("releases/current.json").data
    pointers_before = {
        dataset: repository.objects.get(repository.current_key(dataset)).data
        for dataset in script.WRITE_DATASETS
    }

    def fail(stage: str) -> None:
        if stage == "after_write:corporate_actions":
            raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        script.apply_repair(
            repository,
            prepared,
            private_internal_only_ack=True,
            inject_failure=fail,
        )
    assert repository.objects.get("releases/current.json").data == release_before
    assert {
        dataset: repository.objects.get(repository.current_key(dataset)).data
        for dataset in script.WRITE_DATASETS
    } == pointers_before


def test_frozen_metadata_or_extract_tamper_fails_before_candidate_build(
    tmp_path: Path,
) -> None:
    _repository_value, zip_path, metadata_path, pins = _fixture(tmp_path)
    metadata_path.write_bytes(metadata_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="metadata size changed"):
        script.load_evidence_bundle(zip_path, metadata_path, pins=pins)
