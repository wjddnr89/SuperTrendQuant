from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from supertrend_quant.market_store.lifecycle import canonical_lifecycle_event_id
from supertrend_quant.market_store.manifest import CurrentPointer, DataRelease, sha256_bytes
from supertrend_quant.market_store.schemas import dataset_spec


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_confirmed_identity_histories.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_confirmed_identity_histories", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)

RETRIEVED_AT = "2026-07-18T00:00:00Z"


class _ValidReport:
    def raise_for_errors(self) -> None:
        return None


def _row(dataset_name: str, **values: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        column: "" for column in dataset_spec(dataset_name).required_columns
    }
    defaults.update(
        {
            "source": "fixture",
            "retrieved_at": RETRIEVED_AT,
            "source_hash": "fixture-hash",
        }
    )
    defaults.update(values)
    return defaults


def _price(
    security_id: str,
    session: str,
    close: float,
    source_hash: str,
) -> dict[str, object]:
    return _row(
        "daily_price_raw",
        security_id=security_id,
        session=session,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1_000.0,
        currency="USD",
        source="eodhd_eod",
        source_hash=source_hash,
        source_url="https://eodhd.test/eod",
    )


def _action(
    event_id: str,
    security_id: str,
    action_type: str,
    effective_date: str,
    *,
    cash_amount: float | None = None,
    ratio: float | None = None,
    new_security_id: str = "",
    new_symbol: str = "",
    source_hash: str = "fixture-action",
    official: bool = False,
) -> dict[str, object]:
    return _row(
        "corporate_actions",
        event_id=event_id,
        security_id=security_id,
        action_type=action_type,
        effective_date=effective_date,
        ex_date=effective_date,
        announcement_date=effective_date if official else "",
        cash_amount=cash_amount,
        ratio=ratio,
        currency="USD",
        new_security_id=new_security_id,
        new_symbol=new_symbol,
        official=official,
        source_url="https://www.sec.gov/fixture" if official else "https://eodhd.test",
        source_kind="official_crosscheck" if official else "provider",
        source="fixture_action",
        source_hash=source_hash,
    )


def _evidence() -> dict[str, script.EvidenceArtifact]:
    return {
        spec.label: script.EvidenceArtifact(
            spec=spec,
            content=b"fixture",
            retrieved_at=RETRIEVED_AT,
        )
        for spec in script.EVIDENCE_SPECS
    }


def _frames() -> dict[str, pd.DataFrame]:
    symbols = {
        script.KHC_ID: ("KHC", "Kraft Heinz", "2015-01-02", ""),
        script.KRFT_ID: ("KRFT", "Kraft Foods", "2015-01-02", "2015-07-02"),
        script.CHK_DUPLICATE_ID: ("CHK", "Chesapeake new", "2015-01-02", "2024-10-01"),
        script.CHK_LEGACY_ID: ("CHK", "Chesapeake legacy", "2015-01-02", "2020-06-26"),
        script.EXE_ID: ("EXE", "Expand Energy", "2015-01-01", ""),
        script.FISV_OLD_ID: ("FISV", "Fiserv old", "2015-01-02", "2023-06-07"),
        script.FI_ID: ("FI", "Fiserv FI", "2015-01-02", "2025-11-10"),
        script.FISV_ACTIVE_ID: ("FISV", "Fiserv active", "2015-01-02", ""),
    }
    master = pd.DataFrame(
        [
            _row(
                "security_master",
                security_id=security_id,
                primary_symbol=symbol,
                name=name,
                exchange="NASDAQ" if symbol != "FI" else "NYSE",
                asset_type="STOCK",
                currency="USD",
                country="US",
                active_from=start,
                active_to=end,
                provider_symbol=f"{symbol}.US",
                action_provider_symbol=f"{symbol}.US",
                source_url="https://eodhd.test/catalog",
            )
            for security_id, (symbol, name, start, end) in symbols.items()
        ]
    )
    history = pd.DataFrame(
        [
            _row(
                "symbol_history",
                security_id=security_id,
                symbol=symbol,
                exchange="NASDAQ" if symbol != "FI" else "NYSE",
                effective_from="2015-01-01",
                effective_to=end,
                source_url="https://eodhd.test/catalog",
            )
            for security_id, (symbol, _name, _start, end) in symbols.items()
        ]
    )

    prices = pd.DataFrame(
        [
            _price(script.KHC_ID, "2015-01-02", 80.0, script.KHC_EOD.source_hash),
            _price(script.KHC_ID, "2015-07-02", 88.0, script.KHC_EOD.source_hash),
            _price(script.KHC_ID, "2015-07-06", 73.0, script.KHC_EOD.source_hash),
            _price(script.KHC_ID, "2026-07-15", 25.0, script.KHC_EOD.source_hash),
            _price(script.KRFT_ID, "2015-01-02", 80.0, script.KRFT_EOD.source_hash),
            _price(script.KRFT_ID, "2015-07-02", 88.0, script.KRFT_EOD.source_hash),
            _price(script.CHK_DUPLICATE_ID, "2015-01-02", 10.0, script.CHK_EOD.source_hash),
            _price(script.CHK_DUPLICATE_ID, "2021-02-10", 45.0, script.CHK_EOD.source_hash),
            _price(script.CHK_DUPLICATE_ID, "2024-10-01", 82.0, script.CHK_EOD.source_hash),
            _price(script.CHK_LEGACY_ID, "2015-01-02", 10.0, "legacy-chk"),
            _price(script.EXE_ID, "2021-02-10", 45.0, script.EXE_EOD.source_hash),
            _price(script.EXE_ID, "2024-10-01", 82.0, script.EXE_EOD.source_hash),
            _price(script.EXE_ID, "2026-07-15", 95.0, script.EXE_EOD.source_hash),
            _price(script.FISV_OLD_ID, "2015-01-02", 35.0, script.FISV_OLD_EOD.source_hash),
            _price(script.FISV_OLD_ID, "2023-06-07", 115.78, script.FISV_OLD_EOD.source_hash),
            _price(script.FI_ID, "2015-01-02", 35.0, script.FI_EOD.source_hash),
            _price(script.FI_ID, "2023-06-07", 115.78, script.FI_EOD.source_hash),
            _price(script.FI_ID, "2025-11-10", 63.8, script.FI_EOD.source_hash),
            _price(script.FISV_ACTIVE_ID, "2015-01-02", 35.0, script.FISV_ACTIVE_EOD.source_hash),
            _price(script.FISV_ACTIVE_ID, "2023-06-07", 115.78, script.FISV_ACTIVE_EOD.source_hash),
            _price(script.FISV_ACTIVE_ID, "2025-11-10", 63.8, script.FISV_ACTIVE_EOD.source_hash),
            _price(script.FISV_ACTIVE_ID, "2026-07-15", 70.0, script.FISV_ACTIVE_EOD.source_hash),
        ]
    )
    factors = pd.DataFrame(
        [
            _row(
                "adjustment_factors",
                security_id=row.security_id,
                session=row.session,
                split_factor=1.0,
                total_return_factor=1.0,
                source_version="fixture",
                calculated_at=RETRIEVED_AT,
                source="derived",
            )
            for row in prices.itertuples(index=False)
        ]
    )
    actions = pd.DataFrame(
        [
            _action("khc-copied-div", script.KHC_ID, "cash_dividend", "2015-04-08", cash_amount=0.55),
            _action("krft-div", script.KRFT_ID, "cash_dividend", "2015-04-08", cash_amount=0.55),
            _action(
                "krft-merger", script.KRFT_ID, "stock_merger", "2015-07-02",
                ratio=1.0, new_security_id=script.KHC_ID, new_symbol="KHC",
                source_hash=script.KHC_SEC.source_hash, official=True,
            ),
            _action(
                "chk-exe", script.CHK_DUPLICATE_ID, "ticker_change", script.EXE_FIRST_SESSION,
                new_security_id=script.EXE_ID, new_symbol="EXE",
                source_hash=script.EXE_SEC.source_hash, official=True,
            ),
            _action("exe-div", script.EXE_ID, "cash_dividend", "2024-10-01", cash_amount=0.575),
            _action(
                "fisv-fi", script.FISV_OLD_ID, "ticker_change", script.FISV_TO_FI_DATE,
                new_security_id=script.FI_ID, new_symbol="FI",
                source_hash=script.FISV_TO_FI_SEC.source_hash, official=True,
            ),
            _action(
                "fi-fisv", script.FI_ID, "ticker_change", script.FI_TO_FISV_DATE,
                new_security_id=script.FISV_ACTIVE_ID, new_symbol="FISV",
                source_hash=script.FI_TO_FISV_SEC.source_hash, official=True,
            ),
            _action("fisv-split", script.FISV_ACTIVE_ID, "split", "2023-06-07", ratio=2.0),
            _action(
                "swn-merger", "US:EODHD:e38dbe48-7597-54e3-b3f5-4dcc84b7a7f2",
                "stock_merger", "2024-10-01", ratio=0.0867,
                new_security_id=script.CHK_DUPLICATE_ID, new_symbol="CHK",
                source_hash="swn-official", official=True,
            ),
        ]
    )
    anchors = pd.DataFrame(
        [
            _row(
                "index_constituent_anchors", index_id="sp500", anchor_date="2015-01-07",
                security_id=script.FISV_OLD_ID, official=False,
                source_url="https://community.test/sp500", source_kind="community",
            ),
            _row(
                "index_constituent_anchors", index_id="nasdaq100", anchor_date="2015-01-01",
                security_id=script.FISV_ACTIVE_ID, official=False,
                source_url="https://community.test/nasdaq", source_kind="community",
            ),
        ]
    )
    event_rows = [
        ("old-remove", "sp500", script.FISV_TO_FI_DATE, "REMOVE", script.FISV_OLD_ID),
        ("fi-add", "sp500", script.FISV_TO_FI_DATE, "ADD", script.FI_ID),
        ("fi-remove", "sp500", script.FI_TO_FISV_DATE, "REMOVE", script.FI_ID),
        ("new-add", "sp500", script.FI_TO_FISV_DATE, "ADD", script.FISV_ACTIVE_ID),
        ("nasdaq-remove", "nasdaq100", script.FISV_TO_FI_DATE, "REMOVE", script.FISV_ACTIVE_ID),
    ]
    events = pd.DataFrame(
        [
            _row(
                "index_membership_events", event_id=event_id, index_id=index_id,
                effective_date=date, operation=operation, security_id=security_id,
                official=False, source_url="https://community.test/index",
                source_kind="community",
            )
            for event_id, index_id, date, operation, security_id in event_rows
        ]
    )
    archive = pd.DataFrame(
        [
            _row(
                "source_archive",
                archive_id=spec.source_hash,
                dataset="sec_edgar_filing" if spec.required_text_groups else "eodhd_eod",
                object_path=f"archives/2026-07-15/{spec.source_hash}{spec.object_suffix}",
                content_type="text/plain" if spec.required_text_groups else "application/json",
                source="fixture_archive",
                source_hash=spec.source_hash,
                source_url=spec.source_url,
            )
            for spec in script.EVIDENCE_SPECS
        ]
    )
    return {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "adjustment_factors": factors,
        "index_constituent_anchors": anchors,
        "index_membership_events": events,
        "source_archive": archive,
    }


def _patch_fixture_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        script,
        "EXPECTED_PRICE_ROWS",
        {
            script.KHC_ID: 4,
            script.KRFT_ID: 2,
            script.CHK_DUPLICATE_ID: 3,
            script.CHK_LEGACY_ID: 1,
            script.EXE_ID: 3,
            script.FISV_OLD_ID: 2,
            script.FI_ID: 3,
            script.FISV_ACTIVE_ID: 4,
        },
    )
    monkeypatch.setattr(
        script,
        "EXPECTED_REPAIRED_PRICE_ROWS",
        {
            script.KHC_ID: 2,
            script.KRFT_ID: 2,
            script.CHK_LEGACY_ID: 1,
            script.EXE_ID: 3,
            script.FISV_ACTIVE_ID: 4,
        },
    )
    monkeypatch.setattr(script, "EXPECTED_KHC_CONTAMINATED_ROWS", 2)
    monkeypatch.setattr(script, "EXPECTED_CHK_EXE_OVERLAP_ROWS", 2)
    monkeypatch.setattr(script, "EXPECTED_FISV_OLD_OVERLAP_ROWS", 2)
    monkeypatch.setattr(script, "EXPECTED_FI_OVERLAP_ROWS", 3)
    monkeypatch.setattr(
        script,
        "EXPECTED_CANONICAL_ACTION_ROWS",
        {script.KHC_ID: 0, script.EXE_ID: 2, script.FISV_ACTIVE_ID: 3},
    )
    monkeypatch.setattr(script, "validate_dataset", lambda *_a, **_k: _ValidReport())


def test_repair_is_exact_fail_closed_and_idempotent(monkeypatch: pytest.MonkeyPatch):
    _patch_fixture_counts(monkeypatch)
    frames = _frames()
    repaired, summary = script.prepare_repair_frames(
        frames,
        _evidence(),
        completed_session="2026-07-15",
        source_version="fixture-repair",
    )

    assert summary["status"] == "validated_offline_plan"
    assert summary["khc_contaminated_price_rows"] == 2
    assert summary["same_security_transition_events_collapsed"] == 4
    assert script._looks_repaired(repaired)
    for dataset in script.WRITE_DATASETS:
        assert not repaired[dataset].security_id.astype(str).isin(script.RETIRED_IDS).any()
    assert set(
        repaired["daily_price_raw"].loc[
            repaired["daily_price_raw"].security_id.eq(script.KHC_ID), "session"
        ]
    ) == {script.KHC_FIRST_SESSION, "2026-07-15"}
    assert set(
        repaired["daily_price_raw"].loc[
            repaired["daily_price_raw"].security_id.eq(script.KRFT_ID), "session"
        ]
    ) == {"2015-01-02", "2015-07-02"}
    assert script._history_signature(repaired["symbol_history"], script.EXE_ID) == {
        ("CHK", "NASDAQ", script.NEW_CHK_FIRST_SESSION, script.CHK_LAST_SESSION),
        ("EXE", "NASDAQ", script.EXE_FIRST_SESSION, ""),
    }
    assert script._history_signature(
        repaired["symbol_history"], script.FISV_ACTIVE_ID
    ) == {
        ("FISV", "NASDAQ", "2015-01-01", script.FISV_OLD_LAST_SESSION),
        ("FI", "NYSE", script.FISV_TO_FI_DATE, script.FI_LAST_SESSION),
        ("FISV", "NASDAQ", script.FI_TO_FISV_DATE, ""),
    }
    actions = repaired["corporate_actions"]
    for security_id, date in (
        (script.EXE_ID, script.EXE_FIRST_SESSION),
        (script.FISV_ACTIVE_ID, script.FISV_TO_FI_DATE),
        (script.FISV_ACTIVE_ID, script.FI_TO_FISV_DATE),
    ):
        event_id = canonical_lifecycle_event_id(security_id, "ticker_change", date)
        row = actions.loc[actions.event_id.eq(event_id)].iloc[0]
        assert row.security_id == row.new_security_id == security_id
    script.validate_repaired_frames(
        repaired,
        _evidence(),
        completed_session="2026-07-15",
    )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("khc_close", "KHC/KRFT reviewed close overlap changed"),
        ("chk_close", "Close overlap changed"),
        ("ticker_hash", "ticker actions changed"),
    ],
)
def test_preflight_rejects_unreviewed_drift(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    message: str,
):
    _patch_fixture_counts(monkeypatch)
    frames = _frames()
    if tamper == "khc_close":
        mask = frames["daily_price_raw"].security_id.eq(script.KHC_ID)
        frames["daily_price_raw"].loc[mask, "close"] += 1.0
    elif tamper == "chk_close":
        mask = (
            frames["daily_price_raw"].security_id.eq(script.CHK_DUPLICATE_ID)
            & frames["daily_price_raw"].session.eq(script.NEW_CHK_FIRST_SESSION)
        )
        frames["daily_price_raw"].loc[mask, "close"] += 1.0
    else:
        mask = (
            frames["corporate_actions"].security_id.eq(script.FI_ID)
            & frames["corporate_actions"].action_type.eq("ticker_change")
        )
        frames["corporate_actions"].loc[mask, "source_hash"] = "changed"
    with pytest.raises(ValueError, match=message):
        script._preflight(frames, _evidence())


def test_hash_pinned_archive_loader_rejects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    content = b"official exact identity evidence"
    digest = sha256_bytes(content)
    spec = script.EvidenceSpec(
        label="fixture",
        source_url="https://www.sec.gov/fixture",
        source_hash=digest,
        exact_bytes=len(content),
        object_suffix=".txt.gz",
        required_text_groups=(("identity evidence",),),
    )
    monkeypatch.setattr(script, "EVIDENCE_SPECS", (spec,))
    object_path = f"archives/2026-07-15/{digest}.txt.gz"
    path = tmp_path / object_path
    path.parent.mkdir(parents=True)
    path.write_bytes(gzip.compress(content, mtime=0))
    archive = pd.DataFrame(
        [
            {
                "archive_id": digest,
                "object_path": object_path,
                "source_url": spec.source_url,
                "source_hash": digest,
                "retrieved_at": RETRIEVED_AT,
            }
        ]
    )
    assert script._load_evidence(tmp_path, archive)["fixture"].content == content
    path.write_bytes(gzip.compress(content + b"tamper", mtime=0))
    with pytest.raises(ValueError, match="persisted bytes changed"):
        script._load_evidence(tmp_path, archive)


def test_release_cas_rejects_stale_plan_before_object_access(tmp_path: Path):
    release = DataRelease(
        version="base-release",
        created_at=RETRIEVED_AT,
        completed_session="2026-07-15",
        dataset_versions={dataset: f"{dataset}-v1" for dataset in script.REQUIRED_DATASETS},
    )
    prepared = SimpleNamespace(
        summary={"status": "validated_offline_plan"},
        release=release,
        release_etag="etag-v1",
    )

    class _StaleRepository:
        root = tmp_path

        @staticmethod
        def current_release():
            return release, "etag-v2"

        @property
        def objects(self):
            pytest.fail("stale CAS must fail before object-store access")

    with pytest.raises(RuntimeError, match="Current release changed"):
        script.apply_repair(_StaleRepository(), prepared)


def test_writer_lock_blocks_interrupted_transaction(tmp_path: Path):
    marker = tmp_path / "transactions/other/prepared.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"status":"prepared"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="interrupted transaction blocks"):
        with script._exclusive_repository_lock(SimpleNamespace(root=tmp_path)):
            pytest.fail("interrupted transaction must block a new writer")


class _ObjectStore:
    def __init__(self, values: dict[str, bytes]):
        self.values = dict(values)
        self.etags = {key: f"etag-{number}" for number, key in enumerate(values)}

    def get(self, key: str):
        return SimpleNamespace(data=self.values[key], etag=self.etags[key])

    def put(self, key: str, data: bytes, *, if_match: str):
        assert if_match == self.etags[key]
        self.values[key] = data
        self.etags[key] += "-next"
        return self.get(key)


def test_rollback_restores_release_and_every_dataset_pointer():
    planned = {dataset: f"{dataset}-planned" for dataset in script.WRITE_DATASETS}
    old_versions = {dataset: f"{dataset}-old" for dataset in script.WRITE_DATASETS}
    old_release = DataRelease(
        version="old-release",
        created_at=RETRIEVED_AT,
        completed_session="2026-07-15",
        dataset_versions=old_versions,
    ).to_bytes()
    committed_release = DataRelease(
        version="committed-release",
        created_at=RETRIEVED_AT,
        completed_session="2026-07-15",
        dataset_versions=planned,
    ).to_bytes()
    old_pointers = {
        dataset: CurrentPointer(
            dataset=dataset,
            version=old_versions[dataset],
            manifest_path=f"datasets/{dataset}/old.json",
            manifest_sha256="old",
            updated_at=RETRIEVED_AT,
        ).to_bytes()
        for dataset in script.WRITE_DATASETS
    }
    current_pointers = {
        dataset: CurrentPointer(
            dataset=dataset,
            version=planned[dataset],
            manifest_path=f"datasets/{dataset}/planned.json",
            manifest_sha256="planned",
            updated_at=RETRIEVED_AT,
        ).to_bytes()
        for dataset in script.WRITE_DATASETS
    }
    values = {"releases/current.json": committed_release}
    repository = SimpleNamespace(
        objects=_ObjectStore(values),
        current_key=lambda dataset: f"datasets/{dataset}/current.json",
    )
    repository.objects.values.update(
        {
            repository.current_key(dataset): value
            for dataset, value in current_pointers.items()
        }
    )
    repository.objects.etags.update(
        {
            repository.current_key(dataset): f"pointer-etag-{number}"
            for number, dataset in enumerate(script.WRITE_DATASETS)
        }
    )

    errors = script._restore_transaction(
        repository,
        old_release_bytes=old_release,
        old_pointer_bytes=old_pointers,
        planned_versions=planned,
        committed_release_version="committed-release",
    )

    assert errors == ()
    assert repository.objects.values["releases/current.json"] == old_release
    for dataset in script.WRITE_DATASETS:
        assert (
            repository.objects.values[repository.current_key(dataset)]
            == old_pointers[dataset]
        )
