from __future__ import annotations

import base64
import gzip
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.ingest import SourceArtifact
from supertrend_quant.market_store.manifest import DataRelease, sha256_bytes
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_mallinckrodt_identity.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_us_mallinckrodt_identity",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


COMPLETED = "2026-07-15"
RETRIEVED = "2026-07-18T12:00:00Z"
CONTROL_SECURITY_ID = "US:FIXTURE:CURRENT"


def _frame(dataset: str, rows: list[dict] | None = None) -> pd.DataFrame:
    columns = list(dataset_spec(dataset).required_columns)
    extras = []
    for row in rows or []:
        extras.extend(column for column in row if column not in columns)
    columns.extend(dict.fromkeys(extras))
    return pd.DataFrame(
        [
            {column: row.get(column, "") for column in columns}
            for row in (rows or [])
        ],
        columns=columns,
    )


def _artifact(code: str, endpoint: str, rows: list[dict]) -> SourceArtifact:
    return SourceArtifact(
        source=f"eodhd_{endpoint}",
        source_url=script._public_url(code, endpoint),
        retrieved_at=RETRIEVED,
        content=script._canonical_json_bytes(rows),
        content_type="application/json",
    )


def _bars(code: str) -> list[dict]:
    start, end = script.PROVIDER_RANGES[code]
    dates = script._expected_sessions(start, end)
    base = 80.0 if code == script.LEGACY_CODE else 12.0
    rows = []
    for index, value in enumerate(dates):
        if value in script.DOCUMENTED_NON_TRADING_SESSIONS[code]:
            continue
        close = base + index / 100.0
        rows.append(
            {
                "date": value,
                "open": close - 0.10,
                "high": close + 0.25,
                "low": close - 0.30,
                "close": close,
                "volume": 10_000 + index,
            }
        )
    return rows


def _provider_bundle(*, http_attempts: int = 6) -> script.ProviderBundle:
    artifacts = []
    prices = []
    actions = []
    for code in (script.LEGACY_CODE, script.REORGANIZED_CODE):
        eod = _artifact(code, "eod", _bars(code))
        div = _artifact(code, "div", [])
        splits = _artifact(code, "splits", [])
        artifacts.extend((eod, div, splits))
        prices.append(script._eodhd_price_frame(code, eod))
        actions.append(script._eodhd_action_frame(code, div, splits))
    return script.ProviderBundle(
        prices=pd.concat(prices, ignore_index=True, sort=False),
        actions=pd.concat(actions, ignore_index=True, sort=False),
        artifacts=tuple(artifacts),
        http_attempts=http_attempts,
    )


def _source_row(source_hash: str = "a" * 64) -> dict:
    return {
        "source": "fixture",
        "retrieved_at": RETRIEVED,
        "source_hash": source_hash,
    }


def _base_frames(bundle: script.ProviderBundle) -> dict[str, pd.DataFrame]:
    legacy_provider = bundle.prices.loc[
        bundle.prices["security_id"].astype(str).eq(script.LEGACY_SECURITY_ID)
    ]
    reorg_provider = bundle.prices.loc[
        bundle.prices["security_id"].astype(str).eq(
            script.REORGANIZED_SECURITY_ID
        )
    ]
    legacy = legacy_provider.loc[
        legacy_provider["session"].astype(str).between(
            "2015-01-02", "2020-10-12"
        )
    ].copy()
    carry = legacy.iloc[[-1]].copy()
    carry.loc[:, "session"] = "2020-10-12"
    carry.loc[:, ["open", "high", "low", "close"]] = float(
        legacy.iloc[-1]["close"]
    )
    carry.loc[:, "volume"] = 0
    legacy = pd.concat([legacy, carry], ignore_index=True, sort=False)
    reorg = reorg_provider.loc[
        reorg_provider["session"].astype(str).between(
            "2022-10-27", "2023-08-28"
        )
    ].copy()
    legacy.loc[:, "source"] = "eodhd_eod"
    legacy.loc[:, "source_url"] = "https://eodhd.com/api/eod/MNK_old.US"
    legacy.loc[:, "source_hash"] = "1" * 64
    reorg.loc[:, "source"] = "eodhd_eod"
    reorg.loc[:, "source_url"] = "https://eodhd.com/api/eod/MNK.US"
    reorg.loc[:, "source_hash"] = "2" * 64
    current = legacy_provider.iloc[[0]].copy()
    current.loc[:, "security_id"] = CONTROL_SECURITY_ID
    current.loc[:, "session"] = COMPLETED
    current.loc[:, ["open", "high", "low", "close"]] = [100.0, 101.0, 99.0, 100.5]
    current.loc[:, "volume"] = 1_000
    current.loc[:, "source"] = "fixture"
    current.loc[:, "source_url"] = "https://example.test/current.json"
    current.loc[:, "source_hash"] = "9" * 64
    existing_prices = pd.concat(
        [legacy, reorg, current], ignore_index=True, sort=False
    )

    master = _frame(
        "security_master",
        [
            {
                "security_id": script.LEGACY_SECURITY_ID,
                "primary_symbol": "MNK",
                "provider_symbol": "MNK_old.US",
                "action_provider_symbol": "MNK_old.US",
                "name": "Muniholdings New York Insured Fund III Inc",
                "exchange": "NYSE MKT",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "active_from": "2015-01-02",
                "active_to": "2020-10-12",
                **_source_row(),
            },
            {
                "security_id": script.REORGANIZED_SECURITY_ID,
                "primary_symbol": "MNK",
                "provider_symbol": "MNK.US",
                "action_provider_symbol": "MNK.US",
                "name": "Mallinckrodt plc",
                "exchange": "NYSE MKT",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "active_from": "2022-10-27",
                "active_to": "2023-08-28",
                **_source_row(),
            },
            {
                "security_id": CONTROL_SECURITY_ID,
                "primary_symbol": "CTRL",
                "provider_symbol": "CTRL.US",
                "action_provider_symbol": "CTRL.US",
                "name": "Current Fixture Security",
                "exchange": "NYSE",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "active_from": COMPLETED,
                "active_to": "",
                **_source_row("9" * 64),
            },
        ],
    )
    history = _frame(
        "symbol_history",
        [
            {
                "security_id": script.LEGACY_SECURITY_ID,
                "symbol": "MNK",
                "exchange": "NYSE MKT",
                "effective_from": "2015-01-01",
                "effective_to": "",
                **_source_row(),
            },
            {
                "security_id": script.REORGANIZED_SECURITY_ID,
                "symbol": "MNK",
                "exchange": "NYSE MKT",
                "effective_from": "2015-01-01",
                "effective_to": "",
                **_source_row(),
            },
            {
                "security_id": CONTROL_SECURITY_ID,
                "symbol": "CTRL",
                "exchange": "NYSE",
                "effective_from": COMPLETED,
                "effective_to": "",
                **_source_row("9" * 64),
            },
        ],
    )
    actions = _frame("corporate_actions")
    factors = build_adjustment_factors(
        existing_prices,
        actions,
        source_version="fixture-base",
    )
    anchors = _frame(
        "index_constituent_anchors",
        [
            {
                "index_id": "sp500",
                "anchor_date": "2015-01-07",
                "security_id": script.LEGACY_SECURITY_ID,
                "official": False,
                "source_url": "https://example.test/sp500.csv",
                "source_kind": "community",
                **_source_row("3" * 64),
            }
        ],
    )
    events = _frame(
        "index_membership_events",
        [
            {
                "event_id": "4" * 64,
                "index_id": "sp500",
                "announcement_date": "",
                "effective_date": "2017-07-26",
                "operation": "REMOVE",
                "security_id": script.LEGACY_SECURITY_ID,
                "official": False,
                "source_url": "https://example.test/sp500.csv",
                "source_kind": "community",
                **_source_row("3" * 64),
            }
        ],
    )
    return {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": existing_prices,
        "corporate_actions": actions,
        "adjustment_factors": factors,
        "index_constituent_anchors": anchors,
        "index_membership_events": events,
        "source_archive": _frame("source_archive"),
    }


def _release() -> DataRelease:
    return DataRelease(
        version="release-mnk-fixture",
        created_at=RETRIEVED,
        completed_session=COMPLETED,
        dataset_versions={dataset: f"base-{dataset}" for dataset in script.WRITE_DATASETS},
    )


class FakeRepository:
    def __init__(self, root: Path, frames: dict[str, pd.DataFrame]):
        self.root = root
        self.release = _release()
        self.release_etag = "release-etag"
        self.frames = frames
        self.pointer_etags = {
            dataset: f"{dataset}-etag" for dataset in script.WRITE_DATASETS
        }

    def current_release(self):
        return self.release, self.release_etag

    def current_pointer(self, dataset: str):
        return (
            SimpleNamespace(version=self.release.dataset_versions[dataset]),
            self.pointer_etags[dataset],
        )

    def read_frame(self, dataset: str, _version: str | None = None):
        return self.frames[dataset].copy()


@pytest.fixture(scope="module")
def bundle() -> script.ProviderBundle:
    return _provider_bundle()


@pytest.fixture
def fixture(bundle: script.ProviderBundle, tmp_path: Path):
    frames = _base_frames(bundle)
    repository = FakeRepository(tmp_path, frames)
    preflight = script.build_local_preflight(repository, repository.release)
    return repository, preflight


def test_http_inventory_is_six_and_preferred_security_is_excluded() -> None:
    assert script.MAX_EODHD_HTTP_ATTEMPTS == 6
    assert len(script.EODHD_REQUESTS) == 6
    assert {code for code, _endpoint in script.EODHD_REQUESTS} == {
        "MNKKQ",
        "MNKTQ",
    }
    assert all("MNKPF" not in script._public_url(*request) for request in script.EODHD_REQUESTS)


def test_offline_plan_constructs_no_network_client(
    fixture,
) -> None:
    repository, _preflight = fixture
    args = SimpleNamespace(
        cache_root=str(repository.root),
        offline_plan=True,
        fetch_eodhd_mnk=False,
        apply=False,
    )

    result = script.run(
        args,
        repository_factory=lambda _root: repository,
        source_factory=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("network source constructed")
        ),
    )

    assert result["status"] == "offline_plan"
    assert result["network_clients_constructed"] == 0
    assert result["http_attempts"] == 0
    assert result["next_run_maximum_http_attempts"] == 6
    assert result["official_evidence_bindings"] == script.OFFICIAL_BINDINGS


def test_provider_bundle_strict_overlap_and_tail_trim(
    fixture,
    bundle: script.ProviderBundle,
) -> None:
    _repository, preflight = fixture

    result = script.validate_provider_bundle(preflight, bundle)

    assert result["overlap"]["MNKKQ"]["rows"] > 1_000
    assert result["overlap"]["MNKTQ"]["rows"] > 100
    assert result["overlap"]["MNKKQ"]["volume_match_ratio"] == 1.0
    assert result["overlap"]["MNKKQ"][
        "documented_non_trading_sessions_removed"
    ] == ["2020-10-12"]
    assert result["overlap"]["MNKTQ"]["independent_cross_validation"] is False
    assert result["provider_tail_rows_trimmed"]["MNKKQ"] > 0
    assert result["provider_tail_rows_trimmed"]["MNKTQ"] > 0
    prices, _actions = script._trim_provider_frames(bundle)
    ends = prices.groupby("security_id")["session"].max().astype(str).to_dict()
    assert ends == {
        script.LEGACY_SECURITY_ID: script.LEGACY_LAST,
        script.REORGANIZED_SECURITY_ID: script.REORGANIZED_LAST,
    }


def test_overlap_rejects_single_ohlcv_mismatch(
    fixture,
    bundle: script.ProviderBundle,
) -> None:
    _repository, preflight = fixture
    prices = bundle.prices.copy()
    mask = (
        prices["security_id"].astype(str).eq(script.LEGACY_SECURITY_ID)
        & prices["session"].astype(str).eq("2015-01-02")
    )
    prices.loc[mask, "close"] = pd.to_numeric(prices.loc[mask, "close"]) + 0.01
    changed = script.ProviderBundle(
        prices=prices,
        actions=bundle.actions,
        artifacts=bundle.artifacts,
        http_attempts=bundle.http_attempts,
    )

    with pytest.raises(ValueError, match="close overlap mismatch"):
        script.validate_provider_bundle(preflight, changed)


def test_provider_tail_must_still_match_archived_raw_artifact(
    fixture,
    bundle: script.ProviderBundle,
) -> None:
    _repository, preflight = fixture
    prices = bundle.prices.copy()
    mask = (
        prices["security_id"].astype(str).eq(script.LEGACY_SECURITY_ID)
        & prices["session"].astype(str).eq("2022-07-12")
    )
    prices.loc[mask, "close"] = pd.to_numeric(prices.loc[mask, "close"]) + 0.01
    changed = script.ProviderBundle(
        prices=prices,
        actions=bundle.actions,
        artifacts=bundle.artifacts,
        http_attempts=bundle.http_attempts,
    )

    with pytest.raises(ValueError, match="do not match archived raw artifacts"):
        script.validate_provider_bundle(preflight, changed)


def test_identity_rewrite_removes_muniholdings_and_keeps_two_disjoint_ids(
    fixture,
    bundle: script.ProviderBundle,
) -> None:
    _repository, preflight = fixture

    master, history = script.rewrite_security_identities(
        preflight.existing["security_master"],
        preflight.existing["symbol_history"],
        bundle=bundle,
    )

    affected = master.loc[
        master["security_id"].astype(str).isin(
            {script.LEGACY_SECURITY_ID, script.REORGANIZED_SECURITY_ID}
        )
    ]
    assert not affected["name"].str.contains("Muniholdings", case=False).any()
    by_id = affected.set_index("security_id")
    assert by_id.loc[script.LEGACY_SECURITY_ID, "provider_symbol"] == "MNKKQ.US"
    assert by_id.loc[script.REORGANIZED_SECURITY_ID, "provider_symbol"] == "MNKTQ.US"
    assert by_id.loc[script.LEGACY_SECURITY_ID, "active_to"] == "2022-06-15"
    assert by_id.loc[script.REORGANIZED_SECURITY_ID, "active_from"] == "2022-06-17"
    assert len(history[history["security_id"].isin(by_id.index)]) == 2


def test_index_rewrite_migrates_historical_current_id_and_regenerates_event(
    fixture,
    bundle: script.ProviderBundle,
) -> None:
    _repository, preflight = fixture
    anchors = preflight.existing["index_constituent_anchors"].copy()
    events = preflight.existing["index_membership_events"].copy()
    anchors.loc[:, "security_id"] = script.REORGANIZED_SECURITY_ID
    events.loc[:, "security_id"] = script.REORGANIZED_SECURITY_ID
    prior_event = events.iloc[0]["event_id"]

    output_anchors, output_events, stats = script.rewrite_index_references(
        anchors,
        events,
        legacy_artifact=script._artifact_for_code(bundle, script.LEGACY_CODE),
    )

    assert set(output_anchors["security_id"]) == {script.LEGACY_SECURITY_ID}
    assert set(output_events["security_id"]) == {script.LEGACY_SECURITY_ID}
    assert output_events.iloc[0]["event_id"] != prior_event
    assert stats == {
        "anchors_remapped_to_legacy": 1,
        "events_remapped_to_legacy": 1,
    }


def test_request_envelopes_keep_equal_empty_payloads_url_unique(
    bundle: script.ProviderBundle,
) -> None:
    raw_empty = [artifact for artifact in bundle.artifacts if artifact.content == b"[]\n"]
    assert len(raw_empty) == 4
    assert len({artifact.source_hash for artifact in raw_empty}) == 1
    archived = tuple(script._request_archive_artifact(item) for item in bundle.artifacts)

    script._validate_request_envelopes(bundle.artifacts, archived)

    assert len({artifact.source_hash for artifact in archived}) == 6
    assert len({artifact.source_url for artifact in archived}) == 6


def test_prepare_repair_closes_histories_and_outputs_official_bindings(
    fixture,
    bundle: script.ProviderBundle,
) -> None:
    repository, preflight = fixture

    prepared = script.prepare_repair(
        repository,
        repository.release,
        repository.release_etag,
        preflight,
        bundle=bundle,
    )

    assert prepared.summary["status"] == "validated_dry_run"
    assert prepared.summary["no_successor_link"] is True
    assert prepared.summary["independent_cross_validation"] is False
    assert prepared.summary["preferred_code_excluded"] == "MNKPF.US"
    bindings = prepared.summary["official_evidence_bindings"]
    assert bindings["mallinckrodt_2022_cancellation"][
        "candidate_last_price_date"
    ] == "2022-06-15"
    assert bindings["mallinckrodt_2023_cancellation"][
        "candidate_last_price_date"
    ] == "2023-11-13"
    assert bindings["mallinckrodt_2022_cancellation"]["binding_proof"] == (
        "indexed_lifecycle_candidate"
    )
    assert bindings["mallinckrodt_2023_cancellation"]["binding_proof"] == (
        "exact_terminal_history"
    )
    assert all(
        value["binding_status_after_repair"] == "ready_for_hash_pin"
        for value in bindings.values()
    )
    affected_actions = prepared.frames["corporate_actions"].loc[
        prepared.frames["corporate_actions"]["security_id"].astype(str).isin(
            {script.LEGACY_SECURITY_ID, script.REORGANIZED_SECURITY_ID}
        )
    ]
    assert not affected_actions["new_security_id"].fillna("").astype(str).str.strip().any()
    affected_prices = prepared.frames["daily_price_raw"].loc[
        prepared.frames["daily_price_raw"]["security_id"].astype(str).isin(
            {script.LEGACY_SECURITY_ID, script.REORGANIZED_SECURITY_ID}
        )
    ]
    affected_factors = prepared.frames["adjustment_factors"].loc[
        prepared.frames["adjustment_factors"]["security_id"].astype(str).isin(
            {script.LEGACY_SECURITY_ID, script.REORGANIZED_SECURITY_ID}
        )
    ]
    price_keys = set(
        zip(
            affected_prices["security_id"].astype(str),
            pd.to_datetime(affected_prices["session"]).dt.date.astype(str),
        )
    )
    factor_keys = set(
        zip(
            affected_factors["security_id"].astype(str),
            pd.to_datetime(affected_factors["session"]).dt.date.astype(str),
        )
    )
    assert price_keys == factor_keys
    assert affected_factors["source_version"].astype(str).str.startswith(
        "mallinckrodt-identity:"
    ).all()
    assert len(prepared.archive_artifacts) == 7
    assert prepared.summary["gates"]["archive"] == "passed"

    repository.frames = prepared.frames
    repaired_preflight = script.build_local_preflight(
        repository, repository.release
    )
    assert repaired_preflight.already_repaired is True


class FakeClient:
    def __init__(self):
        self.attempt_count = 0
        self.calls: list[tuple[str, dict]] = []

    def get_json(self, endpoint: str, *, params: dict):
        self.attempt_count += 1
        self.calls.append((endpoint, params))
        code = endpoint.split("/")[-1].split(".")[0]
        kind = endpoint.split("/")[0]
        return _bars(code) if kind == "eod" else []


def test_six_endpoint_cache_is_immutable_and_offline_replayable(tmp_path: Path) -> None:
    client = FakeClient()
    source = script.EodhdMallinckrodtSource(
        tmp_path,
        allow_http=True,
        client_factory=lambda: client,
    )

    first = source.fetch()

    assert client.attempt_count == 6
    assert source.http_attempts == 6
    assert len(list(tmp_path.glob("*.json.gz"))) == 6
    offline = script.EodhdMallinckrodtSource(
        tmp_path,
        allow_http=False,
        client_factory=lambda: (_ for _ in ()).throw(
            AssertionError("client should not be created")
        ),
    )
    second = offline.fetch()
    assert offline.http_attempts == 0
    pd.testing.assert_frame_equal(first.prices, second.prices, check_dtype=False)
    assert [item.source_hash for item in first.artifacts] == [
        item.source_hash for item in second.artifacts
    ]

    path = offline.path(script.LEGACY_CODE, "div")
    value = json.loads(gzip.decompress(path.read_bytes()))
    value["content_base64"] = base64.b64encode(b"[{\"changed\":true}]\n").decode()
    path.write_bytes(gzip.compress(script._canonical_json_bytes(value), mtime=0))
    with pytest.raises(ValueError, match="cache identity mismatch"):
        offline.get(script.LEGACY_CODE, "div")


def test_source_stops_on_first_failed_single_attempt(tmp_path: Path) -> None:
    client = SimpleNamespace(
        attempt_count=1,
        get_json=Mock(side_effect=RuntimeError("one shot failure")),
    )
    source = script.EodhdMallinckrodtSource(
        tmp_path,
        allow_http=True,
        client_factory=lambda: client,
    )

    with pytest.raises(RuntimeError, match="one shot failure"):
        source.fetch()

    assert client.get_json.call_count == 1
    assert list(tmp_path.glob("*.json.gz")) == []


def _real_repository(
    root: Path,
    frames: dict[str, pd.DataFrame],
) -> tuple[LocalDatasetRepository, DataRelease]:
    repository = LocalDatasetRepository(root)
    versions = {}
    for dataset in script.WRITE_DATASETS:
        result = repository.write_frame(
            dataset,
            frames[dataset],
            completed_session=COMPLETED,
            incomplete_action_policy="warn",
            version=f"base-{dataset}",
        )
        versions[dataset] = result.manifest.version
    release = repository.commit_release(
        COMPLETED,
        versions,
        quality="valid",
        warnings=("fixture warning",),
    )
    return repository, release


@pytest.mark.parametrize(
    "failure_stage",
    ("after_write:daily_price_raw", "after_release_commit"),
)
def test_atomic_apply_rolls_back_release_and_all_pointers(
    bundle: script.ProviderBundle,
    failure_stage: str,
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        repository, release = _real_repository(
            Path(directory),
            _base_frames(bundle),
        )
        current, release_etag = repository.current_release()
        assert current is not None
        preflight = script.build_local_preflight(repository, current)
        prepared = script.prepare_repair(
            repository,
            current,
            release_etag,
            preflight,
            bundle=bundle,
        )
        old_release = repository.objects.get("releases/current.json").data
        old_pointers = {
            dataset: repository.objects.get(repository.current_key(dataset)).data
            for dataset in script.WRITE_DATASETS
        }

        with pytest.raises(RuntimeError, match="injected rollback"):
            script.apply_repair(
                repository,
                prepared,
                inject_failure=lambda stage: (
                    (_ for _ in ()).throw(RuntimeError("injected rollback"))
                    if stage == failure_stage
                    else None
                ),
            )

        assert repository.objects.get("releases/current.json").data == old_release
        assert all(
            repository.objects.get(repository.current_key(dataset)).data
            == old_pointers[dataset]
            for dataset in script.WRITE_DATASETS
        )
        marker = next(
            (repository.root / "transactions/mallinckrodt-identity-repair").glob(
                "*.json"
            )
        )
        assert json.loads(marker.read_bytes())["status"] == "rolled_back"
        observed, _ = repository.current_release()
        assert observed is not None and observed.version == release.version
