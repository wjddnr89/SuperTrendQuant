from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from supertrend_quant.market_store import cross_validation
from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.ingest import EodhdCallBudget, SourceArtifact
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    sha256_bytes,
)
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.ledger import PortfolioLedger
from supertrend_quant.portfolio import Position


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "collect_us_spinoff_children.py"
)
SPEC = importlib.util.spec_from_file_location(
    "collect_us_spinoff_children", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def _catalog() -> object:
    return script.FrozenCatalogSelection(
        row=dict(script.EXPECTED_CATALOG_ROW),
        source_url=script.CATALOG_URL,
        retrieved_at="2026-07-16T15:56:01Z",
        source_hash=script.CATALOG_SHA256,
        object_path=(
            f"archives/{script.FETCH_END}/{script.CATALOG_SHA256}.json.gz"
        ),
    )


def _eod_rows() -> list[dict[str, object]]:
    return [
        {
            "date": session,
            "open": 10.0 + number / 1000,
            "high": 10.5 + number / 1000,
            "low": 9.5 + number / 1000,
            "close": 10.1 + number / 1000,
            "adjusted_close": 10.1 + number / 1000,
            "volume": 100_000 + number,
        }
        for number, session in enumerate(script._expected_sessions())
    ]


def _raw_artifacts() -> tuple[SourceArtifact, ...]:
    rows = {
        "eod": _eod_rows(),
        "div": [
            {
                "date": "2023-03-01",
                "value": 0.1,
                "unadjustedValue": 0.1,
                "currency": "USD",
                "declarationDate": "2023-02-01",
                "recordDate": "2023-03-02",
                "paymentDate": "2023-03-15",
            }
        ],
        "splits": [],
    }
    return tuple(
        script._source_artifact(
            endpoint, rows[endpoint], "2026-07-18T00:00:00Z"
        )
        for endpoint in script.ENDPOINTS
    )


def _bundle() -> object:
    return script._bundle_from_artifacts(
        _raw_artifacts(),
        http_attempts=3,
        fetched_against_release="base-release",
        budget_used_before=8_393,
        budget_used_after=8_396,
    )


def _source(label: str) -> dict[str, object]:
    return {
        "source": label,
        "source_url": f"https://example.test/{label}",
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": sha256_bytes(label.encode()),
    }


def _master(
    security_id: str,
    symbol: str,
    *,
    provider_symbol: str,
) -> dict[str, object]:
    return {
        "security_id": security_id,
        "primary_symbol": symbol,
        "provider_symbol": provider_symbol,
        "action_provider_symbol": provider_symbol,
        "name": "Fortune Brands Innovations, Inc.",
        "exchange": "NYSE",
        "asset_type": "STOCK",
        "currency": "USD",
        "country": "US",
        "active_from": "2015-01-02",
        "active_to": "",
        "isin": "",
        **_source("master"),
    }


def _history(symbol: str, start: str, end: str) -> dict[str, object]:
    return {
        "security_id": script.FBIN_SECURITY_ID,
        "symbol": symbol,
        "exchange": "NYSE",
        "effective_from": start,
        "effective_to": end,
        **_source(f"history-{symbol}"),
    }


def _price(session: str, close: float) -> dict[str, object]:
    return {
        "security_id": script.FBIN_SECURITY_ID,
        "session": session,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 1_000_000,
        "currency": "USD",
        **_source("fbin-eod"),
    }


def _ticker_action() -> dict[str, object]:
    return {
        "event_id": script.FBIN_TICKER_EVENT_ID,
        "security_id": script.FBIN_SECURITY_ID,
        "action_type": "ticker_change",
        "effective_date": script.FBIN_HISTORY_START,
        "ex_date": script.FBIN_HISTORY_START,
        "announcement_date": "2022-12-16",
        "record_date": "",
        "payment_date": "",
        "cash_amount": None,
        "ratio": None,
        "currency": "USD",
        "new_security_id": script.FBIN_SECURITY_ID,
        "new_symbol": script.FBIN_SYMBOL,
        "official": True,
        "source_kind": "official_filing",
        **_source("ticker"),
    }


def _pseudo_split() -> dict[str, object]:
    return {
        "event_id": script.PSEUDO_SPLIT_EVENT_ID,
        "security_id": script.FBIN_SECURITY_ID,
        "action_type": "split",
        "effective_date": script.SPINOFF_EFFECTIVE_DATE,
        "ex_date": script.SPINOFF_EFFECTIVE_DATE,
        "announcement_date": "",
        "record_date": "",
        "payment_date": "",
        "cash_amount": None,
        "ratio": 1.17,
        "currency": "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": False,
        "source_kind": "provider",
        "source": "eodhd_splits",
        "source_url": script.PSEUDO_SPLIT_URL,
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": script.PSEUDO_SPLIT_RAW_SHA256,
    }


def _base_frames(*, merged: bool = True) -> dict[str, pd.DataFrame]:
    master_id = (
        script.FBIN_SECURITY_ID if merged else script.RETIRED_FBHS_SECURITY_ID
    )
    master = pd.DataFrame(
        [
            _master(
                master_id,
                script.FBIN_SYMBOL if merged else script.FBHS_SYMBOL,
                provider_symbol="FBIN.US" if merged else "FBHS.US",
            )
        ]
    )
    history = pd.DataFrame(
        [
            _history(
                script.FBHS_SYMBOL,
                script.FBHS_HISTORY_START,
                script.FBHS_HISTORY_END,
            ),
            _history(script.FBIN_SYMBOL, script.FBIN_HISTORY_START, ""),
        ]
    )
    if not merged:
        history["security_id"] = script.RETIRED_FBHS_SECURITY_ID
    prices = pd.DataFrame(
        [
            _price("2022-12-14", 61.90),
            _price("2022-12-15", 57.62),
            _price(script.FETCH_END, 70.0),
        ]
    )
    if not merged:
        prices["security_id"] = script.RETIRED_FBHS_SECURITY_ID
    actions = pd.DataFrame([_ticker_action(), _pseudo_split()])
    if not merged:
        actions["security_id"] = script.RETIRED_FBHS_SECURITY_ID
    factors = build_adjustment_factors(
        prices,
        actions,
        source_version="fixture-before",
    )
    empty_archive = pd.DataFrame(
        columns=dataset_spec("source_archive").required_columns
    )
    return {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "adjustment_factors": factors,
        "source_archive": empty_archive,
    }


class _BaseRepository:
    def __init__(self, frames: dict[str, pd.DataFrame], tmp_path: Path):
        self.frames = frames
        self.root = tmp_path

    def manifest_for_version(self, dataset: str, version: str):
        return SimpleNamespace(dataset=dataset, version=version)

    def read_frame(self, dataset: str, _version: str | None = None):
        if dataset in self.frames:
            return self.frames[dataset].copy()
        return pd.DataFrame(columns=dataset_spec(dataset).required_columns)

    def current_pointer(self, dataset: str):
        return SimpleNamespace(version=f"v-{dataset}"), f"etag-{dataset}"


def _release(*, warnings: tuple[str, ...] = ()) -> DataRelease:
    return DataRelease(
        version="base-release",
        created_at="2026-07-18T00:00:00Z",
        completed_session=script.FETCH_END,
        dataset_versions={dataset: f"v-{dataset}" for dataset in script.WRITE_DATASETS},
        quality="degraded" if warnings else "valid",
        warnings=warnings,
    )


def _sec() -> SourceArtifact:
    return SourceArtifact(
        source="sec_edgar_filing",
        source_url=script.SEC_URL,
        retrieved_at="2026-07-18T00:00:00Z",
        content=b"official one-for-one terms",
        content_type="text/plain",
    )


def _form_8937() -> SourceArtifact:
    return SourceArtifact(
        source="fbin_investor_relations_form_8937",
        source_url=script.FORM_8937_URL,
        retrieved_at="2026-07-18T00:00:00Z",
        content=b"%PDF-1.7 official issuer basis allocation",
        content_type="application/pdf",
    )


def test_exact_client_makes_three_single_attempt_calls_and_advances_budget(
    tmp_path: Path,
):
    payloads = {
        "eod": _eod_rows(),
        "div": [],
        "splits": [],
    }

    class Response:
        def __init__(self, value):
            self.value = value

        def raise_for_status(self):
            return None

        def json(self):
            return self.value

    class Session:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            endpoint = url.split("/")[-2]
            return Response(payloads[endpoint])

    budget = EodhdCallBudget(
        tmp_path / "budget.json",
        limit=100,
        reserve=10,
        seed_used=8,
        period="2026-07-18",
    )
    session = Session()
    client = script.ExactThreeEodhdClient(
        session=session, token="secret", budget=budget
    )
    bundle = script.fetch_exact_bundle(
        client,
        release_version="base-release",
        budget_used_before=8,
    )
    assert len(session.calls) == 3
    assert client.attempted_endpoints == [
        "eod/MBC.US",
        "div/MBC.US",
        "splits/MBC.US",
    ]
    assert bundle.http_attempts == 3
    assert bundle.budget_used_after == 11
    assert all("secret" not in item.source_url for item in bundle.artifacts)
    assert json.loads((tmp_path / "budget.json").read_text())["used"] == 11


def test_bundle_cache_round_trip_is_exact_and_tamper_evident(tmp_path: Path):
    bundle = _bundle()
    path = tmp_path / "bundle.json.gz"
    script._write_bundle_cache(path, _catalog(), bundle)
    replay = script._read_bundle_cache(path, _catalog())
    assert replay is not None
    assert replay.http_attempts == 3
    assert [item.source_hash for item in replay.artifacts] == [
        item.source_hash for item in bundle.artifacts
    ]

    wrapper = json.loads(gzip.decompress(path.read_bytes()))
    wrapper["payload_sha256"] = "0" * 64
    path.write_bytes(gzip.compress(script._canonical_json_bytes(wrapper), mtime=0))
    with pytest.raises(ValueError, match="wrapper hash mismatch"):
        script._read_bundle_cache(path, _catalog())


def test_sec_evidence_decodes_html_entities_before_exact_term_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    content = (
        b"one share of MasterBrand common stock for every one share; "
        b"distribution of MasterBrand shares was completed at 5:00 p.m.; "
        b"Wednesday, December&nbsp;14, 2022; regular way; "
        b"symbol &#147;MBC&#148;"
    )
    monkeypatch.setattr(script, "SEC_EXACT_BYTES", len(content))
    monkeypatch.setattr(script, "SEC_SHA256", sha256_bytes(content))
    monkeypatch.setattr(script, "SEC_CACHE_KEY", "sec-unit-test")
    path = tmp_path / "state/sec_lifecycle/sec-unit-test.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)

    artifact = script.load_sec_evidence(tmp_path)

    assert artifact.content == content
    assert artifact.source_hash == sha256_bytes(content)


def test_form_8937_cache_is_exact_and_tamper_evident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    content = b"%PDF-1.7\nissuer basis allocation\n%%EOF\n"
    relative_path = Path("state/issuer_lifecycle/form-8937-test.pdf")
    monkeypatch.setattr(script, "FORM_8937_EXACT_BYTES", len(content))
    monkeypatch.setattr(script, "FORM_8937_SHA256", sha256_bytes(content))
    monkeypatch.setattr(script, "FORM_8937_CACHE_RELATIVE_PATH", relative_path)
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(content)

    artifact = script.load_form_8937_evidence(tmp_path)

    assert artifact.content == content
    assert artifact.content_type == "application/pdf"
    assert artifact.source_url == script.FORM_8937_URL

    path.write_bytes(content[:-1] + b"X")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        script.load_form_8937_evidence(tmp_path)


def test_fbhs_merge_precondition_blocks_before_mbc_rewrite():
    with pytest.raises(RuntimeError, match="identity merge must be applied"):
        script.assert_fbhs_identity_merged(_base_frames(merged=False))


def test_offline_prepare_replaces_pseudo_split_with_exact_spinoff(
    tmp_path: Path,
):
    frames = _base_frames()
    repository = _BaseRepository(frames, tmp_path)
    form_8937 = _form_8937()
    prepared = script.prepare_collection(
        repository,
        _release(),
        "release-etag",
        frames,
        _catalog(),
        _bundle(),
        _sec(),
        form_8937,
    )
    assert prepared.summary["status"] == "validated_offline_plan"
    assert prepared.summary["fake_split_rows_removed"] == 1
    assert prepared.summary["actual_eodhd_calls"] == 3
    actions = prepared.frames["corporate_actions"]
    assert not script._pseudo_split_mask(actions).any()
    spinoff = actions.loc[actions.event_id.eq(script.SPINOFF_EVENT_ID)].iloc[0]
    assert spinoff.action_type == "spinoff"
    assert spinoff.ratio == pytest.approx(1.0)
    assert spinoff.new_security_id == script.MBC_SECURITY_ID
    assert spinoff.new_symbol == "MBC"
    assert spinoff.metadata == script._spinoff_metadata(form_8937)
    assert json.loads(spinoff.metadata)["cost_basis_fraction"] == pytest.approx(
        0.123028
    )
    master = prepared.frames["security_master"]
    assert master.security_id.eq(script.MBC_SECURITY_ID).sum() == 1
    factors = prepared.frames["adjustment_factors"]
    fbin = factors.loc[factors.security_id.eq(script.FBIN_SECURITY_ID)]
    assert fbin.split_factor.eq(1.0).all()
    archive_urls = set(prepared.frames["source_archive"].source_url)
    assert archive_urls.issuperset(
        {*script.REQUEST_URLS.values(), script.SEC_URL, script.FORM_8937_URL}
    )
    issuer_row = prepared.frames["source_archive"].loc[
        prepared.frames["source_archive"].source_url.eq(script.FORM_8937_URL)
    ].iloc[0]
    assert issuer_row.dataset == "fbin_investor_relations_form_8937"
    assert issuer_row.content_type == "application/pdf"
    assert issuer_row.object_path.endswith(
        f"/{form_8937.source_hash}.pdf.gz"
    )
    assert prepared.summary["r2_accessed"] is False


def test_warning_cleanup_removes_only_the_exact_completed_followup():
    other = "A separate lifecycle warning remains"
    near_match = script.PENDING_MBC_FOLLOWUP_WARNING + "."

    assert script._warnings_after_mbc_completion(
        (script.PENDING_MBC_FOLLOWUP_WARNING, other, near_match)
    ) == (other, near_match)


def test_normal_prepare_drops_completed_warning_and_can_commit_valid(
    tmp_path: Path,
):
    frames = _base_frames()
    repository = _BaseRepository(frames, tmp_path)
    prepared = script.prepare_collection(
        repository,
        _release(warnings=(script.PENDING_MBC_FOLLOWUP_WARNING,)),
        "release-etag",
        frames,
        _catalog(),
        _bundle(),
        _sec(),
        _form_8937(),
    )

    assert prepared.warnings == ()
    assert prepared.summary["warning_cleanup_required"] is True
    assert prepared.summary["remaining_release_warnings"] == []
    assert script._quality_for_release_warnings(prepared.warnings) == "valid"


def test_candidate_rejects_tampered_spinoff_cost_basis_metadata(tmp_path: Path):
    frames = _base_frames()
    repository = _BaseRepository(frames, tmp_path)
    bundle = _bundle()
    catalog = _catalog()
    sec = _sec()
    form_8937 = _form_8937()
    prepared = script.prepare_collection(
        repository,
        _release(),
        "release-etag",
        frames,
        catalog,
        bundle,
        sec,
        form_8937,
    )
    tampered = {
        dataset: frame.copy() for dataset, frame in prepared.frames.items()
    }
    actions = tampered["corporate_actions"]
    index = actions.index[actions.event_id.eq(script.SPINOFF_EVENT_ID)].item()
    metadata = json.loads(actions.at[index, "metadata"])
    metadata["cost_basis_fraction"] = 0.12
    actions.at[index, "metadata"] = script._canonical_json_bytes(metadata).decode()

    with pytest.raises(ValueError, match="cost-basis metadata is not exact"):
        script.validate_candidate_frames(
            tampered,
            bundle,
            catalog,
            sec,
            form_8937,
            completed_session=script.FETCH_END,
        )


def test_canonical_spinoff_metadata_reaches_ledger_cost_basis():
    action = script._official_spinoff_action(_sec(), _form_8937())
    action["symbol"] = script.FBIN_SYMBOL
    stored_action = pd.DataFrame([action]).to_dict(orient="records")[0]
    ledger = PortfolioLedger(
        cash=0.0,
        positions={script.FBIN_SYMBOL: Position(script.FBIN_SYMBOL, 10.0, 100.0)},
    )

    events = ledger.apply_actions(
        [stored_action], through=script.SPINOFF_EFFECTIVE_DATE
    )

    assert len(events) == 1
    assert ledger.positions[script.FBIN_SYMBOL].avg_price == pytest.approx(
        87.6972
    )
    assert ledger.positions[script.MBC_SYMBOL].quantity == pytest.approx(10.0)
    assert ledger.positions[script.MBC_SYMBOL].avg_price == pytest.approx(
        12.3028
    )


def test_cli_defaults_to_read_only_plan_and_modes_are_exclusive():
    assert script._parse_args([]).mode == "plan"
    assert script._parse_args(["--fetch-missing"]).mode == "fetch_missing"
    assert script._parse_args(["--offline-plan"]).mode == "offline_plan"
    assert script._parse_args(["--apply"]).mode == "apply"
    with pytest.raises(SystemExit):
        script._parse_args(["--fetch-missing", "--apply"])


def test_already_applied_warning_cleanup_is_metadata_only_cas_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    release = _release(warnings=(script.PENDING_MBC_FOLLOWUP_WARNING,))

    class Repository:
        def __init__(self):
            self.root = tmp_path
            self.release = release
            self.etag = "release-etag-1"
            self.commits: list[dict[str, object]] = []
            self.dataset_write_calls = 0

        def current_release(self):
            return self.release, self.etag

        def current_pointer(self, dataset: str):
            return (
                SimpleNamespace(version=self.release.dataset_versions[dataset]),
                f"pointer-etag-{dataset}",
            )

        def commit_release(
            self,
            completed_session: str,
            dataset_versions: dict[str, str],
            *,
            quality: str,
            warnings: tuple[str, ...],
            expected_etag: str | None,
        ):
            assert expected_etag == self.etag
            value = {
                "completed_session": completed_session,
                "dataset_versions": dict(dataset_versions),
                "quality": str(quality),
                "warnings": tuple(warnings),
            }
            self.commits.append(value)
            self.release = DataRelease(
                version=f"metadata-release-{len(self.commits)}",
                created_at="2026-07-18T01:00:00Z",
                completed_session=completed_session,
                dataset_versions=dict(dataset_versions),
                quality=str(quality),
                warnings=tuple(warnings),
            )
            self.etag = f"release-etag-{len(self.commits) + 1}"
            return self.release

        def write_frame(self, *_args, **_kwargs):
            self.dataset_write_calls += 1
            raise AssertionError("metadata-only repair must not rewrite datasets")

    repository = Repository()
    validation = SimpleNamespace(issues=(), raise_for_errors=lambda: None)
    monkeypatch.setattr(
        script,
        "validate_repository_snapshot",
        lambda _repository: validation,
    )

    def prepared_for_current_release() -> object:
        current, etag = repository.current_release()
        remaining = script._warnings_after_mbc_completion(current.warnings)
        return script.PreparedCollection(
            release=current,
            release_etag=etag,
            pointer_etags={},
            frames={},
            archive_artifacts=(),
            warnings=remaining,
            summary={
                "status": "already_applied",
                "network_accessed": False,
                "r2_accessed": False,
                "warning_cleanup_required": remaining != current.warnings,
            },
        )

    first = script.apply_collection(repository, prepared_for_current_release())

    assert first["status"] == "applied_metadata_only"
    assert first["metadata_only_release_repair"] is True
    assert first["dataset_writes_performed"] is False
    assert first["network_accessed"] is False
    assert first["r2_accessed"] is False
    assert first["quality"] == "valid"
    assert first["warnings"] == []
    assert repository.dataset_write_calls == 0
    assert repository.commits == [
        {
            "completed_session": script.FETCH_END,
            "dataset_versions": release.dataset_versions,
            "quality": "valid",
            "warnings": (),
        }
    ]

    second = script.apply_collection(repository, prepared_for_current_release())

    assert second["status"] == "already_applied"
    assert second["metadata_only_release_repair"] is False
    assert second["writes_performed"] is False
    assert len(repository.commits) == 1
    assert repository.dataset_write_calls == 0


def test_run_apply_routes_already_installed_data_to_metadata_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repository = SimpleNamespace(root=tmp_path)
    release = _release(warnings=(script.PENDING_MBC_FOLLOWUP_WARNING,))
    frames = {
        "security_master": pd.DataFrame(
            [{"security_id": script.MBC_SECURITY_ID}]
        ),
        "corporate_actions": pd.DataFrame(
            [
                {
                    "event_id": script.SPINOFF_EVENT_ID,
                    "security_id": script.FBIN_SECURITY_ID,
                    "action_type": "spinoff",
                    "effective_date": script.SPINOFF_EFFECTIVE_DATE,
                    "ratio": 1.0,
                    "source": "official_fbin_mbc_spinoff",
                    "source_hash": script.SEC_SHA256,
                }
            ]
        ),
    }
    prepared = script.PreparedCollection(
        release=release,
        release_etag="release-etag",
        pointer_etags={},
        frames={},
        archive_artifacts=(),
        warnings=(),
        summary={"status": "already_applied"},
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        script,
        "_base_context",
        lambda _repository: (
            release,
            "release-etag",
            frames,
            _catalog(),
        ),
    )
    monkeypatch.setattr(script, "load_sec_evidence", lambda _root: _sec())
    monkeypatch.setattr(
        script, "load_form_8937_evidence", lambda _root: _form_8937()
    )
    monkeypatch.setattr(
        script,
        "_read_bundle_cache",
        lambda *_args: pytest.fail(
            "already-applied metadata cleanup must not require the bundle cache"
        ),
    )
    monkeypatch.setattr(
        script,
        "_prepare_already_applied",
        lambda *_args: prepared,
    )

    def apply_spy(actual_repository, actual_prepared):
        observed["repository"] = actual_repository
        observed["prepared"] = actual_prepared
        return {"status": "applied_metadata_only"}

    monkeypatch.setattr(script, "apply_collection", apply_spy)

    result = script.run(
        SimpleNamespace(cache_root=tmp_path, mode="apply"),
        repository_factory=lambda _root: repository,
    )

    assert result == {"status": "applied_metadata_only"}
    assert observed == {"repository": repository, "prepared": prepared}


def test_transaction_restore_reinstalls_release_and_all_pointer_preimages():
    class Value:
        def __init__(self, data: bytes, etag: str):
            self.data = data
            self.etag = etag

    class Objects:
        def __init__(self, values: dict[str, bytes]):
            self.values = dict(values)
            self.revisions = {key: 0 for key in values}

        def get(self, key: str):
            return Value(self.values[key], f"{key}-{self.revisions[key]}")

        def put(self, key: str, data: bytes, *, if_match: str):
            assert if_match == f"{key}-{self.revisions[key]}"
            self.values[key] = data
            self.revisions[key] += 1

    old_versions = {dataset: f"old-{dataset}" for dataset in script.WRITE_DATASETS}
    planned = {dataset: f"new-{dataset}" for dataset in script.WRITE_DATASETS}
    old_release = DataRelease(
        "old-release",
        "now",
        script.FETCH_END,
        old_versions,
    ).to_bytes()
    new_release = DataRelease(
        "new-release",
        "later",
        script.FETCH_END,
        planned,
    ).to_bytes()
    old_pointers = {
        dataset: CurrentPointer(
            dataset,
            old_versions[dataset],
            f"old/{dataset}",
            "a" * 64,
            "now",
        ).to_bytes()
        for dataset in script.WRITE_DATASETS
    }
    new_pointers = {
        dataset: CurrentPointer(
            dataset,
            planned[dataset],
            f"new/{dataset}",
            "b" * 64,
            "later",
        ).to_bytes()
        for dataset in script.WRITE_DATASETS
    }
    values = {"releases/current.json": new_release}
    values.update(
        {
            f"datasets/{dataset}/current.json": new_pointers[dataset]
            for dataset in script.WRITE_DATASETS
        }
    )

    class Repository:
        def __init__(self):
            self.objects = Objects(values)

        @staticmethod
        def current_key(dataset: str) -> str:
            return f"datasets/{dataset}/current.json"

    repository = Repository()
    errors = script._restore_transaction_state(
        repository,
        old_release_bytes=old_release,
        old_pointer_bytes=old_pointers,
        planned_versions=planned,
        committed_release_version="new-release",
    )
    assert errors == ()
    assert repository.objects.values["releases/current.json"] == old_release
    for dataset in script.WRITE_DATASETS:
        assert (
            repository.objects.values[repository.current_key(dataset)]
            == old_pointers[dataset]
        )


def test_spinoff_is_hash_pinned_as_an_exact_reviewed_nonterminal_event():
    policy = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "configs/us_cross_validation.yaml"
        ).read_text(encoding="utf-8")
    )["events"]
    reviewed = cross_validation.reviewed_nonterminal_extractions(policy)

    assert (
        cross_validation.reviewed_nonterminal_inventory_sha256(policy)
        == cross_validation.TRUSTED_REVIEWED_NONTERMINAL_EXTRACTIONS_SHA256
    )
    extraction = reviewed[script.SPINOFF_EVENT_ID]
    assert extraction == {
        "event_id": script.SPINOFF_EVENT_ID,
        "security_id": script.FBIN_SECURITY_ID,
        "action_type": "spinoff",
        "effective_date": script.SPINOFF_EFFECTIVE_DATE,
        "new_security_id": script.MBC_SECURITY_ID,
        "new_symbol": script.MBC_SYMBOL,
        "ratio": "1",
        "cash_amount": None,
        "currency": "USD",
        "source_kind": "official_filing",
        "source_url": script.SEC_URL,
        "source_hash": script.SEC_SHA256,
    }

    action = script._official_spinoff_action(_sec(), _form_8937())
    action["source_hash"] = script.SEC_SHA256
    assert cross_validation._nonterminal_terms_complete(action)
    assert (
        cross_validation.reviewed_nonterminal_extraction_mismatches(
            action, extraction
        )
        == ()
    )

    action["ratio"] = 1.17
    assert cross_validation.reviewed_nonterminal_extraction_mismatches(
        action, extraction
    ) == ("ratio",)
