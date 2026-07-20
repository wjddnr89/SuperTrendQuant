from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import yaml
from supertrend_quant.market_store import official_lifecycle_evidence
from supertrend_quant.market_store.repository import LocalDatasetRepository


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "collect_us_lifecycle_actions.py"
)
SPEC = importlib.util.spec_from_file_location("collect_us_lifecycle_actions", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
COLLECTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COLLECTOR
SPEC.loader.exec_module(COLLECTOR)

FINALIZER_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "finalize_us_lifecycle_coverage.py"
)
FINALIZER_SPEC = importlib.util.spec_from_file_location(
    "finalize_us_lifecycle_coverage_cov_smoke",
    FINALIZER_PATH,
)
assert FINALIZER_SPEC is not None and FINALIZER_SPEC.loader is not None
FINALIZER = importlib.util.module_from_spec(FINALIZER_SPEC)
sys.modules[FINALIZER_SPEC.name] = FINALIZER
FINALIZER_SPEC.loader.exec_module(FINALIZER)

NTCO_PLAN_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plan_us_ntco_ntcoy_transition.py"
)
NTCO_PLAN_SPEC = importlib.util.spec_from_file_location(
    "plan_us_ntco_ntcoy_transition_collector_smoke",
    NTCO_PLAN_PATH,
)
assert NTCO_PLAN_SPEC is not None and NTCO_PLAN_SPEC.loader is not None
NTCO_PLAN = importlib.util.module_from_spec(NTCO_PLAN_SPEC)
sys.modules[NTCO_PLAN_SPEC.name] = NTCO_PLAN
NTCO_PLAN_SPEC.loader.exec_module(NTCO_PLAN)


def test_economic_crosscheck_uses_old_close_before_and_successor_close_after() -> None:
    histories = {
        "OLD": pd.DataFrame(
            {
                "session": pd.to_datetime(["2016-05-27"]),
                "close": [51.55],
            }
        ),
        "NEW": pd.DataFrame(
            {
                "session": pd.to_datetime(["2016-05-27", "2016-05-31"]),
                "close": [51.55, 38.81],
            }
        ),
    }
    evidence = {
        "candidate": {"security_id": "OLD"},
        "parsed": {
            "action_type": "stock_merger",
            "effective_date": "2016-05-28",
            "ratio": 1.0,
            "cash_amount": 14.5,
        },
    }

    result = COLLECTOR._economic_crosscheck(
        evidence,
        successor_security_id="NEW",
        price_histories=histories,
    )

    assert result["old_price_session"] == "2016-05-27"
    assert result["successor_price_session"] == "2016-05-31"
    assert result["implied_consideration"] == 53.31
    assert result["economic_terms_passed"] is True


def test_nearest_close_rejects_missing_directional_session() -> None:
    histories = {
        "ONLY_BEFORE": pd.DataFrame(
            {
                "session": pd.to_datetime(["2024-01-02"]),
                "close": [10.0],
            }
        )
    }

    result = COLLECTOR._nearest_close(
        histories,
        "ONLY_BEFORE",
        pd.Timestamp("2024-01-03"),
        direction="on_or_after",
    )

    assert result is None


def test_cov_verified_hint_uses_sec_url_cache_and_finalizer_artifact_override() -> None:
    hint = COLLECTOR._load_hints(COLLECTOR.DEFAULT_HINTS)["COV"]
    verified = hint["verified_event"]
    assert verified["action_type"] == "stock_merger"
    assert str(verified["effective_date"]) == "2015-01-26"
    assert float(verified["cash_amount"]) == 35.19
    assert float(verified["ratio"]) == 0.956
    assert verified["new_symbol"] == "MDT"
    assert verified["confidence"] == "high"

    candidate = FINALIZER.LifecycleCandidate(
        security_id="COV-ID",
        symbol="COV",
        name="Covidien plc",
        exchange="NYSE",
        last_price_date="2015-01-26",
        active_to="2015-01-26",
        index_remove_dates=("2015-01-26",),
    )
    urls = tuple(verified["source_urls"])
    contents = (
        b"SEC Covidien completion filing: 35.19 cash plus 0.956 MDT",
        b"SEC-filed Medtronic issuer release: COV ceased trading at the close",
    )
    with tempfile.TemporaryDirectory() as directory:
        cache_root = Path(directory)
        session = SimpleNamespace(headers={}, get=Mock())
        source = COLLECTOR.SecEdgarLifecycleSource(
            session=session,
            user_agent="SuperTrendQuant test@example.com",
            cache_dir=cache_root,
        )
        for url, content in zip(urls, contents, strict=True):
            cache_key = hashlib.sha256(f"{url}?".encode()).hexdigest()
            (cache_root / f"{cache_key}.bin").write_bytes(content)

        evidence, artifacts = COLLECTOR._collect_verified_event(
            source,
            candidate,
            dict(verified),
        )

        session.get.assert_not_called()
        assert evidence.parsed is not None
        assert evidence.parsed.confidence == "high"
        assert evidence.parsed.effective_date == "2015-01-26"
        assert evidence.parsed.cash_amount == 35.19
        assert evidence.parsed.ratio == 0.956
        assert evidence.parsed.new_symbol == "MDT"
        assert len(artifacts) == 2

        successor_id = COLLECTOR.resolve_new_security_id(
            pd.DataFrame(
                [
                    {
                        "security_id": "MDT-ID",
                        "primary_symbol": "MDT",
                        "provider_symbol": "MDT.US",
                        "active_from": "2015-01-01",
                        "active_to": "",
                    }
                ]
            ),
            new_symbol=evidence.parsed.new_symbol,
            effective_date=evidence.parsed.effective_date,
            symbol_history=pd.DataFrame(
                [
                    {
                        "security_id": "MDT-ID",
                        "symbol": "MDT",
                        "effective_from": "2015-01-01",
                        "effective_to": "",
                    }
                ]
            ),
        )
        assert successor_id == "MDT-ID"
        crosscheck = COLLECTOR._crosscheck(
            evidence.to_dict(),
            successor_security_id=successor_id,
            price_histories={
                "COV-ID": pd.DataFrame(
                    {
                        "session": pd.to_datetime(["2015-01-26"]),
                        "close": [106.70],
                    }
                ),
                "MDT-ID": pd.DataFrame(
                    {
                        "session": pd.to_datetime(
                            ["2015-01-26", "2015-01-27"]
                        ),
                        "close": [75.59, 75.26],
                    }
                ),
            },
        )
        assert crosscheck["passed"] is True
        assert crosscheck["economic_terms_passed"] is True
        eligible = bool(
            evidence.parsed.confidence == "high"
            and crosscheck["passed"]
            and successor_id
        )
        assert eligible is True

        finalizer_cache = FINALIZER._ArtifactCache(cache_root)
        for artifact, content in zip(artifacts, contents, strict=True):
            assert finalizer_cache.content(artifact.source_hash) == content
        trusted = FINALIZER._artifact_from_event(
            {
                "source_url": artifacts[0].source_url,
                "source_hash": artifacts[0].source_hash,
            },
            {},
            finalizer_cache,
            trusted_override=True,
        )
        assert trusted.content == contents[0]


def test_terminal_zero_recovery_hints_use_legal_cancellation_dates() -> None:
    hints = COLLECTOR._load_hints(COLLECTOR.DEFAULT_HINTS)
    expected = {
        "SIVB": ("2024-11-07", "719739"),
        "MNK": ("2022-06-16", "1567892"),
        "BBBY": ("2023-09-29", "886158"),
    }

    for symbol, (effective_date, cik) in expected.items():
        event = hints[symbol]["verified_event"]
        assert event["action_type"] == "delisting"
        assert str(event["effective_date"]) == effective_date
        assert float(event["cash_amount"]) == 0.0
        assert event["cik"] == cik
        assert event["source_urls"][0].startswith(
            "https://www.sec.gov/Archives/edgar/data/"
        )

    mnk = hints["MNK"]
    assert mnk["manual_review"] is True
    assert "separate security identities" in mnk["manual_review_reason"]
    assert "no successor link" in mnk["manual_review_reason"]


def test_identity_bound_hints_separate_reused_agn_and_exact_successors() -> None:
    symbols = COLLECTOR._load_hints(COLLECTOR.DEFAULT_HINTS)
    exact = COLLECTOR._load_identity_bound_hints(COLLECTOR.DEFAULT_HINTS)
    assert "verified_event" not in symbols["AGN"]
    assert len(exact) == 10
    eca_key = "US:EODHD:53a2ef22-39f2-506c-80b4-bb0f698f43dd|2020-01-24"
    assert exact[eca_key]["candidate_symbol"] == "ECA"
    assert exact[eca_key]["verified_event"]["new_symbol"] == "OVV"
    assert float(exact[eca_key]["verified_event"]["ratio"]) == 0.2
    ntco_key = COLLECTOR._NTCO_RELEASE_EVENT_KEY
    assert exact[ntco_key] == {
        "candidate_symbol": "NTCOY",
        "expected_action": "delisting",
        "existing_release_event": {
            "contract": COLLECTOR._NTCO_RELEASE_EVENT_CONTRACT
        },
    }

    legacy = FINALIZER.LifecycleCandidate(
        security_id="US:EODHD:9f13974d-7f81-5aac-a3a7-ed1d184bd76b",
        symbol="AGN",
        name="Allergan Inc",
        exchange="NYSE",
        last_price_date="2015-03-16",
        active_to="2015-03-16",
    )
    actavis = replace(
        legacy,
        security_id="US:EODHD:79ce1c42-8ff6-5c13-b9cf-3df82c913734",
        name="Allergan plc (formerly Actavis plc)",
        last_price_date="2020-05-08",
    )
    legacy_event = COLLECTOR._hint_for_candidate(legacy, symbols, exact)[
        "verified_event"
    ]
    actavis_event = COLLECTOR._hint_for_candidate(actavis, symbols, exact)[
        "verified_event"
    ]
    assert str(legacy_event["effective_date"]) == "2015-03-17"
    assert legacy_event["new_symbol"] == "ACT"
    assert str(actavis_event["effective_date"]) == "2020-05-08"
    assert float(actavis_event["cash_amount"]) == 120.30
    assert float(actavis_event["ratio"]) == 0.866
    assert actavis_event["new_symbol"] == "ABBV"

    dupont = replace(
        legacy,
        security_id="US:SEC:7992d65b-3a26-5cae-b96d-bd01f695d1c1",
        symbol="DD",
        name="E. I. du Pont de Nemours and Company",
        last_price_date="2017-08-31",
    )
    dupont_event = COLLECTOR._hint_for_candidate(dupont, symbols, exact)[
        "verified_event"
    ]
    assert str(dupont_event["effective_date"]) == "2017-09-01"
    assert float(dupont_event["ratio"]) == 1.282
    assert dupont_event["new_symbol"] == "DWDP"
    assert dupont_event["source_urls"] == [
        "https://www.sec.gov/Archives/edgar/data/30554/"
        "000119312517274840/0001193125-17-274840.txt"
    ]

    colliding = {**symbols, "AGN": {"verified_event": legacy_event}}
    with pytest.raises(RuntimeError, match="collides with an exact identity-bound"):
        COLLECTOR._hint_for_candidate(actavis, colliding, exact)

    spectra = replace(
        legacy,
        security_id="US:EODHD:5fa7bd33-c752-57c7-873c-e9d812d90e05",
        symbol="SE",
        name="Spectra Energy Corp",
        last_price_date="2017-02-24",
    )
    spectra_event = COLLECTOR._hint_for_candidate(spectra, symbols, exact)[
        "verified_event"
    ]
    assert str(spectra_event["effective_date"]) == "2017-02-27"
    assert float(spectra_event["ratio"]) == 0.984
    assert spectra_event["new_symbol"] == "ENB"
    assert spectra_event["source_urls"] == [
        "https://www.sec.gov/Archives/edgar/data/1373835/"
        "000119312517057856/0001193125-17-057856.txt"
    ]

    reused_sea = replace(
        spectra,
        security_id="US:EODHD:cec57207-c56c-51c0-955f-204bca9b27c8",
        name="Sea Ltd",
        last_price_date="2026-07-15",
    )
    assert COLLECTOR._hint_for_candidate(reused_sea, symbols, exact) == {}


def _ntco_postapply_repository_fixture(tmp_path: Path):  # type: ignore[no-untyped-def]
    base = LocalDatasetRepository(Path(__file__).resolve().parents[2] / "data/cache")
    release, _ = base.current_release()
    assert release is not None
    quarantine = NTCO_PLAN.read_quarantine(
        base.root, NTCO_PLAN.OBSERVED_UNREVIEWED_QUARANTINE_ID
    )
    supplemental = NTCO_PLAN._load_supplemental_official_artifacts(
        NTCO_PLAN.validate_pin_contract()
    )
    current_prices, current_dividends = NTCO_PLAN.current_overlap_records(
        base, release
    )
    decision_contract = next(
        row
        for row in COLLECTOR._NTCO_ARCHIVE_CONTRACT
        if row["source"] == "reviewed_ntco_ntcoy_transition_decision"
    )
    current_archive = base.read_frame(
        "source_archive", release.dataset_versions["source_archive"]
    )
    decision_row = current_archive.loc[
        current_archive["archive_id"].astype(str).eq(
            decision_contract["archive_id"]
        )
    ]
    assert len(decision_row) == 1
    decision_path = base.root / str(decision_row.iloc[0]["object_path"])
    exact_scope_audit = json.loads(
        gzip.decompress(decision_path.read_bytes())
    )["release_scope_audit"]
    bundle = NTCO_PLAN.bundle_from_artifacts(
        quarantine.artifacts,
        current_prices=current_prices,
        current_dividends=current_dividends,
        budget_receipt=quarantine.budget_receipt,
        base_release_version=release.version,
        supplemental_artifacts=supplemental,
        # The reviewed decision artifact binds the exact pre-apply release
        # scope audit.  Reusing that immutable audit keeps this fixture stable
        # when the actual cache has since advanced through later finalizers.
        release_scope_audit=exact_scope_audit,
    )
    artifacts = (
        *bundle.artifacts,
        *bundle.supplemental_artifacts,
        *NTCO_PLAN._reviewed_extraction_artifacts(bundle),
    )
    archive = pd.DataFrame(
        [
            NTCO_PLAN._archive_row(item, release.completed_session)
            for item in artifacts
        ]
    )
    for item in artifacts:
        suffix = "json" if item.content_type == "application/json" else "bin"
        path = (
            tmp_path
            / "repo"
            / "archives"
            / release.completed_session
            / f"{item.source_hash}.{suffix}.gz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(item.content, mtime=0))

    actions = base.read_frame(
        "corporate_actions", release.dataset_versions["corporate_actions"]
    )
    official_actions = NTCO_PLAN._official_actions(bundle)
    # The actual cache may already be post-NTCO.  Rebuild the fixture from one
    # exact copy of the reviewed action inventory instead of appending a second
    # copy to whichever release happens to be current.
    actions = actions.loc[
        ~actions["event_id"].astype(str).isin(
            set(official_actions["event_id"].astype(str))
        )
    ].copy()
    actions = pd.concat(
        [actions, official_actions],
        ignore_index=True,
        sort=False,
    )
    resolutions = base.read_frame(
        "lifecycle_resolutions",
        release.dataset_versions["lifecycle_resolutions"],
    )
    resolutions = resolutions.loc[
        ~resolutions["security_id"].astype(str).eq(COLLECTOR._NTCO_SECURITY_ID)
    ].copy()
    row = {column: "" for column in resolutions.columns}
    row.update(
        {
            "candidate_id": COLLECTOR.lifecycle_candidate_id(
                COLLECTOR._NTCO_SECURITY_ID, "2024-08-07"
            ),
            "security_id": COLLECTOR._NTCO_SECURITY_ID,
            "symbol": "NTCOY",
            "last_price_date": "2024-08-07",
            "resolution": "applied",
            "event_id": COLLECTOR._NTCO_TERMINAL_EVENT_ID,
            "reviewed_by": "us_ntco_ntcoy_transition_repair_v1",
            "reviewed_at": COLLECTOR._NTCO_REVIEWED_AT,
            "source_url": (
                "https://www.adrbny.com/content/dam/adr/documents/"
                "corporate-actions-dr/files/ad1145447.pdf"
            ),
            "source": "official_ntcoy_cash_termination",
            "retrieved_at": COLLECTOR._NTCO_RETRIEVED_AT,
            "source_hash": COLLECTOR._NTCO_TERMINAL_SOURCE_HASH,
        }
    )
    resolutions = pd.concat(
        [resolutions, pd.DataFrame([row]).loc[:, resolutions.columns]],
        ignore_index=True,
        sort=False,
    )
    frames = {
        "corporate_actions": actions,
        "lifecycle_resolutions": resolutions,
        "source_archive": archive,
    }

    class Repository:
        root = tmp_path / "repo"

        def current_release(self):  # type: ignore[no-untyped-def]
            return release, None

        def read_frame(self, dataset, _version):  # type: ignore[no-untyped-def]
            return frames[dataset].copy(deep=True)

    candidate = FINALIZER.LifecycleCandidate(
        security_id=COLLECTOR._NTCO_SECURITY_ID,
        symbol="NTCOY",
        name="Natura &Co Holding S.A.",
        exchange="OTC",
        last_price_date="2024-08-07",
        active_to="2024-09-04",
    )
    return Repository(), release, candidate, frames


def test_ntco_postapply_event_replays_action_resolution_and_archives_without_http(
    tmp_path: Path,
) -> None:
    repository, release, candidate, _ = _ntco_postapply_repository_fixture(tmp_path)
    evidence, artifacts = COLLECTOR._collect_existing_release_event(
        repository,
        release,
        candidate,
        {"contract": COLLECTOR._NTCO_RELEASE_EVENT_CONTRACT},
    )

    assert evidence.parsed is not None
    assert evidence.parsed.action_type == "delisting"
    assert evidence.parsed.effective_date == "2024-09-04"
    assert evidence.parsed.cash_amount == pytest.approx(5.043659)
    assert evidence.source_hash == COLLECTOR._NTCO_TERMINAL_SOURCE_HASH
    assert len(artifacts) == len(COLLECTOR._NTCO_ARCHIVE_CONTRACT) == 11
    assert {item.source_hash for item in artifacts} == {
        row["source_hash"] for row in COLLECTOR._NTCO_ARCHIVE_CONTRACT
    }


@pytest.mark.parametrize("target", ("action", "resolution", "archive"))
def test_ntco_postapply_event_fails_closed_on_exact_contract_tampering(
    tmp_path: Path,
    target: str,
) -> None:
    repository, release, candidate, frames = _ntco_postapply_repository_fixture(
        tmp_path
    )
    if target == "action":
        mask = frames["corporate_actions"]["event_id"].astype(str).eq(
            COLLECTOR._NTCO_TERMINAL_EVENT_ID
        )
        frames["corporate_actions"].loc[mask, "cash_amount"] = 5.0
    elif target == "resolution":
        frames["lifecycle_resolutions"].loc[
            frames["lifecycle_resolutions"]["security_id"]
            .astype(str)
            .eq(COLLECTOR._NTCO_SECURITY_ID),
            "source_hash",
        ] = "0" * 64
    else:
        decision = next(
            row
            for row in COLLECTOR._NTCO_ARCHIVE_CONTRACT
            if row["source"] == "reviewed_ntco_ntcoy_transition_decision"
        )
        path = (
            repository.root
            / "archives"
            / release.completed_session
            / f"{decision['source_hash']}.json.gz"
        )
        path.write_bytes(gzip.compress(b"tampered", mtime=0))

    with pytest.raises(RuntimeError, match="NTCO"):
        COLLECTOR._collect_existing_release_event(
            repository,
            release,
            candidate,
            {"contract": COLLECTOR._NTCO_RELEASE_EVENT_CONTRACT},
        )


def test_existing_release_event_hint_cannot_expand_beyond_exact_ntco_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hints.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "identity_bound_hints": {
                    "OTHER|2024-08-07": {
                        "candidate_symbol": "OTHER",
                        "existing_release_event": {
                            "contract": COLLECTOR._NTCO_RELEASE_EVENT_CONTRACT
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="restricted to the exact NTCOY"):
        COLLECTOR._load_identity_bound_hints(path)


def test_official_exception_inventory_is_exact_and_identity_bound() -> None:
    specs = COLLECTOR.load_official_lifecycle_exception_evidence(
        COLLECTOR.DEFAULT_HINTS
    )

    assert set(specs) == {
        "aaba_2019_liquidation_distributions",
        "abmd_2022_cvr_consideration",
        "brcm_2016_election_proration",
        "celg_2019_cvr_consideration",
        "dvmt_2018_class_v_election_proration",
        "ggp_2018_election_proration",
        "legacy_dnr_2020_warrant_consideration",
        "legacy_do_2021_warrant_consideration",
        "legacy_ne_2021_warrant_consideration",
        "legacy_val_2021_warrant_consideration",
        "mallinckrodt_2022_cancellation",
        "mallinckrodt_2023_cancellation",
        "para_2025_election_proration",
        "tfcf_2019_disney_proration",
        "tfcfa_2019_disney_proration",
        "twc_2016_election_proration",
        "utx_2020_carr_otis_distributions",
    }
    reviewed_pins = {
        "aaba_2019_liquidation_distributions",
        "abmd_2022_cvr_consideration",
        "brcm_2016_election_proration",
        "celg_2019_cvr_consideration",
        "dvmt_2018_class_v_election_proration",
        "ggp_2018_election_proration",
        "legacy_dnr_2020_warrant_consideration",
        "legacy_do_2021_warrant_consideration",
        "legacy_ne_2021_warrant_consideration",
        "legacy_val_2021_warrant_consideration",
        "mallinckrodt_2022_cancellation",
        "mallinckrodt_2023_cancellation",
        "para_2025_election_proration",
        "tfcf_2019_disney_proration",
        "tfcfa_2019_disney_proration",
        "twc_2016_election_proration",
        "utx_2020_carr_otis_distributions",
    }
    assert all(specs[evidence_id].pinned for evidence_id in reviewed_pins)
    assert all(
        not specs[evidence_id].pinned
        for evidence_id in set(specs) - reviewed_pins
    )
    assert "frc_fdic_receivership" not in specs
    assert (
        "frc_fdic_receivership"
        not in official_lifecycle_evidence.OFFICIAL_EXCEPTION_EVIDENCE_URL_ALLOWLIST
    )
    assert specs["legacy_val_2021_warrant_consideration"].binding_complete
    dvmt = specs["dvmt_2018_class_v_election_proration"]
    assert dvmt.candidate_security_ids == (
        "US:EODHD:1123af95-e37a-5697-b0a1-1a6c126fa501",
    )
    assert dvmt.candidate_last_price_dates == ("2018-12-27",)
    assert dvmt.effective_date == "2018-12-28"
    assert dvmt.exception_code == "unsupported_consideration"
    assert dvmt.source_sha256 == (
        "06755cccff50fabc62963bc7817c15575618a57d7954d415862dd82da36aa25f"
    )
    for evidence_id, security_id, terminal_date, effective_date, source_sha256 in (
        (
            "legacy_do_2021_warrant_consideration",
            "US:EODHD:2826c370-0467-5e82-9617-dcece5be407f",
            "2020-04-24",
            "2021-04-23",
            "048886a54f9b70198d2e70805aedc56a4eb11c4e84f970f89bc931006594d210",
        ),
        (
            "legacy_dnr_2020_warrant_consideration",
            "US:EODHD:6d9d4638-4922-5f6c-89fd-6b79db60c1c3",
            "2020-07-28",
            "2020-09-18",
            "79295f28796f0e0c5de91e88a6accde76517e25267a189eccbf852674cb78229",
        ),
        (
            "legacy_ne_2021_warrant_consideration",
            "US:EODHD:81b3ca1f-cf1b-5234-bc24-4399b8ecf149",
            "2020-10-22",
            "2021-02-05",
            "9bdaadb02c741ee474a15aa6a95f0674a6fa41d4acd3f99384cc10d9dc60e65f",
        ),
    ):
        spec = specs[evidence_id]
        assert spec.binding_complete
        assert spec.candidate_security_ids == (security_id,)
        assert spec.candidate_last_price_dates == (terminal_date,)
        assert spec.effective_date == effective_date
        assert spec.exception_code == "unsupported_consideration"
        assert spec.pinned
        assert spec.source_sha256 == source_sha256
    assert specs["mallinckrodt_2022_cancellation"].binding_complete
    assert specs["mallinckrodt_2023_cancellation"].binding_complete
    assert specs["mallinckrodt_2022_cancellation"].candidate_security_ids == (
        "US:EODHD:81d711c5-9688-5f2b-9f36-63c8fe3211bf",
    )
    assert specs["mallinckrodt_2023_cancellation"].candidate_security_ids == (
        "US:EODHD:647b8b62-0015-5a56-8a63-4da7ba287025",
    )
    assert specs["legacy_val_2021_warrant_consideration"].required_text_groups[1] == (
        "5,645,161 warrants",
    )
    assert specs["mallinckrodt_2022_cancellation"].required_text_groups[0] == (
        "June 16, 2022",
    )
    assert specs["mallinckrodt_2023_cancellation"].required_text_groups[0] == (
        "November 14, 2023",
    )
    assert (
        specs["mallinckrodt_2022_cancellation"].resolution_kind
        == "applied_event"
    )
    assert specs["mallinckrodt_2022_cancellation"].cash_amount == 0.0
    assert (
        specs["mallinckrodt_2023_cancellation"].resolution_kind
        == "applied_event"
    )
    assert specs["tfcf_2019_disney_proration"].candidate_security_ids == (
        "US:EODHD:acd9ed55-bf0c-5b15-b624-1a917bf6078e",
    )
    assert specs["tfcfa_2019_disney_proration"].candidate_security_ids == (
        "US:EODHD:9398e16f-425d-5a51-8720-35fba7433f28",
    )
    assert specs["tfcf_2019_disney_proration"].candidate_last_price_dates == (
        "2019-03-19",
    )
    assert specs["tfcf_2019_disney_proration"].exception_code == (
        "unsupported_consideration"
    )
    assert specs["tfcf_2019_disney_proration"].source_sha256 == (
        "08ba720b0e5326b652fb94cde8ba44c45bcac09a81b77d70f006e934e9d36d93"
    )
    assert (
        specs["tfcfa_2019_disney_proration"].source_sha256
        == specs["tfcf_2019_disney_proration"].source_sha256
    )
    assert specs["tfcf_2019_disney_proration"].required_text_groups == (
        ("Twenty-First Century Fox, Inc.",),
        ("NASDAQ: TFCFA, TFCF",),
        ("acquisition of 21CF",),
        ("March 20, 2019",),
        ("each share of 21CF common stock",),
        ("$51.572626", "51.572626"),
        ("0.4517",),
        ("election", "elections"),
        ("proration", "prorated"),
    )
    assert specs["abmd_2022_cvr_consideration"].candidate_last_price_dates == (
        "2022-12-21",
    )
    assert specs["celg_2019_cvr_consideration"].candidate_security_ids == (
        "US:EODHD:0337dd23-67ad-5354-b972-50babd1ae5a0",
    )
    assert specs["tfcf_2019_disney_proration"].source_url == (
        "https://www.sec.gov/Archives/edgar/data/1308161/"
        "000119312519079716/d710665dex991.htm"
    )

    muniholdings = FINALIZER.LifecycleCandidate(
        security_id="US:EODHD:81d711c5-9688-5f2b-9f36-63c8fe3211bf",
        symbol="MNK",
        name="Muniholdings New York Insured Fund III Inc",
        exchange="NYSE MKT",
        last_price_date="2020-10-12",
        active_to="2020-10-12",
    )
    assert not specs["mallinckrodt_2023_cancellation"].matches_candidate(
        muniholdings
    )


def test_bound_applied_event_adds_terminal_non_index_candidate() -> None:
    spec = COLLECTOR.load_official_lifecycle_exception_evidence(
        COLLECTOR.DEFAULT_HINTS
    )["mallinckrodt_2023_cancellation"]
    security_id = spec.candidate_security_ids[0]
    frames = {
        "security_master": pd.DataFrame(
            [
                {
                    "security_id": security_id,
                    "primary_symbol": "MNK",
                    "name": "Mallinckrodt plc reorganized ordinary shares",
                    "exchange": "NYSE MKT",
                    "active_to": "2023-11-13",
                }
            ]
        ),
        "symbol_history": pd.DataFrame(
            [
                {
                    "security_id": security_id,
                    "symbol": "MNK",
                    "effective_from": "2022-06-17",
                    "effective_to": "2023-11-13",
                }
            ]
        ),
        "daily_price_raw": pd.DataFrame(
            [{"security_id": security_id, "session": "2023-11-13"}]
        ),
    }

    class Repository:
        def read_frame(self, dataset, _version):
            return frames[dataset].copy()

    release = SimpleNamespace(
        dataset_versions={dataset: "v1" for dataset in frames}
    )
    result = COLLECTOR.include_bound_official_applied_event_candidates(
        (), Repository(), release, {spec.evidence_id: spec}
    )

    assert len(result) == 1
    assert result[0].security_id == security_id
    assert result[0].last_price_date == "2023-11-13"
    assert spec.matches_candidate(result[0])


def test_official_exception_two_stage_fetch_then_pinned_offline_replay() -> None:
    spec = COLLECTOR.load_official_lifecycle_exception_evidence(
        COLLECTOR.DEFAULT_HINTS
    )["aaba_2019_liquidation_distributions"]
    spec = replace(spec, source_sha256="")
    candidate = FINALIZER.LifecycleCandidate(
        security_id="US:EODHD:9b1bbdaa-839c-5d59-8bda-99b2087022e6",
        symbol="AABA",
        name="Altaba Inc",
        exchange="NASDAQ",
        last_price_date="2019-10-02",
        active_to="2019-10-02",
    )
    content = (
        b"<html>October 4, 2019. Plan of Complete Liquidation and Dissolution. "
        b"Rights to receive any post-dissolution liquidating distributions.</html>"
    )
    response = SimpleNamespace(
        status_code=200,
        url=spec.source_url,
        content=content,
        headers={"Content-Type": "text/html; charset=utf-8"},
    )

    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory)
        session = SimpleNamespace(headers={}, get=Mock(return_value=response))
        source = COLLECTOR.OfficialLifecycleExceptionEvidenceSource(
            cache,
            allow_http=True,
            session=session,
            user_agent="SuperTrendQuant test@example.com",
        )
        report = {
            "records": {
                candidate.security_id: {
                    "candidate": {
                        "security_id": candidate.security_id,
                        "symbol": candidate.symbol,
                        "name": candidate.name,
                        "last_price_date": candidate.last_price_date,
                    },
                    "artifacts": [],
                }
            },
            "official_exception_evidence": {
                "frc_fdic_receivership": {"status": "verified_pinned_attached"}
            },
        }

        COLLECTOR._collect_official_exception_evidence(
            report,
            candidates=[candidate],
            specs={spec.evidence_id: spec},
            source=source,
            requested_symbols={"AABA"},
            require_pinned=False,
        )

        session.get.assert_called_once_with(
            spec.source_url,
            timeout=60,
            allow_redirects=False,
        )
        observed = hashlib.sha256(content).hexdigest()
        first = report["official_exception_evidence"][spec.evidence_id]
        assert "frc_fdic_receivership" not in report["official_exception_evidence"]
        assert first["status"] == "observed_unpinned_attached"
        assert first["observed_sha256"] == observed
        assert first["pinned_sha256"] == ""
        assert len(list(cache.glob("official-exception-*.bin"))) == 1
        assert len(list(cache.glob("official-exception-*.json"))) == 1

        pinned = replace(spec, source_sha256=observed)
        offline_session = SimpleNamespace(headers={}, get=Mock())
        replay = COLLECTOR.OfficialLifecycleExceptionEvidenceSource(
            cache,
            allow_http=False,
            session=offline_session,
        )
        COLLECTOR._collect_official_exception_evidence(
            report,
            candidates=[candidate],
            specs={pinned.evidence_id: pinned},
            source=replay,
            requested_symbols={"AABA"},
            require_pinned=True,
        )

        offline_session.get.assert_not_called()
        second = report["official_exception_evidence"][spec.evidence_id]
        assert second["status"] == "verified_pinned_attached"
        artifacts = report["records"][candidate.security_id]["artifacts"]
        assert len(artifacts) == 1
        assert artifacts[0]["source_hash"] == observed
        assert artifacts[0]["pin_status"] == "verified_pinned"


@pytest.mark.parametrize(
    "evidence_ids",
    [
        ("tfcf_2019_disney_proration", "tfcfa_2019_disney_proration"),
        ("tfcfa_2019_disney_proration", "tfcf_2019_disney_proration"),
    ],
)
def test_shared_official_url_cache_is_evidence_id_independent_in_any_order(
    evidence_ids: tuple[str, str],
) -> None:
    specs = COLLECTOR.load_official_lifecycle_exception_evidence(
        COLLECTOR.DEFAULT_HINTS
    )
    first, second = (
        replace(specs[evidence_id], source_sha256="")
        for evidence_id in evidence_ids
    )
    assert first.source_url == second.source_url
    content = (
        b"<html>Twenty-First Century Fox, Inc. (NASDAQ: TFCFA, TFCF), in "
        b"connection with Disney's acquisition of 21CF. March 20, 2019. "
        b"Each share of 21CF common stock: $51.572626 or 0.4517, subject to "
        b"election and proration.</html>"
    )
    response = SimpleNamespace(
        status_code=200,
        url=first.source_url,
        content=content,
        headers={"Content-Type": "text/html; charset=utf-8"},
    )

    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory)
        session = SimpleNamespace(headers={}, get=Mock(return_value=response))
        source = COLLECTOR.OfficialLifecycleExceptionEvidenceSource(
            cache,
            allow_http=True,
            session=session,
            user_agent="SuperTrendQuant test@example.com",
        )

        first_artifact, _ = source.load(first, require_pinned=False)
        second_artifact, _ = source.load(second, require_pinned=False)

        assert source.http_attempts == 1
        session.get.assert_called_once()
        assert first_artifact.content == second_artifact.content == content
        assert first_artifact.source_hash == second_artifact.source_hash
        assert len(list(cache.glob("official-exception-*.bin"))) == 1
        metadata_files = list(cache.glob("official-exception-*.json"))
        assert len(metadata_files) == 1
        metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
        assert (
            metadata["schema"]
            == official_lifecycle_evidence.OFFICIAL_EXCEPTION_EVIDENCE_SCHEMA
        )
        assert metadata["source_url"] == first.source_url
        assert "evidence_id" not in metadata

        raw_path = next(cache.glob("official-exception-*.bin"))
        raw_path.write_bytes(content + b" tampered")
        offline = COLLECTOR.OfficialLifecycleExceptionEvidenceSource(
            cache, allow_http=False
        )
        with pytest.raises(ValueError, match="cache content hash mismatch"):
            offline.load(second, require_pinned=False)


def test_shared_official_url_legacy_cache_is_compatible_but_id_tampering_fails() -> None:
    specs = COLLECTOR.load_official_lifecycle_exception_evidence(
        COLLECTOR.DEFAULT_HINTS
    )
    tfcf = replace(specs["tfcf_2019_disney_proration"], source_sha256="")
    tfcfa = replace(specs["tfcfa_2019_disney_proration"], source_sha256="")
    content = (
        b"<html>Twenty-First Century Fox, Inc. (NASDAQ: TFCFA, TFCF), in "
        b"connection with Disney's acquisition of 21CF. March 20, 2019. "
        b"Each share of 21CF common stock: $51.572626 or 0.4517; elections "
        b"were prorated.</html>"
    )
    response = SimpleNamespace(
        status_code=200,
        url=tfcf.source_url,
        content=content,
        headers={"Content-Type": "text/html"},
    )

    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory)
        source = COLLECTOR.OfficialLifecycleExceptionEvidenceSource(
            cache,
            allow_http=True,
            session=SimpleNamespace(headers={}, get=Mock(return_value=response)),
            user_agent="SuperTrendQuant test@example.com",
        )
        source.load(tfcf, require_pinned=False)
        metadata_path = next(cache.glob("official-exception-*.json"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["schema"] = (
            official_lifecycle_evidence.LEGACY_OFFICIAL_EXCEPTION_EVIDENCE_SCHEMA
        )
        metadata["evidence_id"] = tfcf.evidence_id
        metadata_path.write_text(
            json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
        )

        offline = COLLECTOR.OfficialLifecycleExceptionEvidenceSource(
            cache, allow_http=False
        )
        assert offline.load(tfcfa, require_pinned=False)[0].content == content

        metadata["evidence_id"] = "frc_fdic_receivership"
        metadata_path.write_text(
            json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="legacy cache id/URL mismatch"):
            offline.load(tfcf, require_pinned=False)


def test_official_exception_verify_rejects_unpinned_and_pending_identity() -> None:
    specs = COLLECTOR.load_official_lifecycle_exception_evidence(
        COLLECTOR.DEFAULT_HINTS
    )
    mnk = replace(
        specs["mallinckrodt_2023_cancellation"],
        candidate_security_ids=(),
        candidate_last_price_dates=(),
        binding_status="pending_identity_repair",
    )
    with tempfile.TemporaryDirectory() as directory:
        source = COLLECTOR.OfficialLifecycleExceptionEvidenceSource(
            Path(directory), allow_http=False
        )
        with pytest.raises(RuntimeError, match="candidate binding is pending"):
            COLLECTOR._collect_official_exception_evidence(
                {"records": {}},
                candidates=[],
                specs={mnk.evidence_id: mnk},
                source=source,
                requested_symbols={"MNK"},
                require_pinned=True,
            )


def test_pinned_mallinckrodt_evidence_promotes_only_repaired_identity() -> None:
    spec = COLLECTOR.load_official_lifecycle_exception_evidence(
        COLLECTOR.DEFAULT_HINTS
    )["mallinckrodt_2023_cancellation"]
    spec = replace(spec, source_sha256="")
    candidate = FINALIZER.LifecycleCandidate(
        security_id="REPAIRED-MNK-2023",
        symbol="MNK",
        name="Mallinckrodt plc predecessor equity",
        exchange="NYSE",
        last_price_date="2023-11-13",
        active_to="2023-11-13",
    )
    content = (
        b"<html>On November 14, 2023, all ordinary shares were cancelled "
        b"with no consideration.</html>"
    )
    response = SimpleNamespace(
        status_code=200,
        url=spec.source_url,
        content=content,
        headers={"Content-Type": "text/html"},
    )
    report = {
        "records": {
            candidate.security_id: {
                "candidate": {
                    "security_id": candidate.security_id,
                    "symbol": candidate.symbol,
                    "name": candidate.name,
                    "last_price_date": candidate.last_price_date,
                },
                "artifacts": [],
                "manual_review": True,
                "manual_review_reason": "identity pending",
                "eligible_for_apply": False,
            }
        }
    }

    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory)
        source = COLLECTOR.OfficialLifecycleExceptionEvidenceSource(
            cache,
            allow_http=True,
            session=SimpleNamespace(headers={}, get=Mock(return_value=response)),
            user_agent="SuperTrendQuant test@example.com",
        )
        COLLECTOR._collect_official_exception_evidence(
            report,
            candidates=[candidate],
            specs={spec.evidence_id: spec},
            source=source,
            requested_symbols={"MNK"},
            require_pinned=False,
        )
        assert "verified_event" not in report["records"][candidate.security_id]
        assert (
            report["official_exception_evidence"][spec.evidence_id]["status"]
            == "observed_unpinned_unbound"
        )

        bound = replace(
            spec,
            candidate_security_ids=(candidate.security_id,),
            candidate_last_price_dates=(candidate.last_price_date,),
            binding_status="bound",
            source_sha256=hashlib.sha256(content).hexdigest(),
        )
        replay = COLLECTOR.OfficialLifecycleExceptionEvidenceSource(
            cache,
            allow_http=False,
            session=SimpleNamespace(headers={}, get=Mock()),
        )
        COLLECTOR._collect_official_exception_evidence(
            report,
            candidates=[candidate],
            specs={bound.evidence_id: bound},
            source=replay,
            requested_symbols={"MNK"},
            require_pinned=True,
        )

        record = report["records"][candidate.security_id]
        assert record["verified_event_evidence_id"] == bound.evidence_id
        assert record["verified_event"]["action_type"] == "delisting"
        assert record["verified_event"]["effective_date"] == "2023-11-14"
        assert record["verified_event"]["cash_amount"] == 0.0
        assert record["verified_event"]["source_hash"] == bound.source_sha256
        assert record["manual_review"] is False
        assert record["manual_review_reason"] == ""
        assert (
            report["official_exception_evidence"][bound.evidence_id]["status"]
            == "verified_pinned_promoted"
        )
        event, override = FINALIZER._event_from_record(record)
        assert override is True
        FINALIZER._validate_applied_record(record, override=override)
        accepted = FINALIZER._artifact_from_event(
            event,
            record,
            FINALIZER._ArtifactCache(cache),
            trusted_override=True,
        )
        assert accepted.source_hash == bound.source_sha256
        assert FINALIZER._exception_for(
            candidate,
            {},
            {bound.evidence_id: bound},
        ) is None


def test_official_exception_config_rejects_non_allowlisted_url() -> None:
    document = yaml.safe_load(COLLECTOR.DEFAULT_HINTS.read_text(encoding="utf-8"))
    document["official_exception_evidence"]["aaba_2019_liquidation_distributions"][
        "source_url"
    ] = "https://example.com/altaba.html"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "hints.yaml"
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError, match="exact reviewed allow-list"):
            COLLECTOR.load_official_lifecycle_exception_evidence(path)


def test_official_sec_exception_fetch_requires_identifying_user_agent() -> None:
    spec = COLLECTOR.load_official_lifecycle_exception_evidence(
        COLLECTOR.DEFAULT_HINTS
    )["legacy_val_2021_warrant_consideration"]
    session = SimpleNamespace(headers={}, get=Mock())
    with tempfile.TemporaryDirectory() as directory:
        source = COLLECTOR.OfficialLifecycleExceptionEvidenceSource(
            Path(directory),
            allow_http=True,
            session=session,
            user_agent="",
        )
        with pytest.raises(RuntimeError, match="SEC_USER_AGENT is required"):
            source.fetch(spec)
    session.get.assert_not_called()


def _report_binding_fixture(
    root: Path,
    *,
    release_version: str = "release-v1",
    candidates: tuple[SimpleNamespace, ...] | None = None,
) -> tuple[dict[str, object], Path]:
    hints_path = root / "hints.yaml"
    if not hints_path.is_file():
        hints_path.write_bytes(b"symbols: {}\n")
    candidate_values = candidates or (
        SimpleNamespace(security_id="SEC-A", last_price_date="2025-01-02"),
    )
    return (
        COLLECTOR.build_lifecycle_report_binding(
            release_version=release_version,
            completed_session="2025-01-03",
            dataset_versions={
                "daily_price_raw": "daily-v1",
                "security_master": "master-v1",
            },
            candidates=candidate_values,
            hints_path=hints_path,
        ),
        hints_path,
    )


def _write_resume_fixture(path: Path, binding: dict[str, object]) -> None:
    report = COLLECTOR._empty_report(binding)
    report["records"] = {"SEC-A": {"sentinel": "must-not-be-laundered"}}
    COLLECTOR._write_report(path, report)


def test_normal_resume_accepts_only_the_same_collection_context() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        binding, _ = _report_binding_fixture(root)
        path = root / "report.json"
        _write_resume_fixture(path, binding)

        resumed = COLLECTOR._load_or_initialize_report(
            path,
            resume=True,
            expected_binding=binding,
        )

        assert resumed["records"]["SEC-A"]["sentinel"] == "must-not-be-laundered"
        assert resumed["collection_context_sha256"] == binding[
            "collection_context_sha256"
        ]


def test_normal_resume_rejects_stale_release_instead_of_relabeling_records() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        stale, _ = _report_binding_fixture(root, release_version="release-old")
        current, _ = _report_binding_fixture(root, release_version="release-current")
        path = root / "report.json"
        _write_resume_fixture(path, stale)

        with pytest.raises(RuntimeError, match="field=release_version"):
            COLLECTOR._load_or_initialize_report(
                path,
                resume=True,
                expected_binding=current,
            )
        persisted = json.loads(path.read_text(encoding="utf-8"))
        assert persisted["release_version"] == "release-old"


def test_normal_resume_rejects_changed_exact_hints_bytes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        original, hints_path = _report_binding_fixture(root)
        path = root / "report.json"
        _write_resume_fixture(path, original)
        hints_path.write_bytes(b"symbols:\n  AAA:\n    expected_action: merger\n")
        changed, _ = _report_binding_fixture(root)

        with pytest.raises(RuntimeError, match="field=hints_sha256"):
            COLLECTOR._load_or_initialize_report(
                path,
                resume=True,
                expected_binding=changed,
            )


def test_normal_resume_rejects_changed_candidate_set() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        original, _ = _report_binding_fixture(root)
        changed, _ = _report_binding_fixture(
            root,
            candidates=(
                SimpleNamespace(
                    security_id="SEC-B", last_price_date="2025-01-02"
                ),
            ),
        )
        path = root / "report.json"
        _write_resume_fixture(path, original)

        with pytest.raises(RuntimeError, match="field=candidate_set_sha256"):
            COLLECTOR._load_or_initialize_report(
                path,
                resume=True,
                expected_binding=changed,
            )


def test_normal_resume_rejects_tampered_collector_version() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        binding, _ = _report_binding_fixture(root)
        path = root / "report.json"
        report = COLLECTOR._empty_report(binding)
        report["collector_version"] = "tampered-collector"
        COLLECTOR._write_report(path, report)

        with pytest.raises(RuntimeError, match="field=collector_version"):
            COLLECTOR._load_or_initialize_report(
                path,
                resume=True,
                expected_binding=binding,
            )


def test_sec_source_defaults_to_offline_and_names_exact_cache_miss_url() -> None:
    url = "https://www.sec.gov/Archives/edgar/data/1/fixture.txt"
    session = SimpleNamespace(headers={}, get=Mock())
    with tempfile.TemporaryDirectory() as directory:
        source = COLLECTOR.SecEdgarLifecycleSource(
            cache_dir=Path(directory),
            session=session,
        )

        with pytest.raises(RuntimeError, match="offline/cache-only.*fixture.txt"):
            source.fetch_url(url)

        candidate = FINALIZER.LifecycleCandidate(
            security_id="SEC-OFFLINE",
            symbol="OFF",
            name="Offline Corp",
            exchange="NYSE",
            last_price_date="2025-01-02",
            active_to="2025-01-02",
        )
        with source.candidate_http_scope(candidate):
            with pytest.raises(
                RuntimeError,
                match="offline/cache-only.*search-index",
            ):
                source.collect(candidate)

        assert source.allow_http is False
        assert source.http_attempts == 0
        session.get.assert_not_called()


def _release_archive_replay_fixture(
    root: Path,
    *,
    candidate,
    url: str,
    payload: bytes = b"exact archived SEC filing",
    source: str = "official_identity_evidence_raw",
    content_type: str = "text/html",
):
    source_hash = hashlib.sha256(payload).hexdigest()
    object_path = f"archives/replay/{source_hash}.html.gz"
    archive_path = root / object_path
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(gzip.compress(payload, mtime=0))
    release = SimpleNamespace(
        version="release-bound",
        completed_session="2025-01-02",
        dataset_versions={
            "security_master": "master-v1",
            "source_archive": "archive-v1",
        },
    )
    archive = pd.DataFrame(
        [
            {
                "archive_id": source_hash,
                "dataset": source,
                "object_path": object_path,
                "content_type": content_type,
                "source": source,
                "source_hash": source_hash,
                "source_url": url,
            }
        ]
    )
    repository = SimpleNamespace(
        root=root,
        read_frame=Mock(return_value=archive),
        current_release=Mock(return_value=(release, "etag-bound")),
    )
    replay = COLLECTOR._CurrentReleaseSecArchiveReplay(
        repository,
        release,
        {candidate: (url,)},
    )
    return replay, repository, release, archive, payload


def test_sec_source_replays_exact_current_release_archive_without_cache_writes() -> None:
    url = "https://www.sec.gov/Archives/edgar/data/1/000000000125000001/event.htm"
    candidate = FINALIZER.LifecycleCandidate(
        security_id="SEC-ARCHIVE",
        symbol="ARC",
        name="Archive Corp",
        exchange="NYSE",
        last_price_date="2025-01-02",
        active_to="2025-01-02",
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        replay, repository, _, _, payload = _release_archive_replay_fixture(
            root,
            candidate=candidate,
            url=url,
        )
        cache = root / "state/sec_lifecycle"
        session = SimpleNamespace(headers={}, get=Mock())
        source = COLLECTOR.SecEdgarLifecycleSource(
            cache_dir=cache,
            session=session,
            archive_replay=replay,
        )

        with source.candidate_http_scope(candidate):
            observed, artifact = source.fetch_url(url)

        assert observed == payload
        assert artifact.source_url == url
        assert artifact.source_hash == hashlib.sha256(payload).hexdigest()
        assert source.http_attempts == 0
        assert not cache.exists()
        repository.read_frame.assert_called_once_with("source_archive", "archive-v1")
        session.get.assert_not_called()


def test_sec_replay_delegates_provenance_qualified_id_to_exact_validator() -> None:
    url = "https://www.sec.gov/Archives/edgar/data/876661/fixture/ruleprovisionnotice.htm"
    candidate = FINALIZER.LifecycleCandidate(
        security_id="SEC-RULE-NOTICE",
        symbol="ECA",
        name="Encana Corporation",
        exchange="NYSE",
        last_price_date="2020-01-24",
        active_to="2020-01-24",
    )
    with tempfile.TemporaryDirectory() as directory:
        replay, _, _, archive, payload = _release_archive_replay_fixture(
            Path(directory),
            candidate=candidate,
            url=url,
            source="sec_rule_provision_notice",
        )
        source_hash = str(archive.loc[0, "source_hash"])
        archive.loc[0, "archive_id"] = hashlib.sha256(
            f"sec_rule_provision_notice|{url}|{source_hash}".encode()
        ).hexdigest()
        replay.archive = archive

        with patch.object(
            COLLECTOR,
            "validate_source_archive_id",
            return_value=str(archive.loc[0, "archive_id"]),
        ) as validate_id:
            assert replay(url, candidate) == payload
        validate_id.assert_called_once_with(
            str(archive.loc[0, "archive_id"]),
            source="sec_rule_provision_notice",
            source_url=url,
            source_hash=source_hash,
        )


def test_sec_replay_rejects_unbound_provenance_qualified_id() -> None:
    url = "https://www.sec.gov/Archives/edgar/data/1/fixture.htm"
    candidate = FINALIZER.LifecycleCandidate(
        security_id="SEC-BAD-ROW-ID",
        symbol="BAD",
        name="Bad Row Id Corp",
        exchange="NYSE",
        last_price_date="2025-01-02",
        active_to="2025-01-02",
    )
    with tempfile.TemporaryDirectory() as directory:
        replay, _, _, archive, _ = _release_archive_replay_fixture(
            Path(directory),
            candidate=candidate,
            url=url,
        )
        archive.loc[0, "archive_id"] = "0" * 64
        replay.archive = archive

        with pytest.raises(RuntimeError, match="invalid archive_id/source_hash"):
            replay(url, candidate)


def test_release_archive_replay_rejects_candidate_or_release_mismatch() -> None:
    url = "https://www.sec.gov/Archives/edgar/data/1/000000000125000001/event.htm"
    candidate = FINALIZER.LifecycleCandidate(
        security_id="SEC-OWNER",
        symbol="OWN",
        name="Owner Corp",
        exchange="NYSE",
        last_price_date="2025-01-02",
        active_to="2025-01-02",
    )
    other = FINALIZER.LifecycleCandidate(
        security_id="SEC-OTHER",
        symbol="OTH",
        name="Other Corp",
        exchange="NYSE",
        last_price_date="2025-01-02",
        active_to="2025-01-02",
    )
    with tempfile.TemporaryDirectory() as directory:
        replay, repository, release, _, _ = _release_archive_replay_fixture(
            Path(directory),
            candidate=candidate,
            url=url,
        )

        with pytest.raises(RuntimeError, match="not bound.*active.*candidate"):
            replay(url, other)

        changed = SimpleNamespace(
            version="release-changed",
            completed_session=release.completed_session,
            dataset_versions=dict(release.dataset_versions),
        )
        repository.current_release.return_value = (changed, "etag-changed")
        with pytest.raises(RuntimeError, match="Current release changed"):
            replay(url, candidate)


def test_release_archive_replay_rejects_manifest_ambiguity_and_hash_mismatch() -> None:
    url = "https://www.sec.gov/Archives/edgar/data/1/000000000125000001/event.htm"
    candidate = FINALIZER.LifecycleCandidate(
        security_id="SEC-STRICT",
        symbol="STC",
        name="Strict Corp",
        exchange="NYSE",
        last_price_date="2025-01-02",
        active_to="2025-01-02",
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        replay, _, _, archive, _ = _release_archive_replay_fixture(
            root,
            candidate=candidate,
            url=url,
        )
        duplicate = pd.concat([archive, archive], ignore_index=True)
        replay.archive = duplicate
        with pytest.raises(RuntimeError, match="ambiguous.*matches=2"):
            replay(url, candidate)

        replay.archive = archive.copy()
        replay.archive.loc[0, "dataset"] = "cov_official_evidence_manifest"
        replay.archive.loc[0, "source"] = "cov_official_evidence_manifest"
        replay.archive.loc[0, "content_type"] = "application/json"
        with pytest.raises(RuntimeError, match="not a direct raw official"):
            replay(url, candidate)

        replay.archive = archive.copy()
        path = root / str(replay.archive.loc[0, "object_path"])
        path.write_bytes(gzip.compress(b"tampered SEC payload", mtime=0))
        with pytest.raises(RuntimeError, match="payload hash does not match"):
            replay(url, candidate)


def test_release_archive_replay_collapses_distinct_roles_for_same_immutable_bytes() -> None:
    url = "https://www.sec.gov/Archives/edgar/data/876661/fixture/ruleprovisionnotice.htm"
    candidate = FINALIZER.LifecycleCandidate(
        security_id="SEC-SAME-BYTES",
        symbol="ECA",
        name="Encana Corporation",
        exchange="NYSE",
        last_price_date="2020-01-24",
        active_to="2020-01-24",
    )
    with tempfile.TemporaryDirectory() as directory:
        replay, _, _, archive, payload = _release_archive_replay_fixture(
            Path(directory),
            candidate=candidate,
            url=url,
            source="sec_rule_provision_notice",
        )
        source_hash = str(archive.loc[0, "source_hash"])
        archive.loc[0, "archive_id"] = hashlib.sha256(
            f"sec_rule_provision_notice|{url}|{source_hash}".encode()
        ).hexdigest()
        generic = archive.iloc[0].copy()
        generic["archive_id"] = source_hash
        generic["dataset"] = "sec_edgar_filing"
        generic["source"] = "sec_edgar_filing"
        replay.archive = pd.concat(
            [archive, pd.DataFrame([generic])], ignore_index=True
        )

        with patch.object(
            COLLECTOR,
            "validate_source_archive_id",
            side_effect=lambda archive_id, **_kwargs: archive_id,
        ):
            assert replay(url, candidate) == payload

            conflicting = replay.archive.copy(deep=True)
            conflicting.loc[1, "source_hash"] = "0" * 64
            replay.archive = conflicting
            with pytest.raises(RuntimeError, match="ambiguous.*matches=2"):
                replay(url, candidate)


def test_sec_source_cache_hit_replays_with_zero_http_attempts() -> None:
    url = "https://www.sec.gov/Archives/edgar/data/1/cached.txt"
    content = b"immutable cached SEC filing"
    session = SimpleNamespace(headers={}, get=Mock())
    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory)
        cache_key = hashlib.sha256(f"{url}?".encode()).hexdigest()
        (cache / f"{cache_key}.bin").write_bytes(content)
        source = COLLECTOR.SecEdgarLifecycleSource(
            cache_dir=cache,
            session=session,
        )

        observed, artifact = source.fetch_url(url)

        assert observed == content
        assert artifact.source_hash == hashlib.sha256(content).hexdigest()
        assert source.http_attempts == 0
        assert source.http_attempts_by_candidate == {}
        session.get.assert_not_called()


def test_sec_source_opt_in_counts_retries_and_stops_at_global_hard_cap() -> None:
    url = "https://www.sec.gov/Archives/edgar/data/1/retry.txt"
    response = SimpleNamespace(status_code=503)
    session = SimpleNamespace(headers={}, get=Mock(return_value=response))
    candidate = FINALIZER.LifecycleCandidate(
        security_id="SEC-CAP",
        symbol="CAP",
        name="Cap Corp",
        exchange="NYSE",
        last_price_date="2025-01-02",
        active_to="2025-01-02",
    )
    with tempfile.TemporaryDirectory() as directory:
        source = COLLECTOR.SecEdgarLifecycleSource(
            cache_dir=Path(directory),
            session=session,
            user_agent="SuperTrendQuant test@example.com",
            allow_http=True,
            max_http_attempts=2,
            max_http_attempts_per_candidate=2,
            max_http_attempts_per_request=4,
            min_interval_seconds=0.1,
        )

        with patch("supertrend_quant.market_store.lifecycle.time.sleep"):
            with source.candidate_http_scope(candidate):
                with pytest.raises(RuntimeError, match="global HTTP attempt hard cap"):
                    source.fetch_url(url)

        assert source.http_attempts == 2
        assert source.http_attempts_by_candidate == {
            "SEC-CAP|2025-01-02": 2
        }
        assert session.get.call_count == 2


def test_sec_source_enforces_per_candidate_hard_cap_before_next_retry() -> None:
    url = "https://www.sec.gov/Archives/edgar/data/1/candidate-cap.txt"
    response = SimpleNamespace(status_code=503)
    session = SimpleNamespace(headers={}, get=Mock(return_value=response))
    candidate = FINALIZER.LifecycleCandidate(
        security_id="SEC-CANDIDATE-CAP",
        symbol="CCP",
        name="Candidate Cap Corp",
        exchange="NYSE",
        last_price_date="2025-01-02",
        active_to="2025-01-02",
    )
    with tempfile.TemporaryDirectory() as directory:
        source = COLLECTOR.SecEdgarLifecycleSource(
            cache_dir=Path(directory),
            session=session,
            user_agent="SuperTrendQuant test@example.com",
            allow_http=True,
            max_http_attempts=5,
            max_http_attempts_per_candidate=1,
            max_http_attempts_per_request=4,
            min_interval_seconds=0.1,
        )

        with patch("supertrend_quant.market_store.lifecycle.time.sleep"):
            with source.candidate_http_scope(candidate):
                with pytest.raises(
                    RuntimeError,
                    match="per-candidate HTTP attempt hard cap",
                ):
                    source.fetch_url(url)

        assert source.http_attempts == 1
        assert session.get.call_count == 1


def test_sec_attempt_policy_cap_and_actual_are_in_report_and_summary() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        binding, _ = _report_binding_fixture(root)
        binding = COLLECTOR.build_lifecycle_report_binding(
            release_version=binding["release_version"],
            completed_session=binding["completed_session"],
            dataset_versions=binding["input_dataset_versions"],
            candidates=(
                SimpleNamespace(
                    security_id="SEC-A", last_price_date="2025-01-02"
                ),
            ),
            hints_path=root / "hints.yaml",
            sec_fetch_policy=COLLECTOR.SEC_FETCH_POLICY_FETCH_MISSING,
            sec_max_http_attempts=7,
            sec_max_http_attempts_per_candidate=3,
            sec_http_attempts=2,
            sec_http_attempts_by_candidate={"SEC-A|2025-01-02": 2},
        )
        report = COLLECTOR._empty_report(binding)

        COLLECTOR._finalize_report(report, binding, 1)

        assert report["sec_fetch_policy"] == "fetch_missing_opt_in"
        assert report["sec_max_http_attempts"] == 7
        assert report["sec_http_attempts"] == 2
        assert report["summary"]["sec_fetch_policy"] == "fetch_missing_opt_in"
        assert report["summary"]["sec_http_attempts"] == 2
        assert report["summary"]["sec_http_attempts_remaining"] == 5


def test_sec_global_cap_is_cumulative_across_report_resume() -> None:
    url = "https://www.sec.gov/Archives/edgar/data/1/resume-cap.txt"
    session = SimpleNamespace(headers={}, get=Mock())
    candidate = FINALIZER.LifecycleCandidate(
        security_id="SEC-RESUME-CAP",
        symbol="RCP",
        name="Resume Cap Corp",
        exchange="NYSE",
        last_price_date="2025-01-02",
        active_to="2025-01-02",
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hints_path = root / "hints.yaml"
        hints_path.write_bytes(b"symbols: {}\n")
        binding = COLLECTOR.build_lifecycle_report_binding(
            release_version="release-v1",
            completed_session="2025-01-03",
            dataset_versions={"daily_price_raw": "daily-v1"},
            candidates=(candidate,),
            hints_path=hints_path,
            sec_fetch_policy=COLLECTOR.SEC_FETCH_POLICY_FETCH_MISSING,
            sec_max_http_attempts=2,
            sec_max_http_attempts_per_candidate=2,
            sec_http_attempts=2,
            sec_http_attempts_by_candidate={"SEC-RESUME-CAP|2025-01-02": 2},
        )
        report_path = root / "report.json"
        COLLECTOR._write_report(
            report_path,
            COLLECTOR._empty_report(binding),
        )
        resumed = COLLECTOR._load_or_initialize_report(
            report_path,
            resume=True,
            expected_binding=binding,
        )
        source = COLLECTOR.SecEdgarLifecycleSource(
            cache_dir=root / "cache",
            session=session,
            user_agent="SuperTrendQuant test@example.com",
            allow_http=True,
            max_http_attempts=2,
            max_http_attempts_per_candidate=2,
            initial_http_attempts=resumed["sec_http_attempts"],
            initial_http_attempts_by_candidate=resumed[
                "sec_http_attempts_by_candidate"
            ],
        )

        with source.candidate_http_scope(candidate):
            with pytest.raises(RuntimeError, match="global HTTP attempt hard cap"):
                source.fetch_url(url)

        assert source.http_attempts == 2
        session.get.assert_not_called()
