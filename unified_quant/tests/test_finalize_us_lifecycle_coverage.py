from __future__ import annotations

import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.lifecycle import LifecycleCandidate
from supertrend_quant.market_store.lifecycle_coverage import LifecycleExceptionCode
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    sha256_bytes,
)
from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "finalize_us_lifecycle_coverage.py"
)
SPEC = importlib.util.spec_from_file_location(
    "finalize_us_lifecycle_coverage",
    SCRIPT_PATH,
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Could not load {SCRIPT_PATH}")
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)

FRC_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_frc_para_lifecycle.py"
)
FRC_SPEC = importlib.util.spec_from_file_location(
    "repair_us_frc_para_lifecycle_finalizer_test",
    FRC_SCRIPT_PATH,
)
if FRC_SPEC is None or FRC_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Could not load {FRC_SCRIPT_PATH}")
frc_repair = importlib.util.module_from_spec(FRC_SPEC)
sys.modules[FRC_SPEC.name] = frc_repair
FRC_SPEC.loader.exec_module(frc_repair)

KRAFT_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_us_kraft_special_dividend.py"
)
KRAFT_SPEC = importlib.util.spec_from_file_location(
    "repair_us_kraft_special_dividend_finalizer_test",
    KRAFT_SCRIPT_PATH,
)
if KRAFT_SPEC is None or KRAFT_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Could not load {KRAFT_SCRIPT_PATH}")
kraft_repair = importlib.util.module_from_spec(KRAFT_SPEC)
sys.modules[KRAFT_SPEC.name] = kraft_repair
KRAFT_SPEC.loader.exec_module(kraft_repair)


COMPLETED = "2025-01-03"


def _frame(dataset: str, rows: list[dict] | None = None) -> pd.DataFrame:
    columns = list(dataset_spec(dataset).required_columns)
    values = []
    for row in rows or []:
        values.append({column: row.get(column, "") for column in columns})
    return pd.DataFrame(values, columns=columns)


def _source() -> dict:
    return {
        "source": "fixture",
        "retrieved_at": "2025-01-03T12:00:00Z",
        "source_hash": "a" * 64,
    }


def _master(candidate: LifecycleCandidate) -> dict:
    return {
        "security_id": candidate.security_id,
        "primary_symbol": candidate.symbol,
        "name": candidate.name,
        "exchange": candidate.exchange,
        "asset_type": "STOCK",
        "currency": "USD",
        "country": "US",
        "active_from": "2020-01-01",
        "active_to": candidate.active_to,
        **_source(),
    }


def _history(candidate: LifecycleCandidate) -> dict:
    return {
        "security_id": candidate.security_id,
        "symbol": candidate.symbol,
        "exchange": candidate.exchange,
        "effective_from": "2020-01-01",
        "effective_to": candidate.active_to,
        **_source(),
    }


def _price(security_id: str, session: str, close: float) -> dict:
    return {
        "security_id": security_id,
        "session": session,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1000,
        "currency": "USD",
        **_source(),
    }


class FakeRepository:
    def __init__(
        self,
        root: Path,
        release: DataRelease,
        frames: dict[str, pd.DataFrame],
    ):
        self.root = root
        self.release = release
        self.frames = frames
        self.release_etag = "release-etag"
        self.pointer_etags = {
            dataset: f"{dataset}-etag"
            for dataset in script.WRITE_DATASETS
        }
        self.writes: list[str] = []
        self.commits = 0

    def current_release(self):
        return self.release, self.release_etag

    def read_frame(self, dataset: str, version: str | None = None):
        return self.frames[dataset].copy()

    def current_pointer(self, dataset: str):
        return None, self.pointer_etags.get(dataset)

    def write_frame(self, *args, **kwargs):  # pragma: no cover - plan must not write
        self.writes.append(str(args[0]))
        raise AssertionError("plan attempted a dataset write")

    def commit_release(self, *args, **kwargs):  # pragma: no cover - plan must not commit
        self.commits += 1
        raise AssertionError("plan attempted a release commit")


def _candidate(
    security_id: str,
    symbol: str,
    *,
    last: str = "2025-01-02",
) -> LifecycleCandidate:
    return LifecycleCandidate(
        security_id=security_id,
        symbol=symbol,
        name=f"{symbol} Corp",
        exchange="NYSE",
        last_price_date=last,
        active_to=last,
        index_remove_dates=(),
    )


def _artifact(cache: Path, content: bytes) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"{len(list(cache.glob('*.bin')))}.bin").write_bytes(content)
    return {
        "source": "sec_edgar_filing",
        "source_url": "https://www.sec.gov/Archives/fixture.txt",
        "retrieved_at": "2025-01-03T12:00:00Z",
        "content_type": "text/plain",
        "source_hash": sha256_bytes(content),
    }


def _record(
    candidate: LifecycleCandidate,
    *,
    artifact: dict | None,
    eligible: bool = True,
    parsed: dict | None = None,
    verified_event: dict | None = None,
) -> dict:
    value = {
        "candidate": asdict(candidate),
        "filing": {"filing_date": "2025-01-03"},
        "parsed": parsed,
        "source_url": artifact["source_url"] if artifact else "",
        "source_hash": artifact["source_hash"] if artifact else "",
        "artifacts": [artifact] if artifact else [],
        "successor_security_id": "",
        "crosscheck": {
            "passed": eligible,
            "date_passed": eligible,
            "economic_terms_passed": eligible,
        },
        "eligible_for_apply": eligible,
        "manual_review_reason": "",
        "error": "" if eligible else "unresolved",
    }
    if verified_event is not None:
        value["verified_event"] = verified_event
    return value


def _report(
    root: Path,
    release: DataRelease,
    records: dict[str, dict],
    *,
    hints_path: Path = script.DEFAULT_HINTS,
) -> script.ReportDocument:
    eligible = sum(bool(value.get("eligible_for_apply")) for value in records.values())
    binding = script.build_lifecycle_report_binding(
        release_version=release.version,
        completed_session=release.completed_session,
        dataset_versions=release.dataset_versions,
        candidates=[value["candidate"] for value in records.values()],
        hints_path=hints_path,
    )
    value = {
        **binding,
        "records": records,
        "summary": {
            "candidate_count": len(records),
            "collected_count": len(records),
            "eligible_count": eligible,
            "unresolved_count": len(records) - eligible,
            "sec_fetch_policy": binding["sec_fetch_policy"],
            "sec_max_http_attempts": binding["sec_max_http_attempts"],
            "sec_max_http_attempts_per_candidate": binding[
                "sec_max_http_attempts_per_candidate"
            ],
            "sec_max_http_attempts_per_request": binding[
                "sec_max_http_attempts_per_request"
            ],
            "sec_http_attempts": binding["sec_http_attempts"],
            "sec_http_attempts_remaining": binding["sec_max_http_attempts"]
            - binding["sec_http_attempts"],
            "sec_http_attempts_by_candidate": binding[
                "sec_http_attempts_by_candidate"
            ],
        },
    }
    path = root / "report.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return script.load_report_document(path)


def _repository(
    root: Path,
    candidates: list[LifecycleCandidate],
    prices: list[dict],
    *,
    extra_master: list[LifecycleCandidate] | None = None,
) -> tuple[FakeRepository, DataRelease]:
    datasets = (
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "corporate_actions",
        "adjustment_factors",
        "source_archive",
        "index_constituent_anchors",
        "index_membership_events",
    )
    release = DataRelease(
        version="release-v1",
        created_at="2025-01-03T13:00:00Z",
        completed_session=COMPLETED,
        dataset_versions={dataset: f"{dataset}-v1" for dataset in datasets},
    )
    identities = [*candidates, *(extra_master or [])]
    frames = {
        "security_master": _frame("security_master", [_master(item) for item in identities]),
        "symbol_history": _frame("symbol_history", [_history(item) for item in identities]),
        "daily_price_raw": _frame("daily_price_raw", prices),
        "corporate_actions": _frame("corporate_actions"),
        "adjustment_factors": _frame("adjustment_factors"),
        "source_archive": _frame("source_archive"),
        "index_constituent_anchors": _frame("index_constituent_anchors"),
        "index_membership_events": _frame("index_membership_events"),
    }
    return FakeRepository(root, release, frames), release


def _cash_event(amount: float = 10.0) -> dict:
    return {
        "action_type": "cash_merger",
        "effective_date": "2025-01-03",
        "cash_amount": amount,
        "ratio": None,
        "new_symbol": "",
        "confidence": "high",
        "reason": "official fixture",
    }


def _transaction_fixture(
    root: Path,
    *,
    release_warnings: tuple[str, ...] = (),
) -> tuple[LocalDatasetRepository, script.PreparedFinalization]:
    repository = LocalDatasetRepository(root)
    versions: dict[str, str] = {}
    for dataset in (
        "daily_price_raw",
        "corporate_actions",
        "adjustment_factors",
        "source_archive",
    ):
        version = f"base-{dataset}"
        result = repository.write_frame(
            dataset,
            _frame(dataset),
            completed_session=COMPLETED,
            version=version,
        )
        versions[dataset] = result.manifest.version
    release = repository.commit_release(
        COMPLETED,
        versions,
        quality=(DataQuality.DEGRADED if release_warnings else DataQuality.VALID),
        warnings=release_warnings,
    )
    current, release_etag = repository.current_release()
    if current is None:  # pragma: no cover - fixture construction invariant
        raise AssertionError("fixture release was not committed")
    pointer_etags = {
        dataset: repository.current_pointer(dataset)[1]
        for dataset in script.WRITE_DATASETS
    }
    planned_versions = script._new_planned_versions(release)
    prepared = script.PreparedFinalization(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions=planned_versions,
        input_versions=dict(release.dataset_versions),
        frames={
            dataset: _frame(dataset)
            for dataset in script.WRITE_DATASETS
        },
        artifacts=(),
        coverage_report=SimpleNamespace(),
        evidence_report_sha256="f" * 64,
        lifecycle_metadata={"operation": "transaction-fixture"},
        warnings=(),
        summary={"status": "validated_plan"},
    )
    return repository, prepared


def _frc_exact_correction_warning() -> str:
    bundle = SimpleNamespace(
        envelope_corrections=(
            {
                "session": "2024-12-30",
                "field": "low",
                "observed": 0.0,
                "corrected": 0.003,
            },
        ),
        artifacts=(
            SimpleNamespace(
                source_hash=(
                    "3c96be232fb5f567e77a94fe315b67aa61e520d1611842620c04680fb5df6ab3"
                )
            ),
        ),
    )
    warnings = frc_repair._correction_release_warnings(bundle)
    if len(warnings) != 1:  # pragma: no cover - fixture invariant
        raise AssertionError("FRC exact correction warning changed")
    return warnings[0]


class InjectedFailure(BaseException):
    pass


class OfflineLifecycleFinalizerTest(unittest.TestCase):
    def test_finalizer_rejects_tampered_collection_provenance_before_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = _candidate("SEC-PROVENANCE", "PRV")
            repository, release = _repository(
                root,
                [candidate],
                [_price(candidate.security_id, COMPLETED, 10.0)],
            )
            document = _report(
                root,
                release,
                {
                    candidate.security_id: _record(
                        candidate,
                        artifact=None,
                        eligible=False,
                        verified_event=_cash_event(),
                    )
                },
            )
            tampered_values = {
                "report_schema": "us_lifecycle_sec_collection/tampered",
                "release_version": "release-stale",
                "candidate_set_sha256": "0" * 64,
                "hints_sha256": "1" * 64,
                "collector_version": "tampered-collector",
            }

            for field, value in tampered_values.items():
                with self.subTest(field=field):
                    report = dict(document.value)
                    report[field] = value
                    path = root / f"tampered-{field}.json"
                    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
                    tampered = script.load_report_document(path)
                    with self.assertRaisesRegex(
                        RuntimeError,
                        rf"provenance mismatch.*field={field}",
                    ):
                        script.prepare_finalization(
                            repository,
                            release,
                            repository.release_etag,
                            tampered,
                            sec_cache=root / "sec",
                            exception_mapping={},
                            candidates=[candidate],
                        )

    def test_repaired_terminal_events_are_exact_identity_and_date_bound(self):
        expected_keys = {
            script._key(
                "US:EODHD:9f13974d-7f81-5aac-a3a7-ed1d184bd76b",
                "2015-03-16",
            ),
            script._key(
                "US:EODHD:79ce1c42-8ff6-5c13-b9cf-3df82c913734",
                "2020-05-08",
            ),
            script._key(
                "US:SEC:7992d65b-3a26-5cae-b96d-bd01f695d1c1",
                "2017-08-31",
            ),
            script._key(
                "US:EODHD:5c946c06-0214-5b7b-8e7c-31f91485a215",
                "2017-12-29",
            ),
            script._key(
                "US:EODHD:24bfb026-6327-5e04-9e32-15589dcb45ba",
                "2017-12-29",
            ),
            script._key(
                "US:EODHD:de36b2d8-e15a-5d33-8493-4cc37d0c6ce0",
                "2017-07-03",
            ),
            script._key(
                "US:EODHD:5fa7bd33-c752-57c7-873c-e9d812d90e05",
                "2017-02-24",
            ),
        }
        self.assertEqual(set(script.IDENTITY_BOUND_TERMINAL_EVENTS), expected_keys)
        lila_source_hash = (
            "0efad7b02b77a0daefab021c58fdbbb40f03955f069f42eac3e24d403f2813e4"
        )
        for security_id in (
            "US:EODHD:5c946c06-0214-5b7b-8e7c-31f91485a215",
            "US:EODHD:24bfb026-6327-5e04-9e32-15589dcb45ba",
        ):
            exact = script.IDENTITY_BOUND_TERMINAL_EVENTS[
                script._key(security_id, "2017-12-29")
            ]
            self.assertEqual(exact["source_hash"], lila_source_hash)
            self.assertEqual(exact["source_content_bytes"], 29_731)
            self.assertEqual(
                exact["source_url"],
                "https://www.sec.gov/Archives/edgar/data/1570585/"
                "000157058517000401/ex991split-offrecordanddis.htm",
            )
        actavis = _candidate(
            "US:EODHD:79ce1c42-8ff6-5c13-b9cf-3df82c913734",
            "AGN",
            last="2020-05-08",
        )
        event = {
            "action_type": "stock_merger",
            "effective_date": "2020-05-08",
            "cash_amount": 120.30,
            "ratio": 0.866,
            "new_symbol": "ABBV",
        }
        script._validate_identity_bound_terminal_event(
            actavis,
            event,
            "US:EODHD:3f3cd70b-d1b0-5b4e-a702-d3ab94fc57fe",
        )
        with self.assertRaisesRegex(RuntimeError, "reviewed exact identity/date"):
            script._validate_identity_bound_terminal_event(
                replace(actavis, security_id="WRONG-AGN"),
                event,
                "US:EODHD:3f3cd70b-d1b0-5b4e-a702-d3ab94fc57fe",
            )
        with self.assertRaisesRegex(RuntimeError, "differs from its reviewed exact"):
            script._validate_identity_bound_terminal_event(
                actavis,
                {**event, "cash_amount": 129.22, "new_symbol": "ACT"},
                "US:EODHD:79ce1c42-8ff6-5c13-b9cf-3df82c913734",
            )

        spectra = _candidate(
            "US:EODHD:5fa7bd33-c752-57c7-873c-e9d812d90e05",
            "SE",
            last="2017-02-24",
        )
        spectra_event = {
            "action_type": "stock_merger",
            "effective_date": "2017-02-27",
            "cash_amount": None,
            "ratio": 0.984,
            "new_symbol": "ENB",
        }
        script._validate_identity_bound_terminal_event(
            spectra,
            spectra_event,
            "US:EODHD:8b62832f-27a7-5139-a199-62f9632c21bd",
        )
        with self.assertRaisesRegex(RuntimeError, "reviewed exact identity/date"):
            script._validate_identity_bound_terminal_event(
                replace(
                    spectra,
                    security_id="US:EODHD:cec57207-c56c-51c0-955f-204bca9b27c8",
                    name="Sea Ltd",
                    last_price_date="2026-07-15",
                ),
                spectra_event,
                "US:EODHD:8b62832f-27a7-5139-a199-62f9632c21bd",
            )

        dupont = _candidate(
            "US:SEC:7992d65b-3a26-5cae-b96d-bd01f695d1c1",
            "DD",
            last="2017-08-31",
        )
        dupont_event = {
            "action_type": "stock_merger",
            "effective_date": "2017-09-01",
            "cash_amount": None,
            "ratio": 1.282,
            "new_symbol": "DWDP",
            "source_url": (
                "https://www.sec.gov/Archives/edgar/data/30554/"
                "000119312517274840/0001193125-17-274840.txt"
            ),
            "source_hash": (
                "098828aa2714df3fdd52a18b1fffb91d6a72865ff8dd4e94e84f7bc079cf0e64"
            ),
        }
        script._validate_identity_bound_terminal_event(
            dupont,
            dupont_event,
            "US:EODHD:f174ac51-b4a4-5f29-84b0-12b8b73f0bc3",
        )
        for changed in (
            {"ratio": 1.0},
            {"source_hash": "0" * 64},
            {"source_url": "https://www.sec.gov/Archives/edgar/data/wrong.txt"},
        ):
            with self.subTest(changed=changed), self.assertRaisesRegex(
                RuntimeError, "differs from its reviewed exact"
            ):
                script._validate_identity_bound_terminal_event(
                    dupont,
                    {**dupont_event, **changed},
                    "US:EODHD:f174ac51-b4a4-5f29-84b0-12b8b73f0bc3",
                )
        # DD is reused by a current issuer.  Every future terminal DD identity
        # needs its own reviewed identity/date binding; symbol fallback is unsafe.
        self.assertIn("DD", script.IDENTITY_BOUND_TERMINAL_SYMBOLS)
        with self.assertRaisesRegex(RuntimeError, "no reviewed exact identity/date"):
            script._validate_identity_bound_terminal_event(
                replace(
                    dupont,
                    security_id="US:EODHD:CURRENT-DD",
                    name="DuPont de Nemours Inc",
                    last_price_date="2026-07-15",
                ),
                dupont_event,
                "US:EODHD:f174ac51-b4a4-5f29-84b0-12b8b73f0bc3",
            )

    def test_identity_bound_record_shape_derives_exact_source_provenance(self):
        candidate = _candidate(
            "US:SEC:7992d65b-3a26-5cae-b96d-bd01f695d1c1",
            "DD",
            last="2017-08-31",
        )
        source_url = (
            "https://www.sec.gov/Archives/edgar/data/30554/"
            "000119312517274840/0001193125-17-274840.txt"
        )
        source_hash = (
            "098828aa2714df3fdd52a18b1fffb91d6a72865ff8dd4e94e84f7bc079cf0e64"
        )
        # This mirrors collector output for an identity-bound hint: there is no
        # report-level verified_event.  The parsed terms and exact record-level
        # artifact provenance are materialized instead.
        record = {
            "parsed": {
                "action_type": "stock_merger",
                "effective_date": "2017-09-01",
                "cash_amount": None,
                "ratio": 1.282,
                "new_symbol": "DWDP",
                "confidence": "high",
            },
            "filing": {"filing_date": "2017-09-01"},
            "source_url": source_url,
            "source_hash": source_hash,
        }
        event, override = script._event_from_record(record)
        self.assertFalse(override)
        self.assertEqual(event["source_url"], source_url)
        self.assertEqual(event["source_hash"], source_hash)
        script._validate_identity_bound_terminal_event(
            candidate,
            event,
            "US:EODHD:f174ac51-b4a4-5f29-84b0-12b8b73f0bc3",
        )
        tampered, _ = script._event_from_record({**record, "source_hash": "0" * 64})
        with self.assertRaisesRegex(RuntimeError, "differs from its reviewed exact"):
            script._validate_identity_bound_terminal_event(
                candidate,
                tampered,
                "US:EODHD:f174ac51-b4a4-5f29-84b0-12b8b73f0bc3",
            )

    def test_dd_reuses_only_the_exact_existing_reviewed_action(self):
        candidate = _candidate(
            "US:SEC:7992d65b-3a26-5cae-b96d-bd01f695d1c1",
            "DD",
            last="2017-08-31",
        )
        exact = {**script.DD_EXISTING_ACTION, "retrieved_at": "2026-07-17T18:13:39Z"}
        reused = script._reuse_identity_bound_existing_action(
            candidate, pd.DataFrame([exact])
        )
        self.assertIsNotNone(reused)
        self.assertEqual(reused["event_id"], script.DD_EXISTING_ACTION["event_id"])
        for changed in (
            {"event_id": "0" * 64},
            {"ratio": 1.0},
            {"new_security_id": "WRONG-DWDP"},
            {"source_hash": "0" * 64},
            {"source": "lifecycle_finalizer"},
            {"official": "True"},
        ):
            with self.subTest(changed=changed), self.assertRaisesRegex(
                RuntimeError, "differs from the exact event/source"
            ):
                script._reuse_identity_bound_existing_action(
                    candidate, pd.DataFrame([{**exact, **changed}])
                )
        with self.assertRaisesRegex(RuntimeError, "exactly one pre-existing"):
            script._reuse_identity_bound_existing_action(
                candidate, pd.DataFrame([exact, exact])
            )

    def test_release_archive_binding_is_exact_and_byte_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"immutable reviewed archive\n"
            source_hash = sha256_bytes(content)
            source_url = "https://www.sec.gov/Archives/exact.txt"
            object_path = Path("archives/exact.txt.gz")
            destination = root / object_path
            destination.parent.mkdir(parents=True)
            destination.write_bytes(gzip.compress(content, mtime=0))
            row = {
                "archive_id": source_hash,
                "object_path": str(object_path),
                "source_url": source_url,
                "source_hash": source_hash,
                "source": "sec_edgar_filing",
            }
            repository = SimpleNamespace(root=root)
            self.assertEqual(
                script._release_archive_content(
                    repository,
                    pd.DataFrame([row]),
                    source_url=source_url,
                    source_hash=source_hash,
                    content_bytes=len(content),
                    source="sec_edgar_filing",
                ),
                content,
            )
            for archive in (
                pd.DataFrame([{**row, "source_url": source_url + "?changed=1"}]),
                pd.DataFrame([row, row]),
                pd.DataFrame([{**row, "source": "wrong"}]),
            ):
                with self.subTest(rows=len(archive)), self.assertRaisesRegex(
                    RuntimeError, "missing or ambiguous"
                ):
                    script._release_archive_content(
                        repository,
                        archive,
                        source_url=source_url,
                        source_hash=source_hash,
                        content_bytes=len(content),
                        source="sec_edgar_filing",
                    )
            destination.write_bytes(gzip.compress(b"tampered\n", mtime=0))
            with self.assertRaisesRegex(RuntimeError, "bytes differ"):
                script._release_archive_content(
                    repository,
                    pd.DataFrame([row]),
                    source_url=source_url,
                    source_hash=source_hash,
                    content_bytes=len(content),
                    source="sec_edgar_filing",
                )

    def test_cross_basis_terminal_prices_require_all_three_exact_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05")
            closes = (10.0, 11.0, 12.0, 13.0)
            timestamps = [
                int((pd.Timestamp(value, tz="UTC") + pd.Timedelta(hours=16)).timestamp())
                for value in sessions
            ]
            primary_payload = json.dumps(
                {
                    "chart": {
                        "result": [
                            {
                                "meta": {
                                    "symbol": "ZZZ",
                                    "currency": "USD",
                                    "instrumentType": "EQUITY",
                                    "exchangeName": "NMS",
                                    "exchangeTimezoneName": "America/New_York",
                                    "dataGranularity": "1d",
                                },
                                "timestamp": timestamps,
                                "indicators": {
                                    "quote": [
                                        {
                                            "open": list(closes),
                                            "high": [value + 1.0 for value in closes],
                                            "low": [value - 1.0 for value in closes],
                                            "close": list(closes),
                                            "volume": [100.0] * len(closes),
                                        }
                                    ]
                                },
                            }
                        ],
                        "error": None,
                    }
                },
                separators=(",", ":"),
            ).encode()
            external_payload = (
                "Date,Open,High,Low,Close,Volume,OpenInt\n"
                "2024-01-02,20,21,19,20,100,0\n"
                "2024-01-03,22,23,21,22,100,0\n"
                "2024-01-04,24,25,23,24,100,0\n"
            ).encode()
            successor_payload = json.dumps(
                [
                    {
                        "date": "2024-01-08",
                        "open": 20.0,
                        "high": 21.0,
                        "low": 19.0,
                        "close": 20.0,
                        "volume": 200,
                    }
                ],
                separators=(",", ":"),
            ).encode()
            values = (
                ("primary", "https://primary.example/ZZZ", primary_payload),
                ("external", "https://external.example/ZZZ", external_payload),
                ("successor", "https://successor.example/ZZZ", successor_payload),
            )
            archive_rows = []
            hashes = {}
            for source, source_url, content in values:
                source_hash = sha256_bytes(content)
                hashes[source] = source_hash
                object_path = Path("archives") / f"{source_hash}.bin.gz"
                destination = root / object_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(gzip.compress(content, mtime=0))
                archive_rows.append(
                    {
                        "archive_id": source_hash,
                        "object_path": str(object_path),
                        "source_url": source_url,
                        "source_hash": source_hash,
                        "source": source,
                    }
                )
            prices = pd.DataFrame(
                [
                    {
                        "security_id": "OLD-ZZZ",
                        "session": session,
                        "open": close,
                        "high": close + 1.0,
                        "low": close - 1.0,
                        "close": close,
                        "volume": 100.0,
                        "currency": "USD",
                        "source": "primary",
                        "source_url": "https://primary.example/ZZZ",
                        "source_hash": hashes["primary"],
                    }
                    for session, close in zip(sessions, closes)
                ]
                + [
                    {
                        "security_id": "NEW-ZZZ",
                        "session": "2024-01-08",
                        "open": 20.0,
                        "high": 21.0,
                        "low": 19.0,
                        "close": 20.0,
                        "volume": 200.0,
                        "currency": "USD",
                        "source": "successor",
                        "source_url": "",
                        "source_hash": hashes["successor"],
                    }
                ]
            )
            spec = {
                "symbol": "ZZZ",
                "active_from": sessions[0],
                "active_to": sessions[-1],
                "primary_sessions": 4,
                "primary_archive_sessions": 4,
                "primary_source": "primary",
                "primary_source_url": "https://primary.example/ZZZ",
                "primary_source_hash": hashes["primary"],
                "primary_content_bytes": len(primary_payload),
                "external_source": "external",
                "external_source_url": "https://external.example/ZZZ",
                "external_source_hash": hashes["external"],
                "external_content_bytes": len(external_payload),
                "external_raw_rows": 3,
                "overlap_start": sessions[0],
                "overlap_end": sessions[2],
                "overlap_sessions": 3,
                "uncrosschecked_tail_sessions": 1,
                "minimum_return_correlation": 0.995,
                "maximum_p99_scaled_close_error": 0.05,
                "successor_security_id": "NEW-ZZZ",
                "successor_first_session": "2024-01-08",
                "successor_sessions": 1,
                "successor_source": "successor",
                "successor_source_url": "https://successor.example/ZZZ",
                "successor_source_hash": hashes["successor"],
                "successor_content_bytes": len(successor_payload),
            }
            candidate = _candidate("OLD-ZZZ", "ZZZ", last="2024-01-05")
            repository = SimpleNamespace(root=root)
            archive = pd.DataFrame(archive_rows)
            self.assertTrue(
                script._validate_reviewed_cross_basis_terminal_prices(
                    candidate,
                    "NEW-ZZZ",
                    repository,
                    prices,
                    archive,
                    spec=spec,
                )
            )
            changed = prices.copy()
            changed.loc[
                changed["security_id"].eq("OLD-ZZZ")
                & changed["session"].eq("2024-01-03"),
                "close",
            ] = 11.25
            with self.assertRaisesRegex(RuntimeError, "differs from archived bytes"):
                script._validate_reviewed_cross_basis_terminal_prices(
                    candidate,
                    "NEW-ZZZ",
                    repository,
                    changed,
                    archive,
                    spec=spec,
                )
            with self.assertRaisesRegex(RuntimeError, "wrong identity"):
                script._validate_reviewed_cross_basis_terminal_prices(
                    candidate,
                    "WRONG-ZZZ",
                    repository,
                    prices,
                    archive,
                    spec=spec,
                )

            # Full finalizer integration: this is the actual collector state
            # that exposed the bug.  Generic raw-level economics failed, so
            # eligible_for_apply is false and no verified_event is present.
            cache = root / "sec"
            evidence = _artifact(cache, b"exact official cross-basis event")
            official_object_path = Path("archives") / f"{evidence['source_hash']}.txt.gz"
            (root / official_object_path).write_bytes(
                gzip.compress(b"exact official cross-basis event", mtime=0)
            )
            archive = archive.assign(
                dataset=archive["source"],
                content_type=archive["source"].map(
                    {
                        "primary": "application/json",
                        "external": "text/plain",
                        "successor": "application/json",
                    }
                ),
                effective_date=COMPLETED,
                retrieved_at="2025-01-03T12:00:00Z",
            )
            archive = pd.concat(
                [
                    archive,
                    pd.DataFrame(
                        [
                            {
                                "archive_id": evidence["source_hash"],
                                "dataset": evidence["source"],
                                "object_path": str(official_object_path),
                                "content_type": evidence["content_type"],
                                "effective_date": COMPLETED,
                                "source": evidence["source"],
                                "source_url": evidence["source_url"],
                                "retrieved_at": evidence["retrieved_at"],
                                "source_hash": evidence["source_hash"],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
                sort=False,
            )
            prices["retrieved_at"] = "2025-01-03T12:00:00Z"
            successor_candidate = _candidate("NEW-ZZZ", "ZZZ", last=COMPLETED)
            active_candidate = _candidate("ACTIVE-ZZZ", "ACT", last=COMPLETED)
            prices = pd.concat(
                [
                    prices,
                    pd.DataFrame(
                        [
                            {
                                **_price("ACTIVE-ZZZ", COMPLETED, 1.0),
                                "source_url": "",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
                sort=False,
            )
            repository, release = _repository(
                root,
                [candidate],
                prices.to_dict(orient="records"),
                extra_master=[successor_candidate, active_candidate],
            )
            repository.frames["daily_price_raw"] = prices.copy()
            repository.frames["source_archive"] = archive.copy()
            master = repository.frames["security_master"]
            master.loc[master["security_id"].eq("OLD-ZZZ"), "active_from"] = sessions[0]
            master.loc[master["security_id"].eq("NEW-ZZZ"), "active_from"] = "2024-01-08"
            history = repository.frames["symbol_history"]
            history.loc[history["security_id"].eq("OLD-ZZZ"), "effective_from"] = sessions[0]
            history.loc[history["security_id"].eq("OLD-ZZZ"), "effective_to"] = sessions[-1]
            history.loc[history["security_id"].eq("NEW-ZZZ"), "effective_from"] = "2024-01-08"

            parsed = {
                "action_type": "stock_merger",
                "effective_date": "2024-01-08",
                "cash_amount": None,
                "ratio": 1.0,
                "new_symbol": "ZZZ",
                "confidence": "high",
                "reason": "exact synthetic cross-basis fixture",
            }
            record = _record(
                candidate,
                artifact=evidence,
                eligible=False,
                parsed=parsed,
            )
            record["filing"] = {"filing_date": "2024-01-05"}
            record["successor_security_id"] = "NEW-ZZZ"
            record["crosscheck"] = {
                "basis": "eodhd_terminal_price",
                "passed": False,
                "date_passed": True,
                "economic_terms_passed": False,
                "relative_deviation": 0.40,
            }
            record["error"] = ""
            self.assertFalse(record["eligible_for_apply"])
            self.assertNotIn("verified_event", record)
            key = script._key(candidate.security_id, candidate.last_price_date)
            exact_event = {
                "symbol": "ZZZ",
                "action_type": "stock_merger",
                "effective_date": "2024-01-08",
                "cash_amount": None,
                "ratio": 1.0,
                "new_symbol": "ZZZ",
                "successor_security_id": "NEW-ZZZ",
                "source_url": evidence["source_url"],
                "source_hash": evidence["source_hash"],
                "source_content_bytes": len(b"exact official cross-basis event"),
            }
            document = _report(root, release, {candidate.security_id: record})
            with patch.dict(
                script.IDENTITY_BOUND_TERMINAL_EVENTS,
                {key: exact_event},
            ), patch.dict(
                script.REVIEWED_CROSS_BASIS_TERMINAL_PRICE_PROVENANCE,
                {key: spec},
            ):
                prepared = script.prepare_finalization(
                    repository,
                    release,
                    repository.release_etag,
                    document,
                    sec_cache=cache,
                    exception_mapping={},
                    candidates=[candidate],
                )
                self.assertEqual(prepared.coverage_report.applied_count, 1)
                self.assertEqual(
                    prepared.frames["lifecycle_resolutions"].iloc[0]["resolution"],
                    "applied",
                )

                tampered_records = []
                wrong_ratio = json.loads(json.dumps(record))
                wrong_ratio["parsed"]["ratio"] = 0.5
                tampered_records.append(wrong_ratio)
                wrong_source = json.loads(json.dumps(record))
                wrong_source["source_hash"] = "0" * 64
                tampered_records.append(wrong_source)
                wrong_identity = json.loads(json.dumps(record))
                wrong_identity["candidate"]["security_id"] = "WRONG-OLD-ZZZ"
                tampered_records.append(wrong_identity)
                for index, tampered in enumerate(tampered_records):
                    with self.subTest(tamper=index), self.assertRaises(RuntimeError):
                        script.prepare_finalization(
                            repository,
                            release,
                            repository.release_etag,
                            _report(
                                root,
                                release,
                                {candidate.security_id: tampered},
                            ),
                            sec_cache=cache,
                            exception_mapping={},
                            candidates=[candidate],
                        )

    def test_tfcf_exceptions_are_exact_official_and_symbol_fallback_is_blocked(self):
        specs = script.load_official_lifecycle_exception_evidence(
            script.DEFAULT_HINTS
        )
        candidate = _candidate(
            "US:EODHD:acd9ed55-bf0c-5b15-b624-1a917bf6078e",
            "TFCF",
            last="2019-03-19",
        )
        candidate = replace(candidate, name="Twenty-First Century Fox Inc Class B")
        exception = script._exception_for(candidate, {}, specs)
        self.assertIsNotNone(exception)
        self.assertEqual(
            exception.code,
            LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION,
        )
        self.assertEqual(exception.evidence_id, "tfcf_2019_disney_proration")
        self.assertIsNone(
            script._exception_for(replace(candidate, security_id="WRONG-TFCF"), {}, specs)
        )
        fallback = {
            script._symbol_key("TFCF", "2019-03-19"): script.ExceptionSpec(
                LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION,
                "unsafe symbol fallback",
            )
        }
        with self.assertRaisesRegex(RuntimeError, "fallback is forbidden"):
            script._exception_for(candidate, fallback, {})

    def test_bankruptcy_warrant_exceptions_replace_temporary_exact_mappings(self):
        registry = script.load_official_lifecycle_exception_evidence(
            script.DEFAULT_HINTS
        )
        cases = (
            (
                "US:EODHD:2826c370-0467-5e82-9617-dcece5be407f",
                "DO",
                "Diamond Offshore Drilling Inc",
                "2020-04-24",
                "legacy_do_2021_warrant_consideration",
            ),
            (
                "US:EODHD:6d9d4638-4922-5f6c-89fd-6b79db60c1c3",
                "DNR",
                "Denbury Resources Inc",
                "2020-07-28",
                "legacy_dnr_2020_warrant_consideration",
            ),
            (
                "US:EODHD:81b3ca1f-cf1b-5234-bc24-4399b8ecf149",
                "NE",
                "Noble Corporation plc",
                "2020-10-22",
                "legacy_ne_2021_warrant_consideration",
            ),
        )
        for security_id, symbol, name, last_price_date, evidence_id in cases:
            with self.subTest(symbol=symbol):
                key = script._key(security_id, last_price_date)
                exception = script.EXPLICIT_EXCEPTION_MAPPING[key]
                evidence = registry[evidence_id]
                candidate = replace(
                    _candidate(security_id, symbol, last=last_price_date),
                    name=name,
                )
                self.assertEqual(
                    exception.code,
                    LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION,
                )
                self.assertEqual(exception.recheck_after, "")
                self.assertTrue(exception.require_official_provenance)
                self.assertEqual(exception.evidence_id, evidence_id)
                self.assertEqual(exception.reason, evidence.claim)
                self.assertTrue(evidence.matches_candidate(candidate))
                self.assertFalse(
                    evidence.matches_candidate(
                        replace(candidate, security_id=f"WRONG-{symbol}")
                    )
                )
                self.assertFalse(
                    evidence.matches_candidate(
                        replace(candidate, last_price_date="2020-01-01")
                    )
                )

        self.assertNotIn(
            script._key(
                "US:EODHD:2826c370-0467-5e82-9617-dcece5be407f",
                "2020-04-29",
            ),
            script.EXPLICIT_EXCEPTION_MAPPING,
        )
        self.assertNotIn(
            script._key(
                "US:EODHD:6d9d4638-4922-5f6c-89fd-6b79db60c1c3",
                "2020-07-30",
            ),
            script.EXPLICIT_EXCEPTION_MAPPING,
        )

    def test_frc_exception_is_retired_but_esv_waits_for_same_identity_continuity(self):
        self.assertNotIn(
            script._key(
                "US:EODHD:3453b450-03d1-52ee-9100-cf80856b06ef",
                "2023-05-02",
            ),
            script.EXPLICIT_EXCEPTION_MAPPING,
        )
        esv = script.EXPLICIT_EXCEPTION_MAPPING[
            script._key(
                "US:EODHD:b0395c88-1e0d-5135-b79f-240ac991e540",
                "2019-07-30",
            )
        ]
        mnk_collision = script.EXPLICIT_EXCEPTION_MAPPING[
            script._key(
                "US:EODHD:81d711c5-9688-5f2b-9f36-63c8fe3211bf",
                "2020-10-12",
            )
        ]
        bbby_collision = script.EXPLICIT_EXCEPTION_MAPPING[
            script._key(
                "US:EODHD:dbd287da-b35f-5aaa-873c-76f941b4d93b",
                "2025-08-29",
            )
        ]

        self.assertEqual(esv.code, LifecycleExceptionCode.SUCCESSOR_UNRESOLVED)
        self.assertEqual(esv.recheck_after, script.DEFAULT_RECHECK_AFTER)
        self.assertTrue(esv.require_official_provenance)
        self.assertEqual(
            esv.source_hash,
            "596701a3f09e484f60489e5df3501c0f09e4c908905bab2f81cefb684e338fac",
        )
        self.assertEqual(
            mnk_collision.code,
            LifecycleExceptionCode.PRICE_IDENTITY_CONFLICT,
        )
        self.assertIn("unrelated Muniholdings fund", mnk_collision.reason)
        self.assertEqual(
            bbby_collision.code,
            LifecycleExceptionCode.PRICE_IDENTITY_CONFLICT,
        )
        self.assertIn("reused BBBY ticker", bbby_collision.reason)

    def test_every_permanent_mapping_uses_exact_identity_date_bound_registry(self):
        registry = script.load_official_lifecycle_exception_evidence(
            script.DEFAULT_HINTS
        )
        permanent_mapping = {
            key: value
            for key, value in script.EXPLICIT_EXCEPTION_MAPPING.items()
            if str(value.code) in script.PERMANENT_EXCEPTION_CODES
        }
        expected_evidence_ids = {
            "aaba_2019_liquidation_distributions",
            "abmd_2022_cvr_consideration",
            "brcm_2016_election_proration",
            "celg_2019_cvr_consideration",
            "ggp_2018_election_proration",
            "legacy_dnr_2020_warrant_consideration",
            "legacy_do_2021_warrant_consideration",
            "legacy_ne_2021_warrant_consideration",
            "para_2025_election_proration",
            "twc_2016_election_proration",
            "utx_2020_carr_otis_distributions",
        }
        self.assertEqual(
            {value.evidence_id for value in permanent_mapping.values()},
            expected_evidence_ids,
        )
        for key, exception in permanent_mapping.items():
            security_id, last_price_date = key.rsplit("|", 1)
            evidence = registry[exception.evidence_id]
            with self.subTest(evidence_id=exception.evidence_id):
                self.assertTrue(exception.require_official_provenance)
                self.assertFalse(exception.source_url)
                self.assertFalse(exception.source_hash)
                self.assertEqual(evidence.resolution_kind, "exception")
                self.assertEqual(evidence.exception_code, str(exception.code))
                self.assertEqual(evidence.claim, exception.reason)
                self.assertIn(security_id, evidence.candidate_security_ids)
                self.assertIn(
                    last_price_date, evidence.candidate_last_price_dates
                )
                self.assertTrue(evidence.candidate_symbols)
                self.assertTrue(evidence.candidate_name_contains)
                self.assertTrue(evidence.source_url)

        registry_exception_ids = {
            evidence.evidence_id
            for evidence in registry.values()
            if evidence.resolution_kind == "exception"
        }
        self.assertEqual(
            registry_exception_ids,
            {
                *expected_evidence_ids,
                "dvmt_2018_class_v_election_proration",
                "legacy_val_2021_warrant_consideration",
                "legacy_dnr_2020_warrant_consideration",
                "legacy_do_2021_warrant_consideration",
                "legacy_ne_2021_warrant_consideration",
                "tfcf_2019_disney_proration",
                "tfcfa_2019_disney_proration",
            },
        )

    def test_bound_val_evidence_creates_only_exact_dynamic_exception(self):
        evidence = script.load_official_lifecycle_exception_evidence(
            script.DEFAULT_HINTS
        )["legacy_val_2021_warrant_consideration"]
        candidate = LifecycleCandidate(
            security_id="US:EODHD:b0395c88-1e0d-5135-b79f-240ac991e540",
            symbol="VAL",
            name="Valaris plc",
            exchange="NYSE",
            last_price_date="2021-04-27",
            active_to="2021-04-30",
        )

        exception = script._exception_for(
            candidate,
            {},
            {evidence.evidence_id: evidence},
        )

        self.assertIsNotNone(exception)
        self.assertEqual(
            exception.code,
            LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION,
        )
        self.assertEqual(exception.evidence_id, evidence.evidence_id)
        wrong_identity = replace(candidate, security_id="REUSED-VAL")
        self.assertIsNone(
            script._exception_for(
                wrong_identity,
                {},
                {evidence.evidence_id: evidence},
            )
        )

    def test_registry_exception_requires_reviewer_pin_and_exact_cached_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            evidence = script.load_official_lifecycle_exception_evidence(
                script.DEFAULT_HINTS
            )["legacy_val_2021_warrant_consideration"]
            evidence = replace(evidence, source_sha256="")
            artifact = _artifact(cache, b"reviewed legacy VAL warrant evidence")
            artifact["source_url"] = evidence.source_url
            record = {
                "candidate": {
                    "security_id": "US:EODHD:b0395c88-1e0d-5135-b79f-240ac991e540",
                    "symbol": "VAL",
                    "name": "Valaris plc",
                    "last_price_date": "2021-04-27",
                },
                "artifacts": [artifact],
            }
            exception = script.ExceptionSpec(
                code=evidence.exception_code,
                reason=evidence.claim,
                require_official_provenance=True,
                evidence_id=evidence.evidence_id,
            )

            with self.assertRaisesRegex(RuntimeError, "not reviewer-pinned"):
                script._artifact_from_exception(
                    exception,
                    record,
                    script._ArtifactCache(cache),
                    {evidence.evidence_id: evidence},
                )

            pinned = replace(evidence, source_sha256=artifact["source_hash"])
            accepted = script._artifact_from_exception(
                exception,
                record,
                script._ArtifactCache(cache),
                {pinned.evidence_id: pinned},
            )
            self.assertIsNotNone(accepted)
            self.assertEqual(accepted.source_hash, artifact["source_hash"])
            self.assertEqual(accepted.source_url, evidence.source_url)

    def test_esv_to_legacy_val_can_preserve_the_same_security_id(self):
        candidate = _candidate(
            "US:EODHD:b0395c88-1e0d-5135-b79f-240ac991e540",
            "ESV",
            last="2019-07-30",
        )
        master = pd.DataFrame(
            [
                {
                    **_master(candidate),
                    "primary_symbol": "VAL",
                    "provider_symbol": "VALPQ.US",
                    "active_to": "2021-04-30",
                }
            ]
        )
        history = pd.DataFrame(
            [
                {
                    **_history(candidate),
                    "effective_to": "2019-07-30",
                },
                {
                    **_history(candidate),
                    "symbol": "VAL",
                    "effective_from": "2019-07-31",
                    "effective_to": "2021-04-30",
                },
            ]
        )
        event = {
            "action_type": "ticker_change",
            "effective_date": "2019-07-31",
            "new_symbol": "VAL",
        }

        successor = script._successor_for_event(event, master, history)
        script._crosscheck_event(
            candidate,
            event,
            successor,
            script._price_histories(
                pd.DataFrame(
                    [
                        _price(candidate.security_id, "2019-07-30", 8.27),
                        _price(candidate.security_id, "2019-07-31", 8.25),
                    ]
                )
            ),
        )

        self.assertEqual(successor, candidate.security_id)

    def test_complete_closure_has_exact_applied_or_explicit_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "sec"
            applied = _candidate("SEC-A", "AAA")
            exception = _candidate("SEC-B", "BBB")
            evidence = _artifact(cache, b"official applied evidence")
            repository, release = _repository(
                root,
                [applied, exception],
                [
                    _price(applied.security_id, "2025-01-02", 10.0),
                    _price(exception.security_id, COMPLETED, 5.0),
                ],
            )
            records = {
                applied.security_id: _record(
                    applied,
                    artifact=evidence,
                    parsed=_cash_event(),
                ),
                exception.security_id: _record(
                    exception,
                    artifact=None,
                    eligible=False,
                ),
            }
            document = _report(root, release, records)
            mapping = {
                script._key(exception.security_id, exception.last_price_date): script.ExceptionSpec(
                    LifecycleExceptionCode.NOT_LIFECYCLE_EVENT,
                    "Reviewed fixture is not a lifecycle event.",
                )
            }

            prepared = script.prepare_finalization(
                repository,
                release,
                repository.release_etag,
                document,
                sec_cache=cache,
                exception_mapping=mapping,
                candidates=[applied, exception],
            )

            self.assertTrue(prepared.coverage_report.valid)
            self.assertEqual(prepared.coverage_report.applied_count, 1)
            self.assertEqual(prepared.coverage_report.exception_count, 1)
            self.assertEqual(prepared.coverage_report.open_count, 0)
            self.assertEqual(len(prepared.frames["lifecycle_resolutions"]), 2)

    def test_exception_resolution_retains_and_archives_exact_official_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "sec"
            candidate = _candidate("SEC-E", "EXC")
            evidence = _artifact(cache, b"official exception evidence")
            repository, release = _repository(
                root,
                [candidate],
                [_price(candidate.security_id, COMPLETED, 5.0)],
            )
            document = _report(
                root,
                release,
                {
                    candidate.security_id: _record(
                        candidate,
                        artifact=evidence,
                        eligible=False,
                    )
                },
            )
            mapping = {
                script._key(candidate.security_id, candidate.last_price_date): script.ExceptionSpec(
                    LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION,
                    "Reviewed fixture has nonstandard consideration.",
                    require_official_provenance=True,
                    evidence_id="fixture_permanent_exception",
                )
            }
            template = script.load_official_lifecycle_exception_evidence(
                script.DEFAULT_HINTS
            )["aaba_2019_liquidation_distributions"]
            official_spec = replace(
                template,
                evidence_id="fixture_permanent_exception",
                candidate_symbols=(candidate.symbol,),
                candidate_name_contains=(candidate.name,),
                candidate_security_ids=(candidate.security_id,),
                candidate_last_price_dates=(candidate.last_price_date,),
                effective_date=candidate.last_price_date,
                resolution_kind="exception",
                exception_code=str(LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION),
                claim="Reviewed fixture has nonstandard consideration.",
                source_url=evidence["source_url"],
                source_sha256=evidence["source_hash"],
            )

            prepared = script.prepare_finalization(
                repository,
                release,
                repository.release_etag,
                document,
                sec_cache=cache,
                exception_mapping=mapping,
                official_evidence_specs={official_spec.evidence_id: official_spec},
                candidates=[candidate],
            )

            resolution = prepared.frames["lifecycle_resolutions"].iloc[0]
            self.assertEqual(resolution["source_url"], evidence["source_url"])
            self.assertEqual(resolution["source_hash"], evidence["source_hash"])
            self.assertIn(
                evidence["source_hash"],
                set(prepared.frames["source_archive"]["archive_id"]),
            )

    def test_permanent_exception_fails_without_required_official_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = _candidate("SEC-F", "FRC")
            repository, release = _repository(
                root,
                [candidate],
                [_price(candidate.security_id, COMPLETED, 5.0)],
            )
            document = _report(
                root,
                release,
                {
                    candidate.security_id: _record(
                        candidate,
                        artifact=None,
                        eligible=False,
                    )
                },
            )
            mapping = {
                script._key(candidate.security_id, candidate.last_price_date): script.ExceptionSpec(
                    LifecycleExceptionCode.RECOVERY_UNCERTAIN,
                    "The receivership has no final per-share recovery.",
                    require_official_provenance=True,
                )
            }

            with self.assertRaisesRegex(
                RuntimeError,
                "exact identity/date-bound official evidence registry entry",
            ):
                script.prepare_finalization(
                    repository,
                    release,
                    repository.release_etag,
                    document,
                    sec_cache=root / "sec",
                    exception_mapping=mapping,
                    candidates=[candidate],
                )

    def test_unclassified_candidate_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = _candidate("SEC-U", "UNK")
            repository, release = _repository(
                root,
                [candidate],
                [_price(candidate.security_id, COMPLETED, 5.0)],
            )
            document = _report(
                root,
                release,
                {
                    candidate.security_id: _record(
                        candidate,
                        artifact=None,
                        eligible=False,
                    )
                },
            )

            with self.assertRaisesRegex(RuntimeError, "Unclassified lifecycle candidate"):
                script.prepare_finalization(
                    repository,
                    release,
                    repository.release_etag,
                    document,
                    sec_cache=root / "sec",
                    exception_mapping={},
                    candidates=[candidate],
                )

    def test_verified_event_override_is_recomputed_and_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "sec"
            candidate = _candidate("SEC-V", "VRF")
            evidence = _artifact(cache, b"verified override evidence")
            event = {
                **_cash_event(),
                "source_url": evidence["source_url"],
                "source_hash": evidence["source_hash"],
                "retrieved_at": evidence["retrieved_at"],
                "content_type": evidence["content_type"],
                "filing_date": "2025-01-03",
            }
            repository, release = _repository(
                root,
                [candidate],
                [
                    _price(candidate.security_id, "2025-01-02", 10.0),
                    _price("ACTIVE", COMPLETED, 1.0),
                ],
                extra_master=[_candidate("ACTIVE", "ACT", last=COMPLETED)],
            )
            document = _report(
                root,
                release,
                {
                    candidate.security_id: _record(
                        candidate,
                        artifact=None,
                        eligible=False,
                        verified_event=event,
                    )
                },
            )

            prepared = script.prepare_finalization(
                repository,
                release,
                repository.release_etag,
                document,
                sec_cache=cache,
                exception_mapping={},
                candidates=[candidate],
            )

            self.assertEqual(prepared.coverage_report.applied_count, 1)
            self.assertEqual(
                prepared.frames["lifecycle_resolutions"].iloc[0]["resolution"],
                "applied",
            )

    def test_verified_override_requires_cached_official_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = _candidate("SEC-M", "MISS")
            missing = {
                **_cash_event(),
                "source_url": "https://www.sec.gov/Archives/missing.txt",
                "source_hash": "b" * 64,
                "filing_date": "2025-01-03",
            }
            repository, release = _repository(
                root,
                [candidate],
                [
                    _price(candidate.security_id, "2025-01-02", 10.0),
                    _price("ACTIVE", COMPLETED, 1.0),
                ],
                extra_master=[_candidate("ACTIVE", "ACT", last=COMPLETED)],
            )
            document = _report(
                root,
                release,
                {
                    candidate.security_id: _record(
                        candidate,
                        artifact=None,
                        eligible=False,
                        verified_event=missing,
                    )
                },
            )

            with self.assertRaisesRegex(FileNotFoundError, "Cached official SEC artifact"):
                script.prepare_finalization(
                    repository,
                    release,
                    repository.release_etag,
                    document,
                    sec_cache=root / "empty-sec-cache",
                    exception_mapping={},
                    candidates=[candidate],
                )

    def test_report_and_sec_payload_are_archived_and_metadata_matches_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "sec"
            candidate = _candidate("SEC-H", "HASH")
            evidence = _artifact(cache, b"archive this exact SEC payload")
            repository, release = _repository(
                root,
                [candidate],
                [
                    _price(candidate.security_id, "2025-01-02", 10.0),
                    _price("ACTIVE", COMPLETED, 1.0),
                ],
                extra_master=[_candidate("ACTIVE", "ACT", last=COMPLETED)],
            )
            document = _report(
                root,
                release,
                {
                    candidate.security_id: _record(
                        candidate,
                        artifact=evidence,
                        parsed=_cash_event(),
                    )
                },
            )

            prepared = script.prepare_finalization(
                repository,
                release,
                repository.release_etag,
                document,
                sec_cache=cache,
                exception_mapping={},
                candidates=[candidate],
            )

            archive_ids = set(prepared.frames["source_archive"]["archive_id"])
            self.assertEqual(prepared.evidence_report_sha256, sha256_bytes(document.content))
            self.assertIn(document.sha256, archive_ids)
            self.assertIn(evidence["source_hash"], archive_ids)
            archive = prepared.frames["source_archive"].set_index("archive_id")
            sec_row = archive.loc[evidence["source_hash"]]
            report_row = archive.loc[document.sha256]
            self.assertEqual(sec_row["source_hash"], evidence["source_hash"])
            self.assertEqual(sec_row["source_url"], evidence["source_url"])
            self.assertTrue(str(sec_row["object_path"]).endswith(".txt.gz"))
            self.assertEqual(report_row["source_hash"], document.sha256)
            self.assertEqual(report_row["source_url"], f"file://{document.path}")
            self.assertTrue(str(report_row["object_path"]).endswith(".json.gz"))
            self.assertEqual(
                prepared.lifecycle_metadata["evidence_report_sha256"],
                document.sha256,
            )
            self.assertEqual(
                prepared.lifecycle_metadata["input_versions"],
                release.dataset_versions,
            )
            for key, value in prepared.coverage_report.manifest_metadata().items():
                self.assertEqual(prepared.lifecycle_metadata[key], value)

    def test_temporary_exception_requires_future_recheck(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = _candidate("SEC-T", "TEMP")
            repository, release = _repository(
                root,
                [candidate],
                [_price(candidate.security_id, COMPLETED, 5.0)],
            )
            document = _report(
                root,
                release,
                {
                    candidate.security_id: _record(
                        candidate,
                        artifact=None,
                        eligible=False,
                    )
                },
            )
            key = script._key(candidate.security_id, candidate.last_price_date)
            expired = {
                key: script.ExceptionSpec(
                    LifecycleExceptionCode.SUCCESSOR_UNRESOLVED,
                    "temporary fixture",
                    recheck_after=COMPLETED,
                )
            }
            valid = {
                key: script.ExceptionSpec(
                    LifecycleExceptionCode.SUCCESSOR_UNRESOLVED,
                    "temporary fixture",
                    recheck_after="2025-02-03",
                )
            }

            with self.assertRaisesRegex(ValueError, "recheck"):
                script.prepare_finalization(
                    repository,
                    release,
                    repository.release_etag,
                    document,
                    sec_cache=root / "sec",
                    exception_mapping=expired,
                    candidates=[candidate],
                )
            prepared = script.prepare_finalization(
                repository,
                release,
                repository.release_etag,
                document,
                sec_cache=root / "sec",
                exception_mapping=valid,
                candidates=[candidate],
            )
            self.assertEqual(prepared.coverage_report.exception_count, 1)

    def test_jwn_dividends_precede_verified_cash_merger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "sec"
            candidate = _candidate(
                script.JWN_SECURITY_ID,
                "JWN",
                last="2025-05-20",
            )
            evidence = _artifact(cache, b"Nordstrom 24.25 plus both dividends")
            event = {
                **_cash_event(24.25),
                "effective_date": "2025-05-20",
            }
            release_completed = "2025-05-20"
            repository, release = _repository(
                root,
                [candidate],
                [
                    _price(candidate.security_id, "2025-05-16", 24.65),
                    _price(candidate.security_id, "2025-05-19", 24.25),
                    _price(candidate.security_id, release_completed, 24.25),
                ],
            )
            release = DataRelease(
                version=release.version,
                created_at=release.created_at,
                completed_session=release_completed,
                dataset_versions=release.dataset_versions,
            )
            repository.release = release
            document = _report(
                root,
                release,
                {
                    candidate.security_id: _record(
                        candidate,
                        artifact=evidence,
                        parsed=event,
                    )
                },
            )

            prepared = script.prepare_finalization(
                repository,
                release,
                repository.release_etag,
                document,
                sec_cache=cache,
                exception_mapping={},
                candidates=[candidate],
            )

            own = prepared.frames["corporate_actions"].loc[
                prepared.frames["corporate_actions"]["security_id"].eq(candidate.security_id)
            ]
            dividends = own.loc[own["action_type"].eq("special_dividend")]
            merger = own.loc[own["action_type"].eq("cash_merger")].iloc[0]
            self.assertEqual(set(dividends["cash_amount"].astype(float)), {0.25, 0.1462})
            self.assertEqual(set(dividends["ex_date"]), {"2025-05-19"})
            self.assertEqual(set(dividends["record_date"]), {"2025-05-19"})
            self.assertEqual(set(dividends["payment_date"]), {"2025-05-27"})
            self.assertEqual(len(set(dividends["event_id"])), 2)
            self.assertLess(max(dividends["effective_date"]), merger["effective_date"])
            self.assertEqual(float(merger["cash_amount"]), 24.25)
            factors = prepared.frames["adjustment_factors"]
            before_ex_date = factors.loc[
                factors["security_id"].eq(candidate.security_id)
                & pd.to_datetime(factors["session"]).eq(pd.Timestamp("2025-05-16"))
            ].iloc[0]
            expected_total_return_factor = (
                (24.65 - 0.25) / 24.65
            ) * ((24.65 - 0.1462) / 24.65)
            self.assertAlmostEqual(
                float(before_ex_date["total_return_factor"]),
                expected_total_return_factor,
            )
            expected_source_version = script._adjustment_source_version(
                release.dataset_versions["daily_price_raw"],
                prepared.planned_versions["corporate_actions"],
            )
            self.assertEqual(
                set(factors["source_version"].astype(str)),
                {expected_source_version},
            )
            self.assertEqual(
                prepared.lifecycle_metadata["adjustment_source_version"],
                expected_source_version,
            )

    def test_swn_applies_only_with_cached_chk_to_exe_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "sec"
            swn = _candidate("SWN-ID", "SWN", last="2024-09-30")
            chk = _candidate("CHK-ID", "CHK", last="2024-10-01")
            exe = _candidate("EXE-ID", "EXE", last=COMPLETED)
            source = _artifact(cache, b"SWN to CHK official evidence")
            chain = _artifact(cache, b"CHK to EXE official evidence")
            event = {
                "action_type": "stock_merger",
                "effective_date": "2024-10-01",
                "cash_amount": None,
                "ratio": 0.1,
                "new_symbol": "CHK",
                "confidence": "high",
                "reason": "official fixture",
            }
            repository, release = _repository(
                root,
                [swn],
                [
                    _price(swn.security_id, "2024-09-30", 8.0),
                    _price(chk.security_id, "2024-10-01", 80.0),
                    _price(exe.security_id, "2024-10-02", 80.0),
                    _price(exe.security_id, COMPLETED, 82.0),
                ],
                extra_master=[chk, exe],
            )
            document = _report(
                root,
                release,
                {
                    swn.security_id: _record(
                        swn,
                        artifact=source,
                        parsed=event,
                    )
                },
            )
            chain_event = {
                **script.CHK_EXE_EVIDENCE,
                "source_url": chain["source_url"],
                "source_hash": chain["source_hash"],
                "retrieved_at": chain["retrieved_at"],
                "content_type": chain["content_type"],
            }

            with patch.object(script, "CHK_EXE_EVIDENCE", chain_event):
                prepared = script.prepare_finalization(
                    repository,
                    release,
                    repository.release_etag,
                    document,
                    sec_cache=cache,
                    exception_mapping={},
                    candidates=[swn],
                )

            actions = prepared.frames["corporate_actions"]
            chain_actions = actions.loc[
                actions["security_id"].eq(chk.security_id)
                & actions["action_type"].eq("ticker_change")
            ]
            self.assertEqual(len(chain_actions), 1)
            self.assertEqual(chain_actions.iloc[0]["effective_date"], "2024-10-02")
            self.assertEqual(chain_actions.iloc[0]["new_security_id"], exe.security_id)
            self.assertEqual(chain_actions.iloc[0]["new_symbol"], "EXE")
            self.assertEqual(prepared.coverage_report.applied_count, 1)
            self.assertEqual(prepared.summary["actions"]["chk_exe_action_count"], 1)

    def test_finalizer_preserves_kraft_action_archive_and_rebuilds_factor_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "sec"
            krft = LifecycleCandidate(
                security_id=kraft_repair.KRFT_ID,
                symbol=kraft_repair.KRFT_SYMBOL,
                name="Kraft Foods Group Inc",
                exchange="NASDAQ",
                last_price_date=kraft_repair.EFFECTIVE_DATE,
                active_to=kraft_repair.EFFECTIVE_DATE,
                index_remove_dates=(),
            )
            khc = LifecycleCandidate(
                security_id=kraft_repair.KHC_ID,
                symbol=kraft_repair.KHC_SYMBOL,
                name="The Kraft Heinz Company",
                exchange="NASDAQ",
                last_price_date=COMPLETED,
                active_to=COMPLETED,
                index_remove_dates=(),
            )
            prices = [
                _price(krft.security_id, "2015-06-30", 85.0),
                _price(krft.security_id, "2015-07-01", 88.30),
                _price(krft.security_id, kraft_repair.EFFECTIVE_DATE, 88.19),
                _price(khc.security_id, "2015-07-06", 72.50),
                _price(khc.security_id, COMPLETED, 75.00),
            ]
            repository, release = _repository(
                root,
                [krft],
                prices,
                extra_master=[khc],
            )
            master = repository.frames["security_master"]
            master.loc[master["security_id"].eq(krft.security_id), "active_from"] = (
                "2015-01-02"
            )
            master.loc[master["security_id"].eq(khc.security_id), "active_from"] = (
                kraft_repair.EFFECTIVE_DATE
            )
            history = repository.frames["symbol_history"]
            history.loc[
                history["security_id"].eq(krft.security_id), "effective_from"
            ] = "2015-01-01"
            history.loc[
                history["security_id"].eq(khc.security_id), "effective_from"
            ] = kraft_repair.EFFECTIVE_DATE

            action_columns = list(dataset_spec("corporate_actions").required_columns)
            if "metadata" not in action_columns:
                action_columns.append("metadata")
            special = kraft_repair._expected_action(pd.Index(action_columns))
            merger = {column: "" for column in action_columns}
            merger.update(
                {
                    "event_id": kraft_repair.STOCK_MERGER_EVENT_ID,
                    "security_id": kraft_repair.KRFT_ID,
                    "action_type": "stock_merger",
                    "effective_date": kraft_repair.EFFECTIVE_DATE,
                    "ex_date": kraft_repair.EFFECTIVE_DATE,
                    "announcement_date": kraft_repair.EFFECTIVE_DATE,
                    "cash_amount": None,
                    "ratio": 1.0,
                    "currency": "USD",
                    "new_security_id": kraft_repair.KHC_ID,
                    "new_symbol": kraft_repair.KHC_SYMBOL,
                    "official": True,
                    "source_url": kraft_repair.COMPLETION_EVIDENCE.source_url,
                    "source_kind": "sec_filing",
                    "source": "sec_edgar_filing",
                    "retrieved_at": kraft_repair.COMPLETION_EVIDENCE.retrieved_at,
                    "source_hash": kraft_repair.COMPLETION_EVIDENCE.source_hash,
                    "metadata": "{}",
                }
            )
            existing_actions = pd.DataFrame(
                [special, merger], columns=action_columns
            )
            repository.frames["corporate_actions"] = existing_actions
            repository.frames["adjustment_factors"] = build_adjustment_factors(
                repository.frames["daily_price_raw"],
                existing_actions,
                source_version="kraft-input-lineage",
            )

            archive_columns = list(dataset_spec("source_archive").required_columns)
            if "source_url" not in archive_columns:
                archive_columns.append("source_url")
            archive_rows = []
            for spec in kraft_repair.EVIDENCE_SPECS:
                row = {column: "" for column in archive_columns}
                row.update(
                    {
                        "archive_id": spec.source_hash,
                        "dataset": "sec_edgar_filing",
                        "object_path": spec.archive_object_path,
                        "content_type": spec.content_type,
                        "effective_date": release.completed_session,
                        "source": "sec_edgar_filing",
                        "retrieved_at": spec.retrieved_at,
                        "source_hash": spec.source_hash,
                        "source_url": spec.source_url,
                    }
                )
                archive_rows.append(row)
            repository.frames["source_archive"] = pd.DataFrame(
                archive_rows, columns=archive_columns
            )

            report_artifact = _artifact(cache, b"official Kraft merger evidence")
            record = _record(
                krft,
                artifact=report_artifact,
                parsed={
                    "action_type": "stock_merger",
                    "effective_date": kraft_repair.EFFECTIVE_DATE,
                    "cash_amount": None,
                    "ratio": 1.0,
                    "new_symbol": kraft_repair.KHC_SYMBOL,
                    "confidence": "high",
                    "reason": "official one-for-one Kraft merger fixture",
                },
            )
            record["filing"] = {"filing_date": kraft_repair.EFFECTIVE_DATE}
            record["successor_security_id"] = kraft_repair.KHC_ID
            document = _report(root, release, {krft.security_id: record})

            prepared = script.prepare_finalization(
                repository,
                release,
                repository.release_etag,
                document,
                sec_cache=cache,
                exception_mapping={},
                candidates=[krft],
            )

            special_rows = prepared.frames["corporate_actions"].loc[
                prepared.frames["corporate_actions"]["event_id"]
                .astype(str)
                .eq(kraft_repair.SPECIAL_DIVIDEND_EVENT_ID)
            ]
            self.assertEqual(len(special_rows), 1)
            self.assertTrue(kraft_repair._action_is_exact(special_rows.iloc[0]))
            archived_hashes = set(
                prepared.frames["source_archive"]["source_hash"].astype(str)
            )
            self.assertTrue(
                {spec.source_hash for spec in kraft_repair.EVIDENCE_SPECS}.issubset(
                    archived_hashes
                )
            )
            lineage = prepared.summary["adjustment_source_version"]
            rebuilt = prepared.frames["adjustment_factors"]
            self.assertEqual(set(rebuilt["source_version"].astype(str)), {lineage})
            self.assertEqual(set(rebuilt["source_hash"].astype(str)), {lineage})
            krft_factors = rebuilt.loc[
                rebuilt["security_id"].astype(str).eq(kraft_repair.KRFT_ID)
            ]
            self.assertTrue(
                pd.to_numeric(krft_factors["total_return_factor"])
                .ne(1.0)
                .any()
            )

    def test_finalizer_rejects_a_partial_synthetic_frc_repair_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_hash = (
                "3c96be232fb5f567e77a94fe315b67aa61e520d1611842620c04680fb5df6ab3"
            )
            correction_hash = "5" * 64
            source = {
                "source": "fixture",
                "retrieved_at": "2026-07-18T09:25:56Z",
                "source_hash": "a" * 64,
            }
            master = _frame(
                "security_master",
                [
                    {
                        "security_id": frc_repair.FRC_SECURITY_ID,
                        "primary_symbol": frc_repair.FRC_NEW_SYMBOL,
                        "name": "First Republic Bank",
                        "exchange": "PINK",
                        "asset_type": "STOCK",
                        "currency": "USD",
                        "country": "US",
                        "active_from": "2015-01-02",
                        "active_to": "",
                        **source,
                    }
                ],
            )
            history = _frame(
                "symbol_history",
                [
                    {
                        "security_id": frc_repair.FRC_SECURITY_ID,
                        "symbol": frc_repair.FRC_OLD_SYMBOL,
                        "exchange": "NYSE",
                        "effective_from": "2015-01-01",
                        "effective_to": frc_repair.FRC_OLD_LAST,
                        **source,
                    },
                    {
                        "security_id": frc_repair.FRC_SECURITY_ID,
                        "symbol": frc_repair.FRC_NEW_SYMBOL,
                        "exchange": "PINK",
                        "effective_from": frc_repair.FRC_TRANSITION,
                        "effective_to": "",
                        **source,
                    },
                ],
            )
            prices = _frame(
                "daily_price_raw",
                [
                    _price(frc_repair.FRC_SECURITY_ID, frc_repair.FRC_TRANSITION, 0.33),
                    _price(frc_repair.FRC_SECURITY_ID, frc_repair.FRC_INDEX_EXIT, 0.30),
                    {
                        **_price(frc_repair.FRC_SECURITY_ID, "2024-12-30", 0.004),
                        "open": 0.003,
                        "high": 0.006,
                        "low": 0.003,
                        "source": "eodhd_eod",
                        "source_hash": raw_hash,
                    },
                    _price(frc_repair.FRC_SECURITY_ID, COMPLETED, 0.003),
                ],
            )
            action = {
                "event_id": frc_repair.FRC_EVENT_ID,
                "security_id": frc_repair.FRC_SECURITY_ID,
                "action_type": "ticker_change",
                "effective_date": frc_repair.FRC_TRANSITION,
                "ex_date": frc_repair.FRC_TRANSITION,
                "announcement_date": "2023-05-02",
                "record_date": "",
                "payment_date": "",
                "cash_amount": None,
                "ratio": None,
                "currency": "USD",
                "new_security_id": frc_repair.FRC_SECURITY_ID,
                "new_symbol": frc_repair.FRC_NEW_SYMBOL,
                "official": True,
                "source_url": frc_repair.OCC_MEMO_URL,
                "source_kind": "clearing_notice_reviewed_extraction",
                "source": "occ_reviewed_memo_extraction",
                "retrieved_at": "2026-07-18T00:00:00Z",
                "source_hash": frc_repair._occ_artifact().source_hash,
            }
            actions = _frame("corporate_actions", [action])
            archives = _frame(
                "source_archive",
                [
                    {
                        "archive_id": raw_hash,
                        "dataset": "daily_price_raw",
                        "object_path": f"archives/{raw_hash}.json.gz",
                        "content_type": "application/json",
                        "effective_date": "2024-12-30",
                        "source": "eodhd_eod",
                        "retrieved_at": "2026-07-18T09:25:56Z",
                        "source_hash": raw_hash,
                    },
                    {
                        "archive_id": correction_hash,
                        "dataset": "daily_price_raw",
                        "object_path": f"archives/{correction_hash}.json.gz",
                        "content_type": "application/json",
                        "effective_date": "2024-12-30",
                        "source": "frcb_reviewed_ohlcv_envelope_correction",
                        "retrieved_at": "2026-07-18T09:25:56Z",
                        "source_hash": correction_hash,
                    },
                ],
            )
            datasets = (
                "security_master",
                "symbol_history",
                "daily_price_raw",
                "corporate_actions",
                "adjustment_factors",
                "source_archive",
                "index_constituent_anchors",
                "index_membership_events",
            )
            warning = _frc_exact_correction_warning()
            release = DataRelease(
                version="frc-repaired-release",
                created_at="2026-07-18T09:25:56Z",
                completed_session=COMPLETED,
                dataset_versions={dataset: f"{dataset}-frc" for dataset in datasets},
                quality=DataQuality.DEGRADED,
                warnings=(warning,),
            )
            frames = {
                "security_master": master,
                "symbol_history": history,
                "daily_price_raw": prices,
                "corporate_actions": actions,
                "adjustment_factors": build_adjustment_factors(
                    prices, actions, source_version="frc-repair"
                ),
                "source_archive": archives,
                "index_constituent_anchors": _frame("index_constituent_anchors"),
                "index_membership_events": _frame("index_membership_events"),
            }
            repository = FakeRepository(root, release, frames)
            document = _report(root, release, {})

            # The exact FRC marker is present, but the fixture intentionally
            # omits PARA/PSKY, OCC/FDIC, the correction metadata hash, and the
            # input resolution dataset.  A partial repaired state must never
            # be silently accepted merely because the ticker-change row looks
            # plausible.
            with self.assertRaisesRegex(
                RuntimeError,
                "Exact FRC/PARA repair is partial",
            ):
                script.prepare_finalization(
                    repository,
                    release,
                    repository.release_etag,
                    document,
                    sec_cache=root / "sec",
                    exception_mapping={},
                    candidates=[],
                )

    def test_apply_preserves_existing_frc_degraded_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            warning = _frc_exact_correction_warning()
            repository, prepared = _transaction_fixture(
                Path(directory), release_warnings=(warning,)
            )
            valid_snapshot = SimpleNamespace(raise_for_errors=lambda: None)
            with patch.object(
                script,
                "validate_operational_repository_snapshot",
                return_value=valid_snapshot,
            ):
                script.apply_finalization(repository, prepared)

            release, _ = repository.current_release()
            self.assertIsNotNone(release)
            assert release is not None
            self.assertEqual(release.quality, DataQuality.DEGRADED)
            self.assertIn(warning, release.warnings)

    def test_apply_rolls_back_release_and_pointers_at_every_failure_stage(self):
        stages = (
            "after_archive_payloads",
            "after_corporate_actions",
            "after_adjustment_factors",
            "after_source_archive",
            "after_lifecycle_resolutions",
            "after_release_commit",
        )
        for failure_stage in stages:
            with self.subTest(failure_stage=failure_stage):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    repository, prepared = _transaction_fixture(root)
                    old_release = repository.objects.get("releases/current.json").data
                    old_pointers = {
                        dataset: repository.objects.get(repository.current_key(dataset)).data
                        for dataset in (
                            "corporate_actions",
                            "adjustment_factors",
                            "source_archive",
                        )
                    }

                    def inject(stage: str) -> None:
                        if stage == failure_stage:
                            raise InjectedFailure(stage)

                    valid_snapshot = SimpleNamespace(raise_for_errors=lambda: None)
                    with patch.object(
                        script,
                        "validate_operational_repository_snapshot",
                        return_value=valid_snapshot,
                    ):
                        with self.assertRaises(InjectedFailure):
                            script.apply_finalization(
                                repository,
                                prepared,
                                failure_injector=inject,
                            )

                    self.assertEqual(
                        repository.objects.get("releases/current.json").data,
                        old_release,
                    )
                    for dataset, old_pointer in old_pointers.items():
                        self.assertEqual(
                            repository.objects.get(repository.current_key(dataset)).data,
                            old_pointer,
                        )
                    self.assertIsNone(
                        repository.current_pointer("lifecycle_resolutions")[0]
                    )
                    journals = list(
                        (root / "transactions/lifecycle-finalizer").glob("*.json")
                    )
                    self.assertEqual(len(journals), 1)
                    journal = json.loads(journals[0].read_bytes())
                    self.assertEqual(journal["status"], "rolled_back")
                    self.assertEqual(
                        set(journal["planned_versions"]),
                        set(script.WRITE_DATASETS),
                    )
                    self.assertFalse(
                        (root / "recovery/lifecycle-finalizer").exists()
                    )

    def test_apply_release_keeps_action_and_factor_source_versions_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, prepared = _transaction_fixture(Path(directory))
            valid_snapshot = SimpleNamespace(raise_for_errors=lambda: None)
            with patch.object(
                script,
                "validate_operational_repository_snapshot",
                return_value=valid_snapshot,
            ):
                result = script.apply_finalization(repository, prepared)

            release, _etag = repository.current_release()
            self.assertIsNotNone(release)
            assert release is not None
            self.assertEqual(
                release.dataset_versions["corporate_actions"],
                prepared.planned_versions["corporate_actions"],
            )
            self.assertEqual(
                release.dataset_versions["adjustment_factors"],
                prepared.planned_versions["adjustment_factors"],
            )
            expected_source_version = script._adjustment_source_version(
                release.dataset_versions["daily_price_raw"],
                release.dataset_versions["corporate_actions"],
            )
            factor_manifest = repository.manifest_for_version(
                "adjustment_factors",
                release.dataset_versions["adjustment_factors"],
            )
            self.assertEqual(
                factor_manifest.metadata["source_version"],
                expected_source_version,
            )
            self.assertEqual(
                factor_manifest.metadata["source_corporate_actions_version"],
                release.dataset_versions["corporate_actions"],
            )
            self.assertEqual(
                result["new_dataset_versions"],
                {
                    **prepared.release.dataset_versions,
                    **prepared.planned_versions,
                },
            )

    def test_failed_rollback_writes_recovery_marker_and_blocks_next_apply(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, prepared = _transaction_fixture(root)

            def inject(stage: str) -> None:
                if stage != "after_corporate_actions":
                    return
                key = repository.current_key("corporate_actions")
                current = repository.objects.get(key)
                alien = CurrentPointer(
                    dataset="corporate_actions",
                    version="external-writer-version",
                    manifest_path="external/manifest.json",
                    manifest_sha256="e" * 64,
                    updated_at="2025-01-03T13:30:00Z",
                )
                repository.objects.put(key, alien.to_bytes(), if_match=current.etag)
                raise InjectedFailure(stage)

            valid_snapshot = SimpleNamespace(raise_for_errors=lambda: None)
            with patch.object(
                script,
                "validate_operational_repository_snapshot",
                return_value=valid_snapshot,
            ):
                with self.assertRaisesRegex(RuntimeError, "rollback failed"):
                    script.apply_finalization(
                        repository,
                        prepared,
                        failure_injector=inject,
                    )

            markers = list(
                (root / "recovery/lifecycle-finalizer").glob("*.json")
            )
            self.assertEqual(len(markers), 1)
            marker = json.loads(markers[0].read_bytes())
            self.assertEqual(marker["status"], "rollback_failed")
            self.assertTrue(marker["rollback_errors"])
            with self.assertRaisesRegex(RuntimeError, "recovery marker blocks writes"):
                script.apply_finalization(repository, prepared)

    def test_default_plan_performs_no_repository_or_archive_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "sec"
            candidate = _candidate("SEC-P", "PLAN")
            evidence = _artifact(cache, b"plan evidence")
            repository, release = _repository(
                root,
                [candidate],
                [
                    _price(candidate.security_id, "2025-01-02", 10.0),
                    _price("ACTIVE", COMPLETED, 1.0),
                ],
                extra_master=[_candidate("ACTIVE", "ACT", last=COMPLETED)],
            )
            document = _report(
                root,
                release,
                {
                    candidate.security_id: _record(
                        candidate,
                        artifact=evidence,
                        parsed=_cash_event(),
                    )
                },
            )
            before = sorted(path.relative_to(root) for path in root.rglob("*"))
            args = SimpleNamespace(
                cache_root=str(root),
                report=str(document.path),
                sec_cache=str(cache),
                apply=False,
            )

            result = script.run(
                args,
                repository_factory=lambda _root: repository,
                candidates=[candidate],
                exception_mapping={},
            )

            after = sorted(path.relative_to(root) for path in root.rglob("*"))
            self.assertEqual(result["status"], "validated_plan")
            self.assertFalse(result["writes_performed"])
            self.assertEqual(repository.writes, [])
            self.assertEqual(repository.commits, 0)
            self.assertEqual(before, after)

    def test_plan_replays_exact_report_artifact_from_bound_release_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = _candidate("SEC-ARCHIVE-PLAN", "ARP")
            payload = b"exact current-release SEC filing"
            source_hash = sha256_bytes(payload)
            source_url = (
                "https://www.sec.gov/Archives/edgar/data/1/"
                "000000000125000001/event.htm"
            )
            object_path = f"archives/replay/{source_hash}.html.gz"
            archive_path = root / object_path
            archive_path.parent.mkdir(parents=True)
            archive_path.write_bytes(gzip.compress(payload, mtime=0))
            evidence = {
                "source": "sec_edgar_filing",
                "source_url": source_url,
                "retrieved_at": "2025-01-03T12:00:00Z",
                "content_type": "text/html",
                "source_hash": source_hash,
            }
            repository, release = _repository(
                root,
                [candidate],
                [
                    _price(candidate.security_id, "2025-01-02", 10.0),
                    _price("ACTIVE", COMPLETED, 1.0),
                ],
                extra_master=[_candidate("ACTIVE", "ACT", last=COMPLETED)],
            )
            repository.frames["source_archive"] = _frame(
                "source_archive",
                [
                    {
                        "archive_id": source_hash,
                        "dataset": "sec_edgar_filing",
                        "object_path": object_path,
                        "content_type": "text/html",
                        "effective_date": COMPLETED,
                        "source": "sec_edgar_filing",
                        "retrieved_at": evidence["retrieved_at"],
                        "source_hash": source_hash,
                    }
                ],
            )
            repository.frames["source_archive"]["source_url"] = source_url
            document = _report(
                root,
                release,
                {
                    candidate.security_id: _record(
                        candidate,
                        artifact=evidence,
                        parsed=_cash_event(),
                    )
                },
            )
            before = sorted(path.relative_to(root) for path in root.rglob("*"))

            result = script.run(
                SimpleNamespace(
                    cache_root=str(root),
                    report=str(document.path),
                    sec_cache=str(root / "missing-sec-cache"),
                    apply=False,
                ),
                repository_factory=lambda _root: repository,
                candidates=[candidate],
                exception_mapping={},
            )

            after = sorted(path.relative_to(root) for path in root.rglob("*"))
            self.assertEqual(result["status"], "validated_plan")
            self.assertFalse(result["network_accessed"])
            self.assertFalse(result["writes_performed"])
            self.assertEqual(repository.writes, [])
            self.assertEqual(repository.commits, 0)
            self.assertEqual(before, after)

    def test_bound_release_replay_rejects_report_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = _candidate("SEC-ARCHIVE-HASH", "ARH")
            payload = b"current-release SEC filing"
            archive_hash = sha256_bytes(payload)
            report_hash = sha256_bytes(b"different report artifact")
            source_url = (
                "https://www.sec.gov/Archives/edgar/data/2/"
                "000000000225000001/event.htm"
            )
            object_path = f"archives/replay/{archive_hash}.html.gz"
            archive_path = root / object_path
            archive_path.parent.mkdir(parents=True)
            archive_path.write_bytes(gzip.compress(payload, mtime=0))
            evidence = {
                "source": "sec_edgar_filing",
                "source_url": source_url,
                "retrieved_at": "2025-01-03T12:00:00Z",
                "content_type": "text/html",
                "source_hash": report_hash,
            }
            repository, release = _repository(
                root,
                [candidate],
                [_price(candidate.security_id, "2025-01-02", 10.0)],
            )
            repository.frames["source_archive"] = _frame(
                "source_archive",
                [
                    {
                        "archive_id": archive_hash,
                        "dataset": "sec_edgar_filing",
                        "object_path": object_path,
                        "content_type": "text/html",
                        "effective_date": COMPLETED,
                        "source": "sec_edgar_filing",
                        "retrieved_at": evidence["retrieved_at"],
                        "source_hash": archive_hash,
                    }
                ],
            )
            repository.frames["source_archive"]["source_url"] = source_url
            document = _report(
                root,
                release,
                {
                    candidate.security_id: _record(
                        candidate,
                        artifact=evidence,
                        parsed=_cash_event(),
                    )
                },
            )

            with self.assertRaisesRegex(RuntimeError, "exact report artifact hash"):
                script.prepare_finalization(
                    repository,
                    release,
                    repository.release_etag,
                    document,
                    sec_cache=root / "missing-sec-cache",
                    exception_mapping={},
                    candidates=[candidate],
                )


class ExactShortTerminalReviewedResolutionTests(unittest.TestCase):
    @staticmethod
    def _candidate(expected: dict) -> LifecycleCandidate:
        return LifecycleCandidate(
            security_id=expected["security_id"],
            symbol=expected["symbol"],
            name=f"{expected['symbol']} exact reviewed fixture",
            exchange="NYSE",
            last_price_date=expected["last_price_date"],
            active_to=expected["last_price_date"],
        )

    def test_exact_three_reviewed_resolution_rows_are_hash_pinned(self):
        expected_keys = {
            script._key(
                "US:EODHD:0c47238f-bf19-5faa-a3ae-25a34ef3d3f5",
                "2021-05-13",
            ),
            script._key(
                "US:EODHD:716dea51-f3a0-5381-9696-d097c877695f",
                "2021-03-16",
            ),
            script._key(
                "US:EODHD:865c1483-a99b-5066-b55a-649e24804d68",
                "2025-11-28",
            ),
        }
        self.assertEqual(
            set(script.EXACT_SHORT_TERMINAL_REVIEWED_RESOLUTIONS),
            expected_keys,
        )
        for key, spec in script.EXACT_SHORT_TERMINAL_REVIEWED_RESOLUTIONS.items():
            with self.subTest(key=key):
                expected = spec["resolution"]
                frame = _frame("lifecycle_resolutions", [expected])
                observed = script._preserve_exact_short_terminal_reviewed_resolution(
                    self._candidate(expected),
                    {"lifecycle_resolutions": frame},
                )
                self.assertEqual(observed, expected)
                self.assertEqual(
                    script._lifecycle_resolution_row_sha256(observed),
                    spec["row_sha256"],
                )
                self.assertEqual(
                    observed["reviewed_by"],
                    "short_terminal_boundary_repair_v1",
                )
                self.assertEqual(
                    observed["reviewed_at"],
                    "2026-07-19T00:00:00Z",
                )

    def test_any_exact_reviewed_resolution_field_change_fails_closed(self):
        columns = dataset_spec("lifecycle_resolutions").required_columns
        for key, spec in script.EXACT_SHORT_TERMINAL_REVIEWED_RESOLUTIONS.items():
            expected = spec["resolution"]
            candidate = self._candidate(expected)
            for column in columns:
                with self.subTest(key=key, column=column):
                    changed = dict(expected)
                    changed[column] = (
                        "2000-01-01"
                        if column == "last_price_date"
                        else f"{changed[column]}-changed"
                    )
                    frame = _frame("lifecycle_resolutions", [changed])
                    with self.assertRaises(RuntimeError):
                        script._preserve_exact_short_terminal_reviewed_resolution(
                            candidate,
                            {"lifecycle_resolutions": frame},
                        )

    def test_exact_reviewed_resolution_schema_change_fails_closed(self):
        spec = next(
            iter(script.EXACT_SHORT_TERMINAL_REVIEWED_RESOLUTIONS.values())
        )
        expected = spec["resolution"]
        frame = _frame("lifecycle_resolutions", [expected])
        frame["unexpected_column"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "schema changed"):
            script._preserve_exact_short_terminal_reviewed_resolution(
                self._candidate(expected),
                {"lifecycle_resolutions": frame},
            )

    def test_generated_resolution_must_match_all_thirteen_nonowned_fields(self):
        excluded = {"reviewed_by", "reviewed_at", "source", "retrieved_at"}
        columns = dataset_spec("lifecycle_resolutions").required_columns
        self.assertEqual(len(columns), 17)
        self.assertEqual(len(set(columns) - excluded), 13)
        for key, spec in script.EXACT_SHORT_TERMINAL_REVIEWED_RESOLUTIONS.items():
            prior = spec["resolution"]
            generated = {
                **prior,
                "reviewed_by": script.REVIEWED_BY,
                "reviewed_at": script.REVIEWED_AT,
                "source": "lifecycle_finalizer",
                "retrieved_at": "2026-07-18T12:00:00Z",
            }
            with self.subTest(key=key, state="exact"):
                self.assertEqual(
                    script._restore_exact_short_terminal_reviewed_resolution(
                        prior,
                        generated,
                    ),
                    prior,
                )
            for column in set(columns) - excluded:
                with self.subTest(key=key, column=column):
                    changed = dict(generated)
                    changed[column] = (
                        "2000-01-01"
                        if column == "last_price_date"
                        else f"{changed[column]}-changed"
                    )
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "Generic lifecycle validation disagrees",
                    ):
                        script._restore_exact_short_terminal_reviewed_resolution(
                            prior,
                            changed,
                        )

    def test_exact_restore_runs_only_after_normal_crosscheck(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "sec-cache"
            candidate = _candidate("EXACT-CROSSCHECK", "XCP")
            active = _candidate("EXACT-ACTIVE", "XCA", last=COMPLETED)
            evidence = _artifact(cache, b"exact crosscheck ordering evidence")
            repository, release = _repository(
                root,
                [candidate],
                [
                    _price(candidate.security_id, candidate.last_price_date, 10.0),
                    _price(active.security_id, COMPLETED, 1.0),
                ],
                extra_master=[active],
            )
            release.dataset_versions["lifecycle_resolutions"] = (
                "lifecycle_resolutions-v1"
            )
            repository.frames["lifecycle_resolutions"] = _frame(
                "lifecycle_resolutions"
            )
            record = _record(
                candidate,
                artifact=evidence,
                parsed=_cash_event(),
            )
            document = _report(root, release, {candidate.security_id: record})
            generic = script.prepare_finalization(
                repository,
                release,
                repository.release_etag,
                document,
                sec_cache=cache,
                exception_mapping={},
                candidates=[candidate],
            ).frames["lifecycle_resolutions"].iloc[0].to_dict()
            prior = {
                **generic,
                "reviewed_by": "short_terminal_boundary_repair_v1",
                "reviewed_at": "2026-07-19T00:00:00Z",
                "source": "short_terminal_boundary_repair",
                "retrieved_at": "2026-07-19T00:00:00Z",
            }
            exact_mapping = {
                script._key(candidate.security_id, candidate.last_price_date): {
                    "resolution": prior,
                    "row_sha256": script._lifecycle_resolution_row_sha256(prior),
                }
            }
            repository.frames["lifecycle_resolutions"] = _frame(
                "lifecycle_resolutions", [prior]
            )

            with (
                patch.object(
                    script,
                    "EXACT_SHORT_TERMINAL_REVIEWED_RESOLUTIONS",
                    exact_mapping,
                ),
                patch.object(
                    script,
                    "_preserved_exact_repair_resolution",
                    return_value=prior,
                ),
                patch.object(script, "_crosscheck_event") as crosscheck,
            ):
                with self.assertRaisesRegex(RuntimeError, "overlaps"):
                    script.prepare_finalization(
                        repository,
                        release,
                        repository.release_etag,
                        document,
                        sec_cache=cache,
                        exception_mapping={},
                        candidates=[candidate],
                    )
            crosscheck.assert_not_called()

            with (
                patch.object(
                    script,
                    "EXACT_SHORT_TERMINAL_REVIEWED_RESOLUTIONS",
                    exact_mapping,
                ),
                patch.object(
                    script,
                    "_crosscheck_event",
                    wraps=script._crosscheck_event,
                ) as crosscheck,
                patch.object(
                    script,
                    "_restore_exact_short_terminal_reviewed_resolution",
                    wraps=script._restore_exact_short_terminal_reviewed_resolution,
                ) as restore,
            ):
                prepared = script.prepare_finalization(
                    repository,
                    release,
                    repository.release_etag,
                    document,
                    sec_cache=cache,
                    exception_mapping={},
                    candidates=[candidate],
                )
            crosscheck.assert_called_once()
            restore.assert_called_once()
            self.assertEqual(
                prepared.frames["lifecycle_resolutions"].iloc[0].to_dict(),
                prior,
            )

            with (
                patch.object(
                    script,
                    "EXACT_SHORT_TERMINAL_REVIEWED_RESOLUTIONS",
                    exact_mapping,
                ),
                patch.object(
                    script,
                    "_crosscheck_event",
                    side_effect=RuntimeError("sentinel crosscheck failure"),
                ) as crosscheck,
                patch.object(
                    script,
                    "_restore_exact_short_terminal_reviewed_resolution",
                    wraps=script._restore_exact_short_terminal_reviewed_resolution,
                ) as restore,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "sentinel crosscheck failure",
                ):
                    script.prepare_finalization(
                        repository,
                        release,
                        repository.release_etag,
                        document,
                        sec_cache=cache,
                        exception_mapping={},
                        candidates=[candidate],
                    )
            crosscheck.assert_called_once()
            restore.assert_not_called()

            ineligible = _report(
                root,
                release,
                {
                    candidate.security_id: _record(
                        candidate,
                        artifact=evidence,
                        eligible=False,
                        parsed=None,
                    )
                },
            )
            with (
                patch.object(
                    script,
                    "EXACT_SHORT_TERMINAL_REVIEWED_RESOLUTIONS",
                    exact_mapping,
                ),
                patch.object(script, "_crosscheck_event") as crosscheck,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "cannot bypass the normal applied lifecycle path",
                ):
                    script.prepare_finalization(
                        repository,
                        release,
                        repository.release_etag,
                        ineligible,
                        sec_cache=cache,
                        exception_mapping={},
                        candidates=[candidate],
                    )
            crosscheck.assert_not_called()

    def test_unlisted_candidate_never_uses_exact_preservation(self):
        candidate = _candidate("UNLISTED", "NOPE", last="2021-05-13")
        self.assertIsNone(
            script._preserve_exact_short_terminal_reviewed_resolution(
                candidate,
                {"lifecycle_resolutions": _frame("lifecycle_resolutions")},
            )
        )


if __name__ == "__main__":
    unittest.main()
