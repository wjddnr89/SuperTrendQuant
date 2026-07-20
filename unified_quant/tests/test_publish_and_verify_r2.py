from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
import tempfile
import threading
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pandas as pd
import yaml

from supertrend_quant.config import R2Config
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    DatasetManifest,
    ManifestFile,
    sha256_bytes,
)
from supertrend_quant.market_store.lifecycle import LifecycleCandidate
from supertrend_quant.market_store.lifecycle_coverage import (
    LifecycleExceptionCode,
    lifecycle_candidate_id,
    validate_lifecycle_coverage,
)
from supertrend_quant.market_store.official_lifecycle_evidence import (
    OfficialLifecycleExceptionEvidenceSpec,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.source_archive import (
    TRUSTED_PROVENANCE_ARCHIVE_ID_INVENTORY_SHA256,
    reviewed_provenance_archive_ids,
    validate_source_archive_id,
)
from supertrend_quant.market_store.storage import (
    DatasetCache,
    LocalObjectStore,
    R2ObjectStore,
    R2PrivacyVerificationError,
    R2PrivacyVerifier,
)


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "publish_and_verify_r2.py"
POLICY_PATH = Path(__file__).parents[1] / "configs" / "us_cross_validation.yaml"
SCRIPT_SPEC = importlib.util.spec_from_file_location("publish_and_verify_r2", SCRIPT_PATH)
if SCRIPT_SPEC is None or SCRIPT_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Could not load {SCRIPT_PATH}")
publish_script = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(publish_script)


def _terminal_tail_registry_fixture():
    policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    registry = copy.deepcopy(
        policy["events"]["reviewed_terminal_price_tail_corrections"]
    )
    metadata = {
        dataset: {
            "terminal_tail_registry_draft": copy.deepcopy(registry),
            "terminal_tail_registry_inventory_sha256": (
                publish_script.TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256
            ),
        }
        for dataset in publish_script._TERMINAL_TAIL_WRITE_DATASETS
    }

    class Repository:
        def manifest_for_version(self, dataset, version):
            if version != f"{dataset}-v1":  # pragma: no cover
                raise AssertionError((dataset, version))
            return SimpleNamespace(metadata=metadata[dataset])

    release = SimpleNamespace(
        dataset_versions={
            dataset: f"{dataset}-v1"
            for dataset in publish_script._TERMINAL_TAIL_WRITE_DATASETS
        }
    )
    return Repository(), release, metadata


class _S3UnsupportedError(RuntimeError):
    response = {"Error": {"Code": "NotImplemented"}}


def _s3_privacy_client(**overrides):
    values = {
        "head_bucket": Mock(return_value={}),
        "get_public_access_block": Mock(side_effect=_S3UnsupportedError()),
        "get_bucket_policy_status": Mock(side_effect=_S3UnsupportedError()),
        "get_bucket_policy": Mock(side_effect=_S3UnsupportedError()),
        "get_bucket_acl": Mock(side_effect=_S3UnsupportedError()),
        "put_object": Mock(return_value={"ETag": '"etag"'}),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _api_response(result: dict, *, status_code: int = 200):
    return SimpleNamespace(
        status_code=status_code,
        json=Mock(return_value={"success": True, "result": result}),
    )


def _repository_with_release(
    root: Path,
    *,
    inherited: bool = False,
) -> tuple[LocalDatasetRepository, DataRelease]:
    repository = LocalDatasetRepository(root)
    dataset = "security_master"

    parent_version = "parent-v1"
    parent_prefix = repository.version_prefix(dataset, parent_version)
    parent_payload = b"parent parquet bytes"
    repository.objects.put(f"{parent_prefix}/part.bin", parent_payload)
    parent_manifest = DatasetManifest.create(
        dataset,
        parent_version,
        "2026-07-17",
        (
            ManifestFile(
                path="part.bin",
                sha256=sha256_bytes(parent_payload),
                size_bytes=len(parent_payload),
                row_count=1,
            ),
        ),
        metadata={"inherits_parent": False},
    )
    repository.objects.put(
        f"{parent_prefix}/manifest.json",
        parent_manifest.to_bytes(),
    )

    version = "child-v1" if inherited else parent_version
    if inherited:
        child_prefix = repository.version_prefix(dataset, version)
        child_payload = b"child parquet bytes"
        repository.objects.put(f"{child_prefix}/part.bin", child_payload)
        manifest = DatasetManifest.create(
            dataset,
            version,
            "2026-07-17",
            (
                ManifestFile(
                    path="part.bin",
                    sha256=sha256_bytes(child_payload),
                    size_bytes=len(child_payload),
                    row_count=1,
                ),
            ),
            parent_version=parent_version,
            metadata={"inherits_parent": True},
        )
        repository.objects.put(
            f"{child_prefix}/manifest.json",
            manifest.to_bytes(),
        )
    else:
        manifest = parent_manifest

    manifest_path = f"{repository.version_prefix(dataset, version)}/manifest.json"
    repository.objects.put(
        repository.current_key(dataset),
        CurrentPointer.create(manifest, manifest_path).to_bytes(),
    )
    release = DataRelease(
        version="release-v1",
        created_at="2026-07-18T00:00:00Z",
        completed_session="2026-07-17",
        dataset_versions={dataset: version},
    )
    repository.objects.put(f"releases/{release.version}.json", release.to_bytes())
    repository.objects.put("releases/current.json", release.to_bytes())
    return repository, release


def _lifecycle_gate_fixture():
    candidate = LifecycleCandidate(
        security_id="SEC-A",
        symbol="SECA",
        name="Security A",
        exchange="NYSE",
        last_price_date="2024-01-02",
        active_to="",
    )
    candidate_id = lifecycle_candidate_id(
        candidate.security_id,
        candidate.last_price_date,
    )
    resolutions = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "security_id": candidate.security_id,
                "symbol": candidate.symbol,
                "last_price_date": candidate.last_price_date,
                "resolution": "exception",
                "event_id": "",
                "exception_code": LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION,
                "exception_reason": "The consideration is not representable.",
                "reviewed_by": "reviewer",
                "reviewed_at": "2026-07-18T00:00:00Z",
                "recheck_after": "",
                "successor_security_id": "",
                "successor_symbol": "",
                "source_url": "https://www.sec.gov/Archives/evidence.txt",
                "source": "lifecycle_review",
                "retrieved_at": "2026-07-18T00:00:00Z",
                "source_hash": "resolution-source-hash",
            }
        ]
    )
    candidates = pd.DataFrame([candidate.__dict__])
    actions = pd.DataFrame()
    report = validate_lifecycle_coverage(
        candidates,
        resolutions,
        actions,
        completed_session="2026-07-18",
    )
    metadata = {
        **report.manifest_metadata(),
        "evidence_report_sha256": "evidence-report-sha256",
    }
    frames = {
        "lifecycle_resolutions": resolutions,
        "corporate_actions": actions,
        "source_archive": pd.DataFrame(
            [{"archive_id": "evidence-report-sha256"}]
        ),
    }

    class FixtureRepository:
        def read_frame(self, dataset: str, _version: str):
            return frames[dataset].copy()

        def manifest_for_version(self, dataset: str, _version: str):
            if dataset != "lifecycle_resolutions":  # pragma: no cover
                raise AssertionError(dataset)
            return SimpleNamespace(metadata=metadata)

    release = DataRelease(
        version="release-v1",
        created_at="2026-07-18T00:00:00Z",
        completed_session="2026-07-18",
        dataset_versions={
            "security_master": "master-v1",
            "symbol_history": "history-v1",
            "daily_price_raw": "prices-v1",
            "index_constituent_anchors": "anchors-v1",
            "index_membership_events": "events-v1",
            "corporate_actions": "actions-v1",
            "lifecycle_resolutions": "resolutions-v1",
            "source_archive": "archive-v1",
        },
    )
    return FixtureRepository(), release, candidate, metadata, frames


def _private_archive_gate_fixture(
    root: Path,
    *,
    payload_policy: dict | None = None,
    release_warnings: tuple[str, ...] | None = None,
):
    expected_policy = copy.deepcopy(
        publish_script._PRIVATE_INTERNAL_ONLY_LICENSE_POLICY
    )
    payload = {
        "schema": "fixture_private_archive/v1",
        "license_policy": copy.deepcopy(payload_policy or expected_policy),
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    source_hash = sha256_bytes(raw)
    object_path = f"archives/2026-07-19/{source_hash}.json.gz"
    path = root / object_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(raw, mtime=0))
    source_url = "https://example.test/private-evidence.json"
    dataset = "reviewed_fixture_private_archive"
    source = "reviewed_fixture_private_archive"
    frame = pd.DataFrame(
        [
            {
                "archive_id": source_hash,
                "content_type": "application/json",
                "dataset": dataset,
                "object_path": object_path,
                "source": source,
                "source_hash": source_hash,
                "source_url": source_url,
            }
        ]
    )
    warning = publish_script._WIKI_PRIVATE_INTERNAL_ONLY_WARNING
    release = DataRelease(
        version="release-private-v1",
        created_at="2026-07-19T00:00:00Z",
        completed_session="2026-07-19",
        dataset_versions={"source_archive": "archive-private-v1"},
        warnings=release_warnings if release_warnings is not None else (warning,),
    )
    repository = SimpleNamespace(
        root=root,
        read_frame=Mock(return_value=frame),
    )
    spec = {
        "dataset": dataset,
        "source": source,
        "source_url": source_url,
        "source_hash": source_hash,
        "object_path": object_path,
        "content_type": "application/json",
        "schema": "fixture_private_archive/v1",
        "license_policy": expected_policy,
        "warning": warning,
    }
    return repository, release, spec, path


class PrivateInternalOnlySourceArchiveGateTest(unittest.TestCase):
    def test_cli_acknowledgement_defaults_false_and_is_explicit(self):
        default = publish_script._parser().parse_args([])
        acknowledged = publish_script._parser().parse_args(
            ["--ack-private-internal-only-source-archives"]
        )

        self.assertFalse(default.ack_private_internal_only_source_archives)
        self.assertTrue(acknowledged.ack_private_internal_only_source_archives)
        self.assertFalse(default.preflight_only)

    def test_known_wiki_provenance_inventory_covers_both_audits_and_swy(self):
        by_dataset = {
            item["dataset"]: item
            for item in publish_script._PRIVATE_INTERNAL_ONLY_PROVENANCE_SPECS
        }

        self.assertEqual(
            set(by_dataset),
            {
                "reviewed_us_wiki_price_arbitration",
                "reviewed_us_wiki14_price_only_arbitration",
                "reviewed_swy_wiki_history_provenance",
            },
        )
        for dataset in (
            "reviewed_us_wiki_price_arbitration",
            "reviewed_us_wiki14_price_only_arbitration",
        ):
            self.assertEqual(
                by_dataset[dataset]["license_policy"],
                publish_script._PRIVATE_INTERNAL_ONLY_LICENSE_POLICY,
            )

    def test_hash_verified_gzip_json_exact_policy_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, release, spec, _path = _private_archive_gate_fixture(
                Path(directory)
            )
            with patch.object(
                publish_script,
                "_PRIVATE_INTERNAL_ONLY_PROVENANCE_SPECS",
                (spec,),
            ):
                result = (
                    publish_script._private_internal_only_source_archive_restrictions(
                        repository,
                        release,
                    )
                )

        self.assertTrue(result["restricted"])
        self.assertEqual(len(result["reviewed_provenance"]), 1)
        self.assertEqual(
            result["reviewed_provenance"][0]["raw_sha256"],
            spec["source_hash"],
        )
        self.assertEqual(
            result["release_warning_restrictions"],
            [publish_script._WIKI_PRIVATE_INTERNAL_ONLY_WARNING],
        )

    def test_changed_exact_license_policy_fails_closed(self):
        changed_policy = copy.deepcopy(
            publish_script._PRIVATE_INTERNAL_ONLY_LICENSE_POLICY
        )
        changed_policy["public_publication_allowed"] = True
        with tempfile.TemporaryDirectory() as directory:
            repository, release, spec, _path = _private_archive_gate_fixture(
                Path(directory),
                payload_policy=changed_policy,
            )
            with (
                patch.object(
                    publish_script,
                    "_PRIVATE_INTERNAL_ONLY_PROVENANCE_SPECS",
                    (spec,),
                ),
                self.assertRaisesRegex(RuntimeError, "policy changed"),
            ):
                publish_script._private_internal_only_source_archive_restrictions(
                    repository,
                    release,
                )

    def test_raw_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, release, spec, path = _private_archive_gate_fixture(
                Path(directory)
            )
            path.write_bytes(gzip.compress(b"{}", mtime=0))
            with (
                patch.object(
                    publish_script,
                    "_PRIVATE_INTERNAL_ONLY_PROVENANCE_SPECS",
                    (spec,),
                ),
                self.assertRaisesRegex(RuntimeError, "hash mismatch"),
            ):
                publish_script._private_internal_only_source_archive_restrictions(
                    repository,
                    release,
                )

    def test_near_match_private_wiki_warning_fails_closed(self):
        changed_warning = (
            publish_script._WIKI_PRIVATE_INTERNAL_ONLY_WARNING + " changed"
        )
        release = SimpleNamespace(
            warnings=(changed_warning,),
            dataset_versions={},
        )
        with self.assertRaisesRegex(RuntimeError, "Unrecognized WIKI"):
            publish_script._private_internal_only_source_archive_restrictions(
                SimpleNamespace(),
                release,
            )

    def test_known_provenance_without_exact_release_warning_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, release, spec, _path = _private_archive_gate_fixture(
                Path(directory),
                release_warnings=(),
            )
            with (
                patch.object(
                    publish_script,
                    "_PRIVATE_INTERNAL_ONLY_PROVENANCE_SPECS",
                    (spec,),
                ),
                self.assertRaisesRegex(RuntimeError, "release warning is missing"),
            ):
                publish_script._private_internal_only_source_archive_restrictions(
                    repository,
                    release,
                )

    def test_exact_wiki_and_swy_release_warnings_alone_require_ack(self):
        release = SimpleNamespace(
            warnings=(
                publish_script._WIKI_PRIVATE_INTERNAL_ONLY_WARNING,
                publish_script._SWY_PRIVATE_INTERNAL_ONLY_WARNING,
            ),
            dataset_versions={},
        )
        restrictions = (
            publish_script._private_internal_only_source_archive_restrictions(
                SimpleNamespace(),
                release,
            )
        )

        self.assertTrue(restrictions["restricted"])
        self.assertEqual(
            set(restrictions["release_warning_restrictions"]),
            publish_script._PRIVATE_INTERNAL_ONLY_RELEASE_WARNINGS,
        )
        with self.assertRaisesRegex(RuntimeError, "explicit publisher"):
            publish_script._require_private_internal_only_publisher_ack(
                restrictions,
                acknowledged=False,
            )

    def test_publish_and_verify_only_both_block_before_any_remote_access(self):
        restrictions = {
            "restricted": True,
            "source_archive_version": "archive-v1",
            "reviewed_provenance": [{"dataset": "private"}],
            "release_warning_restrictions": [],
            "evidence_sha256": "evidence",
        }
        release = DataRelease(
            version="release-v1",
            created_at="2026-07-19T00:00:00Z",
            completed_session="2026-07-19",
            dataset_versions={},
        )
        for verify_only in (False, True):
            with self.subTest(verify_only=verify_only):
                repository = SimpleNamespace(
                    current_release=Mock(return_value=(release, None))
                )
                store = SimpleNamespace(
                    verify_private_access=Mock(),
                    get=Mock(),
                    put=Mock(),
                    list=Mock(),
                )
                with (
                    patch.object(
                        publish_script,
                        "_private_internal_only_source_archive_restrictions",
                        return_value=restrictions,
                    ),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "--ack-private-internal-only-source-archives",
                    ),
                ):
                    publish_script.publish_and_verify(
                        repository,
                        store,
                        verify_only=verify_only,
                        keep_verify_cache=False,
                    )

                store.verify_private_access.assert_not_called()
                store.get.assert_not_called()
                store.put.assert_not_called()
                store.list.assert_not_called()

    def test_acknowledgement_does_not_bypass_cloudflare_private_state_gate(self):
        restrictions = {
            "restricted": True,
            "source_archive_version": "archive-v1",
            "reviewed_provenance": [{"dataset": "private"}],
            "release_warning_restrictions": [],
            "evidence_sha256": "evidence",
        }
        for verify_only in (False, True):
            with self.subTest(verify_only=verify_only):
                with tempfile.TemporaryDirectory() as directory:
                    repository, _release = _repository_with_release(
                        Path(directory)
                    )
                    store = SimpleNamespace(
                        verify_private_access=Mock(
                            side_effect=R2PrivacyVerificationError(
                                "private state not proven"
                            )
                        ),
                        get=Mock(),
                        put=Mock(),
                        list=Mock(),
                    )
                    stats = {
                        "snapshot_sha256": "snapshot",
                        "state_sha256": "state",
                    }
                    with (
                        patch.object(
                            publish_script,
                            "_private_internal_only_source_archive_restrictions",
                            return_value=restrictions,
                        ),
                        patch.object(
                            publish_script,
                            "_validate_release_lifecycle_coverage",
                            return_value={"open_count": 0},
                        ),
                        patch.object(
                            publish_script,
                            "validate_cross_validation_gate",
                            return_value={"price_unresolved_count": 0},
                        ),
                        patch.object(
                            publish_script,
                            "_validate_terminal_transition_readiness",
                            return_value={"ready": True, "issue_count": 0},
                        ),
                        patch.object(
                            publish_script,
                            "validate_release_snapshot",
                            return_value=dict(stats),
                        ),
                        patch.object(
                            publish_script,
                            "_fingerprint_release_state",
                            return_value=dict(stats),
                        ),
                        patch.object(
                            publish_script,
                            "publish_repository",
                        ) as publish,
                        self.assertRaisesRegex(
                            R2PrivacyVerificationError,
                            "not proven",
                        ),
                    ):
                        publish_script.publish_and_verify(
                            repository,
                            store,
                            verify_only=verify_only,
                            keep_verify_cache=False,
                            ack_private_internal_only_source_archives=True,
                        )

                    store.verify_private_access.assert_called_once_with()
                    publish.assert_not_called()
                    store.get.assert_not_called()
                    store.put.assert_not_called()
                    store.list.assert_not_called()


class OfflinePreflightTest(unittest.TestCase):
    def test_parser_exposes_mutually_exclusive_preflight_mode(self):
        args = publish_script._parser().parse_args(["--preflight-only"])

        self.assertTrue(args.preflight_only)
        self.assertFalse(args.verify_only)
        self.assertFalse(args.privacy_check_only)
        with self.assertRaises(SystemExit):
            publish_script._parser().parse_args(
                ["--preflight-only", "--verify-only"]
            )

    def test_preflight_accumulates_blockers_and_runs_independent_local_gates(self):
        release = DataRelease(
            version="release-v1",
            created_at="2026-07-19T00:00:00Z",
            completed_session="2026-07-19",
            dataset_versions={"security_master": "master-v1"},
        )
        repository = SimpleNamespace(
            current_release=Mock(return_value=(release, None))
        )
        restrictions = {
            "restricted": True,
            "source_archive_version": "archive-v1",
            "reviewed_provenance": [{"dataset": "private"}],
            "release_warning_restrictions": [],
            "evidence_sha256": "evidence",
        }
        stable = {"snapshot_sha256": "snapshot", "state_sha256": "state"}
        with (
            patch.object(
                publish_script,
                "_private_internal_only_source_archive_restrictions",
                return_value=restrictions,
            ) as archive_gate,
            patch.object(
                publish_script,
                "_validate_release_lifecycle_coverage",
                side_effect=RuntimeError("17 lifecycle candidates remain open"),
            ) as lifecycle_gate,
            patch.object(
                publish_script,
                "validate_cross_validation_gate",
                return_value={"status": "passed"},
            ) as cross_gate,
            patch.object(
                publish_script,
                "_validate_terminal_transition_readiness",
                return_value={"ready": True},
            ) as terminal_gate,
            patch.object(
                publish_script,
                "validate_release_snapshot",
                return_value=dict(stable),
            ) as snapshot_gate,
            patch.object(
                publish_script,
                "_fingerprint_release_state",
                return_value=dict(stable),
            ) as fingerprint_gate,
        ):
            result = publish_script.run_local_preflight(
                repository,
                ack_private_internal_only_source_archives=False,
            )

        self.assertEqual(result["status"], "blocked")
        self.assertTrue(result["local_only"])
        self.assertFalse(result["eodhd_accessed"])
        self.assertEqual(
            {item["gate"] for item in result["blockers"]},
            {"private_archive_publisher_ack", "lifecycle_coverage"},
        )
        self.assertEqual(result["gates"]["cross_validation"]["status"], "passed")
        self.assertEqual(
            result["gates"]["release_state_stability"]["status"],
            "passed",
        )
        self.assertIn(
            "17 lifecycle candidates remain open",
            result["gates"]["lifecycle_coverage"]["error"],
        )
        self.assertEqual(archive_gate.call_count, 2)
        lifecycle_gate.assert_called_once_with(repository, release)
        cross_gate.assert_called_once_with(repository, release)
        terminal_gate.assert_called_once_with(repository, release)
        snapshot_gate.assert_called_once_with(repository, release)
        fingerprint_gate.assert_called_once_with(repository, release)

    def test_preflight_main_never_constructs_r2_store(self):
        config = SimpleNamespace(
            local_cache_dir="cache",
            r2=SimpleNamespace(
                enabled=True,
                bucket="private-bucket",
                prefix="private-prefix",
            ),
        )
        parser = SimpleNamespace(
            parse_args=Mock(
                return_value=SimpleNamespace(
                    data_config="unused",
                    preflight_only=True,
                    privacy_check_only=False,
                    verify_only=False,
                    keep_verify_cache=False,
                    ack_private_internal_only_source_archives=True,
                )
            )
        )
        preflight = {
            "status": "ready",
            "mode": "preflight_only",
            "local_only": True,
            "eodhd_accessed": False,
            "release": {"version": "release-v1"},
            "gates": {},
            "blockers": [],
            "remaining_remote_gates": [],
        }
        with (
            patch.object(publish_script, "_parser", return_value=parser),
            patch.object(
                publish_script,
                "load_data_store_config",
                return_value=config,
            ),
            patch.object(publish_script, "LocalDatasetRepository") as repository,
            patch.object(
                publish_script,
                "run_local_preflight",
                return_value=preflight,
            ) as run_preflight,
            patch.object(publish_script, "R2ObjectStore") as r2_store,
            patch("builtins.print") as output,
        ):
            publish_script.main()

        repository.assert_called_once_with("cache")
        run_preflight.assert_called_once_with(
            repository.return_value,
            ack_private_internal_only_source_archives=True,
        )
        r2_store.assert_not_called()
        summary = json.loads(output.call_args.args[0])
        self.assertEqual(summary["mode"], "preflight_only")
        self.assertEqual(summary["gates"]["r2_configuration"]["status"], "passed")


class R2PrivateVisibilityGateTest(unittest.TestCase):
    account_id = "a" * 32
    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"

    def _config(self, **overrides):
        return R2Config(
            enabled=True,
            bucket="private-bucket",
            **overrides,
        )

    def test_privacy_check_only_cli_mode(self):
        args = publish_script._parser().parse_args(["--privacy-check-only"])
        self.assertTrue(args.privacy_check_only)
        self.assertFalse(args.verify_only)

    def test_privacy_check_only_main_never_constructs_repository_or_writes(self):
        config = SimpleNamespace(
            local_cache_dir="unused",
            r2=self._config(),
        )
        store = SimpleNamespace(
            verify_private_access=Mock(
                return_value={
                    "status": "verified_private",
                    "verification_method": "cloudflare_api",
                    "managed_r2_dev_enabled": False,
                    "enabled_custom_domain_count": 0,
                }
            ),
            put=Mock(),
        )
        parser = SimpleNamespace(
            parse_args=Mock(
                return_value=SimpleNamespace(
                    data_config="unused",
                    privacy_check_only=True,
                    verify_only=False,
                    keep_verify_cache=False,
                )
            )
        )
        with (
            patch.object(publish_script, "load_env") as load_env,
            patch.object(publish_script, "_parser", return_value=parser),
            patch.object(
                publish_script,
                "load_data_store_config",
                return_value=config,
            ),
            patch.object(
                publish_script,
                "R2ObjectStore",
                return_value=store,
            ),
            patch.object(publish_script, "LocalDatasetRepository") as repository,
            patch("builtins.print") as output,
        ):
            publish_script.main()

        load_env.assert_called_once_with()
        repository.assert_not_called()
        store.verify_private_access.assert_called_once_with(force=True)
        store.put.assert_not_called()
        summary = json.loads(output.call_args.args[0])
        self.assertEqual(summary["mode"], "privacy_check_only")

    def test_cloudflare_api_proves_r2_dev_and_custom_domains_are_disabled(self):
        s3 = _s3_privacy_client()

        def response_for(url, **kwargs):
            self.assertNotIn("secret-token", url)
            self.assertEqual(kwargs["timeout"], 30)
            self.assertFalse(kwargs["allow_redirects"])
            self.assertEqual(
                kwargs["headers"]["Authorization"],
                "Bearer secret-token",
            )
            if url.endswith("/managed"):
                return _api_response(
                    {"bucketId": "bucket-id", "domain": "private.r2.dev", "enabled": False}
                )
            if url.endswith("/custom"):
                return _api_response(
                    {"domains": [{"domain": "data.example.com", "enabled": False}]}
                )
            raise AssertionError(url)

        session = SimpleNamespace(get=Mock(side_effect=response_for))
        verifier = R2PrivacyVerifier(
            self._config(),
            s3,
            self.endpoint,
            environ={
                "CLOUDFLARE_ACCOUNT_ID": self.account_id,
                "CLOUDFLARE_API_TOKEN": "secret-token",
            },
            http_session=session,
        )

        result = verifier.verify()

        self.assertEqual(result["status"], "verified_private")
        self.assertEqual(result["verification_method"], "cloudflare_api")
        self.assertFalse(result["managed_r2_dev_enabled"])
        self.assertEqual(result["enabled_custom_domain_count"], 0)
        self.assertNotIn("secret-token", json.dumps(result))
        s3.head_bucket.assert_called_once_with(Bucket="private-bucket")
        s3.get_public_access_block.assert_called_once_with(Bucket="private-bucket")
        s3.get_bucket_policy_status.assert_called_once_with(Bucket="private-bucket")
        s3.get_bucket_policy.assert_called_once_with(Bucket="private-bucket")
        s3.get_bucket_acl.assert_called_once_with(Bucket="private-bucket")
        self.assertEqual(session.get.call_count, 2)

    def test_cloudflare_account_id_is_derived_from_official_r2_endpoint(self):
        session = SimpleNamespace(
            get=Mock(
                side_effect=(
                    _api_response(
                        {
                            "bucketId": "bucket-id",
                            "domain": "private.r2.dev",
                            "enabled": False,
                        }
                    ),
                    _api_response({"domains": []}),
                )
            )
        )
        verifier = R2PrivacyVerifier(
            self._config(),
            _s3_privacy_client(),
            self.endpoint,
            environ={"CLOUDFLARE_API_TOKEN": "secret-token"},
            http_session=session,
        )

        result = verifier.verify()

        self.assertEqual(result["verification_method"], "cloudflare_api")
        requested_urls = [call.args[0] for call in session.get.call_args_list]
        self.assertTrue(
            all(f"/accounts/{self.account_id}/r2/buckets/" in url for url in requested_urls)
        )

    def test_enabled_r2_dev_or_custom_domain_fails_closed(self):
        variants = (
            (
                {"domain": "public.r2.dev", "enabled": True},
                {"domains": []},
                "r2.dev",
            ),
            (
                {"domain": "private.r2.dev", "enabled": False},
                {"domains": [{"domain": "public.example.com", "enabled": True}]},
                "custom domain",
            ),
        )
        for managed, custom, message in variants:
            with self.subTest(message=message):
                session = SimpleNamespace(
                    get=Mock(
                        side_effect=(
                            _api_response(managed),
                            _api_response(custom),
                        )
                    )
                )
                verifier = R2PrivacyVerifier(
                    self._config(),
                    _s3_privacy_client(),
                    self.endpoint,
                    environ={
                        "CLOUDFLARE_ACCOUNT_ID": self.account_id,
                        "CLOUDFLARE_API_TOKEN": "secret-token",
                    },
                    http_session=session,
                )
                with self.assertRaisesRegex(R2PrivacyVerificationError, message):
                    verifier.verify()

    def test_authoritative_public_result_cannot_fall_back_to_attestation(self):
        now = datetime(2026, 7, 18, 2, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "privacy.json"
            content = json.dumps(
                self._attestation(now),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            path.write_bytes(content)
            session = SimpleNamespace(
                get=Mock(
                    return_value=_api_response(
                        {"domain": "public.r2.dev", "enabled": True}
                    )
                )
            )
            verifier = R2PrivacyVerifier(
                self._config(),
                _s3_privacy_client(),
                self.endpoint,
                environ={
                    "CLOUDFLARE_ACCOUNT_ID": self.account_id,
                    "CLOUDFLARE_API_TOKEN": "secret-token",
                    "R2_PRIVACY_ATTESTATION_PATH": str(path),
                    "R2_PRIVACY_ATTESTATION_SHA256": sha256_bytes(content),
                },
                http_session=session,
                now=lambda: now,
            )

            with self.assertRaisesRegex(R2PrivacyVerificationError, "r2.dev"):
                verifier.verify()

    def test_positive_s3_public_policy_blocks_before_cloudflare_api(self):
        s3 = _s3_privacy_client(
            get_bucket_policy_status=Mock(
                return_value={"PolicyStatus": {"IsPublic": True}}
            )
        )
        session = SimpleNamespace(get=Mock())
        verifier = R2PrivacyVerifier(
            self._config(),
            s3,
            self.endpoint,
            environ={
                "CLOUDFLARE_ACCOUNT_ID": self.account_id,
                "CLOUDFLARE_API_TOKEN": "secret-token",
            },
            http_session=session,
        )

        with self.assertRaisesRegex(R2PrivacyVerificationError, "policy status"):
            verifier.verify()
        session.get.assert_not_called()

    def test_missing_management_credentials_and_attestation_fails_closed(self):
        verifier = R2PrivacyVerifier(
            self._config(),
            _s3_privacy_client(),
            self.endpoint,
            environ={},
        )

        with self.assertRaisesRegex(
            R2PrivacyVerificationError,
            "cannot be proven through the S3 API",
        ):
            verifier.verify()

    def _attestation(self, now: datetime) -> dict:
        return {
            "schema_version": 1,
            "verification_method": "cloudflare_api",
            "account_id": self.account_id,
            "bucket": "private-bucket",
            "s3_endpoint_host": f"{self.account_id}.r2.cloudflarestorage.com",
            "api_checks": {
                "custom_domains": "passed",
                "managed_r2_dev": "passed",
            },
            "managed_domain": {"domain": "private.r2.dev", "enabled": False},
            "custom_domains": [],
            "checked_at": (now - timedelta(seconds=30)).isoformat(),
            "expires_at": (now + timedelta(minutes=5)).isoformat(),
        }

    def test_recent_hash_pinned_attestation_is_valid_offline_fallback(self):
        now = datetime(2026, 7, 18, 2, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "privacy.json"
            content = json.dumps(
                self._attestation(now),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            path.write_bytes(content)
            verifier = R2PrivacyVerifier(
                self._config(),
                _s3_privacy_client(),
                self.endpoint,
                environ={
                    "R2_PRIVACY_ATTESTATION_PATH": str(path),
                    "R2_PRIVACY_ATTESTATION_SHA256": sha256_bytes(content),
                },
                now=lambda: now,
            )

            result = verifier.verify()

            self.assertEqual(
                result["verification_method"],
                "hash_pinned_attestation",
            )
            self.assertEqual(result["attestation_sha256"], sha256_bytes(content))

    def test_recent_dashboard_screenshot_attestation_is_valid(self):
        now = datetime(2026, 7, 18, 2, 0, tzinfo=timezone.utc)
        value = {
            "schema_version": 1,
            "verification_method": "cloudflare_dashboard",
            "account_id": self.account_id,
            "bucket": "private-bucket",
            "s3_endpoint_host": f"{self.account_id}.r2.cloudflarestorage.com",
            "dashboard_checks": {
                "custom_domains": "passed",
                "managed_r2_dev": "passed",
            },
            "managed_domain": {
                "enabled": False,
                "state": "public_development_url_disabled",
            },
            "custom_domains": [],
            "dashboard_evidence": {
                "kind": "user_supplied_dashboard_screenshot",
                "screenshot_sha256": "a" * 64,
            },
            "checked_at": (now - timedelta(seconds=30)).isoformat(),
            "expires_at": (now + timedelta(minutes=5)).isoformat(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "privacy.json"
            content = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            path.write_bytes(content)
            verifier = R2PrivacyVerifier(
                self._config(),
                _s3_privacy_client(),
                self.endpoint,
                environ={
                    "R2_PRIVACY_ATTESTATION_PATH": str(path),
                    "R2_PRIVACY_ATTESTATION_SHA256": sha256_bytes(content),
                },
                now=lambda: now,
            )

            result = verifier.verify()

            self.assertEqual(result["verification_method"], "hash_pinned_attestation")
            self.assertEqual(result["attestation_source"], "cloudflare_dashboard")

    def test_dashboard_attestation_rejects_nonempty_custom_domains(self):
        now = datetime(2026, 7, 18, 2, 0, tzinfo=timezone.utc)
        value = {
            "schema_version": 1,
            "verification_method": "cloudflare_dashboard",
            "account_id": self.account_id,
            "bucket": "private-bucket",
            "s3_endpoint_host": f"{self.account_id}.r2.cloudflarestorage.com",
            "dashboard_checks": {
                "custom_domains": "passed",
                "managed_r2_dev": "passed",
            },
            "managed_domain": {
                "enabled": False,
                "state": "public_development_url_disabled",
            },
            "custom_domains": [{"domain": "public.example.com", "enabled": True}],
            "dashboard_evidence": {
                "kind": "user_supplied_dashboard_screenshot",
                "screenshot_sha256": "a" * 64,
            },
            "checked_at": now.isoformat(),
            "expires_at": (now + timedelta(minutes=5)).isoformat(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "privacy.json"
            content = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            path.write_bytes(content)
            verifier = R2PrivacyVerifier(
                self._config(),
                _s3_privacy_client(),
                self.endpoint,
                environ={
                    "R2_PRIVACY_ATTESTATION_PATH": str(path),
                    "R2_PRIVACY_ATTESTATION_SHA256": sha256_bytes(content),
                },
                now=lambda: now,
            )

            with self.assertRaisesRegex(
                R2PrivacyVerificationError, "dashboard attestation"
            ):
                verifier.verify()

    def test_attestation_tamper_and_staleness_are_rejected(self):
        now = datetime(2026, 7, 18, 2, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "privacy.json"
            value = self._attestation(now)
            content = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            path.write_bytes(content + b" ")
            verifier = R2PrivacyVerifier(
                self._config(),
                _s3_privacy_client(),
                self.endpoint,
                environ={
                    "R2_PRIVACY_ATTESTATION_PATH": str(path),
                    "R2_PRIVACY_ATTESTATION_SHA256": sha256_bytes(content),
                },
                now=lambda: now,
            )
            with self.assertRaisesRegex(R2PrivacyVerificationError, "hash mismatch"):
                verifier.verify()

            value["checked_at"] = (now - timedelta(hours=1)).isoformat()
            value["expires_at"] = (now + timedelta(minutes=1)).isoformat()
            stale = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            path.write_bytes(stale)
            verifier = R2PrivacyVerifier(
                self._config(),
                _s3_privacy_client(),
                self.endpoint,
                environ={
                    "R2_PRIVACY_ATTESTATION_PATH": str(path),
                    "R2_PRIVACY_ATTESTATION_SHA256": sha256_bytes(stale),
                },
                now=lambda: now,
            )
            with self.assertRaisesRegex(R2PrivacyVerificationError, "stale"):
                verifier.verify()

    def test_r2_object_store_blocks_before_put_and_caches_success(self):
        client = SimpleNamespace(put_object=Mock(return_value={"ETag": '"etag"'}))
        privacy = Mock()
        privacy.verify.return_value = {
            "status": "verified_private",
            "verification_method": "cloudflare_api",
            "managed_r2_dev_enabled": False,
            "enabled_custom_domain_count": 0,
            "s3_checks": {"head_bucket": "passed"},
        }
        store = object.__new__(R2ObjectStore)
        store.bucket = "private-bucket"
        store.prefix = "prefix"
        store.client = client
        store._privacy_verifier = privacy
        store._privacy_lock = threading.Lock()
        store._privacy_verification = None

        store.put("one", b"1")
        store.put("two", b"2")

        privacy.verify.assert_called_once_with()
        self.assertEqual(client.put_object.call_count, 2)

        blocked_client = SimpleNamespace(put_object=Mock())
        blocked = object.__new__(R2ObjectStore)
        blocked.bucket = "private-bucket"
        blocked.prefix = "prefix"
        blocked.client = blocked_client
        blocked._privacy_verifier = SimpleNamespace(
            verify=Mock(side_effect=R2PrivacyVerificationError("not proven"))
        )
        blocked._privacy_lock = threading.Lock()
        blocked._privacy_verification = None
        with self.assertRaisesRegex(R2PrivacyVerificationError, "not proven"):
            blocked.put("blocked", b"data")
        blocked_client.put_object.assert_not_called()


class PublishAndVerifySafetyTest(unittest.TestCase):
    def test_terminal_tail_registry_allows_only_exact_nbl_identity_gap(self):
        repository, release, _metadata = _terminal_tail_registry_fixture()
        expected = next(
            iter(
                publish_script.TRUSTED_TERMINAL_PRICE_TAIL_SNAPSHOT_IDENTITY_GAPS.values()
            )
        )["fingerprint"]

        self.assertEqual(
            publish_script._terminal_tail_identity_gap_fingerprints(
                repository, release
            ),
            (expected,),
        )

        report = SimpleNamespace(raise_for_errors=Mock())
        with patch.object(
            publish_script,
            "validate_repository_snapshot",
            return_value=report,
        ) as validate:
            result, fingerprints = publish_script._validate_cross_dataset_snapshot(
                repository, release
            )
        self.assertIs(result, report)
        self.assertEqual(fingerprints, (expected,))
        validate.assert_called_once_with(
            repository,
            allowed_index_identity_gap_fingerprints=(expected,),
        )
        report.raise_for_errors.assert_called_once_with()

    def test_terminal_tail_identity_gap_registry_mutations_fail_closed(self):
        variants = ("partial", "registry", "inventory", "fingerprint")
        for variant in variants:
            with self.subTest(variant=variant):
                repository, release, metadata = _terminal_tail_registry_fixture()
                context = patch.object(
                    publish_script,
                    "TRUSTED_TERMINAL_PRICE_TAIL_SNAPSHOT_IDENTITY_GAPS",
                    publish_script.TRUSTED_TERMINAL_PRICE_TAIL_SNAPSHOT_IDENTITY_GAPS,
                )
                if variant == "partial":
                    metadata["symbol_history"] = {}
                    message = "partially installed"
                elif variant == "registry":
                    metadata["security_master"]["terminal_tail_registry_draft"][0][
                        "security_id"
                    ] = "FORGED"
                    message = "not code-pinned"
                elif variant == "inventory":
                    metadata["corporate_actions"][
                        "terminal_tail_registry_inventory_sha256"
                    ] = "f" * 64
                    message = "not code-pinned"
                else:
                    changed = copy.deepcopy(
                        publish_script.TRUSTED_TERMINAL_PRICE_TAIL_SNAPSHOT_IDENTITY_GAPS
                    )
                    changed[next(iter(changed))]["fingerprint"] = "f" * 64
                    context = patch.object(
                        publish_script,
                        "TRUSTED_TERMINAL_PRICE_TAIL_SNAPSHOT_IDENTITY_GAPS",
                        changed,
                    )
                    message = "fingerprint changed"
                with context, self.assertRaisesRegex(RuntimeError, message):
                    publish_script._terminal_tail_identity_gap_fingerprints(
                        repository, release
                    )

    def test_release_without_terminal_tail_registry_gets_no_identity_gap_allowance(self):
        repository, release, metadata = _terminal_tail_registry_fixture()
        for dataset in metadata:
            metadata[dataset] = {}
        self.assertEqual(
            publish_script._terminal_tail_identity_gap_fingerprints(
                repository, release
            ),
            (),
        )

    def test_r2_privacy_failure_is_last_gate_before_any_write(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, _release = _repository_with_release(Path(directory))
            store = SimpleNamespace(
                verify_private_access=Mock(
                    side_effect=R2PrivacyVerificationError("private state not proven")
                ),
                get=Mock(),
                put=Mock(),
                list=Mock(),
            )
            stats = {"snapshot_sha256": "snapshot", "state_sha256": "state"}
            with (
                patch.object(
                    publish_script,
                    "_validate_release_lifecycle_coverage",
                    return_value={"open_count": 0},
                ),
                patch.object(
                    publish_script,
                    "validate_cross_validation_gate",
                    return_value={"price_unresolved_count": 0},
                ),
                patch.object(
                    publish_script,
                    "_validate_terminal_transition_readiness",
                    return_value={"ready": True, "issue_count": 0},
                ),
                patch.object(
                    publish_script,
                    "validate_release_snapshot",
                    return_value=dict(stats),
                ),
                patch.object(
                    publish_script,
                    "_fingerprint_release_state",
                    return_value=dict(stats),
                ),
                patch.object(publish_script, "publish_repository") as publish,
                self.assertRaisesRegex(R2PrivacyVerificationError, "not proven"),
            ):
                publish_script.publish_and_verify(
                    repository,
                    store,
                    verify_only=False,
                    keep_verify_cache=False,
                )

            store.verify_private_access.assert_called_once_with()
            publish.assert_not_called()
            store.get.assert_not_called()
            store.put.assert_not_called()
            store.list.assert_not_called()

    def test_same_version_release_with_changed_raw_bytes_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, release = _repository_with_release(Path(directory))
            changed_raw = json.dumps(
                json.loads(release.to_bytes()),
                separators=(",", ":"),
            ).encode()
            self.assertNotEqual(changed_raw, release.to_bytes())
            repository.objects.put("releases/current.json", changed_raw)
            repository.objects.put(
                f"releases/{release.version}.json",
                changed_raw,
            )

            with self.assertRaisesRegex(RuntimeError, "release bytes changed"):
                publish_script._fingerprint_release_state(repository, release)

    def test_state_fingerprint_changes_when_same_version_pointer_bytes_change(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, release = _repository_with_release(Path(directory))
            before = publish_script._fingerprint_release_state(repository, release)

            pointer_bytes = repository.objects.get(
                repository.current_key("security_master")
            ).data
            pointer = CurrentPointer.from_bytes(pointer_bytes)
            repository.objects.put(
                repository.current_key("security_master"),
                replace(pointer, updated_at="2026-07-18T01:00:00Z").to_bytes(),
            )

            after = publish_script._fingerprint_release_state(repository, release)
            self.assertEqual(before["snapshot_sha256"], after["snapshot_sha256"])
            self.assertNotEqual(before["state_sha256"], after["state_sha256"])

    def test_pointer_change_after_validation_blocks_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, release = _repository_with_release(Path(directory))
            initial = publish_script._fingerprint_release_state(repository, release)

            def validate_then_mutate(_repository, _release):
                pointer_bytes = repository.objects.get(
                    repository.current_key("security_master")
                ).data
                pointer = CurrentPointer.from_bytes(pointer_bytes)
                repository.objects.put(
                    repository.current_key("security_master"),
                    replace(pointer, updated_at="2026-07-18T02:00:00Z").to_bytes(),
                )
                return initial

            with (
                patch.object(
                    publish_script,
                    "_validate_release_lifecycle_coverage",
                    return_value={"open_count": 0},
                ),
                patch.object(
                    publish_script,
                    "validate_cross_validation_gate",
                    return_value={"price_unresolved_count": 0},
                ),
                patch.object(
                    publish_script,
                    "_validate_terminal_transition_readiness",
                    return_value={"ready": True, "issue_count": 0},
                ),
                patch.object(
                    publish_script,
                    "validate_release_snapshot",
                    side_effect=validate_then_mutate,
                ),
                patch.object(publish_script, "publish_repository") as publish,
            ):
                with self.assertRaisesRegex(RuntimeError, "current pointers changed"):
                    publish_script.publish_and_verify(
                        repository,
                        object(),
                        verify_only=False,
                        keep_verify_cache=False,
                    )
            publish.assert_not_called()

    def test_terminal_transition_risk_stops_before_any_remote_call(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, _release = _repository_with_release(Path(directory))
            store = SimpleNamespace(
                verify_private_access=Mock(
                    side_effect=AssertionError("remote privacy check called")
                ),
                get=Mock(side_effect=AssertionError("remote get called")),
                put=Mock(side_effect=AssertionError("remote put called")),
                list=Mock(side_effect=AssertionError("remote list called")),
            )

            with (
                patch.object(
                    publish_script,
                    "_validate_release_lifecycle_coverage",
                    return_value={"open_count": 0},
                ),
                patch.object(
                    publish_script,
                    "validate_cross_validation_gate",
                    return_value={"price_unresolved_count": 0},
                ),
                patch.object(
                    publish_script,
                    "_validate_terminal_transition_readiness",
                    side_effect=RuntimeError(
                        "Terminal-transition readiness is blocked: WFM"
                    ),
                ),
                patch.object(publish_script, "publish_repository") as publish,
                self.assertRaisesRegex(
                    RuntimeError, "Terminal-transition readiness is blocked"
                ),
            ):
                publish_script.publish_and_verify(
                    repository,
                    store,
                    verify_only=False,
                    keep_verify_cache=False,
                )

            publish.assert_not_called()
            store.verify_private_access.assert_not_called()
            store.get.assert_not_called()
            store.put.assert_not_called()
            store.list.assert_not_called()

    def test_missing_lifecycle_release_stops_before_any_remote_call(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, _release = _repository_with_release(Path(directory))
            store = SimpleNamespace(
                get=Mock(side_effect=AssertionError("remote get called")),
                put=Mock(side_effect=AssertionError("remote put called")),
                list=Mock(side_effect=AssertionError("remote list called")),
            )

            with patch.object(publish_script, "publish_repository") as publish:
                with self.assertRaisesRegex(RuntimeError, "lifecycle_resolutions"):
                    publish_script.publish_and_verify(
                        repository,
                        store,
                        verify_only=False,
                        keep_verify_cache=False,
                    )

            publish.assert_not_called()
            store.get.assert_not_called()
            store.put.assert_not_called()
            store.list.assert_not_called()

    def test_missing_cross_validation_stops_before_any_remote_call(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, _release = _repository_with_release(Path(directory))
            store = SimpleNamespace(
                get=Mock(side_effect=AssertionError("remote get called")),
                put=Mock(side_effect=AssertionError("remote put called")),
                list=Mock(side_effect=AssertionError("remote list called")),
            )

            with (
                patch.object(
                    publish_script,
                    "_validate_release_lifecycle_coverage",
                    return_value={"open_count": 0},
                ),
                patch.object(publish_script, "publish_repository") as publish,
            ):
                with self.assertRaisesRegex(RuntimeError, "cross_validation_reports"):
                    publish_script.publish_and_verify(
                        repository,
                        store,
                        verify_only=False,
                        keep_verify_cache=False,
                    )

            publish.assert_not_called()
            store.get.assert_not_called()
            store.put.assert_not_called()
            store.list.assert_not_called()

    def test_lifecycle_gate_checks_manifest_metadata_and_evidence_archive(self):
        repository, release, candidate, metadata, frames = _lifecycle_gate_fixture()
        target = publish_script._build_release_lifecycle_candidates

        with patch.object(publish_script, target.__name__, return_value=(candidate,)):
            result = publish_script._validate_release_lifecycle_coverage(
                repository,
                release,
            )
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["open_count"], 0)
        self.assertEqual(
            result["evidence_report_sha256"],
            "evidence-report-sha256",
        )

        metadata["candidate_count"] = 2
        with (
            patch.object(publish_script, target.__name__, return_value=(candidate,)),
            self.assertRaisesRegex(RuntimeError, "candidate_count"),
        ):
            publish_script._validate_release_lifecycle_coverage(repository, release)

        metadata["candidate_count"] = 1
        frames["source_archive"] = pd.DataFrame([{"archive_id": "other"}])
        with (
            patch.object(publish_script, target.__name__, return_value=(candidate,)),
            self.assertRaisesRegex(RuntimeError, "source_archive.archive_id"),
        ):
            publish_script._validate_release_lifecycle_coverage(repository, release)

    def test_publication_candidate_builder_keeps_bound_official_applied_event(self):
        indexed = LifecycleCandidate(
            security_id="SEC-INDEXED",
            symbol="IDX",
            name="Indexed Security",
            exchange="NYSE",
            last_price_date="2024-01-02",
            active_to="",
        )
        bound_security_id = "SEC-BOUND"
        spec = OfficialLifecycleExceptionEvidenceSpec(
            evidence_id="bound_applied_event",
            candidate_symbols=("BND",),
            candidate_name_contains=("Bound Security",),
            candidate_security_ids=(bound_security_id,),
            candidate_last_price_dates=("2023-11-13",),
            binding_status="bound",
            effective_date="2023-11-14",
            filing_date="2023-11-14",
            resolution_kind="applied_event",
            exception_code="",
            action_type="delisting",
            cash_amount=None,
            claim="Reviewed terminal cancellation.",
            source_url="https://www.sec.gov/Archives/bound.htm",
            source_sha256="a" * 64,
            required_text_groups=(("cancelled",),),
        )
        frames = {
            "security_master": pd.DataFrame(
                [
                    {
                        "security_id": bound_security_id,
                        "name": "Bound Security Inc.",
                        "exchange": "NYSE",
                        "active_to": "2023-11-13",
                    }
                ]
            ),
            "symbol_history": pd.DataFrame(
                [
                    {
                        "security_id": bound_security_id,
                        "symbol": "BND",
                        "effective_from": "2022-01-01",
                        "effective_to": "2023-11-13",
                    }
                ]
            ),
            "daily_price_raw": pd.DataFrame(
                [
                    {
                        "security_id": bound_security_id,
                        "session": "2023-11-13",
                    }
                ]
            ),
        }

        class Repository:
            def read_frame(self, dataset, _version):
                return frames[dataset].copy()

        release = SimpleNamespace(
            dataset_versions={dataset: "v1" for dataset in frames}
        )
        with (
            patch(
                "supertrend_quant.market_store.lifecycle.build_lifecycle_candidates",
                return_value=(indexed,),
            ),
            patch(
                "supertrend_quant.market_store.official_lifecycle_evidence."
                "load_official_lifecycle_exception_evidence",
                return_value={spec.evidence_id: spec},
            ),
        ):
            result = publish_script._build_release_lifecycle_candidates(
                Repository(), release
            )

        self.assertEqual(
            {candidate.security_id for candidate in result},
            {indexed.security_id, bound_security_id},
        )

    def test_temporary_lifecycle_exception_blocks_r2_publication(self):
        repository, release, candidate, _metadata, frames = _lifecycle_gate_fixture()
        frames["lifecycle_resolutions"].loc[0, "recheck_after"] = "2026-10-31"

        with (
            patch.object(
                publish_script,
                "_build_release_lifecycle_candidates",
                return_value=(candidate,),
            ),
            self.assertRaisesRegex(RuntimeError, "temporary exceptions must be zero"),
        ):
            publish_script._validate_release_lifecycle_coverage(repository, release)

    def test_remote_parent_manifest_must_match_local_raw_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, release = _repository_with_release(
                root / "local",
                inherited=True,
            )
            remote = LocalObjectStore(root / "remote")
            remote.put("releases/current.json", release.to_bytes())
            remote.put(f"releases/{release.version}.json", release.to_bytes())
            pointer_path = repository.current_key("security_master")
            remote.put(pointer_path, repository.objects.get(pointer_path).data)

            chain = repository.manifest_chain("security_master", "child-v1")
            for manifest in chain:
                path = (
                    f"{repository.version_prefix('security_master', manifest.version)}"
                    "/manifest.json"
                )
                raw = repository.objects.get(path).data
                if manifest.version == "parent-v1":
                    raw = json.dumps(json.loads(raw), separators=(",", ":")).encode()
                remote.put(path, raw)

            with self.assertRaisesRegex(RuntimeError, "raw manifest mismatch"):
                publish_script.validate_remote_release(remote, repository, release)


class SourceArchivePayloadVerificationTest(unittest.TestCase):
    def test_distinct_provenance_rows_can_share_one_content_addressed_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = b"[]"
            source_hash = sha256_bytes(raw)
            object_path = f"archives/2026-07-15/{source_hash}.json.gz"
            path = root / object_path
            path.parent.mkdir(parents=True)
            path.write_bytes(gzip.compress(raw, mtime=0))
            reviewed = [
                row
                for row in reviewed_provenance_archive_ids()
                if row.source_hash == source_hash
            ]
            self.assertEqual(len(reviewed), 5)
            frame = pd.DataFrame(
                [
                    {
                        "archive_id": row.archive_id,
                        "object_path": object_path,
                        "source": row.source,
                        "source_hash": row.source_hash,
                        "source_url": row.source_url,
                    }
                    for row in reviewed[:2]
                ]
            )
            repository = SimpleNamespace(
                root=root,
                read_frame=Mock(return_value=frame),
            )
            release = SimpleNamespace(
                dataset_versions={"source_archive": "archive-v1"}
            )

            result = publish_script._verify_archive_payloads(repository, release)

            self.assertEqual(result["payloads"], 2)
            self.assertEqual(result["raw_bytes"], 4)

    def test_reviewed_provenance_inventory_is_exact_and_code_pinned(self):
        rows = reviewed_provenance_archive_ids()

        self.assertEqual(len(rows), 15)
        payload = json.dumps(
            [
                {
                    "archive_id": row.archive_id,
                    "source": row.source,
                    "source_url": row.source_url,
                    "source_hash": row.source_hash,
                }
                for row in rows
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self.assertEqual(
            sha256_bytes(payload),
            TRUSTED_PROVENANCE_ARCHIVE_ID_INVENTORY_SHA256,
        )
        for row in rows:
            self.assertEqual(
                validate_source_archive_id(
                    row.archive_id,
                    source=row.source,
                    source_url=row.source_url,
                    source_hash=row.source_hash,
                ),
                row.archive_id,
            )

    def test_unregistered_formula_qualified_id_is_rejected(self):
        source = "unreviewed_source"
        source_url = "https://example.test/unreviewed"
        source_hash = sha256_bytes(b"unreviewed")
        archive_id = hashlib.sha256(
            f"{source}|{source_url}|{source_hash}".encode()
        ).hexdigest()

        with self.assertRaisesRegex(ValueError, "code-reviewed"):
            validate_source_archive_id(
                archive_id,
                source=source,
                source_url=source_url,
                source_hash=source_hash,
            )

    def test_reviewed_qualified_id_rejects_url_drift_and_alternate_formula(self):
        row = next(
            item
            for item in reviewed_provenance_archive_ids()
            if item.source == "ovintiv_issuer_reorganization"
        )
        alternate_id = hashlib.sha256(
            f"{row.source}|{row.source_hash}".encode()
        ).hexdigest()

        with self.assertRaisesRegex(ValueError, "code-reviewed"):
            validate_source_archive_id(
                row.archive_id,
                source=row.source,
                source_url=row.source_url + "#changed",
                source_hash=row.source_hash,
            )
        with self.assertRaisesRegex(ValueError, "code-reviewed"):
            validate_source_archive_id(
                alternate_id,
                source=row.source,
                source_url=row.source_url,
                source_hash=row.source_hash,
            )

    def test_noncanonical_hashes_and_outer_whitespace_are_rejected(self):
        source_hash = sha256_bytes(b"canonical")
        cases = (
            {
                "archive_id": source_hash.upper(),
                "source": "source",
                "source_url": "https://example.test/raw",
                "source_hash": source_hash,
            },
            {
                "archive_id": source_hash,
                "source": " source",
                "source_url": "https://example.test/raw",
                "source_hash": source_hash,
            },
            {
                "archive_id": source_hash,
                "source": "source",
                "source_url": "https://example.test/raw ",
                "source_hash": source_hash,
            },
            {
                "archive_id": source_hash,
                "source": "source",
                "source_url": "https://example.test/raw",
                "source_hash": source_hash + " ",
            },
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_source_archive_id(**values)

    def test_content_id_accepts_storage_null_as_absent_optional_url(self):
        source_hash = sha256_bytes(b"no source URL")

        self.assertEqual(
            validate_source_archive_id(
                source_hash,
                source="local_evidence",
                source_url=float("nan"),
                source_hash=source_hash,
            ),
            source_hash,
        )

    def test_duplicate_provenance_tuple_is_rejected_before_payload_access(self):
        row = reviewed_provenance_archive_ids()[0]
        frame = pd.DataFrame(
            [
                {
                    "archive_id": row.source_hash,
                    "object_path": "archives/missing.bin",
                    "source": row.source,
                    "source_hash": row.source_hash,
                    "source_url": row.source_url,
                },
                {
                    "archive_id": row.archive_id,
                    "object_path": "archives/missing.bin",
                    "source": row.source,
                    "source_hash": row.source_hash,
                    "source_url": row.source_url,
                },
            ]
        )
        repository = SimpleNamespace(
            root=Path("/not-accessed"),
            read_frame=Mock(return_value=frame),
        )
        release = SimpleNamespace(
            dataset_versions={"source_archive": "archive-v1"}
        )

        with self.assertRaisesRegex(RuntimeError, "duplicate provenance tuples"):
            publish_script._verify_archive_payloads(repository, release)
        repository.read_frame.assert_called_once_with(
            "source_archive", "archive-v1"
        )

    def test_arbitrary_archive_row_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = b"official raw"
            source_hash = sha256_bytes(raw)
            object_path = f"archives/2026-07-15/{source_hash}.txt.gz"
            path = root / object_path
            path.parent.mkdir(parents=True)
            path.write_bytes(gzip.compress(raw, mtime=0))
            frame = pd.DataFrame(
                [
                    {
                        "archive_id": "0" * 64,
                        "object_path": object_path,
                        "source": "sec_edgar_filing",
                        "source_hash": source_hash,
                        "source_url": "https://www.sec.gov/Archives/fixture.txt",
                    }
                ]
            )
            repository = SimpleNamespace(
                root=root,
                read_frame=Mock(return_value=frame),
            )
            release = SimpleNamespace(
                dataset_versions={"source_archive": "archive-v1"}
            )

            with self.assertRaisesRegex(RuntimeError, "row identity mismatch"):
                publish_script._verify_archive_payloads(repository, release)


class SourceArchivePublishObjectDeduplicationTest(unittest.TestCase):
    def test_current_source_archive_exposes_each_object_path_once(self):
        frame = pd.DataFrame(
            [
                {"archive_id": "a", "object_path": "archives/shared.json.gz"},
                {"archive_id": "b", "object_path": "archives/shared.json.gz"},
                {"archive_id": "c", "object_path": "archives/unique.json.gz"},
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            repository = publish_script._ArchiveObjectPathPublishRepository(
                directory
            )
            with patch.object(
                LocalDatasetRepository,
                "read_frame",
                return_value=frame,
            ) as read_frame:
                current = repository.read_frame("source_archive")
                immutable = repository.read_frame("source_archive", "archive-v1")

        self.assertEqual(
            list(current["object_path"]),
            ["archives/shared.json.gz", "archives/unique.json.gz"],
        )
        self.assertEqual(len(immutable), 3)
        self.assertEqual(
            read_frame.call_args_list,
            [
                call("source_archive", None),
                call("source_archive", "archive-v1"),
            ],
        )


class DatasetCacheArchivePathSafetyTest(unittest.TestCase):
    def _manifest_with_object_path(
        self,
        root: Path,
        object_path: str,
    ) -> DatasetManifest:
        version_root = root / "datasets/source_archive/versions/v1"
        parquet_path = version_root / "part.parquet"
        parquet_path.parent.mkdir(parents=True)
        pd.DataFrame([{"object_path": object_path}]).to_parquet(
            parquet_path,
            index=False,
        )
        payload = parquet_path.read_bytes()
        return DatasetManifest.create(
            "source_archive",
            "v1",
            "2026-07-17",
            (
                ManifestFile(
                    path="part.parquet",
                    sha256=sha256_bytes(payload),
                    size_bytes=len(payload),
                    row_count=1,
                ),
            ),
        )

    def test_archive_path_escape_is_rejected_before_remote_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            manifest = self._manifest_with_object_path(root, "../escaped.bin")

            class RecordingStore:
                def __init__(self):
                    self.reads: list[str] = []

                def get(self, key: str):
                    self.reads.append(key)
                    return SimpleNamespace(data=b"must not be read")

            store = RecordingStore()
            with self.assertRaisesRegex(ValueError, "escapes cache root"):
                DatasetCache(root, store)._sync_archive_payloads(manifest)

            self.assertEqual(store.reads, [])
            self.assertFalse((root.parent / "escaped.bin").exists())

    def test_safe_archive_path_is_downloaded_inside_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            object_path = "archives/2026-07-17/archive.json.gz"
            manifest = self._manifest_with_object_path(root, object_path)

            class FixtureStore:
                def get(self, key: str):
                    self.last_key = key
                    return SimpleNamespace(data=b"archive")

            store = FixtureStore()
            DatasetCache(root, store)._sync_archive_payloads(manifest)

            self.assertEqual(store.last_key, object_path)
            self.assertEqual((root / object_path).read_bytes(), b"archive")


if __name__ == "__main__":
    unittest.main()
