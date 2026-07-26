from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .adjustments import build_adjustment_factors
from .ingest import SourceArtifact
from .kr_providers import (
    KR_BENCHMARK_SECURITIES,
    KR_INDEX_DEFINITIONS,
    KRX_SOURCE_URL,
    KrDartDividendResult,
    KrIdentityCatalog,
    KrProviderResult,
    canonical_json_bytes,
    fetch_eodhd_actions,
    fetch_eodhd_prices,
    fetch_kis_prices,
    fetch_opendart_dividend_decisions,
    fetch_kr_identity_catalog,
    fetch_krx_membership,
    fetch_krx_session_prices,
    fetch_naver_prices,
    fetch_yahoo_prices,
    is_valid_kr_symbol,
    krx_price_evidence_by_symbol,
    normalize_kr_symbol,
    restore_kr_identity_catalog_from_evidence,
    validate_krx_official_configuration,
)
from .lifecycle import build_lifecycle_candidates
from .lifecycle_coverage import (
    LifecycleExceptionCode,
    TEMPORARY_EXCEPTION_CODES,
    lifecycle_candidate_id,
    validate_lifecycle_coverage,
)
from .manifest import DataRelease, sha256_bytes, utc_now_iso, write_atomic
from .markets import expected_completed_session, market_spec, recent_sessions, release_metadata
from .models import DataQuality
from .repository import LocalDatasetRepository
from .schemas import DATASET_SPECS, dataset_spec
from .validation import (
    INDEX_PRICE_GAP_POLICY_SCHEMA,
    index_member_price_gap_records,
    index_price_gap_policy_sha256,
    validate_dataset,
    validate_index_price_gap_policy,
    validate_manifest_files,
    validate_repository_snapshot,
)


KR_PROFILES = ("kospi200", "kosdaq150")
KR_MEMBERSHIP_CHECKPOINT_SCHEMA_VERSION = 2
KR_PRICE_CHECKPOINT_SCHEMA_VERSION = 4
KR_MIGRATABLE_PRICE_CHECKPOINT_SCHEMA_VERSIONS = frozenset({2, 3, 4})
KR_PRICE_CHECKPOINT_SCHEMA_COLUMN = "__checkpoint_schema_version"
KR_PRICE_CHECKPOINT_SYMBOLS_COLUMN = "__selected_symbols_sha256"
KR_IDENTITY_CHECKPOINT_SCHEMA_VERSION = 2
KR_LIFECYCLE_SELECTION_RULE = "kr_terminal_v1"
KRX_TRADED_STATUS = "traded"
KRX_CLASSIFIED_NO_TRADE_STATUSES = frozenset(
    {
        "suspended_or_no_trade",
        "delisting_effective_date_no_trade",
        "no_regular_session_ohlc",
    }
)
KRX_ALLOWED_OBSERVATION_STATUSES = frozenset(
    {KRX_TRADED_STATUS, *KRX_CLASSIFIED_NO_TRADE_STATUSES}
)
KR_REQUIRED_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "corporate_actions",
    "adjustment_factors",
    "index_constituent_anchors",
    "index_membership_events",
    "lifecycle_resolutions",
    "cross_validation_reports",
    "source_archive",
)
KR_LICENSE_POLICY = {
    "krx_canonical": "allowed_private",
    "krx_raw_payload": "local_only",
    "eodhd": "allowed_private",
    "naver_raw": "local_only",
    "yahoo_raw": "local_only",
    "kis_raw": "local_only",
    "opendart_raw": "local_only",
    "benchmark_report": "allowed_private",
    "lifecycle_evidence_report": "allowed_private",
    "reference_price_audit": "allowed_private",
    "opendart_action_audit": "allowed_private",
    "official_action_evidence": "allowed_private",
}
KR_BENCHMARK_RANKING_WEIGHTS = {
    "accuracy": 0.35,
    "coverage": 0.20,
    "corporate_action": 0.15,
    "survivorship": 0.15,
    "revision_stability": 0.10,
    "operating_cost": 0.05,
}
KRX_EX_DIVIDEND_RULE_URL = (
    "https://global.krx.co.kr/contents/GLB/06/0602/0602010204/"
    "GLB0602010204T1.jsp"
)
KRX_EX_DIVIDEND_RULE = "krx_t_plus_2_second_last_session/v1"


def _kr_price_symbol_inventory_hash(symbols: Iterable[str]) -> str:
    normalized = sorted(
        {normalize_kr_symbol(symbol) for symbol in symbols}
    )
    return sha256_bytes(canonical_json_bytes(normalized))
# A transparent relative-cost heuristic used only after every hard gate passes.
# Free public endpoints score highest; paid/authenticated quotas score lower.
KR_PROVIDER_OPERATING_COST_SCORES = {
    "krx": 0.50,
    "eodhd": 0.25,
    "naver": 1.00,
    "yahoo": 1.00,
    "kis": 0.75,
}
KR_HISTORY_VERIFICATION_SCHEMA = "kr_history_verification/v1"
KR_OFFICIAL_MEMBERSHIP_SOURCES = frozenset(
    {
        "krx_authenticated_web_index_constituents",
        "krx_licensed_index_constituents",
    }
)


@dataclass(frozen=True)
class _ArtifactHashReference:
    source_hash: str


@dataclass(frozen=True)
class KrBenchmarkThresholds:
    identity_mapping_rate: float = 1.0
    expected_session_coverage: float = 1.0
    close_within_one_tick_rate: float = 1.0
    ohlc_within_one_tick_rate: float = 1.0
    volume_exact_rate: float = 1.0
    max_duplicate_keys: int = 0
    max_invalid_ohlc: int = 0
    max_unclassified_missing: int = 0
    max_unexpected_provider_observations: int = 0
    max_misclassified_no_trade: int = 0
    max_unexplained_large_discontinuities: int = 0
    max_large_cross_source_discrepancies: int = 0
    large_return_threshold: float = 0.35


@dataclass(frozen=True)
class KrBenchmarkOutcome:
    report_path: str
    status: str
    start_session: str
    end_session: str
    session_count: int
    symbol_count: int
    primary_provider: str
    secondary_provider: str
    providers: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class KrBootstrapResult:
    completed_session: str
    release_version: str
    row_counts: dict[str, int]
    warnings: tuple[str, ...]
    benchmark_report_sha256: str
    primary_provider: str
    secondary_provider: str


class KrCheckpointStore:
    """Small resumable files kept outside immutable release datasets."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def membership_path(self, profile: str, session: str) -> Path:
        return self.root / "memberships" / profile / f"{session}.json"

    def price_path(self, session: str) -> Path:
        return self.root / "prices" / f"{session}.parquet"

    def identity_catalog_path(self) -> Path:
        return self.root / "identity_catalog.parquet"

    def identity_catalog_metadata_path(self) -> Path:
        return self.root / "identity_catalog.json"

    def load_membership(self, profile: str, session: str) -> dict[str, Any] | None:
        path = self.membership_path(profile, session)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema_version")
            != KR_MEMBERSHIP_CHECKPOINT_SCHEMA_VERSION
        ):
            return None
        return dict(value)

    def save_membership(
        self,
        profile: str,
        session: str,
        symbols: tuple[str, ...],
        artifact,
    ) -> dict[str, Any]:
        payload = {
            "schema_version": KR_MEMBERSHIP_CHECKPOINT_SCHEMA_VERSION,
            "profile": profile,
            "session": session,
            "symbols": list(symbols),
            "source": artifact.source,
            "source_url": artifact.source_url,
            "retrieved_at": artifact.retrieved_at,
            "source_hash": artifact.source_hash,
        }
        write_atomic(
            self.membership_path(profile, session),
            (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
        )
        self.save_local_artifact(artifact, scope="krx-membership")
        return payload

    def load_prices(self, session: str) -> pd.DataFrame | None:
        path = self.price_path(session)
        if not path.is_file():
            return None
        frame = pd.read_parquet(path)
        if KR_PRICE_CHECKPOINT_SCHEMA_COLUMN not in frame:
            return None
        versions = set(
            frame[KR_PRICE_CHECKPOINT_SCHEMA_COLUMN]
            .fillna(-1)
            .astype(int)
            .unique()
        )
        if (
            len(versions) != 1
            or not versions <= KR_MIGRATABLE_PRICE_CHECKPOINT_SCHEMA_VERSIONS
        ):
            return None
        inventory_hash = ""
        if KR_PRICE_CHECKPOINT_SYMBOLS_COLUMN in frame:
            values = (
                frame[KR_PRICE_CHECKPOINT_SYMBOLS_COLUMN]
                .fillna("")
                .astype(str)
                .drop_duplicates()
            )
            if len(values) == 1 and _valid_sha256(values.iloc[0]):
                inventory_hash = str(values.iloc[0])
        output = frame.drop(
            columns=[
                KR_PRICE_CHECKPOINT_SCHEMA_COLUMN,
                KR_PRICE_CHECKPOINT_SYMBOLS_COLUMN,
            ],
            errors="ignore",
        )
        output.attrs["selected_symbols_sha256"] = inventory_hash
        return output

    def save_prices(
        self,
        session: str,
        frame: pd.DataFrame,
        artifact,
        *,
        selected_symbols: Iterable[str],
    ) -> None:
        self.rewrite_prices(
            session,
            frame,
            selected_symbols=selected_symbols,
        )
        self.save_local_artifact(artifact, scope="krx-prices")

    def rewrite_prices(
        self,
        session: str,
        frame: pd.DataFrame,
        *,
        selected_symbols: Iterable[str],
    ) -> None:
        path = self.price_path(session)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_frame = frame.copy()
        checkpoint_frame[KR_PRICE_CHECKPOINT_SCHEMA_COLUMN] = (
            KR_PRICE_CHECKPOINT_SCHEMA_VERSION
        )
        inventory_hash = _kr_price_symbol_inventory_hash(selected_symbols)
        checkpoint_frame[KR_PRICE_CHECKPOINT_SYMBOLS_COLUMN] = inventory_hash
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        checkpoint_frame.to_parquet(
            temporary,
            index=False,
            compression="zstd",
        )
        os.replace(temporary, path)
        frame.attrs["selected_symbols_sha256"] = inventory_hash

    def load_identity_catalog(
        self,
        *,
        start: str,
        end: str,
    ) -> KrIdentityCatalog | None:
        path = self.identity_catalog_path()
        metadata_path = self.identity_catalog_metadata_path()
        if not path.is_file() or not metadata_path.is_file():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("schema_version")
            != KR_IDENTITY_CHECKPOINT_SCHEMA_VERSION
            or str(metadata.get("start")) != str(start)
            or str(metadata.get("end")) != str(end)
        ):
            return None
        frame = pd.read_parquet(path)
        return KrIdentityCatalog(
            frame,
            (),
            str(metadata.get("dart_status") or "checkpoint"),
        )

    def save_identity_catalog(
        self,
        catalog: KrIdentityCatalog,
        *,
        start: str,
        end: str,
    ) -> None:
        path = self.identity_catalog_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        catalog.frame.to_parquet(path, index=False, compression="zstd")
        metadata = {
            "schema_version": KR_IDENTITY_CHECKPOINT_SCHEMA_VERSION,
            "start": str(start),
            "end": str(end),
            "dart_status": catalog.dart_status,
            "row_count": len(catalog.frame),
        }
        write_atomic(
            self.identity_catalog_metadata_path(),
            (
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8"),
        )

    def save_provider_result(self, result: KrProviderResult) -> None:
        provider_root = self.root / "providers" / result.provider
        provider_root.mkdir(parents=True, exist_ok=True)
        artifact_hashes = sorted(
            {
                str(artifact.source_hash).strip().lower()
                for artifact in result.artifacts
                if _valid_sha256(artifact.source_hash)
            }
        )
        result.prices.to_parquet(
            provider_root / "prices.parquet", index=False, compression="zstd"
        )
        write_atomic(
            provider_root / "status.json",
            (
                json.dumps(
                    {
                        "provider": result.provider,
                        "status": result.status,
                        "detail": result.detail,
                        "row_count": len(result.prices),
                        "artifact_hashes": artifact_hashes,
                        "artifact_hash_inventory_sha256": (
                            sha256_bytes(
                                canonical_json_bytes(
                                    artifact_hashes
                                )
                            )
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode(),
        )
        for artifact in result.artifacts:
            if hasattr(artifact, "content"):
                self.save_local_artifact(artifact, scope=result.provider)

    def load_provider_prices(self, provider: str) -> pd.DataFrame | None:
        path = self.root / "providers" / provider / "prices.parquet"
        return pd.read_parquet(path) if path.is_file() else None

    def load_provider_result(self, provider: str) -> KrProviderResult | None:
        provider_root = self.root / "providers" / provider
        status_path = provider_root / "status.json"
        prices = self.load_provider_prices(provider)
        if prices is None or not status_path.is_file():
            return None
        status_payload = json.loads(status_path.read_text(encoding="utf-8"))
        evidence_root = self.root / "evidence_local" / provider
        status_artifact_hashes = status_payload.get(
            "artifact_hashes"
        )
        if isinstance(status_artifact_hashes, list):
            artifact_hashes = tuple(
                sorted(
                    {
                        str(value).strip().lower()
                        for value in status_artifact_hashes
                        if _valid_sha256(value)
                    }
                )
            )
            if (
                len(artifact_hashes) != len(status_artifact_hashes)
                or sha256_bytes(
                    canonical_json_bytes(list(artifact_hashes))
                )
                != str(
                    status_payload.get(
                        "artifact_hash_inventory_sha256"
                    )
                    or ""
                ).lower()
            ):
                return None
            evidence_counts: dict[str, int] = {}
            if evidence_root.is_dir():
                for path in evidence_root.iterdir():
                    if not path.is_file() or not path.name.endswith(
                        ".gz"
                    ):
                        continue
                    source_hash = path.name.split(".", 1)[0].lower()
                    if _valid_sha256(source_hash):
                        evidence_counts[source_hash] = (
                            evidence_counts.get(source_hash, 0) + 1
                        )
            for source_hash in artifact_hashes:
                if evidence_counts.get(source_hash) != 1:
                    return None
        else:
            # A directory-wide fallback can silently mix stale or out-of-scope
            # evidence into a new benchmark. Force one resumable refresh so
            # the exact scoped inventory is written.
            return None
        row_hashes = {
            str(value).strip().lower()
            for value in prices.get("source_hash", pd.Series(dtype=str))
            if _valid_sha256(value)
        }
        if row_hashes and not row_hashes <= set(artifact_hashes):
            return None
        artifacts = tuple(_ArtifactHashReference(value) for value in artifact_hashes)
        return KrProviderResult(
            provider=str(status_payload.get("provider") or provider),
            status=str(status_payload.get("status") or "failed"),
            prices=prices,
            artifacts=artifacts,
            detail=str(status_payload.get("detail") or ""),
        )

    def save_local_artifact(self, artifact, *, scope: str) -> Path:
        content_type = str(artifact.content_type).lower()
        extension = "json" if "json" in content_type else "zip" if "zip" in content_type else "bin"
        path = self.root / "evidence_local" / scope / f"{artifact.source_hash}.{extension}.gz"
        if not path.is_file():
            write_atomic(path, gzip.compress(artifact.content))
        return path


def _read_configured_kr_evidence_table(
    env_name: str,
) -> tuple[pd.DataFrame, SourceArtifact | None]:
    raw_path = os.getenv(env_name, "").strip()
    if not raw_path:
        return pd.DataFrame(), None
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise RuntimeError(f"{env_name} does not exist or is not a file: {path}")
    content = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
        content_type = "application/vnd.apache.parquet"
    elif suffix == ".csv":
        frame = pd.read_csv(path, dtype=str)
        content_type = "text/csv"
    elif suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            frame = pd.read_json(path, lines=True, dtype=False)
            content_type = "application/x-ndjson"
        else:
            value = json.loads(content.decode("utf-8"))
            if isinstance(value, dict):
                preferred_key = (
                    "actions"
                    if env_name == "KR_OFFICIAL_ACTIONS_PATH"
                    else "resolutions"
                )
                value = value.get(preferred_key, value.get("rows", value))
            frame = pd.DataFrame(value)
            content_type = "application/json"
    else:
        raise RuntimeError(
            f"{env_name} must end in .csv, .json, .jsonl, or .parquet."
        )
    if frame is None:
        frame = pd.DataFrame()
    artifact = SourceArtifact(
        source=env_name.lower(),
        source_url=f"local://{path.resolve()}",
        retrieved_at=utc_now_iso(),
        content=content,
        content_type=content_type,
    )
    return frame, artifact


def _identity_catalog_for_run(
    checkpoint: KrCheckpointStore,
    *,
    start: str,
    end: str,
    include_dart: bool,
) -> KrIdentityCatalog:
    cached = checkpoint.load_identity_catalog(start=start, end=end)
    if cached is not None:
        return cached
    try:
        catalog = fetch_kr_identity_catalog(
            start=start,
            end=end,
            include_dart=include_dart,
        )
    except Exception as fetch_error:
        try:
            catalog = restore_kr_identity_catalog_from_evidence(
                checkpoint.root / "evidence_local" / "identity"
            )
        except Exception:
            raise fetch_error
    for artifact in catalog.artifacts:
        checkpoint.save_local_artifact(artifact, scope="identity")
    checkpoint.save_identity_catalog(catalog, start=start, end=end)
    return catalog


def _configured_official_action_symbols() -> set[str]:
    frame, _ = _read_configured_kr_evidence_table("KR_OFFICIAL_ACTIONS_PATH")
    if frame.empty:
        return set()
    output: set[str] = set()
    for column in ("symbol", "new_symbol"):
        if column not in frame:
            continue
        output.update(
            value
            for raw in frame[column]
            if (
                value := (
                    ""
                    if not _cell_text(raw)
                    else normalize_kr_symbol(_cell_text(raw))
                )
            )
            and is_valid_kr_symbol(value)
        )
    return output


def _load_kr_official_actions(
    catalog: KrIdentityCatalog,
    *,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, SourceArtifact | None]:
    frame, artifact = _read_configured_kr_evidence_table(
        "KR_OFFICIAL_ACTIONS_PATH"
    )
    if frame.empty:
        return _empty_dataset("corporate_actions"), artifact
    required = {"action_type", "effective_date", "source_url"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            "KR_OFFICIAL_ACTIONS_PATH is missing columns: " + ", ".join(missing)
        )
    source_hash = str(artifact.source_hash) if artifact is not None else ""
    retrieved_at = str(artifact.retrieved_at) if artifact is not None else utc_now_iso()
    known_ids = set(catalog.frame["security_id"].astype(str))
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        effective = _normalized_date(row.get("effective_date"))
        if not effective or effective < str(start) or effective > str(end):
            continue
        security_id = _import_security_id(row, catalog, effective)
        if security_id not in known_ids:
            raise RuntimeError(
                "KR official action references an identity outside the scoped catalog: "
                + security_id
            )
        source_url = _cell_text(row.get("source_url"))
        if not source_url.lower().startswith(("http://", "https://")):
            raise RuntimeError(
                "Every KR official action requires an HTTP(S) KRX/DART/KIND source_url."
            )
        new_security_id = _import_successor_security_id(row, catalog, effective)
        if new_security_id and new_security_id not in known_ids:
            raise RuntimeError(
                "KR official action successor is outside the scoped catalog: "
                + new_security_id
            )
        action_type = _cell_text(row.get("action_type")).lower()
        identity = {
            "security_id": security_id,
            "action_type": action_type,
            "effective_date": effective,
            "cash_amount": _optional_number(row.get("cash_amount")),
            "ratio": _optional_number(row.get("ratio")),
            "new_security_id": new_security_id,
            "new_symbol": normalize_kr_symbol(row.get("new_symbol"))
            if _cell_text(row.get("new_symbol"))
            else "",
            "source_url": source_url,
        }
        records.append(
            {
                "event_id": sha256_bytes(canonical_json_bytes(identity)),
                **identity,
                "ex_date": _normalized_date(row.get("ex_date")) or effective,
                "announcement_date": _normalized_date(row.get("announcement_date")),
                "record_date": _normalized_date(row.get("record_date")),
                "payment_date": _normalized_date(row.get("payment_date")),
                "currency": (_cell_text(row.get("currency")) or "KRW").upper(),
                "official": True,
                "source_kind": "official",
                "source": _cell_text(row.get("source"))
                or "krx_dart_kind_verified_import",
                "retrieved_at": retrieved_at,
                "source_hash": source_hash,
            }
        )
    return (
        pd.DataFrame(
            records,
            columns=dataset_spec("corporate_actions").required_columns,
        ).drop_duplicates("event_id", keep="last"),
        artifact,
    )


def _import_security_id(
    row: dict[str, Any],
    catalog: KrIdentityCatalog,
    session: str,
) -> str:
    raw = (
        _cell_text(row.get("security_id")) or _cell_text(row.get("isin"))
    ).upper()
    if raw:
        return raw if raw.startswith("KR:") else f"KR:{raw}"
    symbol = normalize_kr_symbol(row.get("symbol"))
    if not is_valid_kr_symbol(symbol):
        raise RuntimeError(
            "KR official action requires security_id/isin or a six-character "
            "KRX short code."
        )
    try:
        return catalog.security_id_for(symbol, session)
    except KeyError as exc:
        raise RuntimeError(str(exc)) from None


def _import_successor_security_id(
    row: dict[str, Any],
    catalog: KrIdentityCatalog,
    session: str,
) -> str:
    raw = (
        _cell_text(row.get("new_security_id"))
        or _cell_text(row.get("new_isin"))
    ).upper()
    if raw:
        return raw if raw.startswith("KR:") else f"KR:{raw}"
    raw_symbol = _cell_text(row.get("new_symbol"))
    if not raw_symbol:
        return ""
    symbol = normalize_kr_symbol(raw_symbol)
    if not is_valid_kr_symbol(symbol):
        raise RuntimeError(
            "KR official action new_symbol must be a six-character KRX short code."
        )
    try:
        return catalog.security_id_for(symbol, session)
    except KeyError as exc:
        raise RuntimeError(str(exc)) from None


def _optional_number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if pd.notna(parsed) else None


def _normalized_date(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else pd.Timestamp(parsed).date().isoformat()


def _normalized_timestamp(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _load_approved_kr_lifecycle_exceptions() -> tuple[pd.DataFrame, SourceArtifact | None]:
    frame, artifact = _read_configured_kr_evidence_table(
        "KR_LIFECYCLE_RESOLUTIONS_PATH"
    )
    if frame.empty:
        return _empty_dataset("lifecycle_resolutions"), artifact
    required = {
        "security_id",
        "last_price_date",
        "exception_code",
        "exception_reason",
        "reviewed_by",
        "reviewed_at",
        "source_url",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            "KR_LIFECYCLE_RESOLUTIONS_PATH is missing columns: "
            + ", ".join(missing)
        )
    source_hash = str(artifact.source_hash) if artifact is not None else ""
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        security_id = _cell_text(row.get("security_id")).upper()
        if not security_id:
            raise RuntimeError(
                "KR lifecycle exception requires a stable security_id/ISIN."
            )
        security_id = (
            security_id
            if security_id.startswith("KR:")
            else f"KR:{security_id}"
        )
        last_price_date = _normalized_date(row.get("last_price_date"))
        source_url = _cell_text(row.get("source_url"))
        if not source_url.lower().startswith(("http://", "https://")):
            raise RuntimeError(
                "Every KR lifecycle exception requires an official HTTP(S) source_url."
            )
        records.append(
            {
                "candidate_id": lifecycle_candidate_id(
                    security_id,
                    last_price_date,
                    selection_rule=KR_LIFECYCLE_SELECTION_RULE,
                ),
                "security_id": security_id,
                "symbol": normalize_kr_symbol(row.get("symbol"))
                if _cell_text(row.get("symbol"))
                else "",
                "last_price_date": last_price_date,
                "resolution": "exception",
                "event_id": "",
                "exception_code": _cell_text(row.get("exception_code")),
                "exception_reason": _cell_text(row.get("exception_reason")),
                "reviewed_by": _cell_text(row.get("reviewed_by")),
                "reviewed_at": _normalized_timestamp(
                    row.get("reviewed_at")
                ),
                "recheck_after": _normalized_date(row.get("recheck_after")),
                "successor_security_id": _cell_text(
                    row.get("successor_security_id")
                ),
                "successor_symbol": _cell_text(row.get("successor_symbol")),
                "source_url": source_url,
                "source": "kr_lifecycle_review_import",
                "retrieved_at": (
                    str(artifact.retrieved_at) if artifact is not None else utc_now_iso()
                ),
                "source_hash": source_hash,
            }
        )
    return (
        pd.DataFrame(
            records,
            columns=dataset_spec("lifecycle_resolutions").required_columns,
        ).drop_duplicates("candidate_id", keep="last"),
        artifact,
    )


def benchmark_kr_providers(
    cache_root: str | Path,
    *,
    session_count: int = 252,
    start_session: str | None = None,
    end_session: str | None = None,
    profiles: tuple[str, ...] = KR_PROFILES,
    providers: tuple[str, ...] = ("krx", "eodhd", "naver", "yahoo", "kis"),
    symbol_limit: int | None = None,
    sleep_seconds: float = 0.2,
    krx_workers: int = 4,
    reuse_provider_cache: bool = True,
    thresholds: KrBenchmarkThresholds | None = None,
) -> KrBenchmarkOutcome:
    """Compare candidate raw prices with KRX over one common session inventory."""

    validate_krx_official_configuration(require_membership=True)
    thresholds = thresholds or KrBenchmarkThresholds()
    sessions = (
        _sessions_between(
            start_session,
            end_session or expected_completed_session("KR"),
        )
        if start_session
        else recent_sessions("KR", session_count, end=end_session)
    )
    if not sessions:
        raise ValueError("KR benchmark session inventory is empty.")
    start, end = sessions[0], sessions[-1]
    run_root = Path(cache_root) / "benchmarks" / f"{start}_{end}_{len(sessions)}"
    checkpoint = KrCheckpointStore(run_root)
    bootstrap_checkpoint_root = Path(cache_root) / "state" / "bootstrap"
    canonical_checkpoint = (
        KrCheckpointStore(bootstrap_checkpoint_root)
        if start_session and bootstrap_checkpoint_root.is_dir()
        else checkpoint
    )
    memberships = _collect_memberships(
        sessions,
        profiles,
        canonical_checkpoint,
        sleep_seconds=sleep_seconds,
        allow_pre_inception=True,
        workers=krx_workers,
    )
    full_symbols = sorted(
        {
            symbol
            for snapshots in memberships.values()
            for snapshot in snapshots.values()
            for symbol in snapshot["symbols"]
        }
        | set(KR_BENCHMARK_SECURITIES.values())
    )
    symbols = list(full_symbols)
    benchmark_scope = "historical_full" if start_session else "window_full"
    if symbol_limit is not None:
        if symbol_limit <= 0:
            raise ValueError("symbol_limit must be positive.")
        benchmarks = set(KR_BENCHMARK_SECURITIES.values())
        symbols = sorted(set(symbols[:symbol_limit]) | benchmarks)
        benchmark_scope = "smoke"

    catalog_start = start_session or start
    catalog = _identity_catalog_for_run(
        canonical_checkpoint,
        start=catalog_start,
        end=end,
        include_dart=True,
    )
    membership_catalog, missing_membership_identity = _catalog_for_symbols(
        catalog,
        full_symbols,
        catalog_start,
        end,
    )
    if missing_membership_identity:
        raise RuntimeError(
            "KR benchmark cannot resolve the full PIT membership union: "
            + ", ".join(sorted(missing_membership_identity))
        )
    scoped_catalog, missing_identity = _catalog_for_symbols(
        catalog,
        symbols,
        catalog_start,
        end,
    )
    mapped_count = len(symbols) - len(missing_identity)
    identity_rate = mapped_count / len(symbols) if symbols else 0.0
    if missing_identity:
        # Continue to report every independent gate, but providers can only be
        # fetched for mapped instruments.
        symbols = [symbol for symbol in symbols if symbol not in missing_identity]

    krx_prices = _collect_krx_prices(
        sessions,
        symbols,
        scoped_catalog,
        canonical_checkpoint,
        sleep_seconds=sleep_seconds,
        workers=krx_workers,
    )
    scoped_catalog = _reconcile_catalog_with_krx_observations(
        scoped_catalog,
        krx_prices,
        completed_session=end,
    )
    if benchmark_scope == "smoke":
        membership_verification = {
            "schema": KR_HISTORY_VERIFICATION_SCHEMA,
            "status": "blocked",
            "market": "KR",
            "calendar": "XKRX",
            "requested_start": start,
            "requested_end": end,
            "expected_snapshot_count": sum(
                len(value) for value in memberships.values()
            ),
            "observed_snapshot_count": 0,
            "missing_snapshot_count": 0,
            "daily_replay_mismatch_count": 0,
            "blocking_issue_count": 1,
            "survivorship_score": 0.0,
            "profiles": {},
            "issues": [
                {
                    "code": "smoke_scope_cannot_verify_survivorship",
                    "profile": "",
                    "session": "",
                    "detail": "",
                }
            ],
        }
    else:
        membership_anchors, membership_events = _membership_datasets(
            memberships,
            scoped_catalog,
        )
        membership_verification = verify_kr_membership_history(
            memberships,
            scoped_catalog,
            membership_anchors,
            membership_events,
            sessions=sessions,
            profiles=profiles,
            checkpoint=canonical_checkpoint,
        )
    provider_results: dict[str, KrProviderResult] = {
        "krx": KrProviderResult("krx", "ok", krx_prices)
    }
    revision_metrics: dict[str, dict[str, Any]] = {
        "krx": {
            "status": "canonical_checkpoint",
            "overlap_key_count": len(krx_prices),
            "revised_key_count": 0,
            "score": 1.0,
        }
    }
    fetchers = {
        "eodhd": fetch_eodhd_prices,
        "naver": fetch_naver_prices,
        "yahoo": fetch_yahoo_prices,
        "kis": fetch_kis_prices,
    }
    for provider in providers:
        normalized = str(provider).strip().lower()
        if normalized == "krx":
            continue
        fetcher = fetchers.get(normalized)
        if fetcher is None:
            raise ValueError(f"Unsupported KR benchmark provider: {provider}")
        previous_prices = checkpoint.load_provider_prices(normalized)
        cached_result = (
            checkpoint.load_provider_result(normalized)
            if reuse_provider_cache
            else None
        )
        if cached_result is not None and cached_result.status != "ok":
            cached_result = None
        if cached_result is not None:
            result = cached_result
            revision_metrics[normalized] = {
                "status": "cached_resume",
                "overlap_key_count": len(result.prices),
                "revised_key_count": 0,
                "score": 1.0,
            }
        else:
            try:
                if normalized == "kis":
                    result = fetcher(
                        scoped_catalog,
                        start=start,
                        end=end,
                        checkpoint_root=checkpoint.root,
                        workers=max(4, krx_workers),
                    )
                elif normalized == "eodhd":
                    result = fetcher(
                        scoped_catalog,
                        start=start,
                        end=end,
                        workers=max(4, krx_workers),
                    )
                elif normalized == "naver":
                    result = fetcher(
                        scoped_catalog,
                        start=start,
                        end=end,
                        workers=min(max(4, krx_workers), 8),
                    )
                else:
                    result = fetcher(
                        scoped_catalog,
                        start=start,
                        end=end,
                    )
            except Exception as exc:
                result = KrProviderResult(
                    normalized,
                    "failed",
                    krx_prices.head(0).copy(),
                    detail=f"{type(exc).__name__}: {exc}",
                )
            revision_metrics[normalized] = _provider_revision_metrics(
                previous_prices,
                result.prices,
            )
        provider_results[normalized] = result
        if cached_result is None:
            checkpoint.save_provider_result(result)

    provider_metrics: dict[str, dict[str, Any]] = {}
    for provider, result in provider_results.items():
        metrics = compare_provider_to_krx(
            krx_prices,
            result,
            identity_mapping_rate=identity_rate,
            thresholds=thresholds,
        )
        metrics["license_class"] = _provider_license_class(provider)
        source_hashes = sorted(
            value
            for value in (
                {
                    str(item).strip().lower()
                    for item in result.prices.get(
                        "source_hash", pd.Series(dtype=str)
                    )
                }
                | {
                    str(artifact.source_hash).strip().lower()
                    for artifact in result.artifacts
                }
            )
            if len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )
        metrics["source_artifact_count"] = len(result.artifacts)
        metrics["source_hash_count"] = len(source_hashes)
        metrics["source_hash_inventory_sha256"] = sha256_bytes(
            canonical_json_bytes(source_hashes)
        )
        row_hashes = {
            str(item).strip().lower()
            for item in result.prices.get("source_hash", pd.Series(dtype=str))
            if _valid_sha256(item)
        }
        artifact_hashes = {
            str(artifact.source_hash).strip().lower()
            for artifact in result.artifacts
            if _valid_sha256(artifact.source_hash)
        }
        source_artifacts_complete = bool(
            provider == "krx"
            or (artifact_hashes and row_hashes and row_hashes <= artifact_hashes)
        )
        metrics["source_artifacts_complete"] = source_artifacts_complete
        metrics["hard_gate_passed"] = bool(
            metrics.get("hard_gate_passed")
            and source_artifacts_complete
            and membership_verification["status"] == "passed"
        )
        metrics["pit_survivorship_status"] = membership_verification["status"]
        metrics["pit_survivorship_score"] = membership_verification[
            "survivorship_score"
        ]
        metrics["ranking"] = _provider_scorecard(
            provider,
            metrics,
            revision_metrics.get(provider),
            pit_survivorship_score=membership_verification[
                "survivorship_score"
            ],
        )
        provider_metrics[provider] = metrics

    composite_result = _composite_independent_price_metrics(
        krx_prices,
        provider_results,
        provider_metrics,
        identity_mapping_rate=identity_rate,
        thresholds=thresholds,
    )
    if composite_result is not None:
        (
            composite_provider,
            composite_metrics,
            composite_revision,
        ) = composite_result
        composite_metrics["pit_survivorship_status"] = (
            membership_verification["status"]
        )
        composite_metrics["pit_survivorship_score"] = (
            membership_verification["survivorship_score"]
        )
        composite_metrics["hard_gate_passed"] = bool(
            composite_metrics.get("hard_gate_passed")
            and membership_verification["status"] == "passed"
        )
        composite_metrics["ranking"] = _provider_scorecard(
            composite_provider,
            composite_metrics,
            composite_revision,
            pit_survivorship_score=membership_verification[
                "survivorship_score"
            ],
        )
        provider_metrics[composite_provider] = composite_metrics
        revision_metrics[composite_provider] = composite_revision

    passed_candidates = [
        (provider, metrics)
        for provider, metrics in provider_metrics.items()
        if provider != "krx" and metrics.get("hard_gate_passed") is True
    ]
    passed_candidates.sort(
        key=lambda item: (
            0 if item[0].startswith("composite:") else 1,
            -float(item[1]["ranking"]["total_score"]),
            -float(item[1].get("close_within_one_tick_rate", 0.0)),
            -float(item[1].get("expected_session_coverage", 0.0)),
            item[0],
        )
    )
    primary = "krx" if provider_metrics["krx"]["hard_gate_passed"] else ""
    secondary = passed_candidates[0][0] if passed_candidates else ""
    status = (
        "ready"
        if (
            primary
            and secondary.startswith("composite:")
            and benchmark_scope == "historical_full"
        )
        else "window_ready"
        if primary and secondary and benchmark_scope == "window_full"
        else "smoke_ready"
        if primary and secondary
        else "blocked"
    )
    report = {
        "schema_version": 5,
        "market": "KR",
        "calendar": "XKRX",
        "currency": "KRW",
        "created_at": utc_now_iso(),
        "status": status,
        "benchmark_scope": benchmark_scope,
        "sessions": {"start": start, "end": end, "count": len(sessions)},
        "profiles": list(profiles),
        "membership_verification": membership_verification,
        "symbols": {
            "full_union_count": len(full_symbols),
            "evaluated_count": len(symbols),
            "identity_mapping_rate": identity_rate,
            "missing_identity": sorted(missing_identity),
        },
        "thresholds": asdict(thresholds),
        "providers": provider_metrics,
        "ranking": [
            {
                "provider": provider,
                **metrics["ranking"],
            }
            for provider, metrics in passed_candidates
        ],
        "selection": {
            "primary": primary,
            "secondary": secondary,
            "rule": (
                "KRX official raw primary; a hard-gated multi-provider "
                "row-level composite is preferred, otherwise the highest "
                "weighted hard-gate provider is secondary; survivorship is "
                "scored only by full official-snapshot anchor/event replay"
            ),
        },
        "license_policy": KR_LICENSE_POLICY,
    }
    report_bytes = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    report_sha = sha256_bytes(report_bytes)
    immutable_path = run_root / f"report-{report_sha}.json"
    write_atomic(immutable_path, report_bytes)
    current_name = (
        "current.json"
        if benchmark_scope == "historical_full"
        else "current-window.json"
        if benchmark_scope == "window_full"
        else "current-smoke.json"
    )
    current_path = Path(cache_root) / "benchmarks" / current_name
    write_atomic(current_path, report_bytes)
    return KrBenchmarkOutcome(
        report_path=str(immutable_path),
        status=status,
        start_session=start,
        end_session=end,
        session_count=len(sessions),
        symbol_count=len(symbols),
        primary_provider=primary,
        secondary_provider=secondary,
        providers=provider_metrics,
    )


def compare_provider_to_krx(
    baseline: pd.DataFrame,
    candidate: KrProviderResult,
    *,
    identity_mapping_rate: float,
    thresholds: KrBenchmarkThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or KrBenchmarkThresholds()
    if candidate.status != "ok":
        return {
            "status": candidate.status,
            "detail": candidate.detail,
            "hard_gate_passed": False,
            "identity_mapping_rate": identity_mapping_rate,
            "row_count": len(candidate.prices),
        }
    key = ["security_id", "session"]
    expected_observations = baseline.copy()
    actual = candidate.prices.copy()
    if "observation_status" not in expected_observations:
        expected_observations["observation_status"] = KRX_TRADED_STATUS
    else:
        expected_observations["observation_status"] = (
            expected_observations["observation_status"]
            .fillna(KRX_TRADED_STATUS)
            .replace("", KRX_TRADED_STATUS)
        )
    if "observation_status" not in actual:
        actual["observation_status"] = KRX_TRADED_STATUS
    else:
        actual["observation_status"] = (
            actual["observation_status"]
            .fillna(KRX_TRADED_STATUS)
            .replace("", KRX_TRADED_STATUS)
        )
    official_status = expected_observations["observation_status"].astype(str)
    classified_no_trade = int(
        official_status.isin(KRX_CLASSIFIED_NO_TRADE_STATUSES).sum()
    )
    official_unclassified = int(
        (~official_status.isin(KRX_ALLOWED_OBSERVATION_STATUSES)).sum()
    )
    expected = expected_observations.loc[
        official_status.eq(KRX_TRADED_STATUS)
    ].copy()
    actual_observations = actual.copy()
    actual_status = actual_observations["observation_status"].astype(str)
    actual = actual_observations.loc[
        actual_status.eq(KRX_TRADED_STATUS)
    ].copy()
    expected["session"] = pd.to_datetime(expected["session"], errors="coerce").dt.date.astype(str)
    expected_observations["session"] = pd.to_datetime(
        expected_observations["session"], errors="coerce"
    ).dt.date.astype(str)
    actual_observations["session"] = pd.to_datetime(
        actual_observations["session"], errors="coerce"
    ).dt.date.astype(str)
    actual["session"] = pd.to_datetime(actual["session"], errors="coerce").dt.date.astype(str)
    duplicate_count = int(
        actual_observations.duplicated(key, keep=False).sum()
    )
    numeric = actual[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid = (
        numeric.isna().any(axis=1)
        | numeric[["open", "high", "low", "close"]].le(0).any(axis=1)
        | numeric["volume"].lt(0)
        | numeric["high"].lt(numeric[["open", "high", "low", "close"]].max(axis=1))
        | numeric["low"].gt(numeric[["open", "high", "low", "close"]].min(axis=1))
    )
    invalid_count = int(invalid.sum())
    source_hash = actual_observations.get(
        "source_hash", pd.Series("", index=actual_observations.index)
    ).astype(str)
    source_url = actual_observations.get(
        "source_url", pd.Series("", index=actual_observations.index)
    ).astype(str)
    source = actual_observations.get(
        "source", pd.Series("", index=actual_observations.index)
    ).astype(str)
    retrieved_at = actual_observations.get(
        "retrieved_at", pd.Series("", index=actual_observations.index)
    ).astype(str)
    reproducible_rows = (
        source_hash.map(_valid_sha256)
        & source_url.str.strip().ne("")
        & source.str.strip().ne("")
        & pd.to_datetime(retrieved_at, errors="coerce", utc=True).notna()
    )
    source_reproducibility_rate = (
        float(reproducible_rows.mean()) if len(actual) else 0.0
    )
    expected_keys = set(expected[key].astype(str).itertuples(index=False, name=None))
    official_observation_keys = set(
        expected_observations[key].astype(str).itertuples(index=False, name=None)
    )
    official_no_trade_keys = set(
        expected_observations.loc[
            official_status.isin(KRX_CLASSIFIED_NO_TRADE_STATUSES),
            key,
        ]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    provider_observation_keys = set(
        actual_observations[key].astype(str).itertuples(index=False, name=None)
    )
    actual_keys = set(actual[key].astype(str).itertuples(index=False, name=None))
    unexpected_provider_observations = (
        provider_observation_keys - official_observation_keys
    )
    misclassified_no_trade = actual_keys & official_no_trade_keys
    provider_classified_no_trade = int(
        actual_observations["observation_status"]
        .astype(str)
        .isin(KRX_CLASSIFIED_NO_TRADE_STATUSES)
        .sum()
    )
    missing = expected_keys - actual_keys
    coverage = len(expected_keys & actual_keys) / len(expected_keys) if expected_keys else 0.0
    joined = expected.merge(actual, on=key, suffixes=("_krx", "_provider"))
    if joined.empty:
        close_rate = 0.0
        ohlc_rate = 0.0
        volume_rate = 0.0
        large_cross_source = 0
        max_close_diff = None
    else:
        ohlc_differences = pd.DataFrame(
            {
                column: (
                    pd.to_numeric(
                        joined[f"{column}_provider"],
                        errors="coerce",
                    )
                    - pd.to_numeric(
                        joined[f"{column}_krx"],
                        errors="coerce",
                    )
                ).abs()
                for column in ("open", "high", "low", "close")
            },
            index=joined.index,
        )
        differences = ohlc_differences["close"]
        ticks = _krx_tick_sizes(
            joined,
            price_column="close_krx",
            exchange_column=(
                "exchange_krx"
                if "exchange_krx" in joined
                else "exchange"
            ),
            asset_type_column=(
                "asset_type_krx"
                if "asset_type_krx" in joined
                else "asset_type"
            ),
        )
        within = differences.le(ticks + 1e-9)
        ohlc_within = ohlc_differences.le(
            ticks,
            axis=0,
        ).all(axis=1)
        volume_within = (
            pd.to_numeric(
                joined["volume_provider"],
                errors="coerce",
            )
            - pd.to_numeric(
                joined["volume_krx"],
                errors="coerce",
            )
        ).abs().le(0.5)
        # Missing rows are failures for the 99.9% gate, rather than being
        # silently excluded from the denominator.
        close_rate = int(within.sum()) / len(expected_keys) if expected_keys else 0.0
        ohlc_rate = (
            int(ohlc_within.sum()) / len(expected_keys)
            if expected_keys
            else 0.0
        )
        volume_rate = (
            int(volume_within.sum()) / len(expected_keys)
            if expected_keys
            else 0.0
        )
        large_cross_source = int(
            differences.gt((ticks * 5).clip(lower=1.0)).sum()
        )
        max_close_diff = float(differences.max()) if differences.notna().any() else None
    unexplained_discontinuities = _unexplained_temporal_discontinuities(
        expected,
        actual,
        threshold=thresholds.large_return_threshold,
    )
    hard_gate = bool(
        identity_mapping_rate >= thresholds.identity_mapping_rate
        and duplicate_count <= thresholds.max_duplicate_keys
        and invalid_count <= thresholds.max_invalid_ohlc
        and source_reproducibility_rate >= 1.0
        and coverage >= thresholds.expected_session_coverage
        and close_rate >= thresholds.close_within_one_tick_rate
        and ohlc_rate >= thresholds.ohlc_within_one_tick_rate
        and volume_rate >= thresholds.volume_exact_rate
        and len(missing) <= thresholds.max_unclassified_missing
        and official_unclassified <= thresholds.max_unclassified_missing
        and len(unexpected_provider_observations)
        <= thresholds.max_unexpected_provider_observations
        and len(misclassified_no_trade)
        <= thresholds.max_misclassified_no_trade
        and unexplained_discontinuities
        <= thresholds.max_unexplained_large_discontinuities
        and large_cross_source
        <= thresholds.max_large_cross_source_discrepancies
    )
    return {
        "status": candidate.status,
        "detail": candidate.detail,
        "hard_gate_passed": hard_gate,
        "identity_mapping_rate": identity_mapping_rate,
        "row_count": len(actual),
        "observation_count": len(actual_observations),
        "expected_row_count": len(expected_keys),
        "duplicate_key_rows": duplicate_count,
        "invalid_ohlc_rows": invalid_count,
        "source_reproducibility_rate": source_reproducibility_rate,
        "expected_session_coverage": coverage,
        "close_within_one_tick_rate": close_rate,
        "ohlc_within_one_tick_rate": ohlc_rate,
        "volume_exact_rate": volume_rate,
        "unclassified_missing": len(missing),
        "krx_classified_no_trade_observations": classified_no_trade,
        "krx_unclassified_observations": official_unclassified,
        "provider_classified_no_trade_observations": provider_classified_no_trade,
        "unexpected_provider_observations": len(
            unexpected_provider_observations
        ),
        "misclassified_official_no_trade_observations": len(
            misclassified_no_trade
        ),
        "unexplained_large_discontinuities": unexplained_discontinuities,
        "large_cross_source_price_discrepancies": large_cross_source,
        "max_absolute_close_difference": max_close_diff,
    }


def _composite_independent_price_metrics(
    baseline: pd.DataFrame,
    provider_results: dict[str, KrProviderResult],
    provider_metrics: dict[str, dict[str, Any]],
    *,
    identity_mapping_rate: float,
    thresholds: KrBenchmarkThresholds,
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """Verify every official traded row against at least one raw provider."""

    participants = sorted(
        provider
        for provider, result in provider_results.items()
        if provider != "krx"
        and result.status == "ok"
        and not result.prices.empty
        and provider_metrics.get(provider, {}).get(
            "source_artifacts_complete"
        )
        is True
    )
    if len(participants) < 2:
        return None

    key = ["security_id", "session"]
    expected_observations = baseline.copy()
    if "observation_status" not in expected_observations:
        expected_observations["observation_status"] = KRX_TRADED_STATUS
    expected_observations["observation_status"] = (
        expected_observations["observation_status"]
        .fillna(KRX_TRADED_STATUS)
        .replace("", KRX_TRADED_STATUS)
        .astype(str)
    )
    expected_observations["session"] = pd.to_datetime(
        expected_observations["session"],
        errors="coerce",
    ).dt.date.astype(str)
    official_status = expected_observations["observation_status"]
    official_unclassified = int(
        (~official_status.isin(KRX_ALLOWED_OBSERVATION_STATUSES)).sum()
    )
    expected = expected_observations.loc[
        official_status.eq(KRX_TRADED_STATUS)
    ].copy()
    expected = expected.drop_duplicates(key, keep="last")
    expected["_tick"] = _krx_tick_sizes(
        expected,
        price_column="close",
        exchange_column="exchange",
        asset_type_column="asset_type",
    )

    matching_frames: list[pd.DataFrame] = []
    evaluation_frames: list[pd.DataFrame] = []
    observation_key_frames: list[pd.DataFrame] = []
    traded_key_frames: list[pd.DataFrame] = []
    verification_counts: dict[str, int] = {}
    for provider in participants:
        observations = provider_results[provider].prices.copy()
        if "observation_status" not in observations:
            observations["observation_status"] = KRX_TRADED_STATUS
        observations["observation_status"] = (
            observations["observation_status"]
            .fillna(KRX_TRADED_STATUS)
            .replace("", KRX_TRADED_STATUS)
            .astype(str)
        )
        observations["session"] = pd.to_datetime(
            observations["session"],
            errors="coerce",
        ).dt.date.astype(str)
        observations = observations.drop_duplicates(
            key,
            keep="last",
        )
        observation_key_frames.append(observations[key])
        actual = observations.loc[
            observations["observation_status"].eq(KRX_TRADED_STATUS)
        ].copy()
        traded_key_frames.append(actual[key])
        joined = expected.merge(
            actual,
            on=key,
            suffixes=("_krx", "_provider"),
            how="inner",
        )
        if joined.empty:
            verification_counts[provider] = 0
            continue
        valid = pd.Series(True, index=joined.index)
        for column in ("open", "high", "low", "close"):
            difference = (
                pd.to_numeric(
                    joined[f"{column}_provider"],
                    errors="coerce",
                )
                - pd.to_numeric(
                    joined[f"{column}_krx"],
                    errors="coerce",
                )
            ).abs()
            valid &= difference.le(joined["_tick"] + 1e-9)
        volume_difference = (
            pd.to_numeric(
                joined["volume_provider"],
                errors="coerce",
            )
            - pd.to_numeric(
                joined["volume_krx"],
                errors="coerce",
            )
        ).abs()
        valid &= volume_difference.le(0.5)
        evaluated = joined[key].copy()
        evaluated["provider"] = provider
        evaluated["matched"] = valid.astype(bool)
        evaluation_frames.append(evaluated)
        matched = evaluated.loc[
            evaluated["matched"],
            [*key, "provider"],
        ]
        matching_frames.append(matched)
        verification_counts[provider] = len(matched)

    matches = (
        pd.concat(matching_frames, ignore_index=True)
        .drop_duplicates([*key, "provider"], keep="last")
        if matching_frames
        else pd.DataFrame(columns=(*key, "provider"))
    )
    verified_counts = (
        matches.groupby(key, sort=False)
        .size()
        .rename("provider_count")
        .reset_index()
    )
    coverage = expected[key].merge(
        verified_counts,
        on=key,
        how="left",
        validate="one_to_one",
    )
    coverage["provider_count"] = (
        coverage["provider_count"].fillna(0).astype(int)
    )
    expected_count = len(coverage)
    verified_count = int(coverage["provider_count"].gt(0).sum())
    unverified_count = expected_count - verified_count

    official_key_index = pd.MultiIndex.from_frame(
        expected_observations[key].drop_duplicates()
    )
    official_no_trade_index = pd.MultiIndex.from_frame(
        expected_observations.loc[
            official_status.isin(KRX_CLASSIFIED_NO_TRADE_STATUSES),
            key,
        ].drop_duplicates()
    )
    provider_observation_index = pd.MultiIndex.from_frame(
        pd.concat(
            observation_key_frames,
            ignore_index=True,
        ).drop_duplicates()
    )
    provider_traded_index = pd.MultiIndex.from_frame(
        pd.concat(
            traded_key_frames,
            ignore_index=True,
        ).drop_duplicates()
    )
    unexpected_provider_observations = len(
        provider_observation_index.difference(official_key_index)
    )
    misclassified_no_trade = len(
        provider_traded_index.intersection(
            official_no_trade_index
        )
    )

    evaluations = (
        pd.concat(evaluation_frames, ignore_index=True)
        if evaluation_frames
        else pd.DataFrame(columns=(*key, "provider", "matched"))
    )
    if evaluations.empty:
        provider_disagreement_count = 0
        unresolved_disagreement_count = 0
    else:
        evaluation_summary = evaluations.groupby(
            key,
            sort=False,
        )["matched"].agg(["any", "all"])
        provider_disagreement_count = int(
            (
                evaluation_summary["any"]
                & ~evaluation_summary["all"]
            ).sum()
        )
        unresolved_disagreement_count = int(
            (~evaluation_summary["any"]).sum()
        )

    assignment_digest = hashlib.sha256()
    for row in matches.sort_values(
        [*key, "provider"],
        kind="stable",
    ).itertuples(index=False, name=None):
        assignment_digest.update(
            ("\0".join(str(value) for value in row) + "\n").encode()
        )

    source_inventory = [
        {
            "provider": provider,
            "source_hash_count": int(
                provider_metrics[provider].get(
                    "source_hash_count",
                    0,
                )
            ),
            "source_hash_inventory_sha256": str(
                provider_metrics[provider].get(
                    "source_hash_inventory_sha256",
                    "",
                )
            ),
        }
        for provider in participants
    ]
    source_artifacts_complete = all(
        provider_metrics[provider].get(
            "source_artifacts_complete"
        )
        is True
        for provider in participants
    )
    # Provider-only rows are evidence about that provider's quality, not a
    # reason to overwrite or reject an otherwise fully corroborated official
    # KRX row inventory.  They remain counted and quarantined below.  The
    # composite gate is concerned with the opposite direction: every official
    # traded row must have at least one independently matching raw observation.
    hard_gate = bool(
        identity_mapping_rate >= thresholds.identity_mapping_rate
        and source_artifacts_complete
        and official_unclassified == 0
        and unverified_count == 0
    )
    verified_rate = (
        verified_count / expected_count
        if expected_count
        else 0.0
    )
    composite_name = "composite:" + "+".join(participants)
    revision = {
        "status": "composite_minimum",
        "overlap_key_count": expected_count,
        "revised_key_count": 0,
        "score": min(
            float(
                provider_metrics[provider]
                .get("ranking", {})
                .get("revision_observation", {})
                .get("score", 0.0)
            )
            for provider in participants
        ),
    }
    metrics = {
        "status": "ok" if hard_gate else "blocked",
        "detail": (
            "at_least_one_exact_raw_provider_per_krx_row;"
            "provider_only_anomalies_quarantined"
        ),
        "hard_gate_passed": hard_gate,
        "identity_mapping_rate": identity_mapping_rate,
        "participant_providers": participants,
        "row_count": int(
            sum(
                len(provider_results[provider].prices)
                for provider in participants
            )
        ),
        "observation_count": int(
            sum(
                len(provider_results[provider].prices)
                for provider in participants
            )
        ),
        "expected_row_count": expected_count,
        "verified_row_count": verified_count,
        "expected_session_coverage": verified_rate,
        "close_within_one_tick_rate": verified_rate,
        "ohlc_within_one_tick_rate": verified_rate,
        "volume_exact_rate": verified_rate,
        "unclassified_missing": unverified_count,
        "krx_unclassified_observations": official_unclassified,
        "unexpected_provider_observations": (
            unexpected_provider_observations
        ),
        "misclassified_official_no_trade_observations": (
            misclassified_no_trade
        ),
        "quarantined_provider_observation_count": (
            unexpected_provider_observations
            + misclassified_no_trade
        ),
        "provider_disagreement_count": provider_disagreement_count,
        "unresolved_cross_source_disagreement_count": (
            unresolved_disagreement_count
        ),
        "unexplained_large_discontinuities": unverified_count,
        "large_cross_source_price_discrepancies": sum(
            int(
                provider_metrics[provider].get(
                    "large_cross_source_price_discrepancies",
                    0,
                )
            )
            for provider in participants
        ),
        "duplicate_key_rows": sum(
            int(
                provider_metrics[provider].get(
                    "duplicate_key_rows",
                    0,
                )
            )
            for provider in participants
        ),
        "invalid_ohlc_rows": sum(
            int(
                provider_metrics[provider].get(
                    "invalid_ohlc_rows",
                    0,
                )
            )
            for provider in participants
        ),
        "source_reproducibility_rate": min(
            float(
                provider_metrics[provider].get(
                    "source_reproducibility_rate",
                    0.0,
                )
            )
            for provider in participants
        ),
        "source_artifacts_complete": source_artifacts_complete,
        "source_artifact_count": sum(
            int(
                provider_metrics[provider].get(
                    "source_artifact_count",
                    0,
                )
            )
            for provider in participants
        ),
        "source_hash_count": sum(
            int(
                provider_metrics[provider].get(
                    "source_hash_count",
                    0,
                )
            )
            for provider in participants
        ),
        "source_hash_inventory_sha256": sha256_bytes(
            canonical_json_bytes(source_inventory)
        ),
        "verification_assignment_sha256": (
            assignment_digest.hexdigest()
        ),
        "verification_provider_counts": verification_counts,
        "single_provider_verified_count": int(
            coverage["provider_count"].eq(1).sum()
        ),
        "multiple_provider_verified_count": int(
            coverage["provider_count"].gt(1).sum()
        ),
        "license_class": KR_LICENSE_POLICY["benchmark_report"],
        "operating_cost_score": min(
            KR_PROVIDER_OPERATING_COST_SCORES.get(
                provider,
                0.0,
            )
            for provider in participants
        ),
    }
    return composite_name, metrics, revision


def _valid_sha256(value: Any) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _provider_revision_metrics(
    previous: pd.DataFrame | None,
    current: pd.DataFrame,
) -> dict[str, Any]:
    """Measure provider changes to the same raw observations on repeat runs."""

    if previous is None or previous.empty:
        return {
            "status": "first_observation",
            "overlap_key_count": 0,
            "revised_key_count": 0,
            "score": 0.5,
        }
    columns = ("open", "high", "low", "close", "volume")

    def normalize(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
        values = frame.copy()
        values["session"] = pd.to_datetime(
            values["session"], errors="coerce"
        ).dt.date.astype(str)
        values = values.drop_duplicates(["security_id", "session"], keep="last")
        for column in columns:
            values[column] = pd.to_numeric(values[column], errors="coerce")
        return values[["security_id", "session", *columns]].rename(
            columns={column: f"{column}_{suffix}" for column in columns}
        )

    left = normalize(previous, "previous")
    right = normalize(current, "current")
    overlap = left.merge(right, on=["security_id", "session"], how="inner")
    if overlap.empty:
        return {
            "status": "no_overlap",
            "overlap_key_count": 0,
            "revised_key_count": 0,
            "score": 0.0,
        }
    revised = pd.Series(False, index=overlap.index)
    for column in columns:
        old = overlap[f"{column}_previous"]
        new = overlap[f"{column}_current"]
        tolerance = old.abs().mul(1e-12).add(1e-9)
        equal = (old.isna() & new.isna()) | (
            old.notna() & new.notna() & old.sub(new).abs().le(tolerance)
        )
        revised |= ~equal
    revised_count = int(revised.sum())
    return {
        "status": "measured",
        "overlap_key_count": len(overlap),
        "revised_key_count": revised_count,
        "score": round(1.0 - (revised_count / len(overlap)), 12),
    }


def _provider_scorecard(
    provider: str,
    metrics: dict[str, Any],
    revision: dict[str, Any] | None,
    *,
    pit_survivorship_score: float = 0.0,
) -> dict[str, Any]:
    expected = max(1, int(metrics.get("expected_row_count", 0)))
    missing = int(metrics.get("unclassified_missing", expected))
    discontinuities = int(
        metrics.get("unexplained_large_discontinuities", expected)
    )

    def bounded(value: Any) -> float:
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    dimensions = {
        "accuracy": bounded(metrics.get("close_within_one_tick_rate", 0.0)),
        "coverage": bounded(metrics.get("expected_session_coverage", 0.0)),
        "corporate_action": bounded(1.0 - discontinuities / expected),
        # A price feed cannot prove point-in-time index membership. This score
        # is supplied only by the independent anchor/event replay against every
        # official daily KRX constituent snapshot.
        "survivorship": bounded(pit_survivorship_score),
        "revision_stability": bounded((revision or {}).get("score", 0.0)),
        "operating_cost": bounded(
            metrics.get(
                "operating_cost_score",
                KR_PROVIDER_OPERATING_COST_SCORES.get(provider, 0.0),
            )
        ),
    }
    total = sum(
        dimensions[name] * weight
        for name, weight in KR_BENCHMARK_RANKING_WEIGHTS.items()
    )
    return {
        "total_score": round(total, 12),
        "dimensions": dimensions,
        "weights": KR_BENCHMARK_RANKING_WEIGHTS,
        "revision_observation": revision or {
            "status": "unavailable",
            "overlap_key_count": 0,
            "revised_key_count": 0,
            "score": 0.0,
        },
    }


def _unexplained_temporal_discontinuities(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    threshold: float,
) -> int:
    """Count provider-only jumps while accepting the same raw jump in KRX.

    A split, capital reduction, or genuine market move appears in both raw
    series. A provider adjustment or splice generally appears in only one.
    This keeps the benchmark independent of a single corporate-action feed.
    """

    def returns(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
        values = frame[["security_id", "session", "close"]].copy()
        values["session"] = pd.to_datetime(values["session"], errors="coerce")
        values["close"] = pd.to_numeric(values["close"], errors="coerce")
        values = values.dropna(subset=["security_id", "session", "close"]).sort_values(
            ["security_id", "session"], kind="stable"
        )
        values[f"return_{suffix}"] = values.groupby("security_id")["close"].pct_change(
            fill_method=None
        )
        return values[["security_id", "session", f"return_{suffix}"]]

    candidate_returns = returns(candidate, "provider")
    large = candidate_returns.loc[
        candidate_returns["return_provider"].abs().ge(float(threshold))
    ]
    if large.empty:
        return 0
    baseline_returns = returns(baseline, "krx")
    joined = large.merge(
        baseline_returns,
        on=["security_id", "session"],
        how="left",
    )
    provider_return = joined["return_provider"]
    krx_return = joined["return_krx"]
    same_direction = provider_return.mul(krx_return).gt(0)
    close_magnitude = provider_return.sub(krx_return).abs().le(0.05)
    explained = krx_return.notna() & same_direction & close_magnitude
    return int((~explained).sum())


def krx_tick_size(
    price: float,
    *,
    exchange: str = "KOSPI",
    asset_type: str = "STOCK",
    session: str = "",
) -> float:
    """Return the applicable KRX cash-equity tick for one observation.

    The stock schedule was unified and narrowed on 2023-01-25. KOSDAQ keeps a
    100-won maximum tick after that date, while KOSPI retains 500/1,000-won
    high-price bands. ETFs/ETNs use their product tick independently.
    """

    try:
        value = float(price)
    except (TypeError, ValueError):
        return float("nan")
    kind = str(asset_type).strip().upper()
    if kind in {"ETF", "ETN", "ELW"}:
        return 5.0
    market = str(exchange).strip().upper()
    when = pd.to_datetime(session, errors="coerce")
    current_schedule = pd.isna(when) or pd.Timestamp(when) >= pd.Timestamp(
        "2023-01-25"
    )
    if not current_schedule:
        if market == "KOSDAQ":
            if value < 1_000:
                return 1.0
            if value < 5_000:
                return 5.0
            if value < 10_000:
                return 10.0
            if value < 50_000:
                return 50.0
            return 100.0
        if value < 2_000:
            return 1.0
        if value < 5_000:
            return 5.0
        if value < 20_000:
            return 10.0
        if value < 50_000:
            return 50.0
        if value < 200_000:
            return 100.0
        if value < 500_000:
            return 500.0
        return 1_000.0
    if value < 1_000:
        return 1.0
    if value < 5_000:
        return 5.0
    if value < 10_000:
        return 10.0
    if value < 50_000:
        return 50.0
    if value < 100_000:
        return 100.0
    if market == "KOSDAQ":
        return 100.0
    if value < 500_000:
        return 500.0
    return 1_000.0


def _krx_tick_sizes(
    frame: pd.DataFrame,
    *,
    price_column: str,
    exchange_column: str,
    asset_type_column: str,
    session_column: str = "session",
) -> pd.Series:
    """Vectorized equivalent of ``krx_tick_size`` for full-history gates."""

    prices = pd.to_numeric(
        frame.get(
            price_column,
            pd.Series(float("nan"), index=frame.index),
        ),
        errors="coerce",
    )
    exchanges = (
        frame.get(
            exchange_column,
            pd.Series("KOSPI", index=frame.index),
        )
        .fillna("KOSPI")
        .astype(str)
        .str.upper()
    )
    asset_types = (
        frame.get(
            asset_type_column,
            pd.Series("STOCK", index=frame.index),
        )
        .fillna("STOCK")
        .astype(str)
        .str.upper()
    )
    sessions = pd.to_datetime(
        frame.get(
            session_column,
            pd.Series("", index=frame.index),
        ),
        errors="coerce",
    )
    ticks = pd.Series(
        float("nan"),
        index=frame.index,
        dtype=float,
    )
    valid = prices.notna()
    product = valid & asset_types.isin({"ETF", "ETN", "ELW"})
    ticks.loc[product] = 5.0
    stock = valid & ~product
    current = sessions.isna() | sessions.ge(
        pd.Timestamp("2023-01-25")
    )
    pre_current = stock & ~current
    current_stock = stock & current
    kosdaq = exchanges.eq("KOSDAQ")

    def assign(mask: pd.Series, value: float) -> None:
        ticks.loc[mask] = value

    assign(pre_current & kosdaq & prices.lt(1_000), 1.0)
    assign(
        pre_current
        & kosdaq
        & prices.ge(1_000)
        & prices.lt(5_000),
        5.0,
    )
    assign(
        pre_current
        & kosdaq
        & prices.ge(5_000)
        & prices.lt(10_000),
        10.0,
    )
    assign(
        pre_current
        & kosdaq
        & prices.ge(10_000)
        & prices.lt(50_000),
        50.0,
    )
    assign(pre_current & kosdaq & prices.ge(50_000), 100.0)

    pre_kospi = pre_current & ~kosdaq
    assign(pre_kospi & prices.lt(2_000), 1.0)
    assign(
        pre_kospi & prices.ge(2_000) & prices.lt(5_000),
        5.0,
    )
    assign(
        pre_kospi & prices.ge(5_000) & prices.lt(20_000),
        10.0,
    )
    assign(
        pre_kospi & prices.ge(20_000) & prices.lt(50_000),
        50.0,
    )
    assign(
        pre_kospi
        & prices.ge(50_000)
        & prices.lt(200_000),
        100.0,
    )
    assign(
        pre_kospi
        & prices.ge(200_000)
        & prices.lt(500_000),
        500.0,
    )
    assign(pre_kospi & prices.ge(500_000), 1_000.0)

    assign(current_stock & prices.lt(1_000), 1.0)
    assign(
        current_stock
        & prices.ge(1_000)
        & prices.lt(5_000),
        5.0,
    )
    assign(
        current_stock
        & prices.ge(5_000)
        & prices.lt(10_000),
        10.0,
    )
    assign(
        current_stock
        & prices.ge(10_000)
        & prices.lt(50_000),
        50.0,
    )
    assign(
        current_stock
        & prices.ge(50_000)
        & prices.lt(100_000),
        100.0,
    )
    assign(
        current_stock & kosdaq & prices.ge(100_000),
        100.0,
    )
    current_kospi = current_stock & ~kosdaq
    assign(
        current_kospi
        & prices.ge(100_000)
        & prices.lt(500_000),
        500.0,
    )
    assign(current_kospi & prices.ge(500_000), 1_000.0)
    return ticks


def bootstrap_kr_market_data(
    repository: LocalDatasetRepository,
    *,
    start_date: str = "2015-01-01",
    end_date: str | None = None,
    profiles: tuple[str, ...] = KR_PROFILES,
    sleep_seconds: float = 0.2,
    krx_workers: int = 4,
    require_eodhd_actions: bool = True,
) -> KrBootstrapResult:
    """Build the initial immutable KR point-in-time release."""

    if repository.market != "KR":
        raise ValueError("KR bootstrap requires LocalDatasetRepository(..., market='KR').")
    validate_krx_official_configuration(require_membership=True)
    completed = end_date or expected_completed_session("KR")
    benchmark, benchmark_bytes = _load_ready_benchmark(
        repository.root,
        start_session=start_date,
        end_session=completed,
    )
    sessions = _sessions_between(start_date, completed)
    checkpoint = KrCheckpointStore(repository.root / "state" / "bootstrap")
    memberships = _collect_memberships(
        sessions,
        profiles,
        checkpoint,
        sleep_seconds=sleep_seconds,
        allow_pre_inception=True,
        workers=krx_workers,
    )
    constituent_symbols = sorted(
        {
            symbol
            for snapshots in memberships.values()
            for snapshot in snapshots.values()
            for symbol in snapshot["symbols"]
        }
    )
    all_symbols = sorted(
        set(constituent_symbols)
        | set(KR_BENCHMARK_SECURITIES.values())
        | _configured_official_action_symbols()
    )
    catalog = _identity_catalog_for_run(
        checkpoint,
        start=start_date,
        end=completed,
        include_dart=True,
    )
    scoped_catalog, missing_identity = _catalog_for_symbols(
        catalog, all_symbols, start_date, completed
    )
    if missing_identity:
        raise RuntimeError(
            "KR bootstrap identity mapping is incomplete: "
            + ", ".join(sorted(missing_identity))
        )
    price_observations = _collect_krx_prices(
        sessions,
        all_symbols,
        scoped_catalog,
        checkpoint,
        sleep_seconds=sleep_seconds,
        workers=krx_workers,
    )
    scoped_catalog = _reconcile_catalog_with_krx_observations(
        scoped_catalog,
        price_observations,
        completed_session=completed,
    )
    _assert_krx_observations_classified(price_observations, context="bootstrap")
    raw_prices = _traded_krx_prices(price_observations)
    if raw_prices.empty:
        raise RuntimeError("KRX bootstrap returned no raw prices.")
    unmapped_rows = raw_prices["security_id"].astype(str).str.strip().eq("")
    if unmapped_rows.any():
        raise RuntimeError(
            f"KRX bootstrap has {int(unmapped_rows.sum())} price rows without stable identity."
        )

    master, history = _identity_datasets(scoped_catalog, all_symbols, start_date)
    anchors, events = _membership_datasets(memberships, scoped_catalog)
    index_price_gap_policy = _krx_index_price_gap_policy(
        anchors,
        events,
        raw_prices,
        price_observations,
    )
    warnings: list[str] = []
    try:
        actions, action_artifacts, action_failures = fetch_eodhd_actions(
            scoped_catalog,
            start=start_date,
            end=completed,
            workers=6,
        )
    except Exception as exc:
        if require_eodhd_actions:
            raise RuntimeError(
                f"EODHD KR corporate-action collection failed: {type(exc).__name__}: {exc}"
            ) from None
        actions = _empty_dataset("corporate_actions")
        action_artifacts = ()
        action_failures = (type(exc).__name__,)
    known_action_gaps = tuple(
        value
        for value in action_failures
        if value.startswith("known_unavailable_pre2019_delisted_actions:")
    )
    hard_action_failures = tuple(
        value
        for value in action_failures
        if value not in known_action_gaps
    )
    if hard_action_failures:
        message = (
            "EODHD action requests failed unexpectedly for "
            f"{len(hard_action_failures)} symbols: "
            + ", ".join(hard_action_failures[:20])
        )
        if require_eodhd_actions:
            raise RuntimeError(message)
        warnings.append(message)
    official_actions, official_action_artifact = _load_kr_official_actions(
        scoped_catalog,
        start=start_date,
        end=completed,
    )
    if official_action_artifact is not None:
        checkpoint.save_local_artifact(
            official_action_artifact,
            scope="official-actions",
        )
    actions = pd.concat([actions, official_actions], ignore_index=True).drop_duplicates(
        "event_id", keep="last"
    )
    provider_actions = actions.copy()
    reference_actions, reference_price_audit_artifact = (
        _krx_reference_price_adjustments(
            price_observations,
            actions,
        )
    )
    actions = pd.concat(
        [actions, reference_actions],
        ignore_index=True,
    ).drop_duplicates("event_id", keep="last")
    dart_dividends = fetch_opendart_dividend_decisions(
        scoped_catalog,
        start=start_date,
        end=completed,
        checkpoint_root=checkpoint.root,
        workers=krx_workers,
    )
    actions = _officialize_opendart_cash_dividends(
        actions,
        dart_dividends,
        start=start_date,
        end=completed,
    )
    checkpoint.save_local_artifact(
        dart_dividends.report,
        scope="opendart-action-audit",
    )
    corporate_action_audit_artifact = _kr_corporate_action_audit(
        actions,
        scoped_catalog,
        dart_dividends,
        reference_price_audit_artifact,
        start=start_date,
        end=completed,
        provider_failures=action_failures,
        provider_actions=provider_actions,
    )
    checkpoint.save_local_artifact(
        corporate_action_audit_artifact,
        scope="opendart-action-audit",
    )
    corporate_action_audit = json.loads(
        corporate_action_audit_artifact.content
    )
    if corporate_action_audit["status"] != "passed":
        raise RuntimeError(
            "KR corporate-action verification failed: "
            f"{corporate_action_audit['blocking_issue_count']} blocking "
            "issues; inspect the OpenDART action audit checkpoint."
        )

    factors = build_adjustment_factors(
        raw_prices,
        actions,
        source_version=f"kr-bootstrap:{completed}",
    )
    report_artifact = SourceArtifact(
        source="kr_provider_benchmark_report",
        source_url="local://benchmarks/current.json",
        retrieved_at=utc_now_iso(),
        content=benchmark_bytes,
        content_type="application/json",
    )
    source_archive = _archive_allowed_artifacts(
        repository,
        (
            *action_artifacts,
            *((official_action_artifact,) if official_action_artifact else ()),
            reference_price_audit_artifact,
            dart_dividends.report,
            corporate_action_audit_artifact,
            report_artifact,
        ),
        effective_date=completed,
    )

    index_coverage = _index_coverage_metadata(memberships)
    frames: dict[str, pd.DataFrame] = {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": raw_prices.drop(
            columns=["symbol", "security_name"],
            errors="ignore",
        ),
        "corporate_actions": actions,
        "adjustment_factors": factors,
        "index_constituent_anchors": anchors,
        "index_membership_events": events,
        "source_archive": source_archive,
    }
    versions: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for dataset in (
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "corporate_actions",
        "adjustment_factors",
        "index_constituent_anchors",
        "index_membership_events",
        "source_archive",
    ):
        dataset_metadata = _dataset_metadata(
            dataset,
            benchmark,
            index_coverage=index_coverage,
        )
        if dataset == "daily_price_raw":
            dataset_metadata["index_price_gap_policy_sha256"] = (
                index_price_gap_policy["policy_sha256"]
            )
        result = repository.write_frame(
            dataset,
            frames[dataset],
            completed_session=completed,
            incomplete_action_policy="block",
            metadata=dataset_metadata,
        )
        if result.conflict:
            raise RuntimeError(f"KR bootstrap pointer conflict: {result.conflict_path}")
        versions[dataset] = result.manifest.version
        row_counts[dataset] = len(frames[dataset])

    provisional = DataRelease.create(
        completed,
        versions,
        metadata=release_metadata("KR"),
    )
    lifecycle, lifecycle_coverage = _resolve_kr_lifecycle_candidates(
        repository,
        provisional,
        actions,
    )
    lifecycle_evidence_artifact = _kr_lifecycle_evidence_artifact(
        completed_session=completed,
        lifecycle_coverage=lifecycle_coverage,
        lifecycle_resolutions=lifecycle,
        corporate_actions=actions,
    )
    lifecycle_archive = _archive_allowed_artifacts(
        repository,
        (lifecycle_evidence_artifact,),
        effective_date=completed,
    )
    source_archive = pd.concat(
        [source_archive, lifecycle_archive],
        ignore_index=True,
    ).drop_duplicates("archive_id", keep="last")
    source_archive_result = repository.write_frame(
        "source_archive",
        source_archive,
        completed_session=completed,
        metadata=_dataset_metadata("source_archive", benchmark),
    )
    if source_archive_result.conflict:
        raise RuntimeError(
            "KR lifecycle source-archive pointer conflict: "
            + source_archive_result.conflict_path
        )
    versions["source_archive"] = source_archive_result.manifest.version
    row_counts["source_archive"] = len(source_archive)
    lifecycle_result = repository.write_frame(
        "lifecycle_resolutions",
        lifecycle,
        completed_session=completed,
        metadata={
            **_dataset_metadata("lifecycle_resolutions", benchmark),
            **lifecycle_coverage.manifest_metadata(),
            "evidence_report_sha256": (
                lifecycle_evidence_artifact.source_hash
            ),
        },
    )
    if lifecycle_result.conflict:
        raise RuntimeError(f"KR lifecycle pointer conflict: {lifecycle_result.conflict_path}")
    versions["lifecycle_resolutions"] = lifecycle_result.manifest.version
    row_counts["lifecycle_resolutions"] = len(lifecycle)

    cross_report = _cross_validation_report(
        benchmark,
        benchmark_bytes,
        versions,
        report_artifact.source_hash,
        completed,
        lifecycle_coverage=lifecycle_coverage,
        lifecycle_resolutions=lifecycle,
        index_price_gap_policy=index_price_gap_policy,
        lifecycle_evidence_report_sha256=(
            lifecycle_evidence_artifact.source_hash
        ),
        corporate_action_audit=corporate_action_audit_artifact,
        opendart_dividend_report=dart_dividends.report,
        official_action_evidence=official_action_artifact,
    )
    cross_result = repository.write_frame(
        "cross_validation_reports",
        cross_report,
        completed_session=completed,
        metadata=_dataset_metadata("cross_validation_reports", benchmark),
    )
    if cross_result.conflict:
        raise RuntimeError(f"KR cross-validation pointer conflict: {cross_result.conflict_path}")
    versions["cross_validation_reports"] = cross_result.manifest.version
    row_counts["cross_validation_reports"] = len(cross_report)

    validate_kr_repository(
        repository,
        release=None,
        require_current_release=False,
        index_price_gap_policy=index_price_gap_policy,
    )
    release = repository.commit_release(
        completed,
        versions,
        quality=DataQuality.DEGRADED if warnings else DataQuality.VALID,
        warnings=tuple(warnings),
        metadata=release_metadata(
            "KR",
            primary_provider=benchmark["selection"]["primary"],
            secondary_provider=benchmark["selection"]["secondary"],
            benchmark_report_sha256=sha256_bytes(benchmark_bytes),
            history_verification_sha256=sha256_bytes(
                canonical_json_bytes(
                    benchmark["membership_verification"]
                )
            ),
            history_verification_start=benchmark["sessions"]["start"],
            history_verification_end=benchmark["sessions"]["end"],
            reference_price_audit_sha256=(
                reference_price_audit_artifact.source_hash
            ),
            opendart_dividend_collection_sha256=(
                dart_dividends.report.source_hash
            ),
            corporate_action_verification_sha256=(
                corporate_action_audit_artifact.source_hash
            ),
            official_action_evidence_sha256=(
                official_action_artifact.source_hash
                if official_action_artifact is not None
                else ""
            ),
            license_policy=KR_LICENSE_POLICY,
            index_price_gap_policy=index_price_gap_policy,
        ),
    )
    return KrBootstrapResult(
        completed_session=completed,
        release_version=release.version,
        row_counts=row_counts,
        warnings=tuple(warnings),
        benchmark_report_sha256=sha256_bytes(benchmark_bytes),
        primary_provider=benchmark["selection"]["primary"],
        secondary_provider=benchmark["selection"]["secondary"],
    )


def sync_kr_market_data(
    repository: LocalDatasetRepository,
    *,
    expected_session: str | None = None,
    overlap_days: int = 7,
    sleep_seconds: float = 0.2,
    krx_workers: int = 4,
) -> KrBootstrapResult:
    """Append a KR overlap window and quarantine any historical revision."""

    validate_krx_official_configuration(require_membership=True)
    release, release_etag = repository.current_release()
    expected = expected_session or expected_completed_session("KR")
    if release is None:
        return bootstrap_kr_market_data(
            repository,
            end_date=expected,
            sleep_seconds=sleep_seconds,
            krx_workers=krx_workers,
        )
    if release.completed_session >= expected:
        benchmark, benchmark_bytes = _load_ready_benchmark(
            repository.root,
            start_session="2015-01-01",
            end_session=release.completed_session,
        )
        return KrBootstrapResult(
            completed_session=release.completed_session,
            release_version=release.version,
            row_counts={},
            warnings=("already_current",),
            benchmark_report_sha256=sha256_bytes(benchmark_bytes),
            primary_provider=benchmark["selection"]["primary"],
            secondary_provider=benchmark["selection"]["secondary"],
        )
    benchmark, benchmark_bytes = _load_ready_benchmark(
        repository.root,
        start_session="2015-01-01",
        end_session=expected,
    )
    overlap_start = (
        pd.Timestamp(release.completed_session) - pd.Timedelta(days=max(1, overlap_days))
    ).date().isoformat()
    sessions = _sessions_between(overlap_start, expected)
    checkpoint = KrCheckpointStore(repository.root / "state" / "daily" / expected)
    memberships = _collect_memberships(
        sessions,
        KR_PROFILES,
        checkpoint,
        sleep_seconds=sleep_seconds,
        allow_pre_inception=False,
        workers=krx_workers,
    )
    current_master = repository.read_frame("security_master", release.dataset_versions["security_master"])
    current_history = repository.read_frame("symbol_history", release.dataset_versions["symbol_history"])
    symbols = sorted(
        set(current_master["primary_symbol"].astype(str))
        | {
            symbol
            for snapshots in memberships.values()
            for snapshot in snapshots.values()
            for symbol in snapshot["symbols"]
        }
        | set(KR_BENCHMARK_SECURITIES.values())
        | _configured_official_action_symbols()
    )
    active_start = pd.to_datetime(
        current_master["active_from"], errors="coerce"
    ).min()
    catalog_start = (
        active_start.date().isoformat()
        if pd.notna(active_start)
        else "2015-01-01"
    )
    catalog = fetch_kr_identity_catalog(
        start=catalog_start,
        end=expected,
        include_dart=True,
    )
    scoped_catalog, missing = _catalog_for_symbols(
        catalog,
        symbols,
        catalog_start,
        expected,
    )
    if missing:
        raise RuntimeError("KR daily identity mapping is incomplete: " + ", ".join(missing))
    price_observations = _collect_krx_prices(
        sessions,
        symbols,
        scoped_catalog,
        checkpoint,
        sleep_seconds=sleep_seconds,
        workers=krx_workers,
    )
    scoped_catalog = _reconcile_catalog_with_krx_observations(
        scoped_catalog,
        price_observations,
        completed_session=expected,
    )
    _assert_krx_observations_classified(price_observations, context="daily sync")
    price_delta = _traded_krx_prices(price_observations).drop(
        columns=["symbol", "security_name"], errors="ignore"
    )
    price_result = repository.append_frame(
        "daily_price_raw",
        price_delta,
        completed_session=expected,
        metadata={**_dataset_metadata("daily_price_raw", benchmark), "overlap_days": overlap_days},
    )
    if price_result.conflict:
        raise RuntimeError(
            "KR historical price revision was quarantined; release was not advanced: "
            + price_result.conflict_path
        )

    fresh_master, fresh_history = _identity_datasets(scoped_catalog, symbols, "2015-01-01")
    master = _merge_by_key(current_master, fresh_master, ("security_id",))
    history = _merge_by_key(
        current_history,
        fresh_history,
        ("security_id", "symbol", "effective_from"),
    )
    master_result = repository.write_frame(
        "security_master",
        master,
        completed_session=expected,
        metadata=_dataset_metadata("security_master", benchmark),
    )
    history_result = repository.write_frame(
        "symbol_history",
        history,
        completed_session=expected,
        metadata=_dataset_metadata("symbol_history", benchmark),
    )
    if master_result.conflict:
        raise RuntimeError(f"KR security-master conflict: {master_result.conflict_path}")
    if history_result.conflict:
        raise RuntimeError(f"KR symbol-history conflict: {history_result.conflict_path}")

    current_anchors = repository.read_frame(
        "index_constituent_anchors", release.dataset_versions["index_constituent_anchors"]
    )
    current_events = repository.read_frame(
        "index_membership_events", release.dataset_versions["index_membership_events"]
    )
    event_delta = _incremental_membership_events(
        current_anchors,
        current_events,
        memberships,
        scoped_catalog,
    )
    # Advance the verified coverage boundary even on a no-change day. An empty
    # inherited delta proves that both official snapshots were checked through
    # ``expected``; leaving the old manifest in place would make that fact
    # indistinguishable from a skipped membership refresh.
    index_coverage = _index_coverage_metadata(memberships)
    anchor_starts = pd.to_datetime(
        current_anchors["anchor_date"], errors="coerce"
    ).dropna()
    if not anchor_starts.empty:
        index_coverage["official_coverage_start"] = (
            anchor_starts.min().date().isoformat()
        )
    index_coverage["official_coverage_end"] = expected
    event_result = repository.append_frame(
        "index_membership_events",
        event_delta,
        completed_session=expected,
        metadata=_dataset_metadata(
            "index_membership_events",
            benchmark,
            index_coverage=index_coverage,
        ),
    )
    if event_result.conflict:
        raise RuntimeError(f"KR membership event conflict: {event_result.conflict_path}")

    current_actions = repository.read_frame(
        "corporate_actions",
        release.dataset_versions["corporate_actions"],
    )
    actions_delta, action_artifacts, action_failures = fetch_eodhd_actions(
        scoped_catalog,
        start=overlap_start,
        end=expected,
        workers=6,
    )
    hard_action_failures = tuple(
        value
        for value in action_failures
        if not value.startswith(
            "known_unavailable_pre2019_delisted_actions:"
        )
    )
    if hard_action_failures:
        raise RuntimeError(
            "KR daily EODHD action collection failed for "
            f"{len(hard_action_failures)} symbols."
        )
    official_actions_delta, official_action_artifact = _load_kr_official_actions(
        scoped_catalog,
        # Re-read the complete reviewed evidence inventory so a newly verified
        # historical merger/delisting can close an old lifecycle candidate
        # without rebuilding unrelated price partitions.
        start=catalog_start,
        end=expected,
    )
    if official_action_artifact is not None:
        checkpoint.save_local_artifact(
            official_action_artifact,
            scope="official-actions",
        )
    actions_delta = pd.concat(
        [actions_delta, official_actions_delta], ignore_index=True
    ).drop_duplicates(
        "event_id", keep="last"
    )
    candidate_actions = pd.concat(
        [current_actions, actions_delta],
        ignore_index=True,
    ).drop_duplicates("event_id", keep="last")
    provider_actions = candidate_actions.copy()
    reference_actions_delta, incremental_reference_audit = (
        _krx_reference_price_adjustments(
            price_observations,
            candidate_actions,
        )
    )
    actions_delta = pd.concat(
        [actions_delta, reference_actions_delta],
        ignore_index=True,
    ).drop_duplicates("event_id", keep="last")
    candidate_actions = pd.concat(
        [current_actions, actions_delta],
        ignore_index=True,
    ).drop_duplicates("event_id", keep="last")
    previous_reference_audit = _release_source_artifact(
        repository,
        release,
        metadata_key="reference_price_audit_sha256",
    )
    reference_price_audit_artifact = (
        _combine_krx_reference_price_audits(
            previous_reference_audit,
            incremental_reference_audit,
        )
    )
    dart_dividends = fetch_opendart_dividend_decisions(
        scoped_catalog,
        start=catalog_start,
        end=expected,
        checkpoint_root=repository.root / "state" / "bootstrap",
        workers=krx_workers,
    )
    candidate_actions = _officialize_opendart_cash_dividends(
        candidate_actions,
        dart_dividends,
        start=catalog_start,
        end=expected,
    )
    corporate_action_audit_artifact = _kr_corporate_action_audit(
        candidate_actions,
        scoped_catalog,
        dart_dividends,
        reference_price_audit_artifact,
        start=catalog_start,
        end=expected,
        provider_failures=action_failures,
        provider_actions=provider_actions,
    )
    for artifact in (
        incremental_reference_audit,
        reference_price_audit_artifact,
        dart_dividends.report,
        corporate_action_audit_artifact,
    ):
        checkpoint.save_local_artifact(
            artifact,
            scope="opendart-action-audit",
        )
    corporate_action_audit = json.loads(
        corporate_action_audit_artifact.content
    )
    if corporate_action_audit["status"] != "passed":
        raise RuntimeError(
            "KR corporate-action verification failed during sync: "
            f"{corporate_action_audit['blocking_issue_count']} blocking "
            "issues."
        )
    action_result = repository.write_frame(
        "corporate_actions",
        candidate_actions,
        completed_session=expected,
        incomplete_action_policy="block",
        metadata=_dataset_metadata("corporate_actions", benchmark),
    )
    if action_result.conflict:
        raise RuntimeError(
            f"KR corporate-action conflict: {action_result.conflict_path}"
        )

    all_prices = repository.read_frame("daily_price_raw")
    all_actions = repository.read_frame("corporate_actions")
    all_events = repository.read_frame("index_membership_events")
    existing_index_price_gap_policy = release.metadata.get(
        "index_price_gap_policy"
    )
    if isinstance(existing_index_price_gap_policy, dict):
        previous_cross = repository.read_frame(
            "cross_validation_reports",
            release.dataset_versions["cross_validation_reports"],
        )
        validate_index_price_gap_policy(
            existing_index_price_gap_policy,
            previous_cross,
        )
    else:
        existing_index_price_gap_policy = None
    index_price_gap_policy = _krx_index_price_gap_policy(
        current_anchors,
        all_events,
        all_prices,
        price_observations,
        existing_policy=existing_index_price_gap_policy,
    )
    factors = build_adjustment_factors(
        all_prices,
        all_actions,
        source_version=f"{repository.current_manifest('daily_price_raw').version}+{repository.current_manifest('corporate_actions').version}",
    )
    factor_result = repository.write_frame(
        "adjustment_factors",
        factors,
        completed_session=expected,
        metadata=_dataset_metadata("adjustment_factors", benchmark),
    )
    if factor_result.conflict:
        raise RuntimeError(f"KR factor conflict: {factor_result.conflict_path}")

    report_artifact = SourceArtifact(
        source="kr_provider_benchmark_report",
        source_url="local://benchmarks/current.json",
        retrieved_at=utc_now_iso(),
        content=benchmark_bytes,
        content_type="application/json",
    )
    archive_delta = _archive_allowed_artifacts(
        repository,
        (
            *action_artifacts,
            *((official_action_artifact,) if official_action_artifact else ()),
            reference_price_audit_artifact,
            dart_dividends.report,
            corporate_action_audit_artifact,
            report_artifact,
        ),
        effective_date=expected,
    )
    if not archive_delta.empty:
        archive_result = repository.append_frame(
            "source_archive",
            archive_delta,
            completed_session=expected,
            metadata=_dataset_metadata("source_archive", benchmark),
        )
        if archive_result.conflict:
            raise RuntimeError(f"KR source archive conflict: {archive_result.conflict_path}")

    versions = {
        dataset: manifest.version
        for dataset in KR_REQUIRED_DATASETS
        if (manifest := repository.current_manifest(dataset)) is not None
    }
    provisional = DataRelease.create(expected, versions, metadata=release_metadata("KR"))
    lifecycle, lifecycle_coverage = _resolve_kr_lifecycle_candidates(
        repository,
        provisional,
        all_actions,
    )
    lifecycle_evidence_artifact = _kr_lifecycle_evidence_artifact(
        completed_session=expected,
        lifecycle_coverage=lifecycle_coverage,
        lifecycle_resolutions=lifecycle,
        corporate_actions=all_actions,
    )
    lifecycle_archive = _archive_allowed_artifacts(
        repository,
        (lifecycle_evidence_artifact,),
        effective_date=expected,
    )
    lifecycle_archive_result = repository.append_frame(
        "source_archive",
        lifecycle_archive,
        completed_session=expected,
        metadata=_dataset_metadata("source_archive", benchmark),
    )
    if lifecycle_archive_result.conflict:
        raise RuntimeError(
            "KR lifecycle source-archive conflict: "
            + lifecycle_archive_result.conflict_path
        )
    versions["source_archive"] = lifecycle_archive_result.manifest.version
    lifecycle_result = repository.write_frame(
        "lifecycle_resolutions",
        lifecycle,
        completed_session=expected,
        metadata={
            **_dataset_metadata("lifecycle_resolutions", benchmark),
            **lifecycle_coverage.manifest_metadata(),
            "evidence_report_sha256": (
                lifecycle_evidence_artifact.source_hash
            ),
        },
    )
    if lifecycle_result.conflict:
        raise RuntimeError(f"KR lifecycle conflict: {lifecycle_result.conflict_path}")
    versions["lifecycle_resolutions"] = lifecycle_result.manifest.version
    cross = _cross_validation_report(
        benchmark,
        benchmark_bytes,
        versions,
        sha256_bytes(benchmark_bytes),
        expected,
        lifecycle_coverage=lifecycle_coverage,
        lifecycle_resolutions=lifecycle,
        index_price_gap_policy=index_price_gap_policy,
        lifecycle_evidence_report_sha256=(
            lifecycle_evidence_artifact.source_hash
        ),
        corporate_action_audit=corporate_action_audit_artifact,
        opendart_dividend_report=dart_dividends.report,
        official_action_evidence=official_action_artifact,
    )
    cross_result = repository.append_frame(
        "cross_validation_reports",
        cross,
        completed_session=expected,
        metadata=_dataset_metadata("cross_validation_reports", benchmark),
    )
    if cross_result.conflict:
        raise RuntimeError(f"KR cross-validation conflict: {cross_result.conflict_path}")
    versions = {
        dataset: repository.current_manifest(dataset).version
        for dataset in KR_REQUIRED_DATASETS
        if repository.current_manifest(dataset) is not None
    }
    validate_kr_repository(
        repository,
        release=None,
        require_current_release=False,
        index_price_gap_policy=index_price_gap_policy,
    )
    final = repository.commit_release(
        expected,
        versions,
        quality=DataQuality.VALID,
        warnings=(),
        expected_etag=release_etag,
        metadata=release_metadata(
            "KR",
            primary_provider=benchmark["selection"]["primary"],
            secondary_provider=benchmark["selection"]["secondary"],
            benchmark_report_sha256=sha256_bytes(benchmark_bytes),
            history_verification_sha256=sha256_bytes(
                canonical_json_bytes(
                    benchmark["membership_verification"]
                )
            ),
            history_verification_start=benchmark["sessions"]["start"],
            history_verification_end=benchmark["sessions"]["end"],
            reference_price_audit_sha256=(
                reference_price_audit_artifact.source_hash
            ),
            opendart_dividend_collection_sha256=(
                dart_dividends.report.source_hash
            ),
            corporate_action_verification_sha256=(
                corporate_action_audit_artifact.source_hash
            ),
            official_action_evidence_sha256=(
                official_action_artifact.source_hash
                if official_action_artifact is not None
                else ""
            ),
            license_policy=KR_LICENSE_POLICY,
            index_price_gap_policy=index_price_gap_policy,
        ),
    )
    return KrBootstrapResult(
        completed_session=expected,
        release_version=final.version,
        row_counts={
            "daily_price_raw": len(price_delta),
            "index_membership_events": len(event_delta),
            "corporate_actions": len(actions_delta),
            "adjustment_factors": len(factors),
        },
        warnings=(),
        benchmark_report_sha256=sha256_bytes(benchmark_bytes),
        primary_provider=benchmark["selection"]["primary"],
        secondary_provider=benchmark["selection"]["secondary"],
    )


def validate_kr_repository(
    repository: LocalDatasetRepository,
    *,
    release: DataRelease | None = None,
    require_current_release: bool = True,
    index_price_gap_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if repository.market != "KR":
        raise ValueError("KR validation requires repository.market='KR'.")
    current, _ = repository.current_release()
    if release is None and require_current_release:
        release = current
    if require_current_release and release is None:
        raise RuntimeError("KR current release is missing.")
    versions = release.dataset_versions if release is not None else {
        dataset: manifest.version
        for dataset in KR_REQUIRED_DATASETS
        if (manifest := repository.current_manifest(dataset)) is not None
    }
    missing = sorted(set(KR_REQUIRED_DATASETS) - set(versions))
    if missing:
        raise RuntimeError("KR release is missing datasets: " + ", ".join(missing))
    if release is not None:
        metadata = dict(release.metadata)
        expected = release_metadata("KR")
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(
                    f"KR release metadata mismatch: {key}={metadata.get(key)!r}, expected {value!r}"
                )
    dataset_stats: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}
    completed_sessions: list[str] = []
    for dataset, version in versions.items():
        manifest = repository.manifest_for_version(dataset, version)
        completed_sessions.append(str(manifest.completed_session))
        for item in repository.manifest_chain(dataset, version):
            validate_manifest_files(
                repository.root / repository.version_prefix(dataset, item.version), item
            ).raise_for_errors()
        frame = repository.read_frame(dataset, version)
        frames[dataset] = frame
        validate_dataset(
            dataset,
            frame,
            incomplete_action_policy="block",
            completed_session=manifest.completed_session,
            market="KR",
        ).raise_for_errors()
        dataset_stats[dataset] = {"version": version, "rows": len(frame)}
    master = frames["security_master"]
    if master["security_id"].astype(str).str.startswith("KR:UNMAPPED:").any():
        raise RuntimeError("KR release contains unmapped ticker-based security identities.")
    prices = frames["daily_price_raw"]
    if not prices["currency"].astype(str).eq("KRW").all():
        raise RuntimeError("KR daily_price_raw contains non-KRW rows.")
    cross = frames["cross_validation_reports"]
    if cross.empty or not cross["status"].astype(str).str.lower().eq("passed").any():
        raise RuntimeError("KR release has no passed cross-validation report.")
    required_history_columns = {
        "verification_schema",
        "verification_scope",
        "verification_start",
        "verification_end",
        "membership_verification_sha256",
        "membership_expected_snapshot_count",
        "membership_observed_snapshot_count",
        "membership_missing_snapshot_count",
        "membership_replay_mismatch_count",
        "membership_blocking_issue_count",
        "membership_invalid_artifact_count",
        "membership_event_count",
        "snapshot_inventory_sha256",
        "membership_event_inventory_sha256",
        "canonical_price_provider",
        "independent_price_provider",
        "independent_price_hard_gate_passed",
        "independent_price_source_artifacts_complete",
        "independent_price_participants_json",
        "price_ohlc_within_one_tick_rate",
        "price_volume_exact_rate",
        "price_provider_disagreement_count",
        "price_verification_assignment_sha256",
        "corporate_action_verification_sha256",
        "reference_price_audit_sha256",
        "opendart_dividend_collection_sha256",
        "official_action_evidence_sha256",
        "corporate_action_inventory_sha256",
        "corporate_action_decision_inventory_sha256",
        "corporate_action_count",
        "reference_price_discontinuity_count",
        "reference_price_generated_adjustment_count",
        "reference_price_unresolved_count",
        "opendart_dividend_decision_count",
        "matched_dividend_count",
        "missing_dividend_count",
        "unmatched_provider_dividend_count",
        "known_provider_action_gap_count",
        "resolved_provider_action_gap_count",
        "corporate_action_blocking_issue_count",
    }
    if missing_history_columns := required_history_columns - set(cross):
        raise RuntimeError(
            "KR cross-validation report predates the full-history contract: "
            + ", ".join(sorted(missing_history_columns))
        )
    passed_cross = cross.loc[
        cross["status"].astype(str).str.lower().eq("passed")
    ].copy()
    passed_cross["_validated_at"] = pd.to_datetime(
        passed_cross["validated_at"],
        errors="coerce",
        utc=True,
    )
    latest_cross = passed_cross.sort_values(
        ["_validated_at", "report_id"],
        kind="stable",
    ).iloc[-1]
    completed = (
        str(release.completed_session)
        if release is not None
        else max(completed_sessions)
    )
    anchor_start_values = pd.to_datetime(
        frames["index_constituent_anchors"]["anchor_date"],
        errors="coerce",
    ).dropna()
    if anchor_start_values.empty:
        raise RuntimeError("KR release has no historical index anchor.")
    history_start = anchor_start_values.min().date().isoformat()
    history_sessions = _sessions_between(history_start, completed)
    expected_membership_snapshots = sum(
        1
        for profile in KR_PROFILES
        for session in history_sessions
        if session >= str(KR_INDEX_DEFINITIONS[profile]["announcement_date"])
    )

    def cross_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes"}
        return bool(value)

    history_failures: list[str] = []
    if str(latest_cross["verification_schema"]) != KR_HISTORY_VERIFICATION_SCHEMA:
        history_failures.append("verification_schema")
    if str(latest_cross["verification_scope"]) != "historical_full":
        history_failures.append("verification_scope")
    if str(latest_cross["verification_start"]) > history_start:
        history_failures.append("verification_start")
    if str(latest_cross["verification_end"]) < completed:
        history_failures.append("verification_end")
    if int(latest_cross["overlap_session_count"]) != len(history_sessions):
        history_failures.append("overlap_session_count")
    if int(latest_cross["membership_expected_snapshot_count"]) != (
        expected_membership_snapshots
    ):
        history_failures.append("membership_expected_snapshot_count")
    if int(latest_cross["membership_observed_snapshot_count"]) != (
        expected_membership_snapshots
    ):
        history_failures.append("membership_observed_snapshot_count")
    for column in (
        "membership_missing_snapshot_count",
        "membership_replay_mismatch_count",
        "membership_blocking_issue_count",
        "membership_invalid_artifact_count",
        "price_unresolved_count",
        "price_mismatch_count",
        "reference_price_unresolved_count",
        "missing_dividend_count",
        "unmatched_provider_dividend_count",
        "corporate_action_blocking_issue_count",
    ):
        if int(latest_cross[column]) != 0:
            history_failures.append(column)
    if int(latest_cross["membership_event_count"]) != len(
        frames["index_membership_events"]
    ):
        history_failures.append("membership_event_count")
    if int(latest_cross["price_pass_count"]) != int(
        latest_cross["price_target_count"]
    ):
        history_failures.append("price_pass_count")
    if float(
        latest_cross["price_ohlc_within_one_tick_rate"]
    ) != 1.0:
        history_failures.append(
            "price_ohlc_within_one_tick_rate"
        )
    if float(latest_cross["price_volume_exact_rate"]) != 1.0:
        history_failures.append("price_volume_exact_rate")
    if int(latest_cross["corporate_action_count"]) != len(
        frames["corporate_actions"]
    ):
        history_failures.append("corporate_action_count")
    if int(latest_cross["matched_dividend_count"]) != int(
        latest_cross["opendart_dividend_decision_count"]
    ):
        history_failures.append("matched_dividend_count")
    if int(latest_cross["resolved_provider_action_gap_count"]) != int(
        latest_cross["known_provider_action_gap_count"]
    ):
        history_failures.append("resolved_provider_action_gap_count")
    if str(latest_cross["canonical_price_provider"]) != "krx":
        history_failures.append("canonical_price_provider")
    independent_provider = str(latest_cross["independent_price_provider"])
    if not independent_provider.startswith("composite:"):
        history_failures.append("independent_price_provider")
    try:
        independent_participants = json.loads(
            str(latest_cross["independent_price_participants_json"])
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        independent_participants = []
    if (
        not isinstance(independent_participants, list)
        or len(set(map(str, independent_participants))) < 2
        or independent_provider
        != "composite:"
        + "+".join(
            sorted(set(map(str, independent_participants)))
        )
    ):
        history_failures.append(
            "independent_price_participants_json"
        )
    if not cross_bool(latest_cross["independent_price_hard_gate_passed"]):
        history_failures.append("independent_price_hard_gate_passed")
    if not cross_bool(
        latest_cross["independent_price_source_artifacts_complete"]
    ):
        history_failures.append(
            "independent_price_source_artifacts_complete"
        )
    for column in (
        "membership_verification_sha256",
        "snapshot_inventory_sha256",
        "membership_event_inventory_sha256",
        "source_hash",
        "report_archive_id",
        "corporate_action_verification_sha256",
        "reference_price_audit_sha256",
        "opendart_dividend_collection_sha256",
        "official_action_evidence_sha256",
        "corporate_action_inventory_sha256",
        "corporate_action_decision_inventory_sha256",
        "price_verification_assignment_sha256",
    ):
        if not _valid_sha256(latest_cross[column]):
            history_failures.append(column)
    if release is not None:
        benchmark_sha = str(
            release.metadata.get("benchmark_report_sha256") or ""
        ).lower()
        if (
            not _valid_sha256(benchmark_sha)
            or str(latest_cross["source_hash"]).lower() != benchmark_sha
        ):
            history_failures.append("release_benchmark_report_sha256")
        if str(
            release.metadata.get("history_verification_sha256") or ""
        ).lower() != str(
            latest_cross["membership_verification_sha256"]
        ).lower():
            history_failures.append("release_history_verification_sha256")
        if str(
            release.metadata.get("history_verification_start") or ""
        ) != str(latest_cross["verification_start"]):
            history_failures.append("release_history_verification_start")
        if str(
            release.metadata.get("history_verification_end") or ""
        ) != str(latest_cross["verification_end"]):
            history_failures.append("release_history_verification_end")
        for metadata_key, cross_column in (
            (
                "reference_price_audit_sha256",
                "reference_price_audit_sha256",
            ),
            (
                "opendart_dividend_collection_sha256",
                "opendart_dividend_collection_sha256",
            ),
            (
                "corporate_action_verification_sha256",
                "corporate_action_verification_sha256",
            ),
            (
                "official_action_evidence_sha256",
                "official_action_evidence_sha256",
            ),
        ):
            if str(
                release.metadata.get(metadata_key) or ""
            ).lower() != str(latest_cross[cross_column]).lower():
                history_failures.append(f"release_{metadata_key}")
    action_inventory = (
        frames["corporate_actions"][
            [
                "event_id",
                "security_id",
                "action_type",
                "effective_date",
                "ex_date",
                "record_date",
                "cash_amount",
                "ratio",
                "source_hash",
            ]
        ]
        .fillna("")
        .sort_values("event_id", kind="stable")
        .to_dict("records")
    )
    if sha256_bytes(canonical_json_bytes(action_inventory)) != str(
        latest_cross["corporate_action_inventory_sha256"]
    ).lower():
        history_failures.append("corporate_action_inventory_sha256")
    if history_failures:
        raise RuntimeError(
            "KR full-history verification gate failed: "
            + ", ".join(sorted(set(history_failures)))
        )

    checkpoint = KrCheckpointStore(repository.root / "state" / "bootstrap")
    if checkpoint.identity_catalog_path().is_file() and (
        checkpoint.root / "memberships"
    ).is_dir():
        local_memberships: dict[str, dict[str, dict[str, Any]]] = {
            profile: {} for profile in KR_PROFILES
        }
        for profile in KR_PROFILES:
            inception = str(KR_INDEX_DEFINITIONS[profile]["announcement_date"])
            for session in history_sessions:
                if session < inception:
                    continue
                snapshot = checkpoint.load_membership(profile, session)
                if snapshot is not None:
                    local_memberships[profile][session] = snapshot
        local_verification = verify_kr_membership_history(
            local_memberships,
            _identity_catalog_from_release_frames(
                master,
                frames["symbol_history"],
            ),
            frames["index_constituent_anchors"],
            frames["index_membership_events"],
            sessions=history_sessions,
            profiles=KR_PROFILES,
            checkpoint=checkpoint,
        )
        local_sha = sha256_bytes(canonical_json_bytes(local_verification))
        if (
            local_verification["status"] != "passed"
            or local_sha
            != str(latest_cross["membership_verification_sha256"])
        ):
            raise RuntimeError(
                "KR local official snapshots do not reproduce the release "
                "anchor/event verification report."
            )
        dataset_stats["__pit_membership_verification__"] = {
            "status": "passed",
            "expected_snapshot_count": expected_membership_snapshots,
            "daily_replay_mismatch_count": 0,
            "membership_verification_sha256": local_sha,
        }
    if index_price_gap_policy is None and release is not None:
        raw_policy = release.metadata.get("index_price_gap_policy")
        if isinstance(raw_policy, dict):
            index_price_gap_policy = raw_policy
    allowed_index_price_gap_ids: tuple[str, ...] = ()
    if index_price_gap_policy is not None:
        allowed_index_price_gap_ids = validate_index_price_gap_policy(
            index_price_gap_policy,
            cross,
        )
    archive = frames["source_archive"]
    if "license_class" not in archive or not archive["license_class"].astype(str).eq(
        "allowed_private"
    ).all():
        raise RuntimeError("KR source_archive contains non-publishable license classes.")
    archive_ids = set(archive["archive_id"].astype(str).str.lower())
    required_archive_ids = {
        str(latest_cross[column]).lower()
        for column in (
            "report_archive_id",
            "corporate_action_verification_sha256",
            "reference_price_audit_sha256",
            "opendart_dividend_collection_sha256",
            "official_action_evidence_sha256",
        )
    }
    if not required_archive_ids.issubset(archive_ids):
        raise RuntimeError(
            "KR full-history or corporate-action verification evidence is "
            "not pinned in source_archive."
        )
    archived_content: dict[str, bytes] = {}
    for source_hash in required_archive_ids:
        matches = archive.loc[
            archive["archive_id"].astype(str).str.lower().eq(source_hash)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "KR verification archive identity is not unique: "
                + source_hash
            )
        try:
            content = gzip.decompress(
                (
                    repository.root
                    / str(matches.iloc[0]["object_path"])
                ).read_bytes()
            )
        except (OSError, EOFError):
            raise RuntimeError(
                "KR verification archive is unreadable: " + source_hash
            ) from None
        if sha256_bytes(content) != source_hash:
            raise RuntimeError(
                "KR verification archive hash mismatch: " + source_hash
            )
        archived_content[source_hash] = content
    try:
        action_report = json.loads(
            archived_content[
                str(
                    latest_cross[
                        "corporate_action_verification_sha256"
                    ]
                ).lower()
            ]
        )
        reference_report = json.loads(
            archived_content[
                str(latest_cross["reference_price_audit_sha256"]).lower()
            ]
        )
        dividend_report = json.loads(
            archived_content[
                str(
                    latest_cross[
                        "opendart_dividend_collection_sha256"
                    ]
                ).lower()
            ]
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise RuntimeError(
            "KR corporate-action verification archives are invalid JSON."
        ) from None
    if (
        action_report.get("status") != "passed"
        or reference_report.get("status") != "passed"
        or dividend_report.get("status") != "passed"
        or str(
            action_report.get("reference_price_audit_sha256") or ""
        ).lower()
        != str(latest_cross["reference_price_audit_sha256"]).lower()
        or str(
            action_report.get("opendart_collection_sha256") or ""
        ).lower()
        != str(
            latest_cross["opendart_dividend_collection_sha256"]
        ).lower()
    ):
        raise RuntimeError(
            "KR corporate-action verification archive binding failed."
        )
    if (
        reference_report.get("schema")
        != "krx_reference_price_audit/v1"
        or int(reference_report.get("unresolved_count") or 0) != 0
        or int(
            reference_report.get(
                "provider_ratio_unaccounted_count"
            )
            or 0
        )
        != 0
        or int(
            reference_report.get(
                "provider_ratio_outside_audit_window_count"
            )
            or 0
        )
        != 0
    ):
        raise RuntimeError(
            "KR KRX reference-price audit is incomplete."
        )
    reference_records = [
        dict(value)
        for value in reference_report.get("records") or ()
    ]
    reference_record_keys = [
        (
            str(value.get("security_id") or ""),
            str(value.get("session") or ""),
            str(
                value.get("record_kind")
                or "reference_discontinuity"
            ),
        )
        for value in reference_records
    ]
    if (
        any(not key[0] or not key[1] for key in reference_record_keys)
        or len(reference_record_keys)
        != len(set(reference_record_keys))
        or any(
            key[2] == "provider_ratio_outside_audit_window"
            for key in reference_record_keys
        )
    ):
        raise RuntimeError(
            "KR KRX reference-price audit record inventory is invalid."
        )
    related_ratio_event_ids = {
        str(event_id)
        for value in reference_records
        for event_id in value.get("related_ratio_event_ids") or ()
        if str(event_id)
    }
    provider_ratio_actions = frames["corporate_actions"].loc[
        frames["corporate_actions"]["action_type"]
        .astype(str)
        .isin({"split", "capital_reduction", "stock_dividend"})
    ]
    provider_ratio_event_ids = set(
        provider_ratio_actions["event_id"].astype(str)
    )
    reference_action_records = [
        value
        for value in reference_records
        if str(
            value.get("record_kind")
            or "reference_discontinuity"
        )
        in {
            "reference_discontinuity",
            "provider_ratio_restatement_noop",
        }
    ]
    reference_action_event_ids = {
        str(value.get("event_id") or "")
        for value in reference_action_records
        if str(value.get("event_id") or "")
    }
    stored_reference_actions = frames["corporate_actions"].loc[
        frames["corporate_actions"]["action_type"]
        .astype(str)
        .eq("reference_price_adjustment")
    ]
    stored_reference_event_ids = set(
        stored_reference_actions["event_id"].astype(str)
    )
    record_kind_counts = pd.Series(
        [
            str(
                value.get("record_kind")
                or "reference_discontinuity"
            )
            for value in reference_records
        ],
        dtype=str,
    ).value_counts()
    outside_factor_action_count = sum(
        len(value.get("related_ratio_event_ids") or ())
        for value in reference_records
        if value.get("record_kind")
        == "provider_ratio_outside_factor_domain"
    )
    reference_count_checks = {
        "reference_discontinuity_count": int(
            record_kind_counts.get("reference_discontinuity", 0)
        ),
        "reference_adjustment_event_count": len(
            reference_action_event_ids
        ),
        "provider_ratio_action_count": len(
            provider_ratio_event_ids
        ),
        "provider_ratio_accounted_action_count": len(
            related_ratio_event_ids
        ),
        "provider_ratio_outside_factor_domain_count": int(
            outside_factor_action_count
        ),
        "provider_ratio_restatement_noop_count": int(
            record_kind_counts.get(
                "provider_ratio_restatement_noop",
                0,
            )
        ),
    }
    inconsistent_reference_counts = [
        key
        for key, expected_value in reference_count_checks.items()
        if int(reference_report.get(key) or 0) != expected_value
    ]
    if (
        inconsistent_reference_counts
        or related_ratio_event_ids != provider_ratio_event_ids
        or reference_action_event_ids != stored_reference_event_ids
        or (
            not stored_reference_actions.empty
            and not stored_reference_actions["official"]
            .map(cross_bool)
            .all()
        )
    ):
        raise RuntimeError(
            "KR KRX reference-price audit does not account for the "
            "stored ratio actions: "
            + ", ".join(inconsistent_reference_counts)
        )
    if reference_action_records:
        factor_checks = pd.DataFrame(
            [
                {
                    "security_id": str(value["security_id"]),
                    "session": pd.Timestamp(value["session"]),
                    "official_ratio": pd.to_numeric(
                        value.get("official_ratio"),
                        errors="coerce",
                    ),
                }
                for value in reference_action_records
            ]
        )
        factors = frames["adjustment_factors"][
            ["security_id", "session", "split_factor"]
        ].copy()
        factors["session"] = pd.to_datetime(
            factors["session"],
            errors="coerce",
        ).dt.normalize()
        factors["split_factor"] = pd.to_numeric(
            factors["split_factor"],
            errors="coerce",
        )
        factors = factors.sort_values(
            ["security_id", "session"],
            kind="stable",
        )
        factors["previous_split_factor"] = factors.groupby(
            factors["security_id"].astype(str),
            sort=False,
        )["split_factor"].shift(1)
        factor_checks = factor_checks.merge(
            factors,
            on=["security_id", "session"],
            how="left",
            validate="one_to_one",
        )
        factor_checks["actual_transition"] = (
            factor_checks["previous_split_factor"]
            / factor_checks["split_factor"]
        )
        factor_checks["expected_transition"] = (
            1.0 / factor_checks["official_ratio"]
        )
        factor_checks["transition_error"] = (
            factor_checks["actual_transition"]
            - factor_checks["expected_transition"]
        ).abs()
        invalid_factor_transitions = factor_checks.loc[
            factor_checks[
                [
                    "official_ratio",
                    "split_factor",
                    "previous_split_factor",
                    "actual_transition",
                    "expected_transition",
                ]
            ]
            .isna()
            .any(axis=1)
            | factor_checks["official_ratio"].le(0)
            | factor_checks["split_factor"].le(0)
            | factor_checks["previous_split_factor"].le(0)
            | factor_checks["transition_error"].gt(
                1e-12
                + 1e-9
                * factor_checks["expected_transition"].abs()
            )
        ]
        if not invalid_factor_transitions.empty:
            raise RuntimeError(
                "KR adjustment factors do not reproduce every official "
                "KRX reference-price transition."
            )
    raw_dividend_hashes = [
        str(value).lower()
        for value in dividend_report.get("raw_artifact_hashes") or ()
    ]
    if (
        len(raw_dividend_hashes)
        != int(dividend_report.get("raw_artifact_count") or 0)
        or len(set(raw_dividend_hashes)) != len(raw_dividend_hashes)
        or not raw_dividend_hashes
        or not all(_valid_sha256(value) for value in raw_dividend_hashes)
        or sha256_bytes(
            canonical_json_bytes(sorted(raw_dividend_hashes))
        )
        != str(
            dividend_report.get("raw_artifact_inventory_sha256") or ""
        ).lower()
    ):
        raise RuntimeError(
            "KR OpenDART raw-artifact inventory is incomplete or invalid."
        )
    local_dividend_evidence = (
        repository.root
        / "state"
        / "bootstrap"
        / "evidence_local"
        / "opendart-actions"
    )
    if local_dividend_evidence.is_dir():
        for source_hash in raw_dividend_hashes:
            matches = sorted(
                local_dividend_evidence.glob(
                    f"{source_hash}.*.gz"
                )
            )
            if len(matches) != 1:
                raise RuntimeError(
                    "KR OpenDART local evidence is missing or ambiguous: "
                    + source_hash
                )
            try:
                content = gzip.decompress(matches[0].read_bytes())
            except (OSError, EOFError):
                raise RuntimeError(
                    "KR OpenDART local evidence is unreadable: "
                    + source_hash
                ) from None
            if sha256_bytes(content) != source_hash:
                raise RuntimeError(
                    "KR OpenDART local evidence hash mismatch: "
                    + source_hash
                )
    coverage_release = release or DataRelease.create(
        max(completed_sessions),
        versions,
        metadata=release_metadata("KR"),
    )
    candidates = build_lifecycle_candidates(
        repository,
        release=coverage_release,
        stale_days=30,
    )
    candidate_frame = pd.DataFrame(
        [
            {
                "candidate_id": lifecycle_candidate_id(
                    candidate.security_id,
                    candidate.last_price_date,
                    selection_rule=KR_LIFECYCLE_SELECTION_RULE,
                ),
                "security_id": candidate.security_id,
                "last_price_date": candidate.last_price_date,
            }
            for candidate in candidates
        ],
        columns=("candidate_id", "security_id", "last_price_date"),
    )
    lifecycle_coverage = validate_lifecycle_coverage(
        candidate_frame,
        frames["lifecycle_resolutions"],
        frames["corporate_actions"],
        completed_session=coverage_release.completed_session,
        selection_rule=KR_LIFECYCLE_SELECTION_RULE,
    )
    if not lifecycle_coverage.valid:
        raise RuntimeError(
            "KR lifecycle coverage validation failed: "
            + "; ".join(issue.message for issue in lifecycle_coverage.issues)
        )
    dataset_stats["__lifecycle_coverage__"] = (
        lifecycle_coverage.manifest_metadata()
    )
    if repository.conflicts():
        raise RuntimeError("KR repository contains quarantined conflicts.")
    cross_report = validate_repository_snapshot(
        _FrameRepositoryView(frames),
        allowed_index_price_gap_ids=allowed_index_price_gap_ids,
    )
    cross_report.raise_for_errors()
    return {
        "market": "KR",
        "calendar": "XKRX",
        "release": release.version if release is not None else "candidate",
        "datasets": dataset_stats,
        "cross_dataset_issue_count": len(cross_report.issues),
    }


def _collect_memberships(
    sessions: tuple[str, ...],
    profiles: tuple[str, ...],
    checkpoint: KrCheckpointStore,
    *,
    sleep_seconds: float,
    allow_pre_inception: bool,
    workers: int = 4,
) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {profile: {} for profile in profiles}
    for profile in profiles:
        missing_sessions: list[str] = []
        for session in sessions:
            cached = checkpoint.load_membership(profile, session)
            if cached is not None:
                output[profile][session] = cached
                continue
            if (
                allow_pre_inception
                and session
                < str(KR_INDEX_DEFINITIONS[profile]["announcement_date"])
            ):
                continue
            missing_sessions.append(session)

        fetched: dict[str, dict[str, Any]] = {}
        failures: dict[str, Exception] = {}
        if missing_sessions:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = {
                    executor.submit(
                        fetch_krx_membership,
                        profile,
                        session,
                        sleep_seconds=sleep_seconds,
                    ): session
                    for session in missing_sessions
                }
                for future in as_completed(futures):
                    session = futures[future]
                    try:
                        symbols, artifact = future.result()
                        fetched[session] = checkpoint.save_membership(
                            profile,
                            session,
                            symbols,
                            artifact,
                        )
                    except Exception as exc:
                        failures[session] = exc

        known_sessions = sorted((*output[profile], *fetched))
        first_known = known_sessions[0] if known_sessions else ""
        actionable_failures = {
            session: exc
            for session, exc in failures.items()
            if not (
                allow_pre_inception
                and profile == "kosdaq150"
                and first_known
                and session < first_known
            )
        }
        if actionable_failures:
            failed_session = min(actionable_failures)
            exc = actionable_failures[failed_session]
            raise RuntimeError(
                f"KRX membership fetch failed for {profile} on {failed_session}: "
                f"{type(exc).__name__}: {exc}"
            ) from None
        for session in sorted(fetched):
            output[profile][session] = fetched[session]
        if not output[profile]:
            raise RuntimeError(f"No verified KRX membership history for {profile}.")
    return output


def _collect_krx_prices(
    sessions: tuple[str, ...],
    symbols: Iterable[str],
    catalog: KrIdentityCatalog,
    checkpoint: KrCheckpointStore,
    *,
    sleep_seconds: float,
    workers: int = 4,
) -> pd.DataFrame:
    symbol_tuple = tuple(sorted(set(symbols)))
    by_session: dict[str, pd.DataFrame] = {}
    missing_sessions: list[str] = []
    for session in sessions:
        cached = checkpoint.load_prices(session)
        if cached is not None:
            cached, upgraded = _upgrade_cached_krx_price_checkpoint(
                cached,
                session=session,
                catalog=catalog,
                checkpoint=checkpoint,
            )
            if cached is not None and upgraded:
                checkpoint.rewrite_prices(
                    session,
                    cached,
                    selected_symbols=symbol_tuple,
                )
        if cached is not None and _krx_price_checkpoint_covers_symbols(
            cached,
            symbol_tuple,
            catalog,
            session,
        ):
            cached = _classify_cached_delisting_effective_absences(
                cached,
                catalog,
                session,
            )
            if "observation_status" not in cached:
                cached["observation_status"] = KRX_TRADED_STATUS
            by_session[session] = cached
        else:
            missing_sessions.append(session)
    if missing_sessions and workers <= 1:
        for session in missing_sessions:
            for attempt in range(3):
                try:
                    frame, artifact = fetch_krx_session_prices(
                        session,
                        catalog,
                        symbols=symbol_tuple,
                        sleep_seconds=sleep_seconds,
                    )
                    break
                except Exception as exc:
                    if attempt < 2:
                        time.sleep(float(2**attempt))
                        continue
                    raise RuntimeError(
                        f"KRX price fetch failed on {session}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from None
            if frame.empty:
                raise RuntimeError(
                    "KRX price fetch returned no selected rows on XKRX "
                    f"session {session}."
                )
            checkpoint.save_prices(
                session,
                frame,
                artifact,
                selected_symbols=symbol_tuple,
            )
            by_session[session] = frame
    elif missing_sessions:
        executor = ThreadPoolExecutor(max_workers=max(1, workers))
        futures = {
            executor.submit(
                fetch_krx_session_prices,
                session,
                catalog,
                symbols=symbol_tuple,
                sleep_seconds=sleep_seconds,
            ): session
            for session in missing_sessions
        }
        failure: RuntimeError | None = None
        try:
            for future in as_completed(futures):
                session = futures[future]
                try:
                    frame, artifact = future.result()
                except Exception as exc:
                    failure = RuntimeError(
                        f"KRX price fetch failed on {session}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    break
                if frame.empty:
                    failure = RuntimeError(
                        f"KRX price fetch returned no selected rows on XKRX session {session}."
                    )
                    break
                checkpoint.save_prices(
                    session,
                    frame,
                    artifact,
                    selected_symbols=symbol_tuple,
                )
                by_session[session] = frame
        except Exception as exc:
            failure = RuntimeError(
                "KRX price checkpoint processing failed: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            if failure is not None:
                for pending in futures:
                    pending.cancel()
            executor.shutdown(
                wait=True,
                cancel_futures=failure is not None,
            )
        if failure is not None:
            raise failure from None
    frames = [by_session[session] for session in sessions if session in by_session]
    if not frames:
        return pd.DataFrame()
    output = pd.concat(frames, ignore_index=True)
    if "observation_status" not in output:
        output["observation_status"] = KRX_TRADED_STATUS
    return output.drop_duplicates(
        ["security_id", "session"], keep="last"
    )


def _upgrade_cached_krx_price_checkpoint(
    frame: pd.DataFrame,
    *,
    session: str,
    catalog: KrIdentityCatalog,
    checkpoint: KrCheckpointStore,
) -> tuple[pd.DataFrame | None, bool]:
    """Rebuild identity and official reference fields from hashed KRX payload."""

    needs_upgrade = (
        "security_name" not in frame
        or "official_reference_price" not in frame
        or "official_fluctuation_rate" not in frame
        or frame["security_id"].fillna("").astype(str).str.strip().eq("").any()
        or (
            "observation_status" in frame
            and frame["observation_status"]
            .astype(str)
            .eq("invalid_official_ohlc")
            .any()
        )
    )
    if not needs_upgrade:
        return frame, False
    hashes = {
        str(value).strip().lower()
        for value in frame.get("source_hash", pd.Series(dtype=str))
        if _valid_sha256(value)
    }
    if not hashes:
        return None, False
    evidence_by_symbol: dict[str, dict[str, object]] = {}
    for source_hash in sorted(hashes):
        evidence_root = checkpoint.root / "evidence_local" / "krx-prices"
        paths = sorted(evidence_root.glob(f"{source_hash}.*.gz"))
        if len(paths) != 1:
            return None, False
        try:
            content = gzip.decompress(paths[0].read_bytes())
            if sha256_bytes(content) != source_hash:
                return None, False
            payload = json.loads(content.decode("utf-8"))
            artifact_session = str(
                (payload.get("request") or {}).get("session") or ""
            )
            if artifact_session and artifact_session != session:
                return None, False
            decoded = krx_price_evidence_by_symbol(payload)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None, False
        for symbol, evidence in decoded.items():
            existing = evidence_by_symbol.get(symbol)
            if (
                existing is not None
                and existing.get("security_id")
                and evidence.get("security_id")
                and existing["security_id"] != evidence["security_id"]
            ):
                raise RuntimeError(
                    "Cached KRX evidence has conflicting ISINs for "
                    f"{symbol} on {session}."
                )
            evidence_by_symbol[symbol] = evidence

    output = frame.copy()
    if "security_name" not in output:
        output["security_name"] = ""
    if "official_reference_price" not in output:
        output["official_reference_price"] = float("nan")
    if "official_fluctuation_rate" not in output:
        output["official_fluctuation_rate"] = float("nan")
    symbols = output["symbol"].map(normalize_kr_symbol)
    evidence_symbols = set(evidence_by_symbol)
    matched = symbols.isin(evidence_symbols)

    def evidence_map(field: str) -> pd.Series:
        values = {
            symbol: evidence.get(field)
            for symbol, evidence in evidence_by_symbol.items()
        }
        return symbols.map(values)

    raw_security_ids = evidence_map("security_id").fillna("").astype(str)
    cached_security_ids = (
        output["security_id"].fillna("").astype(str).str.strip()
    )
    conflicts = (
        matched
        & raw_security_ids.ne("")
        & cached_security_ids.ne("")
        & raw_security_ids.ne(cached_security_ids)
    )
    if conflicts.any():
        index = conflicts.loc[conflicts].index[0]
        raise RuntimeError(
            "Cached KRX row conflicts with its raw official ISIN for "
            f"{symbols.at[index]} on {session}: "
            f"{cached_security_ids.at[index]} != "
            f"{raw_security_ids.at[index]}."
        )
    if (matched & raw_security_ids.eq("")).any():
        return None, False
    output.loc[matched, "security_id"] = raw_security_ids.loc[matched]
    output.loc[matched, "security_name"] = (
        evidence_map("security_name")
        .fillna(symbols)
        .astype(str)
        .loc[matched]
    )
    output.loc[~matched, "security_name"] = (
        output.loc[~matched, "security_name"]
        .fillna("")
        .astype(str)
        .where(
            output.loc[~matched, "security_name"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne(""),
            symbols.loc[~matched],
        )
    )
    output.loc[matched, "exchange"] = (
        evidence_map("exchange").fillna("").astype(str).loc[matched]
    )
    output.loc[matched, "asset_type"] = (
        evidence_map("asset_type")
        .fillna("STOCK")
        .astype(str)
        .loc[matched]
    )
    statuses = (
        evidence_map("observation_status")
        .fillna("")
        .astype(str)
    )
    output.loc[matched, "observation_status"] = statuses.loc[matched]
    volumes = pd.to_numeric(evidence_map("volume"), errors="coerce").fillna(0.0)
    output.loc[matched, "volume"] = volumes.loc[matched]
    output.loc[matched, "official_reference_price"] = pd.to_numeric(
        evidence_map("official_reference_price"),
        errors="coerce",
    ).loc[matched]
    output.loc[matched, "official_fluctuation_rate"] = pd.to_numeric(
        evidence_map("official_fluctuation_rate"),
        errors="coerce",
    ).loc[matched]
    traded = matched & statuses.eq(KRX_TRADED_STATUS)
    for column in ("open", "high", "low", "close"):
        values = pd.to_numeric(evidence_map(column), errors="coerce")
        if values.loc[traded].isna().any():
            return None, False
        output.loc[traded, column] = values.loc[traded]
        output.loc[matched & ~traded, column] = float("nan")
    final_ids = output["security_id"].fillna("").astype(str).str.strip()
    if final_ids.eq("").any():
        return None, False
    return output, True


def _krx_price_checkpoint_covers_symbols(
    frame: pd.DataFrame,
    symbols: Iterable[str],
    catalog: KrIdentityCatalog,
    session: str,
) -> bool:
    """Reject a stale checkpoint when the selected identity inventory grew."""

    if frame.empty or not {"symbol", "security_id"} <= set(frame):
        return False
    selected = {normalize_kr_symbol(symbol) for symbol in symbols}
    relevant = frame.loc[
        frame["symbol"].map(normalize_kr_symbol).isin(selected)
    ].copy()
    stable_ids = relevant["security_id"].fillna("").astype(str).str.strip()
    if stable_ids.eq("").any() or not stable_ids.str.fullmatch(
        r"KR:[A-Z0-9]{12}"
    ).all():
        return False
    inventory_hash = str(
        frame.attrs.get("selected_symbols_sha256") or ""
    )
    if inventory_hash:
        return inventory_hash == _kr_price_symbol_inventory_hash(selected)
    expected_rows = catalog.active_rows_for(selected, session)
    expected = {
        (symbol, str(identity["security_id"]))
        for symbol, identity in expected_rows.items()
    }
    observed = set(
        zip(
            relevant["symbol"].map(normalize_kr_symbol),
            stable_ids,
        )
    )
    return expected <= observed


def _reconcile_catalog_with_krx_observations(
    catalog: KrIdentityCatalog,
    observations: pd.DataFrame,
    *,
    completed_session: str,
) -> KrIdentityCatalog:
    """Build symbol/exchange intervals from each session's official ISIN."""

    required = {
        "security_id",
        "symbol",
        "session",
        "exchange",
        "asset_type",
        "source",
        "source_url",
        "retrieved_at",
        "source_hash",
    }
    if observations.empty or not required <= set(observations):
        return catalog
    working = observations.copy()
    working["security_id"] = (
        working["security_id"].fillna("").astype(str).str.strip()
    )
    working["symbol"] = working["symbol"].map(normalize_kr_symbol)
    working["session"] = working["session"].map(
        lambda value: pd.Timestamp(value).date().isoformat()
    )
    working = working.loc[
        working["security_id"].str.fullmatch(r"KR:[A-Z0-9]{12}")
        & working["symbol"].map(is_valid_kr_symbol)
    ].copy()
    if working.empty:
        return catalog
    conflicts = (
        working.groupby(["session", "symbol"])["security_id"]
        .nunique()
        .loc[lambda values: values.gt(1)]
    )
    if not conflicts.empty:
        session, symbol = conflicts.index[0]
        raise RuntimeError(
            "KRX observations map one symbol to multiple ISINs on one "
            f"session: {symbol} on {session}."
        )
    if "security_name" not in working:
        working["security_name"] = working["symbol"]
    working["exchange"] = working["exchange"].fillna("").astype(str)
    working["asset_type"] = (
        working["asset_type"].fillna("STOCK").astype(str)
    )
    working = working.sort_values(
        ["symbol", "session", "security_id", "exchange"],
        kind="stable",
    ).drop_duplicates(
        ["session", "symbol", "security_id"],
        keep="last",
    )
    working["_state"] = (
        working["security_id"]
        + "\0"
        + working["exchange"]
        + "\0"
        + working["asset_type"]
    )
    previous = working.groupby("symbol", sort=False)["_state"].shift()
    working["_segment_start"] = working["_state"].ne(previous)
    working["_segment"] = (
        working.groupby("symbol", sort=False)["_segment_start"].cumsum()
    )

    existing = catalog.frame.copy()
    observed_pairs = set(zip(working["security_id"], working["symbol"]))
    untouched_mask = [
        (str(row.security_id), str(row.primary_symbol)) not in observed_pairs
        for row in existing.itertuples(index=False)
    ]
    untouched = existing.loc[untouched_mask].copy()
    rows: list[dict[str, Any]] = []
    for symbol, symbol_rows in working.groupby("symbol", sort=False):
        segments = [
            group.copy()
            for _, group in symbol_rows.groupby("_segment", sort=True)
        ]
        for position, segment in enumerate(segments):
            first = segment.iloc[0]
            last = segment.iloc[-1]
            security_id = str(first["security_id"])
            exchange = str(first["exchange"])
            exact = existing.loc[
                existing["security_id"].astype(str).eq(security_id)
                & existing["primary_symbol"].astype(str).eq(symbol)
            ].copy()
            exchange_exact = exact.loc[
                exact["exchange"].fillna("").astype(str).eq(exchange)
            ]
            base_options = exchange_exact if not exchange_exact.empty else exact
            base = (
                base_options.sort_values(
                    ["active_from", "active_to"],
                    kind="stable",
                ).iloc[-1].to_dict()
                if not base_options.empty
                else {}
            )
            active_from = str(first["session"])
            if not exchange_exact.empty:
                known_starts = pd.to_datetime(
                    exchange_exact["active_from"],
                    errors="coerce",
                ).dropna()
                if not known_starts.empty:
                    active_from = min(
                        pd.Timestamp(active_from),
                        known_starts.min(),
                    ).date().isoformat()
            next_segment = (
                segments[position + 1]
                if position + 1 < len(segments)
                else None
            )
            if next_segment is not None:
                next_first = next_segment.iloc[0]
                if str(next_first["security_id"]) == security_id:
                    active_to = (
                        pd.Timestamp(next_first["session"])
                        - pd.Timedelta(days=1)
                    ).date().isoformat()
                else:
                    active_to = str(last["session"])
            else:
                known_ends = pd.to_datetime(
                    exact["active_to"],
                    errors="coerce",
                ).dropna()
                if not known_ends.empty:
                    active_to = known_ends.max().date().isoformat()
                elif (
                    not exact.empty
                    and exact["active_to"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .eq("")
                    .any()
                ) or str(last["session"]) >= str(completed_session):
                    active_to = ""
                else:
                    active_to = str(last["session"])
            record = dict(base)
            record.update(
                {
                    "security_id": security_id,
                    "primary_symbol": symbol,
                    "name": str(
                        last.get("security_name")
                        or base.get("name")
                        or symbol
                    ),
                    "exchange": exchange,
                    "asset_type": str(
                        last.get("asset_type")
                        or base.get("asset_type")
                        or "STOCK"
                    ),
                    "currency": "KRW",
                    "country": "KR",
                    "active_from": active_from,
                    "active_to": active_to,
                    "isin": security_id.removeprefix("KR:"),
                    "identity_mapped": True,
                    "provider_symbol": (
                        f"{symbol}.{'KO' if exchange == 'KOSPI' else 'KQ'}"
                    ),
                    "yahoo_symbol": (
                        f"{symbol}.{'KS' if exchange == 'KOSPI' else 'KQ'}"
                    ),
                    "source": "krx_official_daily_identity",
                    "source_url": str(first["source_url"]),
                    "retrieved_at": str(first["retrieved_at"]),
                    "source_hash": str(first["source_hash"]),
                }
            )
            record.setdefault("dart_corp_code", "")
            rows.append(record)
    reconciled = pd.concat(
        [untouched, pd.DataFrame(rows)],
        ignore_index=True,
        sort=False,
    ).sort_values(
        ["primary_symbol", "active_from", "active_to", "security_id"],
        kind="stable",
    ).drop_duplicates(
        ["security_id", "primary_symbol", "active_from", "exchange"],
        keep="last",
    )
    return KrIdentityCatalog(
        reconciled.reset_index(drop=True),
        catalog.artifacts,
        catalog.dart_status,
    )


def _classify_cached_delisting_effective_absences(
    frame: pd.DataFrame,
    catalog: KrIdentityCatalog,
    session: str,
) -> pd.DataFrame:
    if frame.empty or "observation_status" not in frame:
        return frame
    statuses = frame["observation_status"].astype(str)
    candidates = statuses.isin(
        {
            "missing_from_krx_response",
            "delisting_effective_date_no_trade",
        }
    )
    if not candidates.any():
        return frame
    output = frame.copy()
    identities = catalog.active_rows_for(
        output.loc[candidates, "symbol"],
        session,
    )
    session_text = pd.Timestamp(session).date().isoformat()
    for index in output.index[candidates]:
        symbol = normalize_kr_symbol(output.at[index, "symbol"])
        identity = identities.get(symbol)
        if (
            identity is not None
            and str(identity.get("active_to") or "") == session_text
            and catalog.interval_is_terminal(identity, session)
        ):
            output.at[index, "observation_status"] = (
                "delisting_effective_date_no_trade"
            )
        elif statuses.at[index] == "delisting_effective_date_no_trade":
            output.at[index, "observation_status"] = (
                "missing_from_krx_response"
            )
    return output


def _traded_krx_prices(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        return observations.copy()
    if "observation_status" not in observations:
        return observations.copy()
    return observations.loc[
        observations["observation_status"].astype(str).eq(KRX_TRADED_STATUS)
    ].reset_index(drop=True)


def _krx_reference_price_adjustments(
    observations: pd.DataFrame,
    existing_actions: pd.DataFrame,
) -> tuple[pd.DataFrame, SourceArtifact]:
    """Create an authoritative price-only factor for every KRX reset."""

    required = {
        "security_id",
        "session",
        "close",
        "official_reference_price",
        "source_url",
        "retrieved_at",
        "source_hash",
    }
    if missing := required - set(observations):
        raise RuntimeError(
            "KRX observations lack official reference-price fields: "
            + ", ".join(sorted(missing))
        )
    traded = _traded_krx_prices(observations).copy()
    traded["session"] = pd.to_datetime(
        traded["session"], errors="coerce"
    ).dt.normalize()
    traded["close"] = pd.to_numeric(traded["close"], errors="coerce")
    traded["official_reference_price"] = pd.to_numeric(
        traded["official_reference_price"], errors="coerce"
    )
    traded = traded.dropna(
        subset=["security_id", "session", "close"]
    ).sort_values(["security_id", "session"], kind="stable")
    traded["previous_close"] = traded.groupby(
        traded["security_id"].astype(str),
        sort=False,
    )["close"].shift(1)
    candidate_mask = (
        traded["previous_close"].notna()
        & traded["official_reference_price"].notna()
        & traded["official_reference_price"].gt(0)
        & traded["previous_close"]
        .sub(traded["official_reference_price"])
        .abs()
        .gt(0.5)
    )
    candidates = traded.loc[candidate_mask].copy()

    ratio_types = {
        "split",
        "capital_reduction",
        "stock_dividend",
        "reference_price_adjustment",
    }
    actions = existing_actions.copy()
    if not actions.empty:
        action_dates = actions["ex_date"].where(
            actions["ex_date"].notna()
            & actions["ex_date"].astype(str).str.strip().ne(""),
            actions["effective_date"],
        )
        actions["_audit_date"] = pd.to_datetime(
            action_dates, errors="coerce"
        ).dt.normalize()
        actions["_audit_ratio"] = pd.to_numeric(
            actions["ratio"], errors="coerce"
        )
        actions = actions.loc[
            actions["action_type"].astype(str).isin(ratio_types)
            & actions["_audit_ratio"].gt(0)
        ].copy()
    else:
        actions = pd.DataFrame(
            columns=(
                "security_id",
                "event_id",
                "action_type",
                "_audit_date",
                "_audit_ratio",
            )
        )

    traded = traded.drop_duplicates(
        ["security_id", "session"],
        keep="last",
    )
    traded_by_security = {
        str(security_id): group.sort_values("session", kind="stable")
        for security_id, group in traded.groupby(
            traded["security_id"].astype(str),
            sort=False,
        )
    }
    audit_start = (
        pd.Timestamp(traded["session"].min())
        if not traded.empty
        else pd.NaT
    )
    audit_end = (
        pd.Timestamp(traded["session"].max())
        if not traded.empty
        else pd.NaT
    )
    action_sessions: list[pd.Timestamp | pd.NaT] = []
    action_scopes: list[str] = []
    for action in actions.to_dict("records"):
        security_prices = traded_by_security.get(
            str(action["security_id"])
        )
        action_date = pd.Timestamp(action["_audit_date"])
        if pd.isna(action_date):
            action_sessions.append(pd.NaT)
            action_scopes.append("invalid_date")
            continue
        if (
            pd.isna(audit_start)
            or pd.isna(audit_end)
            or action_date < audit_start
            or action_date > audit_end
        ):
            action_sessions.append(pd.NaT)
            action_scopes.append("outside_audit_window")
            continue
        if (
            security_prices is None
            or security_prices.empty
            or action_date < security_prices["session"].iloc[0]
            or action_date > security_prices["session"].iloc[-1]
        ):
            action_sessions.append(pd.NaT)
            action_scopes.append("outside_factor_domain")
            continue
        sessions = security_prices["session"].to_numpy(
            dtype="datetime64[ns]"
        )
        position = int(
            sessions.searchsorted(
                action_date.to_datetime64(),
                side="left",
            )
        )
        action_sessions.append(
            pd.Timestamp(sessions[position])
            if position < len(sessions)
            else pd.NaT
        )
        action_scopes.append(
            "mapped"
            if 0 < position < len(sessions)
            else "outside_factor_domain"
        )
    actions["_audit_session"] = action_sessions
    actions["_audit_scope"] = action_scopes

    generated: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    covered_ratio_event_ids: set[str] = set()
    unresolved_records: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        security_id = str(row.security_id)
        session = pd.Timestamp(row.session).date().isoformat()
        previous_close = float(row.previous_close)
        reference_price = float(row.official_reference_price)
        official_ratio = previous_close / reference_price
        matching = actions.loc[
            actions["security_id"].astype(str).eq(security_id)
            & actions["_audit_session"].eq(pd.Timestamp(row.session))
        ]
        existing_reference = matching.loc[
            matching["action_type"]
            .astype(str)
            .eq("reference_price_adjustment")
        ]
        if len(existing_reference) > 1:
            raise RuntimeError(
                "KR has multiple reference-price adjustments for "
                f"{security_id} on {session}."
            )
        related_actions = matching.loc[
            ~matching["action_type"]
            .astype(str)
            .eq("reference_price_adjustment")
        ]
        covered_ratio_event_ids.update(
            related_actions["event_id"].astype(str)
        )
        related_ratio = (
            float(related_actions["_audit_ratio"].prod())
            if not related_actions.empty
            else None
        )
        tolerance = max(1.0, float(krx_tick_size(reference_price)))
        if not existing_reference.empty:
            existing_ratio = float(
                existing_reference.iloc[0]["_audit_ratio"]
            )
            explained_reference = previous_close / existing_ratio
            if abs(explained_reference - reference_price) > tolerance:
                raise RuntimeError(
                    "KR reference-price adjustment conflicts with the official "
                    f"price for {security_id} on {session}: "
                    f"{explained_reference} != {reference_price}."
                )
            resolution = "covered_by_existing_reference_adjustment"
            event_id = str(existing_reference.iloc[0]["event_id"])
        else:
            identity = {
                "security_id": security_id,
                "action_type": "reference_price_adjustment",
                "effective_date": session,
                "ratio": round(official_ratio, 12),
                "source_hash": str(row.source_hash).lower(),
            }
            event_id = sha256_bytes(canonical_json_bytes(identity))
            generated.append(
                {
                    "event_id": event_id,
                    "security_id": security_id,
                    "action_type": "reference_price_adjustment",
                    "effective_date": session,
                    "ex_date": session,
                    "announcement_date": "",
                    "record_date": "",
                    "payment_date": "",
                    "cash_amount": None,
                    "ratio": official_ratio,
                    "currency": "KRW",
                    "new_security_id": "",
                    "new_symbol": "",
                    "official": True,
                    "source_url": str(row.source_url),
                    "source_kind": "official",
                    "source": "krx_official_reference_price_adjustment",
                    "retrieved_at": str(row.retrieved_at),
                    "source_hash": str(row.source_hash).lower(),
                }
            )
            resolution = "generated_official_reference_adjustment"
        audit_records.append(
            {
                "security_id": security_id,
                "session": session,
                "record_kind": "reference_discontinuity",
                "previous_close": previous_close,
                "official_reference_price": reference_price,
                "official_ratio": round(official_ratio, 12),
                "resolution": resolution,
                "event_id": event_id,
                "related_ratio_event_ids": sorted(
                    related_actions["event_id"].astype(str)
                ),
                "related_share_ratio": (
                    round(related_ratio, 12)
                    if related_ratio is not None
                    else None
                ),
                "related_ratio_reference_price": (
                    round(previous_close / related_ratio, 12)
                    if related_ratio is not None
                    else None
                ),
                "source_hash": str(row.source_hash).lower(),
            }
        )

    provider_ratio_actions = actions.loc[
        ~actions["action_type"]
        .astype(str)
        .eq("reference_price_adjustment")
    ].copy()
    uncovered_ratio_actions = provider_ratio_actions.loc[
        ~provider_ratio_actions["event_id"]
        .astype(str)
        .isin(covered_ratio_event_ids)
    ].copy()
    outside_audit_window_count = int(
        uncovered_ratio_actions["_audit_scope"]
        .eq("outside_audit_window")
        .sum()
    )
    outside_audit_window_actions = uncovered_ratio_actions.loc[
        uncovered_ratio_actions["_audit_scope"].eq(
            "outside_audit_window"
        )
    ]
    for (security_id, audit_date), related_actions in (
        outside_audit_window_actions.groupby(
            ["security_id", "_audit_date"],
            sort=True,
        )
    ):
        audit_records.append(
            {
                "security_id": str(security_id),
                "session": pd.Timestamp(
                    audit_date
                ).date().isoformat(),
                "record_kind": "provider_ratio_outside_audit_window",
                "previous_close": None,
                "official_reference_price": None,
                "official_ratio": None,
                "resolution": "outside_incremental_audit_window",
                "event_id": "",
                "related_ratio_event_ids": sorted(
                    related_actions["event_id"].astype(str)
                ),
                "related_share_ratio": round(
                    float(related_actions["_audit_ratio"].prod()),
                    12,
                ),
                "related_ratio_reference_price": None,
                "source_hash": "",
            }
        )
    outside_factor_domain_actions = uncovered_ratio_actions.loc[
        uncovered_ratio_actions["_audit_scope"].eq(
            "outside_factor_domain"
        )
    ]
    outside_factor_domain_count = len(outside_factor_domain_actions)
    for (security_id, audit_date), related_actions in (
        outside_factor_domain_actions.groupby(
            ["security_id", "_audit_date"],
            sort=True,
        )
    ):
        related_ratio = float(
            related_actions["_audit_ratio"].prod()
        )
        audit_records.append(
            {
                "security_id": str(security_id),
                "session": pd.Timestamp(
                    audit_date
                ).date().isoformat(),
                "record_kind": "provider_ratio_outside_factor_domain",
                "previous_close": None,
                "official_reference_price": None,
                "official_ratio": None,
                "resolution": "outside_factor_domain_no_price_effect",
                "event_id": "",
                "related_ratio_event_ids": sorted(
                    related_actions["event_id"].astype(str)
                ),
                "related_share_ratio": round(
                    related_ratio,
                    12,
                ),
                "related_ratio_reference_price": None,
                "source_hash": "",
            }
        )
    invalid_date_actions = uncovered_ratio_actions.loc[
        uncovered_ratio_actions["_audit_scope"].eq("invalid_date")
    ]
    for action in invalid_date_actions.itertuples(index=False):
        unresolved_records.append(
            {
                "security_id": str(action.security_id),
                "session": "",
                "code": "provider_ratio_has_invalid_effective_date",
                "related_ratio_event_ids": [str(action.event_id)],
            }
        )
    restated_actions = uncovered_ratio_actions.loc[
        uncovered_ratio_actions["_audit_scope"].eq("mapped")
    ]
    for (security_id, audit_session), related_actions in (
        restated_actions.groupby(
            ["security_id", "_audit_session"],
            sort=True,
        )
    ):
        session_row = traded.loc[
            traded["security_id"].astype(str).eq(str(security_id))
            & traded["session"].eq(pd.Timestamp(audit_session))
        ]
        if len(session_row) != 1:
            unresolved_records.append(
                {
                    "security_id": str(security_id),
                    "session": pd.Timestamp(
                        audit_session
                    ).date().isoformat(),
                    "code": "missing_unique_krx_price_position",
                    "related_ratio_event_ids": sorted(
                        related_actions["event_id"].astype(str)
                    ),
                }
            )
            continue
        official_row = session_row.iloc[0]
        previous_close = _optional_number(
            official_row.get("previous_close")
        )
        reference_price = _optional_number(
            official_row.get("official_reference_price")
        )
        if (
            previous_close is None
            or reference_price is None
            or reference_price <= 0
            or abs(previous_close - reference_price) > 0.5
        ):
            unresolved_records.append(
                {
                    "security_id": str(security_id),
                    "session": pd.Timestamp(
                        audit_session
                    ).date().isoformat(),
                    "code": "provider_ratio_lacks_official_restatement_proof",
                    "related_ratio_event_ids": sorted(
                        related_actions["event_id"].astype(str)
                    ),
                }
            )
            continue
        matching_reference = actions.loc[
            actions["security_id"].astype(str).eq(str(security_id))
            & actions["_audit_session"].eq(pd.Timestamp(audit_session))
            & actions["action_type"]
            .astype(str)
            .eq("reference_price_adjustment")
        ]
        if len(matching_reference) > 1:
            raise RuntimeError(
                "KR has multiple restatement no-op adjustments for "
                f"{security_id} on "
                f"{pd.Timestamp(audit_session).date().isoformat()}."
            )
        session = pd.Timestamp(audit_session).date().isoformat()
        if not matching_reference.empty:
            existing_ratio = float(
                matching_reference.iloc[0]["_audit_ratio"]
            )
            if abs(existing_ratio - 1.0) > 1e-12:
                unresolved_records.append(
                    {
                        "security_id": str(security_id),
                        "session": session,
                        "code": "restated_series_conflicts_with_existing_adjustment",
                        "related_ratio_event_ids": sorted(
                            related_actions["event_id"].astype(str)
                        ),
                    }
                )
                continue
            event_id = str(
                matching_reference.iloc[0]["event_id"]
            )
            resolution = "covered_by_existing_restatement_noop"
        else:
            identity = {
                "security_id": str(security_id),
                "action_type": "reference_price_adjustment",
                "effective_date": session,
                "ratio": 1.0,
                "source_hash": str(
                    official_row["source_hash"]
                ).lower(),
                "contract": "krx_official_series_already_restated/v1",
            }
            event_id = sha256_bytes(canonical_json_bytes(identity))
            generated.append(
                {
                    "event_id": event_id,
                    "security_id": str(security_id),
                    "action_type": "reference_price_adjustment",
                    "effective_date": session,
                    "ex_date": session,
                    "announcement_date": "",
                    "record_date": "",
                    "payment_date": "",
                    "cash_amount": None,
                    "ratio": 1.0,
                    "currency": "KRW",
                    "new_security_id": "",
                    "new_symbol": "",
                    "official": True,
                    "source_url": str(official_row["source_url"]),
                    "source_kind": "official",
                    "source": "krx_official_series_restatement_noop",
                    "retrieved_at": str(
                        official_row["retrieved_at"]
                    ),
                    "source_hash": str(
                        official_row["source_hash"]
                    ).lower(),
                }
            )
            resolution = "generated_official_restatement_noop"
        related_ratio = float(
            related_actions["_audit_ratio"].prod()
        )
        audit_records.append(
            {
                "security_id": str(security_id),
                "session": session,
                "record_kind": "provider_ratio_restatement_noop",
                "previous_close": previous_close,
                "official_reference_price": reference_price,
                "official_ratio": 1.0,
                "resolution": resolution,
                "event_id": event_id,
                "related_ratio_event_ids": sorted(
                    related_actions["event_id"].astype(str)
                ),
                "related_share_ratio": round(
                    related_ratio,
                    12,
                ),
                "related_ratio_reference_price": round(
                    previous_close / related_ratio,
                    12,
                ),
                "source_hash": str(
                    official_row["source_hash"]
                ).lower(),
            }
        )
    accounted_ratio_event_ids = {
        str(event_id)
        for record in audit_records
        for event_id in record.get("related_ratio_event_ids") or ()
        if str(event_id)
    }
    invalid_ratio_event_ids = set(
        invalid_date_actions["event_id"].astype(str)
    )
    unaccounted_ratio_event_ids = sorted(
        set(provider_ratio_actions["event_id"].astype(str))
        - accounted_ratio_event_ids
        - invalid_ratio_event_ids
    )
    if unaccounted_ratio_event_ids:
        unresolved_records.append(
            {
                "security_id": "",
                "session": "",
                "code": "provider_ratio_not_accounted_by_krx_audit",
                "related_ratio_event_ids": (
                    unaccounted_ratio_event_ids
                ),
            }
        )
    generated_frame = pd.DataFrame(
        generated,
        columns=dataset_spec("corporate_actions").required_columns,
    )
    observation_counts_by_session = {
        pd.Timestamp(session).date().isoformat(): int(count)
        for session, count in traded.groupby(
            "session",
            sort=True,
        ).size().items()
    }
    payload = {
        "schema": "krx_reference_price_audit/v1",
        "status": "passed" if not unresolved_records else "blocked",
        "market": "KR",
        "observation_count": len(traded),
        "audit_start": (
            audit_start.date().isoformat()
            if pd.notna(audit_start)
            else ""
        ),
        "audit_end": (
            audit_end.date().isoformat()
            if pd.notna(audit_end)
            else ""
        ),
        "observation_counts_by_session": (
            observation_counts_by_session
        ),
        "reference_discontinuity_count": len(candidates),
        "covered_existing_action_count": sum(
            value["resolution"]
            == "covered_by_existing_reference_adjustment"
            for value in audit_records
        ),
        "generated_adjustment_count": len(generated_frame),
        "generated_restatement_noop_count": sum(
            value["resolution"]
            == "generated_official_restatement_noop"
            for value in audit_records
        ),
        "reference_adjustment_event_count": sum(
            value["record_kind"]
            in {
                "reference_discontinuity",
                "provider_ratio_restatement_noop",
            }
            and bool(value["event_id"])
            for value in audit_records
        ),
        "provider_ratio_action_count": len(provider_ratio_actions),
        "provider_ratio_accounted_action_count": len(
            accounted_ratio_event_ids
        ),
        "provider_ratio_unaccounted_count": len(
            unaccounted_ratio_event_ids
        ),
        "provider_ratio_outside_factor_domain_count": (
            outside_factor_domain_count
        ),
        "provider_ratio_outside_audit_window_count": (
            outside_audit_window_count
        ),
        "provider_ratio_restatement_noop_count": sum(
            value["record_kind"]
            == "provider_ratio_restatement_noop"
            for value in audit_records
        ),
        "unresolved_count": len(unresolved_records),
        "unresolved_records": unresolved_records,
        "records": audit_records,
    }
    return generated_frame, SourceArtifact(
        source="krx_reference_price_audit",
        source_url=KRX_SOURCE_URL,
        retrieved_at=utc_now_iso(),
        content=canonical_json_bytes(payload),
        content_type="application/json",
    )


def _kr_dividend_sessions(*, start: str, end: str) -> pd.DatetimeIndex:
    """Return enough official exchange sessions to resolve T+2 ex dates."""

    from .markets import exchange_calendar

    calendar = exchange_calendar("KR")
    requested_start = pd.Timestamp(start) - pd.Timedelta(days=31)
    calendar_start = max(
        requested_start.normalize(),
        pd.Timestamp(calendar.first_session).tz_localize(None).normalize(),
    ).date().isoformat()
    return pd.DatetimeIndex(
        pd.to_datetime(_sessions_between(calendar_start, end))
    ).normalize()


def _kr_dividend_ex_date(
    record_date: Any,
    sessions: pd.DatetimeIndex,
) -> str:
    """Resolve the KRX ex date for a T+2 security from its record date.

    The entitled last trade is two exchange sessions before the shareholder
    record date. The following session is therefore the ex date, which is the
    second-last KRX session on or before the record date. This also handles the
    year-end closure on December 31 without a calendar-day heuristic.
    """

    record = pd.Timestamp(record_date).normalize()
    position = int(sessions.searchsorted(record, side="right"))
    if position < 2:
        raise RuntimeError(
            "KR dividend record date lacks two prior KRX sessions: "
            + record.date().isoformat()
        )
    return sessions[position - 2].date().isoformat()


def _officialize_opendart_cash_dividends(
    actions: pd.DataFrame,
    dart_result: KrDartDividendResult,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Replace provider cash rows with official DART decisions.

    Raw provider rows are passed separately to the immutable action audit.
    They must not remain in the canonical action table: a provider-only false
    positive would otherwise change total returns despite the complete
    OpenDART filing inventory proving no official decision exists.
    """

    if dart_result.status != "passed":
        return actions.copy()
    sessions = _kr_dividend_sessions(start=start, end=end)
    records: list[dict[str, Any]] = []
    for decision in dart_result.decisions.sort_values(
        ["security_id", "record_date", "cash_amount", "rcept_no"],
        kind="stable",
    ).itertuples(index=False):
        security_id = str(decision.security_id).strip()
        record_date = _normalized_date(decision.record_date)
        cash_amount = _optional_number(decision.cash_amount)
        rcept_no = str(decision.rcept_no).strip()
        source_url = str(decision.source_url).strip()
        source_hash = str(decision.source_hash).strip().lower()
        if (
            not security_id
            or not record_date
            or cash_amount is None
            or cash_amount <= 0
            or not rcept_no
            or not source_url.lower().startswith(("http://", "https://"))
            or not _valid_sha256(source_hash)
        ):
            raise RuntimeError(
                "OpenDART dividend decision lacks an auditable identity, "
                f"amount, date, or source: {security_id}:{rcept_no}"
            )
        ex_date = _kr_dividend_ex_date(record_date, sessions)
        identity = {
            "security_id": security_id,
            "action_type": "cash_dividend",
            "effective_date": ex_date,
            "record_date": record_date,
            "cash_amount": cash_amount,
            "rcept_no": rcept_no,
            "source_hash": source_hash,
            "ex_date_rule": KRX_EX_DIVIDEND_RULE,
        }
        records.append(
            {
                "event_id": sha256_bytes(canonical_json_bytes(identity)),
                "security_id": security_id,
                "action_type": "cash_dividend",
                "effective_date": ex_date,
                "ex_date": ex_date,
                "announcement_date": _normalized_date(
                    decision.announcement_date
                ),
                "record_date": record_date,
                "payment_date": _normalized_date(decision.payment_date),
                "cash_amount": cash_amount,
                "ratio": None,
                "currency": "KRW",
                "new_security_id": "",
                "new_symbol": "",
                "official": True,
                "source_url": source_url,
                "source_kind": "official",
                "source": "opendart_cash_dividend_decision",
                "retrieved_at": dart_result.report.retrieved_at,
                "source_hash": source_hash,
            }
        )
    official = pd.DataFrame(
        records,
        columns=dataset_spec("corporate_actions").required_columns,
    )
    prior_official = actions["source"].astype(str).eq(
        "opendart_cash_dividend_decision"
    )
    provider_cash = (
        actions["action_type"].astype(str).eq("cash_dividend")
        & ~actions["source_kind"]
        .astype(str)
        .str.lower()
        .eq("official")
    )
    return pd.concat(
        [actions.loc[~(prior_official | provider_cash)], official],
        ignore_index=True,
    ).drop_duplicates(
        "event_id",
        keep="last",
    )


def _kr_corporate_action_audit(
    actions: pd.DataFrame,
    catalog: KrIdentityCatalog,
    dart_result: KrDartDividendResult,
    reference_price_audit: SourceArtifact,
    *,
    start: str,
    end: str,
    provider_failures: Iterable[str] = (),
    provider_actions: pd.DataFrame | None = None,
) -> SourceArtifact:
    """Cross-check cash dividends and every price-changing KRX reset."""

    try:
        reference_report = json.loads(reference_price_audit.content)
    except (TypeError, ValueError, json.JSONDecodeError):
        reference_report = {}
    issues: list[dict[str, str]] = []

    def issue(
        code: str,
        *,
        security_id: str = "",
        detail: str = "",
    ) -> None:
        issues.append(
            {
                "code": code,
                "security_id": security_id,
                "detail": detail,
            }
        )

    if (
        reference_report.get("status") != "passed"
        or int(reference_report.get("unresolved_count") or 0) != 0
    ):
        issue("reference_price_audit_blocked")
    if dart_result.status != "passed":
        issue("opendart_dividend_collection_blocked", detail=dart_result.detail)

    stock_ids = set(
        catalog.frame.loc[
            catalog.frame.get(
                "asset_type",
                pd.Series("STOCK", index=catalog.frame.index),
            )
            .fillna("STOCK")
            .astype(str)
            .str.upper()
            .eq("STOCK"),
            "security_id",
        ].astype(str)
    )
    official_cash_actions = actions.loc[
        actions["action_type"].astype(str).eq("cash_dividend")
        & actions["security_id"].astype(str).isin(stock_ids)
        & actions["source"]
        .astype(str)
        .eq("opendart_cash_dividend_decision")
    ].copy()
    official_cash_actions["audit_cash_amount"] = pd.to_numeric(
        official_cash_actions["cash_amount"],
        errors="coerce",
    )
    official_cash_actions = official_cash_actions.loc[
        official_cash_actions["audit_cash_amount"].gt(0)
    ]
    official_cash_actions["audit_record_date"] = pd.to_datetime(
        official_cash_actions["record_date"],
        errors="coerce",
    ).dt.normalize()
    official_dates = official_cash_actions["ex_date"].where(
        official_cash_actions["ex_date"].notna()
        & official_cash_actions["ex_date"].astype(str).str.strip().ne(""),
        official_cash_actions["effective_date"],
    )
    official_cash_actions["audit_action_date"] = pd.to_datetime(
        official_dates,
        errors="coerce",
    ).dt.normalize()

    raw_action_inventory = (
        provider_actions.copy()
        if provider_actions is not None
        else actions.copy()
    )
    provider_cash_actions = raw_action_inventory.loc[
        raw_action_inventory["action_type"].astype(str).eq("cash_dividend")
        & raw_action_inventory["security_id"].astype(str).isin(stock_ids)
        & ~raw_action_inventory["source"]
        .astype(str)
        .eq("opendart_cash_dividend_decision")
        & ~raw_action_inventory["source_kind"]
        .astype(str)
        .str.lower()
        .eq("official")
    ].copy()
    provider_cash_actions["audit_cash_amount"] = pd.to_numeric(
        provider_cash_actions["cash_amount"],
        errors="coerce",
    )
    provider_cash_actions = provider_cash_actions.loc[
        provider_cash_actions["audit_cash_amount"].gt(0)
    ]
    provider_dates = provider_cash_actions["ex_date"].where(
        provider_cash_actions["ex_date"].notna()
        & provider_cash_actions["ex_date"].astype(str).str.strip().ne(""),
        provider_cash_actions["effective_date"],
    )
    provider_cash_actions["audit_action_date"] = pd.to_datetime(
        provider_dates,
        errors="coerce",
    ).dt.normalize()
    dividend_sessions = _kr_dividend_sessions(start=start, end=end)

    def normalized_provider_ex_date(value: Any) -> pd.Timestamp:
        action_date = pd.Timestamp(value).normalize()
        position = int(
            dividend_sessions.searchsorted(action_date, side="left")
        )
        if position >= len(dividend_sessions):
            return pd.NaT
        return pd.Timestamp(dividend_sessions[position]).normalize()

    if not provider_cash_actions.empty:
        provider_cash_actions["audit_normalized_ex_date"] = (
            provider_cash_actions["audit_action_date"].map(
                normalized_provider_ex_date
            )
        )

    decisions = dart_result.decisions.copy()
    decisions["audit_record_date"] = pd.to_datetime(
        decisions.get("record_date"),
        errors="coerce",
    ).dt.normalize()
    decisions["audit_cash_amount"] = pd.to_numeric(
        decisions.get("cash_amount"),
        errors="coerce",
    )
    matched_official_action_ids: set[str] = set()
    matched_provider_action_ids: set[str] = set()
    matched_decisions = 0
    provider_amount_match_count = 0
    provider_amount_mismatch_count = 0
    provider_date_mismatch_count = 0
    provider_missing_count = 0
    decision_records: list[dict[str, Any]] = []
    for decision in decisions.sort_values(
        ["security_id", "audit_record_date", "audit_cash_amount"],
        kind="stable",
    ).itertuples(index=False):
        security_id = str(decision.security_id)
        record_date = pd.Timestamp(decision.audit_record_date)
        amount = float(decision.audit_cash_amount)
        expected_ex_date = pd.Timestamp(
            _kr_dividend_ex_date(record_date, dividend_sessions)
        )
        official_candidates = official_cash_actions.loc[
            official_cash_actions["security_id"]
            .astype(str)
            .eq(security_id)
            & official_cash_actions["audit_record_date"].eq(record_date)
            & official_cash_actions["audit_action_date"].eq(
                expected_ex_date
            )
            & official_cash_actions["audit_cash_amount"]
            .sub(amount)
            .abs()
            .le(max(0.05, abs(amount) * 0.0001))
        ].copy()
        official_ids = sorted(
            official_candidates["event_id"].astype(str).unique()
        )
        if len(official_ids) == 1:
            matched_official_action_ids.add(official_ids[0])
            matched_decisions += 1
            disposition = "matched"
        elif not official_ids:
            disposition = "missing_action"
            issue(
                "missing_official_dividend_action",
                security_id=security_id,
                detail=(
                    f"{record_date.date().isoformat()}:"
                    f"{amount}:{decision.rcept_no}"
                ),
            )
        else:
            disposition = "duplicate_actions"
            issue(
                "duplicate_dividend_actions",
                security_id=security_id,
                detail=",".join(official_ids),
            )

        provider_candidates = provider_cash_actions.loc[
            provider_cash_actions["security_id"]
            .astype(str)
            .eq(security_id)
            & provider_cash_actions["audit_normalized_ex_date"].eq(
                expected_ex_date
            )
            & ~provider_cash_actions["event_id"]
            .astype(str)
            .isin(matched_provider_action_ids)
        ].copy()
        provider_ids: list[str] = []
        provider_amounts: list[float] = []
        provider_disposition = "missing_provider_action"
        if provider_candidates.empty:
            provider_candidates = provider_cash_actions.loc[
                provider_cash_actions["security_id"]
                .astype(str)
                .eq(security_id)
                & provider_cash_actions["audit_action_date"].le(
                    record_date
                )
                & provider_cash_actions["audit_action_date"].ge(
                    record_date - pd.Timedelta(days=20)
                )
                & ~provider_cash_actions["event_id"]
                .astype(str)
                .isin(matched_provider_action_ids)
            ].copy()
            provider_date_mismatch = not provider_candidates.empty
        else:
            provider_date_mismatch = False
        if provider_candidates.empty:
            provider_missing_count += 1
        else:
            provider_candidates["_amount_distance"] = (
                provider_candidates["audit_cash_amount"]
                .sub(amount)
                .abs()
            )
            provider_candidates["_date_distance"] = (
                provider_candidates["audit_normalized_ex_date"]
                .sub(expected_ex_date)
                .abs()
                .dt.days
            )
            provider_match = provider_candidates.sort_values(
                ["_date_distance", "_amount_distance", "event_id"],
                kind="stable",
            ).iloc[0]
            provider_event_id = str(provider_match["event_id"])
            provider_ids = [provider_event_id]
            provider_amount = float(
                provider_match["audit_cash_amount"]
            )
            provider_amounts = [provider_amount]
            matched_provider_action_ids.add(provider_event_id)
            amount_matches = abs(provider_amount - amount) <= max(
                0.05,
                abs(amount) * 0.0001,
            )
            if amount_matches:
                provider_amount_match_count += 1
            else:
                provider_amount_mismatch_count += 1
            if provider_date_mismatch:
                provider_disposition = (
                    "date_mismatch_official_wins"
                    if amount_matches
                    else "date_and_amount_mismatch_official_wins"
                )
                provider_date_mismatch_count += 1
            elif amount_matches:
                provider_disposition = "amount_match"
            else:
                # The official DART amount is applied to returns. Keep the
                # provider discrepancy in the immutable audit instead of
                # rejecting the correct value or silently averaging it.
                provider_disposition = "amount_mismatch_official_wins"
        decision_records.append(
            {
                "security_id": security_id,
                "record_date": record_date.date().isoformat(),
                "ex_date": expected_ex_date.date().isoformat(),
                "ex_date_rule": KRX_EX_DIVIDEND_RULE,
                "cash_amount": amount,
                "rcept_no": str(decision.rcept_no),
                "disposition": disposition,
                "matched_event_ids": official_ids,
                "provider_disposition": provider_disposition,
                "provider_event_ids": provider_ids,
                "provider_cash_amounts": provider_amounts,
            }
        )

    dividend_ex_keys = {
        (value["security_id"], value["ex_date"])
        for value in decision_records
    }
    cash_reference_overlap_count = 0
    for record in reference_report.get("records") or ():
        if (
            str(record.get("record_kind") or "")
            != "reference_discontinuity"
        ):
            continue
        key = (
            str(record.get("security_id") or ""),
            str(record.get("session") or ""),
        )
        if key not in dividend_ex_keys:
            continue
        cash_reference_overlap_count += 1
        related_ratio = record.get("related_share_ratio")
        try:
            has_share_ratio = (
                related_ratio is not None
                and float(related_ratio) > 0
            )
        except (TypeError, ValueError):
            has_share_ratio = False
        if not has_share_ratio:
            issue(
                "cash_dividend_reference_overlap_unclassified",
                security_id=key[0],
                detail=key[1],
            )

    unmatched_actions = provider_cash_actions.loc[
        ~provider_cash_actions["event_id"]
        .astype(str)
        .isin(matched_provider_action_ids)
    ]
    canonical_event_ids = set(actions["event_id"].astype(str))
    rejected_provider_action_ids: set[str] = set()
    for action in unmatched_actions.itertuples(index=False):
        event_id = str(action.event_id)
        if event_id in canonical_event_ids:
            issue(
                "unsupported_provider_dividend_retained",
                security_id=str(action.security_id),
                detail=(
                    f"{action.effective_date}:"
                    f"{float(action.audit_cash_amount)}:{event_id}"
                ),
            )
        else:
            rejected_provider_action_ids.add(event_id)

    known_gaps = sorted(
        {
            str(value)
            for value in provider_failures
            if str(value).startswith(
                "known_unavailable_pre2019_delisted_actions:"
            )
        }
    )
    gap_records: list[dict[str, str]] = []
    provider_symbol_to_id = (
        catalog.frame.loc[
            catalog.frame["provider_symbol"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        ]
        .drop_duplicates("provider_symbol", keep="last")
        .set_index("provider_symbol")["security_id"]
        .astype(str)
        .to_dict()
    )
    decision_issue_ids = {
        value["security_id"]
        for value in issues
        if value["code"]
        in {
            "missing_official_dividend_action",
            "duplicate_dividend_actions",
            "provider_dividend_without_official_filing",
        }
    }
    reference_passed = (
        reference_report.get("status") == "passed"
        and int(reference_report.get("unresolved_count") or 0) == 0
    )
    for value in known_gaps:
        _, provider_symbol, endpoint = value.rsplit(":", 2)
        security_id = str(provider_symbol_to_id.get(provider_symbol) or "")
        if endpoint == "div":
            resolved = bool(
                dart_result.status == "passed"
                and security_id
                and security_id not in decision_issue_ids
            )
        else:
            resolved = bool(reference_passed and security_id)
        if not resolved:
            issue(
                "unresolved_provider_action_gap",
                security_id=security_id,
                detail=value,
            )
        gap_records.append(
            {
                "provider_gap": value,
                "security_id": security_id,
                "resolution": (
                    "official_cross_check_passed"
                    if resolved
                    else "unresolved"
                ),
            }
        )

    action_inventory = (
        actions[
            [
                "event_id",
                "security_id",
                "action_type",
                "effective_date",
                "ex_date",
                "record_date",
                "cash_amount",
                "ratio",
                "source_hash",
            ]
        ]
        .fillna("")
        .sort_values("event_id", kind="stable")
        .to_dict("records")
    )
    payload = {
        "schema": "kr_corporate_action_verification/v1",
        "status": "passed" if not issues else "blocked",
        "market": "KR",
        "start": str(start),
        "end": str(end),
        "action_count": len(actions),
        "action_inventory_sha256": sha256_bytes(
            canonical_json_bytes(action_inventory)
        ),
        "reference_price_audit_sha256": (
            reference_price_audit.source_hash
        ),
        "reference_discontinuity_count": int(
            reference_report.get("reference_discontinuity_count") or 0
        ),
        "reference_generated_adjustment_count": int(
            reference_report.get("generated_adjustment_count") or 0
        ),
        "reference_generated_restatement_noop_count": int(
            reference_report.get(
                "generated_restatement_noop_count"
            )
            or 0
        ),
        "reference_provider_ratio_restatement_noop_count": int(
            reference_report.get(
                "provider_ratio_restatement_noop_count"
            )
            or 0
        ),
        "reference_provider_ratio_outside_factor_domain_count": int(
            reference_report.get(
                "provider_ratio_outside_factor_domain_count"
            )
            or 0
        ),
        "reference_unresolved_count": int(
            reference_report.get("unresolved_count") or 0
        ),
        "ex_dividend_rule": KRX_EX_DIVIDEND_RULE,
        "ex_dividend_rule_url": KRX_EX_DIVIDEND_RULE_URL,
        "opendart_collection_sha256": dart_result.report.source_hash,
        "opendart_decision_count": len(decisions),
        "matched_dividend_count": matched_decisions,
        "missing_dividend_count": sum(
            value["code"] == "missing_official_dividend_action"
            for value in issues
        ),
        "unmatched_provider_dividend_count": sum(
            str(value) in canonical_event_ids
            for value in unmatched_actions["event_id"].astype(str)
        ),
        "rejected_provider_dividend_count": len(
            rejected_provider_action_ids
        ),
        "provider_dividend_without_official_filing_count": len(
            unmatched_actions
        ),
        "provider_dividend_amount_match_count": (
            provider_amount_match_count
        ),
        "provider_dividend_amount_mismatch_count": (
            provider_amount_mismatch_count
        ),
        "provider_dividend_date_mismatch_count": (
            provider_date_mismatch_count
        ),
        "provider_dividend_missing_count": provider_missing_count,
        "cash_dividend_reference_overlap_count": (
            cash_reference_overlap_count
        ),
        "known_provider_gap_count": len(known_gaps),
        "resolved_provider_gap_count": sum(
            value["resolution"] == "official_cross_check_passed"
            for value in gap_records
        ),
        "blocking_issue_count": len(issues),
        "decision_inventory_sha256": sha256_bytes(
            canonical_json_bytes(decision_records)
        ),
        "decisions": decision_records,
        "provider_gaps": gap_records,
        "issues": issues,
    }
    return SourceArtifact(
        source="opendart_action_audit",
        source_url="https://opendart.fss.or.kr/",
        retrieved_at=utc_now_iso(),
        content=canonical_json_bytes(payload),
        content_type="application/json",
    )


def _combine_krx_reference_price_audits(
    previous: SourceArtifact | None,
    current: SourceArtifact,
) -> SourceArtifact:
    """Merge an overlap audit into the prior full-history KRX audit."""

    reports: list[dict[str, Any]] = []
    for artifact in (previous, current):
        if artifact is None:
            continue
        try:
            report = json.loads(artifact.content)
        except (TypeError, ValueError, json.JSONDecodeError):
            report = {}
        if (
            report.get("schema") != "krx_reference_price_audit/v1"
            or report.get("status") != "passed"
            or int(report.get("unresolved_count") or 0) != 0
        ):
            raise RuntimeError(
                "Cannot extend a blocked or invalid KRX reference-price audit."
            )
        reports.append(report)
    previous_ratio_event_ids = {
        str(event_id)
        for record in (
            (reports[0].get("records") or ())
            if previous is not None
            else ()
        )
        for event_id in record.get("related_ratio_event_ids") or ()
        if str(event_id)
    }
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    observation_count = 0
    observation_counts_by_session: dict[str, int] = {}
    exact_observation_count = True
    for report_index, report in enumerate(reports):
        observation_count += int(report.get("observation_count") or 0)
        session_counts = report.get(
            "observation_counts_by_session"
        )
        if not isinstance(session_counts, dict):
            exact_observation_count = False
        else:
            for session, count in session_counts.items():
                observation_counts_by_session[str(session)] = int(
                    count
                )
        for record in report.get("records") or ():
            record_kind = str(
                record.get("record_kind")
                or "reference_discontinuity"
            )
            if record_kind == "provider_ratio_outside_audit_window":
                related_ids = {
                    str(value)
                    for value in (
                        record.get("related_ratio_event_ids") or ()
                    )
                    if str(value)
                }
                if (
                    previous is None
                    or report_index == 0
                    or not related_ids.issubset(
                        previous_ratio_event_ids
                    )
                ):
                    raise RuntimeError(
                        "A historical KR ratio event falls outside the "
                        "incremental KRX audit window. Run a full KR "
                        "bootstrap before publishing the sync."
                    )
                continue
            key = (
                str(record.get("security_id") or ""),
                str(record.get("session") or ""),
                record_kind,
            )
            if all(key[:2]):
                candidate = dict(record)
                prior = records.get(key)
                if (
                    prior is not None
                    and str(prior.get("resolution") or "").startswith(
                        "generated_official_"
                    )
                    and str(
                        candidate.get("resolution") or ""
                    ).startswith("covered_by_existing_")
                    and str(prior.get("event_id") or "")
                    == str(candidate.get("event_id") or "")
                ):
                    candidate["resolution"] = prior["resolution"]
                records[key] = candidate
    if exact_observation_count:
        observation_count = sum(
            observation_counts_by_session.values()
        )
    audit_starts = sorted(
        str(report.get("audit_start") or "")
        for report in reports
        if str(report.get("audit_start") or "")
    )
    audit_ends = sorted(
        str(report.get("audit_end") or "")
        for report in reports
        if str(report.get("audit_end") or "")
    )
    ordered = [
        records[key]
        for key in sorted(records)
    ]
    related_ratio_event_ids = {
        str(event_id)
        for value in ordered
        for event_id in value.get("related_ratio_event_ids") or ()
        if str(event_id)
    }
    payload = {
        "schema": "krx_reference_price_audit/v1",
        "status": "passed",
        "market": "KR",
        "observation_count": observation_count,
        "audit_start": (
            audit_starts[0]
            if audit_starts
            else ""
        ),
        "audit_end": (
            audit_ends[-1]
            if audit_ends
            else ""
        ),
        "observation_counts_by_session": (
            observation_counts_by_session
            if exact_observation_count
            else {}
        ),
        "reference_discontinuity_count": sum(
            value.get("record_kind", "reference_discontinuity")
            == "reference_discontinuity"
            for value in ordered
        ),
        "covered_existing_action_count": sum(
            value.get("resolution")
            == "covered_by_existing_reference_adjustment"
            for value in ordered
        ),
        "generated_adjustment_count": sum(
            value.get("resolution")
            in {
                "generated_official_reference_adjustment",
                "generated_official_restatement_noop",
            }
            for value in ordered
        ),
        "generated_restatement_noop_count": sum(
            value.get("resolution")
            == "generated_official_restatement_noop"
            for value in ordered
        ),
        "reference_adjustment_event_count": sum(
            value.get("record_kind", "reference_discontinuity")
            in {
                "reference_discontinuity",
                "provider_ratio_restatement_noop",
            }
            and bool(value.get("event_id"))
            for value in ordered
        ),
        "provider_ratio_action_count": len(
            related_ratio_event_ids
        ),
        "provider_ratio_accounted_action_count": len(
            related_ratio_event_ids
        ),
        "provider_ratio_unaccounted_count": 0,
        "provider_ratio_outside_factor_domain_count": sum(
            len(value.get("related_ratio_event_ids") or ())
            for value in ordered
            if value.get("record_kind")
            == "provider_ratio_outside_factor_domain"
        ),
        "provider_ratio_outside_audit_window_count": 0,
        "provider_ratio_restatement_noop_count": sum(
            value.get("record_kind")
            == "provider_ratio_restatement_noop"
            for value in ordered
        ),
        "unresolved_count": 0,
        "records": ordered,
        "parent_audit_sha256": (
            previous.source_hash if previous is not None else ""
        ),
        "incremental_audit_sha256": current.source_hash,
    }
    return SourceArtifact(
        source="krx_reference_price_audit",
        source_url=KRX_SOURCE_URL,
        retrieved_at=utc_now_iso(),
        content=canonical_json_bytes(payload),
        content_type="application/json",
    )


def _release_source_artifact(
    repository: LocalDatasetRepository,
    release: DataRelease,
    *,
    metadata_key: str,
) -> SourceArtifact | None:
    source_hash = str(release.metadata.get(metadata_key) or "").lower()
    if not _valid_sha256(source_hash):
        return None
    archive = repository.read_frame(
        "source_archive",
        release.dataset_versions["source_archive"],
    )
    matches = archive.loc[
        archive["archive_id"].astype(str).str.lower().eq(source_hash)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"KR release artifact {metadata_key} is not uniquely archived."
        )
    row = matches.iloc[0]
    path = repository.root / str(row["object_path"])
    try:
        content = gzip.decompress(path.read_bytes())
    except (OSError, EOFError):
        raise RuntimeError(
            f"KR release artifact {metadata_key} is unreadable."
        ) from None
    if sha256_bytes(content) != source_hash:
        raise RuntimeError(
            f"KR release artifact {metadata_key} hash mismatch."
        )
    return SourceArtifact(
        source=str(row["source"]),
        source_url=str(row.get("source_url") or ""),
        retrieved_at=str(row["retrieved_at"]),
        content=content,
        content_type=str(row["content_type"]),
    )


def _assert_krx_observations_classified(
    observations: pd.DataFrame,
    *,
    context: str,
) -> None:
    if observations.empty or "observation_status" not in observations:
        return
    unresolved = observations.loc[
        ~observations["observation_status"]
        .astype(str)
        .isin(KRX_ALLOWED_OBSERVATION_STATUSES)
    ]
    if unresolved.empty:
        return
    examples = ", ".join(
        f"{row.security_id}/{row.session}/{row.observation_status}"
        for row in unresolved.head(20).itertuples(index=False)
    )
    raise RuntimeError(
        f"KRX {context} contains {len(unresolved)} unclassified official "
        f"observations: {examples}"
    )


def _krx_index_price_gap_policy(
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    raw_prices: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    existing_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind every index/price edge gap to exact official no-trade rows."""

    gap_records = index_member_price_gap_records(
        anchors,
        events,
        raw_prices,
    )
    prepared = observations.copy()
    if not prepared.empty:
        prepared["_session"] = pd.to_datetime(
            prepared["session"], errors="coerce"
        ).dt.normalize()
        prepared["_security_id"] = prepared["security_id"].astype(str)
        prepared["_status"] = prepared["observation_status"].astype(str)
        prepared["_source_url"] = prepared["source_url"].astype(str)
        prepared["_source_hash"] = (
            prepared["source_hash"].astype(str).str.lower()
        )
        prepared = prepared.loc[
            prepared["_session"].notna()
        ].drop_duplicates(
            ["_security_id", "_session"],
            keep="last",
        )

    identity_fields = (
        "code",
        "index_id",
        "security_id",
        "membership_start",
        "membership_end",
        "missing_from",
        "missing_through",
    )
    existing_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    if isinstance(existing_policy, dict):
        for raw in existing_policy.get("gaps", []):
            if isinstance(raw, dict):
                existing_by_key[
                    tuple(str(raw.get(field) or "") for field in identity_fields)
                ] = dict(raw)

    reviewed_gaps: list[dict[str, Any]] = []
    for gap in gap_records:
        key = tuple(str(gap.get(field) or "") for field in identity_fields)
        start = pd.Timestamp(gap["missing_from"]).normalize()
        end = pd.Timestamp(gap["missing_through"]).normalize()
        expected_sessions = set(
            _sessions_between(
                start.date().isoformat(),
                end.date().isoformat(),
            )
        )
        evidence = prepared.loc[
            prepared["_security_id"].eq(str(gap["security_id"]))
            & prepared["_session"].ge(start)
            & prepared["_session"].le(end)
        ].copy()
        observed_sessions = {
            value.date().isoformat()
            for value in evidence["_session"]
        }
        evidence_valid = (
            bool(expected_sessions)
            and observed_sessions == expected_sessions
            and evidence["_status"]
            .isin(KRX_CLASSIFIED_NO_TRADE_STATUSES)
            .all()
            and evidence["_source_url"]
            .str.startswith(KRX_SOURCE_URL)
            .all()
            and evidence["_source_hash"].map(_valid_sha256).all()
        )
        if not evidence_valid:
            previous = existing_by_key.get(key)
            if previous is not None:
                reviewed_gaps.append(previous)
                continue
            missing = sorted(expected_sessions - observed_sessions)
            unexpected = sorted(observed_sessions - expected_sessions)
            bad_statuses = sorted(
                set(evidence["_status"])
                - set(KRX_CLASSIFIED_NO_TRADE_STATUSES)
            )
            raise RuntimeError(
                "KR index price gap lacks complete classified KRX evidence: "
                f"{gap['index_id']}/{gap['security_id']}/"
                f"{gap['missing_from']}..{gap['missing_through']}; "
                f"missing_sessions={missing[:10]}, "
                f"unexpected_sessions={unexpected[:10]}, "
                f"bad_statuses={bad_statuses}"
            )
        rows = [
            {
                "session": pd.Timestamp(row["_session"]).date().isoformat(),
                "observation_status": str(row["_status"]),
                "source_hash": str(row["_source_hash"]),
            }
            for row in evidence.sort_values("_session").to_dict("records")
        ]
        status_counts = {
            str(status): int(count)
            for status, count in evidence["_status"]
            .value_counts()
            .sort_index()
            .items()
        }
        reviewed_gaps.append(
            {
                **gap,
                "observation_count": len(rows),
                "status_counts": status_counts,
                "evidence_sha256": sha256_bytes(
                    canonical_json_bytes(rows)
                ),
            }
        )

    reviewed_gaps.sort(
        key=lambda item: tuple(
            str(item.get(field) or "") for field in identity_fields
        )
    )
    policy: dict[str, Any] = {
        "schema": INDEX_PRICE_GAP_POLICY_SCHEMA,
        "source": "krx_official_daily_price_observations",
        "source_url": KRX_SOURCE_URL,
        "security_ids": sorted(
            {str(item["security_id"]) for item in reviewed_gaps}
        ),
        "gap_count": len(reviewed_gaps),
        "observation_count": sum(
            int(item["observation_count"]) for item in reviewed_gaps
        ),
        "gaps": reviewed_gaps,
    }
    policy["policy_sha256"] = index_price_gap_policy_sha256(policy)
    return policy


def _catalog_for_symbols(
    catalog: KrIdentityCatalog,
    symbols: Iterable[str],
    start: str,
    end: str,
) -> tuple[KrIdentityCatalog, tuple[str, ...]]:
    selected = {normalize_kr_symbol(symbol) for symbol in symbols}
    frame = catalog.frame.loc[
        catalog.frame["primary_symbol"].astype(str).isin(selected)
    ].copy()
    starts = pd.to_datetime(frame["active_from"], errors="coerce")
    ends = pd.to_datetime(frame["active_to"], errors="coerce")
    frame = frame.loc[
        (starts.isna() | starts.le(pd.Timestamp(end)))
        & (ends.isna() | ends.ge(pd.Timestamp(start)))
    ]
    mapped = set(
        frame.loc[frame["identity_mapped"].eq(True), "primary_symbol"].astype(str)  # noqa: E712
    )
    missing = tuple(sorted(selected - mapped))
    return KrIdentityCatalog(frame.reset_index(drop=True), catalog.artifacts, catalog.dart_status), missing


def _identity_datasets(
    catalog: KrIdentityCatalog,
    symbols: Iterable[str],
    default_start: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = {normalize_kr_symbol(symbol) for symbol in symbols}
    frame = catalog.frame.loc[catalog.frame["primary_symbol"].astype(str).isin(selected)].copy()
    history = pd.DataFrame(
        {
            "security_id": frame["security_id"].astype(str),
            "symbol": frame["primary_symbol"].astype(str),
            "exchange": frame["exchange"].astype(str),
            "effective_from": frame["active_from"].where(
                frame["active_from"].astype(str).str.strip().ne(""), default_start
            ),
            "effective_to": frame["active_to"].fillna(""),
            "source": frame["source"],
            "source_url": frame["source_url"],
            "retrieved_at": frame["retrieved_at"],
            "source_hash": frame["source_hash"],
        }
    ).drop_duplicates(["security_id", "symbol", "effective_from"], keep="last")
    master = frame.sort_values(
        ["security_id", "active_from", "active_to"], kind="stable"
    ).drop_duplicates("security_id", keep="last")
    required = list(dataset_spec("security_master").required_columns)
    extras = [
        "isin",
        "dart_corp_code",
        "identity_mapped",
        "provider_symbol",
        "yahoo_symbol",
    ]
    master = master.loc[:, [column for column in (*required, *extras) if column in master]]
    return master.reset_index(drop=True), history.reset_index(drop=True)


def _identity_catalog_from_release_frames(
    master: pd.DataFrame,
    history: pd.DataFrame,
) -> KrIdentityCatalog:
    """Reconstruct the exact historical symbol/ISIN intervals in a release."""

    identity_columns = [
        column
        for column in master
        if column not in {"active_from", "active_to", "primary_symbol"}
    ]
    values = history.merge(
        master[identity_columns].drop_duplicates("security_id", keep="last"),
        on="security_id",
        how="left",
        suffixes=("", "_master"),
        validate="many_to_one",
    )
    values["primary_symbol"] = values["symbol"].map(normalize_kr_symbol)
    values["active_from"] = values["effective_from"]
    values["active_to"] = values["effective_to"]
    values["identity_mapped"] = True
    return KrIdentityCatalog(values, ())


def _membership_datasets(
    memberships: dict[str, dict[str, dict[str, Any]]],
    catalog: KrIdentityCatalog,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def interval_date(value: Any) -> str:
        if value is None or pd.isna(value) or not str(value).strip():
            return ""
        return pd.Timestamp(value).date().isoformat()

    intervals: dict[str, list[tuple[str, str, str]]] = {}
    for row in catalog.frame.itertuples(index=False):
        if hasattr(row, "identity_mapped") and not bool(row.identity_mapped):
            continue
        symbol = normalize_kr_symbol(row.primary_symbol)
        intervals.setdefault(symbol, []).append(
            (
                interval_date(row.active_from),
                interval_date(row.active_to),
                str(row.security_id),
            )
        )

    def stable_ids(snapshot: dict[str, Any], session: str) -> set[str]:
        symbols = {
            normalize_kr_symbol(symbol)
            for symbol in snapshot["symbols"]
        }
        resolved: dict[str, str] = {}
        missing: list[str] = []
        ambiguous: list[str] = []
        for symbol in symbols:
            candidates = {
                security_id
                for active_from, active_to, security_id in intervals.get(
                    symbol,
                    (),
                )
                if (not active_from or active_from <= session)
                and (not active_to or active_to >= session)
            }
            if len(candidates) == 1:
                resolved[symbol] = next(iter(candidates))
            elif not candidates:
                missing.append(symbol)
            else:
                ambiguous.append(symbol)
        if missing:
            raise RuntimeError(
                "KRX membership cannot resolve official daily ISIN identity "
                f"on {session}: {', '.join(missing[:20])}"
            )
        if ambiguous:
            raise RuntimeError(
                "KRX membership resolves one symbol to multiple active ISINs "
                f"on {session}: {', '.join(sorted(ambiguous)[:20])}"
            )
        security_ids = set(resolved.values())
        if len(security_ids) != len(symbols):
            raise RuntimeError(
                "KRX membership maps multiple symbols to one security identity "
                f"on {session}."
            )
        return security_ids

    anchors: list[dict] = []
    events: list[dict] = []
    for profile, snapshots in memberships.items():
        ordered = sorted(snapshots.items())
        anchor_session, anchor = ordered[0]
        previous = stable_ids(anchor, anchor_session)
        for security_id in sorted(previous):
            anchors.append(
                {
                    "index_id": profile,
                    "anchor_date": anchor_session,
                    "security_id": security_id,
                    "official": True,
                    "source_url": anchor["source_url"],
                    "source_kind": "official",
                    "source": anchor["source"],
                    "retrieved_at": anchor["retrieved_at"],
                    "source_hash": anchor["source_hash"],
                }
            )
        for session, snapshot in ordered[1:]:
            current = stable_ids(snapshot, session)
            for operation, security_ids in (
                ("REMOVE", previous - current),
                ("ADD", current - previous),
            ):
                for security_id in sorted(security_ids):
                    event_id = sha256_bytes(
                        f"KRX|{profile}|{session}|{operation}|{security_id}".encode()
                    )
                    events.append(
                        {
                            "event_id": event_id,
                            "index_id": profile,
                            "announcement_date": "",
                            "effective_date": session,
                            "operation": operation,
                            "security_id": security_id,
                            "official": True,
                            "source_url": snapshot["source_url"],
                            "source_kind": "official",
                            "source": snapshot["source"],
                            "retrieved_at": snapshot["retrieved_at"],
                            "source_hash": snapshot["source_hash"],
                        }
                    )
            previous = current
    anchor_frame = pd.DataFrame(anchors, columns=dataset_spec("index_constituent_anchors").required_columns)
    event_frame = pd.DataFrame(events, columns=dataset_spec("index_membership_events").required_columns)
    return anchor_frame, event_frame


def verify_kr_membership_history(
    memberships: dict[str, dict[str, dict[str, Any]]],
    catalog: KrIdentityCatalog,
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    *,
    sessions: Iterable[str],
    profiles: Iterable[str] = KR_PROFILES,
    checkpoint: KrCheckpointStore | None = None,
) -> dict[str, Any]:
    """Replay release events against every official KRX daily snapshot.

    This verifier deliberately does not infer survivorship quality from price
    coverage. It resolves each snapshot to the active ISIN inventory, replays
    only the persisted anchor plus ADD/REMOVE events, and requires exact daily
    equality for the complete requested XKRX session inventory.
    """

    requested_sessions = tuple(
        sorted(
            {
                pd.Timestamp(value).date().isoformat()
                for value in sessions
            }
        )
    )
    selected_profiles = tuple(dict.fromkeys(str(value).lower() for value in profiles))
    issues: list[dict[str, str]] = []

    def issue(
        code: str,
        *,
        profile: str = "",
        session: str = "",
        detail: str = "",
    ) -> None:
        if len(issues) < 100:
            issues.append(
                {
                    "code": code,
                    "profile": profile,
                    "session": session,
                    "detail": detail,
                }
            )

    def interval_date(value: Any) -> str:
        if value is None or pd.isna(value) or not str(value).strip():
            return ""
        return pd.Timestamp(value).date().isoformat()

    intervals: dict[str, list[tuple[str, str, str]]] = {}
    for row in catalog.frame.itertuples(index=False):
        if hasattr(row, "identity_mapped") and not bool(row.identity_mapped):
            continue
        symbol = normalize_kr_symbol(row.primary_symbol)
        intervals.setdefault(symbol, []).append(
            (
                interval_date(row.active_from),
                interval_date(row.active_to),
                str(row.security_id),
            )
        )

    def resolve_snapshot(
        profile: str,
        session: str,
        snapshot: dict[str, Any],
    ) -> set[str]:
        resolved: dict[str, str] = {}
        for raw_symbol in snapshot.get("symbols") or ():
            symbol = normalize_kr_symbol(raw_symbol)
            candidates = {
                security_id
                for active_from, active_to, security_id in intervals.get(symbol, ())
                if (not active_from or active_from <= session)
                and (not active_to or active_to >= session)
            }
            if len(candidates) != 1:
                issue(
                    "identity_resolution_mismatch",
                    profile=profile,
                    session=session,
                    detail=f"{symbol}:{len(candidates)}",
                )
                continue
            resolved[symbol] = next(iter(candidates))
        if len(set(resolved.values())) != len(resolved):
            issue(
                "duplicate_security_identity",
                profile=profile,
                session=session,
            )
        return set(resolved.values())

    anchor_values = anchors.copy()
    event_values = events.copy()
    for frame, column in (
        (anchor_values, "anchor_date"),
        (event_values, "effective_date"),
    ):
        if column in frame:
            frame[column] = pd.to_datetime(
                frame[column], errors="coerce"
            ).dt.date.astype("string")
    if "index_id" in anchor_values:
        anchor_values["index_id"] = anchor_values["index_id"].astype(str).str.lower()
    if "index_id" in event_values:
        event_values["index_id"] = event_values["index_id"].astype(str).str.lower()
    if "operation" in event_values:
        event_values["operation"] = event_values["operation"].astype(str).str.upper()

    duplicate_event_count = (
        int(event_values["event_id"].astype(str).duplicated(keep=False).sum())
        if "event_id" in event_values
        else 0
    )
    if duplicate_event_count:
        issue("duplicate_event_id", detail=str(duplicate_event_count))

    inventory_records: list[dict[str, str]] = []
    profile_reports: dict[str, dict[str, Any]] = {}
    total_expected = 0
    total_observed = 0
    total_missing = 0
    total_replay_mismatches = 0
    total_source_issues = 0
    total_artifact_issues = 0
    total_count_issues = 0
    total_identity_issues_before = 0
    artifact_validity: dict[str, bool] = {}

    for profile in selected_profiles:
        definition = KR_INDEX_DEFINITIONS.get(profile)
        if definition is None:
            issue("unknown_profile", profile=profile)
            continue
        inception = str(definition["announcement_date"])
        expected_sessions = tuple(
            value for value in requested_sessions if value >= inception
        )
        snapshots = memberships.get(profile) or {}
        observed_sessions = tuple(sorted(str(value) for value in snapshots))
        expected_set = set(expected_sessions)
        observed_set = set(observed_sessions)
        missing_sessions = sorted(expected_set - observed_set)
        unexpected_sessions = sorted(observed_set - expected_set)
        total_expected += len(expected_sessions)
        total_observed += len(expected_set & observed_set)
        total_missing += len(missing_sessions)
        for session in missing_sessions[:20]:
            issue("missing_official_snapshot", profile=profile, session=session)
        for session in unexpected_sessions[:20]:
            issue("unexpected_snapshot", profile=profile, session=session)

        resolved_by_session: dict[str, set[str]] = {}
        source_issue_count = 0
        artifact_issue_count = 0
        count_issue_count = 0
        identity_issue_start = len(issues)
        expected_count = int(definition["expected_count"])
        tolerance = int(definition.get("count_tolerance", 0))
        for session in sorted(expected_set & observed_set):
            snapshot = snapshots[session]
            symbols = tuple(
                sorted(
                    {
                        normalize_kr_symbol(value)
                        for value in snapshot.get("symbols") or ()
                    }
                )
            )
            source = str(snapshot.get("source") or "")
            source_url = str(snapshot.get("source_url") or "")
            source_hash = str(snapshot.get("source_hash") or "").lower()
            source_valid = (
                source in KR_OFFICIAL_MEMBERSHIP_SOURCES
                and source_url.startswith("https://")
                and _valid_sha256(source_hash)
            )
            if not source_valid:
                source_issue_count += 1
                issue(
                    "invalid_official_snapshot_provenance",
                    profile=profile,
                    session=session,
                    detail=source,
                )
            if checkpoint is not None and _valid_sha256(source_hash):
                artifact_valid = artifact_validity.get(source_hash)
                if artifact_valid is None:
                    evidence_root = (
                        checkpoint.root
                        / "evidence_local"
                        / "krx-membership"
                    )
                    paths = sorted(
                        evidence_root.glob(f"{source_hash}.*.gz")
                    )
                    try:
                        artifact_valid = bool(
                            len(paths) == 1
                            and sha256_bytes(
                                gzip.decompress(paths[0].read_bytes())
                            )
                            == source_hash
                        )
                    except (OSError, EOFError):
                        artifact_valid = False
                    artifact_validity[source_hash] = artifact_valid
                if not artifact_valid:
                    artifact_issue_count += 1
                    issue(
                        "missing_or_corrupt_snapshot_artifact",
                        profile=profile,
                        session=session,
                        detail=source_hash,
                    )
            if not (
                expected_count - tolerance
                <= len(symbols)
                <= expected_count + tolerance
            ):
                count_issue_count += 1
                issue(
                    "invalid_member_count",
                    profile=profile,
                    session=session,
                    detail=str(len(symbols)),
                )
            resolved_by_session[session] = resolve_snapshot(
                profile,
                session,
                {**snapshot, "symbols": symbols},
            )
            inventory_records.append(
                {
                    "profile": profile,
                    "session": session,
                    "member_inventory_sha256": sha256_bytes(
                        canonical_json_bytes(symbols)
                    ),
                    "source_hash": source_hash,
                }
            )

        profile_anchors = anchor_values.loc[
            anchor_values.get(
                "index_id",
                pd.Series(index=anchor_values.index, dtype=str),
            ).astype(str).eq(profile)
        ]
        if profile_anchors.empty or not expected_sessions:
            issue("missing_anchor", profile=profile)
            replay_mismatch_count = len(resolved_by_session)
            current: set[str] = set()
            anchor_session = ""
        else:
            anchor_dates = sorted(
                {
                    str(value)
                    for value in profile_anchors["anchor_date"]
                    if str(value) not in {"", "<NA>", "NaT"}
                }
            )
            anchor_session = anchor_dates[0] if anchor_dates else ""
            if len(anchor_dates) != 1 or anchor_session != expected_sessions[0]:
                issue(
                    "anchor_session_mismatch",
                    profile=profile,
                    session=anchor_session,
                    detail=expected_sessions[0],
                )
            current = set(
                profile_anchors["security_id"].astype(str)
            )
            replay_mismatch_count = 0

        profile_events = event_values.loc[
            event_values.get(
                "index_id",
                pd.Series(index=event_values.index, dtype=str),
            ).astype(str).eq(profile)
        ]
        events_by_session: dict[str, list[Any]] = {}
        for row in profile_events.itertuples(index=False):
            effective = str(row.effective_date)
            events_by_session.setdefault(effective, []).append(row)
            if effective not in expected_set:
                issue(
                    "event_outside_verified_sessions",
                    profile=profile,
                    session=effective,
                )

        change_session_count = 0
        for session in expected_sessions:
            if session != anchor_session:
                session_events = events_by_session.get(session, ())
                if session_events:
                    change_session_count += 1
                removes = {
                    str(row.security_id)
                    for row in session_events
                    if str(row.operation).upper() == "REMOVE"
                }
                adds = {
                    str(row.security_id)
                    for row in session_events
                    if str(row.operation).upper() == "ADD"
                }
                invalid_operations = [
                    str(row.operation)
                    for row in session_events
                    if str(row.operation).upper() not in {"ADD", "REMOVE"}
                ]
                if invalid_operations:
                    issue(
                        "invalid_event_operation",
                        profile=profile,
                        session=session,
                        detail=",".join(sorted(set(invalid_operations))),
                    )
                if removes - current:
                    issue(
                        "remove_nonmember",
                        profile=profile,
                        session=session,
                        detail=",".join(sorted(removes - current)[:10]),
                    )
                if adds & current:
                    issue(
                        "add_existing_member",
                        profile=profile,
                        session=session,
                        detail=",".join(sorted(adds & current)[:10]),
                    )
                current = (current - removes) | adds
                snapshot = snapshots.get(session)
                if snapshot is not None and session_events:
                    expected_source_hash = str(
                        snapshot.get("source_hash") or ""
                    ).lower()
                    for row in session_events:
                        if (
                            not bool(row.official)
                            or str(row.source_kind) != "official"
                            or str(row.source_hash).lower()
                            != expected_source_hash
                        ):
                            issue(
                                "event_provenance_mismatch",
                                profile=profile,
                                session=session,
                                detail=str(row.security_id),
                            )
            expected_members = resolved_by_session.get(session)
            if expected_members is not None and current != expected_members:
                replay_mismatch_count += 1
                issue(
                    "daily_replay_mismatch",
                    profile=profile,
                    session=session,
                    detail=(
                        f"missing={len(expected_members - current)},"
                        f"unexpected={len(current - expected_members)}"
                    ),
                )

        unique_members = set().union(*resolved_by_session.values()) if resolved_by_session else set()
        final_members = (
            resolved_by_session[max(resolved_by_session)]
            if resolved_by_session
            else set()
        )
        identity_issue_count = sum(
            1
            for value in issues[identity_issue_start:]
            if value["profile"] == profile
            and value["code"]
            in {"identity_resolution_mismatch", "duplicate_security_identity"}
        )
        total_replay_mismatches += replay_mismatch_count
        total_source_issues += source_issue_count
        total_artifact_issues += artifact_issue_count
        total_count_issues += count_issue_count
        total_identity_issues_before += identity_issue_count
        profile_reports[profile] = {
            "inception": inception,
            "expected_start": expected_sessions[0] if expected_sessions else "",
            "expected_end": expected_sessions[-1] if expected_sessions else "",
            "expected_snapshot_count": len(expected_sessions),
            "observed_snapshot_count": len(expected_set & observed_set),
            "missing_snapshot_count": len(missing_sessions),
            "unexpected_snapshot_count": len(unexpected_sessions),
            "invalid_source_count": source_issue_count,
            "invalid_artifact_count": artifact_issue_count,
            "invalid_member_count": count_issue_count,
            "identity_resolution_issue_count": identity_issue_count,
            "anchor_session": anchor_session,
            "event_count": len(profile_events),
            "change_session_count": change_session_count,
            "daily_replay_mismatch_count": replay_mismatch_count,
            "unique_member_count": len(unique_members),
            "current_member_count": len(final_members),
            "historical_not_current_count": len(unique_members - final_members),
        }

    blocking_issue_count = (
        total_missing
        + total_replay_mismatches
        + total_source_issues
        + total_artifact_issues
        + total_count_issues
        + total_identity_issues_before
        + duplicate_event_count
        + sum(
            1
            for value in issues
            if value["code"]
            in {
                "unknown_profile",
                "unexpected_snapshot",
                "missing_anchor",
                "anchor_session_mismatch",
                "event_outside_verified_sessions",
                "invalid_event_operation",
                "remove_nonmember",
                "add_existing_member",
                "event_provenance_mismatch",
            }
        )
    )
    status = "passed" if total_expected and blocking_issue_count == 0 else "blocked"
    event_inventory = event_values.loc[
        event_values.get(
            "index_id",
            pd.Series(index=event_values.index, dtype=str),
        ).astype(str).isin(selected_profiles)
    ]
    event_records = (
        event_inventory[
            [
                column
                for column in (
                    "event_id",
                    "index_id",
                    "effective_date",
                    "operation",
                    "security_id",
                    "source_hash",
                )
                if column in event_inventory
            ]
        ]
        .fillna("")
        .sort_values(
            [
                column
                for column in ("index_id", "effective_date", "operation", "security_id")
                if column in event_inventory
            ],
            kind="stable",
        )
        .to_dict("records")
    )
    return {
        "schema": KR_HISTORY_VERIFICATION_SCHEMA,
        "status": status,
        "market": "KR",
        "calendar": "XKRX",
        "requested_start": requested_sessions[0] if requested_sessions else "",
        "requested_end": requested_sessions[-1] if requested_sessions else "",
        "expected_snapshot_count": total_expected,
        "observed_snapshot_count": total_observed,
        "missing_snapshot_count": total_missing,
        "daily_replay_mismatch_count": total_replay_mismatches,
        "invalid_source_count": total_source_issues,
        "invalid_artifact_count": total_artifact_issues,
        "invalid_member_count": total_count_issues,
        "identity_resolution_issue_count": total_identity_issues_before,
        "duplicate_event_count": duplicate_event_count,
        "blocking_issue_count": blocking_issue_count,
        "survivorship_score": 1.0 if status == "passed" else 0.0,
        "snapshot_inventory_sha256": sha256_bytes(
            canonical_json_bytes(inventory_records)
        ),
        "event_inventory_sha256": sha256_bytes(
            canonical_json_bytes(event_records)
        ),
        "profiles": profile_reports,
        "issues": issues,
    }


def _incremental_membership_events(
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    memberships: dict[str, dict[str, dict[str, Any]]],
    catalog: KrIdentityCatalog,
) -> pd.DataFrame:
    records: list[dict] = []
    for profile, snapshots in memberships.items():
        first_session = min(snapshots)
        previous_ids = _replay_members(anchors, events, profile, first_session)
        for session, snapshot in sorted(snapshots.items()):
            current_ids = {
                catalog.security_id_for(symbol, session) for symbol in snapshot["symbols"]
            }
            for operation, ids in (("REMOVE", previous_ids - current_ids), ("ADD", current_ids - previous_ids)):
                for security_id in sorted(ids):
                    event_id = sha256_bytes(
                        f"KRX|{profile}|{session}|{operation}|{security_id}".encode()
                    )
                    records.append(
                        {
                            "event_id": event_id,
                            "index_id": profile,
                            "announcement_date": "",
                            "effective_date": session,
                            "operation": operation,
                            "security_id": security_id,
                            "official": True,
                            "source_url": snapshot["source_url"],
                            "source_kind": "official",
                            "source": snapshot["source"],
                            "retrieved_at": snapshot["retrieved_at"],
                            "source_hash": snapshot["source_hash"],
                        }
                    )
            previous_ids = current_ids
    return pd.DataFrame(records, columns=dataset_spec("index_membership_events").required_columns).drop_duplicates(
        "event_id", keep="last"
    )


def _replay_members(
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    profile: str,
    session: str,
) -> set[str]:
    profile_anchors = anchors.loc[anchors["index_id"].astype(str).eq(profile)].copy()
    profile_anchors["_date"] = pd.to_datetime(profile_anchors["anchor_date"], errors="coerce")
    eligible_dates = profile_anchors.loc[profile_anchors["_date"].le(pd.Timestamp(session)), "_date"]
    if eligible_dates.empty:
        return set()
    anchor_date = eligible_dates.max()
    members = set(
        profile_anchors.loc[profile_anchors["_date"].eq(anchor_date), "security_id"].astype(str)
    )
    profile_events = events.loc[events["index_id"].astype(str).eq(profile)].copy()
    profile_events["_date"] = pd.to_datetime(profile_events["effective_date"], errors="coerce")
    for row in profile_events.loc[
        profile_events["_date"].gt(anchor_date)
        & profile_events["_date"].le(pd.Timestamp(session))
    ].sort_values(["_date", "event_id"]).itertuples(index=False):
        if str(row.operation).upper() == "ADD":
            members.add(str(row.security_id))
        else:
            members.discard(str(row.security_id))
    return members


def _resolve_kr_lifecycle_candidates(
    repository: LocalDatasetRepository,
    release: DataRelease,
    actions: pd.DataFrame,
) -> tuple[pd.DataFrame, Any]:
    candidates = build_lifecycle_candidates(repository, release=release, stale_days=30)
    membership_events = repository.read_frame(
        "index_membership_events",
        release.dataset_versions.get("index_membership_events"),
    )
    security_master = repository.read_frame(
        "security_master",
        release.dataset_versions.get("security_master"),
    )
    lifecycle_actions = actions.loc[
        actions["action_type"].astype(str).isin(
            {"delisting", "cash_merger", "stock_merger", "ticker_change"}
        )
        & actions["official"].eq(True)  # noqa: E712
    ].copy()
    lifecycle_actions["_effective"] = pd.to_datetime(
        lifecycle_actions["effective_date"], errors="coerce"
    )
    records: list[dict] = []
    for candidate in candidates:
        last_price = pd.Timestamp(candidate.last_price_date)
        matches = lifecycle_actions.loc[
            lifecycle_actions["security_id"].astype(str).eq(candidate.security_id)
            & lifecycle_actions["_effective"].notna()
            & lifecycle_actions["_effective"].ge(
                last_price - pd.Timedelta(days=10)
            )
        ].copy()
        if matches.empty:
            continue
        matches["_distance"] = (
            matches["_effective"] - last_price
        ).abs()
        action = matches.sort_values(
            ["_distance", "_effective", "event_id"], kind="stable"
        ).iloc[0]
        records.append(
            {
                "candidate_id": lifecycle_candidate_id(
                    candidate.security_id,
                    candidate.last_price_date,
                    selection_rule=KR_LIFECYCLE_SELECTION_RULE,
                ),
                "security_id": candidate.security_id,
                "symbol": candidate.symbol,
                "last_price_date": candidate.last_price_date,
                "resolution": "applied",
                "event_id": str(action["event_id"]),
                "exception_code": "",
                "exception_reason": "",
                "reviewed_by": "kr_pipeline_official_action_v1",
                "reviewed_at": utc_now_iso(),
                "recheck_after": "",
                "successor_security_id": str(action.get("new_security_id") or ""),
                "successor_symbol": str(action.get("new_symbol") or ""),
                "source_url": str(action["source_url"]),
                "source": "kr_lifecycle_resolution",
                "retrieved_at": str(action["retrieved_at"]),
                "source_hash": str(action["source_hash"]),
            }
        )
    resolved_candidate_ids = {
        str(record["candidate_id"]) for record in records
    }
    remove_events = membership_events.loc[
        membership_events["operation"].astype(str).str.upper().eq("REMOVE")
    ].copy()
    remove_events["_effective"] = pd.to_datetime(
        remove_events["effective_date"], errors="coerce"
    )
    master_by_id = security_master.sort_values(
        ["security_id", "active_from", "active_to"], kind="stable"
    ).drop_duplicates("security_id", keep="last")
    master_by_id = master_by_id.set_index(
        master_by_id["security_id"].astype(str),
        drop=False,
    )
    recheck_after = (
        pd.Timestamp(release.completed_session) + pd.Timedelta(days=31)
    ).date().isoformat()
    for candidate in candidates:
        candidate_id = lifecycle_candidate_id(
            candidate.security_id,
            candidate.last_price_date,
            selection_rule=KR_LIFECYCLE_SELECTION_RULE,
        )
        if candidate_id in resolved_candidate_ids:
            continue
        last_price = pd.Timestamp(candidate.last_price_date)
        prior_removes = remove_events.loc[
            remove_events["security_id"].astype(str).eq(candidate.security_id)
            & remove_events["_effective"].notna()
            & remove_events["_effective"].le(last_price)
        ].copy()
        if not prior_removes.empty:
            evidence = prior_removes.sort_values(
                ["_effective", "event_id"], kind="stable"
            ).iloc[-1]
            removal_date = pd.Timestamp(evidence["_effective"]).date().isoformat()
            records.append(
                {
                    "candidate_id": candidate_id,
                    "security_id": candidate.security_id,
                    "symbol": candidate.symbol,
                    "last_price_date": candidate.last_price_date,
                    "resolution": "exception",
                    "event_id": "",
                    "exception_code": str(
                        LifecycleExceptionCode.ALREADY_REPRESENTED
                    ),
                    "exception_reason": (
                        f"Official {str(evidence['index_id'])} removal became "
                        f"effective on {removal_date}, while the security "
                        f"remained tradable through {candidate.last_price_date}. "
                        "An index-constrained position therefore had an official "
                        "tradable exit before the later terminal lifecycle event."
                    ),
                    "reviewed_by": "kr_pipeline_official_index_exit_v1",
                    "reviewed_at": str(evidence["retrieved_at"]),
                    "recheck_after": "",
                    "successor_security_id": "",
                    "successor_symbol": "",
                    "source_url": str(evidence["source_url"]),
                    "source": "kr_lifecycle_resolution",
                    "retrieved_at": str(evidence["retrieved_at"]),
                    "source_hash": str(evidence["source_hash"]),
                }
            )
            resolved_candidate_ids.add(candidate_id)
            continue
        if candidate.active_to:
            continue
        if candidate.security_id not in master_by_id.index:
            continue
        evidence = master_by_id.loc[candidate.security_id]
        if isinstance(evidence, pd.DataFrame):
            evidence = evidence.iloc[-1]
        records.append(
            {
                "candidate_id": candidate_id,
                "security_id": candidate.security_id,
                "symbol": candidate.symbol,
                "last_price_date": candidate.last_price_date,
                "resolution": "exception",
                "event_id": "",
                "exception_code": str(
                    LifecycleExceptionCode.INSUFFICIENT_OFFICIAL_EVIDENCE
                ),
                "exception_reason": (
                    "KRX still carries this stable identity without an official "
                    "delisting date, while regular trading is stale. Treat it as "
                    "a suspended/nonterminal security and recheck before the "
                    "temporary exception expires."
                ),
                "reviewed_by": "kr_pipeline_krx_active_identity_v1",
                "reviewed_at": str(evidence["retrieved_at"]),
                "recheck_after": recheck_after,
                "successor_security_id": "",
                "successor_symbol": "",
                "source_url": KRX_SOURCE_URL,
                "source": "kr_lifecycle_resolution",
                "retrieved_at": str(evidence["retrieved_at"]),
                "source_hash": str(evidence["source_hash"]),
            }
        )
        resolved_candidate_ids.add(candidate_id)
    automatic = pd.DataFrame(
        records,
        columns=dataset_spec("lifecycle_resolutions").required_columns,
    )
    approved, approved_artifact = _load_approved_kr_lifecycle_exceptions()
    if approved_artifact is not None:
        KrCheckpointStore(repository.root / "state" / "lifecycle").save_local_artifact(
            approved_artifact,
            scope="approved-exceptions",
        )
    if not approved.empty:
        candidate_symbols = {
            lifecycle_candidate_id(
                candidate.security_id,
                candidate.last_price_date,
                selection_rule=KR_LIFECYCLE_SELECTION_RULE,
            ): candidate.symbol
            for candidate in candidates
        }
        blank_symbol = approved["symbol"].fillna("").astype(str).str.strip().eq("")
        approved.loc[blank_symbol, "symbol"] = approved.loc[
            blank_symbol, "candidate_id"
        ].map(candidate_symbols).fillna("")
    # A reviewed exception carries more specific official evidence than the
    # automatic KRX-state fallback for the same exact candidate.
    resolutions = pd.concat([automatic, approved], ignore_index=True).drop_duplicates(
        "candidate_id", keep="last"
    )
    resolutions["reviewed_at"] = resolutions[
        "reviewed_at"
    ].map(_normalized_timestamp)
    _save_kr_lifecycle_review_checkpoint(
        repository,
        release,
        candidates,
        resolutions,
    )
    candidate_frame = pd.DataFrame(
        [
            {
                "candidate_id": lifecycle_candidate_id(
                    candidate.security_id,
                    candidate.last_price_date,
                    selection_rule=KR_LIFECYCLE_SELECTION_RULE,
                ),
                "security_id": candidate.security_id,
                "last_price_date": candidate.last_price_date,
            }
            for candidate in candidates
        ],
        columns=("candidate_id", "security_id", "last_price_date"),
    )
    coverage = validate_lifecycle_coverage(
        candidate_frame,
        resolutions,
        actions,
        completed_session=release.completed_session,
        selection_rule=KR_LIFECYCLE_SELECTION_RULE,
    )
    if not coverage.valid:
        detail = "; ".join(issue.message for issue in coverage.issues)
        raise RuntimeError(
            "KR lifecycle coverage is incomplete. Add verified KRX/DART/KIND "
            "actions to KR_OFFICIAL_ACTIONS_PATH or reviewed exceptions to "
            f"KR_LIFECYCLE_RESOLUTIONS_PATH: {detail}"
        )
    return resolutions, coverage


def _save_kr_lifecycle_review_checkpoint(
    repository: LocalDatasetRepository,
    release: DataRelease,
    candidates: Iterable,
    resolutions: pd.DataFrame,
) -> None:
    """Persist the exact open/closed inventory before the fail-closed gate."""

    resolution_by_id = {
        str(row["candidate_id"]): row
        for row in resolutions.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = lifecycle_candidate_id(
            candidate.security_id,
            candidate.last_price_date,
            selection_rule=KR_LIFECYCLE_SELECTION_RULE,
        )
        resolution = resolution_by_id.get(candidate_id, {})
        rows.append(
            {
                "candidate_id": candidate_id,
                "security_id": candidate.security_id,
                "symbol": candidate.symbol,
                "name": candidate.name,
                "exchange": candidate.exchange,
                "last_price_date": candidate.last_price_date,
                "active_to": candidate.active_to,
                "index_remove_dates": list(candidate.index_remove_dates),
                "resolution": _cell_text(resolution.get("resolution")),
                "event_id": _cell_text(resolution.get("event_id")),
                "exception_code": _cell_text(
                    resolution.get("exception_code")
                ),
                "recheck_after": _cell_text(
                    resolution.get("recheck_after")
                ),
                "source_url": _cell_text(resolution.get("source_url")),
                "source_hash": _cell_text(resolution.get("source_hash")),
            }
        )
    payload = {
        "schema_version": 1,
        "market": "KR",
        "completed_session": release.completed_session,
        "selection_rule": KR_LIFECYCLE_SELECTION_RULE,
        "candidate_count": len(rows),
        "resolved_count": sum(bool(row["resolution"]) for row in rows),
        "open_count": sum(not bool(row["resolution"]) for row in rows),
        "candidates": rows,
    }
    write_atomic(
        repository.root / "state" / "lifecycle" / "candidates.json",
        (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _archive_allowed_artifacts(
    repository: LocalDatasetRepository,
    artifacts: Iterable,
    *,
    effective_date: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for artifact in artifacts:
        source = str(artifact.source)
        if source.startswith("eodhd"):
            license_class = KR_LICENSE_POLICY["eodhd"]
        elif source == "kr_provider_benchmark_report":
            license_class = KR_LICENSE_POLICY["benchmark_report"]
        elif source == "kr_lifecycle_evidence_report":
            license_class = KR_LICENSE_POLICY[
                "lifecycle_evidence_report"
            ]
        elif source == "krx_reference_price_audit":
            license_class = KR_LICENSE_POLICY["reference_price_audit"]
        elif source == "opendart_action_audit":
            license_class = KR_LICENSE_POLICY["opendart_action_audit"]
        elif source == "kr_official_actions_path":
            license_class = KR_LICENSE_POLICY["official_action_evidence"]
        else:
            continue
        if license_class != "allowed_private":
            continue
        content_type = str(artifact.content_type).lower()
        extension = "json" if "json" in content_type else "bin"
        object_path = f"archives/KR/{effective_date}/{artifact.source_hash}.{extension}.gz"
        destination = repository.root / object_path
        if not destination.is_file():
            write_atomic(destination, gzip.compress(artifact.content))
        rows.append(
            {
                "archive_id": artifact.source_hash,
                "dataset": source,
                "object_path": object_path,
                "content_type": artifact.content_type,
                "effective_date": effective_date,
                "license_class": license_class,
                "source": source,
                "source_url": artifact.source_url,
                "retrieved_at": artifact.retrieved_at,
                "source_hash": artifact.source_hash,
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            *dataset_spec("source_archive").required_columns,
            "source_url",
            "license_class",
        ),
    ).drop_duplicates(
        "archive_id", keep="last"
    )


def _kr_lifecycle_evidence_artifact(
    *,
    completed_session: str,
    lifecycle_coverage: Any,
    lifecycle_resolutions: pd.DataFrame,
    corporate_actions: pd.DataFrame,
) -> SourceArtifact:
    """Create a publishable summary bound to all lifecycle evidence rows."""

    action_by_id = {
        str(row["event_id"]): row
        for row in corporate_actions.to_dict("records")
        if str(row.get("event_id") or "").strip()
    }
    records: list[dict[str, Any]] = []
    for resolution in lifecycle_resolutions.sort_values(
        "candidate_id",
        kind="stable",
    ).to_dict("records"):
        event_id = str(resolution.get("event_id") or "").strip()
        action = action_by_id.get(event_id, {})
        records.append(
            {
                "candidate_id": str(
                    resolution.get("candidate_id") or ""
                ),
                "security_id": str(
                    resolution.get("security_id") or ""
                ),
                "last_price_date": _normalized_date(
                    resolution.get("last_price_date")
                ),
                "resolution": str(
                    resolution.get("resolution") or ""
                ),
                "event_id": event_id,
                "exception_code": str(
                    resolution.get("exception_code") or ""
                ),
                "recheck_after": _normalized_date(
                    resolution.get("recheck_after")
                ),
                "source_url": str(
                    (
                        action.get("source_url")
                        if event_id
                        else resolution.get("source_url")
                    )
                    or ""
                ),
                "source_hash": str(
                    (
                        action.get("source_hash")
                        if event_id
                        else resolution.get("source_hash")
                    )
                    or ""
                ).lower(),
            }
        )
    payload = {
        "schema": "kr_lifecycle_evidence_report/v1",
        "market": "KR",
        "completed_session": completed_session,
        "coverage": lifecycle_coverage.manifest_metadata(),
        "records": records,
    }
    return SourceArtifact(
        source="kr_lifecycle_evidence_report",
        source_url=(
            "local://kr-lifecycle-evidence-report/"
            f"{completed_session}"
        ),
        retrieved_at=utc_now_iso(),
        content=canonical_json_bytes(payload),
        content_type="application/json",
    )


def _cross_validation_report(
    benchmark: dict[str, Any],
    benchmark_bytes: bytes,
    versions: dict[str, str],
    report_archive_id: str,
    completed: str,
    *,
    lifecycle_coverage: Any,
    lifecycle_resolutions: pd.DataFrame,
    index_price_gap_policy: dict[str, Any],
    lifecycle_evidence_report_sha256: str,
    corporate_action_audit: SourceArtifact,
    opendart_dividend_report: SourceArtifact,
    official_action_evidence: SourceArtifact | None,
) -> pd.DataFrame:
    provider = str(benchmark["selection"]["secondary"])
    metrics = dict(benchmark["providers"][provider])
    membership = dict(benchmark.get("membership_verification") or {})
    membership_profiles = membership.get("profiles") or {}
    membership_event_count = sum(
        int(value.get("event_count") or 0)
        for value in membership_profiles.values()
        if isinstance(value, dict)
    )
    membership_sha256 = sha256_bytes(canonical_json_bytes(membership))
    sessions = benchmark.get("sessions") or {}
    action_audit = json.loads(corporate_action_audit.content)
    historical_gate_passed = bool(
        benchmark.get("status") == "ready"
        and benchmark.get("benchmark_scope") == "historical_full"
        and membership.get("status") == "passed"
        and metrics.get("hard_gate_passed") is True
        and provider != "krx"
        and action_audit.get("status") == "passed"
    )
    report_id = sha256_bytes(
        canonical_json_bytes(
            {
                "benchmark_sha256": sha256_bytes(benchmark_bytes),
                "completed": completed,
                "versions": versions,
            }
        )
    )
    exception_count = int(lifecycle_coverage.exception_count)
    exception_rows = lifecycle_resolutions.loc[
        lifecycle_resolutions["resolution"].astype(str).eq("exception")
    ]
    permanent_exception_count = (
        int(
            (
                ~exception_rows["exception_code"].astype(str).isin(
                    {str(value) for value in TEMPORARY_EXCEPTION_CODES}
                )
            ).sum()
        )
        if not exception_rows.empty
        else 0
    )
    return pd.DataFrame(
        [
            {
                "report_id": report_id,
                "base_release_version": "kr-candidate",
                "validated_at": utc_now_iso(),
                "status": "passed" if historical_gate_passed else "blocked",
                "provider": provider,
                "policy_sha256": sha256_bytes(canonical_json_bytes(benchmark["thresholds"])),
                "lifecycle_evidence_report_sha256": (
                    lifecycle_evidence_report_sha256
                ),
                "validated_versions_json": json.dumps(versions, sort_keys=True),
                "event_count": int(lifecycle_coverage.candidate_count),
                "event_mismatch_count": len(lifecycle_coverage.issues),
                "nonterminal_event_count": exception_count,
                "reviewed_nonterminal_event_count": exception_count,
                "permanent_exception_count": permanent_exception_count,
                "permanent_exception_mismatch_count": 0,
                "price_target_count": int(metrics.get("expected_row_count", 0)),
                "price_pass_count": int(
                    metrics.get(
                        "verified_row_count",
                        round(
                            float(
                                metrics.get(
                                    "close_within_one_tick_rate",
                                    0.0,
                                )
                            )
                            * int(
                                metrics.get(
                                    "expected_row_count",
                                    0,
                                )
                            )
                        ),
                    )
                ),
                "price_ohlc_within_one_tick_rate": float(
                    metrics.get(
                        "ohlc_within_one_tick_rate",
                        0.0,
                    )
                ),
                "price_volume_exact_rate": float(
                    metrics.get("volume_exact_rate", 0.0)
                ),
                "price_provider_disagreement_count": int(
                    metrics.get(
                        "provider_disagreement_count",
                        0,
                    )
                ),
                "price_verification_assignment_sha256": str(
                    metrics.get(
                        "verification_assignment_sha256",
                        "",
                    )
                ),
                "independent_price_participants_json": json.dumps(
                    metrics.get(
                        "participant_providers",
                        [provider],
                    ),
                    sort_keys=True,
                ),
                "price_exception_count": int(
                    index_price_gap_policy["gap_count"]
                ),
                "price_unresolved_count": int(metrics.get("unclassified_missing", 0)),
                "price_mismatch_count": int(
                    metrics.get("unexplained_large_discontinuities", 0)
                ),
                "overlap_session_count": int(benchmark["sessions"]["count"]),
                "verification_schema": KR_HISTORY_VERIFICATION_SCHEMA,
                "verification_scope": str(benchmark.get("benchmark_scope") or ""),
                "verification_start": str(sessions.get("start") or ""),
                "verification_end": str(sessions.get("end") or ""),
                "membership_verification_sha256": membership_sha256,
                "membership_expected_snapshot_count": int(
                    membership.get("expected_snapshot_count") or 0
                ),
                "membership_observed_snapshot_count": int(
                    membership.get("observed_snapshot_count") or 0
                ),
                "membership_missing_snapshot_count": int(
                    membership.get("missing_snapshot_count") or 0
                ),
                "membership_replay_mismatch_count": int(
                    membership.get("daily_replay_mismatch_count") or 0
                ),
                "membership_blocking_issue_count": int(
                    membership.get("blocking_issue_count") or 0
                ),
                "membership_invalid_artifact_count": int(
                    membership.get("invalid_artifact_count") or 0
                ),
                "membership_event_count": membership_event_count,
                "snapshot_inventory_sha256": str(
                    membership.get("snapshot_inventory_sha256") or ""
                ),
                "membership_event_inventory_sha256": str(
                    membership.get("event_inventory_sha256") or ""
                ),
                "canonical_price_provider": "krx",
                "independent_price_provider": provider,
                "independent_price_hard_gate_passed": bool(
                    metrics.get("hard_gate_passed")
                ),
                "independent_price_source_artifacts_complete": bool(
                    metrics.get("source_artifacts_complete")
                ),
                "corporate_action_verification_sha256": (
                    corporate_action_audit.source_hash
                ),
                "reference_price_audit_sha256": str(
                    action_audit.get("reference_price_audit_sha256") or ""
                ),
                "opendart_dividend_collection_sha256": (
                    opendart_dividend_report.source_hash
                ),
                "official_action_evidence_sha256": (
                    official_action_evidence.source_hash
                    if official_action_evidence is not None
                    else ""
                ),
                "corporate_action_inventory_sha256": str(
                    action_audit.get("action_inventory_sha256") or ""
                ),
                "corporate_action_decision_inventory_sha256": str(
                    action_audit.get("decision_inventory_sha256") or ""
                ),
                "corporate_action_count": int(
                    action_audit.get("action_count") or 0
                ),
                "reference_price_discontinuity_count": int(
                    action_audit.get("reference_discontinuity_count") or 0
                ),
                "reference_price_generated_adjustment_count": int(
                    action_audit.get(
                        "reference_generated_adjustment_count"
                    )
                    or 0
                ),
                "reference_price_unresolved_count": int(
                    action_audit.get("reference_unresolved_count") or 0
                ),
                "opendart_dividend_decision_count": int(
                    action_audit.get("opendart_decision_count") or 0
                ),
                "matched_dividend_count": int(
                    action_audit.get("matched_dividend_count") or 0
                ),
                "missing_dividend_count": int(
                    action_audit.get("missing_dividend_count") or 0
                ),
                "unmatched_provider_dividend_count": int(
                    action_audit.get(
                        "unmatched_provider_dividend_count"
                    )
                    or 0
                ),
                "known_provider_action_gap_count": int(
                    action_audit.get("known_provider_gap_count") or 0
                ),
                "resolved_provider_action_gap_count": int(
                    action_audit.get("resolved_provider_gap_count") or 0
                ),
                "corporate_action_blocking_issue_count": int(
                    action_audit.get("blocking_issue_count") or 0
                ),
                "index_price_gap_policy_sha256": str(
                    index_price_gap_policy["policy_sha256"]
                ),
                "index_price_gap_policy_json": json.dumps(
                    index_price_gap_policy,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "report_archive_id": report_archive_id,
                "source": "kr_provider_benchmark_report",
                "retrieved_at": utc_now_iso(),
                "source_hash": sha256_bytes(benchmark_bytes),
            }
        ]
    )


def _dataset_metadata(
    dataset: str,
    benchmark: dict[str, Any],
    *,
    index_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if dataset.startswith("index_"):
        source_mode = "official_only"
    elif dataset == "daily_price_raw":
        source_mode = "official_primary_cross_validated"
    elif dataset == "corporate_actions":
        source_mode = "provider_plus_official_exception_evidence"
    else:
        source_mode = "derived_or_multi_source"
    metadata = {
        "operation": "kr_market_pipeline",
        "market": "KR",
        "calendar": "XKRX",
        "timezone": "Asia/Seoul",
        "currency": "KRW",
        "primary_provider": benchmark["selection"]["primary"],
        "secondary_provider": benchmark["selection"]["secondary"],
        "license_class": "allowed_private",
        "license_policy": KR_LICENSE_POLICY,
        "source_mode": source_mode,
        "official_coverage_start": "",
        "official_coverage_end": "",
    }
    if dataset.startswith("index_") and index_coverage:
        metadata.update(index_coverage)
    if dataset == "daily_price_raw":
        sessions = benchmark.get("sessions") or {}
        metadata["official_coverage_start"] = str(sessions.get("start") or "")
        metadata["official_coverage_end"] = str(sessions.get("end") or "")
        metadata["independent_validation_provider"] = str(
            (benchmark.get("selection") or {}).get("secondary") or ""
        )
    return metadata


def _index_coverage_metadata(
    memberships: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    starts = {
        profile: min(snapshots)
        for profile, snapshots in memberships.items()
        if snapshots
    }
    ends = {
        profile: max(snapshots)
        for profile, snapshots in memberships.items()
        if snapshots
    }
    if not starts or not ends:
        return {}
    return {
        "official_coverage_start": min(starts.values()),
        "official_coverage_end": min(ends.values()),
        "official_coverage_by_index": {
            profile: {"start": starts[profile], "end": ends[profile]}
            for profile in sorted(starts)
        },
    }


class _FrameRepositoryView:
    """Avoid re-reading every Parquet dataset for cross-dataset validation."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames

    def current_manifest(self, dataset: str):
        return object() if dataset in self.frames else None

    def read_frame(self, dataset: str) -> pd.DataFrame:
        return self.frames.get(dataset, pd.DataFrame())


def _load_ready_benchmark(
    root: Path,
    *,
    start_session: str | None = None,
    end_session: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    path = root / "benchmarks" / "current.json"
    if not path.is_file():
        raise RuntimeError("KR provider benchmark is missing. Run quant-data benchmark-kr first.")
    content = path.read_bytes()
    report = json.loads(content)
    if report.get("status") != "ready":
        raise RuntimeError("KR provider benchmark hard gates are not ready.")
    if report.get("benchmark_scope") != "historical_full":
        raise RuntimeError(
            "KR provider benchmark is not a full PIT-union historical "
            "comparison. Run benchmark-kr without --symbols-limit."
        )
    if int(report.get("schema_version") or 0) < 5:
        raise RuntimeError(
            "KR provider benchmark predates the full-history verification "
            "contract. Re-run quant-data benchmark-kr."
        )
    sessions = report.get("sessions") or {}
    report_start = str(sessions.get("start") or "")
    report_end = str(sessions.get("end") or "")
    report_count = int(sessions.get("count") or 0)
    if not report_start or not report_end or report_count <= 0:
        raise RuntimeError("KR provider benchmark has no verified session inventory.")
    expected_inventory = _sessions_between(report_start, report_end)
    if report_count != len(expected_inventory):
        raise RuntimeError(
            "KR provider benchmark session count does not match the XKRX "
            "calendar inventory."
        )
    if start_session:
        requested = _sessions_between(start_session, end_session or report_end)
        requested_start = requested[0] if requested else str(start_session)
        if report_start > requested_start:
            raise RuntimeError(
                "KR provider benchmark starts after the requested history: "
                f"{report_start} > {requested_start}."
            )
    if end_session and report_end < str(end_session):
        raise RuntimeError(
            "KR provider benchmark ends before the requested release: "
            f"{report_end} < {end_session}."
        )
    symbols = report.get("symbols") or {}
    if (
        int(symbols.get("evaluated_count") or 0)
        != int(symbols.get("full_union_count") or -1)
        or float(symbols.get("identity_mapping_rate") or 0.0) != 1.0
        or symbols.get("missing_identity")
    ):
        raise RuntimeError(
            "KR provider benchmark does not cover the complete historical "
            "constituent union with stable identities."
        )
    membership = report.get("membership_verification") or {}
    if (
        membership.get("schema") != KR_HISTORY_VERIFICATION_SCHEMA
        or membership.get("status") != "passed"
        or float(membership.get("survivorship_score") or 0.0) != 1.0
        or int(membership.get("missing_snapshot_count") or 0) != 0
        or int(membership.get("daily_replay_mismatch_count") or 0) != 0
        or int(membership.get("invalid_artifact_count") or 0) != 0
        or int(membership.get("blocking_issue_count") or 0) != 0
    ):
        raise RuntimeError(
            "KR provider benchmark did not pass full official-snapshot "
            "anchor/event replay."
        )
    selection = report.get("selection") or {}
    secondary = str(selection.get("secondary") or "")
    if (
        selection.get("primary") != "krx"
        or not secondary.startswith("composite:")
    ):
        raise RuntimeError(
            "KR provider benchmark has no approved KRX/multi-provider "
            "primary/secondary selection."
        )
    secondary_metrics = (report.get("providers") or {}).get(secondary) or {}
    participants = secondary_metrics.get("participant_providers") or ()
    if (
        secondary_metrics.get("hard_gate_passed") is not True
        or secondary_metrics.get("pit_survivorship_status") != "passed"
        or secondary_metrics.get("source_artifacts_complete") is not True
        or len(set(map(str, participants))) < 2
        or secondary
        != "composite:" + "+".join(sorted(set(map(str, participants))))
        or float(
            secondary_metrics.get(
                "ohlc_within_one_tick_rate",
                0.0,
            )
        )
        != 1.0
        or float(
            secondary_metrics.get("volume_exact_rate", 0.0)
        )
        != 1.0
        or int(secondary_metrics.get("unclassified_missing", -1))
        != 0
        or not _valid_sha256(
            secondary_metrics.get(
                "verification_assignment_sha256"
            )
        )
    ):
        raise RuntimeError(
            "KR provider benchmark secondary provider did not pass the "
            "full-history price and provenance gates."
        )
    return report, content


def _sessions_between(start: str, end: str) -> tuple[str, ...]:
    from .markets import sessions_between

    return sessions_between("KR", start, end)


def _provider_license_class(provider: str) -> str:
    return {
        "krx": KR_LICENSE_POLICY["krx_canonical"],
        "eodhd": KR_LICENSE_POLICY["eodhd"],
        "naver": KR_LICENSE_POLICY["naver_raw"],
        "yahoo": KR_LICENSE_POLICY["yahoo_raw"],
        "kis": KR_LICENSE_POLICY["kis_raw"],
    }[provider]


def _merge_by_key(
    previous: pd.DataFrame,
    latest: pd.DataFrame,
    key: tuple[str, ...],
) -> pd.DataFrame:
    return pd.concat([previous, latest], ignore_index=True).drop_duplicates(
        list(key), keep="last"
    )


def _empty_dataset(dataset: str) -> pd.DataFrame:
    return pd.DataFrame(columns=dataset_spec(dataset).required_columns)


__all__ = [
    "KR_LICENSE_POLICY",
    "KR_PROFILES",
    "KR_REQUIRED_DATASETS",
    "KrBenchmarkOutcome",
    "KrBenchmarkThresholds",
    "KrBootstrapResult",
    "benchmark_kr_providers",
    "bootstrap_kr_market_data",
    "compare_provider_to_krx",
    "krx_tick_size",
    "sync_kr_market_data",
    "validate_kr_repository",
    "verify_kr_membership_history",
]
