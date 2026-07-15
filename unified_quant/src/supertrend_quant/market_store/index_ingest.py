from __future__ import annotations

import hashlib
import gzip
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .manifest import sha256_bytes, utc_now_iso, write_atomic
from .models import DataQuality
from .repository import DatasetWriteResult, LocalDatasetRepository
from .schemas import DATASET_SPECS
from .validation import validate_repository_snapshot


@dataclass(frozen=True)
class IndexImportResult:
    dataset: str
    version: str
    row_count: int
    conflict: bool
    release_version: str = ""


class IndexDataImporter:
    """Import archived official/best-effort constituent files without inventing rules."""

    def __init__(self, repository: LocalDatasetRepository):
        self.repository = repository

    def import_anchor(
        self,
        index_id: str,
        anchor_date: str,
        members: pd.DataFrame,
        *,
        source: str,
        source_url: str = "",
        official: bool,
        raw_content: bytes | None = None,
    ) -> IndexImportResult:
        normalized = members.copy()
        if "security_id" not in normalized:
            if "symbol" not in normalized:
                raise ValueError("Anchor import requires security_id or symbol.")
            normalized["security_id"] = self._resolve_symbols(
                normalized["symbol"].astype(str),
                pd.Series([anchor_date] * len(normalized)),
            )
        metadata = _source_metadata(
            source,
            source_url or "memory://index-anchor",
            raw_content or normalized.to_csv(index=False).encode(),
            source_kind="official" if official else "community",
        )
        frame = pd.DataFrame(
            {
                "index_id": index_id,
                "anchor_date": anchor_date,
                "security_id": normalized["security_id"].astype(str),
                "official": bool(official),
                **metadata,
            }
        ).drop_duplicates(["index_id", "anchor_date", "security_id"])
        content = raw_content or normalized.to_csv(index=False).encode()
        result = self._write(
            "index_constituent_anchors",
            frame,
            anchor_date,
            "anchor_import",
            {
                "source_mode": "official_only" if official else "best_effort",
                "official_coverage_start": anchor_date if official else "",
                "official_coverage_end": anchor_date if official else "",
            },
        )
        release_version = self._finalize_import(
            result,
            content,
            source=source,
            source_url=metadata["source_url"],
            effective_date=anchor_date,
        )
        return IndexImportResult(
            "index_constituent_anchors",
            result.manifest.version,
            len(frame),
            result.conflict,
            release_version,
        )

    def import_events(
        self,
        index_id: str,
        events: pd.DataFrame,
        *,
        source: str,
        source_url: str = "",
        official: bool,
        raw_content: bytes | None = None,
    ) -> IndexImportResult:
        normalized = events.copy()
        required = {"effective_date", "operation"}
        if missing := required - set(normalized):
            raise ValueError(f"Event import missing columns: {', '.join(sorted(missing))}")
        if "security_id" not in normalized:
            if "symbol" not in normalized:
                raise ValueError("Event import requires security_id or symbol.")
            normalized["security_id"] = self._resolve_symbols(
                normalized["symbol"].astype(str),
                normalized["effective_date"].astype(str),
            )
        metadata = _source_metadata(
            source,
            source_url or "memory://index-events",
            raw_content or normalized.to_csv(index=False).encode(),
            source_kind="official" if official else "community",
        )
        records = []
        for row in normalized.to_dict("records"):
            operation = str(row["operation"]).upper()
            event_id = str(row.get("event_id") or "")
            if not event_id:
                key = f"{source}|{index_id}|{row['effective_date']}|{operation}|{row['security_id']}"
                event_id = hashlib.sha256(key.encode()).hexdigest()
            records.append(
                {
                    "event_id": event_id,
                    "index_id": index_id,
                    "announcement_date": (
                        pd.Timestamp(row["announcement_date"]).date().isoformat()
                        if str(row.get("announcement_date") or "").strip()
                        else ""
                    ),
                    "effective_date": pd.Timestamp(row["effective_date"]).date().isoformat(),
                    "operation": operation,
                    "security_id": str(row["security_id"]),
                    "official": bool(official),
                    "source": metadata["source"],
                    "source_url": metadata["source_url"],
                    "source_kind": metadata["source_kind"],
                    "retrieved_at": metadata["retrieved_at"],
                    "source_hash": metadata["source_hash"],
                }
            )
        frame = pd.DataFrame(records).drop_duplicates("event_id", keep="last")
        completed = max(frame["effective_date"]) if not frame.empty else utc_now_iso()[:10]
        content = raw_content or normalized.to_csv(index=False).encode()
        result = self._write(
            "index_membership_events",
            frame,
            completed,
            "event_import",
            {
                "source_mode": "official_only" if official else "best_effort",
                "official_coverage_start": min(frame["effective_date"]) if official and not frame.empty else "",
                "official_coverage_end": completed if official else "",
            },
        )
        release_version = self._finalize_import(
            result,
            content,
            source=source,
            source_url=metadata["source_url"],
            effective_date=completed,
        )
        return IndexImportResult(
            "index_membership_events",
            result.manifest.version,
            len(frame),
            result.conflict,
            release_version,
        )

    def import_overlays(
        self,
        index_id: str,
        overlays: pd.DataFrame,
        *,
        source: str = "user_overlay",
    ) -> IndexImportResult:
        required = {"effective_from", "operation"}
        if missing := required - set(overlays):
            raise ValueError(f"Overlay import missing columns: {', '.join(sorted(missing))}")
        normalized = overlays.copy()
        if "effective_to" not in normalized:
            normalized["effective_to"] = ""
        if "security_id" not in normalized:
            if "symbol" not in normalized:
                raise ValueError("Overlay import requires security_id or symbol.")
            normalized["security_id"] = self._resolve_symbols(
                normalized["symbol"].astype(str),
                normalized["effective_from"].astype(str),
            )
        content = normalized.to_csv(index=False).encode()
        metadata = _source_metadata(source, "memory://user-overlay", content, source_kind="user")
        records = []
        for row in normalized.to_dict("records"):
            overlay_id = str(row.get("overlay_id") or "")
            if not overlay_id:
                key = f"{index_id}|{row['effective_from']}|{row['operation']}|{row['security_id']}"
                overlay_id = hashlib.sha256(key.encode()).hexdigest()
            records.append(
                {
                    "overlay_id": overlay_id,
                    "index_id": index_id,
                    "effective_from": pd.Timestamp(row["effective_from"]).date().isoformat(),
                    "effective_to": (
                        pd.Timestamp(row["effective_to"]).date().isoformat()
                        if str(row.get("effective_to") or "").strip()
                        else ""
                    ),
                    "operation": str(row["operation"]).upper(),
                    "security_id": str(row["security_id"]),
                    "reason": str(row.get("reason") or ""),
                    "source": metadata["source"],
                    "source_url": metadata["source_url"],
                    "source_kind": metadata["source_kind"],
                    "retrieved_at": metadata["retrieved_at"],
                    "source_hash": metadata["source_hash"],
                }
            )
        frame = pd.DataFrame(records).drop_duplicates("overlay_id", keep="last")
        completed = max(frame["effective_from"]) if not frame.empty else utc_now_iso()[:10]
        result = self._write(
            "custom_universe_overlays",
            frame,
            completed,
            "overlay_import",
            {"source_mode": "user_overlay"},
        )
        release_version = self._finalize_import(
            result,
            content,
            source=source,
            source_url=metadata["source_url"],
            effective_date=completed,
        )
        return IndexImportResult(
            "custom_universe_overlays",
            result.manifest.version,
            len(frame),
            result.conflict,
            release_version,
        )

    def _resolve_symbols(self, symbols: pd.Series, dates: pd.Series) -> list[str]:
        history = self.repository.read_frame("symbol_history")
        output = []
        for symbol, value in zip(symbols, dates):
            when = pd.Timestamp(value)
            candidates = history.loc[history["symbol"].astype(str) == str(symbol)].copy()
            starts = pd.to_datetime(candidates["effective_from"], errors="coerce")
            ends = pd.to_datetime(candidates["effective_to"], errors="coerce")
            active = candidates.loc[(starts <= when) & (ends.isna() | (ends >= when))]
            if len(active) != 1:
                raise ValueError(
                    f"Expected one security_id for {symbol} on {when.date()}, found {len(active)}."
                )
            output.append(str(active.iloc[0]["security_id"]))
        return output

    def _write(
        self,
        dataset: str,
        frame: pd.DataFrame,
        completed_session: str,
        operation: str,
        audit_metadata: dict[str, str] | None = None,
    ) -> DatasetWriteResult:
        metadata = {"operation": operation, **(audit_metadata or {})}
        current = self.repository.current_manifest(dataset)
        if current is None:
            return self.repository.write_frame(
                dataset,
                frame,
                completed_session=completed_session,
                metadata=metadata,
            )
        if current.official_coverage_start and metadata.get("official_coverage_start"):
            metadata["official_coverage_start"] = min(
                current.official_coverage_start,
                str(metadata["official_coverage_start"]),
            )
        if current.official_coverage_end and metadata.get("official_coverage_end"):
            metadata["official_coverage_end"] = max(
                current.official_coverage_end,
                str(metadata["official_coverage_end"]),
            )
        if current.source_mode and current.source_mode != metadata.get("source_mode"):
            metadata["source_mode"] = "best_effort"
        return self.repository.append_frame(
            dataset,
            frame,
            completed_session=completed_session,
            metadata=metadata,
        )

    def _finalize_import(
        self,
        result: DatasetWriteResult,
        content: bytes,
        *,
        source: str,
        source_url: str,
        effective_date: str,
    ) -> str:
        if result.conflict:
            raise RuntimeError(f"Index import was quarantined: {result.conflict_path}")
        archive_id = sha256_bytes(content)
        object_path = f"archives/index/{archive_id}.source.gz"
        destination = self.repository.root / object_path
        if not destination.is_file():
            write_atomic(destination, gzip.compress(content))
        archive = pd.DataFrame(
            [
                {
                    "archive_id": archive_id,
                    "dataset": result.manifest.dataset,
                    "object_path": object_path,
                    "content_type": "application/octet-stream",
                    "effective_date": effective_date,
                    "source": source,
                    "source_url": source_url,
                    "retrieved_at": utc_now_iso(),
                    "source_hash": archive_id,
                }
            ]
        )
        if self.repository.current_manifest("source_archive") is None:
            archive_result = self.repository.write_frame(
                "source_archive",
                archive,
                completed_session=effective_date,
                metadata={"operation": "initial_index_source_archive"},
            )
        else:
            archive_result = self.repository.append_frame(
                "source_archive",
                archive,
                completed_session=effective_date,
                metadata={"operation": "index_source_archive"},
            )
        if archive_result.conflict:
            raise RuntimeError(f"Index source archive was quarantined: {archive_result.conflict_path}")
        report = validate_repository_snapshot(self.repository)
        report.raise_for_errors()
        current_release, _ = self.repository.current_release()
        completed_session = max(
            filter(
                None,
                (
                    effective_date,
                    current_release.completed_session if current_release is not None else "",
                    self.repository.current_manifest("daily_price_raw").completed_session
                    if self.repository.current_manifest("daily_price_raw") is not None
                    else "",
                ),
            )
        )
        versions = {
            dataset: manifest.version
            for dataset in DATASET_SPECS
            if (manifest := self.repository.current_manifest(dataset)) is not None
        }
        manifests = [
            self.repository.current_manifest(dataset)
            for dataset in versions
        ]
        warnings = tuple(
            warning
            for manifest in manifests
            if manifest is not None
            for warning in manifest.warnings
        )
        quality = (
            DataQuality.DEGRADED
            if warnings or any(manifest.quality != DataQuality.VALID for manifest in manifests if manifest)
            else DataQuality.VALID
        )
        release = self.repository.commit_release(
            completed_session,
            versions,
            quality=quality,
            warnings=warnings,
        )
        return release.version


def read_tabular(path: str | Path) -> tuple[pd.DataFrame, bytes]:
    source = Path(path)
    content = source.read_bytes()
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(source), content
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(source), content
    raise ValueError("Index imports support CSV, TXT, or Parquet files.")


def _source_metadata(
    source: str,
    source_url: str,
    content: bytes,
    *,
    source_kind: str,
) -> dict[str, str]:
    return {
        "source": source,
        "source_url": source_url,
        "source_kind": source_kind,
        "retrieved_at": utc_now_iso(),
        "source_hash": sha256_bytes(content),
    }
