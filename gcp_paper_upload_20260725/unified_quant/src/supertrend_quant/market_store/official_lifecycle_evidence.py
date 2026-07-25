from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

import pandas as pd
import yaml

from .ingest import SourceArtifact
from .lifecycle import LifecycleCandidate
from .manifest import sha256_bytes, utc_now_iso, write_atomic


OFFICIAL_EXCEPTION_EVIDENCE_SCHEMA = "official_lifecycle_exception_evidence/v2"
LEGACY_OFFICIAL_EXCEPTION_EVIDENCE_SCHEMA = (
    "official_lifecycle_exception_evidence/v1"
)
OFFICIAL_EXCEPTION_EVIDENCE_URL_ALLOWLIST = {
    "aaba_2019_liquidation_distributions": (
        "https://www.sec.gov/Archives/edgar/data/1011006/"
        "000119312519262790/0001193125-19-262790.txt"
    ),
    "abmd_2022_cvr_consideration": (
        "https://www.sec.gov/Archives/edgar/data/815094/"
        "000119312522311074/0001193125-22-311074.txt"
    ),
    "brcm_2016_election_proration": (
        "https://www.sec.gov/Archives/edgar/data/1649345/"
        "000119312516446881/0001193125-16-446881.txt"
    ),
    "celg_2019_cvr_consideration": (
        "https://www.sec.gov/Archives/edgar/data/14272/"
        "000114036119021048/0001140361-19-021048.txt"
    ),
    "dvmt_2018_class_v_election_proration": (
        "https://www.sec.gov/Archives/edgar/data/1571996/"
        "000119312518360943/0001193125-18-360943.txt"
    ),
    "ggp_2018_election_proration": (
        "https://www.sec.gov/Archives/edgar/data/1496048/"
        "000119312518260735/0001193125-18-260735.txt"
    ),
    "legacy_dnr_2020_warrant_consideration": (
        "https://www.sec.gov/Archives/edgar/data/945764/"
        "000094576420000137/den-20200918x8kemergen.htm"
    ),
    "legacy_do_2021_warrant_consideration": (
        "https://www.sec.gov/Archives/edgar/data/949039/"
        "000119312521196262/d180977ds1.htm"
    ),
    "legacy_ne_2021_warrant_consideration": (
        "https://www.sec.gov/Archives/edgar/data/1458891/"
        "000119312521032043/d109435d8k.htm"
    ),
    "legacy_val_2021_warrant_consideration": (
        "https://www.sec.gov/Archives/edgar/data/314808/"
        "000110465921058903/tm2114630d1_8k.htm"
    ),
    "mallinckrodt_2022_cancellation": (
        "https://www.sec.gov/Archives/edgar/data/1567892/"
        "000119312522178880/d311224d8k.htm"
    ),
    "mallinckrodt_2023_cancellation": (
        "https://www.sec.gov/Archives/edgar/data/1567892/"
        "000156789224000008/mnk-20231229.htm"
    ),
    "para_2025_election_proration": (
        "https://www.sec.gov/Archives/edgar/data/813828/"
        "000119312525175027/0001193125-25-175027.txt"
    ),
    "tfcf_2019_disney_proration": (
        "https://www.sec.gov/Archives/edgar/data/1308161/"
        "000119312519079716/d710665dex991.htm"
    ),
    "tfcfa_2019_disney_proration": (
        "https://www.sec.gov/Archives/edgar/data/1308161/"
        "000119312519079716/d710665dex991.htm"
    ),
    "twc_2016_election_proration": (
        "https://www.sec.gov/Archives/edgar/data/1091667/"
        "000119312516596195/0001193125-16-596195.txt"
    ),
    "utx_2020_carr_otis_distributions": (
        "https://www.sec.gov/Archives/edgar/data/101829/"
        "000114036120008397/0001140361-20-008397.txt"
    ),
}
OFFICIAL_EXCEPTION_CODES = frozenset(
    {"recovery_uncertain", "unsupported_consideration"}
)
OFFICIAL_EVIDENCE_RESOLUTION_KINDS = frozenset({"exception", "applied_event"})
MAX_OFFICIAL_EXCEPTION_HTTP_ATTEMPTS = len(
    OFFICIAL_EXCEPTION_EVIDENCE_URL_ALLOWLIST
)
MAX_OFFICIAL_EXCEPTION_BYTES = 25 * 1024 * 1024
_CACHE_METADATA_FIELDS = frozenset(
    {
        "schema",
        "source_url",
        "retrieved_at",
        "content_type",
        "source_sha256",
        "content_bytes",
    }
)
_LEGACY_CACHE_METADATA_FIELDS = _CACHE_METADATA_FIELDS | {"evidence_id"}
_ALLOWED_CONFIG_FIELDS = frozenset(
    {
        "candidate_symbols",
        "candidate_name_contains",
        "candidate_security_ids",
        "candidate_last_price_dates",
        "binding_status",
        "effective_date",
        "filing_date",
        "resolution_kind",
        "exception_code",
        "action_type",
        "cash_amount",
        "claim",
        "source_url",
        "source_sha256",
        "required_text_groups",
    }
)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _tuple_of_text(value: Any, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a YAML list.")
    return tuple(_text(item) for item in value if _text(item))


def _validated_date(value: Any, *, field: str) -> str:
    text = _text(value)
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date, found {value!r}.") from exc


def _validated_sha256(value: Any, *, field: str, allow_blank: bool) -> str:
    text = _text(value).lower()
    if not text and allow_blank:
        return ""
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 or blank after stage one.")
    return text


def _validate_allowlisted_url(evidence_id: str, source_url: str) -> None:
    wanted = OFFICIAL_EXCEPTION_EVIDENCE_URL_ALLOWLIST.get(evidence_id)
    if wanted is None or source_url != wanted:
        raise ValueError(
            "Official lifecycle exception URL is not the exact reviewed allow-list entry: "
            f"{evidence_id}/{source_url}"
        )
    parsed = urlparse(source_url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.query
        or parsed.fragment
        or (parsed.hostname or "").lower() not in {"www.sec.gov", "www.fdic.gov"}
    ):
        raise ValueError(f"Official lifecycle exception URL is malformed: {source_url}")
    host = (parsed.hostname or "").lower()
    if host == "www.sec.gov" and not parsed.path.startswith("/Archives/edgar/data/"):
        raise ValueError(f"SEC evidence must be an EDGAR archive document: {source_url}")
    if host == "www.fdic.gov" and parsed.path != (
        "/resources/resolutions/bank-failures/failed-bank-list/first-republic.html"
    ):
        raise ValueError(f"FDIC evidence path is not reviewed: {source_url}")


@dataclass(frozen=True)
class OfficialLifecycleExceptionEvidenceSpec:
    evidence_id: str
    candidate_symbols: tuple[str, ...]
    candidate_name_contains: tuple[str, ...]
    candidate_security_ids: tuple[str, ...]
    candidate_last_price_dates: tuple[str, ...]
    binding_status: str
    effective_date: str
    filing_date: str
    resolution_kind: str
    exception_code: str
    action_type: str
    cash_amount: float | None
    claim: str
    source_url: str
    source_sha256: str
    required_text_groups: tuple[tuple[str, ...], ...]

    @property
    def pinned(self) -> bool:
        return bool(self.source_sha256)

    @property
    def binding_complete(self) -> bool:
        return bool(self.candidate_security_ids and self.candidate_last_price_dates)

    def targets_symbol(self, symbol: str) -> bool:
        return str(symbol).strip().upper() in self.candidate_symbols

    def matches_candidate(self, candidate: Any) -> bool:
        if not self.binding_complete:
            return False
        security_id = _text(
            candidate.get("security_id")
            if isinstance(candidate, Mapping)
            else getattr(candidate, "security_id", "")
        )
        symbol = _text(
            candidate.get("symbol")
            if isinstance(candidate, Mapping)
            else getattr(candidate, "symbol", "")
        ).upper()
        name = _text(
            candidate.get("name")
            if isinstance(candidate, Mapping)
            else getattr(candidate, "name", "")
        ).casefold()
        last_price_date = _text(
            candidate.get("last_price_date")
            if isinstance(candidate, Mapping)
            else getattr(candidate, "last_price_date", "")
        )
        return bool(
            security_id in self.candidate_security_ids
            and symbol in self.candidate_symbols
            and last_price_date in self.candidate_last_price_dates
            and any(token.casefold() in name for token in self.candidate_name_contains)
        )


def _parse_spec(
    evidence_id: str,
    raw: Mapping[str, Any],
) -> OfficialLifecycleExceptionEvidenceSpec:
    unknown = sorted(set(raw) - _ALLOWED_CONFIG_FIELDS)
    if unknown:
        raise ValueError(
            f"Unknown official exception evidence fields for {evidence_id}: {unknown}"
        )
    source_url = _text(raw.get("source_url"))
    _validate_allowlisted_url(evidence_id, source_url)
    symbols = tuple(
        dict.fromkeys(
            item.upper()
            for item in _tuple_of_text(
                raw.get("candidate_symbols"), field="candidate_symbols"
            )
        )
    )
    names = _tuple_of_text(
        raw.get("candidate_name_contains"), field="candidate_name_contains"
    )
    security_ids = tuple(
        dict.fromkeys(
            _tuple_of_text(
                raw.get("candidate_security_ids"), field="candidate_security_ids"
            )
        )
    )
    terminal_dates = tuple(
        dict.fromkeys(
            _validated_date(item, field="candidate_last_price_dates")
            for item in _tuple_of_text(
                raw.get("candidate_last_price_dates"),
                field="candidate_last_price_dates",
            )
        )
    )
    binding_status = _text(raw.get("binding_status"))
    binding_complete = bool(security_ids and terminal_dates)
    if binding_status not in {"bound", "pending_identity_repair"}:
        raise ValueError(
            f"Official evidence {evidence_id} has invalid binding_status: {binding_status!r}"
        )
    if binding_complete != (binding_status == "bound"):
        raise ValueError(
            f"Official evidence {evidence_id} binding_status does not match its exact "
            "security/date binding."
        )
    resolution_kind = _text(raw.get("resolution_kind"))
    if resolution_kind not in OFFICIAL_EVIDENCE_RESOLUTION_KINDS:
        raise ValueError(
            f"Official evidence {evidence_id} has invalid resolution_kind: "
            f"{resolution_kind!r}"
        )
    exception_code = _text(raw.get("exception_code"))
    action_type = _text(raw.get("action_type")).lower()
    cash_raw = raw.get("cash_amount")
    cash_amount = float(cash_raw) if cash_raw is not None and _text(cash_raw) else None
    if resolution_kind == "exception" and exception_code not in OFFICIAL_EXCEPTION_CODES:
        raise ValueError(
            f"Official evidence {evidence_id} has unsupported exception_code: "
            f"{exception_code!r}"
        )
    if resolution_kind == "exception" and (action_type or cash_amount is not None):
        raise ValueError(
            f"Official exception evidence {evidence_id} cannot also define an applied event."
        )
    if resolution_kind == "applied_event" and (
        exception_code or action_type != "delisting" or cash_amount != 0.0
    ):
        raise ValueError(
            f"Applied official evidence {evidence_id} must be an exact zero-recovery "
            "delisting and cannot define an exception code."
        )
    claim = _text(raw.get("claim"))
    if not symbols or not names or not claim:
        raise ValueError(
            f"Official evidence {evidence_id} requires symbols, name tokens, and a claim."
        )
    groups_raw = raw.get("required_text_groups")
    if not isinstance(groups_raw, (list, tuple)) or not groups_raw:
        raise ValueError(
            f"Official evidence {evidence_id} requires non-empty required_text_groups."
        )
    groups = tuple(
        _tuple_of_text(group, field=f"{evidence_id}.required_text_groups")
        for group in groups_raw
    )
    if any(not group for group in groups):
        raise ValueError(f"Official evidence {evidence_id} has an empty phrase group.")
    filing_date = _text(raw.get("filing_date"))
    if filing_date:
        filing_date = _validated_date(filing_date, field="filing_date")
    return OfficialLifecycleExceptionEvidenceSpec(
        evidence_id=evidence_id,
        candidate_symbols=symbols,
        candidate_name_contains=names,
        candidate_security_ids=security_ids,
        candidate_last_price_dates=terminal_dates,
        binding_status=binding_status,
        effective_date=_validated_date(raw.get("effective_date"), field="effective_date"),
        filing_date=filing_date,
        resolution_kind=resolution_kind,
        exception_code=exception_code,
        action_type=action_type,
        cash_amount=cash_amount,
        claim=claim,
        source_url=source_url,
        source_sha256=_validated_sha256(
            raw.get("source_sha256"), field=f"{evidence_id}.source_sha256", allow_blank=True
        ),
        required_text_groups=groups,
    )


def load_official_lifecycle_exception_evidence(
    path: str | Path,
) -> dict[str, OfficialLifecycleExceptionEvidenceSpec]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Lifecycle hints are missing: {source}")
    raw_document = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    raw_specs = raw_document.get("official_exception_evidence")
    if not isinstance(raw_specs, dict):
        raise ValueError("Lifecycle hints require official_exception_evidence mapping.")
    expected = set(OFFICIAL_EXCEPTION_EVIDENCE_URL_ALLOWLIST)
    actual = {str(key) for key in raw_specs}
    if actual != expected:
        raise ValueError(
            "Official lifecycle exception evidence inventory changed without a reviewed "
            f"code allow-list update: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return {
        evidence_id: _parse_spec(evidence_id, raw_specs[evidence_id] or {})
        for evidence_id in sorted(expected)
    }


def include_bound_official_applied_event_candidates(
    candidates: Iterable[LifecycleCandidate],
    repository: Any,
    release: Any,
    specs: Mapping[str, OfficialLifecycleExceptionEvidenceSpec],
) -> tuple[LifecycleCandidate, ...]:
    """Add reviewed terminal securities that no longer belong to an index graph.

    A reorganized security can itself be cancelled after the predecessor has
    already left an index.  The normal candidate builder intentionally starts
    from index references, so that later cancellation would otherwise be
    invisible.  Only exact, reviewed ``applied_event`` bindings may expand the
    candidate set; exception records never do so.
    """

    output = {candidate.security_id: candidate for candidate in candidates}
    applied = tuple(
        spec for spec in specs.values() if spec.resolution_kind == "applied_event"
    )
    if not applied:
        return tuple(
            sorted(
                output.values(),
                key=lambda item: (item.last_price_date, item.symbol, item.security_id),
            )
        )
    incomplete = [spec.evidence_id for spec in applied if not spec.binding_complete]
    if incomplete:
        raise RuntimeError(
            "Official applied-event candidate binding is incomplete: "
            + ", ".join(sorted(incomplete))
        )

    versions = release.dataset_versions
    master = repository.read_frame("security_master", versions.get("security_master"))
    history = repository.read_frame("symbol_history", versions.get("symbol_history"))
    prices = repository.read_frame("daily_price_raw", versions.get("daily_price_raw"))
    last_prices = (
        prices.groupby(prices["security_id"].astype(str))["session"].max().to_dict()
    )

    for spec in applied:
        for security_id in spec.candidate_security_ids:
            existing = output.get(security_id)
            if existing is not None:
                if not spec.matches_candidate(existing):
                    raise RuntimeError(
                        "Official applied-event binding disagrees with an indexed "
                        f"candidate: {spec.evidence_id}/{security_id}"
                    )
                continue
            master_rows = master.loc[master["security_id"].astype(str).eq(security_id)]
            if len(master_rows) != 1:
                raise RuntimeError(
                    "Official applied-event security is not uniquely present in the "
                    f"master: {spec.evidence_id}/{security_id}/rows={len(master_rows)}"
                )
            last_value = last_prices.get(security_id)
            if last_value is None or pd.isna(last_value):
                raise RuntimeError(
                    "Official applied-event security has no terminal price history: "
                    f"{spec.evidence_id}/{security_id}"
                )
            last_date = pd.Timestamp(last_value).date().isoformat()
            if last_date not in spec.candidate_last_price_dates:
                raise RuntimeError(
                    "Official applied-event terminal date differs from its reviewed "
                    f"binding: {spec.evidence_id}/{security_id}/actual={last_date}/"
                    f"expected={spec.candidate_last_price_dates}"
                )
            intervals = history.loc[
                history["security_id"].astype(str).eq(security_id)
                & history["symbol"].astype(str).str.upper().isin(spec.candidate_symbols)
            ].copy()
            starts = intervals.get("effective_from", pd.Series(index=intervals.index, dtype=str))
            ends = intervals.get("effective_to", pd.Series(index=intervals.index, dtype=str))
            intervals = intervals.loc[
                starts.astype(str).le(last_date)
                & (ends.fillna("").astype(str).eq("") | ends.astype(str).ge(last_date))
            ]
            symbols = tuple(sorted(set(intervals["symbol"].astype(str).str.upper())))
            if len(symbols) != 1:
                raise RuntimeError(
                    "Official applied-event terminal symbol interval is not unique: "
                    f"{spec.evidence_id}/{security_id}/{symbols}"
                )
            row = master_rows.iloc[0]
            active_to_value = row.get("active_to", "")
            active_to = "" if pd.isna(active_to_value) else _text(active_to_value)
            candidate = LifecycleCandidate(
                security_id=security_id,
                symbol=symbols[0],
                name=_text(row.get("name")),
                exchange=_text(row.get("exchange")),
                last_price_date=last_date,
                active_to=active_to,
            )
            if not spec.matches_candidate(candidate):
                raise RuntimeError(
                    "Official applied-event master identity differs from its reviewed "
                    f"binding: {spec.evidence_id}/{security_id}"
                )
            output[security_id] = candidate

    return tuple(
        sorted(
            output.values(),
            key=lambda item: (item.last_price_date, item.symbol, item.security_id),
        )
    )


def _normalized_document_text(content: bytes) -> str:
    decoded = content.decode("utf-8", errors="ignore")
    decoded = html.unescape(re.sub(r"<[^>]+>", " ", decoded))
    return " ".join(decoded.casefold().split())


def validate_official_evidence_content(
    spec: OfficialLifecycleExceptionEvidenceSpec,
    content: bytes,
) -> tuple[str, ...]:
    if not content:
        raise ValueError(f"Official evidence payload is empty: {spec.evidence_id}")
    normalized = _normalized_document_text(content)
    matched: list[str] = []
    for alternatives in spec.required_text_groups:
        selected = next(
            (
                phrase
                for phrase in alternatives
                if " ".join(phrase.casefold().split()) in normalized
            ),
            "",
        )
        if not selected:
            raise ValueError(
                "Official evidence does not prove every reviewed fact for "
                f"{spec.evidence_id}; missing one of {alternatives!r}."
            )
        matched.append(selected)
    return tuple(matched)


class OfficialLifecycleExceptionEvidenceSource:
    """Exact-URL, one-attempt source with an immutable raw-byte cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        allow_http: bool,
        session: Any | None = None,
        user_agent: str = "",
    ):
        self.cache_dir = Path(cache_dir)
        self.allow_http = bool(allow_http)
        self.session = session
        self.user_agent = _text(user_agent)
        self.http_attempts = 0

    def _key(self, source_url: str) -> str:
        return sha256_bytes(source_url.encode("utf-8"))

    def _raw_path(self, source_url: str) -> Path:
        return self.cache_dir / f"official-exception-{self._key(source_url)}.bin"

    def _metadata_path(self, source_url: str) -> Path:
        return self.cache_dir / f"official-exception-{self._key(source_url)}.json"

    def _decode_cached(self, spec: OfficialLifecycleExceptionEvidenceSpec) -> SourceArtifact | None:
        raw_path = self._raw_path(spec.source_url)
        metadata_path = self._metadata_path(spec.source_url)
        if not raw_path.exists() and not metadata_path.exists():
            return None
        if not raw_path.is_file() or not metadata_path.is_file():
            raise ValueError(
                f"Official evidence cache is incomplete for {spec.evidence_id}."
            )
        content = raw_path.read_bytes()
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Official evidence cache metadata is unreadable: {metadata_path}"
            ) from exc
        schema = _text(metadata.get("schema"))
        if schema not in {
            OFFICIAL_EXCEPTION_EVIDENCE_SCHEMA,
            LEGACY_OFFICIAL_EXCEPTION_EVIDENCE_SCHEMA,
        }:
            raise ValueError("Official evidence cache schema mismatch.")
        expected_fields = (
            _LEGACY_CACHE_METADATA_FIELDS
            if schema == LEGACY_OFFICIAL_EXCEPTION_EVIDENCE_SCHEMA
            else _CACHE_METADATA_FIELDS
        )
        if set(metadata) != expected_fields:
            raise ValueError("Official evidence cache metadata fields mismatch.")
        if _text(metadata.get("source_url")) != spec.source_url:
            raise ValueError("Official evidence cache URL mismatch.")
        if schema == LEGACY_OFFICIAL_EXCEPTION_EVIDENCE_SCHEMA:
            # v1 bound URL-addressed bytes to one evidence id.  Preserve those
            # caches, including shared documents such as the TFCF/TFCFA filing,
            # only when the recorded id is itself reviewed for this exact URL.
            legacy_evidence_id = _text(metadata.get("evidence_id"))
            if (
                not legacy_evidence_id
                or OFFICIAL_EXCEPTION_EVIDENCE_URL_ALLOWLIST.get(legacy_evidence_id)
                != spec.source_url
            ):
                raise ValueError("Official evidence legacy cache id/URL mismatch.")
        observed = sha256_bytes(content)
        if _text(metadata.get("source_sha256")).lower() != observed:
            raise ValueError("Official evidence cache content hash mismatch.")
        if int(metadata.get("content_bytes", -1)) != len(content):
            raise ValueError("Official evidence cache byte count mismatch.")
        return SourceArtifact(
            source=(
                "fdic_failed_bank_receivership"
                if urlparse(spec.source_url).hostname == "www.fdic.gov"
                else "sec_edgar_lifecycle_exception"
            ),
            source_url=spec.source_url,
            retrieved_at=_text(metadata.get("retrieved_at")),
            content=content,
            content_type=_text(metadata.get("content_type")) or "application/octet-stream",
        )

    def get(self, spec: OfficialLifecycleExceptionEvidenceSpec) -> SourceArtifact | None:
        _validate_allowlisted_url(spec.evidence_id, spec.source_url)
        return self._decode_cached(spec)

    def _http_session(self) -> Any:
        if self.session is None:
            import requests

            self.session = requests.Session()
        if hasattr(self.session, "headers"):
            self.session.headers.update(
                {
                    "User-Agent": self.user_agent
                    or "SuperTrendQuant lifecycle-evidence contact-required",
                    "Accept-Encoding": "identity",
                }
            )
        return self.session

    def fetch(self, spec: OfficialLifecycleExceptionEvidenceSpec) -> SourceArtifact:
        _validate_allowlisted_url(spec.evidence_id, spec.source_url)
        cached = self.get(spec)
        if cached is not None:
            return cached
        if not self.allow_http:
            raise FileNotFoundError(
                "Official lifecycle exception evidence is not cached and HTTP was not "
                f"explicitly allowed: {spec.evidence_id}"
            )
        if (
            urlparse(spec.source_url).hostname == "www.sec.gov"
            and not self.user_agent
        ):
            raise RuntimeError(
                "SEC_USER_AGENT is required before fetching official SEC lifecycle "
                f"evidence: {spec.evidence_id}"
            )
        if self.http_attempts >= MAX_OFFICIAL_EXCEPTION_HTTP_ATTEMPTS:
            raise RuntimeError("Official lifecycle exception evidence HTTP cap reached.")
        self.http_attempts += 1
        response = self._http_session().get(
            spec.source_url,
            timeout=60,
            allow_redirects=False,
        )
        status = int(getattr(response, "status_code", 0))
        final_url = _text(getattr(response, "url", "")) or spec.source_url
        content = bytes(getattr(response, "content", b""))
        if status != 200 or final_url != spec.source_url:
            raise RuntimeError(
                "Official lifecycle exception evidence response rejected: "
                f"id={spec.evidence_id}, status={status}, final_url={final_url}"
            )
        if not content or len(content) > MAX_OFFICIAL_EXCEPTION_BYTES:
            raise RuntimeError(
                "Official lifecycle exception evidence response size rejected: "
                f"id={spec.evidence_id}, bytes={len(content)}"
            )
        validate_official_evidence_content(spec, content)
        headers = getattr(response, "headers", {}) or {}
        metadata = {
            "schema": OFFICIAL_EXCEPTION_EVIDENCE_SCHEMA,
            "source_url": spec.source_url,
            "retrieved_at": utc_now_iso(),
            "content_type": _text(headers.get("Content-Type"))
            or "application/octet-stream",
            "source_sha256": sha256_bytes(content),
            "content_bytes": len(content),
        }
        raw_path = self._raw_path(spec.source_url)
        metadata_path = self._metadata_path(spec.source_url)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if raw_path.exists() or metadata_path.exists():
            existing = self.get(spec)
            if existing is None or existing.content != content:
                raise RuntimeError(
                    f"Official evidence changed for immutable URL cache: {spec.source_url}"
                )
            return existing
        write_atomic(raw_path, content)
        write_atomic(
            metadata_path,
            (
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8"),
        )
        artifact = self.get(spec)
        if artifact is None:  # pragma: no cover - both cache files were just written
            raise RuntimeError("Official evidence cache write did not become visible.")
        return artifact

    def load(
        self,
        spec: OfficialLifecycleExceptionEvidenceSpec,
        *,
        require_pinned: bool,
    ) -> tuple[SourceArtifact, tuple[str, ...]]:
        artifact = self.fetch(spec) if self.allow_http else self.get(spec)
        if artifact is None:
            raise FileNotFoundError(
                f"Official lifecycle exception evidence cache is missing: {spec.evidence_id}"
            )
        matched_phrases = validate_official_evidence_content(spec, artifact.content)
        observed = artifact.source_hash
        if require_pinned and not spec.pinned:
            raise RuntimeError(
                "Official lifecycle exception evidence is observed but not reviewer-pinned: "
                f"{spec.evidence_id}; observed_sha256={observed}"
            )
        if spec.pinned and observed != spec.source_sha256:
            raise RuntimeError(
                "Official lifecycle exception evidence differs from the reviewed pin: "
                f"{spec.evidence_id}; expected={spec.source_sha256}, observed={observed}"
            )
        return artifact, matched_phrases


def matching_official_exception_specs(
    candidate: Any,
    specs: Mapping[str, OfficialLifecycleExceptionEvidenceSpec] | Iterable[
        OfficialLifecycleExceptionEvidenceSpec
    ],
) -> tuple[OfficialLifecycleExceptionEvidenceSpec, ...]:
    values = specs.values() if isinstance(specs, Mapping) else specs
    return tuple(spec for spec in values if spec.matches_candidate(candidate))
