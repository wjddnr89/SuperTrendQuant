from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlencode

import pandas as pd

from ..env import load_env
from .ingest import SourceArtifact
from .lifecycle_report_provenance import (
    DEFAULT_SEC_MAX_HTTP_ATTEMPTS,
    DEFAULT_SEC_MAX_HTTP_ATTEMPTS_PER_CANDIDATE,
    SEC_MAX_HTTP_ATTEMPTS_PER_REQUEST,
)


SEC_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"


@dataclass(frozen=True)
class LifecycleCandidate:
    security_id: str
    symbol: str
    name: str
    exchange: str
    last_price_date: str
    active_to: str
    index_remove_dates: tuple[str, ...] = ()


@dataclass(frozen=True)
class SecFiling:
    cik: str
    accession_number: str
    filing_date: str
    form: str
    items: tuple[str, ...]
    display_name: str
    score: float

    @property
    def source_url(self) -> str:
        cik = str(int(self.cik))
        accession_path = self.accession_number.replace("-", "")
        filename = f"{self.accession_number}.txt"
        return f"{SEC_ARCHIVES_URL}/{cik}/{accession_path}/{filename}"


@dataclass(frozen=True)
class ParsedLifecycleEvent:
    action_type: str
    effective_date: str
    cash_amount: float | None = None
    ratio: float | None = None
    new_symbol: str = ""
    confidence: str = ""
    reason: str = ""


@dataclass(frozen=True)
class LifecycleEvidence:
    candidate: LifecycleCandidate
    filing: SecFiling | None
    parsed: ParsedLifecycleEvent | None
    source_url: str = ""
    source_hash: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["candidate"]["index_remove_dates"] = list(
            value["candidate"]["index_remove_dates"]
        )
        return value


class SecCacheMissError(RuntimeError):
    """Raised when an offline SEC replay needs an uncached exact request."""


class SecHttpAttemptLimitError(RuntimeError):
    """Raised before an SEC request would exceed an audited hard cap."""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.values: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.values.append(data)


class SecEdgarLifecycleSource:
    """Find lifecycle evidence in official SEC filings with a local HTTP cache."""

    def __init__(
        self,
        *,
        session=None,
        user_agent: str | None = None,
        cache_dir: str | Path = "data/cache/state/sec_lifecycle",
        min_interval_seconds: float = 0.12,
        allow_http: bool = False,
        max_http_attempts: int = DEFAULT_SEC_MAX_HTTP_ATTEMPTS,
        max_http_attempts_per_candidate: int = (
            DEFAULT_SEC_MAX_HTTP_ATTEMPTS_PER_CANDIDATE
        ),
        max_http_attempts_per_request: int = SEC_MAX_HTTP_ATTEMPTS_PER_REQUEST,
        initial_http_attempts: int = 0,
        initial_http_attempts_by_candidate: dict[str, int] | None = None,
        archive_replay: Callable[
            [str, LifecycleCandidate | None], bytes | None
        ]
        | None = None,
    ):
        load_env()
        self.allow_http = bool(allow_http)
        if session is None and self.allow_http:
            import requests

            session = requests.Session()
        self.session = session
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT", "").strip()
        if self.allow_http and not self.user_agent:
            raise RuntimeError(
                "SEC_USER_AGENT must identify the requester, for example "
                "'SuperTrendQuant contact@example.com'."
            )
        if self.allow_http:
            if self.session is None:  # pragma: no cover - requests created above
                raise RuntimeError("An HTTP session is required when SEC HTTP is enabled.")
            self.session.headers.update(
                {
                    "User-Agent": self.user_agent,
                    "Accept-Encoding": "gzip, deflate",
                }
            )
        self.cache_dir = Path(cache_dir)
        self.min_interval_seconds = max(0.1, float(min_interval_seconds))
        self.max_http_attempts = _positive_attempt_cap(
            max_http_attempts,
            "max_http_attempts",
        )
        self.max_http_attempts_per_candidate = _positive_attempt_cap(
            max_http_attempts_per_candidate,
            "max_http_attempts_per_candidate",
        )
        self.max_http_attempts_per_request = _positive_attempt_cap(
            max_http_attempts_per_request,
            "max_http_attempts_per_request",
        )
        self.http_attempts = _nonnegative_attempt_count(
            initial_http_attempts,
            "initial_http_attempts",
        )
        self.http_attempts_by_candidate = {
            str(key): _nonnegative_attempt_count(
                value,
                f"initial_http_attempts_by_candidate[{key!r}]",
            )
            for key, value in sorted(
                (initial_http_attempts_by_candidate or {}).items()
            )
        }
        if self.http_attempts > self.max_http_attempts:
            raise ValueError("Initial SEC HTTP attempts exceed the global hard cap.")
        if any(
            value > self.max_http_attempts_per_candidate
            for value in self.http_attempts_by_candidate.values()
        ):
            raise ValueError("Initial SEC HTTP attempts exceed a candidate hard cap.")
        if sum(self.http_attempts_by_candidate.values()) > self.http_attempts:
            raise ValueError(
                "Initial per-candidate SEC attempts exceed the global attempt count."
            )
        self._active_candidate_key = ""
        self._active_candidate: LifecycleCandidate | None = None
        self.archive_replay = archive_replay
        self._last_request_at = 0.0

    @contextmanager
    def candidate_http_scope(self, candidate: LifecycleCandidate):
        if self._active_candidate_key:
            raise RuntimeError("Nested SEC candidate HTTP scopes are not allowed.")
        key = f"{candidate.security_id}|{candidate.last_price_date}"
        self._active_candidate_key = key
        self._active_candidate = candidate
        try:
            yield self
        finally:
            self._active_candidate_key = ""
            self._active_candidate = None

    def collect(
        self,
        candidate: LifecycleCandidate,
        *,
        known_symbols: Iterable[str] = (),
        related_symbols: Iterable[str] = (),
        related_names: Iterable[str] = (),
        preferred_symbols: Iterable[str] = (),
        expected_action: str = "",
        anchor_dates: Iterable[str] = (),
    ) -> tuple[LifecycleEvidence, tuple[SourceArtifact, ...]]:
        artifacts: list[SourceArtifact] = []
        filing_errors: list[str] = []
        try:
            filings, search_artifacts = self.search(
                candidate,
                related_symbols=related_symbols,
                related_names=related_names,
                anchor_dates=anchor_dates,
                expected_action=expected_action,
            )
            artifacts.extend(search_artifacts)
            parsed_values: list[
                tuple[ParsedLifecycleEvent, SecFiling, SourceArtifact]
            ] = []
            filing_limit = 16 if expected_action == "ticker_change" else 8
            for filing in filings[:filing_limit]:
                try:
                    content, artifact = self.fetch_filing(filing)
                except (SecCacheMissError, SecHttpAttemptLimitError):
                    raise
                except Exception as exc:
                    filing_errors.append(
                        f"{filing.accession_number}: {type(exc).__name__}: {exc}"
                    )
                    continue
                artifacts.append(artifact)
                parsed = parse_sec_lifecycle_filing(
                    content,
                    candidate=candidate,
                    filing=filing,
                    known_symbols=known_symbols,
                    preferred_symbols=preferred_symbols,
                    expected_action=expected_action,
                )
                if expected_action and parsed is not None and parsed.action_type != expected_action:
                    parsed = None
                if parsed is not None:
                    parsed_values.append((parsed, filing, artifact))
                    if parsed.confidence == "high" and filing.score >= 8.0:
                        break
            if parsed_values:
                parsed, filing, artifact = max(
                    parsed_values,
                    key=lambda item: _parsed_event_score(item[0], item[1]),
                )
                return (
                    LifecycleEvidence(
                        candidate=candidate,
                        filing=filing,
                        parsed=parsed,
                        source_url=filing.source_url,
                        source_hash=artifact.source_hash,
                    ),
                    tuple(artifacts),
                )
            return (
                LifecycleEvidence(
                    candidate=candidate,
                    filing=filings[0] if filings else None,
                    parsed=None,
                    error=(
                        "No supported lifecycle terms found in candidate SEC filings."
                        + (
                            " Fetch errors: " + " | ".join(filing_errors[:3])
                            if filing_errors
                            else ""
                        )
                    ),
                ),
                tuple(artifacts),
            )
        except (SecCacheMissError, SecHttpAttemptLimitError):
            raise
        except Exception as exc:
            return (
                LifecycleEvidence(
                    candidate=candidate,
                    filing=None,
                    parsed=None,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                tuple(artifacts),
            )

    def search(
        self,
        candidate: LifecycleCandidate,
        *,
        related_symbols: Iterable[str] = (),
        related_names: Iterable[str] = (),
        anchor_dates: Iterable[str] = (),
        expected_action: str = "",
    ) -> tuple[tuple[SecFiling, ...], tuple[SourceArtifact, ...]]:
        related_symbols = tuple(str(value).strip() for value in related_symbols if str(value).strip())
        related_names = tuple(str(value).strip() for value in related_names if str(value).strip())
        anchors = tuple(
            dict.fromkeys(
                str(value) for value in anchor_dates if str(value).strip()
            )
        ) or (candidate.last_price_date,)
        company_name = _searchable_company_name(candidate.name)
        company_core = _company_core_name(candidate.name)
        query_values = tuple(
            dict.fromkeys(
                value
                for value in (
                    company_name,
                    company_core,
                    *(str(value).strip() for value in related_names),
                    candidate.symbol,
                    *(str(value).strip() for value in related_symbols),
                )
                if value
            )
        )
        hits: dict[str, SecFiling] = {}
        artifacts: list[SourceArtifact] = []
        for anchor in anchors:
            anchor_date = pd.Timestamp(anchor)
            start = (anchor_date - pd.Timedelta(days=75)).date().isoformat()
            end = (anchor_date + pd.Timedelta(days=45)).date().isoformat()
            anchor_hits: set[str] = set()
            for query in query_values:
                for form in ("8-K", "6-K", "25-NSE"):
                    params = {
                        "q": f'"{query}"',
                        "forms": form,
                        "startdt": start,
                        "enddt": end,
                        "from": 0,
                        "size": 50,
                    }
                    content = self._get(SEC_SEARCH_URL, params=params)
                    artifacts.append(
                        SourceArtifact(
                            source="sec_edgar_search",
                            source_url=f"{SEC_SEARCH_URL}?{urlencode(params)}",
                            retrieved_at=_utc_now_iso(),
                            content=content,
                            content_type="application/json",
                        )
                    )
                    payload = json.loads(content)
                    for item in payload.get("hits", {}).get("hits", ()):
                        source = item.get("_source") or {}
                        accession = str(source.get("adsh") or "")
                        ciks = source.get("ciks") or ()
                        if not accession or not ciks:
                            continue
                        filing = _filing_from_search_hit(
                            source,
                            candidate=candidate,
                            related_symbols=related_symbols,
                            related_names=related_names,
                        )
                        if filing.score < 1.0:
                            continue
                        prior = hits.get(accession)
                        if prior is None or filing.score > prior.score:
                            hits[accession] = filing
                        anchor_hits.add(accession)
                if any(hits[item].score >= 8.0 for item in anchor_hits):
                    break
        ordered = tuple(
            sorted(
                hits.values(),
                key=lambda item: _filing_sort_key(
                    item,
                    anchors=anchors,
                    expected_action=expected_action,
                ),
            )
        )
        return ordered, tuple(artifacts)

    def fetch_filing(self, filing: SecFiling) -> tuple[bytes, SourceArtifact]:
        content = self._get(filing.source_url)
        return content, SourceArtifact(
            source="sec_edgar_filing",
            source_url=filing.source_url,
            retrieved_at=_utc_now_iso(),
            content=content,
            content_type="text/plain",
        )

    def fetch_url(self, url: str) -> tuple[bytes, SourceArtifact]:
        """Fetch an explicitly reviewed SEC filing URL through the same cache."""

        content = self._get(url)
        return content, SourceArtifact(
            source="sec_edgar_filing",
            source_url=url,
            retrieved_at=_utc_now_iso(),
            content=content,
            content_type="text/html" if url.lower().endswith((".htm", ".html")) else "text/plain",
        )

    def _get(self, url: str, *, params: dict[str, Any] | None = None) -> bytes:
        encoded = f"{url}?{urlencode(sorted((params or {}).items()))}"
        key = hashlib.sha256(encoded.encode()).hexdigest()
        cache_path = self.cache_dir / f"{key}.bin"
        if cache_path.is_file():
            return cache_path.read_bytes()
        request_url = encoded[:-1] if encoded.endswith("?") else encoded
        if self.archive_replay is not None:
            archived = self.archive_replay(request_url, self._active_candidate)
            if archived is not None:
                if not isinstance(archived, bytes):
                    raise TypeError("SEC archive replay must return bytes or None.")
                return archived
        if not self.allow_http:
            raise SecCacheMissError(
                "SEC cache miss in offline/cache-only mode for exact URL: "
                f"{request_url}. Re-run the collector with --fetch-missing-sec "
                "only when reviewed network acquisition is intended."
            )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        last_error = ""
        for attempt in range(self.max_http_attempts_per_request):
            self._claim_http_attempt(request_url)
            wait = self.min_interval_seconds - (time.monotonic() - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            try:
                response = self.session.get(url, params=params, timeout=60)
                self._last_request_at = time.monotonic()
                if response.status_code in {429, 500, 502, 503, 504}:
                    last_error = f"HTTP {response.status_code}"
                    if attempt + 1 < self.max_http_attempts_per_request:
                        time.sleep(min(8.0, 0.5 * (2**attempt)))
                    continue
                response.raise_for_status()
                cache_path.write_bytes(response.content)
                return response.content
            except Exception as exc:
                self._last_request_at = time.monotonic()
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < self.max_http_attempts_per_request:
                    time.sleep(min(8.0, 0.5 * (2**attempt)))
        raise RuntimeError(f"SEC request failed for {url}: {last_error}")

    def _claim_http_attempt(self, request_url: str) -> None:
        if self.http_attempts >= self.max_http_attempts:
            raise SecHttpAttemptLimitError(
                "SEC global HTTP attempt hard cap reached before request: "
                f"attempts={self.http_attempts}, cap={self.max_http_attempts}, "
                f"url={request_url}"
            )
        candidate_key = self._active_candidate_key
        candidate_attempts = self.http_attempts_by_candidate.get(candidate_key, 0)
        if (
            candidate_key
            and candidate_attempts >= self.max_http_attempts_per_candidate
        ):
            raise SecHttpAttemptLimitError(
                "SEC per-candidate HTTP attempt hard cap reached before request: "
                f"candidate={candidate_key}, attempts={candidate_attempts}, "
                f"cap={self.max_http_attempts_per_candidate}, url={request_url}"
            )
        self.http_attempts += 1
        if candidate_key:
            self.http_attempts_by_candidate[candidate_key] = candidate_attempts + 1


def _positive_attempt_cap(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer.") from exc
    if parsed < 1 or str(parsed) != str(value).strip():
        raise ValueError(f"{field} must be a positive canonical integer.")
    return parsed


def _nonnegative_attempt_count(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative integer.") from exc
    if parsed < 0 or str(parsed) != str(value).strip():
        raise ValueError(f"{field} must be a non-negative canonical integer.")
    return parsed


def _expand_referenced_security_ids_via_actions(
    referenced_ids: set[str],
    actions: pd.DataFrame,
) -> set[str]:
    """Follow every explicit corporate-action security transition transitively.

    Index membership identifies the initial holdings, but mergers, ticker
    changes, and spinoffs can create a position in a security that never appears
    in an index anchor or membership event.  ``new_security_id`` is the canonical
    identity edge for those transitions.  Starting only from the index graph
    keeps unrelated securities out, while the visited set makes malformed cycles
    harmless.

    Lifecycle resolutions and exception hints are deliberately not inputs here:
    an exception cannot make its own security a candidate.
    """

    output = {str(value).strip() for value in referenced_ids if str(value).strip()}
    if actions.empty:
        return output

    predecessors = actions["security_id"].fillna("").astype(str).str.strip()
    successors = actions["new_security_id"].fillna("").astype(str).str.strip()
    links: dict[str, set[str]] = {}
    for predecessor, successor in zip(predecessors, successors, strict=True):
        if predecessor and successor:
            links.setdefault(predecessor, set()).add(successor)

    frontier = list(output)
    while frontier:
        predecessor = frontier.pop()
        for successor in links.get(predecessor, ()):
            if successor in output:
                continue
            output.add(successor)
            frontier.append(successor)
    return output


def build_lifecycle_candidates(
    repository,
    *,
    release=None,
    stale_days: int = 30,
) -> tuple[LifecycleCandidate, ...]:
    if release is None:
        release, _ = repository.current_release()
    if release is None:
        raise RuntimeError("A data release is required to build lifecycle candidates.")
    versions = release.dataset_versions
    master = repository.read_frame("security_master", versions.get("security_master"))
    prices = repository.read_frame("daily_price_raw", versions.get("daily_price_raw"))
    anchors = repository.read_frame(
        "index_constituent_anchors",
        versions.get("index_constituent_anchors"),
    )
    events = repository.read_frame(
        "index_membership_events",
        versions.get("index_membership_events"),
    )
    actions = repository.read_frame(
        "corporate_actions",
        versions.get("corporate_actions"),
    )
    referenced_ids = set(anchors.get("security_id", ()).astype(str)) | set(
        events.get("security_id", ()).astype(str)
    )
    referenced_ids = _expand_referenced_security_ids_via_actions(
        referenced_ids,
        actions,
    )
    last_prices = (
        prices.groupby(prices["security_id"].astype(str))["session"].max().to_dict()
    )
    removes = events.loc[
        events["operation"].astype(str).str.upper().eq("REMOVE")
    ].copy()
    remove_dates = {
        security_id: tuple(
            sorted(pd.to_datetime(group["effective_date"]).dt.date.astype(str).unique())
        )
        for security_id, group in removes.groupby(removes["security_id"].astype(str))
    }
    cutoff = pd.Timestamp(release.completed_session) - pd.Timedelta(days=max(1, stale_days))
    output: list[LifecycleCandidate] = []
    for row in master.itertuples(index=False):
        security_id = str(row.security_id)
        if security_id not in referenced_ids:
            continue
        last_value = last_prices.get(security_id)
        if last_value is None or pd.isna(last_value):
            continue
        last_date = pd.Timestamp(last_value)
        if last_date >= cutoff:
            continue
        active_to = str(getattr(row, "active_to", "") or "")
        output.append(
            LifecycleCandidate(
                security_id=security_id,
                symbol=str(row.primary_symbol),
                name=str(row.name),
                exchange=str(row.exchange),
                last_price_date=last_date.date().isoformat(),
                active_to=active_to,
                index_remove_dates=remove_dates.get(security_id, ()),
            )
        )
    return tuple(sorted(output, key=lambda item: (item.last_price_date, item.symbol)))


def parse_sec_lifecycle_filing(
    content: bytes | str,
    *,
    candidate: LifecycleCandidate,
    filing: SecFiling,
    known_symbols: Iterable[str] = (),
    preferred_symbols: Iterable[str] = (),
    expected_action: str = "",
) -> ParsedLifecycleEvent | None:
    text = _submission_text(content)
    lowered = text.lower()
    known = {str(value).upper() for value in known_symbols if str(value).strip()}
    preferred = {
        str(value).upper() for value in preferred_symbols if str(value).strip()
    }
    old_symbol = candidate.symbol.upper()
    filing_identity_match = _filing_matches_candidate(
        filing, candidate
    ) or _filing_text_matches_candidate(text, candidate)

    merger_language = any(
        phrase in lowered
        for phrase in (
            "consummated the merger",
            "completed the merger",
            "consummated the acquisition",
            "completed the acquisition",
            "completed its acquisition",
            "acquisition was completed",
            "transaction was completed",
            "merger was completed",
            "completion of the merger",
            "completion of the acquisition",
            "completion of the transaction",
            "closing of the merger",
            "closing of the acquisition",
            "consummation of the merger",
            "consummation of the acquisition",
            "effective time of the merger",
            "effective time of the acquisition",
            "merged with and into",
            "completed the previously announced merger",
            "completed the previously announced acquisition",
            "completed its previously announced acquisition",
            "merger became effective",
            "merger was consummated",
            "combination completed",
            "at the effective time",
        )
    ) or bool(
        re.search(
            r"\bmerger\b.{0,180}?\b(?:became?s?|was)\s+effective\b",
            lowered,
            flags=re.I,
        )
    )
    effective, effective_is_explicit = _parse_completion_date_with_evidence(
        text,
        candidate,
        filing,
    )
    ticker = _parse_ticker_change(
        text,
        old_symbol=old_symbol,
        preferred_symbols=preferred,
    )
    cash_amount = _parse_cash_per_share(text) if merger_language else None
    ratio, new_symbol, stock_terms_are_explicit = (
        _parse_stock_exchange(
            text,
            old_symbol=old_symbol,
            target_company_name=candidate.name,
            known_symbols=known,
            preferred_symbols=preferred,
        )
        if merger_language
        else (None, "", False)
    )
    if expected_action == "ticker_change" and ticker is not None:
        ticker_effective, ticker_symbol = ticker
        ticker_effective_is_explicit = bool(ticker_effective)
        ticker_effective = ticker_effective or effective
        return ParsedLifecycleEvent(
            action_type="ticker_change",
            effective_date=ticker_effective,
            new_symbol=ticker_symbol,
            confidence=(
                "high"
                if ticker_effective_is_explicit or effective_is_explicit
                else "review"
            ),
            reason="SEC filing states the old and new trading symbols and effective date.",
        )
    if (
        expected_action not in {"cash_merger", "ticker_change", "delisting"}
        and ratio is not None
        and ratio > 0
        and stock_terms_are_explicit
        and filing_identity_match
    ):
        return ParsedLifecycleEvent(
            action_type="stock_merger",
            effective_date=effective,
            cash_amount=cash_amount,
            ratio=ratio,
            new_symbol=new_symbol,
            confidence="high" if new_symbol and effective_is_explicit else "review",
            reason="SEC completion filing contains a stock exchange ratio.",
        )
    if (
        expected_action not in {"stock_merger", "ticker_change", "delisting"}
        and cash_amount is not None
        and cash_amount > 0
        and merger_language
        and filing_identity_match
    ):
        return ParsedLifecycleEvent(
            action_type="cash_merger",
            effective_date=effective,
            cash_amount=cash_amount,
            confidence="high" if effective_is_explicit else "review",
            reason="SEC completion filing contains per-share cash consideration.",
        )
    if ticker is not None and expected_action in {"", "ticker_change"}:
        ticker_effective, new_symbol = ticker
        ticker_effective_is_explicit = bool(ticker_effective)
        ticker_effective = ticker_effective or effective
        return ParsedLifecycleEvent(
            action_type="ticker_change",
            effective_date=ticker_effective,
            new_symbol=new_symbol,
            confidence=(
                "high"
                if ticker_effective_is_explicit or effective_is_explicit
                else "review"
            ),
            reason="SEC filing states the old and new trading symbols and effective date.",
        )
    delisting_language = any(
        phrase in lowered
        for phrase in (
            "notification of removal from listing",
            "will no longer be listed",
            "suspend trading",
            "delisting of the",
            "withdraw the common stock from listing",
        )
    )
    if expected_action in {"", "delisting"} and filing_identity_match and delisting_language and (
        filing.form == "25-NSE" or "3.01" in filing.items or merger_language
    ):
        return ParsedLifecycleEvent(
            action_type="delisting",
            effective_date=effective,
            cash_amount=None,
            confidence="review",
            reason="SEC filing confirms removal from exchange; recovery terms are unknown.",
        )
    return None


def lifecycle_action_record(
    evidence: LifecycleEvidence,
    *,
    new_security_id: str = "",
) -> dict[str, Any]:
    parsed = evidence.parsed
    if parsed is None or evidence.filing is None:
        raise ValueError("Verified SEC evidence is required for a lifecycle action record.")
    if parsed.confidence != "high":
        raise ValueError("Only high-confidence lifecycle evidence can become an action.")
    if parsed.action_type in {"stock_merger", "ticker_change"} and not new_security_id:
        raise ValueError("A resolved successor security is required for this action.")
    if parsed.action_type == "delisting" and parsed.cash_amount is None:
        raise ValueError("A delisting action requires an explicitly verified recovery amount.")
    event_id = canonical_lifecycle_event_id(
        evidence.candidate.security_id,
        parsed.action_type,
        parsed.effective_date,
    )
    return {
        "event_id": event_id,
        "security_id": evidence.candidate.security_id,
        "action_type": parsed.action_type,
        "effective_date": parsed.effective_date,
        "ex_date": parsed.effective_date,
        "announcement_date": evidence.filing.filing_date,
        "record_date": "",
        "payment_date": "",
        "cash_amount": parsed.cash_amount,
        "ratio": parsed.ratio,
        "currency": "USD",
        "new_security_id": new_security_id,
        "new_symbol": parsed.new_symbol,
        "official": True,
        "source": "sec_edgar+eodhd_terminal_price",
        "source_url": evidence.source_url,
        "source_kind": "official_crosscheck",
        "retrieved_at": _utc_now_iso(),
        "source_hash": evidence.source_hash,
    }


def canonical_lifecycle_event_id(
    security_id: str,
    action_type: str,
    effective_date: str,
) -> str:
    # Source-independent identity prevents the same economic event from being
    # applied twice when a second evidence provider is added later.
    value = f"{security_id}|{action_type}|{effective_date}".encode()
    return hashlib.sha256(value).hexdigest()


def resolve_new_security_id(
    master: pd.DataFrame,
    *,
    new_symbol: str,
    effective_date: str,
    symbol_history: pd.DataFrame | None = None,
) -> str:
    if not new_symbol:
        return ""
    timestamp = pd.Timestamp(effective_date)
    history_ids: set[str] = set()
    if symbol_history is not None and not symbol_history.empty:
        history = symbol_history.loc[
            symbol_history["symbol"].astype(str).str.upper().eq(new_symbol.upper())
        ].copy()
        history["_start"] = pd.to_datetime(history["effective_from"], errors="coerce")
        history["_end"] = pd.to_datetime(history["effective_to"], errors="coerce")
        history = history.loc[
            (history["_start"].isna() | (history["_start"] <= timestamp))
            & (history["_end"].isna() | (history["_end"] >= timestamp))
        ]
        history_ids = set(history["security_id"].astype(str))

    if history_ids:
        # Historical aliases are authoritative at the event date.  A reused
        # ticker can belong to an unrelated current company whose
        # primary_symbol happens to match (for example ACT).
        matches = master.loc[
            master["security_id"].astype(str).isin(history_ids)
        ].copy()
    else:
        matches = master.loc[
            master["primary_symbol"].astype(str).str.upper().eq(new_symbol.upper())
        ].copy()
    if matches.empty:
        return ""
    matches["_start"] = pd.to_datetime(matches["active_from"], errors="coerce")
    matches["_end"] = pd.to_datetime(matches["active_to"], errors="coerce")
    covered = matches.loc[
        (matches["_start"].isna() | (matches["_start"] <= timestamp))
        & (matches["_end"].isna() | (matches["_end"] >= timestamp))
    ]
    if not covered.empty:
        matches = covered
    exact_provider = matches.loc[
        matches.get("provider_symbol", pd.Series(index=matches.index, dtype=str))
        .astype(str)
        .str.upper()
        .eq(f"{new_symbol.upper()}.US")
    ]
    if not exact_provider.empty:
        matches = exact_provider
    active = matches.loc[matches["_end"].isna()]
    if not active.empty:
        matches = active
    security_ids = matches["security_id"].astype(str).drop_duplicates()
    if len(security_ids) != 1:
        return ""
    return str(security_ids.iloc[0])


def _filing_from_search_hit(
    source: dict[str, Any],
    *,
    candidate: LifecycleCandidate,
    related_symbols: Iterable[str] = (),
    related_names: Iterable[str] = (),
) -> SecFiling:
    display_names = source.get("display_names") or ()
    display = str(display_names[0] if display_names else "")
    items = tuple(
        part.strip()
        for value in (source.get("items") or ())
        for part in str(value).split(",")
        if part.strip()
    )
    filing_date = str(source.get("file_date") or "")
    name_score = _company_name_similarity(candidate.name, display)
    symbol_match = bool(
        re.search(rf"\({re.escape(candidate.symbol)}\)(?:\s|$)", display, flags=re.I)
    )
    related_symbol_match = any(
        re.search(rf"(?:\(|,\s*){re.escape(str(symbol))}(?:,|\)|\s)", display, flags=re.I)
        for symbol in related_symbols
        if str(symbol).strip()
    )
    related_name_score = max(
        (
            _company_name_similarity(str(name), display)
            for name in related_names
            if str(name).strip()
        ),
        default=0.0,
    )
    item_score = (
        (5.0 if "2.01" in items else 0.0)
        + (3.0 if "3.01" in items else 0.0)
        + (1.0 if "8.01" in items else 0.0)
    )
    distance = abs(
        (pd.Timestamp(filing_date) - pd.Timestamp(candidate.last_price_date)).days
    )
    proximity = max(0.0, 2.0 - distance / 30.0)
    score = (
        name_score * 6.0
        + (8.0 if symbol_match else 0.0)
        + related_name_score * 6.0
        + (8.0 if related_symbol_match else 0.0)
        + item_score
        + proximity
    )
    return SecFiling(
        cik=str((source.get("ciks") or ("",))[0]),
        accession_number=str(source.get("adsh") or ""),
        filing_date=filing_date,
        form=str(source.get("form") or source.get("file_type") or ""),
        items=items,
        display_name=display,
        score=score,
    )


def _parsed_event_score(parsed: ParsedLifecycleEvent, filing: SecFiling) -> float:
    confidence = {"high": 100.0, "review": 0.0}.get(parsed.confidence, -100.0)
    action = {
        "stock_merger": 40.0,
        "cash_merger": 40.0,
        "ticker_change": 30.0,
        "delisting": 10.0,
    }.get(parsed.action_type, 0.0)
    return confidence + action + min(20.0, filing.score)


def _filing_sort_key(
    filing: SecFiling,
    *,
    anchors: Iterable[str],
    expected_action: str,
) -> tuple[float, ...]:
    distance = min(
        abs((pd.Timestamp(filing.filing_date) - pd.Timestamp(anchor)).days)
        for anchor in anchors
    )
    if expected_action == "ticker_change":
        return (0.0 if filing.score >= 12.0 else 1.0), float(distance), -filing.score
    return -filing.score, float(distance)


def _filing_matches_candidate(
    filing: SecFiling,
    candidate: LifecycleCandidate,
) -> bool:
    if re.search(
        rf"\({re.escape(candidate.symbol)}\)(?:\s|$)",
        filing.display_name,
        flags=re.I,
    ):
        return True
    return _company_name_similarity(candidate.name, filing.display_name) >= 0.35


def _filing_text_matches_candidate(
    text: str,
    candidate: LifecycleCandidate,
) -> bool:
    normalized_text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    core = re.sub(
        r"[^a-z0-9]+",
        " ",
        _company_core_name(candidate.name).lower(),
    ).strip()
    if len(core) >= 5 and f" {core} " in f" {normalized_text} ":
        return True
    candidate_tokens = _company_tokens(candidate.name)
    if len(candidate_tokens) < 2:
        return False
    text_tokens = set(normalized_text.split())
    return candidate_tokens.issubset(text_tokens)


def _submission_text(content: bytes | str) -> str:
    raw = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
    parser = _TextExtractor()
    try:
        parser.feed(raw)
        value = " ".join(parser.values)
    except Exception:
        value = re.sub(r"<[^>]+>", " ", raw)
    return " ".join(html.unescape(value).replace("\xa0", " ").split())


def _parse_cash_per_share(text: str) -> float | None:
    # Prefer the consideration sentence for an outstanding common share.  A
    # filing often contains unrelated option strike prices, award conversion
    # ratios and par values after the merger terms; those must never become a
    # shareholder cash consideration.
    patterns = (
        r"each\s+(?:(?:issued\s+and\s+)?outstanding\s+)?share"
        r".{0,700}?converted\s+into.{0,350}?"
        r"(?:USD\s*|\$\s*)([0-9][0-9,]*(?:\.[0-9]+)?)\s+in\s+cash",
        r"each\s+(?:(?:issued\s+and\s+)?outstanding\s+)?share"
        r".{0,700}?converted\s+into.{0,350}?"
        r"(?:and|plus)\s+([0-9][0-9,]*(?:\.[0-9]+)?)\s+USD\b",
        r"each\s+(?:(?:issued\s+and\s+)?outstanding\s+)?share"
        r".{0,700}?(?:converted\s+into\s+the\s+right\s+to\s+receive|"
        r"became\s+entitled\s+to\s+receive).{0,220}?"
        r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s+in\s+cash",
        r"each\s+(?:(?:issued\s+and\s+)?outstanding\s+)?share"
        r".{0,700}?had\s+the\s+right\s+to\s+receive.{0,350}?"
        r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s+in\s+cash",
        r"each\s+holder\s+of\s+a\s+share"
        r".{0,900}?had\s+the\s+right\s+to\s+receive.{0,500}?"
        r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s+in\s+cash",
        r"converted\s+into\s+the\s+right\s+to\s+receive"
        r".{0,350}?\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s+in\s+cash",
        r"right\s+to\s+receive\s+\$\s*"
        r"([0-9][0-9,]*(?:\.[0-9]+)?)\s+in\s+cash",
        r"converted\s+into\s+the\s+right\s+to\s+receive"
        r".{0,350}?amount\s+in\s+cash\s+equal\s+to\s+\$\s*"
        r"([0-9][0-9,]*(?:\.[0-9]+)?)",
        r"converted\s+into\s+the\s+right\s+to\s+receive\s+cash\s+"
        r"in\s+the\s+amount\s+of\s+\$\s*"
        r"([0-9][0-9,]*(?:\.[0-9]+)?)\s+per\s+share",
        r"amount\s+in\s+cash\s+equal\s+to\s+\$\s*"
        r"([0-9][0-9,]*(?:\.[0-9]+)?)\s+per\s+share",
        r"payment\s+of\s+\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s+in\s+cash\s+for\s+each",
        r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s+per\s+(?:share|Share)\s+in\s+cash",
        r"at\s+a\s+price\s+of\s+\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s+per\s+(?:share|Share).{0,160}?in\s+cash",
        r"purchase\s+price\s+of\s+\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s+per\s+(?:share|Share)",
        r"offer\s+price\s+(?:of|equal\s+to)\s+\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s+per\s+(?:share|Share)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return float(match.group(1).replace(",", ""))
    return None


def _parse_stock_exchange(
    text: str,
    *,
    old_symbol: str,
    target_company_name: str,
    known_symbols: set[str],
    preferred_symbols: set[str] | None = None,
) -> tuple[float | None, str, bool]:
    candidates: list[tuple[float, int, float, re.Match[str]]] = []
    target_tokens = _company_tokens(target_company_name)
    ratio_patterns = (
        # SEC completion filings commonly put the defined term directly after
        # the operative number.  This is the strongest form and survives
        # intervening adjectives such as "fully paid and non-assessable".
        r"(?<![$\d.])([0-9]+(?:\.[0-9]+)?)\s*"
        r"\(\s*(?:the\s+)?[\"'“”]?\s*exchange\s+ratio\b[^)]*\)",
        # Covers both "0.118 of a share of" and "1.0192 shares of", including
        # a defined term between "shares" and "of".
        r"(?<![$\d.])([0-9]+(?:\.[0-9]+)?)\s+"
        r"(?:(?:of\s+)?a\s+share|shares?)"
        r"(?:\s*\([^)]{0,100}\))?\s+of\b",
        r"exchange\s+ratio\s*(?:of|:)?\s*([0-9]+(?:\.[0-9]+)?)\b",
        # Some filings omit the defined term entirely: "1.60 Weyerhaeuser
        # common shares" or "0.8708 of an Abbott common share".  The
        # surrounding common-share conversion score below keeps this broad
        # numeric candidate out of option and award paragraphs.
        r"(?<![$\d.])([0-9]+(?:\.[0-9]+)?)\b"
        r"(?=[^.;]{0,140}\b(?:american\s+depositary\s+|ordinary\s+|common\s+)?shares?\b)",
        r"\b(one)\b"
        r"(?=[^.;]{0,140}\b(?:american\s+depositary\s+|ordinary\s+|common\s+)?share\b)",
    )
    for pattern_index, pattern in enumerate(ratio_patterns):
        for match in re.finditer(pattern, text, flags=re.I):
            before = text[max(0, match.start() - 900) : match.start()]
            after = text[match.end() : match.end() + 350]
            context = before + after
            lower_before = before.lower()
            lower_context = context.lower()
            raw_ratio = match.group(1).lower()
            ratio_value = 1.0 if raw_ratio == "one" else float(raw_ratio)
            if ratio_value <= 0 or ratio_value > 20:
                continue
            if pattern_index >= 3:
                stripped_before = before.rstrip()
                stripped_after = after.lstrip()
                if stripped_before.endswith("(") and stripped_after.startswith(")"):
                    continue
                if stripped_after.startswith("%"):
                    continue
                if stripped_before.endswith("-") and stripped_after.startswith("-"):
                    continue
            score = 0.0
            common_share_anchor = bool(re.search(
                r"each\s+(?:(?:issued\s+and\s+)?outstanding\s+)?"
                r"(?:eligible\s+)?(?:common\s+)?share",
                before,
                re.I,
            ))
            if common_share_anchor:
                score += 9.0
            if "common stock" in lower_before[-700:]:
                score += 5.0
            if "converted" in lower_before[-650:]:
                score += 9.0
            if "right to receive" in lower_before[-450:]:
                score += 6.0
            if re.search(r"\breceived\b", lower_before[-350:]):
                score += 6.0
            if "merger consideration" in lower_context:
                score += 3.0
            if "exchange ratio" in lower_context:
                score += 4.0
            nearby_tokens = set(re.findall(r"[a-z0-9]+", lower_before[-250:]))
            target_overlap = len(target_tokens & nearby_tokens)
            if target_overlap >= min(2, len(target_tokens)):
                score += 30.0
            elif target_overlap == 1:
                score += 8.0
            if pattern_index == 0:
                score += 12.0
            elif pattern_index == 2:
                score += 2.0
            if raw_ratio == "one" and re.match(
                r"\s*\(\s*1\s*\)",
                text[match.end() : match.end() + 12],
            ):
                score += 12.0

            # If an employee-security paragraph occurs after the last common
            # share conversion phrase, this numeric value is not the merger
            # exchange ratio.
            disqualifiers = tuple(
                lower_before.rfind(value)
                for value in (
                    "stock option",
                    "restricted stock",
                    "performance award",
                    "equity award",
                    "exercise price",
                    "director award",
                    "employee award",
                )
            )
            operative = max(
                lower_before.rfind("each share"),
                lower_before.rfind("each common share"),
                lower_before.rfind("common stock"),
                lower_before.rfind("converted"),
            )
            if max(disqualifiers, default=-1) > operative:
                score -= 30.0
            if re.search(r"par\s+value\s+\$?\s*$", lower_before[-80:]):
                score -= 30.0
            if re.search(r"\$\s*$", before[-12:]):
                score -= 30.0
            if pattern_index >= 3 and re.search(r"\(\s*$", before[-6:]):
                score -= 40.0
            if pattern_index >= 3 and re.search(
                r"(?:form|item|section)[^a-z0-9]{0,6}$",
                lower_before[-20:],
            ):
                score -= 40.0
            if re.search(
                r"each\s+share.{0,180}?preferred\s+stock.{0,500}$",
                lower_before,
            ):
                score -= 60.0
            consideration = max(
                lower_before.rfind("right to receive"),
                lower_before.rfind("converted into"),
                lower_before.rfind(" received"),
            )
            consideration_distance = (
                len(lower_before) - consideration if consideration >= 0 else 10_000
            )
            if (
                pattern_index != 0
                and not common_share_anchor
                and "right to receive" not in lower_before[-450:]
            ):
                score -= 30.0
            if consideration_distance <= 60:
                score += 20.0
            elif consideration_distance <= 150:
                score += 12.0
            elif consideration_distance <= 300:
                score += 5.0
            elif pattern_index >= 3:
                score -= 30.0
            if score >= 14.0:
                candidates.append((score, -match.start(), ratio_value, match))

    if not candidates:
        return None, "", False
    _, _, ratio, ratio_match = max(
        candidates,
        key=lambda value: (value[0], value[1]),
    )
    preferred_symbols = preferred_symbols or set()
    # A filing's phrases such as "Class V Common Stock" or "Series C Common
    # Stock" are share classes, not exchange tickers.  Never map them through
    # the global symbol universe (where V and C are unrelated listed firms).
    # Successors come from the separately reviewed lifecycle hint catalog.
    new_symbol = next(
        (value for value in sorted(preferred_symbols) if value != old_symbol),
        "",
    )
    return ratio, new_symbol, True


def _parse_ticker_change(
    text: str,
    *,
    old_symbol: str,
    preferred_symbols: set[str] | None = None,
) -> tuple[str, str] | None:
    escaped = re.escape(old_symbol)
    direct_patterns = (
        rf"ticker\s+symbol\s+(?:will\s+)?change\s+from\s+['\"“”]?{escaped}['\"“”]?\s+to\s+['\"“”]?([A-Z][A-Z0-9.\-]{{0,8}})",
        rf"replace(?:s|d)?\s+the\s+company(?:'s|’s)\s+current\s+ticker\s+symbol\s+['\"“”]?{escaped}['\"“”]?",
    )
    new_symbol = ""
    direct = re.search(direct_patterns[0], text, flags=re.I)
    if direct:
        new_symbol = direct.group(1).upper()
    else:
        replacement = re.search(direct_patterns[1], text, flags=re.I)
        if replacement:
            before = text[max(0, replacement.start() - 500) : replacement.start()]
            matches = list(
                re.finditer(
                    r"(?i:(?:new\s+)?ticker\s+symbol(?:\s+will\s+be|\s+to)?\s+)"
                    r"['\"“”]([A-Z][A-Z0-9.\-]{0,8})['\"“”]",
                    before,
                )
            )
            if matches:
                new_symbol = matches[-1].group(1).upper()
    if not new_symbol or new_symbol == old_symbol:
        for preferred in sorted(preferred_symbols or ()):
            if preferred == old_symbol:
                continue
            preferred_pattern = re.escape(preferred)
            if re.search(
                rf"(?:begin|commence|continue)\s+trading.{{0,180}}?"
                rf"ticker\s+symbol\s+['\"“”]?{preferred_pattern}['\"“”]?",
                text,
                flags=re.I,
            ) or re.search(
                rf"ticker\s+symbol\s+(?:will\s+)?(?:be\s+)?(?:change(?:d)?\s+)?to\s+"
                rf"['\"“”]?{preferred_pattern}['\"“”]?",
                text,
                flags=re.I,
            ) or re.search(
                rf"(?:change|changing|changed)\s+(?:our|its|the\s+company(?:'s|’s))?\s*"
                rf"ticker(?:\s+symbol)?\s+to\s+['\"“”]?{preferred_pattern}['\"“”]?",
                text,
                flags=re.I,
            ) or re.search(
                rf"under\s+(?:a\s+)?(?:new\s+)?ticker(?:\s+symbol)?\s+"
                rf"[,\s]*['\"“”]?{preferred_pattern}['\"“”]?",
                text,
                flags=re.I,
            ) or re.search(
                rf"ticker(?:\s+symbol)?.{{0,220}}?"
                rf"(?:['\"“”]\s*{preferred_pattern}\s*['\"“”]|\b{preferred_pattern}\b)",
                text,
                flags=re.I,
            ):
                new_symbol = preferred
                break
    if not new_symbol or new_symbol == old_symbol:
        return None
    date_patterns = (
        rf"effective\s+({_MONTH_DATE_PATTERN}).{{0,300}}?"
        rf"(?:new\s+)?ticker\s+symbol[,\s]*['\"“”]?{re.escape(new_symbol)}['\"“”]?",
        rf"(?:under\s+the\s+(?:new\s+)?ticker\s+symbol\s+['\"“”]?{re.escape(new_symbol)}['\"“”]?).{{0,160}}?"
        rf"(?:on|effective)\s+({_MONTH_DATE_PATTERN})",
        rf"({_MONTH_DATE_PATTERN}).{{0,220}}?ticker\s+symbol[,\s]*['\"“”]?{re.escape(new_symbol)}",
        rf"(?:begin|commence|continue)\s+trading.{{0,220}}?"
        rf"(?:on|effective)\s+({_MONTH_DATE_PATTERN}).{{0,120}}?"
        rf"ticker\s+symbol\s+['\"“”]?{re.escape(new_symbol)}",
        rf"ticker\s+symbol\s+['\"“”]?{re.escape(new_symbol)}['\"“”]?"
        rf".{{0,220}}?(?:on|effective)\s+({_MONTH_DATE_PATTERN})",
        rf"(?:changing|change|changed)\s+(?:our|its|the\s+company(?:'s|’s))\s+"
        rf"ticker(?:\s+symbol)?\s+(?:to\s+)?['\"“”]?{re.escape(new_symbol)}['\"“”]?"
        rf".{{0,120}}?effective\s+({_MONTH_DATE_PATTERN})",
        rf"(?:begin|commence)\s+trading.{{0,180}}?({_MONTH_DATE_PATTERN})"
        rf".{{0,180}}?(?:ticker|symbol).{{0,80}}?['\"“”]?{re.escape(new_symbol)}['\"“”]?",
        rf"ticker\s+symbol.{{0,80}}?(?:will\s+)?change\s+to\s+"
        rf"['\"“”]?{re.escape(new_symbol)}['\"“”]?.{{0,180}}?on\s+"
        rf"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)?,?\s*({_MONTH_DATE_PATTERN})",
    )
    for pattern in date_patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            date_value = _coerce_date(match.group(1))
            if date_value:
                return date_value, new_symbol
    return "", new_symbol


_MONTH_DATE_PATTERN = (
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2}\s*,\s+\d{4}"
)


def _parse_completion_date(
    text: str,
    candidate: LifecycleCandidate,
    filing: SecFiling,
) -> str:
    return _parse_completion_date_with_evidence(text, candidate, filing)[0]


def _parse_completion_date_with_evidence(
    text: str,
    candidate: LifecycleCandidate,
    filing: SecFiling,
) -> tuple[str, bool]:
    patterns = (
        rf"consummation,?\s+on\s+({_MONTH_DATE_PATTERN})"
        rf".{{0,240}}?(?:Merger|merger|acquisition|transaction)",
        rf"On\s+({_MONTH_DATE_PATTERN}),?[^.]{{0,500}}?"
        rf"(?:completed|consummated|closed)\s+(?:the\s+|its\s+)?"
        rf"(?:previously\s+announced\s+)?(?:Merger|merger|acquisition|transaction)",
        rf"On\s+({_MONTH_DATE_PATTERN}),.{{0,500}}?"
        rf"(?:completed|consummated|closed)\s+(?:the|its)?\s*(?:Merger|merger|acquisition|transaction)",
        rf"On\s+({_MONTH_DATE_PATTERN}).{{0,650}}?merged\s+with\s+and\s+into",
        rf"On\s+({_MONTH_DATE_PATTERN}).{{0,650}}?(?:Merger|merger)\s+(?:became|was)\s+effective",
        rf"On\s+({_MONTH_DATE_PATTERN}).{{0,900}}?"
        rf"(?:consummated\s+the\s+transactions?|completed\s+the\s+transactions?|"
        rf"became\s+a\s+(?:direct|indirect|wholly[- ]owned)\s+subsidiary)",
        rf"(?:completed|consummated|closed)\s+(?:the|its)?\s*(?:Merger|merger|acquisition|transaction)"
        rf".{{0,180}}?on\s+({_MONTH_DATE_PATTERN})",
        rf"(?:Merger|merger|Acquisition|acquisition|Transaction|transaction)\s+"
        rf"(?:was\s+)?(?:completed|consummated|closed).{{0,180}}?on\s+({_MONTH_DATE_PATTERN})",
        rf"completion\s+of\s+(?:the\s+)?(?:Merger|merger|Acquisition|acquisition|Transaction|transaction)"
        rf".{{0,180}}?on\s+({_MONTH_DATE_PATTERN})",
        rf"effective\s+time.{{0,180}}?on\s+({_MONTH_DATE_PATTERN})",
        rf"(?:Merger|merger).{{0,180}}?\s+(?:became?s?|was)\s+effective\s+on\s+"
        rf"({_MONTH_DATE_PATTERN})",
    )
    candidate_date = pd.Timestamp(candidate.last_price_date)
    strong_explicit_dates: list[tuple[int, str]] = []
    # Scan every explicit "On <date>" start independently.  A prior agreement
    # date can otherwise consume the later closing date in a broad regex, and
    # legal company names such as "Pfizer Inc." contain periods inside the
    # completion sentence.
    for match in re.finditer(rf"On\s+({_MONTH_DATE_PATTERN})", text, flags=re.I):
        following = text[match.end() : match.end() + 500]
        next_explicit = re.search(
            rf"\bOn\s+{_MONTH_DATE_PATTERN}", following, flags=re.I
        )
        if next_explicit:
            following = following[: next_explicit.start()]
        if not re.search(
            r"(?:"
            r"(?:completed|consummated|closed)\s+(?:the\s+|its\s+)?"
            r"(?:previously\s+announced\s+)?"
            r"(?:Merger|merger|acquisition|transaction)"
            r"|merged\s+with\s+and\s+into"
            r"|(?:Merger|merger)s?\s+(?:became|were|was)\s+"
            r"(?:effective|consummated|completed)"
            r")",
            following,
            flags=re.I,
        ):
            continue
        value = _coerce_date(match.group(1))
        if value:
            gap = abs((pd.Timestamp(value) - candidate_date).days)
            if gap <= 75:
                strong_explicit_dates.append((gap, value))
    # A direct ``On <date> ... completed/merged`` statement is stronger than
    # proximity to the last price date.  In mixed-event 8-Ks the cover-page
    # "earliest event" or a shareholder-vote Effective Time can be closer to
    # the terminal session than the later, explicitly stated closing date.
    if strong_explicit_dates:
        return min(strong_explicit_dates)[1], True
    explicit_dates: list[tuple[int, int, str]] = []
    for pattern_index, pattern in enumerate(patterns):
        for match in re.finditer(pattern, text, flags=re.I):
            value = _coerce_date(match.group(1))
            if value:
                gap = abs((pd.Timestamp(value) - candidate_date).days)
                if gap <= 75:
                    explicit_dates.append((gap, pattern_index, value))
    if explicit_dates:
        return min(explicit_dates)[2], True
    filing_date = pd.Timestamp(filing.filing_date)
    if abs((filing_date - candidate_date).days) <= 7:
        return filing_date.date().isoformat(), False
    return candidate.last_price_date, False


def _coerce_date(value: str) -> str:
    try:
        normalized = re.sub(r"\s+,", ",", str(value))
        return pd.Timestamp(normalized).date().isoformat()
    except (TypeError, ValueError):
        return ""


def _searchable_company_name(value: str) -> str:
    normalized = re.sub(r"\([^)]*\)", " ", str(value))
    normalized = re.sub(r"[^A-Za-z0-9& ]+", " ", normalized)
    return " ".join(normalized.split())


def _company_core_name(value: str) -> str:
    tokens = _searchable_company_name(value).split()
    suffixes = {
        "co",
        "company",
        "corp",
        "corporation",
        "inc",
        "incorporated",
        "limited",
        "ltd",
        "llc",
        "lp",
        "plc",
    }
    while tokens and tokens[-1].lower().rstrip(".") in suffixes:
        tokens.pop()
    return " ".join(tokens)


def _company_name_similarity(left: str, right: str) -> float:
    left_tokens = _company_tokens(left)
    right_tokens = _company_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _company_tokens(value: str) -> set[str]:
    ignored = {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "company",
        "co",
        "plc",
        "limited",
        "ltd",
        "holdings",
        "group",
        "the",
        "cik",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value).lower())
        if token not in ignored and not token.isdigit()
    }


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
