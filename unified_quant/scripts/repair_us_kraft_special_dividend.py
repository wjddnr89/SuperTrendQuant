#!/usr/bin/env python3
"""Install Kraft's exact 2015 pre-merger special dividend offline.

Kraft Foods Group holders immediately before the July 2, 2015 effective time
received both one KHC share for each KRFT share and $16.50 cash.  The current
identity repair correctly separates KRFT and KHC and records the 1:1 stock
merger, but the transaction-linked special dividend is absent.

The default command is a strict read-only plan.  It verifies three hash-pinned
official SEC documents, the exact repaired KRFT/KHC identity boundary and the
absence of any duplicate distribution.  ``--apply`` is available for a later
explicitly approved run and uses a single writer lock, release and pointer
compare-and-swap, immutable versions, a durable journal, verified rollback and
post-commit idempotence.  There is no network, EODHD, or R2 code path.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import html
import json
import math
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.lifecycle import canonical_lifecycle_event_id
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.validation import (
    validate_dataset,
    validate_repository_snapshot,
)


DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_EVIDENCE_DIR = DEFAULT_CACHE_ROOT / "state/issuer_lifecycle"
EVIDENCE_REPORT = "kraft_special_dividend_evidence.json"
POLICY_AS_OF = "2026-07-15"
POLICY_VERSION = "kraft_2015_special_dividend/v1"
REVIEWED_AT = "2026-07-18T09:34:38.650933Z"

KRFT_ID = "US:EODHD:e3b28684-f48e-582a-8836-b2f5579f2dd2"
KHC_ID = "US:EODHD:a060087d-109e-58e9-91f9-b4c398a3d415"
KRFT_SYMBOL = "KRFT"
KHC_SYMBOL = "KHC"
DECLARATION_DATE = "2015-06-22"
EFFECTIVE_DATE = "2015-07-02"
RECORD_DATE = EFFECTIVE_DATE
PAYMENT_DATE = EFFECTIVE_DATE
KHC_FIRST_TRADING_SESSION = "2015-07-06"
SPECIAL_DIVIDEND_USD = 16.50
PRE_DIVIDEND_LAST_CLOSE = 88.30
EXPECTED_KRFT_PRICE_ROWS = 126
EXPECTED_KHC_PRICE_ROWS = 2_773
EXPECTED_FACTOR_VALUE_CHANGES = 125

SPECIAL_DIVIDEND_EVENT_ID = canonical_lifecycle_event_id(
    KRFT_ID, "special_dividend", EFFECTIVE_DATE
)
STOCK_MERGER_EVENT_ID = canonical_lifecycle_event_id(
    KRFT_ID, "stock_merger", EFFECTIVE_DATE
)
REQUIRED_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "corporate_actions",
    "adjustment_factors",
    "index_constituent_anchors",
    "index_membership_events",
    "lifecycle_resolutions",
    "source_archive",
)
WRITE_DATASETS = (
    "source_archive",
    "corporate_actions",
    "adjustment_factors",
)


@dataclass(frozen=True)
class EvidenceSpec:
    label: str
    source_url: str
    source_hash: str
    size: int
    retrieved_at: str
    filename: str
    archive_object_path: str
    content_type: str
    required_text_groups: tuple[tuple[str, ...], ...]
    already_archived: bool = False


DECLARATION_EVIDENCE = EvidenceSpec(
    label="kraft_special_dividend_declaration",
    source_url=(
        "https://www.sec.gov/Archives/edgar/data/1545158/"
        "000119312515230632/d947291d425.htm"
    ),
    source_hash="c9d78b9704c3b2b95c018dca5d7e7123a8a1e5d7bae8e1b30152b3037fc26849",
    size=12_892,
    retrieved_at="2026-07-18T09:34:38.349399Z",
    filename="c9d78b9704c3b2b95c018dca5d7e7123a8a1e5d7bae8e1b30152b3037fc26849.html",
    archive_object_path=(
        "archives/2026-07-15/"
        "c9d78b9704c3b2b95c018dca5d7e7123a8a1e5d7bae8e1b30152b3037fc26849.html.gz"
    ),
    content_type="text/html",
    required_text_groups=(
        ("June 22, 2015",),
        ("special cash dividend in the amount of $16.50 per share",),
        ("conditioned upon the closing of the proposed merger",),
        ("payable to Kraft shareholders of record immediately prior",),
    ),
)

PAYMENT_EVIDENCE = EvidenceSpec(
    label="kraft_special_dividend_completion_payment",
    source_url=(
        "https://www.sec.gov/Archives/edgar/data/1637459/"
        "000163745915000021/khc10q62815.htm"
    ),
    source_hash="f138d8464a92839720f8a4441a55b5fc96852a89c917f2b3965d1121075f7875",
    size=1_769_064,
    retrieved_at="2026-07-18T09:34:38.650933Z",
    filename="f138d8464a92839720f8a4441a55b5fc96852a89c917f2b3965d1121075f7875.html",
    archive_object_path=(
        "archives/2026-07-15/"
        "f138d8464a92839720f8a4441a55b5fc96852a89c917f2b3965d1121075f7875.html.gz"
    ),
    content_type="text/html",
    required_text_groups=(
        ("consummated on July 2, 2015",),
        ("on a one-for-one basis",),
        ("Upon the completion of the 2015 Merger",),
        ("received a special cash dividend of $16.50 per share",),
    ),
)

COMPLETION_EVIDENCE = EvidenceSpec(
    label="kraft_heinz_merger_completion",
    source_url=(
        "https://www.sec.gov/Archives/edgar/data/1637459/"
        "000119312515244356/0001193125-15-244356.txt"
    ),
    source_hash="ba6d15da235a1f5f6a186fb7b9e52b7b438c25cf76d4f28d9fac81953b7fc299",
    size=1_440_758,
    retrieved_at="2026-07-18T08:11:54.578547Z",
    filename="",
    archive_object_path=(
        "archives/2026-07-15/"
        "ba6d15da235a1f5f6a186fb7b9e52b7b438c25cf76d4f28d9fac81953b7fc299.txt.gz"
    ),
    content_type="text/plain",
    required_text_groups=(
        ("On July 2, 2015", "On July&nbsp;2, 2015"),
        ("converted into the right to receive one fully paid",),
        ("on June 22, 2015, Kraft declared a special cash dividend",),
        ("$16.50 per share of Kraft Common Stock",),
        ("shareholders of record immediately prior to the closing",),
    ),
    already_archived=True,
)

EVIDENCE_SPECS = (
    DECLARATION_EVIDENCE,
    PAYMENT_EVIDENCE,
    COMPLETION_EVIDENCE,
)


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: Mapping[str, str | None]
    planned_versions: Mapping[str, str]
    frames: Mapping[str, pd.DataFrame]
    evidence_content: Mapping[str, bytes]
    summary: Mapping[str, Any]


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _date(value: Any) -> str:
    raw = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(raw) else raw.date().isoformat()


def _normalized_document_text(content: bytes) -> str:
    decoded = content.decode("utf-8", errors="replace")
    decoded = html.unescape(re.sub(r"<[^>]+>", " ", decoded))
    return re.sub(r"\s+", " ", decoded).strip().casefold()


def _verify_content(content: bytes, spec: EvidenceSpec) -> None:
    digest = hashlib.sha256(content).hexdigest()
    if digest != spec.source_hash or len(content) != spec.size:
        raise ValueError(
            f"{spec.label} hash/size mismatch: sha256={digest}; size={len(content)}."
        )
    normalized = _normalized_document_text(content)
    for alternatives in spec.required_text_groups:
        if not any(value.casefold() in normalized for value in alternatives):
            raise ValueError(
                f"{spec.label} lacks reviewed official term: "
                + " | ".join(alternatives)
            )


def _verified_evidence(
    repository: LocalDatasetRepository,
    source_archive: pd.DataFrame,
    evidence_dir: Path,
) -> dict[str, bytes]:
    report_path = evidence_dir / EVIDENCE_REPORT
    if not report_path.is_file():
        raise FileNotFoundError(f"Pinned Kraft evidence report is missing: {report_path}.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reported = {str(row.get("label")): row for row in report.get("evidence", [])}
    output: dict[str, bytes] = {}
    for spec in EVIDENCE_SPECS:
        if spec.already_archived:
            matches = source_archive.loc[
                source_archive["archive_id"].astype(str).eq(spec.source_hash)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{spec.label} source_archive row is missing or duplicated."
                )
            row = matches.iloc[0]
            expected = {
                "object_path": spec.archive_object_path,
                "source_hash": spec.source_hash,
                "source_url": spec.source_url,
            }
            if any(_text(row.get(key)) != value for key, value in expected.items()):
                raise ValueError(f"{spec.label} archive provenance changed.")
            path = repository.root / spec.archive_object_path
            if not path.is_file():
                raise FileNotFoundError(f"Pinned archive payload is missing: {path}.")
            try:
                content = gzip.decompress(path.read_bytes())
            except (OSError, EOFError) as exc:
                raise ValueError(f"{spec.label} archive is not valid gzip.") from exc
        else:
            row = reported.get(spec.label)
            if row is None:
                raise ValueError(f"Evidence report lacks {spec.label}.")
            expected = {
                "source_url": spec.source_url,
                "source_hash": spec.source_hash,
                "filename": spec.filename,
                "size": spec.size,
                "retrieved_at": spec.retrieved_at,
            }
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Evidence report conflicts for {spec.label}.")
            path = evidence_dir / spec.filename
            if not path.is_file():
                raise FileNotFoundError(f"Pinned evidence payload is missing: {path}.")
            content = path.read_bytes()
        _verify_content(content, spec)
        output[spec.label] = content
    return output


def _one_row(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one {label}; observed {len(rows)}.")
    return rows.iloc[0]


def _preflight(frames: Mapping[str, pd.DataFrame]) -> None:
    master = frames["security_master"]
    history = frames["symbol_history"]
    prices = frames["daily_price_raw"]
    actions = frames["corporate_actions"]
    factors = frames["adjustment_factors"]

    krft_master = _one_row(
        master,
        master["security_id"].astype(str).eq(KRFT_ID),
        "KRFT security master row",
    )
    khc_master = _one_row(
        master,
        master["security_id"].astype(str).eq(KHC_ID),
        "KHC security master row",
    )
    if (
        _text(krft_master.get("primary_symbol")) != KRFT_SYMBOL
        or _date(krft_master.get("active_to")) != EFFECTIVE_DATE
        or _text(khc_master.get("primary_symbol")) != KHC_SYMBOL
        or _date(khc_master.get("active_from")) != KHC_FIRST_TRADING_SESSION
    ):
        raise ValueError("The reviewed KRFT/KHC identity boundary changed.")

    krft_history = _one_row(
        history,
        history["security_id"].astype(str).eq(KRFT_ID)
        & history["symbol"].astype(str).eq(KRFT_SYMBOL),
        "KRFT symbol history row",
    )
    khc_history = _one_row(
        history,
        history["security_id"].astype(str).eq(KHC_ID)
        & history["symbol"].astype(str).eq(KHC_SYMBOL),
        "KHC symbol history row",
    )
    if (
        _date(krft_history.get("effective_to")) != EFFECTIVE_DATE
        or _date(khc_history.get("effective_from")) != EFFECTIVE_DATE
    ):
        raise ValueError("KRFT/KHC legal symbol-history boundary changed.")

    sessions = pd.to_datetime(prices["session"], errors="raise")
    krft_prices = prices.loc[prices["security_id"].astype(str).eq(KRFT_ID)].copy()
    khc_prices = prices.loc[prices["security_id"].astype(str).eq(KHC_ID)].copy()
    if (
        len(krft_prices) != EXPECTED_KRFT_PRICE_ROWS
        or _date(krft_prices["session"].min()) != "2015-01-02"
        or _date(krft_prices["session"].max()) != EFFECTIVE_DATE
        or len(khc_prices) != EXPECTED_KHC_PRICE_ROWS
        or _date(khc_prices["session"].min()) != KHC_FIRST_TRADING_SESSION
        or bool(
            (
                prices["security_id"].astype(str).eq(KHC_ID)
                & sessions.lt(pd.Timestamp(KHC_FIRST_TRADING_SESSION))
            ).any()
        )
    ):
        raise ValueError("Reviewed KRFT/KHC price inventory changed.")
    last_pre_dividend = _one_row(
        krft_prices,
        pd.to_datetime(krft_prices["session"]).dt.date.astype(str).eq("2015-07-01"),
        "KRFT 2015-07-01 close",
    )
    if not math.isclose(
        float(last_pre_dividend["close"]),
        PRE_DIVIDEND_LAST_CLOSE,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("KRFT pre-dividend last close changed.")

    merger = _one_row(
        actions,
        actions["event_id"].astype(str).eq(STOCK_MERGER_EVENT_ID),
        "KRFT stock merger",
    )
    if not (
        _text(merger.get("security_id")) == KRFT_ID
        and _text(merger.get("action_type")) == "stock_merger"
        and _date(merger.get("effective_date")) == EFFECTIVE_DATE
        and math.isclose(float(merger.get("ratio")), 1.0, rel_tol=0, abs_tol=1e-12)
        and _text(merger.get("new_security_id")) == KHC_ID
        and _text(merger.get("new_symbol")) == KHC_SYMBOL
        and bool(merger.get("official"))
        and _text(merger.get("source_hash")) == COMPLETION_EVIDENCE.source_hash
    ):
        raise ValueError("The exact 1:1 KRFT-to-KHC merger action changed.")

    near = actions.loc[
        actions["security_id"].astype(str).isin({KRFT_ID, KHC_ID})
        & pd.to_numeric(actions["cash_amount"], errors="coerce").sub(
            SPECIAL_DIVIDEND_USD
        ).abs().le(1e-12)
    ]
    exact = near["event_id"].astype(str).eq(SPECIAL_DIVIDEND_EVENT_ID)
    if len(near) not in {0, 1} or (len(near) == 1 and not bool(exact.iloc[0])):
        raise ValueError("A conflicting or duplicated Kraft $16.50 action exists.")

    krft_factors = factors.loc[factors["security_id"].astype(str).eq(KRFT_ID)]
    if (
        len(krft_factors) != EXPECTED_KRFT_PRICE_ROWS
        or set(pd.to_datetime(krft_factors["session"]).dt.date.astype(str))
        != set(pd.to_datetime(krft_prices["session"]).dt.date.astype(str))
    ):
        raise ValueError("KRFT adjustment-factor inventory changed.")


def _action_metadata() -> str:
    return json.dumps(
        {
            "policy": POLICY_VERSION,
            "economic_sequence": [
                "KRFT special-dividend entitlement and payment",
                "1:1 KRFT-to-KHC stock merger",
            ],
            "record_time": "immediately_prior_to_2015_merger_effective_time",
            "payment_basis": (
                "KHC 2015 Form 10-Q states record holders received the cash "
                "dividend upon completion of the merger"
            ),
            "evidence": [
                {
                    "label": spec.label,
                    "source_url": spec.source_url,
                    "source_hash": spec.source_hash,
                }
                for spec in EVIDENCE_SPECS
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _expected_action(columns: pd.Index) -> dict[str, Any]:
    row: dict[str, Any] = {column: None for column in columns}
    values = {
        "event_id": SPECIAL_DIVIDEND_EVENT_ID,
        "security_id": KRFT_ID,
        "action_type": "special_dividend",
        "effective_date": EFFECTIVE_DATE,
        "ex_date": EFFECTIVE_DATE,
        "announcement_date": DECLARATION_DATE,
        "record_date": RECORD_DATE,
        "payment_date": PAYMENT_DATE,
        "cash_amount": SPECIAL_DIVIDEND_USD,
        "ratio": None,
        "currency": "USD",
        "new_security_id": "",
        "new_symbol": "",
        "official": True,
        "source_url": PAYMENT_EVIDENCE.source_url,
        "source_kind": "official_crosscheck",
        "source": "sec_edgar+reviewed_special_dividend",
        "retrieved_at": PAYMENT_EVIDENCE.retrieved_at,
        "source_hash": PAYMENT_EVIDENCE.source_hash,
        "metadata": _action_metadata(),
    }
    for key, value in values.items():
        if key in row:
            row[key] = value
    return row


def _action_is_exact(row: Mapping[str, Any]) -> bool:
    expected = _expected_action(pd.Index(row.keys()))
    for key, value in expected.items():
        actual = row.get(key)
        if key in {
            "effective_date",
            "ex_date",
            "announcement_date",
            "record_date",
            "payment_date",
        }:
            if _date(actual) != value:
                return False
        elif key == "cash_amount":
            try:
                if not math.isclose(float(actual), float(value), rel_tol=0, abs_tol=1e-12):
                    return False
            except (TypeError, ValueError):
                return False
        elif key == "ratio":
            if _text(actual):
                return False
        elif key == "official":
            if bool(actual) is not True:
                return False
        elif key == "metadata":
            try:
                actual_json = json.dumps(
                    json.loads(_text(actual)),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError):
                return False
            if actual_json != value:
                return False
        elif _text(actual) != _text(value):
            return False
    return True


def _rewrite_actions(actions: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    output = actions.copy()
    matches = output["event_id"].astype(str).eq(SPECIAL_DIVIDEND_EVENT_ID)
    if matches.any():
        rows = output.loc[matches]
        if len(rows) != 1 or not _action_is_exact(rows.iloc[0].to_dict()):
            raise ValueError("Existing Kraft special-dividend action is not exact.")
        return output.reset_index(drop=True), False
    addition = pd.DataFrame([_expected_action(output.columns)]).loc[:, output.columns]
    output = pd.concat([output, addition], ignore_index=True, sort=False)
    output = output.drop_duplicates(
        list(dataset_spec("corporate_actions").primary_key), keep="last"
    )
    return output.reset_index(drop=True), True


def _archive_row(spec: EvidenceSpec, columns: pd.Index) -> dict[str, Any]:
    row: dict[str, Any] = {column: None for column in columns}
    values = {
        "archive_id": spec.source_hash,
        "dataset": "sec_edgar_filing",
        "object_path": spec.archive_object_path,
        "content_type": spec.content_type,
        "effective_date": EFFECTIVE_DATE,
        "source": "sec_edgar_filing",
        "retrieved_at": spec.retrieved_at,
        "source_hash": spec.source_hash,
        "source_url": spec.source_url,
    }
    for key, value in values.items():
        if key in row:
            row[key] = value
    return row


def _rewrite_source_archive(
    source_archive: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    output = source_archive.copy()
    additions = []
    for spec in (DECLARATION_EVIDENCE, PAYMENT_EVIDENCE):
        expected = _archive_row(spec, output.columns)
        matches = output["archive_id"].astype(str).eq(spec.source_hash)
        if matches.any():
            rows = output.loc[matches]
            if len(rows) != 1 or any(
                _text(rows.iloc[0].get(key)) != _text(value)
                for key, value in expected.items()
            ):
                raise ValueError(f"Conflicting source_archive row for {spec.label}.")
            continue
        additions.append(expected)
    if additions:
        output = pd.concat(
            [output, pd.DataFrame(additions).loc[:, output.columns]],
            ignore_index=True,
            sort=False,
        )
    return output.reset_index(drop=True), len(additions)


def _adjustment_source_version(
    daily_price_version: str,
    corporate_actions_version: str,
) -> str:
    if not daily_price_version or not corporate_actions_version:
        raise RuntimeError(
            "Adjustment factors require exact daily-price and corporate-action versions."
        )
    return f"{daily_price_version}+{corporate_actions_version}"


def _new_planned_versions(release: DataRelease) -> dict[str, str]:
    transaction_id = uuid.uuid4().hex
    session = release.completed_session.replace("-", "")
    return {
        dataset: (
            f"kraft-special-dividend-{session}-{transaction_id}-{dataset}"
        )
        for dataset in WRITE_DATASETS
    }


def _expected_snapshot_factors(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    source_version: str,
) -> pd.DataFrame:
    output = build_adjustment_factors(
        prices,
        actions,
        source_version=source_version,
    )
    output["source_version"] = source_version
    output["calculated_at"] = REVIEWED_AT
    for column, value in {
        "source": "derived",
        "retrieved_at": REVIEWED_AT,
        "source_hash": source_version,
    }.items():
        if column in output:
            output[column] = value
    return output


def _rewrite_factors(
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    source_version: str,
    expected_value_changes: int,
) -> tuple[pd.DataFrame, int, bool]:
    expected = _expected_snapshot_factors(
        prices,
        actions,
        source_version=source_version,
    ).reindex(columns=factors.columns)
    key_columns = ["security_id", "session"]
    value_columns = ["split_factor", "total_return_factor"]
    current = factors.copy()
    current["session"] = pd.to_datetime(current["session"], errors="raise").dt.normalize()
    expected["session"] = pd.to_datetime(expected["session"], errors="raise").dt.normalize()
    joined = current[key_columns + value_columns].merge(
        expected[key_columns + value_columns],
        on=key_columns,
        how="outer",
        suffixes=("_old", "_new"),
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Adjustment-factor inventory changed during Kraft repair.")
    numeric_change = (
        pd.to_numeric(joined["split_factor_old"]).sub(
            pd.to_numeric(joined["split_factor_new"])
        ).abs().gt(1e-12)
        | pd.to_numeric(joined["total_return_factor_old"]).sub(
            pd.to_numeric(joined["total_return_factor_new"])
        ).abs().gt(1e-12)
    )
    changed_rows = int(numeric_change.sum())
    unexpected = joined.loc[
        numeric_change & ~joined["security_id"].astype(str).eq(KRFT_ID),
        key_columns,
    ]
    if not unexpected.empty:
        sample = unexpected.head(10).to_dict("records")
        raise ValueError(
            "Kraft repair would change non-KRFT adjustment economics: "
            + json.dumps(sample, default=str, sort_keys=True)
        )
    if changed_rows != expected_value_changes:
        raise ValueError(
            "Kraft special dividend changed an unexpected factor inventory: "
            f"expected={expected_value_changes}; observed={changed_rows}."
        )
    exact_provenance = (
        set(current["source_version"].astype(str)) == {source_version}
        and set(current["source_hash"].astype(str)) == {source_version}
        and set(current["source"].astype(str)) == {"derived"}
    )
    if changed_rows == 0 and exact_provenance:
        return factors.reset_index(drop=True), 0, False
    return expected.reset_index(drop=True), changed_rows, True


class _CandidateRepository:
    def __init__(
        self,
        base: LocalDatasetRepository,
        versions: Mapping[str, str],
        overrides: Mapping[str, pd.DataFrame],
    ):
        self.base = base
        self.versions = dict(versions)
        self.overrides = dict(overrides)

    def current_manifest(self, dataset: str):
        version = self.versions.get(dataset)
        return self.base.manifest_for_version(dataset, version) if version else None

    def read_frame(self, dataset: str, _version: str | None = None) -> pd.DataFrame:
        if dataset in self.overrides:
            return self.overrides[dataset].copy()
        return self.base.read_frame(dataset, self.versions[dataset])


def _capture_pointer_etags(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for dataset in WRITE_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != release.dataset_versions.get(dataset):
            raise RuntimeError(f"Release/current pointer mismatch: {dataset}.")
        output[dataset] = etag
    return output


def prepare_repair(
    repository: LocalDatasetRepository,
    *,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
) -> PreparedRepair:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    if release.completed_session != POLICY_AS_OF:
        raise RuntimeError(
            f"Kraft repair is frozen to {POLICY_AS_OF}; found {release.completed_session}."
        )
    missing = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks datasets: " + ", ".join(missing))
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in REQUIRED_DATASETS
    }
    evidence_content = _verified_evidence(
        repository, frames["source_archive"], evidence_dir
    )
    _preflight(frames)
    actions, action_added = _rewrite_actions(frames["corporate_actions"])
    source_archive, archive_rows_added = _rewrite_source_archive(
        frames["source_archive"]
    )
    expected_value_changes = EXPECTED_FACTOR_VALUE_CHANGES if action_added else 0
    current_factor_source = _adjustment_source_version(
        release.dataset_versions["daily_price_raw"],
        release.dataset_versions["corporate_actions"],
    )
    planned_versions: dict[str, str] = {}
    factors = frames["adjustment_factors"]
    factor_value_changes = 0
    factors_rewritten = False
    if not action_added and archive_rows_added == 0:
        factors, factor_value_changes, factors_rewritten = _rewrite_factors(
            frames["adjustment_factors"],
            frames["daily_price_raw"],
            actions,
            source_version=current_factor_source,
            expected_value_changes=0,
        )
    data_changed = action_added or archive_rows_added > 0 or factors_rewritten
    factor_source_version = current_factor_source
    if data_changed:
        planned_versions = _new_planned_versions(release)
        factor_source_version = _adjustment_source_version(
            release.dataset_versions["daily_price_raw"],
            planned_versions["corporate_actions"],
        )
        factors, factor_value_changes, factors_rewritten = _rewrite_factors(
            frames["adjustment_factors"],
            frames["daily_price_raw"],
            actions,
            source_version=factor_source_version,
            expected_value_changes=expected_value_changes,
        )
        if not factors_rewritten:
            raise RuntimeError(
                "A changed Kraft snapshot must rebind the full factor provenance."
            )
    overrides = {
        "source_archive": source_archive,
        "corporate_actions": actions,
        "adjustment_factors": factors,
    }
    for dataset, frame in overrides.items():
        validate_dataset(
            dataset,
            frame,
            completed_session=release.completed_session,
            incomplete_action_policy="block",
        ).raise_for_errors()
    validate_repository_snapshot(
        _CandidateRepository(repository, release.dataset_versions, overrides)
    ).raise_for_errors()
    factor_multiplier = (
        PRE_DIVIDEND_LAST_CLOSE - SPECIAL_DIVIDEND_USD
    ) / PRE_DIVIDEND_LAST_CLOSE
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=(
            _capture_pointer_etags(repository, release) if data_changed else {}
        ),
        planned_versions=planned_versions,
        frames=overrides if data_changed else {},
        evidence_content=evidence_content,
        summary={
            "status": "validated_offline_plan" if data_changed else "already_applied",
            "base_release_version": release.version,
            "event_id": SPECIAL_DIVIDEND_EVENT_ID,
            "security_id": KRFT_ID,
            "symbol": KRFT_SYMBOL,
            "action_type": "special_dividend",
            "declaration_date": DECLARATION_DATE,
            "record_date": RECORD_DATE,
            "record_time": "immediately prior to merger effective time",
            "payment_date": PAYMENT_DATE,
            "effective_date": EFFECTIVE_DATE,
            "cash_amount_usd_per_krft_share": SPECIAL_DIVIDEND_USD,
            "stock_merger_event_id": STOCK_MERGER_EVENT_ID,
            "stock_merger_ratio": 1.0,
            "economic_sequence": "KRFT special dividend before same-day 1:1 KRFT-to-KHC merger",
            "action_added": action_added,
            "source_archive_rows_added": archive_rows_added,
            "adjustment_factors_rewritten": factors_rewritten,
            "adjustment_factor_value_changes": factor_value_changes,
            "expected_adjustment_factor_value_changes": expected_value_changes,
            "non_krft_adjustment_factor_value_changes": 0,
            "factor_source_version": factor_source_version,
            "source_daily_price_version": release.dataset_versions[
                "daily_price_raw"
            ],
            "source_corporate_actions_version": (
                planned_versions.get("corporate_actions")
                or release.dataset_versions["corporate_actions"]
            ),
            "factor_provenance_rows_rebound": (
                len(factors) if data_changed else 0
            ),
            "planned_versions": dict(planned_versions),
            "pre_dividend_last_close": PRE_DIVIDEND_LAST_CLOSE,
            "special_dividend_to_last_close_ratio": (
                SPECIAL_DIVIDEND_USD / PRE_DIVIDEND_LAST_CLOSE
            ),
            "total_return_factor_multiplier": factor_multiplier,
            "evidence_sha256": [spec.source_hash for spec in EVIDENCE_SPECS],
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
            "apply_requested": False,
        },
    )


@contextmanager
def _exclusive_repository_lock(repository: LocalDatasetRepository):
    path = repository.root / ".locks/market-store-write.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery = repository.root / "recovery/us-kraft-special-dividend"
        if recovery.exists() and tuple(recovery.glob("*.json")):
            raise RuntimeError("Unresolved Kraft recovery marker blocks writes.")
        transactions = repository.root / "transactions/us-kraft-special-dividend"
        if transactions.exists():
            for journal in transactions.glob("*.json"):
                try:
                    status = str(json.loads(journal.read_bytes()).get("status", ""))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        f"Interrupted Kraft transaction blocks writes: {journal}."
                    )
        yield


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(
        path,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )


def _assert_release_unchanged(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> None:
    release, etag = repository.current_release()
    if (
        release is None
        or release.version != prepared.release.version
        or etag != prepared.release_etag
    ):
        raise RuntimeError("Current release changed during Kraft validation.")


def _persist_evidence(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> None:
    for spec in (DECLARATION_EVIDENCE, PAYMENT_EVIDENCE):
        content = prepared.evidence_content[spec.label]
        _verify_content(content, spec)
        path = repository.root / spec.archive_object_path
        if path.is_file():
            try:
                archived = gzip.decompress(path.read_bytes())
            except (OSError, EOFError) as exc:
                raise ValueError(f"Persisted {spec.label} is not valid gzip.") from exc
            if archived != content:
                raise ValueError(f"Persisted {spec.label} bytes conflict.")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(path, gzip.compress(content, mtime=0))
        if gzip.decompress(path.read_bytes()) != content:
            raise RuntimeError(f"Post-write archive verification failed: {spec.label}.")


def _metadata_for_write(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    dataset: str,
) -> dict[str, Any]:
    current = repository.manifest_for_version(
        dataset, prepared.release.dataset_versions[dataset]
    )
    metadata = dict(current.metadata)
    metadata.update(
        {
            "operation": "repair_us_kraft_special_dividend",
            "policy": POLICY_VERSION,
            "kraft_special_dividend_event_id": SPECIAL_DIVIDEND_EVENT_ID,
            "kraft_special_dividend_usd": SPECIAL_DIVIDEND_USD,
            "official_evidence_sha256": [
                spec.source_hash for spec in EVIDENCE_SPECS
            ],
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
        }
    )
    if dataset == "adjustment_factors":
        if set(prepared.planned_versions) != set(WRITE_DATASETS):
            raise RuntimeError("Kraft factor lineage lacks planned dataset versions.")
        source_daily_price_version = prepared.release.dataset_versions[
            "daily_price_raw"
        ]
        source_corporate_actions_version = prepared.planned_versions[
            "corporate_actions"
        ]
        source_version = _adjustment_source_version(
            source_daily_price_version,
            source_corporate_actions_version,
        )
        factors = prepared.frames["adjustment_factors"]
        if (
            set(factors["source_version"].astype(str)) != {source_version}
            or set(factors["source_hash"].astype(str)) != {source_version}
        ):
            raise RuntimeError(
                "Prepared Kraft factors are not bound to the planned action snapshot."
            )
        metadata.update(
            {
                "source_version": source_version,
                "source_daily_price_version": source_daily_price_version,
                "source_corporate_actions_version": source_corporate_actions_version,
            }
        )
    return metadata


def _assert_applied_release_invariant(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> None:
    current, _etag = repository.current_release()
    if current is None or current.to_bytes() != release.to_bytes():
        raise RuntimeError("Committed Kraft repair release is not current.")
    for dataset, version in release.dataset_versions.items():
        pointer, _pointer_etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != version:
            actual = pointer.version if pointer is not None else "missing"
            raise RuntimeError(
                f"Applied release pointer mismatch for {dataset}: "
                f"expected={version}, actual={actual}."
            )

    daily_version = release.dataset_versions.get("daily_price_raw", "")
    action_version = release.dataset_versions.get("corporate_actions", "")
    factor_version = release.dataset_versions.get("adjustment_factors", "")
    expected_source = _adjustment_source_version(daily_version, action_version)
    factor_manifest = repository.manifest_for_version(
        "adjustment_factors", factor_version
    )
    expected_metadata = {
        "source_version": expected_source,
        "source_daily_price_version": daily_version,
        "source_corporate_actions_version": action_version,
    }
    if any(
        _text(factor_manifest.metadata.get(key)) != value
        for key, value in expected_metadata.items()
    ):
        raise RuntimeError(
            "Applied Kraft factor manifest is not bound to the release inputs."
        )

    factors = repository.read_frame("adjustment_factors", factor_version)
    if (
        set(factors["source_version"].astype(str)) != {expected_source}
        or set(factors["source_hash"].astype(str)) != {expected_source}
        or set(factors["source"].astype(str)) != {"derived"}
    ):
        raise RuntimeError(
            "Applied Kraft factor rows are not bound to the release inputs."
        )
    prices = repository.read_frame("daily_price_raw", daily_version)
    actions = repository.read_frame("corporate_actions", action_version)
    _verified, value_changes, provenance_rewrite = _rewrite_factors(
        factors,
        prices,
        actions,
        source_version=expected_source,
        expected_value_changes=0,
    )
    if value_changes or provenance_rewrite:
        raise RuntimeError(
            "Applied Kraft factors do not reproduce the committed snapshot economics."
        )


def _restore_transaction_state(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: Mapping[str, bytes],
    planned_versions: Mapping[str, str],
    committed_release_version: str,
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        current = repository.objects.get("releases/current.json")
        if current.data != old_release_bytes:
            release = DataRelease.from_bytes(current.data)
            belongs = (
                bool(committed_release_version)
                and release.version == committed_release_version
            ) or all(
                release.dataset_versions.get(dataset) == version
                for dataset, version in planned_versions.items()
            )
            if not belongs:
                raise RuntimeError(f"unexpected release during rollback: {release.version}")
            repository.objects.put(
                "releases/current.json", old_release_bytes, if_match=current.etag
            )
    except Exception as exc:
        errors.append(f"releases/current.json: {type(exc).__name__}: {exc}")
    for dataset in reversed(WRITE_DATASETS):
        key = repository.current_key(dataset)
        try:
            current = repository.objects.get(key)
            old = old_pointer_bytes[dataset]
            if current.data != old:
                pointer = CurrentPointer.from_bytes(current.data)
                if pointer.version != planned_versions[dataset]:
                    raise RuntimeError(
                        f"unexpected pointer during rollback: {pointer.version}"
                    )
                repository.objects.put(key, old, if_match=current.etag)
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
    inject_failure: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if prepared.summary["status"] == "already_applied":
        return {**prepared.summary, "mode": "apply", "writes_performed": False}
    inject_failure = inject_failure or (lambda _stage: None)
    with _exclusive_repository_lock(repository):
        _assert_release_unchanged(repository, prepared)
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != prepared.release.dataset_versions[dataset]
                or value.etag != prepared.pointer_etags[dataset]
            ):
                raise RuntimeError(f"{dataset} pointer changed before Kraft apply.")
            old_pointers[dataset] = value.data

        planned = dict(prepared.planned_versions)
        if set(planned) != set(WRITE_DATASETS) or any(
            not value for value in planned.values()
        ):
            raise RuntimeError("Prepared Kraft repair has incomplete planned versions.")
        if len(set(planned.values())) != len(WRITE_DATASETS):
            raise RuntimeError("Prepared Kraft dataset versions are not unique.")
        transaction_id = uuid.uuid4().hex
        journal_path = (
            repository.root
            / "transactions/us-kraft-special-dividend"
            / f"{transaction_id}.json"
        )
        journal: dict[str, Any] = {
            "schema": "us_kraft_special_dividend_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                key: base64.b64encode(value).decode("ascii")
                for key, value in old_pointers.items()
            },
            "planned_versions": planned,
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        try:
            _persist_evidence(repository, prepared)
            versions = dict(prepared.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                frame = prepared.frames[dataset]
                validate_dataset(
                    dataset,
                    frame,
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy="block",
                ).raise_for_errors()
                result = repository.write_frame(
                    dataset,
                    frame,
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy="block",
                    metadata=_metadata_for_write(repository, prepared, dataset),
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"{dataset} write conflicted: {result.conflict_path}."
                    )
                versions[dataset] = result.manifest.version
                inject_failure(f"after_write:{dataset}")
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=prepared.release.quality,
                warnings=prepared.release.warnings,
                expected_etag=prepared.release_etag,
            )
            inject_failure("after_release_commit")
            validate_repository_snapshot(repository).raise_for_errors()
            _assert_applied_release_invariant(repository, committed)
            replay = prepare_repair(repository, evidence_dir=evidence_dir)
            if replay.summary["status"] != "already_applied":
                raise RuntimeError("Kraft post-commit idempotence check failed.")
            journal.update(
                {
                    "status": "committed",
                    "committed_release_version": committed.version,
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            return {
                **prepared.summary,
                "status": "applied",
                "new_release_version": committed.version,
                "transaction_id": transaction_id,
                "quality": committed.quality,
                "warnings": list(committed.warnings),
                "writes_performed": True,
            }
        except BaseException as original:
            errors = _restore_transaction_state(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_versions=planned,
                committed_release_version=committed.version if committed else "",
            )
            journal.update(
                {
                    "status": "rollback_failed" if errors else "rolled_back",
                    "original_error": f"{type(original).__name__}: {original}",
                    "rollback_errors": list(errors),
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            if errors:
                recovery = (
                    repository.root
                    / "recovery/us-kraft-special-dividend"
                    / f"{transaction_id}.json"
                )
                _write_journal(recovery, journal)
                raise RuntimeError(
                    "Kraft rollback was incomplete; recovery marker blocks writes: "
                    f"{recovery}; errors={errors}."
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install Kraft's exact 2015 pre-merger special dividend offline."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repository = LocalDatasetRepository(args.cache_root)
    prepared = prepare_repair(repository, evidence_dir=args.evidence_dir)
    result = (
        apply_repair(repository, prepared, evidence_dir=args.evidence_dir)
        if args.apply
        else {
            **prepared.summary,
            "mode": "plan",
            "writes_performed": False,
        }
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
