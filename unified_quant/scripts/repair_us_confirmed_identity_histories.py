#!/usr/bin/env python3
"""Repair three confirmed US identity-history defects without networking.

The bootstrap release contains three kinds of provider symbol-history reuse:

* KHC contains 126 KRFT bars before KHC first traded on 2015-07-06;
* the post-bankruptcy CHK and active EXE endpoints describe one security under
  two IDs, while the earlier bankrupt CHK security is genuinely distinct; and
* Fiserv's FISV -> FI -> FISV exchange/ticker history is represented by three
  overlapping IDs.

This command is deliberately fail closed.  It accepts only the reviewed row
inventories and hash-pinned SEC/EODHD archives already stored locally.  It has
no network or R2 code.  ``--offline-plan`` validates a candidate snapshot and
does not write.  ``--apply`` uses dataset-pointer and release compare-and-swap,
an exclusive writer lock, a rollback journal, and a post-commit idempotence
check.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import html
import json
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.lifecycle import canonical_lifecycle_event_id
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    sha256_bytes,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.validation import (
    validate_dataset,
    validate_repository_snapshot,
)


DEFAULT_CACHE_ROOT = Path("data/cache")

KHC_ID = "US:EODHD:a060087d-109e-58e9-91f9-b4c398a3d415"
KRFT_ID = "US:EODHD:e3b28684-f48e-582a-8836-b2f5579f2dd2"
KHC_FIRST_SESSION = "2015-07-06"
KRFT_LAST_SESSION = "2015-07-02"
# The merger completed and the new KHC identity legally existed on July 2;
# the SEC filing separately says its first trading session was July 6.  Keeping
# these boundaries distinct lets the July 2 conversion/index records resolve
# without fabricating July 2 KHC prices.
KHC_IDENTITY_START = "2015-07-02"

CHK_DUPLICATE_ID = "US:EODHD:4a2f472a-4d2d-5c8b-8784-b7b3069f3cfe"
CHK_LEGACY_ID = "US:EODHD:54d04976-15c6-5ba9-a2cc-10701a4b5c1f"
EXE_ID = "US:EODHD:97548dea-74f0-55a8-b906-47d5c2a072e1"
NEW_CHK_FIRST_SESSION = "2021-02-10"
CHK_LAST_SESSION = "2024-10-01"
EXE_FIRST_SESSION = "2024-10-02"

FISV_ACTIVE_ID = "US:EODHD:30662d16-c6e4-5187-9721-2b23ac10e4d0"
FI_ID = "US:EODHD:c17adb03-0b5b-5f4b-86e2-d564d6b96d8e"
FISV_OLD_ID = "US:EODHD:f20c2934-d9ae-539d-9dde-a022873d3131"
FISV_FIRST_SESSION = "2015-01-02"
FISV_TO_FI_DATE = "2023-06-07"
FISV_OLD_LAST_SESSION = "2023-06-06"
FI_LAST_SESSION = "2025-11-10"
FI_TO_FISV_DATE = "2025-11-11"

RETIRED_IDS = frozenset({CHK_DUPLICATE_ID, FI_ID, FISV_OLD_ID})
CANONICAL_REMAP = {
    CHK_DUPLICATE_ID: EXE_ID,
    FI_ID: FISV_ACTIVE_ID,
    FISV_OLD_ID: FISV_ACTIVE_ID,
}

OFFICIAL_SOURCE = "official_confirmed_identity_history_repair"
WRITE_DATASETS = (
    "security_master",
    "symbol_history",
    "daily_price_raw",
    "corporate_actions",
    "adjustment_factors",
    "index_constituent_anchors",
    "index_membership_events",
)
REQUIRED_DATASETS = (*WRITE_DATASETS, "source_archive")


@dataclass(frozen=True)
class EvidenceSpec:
    label: str
    source_url: str
    source_hash: str
    exact_bytes: int
    object_suffix: str
    required_text_groups: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class EvidenceArtifact:
    spec: EvidenceSpec
    content: bytes
    retrieved_at: str


KHC_SEC = EvidenceSpec(
    label="khc_trading_boundary",
    source_url=(
        "https://www.sec.gov/Archives/edgar/data/1637459/"
        "000119312515244356/0001193125-15-244356.txt"
    ),
    source_hash="ba6d15da235a1f5f6a186fb7b9e52b7b438c25cf76d4f28d9fac81953b7fc299",
    exact_bytes=1_440_758,
    object_suffix=".txt.gz",
    required_text_groups=(
        ("ceased trading on, and were delisted from, nasdaq",),
        ("ticker symbol \"khc\"", "ticker symbol “khc”"),
        ("will begin trading on july 6, 2015",),
    ),
)
CHK_EMERGENCE_SEC = EvidenceSpec(
    label="new_chk_emergence",
    source_url=(
        "https://www.sec.gov/Archives/edgar/data/895126/"
        "000089512621000033/chk-20210209.htm"
    ),
    source_hash="80f610bb05f197ef740bae2b23c03af96786118ff08dc23a4c78038a577c4842",
    exact_bytes=148_463,
    object_suffix=".html.gz",
    required_text_groups=(
        ("on february 9, 2021 (the \"effective date\"), the plan became effective", "on february 9, 2021 (the “effective date”), the plan became effective"),
        ("new common stock",),
        ("equity interests outstanding prior to the effective date were cancelled",),
    ),
)
EXE_SEC = EvidenceSpec(
    label="chk_exe_ticker_boundary",
    source_url=(
        "https://www.sec.gov/Archives/edgar/data/895126/"
        "000110465924104976/0001104659-24-104976.txt"
    ),
    source_hash="5112367c6043776743c2532071d2d857d77faae96c8317f77af0aa0c8e9259b1",
    exact_bytes=1_043_155,
    object_suffix=".txt.gz",
    required_text_groups=(
        ("changed its nasdaq ticker symbol \"chk\" to \"exe\"", "changed its nasdaq ticker symbol “chk” to “exe”"),
        ("open of trading",),
        ("october 2, 2024",),
    ),
)
FISV_TO_FI_SEC = EvidenceSpec(
    label="fisv_fi_ticker_boundary",
    source_url=(
        "https://www.sec.gov/Archives/edgar/data/798354/"
        "000119312523154199/0001193125-23-154199.txt"
    ),
    source_hash="f3f09f3deb8f242d01652d48e46077d6646052258e75a9f55752c71dd4559863",
    exact_bytes=233_325,
    object_suffix=".txt.gz",
    required_text_groups=(
        ("cease at the close of trading on or about june 6, 2023",),
        ("begin at market open on or about june 7, 2023",),
        ("trade under the symbol \"fi\"", "trade under the symbol “fi”"),
    ),
)
FI_TO_FISV_SEC = EvidenceSpec(
    label="fi_fisv_ticker_boundary",
    source_url=(
        "https://www.sec.gov/Archives/edgar/data/798354/"
        "000119312525254670/0001193125-25-254670.txt"
    ),
    source_hash="d4cd0c2f981bfd0be14d2ebccfc8e852a94177e5fba86abe2c027c5510fc07d3",
    exact_bytes=311_689,
    object_suffix=".txt.gz",
    required_text_groups=(
        ("trading will begin on nasdaq at market open on or about november 11, 2025",),
        ("trade under the symbols, \"fisv\"", "trade under the symbols, “fisv”"),
    ),
)


def _eod_spec(label: str, symbol: str, source_hash: str, exact_bytes: int) -> EvidenceSpec:
    return EvidenceSpec(
        label=label,
        source_url=(
            f"https://eodhd.com/api/eod/{symbol}.US?"
            "from=2015-01-01&to=2026-07-15"
        ),
        source_hash=source_hash,
        exact_bytes=exact_bytes,
        object_suffix=".json.gz",
    )


KHC_EOD = _eod_spec(
    "khc_eod", "KHC",
    "213b36fee89ff78865bc656ce97f5eac6ee40b886ad83c3c4964ef4a680ccd38",
    336_210,
)
KRFT_EOD = _eod_spec(
    "krft_eod", "KRFT",
    "03de7ec01c810004fcd2357010aab3204215c688c4409402f5175c527428c6f2",
    14_299,
)
CHK_EOD = _eod_spec(
    "chk_duplicate_eod", "CHK",
    "74ac2547dad15740f6abbe75c0093593d7b6bb78ddb46e2aec0aefe7f625f0ab",
    251_620,
)
EXE_EOD = _eod_spec(
    "exe_canonical_eod", "EXE",
    "d43e8a1a466d1d0b4fa54f027a326d68008a437d72a6ea2a40b76d41df8685cf",
    158_420,
)
FISV_OLD_EOD = _eod_spec(
    "fisv_old_eod", "FISV_old",
    "a7faf8d40dae03a9d8721a9cfdf60f19da15f7674e8bae4fbae2073c1d8c8480",
    247_876,
)
FI_EOD = _eod_spec(
    "fi_eod", "FI",
    "b8dd9f1f71acaf14d09ddc9fbfb05cb9af3ef3ddc148a25e4cccd762faf65fec",
    320_054,
)
FISV_ACTIVE_EOD = _eod_spec(
    "fisv_active_eod", "FISV",
    "44a20f47a72dfe278ab23fa2354836b7897da3d0882e5e0268988d252c81b98c",
    338_176,
)

EVIDENCE_SPECS = (
    KHC_SEC,
    CHK_EMERGENCE_SEC,
    EXE_SEC,
    FISV_TO_FI_SEC,
    FI_TO_FISV_SEC,
    KHC_EOD,
    KRFT_EOD,
    CHK_EOD,
    EXE_EOD,
    FISV_OLD_EOD,
    FI_EOD,
    FISV_ACTIVE_EOD,
)

EXPECTED_PRICE_ROWS = {
    KHC_ID: 2_899,
    KRFT_ID: 126,
    CHK_DUPLICATE_ID: 2_201,
    CHK_LEGACY_ID: 1_381,
    EXE_ID: 1_362,
    FISV_OLD_ID: 2_122,
    FI_ID: 2_731,
    FISV_ACTIVE_ID: 2_899,
}
EXPECTED_REPAIRED_PRICE_ROWS = {
    KHC_ID: 2_773,
    KRFT_ID: 126,
    CHK_LEGACY_ID: 1_381,
    EXE_ID: 1_362,
    FISV_ACTIVE_ID: 2_899,
}
EXPECTED_KHC_CONTAMINATED_ROWS = 126
EXPECTED_CHK_EXE_OVERLAP_ROWS = 916
EXPECTED_FISV_OLD_OVERLAP_ROWS = 2_122
EXPECTED_FI_OVERLAP_ROWS = 2_731
EXPECTED_CANONICAL_ACTION_ROWS = {
    KHC_ID: 45,
    EXE_ID: 22,
    FISV_ACTIVE_ID: 14,
}
EXPECTED_FISV_TRANSITION_EVENT_ROWS = 4


@dataclass
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: dict[str, str | None]
    frames: dict[str, pd.DataFrame]
    evidence: dict[str, EvidenceArtifact]
    summary: dict[str, Any]


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


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
    if not _text(value):
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid confirmed-identity date: {value!r}")
    return pd.Timestamp(parsed).date().isoformat()


def _normalized_document_text(content: bytes) -> str:
    decoded = html.unescape(content.decode("utf-8", errors="replace"))
    decoded = re.sub(r"<[^>]+>", " ", decoded)
    return re.sub(r"\s+", " ", decoded).strip().casefold()


def _one_row(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"Expected one {label}; observed={len(rows)}")
    return rows.iloc[0]


def _session_text(frame: pd.DataFrame) -> pd.Series:
    values = pd.to_datetime(frame["session"], errors="coerce")
    if values.isna().any():
        raise ValueError("Confirmed identity repair found invalid price sessions.")
    return values.dt.date.astype(str)


def _load_evidence(
    cache_root: Path,
    source_archive: pd.DataFrame,
) -> dict[str, EvidenceArtifact]:
    output: dict[str, EvidenceArtifact] = {}
    for spec in EVIDENCE_SPECS:
        rows = source_archive.loc[
            source_archive["source_url"].astype(str).eq(spec.source_url)
            & source_archive["source_hash"].astype(str).eq(spec.source_hash)
        ]
        if len(rows) != 1:
            raise ValueError(
                f"{spec.label} exact source_archive URL/hash binding is missing."
            )
        row = rows.iloc[0]
        object_path = _text(row.get("object_path"))
        expected_suffix = f"/{spec.source_hash}{spec.object_suffix}"
        if not (
            _text(row.get("archive_id")) == spec.source_hash
            and object_path.endswith(expected_suffix)
            and _text(row.get("retrieved_at"))
        ):
            raise ValueError(f"{spec.label} archive metadata changed.")
        path = (cache_root / object_path).resolve()
        archive_root = (cache_root / "archives").resolve()
        if archive_root not in path.parents or not path.is_file():
            raise ValueError(f"{spec.label} persisted archive is missing or unsafe.")
        try:
            content = gzip.decompress(path.read_bytes())
        except Exception as exc:
            raise ValueError(f"{spec.label} persisted archive is unreadable.") from exc
        if (
            len(content) != spec.exact_bytes
            or hashlib.sha256(content).hexdigest() != spec.source_hash
        ):
            raise ValueError(f"{spec.label} persisted bytes changed.")
        if spec.required_text_groups:
            normalized = _normalized_document_text(content)
            missing = [
                group
                for group in spec.required_text_groups
                if not any(phrase.casefold() in normalized for phrase in group)
            ]
            if missing:
                raise ValueError(
                    f"{spec.label} no longer proves reviewed claims: {missing!r}"
                )
        output[spec.label] = EvidenceArtifact(
            spec=spec,
            content=content,
            retrieved_at=_text(row["retrieved_at"]),
        )
    return output


def _assert_inventory(frame: pd.DataFrame, expected: Mapping[str, int], label: str) -> None:
    observed = frame["security_id"].astype(str).value_counts()
    changed = {
        security_id: (count, int(observed.get(security_id, 0)))
        for security_id, count in expected.items()
        if int(observed.get(security_id, 0)) != count
    }
    if changed:
        raise ValueError(f"{label} reviewed inventory changed: {changed}")


def _assert_close_overlap(
    prices: pd.DataFrame,
    left_id: str,
    right_id: str,
    *,
    expected_rows: int,
    max_close_difference: float = 0.0,
) -> tuple[str, str]:
    left = prices.loc[
        prices["security_id"].astype(str).eq(left_id), ["session", "close"]
    ].copy()
    right = prices.loc[
        prices["security_id"].astype(str).eq(right_id), ["session", "close"]
    ].copy()
    joined = left.merge(right, on="session", how="inner", suffixes=("_left", "_right"))
    if len(joined) != expected_rows:
        raise ValueError(
            f"Reviewed identity overlap changed for {left_id}/{right_id}: "
            f"expected={expected_rows}, observed={len(joined)}"
        )
    difference = (
        pd.to_numeric(joined["close_left"], errors="coerce")
        - pd.to_numeric(joined["close_right"], errors="coerce")
    ).abs()
    if difference.isna().any() or float(difference.max()) > max_close_difference:
        raise ValueError(f"Close overlap changed for {left_id}/{right_id}.")
    sessions = pd.to_datetime(joined["session"], errors="raise").dt.date.astype(str)
    return sessions.min(), sessions.max()


def _preflight(
    frames: Mapping[str, pd.DataFrame],
    evidence: Mapping[str, EvidenceArtifact],
) -> dict[str, Any]:
    master = frames["security_master"]
    history = frames["symbol_history"]
    prices = frames["daily_price_raw"]
    actions = frames["corporate_actions"]
    factors = frames["adjustment_factors"]

    reviewed_ids = set(EXPECTED_PRICE_ROWS)
    for security_id in reviewed_ids:
        _one_row(
            master,
            master["security_id"].astype(str).eq(security_id),
            f"reviewed security_master {security_id}",
        )
    if set(master.loc[master["security_id"].isin({CHK_LEGACY_ID, CHK_DUPLICATE_ID}), "primary_symbol"].astype(str)) != {"CHK"}:
        raise ValueError("Legacy and post-emergence CHK identities are not both explicit.")
    _assert_inventory(prices, EXPECTED_PRICE_ROWS, "price")
    _assert_inventory(factors, EXPECTED_PRICE_ROWS, "factor")

    sessions = _session_text(prices)
    khc = prices["security_id"].astype(str).eq(KHC_ID)
    krft = prices["security_id"].astype(str).eq(KRFT_ID)
    khc_contamination = prices.loc[khc & sessions.lt(KHC_FIRST_SESSION)].copy()
    krft_prices = prices.loc[krft].copy()
    if len(khc_contamination) != EXPECTED_KHC_CONTAMINATED_ROWS:
        raise ValueError(
            "KHC pre-trading contamination is not exactly "
            f"{EXPECTED_KHC_CONTAMINATED_ROWS} rows."
        )
    comparison = khc_contamination.merge(
        krft_prices,
        on="session",
        how="outer",
        suffixes=("_khc", "_krft"),
        indicator=True,
    )
    if (
        not comparison["_merge"].eq("both").all()
        or len(comparison) != EXPECTED_KHC_CONTAMINATED_ROWS
    ):
        raise ValueError("KHC contamination sessions no longer equal KRFT sessions.")
    for column, tolerance in (
        ("open", 0.0),
        ("high", 0.0005),
        ("low", 0.0005),
        ("close", 0.0),
    ):
        difference = (
            pd.to_numeric(comparison[f"{column}_khc"], errors="coerce")
            - pd.to_numeric(comparison[f"{column}_krft"], errors="coerce")
        ).abs()
        if difference.isna().any() or float(difference.max()) > tolerance:
            raise ValueError(f"KHC/KRFT reviewed {column} overlap changed.")
    volume_left = pd.to_numeric(comparison["volume_khc"], errors="coerce")
    volume_right = pd.to_numeric(comparison["volume_krft"], errors="coerce")
    volume_denominator = pd.concat(
        [volume_left.abs(), volume_right.abs()], axis=1
    ).max(axis=1).replace(0, 1)
    if ((volume_left - volume_right).abs() / volume_denominator).max() > 0.007:
        raise ValueError("KHC/KRFT reviewed volume overlap changed.")
    khc_retained_sessions = sessions.loc[khc & sessions.ge(KHC_FIRST_SESSION)]
    if khc_retained_sessions.min() != KHC_FIRST_SESSION:
        raise ValueError("KHC true first trading session changed.")
    khc_early_actions = actions.loc[
        actions["security_id"].astype(str).eq(KHC_ID)
        & pd.to_datetime(actions["effective_date"], errors="coerce").dt.date.astype(str).lt(KHC_FIRST_SESSION)
    ]
    if not (
        len(khc_early_actions) == 1
        and _text(khc_early_actions.iloc[0]["action_type"]) == "cash_dividend"
        and _date(khc_early_actions.iloc[0]["effective_date"]) == "2015-04-08"
        and float(khc_early_actions.iloc[0]["cash_amount"]) == 0.55
    ):
        raise ValueError("KHC copied pre-trading dividend inventory changed.")

    chk_overlap = _assert_close_overlap(
        prices,
        CHK_DUPLICATE_ID,
        EXE_ID,
        expected_rows=EXPECTED_CHK_EXE_OVERLAP_ROWS,
    )
    if chk_overlap != (NEW_CHK_FIRST_SESSION, CHK_LAST_SESSION):
        raise ValueError("CHK/EXE reviewed overlap boundary changed.")
    exe_sessions = sessions.loc[prices["security_id"].astype(str).eq(EXE_ID)]
    if (exe_sessions.min(), exe_sessions.max()) != (
        NEW_CHK_FIRST_SESSION,
        "2026-07-15",
    ):
        raise ValueError("Canonical EXE provider history boundary changed.")

    fisv_old_overlap = _assert_close_overlap(
        prices,
        FISV_OLD_ID,
        FISV_ACTIVE_ID,
        expected_rows=EXPECTED_FISV_OLD_OVERLAP_ROWS,
        max_close_difference=0.01,
    )
    fi_overlap = _assert_close_overlap(
        prices,
        FI_ID,
        FISV_ACTIVE_ID,
        expected_rows=EXPECTED_FI_OVERLAP_ROWS,
        max_close_difference=0.01,
    )
    if fisv_old_overlap != (FISV_FIRST_SESSION, FISV_TO_FI_DATE):
        raise ValueError("FISV-old/canonical overlap boundary changed.")
    if fi_overlap != (FISV_FIRST_SESSION, FI_LAST_SESSION):
        raise ValueError("FI/canonical-FISV overlap boundary changed.")

    expected_ticker = {
        (CHK_DUPLICATE_ID, EXE_FIRST_SESSION, EXE_ID, "EXE", EXE_SEC.source_hash),
        (FISV_OLD_ID, FISV_TO_FI_DATE, FI_ID, "FI", FISV_TO_FI_SEC.source_hash),
        (FI_ID, FI_TO_FISV_DATE, FISV_ACTIVE_ID, "FISV", FI_TO_FISV_SEC.source_hash),
    }
    observed_ticker = {
        (
            _text(row.security_id),
            _date(row.effective_date),
            _text(row.new_security_id),
            _text(row.new_symbol).upper(),
            _text(row.source_hash),
        )
        for row in actions.loc[
            actions["action_type"].astype(str).eq("ticker_change")
            & actions["security_id"].astype(str).isin(RETIRED_IDS)
        ].itertuples(index=False)
    }
    if observed_ticker != expected_ticker:
        raise ValueError("Reviewed CHK/EXE or FISV/FI ticker actions changed.")

    # Exact reference inventory prevents silently dropping unrelated history.
    anchors = frames["index_constituent_anchors"]
    events = frames["index_membership_events"]
    retired_anchors = anchors.loc[anchors["security_id"].astype(str).isin(RETIRED_IDS)]
    if not (
        len(retired_anchors) == 1
        and _text(retired_anchors.iloc[0]["security_id"]) == FISV_OLD_ID
        and _text(retired_anchors.iloc[0]["index_id"]) == "sp500"
        and _date(retired_anchors.iloc[0]["anchor_date"]) == "2015-01-07"
    ):
        raise ValueError("Retired identity anchor inventory changed.")
    lineage_ids = {FISV_OLD_ID, FI_ID, FISV_ACTIVE_ID}
    transition_events = events.loc[
        events["security_id"].astype(str).isin(lineage_ids)
        & events["index_id"].astype(str).eq("sp500")
        & events["effective_date"].astype(str).isin(
            {FISV_TO_FI_DATE, FI_TO_FISV_DATE}
        )
    ]
    signature = {
        (
            _text(row.security_id),
            _date(row.effective_date),
            _text(row.operation).upper(),
        )
        for row in transition_events.itertuples(index=False)
    }
    if signature != {
        (FISV_OLD_ID, FISV_TO_FI_DATE, "REMOVE"),
        (FI_ID, FISV_TO_FI_DATE, "ADD"),
        (FI_ID, FI_TO_FISV_DATE, "REMOVE"),
        (FISV_ACTIVE_ID, FI_TO_FISV_DATE, "ADD"),
    }:
        raise ValueError("Fiserv S&P transition-event inventory changed.")

    return {
        "evidence_artifact_count": len(evidence),
        "khc_contaminated_price_rows": len(khc_contamination),
        "khc_copied_action_rows": len(khc_early_actions),
        "chk_exe_overlap_rows": EXPECTED_CHK_EXE_OVERLAP_ROWS,
        "fisv_old_overlap_rows": EXPECTED_FISV_OLD_OVERLAP_ROWS,
        "fi_overlap_rows": EXPECTED_FI_OVERLAP_ROWS,
        "retired_identity_count": len(RETIRED_IDS),
    }


def _set_provenance(
    frame: pd.DataFrame,
    mask: pd.Series,
    artifact: EvidenceArtifact,
) -> None:
    frame.loc[mask, "source"] = OFFICIAL_SOURCE
    frame.loc[mask, "source_url"] = artifact.spec.source_url
    frame.loc[mask, "source_hash"] = artifact.spec.source_hash
    frame.loc[mask, "retrieved_at"] = artifact.retrieved_at


def _history_row(
    template: pd.Series,
    *,
    security_id: str,
    symbol: str,
    exchange: str,
    start: str,
    end: str,
    evidence: EvidenceArtifact,
) -> pd.Series:
    row = template.copy()
    row["security_id"] = security_id
    row["symbol"] = symbol
    row["exchange"] = exchange
    row["effective_from"] = start
    row["effective_to"] = end
    row["source"] = OFFICIAL_SOURCE
    row["source_url"] = evidence.spec.source_url
    row["source_hash"] = evidence.spec.source_hash
    row["retrieved_at"] = evidence.retrieved_at
    return row


def _rewrite_master_history(
    frames: Mapping[str, pd.DataFrame],
    evidence: Mapping[str, EvidenceArtifact],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = frames["security_master"].copy()
    history = frames["symbol_history"].copy()
    templates = {
        security_id: _one_row(
            history,
            history["security_id"].astype(str).eq(security_id),
            f"symbol-history template {security_id}",
        ).copy()
        for security_id in {
            KHC_ID,
            KRFT_ID,
            CHK_DUPLICATE_ID,
            EXE_ID,
            FISV_ACTIVE_ID,
            FI_ID,
            FISV_OLD_ID,
        }
    }
    master = master.loc[~master["security_id"].astype(str).isin(RETIRED_IDS)].copy()

    khc_evidence = evidence[KHC_SEC.label]
    khc = master["security_id"].astype(str).eq(KHC_ID)
    krft = master["security_id"].astype(str).eq(KRFT_ID)
    exe = master["security_id"].astype(str).eq(EXE_ID)
    fisv = master["security_id"].astype(str).eq(FISV_ACTIVE_ID)
    if tuple(map(int, (khc.sum(), krft.sum(), exe.sum(), fisv.sum()))) != (1, 1, 1, 1):
        raise ValueError("Canonical identity master rows disappeared during rewrite.")
    master.loc[khc, "active_from"] = KHC_FIRST_SESSION
    master.loc[khc, "active_to"] = ""
    _set_provenance(master, khc, khc_evidence)
    master.loc[krft, "active_to"] = KRFT_LAST_SESSION
    _set_provenance(master, krft, khc_evidence)

    master.loc[exe, "active_from"] = NEW_CHK_FIRST_SESSION
    master.loc[exe, "active_to"] = ""
    master.loc[exe, "primary_symbol"] = "EXE"
    master.loc[exe, "name"] = "Expand Energy Corporation"
    for column in ("provider_symbol", "action_provider_symbol"):
        if column in master:
            master.loc[exe, column] = "EXE.US"
    _set_provenance(master, exe, evidence[EXE_SEC.label])

    master.loc[fisv, "active_from"] = FISV_FIRST_SESSION
    master.loc[fisv, "active_to"] = ""
    master.loc[fisv, "primary_symbol"] = "FISV"
    master.loc[fisv, "name"] = "Fiserv, Inc."
    for column in ("provider_symbol", "action_provider_symbol"):
        if column in master:
            master.loc[fisv, column] = "FISV.US"
    _set_provenance(master, fisv, evidence[FI_TO_FISV_SEC.label])

    affected = {
        KHC_ID,
        KRFT_ID,
        CHK_DUPLICATE_ID,
        EXE_ID,
        FISV_ACTIVE_ID,
        FI_ID,
        FISV_OLD_ID,
    }
    history = history.loc[~history["security_id"].astype(str).isin(affected)].copy()
    rows = [
        _history_row(
            templates[KRFT_ID], security_id=KRFT_ID, symbol="KRFT",
            exchange="NASDAQ", start="2015-01-01", end=KRFT_LAST_SESSION,
            evidence=khc_evidence,
        ),
        _history_row(
            templates[KHC_ID], security_id=KHC_ID, symbol="KHC",
            exchange="NASDAQ", start=KHC_IDENTITY_START, end="",
            evidence=khc_evidence,
        ),
        _history_row(
            templates[CHK_DUPLICATE_ID], security_id=EXE_ID, symbol="CHK",
            exchange="NASDAQ", start=NEW_CHK_FIRST_SESSION, end=CHK_LAST_SESSION,
            evidence=evidence[EXE_SEC.label],
        ),
        _history_row(
            templates[EXE_ID], security_id=EXE_ID, symbol="EXE",
            exchange="NASDAQ", start=EXE_FIRST_SESSION, end="",
            evidence=evidence[EXE_SEC.label],
        ),
        _history_row(
            templates[FISV_OLD_ID], security_id=FISV_ACTIVE_ID, symbol="FISV",
            exchange="NASDAQ", start="2015-01-01", end=FISV_OLD_LAST_SESSION,
            evidence=evidence[FISV_TO_FI_SEC.label],
        ),
        _history_row(
            templates[FI_ID], security_id=FISV_ACTIVE_ID, symbol="FI",
            exchange="NYSE", start=FISV_TO_FI_DATE, end=FI_LAST_SESSION,
            evidence=evidence[FISV_TO_FI_SEC.label],
        ),
        _history_row(
            templates[FISV_ACTIVE_ID], security_id=FISV_ACTIVE_ID, symbol="FISV",
            exchange="NASDAQ", start=FI_TO_FISV_DATE, end="",
            evidence=evidence[FI_TO_FISV_SEC.label],
        ),
    ]
    additions = pd.DataFrame(rows).loc[:, history.columns]
    history = pd.concat([history, additions], ignore_index=True, sort=False)
    history = history.drop_duplicates(
        list(dataset_spec("symbol_history").primary_key), keep="last"
    )
    return master.reset_index(drop=True), history.reset_index(drop=True)


def _canonical_ticker_action(
    original: pd.Series,
    *,
    canonical_id: str,
    new_symbol: str,
) -> pd.Series:
    row = original.copy()
    effective = _date(row["effective_date"])
    row["event_id"] = canonical_lifecycle_event_id(
        canonical_id, "ticker_change", effective
    )
    row["security_id"] = canonical_id
    row["new_security_id"] = canonical_id
    row["new_symbol"] = new_symbol
    return row


def _rewrite_prices_actions_factors(
    frames: Mapping[str, pd.DataFrame],
    *,
    source_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = frames["daily_price_raw"].copy()
    sessions = _session_text(prices)
    remove_prices = prices["security_id"].astype(str).isin(RETIRED_IDS) | (
        prices["security_id"].astype(str).eq(KHC_ID)
        & sessions.lt(KHC_FIRST_SESSION)
    )
    prices = prices.loc[~remove_prices].copy()
    prices = prices.drop_duplicates(
        list(dataset_spec("daily_price_raw").primary_key), keep="last"
    )

    original_actions = frames["corporate_actions"]
    originals: dict[tuple[str, str], pd.Series] = {}
    for security_id, effective in (
        (CHK_DUPLICATE_ID, EXE_FIRST_SESSION),
        (FISV_OLD_ID, FISV_TO_FI_DATE),
        (FI_ID, FI_TO_FISV_DATE),
    ):
        originals[(security_id, effective)] = _one_row(
            original_actions,
            original_actions["security_id"].astype(str).eq(security_id)
            & original_actions["action_type"].astype(str).eq("ticker_change")
            & original_actions["effective_date"].astype(str).eq(effective),
            f"reviewed ticker action {security_id}/{effective}",
        ).copy()
    actions = original_actions.loc[
        ~original_actions["security_id"].astype(str).isin(RETIRED_IDS)
    ].copy()
    action_dates = pd.to_datetime(actions["effective_date"], errors="coerce")
    actions = actions.loc[
        ~(
            actions["security_id"].astype(str).eq(KHC_ID)
            & action_dates.dt.date.astype(str).lt(KHC_FIRST_SESSION)
        )
    ].copy()
    # Southwestern holders received CHK immediately before the same legal
    # issuer began trading as EXE.  Rebind only the successor ID; CHK remains
    # the exact consideration-date symbol and all source provenance is kept.
    successor = actions["new_security_id"].astype(str).eq(CHK_DUPLICATE_ID)
    actions.loc[successor, "new_security_id"] = EXE_ID
    additions = pd.DataFrame(
        [
            _canonical_ticker_action(
                originals[(CHK_DUPLICATE_ID, EXE_FIRST_SESSION)],
                canonical_id=EXE_ID,
                new_symbol="EXE",
            ),
            _canonical_ticker_action(
                originals[(FISV_OLD_ID, FISV_TO_FI_DATE)],
                canonical_id=FISV_ACTIVE_ID,
                new_symbol="FI",
            ),
            _canonical_ticker_action(
                originals[(FI_ID, FI_TO_FISV_DATE)],
                canonical_id=FISV_ACTIVE_ID,
                new_symbol="FISV",
            ),
        ]
    ).loc[:, actions.columns]
    actions = pd.concat([actions, additions], ignore_index=True, sort=False)
    actions = actions.drop_duplicates(
        list(dataset_spec("corporate_actions").primary_key), keep="last"
    )

    rebuild_ids = {KHC_ID, EXE_ID, FISV_ACTIVE_ID}
    factors = frames["adjustment_factors"].loc[
        ~frames["adjustment_factors"]["security_id"].astype(str).isin(
            RETIRED_IDS | rebuild_ids
        )
    ].copy()
    rebuilt = build_adjustment_factors(
        prices.loc[prices["security_id"].astype(str).isin(rebuild_ids)].copy(),
        actions.loc[actions["security_id"].astype(str).isin(rebuild_ids)].copy(),
        source_version=source_version,
    )
    factors = pd.concat([factors, rebuilt], ignore_index=True, sort=False)
    factors = factors.drop_duplicates(
        list(dataset_spec("adjustment_factors").primary_key), keep="last"
    )
    return (
        prices.sort_values(["security_id", "session"]).reset_index(drop=True),
        actions.sort_values(
            ["security_id", "effective_date", "event_id"]
        ).reset_index(drop=True),
        factors.sort_values(["security_id", "session"]).reset_index(drop=True),
    )


def _rewrite_index_references(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    anchors = frames["index_constituent_anchors"].copy()
    events = frames["index_membership_events"].copy()
    retired_anchor = anchors["security_id"].astype(str).eq(FISV_OLD_ID)
    anchors.loc[retired_anchor, "security_id"] = FISV_ACTIVE_ID
    if anchors["security_id"].astype(str).isin(RETIRED_IDS).any():
        raise ValueError("Unexpected retired identity anchor remains after rekey.")

    lineage_ids = {FISV_OLD_ID, FI_ID, FISV_ACTIVE_ID}
    collapse = (
        events["security_id"].astype(str).isin(lineage_ids)
        & events["index_id"].astype(str).eq("sp500")
        & events["effective_date"].astype(str).isin(
            {FISV_TO_FI_DATE, FI_TO_FISV_DATE}
        )
    )
    if int(collapse.sum()) != EXPECTED_FISV_TRANSITION_EVENT_ROWS:
        raise ValueError("Fiserv transition pair inventory changed before collapse.")
    events = events.loc[~collapse].copy()
    if events["security_id"].astype(str).isin(RETIRED_IDS).any():
        raise ValueError("Unexpected retired identity membership event remains.")
    for frame, dataset in (
        (anchors, "index_constituent_anchors"),
        (events, "index_membership_events"),
    ):
        if frame.duplicated(list(dataset_spec(dataset).primary_key), keep=False).any():
            raise ValueError(f"{dataset} identity rewrite created a key collision.")
    return anchors.reset_index(drop=True), events.reset_index(drop=True), {
        "index_anchor_rows_rekeyed": int(retired_anchor.sum()),
        "same_security_transition_events_collapsed": int(collapse.sum()),
    }


def _history_signature(
    history: pd.DataFrame,
    security_id: str,
) -> set[tuple[str, str, str, str]]:
    rows = history.loc[history["security_id"].astype(str).eq(security_id)]
    return {
        (
            _text(row.symbol).upper(),
            _text(row.exchange).upper(),
            _date(row.effective_from),
            _date(row.effective_to),
        )
        for row in rows.itertuples(index=False)
    }


def _validate_ticker_action(
    actions: pd.DataFrame,
    *,
    security_id: str,
    effective_date: str,
    new_symbol: str,
    source_hash: str,
) -> None:
    event_id = canonical_lifecycle_event_id(
        security_id, "ticker_change", effective_date
    )
    row = _one_row(
        actions,
        actions["event_id"].astype(str).eq(event_id),
        f"canonical ticker action {security_id}/{effective_date}",
    )
    if not (
        _text(row["security_id"]) == security_id
        and _text(row["action_type"]) == "ticker_change"
        and _date(row["effective_date"]) == effective_date
        and _text(row["new_security_id"]) == security_id
        and _text(row["new_symbol"]).upper() == new_symbol
        and bool(row["official"])
        and _text(row["source_hash"]) == source_hash
    ):
        raise ValueError(f"Canonical ticker action changed: {security_id}/{effective_date}")


def validate_repaired_frames(
    frames: Mapping[str, pd.DataFrame],
    evidence: Mapping[str, EvidenceArtifact],
    *,
    completed_session: str,
) -> dict[str, Any]:
    missing = sorted(set(REQUIRED_DATASETS) - set(frames))
    if missing:
        raise ValueError("Confirmed identity validation lacks: " + ", ".join(missing))
    for dataset in WRITE_DATASETS:
        validate_dataset(
            dataset,
            frames[dataset],
            incomplete_action_policy="warn",
            completed_session=completed_session,
        ).raise_for_errors()

    for dataset in (
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "corporate_actions",
        "adjustment_factors",
        "index_constituent_anchors",
        "index_membership_events",
    ):
        if frames[dataset]["security_id"].astype(str).isin(RETIRED_IDS).any():
            raise ValueError(f"Retired identity remains in {dataset}.")
    if frames["corporate_actions"]["new_security_id"].astype(str).isin(RETIRED_IDS).any():
        raise ValueError("Corporate action still references a retired successor ID.")

    master = frames["security_master"]
    expected_master = {
        KHC_ID: ("KHC", KHC_FIRST_SESSION, ""),
        KRFT_ID: ("KRFT", "2015-01-02", KRFT_LAST_SESSION),
        EXE_ID: ("EXE", NEW_CHK_FIRST_SESSION, ""),
        FISV_ACTIVE_ID: ("FISV", FISV_FIRST_SESSION, ""),
        CHK_LEGACY_ID: ("CHK", "2015-01-02", "2020-06-26"),
    }
    for security_id, (symbol, start, end) in expected_master.items():
        row = _one_row(
            master,
            master["security_id"].astype(str).eq(security_id),
            f"repaired master {security_id}",
        )
        if (
            _text(row["primary_symbol"]).upper(),
            _date(row["active_from"]),
            _date(row["active_to"]),
        ) != (symbol, start, end):
            raise ValueError(f"Repaired master boundary changed: {security_id}")

    history = frames["symbol_history"]
    if _history_signature(history, KRFT_ID) != {
        ("KRFT", "NASDAQ", "2015-01-01", KRFT_LAST_SESSION)
    }:
        raise ValueError("KRFT history was not preserved exactly.")
    if _history_signature(history, KHC_ID) != {
        ("KHC", "NASDAQ", KHC_IDENTITY_START, "")
    }:
        raise ValueError("KHC legal/trading identity boundary is not exact.")
    if _history_signature(history, EXE_ID) != {
        ("CHK", "NASDAQ", NEW_CHK_FIRST_SESSION, CHK_LAST_SESSION),
        ("EXE", "NASDAQ", EXE_FIRST_SESSION, ""),
    }:
        raise ValueError("CHK/EXE canonical history is not exact.")
    if _history_signature(history, FISV_ACTIVE_ID) != {
        ("FISV", "NASDAQ", "2015-01-01", FISV_OLD_LAST_SESSION),
        ("FI", "NYSE", FISV_TO_FI_DATE, FI_LAST_SESSION),
        ("FISV", "NASDAQ", FI_TO_FISV_DATE, ""),
    }:
        raise ValueError("FISV/FI/FISV canonical history is not exact.")

    prices = frames["daily_price_raw"]
    factors = frames["adjustment_factors"]
    _assert_inventory(prices, EXPECTED_REPAIRED_PRICE_ROWS, "repaired price")
    _assert_inventory(factors, EXPECTED_REPAIRED_PRICE_ROWS, "repaired factor")
    sessions = _session_text(prices)
    expected_boundaries = {
        KHC_ID: (KHC_FIRST_SESSION, "2026-07-15", KHC_EOD.source_hash),
        KRFT_ID: ("2015-01-02", KRFT_LAST_SESSION, KRFT_EOD.source_hash),
        EXE_ID: (NEW_CHK_FIRST_SESSION, "2026-07-15", EXE_EOD.source_hash),
        FISV_ACTIVE_ID: (FISV_FIRST_SESSION, "2026-07-15", FISV_ACTIVE_EOD.source_hash),
    }
    for security_id, (start, end, source_hash) in expected_boundaries.items():
        mask = prices["security_id"].astype(str).eq(security_id)
        if (sessions.loc[mask].min(), sessions.loc[mask].max()) != (start, end):
            raise ValueError(f"Canonical price boundary changed: {security_id}")
        if set(prices.loc[mask, "source_hash"].astype(str)) != {source_hash}:
            raise ValueError(f"Canonical price source basis changed: {security_id}")
        price_sessions = set(sessions.loc[mask])
        factor_sessions = set(
            _session_text(factors.loc[factors["security_id"].astype(str).eq(security_id)])
        )
        if price_sessions != factor_sessions:
            raise ValueError(f"Factor coverage differs from prices: {security_id}")

    actions = frames["corporate_actions"]
    action_counts = actions["security_id"].astype(str).value_counts()
    for security_id, expected in EXPECTED_CANONICAL_ACTION_ROWS.items():
        if int(action_counts.get(security_id, 0)) != expected:
            raise ValueError(f"Canonical action inventory changed: {security_id}")
    if (
        actions["security_id"].astype(str).eq(KHC_ID)
        & actions["effective_date"].astype(str).lt(KHC_FIRST_SESSION)
    ).any():
        raise ValueError("KHC still contains pre-trading actions.")
    _validate_ticker_action(
        actions,
        security_id=EXE_ID,
        effective_date=EXE_FIRST_SESSION,
        new_symbol="EXE",
        source_hash=EXE_SEC.source_hash,
    )
    _validate_ticker_action(
        actions,
        security_id=FISV_ACTIVE_ID,
        effective_date=FISV_TO_FI_DATE,
        new_symbol="FI",
        source_hash=FISV_TO_FI_SEC.source_hash,
    )
    _validate_ticker_action(
        actions,
        security_id=FISV_ACTIVE_ID,
        effective_date=FI_TO_FISV_DATE,
        new_symbol="FISV",
        source_hash=FI_TO_FISV_SEC.source_hash,
    )
    swn = actions.loc[
        actions["security_id"].astype(str).eq(
            "US:EODHD:e38dbe48-7597-54e3-b3f5-4dcc84b7a7f2"
        )
        & actions["action_type"].astype(str).eq("stock_merger")
        & actions["effective_date"].astype(str).eq("2024-10-01")
    ]
    if len(swn) != 1 or not (
        _text(swn.iloc[0]["new_security_id"]) == EXE_ID
        and _text(swn.iloc[0]["new_symbol"]).upper() == "CHK"
        and float(swn.iloc[0]["ratio"]) == 0.0867
    ):
        raise ValueError("SWN consideration-date successor binding changed.")

    anchors = frames["index_constituent_anchors"]
    events = frames["index_membership_events"]
    fisv_sp500_anchor = anchors.loc[
        anchors["security_id"].astype(str).eq(FISV_ACTIVE_ID)
        & anchors["index_id"].astype(str).eq("sp500")
        & anchors["anchor_date"].astype(str).eq("2015-01-07")
    ]
    if len(fisv_sp500_anchor) != 1:
        raise ValueError("Fiserv S&P anchor was not rekeyed exactly once.")
    if (
        events["security_id"].astype(str).eq(FISV_ACTIVE_ID)
        & events["index_id"].astype(str).eq("sp500")
        & events["effective_date"].astype(str).isin(
            {FISV_TO_FI_DATE, FI_TO_FISV_DATE}
        )
    ).any():
        raise ValueError("Same-security Fiserv S&P transition events remain.")
    nasdaq_remove = events.loc[
        events["security_id"].astype(str).eq(FISV_ACTIVE_ID)
        & events["index_id"].astype(str).eq("nasdaq100")
        & events["effective_date"].astype(str).eq(FISV_TO_FI_DATE)
        & events["operation"].astype(str).str.upper().eq("REMOVE")
    ]
    if len(nasdaq_remove) != 1:
        raise ValueError("Fiserv's real Nasdaq-100 removal was not preserved.")

    archive = frames["source_archive"]
    for artifact in evidence.values():
        spec = artifact.spec
        rows = archive.loc[
            archive["source_url"].astype(str).eq(spec.source_url)
            & archive["source_hash"].astype(str).eq(spec.source_hash)
        ]
        if len(rows) != 1:
            raise ValueError(f"Evidence binding disappeared: {spec.label}")
    return {
        "status": "validated_offline_plan",
        "canonical_ids": [KHC_ID, EXE_ID, FISV_ACTIVE_ID],
        "retired_ids": sorted(RETIRED_IDS),
        "khc_price_rows": EXPECTED_REPAIRED_PRICE_ROWS[KHC_ID],
        "exe_price_rows": EXPECTED_REPAIRED_PRICE_ROWS[EXE_ID],
        "fisv_price_rows": EXPECTED_REPAIRED_PRICE_ROWS[FISV_ACTIVE_ID],
        "official_ticker_change_rows": 3,
        "network_accessed": False,
        "eodhd_http_attempts": 0,
        "r2_accessed": False,
        "publication_ready": False,
    }


def prepare_repair_frames(
    frames: Mapping[str, pd.DataFrame],
    evidence: Mapping[str, EvidenceArtifact],
    *,
    completed_session: str,
    source_version: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    missing = sorted(set(REQUIRED_DATASETS) - set(frames))
    if missing:
        raise ValueError("Confirmed identity repair lacks: " + ", ".join(missing))
    preflight = _preflight(frames, evidence)
    master, history = _rewrite_master_history(frames, evidence)
    prices, actions, factors = _rewrite_prices_actions_factors(
        frames, source_version=source_version
    )
    anchors, events, index_summary = _rewrite_index_references(frames)
    rewritten = {
        "security_master": master,
        "symbol_history": history,
        "daily_price_raw": prices,
        "corporate_actions": actions,
        "adjustment_factors": factors,
        "index_constituent_anchors": anchors,
        "index_membership_events": events,
        "source_archive": frames["source_archive"].copy(),
    }
    summary = validate_repaired_frames(
        rewritten, evidence, completed_session=completed_session
    )
    return rewritten, {**summary, **preflight, **index_summary}


def _looks_repaired(frames: Mapping[str, pd.DataFrame]) -> bool:
    master_ids = set(frames["security_master"]["security_id"].astype(str))
    required_events = {
        canonical_lifecycle_event_id(EXE_ID, "ticker_change", EXE_FIRST_SESSION),
        canonical_lifecycle_event_id(
            FISV_ACTIVE_ID, "ticker_change", FISV_TO_FI_DATE
        ),
        canonical_lifecycle_event_id(
            FISV_ACTIVE_ID, "ticker_change", FI_TO_FISV_DATE
        ),
    }
    return bool(
        not (master_ids & RETIRED_IDS)
        and required_events.issubset(
            set(frames["corporate_actions"]["event_id"].astype(str))
        )
    )


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
        version = self.versions.get(dataset)
        if not version:
            return pd.DataFrame(columns=dataset_spec(dataset).required_columns)
        return self.base.read_frame(dataset, version)


def _capture_pointer_etags(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for dataset in WRITE_DATASETS:
        pointer, etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != release.dataset_versions.get(dataset):
            raise RuntimeError(f"Confirmed identity release/pointer mismatch: {dataset}")
        output[dataset] = etag
    return output


def prepare_run(repository: LocalDatasetRepository) -> PreparedRepair:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current release is required for identity repair.")
    missing = sorted(set(REQUIRED_DATASETS) - set(release.dataset_versions))
    if missing:
        raise RuntimeError("Current release lacks datasets: " + ", ".join(missing))
    frames = {
        dataset: repository.read_frame(dataset, release.dataset_versions[dataset])
        for dataset in REQUIRED_DATASETS
    }
    evidence = _load_evidence(repository.root, frames["source_archive"])
    pointer_etags = _capture_pointer_etags(repository, release)
    if _looks_repaired(frames):
        summary = validate_repaired_frames(
            frames, evidence, completed_session=release.completed_session
        )
        validate_repository_snapshot(repository).raise_for_errors()
        return PreparedRepair(
            release=release,
            release_etag=release_etag,
            pointer_etags=pointer_etags,
            frames={dataset: frames[dataset].copy() for dataset in REQUIRED_DATASETS},
            evidence=evidence,
            summary={
                **summary,
                "status": "already_repaired",
                "release_version": release.version,
            },
        )
    rewritten, summary = prepare_repair_frames(
        frames,
        evidence,
        completed_session=release.completed_session,
        source_version=f"confirmed-identity-history-repair/{release.version}",
    )
    candidate = _CandidateRepository(repository, release.dataset_versions, rewritten)
    validate_repository_snapshot(candidate).raise_for_errors()
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        frames=rewritten,
        evidence=evidence,
        summary={**summary, "release_version": release.version},
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
        recovery = repository.root / "recovery"
        pending = tuple(recovery.rglob("*.json")) if recovery.exists() else ()
        if pending:
            raise RuntimeError(
                "A recovery marker blocks confirmed identity writes: "
                + ", ".join(str(item) for item in pending)
            )
        transactions = repository.root / "transactions"
        interrupted: list[Path] = []
        if transactions.exists():
            for item in transactions.rglob("*.json"):
                try:
                    status = _text(json.loads(item.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    interrupted.append(item)
        if interrupted:
            raise RuntimeError(
                "An interrupted transaction blocks confirmed identity writes: "
                + ", ".join(str(item) for item in interrupted)
            )
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, _canonical_json_bytes(dict(value)))


def _restore_transaction(
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
            observed = DataRelease.from_bytes(current.data)
            ours = observed.version == committed_release_version or all(
                observed.dataset_versions.get(dataset) == version
                for dataset, version in planned_versions.items()
            )
            if not ours:
                raise RuntimeError(
                    f"Unexpected release during identity rollback: {observed.version}"
                )
            repository.objects.put(
                "releases/current.json", old_release_bytes, if_match=current.etag
            )
    except Exception as exc:
        errors.append(f"releases/current.json: {type(exc).__name__}: {exc}")
    for dataset in reversed(WRITE_DATASETS):
        key = repository.current_key(dataset)
        try:
            current = repository.objects.get(key)
            if current.data != old_pointer_bytes[dataset]:
                pointer = CurrentPointer.from_bytes(current.data)
                if pointer.version != planned_versions[dataset]:
                    raise RuntimeError(
                        f"Unexpected identity pointer during rollback: {pointer.version}"
                    )
                repository.objects.put(
                    key, old_pointer_bytes[dataset], if_match=current.etag
                )
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> dict[str, Any]:
    if prepared.summary["status"] == "already_repaired":
        return dict(prepared.summary)
    with _exclusive_repository_lock(repository):
        current, current_etag = repository.current_release()
        if (
            current is None
            or current.version != prepared.release.version
            or current_etag != prepared.release_etag
        ):
            raise RuntimeError("Current release changed after identity preflight.")
        old_release = repository.objects.get("releases/current.json")
        old_pointers: dict[str, bytes] = {}
        for dataset in WRITE_DATASETS:
            value = repository.objects.get(repository.current_key(dataset))
            pointer = CurrentPointer.from_bytes(value.data)
            if (
                pointer.version != prepared.release.dataset_versions[dataset]
                or value.etag != prepared.pointer_etags[dataset]
            ):
                raise RuntimeError(f"Identity pointer changed before apply: {dataset}")
            old_pointers[dataset] = value.data

        transaction_id = uuid.uuid4().hex
        session = prepared.release.completed_session.replace("-", "")
        planned = {
            dataset: f"confirmed-identity-{session}-{transaction_id}-{dataset}"
            for dataset in WRITE_DATASETS
        }
        journal_path = (
            repository.root
            / "transactions/confirmed-identity-history-repair"
            / f"{transaction_id}.json"
        )
        journal: dict[str, Any] = {
            "schema": "confirmed_identity_history_repair_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": {
                dataset: base64.b64encode(value).decode("ascii")
                for dataset, value in old_pointers.items()
            },
            "planned_versions": planned,
            "created_at": utc_now_iso(),
            "network_accessed": False,
            "r2_accessed": False,
        }
        _write_journal(journal_path, journal)
        committed: DataRelease | None = None
        try:
            versions = dict(prepared.release.dataset_versions)
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    prepared.frames[dataset],
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy="warn",
                    metadata={
                        "operation": "repair_us_confirmed_identity_histories",
                        "canonical_ids": [KHC_ID, EXE_ID, FISV_ACTIVE_ID],
                        "retired_ids": sorted(RETIRED_IDS),
                        "evidence_sha256": [
                            spec.source_hash for spec in EVIDENCE_SPECS
                        ],
                        "network_accessed": False,
                        "r2_accessed": False,
                    },
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=planned[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"Identity write conflicted: {dataset}/{result.conflict_path}"
                    )
                versions[dataset] = result.manifest.version
            written = {
                dataset: repository.read_frame(dataset, planned[dataset])
                for dataset in WRITE_DATASETS
            }
            written["source_archive"] = prepared.frames["source_archive"].copy()
            validate_repaired_frames(
                written,
                prepared.evidence,
                completed_session=prepared.release.completed_session,
            )
            candidate = _CandidateRepository(repository, versions, written)
            validate_repository_snapshot(candidate).raise_for_errors()
            committed = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=DataQuality.DEGRADED,
                warnings=prepared.release.warnings,
                expected_etag=prepared.release_etag,
            )
            latest, _ = repository.current_release()
            if latest is None or latest.to_bytes() != committed.to_bytes():
                raise RuntimeError("Committed confirmed-identity release is not current.")
            replay = prepare_run(repository)
            if replay.summary.get("status") != "already_repaired":
                raise RuntimeError("Confirmed identity post-commit idempotence failed.")
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
                "transaction_id": transaction_id,
                "new_release_version": committed.version,
                "quality": committed.quality,
                "warnings": list(committed.warnings),
                "idempotence_status": replay.summary["status"],
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointers,
                planned_versions=planned,
                committed_release_version=(committed.version if committed else ""),
            )
            journal.update(
                {
                    "status": "rollback_failed" if rollback_errors else "rolled_back",
                    "original_error": f"{type(original).__name__}: {original}",
                    "rollback_errors": list(rollback_errors),
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            if rollback_errors:
                recovery = (
                    repository.root
                    / "recovery/confirmed-identity-history-repair"
                    / f"{transaction_id}.json"
                )
                _write_journal(recovery, journal)
                raise RuntimeError(
                    "Confirmed identity rollback failed; recovery marker blocks writes: "
                    f"{recovery}; errors={rollback_errors}"
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair KHC, CHK/EXE and FISV/FI/FISV identity histories."
    )
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline-plan", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def run(
    args: argparse.Namespace,
    *,
    repository_factory: Callable[[str | Path], LocalDatasetRepository] = (
        LocalDatasetRepository
    ),
) -> dict[str, Any]:
    repository = repository_factory(args.cache_root)
    prepared = prepare_run(repository)
    if not bool(getattr(args, "apply", False)):
        return prepared.summary
    return apply_repair(repository, prepared)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
