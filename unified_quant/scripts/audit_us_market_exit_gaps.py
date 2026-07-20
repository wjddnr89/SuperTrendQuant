#!/usr/bin/env python3
"""Read-only, release-pinned audit of four US bankruptcy/OTC market-exit gaps.

The command reads only the current local Parquet release and already-cached SEC
bytes.  It never performs HTTP, EODHD, R2, or dataset writes.  Its purpose is
to distinguish a legal cancellation date from an exchange suspension/OTC
continuation, and to show whether the missing tail can affect the two modeled
index backtests.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from supertrend_quant.market_store.cross_validation import canonical_json_bytes  # noqa: E402
from supertrend_quant.market_store.manifest import sha256_bytes  # noqa: E402
from supertrend_quant.market_store.repository import LocalDatasetRepository  # noqa: E402


PINNED_RELEASE_VERSION = "20260715-20260718T230255094849Z"
AUDIT_SCHEMA = "us_market_exit_gap_audit/v1"


@dataclass(frozen=True)
class EvidencePin:
    cache_object: str
    payload_sha256: str
    required_patterns: tuple[str, ...]
    claim: str


@dataclass(frozen=True)
class CaseSpec:
    symbol: str
    security_id: str
    expected_price_rows: int
    expected_first_price: str
    expected_last_price: str
    expected_last_positive_volume: str
    expected_price_rows_sha256: str
    expected_legacy_exchange: str
    event_id: str
    event_effective_date: str
    event_source_hash: str
    event_source_url: str
    resolution_candidate_id: str
    suspension_date: str
    otc_first_date: str
    otc_first_date_status: str
    otc_symbol: str
    exchange_removal_date: str
    post_transition_row_expectation: int
    index_removals: tuple[tuple[str, str], ...]
    successor_relationship: str
    successor_symbol: str
    successor_security_id: str
    minimum_safe_repair: str
    fail_closed_condition: str
    transition_evidence: tuple[EvidencePin, ...]


CASES = (
    CaseSpec(
        symbol="WIN",
        security_id="US:EODHD:2d3e3e74-9be9-5696-94c4-97a5f7598f79",
        expected_price_rows=1118,
        expected_first_price="2015-01-02",
        expected_last_price="2020-07-10",
        expected_last_positive_volume="2019-06-28",
        expected_price_rows_sha256="3fd3dbd065ce606b840e9b62367cbc015851caf6b3e3bb35d833fc5870a2b312",
        expected_legacy_exchange="NASDAQ",
        event_id="4294b6bfa674fab682ba9c299b4fae27ae54b60081c58140f6846b933a47e1ef",
        event_effective_date="2020-09-21",
        event_source_hash="656d5eebc149b51f53a0bb48bc3de4547f5b53986f97f84e4cdc679fc4bca125",
        event_source_url="https://www.sec.gov/Archives/edgar/data/1282266/000114036120021072/brhc10015294_8k.htm",
        resolution_candidate_id="635836bfa46d9be1037d2f96f2af2dc7ac257d30598a3c3186e089ef9d81c40e",
        suspension_date="",
        otc_first_date="",
        otc_first_date_status="not_bound_in_local_evidence",
        otc_symbol="",
        exchange_removal_date="",
        post_transition_row_expectation=0,
        index_removals=(("sp500", "2015-04-07"),),
        successor_relationship=(
            "Legacy common was cancelled for no distribution. New Windstream "
            "Holdings II LLC units went to creditor classes and are not a "
            "public-price successor for legacy shareholders."
        ),
        successor_symbol="",
        successor_security_id="",
        minimum_safe_repair=(
            "Do not infer an OTC ticker/date. First archive official exchange "
            "transition evidence and an independently sourced same-security "
            "OTC price tail; then remove the six trailing zero-volume stale rows."
        ),
        fail_closed_condition=(
            "Keep the exact reviewed degraded exception only while the 2015-04-07 "
            "S&P 500 removal, no later ADD/anchor, all post-2019 price bytes, and "
            "the 2020-09-21 zero-distribution cancellation remain exact."
        ),
        transition_evidence=(
            EvidencePin(
                cache_object="state/sec_lifecycle/36c5c096b38af14320b9f6606cea8bf5c4c54c92c4edf2baeb94c3574f6b28de.bin",
                payload_sha256="4709354a7e186519638633ade9cc3652cf88c9608cbdd287acfb7b8123cc0bf9",
                required_patterns=(
                    r"common stock traded on the OTC Pink Sheets",
                ),
                claim=(
                    "A 2020-07-30 filing confirms OTC Pink status, but the "
                    "cached evidence does not bind the first OTC date or ticker."
                ),
            ),
        ),
    ),
    CaseSpec(
        symbol="CHK",
        security_id="US:EODHD:54d04976-15c6-5ba9-a2cc-10701a4b5c1f",
        expected_price_rows=1381,
        expected_first_price="2015-01-02",
        expected_last_price="2020-06-26",
        expected_last_positive_volume="2020-06-26",
        expected_price_rows_sha256="9ed6e2b5931a308d384f153bf823dbd4fcdb4263bcf7b9fdf4ed72ec5338300b",
        expected_legacy_exchange="NASDAQ",
        event_id="6b6b3440b4c3c0466e5b8d2ee6a8339cd230998a837c7cb573dae75fff565b98",
        event_effective_date="2021-02-09",
        event_source_hash="80f610bb05f197ef740bae2b23c03af96786118ff08dc23a4c78038a577c4842",
        event_source_url="https://www.sec.gov/Archives/edgar/data/895126/000089512621000033/chk-20210209.htm",
        resolution_candidate_id="c8d57e386a5416c5c9a51ba7fd50d35462643d0f6a894ab991b98f3ee108f17d",
        suspension_date="2020-06-29",
        otc_first_date="2020-06-30",
        otc_first_date_status="exact_official",
        otc_symbol="CHKAQ",
        exchange_removal_date="2020-07-31",
        post_transition_row_expectation=0,
        index_removals=(("sp500", "2018-03-19"),),
        successor_relationship=(
            "CHKAQ was the same legacy common security. That equity was cancelled "
            "without distribution on 2021-02-09; the reorganized CHK common that "
            "began trading 2021-02-10 is a distinct security."
        ),
        successor_symbol="CHK",
        successor_security_id="US:EODHD:97548dea-74f0-55a8-b906-47d5c2a072e1",
        minimum_safe_repair=(
            "Close legacy CHK/NYSE on 2020-06-29, add CHKAQ/OTC on the same old "
            "security ID from 2020-06-30 through cancellation, and load a hash-"
            "pinned CHKAQ raw-price tail. Keep reorganized CHK on its distinct ID."
        ),
        fail_closed_condition=(
            "Until the CHKAQ price tail is independently archived, retain a "
            "mismatch; never bridge legacy CHKAQ to reorganized CHK returns."
        ),
        transition_evidence=(
            EvidencePin(
                cache_object="state/sec_lifecycle/8e25805c701e13e19554c35e9c82e857e7ed2f62996b18d7ade5c253d82ab924.bin",
                payload_sha256="f1dc291d2ba3b9e420f9c3e973c3bb622ef54f74597a474dbd1c7515039f263d",
                required_patterns=(
                    r"indefinitely suspended trading .* on June 29, 2020",
                    r"commenced, effective as of June 30, 2020, on the OTC Pink Market .* CHKAQ",
                ),
                claim="NYSE suspension and exact CHKAQ OTC start.",
            ),
            EvidencePin(
                cache_object="state/sec_lifecycle/7286681fa2330335847682eae1a6d43123a53233af28bf025b7f4bd6d41b0e8b.bin",
                payload_sha256="4aee77582bc25e6707f20b4a903f2e95f9d2ec15b7436926ce458153140bf2d8",
                required_patterns=(
                    r"remove .* at the opening of business on July 31, 2020",
                ),
                claim="NYSE Form 25-NSE removal date.",
            ),
        ),
    ),
    CaseSpec(
        symbol="FTR",
        security_id="US:EODHD:62c84ca3-49c6-5a21-ac5b-14c591519d29",
        expected_price_rows=1340,
        expected_first_price="2015-01-02",
        expected_last_price="2020-04-29",
        expected_last_positive_volume="2020-04-23",
        expected_price_rows_sha256="2d8cf5019314e8c64ebfffe46dd7d66312a87333dca83aa188abc7edb2efdcfa",
        expected_legacy_exchange="NYSE",
        event_id="1377981293c7eebce5cb6da722da3c4058a29077c96847b258d21df5af601902",
        event_effective_date="2021-04-30",
        event_source_hash="b581e1ff7cb90e9abf699236ca40f6ccc4b7233fbbb587129bd355d5f23f76cc",
        event_source_url="https://www.sec.gov/Archives/edgar/data/20520/000114036121015200/brhc10023786_8k12g3.htm",
        resolution_candidate_id="91fcbe955bc16c96dec8005954d342d5a4de6dc46ff7ac1f860d226fd148f930",
        suspension_date="2020-04-24",
        otc_first_date="2020-04-24",
        otc_first_date_status="expected_official_symbol_changed_by_confirmation",
        otc_symbol="FTRCQ",
        exchange_removal_date="2020-05-08",
        post_transition_row_expectation=4,
        index_removals=(("sp500", "2017-03-20"),),
        successor_relationship=(
            "FTRCQ was the same legacy common security and was cancelled without "
            "distribution on 2021-04-30. Reorganized FYBR common is a distinct "
            "equity approved to start Nasdaq trading around 2021-05-04."
        ),
        successor_symbol="FYBR",
        successor_security_id="",
        minimum_safe_repair=(
            "Delete the four 2020-04-24..2020-04-29 zero-volume flat placeholders, "
            "model the same-security Nasdaq-to-OTC transition, and load a hash-"
            "pinned FTRCQ tail through the 2021-04-30 cancellation."
        ),
        fail_closed_condition=(
            "Do not use the anticipated FTRQ label as an observed price identity. "
            "Require independent FTRCQ prices and keep FYBR economically separate."
        ),
        transition_evidence=(
            EvidencePin(
                cache_object="state/sec_lifecycle/7f93c5fe83474ceb4d1f16f950f684d8860322d93810114a8c1e43fea1d881b3.bin",
                payload_sha256="0f48e56a3f066e6800e4b5e9940d7b03353cad96f870cd8554e7661ce8cca08e",
                required_patterns=(
                    r"suspended at the opening of business on April 24, 2020",
                    r"commence on April 24, 2020 under the symbol .*FTRQ",
                ),
                claim="Nasdaq suspension and anticipated first OTC label/date.",
            ),
            EvidencePin(
                cache_object="state/sec_lifecycle/f002c8d43ccb3624aa9a29926f98611d6f933474938ca36d7b5fa7cb5f4865f2.bin",
                payload_sha256="32de36ada64faa871090e95248f871016b1c70d1b36016730c285e2b5b2ed7d9",
                required_patterns=(
                    r"Trading of Frontier.*common stock now occurs on the OTC Pink Market under the symbol .*FTRCQ",
                ),
                claim="FTRCQ is confirmed as the actual OTC symbol by 2020-05-01.",
            ),
            EvidencePin(
                cache_object="state/sec_lifecycle/d2c7fba18d02edd0d7ed96a656a9f969eb50f132950b2d4df10344373dddea56.bin",
                payload_sha256="87b825a5c34d16eb07423b26d258060acc01d463d11c50f4ccb3cdbf3ab110d8",
                required_patterns=(
                    r"remove from listing .* effective at the opening of the trading session on May 8, 2020",
                ),
                claim="Nasdaq Form 25-NSE removal date.",
            ),
        ),
    ),
    CaseSpec(
        symbol="ENDP",
        security_id="US:EODHD:f36c4483-5fa7-5866-b266-97130bc35bde",
        expected_price_rows=1970,
        expected_first_price="2015-01-02",
        expected_last_price="2022-10-27",
        expected_last_positive_volume="2022-10-27",
        expected_price_rows_sha256="8811ff519cb9b70db7b6d05a73493076fa3958d79b4ae6323acbf04e7e3b6737",
        expected_legacy_exchange="NASDAQ",
        event_id="1849ca428dda73f0322d36334d81ce5a00f0185702e3ea2b7e5e4fea6fdb7704",
        event_effective_date="2024-04-23",
        event_source_hash="5669e33821e77bf97bd722bf909f649ad10caebb91973f00a2b053515a5c1377",
        event_source_url="https://www.sec.gov/Archives/edgar/data/2008861/000119312524178089/d15705ds1.htm",
        resolution_candidate_id="4f3aa71e1dd7619266eac914379e5a40537699d3957ad6ac21fefc8a56f0ba84",
        suspension_date="2022-08-26",
        otc_first_date="2022-08-26",
        otc_first_date_status="exact_official",
        otc_symbol="ENDPQ",
        exchange_removal_date="",
        post_transition_row_expectation=44,
        index_removals=(
            ("nasdaq100", "2016-07-18"),
            ("sp500", "2017-03-02"),
        ),
        successor_relationship=(
            "ENDPQ was the same legacy Endo International plc equity. The "
            "2024 Endo, Inc. successor was newly formed without the predecessor's "
            "participation; it is not a shareholder-return successor."
        ),
        successor_symbol="",
        successor_security_id="",
        minimum_safe_repair=(
            "Split the identity at 2022-08-26, rebind the 44 already-stored "
            "post-transition rows to ENDPQ/OTC on the same old ID, and supplement "
            "the independently archived ENDPQ tail through 2024-04-23."
        ),
        fail_closed_condition=(
            "Until the missing ENDPQ tail is independently archived, keep the "
            "cross-provider mismatch and never join new Endo, Inc. prices."
        ),
        transition_evidence=(
            EvidencePin(
                cache_object="state/sec_lifecycle/952779e184f2a41e2949806e04783242530d3cd06cc4c319ab94a535205dc026.bin",
                payload_sha256="1c6adf64a35e70602ccca19cfbb957f5abcc8808d7def6ba45eb8f76c49ba971",
                required_patterns=(
                    r"suspended at the opening of business on August 26, 2022",
                    r"began trading exclusively on the over-the-counter .* on August 26, 2022",
                    r"under the symbol ENDPQ",
                ),
                claim="Exact Nasdaq suspension and ENDPQ OTC start.",
            ),
        ),
    ),
)


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
    text = _text(value)
    return pd.Timestamp(text).date().isoformat() if text else ""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _normalized_text(payload: bytes) -> str:
    decoded = payload.decode("utf-8", errors="ignore")
    decoded = re.sub(r"<[^>]+>", " ", decoded)
    decoded = html.unescape(decoded)
    return re.sub(r"\s+", " ", decoded).strip()


def verify_evidence_pin(data_root: Path, pin: EvidencePin) -> dict[str, Any]:
    path = data_root / pin.cache_object
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    _require(actual_sha256 == pin.payload_sha256, f"Cached SEC bytes changed: {path}")
    text = _normalized_text(payload)
    for pattern in pin.required_patterns:
        _require(
            re.search(pattern, text, flags=re.IGNORECASE) is not None,
            f"Pinned SEC claim changed: {path}: {pattern}",
        )
    return {
        "cache_object": pin.cache_object,
        "payload_bytes": len(payload),
        "payload_sha256": actual_sha256,
        "required_pattern_count": len(pin.required_patterns),
        "claim": pin.claim,
        "verified": True,
    }


def _verify_release_archive(
    data_root: Path,
    archive: pd.DataFrame,
    source_hash: str,
    source_url: str,
) -> dict[str, Any]:
    rows = archive.loc[
        archive["source_hash"].map(_text).str.lower().eq(source_hash)
        & archive["source_url"].map(_text).eq(source_url)
    ]
    _require(len(rows) == 1, "Official cancellation archive pair changed: " + source_hash)
    row = rows.iloc[0]
    object_path = _text(row.get("object_path"))
    compressed = (data_root / object_path).read_bytes()
    payload = gzip.decompress(compressed)
    _require(sha256_bytes(payload) == source_hash, "Archived cancellation bytes changed: " + source_hash)
    return {
        "dataset": _text(row.get("dataset")),
        "source_url": source_url,
        "source_hash": source_hash,
        "object_path": object_path,
        "payload_bytes": len(payload),
        "verified": True,
    }


def _row_sha256(frame: pd.DataFrame) -> str:
    columns = sorted(frame.columns)
    rows = frame.loc[:, columns].fillna("").to_dict("records")
    return sha256_bytes(canonical_json_bytes(rows))


def _index_state(
    security_id: str,
    terminal_date: str,
    expected_removals: Sequence[tuple[str, str]],
    anchors: pd.DataFrame,
    events: pd.DataFrame,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index_id, expected_remove in expected_removals:
        scoped_events = events.loc[
            events["security_id"].map(_text).eq(security_id)
            & events["index_id"].map(_text).eq(index_id)
        ].copy()
        scoped_events["_date"] = scoped_events["effective_date"].map(_date)
        removals = scoped_events.loc[
            scoped_events["operation"].map(_text).str.upper().eq("REMOVE")
        ].sort_values("_date")
        _require(
            len(removals) == 1 and _text(removals.iloc[0]["_date"]) == expected_remove,
            f"Reviewed index removal changed: {index_id}:{security_id}",
        )
        later_events = scoped_events.loc[scoped_events["_date"].gt(expected_remove)]
        later_anchors = anchors.loc[
            anchors["security_id"].map(_text).eq(security_id)
            & anchors["index_id"].map(_text).eq(index_id)
            & anchors["anchor_date"].map(_date).gt(expected_remove)
        ]
        before_terminal = scoped_events.loc[scoped_events["_date"].le(terminal_date)].sort_values("_date")
        member = False
        if not before_terminal.empty:
            member = _text(before_terminal.iloc[-1]["operation"]).upper() == "ADD"
        _require(not member and later_events.empty and later_anchors.empty, f"Index re-entry changed: {index_id}:{security_id}")
        output.append(
            {
                "index_id": index_id,
                "remove_date": expected_remove,
                "member_on_stored_terminal_session": False,
                "later_event_count": 0,
                "later_anchor_count": 0,
            }
        )
    return output


def build_audit(repository: LocalDatasetRepository) -> dict[str, Any]:
    release, _ = repository.current_release()
    _require(release is not None, "Current local release is missing.")
    _require(release.version == PINNED_RELEASE_VERSION, "Audit release changed; re-review the finite inventory.")
    names = (
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "corporate_actions",
        "lifecycle_resolutions",
        "source_archive",
        "index_membership_events",
        "index_constituent_anchors",
    )
    frames = {
        name: repository.read_frame(name, release.dataset_versions[name])
        for name in names
    }
    rows: list[dict[str, Any]] = []
    for spec in CASES:
        master = frames["security_master"].loc[
            frames["security_master"]["security_id"].map(_text).eq(spec.security_id)
        ]
        _require(len(master) == 1, "security_master identity changed: " + spec.symbol)
        _require(
            _text(master.iloc[0].get("primary_symbol")).upper() == spec.symbol
            and _date(master.iloc[0].get("active_to")) == spec.expected_last_price,
            "security_master terminal boundary changed: " + spec.symbol,
        )
        history = frames["symbol_history"].loc[
            frames["symbol_history"]["security_id"].map(_text).eq(spec.security_id)
            & frames["symbol_history"]["symbol"].map(_text).str.upper().eq(spec.symbol)
        ]
        _require(len(history) == 1, "Legacy symbol interval changed: " + spec.symbol)
        _require(
            _text(history.iloc[0].get("exchange")).upper() == spec.expected_legacy_exchange
            and not _date(history.iloc[0].get("effective_to")),
            "Legacy open identity interval changed: " + spec.symbol,
        )

        prices = frames["daily_price_raw"].loc[
            frames["daily_price_raw"]["security_id"].map(_text).eq(spec.security_id)
        ].copy()
        prices["_session"] = prices["session"].map(_date)
        prices = prices.sort_values("_session")
        positive = prices.loc[pd.to_numeric(prices["volume"], errors="coerce").fillna(0).gt(0)]
        _require(
            len(prices) == spec.expected_price_rows
            and _text(prices.iloc[0]["_session"]) == spec.expected_first_price
            and _text(prices.iloc[-1]["_session"]) == spec.expected_last_price
            and _text(positive.iloc[-1]["_session"]) == spec.expected_last_positive_volume,
            "Stored price boundary changed: " + spec.symbol,
        )
        price_rows_sha256 = _row_sha256(prices.drop(columns=["_session"]))
        _require(
            price_rows_sha256 == spec.expected_price_rows_sha256,
            "Stored price bytes changed: " + spec.symbol,
        )
        post_transition = prices.iloc[0:0]
        if spec.otc_first_date:
            post_transition = prices.loc[prices["_session"].ge(spec.otc_first_date)]
            _require(len(post_transition) == spec.post_transition_row_expectation, "Post-transition row inventory changed: " + spec.symbol)
        stale_after_last_positive = prices.loc[
            prices["_session"].gt(spec.expected_last_positive_volume)
        ]
        stale_zero_flat = stale_after_last_positive.loc[
            pd.to_numeric(stale_after_last_positive["volume"], errors="coerce").fillna(0).eq(0)
            & stale_after_last_positive["open"].eq(stale_after_last_positive["high"])
            & stale_after_last_positive["high"].eq(stale_after_last_positive["low"])
            & stale_after_last_positive["low"].eq(stale_after_last_positive["close"])
        ]

        actions = frames["corporate_actions"].loc[
            frames["corporate_actions"]["event_id"].map(_text).eq(spec.event_id)
        ]
        _require(len(actions) == 1, "Official cancellation action changed: " + spec.symbol)
        action = actions.iloc[0]
        _require(
            _text(action.get("security_id")) == spec.security_id
            and _text(action.get("action_type")) == "delisting"
            and _date(action.get("effective_date")) == spec.event_effective_date
            and float(action.get("cash_amount")) == 0.0
            and not _text(action.get("new_security_id"))
            and _text(action.get("source_hash")).lower() == spec.event_source_hash,
            "Official zero-distribution action terms changed: " + spec.symbol,
        )
        _require(
            _text(action.get("source_url")) == spec.event_source_url,
            "Official cancellation source URL changed: " + spec.symbol,
        )
        resolutions = frames["lifecycle_resolutions"].loc[
            frames["lifecycle_resolutions"]["candidate_id"].map(_text).eq(spec.resolution_candidate_id)
        ]
        _require(
            len(resolutions) == 1
            and _text(resolutions.iloc[0].get("resolution")) == "applied"
            and _text(resolutions.iloc[0].get("event_id")) == spec.event_id
            and _date(resolutions.iloc[0].get("last_price_date")) == spec.expected_last_price,
            "Lifecycle resolution changed: " + spec.symbol,
        )
        _require(
            _text(resolutions.iloc[0].get("source_hash")).lower()
            == spec.event_source_hash
            and _text(resolutions.iloc[0].get("source_url"))
            == spec.event_source_url,
            "Lifecycle resolution evidence changed: " + spec.symbol,
        )

        successor = {
            "relationship": spec.successor_relationship,
            "symbol": spec.successor_symbol,
            "security_id": spec.successor_security_id,
            "present_in_release": False,
            "first_price_session": "",
        }
        if spec.successor_security_id:
            successor_prices = frames["daily_price_raw"].loc[
                frames["daily_price_raw"]["security_id"].map(_text).eq(spec.successor_security_id)
            ]
            _require(not successor_prices.empty, "Reviewed successor price identity changed: " + spec.symbol)
            successor["present_in_release"] = True
            successor["first_price_session"] = min(successor_prices["session"].map(_date))

        evidence = [
            verify_evidence_pin(repository.root, pin)
            for pin in spec.transition_evidence
        ]
        index_state = _index_state(
            spec.security_id,
            spec.expected_last_price,
            spec.index_removals,
            frames["index_constituent_anchors"],
            frames["index_membership_events"],
        )
        rows.append(
            {
                "symbol": spec.symbol,
                "security_id": spec.security_id,
                "classification": "dataset_repair_required",
                "stored_identity": {
                    "security_master_exchange": _text(master.iloc[0].get("exchange")),
                    "security_master_active_to": _date(master.iloc[0].get("active_to")),
                    "symbol_history_exchange": _text(history.iloc[0].get("exchange")),
                    "symbol_history_effective_to": _date(history.iloc[0].get("effective_to")),
                    "open_interval_conflict": True,
                },
                "stored_price_tail": {
                    "row_count": len(prices),
                    "first_session": spec.expected_first_price,
                    "last_session": spec.expected_last_price,
                    "last_positive_volume_session": spec.expected_last_positive_volume,
                    "rows_on_or_after_otc_transition": len(post_transition),
                    "zero_volume_flat_rows_after_last_positive_volume": len(stale_zero_flat),
                    "rows_sha256": price_rows_sha256,
                },
                "market_exit": {
                    "exchange_suspension_date": spec.suspension_date,
                    "otc_first_date": spec.otc_first_date,
                    "otc_first_date_status": spec.otc_first_date_status,
                    "otc_symbol": spec.otc_symbol,
                    "exchange_removal_date": spec.exchange_removal_date,
                    "legacy_equity_cancellation_date": spec.event_effective_date,
                },
                "official_cancellation_action": {
                    "event_id": spec.event_id,
                    "effective_date": spec.event_effective_date,
                    "cash_amount": 0.0,
                    "new_security_id": "",
                    "source_hash": spec.event_source_hash,
                    "source_url": spec.event_source_url,
                    "archive": _verify_release_archive(
                        repository.root,
                        frames["source_archive"],
                        spec.event_source_hash,
                        spec.event_source_url,
                    ),
                },
                "lifecycle_resolution": {
                    "candidate_id": spec.resolution_candidate_id,
                    "resolution": "applied",
                    "event_id": spec.event_id,
                    "stored_last_price_date": spec.expected_last_price,
                },
                "transition_evidence": evidence,
                "successor": successor,
                "index_scope": index_state,
                "triple_supertrend_backtest": {
                    "direct_delta_expected": 0,
                    "basis": (
                        "Every proposed repair starts after the last target-index "
                        "removal, and the release has no later ADD or anchor."
                    ),
                    "scope": [item[0] for item in spec.index_removals],
                },
                "minimum_safe_repair": spec.minimum_safe_repair,
                "fail_closed_condition": spec.fail_closed_condition,
            }
        )

    exact = sum(
        row["market_exit"]["otc_first_date_status"] == "exact_official"
        for row in rows
    )
    bounded = sum(
        row["market_exit"]["otc_first_date_status"]
        == "expected_official_symbol_changed_by_confirmation"
        for row in rows
    )
    unbound = sum(
        row["market_exit"]["otc_first_date_status"] == "not_bound_in_local_evidence"
        for row in rows
    )
    return {
        "schema": AUDIT_SCHEMA,
        "release_version": release.version,
        "network_accessed": False,
        "http_attempts": 0,
        "eodhd_calls": 0,
        "r2_accessed": False,
        "dataset_writes_performed": False,
        "summary": {
            "case_count": len(rows),
            "dataset_repair_required_count": len(rows),
            "exact_otc_start_count": exact,
            "bounded_otc_start_count": bounded,
            "unbound_otc_start_count": unbound,
            "terminal_index_member_count": 0,
            "expected_triple_supertrend_trade_or_equity_delta": 0,
        },
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/cache")
    args = parser.parse_args()
    report = build_audit(LocalDatasetRepository(Path(args.data_root)))
    payload = canonical_json_bytes(report)
    print(payload.decode("utf-8"))
    print("audit_sha256=" + sha256_bytes(payload), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
