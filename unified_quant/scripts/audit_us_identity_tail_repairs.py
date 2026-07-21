#!/usr/bin/env python3
"""Plan-only audit for eight release-pinned US identity price tails.

The command is deliberately offline and finite.  It reads one immutable local
release, verifies the official lifecycle archive bytes, and produces an exact
repair plan.  It does not perform HTTP, EODHD, R2, dataset, or release-pointer
writes.  Repair recommendations are case-specific; there is no date tolerance
or generic identity fallback.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import exchange_calendars as xcals
import duckdb
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from supertrend_quant.indicators import add_triple_supertrend  # noqa: E402
from supertrend_quant.market_store.adjustments import (  # noqa: E402
    apply_adjustment_factors,
)
from supertrend_quant.market_store.manifest import (  # noqa: E402
    sha256_bytes,
    write_atomic,
)
from supertrend_quant.market_store.repository import (  # noqa: E402
    LocalDatasetRepository,
)
from supertrend_quant.market_store.schemas import dataset_spec  # noqa: E402


AUDIT_SCHEMA = "us_identity_tail_repair_audit/v1"
PINNED_RELEASE_VERSION = "20260715-20260718T230255094849Z"
PINNED_DATASET_VERSIONS = {
    "adjustment_factors": (
        "early-terminal-history-2026-07-15-566e79bcc7ac4e268c4cc304e14b700e-"
        "adjustment_factors"
    ),
    "corporate_actions": (
        "early-terminal-history-2026-07-15-566e79bcc7ac4e268c4cc304e14b700e-"
        "corporate_actions"
    ),
    "daily_price_raw": (
        "early-terminal-history-2026-07-15-566e79bcc7ac4e268c4cc304e14b700e-"
        "daily_price_raw"
    ),
    "index_constituent_anchors": (
        "market-transitions-20260715-0cd5221df34d80c5-index_constituent_anchors"
    ),
    "index_membership_events": (
        "market-transitions-20260715-0cd5221df34d80c5-index_membership_events"
    ),
    "security_master": (
        "early-terminal-history-2026-07-15-566e79bcc7ac4e268c4cc304e14b700e-"
        "security_master"
    ),
    "source_archive": (
        "wiki-price-arbitration-20260715-301b7adc38334f65a4012a095993dce9-"
        "source_archive"
    ),
    "symbol_history": (
        "early-terminal-history-2026-07-15-566e79bcc7ac4e268c4cc304e14b700e-"
        "symbol_history"
    ),
}

TRIPLE_SETTINGS = ((10, 1.0), (11, 2.0), (12, 3.0))
SIGNAL_COLUMNS = (
    "TripleST1_Trend",
    "TripleST2_Trend",
    "TripleST3_Trend",
    "TripleAllUp",
    "TripleDownCount",
    "TripleBuySignal",
    "TripleSellSignal",
)


@dataclass(frozen=True)
class CaseSpec:
    symbol: str
    security_id: str
    successor_symbol: str
    successor_security_id: str
    event_id: str
    action_type: str
    transition_date: str
    old_last_good_session: str
    tail_end: str
    tail_rows: int
    old_tail_source_hash: str
    successor_overlap_source_hash: str
    official_source_hash: str
    old_tail_sha256: str
    successor_overlap_sha256: str
    action_row_sha256: str
    membership_inventory_sha256: str
    identity_inventory_sha256: str
    equal_open: int
    equal_high: int
    equal_low: int
    equal_close: int
    equal_volume: int
    equal_currency: int
    exact_ohlcv_currency_rows: int
    repair_class: str


CASES = (
    CaseSpec(
        "FLT", "US:EODHD:41894e44-7e71-54dd-abfc-3454cc10286a",
        "CPAY", "US:EODHD:cfae90d2-b677-53c6-9405-a6f663155e84",
        "cc034dd705e7770f8504c36b2f27604f7834f6e02cfb04edc355c3b70371a8e6",
        "ticker_change", "2024-03-25", "2024-03-22", "2024-05-24", 44,
        "e11d8c4e03cb4ff5abff23b0a06a01002cce4132689e3f9ca1fcebbab7f00d6d",
        "37e696d98f05e233a7004b36cbb0c844f37dda20cc31c2b603a88e63a8c17a1b",
        "efc96eeaa39ea2e90b85b9fb456e698396766bd1b33bc73b49496b511fb5b6fe",
        "d99f23413929a4a10369d59117fafadd717fa6c18216957398754b4186382959",
        "d73c77e4a69cbbf9dc60eb248b565aaf0e34b9b0aeeac17aa7fcecb65f107836",
        "b5744a4b44836a923c5bee421b938167f54c71e61d914a36182c2c7e0b2ed8cb",
        "de762aa8ab3a77716fc95994829c70518931bf9034ed742c548a400544e0ee4e",
        "f7622c8d6737a63d38c8cbfce32eb9039cc310289ed55981fceab2ef3ce59b99",
        44, 35, 37, 44, 2, 44, 2, "delete_old_tail_close_identity",
    ),
    CaseSpec(
        "CDAY", "US:EODHD:46664b7c-7250-543d-abe8-8df5137c4f4b",
        "DAY", "US:EODHD:9fe90f09-82b7-5358-a46c-09051044402a",
        "efeec2d1b30106a50d0909c6568d5162ec615d0d4e2f05492a37af466718de8c",
        "ticker_change", "2024-02-01", "2024-01-31", "2026-02-04", 504,
        "3ed1dba8225e79f8e374d67343a26d46e5a25973341cd1033aee03fc390c839e",
        "3ed1dba8225e79f8e374d67343a26d46e5a25973341cd1033aee03fc390c839e",
        "3662c4cee13538e74aeef83060398400ed3867243db4ae6612a2f6ecb4f85b5c",
        "136a07c746992717437647315d5530aa5f0f9f22acacdd5106aaf4824d1426f6",
        "f6bf6c3e4bdfe29eb2a9aeb8fb13c2fb46a8b5f76b3691a6a80b87727f64c3ac",
        "24a4951b4dd9d30c6b02ea75ef2f68903b7fc63bf9770161a82ce3ca611452fb",
        "e9509808b8a4bfd070476a6488cb9cba03b9e3c57cd79cd4f303d292c5176134",
        "546916a62d0c00a42cacad660a974665a436291bd300966c7f4feb7e0680fa2e",
        504, 504, 504, 504, 504, 504, 504,
        "delete_old_tail_close_identity",
    ),
    CaseSpec(
        "XEC", "US:EODHD:64cd243f-ab9c-516f-967d-15b06ccfcca2",
        "COG", "US:EODHD:e5169183-3360-57f2-846e-da37cb18541a",
        "4a8be9ecafc3d03c987213c87627904a4aefcf49027d35da3ede76f3c46ffdc2",
        "stock_merger", "2021-10-01", "2021-09-30", "2021-10-04", 2,
        "4121a70a6d6fc305fd9df58646d83036f1e6e021f59560c1abe2794a9253f698",
        "1b9dfefc2f590febe76002b73fb5c0d5d0f838617ba4392957221be29767e6dc",
        "cd18f9c67f680493dce1af86c085622ce643861778066b187d64dc58771c3067",
        "a3885a7aaa6d1a330b54174d56d573383e92de98df58d13183c47b4f669ba723",
        "8047ecbb8b7cfa386068103d45cc55dc6c8ea034f509936e0d2d3812c4dc8e13",
        "2967383180b8e965c8fbe740311cfe912d4ad34f5c3fa517e53aa584c708c26c",
        "5588005a6296f5332d746072947876c77632d88a9858588aef2f68c6bf369fc7",
        "6cb080a887c05364acc052bebc968feba266ace9c8535e13f794b6576bee617f",
        0, 0, 0, 0, 0, 2, 0, "delete_synthetic_merger_tail_close_identity",
    ),
    CaseSpec(
        "HCP", "US:EODHD:ad5cea39-7b69-5c0f-93a2-d20add8a8215",
        "PEAK", "US:EODHD:e2fd43dd-fa96-5923-9657-d0debbb1624f",
        "c1fcf2e0a945029e70d7fec9b7fd1fe4594b55fb1393a7ada6779e5847865cea",
        "ticker_change", "2019-11-05", "2019-11-04", "2019-11-08", 4,
        "a04ad2d7482efbda14c7240d787deb3c240cc3573262d2c6c966f20bcc429c0a",
        "564be2316a26477a93779f456b11acf5b76ecf25bd75f9999cd11238253d216b",
        "95bdb7778547e2ba5f7cffdec705fa82bca3624cdef8687c0acde96aa807aa11",
        "337ab264d94c7e1ef3340008169989b6b88d52892d4e2361cfe96f95a238de90",
        "159c6daba47d88029adc3fe207e7e147910bf66baaf7f716896f0199b97a59d8",
        "127217178b350b491cd235edc0565a5dfdfd5fb844f10196320138559dc243df",
        "c282fb7f13376fe4be2f47aff7cb4eae17bdcf05ab230fc9beef917ecac0da7d",
        "5e0436f453c3224d92f0e163e02a8e3e914d69b810b28e88ccedd61ec0178b2d",
        3, 3, 3, 3, 0, 4, 0,
        "replace_successor_first_session_then_delete_old_tail",
    ),
    CaseSpec(
        "UTX", "US:EODHD:aefd1dd7-529d-5b6a-80e9-65d0e14102a6",
        "RTX", "US:EODHD:88af8b3e-5e83-5344-9e82-923a06cf467f",
        "1eebcc8f193de474779068560d90da76961560ffd1fe459dabc10d3c1085374b",
        "ticker_change", "2020-04-03", "2020-04-02", "2020-04-23", 14,
        "979bf4b38ed0292cb30e2b3f23018dcfec16c33c36238375a4665d03abdcedba",
        "e7dc3c9e3755ef02c551ca117240511fe47a74b54f1e67b0ea6acd1e4944a78a",
        "86de499c88aaeb73714cd648434fc88016ce47ab5fa1ed5dd770c26bb63b523d",
        "ec33ceaaaa8d21fc8912dc303189a84968e2346f4d71d74e60fff86df1e2ccba",
        "2380883dbf0c49ada1c877bd47d121666c132fa3b2180f7a49c5fe6aa7767eac",
        "cb2f68a29ef9662087caf2abebc3e3366df5c49d2332c3bf9dba634c144a12a9",
        "00f9eb000c9545d7112e44ecaea9bd5541c6850cf2a36c22d709d0536273535c",
        "5bebd2e4192ecd14df6d4fccfdf2005ac5b6baee4a6b2733fd0bca12afd71d40",
        8, 8, 8, 8, 4, 14, 4, "delete_old_tail_close_identity",
    ),
    CaseSpec(
        "COG", "US:EODHD:e5169183-3360-57f2-846e-da37cb18541a",
        "CTRA", "US:EODHD:97c04993-9845-5645-a7dd-027c590e3502",
        "4a4620d5793b1264ccbe8c1d2a6467bdd0912e1a20e0c454471607541d2ae8dc",
        "ticker_change", "2021-10-04", "2021-10-01", "2021-12-03", 44,
        "1b9dfefc2f590febe76002b73fb5c0d5d0f838617ba4392957221be29767e6dc",
        "edd9b7c4ecc54c0b3df42e873111f2c96be06a31db1a08a8371519ea0a3d68f4",
        "cd18f9c67f680493dce1af86c085622ce643861778066b187d64dc58771c3067",
        "3917d662e76d464d03563fe739bf1cc1d5f011520604d135af944fd13638acf2",
        "e5ff1b17a07ae146a3fcabe5796f22e75af9a1b9c8b5d3d460daa4081d005161",
        "152951658769992ae36aa4e3f7fc4474d2bacfb4f4ec112eda571db553776c2f",
        "bcdd202627b13efe461b913d9f41498af667c3c3df040ca162ba05c1f0dbe9e2",
        "e6023b7f7f610614509f06942468c5568aa397a72989776d3789e731a7e30aef",
        42, 37, 35, 44, 1, 44, 0, "delete_old_tail_close_identity",
    ),
    CaseSpec(
        "CTRP", "US:EODHD:e7669b68-15d6-5ce8-99c9-19f55990f369",
        "TCOM", "US:EODHD:0cc06130-d930-5632-887c-023597219f54",
        "793f1775c345b9ab44ed255b72a5a2ec62f356329d152607717f5c39d7c568d1",
        "ticker_change", "2019-11-05", "2019-11-04", "2019-11-08", 4,
        "9d811f3540c6269736522cf60f3cecb95ab78181978cdaec790e71308eefc6ad",
        "076e826a2a136387e37cd095b6ae9a518201fbf7562162616874a347a04a6971",
        "ae16823ad948aa72f7fc8517af9309af877764c146ff02b6863f68b700cab855",
        "f096e27d7319239d20cd1329676d23b8538e9493790d13dedd2a26110e6cab04",
        "57a1b6c22f92db46af7e2062d175b93a0a9742602ccd2ddba0ec9958adfbbe2e",
        "cfe5ade7e340fdf4dc1b18cd5336435a3776e7a320fbaff10b70992543e4ee84",
        "cef2086f09d3e655576e251c481d8da5caef2b3270789d9d149b0e767224699c",
        "2eb26840d0d8dc88ad2b83f0ba6f888d58a81d98b70ef29594952606947a301b",
        4, 4, 4, 4, 2, 4, 2, "delete_old_tail_close_identity",
    ),
    CaseSpec(
        "SYMC", "US:EODHD:e92a676b-43f3-5d6b-9b7a-5d314fd6135f",
        "NLOK", "US:EODHD:e9eea478-61d8-5762-9f5b-fbdfd69a02a3",
        "1b19b589542dfaf2e0e07c11188c59beab3db1b9e1aaab1b96570cc54d49a1cc",
        "ticker_change", "2019-11-04", "2019-11-01", "2020-11-05", 237,
        "2251e3d6be6dd058891fd83f9478160d53e5f0d928dee411fab265e7bb09a8e7",
        "4935d389ce3dd31477c0906d12a9b8eda3e67332ec14bc715d57c6ce7a313d4a",
        "87a584813a438f76e5cee9ae800678771bc2df5ed6f3e50461273c1849026e18",
        "b9a386176d6afc428d8e6a06a5b6e8f80d98d086ad2dbb6afede868dab5b9e2b",
        "8964316a5b52cae2f85856b892d4f80b9a5a31cd59e880087c661711dbf54cf8",
        "254b11b7803abac0194df6e07e5cbe0565627cb86d3889b59052a133b1d8d997",
        "4bd2c2d8345d26f1e32a049d140b347f7a6b68978019021243f5830e0d44fc07",
        "1d02b70873cd8bee9d90f3b7753284b95bd28130279ba3595d69ab4689118505",
        234, 237, 237, 237, 200, 237, 197,
        "canonicalize_nlok_identity_then_retire_old_sid",
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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
    if not text:
        return ""
    parsed = pd.Timestamp(text)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_localize(None)
    return parsed.date().isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _projected_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format(float(value), ".17g")
    return str(value)


def _frame_sha256(
    frame: pd.DataFrame,
    *,
    columns: Iterable[str] | None = None,
    sort_by: Iterable[str],
) -> str:
    selected = list(columns or frame.columns)
    ordered = frame.sort_values(list(sort_by), kind="stable")
    records = [
        {column: _projected_value(row[column]) for column in selected}
        for _, row in ordered.iterrows()
    ]
    return sha256_bytes(_canonical_json_bytes(records))


def _one_row(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    rows = frame.loc[mask]
    _require(len(rows) == 1, f"Expected one {label}; found {len(rows)}.")
    return rows.iloc[0]


def _read_security_subset(
    repository: LocalDatasetRepository,
    dataset: str,
    version: str,
    security_ids: Iterable[str],
) -> pd.DataFrame:
    """Read only the finite SID inventory without materializing 2M-row tables."""

    paths = [str(path) for path in repository.parquet_paths(dataset, version)]
    _require(bool(paths), f"Pinned {dataset} Parquet inventory is empty.")
    connection = duckdb.connect()
    try:
        frame = connection.execute(
            "SELECT * FROM read_parquet(?, union_by_name=true) "
            "WHERE security_id = ANY(?)",
            [paths, sorted(set(security_ids))],
        ).fetchdf()
    finally:
        connection.close()
    spec = dataset_spec(dataset)
    derived_partitions = [
        column
        for column in spec.partition_columns
        if column in frame.columns and column not in spec.required_columns
    ]
    if derived_partitions:
        frame = frame.drop(columns=derived_partitions)
    return frame.drop_duplicates(
        list(spec.primary_key), keep="last"
    ).reset_index(drop=True)


def _official_archive_binding(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    source_hash: str,
    source_url: str,
) -> dict[str, Any]:
    rows = archive.loc[
        archive["source_hash"].map(_text).str.lower().eq(source_hash)
        & archive["source_url"].map(_text).eq(source_url)
    ]
    row = _one_row(rows, pd.Series(True, index=rows.index), "official archive row")
    root = repository.root.resolve()
    path = (root / _text(row["object_path"])).resolve()
    _require(root in path.parents and path.is_file(), "Official archive object is missing.")
    try:
        payload = gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise RuntimeError("Official archive object is not valid gzip.") from exc
    _require(sha256_bytes(payload) == source_hash, "Official archive bytes changed.")
    return {
        "dataset": _text(row["dataset"]),
        "source_url": source_url,
        "source_hash": source_hash,
        "object_path": _text(row["object_path"]),
        "payload_bytes": len(payload),
        "payload_sha256_verified": True,
    }


def _identity_inventory(
    master: pd.DataFrame,
    history: pd.DataFrame,
    security_ids: set[str],
) -> pd.DataFrame:
    return pd.concat(
        [
            master.loc[master["security_id"].map(_text).isin(security_ids)].assign(
                _kind="master"
            ),
            history.loc[history["security_id"].map(_text).isin(security_ids)].assign(
                _kind="history"
            ),
        ],
        ignore_index=True,
        sort=False,
    )


def _membership_inventory(
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    security_ids: set[str],
) -> pd.DataFrame:
    return pd.concat(
        [
            anchors.loc[
                anchors["security_id"].map(_text).isin(security_ids)
            ].assign(_kind="anchor"),
            events.loc[
                events["security_id"].map(_text).isin(security_ids)
            ].assign(_kind="event"),
        ],
        ignore_index=True,
        sort=False,
    )


def _membership_signature(frame: pd.DataFrame) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in frame.to_dict(orient="records"):
        kind = _text(row.get("_kind"))
        output.append(
            {
                "kind": kind,
                "index_id": _text(row.get("index_id")),
                "date": _date(
                    row.get("anchor_date") if kind == "anchor" else row.get("effective_date")
                ),
                "operation": "ANCHOR" if kind == "anchor" else _text(row.get("operation")).upper(),
                "security_id": _text(row.get("security_id")),
            }
        )
    return sorted(
        output,
        key=lambda value: (
            value["index_id"], value["date"], value["operation"], value["security_id"]
        ),
    )


def _member_on(
    security_id: str,
    index_id: str,
    session: str,
    anchors: pd.DataFrame,
    events: pd.DataFrame,
) -> bool:
    index_anchors = anchors.loc[
        anchors["index_id"].map(_text).eq(index_id)
        & anchors["anchor_date"].map(_date).le(session)
    ].copy()
    _require(not index_anchors.empty, f"No {index_id} anchor before {session}.")
    index_anchors["_date"] = index_anchors["anchor_date"].map(_date)
    anchor_date = index_anchors["_date"].max()
    member = bool(
        index_anchors.loc[index_anchors["_date"].eq(anchor_date), "security_id"]
        .map(_text)
        .eq(security_id)
        .any()
    )
    relevant = events.loc[
        events["index_id"].map(_text).eq(index_id)
        & events["security_id"].map(_text).eq(security_id)
        & events["effective_date"].map(_date).gt(anchor_date)
        & events["effective_date"].map(_date).le(session)
    ].copy()
    if relevant.empty:
        return member
    relevant["_date"] = relevant["effective_date"].map(_date)
    relevant = relevant.sort_values(["_date", "event_id"], kind="stable")
    return _text(relevant.iloc[-1]["operation"]).upper() == "ADD"


def _tail_member_sessions(
    security_id: str,
    tail: pd.DataFrame,
    anchors: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, list[str]]:
    index_ids = sorted(
        set(
            anchors.loc[anchors["security_id"].map(_text).eq(security_id), "index_id"]
            .map(_text)
        )
        | set(
            events.loc[events["security_id"].map(_text).eq(security_id), "index_id"]
            .map(_text)
        )
    )
    sessions = sorted(tail["session"].map(_date))
    output: dict[str, list[str]] = {}
    for index_id in index_ids:
        index_anchors = anchors.loc[
            anchors["index_id"].map(_text).eq(index_id)
        ].copy()
        index_anchors["_date"] = index_anchors["anchor_date"].map(_date)
        security_events = events.loc[
            events["index_id"].map(_text).eq(index_id)
            & events["security_id"].map(_text).eq(security_id)
        ].copy()
        security_events["_date"] = security_events["effective_date"].map(_date)
        security_events = security_events.sort_values(
            ["_date", "event_id"], kind="stable"
        )
        selected: list[str] = []
        for session in sessions:
            eligible_anchors = index_anchors.loc[
                index_anchors["_date"].le(session)
            ]
            _require(
                not eligible_anchors.empty,
                f"No {index_id} anchor before {session}.",
            )
            anchor_date = eligible_anchors["_date"].max()
            member = bool(
                eligible_anchors.loc[
                    eligible_anchors["_date"].eq(anchor_date), "security_id"
                ]
                .map(_text)
                .eq(security_id)
                .any()
            )
            applied = security_events.loc[
                security_events["_date"].gt(anchor_date)
                & security_events["_date"].le(session)
            ]
            if not applied.empty:
                member = _text(applied.iloc[-1]["operation"]).upper() == "ADD"
            if member:
                selected.append(session)
        output[index_id] = selected
    return output


def _signal_frame(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    mode: str,
) -> pd.DataFrame:
    frame = (
        prices.copy()
        if mode == "raw"
        else apply_adjustment_factors(prices, factors, mode=mode)
    )
    frame["session"] = pd.to_datetime(frame["session"], errors="coerce")
    _require(not frame["session"].isna().any(), "Signal price session changed.")
    values = frame.set_index("session")[["high", "low", "close"]].rename(
        columns={"high": "High", "low": "Low", "close": "Close"}
    )
    return add_triple_supertrend(
        values,
        settings=TRIPLE_SETTINGS,
        atr_method="wilder",
        exit_down_count=2,
    )


def _signal_diff(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, Any]:
    _require(baseline.index.equals(candidate.index), "Signal sessions changed.")
    output: dict[str, Any] = {}
    for column in SIGNAL_COLUMNS:
        changed = baseline[column].ne(candidate[column])
        output[column] = {
            "count": int(changed.sum()),
            "sessions": [_date(value) for value in baseline.index[changed]],
        }
    return output


def _successor_reassignment_signal_impact(
    spec: CaseSpec,
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    old_tail: pd.DataFrame,
) -> dict[str, Any]:
    if spec.action_type == "stock_merger":
        return {
            "evaluated": False,
            "reason": "distinct_issuers_stock_merger_reassignment_forbidden",
        }
    successor = prices.loc[
        prices["security_id"].map(_text).eq(spec.successor_security_id)
    ].sort_values("session").copy()
    successor_factors = factors.loc[
        factors["security_id"].map(_text).eq(spec.successor_security_id)
    ].sort_values("session").copy()
    candidate = successor.set_index("session")
    old = old_tail.set_index("session")
    overlap = candidate.index.intersection(old.index)
    _require(len(overlap) == spec.tail_rows, f"{spec.symbol} successor coverage changed.")
    price_columns = ["open", "high", "low", "close", "volume"]
    candidate.loc[overlap, price_columns] = old.loc[overlap, price_columns]
    candidate = candidate.reset_index()
    output: dict[str, Any] = {"evaluated": True, "reassigned_rows": len(overlap)}
    for mode in ("raw", "total_return_adjusted"):
        output[mode] = _signal_diff(
            _signal_frame(successor, successor_factors, mode=mode),
            _signal_frame(candidate, successor_factors, mode=mode),
        )
    return output


def _price_overlap_diagnostic(
    spec: CaseSpec,
    old_tail: pd.DataFrame,
    successor_overlap: pd.DataFrame,
) -> dict[str, Any]:
    joined = old_tail.merge(
        successor_overlap,
        on="session",
        suffixes=("_old", "_successor"),
        validate="one_to_one",
    )
    _require(len(joined) == spec.tail_rows, f"{spec.symbol} overlap rows changed.")
    expected = {
        "open": spec.equal_open,
        "high": spec.equal_high,
        "low": spec.equal_low,
        "close": spec.equal_close,
        "volume": spec.equal_volume,
        "currency": spec.equal_currency,
    }
    actual: dict[str, int] = {}
    all_equal = pd.Series(True, index=joined.index)
    for field in expected:
        equal = joined[f"{field}_old"].fillna(-math.inf).eq(
            joined[f"{field}_successor"].fillna(-math.inf)
        )
        actual[field] = int(equal.sum())
        all_equal &= equal
    _require(actual == expected, f"{spec.symbol} overlap equality signature changed.")
    _require(
        int(all_equal.sum()) == spec.exact_ohlcv_currency_rows,
        f"{spec.symbol} exact overlap row count changed.",
    )
    return {
        "overlap_rows": len(joined),
        "field_exact_counts": actual,
        "all_ohlcv_currency_exact_rows": int(all_equal.sum()),
    }


def _repair_plan(
    spec: CaseSpec,
    old_tail: pd.DataFrame,
    successor_overlap: pd.DataFrame,
    member_sessions: Mapping[str, list[str]],
) -> dict[str, Any]:
    base = {
        "repair_class": spec.repair_class,
        "old_security_master_active_to": spec.old_last_good_session,
        "old_symbol_history_effective_to": spec.old_last_good_session,
        "delete_old_tail_rows": spec.tail_rows,
        "delete_old_tail_start": spec.transition_date,
        "delete_old_tail_end": spec.tail_end,
        "reassign_all_tail_rows": False,
        "rebuild_adjustment_factors": True,
        "generic_tolerance": False,
    }
    if spec.symbol == "XEC":
        base.update(
            {
                "successor_rows_preserved": spec.tail_rows,
                "reason": (
                    "XEC and COG are distinct issuers linked by a 4.0146-for-1 stock "
                    "merger.  The two XEC tail rows are flat 87.2 with zero volume; "
                    "they must not overwrite the independently traded COG rows."
                ),
            }
        )
    elif spec.symbol == "HCP":
        first_old = old_tail.sort_values("session").iloc[0]
        first_successor = successor_overlap.sort_values("session").iloc[0]
        _require(
            _date(first_old["session"]) == spec.transition_date
            and float(first_old["volume"]) == 8_054_269.0
            and float(first_successor["open"]) == float(first_successor["high"])
            == float(first_successor["low"])
            == float(first_successor["close"])
            == 35.78
            and float(first_successor["volume"]) == 0.0,
            "HCP/PEAK first-session defect signature changed.",
        )
        base.update(
            {
                "replace_successor_sessions_from_old": [spec.transition_date],
                "replacement_rows": 1,
                "replacement_source_hash": spec.old_tail_source_hash,
                "reason": (
                    "The derived PEAK first row is a flat prior-close carry-forward "
                    "with zero volume.  The hash-pinned HCP endpoint has the traded "
                    "2019-11-05 OHLCV row; only that one successor row is replaced."
                ),
            }
        )
    elif spec.symbol == "SYMC":
        _require(
            member_sessions == {"sp500": ["2019-11-04"]},
            "SYMC direct membership tail changed.",
        )
        base.update(
            {
                "delete_old_tail_rows": spec.tail_rows,
                "retire_entire_old_sid": True,
                "canonical_security_id": spec.successor_security_id,
                "canonical_symbol_intervals": [
                    {"symbol": "SYMC", "effective_from": "2015-01-01", "effective_to": "2019-11-01"},
                    {"symbol": "NLOK", "effective_from": "2019-11-04", "effective_to": "2022-11-07"},
                ],
                "rebind_sp500_anchor_and_remove_redundant_2019_11_05_swap": True,
                "reason": (
                    "NLOK already owns the 2015 Nasdaq-100 anchor and a complete "
                    "provider history.  A tail-only deletion would leave a one-session "
                    "S&P data gap and pre-2019 NLOK ticker look-ahead."
                ),
            }
        )
    else:
        base.update(
            {
                "successor_rows_preserved": spec.tail_rows,
                "reason": (
                    "The successor covers every tail date and the index identity swaps "
                    "at the transition boundary.  Reassignment would overwrite the "
                    "current-ticker provider series without adding coverage."
                ),
            }
        )
    return base


def _symc_full_identity_diagnostic(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
) -> dict[str, Any]:
    spec = next(case for case in CASES if case.symbol == "SYMC")
    old = prices.loc[prices["security_id"].map(_text).eq(spec.security_id)].sort_values(
        "session"
    )
    successor = prices.loc[
        prices["security_id"].map(_text).eq(spec.successor_security_id)
    ].sort_values("session")
    joined = old.merge(
        successor,
        on="session",
        suffixes=("_old", "_successor"),
        validate="one_to_one",
    )
    _require(
        len(old) == 1_455 and len(successor) == 1_977 and len(joined) == 1_455,
        "SYMC/NLOK full identity coverage changed.",
    )
    exact = {
        field: int(
            joined[f"{field}_old"].fillna(-math.inf).eq(
                joined[f"{field}_successor"].fillna(-math.inf)
            ).sum()
        )
        for field in ("open", "high", "low", "close", "volume", "currency")
    }
    _require(
        exact
        == {"open": 1_452, "high": 1_455, "low": 1_451, "close": 1_455, "volume": 1_245, "currency": 1_455},
        "SYMC/NLOK full identity equality signature changed.",
    )

    before = spec.transition_date
    old_pre = old.loc[old["session"].map(_date).lt(before)]
    successor_pre = successor.loc[successor["session"].map(_date).lt(before)]
    old_factors = factors.loc[
        factors["security_id"].map(_text).eq(spec.security_id)
        & factors["session"].map(_date).lt(before)
    ]
    successor_factors = factors.loc[
        factors["security_id"].map(_text).eq(spec.successor_security_id)
        & factors["session"].map(_date).lt(before)
    ]
    signals: dict[str, Any] = {}
    for mode in ("raw", "total_return_adjusted"):
        signals[mode] = _signal_diff(
            _signal_frame(old_pre, old_factors, mode=mode),
            _signal_frame(successor_pre, successor_factors, mode=mode),
        )
    _require(
        signals["raw"]["TripleBuySignal"]["count"] == 0
        and signals["raw"]["TripleSellSignal"]["count"] == 0
        and signals["total_return_adjusted"]["TripleBuySignal"]["count"] == 0
        and signals["total_return_adjusted"]["TripleSellSignal"]["count"] == 0
        and signals["total_return_adjusted"]["TripleST1_Trend"]["sessions"]
        == ["2015-08-26"],
        "SYMC/NLOK canonicalization signal signature changed.",
    )
    return {
        "old_sid_rows": len(old),
        "successor_rows": len(successor),
        "old_rows_covered_by_successor": len(joined),
        "full_overlap_field_exact_counts": exact,
        "old_price_inventory_sha256": _frame_sha256(
            old, sort_by=("session",)
        ),
        "successor_price_inventory_sha256": _frame_sha256(
            successor, sort_by=("session",)
        ),
        "pre_transition_triple_supertrend_diff": signals,
        "economic_signal_interpretation": (
            "The canonical NLOK price path changes no TripleBuySignal or "
            "TripleSellSignal before the ticker change.  It changes one adjusted "
            "short-trend state on 2015-08-26 and corrects the displayed historical "
            "ticker from NLOK to SYMC."
        ),
    }


def build_audit(repository: LocalDatasetRepository) -> dict[str, Any]:
    release, _ = repository.current_release()
    _require(release is not None, "Current local release is missing.")
    _require(
        release.version == PINNED_RELEASE_VERSION,
        "Current release changed; re-review the finite identity-tail inventory.",
    )
    for dataset, version in PINNED_DATASET_VERSIONS.items():
        _require(
            release.dataset_versions.get(dataset) == version,
            f"Pinned {dataset} version changed.",
        )
    security_ids = {
        security_id
        for spec in CASES
        for security_id in (spec.security_id, spec.successor_security_id)
    }
    subset_datasets = {
        "adjustment_factors",
        "corporate_actions",
        "daily_price_raw",
        "security_master",
        "symbol_history",
    }
    frames = {
        dataset: (
            _read_security_subset(repository, dataset, version, security_ids)
            if dataset in subset_datasets
            else repository.read_frame(dataset, version)
        )
        for dataset, version in PINNED_DATASET_VERSIONS.items()
    }
    prices = frames["daily_price_raw"]
    factors = frames["adjustment_factors"]
    actions = frames["corporate_actions"]
    master = frames["security_master"]
    history = frames["symbol_history"]
    anchors = frames["index_constituent_anchors"]
    events = frames["index_membership_events"]
    archive = frames["source_archive"]
    calendar = xcals.get_calendar("XNYS")

    cases: list[dict[str, Any]] = []
    for spec in CASES:
        ids = {spec.security_id, spec.successor_security_id}
        action = _one_row(
            actions,
            actions["event_id"].map(_text).eq(spec.event_id),
            f"{spec.symbol} official action",
        )
        _require(
            _text(action["security_id"]) == spec.security_id
            and _text(action["action_type"]) == spec.action_type
            and _date(action["effective_date"]) == spec.transition_date
            and _text(action["new_security_id"]) == spec.successor_security_id
            and _text(action["new_symbol"]).upper() == spec.successor_symbol
            and bool(action["official"])
            and _text(action["source_hash"]).lower() == spec.official_source_hash,
            f"{spec.symbol} official action binding changed.",
        )
        action_rows = actions.loc[actions["event_id"].map(_text).eq(spec.event_id)]
        _require(
            _frame_sha256(action_rows, sort_by=("event_id",))
            == spec.action_row_sha256,
            f"{spec.symbol} action row bytes changed.",
        )
        official_source = _official_archive_binding(
            repository,
            archive,
            source_hash=spec.official_source_hash,
            source_url=_text(action["source_url"]),
        )

        old_master = _one_row(
            master,
            master["security_id"].map(_text).eq(spec.security_id),
            f"{spec.symbol} security master",
        )
        successor_master = _one_row(
            master,
            master["security_id"].map(_text).eq(spec.successor_security_id),
            f"{spec.successor_symbol} security master",
        )
        old_history = _one_row(
            history,
            history["security_id"].map(_text).eq(spec.security_id)
            & history["symbol"].map(_text).str.upper().eq(spec.symbol),
            f"{spec.symbol} symbol history",
        )
        successor_history = _one_row(
            history,
            history["security_id"].map(_text).eq(spec.successor_security_id)
            & history["symbol"].map(_text).str.upper().eq(spec.successor_symbol),
            f"{spec.successor_symbol} symbol history",
        )
        identity_inventory = _identity_inventory(master, history, ids)
        _require(
            _frame_sha256(
                identity_inventory,
                sort_by=("_kind", "security_id"),
            )
            == spec.identity_inventory_sha256,
            f"{spec.symbol} identity inventory changed.",
        )

        old_prices = prices.loc[prices["security_id"].map(_text).eq(spec.security_id)].copy()
        old_prices["_session"] = old_prices["session"].map(_date)
        old_tail = old_prices.loc[old_prices["_session"].ge(spec.transition_date)].drop(
            columns="_session"
        )
        _require(
            len(old_tail) == spec.tail_rows
            and old_tail["session"].map(_date).min() == spec.transition_date
            and old_tail["session"].map(_date).max() == spec.tail_end
            and set(old_tail["source_hash"].map(_text)) == {spec.old_tail_source_hash}
            and _frame_sha256(old_tail, sort_by=("session",)) == spec.old_tail_sha256,
            f"{spec.symbol} old SID tail changed.",
        )
        stored_last_good = old_prices.loc[
            old_prices["_session"].lt(spec.transition_date), "_session"
        ].max()
        previous_xnys = _date(
            calendar.previous_session(pd.Timestamp(spec.transition_date))
        )
        _require(
            stored_last_good == spec.old_last_good_session == previous_xnys,
            f"{spec.symbol} last-good session changed.",
        )

        successor_overlap = prices.loc[
            prices["security_id"].map(_text).eq(spec.successor_security_id)
            & prices["session"].map(_date).isin(set(old_tail["session"].map(_date)))
        ].copy()
        _require(
            len(successor_overlap) == spec.tail_rows
            and set(successor_overlap["source_hash"].map(_text))
            == {spec.successor_overlap_source_hash}
            and _frame_sha256(successor_overlap, sort_by=("session",))
            == spec.successor_overlap_sha256,
            f"{spec.symbol} successor overlap changed.",
        )
        overlap = _price_overlap_diagnostic(spec, old_tail, successor_overlap)

        membership_inventory = _membership_inventory(anchors, events, ids)
        _require(
            _frame_sha256(
                membership_inventory,
                sort_by=("_kind", "index_id", "security_id"),
            )
            == spec.membership_inventory_sha256,
            f"{spec.symbol} membership inventory changed.",
        )
        member_sessions = _tail_member_sessions(
            spec.security_id, old_tail, anchors, events
        )
        direct_count = sum(len(value) for value in member_sessions.values())
        if spec.symbol != "SYMC":
            _require(direct_count == 0, f"{spec.symbol} gained direct tail membership.")

        signal_impact = _successor_reassignment_signal_impact(
            spec, prices, factors, old_tail
        )
        if spec.symbol == "UTX":
            _require(
                signal_impact["total_return_adjusted"]["TripleBuySignal"]["count"] == 2
                and signal_impact["total_return_adjusted"]["TripleSellSignal"]["count"] == 2,
                "UTX/RTX reassignment risk signature changed.",
            )
        elif signal_impact.get("evaluated"):
            _require(
                signal_impact["raw"]["TripleBuySignal"]["count"] == 0
                and signal_impact["raw"]["TripleSellSignal"]["count"] == 0
                and signal_impact["total_return_adjusted"]["TripleBuySignal"]["count"] == 0
                and signal_impact["total_return_adjusted"]["TripleSellSignal"]["count"] == 0,
                f"{spec.symbol} successor replacement signal signature changed.",
            )

        cases.append(
            {
                "symbol": spec.symbol,
                "security_id": spec.security_id,
                "successor_symbol": spec.successor_symbol,
                "successor_security_id": spec.successor_security_id,
                "action_type": spec.action_type,
                "event_id": spec.event_id,
                "transition_date": spec.transition_date,
                "old_last_good_session": spec.old_last_good_session,
                "old_tail": {
                    "rows": spec.tail_rows,
                    "start": spec.transition_date,
                    "end": spec.tail_end,
                    "source_hash": spec.old_tail_source_hash,
                    "projected_sha256": spec.old_tail_sha256,
                },
                "successor_overlap": {
                    "rows": len(successor_overlap),
                    "source_hash": spec.successor_overlap_source_hash,
                    "projected_sha256": spec.successor_overlap_sha256,
                    **overlap,
                },
                "stored_identity": {
                    "old_master_active_from": _date(old_master["active_from"]),
                    "old_master_active_to": _date(old_master["active_to"]),
                    "old_symbol_effective_from": _date(old_history["effective_from"]),
                    "old_symbol_effective_to": _date(old_history["effective_to"]),
                    "successor_master_active_from": _date(successor_master["active_from"]),
                    "successor_master_active_to": _date(successor_master["active_to"]),
                    "successor_symbol_effective_from": _date(successor_history["effective_from"]),
                    "successor_symbol_effective_to": _date(successor_history["effective_to"]),
                    "inventory_sha256": spec.identity_inventory_sha256,
                },
                "official_evidence": official_source,
                "index_membership": {
                    "inventory_sha256": spec.membership_inventory_sha256,
                    "signature": _membership_signature(membership_inventory),
                    "old_sid_member_tail_sessions": member_sessions,
                    "old_sid_member_tail_session_count": direct_count,
                },
                "hypothetical_full_tail_reassignment_signal_impact": signal_impact,
                "safe_minimum_repair": _repair_plan(
                    spec, old_tail, successor_overlap, member_sessions
                ),
            }
        )

    symc_identity = _symc_full_identity_diagnostic(prices, factors)
    _require(
        symc_identity["old_price_inventory_sha256"]
        == "bc3878bbd989b7c2f2d3307a66e93f221e050ac1d14bb507b0354fcdd5160d7d"
        and symc_identity["successor_price_inventory_sha256"]
        == "77f154621cf2f40ca3a8d7a8ace36c06c27fd522845a2ff9058f6212bd184d44",
        "SYMC/NLOK full price inventory changed.",
    )
    repair_classes = pd.Series(
        [case["safe_minimum_repair"]["repair_class"] for case in cases]
    ).value_counts()
    return {
        "schema": AUDIT_SCHEMA,
        "release_version": release.version,
        "dataset_versions": dict(PINNED_DATASET_VERSIONS),
        "network_accessed": False,
        "http_attempts": 0,
        "eodhd_calls": 0,
        "r2_accessed": False,
        "dataset_writes_performed": False,
        "release_pointer_writes_performed": False,
        "generic_tolerance_added": False,
        "summary": {
            "case_count": len(cases),
            "old_sid_tail_rows": sum(case["old_tail"]["rows"] for case in cases),
            "delete_close_cases": int(
                repair_classes.get("delete_old_tail_close_identity", 0)
            ),
            "synthetic_merger_tail_cases": int(
                repair_classes.get("delete_synthetic_merger_tail_close_identity", 0)
            ),
            "successor_first_row_replacement_cases": int(
                repair_classes.get(
                    "replace_successor_first_session_then_delete_old_tail", 0
                )
            ),
            "canonical_identity_cases": int(
                repair_classes.get("canonicalize_nlok_identity_then_retire_old_sid", 0)
            ),
            "direct_index_member_tail_session_count": sum(
                case["index_membership"]["old_sid_member_tail_session_count"]
                for case in cases
            ),
            "direct_index_member_tail_symbols": sorted(
                case["symbol"]
                for case in cases
                if case["index_membership"]["old_sid_member_tail_session_count"]
            ),
        },
        "backtest_interpretation": {
            "delete_only_economic_path_change": False,
            "seven_zero_direct_membership_cases": sorted(
                case["symbol"]
                for case in cases
                if not case["index_membership"]["old_sid_member_tail_session_count"]
            ),
            "symc_requires_coupled_identity_rebind": True,
            "utx_tail_reassignment_is_unsafe": True,
            "hcp_first_session_replacement_triple_signal_changes": 0,
            "scope_note": (
                "This is a signal- and index-membership audit, not a rerun of the "
                "published portfolio.  The proposed delete-only cases remove rows "
                "that the index schedule cannot select.  HCP replacement changes no "
                "Triple Supertrend state; SYMC canonicalization changes no pre-change "
                "buy or sell signal but must be applied with index and symbol history."
            ),
        },
        "symc_full_identity_diagnostic": symc_identity,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/cache")
    parser.add_argument(
        "--output", default="/tmp/us-identity-tail-repair-audit.json"
    )
    args = parser.parse_args()
    repository = LocalDatasetRepository(Path(args.data_root))
    audit = build_audit(repository)
    payload = _canonical_json_bytes(audit)
    write_atomic(Path(args.output), payload)
    print(
        json.dumps(
            {
                "status": "plan_only_complete",
                "output": str(Path(args.output)),
                "audit_sha256": sha256_bytes(payload),
                "summary": audit["summary"],
                "network_accessed": False,
                "dataset_writes_performed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
