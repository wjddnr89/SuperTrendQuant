#!/usr/bin/env python3
"""Read-only free-provider arbitration for seven legacy US price mismatches.

The audit is deliberately unable to modify a dataset, a release pointer, R2,
or the EODHD call ledger.  Optional network acquisition is limited to one
Stooq and one frozen Boris/Kaggle v3 URL per audited symbol.  The complete
missing request set is checked against both per-provider caps and the run-wide
cap before the first HTTP request; neither client retries.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qs, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import duckdb
import numpy as np
import pandas as pd

from supertrend_quant.indicators import add_triple_supertrend
from supertrend_quant.market_store.adjustments import apply_adjustment_factors
from supertrend_quant.market_store.manifest import sha256_bytes, utc_now_iso, write_atomic
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.stooq import StooqHistoricalCache
from supertrend_quant.market_store.yahoo_chart import YahooChartCache


SYMBOLS = ("APC", "HOT", "IR", "LB", "PCL", "POM", "SPLS")
MAX_STOOQ_HTTP_ATTEMPTS = len(SYMBOLS)
MAX_BORIS_HTTP_ATTEMPTS = len(SYMBOLS)
MAX_TOTAL_HTTP_ATTEMPTS = MAX_STOOQ_HTTP_ATTEMPTS + MAX_BORIS_HTTP_ATTEMPTS

DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_STOOQ_CACHE = DEFAULT_CACHE_ROOT / "state/stooq-price-audit"
DEFAULT_BORIS_CACHE = DEFAULT_CACHE_ROOT / "state/boris-kaggle-us-index-identity"
DEFAULT_ATTEMPT_LEDGER = (
    DEFAULT_CACHE_ROOT / "state/free-price-arbiter-audit/http_attempt_ledger.json"
)
DEFAULT_RELEASE = DEFAULT_CACHE_ROOT / "releases/20260715-20260718T230255094849Z.json"
DEFAULT_CROSSVALIDATION_REPORT = Path("/tmp/crossval-current-final.json")
DEFAULT_YAHOO_CACHE = DEFAULT_CACHE_ROOT / "state/us_cross_validation/yahoo_chart"
DEFAULT_WIKI_ZIP = Path("/tmp/marketneutral-quandl-wiki-prices.zip")
DEFAULT_OUTPUT = Path(
    "results/data_quality/us_cross_validation/free_price_arbiter_audit_20260719.json"
)

RELEASE_SHA256 = "9657dc558b571515674d30f6ce6ddd8c38a7d266b2d07ba2f9386ed4e001cfc6"
CROSSVALIDATION_REPORT_SHA256 = (
    "42a5ce96f79a0c8adfc7e49d90f3abac833ff6dca0ac849604abb1db2b6546f0"
)
BASE_RELEASE_VERSION = "20260715-20260718T230255094849Z"
EXPECTED_DATASET_VERSIONS = {
    "daily_price_raw": (
        "early-terminal-history-2026-07-15-566e79bcc7ac4e268c4cc304e14b700e-"
        "daily_price_raw"
    ),
    "adjustment_factors": (
        "early-terminal-history-2026-07-15-566e79bcc7ac4e268c4cc304e14b700e-"
        "adjustment_factors"
    ),
    "security_master": (
        "early-terminal-history-2026-07-15-566e79bcc7ac4e268c4cc304e14b700e-"
        "security_master"
    ),
    "symbol_history": (
        "early-terminal-history-2026-07-15-566e79bcc7ac4e268c4cc304e14b700e-"
        "symbol_history"
    ),
    "source_archive": (
        "wiki-price-arbitration-20260715-301b7adc38334f65a4012a095993dce9-"
        "source_archive"
    ),
}

WIKI_MEMBER = "WIKI_PRICES.csv"
WIKI_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "marketneutral/quandl-wiki-prices-us-equites/WIKI_PRICES.csv"
)
WIKI_ZIP_SHA256 = "36c667bbecf42c43e5b9e8e4e5d9a1268522705bc40a4bac9671d62c1a20cbae"
WIKI_ZIP_SIZE = 463_184_323
WIKI_MEMBER_SHA256 = "ca7fb174c7948db85638917d25ff65d438e27d5cb23675da784c54db01e3d003"
WIKI_MEMBER_SIZE = 1_797_003_576
WIKI_MEMBER_CRC32 = 0x946874CE
WIKI_LICENSE = "Unknown; private/internal-only; redistribution and public publication blocked"

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

# Strategy outputs are pinned independently from the underlying source-byte and
# relation hashes.  This catches indicator/default drift while still emitting
# the exact seven per-session Triple Supertrend field differences in the report.
EXPECTED_BASELINE_SIGNAL_SHA256 = {
    "APC": (
        "14638d5869dd713e5d2f62992460307bfb4fd6ba7dbd0a4bff5e77b8f80140d1",
        "c0ba31726ca221cce215488e247bd3c9b3357488630aac2d32e229b3351b716e",
    ),
    "HOT": (
        "67ee44f5b1c64ca2022ce83224a379aded80b0dfee15187eb9c13f776c7f58ac",
        "4c38414205abda6ba5c3ef03ab9353ad503c4f757294318639958b27bcc4641d",
    ),
    "IR": (
        "90ae60db91376fb14db70bfe8f018a885054723a0006eb9c18b9410bf085c6c6",
        "2788f491311de22940c2ccf73dd6e53a7c2c822fbefa07884023d74fae79208f",
    ),
    "LB": (
        "e2aca0659a69a3caf72b0ec4777ce5b71b1bbba1115e1bad1ed5ada0b2b525d6",
        "5de9c0e40da3ffe0ab304c2d5145a6db306698a84b3b87958ec3007b1d9f65fa",
    ),
    "PCL": (
        "b484883b0e1b5162b4384d9a9449878e058d795b1579a80b4f5179323af53071",
        "3757c51b5cfab62c53f80d1e3572900a1710e341e79f37484cf87642fae4fa68",
    ),
    "POM": (
        "2ae092e49b16972312bed0d3cbaf3e2fae411a9b72a6ebe4642cb8330f6c5e0a",
        "d0ca1af46caeae449048542c130162f2b059b738b834970caf05a3ec257fec04",
    ),
    "SPLS": (
        "16100cebc8b08bb0f3a928dbcd4d64dcf3017ffe1b5d002411ff6efdedf07b98",
        "4b7ca986545686e0343c62f64f2d49785d84509a7d9fab6e7de9c8bc3ab04c1d",
    ),
}

EXPECTED_WIKI_ADJUSTED_DIFF_COUNTS = {
    "APC": (0, 1, 0, 0, 1, 0, 0),
    "HOT": (21, 41, 29, 20, 55, 9, 7),
    "IR": (1, 0, 3, 0, 4, 0, 0),
    "LB": (5, 0, 0, 5, 5, 1, 0),
    "PCL": (0, 3, 0, 0, 3, 0, 1),
    "POM": (0, 1, 1, 1, 1, 1, 0),
    "SPLS": (3, 0, 0, 0, 3, 0, 0),
}

EXPECTED_BORIS_ADJUSTED_DIFF_COUNTS = {
    "APC": (3, 1, 0, 3, 4, 2, 0),
    "IR": (0, 1, 3, 1, 4, 2, 0),
    "LB": (1, 0, 0, 1, 1, 0, 0),
}

EODHD_ARCHIVE_INVENTORY_OVERRIDES = {
    # The immutable provider payload contains a 122-session zero-volume carry
    # tail after the official 2016-09-22 merger.  The release correctly trims
    # it while retaining exact archived bytes for audit.
    "HOT": {"rows": 557, "start": "2015-01-02", "end": "2017-03-20"},
}

BORIS_DATASET_VERSION_URL = (
    "https://www.kaggle.com/datasets/borismarjanovic/"
    "price-volume-data-for-all-us-stocks-etfs/versions/3"
)
BORIS_LICENSE = "CC0: Public Domain"
BORIS_URL_TEMPLATE = (
    "https://www.kaggle.com/api/v1/datasets/download/borismarjanovic/"
    "price-volume-data-for-all-us-stocks-etfs/Data%2FStocks%2F{symbol}.us.txt"
    "?datasetVersionNumber=3"
)

# Filled only after the one-attempt acquisition has captured immutable bytes.
# A normal audit refuses any response whose hash is absent or different.
PINNED_BORIS_SHA256: dict[str, str] = {
    "APC": "e44acd2e028850f9085977a4994e216ecc426c6735cb77746d28dd4537453f0b",
    "IR": "ba2356b248e53f993b80d8192d268a9af8522a6eeb3c1f0d1f8864e4ac817600",
    "LB": "195c53db12a30976c7c051d8a73b667a2d67c76c61a17d452d94d4e8278bd3f5",
    "PCL": "e0ede9ea3729fc573ab4c00ae8934bd659e6088e8883b88a61725f156989602f",
    "POM": "e0ede9ea3729fc573ab4c00ae8934bd659e6088e8883b88a61725f156989602f",
    "SPLS": "e0ede9ea3729fc573ab4c00ae8934bd659e6088e8883b88a61725f156989602f",
}


@dataclass(frozen=True)
class TargetPin:
    symbol: str
    security_id: str
    target_id: str
    master_primary_symbol: str
    master_active_to: str
    history_effective_to: str
    eodhd_source_sha256: str
    eodhd_source_url: str
    eodhd_rows: int
    eodhd_start: str
    eodhd_end: str
    yahoo_source_sha256: str
    yahoo_wrapper_sha256: str
    yahoo_period1: int
    yahoo_period2: int
    yahoo_instrument_type: str
    yahoo_long_name: str
    yahoo_identity_disposition: str
    wiki_rows: int
    wiki_start: str
    wiki_end: str
    wiki_raw_lines_sha256: str
    wiki_extract_sha256: str
    wiki_extract_size: int
    wiki_overlap_rows: int
    wiki_relation_sha256: str
    stooq_source_sha256: str
    stooq_wrapper_sha256: str
    boris_source_sha256: str
    boris_wrapper_sha256: str
    boris_rows: int
    boris_start: str
    boris_end: str
    index_relevance: str


def _target(**values: Any) -> TargetPin:
    return TargetPin(**values)


TARGETS = (
    _target(
        symbol="APC", security_id="US:EODHD:f485cff6-47f0-5c3f-85ec-1c54895aae21",
        target_id="a042abc3b784d48e2c5674ee182357e4fb0dc70053d948cb476b10e9856091b9",
        master_primary_symbol="APC", master_active_to="2019-08-08", history_effective_to="",
        eodhd_source_sha256="90eef3bbeb48107348add4e0b4ea648001870b52d33692d7c444cbe82de5e0ab",
        eodhd_source_url="https://eodhd.com/api/eod/APC_old.US?from=2015-01-01&to=2026-07-15",
        eodhd_rows=1158, eodhd_start="2015-01-02", eodhd_end="2019-08-08",
        yahoo_source_sha256="e6ee01ca7f8ea98dbc594300f430b1e946020c05b7973e3481570e1ee613b12c",
        yahoo_wrapper_sha256="75c0c4266559c4336755dfd9ae529c9ab86eac71245ef8abd7098099ed4bafa7",
        yahoo_period1=1420070400, yahoo_period2=1784160000,
        yahoo_instrument_type="EQUITY", yahoo_long_name="ARKO Petroleum Corp.",
        yahoo_identity_disposition="reused_ticker_wrong_issuer_2026",
        wiki_rows=7952, wiki_start="1986-09-09", wiki_end="2018-03-27",
        wiki_raw_lines_sha256="fb41890836a1802e55cb36fa4700c83cd7ccfce1a28744006015687cd9f39ef9",
        wiki_extract_sha256="2fdbc9bf39054dd0811571ed825417a4742f69a41361a789af64a585ba562e55",
        wiki_extract_size=1021604, wiki_overlap_rows=813,
        wiki_relation_sha256="63cacc3b2bcc7e71dfc35f510bbff86140fcad6a6476094c720a8194653d72c7",
        stooq_source_sha256="98bf1ebd607c79eeb44be8b106ae4a1733396a5b835fface6153148ea7bd2483",
        stooq_wrapper_sha256="76464748f696e4e550203189b3a34a9077f14c2d9c5f839abb35ab93caaae0a7",
        boris_source_sha256="e44acd2e028850f9085977a4994e216ecc426c6735cb77746d28dd4537453f0b",
        boris_wrapper_sha256="e566c342928937f9f81296d01f7eef8ece43dae7f7a656ae704541689b9e9c01",
        boris_rows=7855, boris_start="1986-09-09", boris_end="2017-11-10",
        index_relevance="S&P 500 through 2019-08-09",
    ),
    _target(
        symbol="HOT", security_id="US:EODHD:3073ffd2-9115-5bf6-8bec-fddcd41749e5",
        target_id="118704d83243ecae0cfb838423558950ae729eceba8503b8ae07cece7fa2758f",
        master_primary_symbol="HOT", master_active_to="2016-09-22", history_effective_to="2016-09-22",
        eodhd_source_sha256="8612a211ec6514a093f6d62a9448c82eb75f45778f244f5a67d6c93ee820a40a",
        eodhd_source_url="https://eodhd.com/api/eod/HOT.US?from=2015-01-01&to=2026-07-15",
        eodhd_rows=435, eodhd_start="2015-01-02", eodhd_end="2016-09-22",
        yahoo_source_sha256="ac478ed4db8bd8b768a70da774a64e8ac43c1ff6e2bc6e22a5ae79f1ffeaf642",
        yahoo_wrapper_sha256="5f43c58aeaffdb58efe87baa8fbb260125843f92f7dc0b7aa9827632cd56e742",
        yahoo_period1=1420070400, yahoo_period2=1474588800,
        yahoo_instrument_type="MUTUALFUND", yahoo_long_name="",
        yahoo_identity_disposition="retired_yhd_metadata_with_55_null_bars",
        wiki_rows=7280, wiki_start="1987-11-05", wiki_end="2016-09-22",
        wiki_raw_lines_sha256="5fe823edd493eccefa09a218890a1e3d1a4f89c919e6b25d5c2252508038263e",
        wiki_extract_sha256="9e85b82f0c6fe1138fed54eea99bca884c69b110d5c206270b60d7f28a3f3b81",
        wiki_extract_size=996768, wiki_overlap_rows=435,
        wiki_relation_sha256="4861a53bda386a5c2c6db45817adb5ce429403ba0aa643fcda097652e6c4fa70",
        stooq_source_sha256="d1aeec9d4af46e39fdcea4871108d9dd85c9f6c920984d274cb4a765ea36450b",
        stooq_wrapper_sha256="e08455ec4a3a68105b1541471f97eb08582168d4bcef637c3351beea3a0cdd1e",
        boris_source_sha256="", boris_wrapper_sha256="", boris_rows=0, boris_start="", boris_end="",
        index_relevance="S&P 500 through the 2016 Marriott acquisition",
    ),
    _target(
        symbol="IR", security_id="US:EODHD:cb64587f-5f98-5931-adbf-9804aff1bcf0",
        target_id="34467b4f4e7c4d0a0b3b41d1b38bea89a369cdda27c3eaae0acdd401f1b7bb18",
        master_primary_symbol="TT", master_active_to="", history_effective_to="2020-02-28",
        eodhd_source_sha256="2aa4f00c6ca844b4df25f21bb236125129a4528a71e352be33b5e8dbe835b6b5",
        eodhd_source_url="https://eodhd.com/api/eod/TT.US?from=2015-01-01&to=2026-07-15",
        eodhd_rows=2899, eodhd_start="2015-01-02", eodhd_end="2026-07-15",
        yahoo_source_sha256="00f45d7a4617a4a8f96574bd6c5d729ed510261e44a160b155ae0a288011d729",
        yahoo_wrapper_sha256="26e219197c2265e8521736106885c7417c1df13b791dfe3130b55e10826a8438",
        yahoo_period1=1420070400, yahoo_period2=1582934400,
        yahoo_instrument_type="EQUITY", yahoo_long_name="Ingersoll Rand Inc.",
        yahoo_identity_disposition="reused_ticker_wrong_issuer_gardner_denver",
        wiki_rows=8252, wiki_start="1985-07-01", wiki_end="2018-03-27",
        wiki_raw_lines_sha256="ee91581e935aa62f86a960717916a9995cdf8fdb7cc6245eb7c283e33f02ee18",
        wiki_extract_sha256="b1d8cdb334fb5156d02adfdf31d6ad1ac9375f6cf08d09fac942397b872f7a7d",
        wiki_extract_size=1050997, wiki_overlap_rows=813,
        wiki_relation_sha256="8dcb73e5e3e56499c820aab9e4dc7101da224ab092ec9b67c4acf394838fa1a6",
        stooq_source_sha256="a5b8456adbcfa2b17dea796aaab5ffc07b946bd05341447ef007249ed68480e7",
        stooq_wrapper_sha256="69de9cc64dfc5be27810c361275723bf6f6d3059da0b83fc61dcaa5981a4c6eb",
        boris_source_sha256="ba2356b248e53f993b80d8192d268a9af8522a6eeb3c1f0d1f8864e4ac817600",
        boris_wrapper_sha256="5be7ad375822d1c8838cf9e1e773d1038a365c54ec142a4211200bd09f2ba15b",
        boris_rows=8160, boris_start="1985-07-01", boris_end="2017-11-10",
        index_relevance="legacy S&P 500 lineage continues as TT",
    ),
    _target(
        symbol="LB", security_id="US:EODHD:e144ef86-76af-5fee-9041-4effc6d321bc",
        target_id="65f58d24688230cd3a598ee9cc92a2f483db68ace1c27fa6fdc116cdf397097a",
        master_primary_symbol="LB", master_active_to="2021-08-02", history_effective_to="",
        eodhd_source_sha256="c3e4ce3289d31ee04bd0f97359925bda66a69eb1c0b656644bc57696244b7901",
        eodhd_source_url="https://eodhd.com/api/eod/LB_old1.US?from=2015-01-01&to=2026-07-15",
        eodhd_rows=1657, eodhd_start="2015-01-02", eodhd_end="2021-08-02",
        yahoo_source_sha256="83bc794550b34bc93c6fd11385881678447d54ba49585a12a38288e257e0226d",
        yahoo_wrapper_sha256="c57c31b54ccbedd42447c33e4c86956837a727de2d146f9084eea3b7f15a5f6b",
        yahoo_period1=1420070400, yahoo_period2=1784160000,
        yahoo_instrument_type="EQUITY", yahoo_long_name="LandBridge Company LLC",
        yahoo_identity_disposition="reused_ticker_wrong_issuer_2024",
        wiki_rows=8252, wiki_start="1985-07-01", wiki_end="2018-03-27",
        wiki_raw_lines_sha256="7c32382df55f786f9c43f216f34b177158c2272ec32b7c488aba2f980694bd75",
        wiki_extract_sha256="4e392160964bb55a4585115db2660039f7621a32b8ba67acd8dd30e477de45ed",
        wiki_extract_size=1052505, wiki_overlap_rows=813,
        wiki_relation_sha256="030c2b63f62928a561b586cc65dd55ae992d709cb04bfe5e87a6ef57747b9890",
        stooq_source_sha256="6bcbb8320818e6e0e391b9d20573ce827dd8f47676b34488e3a48a0c2256382d",
        stooq_wrapper_sha256="1a143ace212847abc4fe5c28740286601fd29f8ca13879a9caf45a8ecc74fb91",
        boris_source_sha256="195c53db12a30976c7c051d8a73b667a2d67c76c61a17d452d94d4e8278bd3f5",
        boris_wrapper_sha256="f9795108c7174af67579d0fb4939b022565554bb7bad601b686e077e9bf99b55",
        boris_rows=8160, boris_start="1985-07-01", boris_end="2017-11-10",
        index_relevance="S&P 500 through 2021-08-03 ticker transition",
    ),
    _target(
        symbol="PCL", security_id="US:EODHD:bd9648b7-1b95-5f55-a777-1c7d660cd2db",
        target_id="6641ad2fd50e5a4028b06df94f31600c98c43d08e19315eb938907f4abaaa87f",
        master_primary_symbol="PCL", master_active_to="2016-02-19", history_effective_to="",
        eodhd_source_sha256="8c886049acfdd0bf097225bce70de4512e1fb5dabfee6ead3d3d2bd35babbcea",
        eodhd_source_url="https://eodhd.com/api/eod/PCL_old.US?from=2015-01-01&to=2026-07-15",
        eodhd_rows=285, eodhd_start="2015-01-02", eodhd_end="2016-02-19",
        yahoo_source_sha256="b0fa62a2b50f09a7009ec3f9af94af637a4cbc3b13120a35ab9e17358378aa52",
        yahoo_wrapper_sha256="ffe2e1c8c86de30a630757fb51d1895ba084dbff596da93424623c8110117caf",
        yahoo_period1=1420070400, yahoo_period2=1784160000,
        yahoo_instrument_type="ETF", yahoo_long_name="PGIM Corporate Bond 10+ Year ETF",
        yahoo_identity_disposition="reused_ticker_etf_collision_2025",
        wiki_rows=6733, wiki_start="1989-06-02", wiki_end="2016-02-19",
        wiki_raw_lines_sha256="55c6d5a97129ea244a97650b1c3a93bd8809ae732fe3bfdc46253a31957f0410",
        wiki_extract_sha256="cd716ef088a30b3d3cbdfa36f3eed92287a7825dd87b0e931d8092429db6a466",
        wiki_extract_size=926076, wiki_overlap_rows=285,
        wiki_relation_sha256="83fdf628a2544c7fd32b69d50f2a52f0f9b038a587405803430e88435392a4ea",
        stooq_source_sha256="ee2ee2845e5bda0c4f5e045dda4ce5233bb9e84b430b7d7b6d0b5c6525f8e849",
        stooq_wrapper_sha256="c0f08ff8d69ea3bb1a6d810d13e48a496bd01a9574b2a4cdb88d89ce917f61db",
        boris_source_sha256="e0ede9ea3729fc573ab4c00ae8934bd659e6088e8883b88a61725f156989602f",
        boris_wrapper_sha256="08b07a66e83a86bca0b55c548e349af877962c1d9f78c80a27c97ceb982ed495",
        boris_rows=0, boris_start="", boris_end="",
        index_relevance="S&P 500 through 2016-02-22",
    ),
    _target(
        symbol="POM", security_id="US:EODHD:9f2cfe0f-b5b2-5b9e-8685-6fc38307afd3",
        target_id="e1762ec53fe21e95d39adf0d225656ab39d22857daaebee109e3de74cfa7aad7",
        master_primary_symbol="POM", master_active_to="2016-03-23", history_effective_to="",
        eodhd_source_sha256="a3d79c7ffe208cbe3be44ee3f7e7502b39ef54cae4afddb6088f10c4bba695e1",
        eodhd_source_url="https://eodhd.com/api/eod/POM_old.US?from=2015-01-01&to=2026-07-15",
        eodhd_rows=308, eodhd_start="2015-01-02", eodhd_end="2016-03-23",
        yahoo_source_sha256="ab59b1c1328c15026286e7a03b116830395b545d40f79d8fd21549ccea4097d6",
        yahoo_wrapper_sha256="824b4dcd3dfc4cef8661a3e1c89a2c02375f542b78fd041625cb269f43a9d63a",
        yahoo_period1=1420070400, yahoo_period2=1784160000,
        yahoo_instrument_type="EQUITY", yahoo_long_name="Pomdoctor Limited",
        yahoo_identity_disposition="reused_ticker_wrong_issuer_2025",
        wiki_rows=7367, wiki_start="1987-01-02", wiki_end="2016-03-23",
        wiki_raw_lines_sha256="b07f1667e887ecc16b02d39e32b9845a6b69e13273506475fe3e67676237a23d",
        wiki_extract_sha256="78ad1a696e69210e0bdbd2850e9c2628b710b7ec9c8cb7a5e8bd4b35cf2edf81",
        wiki_extract_size=1014090, wiki_overlap_rows=308,
        wiki_relation_sha256="38025bf218f84db84a40bb7c920e578f1af9141803a827c78a8e6f39d5623084",
        stooq_source_sha256="42477741b9295bc1fd6d86932e0d0e3f5cddfed18ff63d335a557e15c5c74b3d",
        stooq_wrapper_sha256="6e58f97ba94a98b5e7c89f8de8f15ce5859dec2a520b2792344bf19debbec21a",
        boris_source_sha256="e0ede9ea3729fc573ab4c00ae8934bd659e6088e8883b88a61725f156989602f",
        boris_wrapper_sha256="7e42a2bddaf3680f931b3b2ea6778f4ffb48a31dc38773a2d4fc9c65aa979742",
        boris_rows=0, boris_start="", boris_end="",
        index_relevance="S&P 500 through 2016-03-24",
    ),
    _target(
        symbol="SPLS", security_id="US:EODHD:591b1e97-ff78-5a6f-806d-0bb7885d2231",
        target_id="bdd64d073e1941ac21b8edc31bd542204a1adf85d561408ebac51a068fd17728",
        master_primary_symbol="SPLS", master_active_to="2017-09-12", history_effective_to="",
        eodhd_source_sha256="add2e21b817013126c8af70207310e579c849987bebee77f9f065d492bb2a649",
        eodhd_source_url="https://eodhd.com/api/eod/SPLS_old.US?from=2015-01-01&to=2026-07-15",
        eodhd_rows=679, eodhd_start="2015-01-02", eodhd_end="2017-09-12",
        yahoo_source_sha256="eddbafaee6f591e8abf58819a429856efcb14c175148b8cce173e16cf2cf2786",
        yahoo_wrapper_sha256="2f3aaf0943fc1bb18d0cf09c486d8d708356d357f76d6237c67933c0916005a0",
        yahoo_period1=1420070400, yahoo_period2=1784160000,
        yahoo_instrument_type="ETF", yahoo_long_name="PIMCO US Stocks PLUS Active Bond Exchange Traded Fund",
        yahoo_identity_disposition="reused_ticker_etf_collision_2026",
        wiki_rows=6922, wiki_start="1990-03-26", wiki_end="2017-09-12",
        wiki_raw_lines_sha256="1d9938cef66e9a93b918408bcc3164be04f599c40bddafabe80d39a717b7d72a",
        wiki_extract_sha256="9e2161067b3863db1d3e71cbd3ecb59776833d7327361287fa18e4ae68b83ee9",
        wiki_extract_size=911447, wiki_overlap_rows=679,
        wiki_relation_sha256="c6959488b034ab7156f49b960ab0c2a46a1d7b06514b37299cdc79ab865978bd",
        stooq_source_sha256="f7446ca3cde329d66c58a08351d12fc13acc3ba0b9d434878bb920b83d43de02",
        stooq_wrapper_sha256="f2acd96b3ed53000f461e87bcef318378f112599be0768d158820b57ada515d4",
        boris_source_sha256="e0ede9ea3729fc573ab4c00ae8934bd659e6088e8883b88a61725f156989602f",
        boris_wrapper_sha256="38092f9c387a2627560b9a045930a7b226ea8fb65eaff8760866e3a09c5be0e5",
        boris_rows=0, boris_start="", boris_end="",
        index_relevance="S&P 500 and Nasdaq-100 legacy membership",
    ),
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _normalize_symbol(symbol: str) -> str:
    value = str(symbol).strip().upper()
    if value not in SYMBOLS:
        raise ValueError(f"Unsupported audited symbol: {symbol!r}")
    return value


@dataclass(frozen=True)
class BorisCachedResponse:
    symbol: str
    source_url: str
    retrieved_at: str
    content: bytes
    content_type: str
    http_status: int

    @property
    def source_hash(self) -> str:
        return sha256_bytes(self.content)


class BorisPriceAuditCache:
    """Seven-attempt, no-retry immutable cache for frozen version-3 files."""

    SCHEMA = "boris_kaggle_cc0_raw_response/v1"

    def __init__(
        self,
        root: Path,
        *,
        max_http_attempts: int = MAX_BORIS_HTTP_ATTEMPTS,
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 10 * 1024 * 1024,
        permit_initial_unpinned_capture: bool = False,
    ):
        if not 0 < int(max_http_attempts) <= MAX_BORIS_HTTP_ATTEMPTS:
            raise ValueError("Boris HTTP attempt cap must be between one and seven.")
        if timeout_seconds <= 0 or max_response_bytes <= 0:
            raise ValueError("Boris timeout/response cap must be positive.")
        self.root = Path(root)
        self.max_http_attempts = int(max_http_attempts)
        self.timeout_seconds = float(timeout_seconds)
        self.max_response_bytes = int(max_response_bytes)
        self.permit_initial_unpinned_capture = bool(permit_initial_unpinned_capture)
        self.http_attempts = 0

    def url(self, symbol: str) -> str:
        return BORIS_URL_TEMPLATE.format(symbol=_normalize_symbol(symbol).lower())

    def path(self, symbol: str) -> Path:
        return self.root / f"{sha256_bytes(self.url(symbol).encode())}.json.gz"

    def _decode(self, symbol: str, encoded: bytes) -> BorisCachedResponse:
        normalized = _normalize_symbol(symbol)
        try:
            envelope = json.loads(gzip.decompress(encoded))
            payload = envelope["payload"]
            content = base64.b64decode(payload["content_base64"], validate=True)
        except Exception as exc:
            raise RuntimeError(f"Invalid Boris cache envelope: {self.path(symbol)}") from exc
        if envelope.get("schema") != self.SCHEMA or not isinstance(payload, dict):
            raise RuntimeError("Wrong Boris cache schema.")
        if envelope.get("payload_sha256") != sha256_bytes(_canonical_json_bytes(payload)):
            raise RuntimeError("Boris cache payload hash mismatch.")
        if payload.get("symbol") != normalized or payload.get("source_url") != self.url(normalized):
            raise RuntimeError("Boris cache identity changed.")
        if payload.get("content_sha256") != sha256_bytes(content):
            raise RuntimeError("Boris cache content hash mismatch.")
        pinned = PINNED_BORIS_SHA256.get(normalized)
        if pinned and pinned != sha256_bytes(content):
            raise RuntimeError(f"Pinned Boris content changed for {normalized}.")
        if not pinned and not self.permit_initial_unpinned_capture:
            raise RuntimeError(f"Boris content hash is not pinned for {normalized}.")
        return BorisCachedResponse(
            symbol=normalized,
            source_url=self.url(normalized),
            retrieved_at=str(payload["retrieved_at"]),
            content=content,
            content_type=str(payload.get("content_type") or ""),
            http_status=int(payload["http_status"]),
        )

    def get(self, symbol: str) -> BorisCachedResponse | None:
        path = self.path(symbol)
        return self._decode(symbol, path.read_bytes()) if path.is_file() else None

    def fetch(self, symbol: str) -> BorisCachedResponse:
        normalized = _normalize_symbol(symbol)
        if not self.permit_initial_unpinned_capture:
            raise RuntimeError("Unpinned Boris acquisition was not explicitly enabled.")
        if self.http_attempts >= self.max_http_attempts:
            raise RuntimeError("Boris HTTP attempt cap reached.")
        self.http_attempts += 1
        request = Request(
            self.url(normalized),
            headers={
                "Accept": "text/csv,text/plain,application/octet-stream",
                "User-Agent": "SuperTrendQuant free-price-audit/1.0",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                status = int(getattr(response, "status", 200))
                content_type = str(response.headers.get("Content-Type") or "")
                content = response.read(self.max_response_bytes + 1)
        except HTTPError as exc:
            status = int(exc.code)
            content_type = str(exc.headers.get("Content-Type") or "")
            content = exc.read(self.max_response_bytes + 1)
        except URLError as exc:
            raise RuntimeError(
                f"Boris single HTTP attempt failed for {normalized}: {exc.reason}"
            ) from None
        if len(content) > self.max_response_bytes:
            raise RuntimeError("Boris response exceeds configured byte cap.")
        value = BorisCachedResponse(
            symbol=normalized,
            source_url=self.url(normalized),
            retrieved_at=utc_now_iso(),
            content=content,
            content_type=content_type,
            http_status=status,
        )
        payload = {
            "symbol": value.symbol,
            "source_url": value.source_url,
            "retrieved_at": value.retrieved_at,
            "http_status": value.http_status,
            "content_type": value.content_type,
            "content_sha256": value.source_hash,
            "content_base64": base64.b64encode(value.content).decode("ascii"),
        }
        envelope = {
            "schema": self.SCHEMA,
            "payload": payload,
            "payload_sha256": sha256_bytes(_canonical_json_bytes(payload)),
        }
        destination = self.path(normalized)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = self._decode(normalized, destination.read_bytes())
            if existing.content != content:
                raise RuntimeError("Boris cache changed for one pinned URL.")
            return existing
        write_atomic(destination, gzip.compress(_canonical_json_bytes(envelope), mtime=0))
        return self._decode(normalized, destination.read_bytes())

    def fill_missing(self, symbols: Iterable[str]) -> dict[str, BorisCachedResponse]:
        ordered = tuple(dict.fromkeys(_normalize_symbol(item) for item in symbols))
        missing = [symbol for symbol in ordered if self.get(symbol) is None]
        remaining = self.max_http_attempts - self.http_attempts
        if len(missing) > remaining:
            raise RuntimeError(
                "Boris request set exceeds its run cap before network access: "
                f"{len(missing)} > {remaining}."
            )
        return {symbol: self.get(symbol) or self.fetch(symbol) for symbol in ordered}


def acquire_missing_evidence(
    *,
    stooq_root: Path,
    boris_root: Path,
    attempt_ledger_path: Path = DEFAULT_ATTEMPT_LEDGER,
) -> dict[str, Any]:
    stooq = StooqHistoricalCache(
        stooq_root,
        max_http_attempts=MAX_STOOQ_HTTP_ATTEMPTS,
    )
    boris = BorisPriceAuditCache(
        boris_root,
        max_http_attempts=MAX_BORIS_HTTP_ATTEMPTS,
        permit_initial_unpinned_capture=True,
    )
    if attempt_ledger_path.is_file():
        ledger = json.loads(attempt_ledger_path.read_text(encoding="utf-8"))
    else:
        ledger = {"schema": "us_free_price_arbiter_http_attempt_ledger/v1", "failures": []}
    if ledger.get("schema") != "us_free_price_arbiter_http_attempt_ledger/v1":
        raise RuntimeError("Free-price HTTP attempt ledger schema changed.")
    failures = list(ledger.get("failures") or [])
    failed_stooq = {
        _normalize_symbol(item["symbol"])
        for item in failures
        if item.get("provider") == "stooq" and not item.get("cache_written")
    }
    failed_boris = {
        _normalize_symbol(item["symbol"])
        for item in failures
        if item.get("provider") == "boris" and not item.get("cache_written")
    }
    stooq_cached = {symbol: stooq.get(symbol) for symbol in SYMBOLS}
    boris_cached = {symbol: boris.get(symbol) for symbol in SYMBOLS}
    missing_stooq = [
        symbol for symbol in SYMBOLS
        if stooq_cached[symbol] is None and symbol not in failed_stooq
    ]
    missing_boris = [
        symbol for symbol in SYMBOLS
        if boris_cached[symbol] is None and symbol not in failed_boris
    ]
    total_missing = len(missing_stooq) + len(missing_boris)
    if len(missing_stooq) > MAX_STOOQ_HTTP_ATTEMPTS:
        raise RuntimeError("Stooq missing set exceeds its full-run cap.")
    if len(missing_boris) > MAX_BORIS_HTTP_ATTEMPTS:
        raise RuntimeError("Boris missing set exceeds its full-run cap.")
    if total_missing > MAX_TOTAL_HTTP_ATTEMPTS:
        raise RuntimeError("Combined missing set exceeds the 14-attempt run cap.")
    spent_stooq = sum(value is not None for value in stooq_cached.values()) + len(failed_stooq)
    spent_boris = sum(value is not None for value in boris_cached.values()) + len(failed_boris)
    if spent_stooq + len(missing_stooq) > MAX_STOOQ_HTTP_ATTEMPTS:
        raise RuntimeError("Historical plus planned Stooq attempts exceed seven.")
    if spent_boris + len(missing_boris) > MAX_BORIS_HTTP_ATTEMPTS:
        raise RuntimeError("Historical plus planned Boris attempts exceed seven.")
    if spent_stooq + spent_boris + total_missing > MAX_TOTAL_HTTP_ATTEMPTS:
        raise RuntimeError("Historical plus planned HTTP attempts exceed fourteen.")

    # Both complete sets were proven bounded before either provider is touched.
    stooq_values = stooq.fill_missing(
        symbol for symbol in SYMBOLS if symbol not in failed_stooq
    )
    boris_values = boris.fill_missing(
        symbol for symbol in SYMBOLS if symbol not in failed_boris
    )
    return {
        "schema": "us_free_price_arbiter_acquisition/v1",
        "symbols": list(SYMBOLS),
        "preflight": {
            "missing_stooq": missing_stooq,
            "missing_boris": missing_boris,
            "missing_total": total_missing,
            "spent_before_run": {
                "stooq": spent_stooq,
                "boris": spent_boris,
                "total": spent_stooq + spent_boris,
            },
            "maximum_stooq_http_attempts": MAX_STOOQ_HTTP_ATTEMPTS,
            "maximum_boris_http_attempts": MAX_BORIS_HTTP_ATTEMPTS,
            "maximum_total_http_attempts": MAX_TOTAL_HTTP_ATTEMPTS,
            "retry_count": 0,
        },
        "http_attempts": {
            "stooq": stooq.http_attempts,
            "boris": boris.http_attempts,
            "total": stooq.http_attempts + boris.http_attempts,
        },
        "cumulative_http_attempts": {
            "stooq": spent_stooq + stooq.http_attempts,
            "boris": spent_boris + boris.http_attempts,
            "total": spent_stooq + spent_boris + stooq.http_attempts + boris.http_attempts,
        },
        "responses": {
            symbol: {
                "stooq": {
                    "source_url": stooq_values[f"{symbol.lower()}.us"].source_url,
                    "http_status": stooq_values[f"{symbol.lower()}.us"].http_status,
                    "content_type": stooq_values[f"{symbol.lower()}.us"].content_type,
                    "retrieved_at": stooq_values[f"{symbol.lower()}.us"].retrieved_at,
                    "source_sha256": stooq_values[f"{symbol.lower()}.us"].source_hash,
                    "size": len(stooq_values[f"{symbol.lower()}.us"].content),
                },
                "boris": ({
                    "source_url": boris_values[symbol].source_url,
                    "http_status": boris_values[symbol].http_status,
                    "content_type": boris_values[symbol].content_type,
                    "retrieved_at": boris_values[symbol].retrieved_at,
                    "source_sha256": boris_values[symbol].source_hash,
                    "size": len(boris_values[symbol].content),
                } if symbol in boris_values else next(
                    item for item in failures
                    if item.get("provider") == "boris" and item.get("symbol") == symbol
                )),
            }
            for symbol in SYMBOLS
        },
        "boris_license": BORIS_LICENSE,
        "boris_license_url": BORIS_DATASET_VERSION_URL,
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return sha256_bytes(
        (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )


def _date(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    _require(not pd.isna(parsed), "Evidence contains an invalid date.")
    return pd.Timestamp(parsed).date().isoformat()


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _one(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    rows = frame.loc[mask]
    _require(len(rows) == 1, f"{label} inventory changed: expected=1 observed={len(rows)}.")
    return rows.iloc[0]


def _read_subset(
    repository: LocalDatasetRepository,
    dataset: str,
    version: str,
    security_ids: Sequence[str],
) -> pd.DataFrame:
    paths = [str(path) for path in repository.parquet_paths(dataset, version)]
    _require(bool(paths), f"{dataset} has no Parquet files.")
    placeholders = ",".join("?" for _ in security_ids)
    return duckdb.execute(
        f"select * from read_parquet(?) where security_id in ({placeholders})",
        [paths, *security_ids],
    ).fetchdf()


def _load_release_and_report(
    release_path: Path,
    report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(release_path.is_file(), "Pinned release file is missing.")
    _require(_sha256_file(release_path) == RELEASE_SHA256, "Pinned release bytes changed.")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    _require(release.get("version") == BASE_RELEASE_VERSION, "Release version changed.")
    for dataset, version in EXPECTED_DATASET_VERSIONS.items():
        _require(
            release.get("dataset_versions", {}).get(dataset) == version,
            f"Pinned release {dataset} version changed.",
        )
    _require(report_path.is_file(), "Pinned cross-validation report is missing.")
    _require(
        _sha256_file(report_path) == CROSSVALIDATION_REPORT_SHA256,
        "Pinned cross-validation report bytes changed.",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("base_release_version") == BASE_RELEASE_VERSION, "Report release changed.")
    _require(report.get("status") == "incomplete", "Baseline cross-validation status changed.")
    for target in TARGETS:
        rows = [row for row in report["prices"] if row.get("target_id") == target.target_id]
        _require(len(rows) == 1, f"{target.symbol} report target inventory changed.")
        row = rows[0]
        _require(row.get("status") == "mismatch", f"{target.symbol} is no longer a mismatch.")
        _require(row.get("security_id") == target.security_id, f"{target.symbol} SID changed.")
        _require(row.get("source_sha256") == target.yahoo_source_sha256, f"{target.symbol} Yahoo bytes changed.")
        _require(row.get("cache_wrapper_sha256") == target.yahoo_wrapper_sha256, f"{target.symbol} Yahoo wrapper changed.")
    return release, report


def _load_wiki_bundle(path: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    _require(path.is_file(), "Frozen WIKI ZIP is missing.")
    _require(path.stat().st_size == WIKI_ZIP_SIZE, "Frozen WIKI ZIP size changed.")
    _require(_sha256_file(path) == WIKI_ZIP_SHA256, "Frozen WIKI ZIP hash changed.")
    target_by_symbol = {target.symbol: target for target in TARGETS}
    lines: dict[str, list[bytes]] = {symbol: [] for symbol in target_by_symbol}
    digests = {symbol: hashlib.sha256() for symbol in target_by_symbol}
    member_digest = hashlib.sha256()
    header = b""
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        _require(len(infos) == 1 and infos[0].filename == WIKI_MEMBER, "WIKI member changed.")
        info = infos[0]
        _require(
            info.file_size == WIKI_MEMBER_SIZE and info.CRC == WIKI_MEMBER_CRC32,
            "WIKI member size/CRC changed.",
        )
        with archive.open(info) as handle:
            for number, line in enumerate(handle, start=1):
                member_digest.update(line)
                if number == 1:
                    _require(
                        line.startswith(b"ticker,date,open,high,low,close,volume,"),
                        "WIKI header changed.",
                    )
                    header = line
                    continue
                symbol = line.split(b",", 1)[0].decode("ascii", errors="ignore")
                if symbol in lines:
                    lines[symbol].append(line)
                    digests[symbol].update(line)
    _require(member_digest.hexdigest() == WIKI_MEMBER_SHA256, "WIKI member hash changed.")
    frames: dict[str, pd.DataFrame] = {}
    evidence: list[dict[str, Any]] = []
    for target in TARGETS:
        payload = header + b"".join(lines[target.symbol])
        _require(len(lines[target.symbol]) == target.wiki_rows, f"{target.symbol} WIKI rows changed.")
        _require(
            digests[target.symbol].hexdigest() == target.wiki_raw_lines_sha256,
            f"{target.symbol} WIKI raw-line hash changed.",
        )
        _require(sha256_bytes(payload) == target.wiki_extract_sha256, f"{target.symbol} WIKI extract hash changed.")
        _require(len(payload) == target.wiki_extract_size, f"{target.symbol} WIKI extract size changed.")
        frame = pd.read_csv(io.BytesIO(payload))
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.date.astype(str)
        _require(not frame["date"].duplicated().any(), f"{target.symbol} WIKI dates duplicated.")
        _require(
            frame["date"].min() == target.wiki_start and frame["date"].max() == target.wiki_end,
            f"{target.symbol} WIKI date inventory changed.",
        )
        frames[target.symbol] = frame
        evidence.append(
            {
                "symbol": target.symbol,
                "rows": len(frame),
                "start": target.wiki_start,
                "end": target.wiki_end,
                "raw_lines_sha256": target.wiki_raw_lines_sha256,
                "extract_sha256": target.wiki_extract_sha256,
                "extract_size": target.wiki_extract_size,
            }
        )
    return frames, {
        "source_url": WIKI_DOWNLOAD_URL,
        "zip_sha256": WIKI_ZIP_SHA256,
        "zip_size": WIKI_ZIP_SIZE,
        "member_sha256": WIKI_MEMBER_SHA256,
        "member_size": WIKI_MEMBER_SIZE,
        "member_crc32": f"{WIKI_MEMBER_CRC32:08x}",
        "license": WIKI_LICENSE,
        "extracts": evidence,
    }


def _relation_fingerprint(joined: pd.DataFrame) -> str:
    records: list[list[str]] = []
    for row in joined.sort_values("date").itertuples(index=False):
        records.append(
            [str(row.date)]
            + [format(float(getattr(row, f"{column}_eod")), ".17g") for column in ("open", "high", "low", "close", "volume")]
            + [format(float(getattr(row, f"{column}_wiki")), ".17g") for column in ("open", "high", "low", "close", "volume")]
        )
    return _canonical_sha(records)


def _numeric_relation(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_suffix: str,
    right_suffix: str,
) -> dict[str, Any]:
    columns = ("open", "high", "low", "close")
    joined = left[["date", *columns]].merge(
        right[["date", *columns]], on="date", suffixes=(left_suffix, right_suffix), validate="one_to_one"
    ).sort_values("date", ignore_index=True)
    metrics: dict[str, Any] = {}
    for column in columns:
        x = pd.to_numeric(joined[f"{column}{left_suffix}"], errors="raise")
        y = pd.to_numeric(joined[f"{column}{right_suffix}"], errors="raise")
        ratio = x / y
        median = float(ratio.median())
        metrics[column] = {
            "exact_rows": int(np.isclose(x, y, rtol=0.0, atol=1e-12).sum()),
            "median_scale_left_to_right": median,
            "max_scale_relative_deviation": float((ratio / median - 1.0).abs().max()),
            "return_correlation": float(x.pct_change().corr(y.pct_change())),
        }
    records = [
        [
            str(row.date),
            *[format(float(getattr(row, f"{column}{left_suffix}")), ".17g") for column in columns],
            *[format(float(getattr(row, f"{column}{right_suffix}")), ".17g") for column in columns],
        ]
        for row in joined.itertuples(index=False)
    ]
    return {
        "rows": len(joined),
        "start": "" if joined.empty else str(joined["date"].min()),
        "end": "" if joined.empty else str(joined["date"].max()),
        "relation_sha256": _canonical_sha(records),
        "ohlc": metrics,
    }


def _signal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = pd.DataFrame(
        {
            "Date": pd.to_datetime(frame["session"], errors="raise"),
            "Open": pd.to_numeric(frame["open"], errors="raise"),
            "High": pd.to_numeric(frame["high"], errors="raise"),
            "Low": pd.to_numeric(frame["low"], errors="raise"),
            "Close": pd.to_numeric(frame["close"], errors="raise"),
            "Volume": pd.to_numeric(frame["volume"], errors="raise"),
        }
    )
    return add_triple_supertrend(
        prepared,
        settings=TRIPLE_SETTINGS,
        atr_method="wilder",
        exit_down_count=2,
    )


def _signal_hash(frame: pd.DataFrame) -> str:
    records: list[list[Any]] = []
    for date, values in zip(
        frame["Date"].dt.date.astype(str),
        frame[list(SIGNAL_COLUMNS)].itertuples(index=False, name=None),
        strict=True,
    ):
        records.append(
            [date, *[bool(value) if isinstance(value, (bool, np.bool_)) else int(value) for value in values]]
        )
    return _canonical_sha(records)


def signal_differences(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    _require(tuple(left["Date"]) == tuple(right["Date"]), "Signal sessions changed.")
    result: dict[str, Any] = {}
    for column in SIGNAL_COLUMNS:
        mask = ~left[column].eq(right[column])
        result[column] = {
            "count": int(mask.sum()),
            "sessions": left.loc[mask, "Date"].dt.date.astype(str).tolist(),
        }
    return result


def _signal_difference_counts(differences: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(int(differences[column]["count"]) for column in SIGNAL_COLUMNS)


def _replace_prices(base: pd.DataFrame, replacement: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    output = base.sort_values("session", ignore_index=True).copy()
    output["_date"] = pd.to_datetime(output["session"], errors="raise").dt.date.astype(str)
    replacement = replacement.copy()
    replacement["date"] = replacement["date"].astype(str)
    by_date = replacement.set_index("date")
    mask = output["_date"].isin(by_date.index)
    for column in ("open", "high", "low", "close", "volume"):
        if column in by_date:
            output.loc[mask, column] = output.loc[mask, "_date"].map(
                pd.to_numeric(by_date[column], errors="raise")
            )
    return output.drop(columns="_date"), int(mask.sum())


def _safe_archived_payload(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    target: TargetPin,
) -> bytes:
    row = _one(
        archive,
        archive["source_hash"].astype(str).eq(target.eodhd_source_sha256),
        f"{target.symbol} EODHD archive",
    )
    _require(str(row["dataset"]) == "eodhd_eod", f"{target.symbol} EODHD dataset changed.")
    _require(str(row["source_url"]) == target.eodhd_source_url, f"{target.symbol} EODHD URL changed.")
    root = repository.root.resolve()
    path = (root / str(row["object_path"])).resolve()
    _require(root in path.parents and path.is_file(), f"{target.symbol} archived EODHD bytes missing.")
    try:
        payload = gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise RuntimeError(f"{target.symbol} archived EODHD gzip is invalid.") from exc
    _require(sha256_bytes(payload) == target.eodhd_source_sha256, f"{target.symbol} EODHD raw hash changed.")
    return payload


def _audit_eodhd(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    target: TargetPin,
    prices: pd.DataFrame,
) -> dict[str, Any]:
    payload = _safe_archived_payload(repository, archive, target)
    raw = json.loads(payload)
    _require(isinstance(raw, list), f"{target.symbol} EODHD response is not a list.")
    raw_frame = pd.DataFrame(raw)
    raw_frame["date"] = pd.to_datetime(raw_frame["date"], errors="raise").dt.date.astype(str)
    raw_inventory = EODHD_ARCHIVE_INVENTORY_OVERRIDES.get(
        target.symbol,
        {"rows": target.eodhd_rows, "start": target.eodhd_start, "end": target.eodhd_end},
    )
    _require(
        len(raw_frame) == raw_inventory["rows"]
        and raw_frame["date"].min() == raw_inventory["start"]
        and raw_frame["date"].max() == raw_inventory["end"],
        f"{target.symbol} EODHD raw inventory changed.",
    )
    current = prices.loc[prices["security_id"].astype(str).eq(target.security_id)].copy()
    current["date"] = pd.to_datetime(current["session"], errors="raise").dt.date.astype(str)
    _require(
        len(current) == target.eodhd_rows
        and current["date"].min() == target.eodhd_start
        and current["date"].max() == target.eodhd_end,
        f"{target.symbol} Parquet price inventory changed.",
    )
    joined = current.merge(raw_frame, on="date", suffixes=("_parquet", "_raw"), validate="one_to_one")
    _require(len(joined) == len(current), f"{target.symbol} retained EODHD sessions changed.")
    for column in ("open", "high", "low", "close", "volume"):
        _require(
            bool(
                np.isclose(
                    pd.to_numeric(joined[f"{column}_parquet"], errors="raise"),
                    pd.to_numeric(joined[f"{column}_raw"], errors="raise"),
                    rtol=0.0,
                    atol=0.0,
                ).all()
            ),
            f"{target.symbol} Parquet diverges from archived EODHD {column}.",
        )
    return {
        "status": "exact_archived_raw_verified",
        "source_url": target.eodhd_source_url,
        "source_sha256": target.eodhd_source_sha256,
        "content_type": "application/json",
        "size": len(payload),
        "archived_raw_rows": len(raw_frame),
        "archived_raw_start": raw_inventory["start"],
        "archived_raw_end": raw_inventory["end"],
        "curated_parquet_rows": len(current),
        "curated_parquet_start": target.eodhd_start,
        "curated_parquet_end": target.eodhd_end,
        "excluded_provider_tail_rows": len(raw_frame) - len(current),
        "provider_code": target.eodhd_source_url.split("/eod/", 1)[1].split("?", 1)[0],
        "adjustment_basis": "raw open/high/low/close/volume; adjusted_close ignored",
        "parquet_exactly_reproduces_retained_archived_raw": True,
    }


def _parse_yahoo_payload(content: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = json.loads(content)
    chart = payload.get("chart") or {}
    _require(chart.get("error") is None, "Yahoo chart response contains an API error.")
    results = chart.get("result") or []
    _require(len(results) == 1, "Yahoo chart result inventory changed.")
    result = results[0]
    meta = result.get("meta") or {}
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quotes = indicators.get("quote") or []
    _require(len(quotes) == 1, "Yahoo quote inventory changed.")
    quote = quotes[0]
    frame = pd.DataFrame({"epoch": timestamps})
    frame["date"] = (
        pd.to_datetime(frame["epoch"], unit="s", utc=True, errors="raise")
        .dt.tz_convert("America/New_York")
        .dt.date.astype(str)
    )
    for column in ("open", "high", "low", "close", "volume"):
        values = quote.get(column) or []
        _require(len(values) == len(frame), f"Yahoo {column} inventory changed.")
        frame[column] = pd.to_numeric(pd.Series(values), errors="coerce")
    frame["complete"] = frame[["open", "high", "low", "close", "volume"]].notna().all(axis=1)
    adjclose = indicators.get("adjclose") or []
    adjclose_count = len((adjclose[0].get("adjclose") or [])) if len(adjclose) == 1 else 0
    return frame, {
        "symbol": _text(meta.get("symbol")),
        "currency": _text(meta.get("currency")),
        "instrument_type": _text(meta.get("instrumentType")),
        "exchange_name": _text(meta.get("exchangeName")),
        "exchange_timezone": _text(meta.get("exchangeTimezoneName")),
        "long_name": _text(meta.get("longName")),
        "short_name": _text(meta.get("shortName")),
        "first_trade_epoch": meta.get("firstTradeDate"),
        "timestamp_rows": len(frame),
        "complete_rows": int(frame["complete"].sum()),
        "null_or_incomplete_rows": int((~frame["complete"]).sum()),
        "start": "" if frame.empty else str(frame["date"].min()),
        "end": "" if frame.empty else str(frame["date"].max()),
        "adjclose_rows_present_but_ignored": adjclose_count,
    }


def _audit_yahoo(
    cache: YahooChartCache,
    target: TargetPin,
    prices: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    response = cache.get(
        target.symbol,
        period1=target.yahoo_period1,
        period2=target.yahoo_period2,
    )
    _require(response is not None, f"{target.symbol} Yahoo cache missing.")
    _require(response.source_hash == target.yahoo_source_sha256, f"{target.symbol} Yahoo source changed.")
    _require(response.wrapper_hash == target.yahoo_wrapper_sha256, f"{target.symbol} Yahoo wrapper changed.")
    _require(response.http_status == 200, f"{target.symbol} Yahoo HTTP status changed.")
    frame, metadata = _parse_yahoo_payload(response.content)
    _require(metadata["symbol"] == target.symbol, f"{target.symbol} Yahoo symbol metadata changed.")
    _require(metadata["instrument_type"] == target.yahoo_instrument_type, f"{target.symbol} Yahoo type changed.")
    _require(metadata["long_name"] == target.yahoo_long_name, f"{target.symbol} Yahoo issuer changed.")
    local = prices.loc[prices["security_id"].astype(str).eq(target.security_id)].copy()
    local["date"] = pd.to_datetime(local["session"], errors="raise").dt.date.astype(str)
    complete = frame.loc[frame["complete"], ["date", "open", "high", "low", "close", "volume"]].copy()
    overlap = local[["date", "open", "high", "low", "close"]].merge(
        complete[["date", "open", "high", "low", "close"]],
        on="date",
        suffixes=("_eodhd", "_yahoo"),
    )
    metadata.update(
        {
            "status": "rejected_identity_or_inventory_mismatch",
            "source_url": response.source_url,
            "source_sha256": response.source_hash,
            "cache_wrapper_sha256": response.wrapper_hash,
            "cache_file_sha256": _sha256_file(
                cache.path(target.symbol, period1=target.yahoo_period1, period2=target.yahoo_period2)
            ),
            "http_status": response.http_status,
            "content_type": response.content_type,
            "size": len(response.content),
            "identity_disposition": target.yahoo_identity_disposition,
            "identity_accepted": False,
            "overlap_rows_with_eodhd": len(overlap),
            "adjustment_basis": "indicators.quote raw OHLCV; adjclose never used",
            "reason": (
                "Ticker metadata resolves to a different issuer/instrument."
                if target.symbol != "HOT"
                else "Retired YHD/MUTUALFUND metadata lacks USD and 55 bars are incomplete."
            ),
        }
    )
    return metadata, complete


def _audit_stooq(cache: StooqHistoricalCache, target: TargetPin) -> dict[str, Any]:
    response = cache.get(target.symbol)
    _require(response is not None, f"{target.symbol} Stooq cache missing.")
    _require(response.source_hash == target.stooq_source_sha256, f"{target.symbol} Stooq bytes changed.")
    cache_path = cache.path(target.symbol)
    _require(_sha256_file(cache_path) == target.stooq_wrapper_sha256, f"{target.symbol} Stooq cache file changed.")
    html = response.content.lstrip().startswith((b"<", b"<!")) or "html" in response.content_type.lower()
    _require(html, f"{target.symbol} Stooq challenge disposition changed.")
    return {
        "status": "rejected_html_challenge",
        "source_url": response.source_url,
        "source_sha256": response.source_hash,
        "cache_file_sha256": target.stooq_wrapper_sha256,
        "http_status": response.http_status,
        "content_type": response.content_type,
        "size": len(response.content),
        "date_inventory": None,
        "identity_accepted": False,
        "adjustment_basis": "unavailable_html_challenge",
        "triple_supertrend": {"status": "not_computed_non_price_payload"},
    }


def _parse_boris(response: BorisCachedResponse, target: TargetPin) -> pd.DataFrame:
    _require(response.http_status == 200, f"{target.symbol} Boris is not HTTP 200.")
    _require("html" not in response.content_type.lower(), f"{target.symbol} Boris returned HTML.")
    _require(not response.content.lstrip().startswith((b"<", b"<!")), f"{target.symbol} Boris returned a challenge.")
    frame = pd.read_csv(io.BytesIO(response.content))
    _require(
        list(frame.columns) == ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"],
        f"{target.symbol} Boris schema changed.",
    )
    frame = frame.rename(
        columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.date.astype(str)
    for column in ("open", "high", "low", "close", "volume", "OpenInt"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    _require(not frame["date"].duplicated().any() and frame["date"].is_monotonic_increasing, f"{target.symbol} Boris dates invalid.")
    coherent = (
        frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & frame["volume"].ge(0)
        & frame["high"].ge(frame[["open", "low", "close"]].max(axis=1))
        & frame["low"].le(frame[["open", "high", "close"]].min(axis=1))
    )
    _require(bool(coherent.all()), f"{target.symbol} Boris OHLCV invalid.")
    _require(
        len(frame) == target.boris_rows
        and frame["date"].min() == target.boris_start
        and frame["date"].max() == target.boris_end,
        f"{target.symbol} Boris date inventory changed.",
    )
    return frame


def _audit_boris(
    cache: BorisPriceAuditCache,
    target: TargetPin,
    wiki: pd.DataFrame,
    attempt_ledger: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    if target.symbol == "HOT":
        failures = [
            item for item in attempt_ledger.get("failures", [])
            if item.get("provider") == "boris" and item.get("symbol") == "HOT"
        ]
        _require(len(failures) == 1, "HOT Boris failure ledger changed.")
        failure = dict(failures[0])
        _require(
            failure.get("http_status") == 404
            and failure.get("content_type") == "application/json"
            and failure.get("raw_bytes_available") is False
            and failure.get("cache_written") is False,
            "HOT Boris failure evidence changed.",
        )
        return {
            "status": "rejected_404_uncached_no_retry",
            **failure,
            "identity_accepted": False,
            "adjustment_basis": "unavailable",
            "triple_supertrend": {"status": "not_computed_no_price_payload"},
        }, None
    response = cache.get(target.symbol)
    _require(response is not None, f"{target.symbol} Boris cache missing.")
    _require(response.source_hash == target.boris_source_sha256, f"{target.symbol} Boris bytes changed.")
    _require(_sha256_file(cache.path(target.symbol)) == target.boris_wrapper_sha256, f"{target.symbol} Boris cache file changed.")
    common = {
        "source_url": response.source_url,
        "source_sha256": response.source_hash,
        "cache_file_sha256": target.boris_wrapper_sha256,
        "http_status": response.http_status,
        "content_type": response.content_type,
        "size": len(response.content),
        "license": BORIS_LICENSE,
        "license_url": BORIS_DATASET_VERSION_URL,
        "license_evidence": "existing reviewed local v3 client constant; no new metadata HTTP call",
        "upstream_provider_independence_established": False,
    }
    if response.http_status != 200:
        _require(
            response.http_status == 404 and response.content == b'{"code":404,"message":"Dataset not found"}',
            f"{target.symbol} Boris failure payload changed.",
        )
        return {
            "status": "rejected_404_json_cached",
            **common,
            "date_inventory": None,
            "identity_accepted": False,
            "adjustment_basis": "unavailable_json_error",
            "triple_supertrend": {"status": "not_computed_no_price_payload"},
        }, None
    frame = _parse_boris(response, target)
    wiki_adjusted = wiki[["date", "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]].rename(
        columns={"adj_open": "open", "adj_high": "high", "adj_low": "low", "adj_close": "close", "adj_volume": "volume"}
    )
    reviewed_start = "2015-01-02"
    relation = _numeric_relation(
        frame.loc[frame["date"].ge(reviewed_start)],
        wiki_adjusted.loc[wiki_adjusted["date"].ge(reviewed_start)],
        left_suffix="_boris",
        right_suffix="_wiki_adjusted",
    )
    close = relation["ohlc"]["close"]
    strict_stability = (
        relation["rows"] >= 700
        and close["max_scale_relative_deviation"] <= 0.005
        and close["return_correlation"] >= 0.999
        and all(value["max_scale_relative_deviation"] <= 0.01 for value in relation["ohlc"].values())
    )
    return {
        "status": "usable_adjusted_price_diagnostic",
        **common,
        "rows": len(frame),
        "start": target.boris_start,
        "end": target.boris_end,
        "identity_basis": "frozen ticker file plus long date overlap; no issuer metadata in file",
        "identity_accepted_for_diagnostic_only": True,
        "identity_accepted_for_release_pass": False,
        "adjustment_basis": "adjusted OHLCV inferred from close relation to WIKI adj_close; raw basis unavailable",
        "wiki_adjusted_relation": relation,
        "strict_long_scale_return_stability_passed": strict_stability,
    }, frame


def _audit_signals(
    target: TargetPin,
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    wiki: pd.DataFrame,
    yahoo_complete: pd.DataFrame,
    boris: pd.DataFrame | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = prices.loc[prices["security_id"].astype(str).eq(target.security_id)].copy()
    base = base.sort_values("session", ignore_index=True)
    target_factors = factors.loc[factors["security_id"].astype(str).eq(target.security_id)].copy()
    _require(len(target_factors) == len(base), f"{target.symbol} factor inventory changed.")
    base["date"] = pd.to_datetime(base["session"], errors="raise").dt.date.astype(str)
    joined = base[["date", "open", "high", "low", "close", "volume"]].merge(
        wiki[["date", "open", "high", "low", "close", "volume"]],
        on="date",
        suffixes=("_eod", "_wiki"),
        validate="one_to_one",
    ).sort_values("date", ignore_index=True)
    _require(len(joined) == target.wiki_overlap_rows, f"{target.symbol} WIKI overlap changed.")
    relation_sha = _relation_fingerprint(joined)
    _require(relation_sha == target.wiki_relation_sha256, f"{target.symbol} WIKI/EODHD relation changed.")

    base_without_date = base.drop(columns="date")
    wiki_candidate, wiki_replaced = _replace_prices(
        base_without_date,
        wiki[["date", "open", "high", "low", "close", "volume"]],
    )
    _require(wiki_replaced == target.wiki_overlap_rows, f"{target.symbol} WIKI replacement count changed.")
    raw_base_signals = _signal_frame(base_without_date)
    raw_wiki_signals = _signal_frame(wiki_candidate)
    adjusted_base = apply_adjustment_factors(
        base_without_date, target_factors, mode="total_return_adjusted"
    ).sort_values("session", ignore_index=True)
    adjusted_wiki = apply_adjustment_factors(
        wiki_candidate, target_factors, mode="total_return_adjusted"
    ).sort_values("session", ignore_index=True)
    adjusted_base_signals = _signal_frame(adjusted_base)
    adjusted_wiki_signals = _signal_frame(adjusted_wiki)
    baseline_raw_sha = _signal_hash(raw_base_signals)
    baseline_adjusted_sha = _signal_hash(adjusted_base_signals)
    _require(
        (baseline_raw_sha, baseline_adjusted_sha)
        == EXPECTED_BASELINE_SIGNAL_SHA256[target.symbol],
        f"{target.symbol} baseline Triple Supertrend output changed.",
    )
    wiki_raw_differences = signal_differences(raw_base_signals, raw_wiki_signals)
    wiki_adjusted_differences = signal_differences(
        adjusted_base_signals, adjusted_wiki_signals
    )
    _require(
        _signal_difference_counts(wiki_adjusted_differences)
        == EXPECTED_WIKI_ADJUSTED_DIFF_COUNTS[target.symbol],
        f"{target.symbol} WIKI-adjusted Triple Supertrend differences changed.",
    )

    yahoo_candidate, yahoo_replaced = _replace_prices(base_without_date, yahoo_complete)
    yahoo_raw_signals = _signal_frame(yahoo_candidate)
    yahoo_adjusted = apply_adjustment_factors(
        yahoo_candidate, target_factors, mode="total_return_adjusted"
    ).sort_values("session", ignore_index=True)
    yahoo_adjusted_signals = _signal_frame(yahoo_adjusted)
    yahoo_impact = {
        "status": (
            "partial_complete_bar_diagnostic_only"
            if yahoo_replaced
            else "no_identity_bound_overlap"
        ),
        "replaced_complete_rows": yahoo_replaced,
        "identity_collision_means_zero_or_nonzero_differences_cannot_validate_prices": True,
        "raw": signal_differences(raw_base_signals, yahoo_raw_signals),
        "total_return_adjusted": signal_differences(
            adjusted_base_signals, yahoo_adjusted_signals
        ),
    }

    boris_impact: dict[str, Any]
    if boris is None:
        boris_impact = {"status": "not_computed_no_valid_price_payload"}
    else:
        adjusted_dates = adjusted_base.copy()
        adjusted_dates["date"] = pd.to_datetime(
            adjusted_dates["session"], errors="raise"
        ).dt.date.astype(str)
        common = adjusted_dates[["date", "close"]].merge(
            boris[["date", "close"]],
            on="date",
            suffixes=("_current_adjusted", "_boris"),
            validate="one_to_one",
        )
        _require(len(common) >= 700, f"{target.symbol} Boris/current overlap is too short.")
        scale = float(
            (
                pd.to_numeric(common["close_boris"], errors="raise")
                / pd.to_numeric(common["close_current_adjusted"], errors="raise")
            ).median()
        )
        normalized = boris[["date", "open", "high", "low", "close", "volume"]].copy()
        for column in ("open", "high", "low", "close"):
            normalized[column] = pd.to_numeric(normalized[column], errors="raise") / scale
        candidate_adjusted, replaced = _replace_prices(adjusted_base, normalized)
        candidate_signals = _signal_frame(candidate_adjusted)
        boris_differences = signal_differences(
            adjusted_base_signals, candidate_signals
        )
        _require(
            _signal_difference_counts(boris_differences)
            == EXPECTED_BORIS_ADJUSTED_DIFF_COUNTS[target.symbol],
            f"{target.symbol} Boris-adjusted Triple Supertrend differences changed.",
        )
        boris_impact = {
            "status": "diagnostic_adjusted_basis_normalized_by_one_close_scale",
            "replaced_rows": replaced,
            "normalization_scale_boris_to_current_adjusted": scale,
            "volume_not_used_by_triple_supertrend": True,
            "total_return_adjusted": boris_differences,
        }

    relation_summary: dict[str, Any] = {
        "rows": len(joined),
        "start": str(joined["date"].min()),
        "end": str(joined["date"].max()),
        "relation_sha256": relation_sha,
    }
    for column in ("open", "high", "low", "close"):
        left = pd.to_numeric(joined[f"{column}_eod"], errors="raise")
        right = pd.to_numeric(joined[f"{column}_wiki"], errors="raise")
        relative = (left - right).abs() / right.abs()
        relation_summary[column] = {
            "exact_rows": int(np.isclose(left, right, rtol=0.0, atol=1e-12).sum()),
            "max_absolute_difference": float((left - right).abs().max()),
            "max_relative_difference": float(relative.max()),
            "return_correlation": float(left.pct_change().corr(right.pct_change())),
        }
    close_relative = (
        pd.to_numeric(joined["close_eod"], errors="raise")
        - pd.to_numeric(joined["close_wiki"], errors="raise")
    ).abs() / pd.to_numeric(joined["close_wiki"], errors="raise").abs()
    maximum_row = joined.loc[close_relative.idxmax()]
    relation_summary["maximum_close_disagreement"] = {
        "date": str(maximum_row["date"]),
        "eodhd_raw_close": float(maximum_row["close_eod"]),
        "wiki_raw_close": float(maximum_row["close_wiki"]),
        "relative_difference": float(close_relative.max()),
    }
    return {
        "baseline": {
            "raw_signal_sha256": baseline_raw_sha,
            "total_return_adjusted_signal_sha256": baseline_adjusted_sha,
        },
        "wiki_raw_substitution": {
            "status": "changes_strategy_state_fail_closed",
            "raw": wiki_raw_differences,
            "total_return_adjusted": wiki_adjusted_differences,
        },
        "yahoo_raw_quote_substitution": yahoo_impact,
        "stooq": {"status": "not_computed_non_price_payload"},
        "boris_adjusted_substitution": boris_impact,
    }, relation_summary


def _disposition(
    target: TargetPin,
    boris_audit: Mapping[str, Any],
    relation: Mapping[str, Any],
) -> dict[str, Any]:
    if target.symbol == "APC":
        _require(
            boris_audit.get("strict_long_scale_return_stability_passed") is True,
            "APC Boris/WIKI consensus changed.",
        )
        _require(
            relation["maximum_close_disagreement"]["date"] == "2015-11-10",
            "APC maximum close disagreement changed.",
        )
        finding = (
            "Boris adjusted OHLC closely follows frozen WIKI adjusted OHLC over 721+ sessions "
            "and is consistent with WIKI around the 2015-11-10 close anomaly. This identifies a "
            "candidate EODHD raw-close anomaly, not independent confirmation of an EODHD defect: "
            "Boris lacks issuer metadata, its upstream provider independence is not established, "
            "and its raw adjustment inverse is unavailable."
        )
        next_step = "Seek an issuer-bound raw close for 2015-11-10 before any one-row repair."
    elif target.symbol == "IR":
        finding = (
            "Boris generally follows WIKI adjusted history, but strict scale stability fails; "
            "Yahoo is the different Gardner Denver/Ingersoll Rand issuer."
        )
        next_step = "Keep the legacy IR-to-TT identity bound and obtain an issuer-labelled raw arbiter."
    elif target.symbol == "LB":
        finding = (
            "Boris, WIKI adjusted, and current adjusted history do not maintain the required "
            "single stable scale; Yahoo and Stooq cannot arbitrate the legacy L Brands series."
        )
        next_step = "Obtain issuer-labelled legacy L Brands raw OHLCV or retain the mismatch."
    else:
        finding = (
            "Stooq returned only an HTML challenge and Boris supplied no valid price file; "
            "Yahoo is incomplete or a reused issuer/ETF, leaving only EODHD versus WIKI."
        )
        next_step = "Retain the exact mismatch until another issuer-bound raw provider is available."
    return {
        "status": "fail_closed_no_crossvalidation_pass",
        "generic_exception_allowed": False,
        "release_repair_allowed": False,
        "finding": finding,
        "next_step": next_step,
    }


def run_audit(
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    release_path: Path = DEFAULT_RELEASE,
    report_path: Path = DEFAULT_CROSSVALIDATION_REPORT,
    yahoo_cache_root: Path = DEFAULT_YAHOO_CACHE,
    stooq_cache_root: Path = DEFAULT_STOOQ_CACHE,
    boris_cache_root: Path = DEFAULT_BORIS_CACHE,
    wiki_zip_path: Path = DEFAULT_WIKI_ZIP,
    attempt_ledger_path: Path = DEFAULT_ATTEMPT_LEDGER,
) -> dict[str, Any]:
    release, report = _load_release_and_report(release_path, report_path)
    _require(attempt_ledger_path.is_file(), "HTTP attempt ledger is missing.")
    attempt_ledger = json.loads(attempt_ledger_path.read_text(encoding="utf-8"))
    _require(
        attempt_ledger.get("schema") == "us_free_price_arbiter_http_attempt_ledger/v1",
        "HTTP attempt ledger schema changed.",
    )
    wiki, wiki_evidence = _load_wiki_bundle(wiki_zip_path)
    repository = LocalDatasetRepository(cache_root)
    security_ids = [target.security_id for target in TARGETS]
    prices = _read_subset(
        repository,
        "daily_price_raw",
        EXPECTED_DATASET_VERSIONS["daily_price_raw"],
        security_ids,
    )
    factors = _read_subset(
        repository,
        "adjustment_factors",
        EXPECTED_DATASET_VERSIONS["adjustment_factors"],
        security_ids,
    )
    master = _read_subset(
        repository,
        "security_master",
        EXPECTED_DATASET_VERSIONS["security_master"],
        security_ids,
    )
    history = _read_subset(
        repository,
        "symbol_history",
        EXPECTED_DATASET_VERSIONS["symbol_history"],
        security_ids,
    )
    archive = repository.read_frame(
        "source_archive", EXPECTED_DATASET_VERSIONS["source_archive"]
    )
    yahoo_cache = YahooChartCache(yahoo_cache_root)
    stooq_cache = StooqHistoricalCache(stooq_cache_root, max_http_attempts=7)
    boris_cache = BorisPriceAuditCache(boris_cache_root, max_http_attempts=7)

    audits: list[dict[str, Any]] = []
    for target in TARGETS:
        master_row = _one(
            master,
            master["security_id"].astype(str).eq(target.security_id),
            f"{target.symbol} master",
        )
        history_row = _one(
            history,
            history["security_id"].astype(str).eq(target.security_id)
            & history["symbol"].astype(str).eq(target.symbol),
            f"{target.symbol} history",
        )
        _require(str(master_row["primary_symbol"]) == target.master_primary_symbol, f"{target.symbol} primary symbol changed.")
        _require(_text(master_row["active_to"]) == target.master_active_to, f"{target.symbol} master end changed.")
        _require(_text(history_row["effective_to"]) == target.history_effective_to, f"{target.symbol} history end changed.")
        eodhd = _audit_eodhd(repository, archive, target, prices)
        yahoo, yahoo_complete = _audit_yahoo(yahoo_cache, target, prices)
        stooq = _audit_stooq(stooq_cache, target)
        boris, boris_frame = _audit_boris(
            boris_cache, target, wiki[target.symbol], attempt_ledger
        )
        signals, wiki_relation = _audit_signals(
            target,
            prices,
            factors,
            wiki[target.symbol],
            yahoo_complete,
            boris_frame,
        )
        audits.append(
            {
                "symbol": target.symbol,
                "security_id": target.security_id,
                "target_id": target.target_id,
                "index_relevance": target.index_relevance,
                "identity": {
                    "primary_symbol": str(master_row["primary_symbol"]),
                    "name": str(master_row["name"]),
                    "master_active_from": _date(master_row["active_from"]),
                    "master_active_to": _text(master_row["active_to"]),
                    "history_effective_from": _date(history_row["effective_from"]),
                    "history_effective_to": _text(history_row["effective_to"]),
                    "history_source_sha256": str(history_row["source_hash"]),
                },
                "providers": {
                    "eodhd": eodhd,
                    "yahoo": yahoo,
                    "wiki": {
                        "status": "frozen_exact_bytes_verified_private_only",
                        "raw_lines_sha256": target.wiki_raw_lines_sha256,
                        "extract_sha256": target.wiki_extract_sha256,
                        "rows": target.wiki_rows,
                        "start": target.wiki_start,
                        "end": target.wiki_end,
                        "adjustment_basis": "raw and adjusted OHLCV are separate columns",
                        "license": WIKI_LICENSE,
                    },
                    "stooq": stooq,
                    "boris": boris,
                },
                "eodhd_wiki_raw_relation": wiki_relation,
                "triple_supertrend": signals,
                "disposition": _disposition(target, boris, wiki_relation),
            }
        )
    _require(yahoo_cache.http_attempts == 0, "Audit unexpectedly called Yahoo.")
    _require(stooq_cache.http_attempts == 0, "Audit unexpectedly called Stooq.")
    _require(boris_cache.http_attempts == 0, "Audit unexpectedly called Boris/Kaggle.")
    _require(all(item["disposition"]["status"].startswith("fail_closed") for item in audits), "A target was promoted.")
    return {
        "schema": "us_free_price_arbiter_audit/v1",
        "base_release_version": release["version"],
        "baseline_crossvalidation_report_sha256": CROSSVALIDATION_REPORT_SHA256,
        "baseline_crossvalidation_summary": report["summary"],
        "controls": {
            "network_http_attempts_for_acquisition": 14,
            "network_retries": 0,
            "acquisition_transport_count_evidence_limit": (
                "The bounded acquisition run recorded 14 attempts and zero retries; offline "
                "artifacts independently establish 13 cached raw responses plus one uncached "
                "failure-ledger outcome, but do not reconstruct transport retry history."
            ),
            "cached_raw_responses": 13,
            "hot_boris_404_uncached_failures": 1,
            "audit_run_http_attempts": 0,
            "eodhd_api_calls": 0,
            "r2_calls": 0,
            "release_apply_calls": 0,
            "dataset_mutations": 0,
            "common_crossvalidation_files_modified": False,
            "generic_exceptions_added": 0,
            "attempt_ledger_sha256": _sha256_file(attempt_ledger_path),
        },
        "source_licenses": {
            "wiki": WIKI_LICENSE,
            "boris": {
                "license": BORIS_LICENSE,
                "url": BORIS_DATASET_VERSION_URL,
                "evidence_limit": "reused reviewed local client constant; metadata bytes were not fetched",
            },
            "stooq": "not evaluated because every response was an HTML challenge",
        },
        "wiki_bundle": wiki_evidence,
        "summary": {
            "target_count": 7,
            "stooq_valid_price_targets": 0,
            "boris_valid_price_targets": 3,
            "boris_strict_wiki_adjusted_stability_targets": [
                item["symbol"]
                for item in audits
                if item["providers"]["boris"].get("strict_long_scale_return_stability_passed")
            ],
            "promoted_crossvalidation_passes": 0,
            "remaining_fail_closed": 7,
            "candidate_eodhd_raw_close_anomaly_needing_independent_confirmation": ["APC"],
        },
        "targets": audits,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-missing", action="store_true")
    parser.add_argument("--acquire-only", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--crossvalidation-report", type=Path, default=DEFAULT_CROSSVALIDATION_REPORT)
    parser.add_argument("--yahoo-cache", type=Path, default=DEFAULT_YAHOO_CACHE)
    parser.add_argument("--stooq-cache", type=Path, default=DEFAULT_STOOQ_CACHE)
    parser.add_argument("--boris-cache", type=Path, default=DEFAULT_BORIS_CACHE)
    parser.add_argument("--wiki-zip", type=Path, default=DEFAULT_WIKI_ZIP)
    parser.add_argument("--attempt-ledger", type=Path, default=DEFAULT_ATTEMPT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.acquire_only and not args.fetch_missing:
        raise RuntimeError("--acquire-only requires --fetch-missing.")
    if args.fetch_missing:
        result = acquire_missing_evidence(
            stooq_root=args.stooq_cache,
            boris_root=args.boris_cache,
            attempt_ledger_path=args.attempt_ledger,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        if args.acquire_only:
            return 0
    result = run_audit(
        cache_root=args.cache_root,
        release_path=args.release,
        report_path=args.crossvalidation_report,
        yahoo_cache_root=args.yahoo_cache,
        stooq_cache_root=args.stooq_cache,
        boris_cache_root=args.boris_cache,
        wiki_zip_path=args.wiki_zip,
        attempt_ledger_path=args.attempt_ledger,
    )
    rendered = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    if not args.no_write:
        write_atomic(args.output, rendered)
    print(
        json.dumps(
            {
                "status": "completed_fail_closed",
                "output": "" if args.no_write else str(args.output),
                "report_sha256": sha256_bytes(rendered),
                "summary": result["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
