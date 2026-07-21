#!/usr/bin/env python3
"""Archive the exact frozen-WIKI price-only evidence for 14 US identities.

The operation is deliberately narrow.  It streams one hash-pinned local WIKI
ZIP, archives one immutable full-symbol CSV extract per reviewed identity plus
one canonical provenance object, and changes only ``source_archive``.  It does
not rewrite prices, corporate actions, factors, identities, index data, or any
cross-validation policy.

The source has a formal Kaggle license value of ``Unknown``.  Plan mode is
read-only and apply is refused unless the caller explicitly acknowledges a
private/internal-only local archive.  Redistribution and public publication
remain blocked; R2 publication is outside this script.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import io
import json
import math
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd

from supertrend_quant.indicators import add_triple_supertrend
from supertrend_quant.market_store.adjustments import apply_adjustment_factors
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    sha256_bytes,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.storage import (
    ConditionalWriteFailed,
    ObjectNotFound,
)
from supertrend_quant.market_store.validation import (
    validate_dataset,
    validate_manifest_files,
)


OPERATION = "repair_us_wiki14_price_only"
DATASET = "source_archive"
DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_WIKI_ZIP = Path("/tmp/marketneutral-quandl-wiki-prices.zip")
REQUIRED_DATASETS = (
    "daily_price_raw",
    "corporate_actions",
    "adjustment_factors",
    "security_master",
    "symbol_history",
    DATASET,
)
TRANSACTION_DIR = "transactions/us-wiki14-price-only"
RECOVERY_DIR = "recovery/us-wiki14-price-only"

WIKI_MEMBER = "WIKI_PRICES.csv"
WIKI_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "marketneutral/quandl-wiki-prices-us-equites/WIKI_PRICES.csv"
)
WIKI_ZIP_SHA256 = (
    "36c667bbecf42c43e5b9e8e4e5d9a1268522705bc40a4bac9671d62c1a20cbae"
)
WIKI_ZIP_SIZE = 463_184_323
WIKI_MEMBER_SHA256 = (
    "ca7fb174c7948db85638917d25ff65d438e27d5cb23675da784c54db01e3d003"
)
WIKI_MEMBER_SIZE = 1_797_003_576
WIKI_MEMBER_CRC32 = 0x946874CE
WIKI_RETRIEVED_AT = "2026-07-18T03:58:26.808706Z"
REVIEWED_AT = "2026-07-19T12:00:00Z"
REVIEW_CEILING = "2018-03-27"
ARCHIVE_EFFECTIVE_DATE = "2026-07-15"
WIKI_LICENSE_WARNING = (
    "Kaggle Quandl WIKI licenseName=Unknown; private/internal-only; "
    "redistribution/public publication blocked."
)
BBBY_BBT_EXTRACT_SHA256 = (
    "a6a6f651265825ed9ed95a1dfb9889f70586a728aa53eeae8585b8c00e4af52f"
)
BBBY_BBT_PROVENANCE_SHA256 = (
    "d73bf90641034b56b4ce42d9cef2fd4dff23a6db8c101cc7ed9b49af4c7140c8"
)
BBBY_BBT_ARTIFACT_PINS = {
    BBBY_BBT_EXTRACT_SHA256: {
        "dataset": "kaggle_quandl_wiki_bbby_bbt_price_extract",
        "object_path": (
            f"archives/{ARCHIVE_EFFECTIVE_DATE}/{BBBY_BBT_EXTRACT_SHA256}.csv.gz"
        ),
        "content_type": "text/csv",
        "effective_date": ARCHIVE_EFFECTIVE_DATE,
        "source": "kaggle_quandl_wiki_bbby_bbt_price_extract",
        "retrieved_at": WIKI_RETRIEVED_AT,
        "source_hash": BBBY_BBT_EXTRACT_SHA256,
        "source_url": WIKI_DOWNLOAD_URL,
    },
    BBBY_BBT_PROVENANCE_SHA256: {
        "dataset": "reviewed_us_wiki_price_arbitration",
        "object_path": (
            f"archives/{ARCHIVE_EFFECTIVE_DATE}/{BBBY_BBT_PROVENANCE_SHA256}.json.gz"
        ),
        "content_type": "application/json",
        "effective_date": ARCHIVE_EFFECTIVE_DATE,
        "source": "reviewed_us_wiki_price_arbitration",
        "retrieved_at": "2026-07-19T04:30:00Z",
        "source_hash": BBBY_BBT_PROVENANCE_SHA256,
        "source_url": WIKI_DOWNLOAD_URL,
    },
}

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
class TargetPin:
    symbol: str
    security_id: str
    target_id: str
    provider_symbol: str
    active_from: str
    active_to: str
    history_effective_from: str
    history_effective_to: str
    identity_source: str
    identity_source_sha256: str
    terminal_event_id: str
    terminal_source_sha256: str
    full_wiki_rows: int
    full_wiki_start: str
    full_wiki_end: str
    raw_lines_sha256: str
    extract_sha256: str
    extract_size: int
    review_start: str
    review_end: str
    overlap_rows: int
    relation_sha256: str
    price_rows: int
    price_start: str
    price_end: str
    price_source_sha256s: tuple[str, ...]
    raw_economics_sha256: str
    identity_sha256: str
    terminal_actions_sha256: str
    all_actions_sha256: str
    action_coverage_sha256: str
    factor_rows: int
    factor_economics_sha256: str
    signal_sha256: str


@dataclass(frozen=True)
class IdentitySchemaPin:
    master_primary_symbol: str
    master_exchange: str
    master_asset_type: str
    master_currency: str
    master_country: str
    history_symbol: str
    history_exchange: str
    raw_price_currency: str


def _pin(**values: Any) -> TargetPin:
    return TargetPin(**values)


def _identity_pin(
    symbol: str,
    exchange: str,
    *,
    master_primary_symbol: str = "",
) -> IdentitySchemaPin:
    return IdentitySchemaPin(
        master_primary_symbol=master_primary_symbol or symbol,
        master_exchange=exchange,
        master_asset_type="STOCK",
        master_currency="USD",
        master_country="US",
        history_symbol=symbol,
        history_exchange=exchange,
        raw_price_currency="USD",
    )


IDENTITY_SCHEMA_PINS = {
    "ADT": _identity_pin("ADT", "NYSE"),
    "CAM": _identity_pin("CAM", "NYSE"),
    "COL": _identity_pin("COL", "NYSE"),
    "EMC": _identity_pin("EMC", "NYSE"),
    "EVHC": _identity_pin("EVHC", "NYSE"),
    "FB": _identity_pin("FB", "BATS"),
    "FOX": _identity_pin("FOX", "NASDAQ", master_primary_symbol="TFCF"),
    "FOXA": _identity_pin("FOXA", "NASDAQ", master_primary_symbol="TFCFA"),
    "INFO": _identity_pin("INFO", "NYSE"),
    "NFX": _identity_pin("NFX", "NYSE"),
    "SCG": _identity_pin("SCG", "NYSE"),
    "SNDK": _identity_pin("SNDK", "NASDAQ"),
    "STI": _identity_pin("STI", "NASDAQ"),
    "TE": _identity_pin("TE", "NYSE"),
}

# Hash of every exact master/history/raw-price identity classification above.
# The code inventory, not a YAML-only ticker entry, is the authority.
IDENTITY_SCHEMA_INVENTORY_SHA256 = (
    "1c82f703ed1e88790dd1b066614c25389bb4a442c3e0369655fad5583b14880b"
)


COMMON_IDENTITY_SHA = (
    "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99"
)


TARGETS = (
    _pin(
        symbol="ADT", security_id="US:EODHD:154f8739-c6d3-59c7-8abc-f871d1a92683",
        target_id="296e2d263b549d741aea400cfe59b164b8f06e7420426e95d251f7ee926e26b0",
        provider_symbol="ADT_old.US", active_from="2015-01-02", active_to="2016-04-29",
        history_effective_from="2015-01-01", history_effective_to="",
        identity_source="eodhd_exchange_symbols", identity_source_sha256=COMMON_IDENTITY_SHA,
        terminal_event_id="418defe40340005df5628364001a45f2438b7ed292a7b1196325000248aefaf7",
        terminal_source_sha256="c2db6de16c21d9a5797b45b9d45f14a7e895815529133c8604c72d7faf52ca0a",
        full_wiki_rows=905, full_wiki_start="2012-10-01", full_wiki_end="2018-03-07",
        raw_lines_sha256="fb9c6cecafa8b3fab1346787cc5e5fc45364664ec64b12ad1725d1dc629cea51",
        extract_sha256="57ad6f387e01cf1ade18b8d4f8f52b2411277caf43bdfee856e7ec9d67b3a560",
        extract_size=123066, review_start="2015-01-02", review_end="2016-04-29",
        overlap_rows=334, relation_sha256="b1f34098371f11b11532060d7989a8be4b04822e9a9f634e0d9c4d517bd86422",
        price_rows=334, price_start="2015-01-02", price_end="2016-04-29",
        price_source_sha256s=("fa1b494230ec66e68bc30d1a6bbc8cf2111f43d95a546c8aa73c61162010be90",),
        raw_economics_sha256="3de1d55e3202627b8b9fb9e0edbbe1cb2ce0099956c2bb27ff1e206911cba1fd",
        identity_sha256="f5dc5e40e621da92bbdc9be981821c57251a0cb3c55d19b0e77b194f1d00a1c1",
        terminal_actions_sha256="62764d778043fdb95919c8a8c7b10903e54ceb87980ad2ac6a8b52820a6113c4",
        all_actions_sha256="9d2df9ec577b29cbeb4ab1f2f230320d6e6bbb941aaf3f1a2b4e111b88b32920",
        action_coverage_sha256="4c54414a08fbb19e523af898636768f7de62ec858467f53b41b044dc2b3ca358",
        factor_rows=334, factor_economics_sha256="17ac6214c8a2de9af07e157e9d056b96bd09b15b0f65a92fe6a7a5a95f067427",
        signal_sha256="06bf9aef26302433eb9f1360776d6a1b9fd765da2ea68366d16c3c7b1417ad06",
    ),
    _pin(
        symbol="CAM", security_id="US:EODHD:20f4418a-aa9a-5ede-a981-a4bda023f291",
        target_id="d04a3aa3adfa1f2290f64d1fa9747fb1a1d59d5e106a0e4e4b2841f024ff4c08",
        provider_symbol="CAM_old.US", active_from="2015-01-02", active_to="2016-04-01",
        history_effective_from="2015-01-01", history_effective_to="",
        identity_source="eodhd_exchange_symbols", identity_source_sha256=COMMON_IDENTITY_SHA,
        terminal_event_id="5e03acf71792e61458faabda31aed6535512b1cf82bcc03fe9a3fcb73960c079",
        terminal_source_sha256="0acee6cb8286727e437df5fdf890961b2941335dfc92e63a3177633558b84e71",
        full_wiki_rows=5223, full_wiki_start="1995-07-05", full_wiki_end="2016-04-01",
        raw_lines_sha256="fbc31dcb9550fa7e956d72d45fb00bbb64fd2794008817a8985c7324d4e28bc2",
        extract_sha256="5c102159b85a57bf953429506f2988536927925b34489d0594528d492bc27567",
        extract_size=477494, review_start="2015-01-02", review_end="2016-04-01",
        overlap_rows=314, relation_sha256="f64e6e8460ce962cf9727b90eb8080e961ad094041dfa946fcd7ab995c7c8251",
        price_rows=314, price_start="2015-01-02", price_end="2016-04-01",
        price_source_sha256s=("3576b4935e0e5e0240b0566cd75b526279f173ef40003848e90c72b870440191",),
        raw_economics_sha256="e5054e95eb2d80df3c2da46120836ff812652162ea388600cacaadbb2a6d0c4e",
        identity_sha256="a6168a01537acad5768dbe20f2b63a99030f7e66f64a1b97a40c8034329b26db",
        terminal_actions_sha256="075678a24f69ccda0539fa89bdae21659d39e77116ec2722de36257ee1e31437",
        all_actions_sha256="075678a24f69ccda0539fa89bdae21659d39e77116ec2722de36257ee1e31437",
        action_coverage_sha256="b8efadebf946bba0dde4fb6167e5ca8f8fb05054480f149941e1d375edfd5b37",
        factor_rows=314, factor_economics_sha256="e5bf9f2977c6f11241e32bef012824b6c1ca5862361b04c4b9eb9f369deaebac",
        signal_sha256="c0325eb62f1a2022ffd36401194642d9ee1f455c77a84abb45c8349df4492d80",
    ),
    _pin(
        symbol="COL", security_id="US:EODHD:4ddb0638-fe2a-5f9c-97c8-691e9c42d5f3",
        target_id="f3e261c422fd805f271ad9104511d7a1a76e7228adbee3494a964399a2ceb235",
        provider_symbol="COL.US", active_from="2015-01-02", active_to="2018-11-26",
        history_effective_from="2015-01-01", history_effective_to="",
        identity_source="eodhd_exchange_symbols", identity_source_sha256=COMMON_IDENTITY_SHA,
        terminal_event_id="ff2361a64b153fa524e11b69b0bb1fce76b325343558ed17924c005420914e65",
        terminal_source_sha256="d93d3e1d6a03dfe0648deba22be4cbc1869d793cff553ad5493140946c99fafb",
        full_wiki_rows=4220, full_wiki_start="2001-06-15", full_wiki_end="2018-03-27",
        raw_lines_sha256="bef8afd45e986a70c32d43aaed2b43593e5e152bf60d509f9ec224e019d11ed0",
        extract_sha256="0564534662e44aa613027d0a1e42979422ebe82cfc8f00cc5c159095b2ab0743",
        extract_size=538648, review_start="2015-01-02", review_end="2018-03-27",
        overlap_rows=813, relation_sha256="438021cc67e0b737a83b35c495f7a8ae9a04d75abd69e2abc9121cc856e5b665",
        price_rows=983, price_start="2015-01-02", price_end="2018-11-26",
        price_source_sha256s=("2634cc111dc81eba972d563bbf05c9d6b2f79ba9812cb3b0c8540d4ba9dc5b14", "5d99f922bb7c45afe31a473f89b441f42df0cb0769b01fdfa842353304b3d636"),
        raw_economics_sha256="68617ccda7036e3c87c5150d661a86840217b409d2be2b6c9c17da56974fd95c",
        identity_sha256="f631d47e672d2916d2bca628a4bc85fac8a0e7abf90b52d9ef3f9d2ca09fc6e0",
        terminal_actions_sha256="791d029674f35b8eb2b6fdfbe51ca15458e98ba43df9d371306322ecee0edc72",
        all_actions_sha256="773013f0733bbd7b3bb993d8b1a57066d5dc1eacf6dcccd92da53c5ba761bf01",
        action_coverage_sha256="0ee4540e75c7957c7b1debe909d9de1254988e0e030dd178ecedfa16549afc71",
        factor_rows=983, factor_economics_sha256="153abe931a4fddb5ab38712e6de95f0f59ac5afbea4a89f1c586054c3807dd73",
        signal_sha256="b81f628a22fda05b9c08b3ab4d3a6a18276cff1163e8e3737b788bc4447d879c",
    ),
    _pin(
        symbol="EMC", security_id="US:EODHD:d0212dec-b333-5d7f-90ce-cd3d4c6cc035",
        target_id="1bbd0db44c10b63aa9fb4a2ee1399e83b699a97dd343d0e541682b6066af659a",
        provider_symbol="EMC_old.US", active_from="2015-01-02", active_to="2016-09-06",
        history_effective_from="2015-01-01", history_effective_to="",
        identity_source="eodhd_exchange_symbols", identity_source_sha256=COMMON_IDENTITY_SHA,
        terminal_event_id="42745f23f18059ac4772c578967a56d6ba6d19f7053115b97cd3db394a3bff26",
        terminal_source_sha256="9a7457fb4ad4475fcae7937aba4886d4f92eece8c98652049f9bf757e1e3e27d",
        full_wiki_rows=6986, full_wiki_start="1988-12-16", full_wiki_end="2016-09-06",
        raw_lines_sha256="96bc80c99cbcb07d40fb9d4aba03463c8761894466ddab4924f2cc392f43ab8f",
        extract_sha256="07fdbc0505f10ada04f55796f5a65815be9663a50f40d2e203fedc301c658632",
        extract_size=985669, review_start="2015-01-02", review_end="2016-09-06",
        overlap_rows=423, relation_sha256="2fb01f84dae24ee64dcb5daebdf7ea3b9e0b3379543507f6a40dda00be3e30b5",
        price_rows=423, price_start="2015-01-02", price_end="2016-09-06",
        price_source_sha256s=("44805dae7503ea327be204118a262012b3c84c5e3445de561c7a5fe597c871e7",),
        raw_economics_sha256="a4030266d15b69358b7ad428d498324792702e28e6e67c5314eb74f52810bcb2",
        identity_sha256="4cb1607a1aad3676d2f3949414f0bbde84392dcffe3330db667ba8214f9396ca",
        terminal_actions_sha256="c29ea02720482750d31639749071f550195182f9727dad565737dfd07062bf04",
        all_actions_sha256="c29ea02720482750d31639749071f550195182f9727dad565737dfd07062bf04",
        action_coverage_sha256="5626646ffc2bc81accc5321d66d717352d08a79e5301c406f07b9e3434e1b995",
        factor_rows=423, factor_economics_sha256="3c222c34cec451fa2400653954a4f9db927f6f5f0b786da4731df28bc13cb9d4",
        signal_sha256="1bc3aa058ea6f45b84024705f819b4dd1cfff217d51e63432b6d9c261a162675",
    ),
    _pin(
        symbol="EVHC", security_id="US:EODHD:b5c1cb33-a560-58df-8a77-978096a5dbbf",
        target_id="9aff0d4d65ce790b98d6f050f6323044ce01a2fbd378c93b5c0f668f9fffab44",
        provider_symbol="EVHC.US", active_from="2015-01-02", active_to="2018-10-10",
        history_effective_from="2015-01-01", history_effective_to="",
        identity_source="eodhd_exchange_symbols", identity_source_sha256=COMMON_IDENTITY_SHA,
        terminal_event_id="d693ed94d0f1468fe4455c9ccd5d8687ea36d20c33e66e2a12983904aa9e482a",
        terminal_source_sha256="bea291af8e20362bcfdefc717d393f96d088567101a90139cedceb45443b01b4",
        full_wiki_rows=1162, full_wiki_start="2013-08-14", full_wiki_end="2018-03-27",
        raw_lines_sha256="7d1bbdeeb3e355ea351f7c3aabc3c3896e52e035b1e3daeee8befc63d6f78ae1",
        extract_sha256="a745b7d5553b0dedb06652b8c41acbf536114d621e1d702c34719225c82fc697",
        extract_size=105564, review_start="2015-01-02", review_end="2018-03-27",
        overlap_rows=813, relation_sha256="6f0fc3fef42f1a37d81e1bcf619c56d95e647d344fede76d7fede4fdaf6414d5",
        price_rows=951, price_start="2015-01-02", price_end="2018-10-10",
        price_source_sha256s=("758de9d501550201401fc6b75ebb34dac4861c0a50d07819abed3d4eb02b1ca0",),
        raw_economics_sha256="6502f2b7c4841c3319424b02894db66ac4f8274d6a160360882a6f317e5d9528",
        identity_sha256="bdf3e782041b558d9783bc871d8e4b26ad2b8f4ae7caacd010e61c09926119ce",
        terminal_actions_sha256="bee1529a33c688494c5ba5eefb0cee04d733710f39d215231cedaa868bd7bcfa",
        all_actions_sha256="fa038ec0778126652c9a6cb959ec3950328a7ad94647682b04d22bbd83a7a9ad",
        action_coverage_sha256="847185eace0d7f2c9b8caec9ba9eb824b087873b30ee9b0c81a64bea4ead3d01",
        factor_rows=951, factor_economics_sha256="d7268ef79fabdc944afbd2f470b6642419a9ce50e526672d412c321461e72a47",
        signal_sha256="19801dac4e0b8f291047f81947ec3abf5dc7c99bf029069ff2bf74fc02e26c94",
    ),
    _pin(
        symbol="FB", security_id="US:EODHD:50992e1b-839e-528b-bf5a-8b4216b79779",
        target_id="06b8e2e6e06e71b682f4792e0bc3cff9ace1c6bea0a4d07ee4e7aa3f99afa2d7",
        provider_symbol="FB_old.US", active_from="2015-01-02", active_to="2022-06-27",
        history_effective_from="2015-01-01", history_effective_to="",
        identity_source="eodhd_exchange_symbols", identity_source_sha256=COMMON_IDENTITY_SHA,
        terminal_event_id="937e4b724c59ee2e5355832ae14225d50a9ba0239a9e6881b50a5ce017f0bf45",
        terminal_source_sha256="508c7e9504b2c51a4cfad3aaa15ebffbc0f2786b14203af3ebb6b89893326e39",
        full_wiki_rows=1472, full_wiki_start="2012-05-18", full_wiki_end="2018-03-27",
        raw_lines_sha256="aa838c12a7c9c2cea588c1d2597da679d64fedf026de4d8f9ba6f779d7270ade",
        extract_sha256="5470f808aff0e6c420769528b2100c2a45f6672c698f23bdc0e09c40b2a87203",
        extract_size=139672, review_start="2015-01-02", review_end="2018-03-27",
        overlap_rows=813, relation_sha256="acc3fca09da894c3970e67f42c64214cdd5eefb56022bba30c7824dbb90718a3",
        price_rows=1884, price_start="2015-01-02", price_end="2022-06-27",
        price_source_sha256s=("524d4491151c92a53715140b3c7fbf898246259fceea15a9f35f526a3911d6b5",),
        raw_economics_sha256="b2d11361ea2d9bea0cb9bae6d21b9667b7eca5c10a3e5c2249dd95be23b0b5e5",
        identity_sha256="6171a309cfa9bd81835293487c12b34e73875f5350cfecf51975e34e0cf068ab",
        terminal_actions_sha256="a6119e08cce839c500e5c654b12298f41469ddd72a4656bcb4752e7b061d5544",
        all_actions_sha256="a6119e08cce839c500e5c654b12298f41469ddd72a4656bcb4752e7b061d5544",
        action_coverage_sha256="b8efadebf946bba0dde4fb6167e5ca8f8fb05054480f149941e1d375edfd5b37",
        factor_rows=1884, factor_economics_sha256="ddce3b2917295c5a787b2ffa20192e65236dd9992a16c6b5e892ad3f0fc53b92",
        signal_sha256="e914cc6ea38588b3faddc4cd9ca809e4357eb2a41f5fea5de74384541d05d16d",
    ),
    _pin(
        symbol="FOX", security_id="US:EODHD:acd9ed55-bf0c-5b15-b624-1a917bf6078e",
        target_id="7f1cdc57371b12e2913ee5d00f5e262c91a800648786c55d0e78bf97d08270b8",
        provider_symbol="TFCF.US", active_from="2015-01-02", active_to="2019-03-19",
        history_effective_from="2015-01-01", history_effective_to="2019-03-18",
        identity_source="official_identity_repair",
        identity_source_sha256="89fde56d5b8f452b122f2f0656a6814bca2d19ed4e7ae89659d2926391f282a6",
        terminal_event_id="", terminal_source_sha256="",
        full_wiki_rows=7621, full_wiki_start="1987-12-30", full_wiki_end="2018-03-27",
        raw_lines_sha256="beedbf58ea004ff5acb317e90e0ad2293ee9432f9a8724412bcb826cad21b6de",
        extract_sha256="ac4c9bbf710cbcfa07089332dc6ab16bbe8c090023db8a66bb56dc3a0e05dceb",
        extract_size=980932, review_start="2015-01-02", review_end="2018-03-27",
        overlap_rows=813, relation_sha256="1fb94274eeba511ed567a4f56be97fa9c88475deee09a7280623b18ed5f96d9c",
        price_rows=1059, price_start="2015-01-02", price_end="2019-03-19",
        price_source_sha256s=("df1abccdad4353d2f3b17331c3ca7ca52da376805648d437ab7c73c49eed3dc7",),
        raw_economics_sha256="79d8db49283bfe752c68f270a943a538824dfe244edf73b1a61a01c2fadf3e00",
        identity_sha256="d24b0b763aa6bf64e150ffc711058b3302f1f93fbf984d1b358e03793ddcdb8e",
        terminal_actions_sha256="37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570",
        all_actions_sha256="676c20c77167502492147c309e07b27a1a46d375cebd9e5b99849cbfcb335511",
        action_coverage_sha256="1d2ab250c605559bd06308ea382a417c1bede74568912fb6039f57304f3d0762",
        factor_rows=1059, factor_economics_sha256="449c555b2f8ee788887e0ab72b6a460b1752e1d23ff0361da17f7342bd997667",
        signal_sha256="23989f244479f1759130ddbf67926278a15b71566d657b8bd2eef9890c2ba457",
    ),
    _pin(
        symbol="FOXA", security_id="US:EODHD:9398e16f-425d-5a51-8720-35fba7433f28",
        target_id="7516f04c6e27d612002fc1d3468720f45ccc1b141e7e8a5fe78564c261d87729",
        provider_symbol="TFCFA.US", active_from="2015-01-02", active_to="2019-03-19",
        history_effective_from="2015-01-01", history_effective_to="2019-03-18",
        identity_source="official_identity_repair",
        identity_source_sha256="89fde56d5b8f452b122f2f0656a6814bca2d19ed4e7ae89659d2926391f282a6",
        terminal_event_id="", terminal_source_sha256="",
        full_wiki_rows=5549, full_wiki_start="1996-03-11", full_wiki_end="2018-03-27",
        raw_lines_sha256="2573b3e10ba24ce69e7b5acd20ec232f3701d7c78a9f70a2e9c38aec433b639c",
        extract_sha256="8ce27540be2cc5a0ec6ee8ecb264c09504f3db2504bc88205c8c65c323e6dbf4",
        extract_size=722768, review_start="2015-01-02", review_end="2018-03-27",
        overlap_rows=813, relation_sha256="1331326ad9827d7f5c82c004fbe1f69a9425dee166a3d6921633a3fcbbcb4da9",
        price_rows=1059, price_start="2015-01-02", price_end="2019-03-19",
        price_source_sha256s=("46bfceffe1de95ffce022fb841ca822b70f9e309dbea1cf542490359e99e6092",),
        raw_economics_sha256="4fa3162cdc2174e3b1f54418a584b5a23cce6af2eec39180ed79a0e874b1d312",
        identity_sha256="8722850cbbc19b133f039fd445f6b3a488538a99eee317ef33d74857a86ff591",
        terminal_actions_sha256="37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570",
        all_actions_sha256="540eb5ad255a1ea6b953b94b054f8e3430fe9f2a8ca66e419dab5966cbc306ae",
        action_coverage_sha256="1d2ab250c605559bd06308ea382a417c1bede74568912fb6039f57304f3d0762",
        factor_rows=1059, factor_economics_sha256="449c555b2f8ee788887e0ab72b6a460b1752e1d23ff0361da17f7342bd997667",
        signal_sha256="6c1395187d7c203862c7e4031c6798667a95fe84b5f2c2c6031913bbf08ef3ae",
    ),
    _pin(
        symbol="INFO", security_id="US:EODHD:67bc140c-d643-57d1-8a4f-e44be550d27c",
        target_id="599705d408171c2606e2a09958f8fccca2571bd1df4ca7dbadcd4074bbdeebba",
        provider_symbol="INFO_old1.US", active_from="2015-01-02", active_to="2022-02-25",
        history_effective_from="2015-01-01", history_effective_to="",
        identity_source="eodhd_exchange_symbols", identity_source_sha256=COMMON_IDENTITY_SHA,
        terminal_event_id="963c907c1ffbce39087638b273b7eacc7c16a551bc1f612a63a4dfdd2e1b84f0",
        terminal_source_sha256="91aed4ce14bd53aa085ff3efdc252597481efeb750b9006f524b246cfb19b422",
        full_wiki_rows=205, full_wiki_start="2017-06-02", full_wiki_end="2018-03-27",
        raw_lines_sha256="65270c6ad23368c35fe2df4e1602bc086584ea53ab2a781c7178a6b82672ef58",
        extract_sha256="23dfb8ca1ef163d58f767563791ae4e11c597adcfa9ac00986ea4f5753fa79be",
        extract_size=18874, review_start="2015-01-02", review_end="2018-03-27",
        overlap_rows=205, relation_sha256="8ff2717ad7ce6b626a91c05f786a487b0885940f01432876d6a1662379954581",
        price_rows=1801, price_start="2015-01-02", price_end="2022-02-25",
        price_source_sha256s=("03af2411a7689d0867581beaa609467dc2650ea595560d2c041484f79d7162d7",),
        raw_economics_sha256="b330f1f1b85ee3f9935b1f0bfb9d74422fdf3cf0447735c22091877d929ef95e",
        identity_sha256="3f8c39ad3a2393e62f26f1406884e1834a7716f6cc007338b608de31f017d046",
        terminal_actions_sha256="be2457e250ceba8222de00fd8dda409e7f9ae1a0052a987faf3724a2b004ad05",
        all_actions_sha256="8caba017c53e4c7f9b8c7593f1b108e5289fa79cbeffd68be8999052a913f480",
        action_coverage_sha256="ce354fd31029bc42b8c51152cd8d713779c3e1c0fad0f952f493d237ee4521c6",
        factor_rows=1801, factor_economics_sha256="cfe1fdaf758e5b3bd234da28d71c91a08322ee1e9eea0e89e44fb627d6dcbd70",
        signal_sha256="97031cec7d1b7624e8a246b8cc12c954167c0e7662d6300afa7253c21a5fdeb0",
    ),
    _pin(
        symbol="NFX", security_id="US:EODHD:5761939d-b25c-58b8-9f70-674b2b505362",
        target_id="da410eb4d833807a0cb86317cdd3c14011b49d06bf34528abbe7e75c6e32eafa",
        provider_symbol="NFX_old.US", active_from="2015-01-02", active_to="2019-02-13",
        history_effective_from="2015-01-01", history_effective_to="",
        identity_source="eodhd_exchange_symbols", identity_source_sha256=COMMON_IDENTITY_SHA,
        terminal_event_id="135d28c0ee9edf5bd26cf2f06a4ac73e034c42eaddc08b286dbb77540144a04d",
        terminal_source_sha256="853213f50cecdad3991d49964b784b7ed154d2414dc33a9934c82c485e668b31",
        full_wiki_rows=6135, full_wiki_start="1993-11-12", full_wiki_end="2018-03-27",
        raw_lines_sha256="21e2717cbcd3c2f1bb3c8a29a2abeccb1772d1e64c9ac1e3f79dab872f51a80d",
        extract_sha256="09de2f6350f3207df8d9b6b662e9a36077eb19c8c6e0bb04adbb2c145afc9189",
        extract_size=546724, review_start="2015-01-02", review_end="2018-03-27",
        overlap_rows=813, relation_sha256="b9a0a75317d404fee885801c0da9752fcb44fc08a593adcb7974d25aabd44477",
        price_rows=1036, price_start="2015-01-02", price_end="2019-02-13",
        price_source_sha256s=("bd492c1a08c244f57e3c777316aab3763b2e6c7634e6c419eac6439c2f29c5a2",),
        raw_economics_sha256="3ad2c01ac0f126c8b1a73ab125be17d88f310e6cfe73e39bfe190f3016ca3a01",
        identity_sha256="bcbb55dd5d46426cffdec23319eb88828edc63634ce6c3ce6629fd81418b14d6",
        terminal_actions_sha256="58d4b8db6b58fcb37e784dc30aa22b4e7d938ea8d1f028d4517caee3f8549a0a",
        all_actions_sha256="58d4b8db6b58fcb37e784dc30aa22b4e7d938ea8d1f028d4517caee3f8549a0a",
        action_coverage_sha256="b8efadebf946bba0dde4fb6167e5ca8f8fb05054480f149941e1d375edfd5b37",
        factor_rows=1036, factor_economics_sha256="d4696d1d68d90199c1b9d511993b3058a83c2f6d5f1979d977f57bbc5f28bb49",
        signal_sha256="1de6456463fbdcb5796e4b4009d0a87a9ba24f062c91f6cd875cea363c4f72bc",
    ),
    _pin(
        symbol="SCG", security_id="US:EODHD:a839e937-8380-540e-b610-480496ece1bb",
        target_id="addfc00ac2573c1efd61fff6686797169ffdc2b1c6da1552521d936d80b19a03",
        provider_symbol="SCG.US", active_from="2015-01-02", active_to="2018-12-31",
        history_effective_from="2015-01-01", history_effective_to="",
        identity_source="eodhd_exchange_symbols", identity_source_sha256=COMMON_IDENTITY_SHA,
        terminal_event_id="729fc54f87e508f38e78684e897afe84824e7ba2787a4d681a6f7fa4f3f222c8",
        terminal_source_sha256="9118b9de5bd7a1ca7f2fac93dd36d7958d632d01881609889ab35c04b1b487cc",
        full_wiki_rows=7621, full_wiki_start="1987-12-30", full_wiki_end="2018-03-27",
        raw_lines_sha256="5d4b568c6b937720fe9c5b0bb1cba9a227b9a166cd21564d79902f572a5dfb18",
        extract_sha256="a4e7101178eba73d3d53c27b4c51dc5a92eeb4a92b5e024c5d00e4de4c3228e3",
        extract_size=967197, review_start="2015-01-02", review_end="2018-03-27",
        overlap_rows=813, relation_sha256="1e67e6951a66e566e2e5e905c62cfe51da284c24bb7963a49ebc32cc52a27427",
        price_rows=1006, price_start="2015-01-02", price_end="2018-12-31",
        price_source_sha256s=("f5079257ab388a79663c8f7bf1850844c883ce79c98c43ec271ddc2c710415f4",),
        raw_economics_sha256="109a180dc7a797b1318f47aff40fabc2395703bf9e6ae4124d7b9e9fc28c2e49",
        identity_sha256="39405751cb35ccb1f9e79db44a776ea115e0db00232eb2937941cd2e1cb773fb",
        terminal_actions_sha256="ef8f33140fbdb59fb2fb17f79a7767ab55c6f6825af10af24fef4a0258e1b515",
        all_actions_sha256="cd673cb0ada5ee303f1e5aa79622e8f8bc6268108f2ab97a93729054dc726942",
        action_coverage_sha256="2a4c792fdba32f65753e55d0d697137f702ab7365aebae53bf29ee47074ee410",
        factor_rows=1006, factor_economics_sha256="ee10da91f8f10c9210d88e5caad411765bcfce7f01d002be8661e401d68b582d",
        signal_sha256="925b34caf2949bd5d5ec03121d36a0ffc39af81bd3a01fd434da71dbede12863",
    ),
    _pin(
        symbol="SNDK", security_id="US:EODHD:3ae520c3-9c0f-58f2-a47e-9c5b3095f8ec",
        target_id="5dd6c8006be4eeb5f7947aac282769adc1239c83ec9d14169fba5eea8e564f79",
        provider_symbol="SNDK_old.US", active_from="2015-01-02", active_to="2016-05-11",
        history_effective_from="2015-01-01", history_effective_to="",
        identity_source="eodhd_exchange_symbols", identity_source_sha256=COMMON_IDENTITY_SHA,
        terminal_event_id="85f01c5058b4f3954bf004292c38af7f8a0b59afa382313a1efc4d34fe20c63c",
        terminal_source_sha256="f84825ed01af6f889e4498f43c1e5830a190f54f6342bcf6bf1d4cf8dd16f426",
        full_wiki_rows=5162, full_wiki_start="1995-11-08", full_wiki_end="2016-05-11",
        raw_lines_sha256="7b38012925dae32ceb93032bc59208f778d8a87680be3b13cdd0ea5ec74374f6",
        extract_sha256="7c78cb0618223072c2183470dce274a50a81a655aab5e16f81e6797cde32295c",
        extract_size=715394, review_start="2015-01-02", review_end="2016-05-11",
        overlap_rows=342, relation_sha256="b3fa31f0b31892e29126616d10e261bc22d4934fe8df0cf126bfb9dd339f8bcf",
        price_rows=342, price_start="2015-01-02", price_end="2016-05-11",
        price_source_sha256s=("433fabc46c3872f27d8738a144289602ea9d646b55156a494d685e963e4f78ed",),
        raw_economics_sha256="dcf56f2af91ae58c8f823fee12a3d59ea36ebe99eae65447cf7cf05980dbd593",
        identity_sha256="e9f721a6680341a7aa1c4ead753a46505c1d69465925ec8c5a40708e0b5321b3",
        terminal_actions_sha256="a8e20cd4f89952a5867b0db09e3ac869323d8d4449088c659353d106da053d98",
        all_actions_sha256="286d8d7b4e942b54410ead81c5fa8b79ed366fee49903c04ef9b2acd00c53e64",
        action_coverage_sha256="78e9a5080fdba57dceb00c027bb1ab671139b1282c99058570b3a44cffb32419",
        factor_rows=342, factor_economics_sha256="a34fb216a0f027bef9a21e0344a3c68a9004e0aa28eaedda4213c4d39ff7b709",
        signal_sha256="7969dac348d0b81d393f862d1c935b2491934e33399a6be4059322737fcda540",
    ),
    _pin(
        symbol="STI", security_id="US:EODHD:3791f942-824e-5b24-b243-2aabd4cfaa56",
        target_id="b5f68dcfc88bb04c75a326e0667864ba403d443ffb195c6dcab2c94d36ca576a",
        provider_symbol="STI_old.US", active_from="2015-01-02", active_to="2019-12-06",
        history_effective_from="2015-01-01", history_effective_to="",
        identity_source="eodhd_exchange_symbols", identity_source_sha256=COMMON_IDENTITY_SHA,
        terminal_event_id="9a5faa316f359005cf44ebe58b35a43f1b0c5f941514887ce9e76d56ca883d29",
        terminal_source_sha256="094d8ee3bb3b2cdf33ff4492d6cc93a3738098e4d8bfbeb0c7962d8d9bc3208d",
        full_wiki_rows=7621, full_wiki_start="1987-12-30", full_wiki_end="2018-03-27",
        raw_lines_sha256="6de4a417c833280fc944881596394b585b6dec9183c534e354596d6bc6ca50b0",
        extract_sha256="a66f7d6a39b53a2917ca3d228931dd5f7e23ea112bc172380978575aaf93dee0",
        extract_size=975408, review_start="2015-01-02", review_end="2018-03-27",
        overlap_rows=813, relation_sha256="f9f3aad2af50bbe3569a7161705a66f4a37081d02eef2fa48f47486462385d14",
        price_rows=1242, price_start="2015-01-02", price_end="2019-12-06",
        price_source_sha256s=("a0f1765c37931a144a04327063e02abcbc6e76b23dfc751923f685fc970e2df9",),
        raw_economics_sha256="fbc6edbe5f98a09101c649f8adb02130d1a8537de4e6867a279fe8fe071b4ce7",
        identity_sha256="c1f728081e8c17d3f6bbea5efc18d1d92c8424a1918eaad2705ab00758cab32a",
        terminal_actions_sha256="27174a65f2d5002a1b04e1e7692c05205ba2728daf11aec7bf30abbf5e75bcf8",
        all_actions_sha256="50b2b0dbf7c254c914cb566757edacd5808a4994a87d3de1f2eea376062957ab",
        action_coverage_sha256="becebb09f9393ad8637a28a1ae67bfcdd7f42785887202722667e1d5accbc723",
        factor_rows=1242, factor_economics_sha256="7d91be9c5dfb20d2636eb3d83410386e0565b9ecbbe0097201f324c9fa809c25",
        signal_sha256="7dec8d17caee5a3494f7d274b5ac2f460bab102223b7811eb3510e0295215d02",
    ),
    _pin(
        symbol="TE", security_id="US:EODHD:306e4817-7ce8-5279-a033-93031e72f3f1",
        target_id="a17b2c14178839f2206e64fef7f77c68d90a2849d3e81b782401348c0b6e5c71",
        provider_symbol="TE_old1.US", active_from="2015-01-02", active_to="2016-06-30",
        history_effective_from="2015-01-01", history_effective_to="",
        identity_source="eodhd_exchange_symbols", identity_source_sha256=COMMON_IDENTITY_SHA,
        terminal_event_id="4412df48d95a824adb7255e7a61b8746bcc978ebe466566e3e1ea2556988bf1b",
        terminal_source_sha256="6e21a991c2ab18fcfb7cd497034f1eb00977bd405ded39b1a3817bc8b032c264",
        full_wiki_rows=7985, full_wiki_start="1984-10-29", full_wiki_end="2016-06-30",
        raw_lines_sha256="63eb5ae21f69709e3a3e920cf02470b0638fc26988734951282ffc73e14bcc6d",
        extract_sha256="b8886f234d2ff49583aa36ac146fae29bcbd95bd051faa5334159a645c227c51",
        extract_size=1092730, review_start="2015-01-02", review_end="2016-06-30",
        overlap_rows=377, relation_sha256="dd9cd4244d79eaf29288f2cff1e5a79f2fb4f8930ace32d08c951e5ccae373e2",
        price_rows=377, price_start="2015-01-02", price_end="2016-06-30",
        price_source_sha256s=("96552a8424d4e92762fee7908631c59a5c9ed5566b0a80b74b4fcc53e21fc596",),
        raw_economics_sha256="8766a621e6ada0060cb7eeed2901504e6ec08d513eb3127372555f6057425e9e",
        identity_sha256="91cacdddb31b41dd202fa106954a422e07ea6cc018ae820824c9403740fcf568",
        terminal_actions_sha256="6073adf3d78bc5721c247af6b3a59d5633a936ad604be5bcc34dbbc983333f23",
        all_actions_sha256="1a6b4e86951f288c18a77799783b0fb23ae90799d12006b774a8c8470cd82b30",
        action_coverage_sha256="df747c642b0633d29f5b28dcddf90e5b1c9dd3cae90b4e07f0eb1e87a365ca2c",
        factor_rows=377, factor_economics_sha256="48b45c139b6a0058c0faf49e90ac0682a84991efd0909d4e691e8ff015f1c5df",
        signal_sha256="c7bd3999094ef40c2bf1a729ec64a9a2053a3a6aa9d3ac41b0a22b0b7a6ba2fa",
    ),
)


@dataclass(frozen=True)
class ArchiveArtifact:
    dataset: str
    source: str
    source_url: str
    content_type: str
    extension: str
    payload: bytes
    retrieved_at: str

    @property
    def source_hash(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    def object_path(self, completed_session: str) -> str:
        return f"archives/{completed_session}/{self.source_hash}.{self.extension}.gz"


@dataclass(frozen=True)
class EvidenceBundle:
    artifacts: tuple[ArchiveArtifact, ...]
    wiki_rows: Mapping[str, pd.DataFrame]
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedRepair:
    release: DataRelease
    release_etag: str | None
    pointer_etags: Mapping[str, str | None]
    version_state: Mapping[str, Mapping[str, Any]]
    frame: pd.DataFrame
    artifacts: tuple[ArchiveArtifact, ...]
    wiki_zip_path: Path
    targets: tuple[TargetPin, ...]
    allowed_index_identity_gap_fingerprints: tuple[str, ...]
    planned_source_archive_version: str
    planned_release: DataRelease | None
    source_archive_inventory_sha256: str
    summary: Mapping[str, Any]


FailureInjector = Callable[[str], None]


def _noop_injector(_stage: str) -> None:
    return None


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _date(value: Any) -> str:
    parsed = pd.to_datetime(_text(value), errors="coerce")
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _one(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"{label} inventory changed: expected=1; observed={len(rows)}.")
    return rows.iloc[0]


def _safe_path(root: Path, object_path: str) -> Path:
    base = root.resolve()
    path = (base / object_path).resolve()
    if path == base or base not in path.parents:
        raise ValueError(f"Archive path escapes repository: {object_path}.")
    return path


def _economic_actions(frame: pd.DataFrame) -> list[list[Any]]:
    columns = (
        "event_id", "security_id", "action_type", "effective_date", "ex_date",
        "ratio", "cash_amount", "currency", "new_security_id", "new_symbol",
        "official", "source", "source_hash", "source_url",
    )
    output: list[list[Any]] = []
    for row in frame.sort_values(["effective_date", "event_id"]).itertuples(index=False):
        item: list[Any] = []
        for column in columns:
            value = getattr(row, column)
            if pd.isna(value):
                value = None
            elif isinstance(value, (bool, np.bool_)):
                value = bool(value)
            elif isinstance(value, (float, np.floating)):
                value = format(float(value), ".17g")
            else:
                value = str(value)
            item.append(value)
        output.append(item)
    return output


def _economic_factors(frame: pd.DataFrame) -> list[list[str]]:
    return [
        [
            _date(row.session),
            format(float(row.split_factor), ".17g"),
            format(float(row.total_return_factor), ".17g"),
        ]
        for row in frame.sort_values("session").itertuples(index=False)
    ]


def _raw_economics(frame: pd.DataFrame) -> list[list[str]]:
    return [
        [
            str(row.date),
            *[
                format(float(getattr(row, column)), ".17g")
                for column in ("open", "high", "low", "close", "volume")
            ],
        ]
        for row in frame.sort_values("date").itertuples(index=False)
    ]


def _relation_fingerprint(joined: pd.DataFrame) -> str:
    records: list[list[str]] = []
    for row in joined.itertuples(index=False):
        records.append(
            [str(row.date)]
            + [
                format(float(getattr(row, f"{column}_eod")), ".17g")
                for column in ("open", "high", "low", "close", "volume")
            ]
            + [
                format(float(getattr(row, f"{column}_wiki")), ".17g")
                for column in ("open", "high", "low", "close", "volume")
            ]
        )
    return _canonical_sha(records)


def _identity_schema_inventory() -> list[list[str]]:
    return [
        [
            symbol,
            pin.master_primary_symbol,
            pin.master_exchange,
            pin.master_asset_type,
            pin.master_currency,
            pin.master_country,
            pin.history_symbol,
            pin.history_exchange,
            pin.raw_price_currency,
        ]
        for symbol, pin in sorted(IDENTITY_SCHEMA_PINS.items())
    ]


def _assert_identity_schema(
    target: TargetPin,
    master: Mapping[str, Any],
    history: Mapping[str, Any],
) -> dict[str, str]:
    pin = IDENTITY_SCHEMA_PINS.get(target.symbol)
    if pin is None:
        raise ValueError(f"{target.symbol} has no exact identity-schema pin.")
    observed = {
        "master_primary_symbol": _text(master.get("primary_symbol")).upper(),
        "master_exchange": _text(master.get("exchange")).upper(),
        "master_asset_type": _text(master.get("asset_type")).upper(),
        "master_currency": _text(master.get("currency")).upper(),
        "master_country": _text(master.get("country")).upper(),
        "history_symbol": _text(history.get("symbol")).upper(),
        "history_exchange": _text(history.get("exchange")).upper(),
    }
    expected = {
        "master_primary_symbol": pin.master_primary_symbol,
        "master_exchange": pin.master_exchange,
        "master_asset_type": pin.master_asset_type,
        "master_currency": pin.master_currency,
        "master_country": pin.master_country,
        "history_symbol": pin.history_symbol,
        "history_exchange": pin.history_exchange,
    }
    if observed != expected:
        raise ValueError(f"{target.symbol} exact identity schema changed.")
    return observed


def _assert_raw_price_currency(
    target: TargetPin,
    prices: pd.DataFrame,
) -> str:
    pin = IDENTITY_SCHEMA_PINS.get(target.symbol)
    if pin is None:
        raise ValueError(f"{target.symbol} has no exact identity-schema pin.")
    currencies = tuple(
        sorted({_text(value).upper() for value in prices["currency"]})
    )
    if currencies != (pin.raw_price_currency,):
        raise ValueError(f"{target.symbol} exact raw price currency changed.")
    return pin.raw_price_currency


def _triple_signal_frame(frame: pd.DataFrame) -> pd.DataFrame:
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
        settings=((10, 1.0), (11, 2.0), (12, 3.0)),
        atr_method="wilder",
        exit_down_count=2,
    )


def _signal_hash(frame: pd.DataFrame) -> str:
    records: list[list[Any]] = []
    for session, values in zip(
        frame["Date"].dt.date.astype(str),
        frame[list(SIGNAL_COLUMNS)].itertuples(index=False, name=None),
        strict=True,
    ):
        records.append(
            [
                session,
                *[
                    bool(value)
                    if isinstance(value, (bool, np.bool_))
                    else int(value)
                    for value in values
                ],
            ]
        )
    return _canonical_sha(records)


def _extract_inventory(targets: Sequence[TargetPin]) -> list[list[Any]]:
    return [
        [
            target.symbol,
            target.security_id,
            target.full_wiki_rows,
            target.raw_lines_sha256,
            target.extract_sha256,
            target.extract_size,
        ]
        for target in targets
    ]


# Hash of the complete, sorted 14-extract inventory above.  It prevents a
# single target from being silently added, removed, or rebound to another SID.
EXTRACT_INVENTORY_SHA256 = (
    "173635ad3c82264826d118bcfd963cc884e50365539ee623c4b24d682060b0f5"
)


def load_evidence_bundle(
    wiki_zip_path: Path,
    *,
    targets: Sequence[TargetPin] = TARGETS,
    enforce_reviewed_profile: bool = True,
) -> EvidenceBundle:
    if not wiki_zip_path.is_file():
        raise FileNotFoundError(f"Frozen WIKI ZIP is missing: {wiki_zip_path}.")
    if enforce_reviewed_profile and (
        wiki_zip_path.stat().st_size != WIKI_ZIP_SIZE
        or _sha256_file(wiki_zip_path) != WIKI_ZIP_SHA256
    ):
        raise ValueError("Frozen WIKI ZIP hash/size changed.")
    if enforce_reviewed_profile and _canonical_sha(_extract_inventory(targets)) != EXTRACT_INVENTORY_SHA256:
        raise ValueError("Frozen WIKI14 extract inventory pin changed.")

    target_by_symbol = {target.symbol: target for target in targets}
    if len(target_by_symbol) != len(targets):
        raise ValueError("WIKI14 symbol inventory is duplicated.")
    lines: dict[str, list[bytes]] = {symbol: [] for symbol in target_by_symbol}
    digests = {symbol: hashlib.sha256() for symbol in target_by_symbol}
    member_digest = hashlib.sha256()
    header = b""
    with zipfile.ZipFile(wiki_zip_path) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != WIKI_MEMBER:
            raise ValueError("Frozen WIKI member inventory changed.")
        info = infos[0]
        if enforce_reviewed_profile and (
            info.file_size != WIKI_MEMBER_SIZE or info.CRC != WIKI_MEMBER_CRC32
        ):
            raise ValueError("Frozen WIKI member size/CRC changed.")
        with archive.open(info, "r") as member:
            for number, line in enumerate(member, start=1):
                member_digest.update(line)
                if number == 1:
                    if not line.startswith(b"ticker,date,open,high,low,close,volume,"):
                        raise ValueError("Frozen WIKI CSV header changed.")
                    header = line
                    continue
                fields = line.split(b",", 1)
                if len(fields) != 2:
                    continue
                try:
                    symbol = fields[0].decode("ascii")
                except UnicodeDecodeError as exc:
                    raise ValueError("Frozen WIKI ticker encoding changed.") from exc
                if symbol in lines:
                    lines[symbol].append(line)
                    digests[symbol].update(line)
    if enforce_reviewed_profile and member_digest.hexdigest() != WIKI_MEMBER_SHA256:
        raise ValueError("Frozen WIKI member SHA-256 changed.")

    artifacts: list[ArchiveArtifact] = []
    wiki_rows: dict[str, pd.DataFrame] = {}
    target_audits: list[dict[str, Any]] = []
    for target in targets:
        payload = header + b"".join(lines[target.symbol])
        artifact = ArchiveArtifact(
            dataset=f"kaggle_quandl_wiki_{target.symbol.lower()}_full_price_extract",
            source=f"kaggle_quandl_wiki_{target.symbol.lower()}_full_price_extract",
            source_url=WIKI_DOWNLOAD_URL,
            content_type="text/csv",
            extension="csv",
            payload=payload,
            retrieved_at=WIKI_RETRIEVED_AT,
        )
        if enforce_reviewed_profile and (
            len(lines[target.symbol]) != target.full_wiki_rows
            or digests[target.symbol].hexdigest() != target.raw_lines_sha256
            or artifact.source_hash != target.extract_sha256
            or len(payload) != target.extract_size
        ):
            raise ValueError(f"Frozen WIKI {target.symbol} extract pin changed.")
        parsed = pd.read_csv(io.BytesIO(payload))
        parsed["date"] = parsed["date"].astype(str)
        if parsed["date"].duplicated().any():
            raise ValueError(f"Frozen WIKI {target.symbol} dates are duplicated.")
        if enforce_reviewed_profile and (
            str(parsed["date"].min()) != target.full_wiki_start
            or str(parsed["date"].max()) != target.full_wiki_end
        ):
            raise ValueError(f"Frozen WIKI {target.symbol} date inventory changed.")
        artifacts.append(artifact)
        wiki_rows[target.symbol] = parsed
        target_audits.append(
            {
                "symbol": target.symbol,
                "security_id": target.security_id,
                "full_rows": len(parsed),
                "raw_lines_sha256": digests[target.symbol].hexdigest(),
                "extract_sha256": artifact.source_hash,
                "extract_size": len(payload),
            }
        )
    return EvidenceBundle(
        artifacts=tuple(artifacts),
        wiki_rows=wiki_rows,
        audit={
            "zip_sha256": _sha256_file(wiki_zip_path),
            "zip_size": wiki_zip_path.stat().st_size,
            "member_name": WIKI_MEMBER,
            "member_sha256": member_digest.hexdigest(),
            "member_size": info.file_size,
            "member_crc32": f"{info.CRC:08x}",
            "extract_inventory_sha256": _canonical_sha(_extract_inventory(targets)),
            "extracts": target_audits,
        },
    )


def _action_coverage(
    target: TargetPin,
    wiki: pd.DataFrame,
    actions: pd.DataFrame,
) -> dict[str, Any]:
    wiki_div = sorted(
        [str(row.date), format(float(row.amount), ".17g")]
        for row in pd.DataFrame(
            {
                "date": wiki["date"].astype(str),
                "amount": pd.to_numeric(wiki["ex-dividend"], errors="raise"),
            }
        ).loc[pd.to_numeric(wiki["ex-dividend"], errors="raise").gt(0)].itertuples(index=False)
    )
    current_dividend_frame = actions.loc[
        actions["action_type"].astype(str).eq("cash_dividend")
        & actions["effective_date"].astype(str).between(
            target.review_start, target.review_end
        )
    ]
    current_div = sorted(
        [str(row.effective_date), format(float(row.cash_amount), ".17g")]
        for row in current_dividend_frame.itertuples(index=False)
    )
    wiki_split = sorted(
        [str(row.date), format(float(row.ratio), ".17g")]
        for row in pd.DataFrame(
            {
                "date": wiki["date"].astype(str),
                "ratio": pd.to_numeric(wiki["split_ratio"], errors="raise"),
            }
        ).loc[
            ~pd.to_numeric(wiki["split_ratio"], errors="raise").eq(1.0)
        ].itertuples(index=False)
    )
    split_actions = actions.loc[
        actions["action_type"].astype(str).isin(
            ["split", "stock_dividend", "capital_reduction"]
        )
        & actions["effective_date"].astype(str).between(
            target.review_start, target.review_end
        )
    ]
    current_split = sorted(
        [
            str(row.effective_date),
            str(row.action_type),
            None if pd.isna(row.ratio) else format(float(row.ratio), ".17g"),
        ]
        for row in split_actions.itertuples(index=False)
    )
    return {
        "wiki_dividends": wiki_div,
        "current_dividends": current_div,
        "wiki_dividends_missing_from_current": sorted(
            value for value in wiki_div if value not in current_div
        ),
        "current_dividends_missing_from_wiki": sorted(
            value for value in current_div if value not in wiki_div
        ),
        "wiki_splits": wiki_split,
        "current_split_like_actions": current_split,
    }


def _audit_target(
    target: TargetPin,
    wiki_full: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    master = _one(
        frames["security_master"],
        frames["security_master"]["security_id"].astype(str).eq(target.security_id),
        f"{target.symbol} security_master",
    )
    history = _one(
        frames["symbol_history"],
        frames["symbol_history"]["security_id"].astype(str).eq(target.security_id)
        & frames["symbol_history"]["symbol"].astype(str).str.upper().eq(target.symbol),
        f"{target.symbol} symbol_history",
    )
    identity_schema = _assert_identity_schema(target, master, history)
    identity = {
        "security_id": target.security_id,
        "symbol": target.symbol,
        "provider_symbol": _text(master.get("provider_symbol")),
        "active_from": _date(master.get("active_from")),
        "active_to": _date(master.get("active_to")),
        "master_source": _text(master.get("source")),
        "master_source_hash": _text(master.get("source_hash")),
        "history_effective_from": _date(history.get("effective_from")),
        "history_effective_to": _date(history.get("effective_to")),
        "history_source": _text(history.get("source")),
        "history_source_hash": _text(history.get("source_hash")),
    }
    expected_identity = {
        "security_id": target.security_id,
        "symbol": target.symbol,
        "provider_symbol": target.provider_symbol,
        "active_from": target.active_from,
        "active_to": target.active_to,
        "master_source": target.identity_source,
        "master_source_hash": target.identity_source_sha256,
        "history_effective_from": target.history_effective_from,
        "history_effective_to": target.history_effective_to,
        "history_source": target.identity_source,
        "history_source_hash": target.identity_source_sha256,
    }
    if identity != expected_identity or _canonical_sha(identity) != target.identity_sha256:
        raise ValueError(f"{target.symbol} exact identity interval changed.")

    prices = frames["daily_price_raw"].loc[
        frames["daily_price_raw"]["security_id"].astype(str).eq(target.security_id)
    ].copy()
    raw_price_currency = _assert_raw_price_currency(target, prices)
    prices["date"] = pd.to_datetime(prices["session"], errors="raise").dt.date.astype(str)
    if (
        prices["date"].duplicated().any()
        or len(prices) != target.price_rows
        or str(prices["date"].min()) != target.price_start
        or str(prices["date"].max()) != target.price_end
        or set(prices["source"].astype(str))
        != ({"eodhd_eod", "reviewed_col_scaling_repair"} if target.symbol == "COL" else {"eodhd_eod"})
        or tuple(sorted(prices["source_hash"].astype(str).unique()))
        != target.price_source_sha256s
        or _canonical_sha(_raw_economics(prices)) != target.raw_economics_sha256
    ):
        raise ValueError(f"{target.symbol} exact raw price input changed.")

    wiki = wiki_full.loc[
        wiki_full["date"].astype(str).between(target.review_start, target.review_end)
    ].copy()
    overlap = prices.loc[
        prices["date"].between(target.review_start, target.review_end)
    ].copy()
    joined = overlap.merge(
        wiki,
        on="date",
        suffixes=("_eod", "_wiki"),
        validate="one_to_one",
    ).sort_values("date", ignore_index=True)
    if (
        len(wiki) != target.overlap_rows
        or len(joined) != target.overlap_rows
        or _relation_fingerprint(joined) != target.relation_sha256
    ):
        raise ValueError(f"{target.symbol} exact EODHD/WIKI relation changed.")

    actions = frames["corporate_actions"].loc[
        frames["corporate_actions"]["security_id"].astype(str).eq(target.security_id)
    ].copy()
    all_action_economics = _economic_actions(actions)
    terminal = actions.loc[
        actions["action_type"].astype(str).isin(
            ["cash_merger", "stock_merger", "ticker_change", "delisting"]
        )
    ]
    terminal_economics = _economic_actions(terminal)
    if (
        _canonical_sha(all_action_economics) != target.all_actions_sha256
        or _canonical_sha(terminal_economics) != target.terminal_actions_sha256
    ):
        raise ValueError(f"{target.symbol} exact action inventory changed.")
    if target.terminal_event_id:
        terminal_row = _one(
            terminal,
            terminal["event_id"].astype(str).eq(target.terminal_event_id),
            f"{target.symbol} terminal action",
        )
        if _text(terminal_row.get("source_hash")) != target.terminal_source_sha256:
            raise ValueError(f"{target.symbol} terminal source hash changed.")
    elif len(terminal):
        raise ValueError(f"{target.symbol} gained an unreviewed terminal action.")
    coverage = _action_coverage(target, wiki, actions)
    if _canonical_sha(coverage) != target.action_coverage_sha256:
        raise ValueError(f"{target.symbol} action-gap fingerprint changed.")

    factors = frames["adjustment_factors"].loc[
        frames["adjustment_factors"]["security_id"].astype(str).eq(target.security_id)
    ].copy()
    factor_sessions = set(pd.to_datetime(factors["session"]).dt.date.astype(str))
    if (
        len(factors) != target.factor_rows
        or factor_sessions != set(prices["date"])
        or _canonical_sha(_economic_factors(factors)) != target.factor_economics_sha256
    ):
        raise ValueError(f"{target.symbol} factor economics changed.")

    candidate = prices.drop(columns="date").copy()
    candidate["_date"] = pd.to_datetime(candidate["session"]).dt.date.astype(str)
    by_date = wiki.set_index("date")
    replace = candidate["_date"].isin(by_date.index)
    for column in ("open", "high", "low", "close", "volume"):
        candidate.loc[replace, column] = candidate.loc[replace, "_date"].map(
            pd.to_numeric(by_date[column], errors="raise")
        )
    candidate = candidate.drop(columns="_date")
    current_adjusted = apply_adjustment_factors(
        prices.drop(columns="date"), factors, mode="total_return_adjusted"
    ).sort_values("session", ignore_index=True)
    candidate_adjusted = apply_adjustment_factors(
        candidate, factors, mode="total_return_adjusted"
    ).sort_values("session", ignore_index=True)
    current_signals = _triple_signal_frame(current_adjusted)
    substituted_signals = _triple_signal_frame(candidate_adjusted)
    signal_differences = {
        column: int((~current_signals[column].eq(substituted_signals[column])).sum())
        for column in SIGNAL_COLUMNS
    }
    current_signal_hash = _signal_hash(current_signals)
    substituted_signal_hash = _signal_hash(substituted_signals)
    if (
        any(signal_differences.values())
        or current_signal_hash != target.signal_sha256
        or substituted_signal_hash != target.signal_sha256
    ):
        raise ValueError(f"{target.symbol} WIKI substitution changed Triple Supertrend.")

    return {
        "status": "passed_price_only_arbitration",
        "target_id": target.target_id,
        "symbol": target.symbol,
        "security_id": target.security_id,
        "identity": identity,
        "identity_sha256": target.identity_sha256,
        "identity_schema": {
            **identity_schema,
            "raw_price_currency": raw_price_currency,
        },
        "identity_schema_inventory_sha256": IDENTITY_SCHEMA_INVENTORY_SHA256,
        "terminal_event_id": target.terminal_event_id,
        "terminal_source_sha256": target.terminal_source_sha256,
        "raw_price_source_sha256s": list(target.price_source_sha256s),
        "raw_economics_sha256": target.raw_economics_sha256,
        "reviewed_relation": {
            "start": target.review_start,
            "end": target.review_end,
            "session_count": target.overlap_rows,
            "relation_sha256": target.relation_sha256,
        },
        "action_coverage": {
            **coverage,
            "coverage_sha256": target.action_coverage_sha256,
            "all_actions_sha256": target.all_actions_sha256,
            "actions_rewritten": False,
            "price_only_pass_must_not_imply_action_pass": True,
        },
        "factor_coverage": {
            "status": "current_economics_pinned_not_independent_action_factor_pass",
            "row_count": len(factors),
            "economics_sha256": target.factor_economics_sha256,
            "factors_rewritten": False,
            "price_only_pass_must_not_imply_factor_pass": True,
        },
        "triple_supertrend": {
            "current_signal_sha256": current_signal_hash,
            "substituted_signal_sha256": substituted_signal_hash,
            "field_differences": signal_differences,
        },
        "raw_price_rewritten": False,
        "corporate_actions_rewritten": False,
        "adjustment_factors_rewritten": False,
        "identity_rewritten": False,
    }


PROVENANCE_SHA256 = (
    "16691eab9edc01f626d00551ba17e922d3f869d928c13478aa0443fbc329209e"
)
ARCHIVE_ARTIFACT_INVENTORY_SHA256 = (
    "134d0d92fa4e31e6c4deb0ab7fa0a57ccf865e0e7d01f712c08d29e87b493ab2"
)


def _provenance_artifact(
    evidence: EvidenceBundle,
    audits: Sequence[Mapping[str, Any]],
) -> ArchiveArtifact:
    payload = _canonical_json(
        {
            "schema": "us_wiki14_price_only_arbitration/v1",
            "reviewed_at": REVIEWED_AT,
            "scope": {
                "passed_price_only_security_ids": [
                    target.security_id for target in TARGETS
                ],
                "write_dataset": DATASET,
                "non_write_datasets": [
                    "daily_price_raw", "corporate_actions", "adjustment_factors",
                    "security_master", "symbol_history", "index_constituent_anchors",
                    "index_membership_events", "lifecycle_resolutions",
                ],
                "generic_symbol_or_ticker_exception_allowed": False,
                "identity_schema_inventory_sha256": (
                    IDENTITY_SCHEMA_INVENTORY_SHA256
                ),
                "archive_effective_date": ARCHIVE_EFFECTIVE_DATE,
            },
            "frozen_evidence": dict(evidence.audit),
            "price_arbitrations": list(audits),
            "license_policy": {
                "formal_license_name": "Unknown",
                "allowed_scope": "private_internal_only",
                "redistribution_allowed": False,
                "public_publication_allowed": False,
                "local_apply_ack_required": True,
                "private_r2_publisher_ack_required_separately": True,
                "fail_closed": True,
            },
        }
    )
    return ArchiveArtifact(
        dataset="reviewed_us_wiki14_price_only_arbitration",
        source="reviewed_us_wiki14_price_only_arbitration",
        source_url=WIKI_DOWNLOAD_URL,
        content_type="application/json",
        extension="json",
        payload=payload,
        retrieved_at=REVIEWED_AT,
    )


def _read_archived_payload(
    repository: LocalDatasetRepository, row: Mapping[str, Any]
) -> bytes:
    path = _safe_path(repository.root, _text(row.get("object_path")))
    if not path.is_file():
        raise FileNotFoundError(f"Archived payload is missing: {path}.")
    try:
        payload = gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise ValueError(f"Archived payload is not valid gzip: {path}.") from exc
    if hashlib.sha256(payload).hexdigest() != _text(row.get("source_hash")):
        raise ValueError(f"Archived payload hash changed: {path}.")
    return payload


def _artifact_row(
    artifact: ArchiveArtifact,
    *,
    columns: Sequence[str],
) -> dict[str, Any]:
    values = {
        "archive_id": artifact.source_hash,
        "dataset": artifact.dataset,
        "object_path": artifact.object_path(ARCHIVE_EFFECTIVE_DATE),
        "content_type": artifact.content_type,
        "effective_date": ARCHIVE_EFFECTIVE_DATE,
        "source": artifact.source,
        "retrieved_at": artifact.retrieved_at,
        "source_hash": artifact.source_hash,
        "source_url": artifact.source_url,
    }
    return {column: values.get(column, np.nan) for column in columns}


def _verify_artifact_row(
    repository: LocalDatasetRepository,
    row: Mapping[str, Any],
    artifact: ArchiveArtifact,
) -> None:
    expected = {
        "archive_id": artifact.source_hash,
        "dataset": artifact.dataset,
        "object_path": artifact.object_path(ARCHIVE_EFFECTIVE_DATE),
        "content_type": artifact.content_type,
        "effective_date": ARCHIVE_EFFECTIVE_DATE,
        "source": artifact.source,
        "retrieved_at": artifact.retrieved_at,
        "source_hash": artifact.source_hash,
        "source_url": artifact.source_url,
    }
    changed = [
        key
        for key, value in expected.items()
        if (_date(row.get(key)) if key == "effective_date" else _text(row.get(key)))
        != value
    ]
    if changed:
        raise ValueError(
            f"Existing WIKI14 artifact row changed: {', '.join(changed)}."
        )
    if _read_archived_payload(repository, row) != artifact.payload:
        raise ValueError("Existing WIKI14 artifact payload bytes changed.")


def _append_or_verify_artifacts(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    artifacts: Sequence[ArchiveArtifact],
) -> tuple[pd.DataFrame, bool]:
    existence: list[bool] = []
    for artifact in artifacts:
        rows = archive.loc[archive["archive_id"].astype(str).eq(artifact.source_hash)]
        if len(rows) > 1:
            raise ValueError("WIKI14 archive_id is duplicated.")
        existence.append(len(rows) == 1)
        if len(rows) == 1:
            _verify_artifact_row(
                repository,
                rows.iloc[0],
                artifact,
            )
    if any(existence) and not all(existence):
        raise ValueError("WIKI14 evidence is only partially archived.")
    if all(existence):
        return archive.copy(), False
    additions = pd.DataFrame(
        [
            _artifact_row(
                artifact,
                columns=archive.columns,
            )
            for artifact in artifacts
        ],
        columns=archive.columns,
    )
    output = pd.concat([archive, additions], ignore_index=True)
    if output["archive_id"].astype(str).duplicated().any():
        raise ValueError("WIKI14 candidate contains duplicate archive IDs.")
    return output, True


def _source_archive_inventory_sha256(frame: pd.DataFrame) -> str:
    """Hash every logical source-archive field independent of Parquet dtypes."""

    columns = sorted(str(column) for column in frame.columns)
    records: list[list[str]] = []
    ordered = frame.sort_values("archive_id", kind="stable")
    for row in ordered.to_dict(orient="records"):
        records.append(
            [
                _date(row.get(column))
                if column == "effective_date"
                else _text(row.get(column))
                for column in columns
            ]
        )
    return _canonical_sha({"columns": columns, "records": records})


def _read_security_subset(
    repository: LocalDatasetRepository,
    dataset: str,
    version: str,
    security_ids: Sequence[str],
) -> pd.DataFrame:
    """Read only the exact reviewed identities, preserving version-chain order."""

    paths = [str(path.resolve()) for path in repository.parquet_paths(dataset, version)]
    if not paths:
        raise RuntimeError(f"{dataset} Parquet inventory is empty.")
    path_order = pd.DataFrame(
        {"filename": paths, "_path_order": range(len(paths))}
    )
    spec = dataset_spec(dataset)
    primary_key = ", ".join(f'"{column}"' for column in spec.primary_key)
    connection = duckdb.connect()
    try:
        connection.register("wiki14_path_order", path_order)
        frame = connection.execute(
            "SELECT * EXCLUDE (filename, _path_order, _row_number) FROM ("
            "SELECT source_rows.*, path_order._path_order, "
            f"ROW_NUMBER() OVER (PARTITION BY {primary_key} "
            "ORDER BY path_order._path_order DESC) AS _row_number "
            "FROM read_parquet(?, union_by_name=true, filename=true) AS source_rows "
            "JOIN wiki14_path_order AS path_order USING (filename) "
            "WHERE security_id = ANY(?)"
            ") WHERE _row_number = 1",
            [paths, sorted(set(security_ids))],
        ).fetchdf()
    finally:
        connection.close()
    derived_partitions = [
        column
        for column in spec.partition_columns
        if column in frame.columns and column not in spec.required_columns
    ]
    if derived_partitions:
        frame = frame.drop(columns=derived_partitions)
    return frame.reset_index(drop=True)


def _version_state(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> tuple[dict[str, str | None], dict[str, dict[str, Any]]]:
    """Pin every release pointer, manifest and immutable Parquet file."""

    pointer_etags: dict[str, str | None] = {}
    state: dict[str, dict[str, Any]] = {}
    for dataset, version in sorted(release.dataset_versions.items()):
        pointer_value = repository.objects.get(repository.current_key(dataset))
        pointer = CurrentPointer.from_bytes(pointer_value.data)
        expected_manifest_path = (
            f"{repository.version_prefix(dataset, version)}/manifest.json"
        )
        if not (
            pointer.dataset == dataset
            and pointer.version == version
            and pointer.manifest_path == expected_manifest_path
        ):
            raise RuntimeError(f"{dataset} release/current pointer mismatch.")
        chain_state: list[dict[str, Any]] = []
        for manifest in repository.manifest_chain(dataset, version):
            manifest_path = (
                repository.root
                / repository.version_prefix(dataset, manifest.version)
                / "manifest.json"
            )
            manifest_bytes = manifest_path.read_bytes()
            if manifest.version == version and (
                pointer.manifest_sha256 != sha256_bytes(manifest_bytes)
            ):
                raise RuntimeError(f"{dataset} pointer/manifest hash mismatch.")
            validate_manifest_files(
                repository.root
                / repository.version_prefix(dataset, manifest.version),
                manifest,
            ).raise_for_errors()
            chain_state.append(
                {
                    "version": manifest.version,
                    "manifest_sha256": sha256_bytes(manifest_bytes),
                    "files": [
                        {
                            "path": item.path,
                            "sha256": item.sha256,
                            "size_bytes": item.size_bytes,
                            "row_count": item.row_count,
                        }
                        for item in manifest.files
                    ],
                }
            )
        pointer_etags[dataset] = pointer_value.etag
        state[dataset] = {
            "version": version,
            "pointer_sha256": sha256_bytes(pointer_value.data),
            "manifest_chain": chain_state,
        }
    return pointer_etags, state


def _load_target_frames(
    repository: LocalDatasetRepository,
    release: DataRelease,
    targets: Sequence[TargetPin],
) -> dict[str, pd.DataFrame]:
    """Load all source-archive rows and only the 14 reviewed security IDs."""

    missing = [
        dataset for dataset in REQUIRED_DATASETS
        if not release.dataset_versions.get(dataset)
    ]
    if missing:
        raise RuntimeError(
            "Current release lacks required datasets: " + ", ".join(missing) + "."
        )
    security_ids = tuple(target.security_id for target in targets)
    frames = {
        dataset: _read_security_subset(
            repository,
            dataset,
            release.dataset_versions[dataset],
            security_ids,
        )
        for dataset in REQUIRED_DATASETS
        if dataset != DATASET
    }
    frames[DATASET] = repository.read_frame(
        DATASET, release.dataset_versions[DATASET]
    )
    return frames


def _verify_bbby_bbt_evidence(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
) -> None:
    for digest, label in (
        (BBBY_BBT_EXTRACT_SHA256, "BBBY/BBT extract"),
        (BBBY_BBT_PROVENANCE_SHA256, "BBBY/BBT provenance"),
    ):
        row = _one(
            archive,
            archive["archive_id"].astype(str).eq(digest),
            label,
        )
        expected = BBBY_BBT_ARTIFACT_PINS[digest]
        changed = [
            key
            for key, value in expected.items()
            if (
                (
                    _date(row.get(key))
                    if key == "effective_date"
                    else _text(row.get(key))
                )
                != value
            )
        ]
        if changed:
            raise ValueError(
                f"Existing {label} metadata changed: {', '.join(changed)}."
            )
        _read_archived_payload(repository, row)


def prepare_repair(
    repository: LocalDatasetRepository,
    *,
    wiki_zip_path: Path = DEFAULT_WIKI_ZIP,
    targets: Sequence[TargetPin] = TARGETS,
) -> PreparedRepair:
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    pointer_etags, version_state = _version_state(repository, release)
    if (
        set(IDENTITY_SCHEMA_PINS) != {target.symbol for target in targets}
        or _canonical_sha(_identity_schema_inventory())
        != IDENTITY_SCHEMA_INVENTORY_SHA256
        or len(targets) != 14
        or sum(target.full_wiki_rows for target in targets) != 67_867
        or sum(target.overlap_rows for target in targets) != 8_499
    ):
        raise ValueError("WIKI14 exact reviewed inventory pin changed.")
    frames = _load_target_frames(repository, release, targets)
    for dataset in REQUIRED_DATASETS:
        if dataset == DATASET:
            continue
        validate_dataset(
            dataset,
            frames[dataset],
            incomplete_action_policy="warn",
        ).raise_for_errors()
    _verify_bbby_bbt_evidence(repository, frames[DATASET])
    evidence = load_evidence_bundle(wiki_zip_path, targets=targets)
    audits = [
        _audit_target(target, evidence.wiki_rows[target.symbol], frames)
        for target in targets
    ]
    provenance = _provenance_artifact(evidence, audits)
    if provenance.source_hash != PROVENANCE_SHA256:
        raise ValueError("WIKI14 canonical provenance hash changed.")
    artifacts = (*evidence.artifacts, provenance)
    artifact_inventory = [artifact.source_hash for artifact in artifacts]
    observed_inventory_sha = _canonical_sha(artifact_inventory)
    if observed_inventory_sha != ARCHIVE_ARTIFACT_INVENTORY_SHA256:
        raise ValueError("WIKI14 archive artifact inventory hash changed.")
    candidate, changed = _append_or_verify_artifacts(
        repository,
        frames[DATASET],
        artifacts,
    )
    if not changed and WIKI_LICENSE_WARNING not in release.warnings:
        raise RuntimeError("WIKI14 artifacts exist without the Unknown-license warning.")
    validate_dataset(
        DATASET, candidate, completed_session=release.completed_session
    ).raise_for_errors()
    candidate_inventory_sha256 = _source_archive_inventory_sha256(candidate)
    signal_differences = sum(
        sum(audit["triple_supertrend"]["field_differences"].values())
        for audit in audits
    )
    if signal_differences:
        raise ValueError("WIKI14 aggregate Triple Supertrend difference changed.")
    status = "validated_offline_plan" if changed else "already_applied"
    planned_source_archive_version = ""
    planned_release: DataRelease | None = None
    if changed:
        planned_source_archive_version = (
            "wiki14-price-only-"
            f"{release.completed_session.replace('-', '')}-{uuid.uuid4().hex}-{DATASET}"
        )
        versions = dict(release.dataset_versions)
        versions[DATASET] = planned_source_archive_version
        warnings = tuple(dict.fromkeys((*release.warnings, WIKI_LICENSE_WARNING)))
        planned_release = DataRelease.create(
            release.completed_session,
            versions,
            quality=release.quality,
            warnings=warnings,
        )
    return PreparedRepair(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        version_state=version_state,
        frame=candidate,
        artifacts=tuple(artifacts),
        wiki_zip_path=wiki_zip_path,
        targets=tuple(targets),
        allowed_index_identity_gap_fingerprints=(),
        planned_source_archive_version=planned_source_archive_version,
        planned_release=planned_release,
        source_archive_inventory_sha256=candidate_inventory_sha256,
        summary={
            "status": status,
            "base_release_version": release.version,
            "source_archive_base_version": release.dataset_versions[DATASET],
            "passed_price_only_security_ids": [
                audit["security_id"] for audit in audits
            ],
            "price_arbitrations": audits,
            "extract_inventory_sha256": EXTRACT_INVENTORY_SHA256,
            "identity_schema_inventory_sha256": (
                IDENTITY_SCHEMA_INVENTORY_SHA256
            ),
            "archive_effective_date": ARCHIVE_EFFECTIVE_DATE,
            "provenance_sha256": provenance.source_hash,
            "archive_artifact_inventory_sha256": observed_inventory_sha,
            "source_archive_rows_added": len(artifacts) if changed else 0,
            "wiki_full_rows": sum(target.full_wiki_rows for target in targets),
            "reviewed_overlap_rows": sum(target.overlap_rows for target in targets),
            "triple_supertrend_field_differences": signal_differences,
            "source_archive_inventory_sha256": candidate_inventory_sha256,
            "source_archive_only": True,
            "daily_price_raw_rows_changed": 0,
            "corporate_action_rows_changed": 0,
            "adjustment_factor_rows_changed": 0,
            "identity_rows_changed": 0,
            "index_rows_changed": 0,
            "existing_bbby_bbt_evidence_preserved": True,
            "license_name": "Unknown",
            "private_internal_only": True,
            "redistribution_allowed": False,
            "public_publication_allowed": False,
            "local_apply_ack_required": True,
            "generic_symbol_or_ticker_exception_allowed": False,
            "network_accessed": False,
            "eodhd_calls": 0,
            "r2_accessed": False,
            "plan_read_mode": "duckdb_exact_14_security_subset",
            "full_market_price_frames_materialized": False,
            "parent_release_files_hash_validated": True,
            "inherited_index_identity_gap_fingerprints": [],
        },
    )


@contextmanager
def _exclusive_repository_lock(repository: LocalDatasetRepository):
    lock_path = repository.root / ".locks/market-store-write.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Market-store writer lock is already held.") from exc
        recovery = repository.root / RECOVERY_DIR
        if recovery.exists() and tuple(recovery.glob("*.json")):
            raise RuntimeError("Unresolved WIKI14 recovery marker blocks writes.")
        transactions = repository.root / TRANSACTION_DIR
        if transactions.exists():
            for path in transactions.glob("*.json"):
                try:
                    status = _text(json.loads(path.read_bytes()).get("status"))
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    raise RuntimeError(
                        f"Interrupted WIKI14 transaction blocks writes: {path}."
                    )
        yield


def _write_journal(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(
        path,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )


def _assert_inputs_unchanged(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
) -> None:
    release, release_etag = repository.current_release()
    if not (
        release is not None
        and release.to_bytes() == prepared.release.to_bytes()
        and release_etag == prepared.release_etag
    ):
        raise RuntimeError("Current release changed after WIKI14 planning.")
    if not (
        set(prepared.pointer_etags) == set(release.dataset_versions)
        and set(prepared.version_state) == set(release.dataset_versions)
    ):
        raise RuntimeError("Prepared WIKI14 input inventory is incomplete.")
    pointer_etags, version_state = _version_state(repository, release)
    if pointer_etags != dict(prepared.pointer_etags):
        raise RuntimeError("WIKI14 pointer inventory changed after planning.")
    if version_state != dict(prepared.version_state):
        raise RuntimeError("WIKI14 manifest/file inventory changed after planning.")


def _artifact_path(
    repository: LocalDatasetRepository,
    artifact: ArchiveArtifact,
) -> Path:
    return _safe_path(
        repository.root, artifact.object_path(ARCHIVE_EFFECTIVE_DATE)
    )


def _stored_artifact_bytes(artifact: ArchiveArtifact) -> bytes:
    """Return the exact deterministic gzip bytes owned by this operation."""

    return gzip.compress(artifact.payload, mtime=0)


def _write_artifact(
    repository: LocalDatasetRepository,
    artifact: ArchiveArtifact,
) -> bool:
    """Persist exact gzip bytes and report whether this call created the file."""

    path = _artifact_path(repository, artifact)
    if path.exists():
        try:
            existing = gzip.decompress(path.read_bytes())
        except (OSError, EOFError) as exc:
            raise ValueError(f"Existing WIKI14 artifact is invalid gzip: {path}.") from exc
        if existing != artifact.payload:
            raise ValueError(f"Existing WIKI14 artifact bytes conflict: {path}.")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_storage_bytes = _stored_artifact_bytes(artifact)
    write_atomic(path, expected_storage_bytes)
    try:
        observed_storage_bytes = path.read_bytes()
        observed = gzip.decompress(observed_storage_bytes)
    except (OSError, EOFError) as exc:
        raise ValueError(f"Written WIKI14 artifact is invalid gzip: {path}.") from exc
    if (
        observed_storage_bytes != expected_storage_bytes
        or observed != artifact.payload
    ):
        raise ValueError(f"Written WIKI14 artifact verification failed: {path}.")
    return True


def _remove_created_artifacts(
    repository: LocalDatasetRepository,
    artifacts: Sequence[ArchiveArtifact],
    created_source_hashes: set[str],
) -> tuple[str, ...]:
    """Remove only exact files owned by this transaction after safe rollback."""

    selected = [
        artifact
        for artifact in artifacts
        if artifact.source_hash in created_source_hashes
    ]
    paths: list[Path] = []
    try:
        if len(selected) != len(created_source_hashes):
            raise RuntimeError("Created WIKI14 artifact inventory is incomplete.")
        for artifact in selected:
            path = _artifact_path(repository, artifact)
            if not path.is_file():
                raise RuntimeError(f"Created WIKI14 artifact disappeared: {path}.")
            try:
                observed_storage_bytes = path.read_bytes()
                observed = gzip.decompress(observed_storage_bytes)
            except (OSError, EOFError) as exc:
                raise RuntimeError(
                    f"Created WIKI14 artifact is invalid gzip: {path}."
                ) from exc
            if (
                observed_storage_bytes != _stored_artifact_bytes(artifact)
                or observed != artifact.payload
            ):
                raise RuntimeError(f"Created WIKI14 artifact changed: {path}.")
            paths.append(path)
    except Exception as exc:
        return (f"archive preflight: {type(exc).__name__}: {exc}",)
    errors: list[str] = []
    for path in paths:
        try:
            path.unlink()
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _capture_owned_pointer(
    repository: LocalDatasetRepository,
    *,
    version: str,
    manifest_bytes: bytes,
) -> bytes:
    value = repository.objects.get(repository.current_key(DATASET))
    pointer = CurrentPointer.from_bytes(value.data)
    expected_path = f"{repository.version_prefix(DATASET, version)}/manifest.json"
    if not (
        pointer.dataset == DATASET
        and pointer.version == version
        and pointer.manifest_path == expected_path
        and pointer.manifest_sha256 == sha256_bytes(manifest_bytes)
    ):
        raise RuntimeError("Written WIKI14 source_archive pointer is not exact.")
    return value.data


def _restore_transaction(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: bytes,
    planned_release_bytes: bytes,
    owned_pointer_bytes: bytes | None,
) -> tuple[str, ...]:
    """Preflight every mutable byte before restoring an owned publication."""

    try:
        current_release = repository.objects.get("releases/current.json")
        if current_release.data not in {old_release_bytes, planned_release_bytes}:
            observed = DataRelease.from_bytes(current_release.data)
            raise RuntimeError(
                f"unexpected release during rollback: {observed.version}"
            )
        key = repository.current_key(DATASET)
        current_pointer = repository.objects.get(key)
        if current_pointer.data != old_pointer_bytes and (
            owned_pointer_bytes is None
            or current_pointer.data != owned_pointer_bytes
        ):
            observed = CurrentPointer.from_bytes(current_pointer.data)
            raise RuntimeError(
                "unexpected source_archive pointer during rollback: "
                + observed.version
            )
    except Exception as exc:
        return (f"rollback preflight: {type(exc).__name__}: {exc}",)

    errors: list[str] = []
    try:
        if current_release.data != old_release_bytes:
            repository.objects.put(
                "releases/current.json",
                old_release_bytes,
                if_match=current_release.etag,
            )
    except Exception as exc:
        errors.append(f"releases/current.json: {type(exc).__name__}: {exc}")
    try:
        if current_pointer.data != old_pointer_bytes:
            repository.objects.put(
                key, old_pointer_bytes, if_match=current_pointer.etag
            )
    except Exception as exc:
        errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _persist_immutable_release(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> bool:
    key = f"releases/{release.version}.json"
    payload = release.to_bytes()
    try:
        repository.objects.put(key, payload, if_none_match=True)
        return True
    except ConditionalWriteFailed:
        try:
            existing = repository.objects.get(key)
        except ObjectNotFound as exc:  # pragma: no cover - race guard
            raise RuntimeError("Prepared WIKI14 immutable release conflicted.") from exc
        if existing.data != payload:
            raise RuntimeError("Prepared WIKI14 immutable release bytes conflict.")
        return False


def _remove_created_immutable_release(
    repository: LocalDatasetRepository,
    release: DataRelease,
    *,
    created: bool,
) -> tuple[str, ...]:
    if not created:
        return ()
    try:
        path = _safe_path(repository.root, f"releases/{release.version}.json")
        if not path.is_file() or path.read_bytes() != release.to_bytes():
            raise RuntimeError("Created immutable WIKI14 release changed.")
        path.unlink()
        return ()
    except Exception as exc:
        return (f"immutable_release: {type(exc).__name__}: {exc}",)


def _verify_written_source_archive(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    manifest_bytes: bytes,
    owned_pointer_bytes: bytes,
) -> None:
    planned = prepared.planned_release
    if planned is None:
        raise RuntimeError("Prepared WIKI14 release is missing.")
    pointer_value = repository.objects.get(repository.current_key(DATASET))
    if pointer_value.data != owned_pointer_bytes:
        raise RuntimeError("Written WIKI14 source_archive pointer bytes changed.")
    manifest = repository.current_manifest(DATASET)
    if not (
        manifest is not None
        and manifest.version == prepared.planned_source_archive_version
        and manifest.to_bytes() == manifest_bytes
        and sum(item.row_count for item in manifest.files) == len(prepared.frame)
    ):
        raise RuntimeError("Written WIKI14 source_archive manifest changed.")
    frame = repository.read_frame(DATASET, prepared.planned_source_archive_version)
    validate_dataset(
        DATASET, frame, completed_session=planned.completed_session
    ).raise_for_errors()
    if (
        _source_archive_inventory_sha256(frame)
        != prepared.source_archive_inventory_sha256
    ):
        raise RuntimeError("Written WIKI14 source_archive inventory changed.")
    for artifact in prepared.artifacts:
        row = _one(
            frame,
            frame["archive_id"].astype(str).eq(artifact.source_hash),
            f"WIKI14 {artifact.source_hash} archive row",
        )
        _verify_artifact_row(repository, row, artifact)


def _assert_committed_publication(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    owned_pointer_bytes: bytes,
) -> None:
    planned = prepared.planned_release
    if planned is None:
        raise RuntimeError("Prepared WIKI14 release is missing.")
    current, _etag = repository.current_release()
    if current is None or current.to_bytes() != planned.to_bytes():
        raise RuntimeError("Committed WIKI14 release is not current.")
    pointer_etags, version_state = _version_state(repository, planned)
    for dataset in prepared.release.dataset_versions:
        if dataset == DATASET:
            observed = repository.objects.get(repository.current_key(dataset))
            if observed.data != owned_pointer_bytes:
                raise RuntimeError("Committed WIKI14 source_archive pointer changed.")
            continue
        if (
            pointer_etags.get(dataset) != prepared.pointer_etags.get(dataset)
            or version_state.get(dataset) != prepared.version_state.get(dataset)
        ):
            raise RuntimeError(
                f"Out-of-scope WIKI14 dataset changed during apply: {dataset}."
            )
    if WIKI_LICENSE_WARNING not in planned.warnings:
        raise RuntimeError("Committed WIKI14 license warning is missing.")


def apply_repair(
    repository: LocalDatasetRepository,
    prepared: PreparedRepair,
    *,
    ack_private_internal_only_local_repair: bool = False,
    inject_failure: FailureInjector = _noop_injector,
) -> dict[str, Any]:
    if prepared.summary["status"] == "already_applied":
        _assert_inputs_unchanged(repository, prepared)
        return {**prepared.summary, "mode": "apply", "writes_performed": False}
    if not ack_private_internal_only_local_repair:
        raise PermissionError(
            "Unknown-license WIKI14 evidence requires "
            "ack_private_internal_only_local_repair=True."
        )
    with _exclusive_repository_lock(repository):
        _assert_inputs_unchanged(repository, prepared)
        planned_version = prepared.planned_source_archive_version
        planned_release = prepared.planned_release
        expected_versions = dict(prepared.release.dataset_versions)
        expected_versions[DATASET] = planned_version
        if not (
            planned_version
            and planned_release is not None
            and planned_release.dataset_versions == expected_versions
            and WIKI_LICENSE_WARNING in planned_release.warnings
            and prepared.source_archive_inventory_sha256
            == _source_archive_inventory_sha256(prepared.frame)
            and prepared.summary.get("plan_read_mode")
            == "duckdb_exact_14_security_subset"
            and prepared.summary.get("full_market_price_frames_materialized")
            is False
        ):
            raise RuntimeError("Prepared WIKI14 transaction contract is incomplete.")
        old_release = repository.objects.get("releases/current.json")
        old_pointer = repository.objects.get(repository.current_key(DATASET))
        transaction_id = uuid.uuid4().hex
        journal_path = repository.root / TRANSACTION_DIR / f"{transaction_id}.json"
        artifact_preexisting = {
            artifact.source_hash: _artifact_path(repository, artifact).exists()
            for artifact in prepared.artifacts
        }
        journal: dict[str, Any] = {
            "schema": "us_wiki14_price_only_transaction/v1",
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release.data).decode("ascii"),
            "old_pointer_base64": base64.b64encode(old_pointer.data).decode("ascii"),
            "planned_source_archive_version": planned_version,
            "planned_release_version": planned_release.version,
            "planned_release_sha256": sha256_bytes(planned_release.to_bytes()),
            "artifact_preexisting": artifact_preexisting,
            "created_artifact_source_hashes": [],
            "local_private_internal_only_ack": True,
            "redistribution_allowed": False,
            "public_publication_allowed": False,
            "created_at": utc_now_iso(),
        }
        _write_journal(journal_path, journal)
        created_artifact_source_hashes: set[str] = set()
        owned_pointer_bytes: bytes | None = None
        immutable_release_created = False
        try:
            inject_failure("after_journal")
            for artifact in prepared.artifacts:
                existed_before = _artifact_path(repository, artifact).exists()
                try:
                    created = _write_artifact(repository, artifact)
                except BaseException:
                    if not existed_before and _artifact_path(repository, artifact).exists():
                        created_artifact_source_hashes.add(artifact.source_hash)
                    raise
                if created:
                    created_artifact_source_hashes.add(artifact.source_hash)
                journal["created_artifact_source_hashes"] = sorted(
                    created_artifact_source_hashes
                )
                _write_journal(journal_path, journal)
            inject_failure("after_artifacts")
            parent_metadata = dict(
                repository.manifest_for_version(
                    DATASET, prepared.release.dataset_versions[DATASET]
                ).metadata
            )
            result = repository.write_frame(
                DATASET,
                prepared.frame,
                completed_session=prepared.release.completed_session,
                metadata={
                    **parent_metadata,
                    "operation": OPERATION,
                    "passed_price_only_security_ids": [
                        target.security_id for target in prepared.targets
                    ],
                    "extract_inventory_sha256": EXTRACT_INVENTORY_SHA256,
                    "identity_schema_inventory_sha256": (
                        IDENTITY_SCHEMA_INVENTORY_SHA256
                    ),
                    "archive_effective_date": ARCHIVE_EFFECTIVE_DATE,
                    "provenance_sha256": PROVENANCE_SHA256,
                    "archive_artifact_inventory_sha256": (
                        ARCHIVE_ARTIFACT_INVENTORY_SHA256
                    ),
                    "source_archive_rows_added": len(prepared.artifacts),
                    "license_name": "Unknown",
                    "private_internal_only": True,
                    "redistribution_allowed": False,
                    "public_publication_allowed": False,
                    "network_accessed": False,
                    "eodhd_calls": 0,
                    "r2_accessed": False,
                },
                expected_pointer_etag=prepared.pointer_etags[DATASET],
                version=planned_version,
            )
            if result.conflict:
                raise RuntimeError(
                    f"source_archive write conflicted: {result.conflict_path}."
                )
            manifest_bytes = result.manifest.to_bytes()
            owned_pointer_bytes = _capture_owned_pointer(
                repository,
                version=planned_version,
                manifest_bytes=manifest_bytes,
            )
            inject_failure("after_source_archive_write")
            _verify_written_source_archive(
                repository,
                prepared,
                manifest_bytes=manifest_bytes,
                owned_pointer_bytes=owned_pointer_bytes,
            )
            immutable_release_created = _persist_immutable_release(
                repository, planned_release
            )
            repository.objects.put(
                "releases/current.json",
                planned_release.to_bytes(),
                if_match=prepared.release_etag,
            )
            inject_failure("after_release_commit")
            _assert_committed_publication(
                repository,
                prepared,
                owned_pointer_bytes=owned_pointer_bytes,
            )
            journal.update(
                {
                    "status": "committed",
                    "committed_release_version": planned_release.version,
                    "completed_at": utc_now_iso(),
                }
            )
            _write_journal(journal_path, journal)
            return {
                **prepared.summary,
                "status": "applied",
                "mode": "apply",
                "writes_performed": True,
                "new_release_version": planned_release.version,
                "new_source_archive_version": result.manifest.version,
                "transaction_id": transaction_id,
                "warnings": list(planned_release.warnings),
            }
        except BaseException as original:
            rollback_errors = _restore_transaction(
                repository,
                old_release_bytes=old_release.data,
                old_pointer_bytes=old_pointer.data,
                planned_release_bytes=planned_release.to_bytes(),
                owned_pointer_bytes=owned_pointer_bytes,
            )
            if not rollback_errors:
                rollback_errors = (
                    *rollback_errors,
                    *_remove_created_artifacts(
                        repository,
                        prepared.artifacts,
                        created_artifact_source_hashes,
                    ),
                )
            if not rollback_errors:
                rollback_errors = (
                    *rollback_errors,
                    *_remove_created_immutable_release(
                        repository,
                        planned_release,
                        created=immutable_release_created,
                    ),
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
                recovery = repository.root / RECOVERY_DIR / f"{transaction_id}.json"
                _write_journal(recovery, journal)
                raise RuntimeError(
                    f"WIKI14 rollback incomplete: {recovery}; "
                    f"errors={rollback_errors}."
                ) from original
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive exact frozen-WIKI price-only evidence for 14 identities."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--wiki-zip", type=Path, default=DEFAULT_WIKI_ZIP)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--ack-private-internal-only-local-repair", action="store_true"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repository = LocalDatasetRepository(args.cache_root)
    prepared = prepare_repair(repository, wiki_zip_path=args.wiki_zip)
    result = (
        apply_repair(
            repository,
            prepared,
            ack_private_internal_only_local_repair=(
                args.ack_private_internal_only_local_repair
            ),
        )
        if args.apply
        else {**prepared.summary, "mode": "plan", "writes_performed": False}
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
