#!/usr/bin/env python3
"""Finalize US lifecycle coverage from a frozen, fully collected JSON report.

The command is deliberately offline.  Its default mode builds and validates a
complete plan without writing anything.  ``--apply`` is the only mode that may
advance local dataset pointers and the release pointer.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import html
import io
import json
import math
import re
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlparse

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from supertrend_quant.market_store.adjustments import build_adjustment_factors
from supertrend_quant.market_store.ingest import SourceArtifact
from supertrend_quant.market_store.lifecycle import (
    LifecycleCandidate,
    build_lifecycle_candidates,
    canonical_lifecycle_event_id,
    resolve_new_security_id,
)
from supertrend_quant.market_store.lifecycle_coverage import (
    LifecycleCoverageReport,
    LifecycleExceptionCode,
    lifecycle_candidate_id,
    validate_lifecycle_coverage,
)
from supertrend_quant.market_store.lifecycle_report_provenance import (
    DEFAULT_SEC_MAX_HTTP_ATTEMPTS,
    DEFAULT_SEC_MAX_HTTP_ATTEMPTS_PER_CANDIDATE,
    SEC_FETCH_POLICY_CACHE_ONLY,
    SEC_MAX_HTTP_ATTEMPTS_PER_REQUEST,
    build_lifecycle_report_binding,
    validate_lifecycle_report_binding,
)
from supertrend_quant.market_store.manifest import (
    CurrentPointer,
    DataRelease,
    sha256_bytes,
    utc_now_iso,
    write_atomic,
)
from supertrend_quant.market_store.models import DataQuality
from supertrend_quant.market_store.official_lifecycle_evidence import (
    OfficialLifecycleExceptionEvidenceSpec,
    include_bound_official_applied_event_candidates,
    load_official_lifecycle_exception_evidence,
    matching_official_exception_specs,
)
from supertrend_quant.market_store.operational_validation import (
    TRUSTED_OPERATIONAL_NBL_TERMINAL_STATE,
    reviewed_operational_index_identity_gap_fingerprints,
    validate_operational_repository_snapshot,
)
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.schemas import dataset_spec
from supertrend_quant.market_store.storage import ObjectNotFound
from supertrend_quant.market_store.yahoo_chart import parse_yahoo_chart_json
from supertrend_quant.market_store.validation import (
    validate_dataset,
)
try:
    from unified_quant.scripts.collect_us_lifecycle_actions import (
        _CurrentReleaseSecArchiveReplay,
    )
except ModuleNotFoundError as exc:
    if exc.name != "unified_quant":  # pragma: no cover - dependency failure
        raise
    from collect_us_lifecycle_actions import _CurrentReleaseSecArchiveReplay


DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_REPORT = Path("results/data_quality/us_lifecycle/sec_collection_v7.json")
DEFAULT_SEC_CACHE = Path("data/cache/state/sec_lifecycle")
DEFAULT_HINTS = Path("unified_quant/configs/us_lifecycle_hints.yaml")
FINALIZER_VERSION = 1
REVIEWED_BY = "us_lifecycle_finalizer_v1"
REVIEWED_AT = "2026-07-18T00:00:00Z"
DEFAULT_RECHECK_AFTER = "2026-10-31"
WRITE_DATASETS = (
    "corporate_actions",
    "adjustment_factors",
    "source_archive",
    "lifecycle_resolutions",
)
PERMANENT_EXCEPTION_CODES = frozenset(
    {
        str(LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION),
        str(LifecycleExceptionCode.RECOVERY_UNCERTAIN),
    }
)

# Exact, independently reviewed repairs may replace two former permanent
# exceptions.  These pins are deliberately duplicated here: the lifecycle
# finalizer must not trust a newly rehashed action merely because another
# script wrote it.  Preservation is allowed only after every action,
# resolution, identity boundary, archive payload, metadata object, and warning
# below matches this code-reviewed contract.
CELG_EXACT_SECURITY_ID = "US:EODHD:0337dd23-67ad-5354-b972-50babd1ae5a0"
CELG_EXACT_LAST_SESSION = "2019-11-20"
CELG_EXACT_EFFECTIVE_DATE = "2019-11-21"
CELG_EXACT_TERMS_URL = (
    "https://www.sec.gov/Archives/edgar/data/14272/"
    "000114036119021048/0001140361-19-021048.txt"
)
CELG_EXACT_TERMS_SHA256 = (
    "157cae6dae5486f16c63a51e61d79aab2ce2f37d0e8584337fb21d7d0ec6f211"
)
CELG_EXACT_TERMS_BYTES = 1_257_384
CELG_EXACT_TERMINATION_URL = (
    "https://www.sec.gov/Archives/edgar/data/14272/"
    "000001427221000066/bmy-20201231.htm"
)
CELG_EXACT_TERMINATION_SHA256 = (
    "a86e198381d31eacf1fd4b17e93e7a09c8b7f191c1941bec43467729b4f8b055"
)
CELG_EXACT_TERMINATION_BYTES = 5_654_027
CELG_EXACT_BMY_SECURITY_ID = "US:EODHD:25d16784-a5a9-5eee-bf6e-81519b64ef0b"
CELG_EXACT_CVR_LAST_SESSION = "2020-12-31"
CELG_EXACT_CVR_TERMINATION_DATE = "2021-01-01"
CELG_EXACT_CVR_SESSIONS = 280
CELG_EXACT_CVR_FIRST_CLOSE = 2.30
CELG_OFFICIAL_EXIT_PROVIDER_CODE = "CELG-RI"
CELG_OFFICIAL_EXIT_PROVIDER_SYMBOL = "CELG-RI.US"
CELG_OFFICIAL_EXIT_WARNING = (
    "CELG/BMYRT official_exit_mark uses only the hash-pinned BMY 2020 10-K "
    "first-trade close of USD 2.30 as a retrospective first-session close exit "
    "mark for a non-index child, with cash available next session; the "
    "2019-11-21..2020-12-31 280-session trading path is unsupported."
)
CELG_OFFICIAL_EXIT_SECURITY_ID = (
    "US:EODHD:"
    + str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            "eodhd:US:CELG-RI:symbol:BMYRT",
        )
    )
)
CELG_OFFICIAL_EXIT_EVENT_ID = canonical_lifecycle_event_id(
    CELG_OFFICIAL_EXIT_SECURITY_ID, "delisting", CELG_EXACT_EFFECTIVE_DATE
)
CELG_OFFICIAL_EXIT_POLICY_URL = (
    "supertrendquant://policy/celg-bmyrt-official-exit-mark/v1"
)
CELG_OFFICIAL_EXIT_ACTIVE_CATALOG_URL = (
    "https://eodhd.com/api/exchange-symbol-list/US?delisted=0"
)
CELG_OFFICIAL_EXIT_ACTIVE_CATALOG_SHA256 = (
    "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99"
)
CELG_OFFICIAL_EXIT_DELISTED_CATALOG_URL = (
    "https://eodhd.com/api/exchange-symbol-list/US?delisted=1"
)
CELG_OFFICIAL_EXIT_DELISTED_CATALOG_SHA256 = (
    "8a64e65e316b71e5d165265db2796b68a31f821812f74b63367435b8fcb2ed13"
)
CELG_OFFICIAL_RESIDUAL_EVENT_ID = canonical_lifecycle_event_id(
    CELG_OFFICIAL_EXIT_SECURITY_ID,
    "delisting",
    CELG_EXACT_CVR_TERMINATION_DATE,
)
CELG_EXACT_CVR_BASIS_METADATA_SHA256 = (
    "37d0cdf9809d5e96c0065c4a1552855df976b16076fa226c3082d81b8e2ac989"
)
CELG_EXACT_CVR_TERMINATION_METADATA_SHA256 = (
    "7d4f889d89a3aa52d1efe505511182e0d5835e1467851d7a273a75d2ca29c7ad"
)
CELG_EXACT_DISTRIBUTION_EVENT_ID = canonical_lifecycle_event_id(
    CELG_EXACT_SECURITY_ID, "spinoff", CELG_EXACT_EFFECTIVE_DATE
)
CELG_EXACT_MERGER_EVENT_ID = canonical_lifecycle_event_id(
    CELG_EXACT_SECURITY_ID, "stock_merger", CELG_EXACT_EFFECTIVE_DATE
)

ABMD_EXACT_SECURITY_ID = "US:EODHD:faece1b7-4b1a-5c1f-951b-b1178ed57161"
ABMD_EXACT_LAST_SESSION = "2022-12-21"
ABMD_EXACT_LAST_CLOSE = 381.02
ABMD_EXACT_EFFECTIVE_DATE = "2022-12-22"
ABMD_EXACT_EVENT_ID = canonical_lifecycle_event_id(
    ABMD_EXACT_SECURITY_ID, "cash_merger", ABMD_EXACT_EFFECTIVE_DATE
)
ABMD_EXACT_TERMS_URL = (
    "https://www.sec.gov/Archives/edgar/data/815094/"
    "000119312522311074/0001193125-22-311074.txt"
)
ABMD_EXACT_TERMS_SHA256 = (
    "f98bc807432739e4f2447ffbc6a70f7651bd8982b901989fc31dcffaa56ec593"
)
ABMD_EXACT_TERMS_BYTES = 331_457
ABMD_EXACT_VALUATION_URL = (
    "https://d18rn0p25nwr6d.cloudfront.net/CIK-0000200406/"
    "e09b8882-48b1-4fea-a818-66acddf84c4f.pdf"
)
ABMD_EXACT_VALUATION_SHA256 = (
    "65710a85a1f27aa581c1cddce22cab62bec0a3b5848283e163bbdcc1aa67b5e8"
)
ABMD_EXACT_VALUATION_BYTES = 2_184_778
ABMD_EXACT_METADATA_SHA256 = (
    "d96a0c9ce94d676ea228ac2f0dc6fae86b638a5992666a3d3670b4cd7edb4f4b"
)
ABMD_EXACT_WARNING = (
    "ABMD 2022-12-22 merger models only the guaranteed $380/share cash leg; "
    "one non-tradeable CVR per share (up to $35 contingent cash) is marked at "
    "$0 under the as-of-2026-07-15 lower-bound policy, so returns may be "
    "understated; J&J still reported about $0.4bn of aggregate ABMD CVR "
    "liability at 2025 year-end, so $0 is not a fair-value estimate."
)

# AVP completed its combination with Natura on 2020-01-03 after the close.
# The SEC filing says each AVP share converted into 0.300 NTCO ADS, that AVP
# would be suspended before the 2020-01-06 open, and that NTCO was expected to
# begin trading on 2020-01-06.  The generic report crosscheck looks only on the
# legal completion date and therefore cannot see the first priceable successor
# session.  Preserve this repair only after independently attesting the exact
# action, identity boundary, SEC bytes, AVP raw price path, and the original
# NTCO NYSE price path.  These pins are deliberately duplicated from the writer
# so a changed repair cannot authorize itself.
AVP_EXACT_SECURITY_ID = "US:EODHD:529d8af8-043b-542e-8eeb-e8651009a2a8"
AVP_EXACT_SUCCESSOR_ID = "US:EODHD:7e3cea59-409f-5cf0-8429-9d4245013622"
AVP_EXACT_LAST_SESSION = "2020-01-03"
AVP_EXACT_LEGAL_COMPLETION = "2020-01-03"
AVP_EXACT_MARKET_TRANSITION = "2020-01-06"
AVP_EXACT_RATIO = 0.3
AVP_EXACT_OLD_EVENT_ID = (
    "7fd31cc07d0f1fff0c4b17a7b821acfd20e584a770c3119ff18ad074803e67d3"
)
AVP_EXACT_EVENT_ID = (
    "825ea0640b20da42dcfa1c516ff921f272b0fd0a0fd4020de509674832391806"
)
AVP_EXACT_ACTION_METADATA_SHA256 = (
    "cf04f0bd80a624c5c6988b9025589f90939427201f4fe966a28ff5d2a7c3411e"
)
AVP_EXACT_SEC_URL = (
    "https://www.sec.gov/Archives/edgar/data/8868/"
    "000095015720000022/0000950157-20-000022.txt"
)
AVP_EXACT_SEC_SHA256 = (
    "12ca5855e19d9c0c0542f393964ef1e9ee0b1f831c26296f389f143d4bad42a4"
)
AVP_EXACT_SEC_BYTES = 365_795
AVP_EXACT_SEC_RETRIEVED_AT = "2026-07-18T10:30:42.442453Z"
AVP_EXACT_RAW_URL = (
    "https://eodhd.com/api/eod/AVP.US?from=2015-01-01&to=2026-07-15"
)
AVP_EXACT_RAW_SHA256 = (
    "b7a04462fb63c48d389d5a03300e4ba0b6bc0307e4f03c06babd75e10f7ebbe4"
)
AVP_EXACT_RAW_BYTES = 139_550
AVP_EXACT_RAW_ROWS = 1_260
AVP_EXACT_RAW_RETRIEVED_AT = "2026-07-16T15:57:00.046989Z"
AVP_EXACT_TERMINAL_OHLCV = (5.64, 5.92, 5.05, 5.6, 236_653_175.0)
AVP_EXACT_NTCO_RAW_URL = (
    "https://eodhd.com/api/eod/NTCO.US?from=2015-01-01&to=2026-07-15"
)
AVP_EXACT_NTCO_ENVELOPE_SHA256 = (
    "e88684de37208bd947df3140593aff81082126aefbc353d545f3ef0ae9fd8883"
)
AVP_EXACT_NTCO_ENVELOPE_BYTES = 161_099
AVP_EXACT_NTCO_RAW_SHA256 = (
    "91cb9baec50c86d49447d78f2882256a991884e46fda1a6019f5df792cb02dde"
)
AVP_EXACT_NTCO_RAW_BYTES = 120_644
AVP_EXACT_NTCO_RAW_ROWS = 1_075
AVP_EXACT_NTCO_NYSE_ROWS = 1_032
AVP_EXACT_NTCO_NYSE_END = "2024-02-09"
AVP_EXACT_NTCO_RAW_RETRIEVED_AT = "2026-07-17T20:37:19.646249Z"
AVP_EXACT_NTCO_FIRST_OHLCV = (20.6, 20.73, 19.06, 19.46, 9_007_021.0)
AVP_EXACT_REVIEWED_BY = "sivb_avp_terminal_transition_planner_v1"
AVP_EXACT_REVIEWED_AT = "2026-07-18T15:30:00Z"
AVP_EXACT_REPAIR_SOURCE = "official_market_transition_repair"
AVP_EXACT_TEMPORARY_REPORT_SHA256 = (
    "77510685cfa3db900b4f48772447663f7d5a28d0c664cc2db2eb7984ef255f45"
)
AVP_EXACT_TEMPORARY_REPORT_URL = "file:///tmp/us-lifecycle-post-ntco.json"
AVP_EXACT_TEMPORARY_REPORT_BYTES = 742_419
AVP_EXACT_TEMPORARY_REPORT_RETRIEVED_AT = "2026-07-18T19:57:50.471614Z"

# SIVB/SIVBQ is one legal security across the NASDAQ-to-OTC ticker transition.
# The exact repaired path separates the 2024-11-07 legal cancellation and last
# OTC close from the engine's next-XNYS-session market exit on 2024-11-08.  The
# raw OCC PDF is independently pinned here; the legacy reviewed extraction is
# retained as non-authoritative audit history.  These constants are duplicated
# from the writer so a changed repair cannot self-authorize through metadata.
SIVB_EXACT_SECURITY_ID = "US:EODHD:181d58e2-2b82-5f5d-8c33-311c5d7b5129"
SIVB_EXACT_OLD_LAST = "2023-03-27"
SIVB_EXACT_OTC_START = "2023-03-28"
SIVB_EXACT_LAST_SESSION = "2024-11-07"
SIVB_EXACT_ENGINE_EXIT = "2024-11-08"
SIVB_EXACT_TICKER_EVENT_ID = (
    "01419d978e03e608512e4e898e695fdb39953278b08dc8138d97e0d0e21e4caa"
)
SIVB_EXACT_LEGAL_EVENT_ID = (
    "1f4a23cffdf2decb8c26be93d94318d6d5a2be7fc045c33ff9e5abd4e9c69c82"
)
SIVB_EXACT_MARKET_EXIT_EVENT_ID = (
    "f8cbecc851776e882ac85872972ac5c1680a672138276c1a9301d532c2d8ad3f"
)
SIVB_EXACT_OCC_URL = "https://infomemo.theocc.com/infomemos?number=52179"
SIVB_EXACT_OCC_PDF_SHA256 = (
    "28f25f761b61e7f898fa7d1c237520a1a4fed97e88af0b788ef52256608c6035"
)
SIVB_EXACT_OCC_PDF_BYTES = 566_940
SIVB_EXACT_OCC_LEGACY_SHA256 = (
    "11e986df7ea010021dd343353e24da8009673053b555b32cce95d4aae1598c6f"
)
SIVB_EXACT_SEC_MARKET_URL = (
    "https://www.sec.gov/Archives/edgar/data/719739/"
    "000119312523073665/d485308d8k.htm"
)
SIVB_EXACT_SEC_MARKET_SHA256 = (
    "69f3b20dfab4c9c43641a3c38a99f288129665af40e5ae3e6993ec36ccf4fcef"
)
SIVB_EXACT_SEC_CANCEL_URL = (
    "https://www.sec.gov/Archives/edgar/data/719739/"
    "000119312524254186/d904756d8k.htm"
)
SIVB_EXACT_SEC_CANCEL_SHA256 = (
    "14371aef1566bfdcda9ca3171b1ced46d095adf34d899a8bde6d8e038d68e231"
)
SIVB_EXACT_EOD_URL = (
    "https://eodhd.com/api/eod/SIVBQ.US?from=2023-03-28&to=2024-11-08"
)
SIVB_EXACT_EOD_SHA256 = (
    "038c5a1ab7a5b439835a12507ebacc8bd8342ba73005479a0c57acc60ff04a1f"
)
SIVB_EXACT_EOD_ROWS = 409
SIVB_EXACT_STORED_OTC_ROWS = 408
SIVB_EXACT_NON_XNYS_EXCLUSIONS = frozenset({"2024-09-02"})
SIVB_EXACT_TICKER_METADATA_SHA256 = (
    "086360be7d6b0642b95121e8b78f3b23beff00ed5b1ee6adf2a6da3840607b81"
)
SIVB_EXACT_EXIT_METADATA_SHA256 = (
    "28c562187b5e25bfca2b767dbc12d17519c98ebb86bdd861afe27a0916881c08"
)
# The current audited release applied the SIVB and FRC raw-OCC bindings after
# its last full adjustment-factor rebuild.  Finalization may bridge precisely
# this provenance-only gap, then must rebuild factors against its planned
# action version.  Any other stale action lineage remains blocked.
EXACT_PROVENANCE_BRIDGE_FACTOR_ACTION_VERSION = (
    "eca-qvcaq-transition-20260715-e7d95a0a245744b59a357169156ad32c-"
    "corporate_actions"
)
EXACT_PROVENANCE_BRIDGE_CURRENT_ACTION_VERSION = (
    "frc-occ-52352-20260715-9c669b9d1b944d9196b12323080f7f32-"
    "corporate_actions"
)

# The early-terminal-history repair performs a full factor-table rebuild and
# binds that table to its exact planned price/action versions with a compact,
# deterministic digest.  Refinalization must understand this writer-owned
# lineage format; treating it as a stale partial bridge would make a valid
# current release impossible to refinalize after the repair.
EARLY_TERMINAL_HISTORY_FACTOR_OPERATION = (
    "repair_us_early_terminal_history_supplements"
)
EARLY_TERMINAL_HISTORY_FACTOR_PREFIX = "early-terminal-history:"

# FRC/FRCB and PARA/PSKY are a second independently reviewed repair pair.  FRC
# is deliberately *not* preserved as a lifecycle resolution after FRCB prices
# extend the same security through the completed session: its former terminal
# candidate is stale and the finalizer must remove that row.  The same-security
# ticker change, immutable raw payload, one reviewed OHLC correction, official
# provenance, and degraded-quality warning are nevertheless exact invariants.
# PARA remains a real terminal candidate, so its exact 1:1 default-stock PSKY
# resolution is preserved.  These pins are duplicated rather than imported
# from the writer so the finalizer remains an independent fail-closed gate.
FRC_EXACT_SECURITY_ID = "US:EODHD:3453b450-03d1-52ee-9100-cf80856b06ef"
PARA_EXACT_SECURITY_ID = "US:EODHD:f60b749b-3d84-552a-9dc9-39e742f67537"
PSKY_EXACT_SECURITY_ID = "US:EODHD:fe84848c-624b-5aba-b542-24af3959f97f"

FRC_EXACT_OLD_LAST = "2023-05-02"
FRC_EXACT_TRANSITION = "2023-05-03"
FRC_EXACT_PRICE_END = "2026-07-15"
FRC_EXACT_EVENT_ID = canonical_lifecycle_event_id(
    FRC_EXACT_SECURITY_ID, "ticker_change", FRC_EXACT_TRANSITION
)
FRC_EXACT_OCC_URL = "https://infomemo.theocc.com/infomemos?number=52352"
# Legacy deterministic extraction: retained for identity/history audit only.
FRC_EXACT_OCC_SHA256 = (
    "377bcc0663eb9666b9f639edaf541b0ec729fd4b84ac876e345baaa9bf413668"
)
FRC_EXACT_OCC_LEGACY_ARCHIVE_ID = (
    "c568a6ac21ddc05d3c5821c228b94b7bd7e52a602a96b1cfb2f5f08ee24af658"
)
FRC_EXACT_OCC_PDF_SHA256 = (
    "0ab8f0c49076f97049dcde80078affda258108ce7c904241bf1eba46b44aec66"
)
FRC_EXACT_OCC_PDF_BYTES = 566_923
FRC_EXACT_OCC_PDF_OBJECT_PATH = (
    "archives/2026-07-15/"
    "0ab8f0c49076f97049dcde80078affda258108ce7c904241bf1eba46b44aec66.pdf.gz"
)
FRC_EXACT_OCC_PDF_RETRIEVED_AT = "2026-07-18T18:35:42Z"
FRC_EXACT_OCC_ACTION_METADATA_SHA256 = (
    "6cd2c29ee9b870a4fbffadaf984aac9a211d579106e72b5bbf43548fdc2cfbb2"
)
FRC_EXACT_FDIC_URL = (
    "https://www.fdic.gov/resources/resolutions/bank-failures/"
    "failed-bank-list/first-republic.html"
)
FRC_EXACT_FDIC_SHA256 = (
    "30c6bad80710f702144fa3ef61ca1ae14e81503fb0f436abdad7f91ebe8e51eb"
)
FRC_EXACT_RAW_EOD_URL = (
    "https://eodhd.com/api/eod/FRCB.US?"
    f"from={FRC_EXACT_TRANSITION}&to={FRC_EXACT_PRICE_END}"
)
FRC_EXACT_RAW_EOD_SHA256 = (
    "3c96be232fb5f567e77a94fe315b67aa61e520d1611842620c04680fb5df6ab3"
)
FRC_EXACT_RAW_EOD_ROWS = 802
FRC_EXACT_CORRECTION_SHA256 = (
    "53eed10c5d6a7ccc262215b7848d30efa606a1621d2e793ca21b6002f8a5c298"
)
FRC_EXACT_CORRECTION_POLICY_SHA256 = (
    "15f4444f68f513ad20443a72325c506536b837d33ed99f6cf54ff6023bb00626"
)
FRC_EXACT_WARNING = (
    "FRCB EODHD raw OHLCV required an exact-hash reviewed envelope correction "
    "(2024-12-30 low=0.0->0.003; "
    "raw_eod_sha256=3c96be232fb5f567e77a94fe315b67aa61e520d1611842620c04680fb5df6ab3); "
    "all other fields remain unchanged, so release quality remains degraded."
)
FRC_EXACT_OCC_EXTRACTION = {
    "schema": "occ_reviewed_memo_extraction/v1",
    "memo_number": "52352",
    "source_url": FRC_EXACT_OCC_URL,
    "subject": "First Republic Bank - Symbol Change",
    "effective_date": FRC_EXACT_TRANSITION,
    "old_symbol": "FRC",
    "new_symbol": "FRCB",
    "market": "OTC",
    "cusip": "33616C100",
    "contract_multiplier": 1,
    "deliverable_per_contract": "100 First Republic Bank (FRCB) Common Shares",
    "reviewed_claim": (
        "FRC and FRCB are the same First Republic common-share identity; "
        "only the market and ticker changed on 2023-05-03."
    ),
}
FRC_EXACT_CORRECTION_METADATA = {
    "schema": "frcb_reviewed_ohlcv_envelope_correction/v1",
    "provider_symbol": "FRCB.US",
    "raw_eod_sha256": FRC_EXACT_RAW_EOD_SHA256,
    "correction_policy_sha256": FRC_EXACT_CORRECTION_POLICY_SHA256,
    "corrections": [
        {
            "raw_eod_sha256": FRC_EXACT_RAW_EOD_SHA256,
            "session": "2024-12-30",
            "field": "low",
            "observed": 0.0,
            "corrected": 0.003,
            "justification": (
                "Exact-hash reviewed zero-low provider defect; replace only low "
                "with the minimum positive observed OHLC boundary."
            ),
        }
    ],
    "unchanged_fields": ["open", "high", "close", "volume"],
    "review_scope": "exact raw hash, exact session, exact observed row",
}

# NTCO and NTCOY are one Natura ADS security across the NYSE-to-OTC ticker
# transition.  The independently reviewed price-only repair replaces the
# stale NTCO tail with the exact NTCOY raw response, preserves two already
# stored NTCO dividends whose provider aliases disagree slightly, and closes
# the candidate at the BNY-proven 2024-08-07 last trade before the mandatory
# cash exchange.  These pins are deliberately local to the finalizer: a
# changed writer or a merely edited lifecycle report cannot self-authorize the
# preservation path.
NTCO_EXACT_SECURITY_ID = "US:EODHD:7e3cea59-409f-5cf0-8429-9d4245013622"
NTCO_EXACT_OLD_SYMBOL = "NTCO"
NTCO_EXACT_NEW_SYMBOL = "NTCOY"
NTCO_EXACT_ACTIVE_FROM = "2020-01-06"
NTCO_EXACT_TICKER_DATE = "2024-02-12"
NTCO_EXACT_OLD_SYMBOL_END = "2024-02-09"
NTCO_EXACT_LAST_SESSION = "2024-08-07"
NTCO_EXACT_TERMINAL_DATE = "2024-09-04"
NTCO_EXACT_TERMINAL_CASH = 5.043659
NTCO_EXACT_TICKER_EVENT_ID = (
    "0bd231d216016d33c8d6eb1ec9bb85450a21b9298c4abda34a3399727e9b2c00"
)
NTCO_EXACT_TERMINAL_EVENT_ID = (
    "d1eeadf8f4779212ff4b3162af1731bf7bfc804c10756cc4bb8074c307e1f746"
)
NTCO_EXACT_RETRIEVED_AT = "2026-07-18T18:47:16.808110Z"
NTCO_EXACT_REVIEWED_BY = "us_ntco_ntcoy_transition_repair_v1"
NTCO_EXACT_REVIEWED_AT = "2026-07-19T00:00:00Z"

# The NTCO writer intentionally rebuilt only the one affected security's
# factors, preserving every other factor row byte-for-byte until this
# lifecycle finalizer performs the next full rebuild.  Its manifest therefore
# retains the last whole-table ECA lineage while the NTCO partition carries
# the exact new input-pair lineage.  Only this immutable version triplet may
# use that reviewed mixed-partition bridge.
NTCO_EXACT_MIXED_LINEAGE_ROOT = (
    "ntco-ntcoy-20260715-661edf2da0df4a86a70f5c37e81ecd5b"
)
NTCO_EXACT_MIXED_PRICE_VERSION = (
    f"{NTCO_EXACT_MIXED_LINEAGE_ROOT}-daily_price_raw"
)
NTCO_EXACT_MIXED_ACTION_VERSION = (
    f"{NTCO_EXACT_MIXED_LINEAGE_ROOT}-corporate_actions"
)
NTCO_EXACT_MIXED_FACTOR_VERSION = (
    f"{NTCO_EXACT_MIXED_LINEAGE_ROOT}-adjustment_factors"
)

NTCO_EXACT_OCC_URL = "https://infomemo.theocc.com/infomemos?number=54105"
NTCO_EXACT_OCC_RAW_SHA256 = (
    "5eb3c0058195d6d6c22361e362f1623c4eae7dd7262e9497ef7b53f8a564e913"
)
NTCO_EXACT_OCC_RAW_RETRIEVED_AT = "2026-07-18T17:41:32.027461Z"
NTCO_EXACT_CBOE_URL = (
    "https://cdn.cboe.com/resources/product_restriction/2024/"
    "Cboe-Options-Exchanges-Restrictions-on-Transactions-in-Options-on-"
    "Natura-Co-Holding-S-A.pdf"
)
NTCO_EXACT_CBOE_RAW_SHA256 = (
    "e67d8046a4c532daa11dac1faacb9e573238812c90d15be9792b5d6a46e67928"
)
NTCO_EXACT_CBOE_RAW_RETRIEVED_AT = "2026-07-18T17:41:15.684878Z"
NTCO_EXACT_BNY_CASH_URL = (
    "https://www.adrbny.com/content/dam/adr/documents/"
    "corporate-actions-dr/files/ad1145447.pdf"
)
NTCO_EXACT_BNY_CASH_RAW_SHA256 = (
    "057a95798bd3b9ff4f0f0be6fd34d7ed42d6ad1d27a12ad1c41e30700c8da28b"
)
NTCO_EXACT_BNY_CASH_RAW_RETRIEVED_AT = "2026-07-18T17:42:06.789266Z"
NTCO_EXACT_BNY_TERMINATION_URL = (
    "https://www.adrbny.com/content/dam/adr/documents/"
    "corporate-actions-dr/files/ad1140774.pdf"
)
NTCO_EXACT_BNY_TERMINATION_RAW_SHA256 = (
    "abc23f86378f82f5efb475fc23e0a82e398db4201dc7e52fd4064d088eb47b83"
)
NTCO_EXACT_BNY_TERMINATION_RETRIEVED_AT = "2026-07-18T18:42:57.215876Z"
NTCO_EXACT_BNY_BOOKS_CLOSED_URL = (
    "https://www.adrbny.com/content/dam/adr/documents/"
    "books-closed/files/bc1141635.pdf"
)
NTCO_EXACT_BNY_BOOKS_CLOSED_RAW_SHA256 = (
    "b5d449fcad19cd38fb461c8f9bae4c1856c0fb7ccd4903e062f87840315ba675"
)
NTCO_EXACT_BNY_BOOKS_CLOSED_RETRIEVED_AT = "2026-07-18T18:47:16.808110Z"

NTCO_EXACT_IDENTITY_SHA256 = (
    "8c9312d2079c238a4fa47b701d24b8e707c040080cb8a5ce0d62f6bd82fd54cb"
)
NTCO_EXACT_TERMINAL_SHA256 = (
    "3be830009ee6942d4ca604aafdc19730a693f13e62b046641a638e9a1ee112c1"
)
NTCO_EXACT_DECISION_SHA256 = (
    "fa349557d3f433371a2b08015cd7d672801106a2e442b3be4e51478e69587557"
)
NTCO_EXACT_EOD_URL = (
    "https://eodhd.com/api/eod/NTCOY.US?"
    "from=2024-02-12&to=2024-09-03"
)
NTCO_EXACT_EOD_RAW_SHA256 = (
    "3ef3a1f03ec97252ac4db079298cdb90ddc32bdeb41fd64a71aaf6d667153e54"
)
NTCO_EXACT_EOD_RETRIEVED_AT = "2026-07-18T18:28:42.473931Z"
NTCO_EXACT_EOD_ROWS = 123
NTCO_EXACT_DIV_URL = (
    "https://eodhd.com/api/div/NTCOY.US?"
    "from=2024-02-12&to=2024-09-03"
)
NTCO_EXACT_DIV_RAW_SHA256 = (
    "6adc67e2b64dd8dcf0acfc0a3bf20bb0d275844f2305b66c1ff4d2a3789d8175"
)
NTCO_EXACT_SPLITS_URL = (
    "https://eodhd.com/api/splits/NTCOY.US?"
    "from=2024-02-12&to=2024-09-03"
)
NTCO_EXACT_SPLITS_RAW_SHA256 = (
    "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
)
NTCO_EXACT_SPLITS_ARCHIVE_ID = (
    "6a09ccaafcdf8ad57177fd1be2146ce912c84c4269cdc11ce736c7b4faad4461"
)
NTCO_EXACT_DECISION_URL = (
    NTCO_EXACT_DIV_URL
)
NTCO_EXACT_PRESERVED_DIVIDENDS = {
    "658cb5351b78504a2c20ca3ae75d4d5a2660ea884fc1e2650b1c9a0370551cc0": (
        "2024-03-21",
        0.28427,
    ),
    "ebbf2e8b20dfeb94521486d8ed81342ae1fb631c01796857e53795fcafbd163c": (
        "2024-04-09",
        0.01099,
    ),
}
NTCO_EXACT_PRESERVED_DIVIDEND_URL = (
    "https://eodhd.com/api/div/NTCO.US?from=2015-01-01&to=2026-07-15"
)
NTCO_EXACT_PRESERVED_DIVIDEND_SHA256 = (
    "b2a5b7c6a26165cf4f92618e4a76c06b0cd7de55673fd5cc7162073374469fa0"
)
NTCO_EXACT_PRESERVED_DIVIDEND_RETRIEVED_AT = (
    "2026-07-17T20:37:19.646249Z"
)

PARA_EXACT_LAST = "2025-08-06"
PARA_EXACT_TRANSITION = "2025-08-07"
PARA_EXACT_EVENT_ID = canonical_lifecycle_event_id(
    PARA_EXACT_SECURITY_ID, "stock_merger", PARA_EXACT_TRANSITION
)
PARA_EXACT_SEC_URL = (
    "https://www.sec.gov/Archives/edgar/data/813828/"
    "000119312525175027/0001193125-25-175027.txt"
)
PARA_EXACT_SEC_SHA256 = (
    "61ea922a72a55f05b79c2cf00e9c4b0367434c35bf25a78fd4d815d3b20e68be"
)
PARA_EXACT_RETRIEVED_AT = "2026-07-18T08:13:18.114738Z"


@dataclass(frozen=True)
class ExceptionSpec:
    code: str
    reason: str
    reviewed_by: str = REVIEWED_BY
    reviewed_at: str = REVIEWED_AT
    recheck_after: str = ""
    source_url: str = ""
    source_hash: str = ""
    require_official_provenance: bool = False
    evidence_id: str = ""


@dataclass(frozen=True)
class ReportDocument:
    path: Path
    content: bytes
    value: dict[str, Any]

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.content)


@dataclass(frozen=True)
class ArchivedArtifact:
    artifact: SourceArtifact
    object_path: str


@dataclass
class PreparedFinalization:
    release: DataRelease
    release_etag: str | None
    pointer_etags: dict[str, str | None]
    planned_versions: dict[str, str]
    input_versions: dict[str, str]
    frames: dict[str, pd.DataFrame]
    artifacts: tuple[ArchivedArtifact, ...]
    coverage_report: LifecycleCoverageReport
    evidence_report_sha256: str
    lifecycle_metadata: dict[str, Any]
    warnings: tuple[str, ...]
    summary: dict[str, Any]


def _key(security_id: str, last_price_date: str) -> str:
    return f"{str(security_id).strip()}|{pd.Timestamp(last_price_date).date().isoformat()}"


def _symbol_key(symbol: str, last_price_date: str) -> str:
    return f"symbol:{str(symbol).strip().upper()}|{pd.Timestamp(last_price_date).date().isoformat()}"


# KRFT->KHC and YHOO->AABA were independently repaired from legal dates to
# the first successor trading sessions.  The frozen collection report still
# contains the earlier legal-date interpretations, so finalization must reuse
# these exact reviewed market-date actions and resolutions rather than create
# duplicate actions that precede the successor identities.
EXACT_REVIEWED_MARKET_DATE_TRANSITIONS: dict[str, dict[str, Any]] = {
    _key(
        "US:EODHD:e3b28684-f48e-582a-8836-b2f5579f2dd2", "2015-07-02"
    ): {
        "candidate": {
            "security_id": "US:EODHD:e3b28684-f48e-582a-8836-b2f5579f2dd2",
            "symbol": "KRFT",
            "name": "Kraft Foods Group Inc",
            "exchange": "NASDAQ",
            "last_price_date": "2015-07-02",
            "active_to": "2015-07-02",
        },
        "event_id": (
            "39caca00fbedb7b23e8b5294cdf3bdc5aba9014f98bf72b6a6da341602651192"
        ),
        "rejected_legal_date_event_id": (
            "c3209167d547d7e8379cb316ec4910ea62d1c1d28679f137a80364fe876e9b7c"
        ),
        "action": {
            "security_id": "US:EODHD:e3b28684-f48e-582a-8836-b2f5579f2dd2",
            "action_type": "stock_merger",
            "effective_date": "2015-07-06",
            "ex_date": "2015-07-06",
            "announcement_date": "2015-07-02",
            "record_date": "",
            "payment_date": "",
            "cash_amount": None,
            "ratio": 1.0,
            "currency": "USD",
            "new_security_id": "US:EODHD:a060087d-109e-58e9-91f9-b4c398a3d415",
            "new_symbol": "KHC",
            "source_url": (
                "https://www.sec.gov/Archives/edgar/data/1637459/"
                "000119312515244356/0001193125-15-244356.txt"
            ),
            "source_kind": "official_crosscheck",
            "source": "sec_edgar+stored_price_crosscheck",
            "retrieved_at": "2026-07-18T10:30:21.019780Z",
            "source_hash": (
                "ba6d15da235a1f5f6a186fb7b9e52b7b438c25cf76d4f28d9fac81953b7fc299"
            ),
        },
        "metadata": {
            "policy": "us_market_transition_dates/v1",
            "legal_completion_date": "2015-07-02",
            "source_last_trade_date": "2015-07-02",
            "market_effective_date": "2015-07-06",
            "market_date_policy": "first_successor_trading_session",
            "evidence": [
                {
                    "source_url": (
                        "https://www.sec.gov/Archives/edgar/data/1637459/"
                        "000119312515244356/0001193125-15-244356.txt"
                    ),
                    "source_hash": (
                        "ba6d15da235a1f5f6a186fb7b9e52b7b438c25cf76d4f28d9fac81953b7fc299"
                    ),
                }
            ],
        },
        "resolution": {
            "candidate_id": (
                "8afeca7c8f24b790e6b2f234254597ae076e1d6e3d35dcd0f41299e56642e9f6"
            ),
            "security_id": "US:EODHD:e3b28684-f48e-582a-8836-b2f5579f2dd2",
            "symbol": "KRFT",
            "last_price_date": "2015-07-02",
            "resolution": "applied",
            "event_id": (
                "39caca00fbedb7b23e8b5294cdf3bdc5aba9014f98bf72b6a6da341602651192"
            ),
            "exception_code": "",
            "exception_reason": "",
            "reviewed_by": "us_lifecycle_finalizer_v1",
            "reviewed_at": "2026-07-18T00:00:00Z",
            "recheck_after": "",
            "successor_security_id": (
                "US:EODHD:a060087d-109e-58e9-91f9-b4c398a3d415"
            ),
            "successor_symbol": "KHC",
            "source_url": (
                "https://www.sec.gov/Archives/edgar/data/1637459/"
                "000119312515244356/0001193125-15-244356.txt"
            ),
            "source": "lifecycle_finalizer",
            "retrieved_at": "2026-07-18T10:30:21.019780Z",
            "source_hash": (
                "ba6d15da235a1f5f6a186fb7b9e52b7b438c25cf76d4f28d9fac81953b7fc299"
            ),
        },
        "master": (
            {
                "security_id": "US:EODHD:e3b28684-f48e-582a-8836-b2f5579f2dd2",
                "primary_symbol": "KRFT",
                "provider_symbol": "KRFT.US",
                "action_provider_symbol": "KRFT.US",
                "name": "Kraft Foods Group Inc",
                "exchange": "NASDAQ",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "isin": "",
                "active_from": "2015-01-02",
                "active_to": "2015-07-02",
                "source": "official_confirmed_identity_history_repair",
                "source_url": (
                    "https://www.sec.gov/Archives/edgar/data/1637459/"
                    "000119312515244356/0001193125-15-244356.txt"
                ),
                "retrieved_at": "2026-07-18T08:11:54.578547Z",
                "source_hash": (
                    "ba6d15da235a1f5f6a186fb7b9e52b7b438c25cf76d4f28d9fac81953b7fc299"
                ),
            },
            {
                "security_id": "US:EODHD:a060087d-109e-58e9-91f9-b4c398a3d415",
                "primary_symbol": "KHC",
                "provider_symbol": "KHC.US",
                "action_provider_symbol": "KHC.US",
                "name": "Kraft Heinz Co",
                "exchange": "NASDAQ",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "isin": "",
                "active_from": "2015-07-06",
                "active_to": "",
                "source": "official_confirmed_identity_history_repair",
                "source_url": (
                    "https://www.sec.gov/Archives/edgar/data/1637459/"
                    "000119312515244356/0001193125-15-244356.txt"
                ),
                "retrieved_at": "2026-07-18T08:11:54.578547Z",
                "source_hash": (
                    "ba6d15da235a1f5f6a186fb7b9e52b7b438c25cf76d4f28d9fac81953b7fc299"
                ),
            },
        ),
        "history": (
            {
                "security_id": "US:EODHD:e3b28684-f48e-582a-8836-b2f5579f2dd2",
                "symbol": "KRFT",
                "exchange": "NASDAQ",
                "effective_from": "2015-01-01",
                "effective_to": "2015-07-02",
                "source": "official_confirmed_identity_history_repair",
                "source_url": (
                    "https://www.sec.gov/Archives/edgar/data/1637459/"
                    "000119312515244356/0001193125-15-244356.txt"
                ),
                "retrieved_at": "2026-07-18T08:11:54.578547Z",
                "source_hash": (
                    "ba6d15da235a1f5f6a186fb7b9e52b7b438c25cf76d4f28d9fac81953b7fc299"
                ),
            },
            {
                "security_id": "US:EODHD:a060087d-109e-58e9-91f9-b4c398a3d415",
                "symbol": "KHC",
                "exchange": "NASDAQ",
                "effective_from": "2015-07-06",
                "effective_to": "",
                "source": "official_confirmed_identity_history_repair",
                "source_url": (
                    "https://www.sec.gov/Archives/edgar/data/1637459/"
                    "000119312515244356/0001193125-15-244356.txt"
                ),
                "retrieved_at": "2026-07-18T08:11:54.578547Z",
                "source_hash": (
                    "ba6d15da235a1f5f6a186fb7b9e52b7b438c25cf76d4f28d9fac81953b7fc299"
                ),
            },
        ),
        "archive_bytes": 1_440_758,
    },
    _key(
        "US:EODHD:1dfb343a-9d83-5f7b-b683-bdf884111979", "2017-06-16"
    ): {
        "candidate": {
            "security_id": "US:EODHD:1dfb343a-9d83-5f7b-b683-bdf884111979",
            "symbol": "YHOO",
            "name": "Yahoo! Inc",
            "exchange": "NASDAQ",
            "last_price_date": "2017-06-16",
            "active_to": "2017-06-16",
        },
        "event_id": (
            "ea791beeaa569aaf19b1df04e59df3710d45382cd70e5d82078d7247286272c6"
        ),
        "rejected_legal_date_event_id": (
            "b31525699f142a6ed8995b71d86b8480e19e54446ccb0694960296f753ad2be6"
        ),
        "action": {
            "security_id": "US:EODHD:1dfb343a-9d83-5f7b-b683-bdf884111979",
            "action_type": "ticker_change",
            "effective_date": "2017-06-19",
            "ex_date": "2017-06-19",
            "announcement_date": "2017-06-19",
            "record_date": "",
            "payment_date": "",
            "cash_amount": None,
            "ratio": None,
            "currency": "USD",
            "new_security_id": "US:EODHD:9b1bbdaa-839c-5d59-8bda-99b2087022e6",
            "new_symbol": "AABA",
            "source_url": (
                "https://www.sec.gov/Archives/edgar/data/1011006/"
                "000119312517206955/0001193125-17-206955.txt"
            ),
            "source_kind": "official_crosscheck",
            "source": "sec_edgar+stored_price_crosscheck",
            "retrieved_at": "2026-07-18T10:30:27.515561Z",
            "source_hash": (
                "274d98161c66f9480a4825eae722974405fffe2e3a0b73091d22f1a9acb07ca9"
            ),
        },
        "metadata": {
            "policy": "us_market_transition_dates/v1",
            "operating_business_sale_completion_date": "2017-06-13",
            "legal_name_change_date": "2017-06-16",
            "source_last_trade_date": "2017-06-16",
            "market_effective_date": "2017-06-19",
            "market_date_policy": "first_successor_trading_session",
            "holder_action_required": False,
            "evidence": [
                {
                    "source_url": (
                        "https://www.sec.gov/Archives/edgar/data/1011006/"
                        "000119312517206955/0001193125-17-206955.txt"
                    ),
                    "source_hash": (
                        "274d98161c66f9480a4825eae722974405fffe2e3a0b73091d22f1a9acb07ca9"
                    ),
                }
            ],
        },
        "resolution": {
            "candidate_id": (
                "85ae0eb05557268955665853205bf4e684f1cc7317d4d6921e739aacebcfdfd2"
            ),
            "security_id": "US:EODHD:1dfb343a-9d83-5f7b-b683-bdf884111979",
            "symbol": "YHOO",
            "last_price_date": "2017-06-16",
            "resolution": "applied",
            "event_id": (
                "ea791beeaa569aaf19b1df04e59df3710d45382cd70e5d82078d7247286272c6"
            ),
            "exception_code": "",
            "exception_reason": "",
            "reviewed_by": "us_lifecycle_finalizer_v1",
            "reviewed_at": "2026-07-18T00:00:00Z",
            "recheck_after": "",
            "successor_security_id": (
                "US:EODHD:9b1bbdaa-839c-5d59-8bda-99b2087022e6"
            ),
            "successor_symbol": "AABA",
            "source_url": (
                "https://www.sec.gov/Archives/edgar/data/1011006/"
                "000119312517206955/0001193125-17-206955.txt"
            ),
            "source": "lifecycle_finalizer",
            "retrieved_at": "2026-07-18T10:30:27.515561Z",
            "source_hash": (
                "274d98161c66f9480a4825eae722974405fffe2e3a0b73091d22f1a9acb07ca9"
            ),
        },
        "master": (
            {
                "security_id": "US:EODHD:1dfb343a-9d83-5f7b-b683-bdf884111979",
                "primary_symbol": "YHOO",
                "provider_symbol": "YHOO.US",
                "action_provider_symbol": "YHOO.US",
                "name": "Yahoo! Inc",
                "exchange": "NASDAQ",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "isin": "",
                "active_from": "2015-01-02",
                "active_to": "2017-06-16",
                "source": "eodhd_exchange_symbols",
                "source_url": (
                    "https://eodhd.com/api/exchange-symbol-list/US?delisted=0"
                ),
                "retrieved_at": "2026-07-16T15:56:01.033938Z",
                "source_hash": (
                    "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99"
                ),
            },
            {
                "security_id": "US:EODHD:9b1bbdaa-839c-5d59-8bda-99b2087022e6",
                "primary_symbol": "AABA",
                "provider_symbol": "AABA.US",
                "action_provider_symbol": "AABA.US",
                "name": "Altaba Inc",
                "exchange": "NASDAQ",
                "asset_type": "STOCK",
                "currency": "USD",
                "country": "US",
                "isin": "",
                "active_from": "2017-06-19",
                "active_to": "2019-10-02",
                "source": "official_market_transition_repair",
                "source_url": (
                    "https://www.sec.gov/Archives/edgar/data/1011006/"
                    "000119312517206955/0001193125-17-206955.txt"
                ),
                "retrieved_at": "2026-07-18T10:30:27.515561Z",
                "source_hash": (
                    "274d98161c66f9480a4825eae722974405fffe2e3a0b73091d22f1a9acb07ca9"
                ),
            },
        ),
        "history": (
            {
                "security_id": "US:EODHD:1dfb343a-9d83-5f7b-b683-bdf884111979",
                "symbol": "YHOO",
                "exchange": "NASDAQ",
                "effective_from": "2015-01-01",
                "effective_to": "2017-06-16",
                "source": "official_market_transition_repair",
                "source_url": (
                    "https://www.sec.gov/Archives/edgar/data/1011006/"
                    "000119312517206955/0001193125-17-206955.txt"
                ),
                "retrieved_at": "2026-07-18T10:30:27.515561Z",
                "source_hash": (
                    "274d98161c66f9480a4825eae722974405fffe2e3a0b73091d22f1a9acb07ca9"
                ),
            },
            {
                "security_id": "US:EODHD:9b1bbdaa-839c-5d59-8bda-99b2087022e6",
                "symbol": "AABA",
                "exchange": "NASDAQ",
                "effective_from": "2017-06-19",
                "effective_to": "2019-10-02",
                "source": "official_market_transition_repair",
                "source_url": (
                    "https://www.sec.gov/Archives/edgar/data/1011006/"
                    "000119312517206955/0001193125-17-206955.txt"
                ),
                "retrieved_at": "2026-07-18T10:30:27.515561Z",
                "source_hash": (
                    "274d98161c66f9480a4825eae722974405fffe2e3a0b73091d22f1a9acb07ca9"
                ),
            },
        ),
        "archive_bytes": 17_192,
    },
}


# Three terminal-boundary repairs deliberately moved a lifecycle event from
# the issuer/legal date (which is also the last old-symbol price date) to the
# first successor trading session.  The fresh SEC collector still describes
# the legal date, so refinalization must preserve these exact reviewed market
# dates instead of reintroducing duplicate actions one session too early.
EXACT_SHORT_TERMINAL_MARKET_TRANSITIONS: dict[str, dict[str, Any]] = {
    _key(
        "US:EODHD:7623d5d2-1c3d-595f-8e96-408208fc7d37", "2018-12-31"
    ): {
        "candidate": {
            "symbol": "KORS",
            "name": "Michael Kors Holdings Limited",
            "exchange": "NYSE MKT",
            "active_to": "2018-12-31",
            "index_remove_dates": ("2018-09-19",),
        },
        "event_id": "951960822646a8f972002508d162bd37a3ab3aaec42349fbff37caa1919b7b51",
        "rejected_event_id": "51ffb3cf286e69bdcc2d66a6945a33cc8cc3deb41661920b5ab4a7fa7b327f36",
        "action": {
            "action_type": "ticker_change",
            "effective_date": "2019-01-02",
            "ex_date": "2019-01-02",
            "announcement_date": "2018-12-31",
            "record_date": "",
            "payment_date": "",
            "cash_amount": None,
            "ratio": None,
            "currency": "USD",
            "new_security_id": "US:EODHD:27f39ea8-f202-53a2-83bf-41a211b5f3d9",
            "new_symbol": "CPRI",
            "source_url": "https://www.sec.gov/Archives/edgar/data/1530721/000119312518362322/0001193125-18-362322.txt",
            "source_kind": "official_crosscheck",
            "source": "sec_edgar+stored_price_crosscheck",
            "retrieved_at": "2026-07-18T20:58:24.767711Z",
            "source_hash": "ff4732e714524028c56a66c96e6ac8c50a401a36a4a46037cb80b01bb8454d25",
        },
    },
    _key(
        "US:EODHD:99eac7c1-6892-5b3a-bf4b-bc8143e3bfe2", "2022-10-11"
    ): {
        "candidate": {
            "symbol": "NLSN",
            "name": "Nielsen Holdings PLC",
            "exchange": "NYSE",
            "active_to": "2022-10-11",
            "index_remove_dates": ("2022-10-12",),
        },
        "event_id": "2aa7c18ca6ac8f0e4680a7e5456a04ba2f401fabfb2cc7dc0a5326e298f71176",
        "rejected_event_id": "0079876b484ced964d0395f5427978616ac571d778a8eff54a42c994cc459177",
        "action": {
            "action_type": "cash_merger",
            "effective_date": "2022-10-12",
            "ex_date": "2022-10-12",
            "announcement_date": "2022-10-11",
            "record_date": "",
            "payment_date": "",
            "cash_amount": 28.0,
            "ratio": None,
            "currency": "USD",
            "new_security_id": "",
            "new_symbol": "",
            "source_url": "https://www.sec.gov/Archives/edgar/data/1492633/000119312522260583/0001193125-22-260583.txt",
            "source_kind": "official_crosscheck",
            "source": "sec_edgar+stored_price_crosscheck",
            "retrieved_at": "2026-07-18T20:59:19.348906Z",
            "source_hash": "893b9f658c505f40fa304c7b18c89fd7ade6d29a6dcd166590181fae4d8e11fd",
        },
    },
    _key(
        "US:EODHD:e9eea478-61d8-5762-9f5b-fbdfd69a02a3", "2022-11-07"
    ): {
        "candidate": {
            "symbol": "NLOK",
            "name": "NortonLifeLock Inc",
            "exchange": "NASDAQ",
            "active_to": "2022-11-07",
            "index_remove_dates": ("2019-12-23", "2022-11-08"),
        },
        "event_id": "d82975bc819ca47d10c7b2e2ca963422629980682933a4ee13b355fe564e6344",
        "rejected_event_id": "002f1c86383d22157a2bfc5602decaf3880b85300a2c410ba1c84608eb43b967",
        "action": {
            "action_type": "ticker_change",
            "effective_date": "2022-11-08",
            "ex_date": "2022-11-08",
            "announcement_date": "2022-11-07",
            "record_date": "",
            "payment_date": "",
            "cash_amount": None,
            "ratio": None,
            "currency": "USD",
            "new_security_id": "US:EODHD:cb0b8e57-3e09-542c-adf8-fe2c98d97b55",
            "new_symbol": "GEN",
            "source_url": "https://www.sec.gov/Archives/edgar/data/849399/000110465922115277/0001104659-22-115277.txt",
            "source_kind": "official_crosscheck",
            "source": "sec_edgar+stored_price_crosscheck",
            "retrieved_at": "2026-07-18T20:59:19.501210Z",
            "source_hash": "a4732aaa030033aebda1d508bed1742e237694dc97fdb1a71f9af02f20d95d83",
        },
    },
}


def _temporary(
    code: str,
    reason: str,
    *,
    source_url: str = "",
    source_hash: str = "",
) -> ExceptionSpec:
    return ExceptionSpec(
        code=code,
        reason=reason,
        recheck_after=DEFAULT_RECHECK_AFTER,
        source_url=source_url,
        source_hash=source_hash,
        require_official_provenance=bool(source_url or source_hash),
    )


# This is an audited allow-list, not a fallback policy.  A candidate absent
# from this exact mapping blocks finalization.  Dates bind entries to the
# frozen terminal observation so ticker reuse cannot silently inherit a prior
# decision.
EXPLICIT_EXCEPTION_MAPPING: dict[str, ExceptionSpec] = {
    # Permanently unsupported economic outcomes.
    _key("US:EODHD:0337dd23-67ad-5354-b972-50babd1ae5a0", "2019-11-20"): ExceptionSpec(
        LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION,
        "CELG consideration included a separately tradable contingent value right.",
        require_official_provenance=True,
        evidence_id="celg_2019_cvr_consideration",
    ),
    _key("US:EODHD:9533a56f-c357-577e-b72f-b85ffbba62af", "2016-05-17"): ExceptionSpec(
        LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION,
        "TWC shareholder elections and proration produced multiple consideration paths.",
        require_official_provenance=True,
        evidence_id="twc_2016_election_proration",
    ),
    _key("US:EODHD:9b1bbdaa-839c-5d59-8bda-99b2087022e6", "2019-10-02"): ExceptionSpec(
        LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION,
        "AABA multi-stage liquidation distributions are not one terminal cash event.",
        require_official_provenance=True,
        evidence_id="aaba_2019_liquidation_distributions",
    ),
    _key("US:EODHD:b2681128-10c6-54e9-afac-d2aeb03fed8f", "2016-01-29"): ExceptionSpec(
        LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION,
        "BRCM cash-stock elections and proration are not one fixed consideration row.",
        require_official_provenance=True,
        evidence_id="brcm_2016_election_proration",
    ),
    _key("US:EODHD:b9fbee49-31b7-527c-9295-1006f06cb15f", "2018-08-27"): ExceptionSpec(
        LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION,
        "GGP cash-stock elections and proration are not one fixed consideration row.",
        require_official_provenance=True,
        evidence_id="ggp_2018_election_proration",
    ),
    _key("US:EODHD:2826c370-0467-5e82-9617-dcece5be407f", "2020-04-24"): ExceptionSpec(
        LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION,
        "Legacy DO shares were cancelled under the 2021 reorganization and holders received pro-rata Emergence Warrants with nonzero, path-dependent value that cannot be represented as a zero-cash delisting.",
        require_official_provenance=True,
        evidence_id="legacy_do_2021_warrant_consideration",
    ),
    _key("US:EODHD:6d9d4638-4922-5f6c-89fd-6b79db60c1c3", "2020-07-28"): ExceptionSpec(
        LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION,
        "Legacy DNR shares were cancelled under the 2020 reorganization and holders received pro-rata Series B Warrants with nonzero, path-dependent value that cannot be represented as a zero-cash delisting.",
        require_official_provenance=True,
        evidence_id="legacy_dnr_2020_warrant_consideration",
    ),
    _key("US:EODHD:81b3ca1f-cf1b-5234-bc24-4399b8ecf149", "2020-10-22"): ExceptionSpec(
        LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION,
        "Legacy NE shares were cancelled or exchanged under the 2021 reorganization and holders received pro-rata Tranche 3 Warrants with nonzero, path-dependent value that cannot be represented as a zero-cash delisting.",
        require_official_provenance=True,
        evidence_id="legacy_ne_2021_warrant_consideration",
    ),
    _key("US:EODHD:f60b749b-3d84-552a-9dc9-39e742f67537", "2025-08-06"): ExceptionSpec(
        LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION,
        "PARA shareholder election and proration alternatives are mutually exclusive.",
        require_official_provenance=True,
        evidence_id="para_2025_election_proration",
    ),
    _key("US:EODHD:faece1b7-4b1a-5c1f-951b-b1178ed57161", "2022-12-21"): ExceptionSpec(
        LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION,
        "ABMD consideration included cash plus an unrepresentable contingent value right.",
        require_official_provenance=True,
        evidence_id="abmd_2022_cvr_consideration",
    ),
    _key("US:EODHD:aefd1dd7-529d-5b6a-80e9-65d0e14102a6", "2020-04-02"): ExceptionSpec(
        LifecycleExceptionCode.UNSUPPORTED_CONSIDERATION,
        "UTX-to-RTX raw ticker-change price equality is invalid because the current "
        "canonical action inventory omits the CARR 1.0-share and OTIS 0.5-share "
        "distributions; keep fail-closed until exact spin-off actions are modeled.",
        require_official_provenance=True,
        evidence_id="utx_2020_carr_otis_distributions",
    ),
    # Temporary evidence, successor, or identity gaps.  Expiry deliberately
    # fails closed and requires this checked-in catalog to be reviewed again.
    _key("US:EODHD:2c15a3cb-4bdb-5b6f-b82e-8e17e286ee69", "2023-08-30"): _temporary(
        LifecycleExceptionCode.INSUFFICIENT_OFFICIAL_EVIDENCE,
        "COR terminal identity requires a refreshed official ticker-event record.",
    ),
    _key("US:EODHD:3073ffd2-9115-5bf6-8bec-fddcd41749e5", "2017-03-20"): _temporary(
        LifecycleExceptionCode.PRICE_IDENTITY_CONFLICT,
        "HOT prices continue beyond the legal merger and require identity repair.",
    ),
    _key("US:EODHD:6c98b8f3-f222-5def-92e5-a0633c3f0775", "2017-01-24"): _temporary(
        LifecycleExceptionCode.SUCCESSOR_UNRESOLVED,
        "LMCA is the predecessor ticker for same-lineage FWONA; collect and crosscheck the FWONA continuation before applying the 2017-01-24 ticker change.",
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/1560385/"
            "000156038517000008/0001560385-17-000008.txt"
        ),
        source_hash="28e903dbfe48fbd8278786008392ee02f7982c4dddcd8c314a39248b641e13f9",
    ),
    _key("US:EODHD:8e7e0713-31d7-55a7-8878-74ba653d9090", "2017-01-24"): _temporary(
        LifecycleExceptionCode.SUCCESSOR_UNRESOLVED,
        "LMCK is the predecessor ticker for same-lineage FWONK; collect and crosscheck the FWONK continuation before applying the 2017-01-24 ticker change.",
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/1560385/"
            "000156038517000008/0001560385-17-000008.txt"
        ),
        source_hash="28e903dbfe48fbd8278786008392ee02f7982c4dddcd8c314a39248b641e13f9",
    ),
    _key("US:EODHD:f174ac51-b4a4-5f29-84b0-12b8b73f0bc3", "2019-06-28"): _temporary(
        LifecycleExceptionCode.PRICE_IDENTITY_CONFLICT,
        "DWDP requires exact DOW and CTVA distribution events plus same-lineage DD continuity; the stored 2019-06-03/04/28 rows and successor active_from values are identity-inconsistent.",
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/1666700/"
            "000119312519163322/0001193125-19-163322.txt"
        ),
        source_hash="ae9343609e64dcd8421f11462b8782cc8db38a130e03c983714f3c10ba8db311",
    ),
    _key("US:EODHD:faece1b7-4b1a-5c1f-951b-b1178ed57161", "2022-12-23"): _temporary(
        LifecycleExceptionCode.PRICE_IDENTITY_CONFLICT,
        "ABMD contains two zero-volume carry-forward bars after its last real 2022-12-21 session; repair the exact terminal identity before applying the CVR exception.",
    ),
    _key("US:EODHD:3e7acffc-054a-5173-bcb3-af5b1bf93c93", "2024-10-15"): _temporary(
        LifecycleExceptionCode.PRICE_IDENTITY_CONFLICT,
        "GPS provider history requires the official GAP transition and price supplement.",
    ),
    _key("US:EODHD:4e0bf98f-3697-5897-b5e5-724a76456870", "2026-01-30"): _temporary(
        LifecycleExceptionCode.INSUFFICIENT_OFFICIAL_EVIDENCE,
        "AZN terminal observation requires refreshed listing evidence.",
    ),
    _key("US:EODHD:5b0fba54-201e-58fc-9a0c-6985ef098924", "2025-02-21"): _temporary(
        LifecycleExceptionCode.INSUFFICIENT_OFFICIAL_EVIDENCE,
        "QRTEA terminal observation requires verified official disposition evidence.",
    ),
    _key("US:EODHD:81d711c5-9688-5f2b-9f36-63c8fe3211bf", "2020-10-12"): _temporary(
        LifecycleExceptionCode.PRICE_IDENTITY_CONFLICT,
        "The provider MNK_old identity is an unrelated Muniholdings fund, not Mallinckrodt; legacy and reorganized Mallinckrodt cancellations require separate repaired security identities with no successor link.",
    ),
    _key("US:EODHD:9a968d54-1ad6-5daf-9edd-ae838a9569b3", "2023-05-16"): _temporary(
        LifecycleExceptionCode.SUCCESSOR_UNRESOLVED,
        "PKI to RVTY remains open until official evidence and successor prices crosscheck.",
    ),
    _key("US:EODHD:b0395c88-1e0d-5135-b79f-240ac991e540", "2019-07-30"): _temporary(
        LifecycleExceptionCode.SUCCESSOR_UNRESOLVED,
        "SEC evidence proves ESV changed to legacy VAL on 2019-07-31, but the same-security VAL price and symbol history must be present before continuity can be applied; the legacy VAL cancellation and warrant recovery on 2021-04-30 is a separate economic event.",
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/314808/"
            "000031480819000130/0000314808-19-000130.txt"
        ),
        source_hash="596701a3f09e484f60489e5df3501c0f09e4c908905bab2f81cefb684e338fac",
    ),
    _key("US:EODHD:b4ec0c21-2f69-5abf-973e-1f725f81f0ce", "2026-05-20"): _temporary(
        LifecycleExceptionCode.SUCCESSOR_UNRESOLVED,
        "BK to BNY requires official evidence and successor prices at the transition.",
    ),
    _key("US:EODHD:dbd287da-b35f-5aaa-873c-76f941b4d93b", "2025-08-29"): _temporary(
        LifecycleExceptionCode.PRICE_IDENTITY_CONFLICT,
        "The legacy BBBY_old history is contaminated by the later reused BBBY ticker; the repaired legacy identity must terminate at the 2023 cancellation before plan evidence can be applied.",
    ),
    _key("US:EODHD:dc3f4283-a3cc-5bc7-916c-9ffdd71c9874", "2021-02-16"): _temporary(
        LifecycleExceptionCode.CROSSCHECK_FAILED,
        "WYND ticker evidence does not yet crosscheck against the stored price identity.",
    ),
    _key("US:EODHD:fc00ff9c-3a71-5995-968e-bc351f950cb4", "2021-02-01"): _temporary(
        LifecycleExceptionCode.SUCCESSOR_UNRESOLVED,
        "CTL to LUMN requires official evidence and successor prices at the transition.",
    ),
    # Expected to become applied after the successor supplement is present.
    **{
        candidate_key: _temporary(
            LifecycleExceptionCode.SUCCESSOR_UNRESOLVED,
            reason,
        )
        for candidate_key, reason in {
            _key("US:EODHD:234e6b6a-3fdb-53df-8c0e-e6de98a8563a", "2022-03-14"): "HFC to DINO successor chain is not fully crosschecked.",
            _key("US:EODHD:330d4f9a-dd92-5e71-9843-9c3cf5dc1058", "2019-01-07"): "SHPG to TAK successor chain is not fully crosschecked.",
            _key("US:EODHD:46e4e57b-eae7-55bb-94b5-bbc4d595fe49", "2017-07-24"): "RAI to BTI successor chain is not fully crosschecked.",
            _key("US:EODHD:527e931f-3364-53a2-963a-2755a59461cb", "2022-04-01"): "ADS to BFH successor chain is not fully crosschecked.",
            _key("US:EODHD:5761939d-b25c-58b8-9f70-674b2b505362", "2019-02-13"): "NFX to ECA successor chain is not fully crosschecked.",
            _key("US:EODHD:724457bc-0eaf-5959-8c93-f0c2a03c80de", "2023-03-06"): "FBHS to FBIN successor chain is not fully crosschecked.",
            _key("US:EODHD:865c1483-a99b-5066-b55a-649e24804d68", "2025-12-02"): "HBI to GIL successor chain is not fully crosschecked.",
            _key("US:EODHD:9b1e757c-e463-5c2c-92d4-9d3b2137115b", "2021-12-14"): "KSU to CP successor chain is not fully crosschecked.",
            _key("US:EODHD:9f13974d-7f81-5aac-a3a7-ed1d184bd76b", "2015-03-16"): "AGN successor identity is not fully crosschecked.",
            _key("US:EODHD:d0212dec-b333-5d7f-90ce-cd3d4c6cc035", "2016-09-06"): "EMC to DVMT successor chain is not fully crosschecked.",
            _key("US:EODHD:e2c710f1-f687-511b-93ff-233a8b8e40a7", "2022-03-02"): "VIP to VEON successor chain is not fully crosschecked.",
        }.items()
    },
    # SWN is applied only when the complete SWN -> CHK -> EXE chain below is
    # evidenced and crosschecked.  Otherwise this exact, explicit exception is
    # used; no generic fallback exists.
    _key("US:EODHD:e38dbe48-7597-54e3-b3f5-4dcc84b7a7f2", "2024-09-30"): _temporary(
        LifecycleExceptionCode.SUCCESSOR_UNRESOLVED,
        "SWN cannot be applied until the successor CHK to EXE chain is verified.",
    ),
}


# These terminal outcomes are safe only for the exact repaired identity and
# terminal price date.  In particular, AGN was reused by two different legal
# issuers, while LILA/LILAK were reused by the post-split-off securities.
IDENTITY_BOUND_TERMINAL_EVENTS: dict[str, dict[str, Any]] = {
    _key("US:EODHD:9f13974d-7f81-5aac-a3a7-ed1d184bd76b", "2015-03-16"): {
        "symbol": "AGN",
        "action_type": "stock_merger",
        "effective_date": "2015-03-17",
        "cash_amount": 129.22,
        "ratio": 0.3683,
        "new_symbol": "ACT",
        "successor_security_id": "US:EODHD:79ce1c42-8ff6-5c13-b9cf-3df82c913734",
    },
    _key("US:EODHD:79ce1c42-8ff6-5c13-b9cf-3df82c913734", "2020-05-08"): {
        "symbol": "AGN",
        "action_type": "stock_merger",
        "effective_date": "2020-05-08",
        "cash_amount": 120.30,
        "ratio": 0.866,
        "new_symbol": "ABBV",
        "successor_security_id": "US:EODHD:3f3cd70b-d1b0-5b4e-a702-d3ab94fc57fe",
    },
    _key("US:SEC:7992d65b-3a26-5cae-b96d-bd01f695d1c1", "2017-08-31"): {
        "symbol": "DD",
        "action_type": "stock_merger",
        "effective_date": "2017-09-01",
        "cash_amount": None,
        "ratio": 1.282,
        "new_symbol": "DWDP",
        "successor_security_id": "US:EODHD:f174ac51-b4a4-5f29-84b0-12b8b73f0bc3",
        "source_url": (
            "https://www.sec.gov/Archives/edgar/data/30554/"
            "000119312517274840/0001193125-17-274840.txt"
        ),
        "source_hash": (
            "098828aa2714df3fdd52a18b1fffb91d6a72865ff8dd4e94e84f7bc079cf0e64"
        ),
        "source_content_bytes": 204_607,
        "reuse_existing_action": True,
    },
    _key("US:EODHD:5c946c06-0214-5b7b-8e7c-31f91485a215", "2017-12-29"): {
        "symbol": "LILA",
        "action_type": "stock_merger",
        "effective_date": "2018-01-02",
        "cash_amount": None,
        "ratio": 1.0,
        "new_symbol": "LILA",
        "successor_security_id": "US:EODHD:1b6b9beb-42b0-5a06-81f3-23a49627565f",
        "source_url": (
            "https://www.sec.gov/Archives/edgar/data/1570585/"
            "000157058517000401/ex991split-offrecordanddis.htm"
        ),
        "source_hash": (
            "0efad7b02b77a0daefab021c58fdbbb40f03955f069f42eac3e24d403f2813e4"
        ),
        "source_content_bytes": 29_731,
    },
    _key("US:EODHD:24bfb026-6327-5e04-9e32-15589dcb45ba", "2017-12-29"): {
        "symbol": "LILAK",
        "action_type": "stock_merger",
        "effective_date": "2018-01-02",
        "cash_amount": None,
        "ratio": 1.0,
        "new_symbol": "LILAK",
        "successor_security_id": "US:EODHD:7fda02a3-10dd-51a3-96cb-41695fcff341",
        "source_url": (
            "https://www.sec.gov/Archives/edgar/data/1570585/"
            "000157058517000401/ex991split-offrecordanddis.htm"
        ),
        "source_hash": (
            "0efad7b02b77a0daefab021c58fdbbb40f03955f069f42eac3e24d403f2813e4"
        ),
        "source_content_bytes": 29_731,
    },
    _key("US:EODHD:de36b2d8-e15a-5d33-8493-4cc37d0c6ce0", "2017-07-03"): {
        "symbol": "BHI",
        "action_type": "stock_merger",
        "effective_date": "2017-07-05",
        "cash_amount": 17.50,
        "ratio": 1.0,
        "new_symbol": "BHGE",
        "successor_security_id": "US:EODHD:a1542ac4-30f6-57dc-bf2f-79c0ea6aefd2",
    },
    _key("US:EODHD:5fa7bd33-c752-57c7-873c-e9d812d90e05", "2017-02-24"): {
        "symbol": "SE",
        "action_type": "stock_merger",
        "effective_date": "2017-02-27",
        "cash_amount": None,
        "ratio": 0.984,
        "new_symbol": "ENB",
        "successor_security_id": "US:EODHD:8b62832f-27a7-5139-a199-62f9632c21bd",
    },
}
IDENTITY_BOUND_TERMINAL_SYMBOLS = frozenset(
    {
        *(str(value["symbol"]).upper() for value in IDENTITY_BOUND_TERMINAL_EVENTS.values()),
        "TFCF",
        "TFCFA",
    }
)


# These three applied resolutions came from the reviewed short-terminal price
# boundary repair.  The broad lifecycle finalizer sees the same events in its
# frozen report, but that report cannot authorize replacing the independent
# review provenance with its generic reviewer.  Preservation is intentionally
# limited to these exact candidate bindings.  The full required dataset row is
# code-pinned as well as hashed so any field or schema drift fails closed.
EXACT_SHORT_TERMINAL_REVIEWED_RESOLUTIONS: dict[str, dict[str, Any]] = {
    _key("US:EODHD:0c47238f-bf19-5faa-a3ae-25a34ef3d3f5", "2021-05-13"): {
        "resolution": {
            "candidate_id": "f8f7358ae36981dcc5f346f7aaa4c88bf6dbfc3796f685e2bea9648d0442fe3d",
            "security_id": "US:EODHD:0c47238f-bf19-5faa-a3ae-25a34ef3d3f5",
            "symbol": "FLIR",
            "last_price_date": "2021-05-13",
            "resolution": "applied",
            "event_id": "cff77a9d1a8fbd905c0254118710c572c56a14da2086b77d8ba3900a9ac627f6",
            "exception_code": "",
            "exception_reason": "",
            "reviewed_by": "short_terminal_boundary_repair_v1",
            "reviewed_at": "2026-07-19T00:00:00Z",
            "recheck_after": "",
            "successor_security_id": "US:EODHD:02738ac8-c50a-5089-bf68-f174ac71704b",
            "successor_symbol": "TDY",
            "source_url": "https://www.sec.gov/Archives/edgar/data/1094285/000119312521161542/0001193125-21-161542.txt",
            "source": "short_terminal_boundary_repair",
            "retrieved_at": "2026-07-19T00:00:00Z",
            "source_hash": "354312dc20154537f038d2bde1390789b770e8e5c1b62bd20c34449ecadac101",
        },
        "row_sha256": "8d23ae2244b79aa5581eeaa543d8afc67b0c29a180ee85618ae90357d22ab6eb",
    },
    _key("US:EODHD:716dea51-f3a0-5381-9696-d097c877695f", "2021-03-16"): {
        "resolution": {
            "candidate_id": "0134aa07f07f6f8a1ae42fdcf66dc87d2d42752242f8f3508a9cde35759efadb",
            "security_id": "US:EODHD:716dea51-f3a0-5381-9696-d097c877695f",
            "symbol": "QEP",
            "last_price_date": "2021-03-16",
            "resolution": "applied",
            "event_id": "2e8f5c5e5a3a887eb38b579ef45c47ab458770cd37002bcf5634cbfad0ae16da",
            "exception_code": "",
            "exception_reason": "",
            "reviewed_by": "short_terminal_boundary_repair_v1",
            "reviewed_at": "2026-07-19T00:00:00Z",
            "recheck_after": "",
            "successor_security_id": "US:EODHD:c1f9ab83-05e8-57fc-8c07-f775826662c6",
            "successor_symbol": "FANG",
            "source_url": "https://www.sec.gov/Archives/edgar/data/1539838/000119312521084144/0001193125-21-084144.txt",
            "source": "short_terminal_boundary_repair",
            "retrieved_at": "2026-07-19T00:00:00Z",
            "source_hash": "c38c9f61ea9ddcdaef61f393012e620c21dea7c3062634c7aa8dbb53bec8af2a",
        },
        "row_sha256": "a7c64318c23d84c1b5c5b37a72edd41a8f05177e5b439e729f0e418ea6982c22",
    },
    _key("US:EODHD:865c1483-a99b-5066-b55a-649e24804d68", "2025-11-28"): {
        "resolution": {
            "candidate_id": "34a78e61e5ac60c2a56899790d79053e94af92f73ad968069624f22b4f6563d2",
            "security_id": "US:EODHD:865c1483-a99b-5066-b55a-649e24804d68",
            "symbol": "HBI",
            "last_price_date": "2025-11-28",
            "resolution": "applied",
            "event_id": "11d34abfe232c1b262916a2b845cc5987819f7a73b4a76f9cdb95dd1137ae85f",
            "exception_code": "",
            "exception_reason": "",
            "reviewed_by": "short_terminal_boundary_repair_v1",
            "reviewed_at": "2026-07-19T00:00:00Z",
            "recheck_after": "",
            "successor_security_id": "US:EODHD:c18d86ee-dd25-509d-a4de-8a552fd6c69d",
            "successor_symbol": "GIL",
            "source_url": "https://www.sec.gov/Archives/edgar/data/1359841/000119312525303276/0001193125-25-303276.txt",
            "source": "short_terminal_boundary_repair",
            "retrieved_at": "2026-07-19T00:00:00Z",
            "source_hash": "85b0662c74efc9afdbd9e40babd6bb690c65287fd0ea4189cad7981b0148bde8",
        },
        "row_sha256": "6046aa5e767b4d525729218df37b258ec43a69b20fbbb68010b97c89b69ccb48",
    },
}


# Four terminal outcomes were already corrected and reviewed before the broad
# lifecycle finalizer was run.  The frozen report describes the legal close
# for each transaction, while the reviewed actions below deliberately use the
# priceable terminal/market session needed by the backtest engine.  A later
# finalizer replay must therefore retain these exact actions and resolutions;
# it may remove only the exact superseded report-date event.  Every candidate,
# action, resolution, identity boundary, terminal/successor price row, and raw
# SEC payload is pinned so a newly similar-looking event cannot authorize
# itself through this preservation path.
EXACT_PRIOR_TERMINAL_TRANSITIONS: dict[str, dict[str, Any]] = {
    _key("US:EODHD:7b14f17c-6c95-5b75-9e84-86ed9d20f5e3", "2019-06-28"): {
        "candidate": {
            "security_id": "US:EODHD:7b14f17c-6c95-5b75-9e84-86ed9d20f5e3",
            "symbol": "HRS",
            "name": "Harris Corporation",
            "exchange": "NYSE",
            "last_price_date": "2019-06-28",
            "active_to": "2019-06-28",
            "index_remove_dates": ("2019-06-01",),
        },
        "action": {
            "event_id": "2093c4a169a10534ac01ff370ec37aaf240cf6a26bc32c9c0746b89cbe8281d9",
            "security_id": "US:EODHD:7b14f17c-6c95-5b75-9e84-86ed9d20f5e3",
            "action_type": "ticker_change",
            "effective_date": "2019-06-28",
            "ex_date": "2019-06-28",
            "announcement_date": "2019-07-01",
            "record_date": "",
            "payment_date": "",
            "cash_amount": None,
            "ratio": None,
            "currency": "USD",
            "new_security_id": "US:EODHD:ed2a43fc-979c-5915-8f67-131e516279ee",
            "new_symbol": "LHX",
            "source_url": "https://www.sec.gov/Archives/edgar/data/202058/000114036119012139/0001140361-19-012139.txt",
            "source_kind": "official_crosscheck",
            "source": "sec_edgar+stored_price_crosscheck",
            "retrieved_at": "2026-07-18T10:30:29.523939Z",
            "source_hash": "65c7c2925d6d5f671185c1ae3cd9c99806ec14eae86e84758152f677d4f5aa29",
        },
        "superseded_action": {
            "event_id": "84864b3c23dee06565cabd9ec7affba69d0db7041832674e0bd3b1a9f1c3cf18",
            "effective_date": "2019-06-29",
            "ex_date": "2019-06-29",
            "retrieved_at": "2026-07-18T19:58:31.309376Z",
        },
        "resolution": {
            "candidate_id": "3aacbaa34e19afb5ae6668b6fa4618bfbbda3e4d733a7fb5d66e5fbe22f04605",
            "security_id": "US:EODHD:7b14f17c-6c95-5b75-9e84-86ed9d20f5e3",
            "symbol": "HRS",
            "last_price_date": "2019-06-28",
            "resolution": "applied",
            "event_id": "2093c4a169a10534ac01ff370ec37aaf240cf6a26bc32c9c0746b89cbe8281d9",
            "exception_code": "",
            "exception_reason": "",
            "reviewed_by": "us_lifecycle_finalizer_v1",
            "reviewed_at": "2026-07-18T00:00:00Z",
            "recheck_after": "",
            "successor_security_id": "US:EODHD:ed2a43fc-979c-5915-8f67-131e516279ee",
            "successor_symbol": "LHX",
            "source_url": "https://www.sec.gov/Archives/edgar/data/202058/000114036119012139/0001140361-19-012139.txt",
            "source": "lifecycle_finalizer",
            "retrieved_at": "2026-07-18T10:30:29.523939Z",
            "source_hash": "65c7c2925d6d5f671185c1ae3cd9c99806ec14eae86e84758152f677d4f5aa29",
        },
        "superseded_resolution": {
            "event_id": "84864b3c23dee06565cabd9ec7affba69d0db7041832674e0bd3b1a9f1c3cf18",
            "retrieved_at": "2026-07-18T19:58:31.309376Z",
        },
        "target_master": {
            "primary_symbol": "HRS", "provider_symbol": "HRS.US",
            "name": "Harris Corporation", "exchange": "NYSE",
            "active_from": "2015-01-02", "active_to": "2019-06-28",
            "source": "eodhd_exchange_symbols",
            "source_url": "https://eodhd.com/api/exchange-symbol-list/US?delisted=0",
            "retrieved_at": "2026-07-16T15:56:01.033938Z",
            "source_hash": "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99",
        },
        "target_history": {
            "symbol": "HRS", "exchange": "NYSE",
            "effective_from": "2015-01-01", "effective_to": "",
            "source": "eodhd_exchange_symbols",
            "source_url": "https://eodhd.com/api/exchange-symbol-list/US?delisted=0",
            "retrieved_at": "2026-07-16T15:56:01.033938Z",
            "source_hash": "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99",
        },
        "successor_master": {
            "primary_symbol": "LHX", "provider_symbol": "LHX.US",
            "name": "L3Harris Technologies Inc", "exchange": "NYSE",
            "active_from": "2015-01-01", "active_to": "",
            "source": "eodhd_exchange_symbols",
            "source_url": "https://eodhd.com/api/exchange-symbol-list/US?delisted=0",
            "retrieved_at": "2026-07-16T15:56:01.033938Z",
            "source_hash": "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99",
        },
        "target_price": {
            "session": "2019-06-28", "ohlcv": (190.43, 191.44, 187.71, 189.13, 7_197_492.0),
            "retrieved_at": "2026-07-16T15:57:22.793200Z",
            "source_hash": "b5a2192b79c0901c5ec2aa2343cf45d144ab769ffc72856e0c050f3611c42045",
        },
        "successor_price": {
            "session": "2019-06-28", "ohlcv": (190.43, 191.44, 187.71, 189.13, 4_782_100.0),
            "retrieved_at": "2026-07-16T15:58:26.014757Z",
            "source_hash": "38a0b235975582c0e16c36fe5b01f3cdf023878a725bdf615cfb33b4dfaf2609",
        },
        "archive_bytes": 4_605_401,
        "archive_retrieved_at": "2026-07-18T10:30:30.790016Z",
    },
    _key("US:EODHD:e91da1be-75dc-5919-9464-fa580892871a", "2019-06-28"): {
        "candidate": {
            "security_id": "US:EODHD:e91da1be-75dc-5919-9464-fa580892871a",
            "symbol": "LLL", "name": "L3 Technologies Inc", "exchange": "NASDAQ",
            "last_price_date": "2019-06-28", "active_to": "2019-06-28",
            "index_remove_dates": ("2019-07-01",),
        },
        "action": {
            "event_id": "42934ff153b211e42af214f866f5bc4e9e6b9a020168b5a6488030a9e929b8af",
            "security_id": "US:EODHD:e91da1be-75dc-5919-9464-fa580892871a",
            "action_type": "stock_merger", "effective_date": "2019-06-28",
            "ex_date": "2019-06-28", "announcement_date": "2019-07-01",
            "record_date": "", "payment_date": "", "cash_amount": None,
            "ratio": 1.3, "currency": "USD",
            "new_security_id": "US:EODHD:ed2a43fc-979c-5915-8f67-131e516279ee",
            "new_symbol": "LHX",
            "source_url": "https://www.sec.gov/Archives/edgar/data/202058/000114036119012139/0001140361-19-012139.txt",
            "source_kind": "official_crosscheck", "source": "sec_edgar+stored_price_crosscheck",
            "retrieved_at": "2026-07-18T10:30:30.790016Z",
            "source_hash": "65c7c2925d6d5f671185c1ae3cd9c99806ec14eae86e84758152f677d4f5aa29",
        },
        "superseded_action": {
            "event_id": "667753d1424574220c0ee7c788da9bcd9e5ca7873acfc1c5e9e951c6b6d63195",
            "effective_date": "2019-06-29", "ex_date": "2019-06-29",
            "retrieved_at": "2026-07-18T19:58:32.381481Z",
        },
        "resolution": {
            "candidate_id": "f31d2e43f52e4378ed9cbe10c2f11a11f93d83049e39277c84131b538f47594e",
            "security_id": "US:EODHD:e91da1be-75dc-5919-9464-fa580892871a",
            "symbol": "LLL", "last_price_date": "2019-06-28", "resolution": "applied",
            "event_id": "42934ff153b211e42af214f866f5bc4e9e6b9a020168b5a6488030a9e929b8af",
            "exception_code": "", "exception_reason": "",
            "reviewed_by": "us_lifecycle_finalizer_v1", "reviewed_at": "2026-07-18T00:00:00Z",
            "recheck_after": "",
            "successor_security_id": "US:EODHD:ed2a43fc-979c-5915-8f67-131e516279ee",
            "successor_symbol": "LHX",
            "source_url": "https://www.sec.gov/Archives/edgar/data/202058/000114036119012139/0001140361-19-012139.txt",
            "source": "lifecycle_finalizer", "retrieved_at": "2026-07-18T10:30:30.790016Z",
            "source_hash": "65c7c2925d6d5f671185c1ae3cd9c99806ec14eae86e84758152f677d4f5aa29",
        },
        "superseded_resolution": {
            "event_id": "667753d1424574220c0ee7c788da9bcd9e5ca7873acfc1c5e9e951c6b6d63195",
            "retrieved_at": "2026-07-18T19:58:32.381481Z",
        },
        "target_master": {
            "primary_symbol": "LLL", "provider_symbol": "LLL_old.US",
            "name": "L3 Technologies Inc", "exchange": "NASDAQ",
            "active_from": "2015-01-02", "active_to": "2019-06-28",
            "source": "eodhd_exchange_symbols",
            "source_url": "https://eodhd.com/api/exchange-symbol-list/US?delisted=0",
            "retrieved_at": "2026-07-16T15:56:01.033938Z",
            "source_hash": "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99",
        },
        "target_history": {
            "symbol": "LLL", "exchange": "NASDAQ", "effective_from": "2015-01-01",
            "effective_to": "", "source": "eodhd_exchange_symbols",
            "source_url": "https://eodhd.com/api/exchange-symbol-list/US?delisted=0",
            "retrieved_at": "2026-07-16T15:56:01.033938Z",
            "source_hash": "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99",
        },
        "successor_master": {
            "primary_symbol": "LHX", "provider_symbol": "LHX.US",
            "name": "L3Harris Technologies Inc", "exchange": "NYSE",
            "active_from": "2015-01-01", "active_to": "", "source": "eodhd_exchange_symbols",
            "source_url": "https://eodhd.com/api/exchange-symbol-list/US?delisted=0",
            "retrieved_at": "2026-07-16T15:56:01.033938Z",
            "source_hash": "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99",
        },
        "target_price": {
            "session": "2019-06-28", "ohlcv": (247.31, 248.75, 244.0, 245.17, 3_744_933.0),
            "retrieved_at": "2026-07-16T15:58:24.483491Z",
            "source_hash": "06451b6996930a2fabb7d5a3586d2b7fb1911abf929f84087562987aba13d1d9",
        },
        "successor_price": {
            "session": "2019-06-28", "ohlcv": (190.43, 191.44, 187.71, 189.13, 4_782_100.0),
            "retrieved_at": "2026-07-16T15:58:26.014757Z",
            "source_hash": "38a0b235975582c0e16c36fe5b01f3cdf023878a725bdf615cfb33b4dfaf2609",
        },
        "archive_bytes": 4_605_401,
        "archive_retrieved_at": "2026-07-18T10:30:30.790016Z",
    },
    _key("US:EODHD:6637f59f-7ea0-5273-9fcb-9f48277650c2", "2021-07-21"): {
        "candidate": {
            "security_id": "US:EODHD:6637f59f-7ea0-5273-9fcb-9f48277650c2",
            "symbol": "ALXN", "name": "Alexion Pharmaceuticals Inc", "exchange": "NASDAQ",
            "last_price_date": "2021-07-21", "active_to": "2021-07-21",
            "index_remove_dates": ("2021-07-21",),
        },
        "action": {
            "event_id": "50f8bfb2bb620c136dd9f3ce8699d049d5c394f3b2a447cb316019f55f6351f7",
            "security_id": "US:EODHD:6637f59f-7ea0-5273-9fcb-9f48277650c2",
            "action_type": "stock_merger", "effective_date": "2021-07-21",
            "ex_date": "2021-07-21", "announcement_date": "2021-07-22",
            "record_date": "", "payment_date": "", "cash_amount": 60.0,
            "ratio": 2.1243, "currency": "USD",
            "new_security_id": "US:EODHD:431e0049-d4ce-5318-8920-b9115ffe775a",
            "new_symbol": "AZN",
            "source_url": "https://www.sec.gov/Archives/edgar/data/899866/000110465921094624/0001104659-21-094624.txt",
            "source_kind": "official_crosscheck", "source": "sec_edgar+stored_price_crosscheck",
            "retrieved_at": "2026-07-18T10:30:48.961148Z",
            "source_hash": "1bce4c27938a9fc3f8702e67b6360a1a1b3f7934927dd8d93ee832e2c9813264",
        },
        "superseded_action": {
            "event_id": "d3c51529086a1fc56c37286f450191253cc8fa4f9bb2201e4aca0dc8f7f33cf8",
            "effective_date": "2021-07-22", "ex_date": "2021-07-22",
            "retrieved_at": "2026-07-18T19:58:49.957996Z",
        },
        "resolution": {
            "candidate_id": "253fc09bd15169d966e577c988220b366d95e36a851a7ce7eca91dd83123937c",
            "security_id": "US:EODHD:6637f59f-7ea0-5273-9fcb-9f48277650c2",
            "symbol": "ALXN", "last_price_date": "2021-07-21", "resolution": "applied",
            "event_id": "50f8bfb2bb620c136dd9f3ce8699d049d5c394f3b2a447cb316019f55f6351f7",
            "exception_code": "", "exception_reason": "",
            "reviewed_by": "us_lifecycle_finalizer_v1", "reviewed_at": "2026-07-18T00:00:00Z",
            "recheck_after": "",
            "successor_security_id": "US:EODHD:431e0049-d4ce-5318-8920-b9115ffe775a",
            "successor_symbol": "AZN",
            "source_url": "https://www.sec.gov/Archives/edgar/data/899866/000110465921094624/0001104659-21-094624.txt",
            "source": "lifecycle_finalizer", "retrieved_at": "2026-07-18T10:30:48.961148Z",
            "source_hash": "1bce4c27938a9fc3f8702e67b6360a1a1b3f7934927dd8d93ee832e2c9813264",
        },
        "superseded_resolution": {
            "event_id": "d3c51529086a1fc56c37286f450191253cc8fa4f9bb2201e4aca0dc8f7f33cf8",
            "retrieved_at": "2026-07-18T19:58:49.957996Z",
        },
        "target_master": {
            "primary_symbol": "ALXN", "provider_symbol": "ALXN.US",
            "name": "Alexion Pharmaceuticals Inc", "exchange": "NASDAQ",
            "active_from": "2015-01-02", "active_to": "2021-07-21",
            "source": "eodhd_exchange_symbols",
            "source_url": "https://eodhd.com/api/exchange-symbol-list/US?delisted=0",
            "retrieved_at": "2026-07-16T15:56:01.033938Z",
            "source_hash": "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99",
        },
        "target_history": {
            "symbol": "ALXN", "exchange": "NASDAQ", "effective_from": "2015-01-01",
            "effective_to": "", "source": "eodhd_exchange_symbols",
            "source_url": "https://eodhd.com/api/exchange-symbol-list/US?delisted=0",
            "retrieved_at": "2026-07-16T15:56:01.033938Z",
            "source_hash": "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99",
        },
        "successor_master": {
            "primary_symbol": "AZN", "provider_symbol": "AZN.US", "name": "AstraZeneca PLC",
            "exchange": "NYSE", "active_from": "2015-01-02", "active_to": "",
            "source": "eodhd_exchange_symbols",
            "source_url": "https://eodhd.com/api/exchange-symbol-list/US?delisted=0",
            "retrieved_at": "2026-07-16T15:56:01.033938Z",
            "source_hash": "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99",
        },
        "target_price": {
            "session": "2021-07-21", "ohlcv": (182.5, 182.5, 182.5, 182.5, 0.0),
            "retrieved_at": "2026-07-16T15:57:09.647195Z",
            "source_hash": "ce2f97fd2043520db1dfe8a66d9e3e3a6891642d6118bc263f6e719fb8abdd41",
        },
        "successor_price": {
            "session": "2021-07-21", "ohlcv": (57.91, 58.06, 57.53, 57.77, 9_776_550.0),
            "retrieved_at": "2026-07-16T15:56:51.017223Z",
            "source_hash": "82dffed2c4c67f423eee38a6396eadf41ed31bcabd7cde932fbf6753fe574e72",
        },
        "archive_bytes": 347_834,
        "archive_retrieved_at": "2026-07-18T10:30:48.961148Z",
    },
    _key("US:EODHD:b32ce08e-4158-5aea-b7ba-6175e716fa41", "2021-01-15"): {
        "candidate": {
            "security_id": "US:EODHD:b32ce08e-4158-5aea-b7ba-6175e716fa41",
            "symbol": "CXO", "name": "Concho Resources Inc", "exchange": "NYSE",
            "last_price_date": "2021-01-15", "active_to": "2021-01-15",
            "index_remove_dates": ("2021-01-21",),
        },
        "action": {
            "event_id": "162dee2832998354021e79efcafa8a915d3c2f0744b939c5f570c93deb2f1a55",
            "security_id": "US:EODHD:b32ce08e-4158-5aea-b7ba-6175e716fa41",
            "action_type": "stock_merger", "effective_date": "2021-01-19",
            "ex_date": "2021-01-19", "announcement_date": "2021-01-15",
            "record_date": "", "payment_date": "", "cash_amount": None,
            "ratio": 1.46, "currency": "USD",
            "new_security_id": "US:EODHD:cfbc7973-3e6d-5334-a29f-7d0d83693ae0",
            "new_symbol": "COP",
            "source_url": "https://www.sec.gov/Archives/edgar/data/1163165/000110465921004775/0001104659-21-004775.txt",
            "source_kind": "official_crosscheck", "source": "sec_edgar+stored_price_crosscheck",
            "retrieved_at": "2026-07-18T10:30:46.496018Z",
            "source_hash": "65047e754c1838fd1c5cee20d03d8941ee6990540b5b2307beb2e9f8839bcc9e",
        },
        "superseded_action": {
            "event_id": "db752821ea192e7c3ea7ebc90f02f8474b017b053d52334cb0cca7e6a803396b",
            "effective_date": "2021-01-15", "ex_date": "2021-01-15",
            "retrieved_at": "2026-07-18T19:58:47.568055Z",
        },
        "resolution": {
            "candidate_id": "f326b63d0229f68816a663f468e1723b8925db58b8f21607cb3b16e72cfb531c",
            "security_id": "US:EODHD:b32ce08e-4158-5aea-b7ba-6175e716fa41",
            "symbol": "CXO", "last_price_date": "2021-01-15", "resolution": "applied",
            "event_id": "162dee2832998354021e79efcafa8a915d3c2f0744b939c5f570c93deb2f1a55",
            "exception_code": "", "exception_reason": "",
            "reviewed_by": "terminal_boundary_repair_v1", "reviewed_at": "2026-07-18T14:00:00Z",
            "recheck_after": "",
            "successor_security_id": "US:EODHD:cfbc7973-3e6d-5334-a29f-7d0d83693ae0",
            "successor_symbol": "COP",
            "source_url": "https://www.sec.gov/Archives/edgar/data/1163165/000110465921004775/0001104659-21-004775.txt",
            "source": "terminal_boundary_repair", "retrieved_at": "2026-07-18T14:00:00Z",
            "source_hash": "65047e754c1838fd1c5cee20d03d8941ee6990540b5b2307beb2e9f8839bcc9e",
        },
        "superseded_resolution": {
            "event_id": "db752821ea192e7c3ea7ebc90f02f8474b017b053d52334cb0cca7e6a803396b",
            "reviewed_by": "us_lifecycle_finalizer_v1", "reviewed_at": "2026-07-18T00:00:00Z",
            "source": "lifecycle_finalizer", "retrieved_at": "2026-07-18T19:58:47.568055Z",
        },
        "target_master": {
            "primary_symbol": "CXO", "provider_symbol": "CXO.US",
            "name": "Concho Resources Inc", "exchange": "NYSE",
            "active_from": "2015-01-02", "active_to": "2021-01-15",
            "source": "official_terminal_boundary_repair",
            "source_url": "https://www.sec.gov/Archives/edgar/data/1163165/000110465921004775/0001104659-21-004775.txt",
            "retrieved_at": "2026-07-18T14:00:00Z",
            "source_hash": "65047e754c1838fd1c5cee20d03d8941ee6990540b5b2307beb2e9f8839bcc9e",
        },
        "target_history": {
            "symbol": "CXO", "exchange": "NYSE", "effective_from": "2015-01-01",
            "effective_to": "2021-01-15", "source": "official_terminal_boundary_repair",
            "source_url": "https://www.sec.gov/Archives/edgar/data/1163165/000110465921004775/0001104659-21-004775.txt",
            "retrieved_at": "2026-07-18T14:00:00Z",
            "source_hash": "65047e754c1838fd1c5cee20d03d8941ee6990540b5b2307beb2e9f8839bcc9e",
        },
        "successor_master": {
            "primary_symbol": "COP", "provider_symbol": "COP.US", "name": "ConocoPhillips",
            "exchange": "NYSE", "active_from": "2015-01-01", "active_to": "",
            "source": "eodhd_exchange_symbols",
            "source_url": "https://eodhd.com/api/exchange-symbol-list/US?delisted=0",
            "retrieved_at": "2026-07-16T15:56:01.033938Z",
            "source_hash": "2e43e71c491c2eda9e18ee156aac21149b170f84eb3f84ff107f125c269dfb99",
        },
        "target_price": {
            "session": "2021-01-15", "ohlcv": (69.12, 69.12, 64.6, 65.6, 28_565_330.0),
            "retrieved_at": "2026-07-16T15:57:56.073822Z",
            "source_hash": "59e7e2843948065b50ced2daf2012cab37061643a03e0929426b7507997a64bb",
        },
        "successor_price": {
            "session": "2021-01-19", "ohlcv": (45.15, 46.15, 44.86, 46.0, 14_498_800.0),
            "retrieved_at": "2026-07-16T15:58:12.348065Z",
            "source_hash": "4d0dd5898022ca5c32fb548c86be65c6d5ff115e5a7e4532a7ecedafc224841a",
        },
        "archive_bytes": 316_992,
        "archive_retrieved_at": "2026-07-18T10:30:46.496018Z",
    },
}


# Old LiLAC tracking-share prices and the post-split-off issuer prices use
# different adjustment bases, so a raw price-level comparison is invalid.  The
# only allowed alternate check is this exact identity/date-bound, immutable
# primary/external/successor archive inventory.  Any changed URL, byte hash,
# row count, calendar boundary, or overlap statistic fails closed.
REVIEWED_CROSS_BASIS_TERMINAL_PRICE_PROVENANCE: dict[str, dict[str, Any]] = {
    _key("US:EODHD:5c946c06-0214-5b7b-8e7c-31f91485a215", "2017-12-29"): {
        "symbol": "LILA",
        "active_from": "2015-07-02",
        "active_to": "2017-12-29",
        "primary_sessions": 630,
        "primary_archive_sessions": 630,
        "primary_source": "yahoo_chart_adjusted_basis_primary",
        "primary_source_url": (
            "https://query1.finance.yahoo.com/v8/finance/chart/LILA"
            "?period1=1434931200&period2=1514764800&interval=1d&events=history"
        ),
        "primary_source_hash": (
            "93480c09d74adde13f320eac83147df83256f2e86ff28506e97ff096de88394f"
        ),
        "primary_content_bytes": 71_194,
        "external_source": "boris_kaggle_cc0_v3",
        "external_source_url": (
            "https://www.kaggle.com/api/v1/datasets/download/borismarjanovic/"
            "price-volume-data-for-all-us-stocks-etfs/Data%2FStocks%2Flila.us.txt"
            "?datasetVersionNumber=3"
        ),
        "external_source_hash": (
            "9885111c20ca809ce8791c429cd8eb66a62470b53ab71f7c2ac6a573d576f73c"
        ),
        "external_content_bytes": 26_596,
        "external_raw_rows": 599,
        "overlap_start": "2015-07-02",
        "overlap_end": "2017-11-10",
        "overlap_sessions": 597,
        "uncrosschecked_tail_sessions": 33,
        "minimum_return_correlation": 0.995,
        "maximum_p99_scaled_close_error": 0.05,
        "successor_security_id": "US:EODHD:1b6b9beb-42b0-5a06-81f3-23a49627565f",
        "successor_first_session": "2018-01-02",
        "successor_sessions": 2_144,
        "successor_source": "eodhd_eod",
        "successor_source_url": (
            "https://eodhd.com/api/eod/LILA.US?from=2015-01-01&to=2026-07-15"
        ),
        "successor_source_hash": (
            "2d83bb3bc51253905dab9cb15156797f3097fb41e905bf3fff0d60d34aa19bc4"
        ),
        "successor_content_bytes": 256_335,
    },
    _key("US:EODHD:24bfb026-6327-5e04-9e32-15589dcb45ba", "2017-12-29"): {
        "symbol": "LILAK",
        "active_from": "2015-07-02",
        "active_to": "2017-12-29",
        "primary_sessions": 630,
        "primary_archive_sessions": 637,
        "primary_source": "yahoo_chart_adjusted_basis_primary",
        "primary_source_url": (
            "https://query1.finance.yahoo.com/v8/finance/chart/LILAK"
            "?period1=1434931200&period2=1514764800&interval=1d&events=history"
        ),
        "primary_source_hash": (
            "50c1f7cfca4ed8c2f750ed982581a140762aef466be6dddb1b026c3ff842ce61"
        ),
        "primary_content_bytes": 72_216,
        "external_source": "boris_kaggle_cc0_v3",
        "external_source_url": (
            "https://www.kaggle.com/api/v1/datasets/download/borismarjanovic/"
            "price-volume-data-for-all-us-stocks-etfs/Data%2FStocks%2Flilak.us.txt"
            "?datasetVersionNumber=3"
        ),
        "external_source_hash": (
            "b5a56cc0c1b5a478354d85149c2370ccde6146f7f43d94566dcc76382db610e4"
        ),
        "external_content_bytes": 26_729,
        "external_raw_rows": 599,
        "overlap_start": "2015-07-02",
        "overlap_end": "2017-11-10",
        "overlap_sessions": 597,
        "uncrosschecked_tail_sessions": 33,
        "minimum_return_correlation": 0.995,
        "maximum_p99_scaled_close_error": 0.05,
        "successor_security_id": "US:EODHD:7fda02a3-10dd-51a3-96cb-41695fcff341",
        "successor_first_session": "2018-01-02",
        "successor_sessions": 2_144,
        "successor_source": "eodhd_eod",
        "successor_source_url": (
            "https://eodhd.com/api/eod/LILAK.US?from=2015-01-01&to=2026-07-15"
        ),
        "successor_source_hash": (
            "b8cb0bfae114e27c69668639271829b3692d14d892cd00a4d7857fba8172e0a5"
        ),
        "successor_content_bytes": 256_896,
    },
}


DD_EXISTING_ACTION = {
    "event_id": "7ad3b0a7ccdec1034fb7bd56914e8ad20b30d13045831ec776236626db8342c2",
    "security_id": "US:SEC:7992d65b-3a26-5cae-b96d-bd01f695d1c1",
    "action_type": "stock_merger",
    "effective_date": "2017-09-01",
    "ex_date": "2017-09-01",
    "cash_amount": None,
    "ratio": 1.282,
    "currency": "USD",
    "new_security_id": "US:EODHD:f174ac51-b4a4-5f29-84b0-12b8b73f0bc3",
    "new_symbol": "DWDP",
    "official": True,
    "source_url": IDENTITY_BOUND_TERMINAL_EVENTS[
        _key("US:SEC:7992d65b-3a26-5cae-b96d-bd01f695d1c1", "2017-08-31")
    ]["source_url"],
    "source_kind": "official_filing",
    "source": "official_dwdp_identity_repair",
    "source_hash": IDENTITY_BOUND_TERMINAL_EVENTS[
        _key("US:SEC:7992d65b-3a26-5cae-b96d-bd01f695d1c1", "2017-08-31")
    ]["source_hash"],
}


JWN_SECURITY_ID = "US:EODHD:b2e91d46-5fe8-52c5-b587-c33c23de5095"
CHK_EXE_EVIDENCE = {
    "action_type": "ticker_change",
    "effective_date": "2024-10-02",
    "new_symbol": "EXE",
    "confidence": "high",
    "filing_date": "2024-10-01",
    "source_url": (
        "https://www.sec.gov/Archives/edgar/data/895126/"
        "000110465924104976/0001104659-24-104976.txt"
    ),
    "source_hash": "5112367c6043776743c2532071d2d857d77faae96c8317f77af0aa0c8e9259b1",
    "retrieved_at": REVIEWED_AT,
    "content_type": "text/plain",
    "reason": "SEC filing states CHK changed to EXE effective at the open on 2024-10-02.",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline, fail-closed US lifecycle coverage finalizer."
    )
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--sec-cache", default=str(DEFAULT_SEC_CACHE))
    parser.add_argument("--hints", default=str(DEFAULT_HINTS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Validate only (default).")
    mode.add_argument("--apply", action="store_true", help="Write validated outputs and commit a release.")
    return parser.parse_args(argv)


def load_report_document(path: Path) -> ReportDocument:
    if not path.is_file():
        raise FileNotFoundError(f"Lifecycle evidence report is missing: {path}")
    content = path.read_bytes()
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Lifecycle evidence report is invalid JSON: {path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("records"), dict):
        raise ValueError("Lifecycle evidence report must contain a full records object.")
    return ReportDocument(path=path, content=content, value=value)


def _candidate_frame(candidates: Iterable[LifecycleCandidate]) -> pd.DataFrame:
    rows = [asdict(item) for item in candidates]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=("security_id", "last_price_date"))


def _validate_full_report(
    document: ReportDocument,
    release: DataRelease,
    candidates: tuple[LifecycleCandidate, ...],
    *,
    hints_path: Path = DEFAULT_HINTS,
) -> dict[str, dict[str, Any]]:
    report = document.value
    expected_binding = build_lifecycle_report_binding(
        release_version=release.version,
        completed_session=release.completed_session,
        dataset_versions=release.dataset_versions,
        candidates=candidates,
        hints_path=hints_path,
        sec_fetch_policy=report.get(
            "sec_fetch_policy", SEC_FETCH_POLICY_CACHE_ONLY
        ),
        sec_max_http_attempts=report.get(
            "sec_max_http_attempts", DEFAULT_SEC_MAX_HTTP_ATTEMPTS
        ),
        sec_max_http_attempts_per_candidate=report.get(
            "sec_max_http_attempts_per_candidate",
            DEFAULT_SEC_MAX_HTTP_ATTEMPTS_PER_CANDIDATE,
        ),
        sec_max_http_attempts_per_request=report.get(
            "sec_max_http_attempts_per_request",
            SEC_MAX_HTTP_ATTEMPTS_PER_REQUEST,
        ),
        sec_http_attempts=report.get("sec_http_attempts", 0),
        sec_http_attempts_by_candidate=report.get(
            "sec_http_attempts_by_candidate", {}
        ),
    )
    validate_lifecycle_report_binding(
        report,
        expected_binding,
        purpose="lifecycle finalization",
    )
    records = {str(key): value for key, value in report["records"].items()}
    expected = {item.security_id: item for item in candidates}
    missing = sorted(set(expected) - set(records))
    extra = sorted(set(records) - set(expected))
    if missing or extra:
        raise RuntimeError(
            "Lifecycle report is not the full current candidate set: "
            f"missing={missing}, extra={extra}"
        )
    for security_id, candidate in expected.items():
        record = records[security_id]
        if not isinstance(record, dict) or not isinstance(record.get("candidate"), dict):
            raise RuntimeError(f"Lifecycle report record is malformed: {security_id}")
        identity = record["candidate"]
        actual = (
            str(identity.get("security_id") or ""),
            str(identity.get("symbol") or "").upper(),
            _date(identity.get("last_price_date")),
        )
        wanted = (candidate.security_id, candidate.symbol.upper(), _date(candidate.last_price_date))
        if actual != wanted:
            raise RuntimeError(
                f"Lifecycle report candidate identity mismatch for {security_id}: "
                f"expected={wanted!r}, found={actual!r}"
            )
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("Lifecycle report summary is required.")
    expected_count = len(candidates)
    if int(summary.get("candidate_count", -1)) != expected_count or int(
        summary.get("collected_count", -1)
    ) != expected_count:
        raise RuntimeError("Lifecycle report summary does not describe the full candidate set.")
    recomputed_eligible = sum(bool(item.get("eligible_for_apply")) for item in records.values())
    if int(summary.get("eligible_count", -1)) != recomputed_eligible:
        raise RuntimeError("Lifecycle report eligible_count is stale or inconsistent.")
    if int(summary.get("unresolved_count", -1)) != expected_count - recomputed_eligible:
        raise RuntimeError("Lifecycle report unresolved_count is stale or inconsistent.")
    expected_sec_summary = {
        "sec_fetch_policy": report["sec_fetch_policy"],
        "sec_max_http_attempts": int(report["sec_max_http_attempts"]),
        "sec_max_http_attempts_per_candidate": int(
            report["sec_max_http_attempts_per_candidate"]
        ),
        "sec_max_http_attempts_per_request": int(
            report["sec_max_http_attempts_per_request"]
        ),
        "sec_http_attempts": int(report["sec_http_attempts"]),
        "sec_http_attempts_remaining": int(report["sec_max_http_attempts"])
        - int(report["sec_http_attempts"]),
        "sec_http_attempts_by_candidate": dict(
            report["sec_http_attempts_by_candidate"]
        ),
    }
    for field, expected_value in expected_sec_summary.items():
        if summary.get(field) != expected_value:
            raise RuntimeError(
                "Lifecycle report SEC HTTP summary is stale or inconsistent: "
                f"field={field}, expected={expected_value!r}, "
                f"found={summary.get(field)!r}"
            )
    return records


def _date(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Expected a valid date, found {value!r}")
    return pd.Timestamp(parsed).date().isoformat()


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _is_official_sec_url(value: str) -> bool:
    parsed = urlparse(str(value))
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (host == "sec.gov" or host.endswith(".sec.gov"))


def _is_official_exception_url(value: str) -> bool:
    parsed = urlparse(str(value))
    host = (parsed.hostname or "").lower()
    official_hosts = ("sec.gov", "fdic.gov")
    return parsed.scheme == "https" and any(
        host == domain or host.endswith(f".{domain}") for domain in official_hosts
    )


class _ArtifactCache:
    def __init__(
        self,
        root: Path,
        *,
        archive_replay_factory: Callable[[], Any] | None = None,
        archive_authorized_artifacts: Mapping[
            LifecycleCandidate, frozenset[tuple[str, str]]
        ] | None = None,
    ):
        self.root = Path(root)
        self._by_hash: dict[str, bytes] | None = None
        self._archive_replay_factory = archive_replay_factory
        self._archive_replay: Any | None = None
        self._archive_authorized_artifacts = dict(
            archive_authorized_artifacts or {}
        )

    def content(
        self,
        source_hash: str,
        *,
        source_url: str = "",
        candidate: LifecycleCandidate | None = None,
    ) -> bytes:
        wanted = str(source_hash).strip().lower()
        if len(wanted) != 64 or any(value not in "0123456789abcdef" for value in wanted):
            raise ValueError(f"Invalid SEC artifact SHA-256: {source_hash!r}")
        if self._by_hash is None:
            values: dict[str, bytes] = {}
            if self.root.is_dir():
                for path in sorted(self.root.glob("*.bin")):
                    content = path.read_bytes()
                    values.setdefault(sha256_bytes(content), content)
            self._by_hash = values
        cached = self._by_hash.get(wanted)
        if cached is not None:
            return cached
        authorized = bool(
            candidate is not None
            and (source_url, wanted)
            in self._archive_authorized_artifacts.get(candidate, frozenset())
        )
        if self._archive_replay_factory is not None and authorized:
            if self._archive_replay is None:
                self._archive_replay = self._archive_replay_factory()
            archived = self._archive_replay(source_url, candidate)
            if archived is not None:
                observed = sha256_bytes(archived)
                if observed != wanted:
                    raise RuntimeError(
                        "Current-release SEC archive payload does not match the exact "
                        "report artifact hash: "
                        f"url={source_url}, expected={wanted}, observed={observed}"
                    )
                return archived
        raise FileNotFoundError(
            f"Cached official SEC artifact is missing: {wanted}"
        )


def _artifact_from_event(
    event: Mapping[str, Any],
    record: Mapping[str, Any],
    cache: _ArtifactCache,
    *,
    trusted_override: bool,
    candidate: LifecycleCandidate | None = None,
) -> SourceArtifact:
    source_url = _text(event.get("source_url") or record.get("source_url"))
    source_hash = _text(event.get("source_hash") or record.get("source_hash")).lower()
    if not _is_official_sec_url(source_url):
        raise RuntimeError(f"Applied lifecycle evidence is not an official SEC URL: {source_url!r}")
    metadata = None
    for item in record.get("artifacts") or ():
        if (
            isinstance(item, dict)
            and _text(item.get("source_hash")).lower() == source_hash
            and _text(item.get("source_url")) == source_url
        ):
            metadata = item
            break
    if metadata is None and not trusted_override:
        raise RuntimeError("Applied lifecycle evidence is absent from the report artifact list.")
    metadata = dict(metadata or event)
    content = cache.content(
        source_hash,
        source_url=source_url,
        candidate=candidate,
    )
    if sha256_bytes(content) != source_hash:  # pragma: no cover - cache indexes by hash
        raise RuntimeError(f"Cached SEC artifact hash mismatch: {source_hash}")
    return SourceArtifact(
        source=_text(metadata.get("source")) or "sec_edgar_filing",
        source_url=source_url,
        retrieved_at=_text(metadata.get("retrieved_at")) or REVIEWED_AT,
        content=content,
        content_type=_text(metadata.get("content_type")) or "text/plain",
    )


def _artifact_from_exception(
    spec: ExceptionSpec,
    record: Mapping[str, Any],
    cache: _ArtifactCache,
    official_evidence_specs: Mapping[
        str, OfficialLifecycleExceptionEvidenceSpec
    ] | None = None,
    *,
    candidate: LifecycleCandidate | None = None,
) -> SourceArtifact | None:
    permanent = str(spec.code) in PERMANENT_EXCEPTION_CODES
    if permanent and not spec.evidence_id:
        raise RuntimeError(
            "Permanent lifecycle exception requires one exact identity/date-bound "
            "official evidence registry entry; direct or report-derived provenance "
            f"is forbidden: {spec.code}"
        )
    source_url = _text(spec.source_url)
    source_hash = _text(spec.source_hash).lower()
    if spec.evidence_id:
        evidence = (official_evidence_specs or {}).get(spec.evidence_id)
        if evidence is None:
            raise RuntimeError(
                "Official lifecycle exception evidence registry entry is missing: "
                f"{spec.evidence_id}"
            )
        if evidence.exception_code != str(spec.code):
            raise RuntimeError(
                "Official lifecycle exception evidence code does not match the "
                f"reviewed exception: {spec.evidence_id}"
            )
        if evidence.resolution_kind != "exception":
            raise RuntimeError(
                "Applied-event evidence cannot be used as a lifecycle exception: "
                f"{spec.evidence_id}"
            )
        if not evidence.matches_candidate(record.get("candidate") or {}):
            raise RuntimeError(
                "Official lifecycle exception evidence is not bound to this exact "
                f"candidate: {spec.evidence_id}"
            )
        if evidence.claim != spec.reason:
            raise RuntimeError(
                "Official lifecycle exception evidence claim does not match the "
                f"reviewed exception: {spec.evidence_id}"
            )
        source_url = evidence.source_url
        source_hash = evidence.source_sha256
        if not source_hash:
            raise RuntimeError(
                "Official lifecycle exception evidence is not reviewer-pinned: "
                f"{spec.evidence_id}"
            )
    if not source_url and not source_hash:
        if spec.require_official_provenance:
            raise RuntimeError(
                "Required official exception provenance is missing: "
                f"{spec.code}"
            )
        return None
    if not source_url or not source_hash:
        raise RuntimeError(
            "Official exception provenance requires both source_url and source_hash: "
            f"{spec.code}"
        )
    if not _is_official_exception_url(source_url):
        raise RuntimeError(
            f"Lifecycle exception evidence is not an official URL: {source_url!r}"
        )
    metadata = None
    for item in record.get("artifacts") or ():
        if (
            isinstance(item, dict)
            and _text(item.get("source_hash")).lower() == source_hash
            and _text(item.get("source_url")) == source_url
        ):
            metadata = item
            break
    if metadata is None:
        raise RuntimeError(
            "Official exception evidence is absent from the report artifact list: "
            f"{source_url}"
        )
    content = cache.content(
        source_hash,
        source_url=source_url,
        candidate=candidate,
    )
    if sha256_bytes(content) != source_hash:  # pragma: no cover - cache indexes by hash
        raise RuntimeError(
            f"Cached official exception artifact hash mismatch: {source_hash}"
        )
    return SourceArtifact(
        source=_text(metadata.get("source")) or "official_lifecycle_exception",
        source_url=source_url,
        retrieved_at=_text(metadata.get("retrieved_at")) or REVIEWED_AT,
        content=content,
        content_type=_text(metadata.get("content_type")) or "text/plain",
    )


def _price_histories(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    value = prices.loc[:, ["security_id", "session", "close"]].copy()
    value["session"] = pd.to_datetime(value["session"], errors="coerce")
    value["close"] = pd.to_numeric(value["close"], errors="coerce")
    value = value.dropna(subset=["session", "close"])
    return {
        str(security_id): group.sort_values("session").reset_index(drop=True)
        for security_id, group in value.groupby(value["security_id"].astype(str))
    }


def _xnys_sessions(start: str, end: str) -> tuple[str, ...]:
    values = xcals.get_calendar("XNYS").sessions_in_range(start, end)
    return tuple(
        pd.Timestamp(value).tz_localize(None).date().isoformat()
        for value in values
    )


def _release_archive_content(
    repository: LocalDatasetRepository,
    source_archive: pd.DataFrame,
    *,
    source_url: str,
    source_hash: str,
    content_bytes: int,
    source: str = "",
) -> bytes:
    required = {"archive_id", "object_path", "source_url", "source_hash"}
    missing = sorted(required - set(source_archive.columns))
    if missing:
        raise RuntimeError(
            f"Current-release source_archive lacks exact provenance columns: {missing}"
        )
    wanted_hash = _text(source_hash).lower()
    rows = source_archive.loc[
        source_archive["archive_id"].astype(str).str.lower().eq(wanted_hash)
        & source_archive["source_hash"].astype(str).str.lower().eq(wanted_hash)
        & source_archive["source_url"].astype(str).eq(source_url)
    ]
    if source:
        if "source" not in source_archive.columns:
            raise RuntimeError("Current-release source_archive lacks source provenance.")
        rows = rows.loc[rows["source"].astype(str).eq(source)]
    if len(rows) != 1:
        raise RuntimeError(
            "Exact current-release archive binding is missing or ambiguous: "
            f"url={source_url}, hash={wanted_hash}, matches={len(rows)}"
        )
    object_path = Path(_text(rows.iloc[0].get("object_path")))
    if object_path.is_absolute() or not object_path.parts:
        raise RuntimeError(f"Current-release archive object path is unsafe: {object_path}")
    root = Path(repository.root).resolve()
    path = (root / object_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"Current-release archive object escapes repository root: {object_path}"
        ) from exc
    if not path.is_file():
        raise RuntimeError(f"Current-release archive object is missing: {object_path}")
    encoded = path.read_bytes()
    try:
        content = gzip.decompress(encoded) if path.suffix.lower() == ".gz" else encoded
    except Exception as exc:
        raise RuntimeError(
            f"Current-release archive object is not valid gzip: {object_path}"
        ) from exc
    observed_hash = sha256_bytes(content)
    if observed_hash != wanted_hash or len(content) != int(content_bytes):
        raise RuntimeError(
            "Current-release archive bytes differ from the reviewed binding: "
            f"hash={observed_hash}/{wanted_hash}, bytes={len(content)}/{content_bytes}"
        )
    return content


def _validate_identity_bound_terminal_archive(
    candidate: LifecycleCandidate,
    repository: LocalDatasetRepository | None,
    source_archive: pd.DataFrame | None,
) -> None:
    expected = IDENTITY_BOUND_TERMINAL_EVENTS.get(
        _key(candidate.security_id, candidate.last_price_date)
    )
    if expected is None or not expected.get("source_hash"):
        return
    if repository is None or source_archive is None:
        raise RuntimeError(
            "Identity-bound terminal event requires its exact current-release archive: "
            f"{candidate.security_id}/{candidate.last_price_date}"
        )
    _release_archive_content(
        repository,
        source_archive,
        source_url=str(expected["source_url"]),
        source_hash=str(expected["source_hash"]),
        content_bytes=int(expected["source_content_bytes"]),
    )


def _parse_external_price_csv(
    content: bytes,
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    if content.lstrip().startswith((b"<", b"<!")):
        raise RuntimeError("Pinned external price archive contains HTML.")
    try:
        raw = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise RuntimeError("Pinned external price archive is unreadable.") from exc
    expected_columns = ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"]
    if list(raw.columns) != expected_columns or len(raw) != int(
        spec["external_raw_rows"]
    ):
        raise RuntimeError("Pinned external price schema or raw row count changed.")
    sessions = pd.to_datetime(raw["Date"], format="%Y-%m-%d", errors="coerce")
    if sessions.isna().any():
        raise RuntimeError("Pinned external prices contain invalid sessions.")
    raw["session"] = sessions.dt.date.astype(str)
    if raw["session"].duplicated().any() or not raw["session"].is_monotonic_increasing:
        raise RuntimeError("Pinned external price sessions are not unique and sorted.")
    numeric_columns = ("Open", "High", "Low", "Close", "Volume", "OpenInt")
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    numeric = raw.loc[:, numeric_columns]
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all()
    coherent = (
        numeric[["Open", "High", "Low", "Close"]].gt(0).all(axis=1)
        & numeric["Volume"].ge(0)
        & numeric["OpenInt"].ge(0)
        & numeric["High"].ge(numeric[["Open", "Low", "Close"]].max(axis=1))
        & numeric["Low"].le(numeric[["Open", "High", "Close"]].min(axis=1))
    )
    if not finite or not bool(coherent.all()):
        raise RuntimeError("Pinned external prices contain invalid OHLCV values.")
    overlap = raw.loc[
        raw["session"].ge(str(spec["overlap_start"]))
        & raw["session"].le(str(spec["overlap_end"]))
    ].copy()
    expected = _xnys_sessions(str(spec["overlap_start"]), str(spec["overlap_end"]))
    if (
        len(expected) != int(spec["overlap_sessions"])
        or tuple(overlap["session"].astype(str)) != expected
    ):
        raise RuntimeError("Pinned external price overlap calendar changed.")
    return overlap.rename(columns={"Close": "close"}).loc[:, ["session", "close"]]


def _validate_reviewed_cross_basis_terminal_prices(
    candidate: LifecycleCandidate,
    successor_security_id: str,
    repository: LocalDatasetRepository | None,
    prices: pd.DataFrame | None,
    source_archive: pd.DataFrame | None,
    *,
    spec: Mapping[str, Any] | None = None,
) -> bool:
    reviewed = dict(
        spec
        or REVIEWED_CROSS_BASIS_TERMINAL_PRICE_PROVENANCE.get(
            _key(candidate.security_id, candidate.last_price_date)
        )
        or {}
    )
    if not reviewed:
        return False
    if repository is None or prices is None or source_archive is None:
        raise RuntimeError(
            f"Reviewed cross-basis terminal prices are unavailable for {candidate.symbol}."
        )
    if (
        candidate.symbol.upper() != str(reviewed["symbol"]).upper()
        or successor_security_id != str(reviewed["successor_security_id"])
    ):
        raise RuntimeError(
            f"Reviewed cross-basis price binding has the wrong identity for {candidate.symbol}."
        )
    primary = prices.loc[
        prices["security_id"].astype(str).eq(candidate.security_id)
    ].copy()
    primary["session"] = pd.to_datetime(primary["session"], errors="coerce")
    if primary["session"].isna().any():
        raise RuntimeError(f"Stored primary prices have invalid sessions for {candidate.symbol}.")
    primary = primary.sort_values("session", kind="stable").reset_index(drop=True)
    primary_sessions = tuple(primary["session"].dt.date.astype(str))
    expected_primary = _xnys_sessions(
        str(reviewed["active_from"]), str(reviewed["active_to"])
    )
    required_columns = {
        "open", "high", "low", "close", "volume", "currency", "source",
        "source_url", "source_hash",
    }
    if not required_columns.issubset(primary.columns):
        raise RuntimeError(f"Stored primary provenance is incomplete for {candidate.symbol}.")
    if (
        len(expected_primary) != int(reviewed["primary_sessions"])
        or primary_sessions != expected_primary
        or set(primary["currency"].astype(str).str.upper()) != {"USD"}
        or set(primary["source"].astype(str)) != {str(reviewed["primary_source"])}
        or set(primary["source_url"].astype(str))
        != {str(reviewed["primary_source_url"])}
        or set(primary["source_hash"].astype(str).str.lower())
        != {str(reviewed["primary_source_hash"]).lower()}
    ):
        raise RuntimeError(f"Stored primary coverage/provenance changed for {candidate.symbol}.")
    primary_payload = _release_archive_content(
        repository,
        source_archive,
        source_url=str(reviewed["primary_source_url"]),
        source_hash=str(reviewed["primary_source_hash"]),
        content_bytes=int(reviewed["primary_content_bytes"]),
        source=str(reviewed["primary_source"]),
    )
    parsed_primary = parse_yahoo_chart_json(
        primary_payload, str(reviewed["symbol"])
    ).bars
    if len(parsed_primary) != int(reviewed["primary_archive_sessions"]):
        raise RuntimeError(
            f"Archived primary raw session count changed for {candidate.symbol}."
        )
    parsed_primary = parsed_primary.loc[
        pd.to_datetime(parsed_primary["session"])
        .dt.date.astype(str)
        .ge(str(reviewed["active_from"]))
        & pd.to_datetime(parsed_primary["session"])
        .dt.date.astype(str)
        .le(str(reviewed["active_to"]))
    ].reset_index(drop=True)
    parsed_sessions = tuple(
        pd.to_datetime(parsed_primary["session"]).dt.date.astype(str)
    )
    if parsed_sessions != expected_primary:
        raise RuntimeError(f"Archived primary calendar changed for {candidate.symbol}.")
    for column in ("open", "high", "low", "close", "volume"):
        stored = pd.to_numeric(primary[column], errors="coerce").to_numpy(dtype=float)
        archived = pd.to_numeric(
            parsed_primary[column], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.array_equal(stored, archived):
            raise RuntimeError(
                f"Stored primary {column} differs from archived bytes for {candidate.symbol}."
            )
    external_payload = _release_archive_content(
        repository,
        source_archive,
        source_url=str(reviewed["external_source_url"]),
        source_hash=str(reviewed["external_source_hash"]),
        content_bytes=int(reviewed["external_content_bytes"]),
        source=str(reviewed["external_source"]),
    )
    external = _parse_external_price_csv(external_payload, reviewed)
    joined = primary.assign(session=primary_sessions).loc[:, ["session", "close"]].merge(
        external,
        on="session",
        suffixes=("_primary", "_external"),
        validate="one_to_one",
    ).sort_values("session", kind="stable")
    expected_overlap = _xnys_sessions(
        str(reviewed["overlap_start"]), str(reviewed["overlap_end"])
    )
    primary_close = pd.to_numeric(joined["close_primary"], errors="coerce")
    external_close = pd.to_numeric(joined["close_external"], errors="coerce")
    ratio = primary_close / external_close
    scale = float(ratio.median())
    normalized_error = (ratio / scale - 1.0).abs()
    correlation = float(primary_close.pct_change().corr(external_close.pct_change()))
    p99 = float(normalized_error.quantile(0.99))
    tail = tuple(value for value in primary_sessions if value > str(reviewed["overlap_end"]))
    if not (
        tuple(joined["session"].astype(str)) == expected_overlap
        and len(joined) == int(reviewed["overlap_sessions"])
        and len(tail) == int(reviewed["uncrosschecked_tail_sessions"])
        and math.isfinite(scale)
        and scale > 0
        and math.isfinite(correlation)
        and correlation >= float(reviewed["minimum_return_correlation"])
        and math.isfinite(p99)
        and p99 <= float(reviewed["maximum_p99_scaled_close_error"])
    ):
        raise RuntimeError(f"Pinned external price overlap failed for {candidate.symbol}.")
    successor = prices.loc[
        prices["security_id"].astype(str).eq(successor_security_id)
    ].copy()
    successor["session"] = pd.to_datetime(successor["session"], errors="coerce")
    successor = successor.sort_values("session", kind="stable").reset_index(drop=True)
    if successor.empty or successor["session"].isna().any():
        raise RuntimeError(f"Successor boundary prices are missing for {candidate.symbol}.")
    if (
        successor.iloc[0]["session"].date().isoformat()
        != str(reviewed["successor_first_session"])
        or len(successor) != int(reviewed["successor_sessions"])
        or set(successor["currency"].astype(str).str.upper()) != {"USD"}
        or set(successor["source"].astype(str)) != {str(reviewed["successor_source"])}
        or set(successor["source_hash"].astype(str).str.lower())
        != {str(reviewed["successor_source_hash"]).lower()}
    ):
        raise RuntimeError(f"Successor boundary provenance changed for {candidate.symbol}.")
    successor_payload = _release_archive_content(
        repository,
        source_archive,
        source_url=str(reviewed["successor_source_url"]),
        source_hash=str(reviewed["successor_source_hash"]),
        content_bytes=int(reviewed["successor_content_bytes"]),
        source=str(reviewed["successor_source"]),
    )
    try:
        successor_raw = json.loads(successor_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Successor archive is invalid JSON for {candidate.symbol}.") from exc
    if (
        not isinstance(successor_raw, list)
        or len(successor_raw) != int(reviewed["successor_sessions"])
    ):
        raise RuntimeError(f"Successor archive row count changed for {candidate.symbol}.")
    first_raw = successor_raw[0]
    if not isinstance(first_raw, dict) or _date(first_raw.get("date")) != str(
        reviewed["successor_first_session"]
    ):
        raise RuntimeError(f"Successor archive boundary changed for {candidate.symbol}.")
    first_stored = successor.iloc[0]
    for stored_column, raw_column in (
        ("open", "open"), ("high", "high"), ("low", "low"),
        ("close", "close"), ("volume", "volume"),
    ):
        stored = pd.to_numeric(first_stored.get(stored_column), errors="coerce")
        archived = pd.to_numeric(first_raw.get(raw_column), errors="coerce")
        if pd.isna(stored) or pd.isna(archived) or float(stored) != float(archived):
            raise RuntimeError(
                f"Successor boundary {stored_column} changed for {candidate.symbol}."
            )
    return True


def _nearest_close(
    histories: Mapping[str, pd.DataFrame],
    security_id: str,
    effective: pd.Timestamp,
    *,
    after: bool,
) -> float | None:
    frame = histories.get(str(security_id))
    if frame is None or frame.empty:
        return None
    eligible = frame.loc[
        frame["session"].ge(effective) if after else frame["session"].le(effective)
    ]
    if eligible.empty:
        return None
    distances = (
        eligible["session"] - effective if after else effective - eligible["session"]
    ).dt.days
    index = distances.idxmin()
    if int(distances.loc[index]) > 10:
        return None
    return float(eligible.loc[index, "close"])


def _crosscheck_event(
    candidate: LifecycleCandidate,
    event: Mapping[str, Any],
    successor_security_id: str,
    histories: Mapping[str, pd.DataFrame],
    *,
    repository: LocalDatasetRepository | None = None,
    prices: pd.DataFrame | None = None,
    source_archive: pd.DataFrame | None = None,
) -> None:
    action_type = _text(event.get("action_type")).lower()
    effective = pd.Timestamp(_date(event.get("effective_date")))
    last = pd.Timestamp(candidate.last_price_date)
    terminal_gap = abs((effective - last).days)
    remove_gaps = [
        abs((effective - pd.Timestamp(value)).days)
        for value in candidate.index_remove_dates
    ]
    date_passed = terminal_gap <= 7 or (remove_gaps and min(remove_gaps) <= 7)
    if action_type == "delisting" and event.get("cash_amount") is not None and effective >= last:
        date_passed = True
    if not date_passed:
        raise RuntimeError(f"Lifecycle event date crosscheck failed for {candidate.symbol}.")
    _validate_identity_bound_terminal_archive(candidate, repository, source_archive)
    if action_type == "delisting":
        amount = pd.to_numeric(event.get("cash_amount"), errors="coerce")
        if pd.isna(amount) or float(amount) < 0:
            raise RuntimeError(f"Delisting recovery is unverified for {candidate.symbol}.")
        return
    old_close = _nearest_close(histories, candidate.security_id, effective, after=False)
    if old_close is None:
        raise RuntimeError(f"No terminal price for economic crosscheck: {candidate.symbol}")
    if action_type == "cash_merger":
        implied = pd.to_numeric(event.get("cash_amount"), errors="coerce")
        if pd.isna(implied) or float(implied) <= 0:
            raise RuntimeError(f"Cash merger terms are invalid for {candidate.symbol}.")
        implied = float(implied)
    elif action_type in {"stock_merger", "ticker_change"}:
        if _validate_reviewed_cross_basis_terminal_prices(
            candidate,
            successor_security_id,
            repository,
            prices,
            source_archive,
        ):
            return
        successor_close = _nearest_close(histories, successor_security_id, effective, after=True)
        if successor_close is None:
            raise RuntimeError(f"No successor price for economic crosscheck: {candidate.symbol}")
        if action_type == "ticker_change":
            implied = successor_close
        else:
            ratio = pd.to_numeric(event.get("ratio"), errors="coerce")
            if pd.isna(ratio) or float(ratio) <= 0:
                raise RuntimeError(f"Stock merger ratio is invalid for {candidate.symbol}.")
            implied = float(ratio) * successor_close + float(event.get("cash_amount") or 0.0)
    else:
        raise RuntimeError(f"Unsupported applied lifecycle action: {action_type}")
    deviation = abs(old_close - implied) / max(abs(old_close), abs(implied), 1e-12)
    if deviation > 0.20:
        raise RuntimeError(
            f"Lifecycle economic crosscheck failed for {candidate.symbol}: {deviation:.6f}"
        )


def _successor_for_event(
    event: Mapping[str, Any],
    master: pd.DataFrame,
    history: pd.DataFrame,
) -> str:
    action_type = _text(event.get("action_type")).lower()
    if action_type not in {"stock_merger", "ticker_change"}:
        return ""
    symbol = _text(event.get("new_symbol")).upper()
    resolved = resolve_new_security_id(
        master,
        new_symbol=symbol,
        effective_date=_date(event.get("effective_date")),
        symbol_history=history,
    )
    supplied = _text(event.get("successor_security_id"))
    if not resolved or (supplied and supplied != resolved):
        raise RuntimeError(
            f"Lifecycle successor is unresolved or inconsistent: {symbol}/{supplied}"
        )
    return resolved


def _same_optional_number(actual: Any, expected: float | None) -> bool:
    if expected is None:
        return actual is None or not _text(actual)
    parsed = pd.to_numeric(actual, errors="coerce")
    return bool(not pd.isna(parsed) and abs(float(parsed) - expected) <= 1e-9)


def _canonical_metadata_text(value: Any) -> str:
    if isinstance(value, Mapping):
        parsed = dict(value)
    else:
        raw = _text(value)
        if not raw:
            raise RuntimeError("Exact repaired lifecycle action lacks metadata.")
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Exact repaired lifecycle action metadata is not JSON."
            ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Exact repaired lifecycle action metadata is not an object.")
    return json.dumps(
        parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _metadata_sha256(value: Any) -> str:
    return sha256_bytes(_canonical_metadata_text(value).encode())


def _archive_binding_row(
    source_archive: pd.DataFrame,
    *,
    source_url: str,
    source_hash: str,
    source: str,
    require_archive_id_hash: bool = True,
) -> Mapping[str, Any]:
    required = {"archive_id", "object_path", "source_url", "source_hash", "source"}
    missing = sorted(required - set(source_archive.columns))
    if missing:
        raise RuntimeError(
            "Exact repaired lifecycle archive lacks columns: " + ", ".join(missing)
        )
    rows = source_archive.loc[
        source_archive["source_hash"].astype(str).str.lower().eq(source_hash.lower())
        & source_archive["source_url"].astype(str).eq(source_url)
        & source_archive["source"].astype(str).eq(source)
    ]
    if require_archive_id_hash:
        rows = rows.loc[
            rows["archive_id"].astype(str).str.lower().eq(source_hash.lower())
        ]
    if len(rows) != 1:
        raise RuntimeError(
            "Exact repaired lifecycle archive binding is missing or ambiguous: "
            f"source={source}, url={source_url}, hash={source_hash}, matches={len(rows)}"
        )
    return rows.iloc[0].to_dict()


def _archive_pair_content(
    repository: LocalDatasetRepository,
    source_archive: pd.DataFrame,
    *,
    source_url: str,
    source_hash: str,
    source: str,
) -> tuple[bytes, Mapping[str, Any]]:
    row = _archive_binding_row(
        source_archive,
        source_url=source_url,
        source_hash=source_hash,
        source=source,
        require_archive_id_hash=False,
    )
    object_path = Path(_text(row.get("object_path")))
    if object_path.is_absolute() or not object_path.parts:
        raise RuntimeError("Exact repaired lifecycle archive object path is unsafe.")
    root = Path(repository.root).resolve()
    path = (root / object_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            "Exact repaired lifecycle archive object escapes repository root."
        ) from exc
    if not path.is_file():
        raise RuntimeError(
            f"Exact repaired lifecycle archive object is missing: {object_path}"
        )
    encoded = path.read_bytes()
    try:
        content = gzip.decompress(encoded) if path.suffix.lower() == ".gz" else encoded
    except Exception as exc:
        raise RuntimeError(
            f"Exact repaired lifecycle archive object is invalid gzip: {object_path}"
        ) from exc
    if sha256_bytes(content) != source_hash.lower():
        raise RuntimeError("Exact repaired lifecycle archive payload hash changed.")
    return content, row


def _require_exact_repaired_action(
    actions: pd.DataFrame,
    *,
    event_id: str,
    expected_text: Mapping[str, Any],
    cash_amount: float | None,
    ratio: float | None,
    metadata: Mapping[str, Any] | None = None,
    metadata_sha256: str = "",
) -> dict[str, Any]:
    rows = actions.loc[actions["event_id"].astype(str).eq(event_id)]
    if len(rows) != 1:
        raise RuntimeError(
            f"Exact repaired lifecycle action is missing or duplicated: {event_id}"
        )
    row = rows.iloc[0].to_dict()
    semantic = actions.loc[
        actions["security_id"].astype(str).eq(_text(expected_text["security_id"]))
        & actions["action_type"]
        .astype(str)
        .str.lower()
        .eq(_text(expected_text["action_type"]).lower())
        & pd.to_datetime(actions["effective_date"], errors="coerce")
        .dt.date.astype(str)
        .eq(_date(expected_text["effective_date"]))
    ]
    if len(semantic) != 1:
        raise RuntimeError(
            f"Exact repaired lifecycle action semantic key is ambiguous: {event_id}"
        )
    date_fields = {
        "effective_date",
        "ex_date",
        "announcement_date",
        "record_date",
        "payment_date",
    }
    for field, expected in expected_text.items():
        actual_text = _text(row.get(field))
        expected_text_value = _text(expected)
        actual = (
            _date(actual_text)
            if field in date_fields and actual_text
            else actual_text
        )
        expected_value = (
            _date(expected_text_value)
            if field in date_fields and expected_text_value
            else expected_text_value
        )
        if actual != expected_value:
            raise RuntimeError(
                f"Exact repaired lifecycle action field changed: {event_id}/{field}"
            )
    if not _same_optional_number(row.get("cash_amount"), cash_amount) or not (
        _same_optional_number(row.get("ratio"), ratio)
    ):
        raise RuntimeError(
            f"Exact repaired lifecycle action economics changed: {event_id}"
        )
    official = row.get("official")
    if not isinstance(official, (bool, np.bool_)) or not bool(official):
        raise RuntimeError(
            f"Exact repaired lifecycle action is not strictly official: {event_id}"
        )
    canonical = _canonical_metadata_text(row.get("metadata"))
    if metadata is not None:
        expected_metadata = json.dumps(
            dict(metadata), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if canonical != expected_metadata:
            raise RuntimeError(
                f"Exact repaired lifecycle action metadata changed: {event_id}"
            )
    if metadata_sha256 and sha256_bytes(canonical.encode()) != metadata_sha256:
        raise RuntimeError(
            f"Exact repaired lifecycle action metadata hash changed: {event_id}"
        )
    return row


def _require_exact_repaired_resolution(
    resolutions: pd.DataFrame,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    rows = resolutions.loc[
        resolutions["security_id"].astype(str).eq(_text(expected["security_id"]))
    ]
    if len(rows) != 1:
        raise RuntimeError("Exact repaired lifecycle resolution is missing or duplicated.")
    row = rows.iloc[0].to_dict()
    for field, value in expected.items():
        actual = (
            _date(row.get(field))
            if field == "last_price_date"
            else _text(row.get(field))
        )
        wanted = (
            _date(value) if field == "last_price_date" else _text(value)
        )
        if actual != wanted:
            raise RuntimeError(
                f"Exact repaired lifecycle resolution field changed: {field}"
            )
    return row


def _canonical_lifecycle_resolution_row(
    row: Mapping[str, Any],
) -> dict[str, str]:
    columns = dataset_spec("lifecycle_resolutions").required_columns
    if set(row) != set(columns):
        raise RuntimeError(
            "Exact short-terminal lifecycle resolution schema changed."
        )
    return {
        column: (
            _date(row.get(column))
            if column == "last_price_date" and _text(row.get(column))
            else _text(row.get(column))
        )
        for column in columns
    }


def _lifecycle_resolution_row_sha256(row: Mapping[str, Any]) -> str:
    canonical = _canonical_lifecycle_resolution_row(row)
    return sha256_bytes(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


def _preserve_exact_short_terminal_reviewed_resolution(
    candidate: LifecycleCandidate,
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any] | None:
    spec = EXACT_SHORT_TERMINAL_REVIEWED_RESOLUTIONS.get(
        _key(candidate.security_id, candidate.last_price_date)
    )
    if spec is None:
        return None
    expected = spec["resolution"]
    expected_candidate_id = lifecycle_candidate_id(
        candidate.security_id, candidate.last_price_date
    )
    if not (
        candidate.security_id == expected["security_id"]
        and candidate.symbol.upper() == expected["symbol"]
        and _date(candidate.last_price_date) == expected["last_price_date"]
        and expected_candidate_id == expected["candidate_id"]
        and expected["reviewed_by"] == "short_terminal_boundary_repair_v1"
        and expected["reviewed_at"] == "2026-07-19T00:00:00Z"
    ):
        raise RuntimeError(
            "Exact short-terminal reviewed candidate binding changed."
        )
    resolutions = frames.get("lifecycle_resolutions")
    if resolutions is None:
        raise RuntimeError(
            "Exact short-terminal reviewed lifecycle resolutions are missing."
        )
    row = _require_exact_repaired_resolution(
        resolutions,
        expected=expected,
    )
    expected_hash = _text(spec["row_sha256"]).lower()
    if (
        _lifecycle_resolution_row_sha256(expected) != expected_hash
        or _lifecycle_resolution_row_sha256(row) != expected_hash
    ):
        raise RuntimeError(
            "Exact short-terminal reviewed lifecycle resolution hash changed."
        )
    return row


def _restore_exact_short_terminal_reviewed_resolution(
    prior: Mapping[str, Any],
    generated: Mapping[str, Any],
) -> dict[str, Any]:
    prior_row = _canonical_lifecycle_resolution_row(prior)
    generated_row = _canonical_lifecycle_resolution_row(generated)
    intentionally_distinct_provenance = {
        "reviewed_by",
        "reviewed_at",
        "source",
        "retrieved_at",
    }
    mismatches = sorted(
        column
        for column in prior_row
        if column not in intentionally_distinct_provenance
        and prior_row[column] != generated_row[column]
    )
    if mismatches:
        raise RuntimeError(
            "Generic lifecycle validation disagrees with the exact "
            "short-terminal reviewed resolution: "
            + ", ".join(mismatches)
        )
    if not (
        generated_row["reviewed_by"] == REVIEWED_BY
        and generated_row["reviewed_at"] == REVIEWED_AT
        and generated_row["source"] == "lifecycle_finalizer"
        and bool(generated_row["retrieved_at"])
    ):
        raise RuntimeError(
            "Generic lifecycle resolution provenance is not finalizer-owned."
        )
    return dict(prior)


def _frame_contains_exact_value(
    frame: pd.DataFrame | None,
    column: str,
    values: Iterable[str],
) -> bool:
    if frame is None or column not in frame.columns:
        return False
    wanted = {str(value) for value in values}
    return bool(frame[column].astype(str).isin(wanted).any())


def _require_exact_identity_row(
    frame: pd.DataFrame,
    *,
    security_id: str,
    expected: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    if "security_id" not in frame.columns:
        raise RuntimeError(f"Exact {label} identity dataset lacks security_id.")
    rows = frame.loc[frame["security_id"].astype(str).eq(security_id)]
    if len(rows) != 1:
        raise RuntimeError(
            f"Exact {label} identity row is missing or duplicated: matches={len(rows)}"
        )
    row = rows.iloc[0].to_dict()
    date_fields = {"active_from", "active_to", "effective_from", "effective_to"}
    for field, wanted in expected.items():
        actual_text = _text(row.get(field))
        wanted_text = _text(wanted)
        actual = (
            _date(actual_text) if field in date_fields and actual_text else actual_text
        )
        target = (
            _date(wanted_text) if field in date_fields and wanted_text else wanted_text
        )
        if actual != target:
            raise RuntimeError(f"Exact {label} identity field changed: {field}")
    return row


def _require_exact_history_row(
    history: pd.DataFrame,
    *,
    security_id: str,
    symbol: str,
    expected: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    required = {"security_id", "symbol"}
    missing = sorted(required - set(history.columns))
    if missing:
        raise RuntimeError(f"Exact {label} history lacks columns: {missing}")
    rows = history.loc[
        history["security_id"].astype(str).eq(security_id)
        & history["symbol"].astype(str).str.upper().eq(symbol.upper())
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"Exact {label} symbol-history row is missing or duplicated: "
            f"symbol={symbol}, matches={len(rows)}"
        )
    row = rows.iloc[0].to_dict()
    for field, wanted in expected.items():
        actual_text = _text(row.get(field))
        wanted_text = _text(wanted)
        actual = (
            _date(actual_text)
            if field in {"effective_from", "effective_to"} and actual_text
            else actual_text
        )
        target = (
            _date(wanted_text)
            if field in {"effective_from", "effective_to"} and wanted_text
            else wanted_text
        )
        if actual != target:
            raise RuntimeError(f"Exact {label} symbol-history field changed: {field}")
    return row


def _exact_security_session_keys(
    frame: pd.DataFrame,
    security_ids: set[str],
    *,
    label: str,
) -> set[tuple[str, str]]:
    required = {"security_id", "session"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Exact {label} lacks key columns: {missing}")
    rows = frame.loc[frame["security_id"].astype(str).isin(security_ids)]
    keys = [
        (_text(row.security_id), _date(row.session))
        for row in rows.itertuples(index=False)
    ]
    if any(not session for _security_id, session in keys) or len(keys) != len(set(keys)):
        raise RuntimeError(f"Exact {label} contains invalid or duplicated keys.")
    return set(keys)


def _require_exact_frc_occ_action(actions: pd.DataFrame) -> dict[str, Any]:
    """Attest the raw OCC 52352 action; the legacy JSON cannot authorize it."""

    return _require_exact_repaired_action(
        actions,
        event_id=FRC_EXACT_EVENT_ID,
        expected_text={
            "event_id": FRC_EXACT_EVENT_ID,
            "security_id": FRC_EXACT_SECURITY_ID,
            "action_type": "ticker_change",
            "effective_date": FRC_EXACT_TRANSITION,
            "ex_date": FRC_EXACT_TRANSITION,
            "announcement_date": FRC_EXACT_OLD_LAST,
            "record_date": "",
            "payment_date": "",
            "currency": "USD",
            "new_security_id": FRC_EXACT_SECURITY_ID,
            "new_symbol": "FRCB",
            "source_url": FRC_EXACT_OCC_URL,
            "source_kind": "official_crosscheck",
            "source": "occ_information_memo",
            "retrieved_at": FRC_EXACT_OCC_PDF_RETRIEVED_AT,
            "source_hash": FRC_EXACT_OCC_PDF_SHA256,
        },
        cash_amount=None,
        ratio=None,
        metadata_sha256=FRC_EXACT_OCC_ACTION_METADATA_SHA256,
    )


def _preserved_exact_frc_para_repairs(
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
    candidates: Iterable[LifecycleCandidate],
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    """Validate the complete FRC/PARA repair and retain only PARA's resolution.

    The old FRC terminal row is accepted on input for the first finalization,
    but it is intentionally omitted from the returned mapping because FRCB's
    continued prices mean that FRC is no longer a candidate.  A subsequent
    finalization may therefore have no FRC resolution row at all.  Any partial
    or changed repair structure fails closed.
    """

    actions = frames.get("corporate_actions")
    resolutions = frames.get("lifecycle_resolutions")
    master = frames.get("security_master")
    archive = frames.get("source_archive")
    relevant = (
        _frame_contains_exact_value(
            actions,
            "event_id",
            {FRC_EXACT_EVENT_ID, PARA_EXACT_EVENT_ID},
        )
        or _frame_contains_exact_value(
            resolutions,
            "event_id",
            {FRC_EXACT_EVENT_ID, PARA_EXACT_EVENT_ID},
        )
        or _frame_contains_exact_value(
            archive,
            "source_hash",
            {
                FRC_EXACT_OCC_SHA256,
                FRC_EXACT_OCC_PDF_SHA256,
                FRC_EXACT_RAW_EOD_SHA256,
                FRC_EXACT_CORRECTION_SHA256,
                PARA_EXACT_SEC_SHA256,
            },
        )
        or (
            master is not None
            and {"security_id", "primary_symbol"}.issubset(master.columns)
            and bool(
                (
                    master["security_id"].astype(str).eq(FRC_EXACT_SECURITY_ID)
                    & master["primary_symbol"].astype(str).str.upper().eq("FRCB")
                ).any()
            )
        )
        or FRC_EXACT_WARNING in tuple(getattr(release, "warnings", ()))
    )
    if not relevant:
        return {}, ()

    required_frames = {
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "corporate_actions",
        "adjustment_factors",
        "source_archive",
        "lifecycle_resolutions",
    }
    missing_frames = sorted(required_frames - set(frames))
    if missing_frames:
        raise RuntimeError(
            "Exact FRC/PARA repair is partial; missing datasets: "
            + ", ".join(missing_frames)
        )
    if _date(getattr(release, "completed_session", "")) != FRC_EXACT_PRICE_END:
        raise RuntimeError("Exact FRC/PARA repair completed-session binding changed.")
    if FRC_EXACT_WARNING not in tuple(getattr(release, "warnings", ())):
        raise RuntimeError("Exact FRCB envelope-correction release warning is missing.")

    candidate_values = tuple(candidates)
    if any(item.security_id == FRC_EXACT_SECURITY_ID for item in candidate_values):
        raise RuntimeError("FRC remained a stale lifecycle candidate after FRCB continuity.")
    para_candidates = [
        item for item in candidate_values if item.security_id == PARA_EXACT_SECURITY_ID
    ]
    if len(para_candidates) != 1:
        raise RuntimeError("Exact PARA lifecycle candidate is missing or duplicated.")
    para_candidate = para_candidates[0]
    if (
        para_candidate.symbol.upper() != "PARA"
        or _date(para_candidate.last_price_date) != PARA_EXACT_LAST
        or lifecycle_candidate_id(
            para_candidate.security_id, para_candidate.last_price_date
        )
        != lifecycle_candidate_id(PARA_EXACT_SECURITY_ID, PARA_EXACT_LAST)
    ):
        raise RuntimeError("Exact PARA lifecycle candidate identity/date changed.")
    if any(item.security_id == PSKY_EXACT_SECURITY_ID for item in candidate_values):
        raise RuntimeError("PSKY incorrectly remained a pre-transition lifecycle candidate.")

    archive = frames["source_archive"]
    occ_content, occ_archive_row = _archive_pair_content(
        repository,
        archive,
        source_url=FRC_EXACT_OCC_URL,
        source_hash=FRC_EXACT_OCC_SHA256,
        source="occ_reviewed_memo_extraction",
    )
    occ_pdf_content, occ_pdf_archive_row = _archive_pair_content(
        repository,
        archive,
        source_url=FRC_EXACT_OCC_URL,
        source_hash=FRC_EXACT_OCC_PDF_SHA256,
        source="occ_information_memo",
    )
    _archive_pair_content(
        repository,
        archive,
        source_url=FRC_EXACT_FDIC_URL,
        source_hash=FRC_EXACT_FDIC_SHA256,
        source="fdic_failed_bank_receivership",
    )
    raw_eod_content, raw_eod_archive_row = _archive_pair_content(
        repository,
        archive,
        source_url=FRC_EXACT_RAW_EOD_URL,
        source_hash=FRC_EXACT_RAW_EOD_SHA256,
        source="eodhd_eod",
    )
    correction_content, correction_archive_row = _archive_pair_content(
        repository,
        archive,
        source_url=FRC_EXACT_RAW_EOD_URL,
        source_hash=FRC_EXACT_CORRECTION_SHA256,
        source="frcb_reviewed_ohlcv_envelope_correction",
    )
    _sec_content, sec_archive_row = _archive_pair_content(
        repository,
        archive,
        source_url=PARA_EXACT_SEC_URL,
        source_hash=PARA_EXACT_SEC_SHA256,
        source="sec_edgar_filing",
    )
    try:
        occ_value = json.loads(occ_content)
        correction_value = json.loads(correction_content)
        raw_eod_value = json.loads(raw_eod_content)
    except (UnicodeDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError("Exact FRC/PARA archive payload is not valid JSON.") from exc
    if occ_value != FRC_EXACT_OCC_EXTRACTION:
        raise RuntimeError("Exact FRC OCC reviewed extraction changed.")
    if (
        len(occ_pdf_content) != FRC_EXACT_OCC_PDF_BYTES
        or not occ_pdf_content.startswith(b"%PDF-")
        or b"%%EOF" not in occ_pdf_content[-4096:]
        or _text(occ_pdf_archive_row.get("archive_id")).lower()
        != FRC_EXACT_OCC_PDF_SHA256
        or _text(occ_pdf_archive_row.get("dataset")) != "occ_information_memo"
        or _text(occ_pdf_archive_row.get("content_type")) != "application/pdf"
        or _text(occ_pdf_archive_row.get("object_path"))
        != FRC_EXACT_OCC_PDF_OBJECT_PATH
        or _date(occ_pdf_archive_row.get("effective_date")) != "2026-07-15"
        or _text(occ_pdf_archive_row.get("retrieved_at"))
        != FRC_EXACT_OCC_PDF_RETRIEVED_AT
    ):
        raise RuntimeError("Exact FRC OCC 52352 raw-PDF archive binding changed.")
    if (
        len(occ_content) != 516
        or _text(occ_archive_row.get("archive_id")).lower()
        != FRC_EXACT_OCC_LEGACY_ARCHIVE_ID
        or _text(occ_archive_row.get("dataset"))
        != "occ_reviewed_memo_extraction"
        or _text(occ_archive_row.get("content_type")) != "application/json"
        or _text(occ_archive_row.get("object_path"))
        != f"archives/2026-07-15/{FRC_EXACT_OCC_SHA256}.json.gz"
        or _date(occ_archive_row.get("effective_date")) != "2026-07-15"
        or _text(occ_archive_row.get("retrieved_at")) != "2026-07-18T00:00:00Z"
    ):
        raise RuntimeError("Exact FRC OCC legacy extraction archive binding changed.")
    if correction_value != FRC_EXACT_CORRECTION_METADATA:
        raise RuntimeError("Exact FRCB correction metadata changed.")
    if not isinstance(raw_eod_value, list) or len(raw_eod_value) != FRC_EXACT_RAW_EOD_ROWS:
        raise RuntimeError("Exact FRCB raw EOD row inventory changed.")

    actions = frames["corporate_actions"]
    _require_exact_frc_occ_action(actions)
    _require_exact_repaired_action(
        actions,
        event_id=PARA_EXACT_EVENT_ID,
        expected_text={
            "event_id": PARA_EXACT_EVENT_ID,
            "security_id": PARA_EXACT_SECURITY_ID,
            "action_type": "stock_merger",
            "effective_date": PARA_EXACT_TRANSITION,
            "ex_date": PARA_EXACT_TRANSITION,
            "announcement_date": PARA_EXACT_TRANSITION,
            "record_date": "",
            "payment_date": "",
            "currency": "USD",
            "new_security_id": PSKY_EXACT_SECURITY_ID,
            "new_symbol": "PSKY",
            "source_url": PARA_EXACT_SEC_URL,
            "source_kind": "sec_filing_default_stock_policy",
            "source": "sec_edgar_filing",
            "retrieved_at": PARA_EXACT_RETRIEVED_AT,
            "source_hash": PARA_EXACT_SEC_SHA256,
        },
        cash_amount=None,
        ratio=1.0,
        metadata={
            "backtest_policy": "no_election_default_stock",
            "cash_elector_proration_excluded": True,
        },
    )

    lifecycle_types = {"cash_merger", "stock_merger", "spinoff", "ticker_change", "delisting"}
    action_dates = pd.to_datetime(actions["effective_date"], errors="coerce")
    action_types = actions["action_type"].astype(str).str.lower()
    frc_terminal = actions.loc[
        actions["security_id"].astype(str).eq(FRC_EXACT_SECURITY_ID)
        & action_types.isin(lifecycle_types)
        & action_dates.ge(pd.Timestamp(FRC_EXACT_TRANSITION))
    ]
    para_terminal = actions.loc[
        actions["security_id"].astype(str).eq(PARA_EXACT_SECURITY_ID)
        & action_types.isin(lifecycle_types)
        & action_dates.ge(pd.Timestamp(PARA_EXACT_TRANSITION))
    ]
    psky_pre_transition = actions.loc[
        actions["security_id"].astype(str).eq(PSKY_EXACT_SECURITY_ID)
        & action_dates.lt(pd.Timestamp(PARA_EXACT_TRANSITION))
    ]
    if set(frc_terminal["event_id"].astype(str)) != {FRC_EXACT_EVENT_ID} or len(frc_terminal) != 1:
        raise RuntimeError("Exact FRC lifecycle action boundary changed.")
    if set(para_terminal["event_id"].astype(str)) != {PARA_EXACT_EVENT_ID} or len(para_terminal) != 1:
        raise RuntimeError("Exact PARA lifecycle action boundary changed.")
    if not psky_pre_transition.empty:
        raise RuntimeError("Exact PSKY pre-transition action boundary changed.")

    master = frames["security_master"]
    _require_exact_identity_row(
        master,
        security_id=FRC_EXACT_SECURITY_ID,
        label="FRC/FRCB master",
        expected={
            "primary_symbol": "FRCB",
            "provider_symbol": "FRCB.US",
            "action_provider_symbol": "FRCB.US",
            "exchange": "PINK",
            "active_to": "",
            "source": "occ_reviewed_memo_extraction",
            "source_url": FRC_EXACT_OCC_URL,
            "retrieved_at": "2026-07-18T00:00:00Z",
            "source_hash": FRC_EXACT_OCC_SHA256,
        },
    )
    _require_exact_identity_row(
        master,
        security_id=PARA_EXACT_SECURITY_ID,
        label="PARA master",
        expected={"primary_symbol": "PARA", "active_to": PARA_EXACT_TRANSITION},
    )
    _require_exact_identity_row(
        master,
        security_id=PSKY_EXACT_SECURITY_ID,
        label="PSKY master",
        expected={
            "primary_symbol": "PSKY",
            "provider_symbol": "PSKY.US",
            "action_provider_symbol": "PSKY.US",
            "exchange": "NASDAQ",
            "active_from": PARA_EXACT_TRANSITION,
            "active_to": "",
            "source": "sec_edgar_filing",
            "source_url": PARA_EXACT_SEC_URL,
            "retrieved_at": PARA_EXACT_RETRIEVED_AT,
            "source_hash": PARA_EXACT_SEC_SHA256,
        },
    )

    history = frames["symbol_history"]
    touched_history = history.loc[
        history["security_id"].astype(str).isin(
            {FRC_EXACT_SECURITY_ID, PARA_EXACT_SECURITY_ID, PSKY_EXACT_SECURITY_ID}
        )
    ]
    if len(touched_history) != 4:
        raise RuntimeError("Exact FRC/PARA symbol-history inventory changed.")
    common_occ = {
        "source": "occ_reviewed_memo_extraction",
        "source_url": FRC_EXACT_OCC_URL,
        "retrieved_at": "2026-07-18T00:00:00Z",
        "source_hash": FRC_EXACT_OCC_SHA256,
    }
    _require_exact_history_row(
        history,
        security_id=FRC_EXACT_SECURITY_ID,
        symbol="FRC",
        label="FRC",
        expected={
            "exchange": "NYSE",
            "effective_from": "2015-01-01",
            "effective_to": FRC_EXACT_OLD_LAST,
            **common_occ,
        },
    )
    _require_exact_history_row(
        history,
        security_id=FRC_EXACT_SECURITY_ID,
        symbol="FRCB",
        label="FRCB",
        expected={
            "exchange": "PINK",
            "effective_from": FRC_EXACT_TRANSITION,
            "effective_to": "",
            **common_occ,
        },
    )
    common_sec = {
        "exchange": "NASDAQ",
        "source": "sec_edgar_filing",
        "source_url": PARA_EXACT_SEC_URL,
        "retrieved_at": PARA_EXACT_RETRIEVED_AT,
        "source_hash": PARA_EXACT_SEC_SHA256,
    }
    _require_exact_history_row(
        history,
        security_id=PARA_EXACT_SECURITY_ID,
        symbol="PARA",
        label="PARA",
        expected={
            "effective_from": "2022-02-17",
            "effective_to": PARA_EXACT_TRANSITION,
            **common_sec,
        },
    )
    _require_exact_history_row(
        history,
        security_id=PSKY_EXACT_SECURITY_ID,
        symbol="PSKY",
        label="PSKY",
        expected={
            "effective_from": PARA_EXACT_TRANSITION,
            "effective_to": "",
            **common_sec,
        },
    )

    raw = pd.DataFrame(raw_eod_value)
    raw_required = {"date", "open", "high", "low", "close", "volume"}
    if not raw_required.issubset(raw.columns):
        raise RuntimeError("Exact FRCB raw EOD fields changed.")
    raw["_session"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.sort_values("_session").reset_index(drop=True)
    raw_session_text = raw["_session"].dt.date.astype(str)
    raw_sessions = tuple(raw_session_text)
    if (
        raw["_session"].isna().any()
        or len(set(raw_sessions)) != FRC_EXACT_RAW_EOD_ROWS
        or raw_sessions[0] != FRC_EXACT_TRANSITION
        or raw_sessions[-1] != FRC_EXACT_PRICE_END
    ):
        raise RuntimeError("Exact FRCB raw EOD session boundary changed.")
    bad_raw = raw.loc[raw_session_text.eq("2024-12-30")]
    if len(bad_raw) != 1:
        raise RuntimeError("Exact FRCB corrected raw session is missing or duplicated.")
    bad_values = {
        field: float(pd.to_numeric(bad_raw.iloc[0][field], errors="coerce"))
        for field in ("open", "high", "low", "close", "volume")
    }
    if bad_values != {
        "open": 0.003,
        "high": 0.006,
        "low": 0.0,
        "close": 0.004,
        "volume": 629864.0,
    }:
        raise RuntimeError("Exact FRCB corrected raw OHLCV row changed.")

    prices = frames["daily_price_raw"].copy()
    prices["_session"] = pd.to_datetime(prices["session"], errors="coerce")
    frc_prices = prices.loc[
        prices["security_id"].astype(str).eq(FRC_EXACT_SECURITY_ID)
    ].sort_values("_session")
    para_prices = prices.loc[
        prices["security_id"].astype(str).eq(PARA_EXACT_SECURITY_ID)
    ].sort_values("_session")
    psky_prices = prices.loc[
        prices["security_id"].astype(str).eq(PSKY_EXACT_SECURITY_ID)
    ].sort_values("_session")
    if (
        frc_prices.empty
        or frc_prices["_session"].isna().any()
        or frc_prices["_session"].duplicated().any()
        or _date(frc_prices.iloc[0]["_session"]) != "2015-01-02"
        or _date(frc_prices.iloc[-1]["_session"]) != FRC_EXACT_PRICE_END
    ):
        raise RuntimeError("Exact FRC/FRCB stored price boundary changed.")
    frc_tail = frc_prices.loc[
        frc_prices["_session"].ge(pd.Timestamp(FRC_EXACT_TRANSITION))
    ].reset_index(drop=True)
    if tuple(frc_tail["_session"].dt.date.astype(str)) != raw_sessions:
        raise RuntimeError("Exact FRCB stored/raw EOD session inventory changed.")
    raw_numeric = raw.loc[:, ["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    expected_tail = raw_numeric.copy()
    correction_mask = raw["_session"].dt.date.astype(str).eq("2024-12-30")
    expected_tail.loc[correction_mask, "low"] = 0.003
    stored_numeric = frc_tail.loc[
        :, ["open", "high", "low", "close", "volume"]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.allclose(
        stored_numeric.to_numpy(dtype=float),
        expected_tail.to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
        equal_nan=False,
    ):
        raise RuntimeError("Exact FRCB stored prices differ from raw plus one correction.")
    raw_retrieved_at = _text(raw_eod_archive_row.get("retrieved_at"))
    correction_retrieved_at = _text(correction_archive_row.get("retrieved_at"))
    if (
        not raw_retrieved_at
        or correction_retrieved_at != raw_retrieved_at
        or frc_tail["source"].astype(str).ne("eodhd_eod").any()
        or frc_tail["source_url"].astype(str).ne(FRC_EXACT_RAW_EOD_URL).any()
        or frc_tail["source_hash"].astype(str).ne(FRC_EXACT_RAW_EOD_SHA256).any()
        or frc_tail["retrieved_at"].astype(str).ne(raw_retrieved_at).any()
    ):
        raise RuntimeError("Exact FRCB stored price provenance changed.")
    if (
        para_prices.empty
        or psky_prices.empty
        or para_prices["_session"].isna().any()
        or psky_prices["_session"].isna().any()
        or para_prices["_session"].duplicated().any()
        or psky_prices["_session"].duplicated().any()
        or _date(para_prices.iloc[-1]["_session"]) != PARA_EXACT_LAST
        or _date(psky_prices.iloc[0]["_session"]) != PARA_EXACT_TRANSITION
        or para_prices["_session"].gt(pd.Timestamp(PARA_EXACT_LAST)).any()
        or psky_prices["_session"].lt(pd.Timestamp(PARA_EXACT_TRANSITION)).any()
    ):
        raise RuntimeError("Exact PARA/PSKY stored price boundary changed.")

    touched = {FRC_EXACT_SECURITY_ID, PARA_EXACT_SECURITY_ID, PSKY_EXACT_SECURITY_ID}
    price_keys = _exact_security_session_keys(prices, touched, label="FRC/PARA prices")
    factor_keys = _exact_security_session_keys(
        frames["adjustment_factors"], touched, label="FRC/PARA adjustment factors"
    )
    if price_keys != factor_keys:
        raise RuntimeError("Exact FRC/PARA adjustment-factor inventory changed.")

    resolutions = frames["lifecycle_resolutions"]
    frc_resolution_rows = resolutions.loc[
        resolutions["security_id"].astype(str).eq(FRC_EXACT_SECURITY_ID)
    ]
    if len(frc_resolution_rows) > 1:
        raise RuntimeError("Exact stale FRC resolution is duplicated.")
    if len(frc_resolution_rows) == 1:
        _require_exact_repaired_resolution(
            resolutions,
            expected={
                "candidate_id": lifecycle_candidate_id(
                    FRC_EXACT_SECURITY_ID, FRC_EXACT_OLD_LAST
                ),
                "security_id": FRC_EXACT_SECURITY_ID,
                "symbol": "FRC",
                "last_price_date": FRC_EXACT_OLD_LAST,
                "resolution": "applied",
                "event_id": FRC_EXACT_EVENT_ID,
                "exception_code": "",
                "exception_reason": "",
                "reviewed_by": "us_frc_para_lifecycle_repair_v1",
                "reviewed_at": "2026-07-18T00:00:00Z",
                "recheck_after": "",
                "successor_security_id": FRC_EXACT_SECURITY_ID,
                "successor_symbol": "FRCB",
                "source_url": FRC_EXACT_OCC_URL,
                "source": "occ_reviewed_memo_extraction",
                "retrieved_at": "2026-07-18T00:00:00Z",
                "source_hash": FRC_EXACT_OCC_SHA256,
            },
        )
    para_resolution = _require_exact_repaired_resolution(
        resolutions,
        expected={
            "candidate_id": lifecycle_candidate_id(
                PARA_EXACT_SECURITY_ID, PARA_EXACT_LAST
            ),
            "security_id": PARA_EXACT_SECURITY_ID,
            "symbol": "PARA",
            "last_price_date": PARA_EXACT_LAST,
            "resolution": "applied",
            "event_id": PARA_EXACT_EVENT_ID,
            "exception_code": "",
            "exception_reason": "",
            "reviewed_by": "us_frc_para_lifecycle_repair_v1",
            "reviewed_at": "2026-07-18T00:00:00Z",
            "recheck_after": "",
            "successor_security_id": PSKY_EXACT_SECURITY_ID,
            "successor_symbol": "PSKY",
            "source_url": PARA_EXACT_SEC_URL,
            "source": "sec_edgar_filing",
            "retrieved_at": PARA_EXACT_RETRIEVED_AT,
            "source_hash": PARA_EXACT_SEC_SHA256,
        },
    )
    if (
        _text(occ_archive_row.get("retrieved_at")) != "2026-07-18T00:00:00Z"
        or _text(sec_archive_row.get("retrieved_at")) != PARA_EXACT_RETRIEVED_AT
    ):
        raise RuntimeError("Exact FRC/PARA official archive retrieval provenance changed.")
    return (
        {_key(PARA_EXACT_SECURITY_ID, PARA_EXACT_LAST): para_resolution},
        ("FRC/FRCB", "PARA/PSKY"),
    )


def _preserve_exact_celg_official_exit_resolution(
    candidate: LifecycleCandidate,
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
    *,
    cvr_security_id: str,
    terms_row: Mapping[str, Any],
    termination_row: Mapping[str, Any],
) -> dict[str, Any]:
    if CELG_OFFICIAL_EXIT_WARNING not in tuple(release.warnings):
        raise RuntimeError("Exact CELG official_exit_mark release warning is missing.")
    if cvr_security_id != CELG_OFFICIAL_EXIT_SECURITY_ID:
        raise RuntimeError("Exact CELG official_exit_mark BMYRT identity changed.")
    actions = frames["corporate_actions"]
    lifecycle_types = {
        "cash_merger",
        "stock_merger",
        "spinoff",
        "ticker_change",
        "delisting",
    }
    effective = pd.to_datetime(actions["effective_date"], errors="coerce")
    relevant_actions = actions.loc[
        actions["action_type"].astype(str).str.lower().isin(lifecycle_types)
        & (
            actions["security_id"].astype(str).eq(cvr_security_id)
            | (
                actions["security_id"].astype(str).eq(CELG_EXACT_SECURITY_ID)
                & effective.ge(pd.Timestamp(CELG_EXACT_EFFECTIVE_DATE))
            )
        )
    ]
    expected_event_ids = {
        CELG_EXACT_DISTRIBUTION_EVENT_ID,
        CELG_EXACT_MERGER_EVENT_ID,
        CELG_OFFICIAL_EXIT_EVENT_ID,
        CELG_OFFICIAL_RESIDUAL_EVENT_ID,
    }
    if len(relevant_actions) != len(expected_event_ids) or set(
        relevant_actions["event_id"].astype(str)
    ) != expected_event_ids:
        raise RuntimeError(
            "Exact CELG official_exit_mark lifecycle action set changed."
        )
    incoming_cvr = actions.loc[
        actions.get("new_security_id", pd.Series("", index=actions.index))
        .astype(str)
        .eq(cvr_security_id)
    ]
    if len(incoming_cvr) != 1 or _text(incoming_cvr.iloc[0].get("event_id")) != (
        CELG_EXACT_DISTRIBUTION_EVENT_ID
    ):
        raise RuntimeError(
            "Exact CELG official_exit_mark BMYRT creation action changed."
        )
    basis_metadata = {
        "asset_kind": "exchange_traded_contingent_value_right",
        "cost_basis_fraction": CELG_EXACT_CVR_FIRST_CLOSE / 108.78,
        "basis_kind": "economic_relative_fair_value_including_cash",
        "cash_consideration": 50.0,
        "bmy_reference_close": 56.48,
        "cvr_reference_close": CELG_EXACT_CVR_FIRST_CLOSE,
        "reference_total_consideration": 108.78,
        "reference_source_url": CELG_EXACT_TERMINATION_URL,
        "reference_source_hash": CELG_EXACT_TERMINATION_SHA256,
        "trade_start": CELG_EXACT_EFFECTIVE_DATE,
        "exit_policy": "official_exit_mark",
        "non_index_child": True,
        "trading_path_supported": False,
        "termination_date": CELG_EXACT_CVR_TERMINATION_DATE,
        "terminal_payment": 0.0,
    }
    _require_exact_repaired_action(
        actions,
        event_id=CELG_EXACT_DISTRIBUTION_EVENT_ID,
        expected_text={
            "event_id": CELG_EXACT_DISTRIBUTION_EVENT_ID,
            "security_id": CELG_EXACT_SECURITY_ID,
            "action_type": "spinoff",
            "effective_date": CELG_EXACT_EFFECTIVE_DATE,
            "ex_date": CELG_EXACT_EFFECTIVE_DATE,
            "announcement_date": "2019-11-20",
            "record_date": "",
            "payment_date": CELG_EXACT_EFFECTIVE_DATE,
            "currency": "USD",
            "new_security_id": cvr_security_id,
            "new_symbol": "BMYRT",
            "source_url": CELG_EXACT_TERMS_URL,
            "source_kind": "official_crosscheck",
            "source": "official_celg_bmy_cvr",
            "retrieved_at": _text(terms_row.get("retrieved_at")),
            "source_hash": CELG_EXACT_TERMS_SHA256,
        },
        cash_amount=None,
        ratio=1.0,
        metadata=basis_metadata,
    )
    _require_exact_repaired_action(
        actions,
        event_id=CELG_EXACT_MERGER_EVENT_ID,
        expected_text={
            "event_id": CELG_EXACT_MERGER_EVENT_ID,
            "security_id": CELG_EXACT_SECURITY_ID,
            "action_type": "stock_merger",
            "effective_date": CELG_EXACT_EFFECTIVE_DATE,
            "ex_date": CELG_EXACT_EFFECTIVE_DATE,
            "announcement_date": "2019-11-20",
            "record_date": "",
            "payment_date": CELG_EXACT_EFFECTIVE_DATE,
            "currency": "USD",
            "new_security_id": CELG_EXACT_BMY_SECURITY_ID,
            "new_symbol": "BMY",
            "source_url": CELG_EXACT_TERMS_URL,
            "source_kind": "official_crosscheck",
            "source": "official_celg_bmy_cvr",
            "retrieved_at": _text(terms_row.get("retrieved_at")),
            "source_hash": CELG_EXACT_TERMS_SHA256,
        },
        cash_amount=50.0,
        ratio=1.0,
        metadata={
            "additional_security_event_id": CELG_EXACT_DISTRIBUTION_EVENT_ID,
            "consideration_sequence": ["BMYRT", "BMY", "USD"],
            "cvr_security_id": cvr_security_id,
            "cvr_symbol": "BMYRT",
            "cvr_exit_event_id": CELG_OFFICIAL_EXIT_EVENT_ID,
            "cvr_exit_policy": "official_exit_mark",
            "source_url": CELG_EXACT_TERMS_URL,
            "source_hash": CELG_EXACT_TERMS_SHA256,
        },
    )
    _require_exact_repaired_action(
        actions,
        event_id=CELG_OFFICIAL_EXIT_EVENT_ID,
        expected_text={
            "event_id": CELG_OFFICIAL_EXIT_EVENT_ID,
            "security_id": cvr_security_id,
            "action_type": "delisting",
            "effective_date": CELG_EXACT_EFFECTIVE_DATE,
            "ex_date": CELG_EXACT_EFFECTIVE_DATE,
            "announcement_date": "",
            "record_date": "",
            "payment_date": "2019-11-22",
            "currency": "USD",
            "new_security_id": "",
            "new_symbol": "",
            "source_url": CELG_EXACT_TERMINATION_URL,
            "source_kind": "official_filing_exit_mark",
            "source": "sec_bmy_2020_10k",
            "retrieved_at": _text(termination_row.get("retrieved_at")),
            "source_hash": CELG_EXACT_TERMINATION_SHA256,
        },
        cash_amount=CELG_EXACT_CVR_FIRST_CLOSE,
        ratio=None,
        metadata={
            "mode": "official_exit_mark",
            "exit_only": True,
            "non_index_child": True,
            "first_tradable_session": CELG_EXACT_EFFECTIVE_DATE,
            "official_first_trade_close": CELG_EXACT_CVR_FIRST_CLOSE,
            "price_row_kind": "official_valuation_mark_not_provider_ohlcv",
            "execution_timing": "first_tradable_session_close",
            "cash_available_session": "2019-11-22",
            "retrospective_official_evidence": True,
            "trading_path_supported": False,
            "reference_source_url": CELG_EXACT_TERMINATION_URL,
            "reference_source_hash": CELG_EXACT_TERMINATION_SHA256,
        },
    )
    _require_exact_repaired_action(
        actions,
        event_id=CELG_OFFICIAL_RESIDUAL_EVENT_ID,
        expected_text={
            "event_id": CELG_OFFICIAL_RESIDUAL_EVENT_ID,
            "security_id": cvr_security_id,
            "action_type": "delisting",
            "effective_date": CELG_EXACT_CVR_TERMINATION_DATE,
            "ex_date": CELG_EXACT_CVR_TERMINATION_DATE,
            "announcement_date": CELG_EXACT_CVR_TERMINATION_DATE,
            "record_date": "",
            "payment_date": CELG_EXACT_CVR_TERMINATION_DATE,
            "currency": "USD",
            "new_security_id": "",
            "new_symbol": "",
            "source_url": CELG_EXACT_TERMINATION_URL,
            "source_kind": "official_filing",
            "source": "sec_bmy_2020_10k",
            "retrieved_at": _text(termination_row.get("retrieved_at")),
            "source_hash": CELG_EXACT_TERMINATION_SHA256,
        },
        cash_amount=0.0,
        ratio=None,
        metadata={
            "contract_terminated_automatically": True,
            "last_trading_session": CELG_EXACT_CVR_LAST_SESSION,
            "milestone_not_met": "liso-cel FDA approval by 2020-12-31",
            "payout_per_right": 0.0,
            "residual_only": True,
            "only_if_position_remains": True,
            "trading_path_supported": False,
        },
    )

    master = frames["security_master"]
    history = frames["symbol_history"]
    cvr_master = master.loc[master["security_id"].astype(str).eq(cvr_security_id)]
    cvr_history = history.loc[
        history["security_id"].astype(str).eq(cvr_security_id)
        & history["symbol"].astype(str).str.upper().eq("BMYRT")
    ]
    celg_master = master.loc[
        master["security_id"].astype(str).eq(CELG_EXACT_SECURITY_ID)
    ]
    celg_history = history.loc[
        history["security_id"].astype(str).eq(CELG_EXACT_SECURITY_ID)
        & history["symbol"].astype(str).str.upper().eq("CELG")
    ]
    if any(len(value) != 1 for value in (cvr_master, cvr_history, celg_master, celg_history)):
        raise RuntimeError("Exact CELG official_exit_mark identity rows are incomplete.")
    master_row = cvr_master.iloc[0].to_dict()
    history_row = cvr_history.iloc[0].to_dict()
    if not (
        _text(master_row.get("primary_symbol")).upper() == "BMYRT"
        and _text(master_row.get("provider_symbol")) == CELG_OFFICIAL_EXIT_PROVIDER_SYMBOL
        and _text(master_row.get("exchange")).upper() == "NYSE"
        and _date(master_row.get("active_from")) == CELG_EXACT_EFFECTIVE_DATE
        and _date(master_row.get("active_to")) == CELG_EXACT_CVR_LAST_SESSION
        and _text(master_row.get("source")) == "celg_bmyrt_identity_resolution"
        and _text(master_row.get("source_url")) == CELG_EXACT_TERMINATION_URL
        and _date(history_row.get("effective_from")) == CELG_EXACT_EFFECTIVE_DATE
        and _date(history_row.get("effective_to")) == CELG_EXACT_CVR_LAST_SESSION
        and _text(history_row.get("source")) == "celg_bmyrt_identity_resolution"
        and _text(history_row.get("source_url")) == CELG_EXACT_TERMINATION_URL
        and _text(history_row.get("source_hash")) == _text(master_row.get("source_hash"))
        and _date(celg_master.iloc[0].get("active_to")) == CELG_EXACT_LAST_SESSION
        and _date(celg_history.iloc[0].get("effective_to")) == CELG_EXACT_LAST_SESSION
    ):
        raise RuntimeError("Exact CELG official_exit_mark identity boundary changed.")

    archive = frames["source_archive"]
    _archive_binding_row(
        archive,
        source_url=CELG_OFFICIAL_EXIT_ACTIVE_CATALOG_URL,
        source_hash=CELG_OFFICIAL_EXIT_ACTIVE_CATALOG_SHA256,
        source="eodhd_exchange_symbols",
    )
    _archive_binding_row(
        archive,
        source_url=CELG_OFFICIAL_EXIT_DELISTED_CATALOG_URL,
        source_hash=CELG_OFFICIAL_EXIT_DELISTED_CATALOG_SHA256,
        source="eodhd_exchange_symbols",
    )
    active_catalog_content, _ = _archive_pair_content(
        repository,
        archive,
        source_url=CELG_OFFICIAL_EXIT_ACTIVE_CATALOG_URL,
        source_hash=CELG_OFFICIAL_EXIT_ACTIVE_CATALOG_SHA256,
        source="eodhd_exchange_symbols",
    )
    delisted_catalog_content, _ = _archive_pair_content(
        repository,
        archive,
        source_url=CELG_OFFICIAL_EXIT_DELISTED_CATALOG_URL,
        source_hash=CELG_OFFICIAL_EXIT_DELISTED_CATALOG_SHA256,
        source="eodhd_exchange_symbols",
    )
    try:
        active_catalog = json.loads(active_catalog_content)
        delisted_catalog = json.loads(delisted_catalog_content)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("Exact CELG catalog evidence is not JSON.") from exc
    exact_catalog_row = {
        "Code": "CELG-RI",
        "Country": "USA",
        "Currency": "USD",
        "Exchange": "NYSE",
        "Isin": "US1101221406",
        "Name": "Bristol-Myers Squibb Company Ce",
        "Type": "Common Stock",
    }
    if not isinstance(active_catalog, list) or [
        row for row in active_catalog if isinstance(row, dict) and row.get("Code") == "CELG-RI"
    ] != [exact_catalog_row]:
        raise RuntimeError("Exact CELG-RI active catalog row changed.")
    expected_alias_rows = [
        {
            "Code": code,
            "Country": "USA",
            "Currency": "USD",
            "Exchange": "NYSE",
            "Isin": None,
            "Name": code,
            "Type": "Common Stock",
        }
        for code in ("BMY-R", "BMY-RI")
    ]
    if not isinstance(delisted_catalog, list) or [
        row
        for row in delisted_catalog
        if isinstance(row, dict) and row.get("Code") in {"BMY-R", "BMY-RI"}
    ] != expected_alias_rows:
        raise RuntimeError("Exact BMY-R/BMY-RI secondary catalog rows changed.")

    policy_payload = {
        "schema": "celg_bmyrt_official_exit_mark_policy/v1",
        "mode": "official_exit_mark",
        "security_id": cvr_security_id,
        "symbol": "BMYRT",
        "session": CELG_EXACT_EFFECTIVE_DATE,
        "mark_usd": CELG_EXACT_CVR_FIRST_CLOSE,
        "row_encoding": {
            "open": CELG_EXACT_CVR_FIRST_CLOSE,
            "high": CELG_EXACT_CVR_FIRST_CLOSE,
            "low": CELG_EXACT_CVR_FIRST_CLOSE,
            "close": CELG_EXACT_CVR_FIRST_CLOSE,
            "volume": 0.0,
            "meaning": "valuation_mark_not_observed_provider_ohlcv",
        },
        "execution_timing": "first_tradable_session_close",
        "cash_available_session": "2019-11-22",
        "non_index_child": True,
        "retrospective_official_evidence": True,
        "trading_path_supported": False,
        "official_source_url": CELG_EXACT_TERMINATION_URL,
        "official_source_hash": CELG_EXACT_TERMINATION_SHA256,
    }
    policy_content = json.dumps(
        policy_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    policy_hash = sha256_bytes(policy_content)
    archived_policy, policy_row = _archive_pair_content(
        repository,
        frames["source_archive"],
        source_url=CELG_OFFICIAL_EXIT_POLICY_URL,
        source_hash=policy_hash,
        source="official_exit_mark_policy",
    )
    if archived_policy != policy_content:
        raise RuntimeError("Exact CELG official exit-mark policy artifact changed.")
    identity_payload = {
        "schema": "celg_bmyrt_identity_resolution/official_exit_mark/v1",
        "mode": "official_exit_mark",
        "security_id": cvr_security_id,
        "symbol": "BMYRT",
        "provider_code": CELG_OFFICIAL_EXIT_PROVIDER_CODE,
        "provider_symbol": CELG_OFFICIAL_EXIT_PROVIDER_SYMBOL,
        "exchange": "NYSE",
        "active_from": CELG_EXACT_EFFECTIVE_DATE,
        "active_to": CELG_EXACT_CVR_LAST_SESSION,
        "catalog_source_url": CELG_OFFICIAL_EXIT_ACTIVE_CATALOG_URL,
        "catalog_source_hash": CELG_OFFICIAL_EXIT_ACTIVE_CATALOG_SHA256,
        "catalog_row": {
            "Code": "CELG-RI",
            "Country": "USA",
            "Currency": "USD",
            "Exchange": "NYSE",
            "Isin": "US1101221406",
            "Name": "Bristol-Myers Squibb Company Ce",
            "Type": "Common Stock",
        },
        "secondary_catalog_evidence": [
            {
                "role": "secondary_ambiguous",
                "reason": "Provider alias lacks the SEC submitter ticker and ISIN binding.",
                "source_url": CELG_OFFICIAL_EXIT_DELISTED_CATALOG_URL,
                "source_hash": CELG_OFFICIAL_EXIT_DELISTED_CATALOG_SHA256,
                "row": {
                    "Code": code,
                    "Country": "USA",
                    "Currency": "USD",
                    "Exchange": "NYSE",
                    "Isin": None,
                    "Name": code,
                    "Type": "Common Stock",
                },
            }
            for code in ("BMY-R", "BMY-RI")
        ],
        "official_exit_mark": CELG_EXACT_CVR_FIRST_CLOSE,
        "official_exit_session": CELG_EXACT_EFFECTIVE_DATE,
        "official_exit_source_url": CELG_EXACT_TERMINATION_URL,
        "official_exit_source_hash": CELG_EXACT_TERMINATION_SHA256,
        "official_exit_policy_url": CELG_OFFICIAL_EXIT_POLICY_URL,
        "official_exit_policy_hash": policy_hash,
        "trading_path_supported": False,
        "provider_price_artifact_claimed": False,
        "official_merger_url": CELG_EXACT_TERMS_URL,
        "official_merger_sha256": CELG_EXACT_TERMS_SHA256,
        "official_termination_url": CELG_EXACT_TERMINATION_URL,
        "official_termination_sha256": CELG_EXACT_TERMINATION_SHA256,
    }
    identity_content, identity_row = _archive_pair_content(
        repository,
        frames["source_archive"],
        source_url=CELG_EXACT_TERMINATION_URL,
        source_hash=_text(master_row.get("source_hash")),
        source="celg_bmyrt_identity_resolution",
    )
    try:
        identity = json.loads(identity_content)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("Exact CELG official exit identity is not JSON.") from exc
    if identity != identity_payload:
        raise RuntimeError("Exact CELG official exit identity artifact changed.")
    if not (
        _text(identity_row.get("retrieved_at"))
        == _text(policy_row.get("retrieved_at"))
        == _text(termination_row.get("retrieved_at"))
    ):
        raise RuntimeError("Exact CELG official exit retrieval provenance changed.")

    prices = frames["daily_price_raw"].loc[
        frames["daily_price_raw"]["security_id"].astype(str).eq(cvr_security_id)
    ]
    if len(prices) != 1:
        raise RuntimeError("Exact BMYRT official exit mark must have one price row.")
    price = prices.iloc[0]
    numeric = pd.to_numeric(
        pd.Series([price.get(field) for field in ("open", "high", "low", "close")]),
        errors="coerce",
    )
    if not (
        _date(price.get("session")) == CELG_EXACT_EFFECTIVE_DATE
        and numeric.notna().all()
        and np.allclose(
            numeric.to_numpy(dtype=float),
            CELG_EXACT_CVR_FIRST_CLOSE,
            rtol=0,
            atol=0,
        )
        and math.isclose(float(price.get("volume")), 0.0, rel_tol=0, abs_tol=0)
        and _text(price.get("currency")) == "USD"
        and _text(price.get("source")) == "official_exit_mark_policy"
        and _text(price.get("source_url")) == CELG_OFFICIAL_EXIT_POLICY_URL
        and _text(price.get("source_hash")) == policy_hash
        and _text(price.get("retrieved_at")) == _text(termination_row.get("retrieved_at"))
    ):
        raise RuntimeError("Exact BMYRT official exit mark row changed.")
    factors = frames["adjustment_factors"].loc[
        frames["adjustment_factors"]["security_id"].astype(str).eq(cvr_security_id)
    ]
    if len(factors) != 1:
        raise RuntimeError("Exact BMYRT official exit mark factor row changed.")
    factor = factors.iloc[0]
    input_factor_source = _validate_input_adjustment_lineage_for_refinalization(
        repository, release, frames
    )
    calculated_at = _text(factor.get("calculated_at"))
    retrieved_at = _text(factor.get("retrieved_at"))
    if not (
        _date(factor.get("session")) == CELG_EXACT_EFFECTIVE_DATE
        and math.isclose(
            float(factor.get("split_factor")), 1.0, rel_tol=0, abs_tol=0
        )
        and math.isclose(
            float(factor.get("total_return_factor")), 1.0, rel_tol=0, abs_tol=0
        )
        and _text(factor.get("source_version")) == input_factor_source
        and bool(calculated_at)
        and calculated_at == retrieved_at
        and _text(factor.get("source")) == "derived"
        and _text(factor.get("source_hash")) == input_factor_source
    ):
        raise RuntimeError("Exact BMYRT official exit mark factor provenance changed.")
    if _key(candidate.security_id, candidate.last_price_date) == _key(
        CELG_OFFICIAL_EXIT_SECURITY_ID, CELG_EXACT_EFFECTIVE_DATE
    ):
        expected_child_resolution = {
            "candidate_id": lifecycle_candidate_id(
                CELG_OFFICIAL_EXIT_SECURITY_ID, CELG_EXACT_EFFECTIVE_DATE
            ),
            "security_id": CELG_OFFICIAL_EXIT_SECURITY_ID,
            "symbol": "BMYRT",
            "last_price_date": CELG_EXACT_EFFECTIVE_DATE,
            "resolution": "applied",
            "event_id": CELG_OFFICIAL_EXIT_EVENT_ID,
            "exception_code": "",
            "exception_reason": "",
            "reviewed_by": "celg_bmy_cvr_official_exit_mark/v1",
            "reviewed_at": "2026-07-18T08:39:00Z",
            "recheck_after": "",
            "successor_security_id": "",
            "successor_symbol": "",
            "source_url": CELG_EXACT_TERMINATION_URL,
            "source": "sec_bmy_2020_10k",
            "retrieved_at": _text(termination_row.get("retrieved_at")),
            "source_hash": CELG_EXACT_TERMINATION_SHA256,
        }
        child_rows = frames["lifecycle_resolutions"].loc[
            frames["lifecycle_resolutions"]["security_id"]
            .astype(str)
            .eq(CELG_OFFICIAL_EXIT_SECURITY_ID)
        ]
        if child_rows.empty:
            # Releases created before successor-graph candidate expansion have
            # no BMYRT resolution to preserve. The exact action/artifact/model
            # checks above are the only permitted migration path.
            return expected_child_resolution
        if len(child_rows) != 1:
            raise RuntimeError(
                "Exact BMYRT official exit resolution is duplicated."
            )
        return _require_exact_repaired_resolution(
            frames["lifecycle_resolutions"],
            expected=expected_child_resolution,
        )

    return _require_exact_repaired_resolution(
        frames["lifecycle_resolutions"],
        expected={
            "candidate_id": lifecycle_candidate_id(
                CELG_EXACT_SECURITY_ID, CELG_EXACT_LAST_SESSION
            ),
            "security_id": CELG_EXACT_SECURITY_ID,
            "symbol": "CELG",
            "last_price_date": CELG_EXACT_LAST_SESSION,
            "resolution": "applied",
            "event_id": CELG_EXACT_MERGER_EVENT_ID,
            "exception_code": "",
            "exception_reason": "",
            "reviewed_by": "celg_bmy_cvr_official_exit_mark/v1",
            "reviewed_at": "2026-07-18T08:39:00Z",
            "recheck_after": "",
            "successor_security_id": CELG_EXACT_BMY_SECURITY_ID,
            "successor_symbol": "BMY",
            "source_url": CELG_EXACT_TERMS_URL,
            "source": "celg_bmy_cvr_official_exit_mark_repair",
            "retrieved_at": _text(terms_row.get("retrieved_at")),
            "source_hash": CELG_EXACT_TERMS_SHA256,
        },
    )


def _preserve_exact_celg_resolution(
    candidate: LifecycleCandidate,
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    archive = frames["source_archive"]
    _release_archive_content(
        repository,
        archive,
        source_url=CELG_EXACT_TERMS_URL,
        source_hash=CELG_EXACT_TERMS_SHA256,
        content_bytes=CELG_EXACT_TERMS_BYTES,
        source="sec_edgar_filing",
    )
    _release_archive_content(
        repository,
        archive,
        source_url=CELG_EXACT_TERMINATION_URL,
        source_hash=CELG_EXACT_TERMINATION_SHA256,
        content_bytes=CELG_EXACT_TERMINATION_BYTES,
        source="sec_bmy_2020_10k",
    )
    terms_row = _archive_binding_row(
        archive,
        source_url=CELG_EXACT_TERMS_URL,
        source_hash=CELG_EXACT_TERMS_SHA256,
        source="sec_edgar_filing",
    )
    termination_row = _archive_binding_row(
        archive,
        source_url=CELG_EXACT_TERMINATION_URL,
        source_hash=CELG_EXACT_TERMINATION_SHA256,
        source="sec_bmy_2020_10k",
    )
    actions = frames["corporate_actions"]
    spin_rows = actions.loc[
        actions["event_id"].astype(str).eq(CELG_EXACT_DISTRIBUTION_EVENT_ID)
    ]
    if len(spin_rows) != 1:
        raise RuntimeError("Exact CELG CVR distribution is missing or duplicated.")
    cvr_security_id = _text(spin_rows.iloc[0].get("new_security_id"))
    if not cvr_security_id:
        raise RuntimeError("Exact CELG CVR distribution lacks the BMYRT identity.")
    official_exit_rows = actions.loc[
        actions["event_id"].astype(str).eq(CELG_OFFICIAL_EXIT_EVENT_ID)
    ]
    if (
        CELG_OFFICIAL_EXIT_WARNING in tuple(release.warnings)
        or not official_exit_rows.empty
    ):
        return _preserve_exact_celg_official_exit_resolution(
            candidate,
            repository,
            release,
            frames,
            cvr_security_id=cvr_security_id,
            terms_row=terms_row,
            termination_row=termination_row,
        )
    cvr_terminal_event_id = canonical_lifecycle_event_id(
        cvr_security_id, "delisting", CELG_EXACT_CVR_TERMINATION_DATE
    )
    _require_exact_repaired_action(
        actions,
        event_id=CELG_EXACT_DISTRIBUTION_EVENT_ID,
        expected_text={
            "event_id": CELG_EXACT_DISTRIBUTION_EVENT_ID,
            "security_id": CELG_EXACT_SECURITY_ID,
            "action_type": "spinoff",
            "effective_date": CELG_EXACT_EFFECTIVE_DATE,
            "ex_date": CELG_EXACT_EFFECTIVE_DATE,
            "announcement_date": "2019-11-20",
            "record_date": "",
            "payment_date": CELG_EXACT_EFFECTIVE_DATE,
            "currency": "USD",
            "new_security_id": cvr_security_id,
            "new_symbol": "BMYRT",
            "source_url": CELG_EXACT_TERMS_URL,
            "source_kind": "official_crosscheck",
            "source": "official_celg_bmy_cvr",
            "retrieved_at": _text(terms_row.get("retrieved_at")),
            "source_hash": CELG_EXACT_TERMS_SHA256,
        },
        cash_amount=None,
        ratio=1.0,
        metadata_sha256=CELG_EXACT_CVR_BASIS_METADATA_SHA256,
    )
    _require_exact_repaired_action(
        actions,
        event_id=CELG_EXACT_MERGER_EVENT_ID,
        expected_text={
            "event_id": CELG_EXACT_MERGER_EVENT_ID,
            "security_id": CELG_EXACT_SECURITY_ID,
            "action_type": "stock_merger",
            "effective_date": CELG_EXACT_EFFECTIVE_DATE,
            "ex_date": CELG_EXACT_EFFECTIVE_DATE,
            "announcement_date": "2019-11-20",
            "record_date": "",
            "payment_date": CELG_EXACT_EFFECTIVE_DATE,
            "currency": "USD",
            "new_security_id": CELG_EXACT_BMY_SECURITY_ID,
            "new_symbol": "BMY",
            "source_url": CELG_EXACT_TERMS_URL,
            "source_kind": "official_crosscheck",
            "source": "official_celg_bmy_cvr",
            "retrieved_at": _text(terms_row.get("retrieved_at")),
            "source_hash": CELG_EXACT_TERMS_SHA256,
        },
        cash_amount=50.0,
        ratio=1.0,
        metadata={
            "additional_security_event_id": CELG_EXACT_DISTRIBUTION_EVENT_ID,
            "consideration_sequence": ["BMYRT", "BMY", "USD"],
            "cvr_security_id": cvr_security_id,
            "cvr_symbol": "BMYRT",
            "source_url": CELG_EXACT_TERMS_URL,
            "source_hash": CELG_EXACT_TERMS_SHA256,
        },
    )
    _require_exact_repaired_action(
        actions,
        event_id=cvr_terminal_event_id,
        expected_text={
            "event_id": cvr_terminal_event_id,
            "security_id": cvr_security_id,
            "action_type": "delisting",
            "effective_date": CELG_EXACT_CVR_TERMINATION_DATE,
            "ex_date": CELG_EXACT_CVR_TERMINATION_DATE,
            "announcement_date": CELG_EXACT_CVR_TERMINATION_DATE,
            "record_date": "",
            "payment_date": CELG_EXACT_CVR_TERMINATION_DATE,
            "currency": "USD",
            "new_security_id": "",
            "new_symbol": "",
            "source_url": CELG_EXACT_TERMINATION_URL,
            "source_kind": "official_filing",
            "source": "sec_bmy_2020_10k",
            "retrieved_at": _text(termination_row.get("retrieved_at")),
            "source_hash": CELG_EXACT_TERMINATION_SHA256,
        },
        cash_amount=0.0,
        ratio=None,
        metadata_sha256=CELG_EXACT_CVR_TERMINATION_METADATA_SHA256,
    )

    master = frames["security_master"]
    history = frames["symbol_history"]
    master_rows = master.loc[
        master["security_id"].astype(str).eq(cvr_security_id)
    ]
    history_rows = history.loc[
        history["security_id"].astype(str).eq(cvr_security_id)
        & history["symbol"].astype(str).str.upper().eq("BMYRT")
    ]
    if len(master_rows) != 1 or len(history_rows) != 1:
        raise RuntimeError("Exact BMYRT master/history identity is missing or duplicated.")
    master_row = master_rows.iloc[0].to_dict()
    history_row = history_rows.iloc[0].to_dict()
    provider_symbol = _text(master_row.get("provider_symbol"))
    if not provider_symbol.upper().endswith(".US"):
        raise RuntimeError("Exact BMYRT provider symbol is invalid.")
    provider_code = provider_symbol[:-3]
    expected_security_id = f"US:EODHD:{uuid.uuid5(uuid.NAMESPACE_URL, f'eodhd:US:{provider_code}:symbol:BMYRT')}"
    identity_fields_ok = (
        cvr_security_id == expected_security_id
        and _text(master_row.get("primary_symbol")).upper() == "BMYRT"
        and _text(master_row.get("exchange")).upper() == "NYSE"
        and _date(master_row.get("active_from")) == CELG_EXACT_EFFECTIVE_DATE
        and _date(master_row.get("active_to")) == CELG_EXACT_CVR_LAST_SESSION
        and _text(master_row.get("source")) == "celg_bmyrt_identity_resolution"
        and _date(history_row.get("effective_from")) == CELG_EXACT_EFFECTIVE_DATE
        and _date(history_row.get("effective_to")) == CELG_EXACT_CVR_LAST_SESSION
        and _text(history_row.get("source")) == "celg_bmyrt_identity_resolution"
        and _text(history_row.get("source_hash")) == _text(master_row.get("source_hash"))
    )
    if not identity_fields_ok:
        raise RuntimeError("Exact BMYRT identity boundary/provenance changed.")
    identity_hash = _text(master_row.get("source_hash")).lower()
    identity_content, identity_archive_row = _archive_pair_content(
        repository,
        archive,
        source_url=CELG_EXACT_TERMS_URL,
        source_hash=identity_hash,
        source="celg_bmyrt_identity_resolution",
    )
    try:
        identity = json.loads(identity_content)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("Exact BMYRT identity artifact is not JSON.") from exc
    expected_identity = {
        "schema": "celg_bmyrt_identity_resolution/v1",
        "security_id": cvr_security_id,
        "symbol": "BMYRT",
        "provider_code": provider_code,
        "provider_symbol": provider_symbol,
        "exchange": "NYSE",
        "active_from": CELG_EXACT_EFFECTIVE_DATE,
        "active_to": CELG_EXACT_CVR_LAST_SESSION,
        "eodhd_search_url": "https://eodhd.com/api/search/BMYRT?limit=10",
        "eodhd_search_sha256": _text(identity.get("eodhd_search_sha256")),
        "official_merger_url": CELG_EXACT_TERMS_URL,
        "official_merger_sha256": CELG_EXACT_TERMS_SHA256,
        "official_termination_url": CELG_EXACT_TERMINATION_URL,
        "official_termination_sha256": CELG_EXACT_TERMINATION_SHA256,
    }
    if identity != expected_identity:
        raise RuntimeError("Exact BMYRT identity artifact changed.")
    if _text(master_row.get("retrieved_at")) != _text(
        identity_archive_row.get("retrieved_at")
    ):
        raise RuntimeError("Exact BMYRT identity retrieval provenance changed.")
    search_content, _ = _archive_pair_content(
        repository,
        archive,
        source_url=expected_identity["eodhd_search_url"],
        source_hash=expected_identity["eodhd_search_sha256"],
        source="eodhd_search",
    )
    try:
        search_rows = json.loads(search_content)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("Exact BMYRT search artifact is not JSON.") from exc
    search_matches = [
        row
        for row in search_rows
        if isinstance(row, dict)
        and _text(row.get("Code") or row.get("code")) == provider_code
        and "BRISTOL" in _text(row.get("Name") or row.get("name")).upper()
        and "RIGHT" in _text(row.get("Name") or row.get("name")).upper()
    ]
    if len(search_matches) != 1:
        raise RuntimeError("Exact BMYRT provider identity is not search-bound.")

    prices = frames["daily_price_raw"].loc[
        frames["daily_price_raw"]["security_id"].astype(str).eq(cvr_security_id)
    ].copy()
    prices["_session"] = pd.to_datetime(prices["session"], errors="coerce")
    prices = prices.sort_values("_session").reset_index(drop=True)
    expected_sessions = _xnys_sessions(
        CELG_EXACT_EFFECTIVE_DATE, CELG_EXACT_CVR_LAST_SESSION
    )
    observed_sessions = tuple(prices["_session"].dt.date.astype(str))
    first_close = pd.to_numeric(prices.get("close"), errors="coerce").iloc[0]
    if (
        len(prices) != CELG_EXACT_CVR_SESSIONS
        or observed_sessions != expected_sessions
        or pd.isna(first_close)
        or not math.isclose(
            float(first_close), CELG_EXACT_CVR_FIRST_CLOSE, rel_tol=0, abs_tol=1e-8
        )
    ):
        raise RuntimeError("Exact BMYRT 280-session price boundary changed.")
    provenance_fields = ("source", "source_url", "source_hash", "retrieved_at")
    if any(prices[field].astype(str).nunique() != 1 for field in provenance_fields):
        raise RuntimeError("Exact BMYRT price provenance is not one immutable artifact.")
    price_source = _text(prices.iloc[0].get("source"))
    price_url = _text(prices.iloc[0].get("source_url"))
    price_hash = _text(prices.iloc[0].get("source_hash")).lower()
    expected_eod_url = (
        f"https://eodhd.com/api/eod/{provider_code}.US?"
        f"from={CELG_EXACT_EFFECTIVE_DATE}&to={CELG_EXACT_CVR_LAST_SESSION}"
    )
    if price_source != "eodhd_eod" or price_url != expected_eod_url:
        raise RuntimeError("Exact BMYRT EODHD price endpoint changed.")
    eod_content, _ = _archive_pair_content(
        repository,
        archive,
        source_url=price_url,
        source_hash=price_hash,
        source=price_source,
    )
    try:
        eod_rows = json.loads(eod_content)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("Exact BMYRT EOD artifact is not JSON.") from exc
    if not isinstance(eod_rows, list) or len(eod_rows) != CELG_EXACT_CVR_SESSIONS:
        raise RuntimeError("Exact BMYRT EOD raw row inventory changed.")
    raw = pd.DataFrame(eod_rows)
    raw_sessions = tuple(pd.to_datetime(raw["date"], errors="coerce").dt.date.astype(str))
    if raw_sessions != expected_sessions:
        raise RuntimeError("Exact BMYRT EOD raw sessions changed.")
    for field in ("open", "high", "low", "close", "volume"):
        left = pd.to_numeric(prices[field], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(raw[field], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=0, atol=1e-12, equal_nan=False):
            raise RuntimeError(f"Exact BMYRT EOD raw field changed: {field}")
    empty_hash = sha256_bytes(b"[]")
    for endpoint in ("div", "splits"):
        content, _ = _archive_pair_content(
            repository,
            archive,
            source_url=(
                f"https://eodhd.com/api/{endpoint}/{provider_code}.US?"
                f"from={CELG_EXACT_EFFECTIVE_DATE}&to={CELG_EXACT_CVR_LAST_SESSION}"
            ),
            source_hash=empty_hash,
            source=f"eodhd_{endpoint}",
        )
        if content != b"[]":
            raise RuntimeError(f"Exact BMYRT {endpoint} artifact is no longer empty.")

    return _require_exact_repaired_resolution(
        frames["lifecycle_resolutions"],
        expected={
            "candidate_id": lifecycle_candidate_id(
                CELG_EXACT_SECURITY_ID, CELG_EXACT_LAST_SESSION
            ),
            "security_id": CELG_EXACT_SECURITY_ID,
            "symbol": "CELG",
            "last_price_date": CELG_EXACT_LAST_SESSION,
            "resolution": "applied",
            "event_id": CELG_EXACT_MERGER_EVENT_ID,
            "exception_code": "",
            "exception_reason": "",
            "reviewed_by": "celg_bmy_cvr_exact_model/v1",
            "reviewed_at": "2026-07-18T08:39:00Z",
            "recheck_after": "",
            "successor_security_id": CELG_EXACT_BMY_SECURITY_ID,
            "successor_symbol": "BMY",
            "source_url": CELG_EXACT_TERMS_URL,
            "source": "celg_bmy_cvr_exact_repair",
            "retrieved_at": _text(terms_row.get("retrieved_at")),
            "source_hash": CELG_EXACT_TERMS_SHA256,
        },
    )


def _preserve_exact_abmd_resolution(
    candidate: LifecycleCandidate,
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    if ABMD_EXACT_WARNING not in release.warnings:
        raise RuntimeError("Exact ABMD lower-bound release warning is missing.")
    archive = frames["source_archive"]
    _release_archive_content(
        repository,
        archive,
        source_url=ABMD_EXACT_TERMS_URL,
        source_hash=ABMD_EXACT_TERMS_SHA256,
        content_bytes=ABMD_EXACT_TERMS_BYTES,
        source="sec_edgar_filing",
    )
    _release_archive_content(
        repository,
        archive,
        source_url=ABMD_EXACT_VALUATION_URL,
        source_hash=ABMD_EXACT_VALUATION_SHA256,
        content_bytes=ABMD_EXACT_VALUATION_BYTES,
        source="jnj_2025_annual_report",
    )
    terms_row = _archive_binding_row(
        archive,
        source_url=ABMD_EXACT_TERMS_URL,
        source_hash=ABMD_EXACT_TERMS_SHA256,
        source="sec_edgar_filing",
    )
    _require_exact_repaired_action(
        frames["corporate_actions"],
        event_id=ABMD_EXACT_EVENT_ID,
        expected_text={
            "event_id": ABMD_EXACT_EVENT_ID,
            "security_id": ABMD_EXACT_SECURITY_ID,
            "action_type": "cash_merger",
            "effective_date": ABMD_EXACT_EFFECTIVE_DATE,
            "ex_date": ABMD_EXACT_EFFECTIVE_DATE,
            "announcement_date": "2022-11-01",
            "record_date": "",
            "payment_date": ABMD_EXACT_EFFECTIVE_DATE,
            "currency": "USD",
            "new_security_id": "",
            "new_symbol": "",
            "source_url": ABMD_EXACT_TERMS_URL,
            "source_kind": "official_lower_bound_policy",
            "source": "sec_edgar+audited_lower_bound_policy",
            "retrieved_at": _text(terms_row.get("retrieved_at")),
            "source_hash": ABMD_EXACT_TERMS_SHA256,
        },
        cash_amount=380.0,
        ratio=None,
        metadata_sha256=ABMD_EXACT_METADATA_SHA256,
    )
    master = frames["security_master"].loc[
        frames["security_master"]["security_id"]
        .astype(str)
        .eq(ABMD_EXACT_SECURITY_ID)
    ]
    history = frames["symbol_history"].loc[
        frames["symbol_history"]["security_id"]
        .astype(str)
        .eq(ABMD_EXACT_SECURITY_ID)
        & frames["symbol_history"]["symbol"].astype(str).str.upper().eq("ABMD")
    ]
    if (
        len(master) != 1
        or len(history) != 1
        or _date(master.iloc[0].get("active_to")) != ABMD_EXACT_LAST_SESSION
        or _date(history.iloc[0].get("effective_to")) != ABMD_EXACT_LAST_SESSION
    ):
        raise RuntimeError("Exact ABMD terminal identity boundary changed.")
    prices = frames["daily_price_raw"].loc[
        frames["daily_price_raw"]["security_id"]
        .astype(str)
        .eq(ABMD_EXACT_SECURITY_ID)
    ].copy()
    prices["_session"] = pd.to_datetime(prices["session"], errors="coerce")
    last = prices.sort_values("_session").iloc[-1]
    close = pd.to_numeric(last.get("close"), errors="coerce")
    if (
        _date(last.get("_session")) != ABMD_EXACT_LAST_SESSION
        or pd.isna(close)
        or not math.isclose(
            float(close), ABMD_EXACT_LAST_CLOSE, rel_tol=0, abs_tol=1e-8
        )
    ):
        raise RuntimeError("Exact ABMD last real session/close changed.")
    return _require_exact_repaired_resolution(
        frames["lifecycle_resolutions"],
        expected={
            "candidate_id": lifecycle_candidate_id(
                ABMD_EXACT_SECURITY_ID, ABMD_EXACT_LAST_SESSION
            ),
            "security_id": ABMD_EXACT_SECURITY_ID,
            "symbol": "ABMD",
            "last_price_date": ABMD_EXACT_LAST_SESSION,
            "resolution": "applied",
            "event_id": ABMD_EXACT_EVENT_ID,
            "exception_code": "",
            "exception_reason": "",
            "reviewed_by": "abmd_cvr_lower_bound_policy_v1",
            "reviewed_at": "2026-07-18T00:00:00Z",
            "recheck_after": "",
            "successor_security_id": "",
            "successor_symbol": "",
            "source_url": ABMD_EXACT_TERMS_URL,
            "source": "abmd_cvr_lower_bound_repair",
            "retrieved_at": _text(terms_row.get("retrieved_at")),
            "source_hash": ABMD_EXACT_TERMS_SHA256,
        },
    )


def _avp_exact_applied_resolution() -> dict[str, Any]:
    return {
        "candidate_id": lifecycle_candidate_id(
            AVP_EXACT_SECURITY_ID, AVP_EXACT_LAST_SESSION
        ),
        "security_id": AVP_EXACT_SECURITY_ID,
        "symbol": "AVP",
        "last_price_date": AVP_EXACT_LAST_SESSION,
        "resolution": "applied",
        "event_id": AVP_EXACT_EVENT_ID,
        "exception_code": "",
        "exception_reason": "",
        "reviewed_by": AVP_EXACT_REVIEWED_BY,
        "reviewed_at": AVP_EXACT_REVIEWED_AT,
        "recheck_after": "",
        "successor_security_id": AVP_EXACT_SUCCESSOR_ID,
        "successor_symbol": "NTCO",
        "source_url": AVP_EXACT_SEC_URL,
        "source": AVP_EXACT_REPAIR_SOURCE,
        "retrieved_at": AVP_EXACT_REVIEWED_AT,
        "source_hash": AVP_EXACT_SEC_SHA256,
    }


def _avp_exact_archive_content(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    release: DataRelease,
    source_url: str,
    source_hash: str,
    source: str,
    content_type: str,
    retrieved_at: str,
    expected_bytes: int,
) -> bytes:
    content, row = _archive_pair_content(
        repository,
        archive,
        source_url=source_url,
        source_hash=source_hash,
        source=source,
    )
    expected_row = {
        "archive_id": source_hash,
        "dataset": source,
        "content_type": content_type,
        "effective_date": release.completed_session,
        "source": source,
        "source_url": source_url,
        "retrieved_at": retrieved_at,
        "source_hash": source_hash,
    }
    for field, wanted in expected_row.items():
        actual = (
            _date(row.get(field))
            if field == "effective_date"
            else _text(row.get(field))
        )
        target = _date(wanted) if field == "effective_date" else _text(wanted)
        if actual != target:
            raise RuntimeError(f"Exact AVP archive row changed: {field}")
    if len(content) != expected_bytes:
        raise RuntimeError("Exact AVP archive byte inventory changed.")
    return content


def _avp_exact_eod_frame(
    content: bytes,
    *,
    label: str,
    expected_rows: int,
) -> pd.DataFrame:
    try:
        values = json.loads(content)
    except (UnicodeDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Exact {label} EOD artifact is not JSON.") from exc
    if (
        not isinstance(values, list)
        or len(values) != expected_rows
        or not all(isinstance(row, dict) for row in values)
    ):
        raise RuntimeError(f"Exact {label} EOD raw row inventory changed.")
    raw = pd.DataFrame(values)
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"Exact {label} EOD raw fields changed: {missing}")
    raw["_session"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.sort_values("_session").reset_index(drop=True)
    sessions = tuple(raw["_session"].dt.date.astype(str))
    if (
        raw["_session"].isna().any()
        or len(sessions) != len(set(sessions))
        or sessions != tuple(sorted(sessions))
    ):
        raise RuntimeError(f"Exact {label} EOD raw sessions changed.")
    numeric = raw.loc[:, ["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any():
        raise RuntimeError(f"Exact {label} EOD raw OHLCV is invalid.")
    if (
        (numeric[["open", "high", "low", "close"]] <= 0).any().any()
        or (numeric["volume"] < 0).any()
        or (numeric["high"] < numeric[["open", "low", "close"]].max(axis=1)).any()
        or (numeric["low"] > numeric[["open", "high", "close"]].min(axis=1)).any()
    ):
        raise RuntimeError(f"Exact {label} EOD raw price geometry changed.")
    return raw


def _validate_avp_exact_stored_prices(
    prices: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    security_id: str,
    label: str,
    source_hash: str,
    retrieved_at: str,
) -> None:
    stored = prices.loc[
        prices["security_id"].astype(str).eq(security_id)
    ].copy()
    stored["_session"] = pd.to_datetime(stored["session"], errors="coerce")
    stored = stored.sort_values("_session").reset_index(drop=True)
    stored_sessions = tuple(stored["_session"].dt.date.astype(str))
    raw_sessions = tuple(raw["_session"].dt.date.astype(str))
    if (
        stored["_session"].isna().any()
        or stored["_session"].duplicated().any()
        or stored_sessions != raw_sessions
    ):
        raise RuntimeError(f"Exact {label} stored price sessions changed.")
    fields = ["open", "high", "low", "close", "volume"]
    stored_values = stored.loc[:, fields].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    raw_values = raw.loc[:, fields].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    if not np.allclose(
        stored_values,
        raw_values,
        rtol=0,
        atol=1e-12,
        equal_nan=False,
    ):
        raise RuntimeError(f"Exact {label} stored prices differ from raw EOD bytes.")
    expected_provenance = {
        "source": {"eodhd_eod"},
        "source_hash": {source_hash},
        "retrieved_at": {retrieved_at},
        "currency": {"USD"},
    }
    observed_provenance = {
        field: set(stored[field].astype(str)) for field in expected_provenance
    }
    if observed_provenance != expected_provenance:
        raise RuntimeError(f"Exact {label} stored price provenance changed.")


def _validate_avp_temporary_resolution_report(
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
) -> None:
    archive = frames["source_archive"]
    report_content = _avp_exact_archive_content(
        repository,
        archive,
        release=release,
        source_url=AVP_EXACT_TEMPORARY_REPORT_URL,
        source_hash=AVP_EXACT_TEMPORARY_REPORT_SHA256,
        source="lifecycle_evidence_report",
        content_type="application/json",
        retrieved_at=AVP_EXACT_TEMPORARY_REPORT_RETRIEVED_AT,
        expected_bytes=AVP_EXACT_TEMPORARY_REPORT_BYTES,
    )
    try:
        report = json.loads(report_content)
        record = report["records"][AVP_EXACT_SECURITY_ID]
    except (KeyError, TypeError, UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("Exact AVP temporary lifecycle report changed.") from exc
    candidate = record.get("candidate") or {}
    parsed = record.get("parsed") or {}
    crosscheck = record.get("crosscheck") or {}
    if not (
        _text(candidate.get("security_id")) == AVP_EXACT_SECURITY_ID
        and _text(candidate.get("symbol")).upper() == "AVP"
        and _date(candidate.get("last_price_date")) == AVP_EXACT_LAST_SESSION
        and _date(candidate.get("active_to")) == AVP_EXACT_LAST_SESSION
        and _text(record.get("source_url")) == AVP_EXACT_SEC_URL
        and _text(record.get("source_hash")) == AVP_EXACT_SEC_SHA256
        and record.get("eligible_for_apply") is False
        and _text(parsed.get("action_type")) == "stock_merger"
        and _date(parsed.get("effective_date")) == AVP_EXACT_LEGAL_COMPLETION
        and _text(parsed.get("new_symbol")).upper() == "NTCO"
        and _text(parsed.get("confidence")).lower() == "high"
        and _same_optional_number(parsed.get("ratio"), AVP_EXACT_RATIO)
        and _date(crosscheck.get("old_price_session")) == AVP_EXACT_LAST_SESSION
        and _same_optional_number(crosscheck.get("old_close"), 5.6)
        and not _text(crosscheck.get("successor_price_session"))
    ):
        raise RuntimeError("Exact AVP temporary lifecycle report semantics changed.")


def _preserve_exact_avp_resolution(
    candidate: LifecycleCandidate,
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    """Attest AVP -> NTCO across the legal-close/next-session boundary."""

    if not (
        candidate.security_id == AVP_EXACT_SECURITY_ID
        and candidate.symbol.upper() == "AVP"
        and candidate.name == "Avon Products Inc"
        and candidate.exchange.upper() == "NYSE"
        and _date(candidate.last_price_date) == AVP_EXACT_LAST_SESSION
        and _date(candidate.active_to) == AVP_EXACT_LAST_SESSION
        and tuple(candidate.index_remove_dates) == ("2015-03-23",)
        and lifecycle_candidate_id(candidate.security_id, candidate.last_price_date)
        == lifecycle_candidate_id(AVP_EXACT_SECURITY_ID, AVP_EXACT_LAST_SESSION)
    ):
        raise RuntimeError("Exact AVP lifecycle candidate changed.")
    if _date(release.completed_session) != "2026-07-15":
        raise RuntimeError("Exact AVP completed-session binding changed.")
    required = {
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "corporate_actions",
        "adjustment_factors",
        "source_archive",
        "lifecycle_resolutions",
    }
    missing = sorted(required - set(frames))
    if missing:
        raise RuntimeError(
            "Exact AVP repair is partial; missing datasets: " + ", ".join(missing)
        )

    archive = frames["source_archive"]
    sec_content = _avp_exact_archive_content(
        repository,
        archive,
        release=release,
        source_url=AVP_EXACT_SEC_URL,
        source_hash=AVP_EXACT_SEC_SHA256,
        source="sec_edgar_filing",
        content_type="text/plain",
        retrieved_at=AVP_EXACT_SEC_RETRIEVED_AT,
        expected_bytes=AVP_EXACT_SEC_BYTES,
    )
    normalized = html.unescape(
        re.sub(r"<[^>]+>", " ", sec_content.decode("utf-8", errors="replace"))
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    patterns = {
        "legal_completion": (
            r"On January 3, 2020.{0,500}?consummated the previously announced "
            r"business combination"
        ),
        "exchange_ratio": (
            r"each share.{0,800}?automatically converted.{0,300}?0\.300 "
            r"validly issued.{0,250}?American Depositary Shares"
        ),
        "avp_suspension": (
            r"AVP.{0,250}?NYSE.{0,250}?suspended from trading.{0,250}?prior "
            r"to the opening of the market on January 6, 2020"
        ),
        "ntco_first_trade": (
            r"expects to begin trading.{0,120}?NYSE \(NTCO\) on January 6"
        ),
    }
    missing_patterns = [
        label for label, pattern in patterns.items()
        if re.search(pattern, normalized, re.I) is None
    ]
    if missing_patterns:
        raise RuntimeError(
            "Exact AVP SEC evidence no longer proves: "
            + ", ".join(missing_patterns)
        )
    if _xnys_sessions(
        AVP_EXACT_LEGAL_COMPLETION, AVP_EXACT_MARKET_TRANSITION
    ) != (AVP_EXACT_LEGAL_COMPLETION, AVP_EXACT_MARKET_TRANSITION):
        raise RuntimeError("Exact AVP next-XNYS-session relation changed.")

    _require_exact_repaired_action(
        frames["corporate_actions"],
        event_id=AVP_EXACT_EVENT_ID,
        expected_text={
            "event_id": AVP_EXACT_EVENT_ID,
            "security_id": AVP_EXACT_SECURITY_ID,
            "action_type": "stock_merger",
            "effective_date": AVP_EXACT_MARKET_TRANSITION,
            "ex_date": AVP_EXACT_MARKET_TRANSITION,
            "announcement_date": AVP_EXACT_LEGAL_COMPLETION,
            "record_date": "",
            "payment_date": "",
            "currency": "USD",
            "new_security_id": AVP_EXACT_SUCCESSOR_ID,
            "new_symbol": "NTCO",
            "source_url": AVP_EXACT_SEC_URL,
            "source_kind": "official_crosscheck",
            "source": "sec_edgar+stored_price_crosscheck",
            "retrieved_at": AVP_EXACT_SEC_RETRIEVED_AT,
            "source_hash": AVP_EXACT_SEC_SHA256,
        },
        cash_amount=None,
        ratio=AVP_EXACT_RATIO,
        metadata_sha256=AVP_EXACT_ACTION_METADATA_SHA256,
    )

    master = frames["security_master"]
    _require_exact_identity_row(
        master,
        security_id=AVP_EXACT_SECURITY_ID,
        label="AVP master",
        expected={
            "primary_symbol": "AVP",
            "provider_symbol": "AVP.US",
            "name": "Avon Products Inc",
            "exchange": "NYSE",
            "active_from": "2015-01-02",
            "active_to": AVP_EXACT_LAST_SESSION,
        },
    )
    _require_exact_identity_row(
        master,
        security_id=AVP_EXACT_SUCCESSOR_ID,
        label="AVP successor master",
        expected={
            "primary_symbol": NTCO_EXACT_NEW_SYMBOL,
            "provider_symbol": "NTCOY.US",
            "active_from": AVP_EXACT_MARKET_TRANSITION,
            "active_to": NTCO_EXACT_TERMINAL_DATE,
        },
    )
    _require_exact_history_row(
        frames["symbol_history"],
        security_id=AVP_EXACT_SECURITY_ID,
        symbol="AVP",
        label="AVP",
        expected={
            "exchange": "NYSE",
            "effective_from": "2015-01-01",
            "effective_to": AVP_EXACT_LAST_SESSION,
            "source": AVP_EXACT_REPAIR_SOURCE,
            "source_url": AVP_EXACT_SEC_URL,
            "retrieved_at": AVP_EXACT_REVIEWED_AT,
            "source_hash": AVP_EXACT_SEC_SHA256,
        },
    )
    _require_exact_history_row(
        frames["symbol_history"],
        security_id=AVP_EXACT_SUCCESSOR_ID,
        symbol="NTCO",
        label="AVP successor NTCO",
        expected={
            "exchange": "NYSE",
            "effective_from": AVP_EXACT_MARKET_TRANSITION,
            "effective_to": AVP_EXACT_NTCO_NYSE_END,
            "source": "official_ntco_ntcoy_identity",
            "source_url": NTCO_EXACT_OCC_URL,
            "retrieved_at": NTCO_EXACT_RETRIEVED_AT,
            "source_hash": NTCO_EXACT_IDENTITY_SHA256,
        },
    )

    avp_content = _avp_exact_archive_content(
        repository,
        archive,
        release=release,
        source_url=AVP_EXACT_RAW_URL,
        source_hash=AVP_EXACT_RAW_SHA256,
        source="eodhd_eod",
        content_type="application/json",
        retrieved_at=AVP_EXACT_RAW_RETRIEVED_AT,
        expected_bytes=AVP_EXACT_RAW_BYTES,
    )
    avp_raw = _avp_exact_eod_frame(
        avp_content, label="AVP", expected_rows=AVP_EXACT_RAW_ROWS
    )
    if (
        _date(avp_raw.iloc[-1]["_session"]) != AVP_EXACT_LAST_SESSION
        or tuple(
            float(avp_raw.iloc[-1][field])
            for field in ("open", "high", "low", "close", "volume")
        )
        != AVP_EXACT_TERMINAL_OHLCV
    ):
        raise RuntimeError("Exact AVP terminal raw OHLCV changed.")
    _validate_avp_exact_stored_prices(
        frames["daily_price_raw"],
        avp_raw,
        security_id=AVP_EXACT_SECURITY_ID,
        label="AVP",
        source_hash=AVP_EXACT_RAW_SHA256,
        retrieved_at=AVP_EXACT_RAW_RETRIEVED_AT,
    )

    ntco_envelope_content = _avp_exact_archive_content(
        repository,
        archive,
        release=release,
        source_url=AVP_EXACT_NTCO_RAW_URL,
        source_hash=AVP_EXACT_NTCO_ENVELOPE_SHA256,
        source="eodhd_eod",
        content_type="application/vnd.supertrendquant.source-envelope+json",
        retrieved_at=AVP_EXACT_NTCO_RAW_RETRIEVED_AT,
        expected_bytes=AVP_EXACT_NTCO_ENVELOPE_BYTES,
    )
    try:
        envelope = json.loads(ntco_envelope_content)
        ntco_content = base64.b64decode(
            envelope["content_base64"], validate=True
        )
    except (KeyError, TypeError, UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("Exact AVP successor EOD envelope changed.") from exc
    expected_envelope = {
        "content_sha256": AVP_EXACT_NTCO_RAW_SHA256,
        "content_type": "application/json",
        "source": "eodhd_eod",
        "source_url": AVP_EXACT_NTCO_RAW_URL,
    }
    if (
        {field: _text(envelope.get(field)) for field in expected_envelope}
        != expected_envelope
        or len(ntco_content) != AVP_EXACT_NTCO_RAW_BYTES
        or sha256_bytes(ntco_content) != AVP_EXACT_NTCO_RAW_SHA256
    ):
        raise RuntimeError("Exact AVP successor EOD envelope binding changed.")
    ntco_raw = _avp_exact_eod_frame(
        ntco_content,
        label="AVP successor NTCO",
        expected_rows=AVP_EXACT_NTCO_RAW_ROWS,
    )
    ntco_nyse_raw = ntco_raw.loc[
        ntco_raw["_session"].dt.date.astype(str).le(AVP_EXACT_NTCO_NYSE_END)
    ].reset_index(drop=True)
    if (
        len(ntco_nyse_raw) != AVP_EXACT_NTCO_NYSE_ROWS
        or _date(ntco_nyse_raw.iloc[0]["_session"])
        != AVP_EXACT_MARKET_TRANSITION
        or _date(ntco_nyse_raw.iloc[-1]["_session"])
        != AVP_EXACT_NTCO_NYSE_END
        or tuple(
            float(ntco_nyse_raw.iloc[0][field])
            for field in ("open", "high", "low", "close", "volume")
        )
        != AVP_EXACT_NTCO_FIRST_OHLCV
    ):
        raise RuntimeError("Exact AVP successor first-session boundary changed.")
    successor_prices = frames["daily_price_raw"].loc[
        frames["daily_price_raw"]["security_id"]
        .astype(str)
        .eq(AVP_EXACT_SUCCESSOR_ID)
        & pd.to_datetime(
            frames["daily_price_raw"]["session"], errors="coerce"
        )
        .dt.date.astype(str)
        .le(AVP_EXACT_NTCO_NYSE_END)
    ].copy()
    _validate_avp_exact_stored_prices(
        successor_prices,
        ntco_nyse_raw,
        security_id=AVP_EXACT_SUCCESSOR_ID,
        label="AVP successor NTCO",
        source_hash=AVP_EXACT_NTCO_RAW_SHA256,
        retrieved_at=AVP_EXACT_NTCO_RAW_RETRIEVED_AT,
    )
    if _exact_security_session_keys(
        frames["daily_price_raw"], {AVP_EXACT_SECURITY_ID}, label="AVP prices"
    ) != _exact_security_session_keys(
        frames["adjustment_factors"],
        {AVP_EXACT_SECURITY_ID},
        label="AVP adjustment factors",
    ):
        raise RuntimeError("Exact AVP adjustment-factor inventory changed.")

    exact_resolution = _avp_exact_applied_resolution()
    current = frames["lifecycle_resolutions"].loc[
        frames["lifecycle_resolutions"]["security_id"]
        .astype(str)
        .eq(AVP_EXACT_SECURITY_ID)
    ]
    if len(current) != 1:
        raise RuntimeError("Exact AVP lifecycle resolution is missing or duplicated.")
    if _text(current.iloc[0].get("resolution")) == "applied":
        return _require_exact_repaired_resolution(
            frames["lifecycle_resolutions"], expected=exact_resolution
        )
    _require_exact_repaired_resolution(
        frames["lifecycle_resolutions"],
        expected={
            "candidate_id": exact_resolution["candidate_id"],
            "security_id": AVP_EXACT_SECURITY_ID,
            "symbol": "AVP",
            "last_price_date": AVP_EXACT_LAST_SESSION,
            "resolution": "exception",
            "event_id": "",
            "exception_code": "successor_unresolved",
            "exception_reason": "AVP to NTCO successor chain is not fully crosschecked.",
            "reviewed_by": REVIEWED_BY,
            "reviewed_at": REVIEWED_AT,
            "recheck_after": DEFAULT_RECHECK_AFTER,
            "successor_security_id": "",
            "successor_symbol": "",
            "source_url": (
                "archive://source_archive/"
                + AVP_EXACT_TEMPORARY_REPORT_SHA256
            ),
            "source": "lifecycle_evidence_report",
            "retrieved_at": AVP_EXACT_TEMPORARY_REPORT_RETRIEVED_AT,
            "source_hash": AVP_EXACT_TEMPORARY_REPORT_SHA256,
        },
    )
    _validate_avp_temporary_resolution_report(
        repository, release, frames
    )
    return exact_resolution


def _preserve_exact_sivb_resolution(
    candidate: LifecycleCandidate,
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    """Preserve only the complete SIVB/SIVBQ repair; reject every partial form."""

    if (
        candidate.security_id != SIVB_EXACT_SECURITY_ID
        or candidate.symbol.upper() != "SIVBQ"
        or _date(candidate.last_price_date) != SIVB_EXACT_LAST_SESSION
        or lifecycle_candidate_id(candidate.security_id, candidate.last_price_date)
        != lifecycle_candidate_id(SIVB_EXACT_SECURITY_ID, SIVB_EXACT_LAST_SESSION)
    ):
        raise RuntimeError("Exact SIVB/SIVBQ lifecycle candidate changed.")
    if _date(release.completed_session) != "2026-07-15":
        raise RuntimeError("Exact SIVB/SIVBQ completed-session binding changed.")

    required = {
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "corporate_actions",
        "adjustment_factors",
        "source_archive",
        "lifecycle_resolutions",
    }
    missing = sorted(required - set(frames))
    if missing:
        raise RuntimeError(
            "Exact SIVB/SIVBQ repair is partial; missing datasets: "
            + ", ".join(missing)
        )

    archive = frames["source_archive"]
    archive_specs = (
        (
            "occ_raw_pdf",
            SIVB_EXACT_OCC_URL,
            SIVB_EXACT_OCC_PDF_SHA256,
            "occ_information_memo",
            "occ_information_memo",
            "application/pdf",
            SIVB_EXACT_OCC_PDF_BYTES,
            "2026-07-18T18:20:45Z",
            "pdf",
        ),
        (
            "occ_legacy_extraction",
            SIVB_EXACT_OCC_URL,
            SIVB_EXACT_OCC_LEGACY_SHA256,
            "occ_reviewed_memo_extraction",
            "occ_reviewed_memo_extraction",
            "application/json",
            659,
            "2026-07-18T14:11:49.785762Z",
            "json",
        ),
        (
            "sec_market_transition",
            SIVB_EXACT_SEC_MARKET_URL,
            SIVB_EXACT_SEC_MARKET_SHA256,
            "sec_edgar_filing",
            "sec_edgar_filing",
            "text/html",
            33_250,
            "2026-07-18T14:11:49.785762Z",
            "html",
        ),
        (
            "eodhd_otc_path",
            SIVB_EXACT_EOD_URL,
            SIVB_EXACT_EOD_SHA256,
            "eodhd_eod",
            "eodhd_eod",
            "application/json",
            44_932,
            "2026-07-18T14:11:49.785762Z",
            "json",
        ),
        (
            "sec_legal_cancellation",
            SIVB_EXACT_SEC_CANCEL_URL,
            SIVB_EXACT_SEC_CANCEL_SHA256,
            "sec_edgar_filing",
            "sec_edgar_filing",
            "text/html",
            54_478,
            "2026-07-18T10:31:24.972232Z",
            "html",
        ),
    )
    contents: dict[str, bytes] = {}
    for (
        label,
        source_url,
        source_hash,
        source,
        dataset,
        content_type,
        content_bytes,
        retrieved_at,
        suffix,
    ) in archive_specs:
        content, row = _archive_pair_content(
            repository,
            archive,
            source_url=source_url,
            source_hash=source_hash,
            source=source,
        )
        expected_object = (
            f"archives/2026-07-15/{source_hash}.{suffix}.gz"
        )
        if (
            len(content) != content_bytes
            or _text(row.get("archive_id")).lower() != source_hash
            or _text(row.get("dataset")) != dataset
            or _text(row.get("content_type")) != content_type
            or _text(row.get("object_path")) != expected_object
            or _date(row.get("effective_date")) != "2026-07-15"
            or _text(row.get("retrieved_at")) != retrieved_at
        ):
            raise RuntimeError(
                "Exact SIVB/SIVBQ archive row changed: " + label
            )
        contents[label] = content
    if (
        not contents["occ_raw_pdf"].startswith(b"%PDF-")
        or b"%%EOF" not in contents["occ_raw_pdf"][-4096:]
    ):
        raise RuntimeError("Exact SIVB OCC 52179 object is not a complete PDF.")

    actions = frames["corporate_actions"]
    _require_exact_repaired_action(
        actions,
        event_id=SIVB_EXACT_TICKER_EVENT_ID,
        expected_text={
            "event_id": SIVB_EXACT_TICKER_EVENT_ID,
            "security_id": SIVB_EXACT_SECURITY_ID,
            "action_type": "ticker_change",
            "effective_date": SIVB_EXACT_OTC_START,
            "ex_date": SIVB_EXACT_OTC_START,
            "announcement_date": "2023-03-27",
            "record_date": "",
            "payment_date": "",
            "currency": "USD",
            "new_security_id": SIVB_EXACT_SECURITY_ID,
            "new_symbol": "SIVBQ",
            "source_url": SIVB_EXACT_OCC_URL,
            "source_kind": "official_crosscheck",
            "source": "occ_information_memo",
            "retrieved_at": "2026-07-18T18:20:45Z",
            "source_hash": SIVB_EXACT_OCC_PDF_SHA256,
        },
        cash_amount=None,
        ratio=None,
        metadata_sha256=SIVB_EXACT_TICKER_METADATA_SHA256,
    )
    _require_exact_repaired_action(
        actions,
        event_id=SIVB_EXACT_MARKET_EXIT_EVENT_ID,
        expected_text={
            "event_id": SIVB_EXACT_MARKET_EXIT_EVENT_ID,
            "security_id": SIVB_EXACT_SECURITY_ID,
            "action_type": "delisting",
            "effective_date": SIVB_EXACT_ENGINE_EXIT,
            "ex_date": SIVB_EXACT_ENGINE_EXIT,
            "announcement_date": SIVB_EXACT_ENGINE_EXIT,
            "record_date": "",
            "payment_date": "",
            "currency": "USD",
            "new_security_id": "",
            "new_symbol": "",
            "source_url": SIVB_EXACT_SEC_CANCEL_URL,
            "source_kind": "official_crosscheck",
            "source": "sec_edgar+stored_price_crosscheck",
            "retrieved_at": "2026-07-18T10:31:24.972232Z",
            "source_hash": SIVB_EXACT_SEC_CANCEL_SHA256,
        },
        cash_amount=0.0,
        ratio=None,
        metadata_sha256=SIVB_EXACT_EXIT_METADATA_SHA256,
    )
    lifecycle_types = {
        "cash_merger",
        "stock_merger",
        "spinoff",
        "ticker_change",
        "delisting",
    }
    touched_actions = actions.loc[
        actions["security_id"].astype(str).eq(SIVB_EXACT_SECURITY_ID)
        & actions["action_type"].astype(str).str.lower().isin(lifecycle_types)
    ]
    if (
        len(touched_actions) != 2
        or set(touched_actions["event_id"].astype(str))
        != {SIVB_EXACT_TICKER_EVENT_ID, SIVB_EXACT_MARKET_EXIT_EVENT_ID}
    ):
        raise RuntimeError("Exact SIVB/SIVBQ action inventory changed.")

    common_identity = {
        "source": "occ_reviewed_memo_extraction",
        "source_url": SIVB_EXACT_OCC_URL,
        "retrieved_at": "2026-07-18T14:11:49.785762Z",
        "source_hash": SIVB_EXACT_OCC_LEGACY_SHA256,
    }
    _require_exact_identity_row(
        frames["security_master"],
        security_id=SIVB_EXACT_SECURITY_ID,
        label="SIVB/SIVBQ master",
        expected={
            "primary_symbol": "SIVBQ",
            "provider_symbol": "SIVBQ.US",
            "action_provider_symbol": "SIVBQ.US",
            "exchange": "PINK",
            "active_from": "2015-01-02",
            "active_to": SIVB_EXACT_LAST_SESSION,
            "name": "SVB Financial Group",
            **common_identity,
        },
    )
    history = frames["symbol_history"]
    touched_history = history.loc[
        history["security_id"].astype(str).eq(SIVB_EXACT_SECURITY_ID)
    ]
    if len(touched_history) != 2:
        raise RuntimeError("Exact SIVB/SIVBQ symbol-history inventory changed.")
    _require_exact_history_row(
        history,
        security_id=SIVB_EXACT_SECURITY_ID,
        symbol="SIVB",
        label="SIVB",
        expected={
            "exchange": "NASDAQ",
            "effective_from": "2015-01-01",
            "effective_to": SIVB_EXACT_OLD_LAST,
            **common_identity,
        },
    )
    _require_exact_history_row(
        history,
        security_id=SIVB_EXACT_SECURITY_ID,
        symbol="SIVBQ",
        label="SIVBQ",
        expected={
            "exchange": "PINK",
            "effective_from": SIVB_EXACT_OTC_START,
            "effective_to": SIVB_EXACT_LAST_SESSION,
            **common_identity,
        },
    )

    try:
        raw_values = json.loads(contents["eodhd_otc_path"])
    except (UnicodeDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError("Exact SIVBQ EOD object is not valid JSON.") from exc
    if not isinstance(raw_values, list) or len(raw_values) != SIVB_EXACT_EOD_ROWS:
        raise RuntimeError("Exact SIVBQ raw EOD inventory changed.")
    raw = pd.DataFrame(raw_values)
    required_raw = {"date", "open", "high", "low", "close", "volume"}
    if not required_raw.issubset(raw.columns):
        raise RuntimeError("Exact SIVBQ raw EOD fields changed.")
    raw["_session"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.sort_values("_session").reset_index(drop=True)
    raw_sessions = tuple(raw["_session"].dt.date.astype(str))
    if (
        raw["_session"].isna().any()
        or len(set(raw_sessions)) != SIVB_EXACT_EOD_ROWS
        or raw_sessions[0] != SIVB_EXACT_OTC_START
        or raw_sessions[-1] != SIVB_EXACT_LAST_SESSION
        or set(raw_sessions) & set(SIVB_EXACT_NON_XNYS_EXCLUSIONS)
        != set(SIVB_EXACT_NON_XNYS_EXCLUSIONS)
    ):
        raise RuntimeError("Exact SIVBQ raw EOD session boundary changed.")
    raw = raw.loc[
        ~raw["_session"].dt.date.astype(str).isin(SIVB_EXACT_NON_XNYS_EXCLUSIONS)
    ].reset_index(drop=True)

    prices = frames["daily_price_raw"].copy()
    prices["_session"] = pd.to_datetime(prices["session"], errors="coerce")
    touched_prices = prices.loc[
        prices["security_id"].astype(str).eq(SIVB_EXACT_SECURITY_ID)
    ].sort_values("_session")
    otc_prices = touched_prices.loc[
        touched_prices["_session"].ge(pd.Timestamp(SIVB_EXACT_OTC_START))
    ].reset_index(drop=True)
    prior_prices = touched_prices.loc[
        touched_prices["_session"].lt(pd.Timestamp(SIVB_EXACT_OTC_START))
    ]
    if (
        touched_prices.empty
        or touched_prices["_session"].isna().any()
        or touched_prices["_session"].duplicated().any()
        or prior_prices.empty
        or _date(prior_prices.iloc[-1]["_session"]) != "2023-03-09"
        or len(otc_prices) != SIVB_EXACT_STORED_OTC_ROWS
        or tuple(otc_prices["_session"].dt.date.astype(str))
        != tuple(raw["_session"].dt.date.astype(str))
        or _date(touched_prices.iloc[-1]["_session"])
        != SIVB_EXACT_LAST_SESSION
    ):
        raise RuntimeError("Exact SIVB/SIVBQ stored price boundary changed.")
    raw_numeric = raw.loc[:, ["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    stored_numeric = otc_prices.loc[
        :, ["open", "high", "low", "close", "volume"]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.allclose(
        stored_numeric.to_numpy(dtype=float),
        raw_numeric.to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
        equal_nan=False,
    ):
        raise RuntimeError("Exact SIVBQ stored prices differ from raw EOD bytes.")
    if (
        otc_prices["source"].astype(str).ne("eodhd_eod").any()
        or otc_prices["source_url"].astype(str).ne(SIVB_EXACT_EOD_URL).any()
        or otc_prices["source_hash"].astype(str).ne(SIVB_EXACT_EOD_SHA256).any()
        or otc_prices["retrieved_at"]
        .astype(str)
        .ne("2026-07-18T14:11:49.785762Z")
        .any()
    ):
        raise RuntimeError("Exact SIVBQ stored price provenance changed.")
    price_keys = _exact_security_session_keys(
        frames["daily_price_raw"], {SIVB_EXACT_SECURITY_ID}, label="SIVB prices"
    )
    factor_keys = _exact_security_session_keys(
        frames["adjustment_factors"],
        {SIVB_EXACT_SECURITY_ID},
        label="SIVB adjustment factors",
    )
    if price_keys != factor_keys:
        raise RuntimeError("Exact SIVB adjustment-factor inventory changed.")

    return _require_exact_repaired_resolution(
        frames["lifecycle_resolutions"],
        expected={
            "candidate_id": lifecycle_candidate_id(
                SIVB_EXACT_SECURITY_ID, SIVB_EXACT_LAST_SESSION
            ),
            "security_id": SIVB_EXACT_SECURITY_ID,
            "symbol": "SIVBQ",
            "last_price_date": SIVB_EXACT_LAST_SESSION,
            "resolution": "applied",
            "event_id": SIVB_EXACT_MARKET_EXIT_EVENT_ID,
            "exception_code": "",
            "exception_reason": "",
            "reviewed_by": "sivb_avp_terminal_transition_planner_v1",
            "reviewed_at": "2026-07-18T15:30:00Z",
            "recheck_after": "",
            "successor_security_id": "",
            "successor_symbol": "",
            "source_url": SIVB_EXACT_SEC_CANCEL_URL,
            "source": "official_market_transition_repair",
            "retrieved_at": "2026-07-18T15:30:00Z",
            "source_hash": SIVB_EXACT_SEC_CANCEL_SHA256,
        },
    )


def _require_exact_ntco_archive_content(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    completed_session: str,
    source: str,
    source_url: str,
    source_hash: str,
    retrieved_at: str,
    content_type: str,
    archive_id: str = "",
) -> bytes:
    """Read one exact NTCO repair artifact and attest its publication row."""

    content, row = _archive_pair_content(
        repository,
        archive,
        source_url=source_url,
        source_hash=source_hash,
        source=source,
    )
    suffix = "json" if content_type == "application/json" else "bin"
    expected = {
        "archive_id": archive_id or source_hash,
        "dataset": source,
        "object_path": f"archives/{completed_session}/{source_hash}.{suffix}.gz",
        "content_type": content_type,
        "effective_date": completed_session,
        "source": source,
        "source_url": source_url,
        "retrieved_at": retrieved_at,
        "source_hash": source_hash,
    }
    for field, wanted in expected.items():
        actual = (
            _date(row.get(field))
            if field == "effective_date"
            else _text(row.get(field))
        )
        target = _date(wanted) if field == "effective_date" else wanted
        if actual != target:
            raise RuntimeError(
                f"Exact NTCO source_archive provenance changed: {source}/{field}"
            )
    return content


def _require_exact_ntco_dividends(actions: pd.DataFrame) -> None:
    for event_id, (effective_date, cash_amount) in (
        NTCO_EXACT_PRESERVED_DIVIDENDS.items()
    ):
        rows = actions.loc[actions["event_id"].astype(str).eq(event_id)]
        if len(rows) != 1:
            raise RuntimeError(
                "Exact NTCO preserved dividend is missing or duplicated: "
                f"{event_id}"
            )
        row = rows.iloc[0]
        expected_text = {
            "security_id": NTCO_EXACT_SECURITY_ID,
            "action_type": "cash_dividend",
            "effective_date": effective_date,
            "ex_date": effective_date,
            "announcement_date": "",
            "record_date": "",
            "payment_date": "",
            "currency": "USD",
            "new_security_id": "",
            "new_symbol": "",
            "source_url": NTCO_EXACT_PRESERVED_DIVIDEND_URL,
            "source_kind": "provider",
            "source": "eodhd_div",
            "retrieved_at": NTCO_EXACT_PRESERVED_DIVIDEND_RETRIEVED_AT,
            "source_hash": NTCO_EXACT_PRESERVED_DIVIDEND_SHA256,
            "metadata": "",
        }
        for field, wanted in expected_text.items():
            actual_text = _text(row.get(field))
            actual = (
                _date(actual_text)
                if field.endswith("_date") and actual_text
                else actual_text
            )
            target = (
                _date(wanted)
                if field.endswith("_date") and wanted
                else wanted
            )
            if actual != target:
                raise RuntimeError(
                    f"Exact NTCO preserved dividend changed: {event_id}/{field}"
                )
        if not _same_optional_number(row.get("cash_amount"), cash_amount) or not (
            _same_optional_number(row.get("ratio"), None)
        ):
            raise RuntimeError(
                f"Exact NTCO preserved dividend economics changed: {event_id}"
            )
        official = row.get("official")
        if not isinstance(official, (bool, np.bool_)) or bool(official):
            raise RuntimeError(
                f"Exact NTCO preserved provider dividend official flag changed: {event_id}"
            )


def _require_exact_ntco_prices(
    prices: pd.DataFrame,
    raw_content: bytes,
) -> pd.DataFrame:
    try:
        raw_rows = json.loads(raw_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Exact NTCOY EOD raw artifact is not JSON.") from exc
    if not isinstance(raw_rows, list) or len(raw_rows) != NTCO_EXACT_EOD_ROWS:
        raise RuntimeError("Exact NTCOY EOD raw row inventory changed.")
    raw_by_session: dict[str, Mapping[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise RuntimeError("Exact NTCOY EOD raw row is malformed.")
        session = _date(raw.get("date"))
        if session in raw_by_session:
            raise RuntimeError("Exact NTCOY EOD raw sessions are duplicated.")
        raw_by_session[session] = raw
    raw_sessions = sorted(raw_by_session)
    if (
        raw_sessions[0] != NTCO_EXACT_TICKER_DATE
        or raw_sessions[-1] != NTCO_EXACT_LAST_SESSION
    ):
        raise RuntimeError("Exact NTCOY EOD raw session boundary changed.")

    own = prices.loc[
        prices["security_id"].astype(str).eq(NTCO_EXACT_SECURITY_ID)
    ].copy()
    own["_session"] = pd.to_datetime(own["session"], errors="coerce")
    if own["_session"].isna().any():
        raise RuntimeError("Exact NTCO stored price session is invalid.")
    tail = own.loc[
        own["_session"].dt.date.astype(str).ge(NTCO_EXACT_TICKER_DATE)
    ].copy()
    stored_sessions = tail["_session"].dt.date.astype(str)
    if (
        len(tail) != NTCO_EXACT_EOD_ROWS
        or stored_sessions.duplicated().any()
        or set(stored_sessions) != set(raw_sessions)
    ):
        raise RuntimeError("Exact NTCOY stored price inventory changed.")
    if own["_session"].dt.date.astype(str).gt(NTCO_EXACT_LAST_SESSION).any():
        raise RuntimeError("Exact NTCOY stored prices extend beyond the BNY boundary.")
    for row in tail.to_dict("records"):
        session = _date(row["_session"])
        raw = raw_by_session[session]
        for field in ("open", "high", "low", "close", "volume"):
            try:
                wanted = float(raw[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Exact NTCOY raw OHLCV is invalid: {session}/{field}"
                ) from exc
            if not _same_optional_number(row.get(field), wanted):
                raise RuntimeError(
                    f"Exact NTCOY stored OHLCV changed: {session}/{field}"
                )
        expected_lineage = {
            "currency": "USD",
            "source": "eodhd_eod",
            "source_url": NTCO_EXACT_EOD_URL,
            "retrieved_at": NTCO_EXACT_EOD_RETRIEVED_AT,
            "source_hash": NTCO_EXACT_EOD_RAW_SHA256,
        }
        if any(_text(row.get(field)) != wanted for field, wanted in expected_lineage.items()):
            raise RuntimeError(
                f"Exact NTCOY stored price provenance changed: {session}"
            )
    return own.drop(columns="_session")


def _require_exact_ntco_factors(
    repository: LocalDatasetRepository,
    release: DataRelease,
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    factors: pd.DataFrame,
) -> None:
    own_factors = factors.loc[
        factors["security_id"].astype(str).eq(NTCO_EXACT_SECURITY_ID)
    ].copy()
    price_keys = _exact_security_session_keys(
        prices,
        {NTCO_EXACT_SECURITY_ID},
        label="NTCO prices",
    )
    factor_keys = _exact_security_session_keys(
        own_factors,
        {NTCO_EXACT_SECURITY_ID},
        label="NTCO adjustment factors",
    )
    if price_keys != factor_keys:
        raise RuntimeError("Exact NTCO adjustment-factor inventory changed.")
    lineages = set(own_factors["source_version"].map(_text))
    if len(lineages) != 1:
        raise RuntimeError("Exact NTCO adjustment-factor lineage is ambiguous.")
    lineage = next(iter(lineages))
    versions = release.dataset_versions
    price_version = _text(versions.get("daily_price_raw"))
    action_version = _text(versions.get("corporate_actions"))
    factor_version = _text(versions.get("adjustment_factors"))
    manifest = None
    metadata: Mapping[str, Any] | None = None
    if factor_version == NTCO_EXACT_MIXED_FACTOR_VERSION:
        expected_lineage = _adjustment_source_version(
            price_version, action_version
        )
    else:
        try:
            manifest = repository.manifest_for_version(
                "adjustment_factors", factor_version
            )
        except Exception as exc:
            raise RuntimeError(
                "Exact NTCO adjustment-factor manifest is unavailable."
            ) from exc
        candidate_metadata = getattr(manifest, "metadata", None)
        if not isinstance(candidate_metadata, Mapping):
            raise RuntimeError(
                "Exact NTCO refinalized adjustment-factor manifest changed."
            )
        metadata = candidate_metadata
        expected_lineage = _manifest_adjustment_source_version(
            metadata,
            price_version,
            action_version,
        )
    if not price_version or not action_version or lineage != expected_lineage:
        raise RuntimeError("Exact NTCO adjustment-factor source version changed.")
    calculated_values = set(own_factors["calculated_at"].map(_text))
    retrieved_values = set(own_factors["retrieved_at"].map(_text))
    if factor_version == NTCO_EXACT_MIXED_FACTOR_VERSION:
        if (
            price_version != NTCO_EXACT_MIXED_PRICE_VERSION
            or action_version != NTCO_EXACT_MIXED_ACTION_VERSION
            or calculated_values != {NTCO_EXACT_REVIEWED_AT}
            or retrieved_values != {NTCO_EXACT_REVIEWED_AT}
        ):
            raise RuntimeError("Exact NTCO mixed adjustment-factor binding changed.")
    else:
        if metadata is None or any(
            _text(metadata.get(field)) != wanted
            for field, wanted in {
                "source_daily_price_version": price_version,
                "source_corporate_actions_version": action_version,
                "source_version": expected_lineage,
            }.items()
        ):
            raise RuntimeError(
                "Exact NTCO refinalized adjustment-factor manifest changed."
            )
        if (
            len(calculated_values) != 1
            or len(retrieved_values) != 1
            or not next(iter(calculated_values))
            or calculated_values != retrieved_values
        ):
            raise RuntimeError(
                "Exact NTCO refinalized adjustment-factor timestamps changed."
            )
    if (
        own_factors["source_version"].map(_text).ne(lineage).any()
        or own_factors["source_hash"].map(_text).ne(lineage).any()
        or own_factors["source"].map(_text).ne("derived").any()
    ):
        raise RuntimeError("Exact NTCO adjustment-factor provenance changed.")
    expected = build_adjustment_factors(
        prices,
        actions,
        source_version=lineage,
    )
    expected_rows = {
        _date(row["session"]): row for row in expected.to_dict("records")
    }
    for row in own_factors.to_dict("records"):
        session = _date(row["session"])
        wanted = expected_rows.get(session)
        if wanted is None or any(
            not _same_optional_number(row.get(field), wanted.get(field))
            for field in ("split_factor", "total_return_factor")
        ):
            raise RuntimeError(
                f"Exact NTCO adjustment-factor economics changed: {session}"
            )


def _preserve_exact_ntco_resolution(
    candidate: LifecycleCandidate,
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    if (
        candidate.security_id != NTCO_EXACT_SECURITY_ID
        or candidate.symbol.upper() != NTCO_EXACT_NEW_SYMBOL
        or _date(candidate.last_price_date) != NTCO_EXACT_LAST_SESSION
        or _date(candidate.active_to) != NTCO_EXACT_TERMINAL_DATE
        or _date(release.completed_session) != "2026-07-15"
    ):
        raise RuntimeError("Exact NTCOY lifecycle candidate boundary changed.")
    required = {
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "corporate_actions",
        "adjustment_factors",
        "source_archive",
        "lifecycle_resolutions",
    }
    missing = sorted(required - set(frames))
    if missing:
        raise RuntimeError(f"Exact NTCO preservation inputs are missing: {missing}")

    actions = frames["corporate_actions"]
    ticker = _require_exact_repaired_action(
        actions,
        event_id=NTCO_EXACT_TICKER_EVENT_ID,
        expected_text={
            "event_id": NTCO_EXACT_TICKER_EVENT_ID,
            "security_id": NTCO_EXACT_SECURITY_ID,
            "action_type": "ticker_change",
            "effective_date": NTCO_EXACT_TICKER_DATE,
            "ex_date": NTCO_EXACT_TICKER_DATE,
            "announcement_date": "2024-02-09",
            "record_date": "",
            "payment_date": "",
            "currency": "USD",
            "new_security_id": NTCO_EXACT_SECURITY_ID,
            "new_symbol": NTCO_EXACT_NEW_SYMBOL,
            "source_url": NTCO_EXACT_OCC_URL,
            "source_kind": "clearing_and_exchange_notices",
            "source": "official_ntco_ntcoy_identity",
            "retrieved_at": NTCO_EXACT_RETRIEVED_AT,
            "source_hash": NTCO_EXACT_IDENTITY_SHA256,
        },
        cash_amount=None,
        ratio=None,
        metadata={
            "cboe_source_url": NTCO_EXACT_CBOE_URL,
            "occ_source_url": NTCO_EXACT_OCC_URL,
            "official_destination_market": "Other-OTC",
            "canonical_exchange": "OTC",
            "cusip": "63884N108",
            "deliverable": "100 American Depositary Shares",
        },
    )
    terminal = _require_exact_repaired_action(
        actions,
        event_id=NTCO_EXACT_TERMINAL_EVENT_ID,
        expected_text={
            "event_id": NTCO_EXACT_TERMINAL_EVENT_ID,
            "security_id": NTCO_EXACT_SECURITY_ID,
            "action_type": "delisting",
            "effective_date": NTCO_EXACT_TERMINAL_DATE,
            "ex_date": NTCO_EXACT_TERMINAL_DATE,
            "announcement_date": "2024-08-26",
            "record_date": "",
            "payment_date": NTCO_EXACT_TERMINAL_DATE,
            "currency": "USD",
            "new_security_id": "",
            "new_symbol": "",
            "source_url": NTCO_EXACT_BNY_CASH_URL,
            "source_kind": "depositary_corporate_action_notice",
            "source": "official_ntcoy_cash_termination",
            "retrieved_at": NTCO_EXACT_RETRIEVED_AT,
            "source_hash": NTCO_EXACT_TERMINAL_SHA256,
        },
        cash_amount=NTCO_EXACT_TERMINAL_CASH,
        ratio=None,
        metadata={
            "mandatory_exchange": True,
            "gross_rate_per_ads": "5.043659",
            "cancellation_fee_per_ads": "0",
            "net_rate_per_ads": "5.043659",
            "ads_to_underlying_ratio": "1:2",
        },
    )
    _require_exact_ntco_dividends(actions)
    action_dates = pd.to_datetime(actions["effective_date"], errors="coerce")
    target_tail = actions.loc[
        actions["security_id"].astype(str).eq(NTCO_EXACT_SECURITY_ID)
        & action_dates.ge(pd.Timestamp(NTCO_EXACT_TICKER_DATE))
    ]
    expected_tail_ids = {
        NTCO_EXACT_TICKER_EVENT_ID,
        NTCO_EXACT_TERMINAL_EVENT_ID,
        *NTCO_EXACT_PRESERVED_DIVIDENDS,
    }
    if len(target_tail) != 4 or set(target_tail["event_id"].astype(str)) != expected_tail_ids:
        raise RuntimeError("Exact NTCO post-transition action inventory changed.")

    master = _require_exact_identity_row(
        frames["security_master"],
        security_id=NTCO_EXACT_SECURITY_ID,
        expected={
            "primary_symbol": NTCO_EXACT_NEW_SYMBOL,
            "provider_symbol": "NTCOY.US",
            "exchange": "OTC",
            "active_from": NTCO_EXACT_ACTIVE_FROM,
            "active_to": NTCO_EXACT_TERMINAL_DATE,
            "source": "official_ntco_ntcoy_identity",
            "source_url": NTCO_EXACT_OCC_URL,
            "retrieved_at": NTCO_EXACT_RETRIEVED_AT,
            "source_hash": NTCO_EXACT_IDENTITY_SHA256,
        },
        label="NTCOY master",
    )
    if "action_provider_symbol" in frames["security_master"].columns and (
        _text(master.get("action_provider_symbol")) != "NTCOY.US"
    ):
        raise RuntimeError("Exact NTCOY action provider symbol changed.")
    history = frames["symbol_history"].loc[
        frames["symbol_history"]["security_id"]
        .astype(str)
        .eq(NTCO_EXACT_SECURITY_ID)
    ]
    if set(history["symbol"].astype(str).str.upper()) != {
        NTCO_EXACT_OLD_SYMBOL,
        NTCO_EXACT_NEW_SYMBOL,
    } or len(history) != 2:
        raise RuntimeError("Exact NTCO/NTCOY symbol-history inventory changed.")
    for symbol, expected in {
        NTCO_EXACT_OLD_SYMBOL: {
            "exchange": "NYSE",
            "effective_from": NTCO_EXACT_ACTIVE_FROM,
            "effective_to": NTCO_EXACT_OLD_SYMBOL_END,
        },
        NTCO_EXACT_NEW_SYMBOL: {
            "exchange": "OTC",
            "effective_from": NTCO_EXACT_TICKER_DATE,
            "effective_to": NTCO_EXACT_LAST_SESSION,
        },
    }.items():
        _require_exact_history_row(
            history,
            security_id=NTCO_EXACT_SECURITY_ID,
            symbol=symbol,
            expected={
                **expected,
                "source": "official_ntco_ntcoy_identity",
                "source_url": NTCO_EXACT_OCC_URL,
                "retrieved_at": NTCO_EXACT_RETRIEVED_AT,
                "source_hash": NTCO_EXACT_IDENTITY_SHA256,
            },
            label=f"NTCO {symbol}",
        )

    archive = frames["source_archive"]
    identity_content = _require_exact_ntco_archive_content(
        repository,
        archive,
        completed_session="2026-07-15",
        source="official_ntco_ntcoy_identity",
        source_url=NTCO_EXACT_OCC_URL,
        source_hash=NTCO_EXACT_IDENTITY_SHA256,
        retrieved_at=NTCO_EXACT_RETRIEVED_AT,
        content_type="application/json",
    )
    terminal_content = _require_exact_ntco_archive_content(
        repository,
        archive,
        completed_session="2026-07-15",
        source="official_ntcoy_cash_termination",
        source_url=NTCO_EXACT_BNY_CASH_URL,
        source_hash=NTCO_EXACT_TERMINAL_SHA256,
        retrieved_at=NTCO_EXACT_RETRIEVED_AT,
        content_type="application/json",
    )
    decision_content = _require_exact_ntco_archive_content(
        repository,
        archive,
        completed_session="2026-07-15",
        source="reviewed_ntco_ntcoy_transition_decision",
        source_url=NTCO_EXACT_DECISION_URL,
        source_hash=NTCO_EXACT_DECISION_SHA256,
        retrieved_at=NTCO_EXACT_RETRIEVED_AT,
        content_type="application/json",
    )
    eod_content = _require_exact_ntco_archive_content(
        repository,
        archive,
        completed_session="2026-07-15",
        source="eodhd_eod",
        source_url=NTCO_EXACT_EOD_URL,
        source_hash=NTCO_EXACT_EOD_RAW_SHA256,
        retrieved_at=NTCO_EXACT_EOD_RETRIEVED_AT,
        content_type="application/json",
    )
    dividend_content = _require_exact_ntco_archive_content(
        repository,
        archive,
        completed_session="2026-07-15",
        source="eodhd_div",
        source_url=NTCO_EXACT_DIV_URL,
        source_hash=NTCO_EXACT_DIV_RAW_SHA256,
        retrieved_at=NTCO_EXACT_EOD_RETRIEVED_AT,
        content_type="application/json",
    )
    splits_content = _require_exact_ntco_archive_content(
        repository,
        archive,
        completed_session="2026-07-15",
        source="eodhd_splits",
        source_url=NTCO_EXACT_SPLITS_URL,
        source_hash=NTCO_EXACT_SPLITS_RAW_SHA256,
        retrieved_at=NTCO_EXACT_EOD_RETRIEVED_AT,
        content_type="application/json",
        archive_id=NTCO_EXACT_SPLITS_ARCHIVE_ID,
    )
    try:
        rejected_dividends = json.loads(dividend_content)
        observed_splits = json.loads(splits_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Exact NTCOY rejected provider action raw is not JSON.") from exc
    if not isinstance(rejected_dividends, list) or len(rejected_dividends) != 2:
        raise RuntimeError("Exact NTCOY rejected dividend raw inventory changed.")
    if observed_splits != []:
        raise RuntimeError("Exact NTCOY splits raw is no longer empty.")
    raw_official_specs = (
        (
            "official_cboe",
            NTCO_EXACT_CBOE_URL,
            NTCO_EXACT_CBOE_RAW_SHA256,
            NTCO_EXACT_CBOE_RAW_RETRIEVED_AT,
            "application/pdf",
        ),
        (
            "official_occ",
            NTCO_EXACT_OCC_URL,
            NTCO_EXACT_OCC_RAW_SHA256,
            NTCO_EXACT_OCC_RAW_RETRIEVED_AT,
            "text/html",
        ),
        (
            "official_bny",
            NTCO_EXACT_BNY_CASH_URL,
            NTCO_EXACT_BNY_CASH_RAW_SHA256,
            NTCO_EXACT_BNY_CASH_RAW_RETRIEVED_AT,
            "application/pdf",
        ),
        (
            "official_bny_termination",
            NTCO_EXACT_BNY_TERMINATION_URL,
            NTCO_EXACT_BNY_TERMINATION_RAW_SHA256,
            NTCO_EXACT_BNY_TERMINATION_RETRIEVED_AT,
            "application/pdf",
        ),
        (
            "official_bny_books_closed",
            NTCO_EXACT_BNY_BOOKS_CLOSED_URL,
            NTCO_EXACT_BNY_BOOKS_CLOSED_RAW_SHA256,
            NTCO_EXACT_BNY_BOOKS_CLOSED_RETRIEVED_AT,
            "application/pdf",
        ),
    )
    for source, url, digest, retrieved_at, content_type in raw_official_specs:
        content = _require_exact_ntco_archive_content(
            repository,
            archive,
            completed_session="2026-07-15",
            source=source,
            source_url=url,
            source_hash=digest,
            retrieved_at=retrieved_at,
            content_type=content_type,
        )
        if not content.startswith(b"%PDF-") or b"%%EOF" not in content[-32:]:
            raise RuntimeError(f"Exact NTCO official raw is not an intact PDF: {source}")

    expected_identity = {
        "schema": "official_ntco_ntcoy_identity/v1",
        "security_id": NTCO_EXACT_SECURITY_ID,
        "effective_date": NTCO_EXACT_TICKER_DATE,
        "old_symbol": NTCO_EXACT_OLD_SYMBOL,
        "new_symbol": NTCO_EXACT_NEW_SYMBOL,
        "canonical_exchange": "OTC",
        "official_destination_market": "Other-OTC",
        "cusip": "63884N108",
        "deliverable": "100 American Depositary Shares",
        "cboe_raw_sha256": NTCO_EXACT_CBOE_RAW_SHA256,
        "occ_raw_sha256": NTCO_EXACT_OCC_RAW_SHA256,
    }
    expected_terminal = {
        "schema": "official_ntcoy_cash_termination/v1",
        "security_id": NTCO_EXACT_SECURITY_ID,
        "action_type": "delisting",
        "effective_date": NTCO_EXACT_TERMINAL_DATE,
        "cash_amount": "5.043659",
        "currency": "USD",
        "ads_to_underlying_ratio": "1:2",
        "fee_per_ads": "0",
        "bny_raw_sha256": NTCO_EXACT_BNY_CASH_RAW_SHA256,
    }
    try:
        identity_value = json.loads(identity_content)
        terminal_value = json.loads(terminal_content)
        decision_value = json.loads(decision_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Exact NTCO reviewed extraction is not JSON.") from exc
    if identity_value != expected_identity or terminal_value != expected_terminal:
        raise RuntimeError("Exact NTCO reviewed extraction content changed.")
    if not isinstance(decision_value, Mapping) or any(
        decision_value.get(field) != wanted
        for field, wanted in {
            "schema": "reviewed_ntco_ntcoy_transition_decision/v1",
            "security_id": NTCO_EXACT_SECURITY_ID,
            "decision_mode": "price_identity_terminal_only",
            "provider_price_raw_sha256": NTCO_EXACT_EOD_RAW_SHA256,
            "provider_splits_raw_sha256": NTCO_EXACT_SPLITS_RAW_SHA256,
            "provider_dividend_economics_accepted": False,
            "provider_dividend_raw_decision": (
                "archive_exact_ntcoy_raw_reject_economics_preserve_ntco_actions"
            ),
            "provider_dividend_raw_sha256": NTCO_EXACT_DIV_RAW_SHA256,
            "maximum_absolute_sensitivity_usd_per_ads": "0.01585",
        }.items()
    ):
        raise RuntimeError("Exact NTCO price-only decision audit changed.")

    own_prices = _require_exact_ntco_prices(frames["daily_price_raw"], eod_content)
    own_actions = actions.loc[
        actions["security_id"].astype(str).eq(NTCO_EXACT_SECURITY_ID)
    ]
    _require_exact_ntco_factors(
        repository,
        release,
        own_prices,
        own_actions,
        frames["adjustment_factors"],
    )
    if (
        _text(ticker.get("retrieved_at")) != NTCO_EXACT_RETRIEVED_AT
        or _text(terminal.get("retrieved_at")) != NTCO_EXACT_RETRIEVED_AT
    ):
        raise RuntimeError("Exact NTCO action retrieval binding changed.")

    return _require_exact_repaired_resolution(
        frames["lifecycle_resolutions"],
        expected={
            "candidate_id": lifecycle_candidate_id(
                NTCO_EXACT_SECURITY_ID, NTCO_EXACT_LAST_SESSION
            ),
            "security_id": NTCO_EXACT_SECURITY_ID,
            "symbol": NTCO_EXACT_NEW_SYMBOL,
            "last_price_date": NTCO_EXACT_LAST_SESSION,
            "resolution": "applied",
            "event_id": NTCO_EXACT_TERMINAL_EVENT_ID,
            "exception_code": "",
            "exception_reason": "",
            "reviewed_by": NTCO_EXACT_REVIEWED_BY,
            "reviewed_at": NTCO_EXACT_REVIEWED_AT,
            "recheck_after": "",
            "successor_security_id": "",
            "successor_symbol": "",
            "source_url": NTCO_EXACT_BNY_CASH_URL,
            "source": "official_ntcoy_cash_termination",
            "retrieved_at": NTCO_EXACT_RETRIEVED_AT,
            "source_hash": NTCO_EXACT_TERMINAL_SHA256,
        },
    )


def _preserve_exact_market_date_transition_resolution(
    candidate: LifecycleCandidate,
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    key = _key(candidate.security_id, candidate.last_price_date)
    spec = EXACT_REVIEWED_MARKET_DATE_TRANSITIONS.get(key)
    if spec is None:
        raise RuntimeError("Reviewed market-date transition candidate is unpinned.")
    expected_candidate = spec["candidate"]
    candidate_values = {
        "security_id": candidate.security_id,
        "symbol": candidate.symbol,
        "name": candidate.name,
        "exchange": candidate.exchange,
        "last_price_date": _date(candidate.last_price_date),
        "active_to": _date(candidate.active_to),
    }
    if any(
        _text(candidate_values[field]) != _text(wanted)
        for field, wanted in expected_candidate.items()
    ):
        raise RuntimeError(
            "Exact reviewed market-date transition candidate boundary changed."
        )
    if _date(release.completed_session) != "2026-07-15":
        raise RuntimeError(
            "Exact reviewed market-date transition release binding changed."
        )
    required = {
        "security_master",
        "symbol_history",
        "corporate_actions",
        "source_archive",
        "lifecycle_resolutions",
    }
    missing = sorted(required - set(frames))
    if missing:
        raise RuntimeError(
            "Exact reviewed market-date transition inputs are missing: "
            + ", ".join(missing)
        )

    action_spec = dict(spec["action"])
    cash_amount = action_spec.pop("cash_amount")
    ratio = action_spec.pop("ratio")
    action = _require_exact_repaired_action(
        frames["corporate_actions"],
        event_id=spec["event_id"],
        expected_text={"event_id": spec["event_id"], **action_spec},
        cash_amount=cash_amount,
        ratio=ratio,
        metadata=spec["metadata"],
    )
    lifecycle_types = {
        "cash_merger",
        "stock_merger",
        "spinoff",
        "ticker_change",
        "delisting",
    }
    own_lifecycle = frames["corporate_actions"].loc[
        frames["corporate_actions"]["security_id"]
        .astype(str)
        .eq(candidate.security_id)
        & frames["corporate_actions"]["action_type"]
        .astype(str)
        .str.lower()
        .isin(lifecycle_types)
    ]
    if (
        len(own_lifecycle) != 1
        or set(own_lifecycle["event_id"].astype(str)) != {spec["event_id"]}
        or frames["corporate_actions"]["event_id"]
        .astype(str)
        .eq(spec["rejected_legal_date_event_id"])
        .any()
    ):
        raise RuntimeError(
            "Exact reviewed market-date transition action inventory changed."
        )

    for expected_master in spec["master"]:
        expected = dict(expected_master)
        security_id = expected.pop("security_id")
        _require_exact_identity_row(
            frames["security_master"],
            security_id=security_id,
            expected=expected,
            label=f"{expected_master['primary_symbol']} market-date master",
        )
    for expected_history in spec["history"]:
        expected = dict(expected_history)
        security_id = expected.pop("security_id")
        symbol = expected.pop("symbol")
        _require_exact_history_row(
            frames["symbol_history"],
            security_id=security_id,
            symbol=symbol,
            expected=expected,
            label=f"{symbol} market-date",
        )

    source_url = _text(action.get("source_url"))
    source_hash = _text(action.get("source_hash")).lower()
    retrieved_at = _text(action.get("retrieved_at"))
    content, archive_row = _archive_pair_content(
        repository,
        frames["source_archive"],
        source_url=source_url,
        source_hash=source_hash,
        source="sec_edgar_filing",
    )
    expected_archive = {
        "archive_id": source_hash,
        "dataset": "sec_edgar_filing",
        "object_path": f"archives/2026-07-15/{source_hash}.txt.gz",
        "content_type": "text/plain",
        "effective_date": "2026-07-15",
        "source": "sec_edgar_filing",
        "source_url": source_url,
        "retrieved_at": retrieved_at,
        "source_hash": source_hash,
    }
    for field, wanted in expected_archive.items():
        actual = (
            _date(archive_row.get(field))
            if field == "effective_date"
            else _text(archive_row.get(field))
        )
        if actual != wanted:
            raise RuntimeError(
                "Exact reviewed market-date archive row changed: "
                f"{candidate.symbol}/{field}"
            )
    if len(content) != int(spec["archive_bytes"]):
        raise RuntimeError(
            "Exact reviewed market-date archive payload size changed: "
            f"{candidate.symbol}"
        )

    return _require_exact_repaired_resolution(
        frames["lifecycle_resolutions"],
        expected=spec["resolution"],
    )


def _preserve_exact_short_terminal_market_resolution(
    candidate: LifecycleCandidate,
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    key = _key(candidate.security_id, candidate.last_price_date)
    spec = EXACT_SHORT_TERMINAL_MARKET_TRANSITIONS.get(key)
    if spec is None:
        raise RuntimeError("Short-terminal market transition candidate is unpinned.")
    expected_candidate = spec["candidate"]
    observed_candidate = {
        "symbol": candidate.symbol,
        "name": candidate.name,
        "exchange": candidate.exchange,
        "active_to": _date(candidate.active_to),
        "index_remove_dates": tuple(candidate.index_remove_dates),
    }
    if any(
        observed_candidate[field] != wanted
        for field, wanted in expected_candidate.items()
    ):
        raise RuntimeError(
            "Exact short-terminal market transition candidate changed."
        )
    if _date(release.completed_session) != "2026-07-15":
        raise RuntimeError(
            "Exact short-terminal market transition release binding changed."
        )
    required = {
        "daily_price_raw",
        "adjustment_factors",
        "corporate_actions",
        "source_archive",
        "lifecycle_resolutions",
    }
    missing = sorted(required - set(frames))
    if missing:
        raise RuntimeError(
            "Exact short-terminal market transition inputs are missing: "
            + ", ".join(missing)
        )

    action_spec = dict(spec["action"])
    action = _require_exact_prior_terminal_action(
        frames["corporate_actions"],
        {
            "event_id": spec["event_id"],
            "security_id": candidate.security_id,
            **action_spec,
        },
        label=candidate.symbol,
    )
    lifecycle_types = {
        "cash_merger",
        "stock_merger",
        "spinoff",
        "ticker_change",
        "delisting",
    }
    own_lifecycle = frames["corporate_actions"].loc[
        frames["corporate_actions"]["security_id"]
        .astype(str)
        .eq(candidate.security_id)
        & frames["corporate_actions"]["action_type"]
        .astype(str)
        .str.lower()
        .isin(lifecycle_types)
    ]
    if (
        len(own_lifecycle) != 1
        or set(own_lifecycle["event_id"].astype(str)) != {spec["event_id"]}
        or frames["corporate_actions"]["event_id"]
        .astype(str)
        .eq(spec["rejected_event_id"])
        .any()
    ):
        raise RuntimeError(
            "Exact short-terminal market transition action inventory changed."
        )

    price_sessions = pd.to_datetime(
        frames["daily_price_raw"].loc[
            frames["daily_price_raw"]["security_id"]
            .astype(str)
            .eq(candidate.security_id),
            "session",
        ],
        errors="coerce",
    ).dropna()
    factor_sessions = pd.to_datetime(
        frames["adjustment_factors"].loc[
            frames["adjustment_factors"]["security_id"]
            .astype(str)
            .eq(candidate.security_id),
            "session",
        ],
        errors="coerce",
    ).dropna()
    if (
        price_sessions.empty
        or price_sessions.dt.date.astype(str).max() != _date(candidate.last_price_date)
        or set(price_sessions.dt.date.astype(str))
        != set(factor_sessions.dt.date.astype(str))
    ):
        raise RuntimeError(
            "Exact short-terminal market transition price/factor boundary changed."
        )

    source_url = _text(action.get("source_url"))
    source_hash = _text(action.get("source_hash")).lower()
    _content, archive_row = _archive_pair_content(
        repository,
        frames["source_archive"],
        source_url=source_url,
        source_hash=source_hash,
        source="sec_edgar_filing",
    )
    expected_archive = {
        "archive_id": source_hash,
        "dataset": "sec_edgar_filing",
        "object_path": f"archives/2026-07-15/{source_hash}.txt.gz",
        "content_type": "text/plain",
        "effective_date": "2026-07-15",
        "source": "sec_edgar_filing",
        "source_url": source_url,
        "retrieved_at": _text(action.get("retrieved_at")),
        "source_hash": source_hash,
    }
    for field, wanted in expected_archive.items():
        actual = (
            _date(archive_row.get(field))
            if field == "effective_date"
            else _text(archive_row.get(field))
        )
        if actual != wanted:
            raise RuntimeError(
                "Exact short-terminal market transition archive changed: "
                f"{candidate.symbol}/{field}"
            )

    return _require_exact_repaired_resolution(
        frames["lifecycle_resolutions"],
        expected={
            "candidate_id": lifecycle_candidate_id(
                candidate.security_id, candidate.last_price_date
            ),
            "security_id": candidate.security_id,
            "symbol": candidate.symbol,
            "last_price_date": candidate.last_price_date,
            "resolution": "applied",
            "event_id": spec["event_id"],
            "exception_code": "",
            "exception_reason": "",
            "reviewed_by": "short_terminal_boundary_repair_v1",
            "reviewed_at": "2026-07-19T00:00:00Z",
            "recheck_after": "",
            "successor_security_id": _text(action.get("new_security_id")),
            "successor_symbol": _text(action.get("new_symbol")),
            "source_url": source_url,
            "source": "short_terminal_boundary_repair",
            "retrieved_at": "2026-07-19T00:00:00Z",
            "source_hash": source_hash,
        },
    )


def _require_exact_prior_terminal_action(
    actions: pd.DataFrame,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Validate one pre-finalizer action whose reviewed metadata is empty."""

    required = set(expected) | {"official", "metadata"}
    missing = sorted(required - set(actions.columns))
    if missing:
        raise RuntimeError(
            f"Exact prior terminal action lacks columns: {label}/{missing}"
        )
    event_id = _text(expected["event_id"])
    rows = actions.loc[actions["event_id"].astype(str).eq(event_id)]
    if len(rows) != 1:
        raise RuntimeError(
            f"Exact prior terminal action is missing or duplicated: {label}/{event_id}"
        )
    row = rows.iloc[0].to_dict()
    for field, wanted in expected.items():
        if field in {"cash_amount", "ratio"}:
            if not _same_optional_number(row.get(field), wanted):
                raise RuntimeError(
                    f"Exact prior terminal action economics changed: {label}/{field}"
                )
            continue
        actual_text = _text(row.get(field))
        wanted_text = _text(wanted)
        if field in {
            "effective_date",
            "ex_date",
            "announcement_date",
            "record_date",
            "payment_date",
        }:
            actual = _date(actual_text) if actual_text else ""
            target = _date(wanted_text) if wanted_text else ""
        else:
            actual, target = actual_text, wanted_text
        if actual != target:
            raise RuntimeError(
                f"Exact prior terminal action field changed: {label}/{field}"
            )
    official = row.get("official")
    if not isinstance(official, (bool, np.bool_)) or not bool(official):
        raise RuntimeError(f"Exact prior terminal action is not official: {label}")
    if _text(row.get("metadata")):
        raise RuntimeError(
            f"Exact prior terminal action metadata changed: {label}"
        )
    return row


def _prior_terminal_resolution_matches(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    for field, wanted in expected.items():
        actual_text = _text(row.get(field))
        wanted_text = _text(wanted)
        actual = (
            _date(actual_text)
            if field == "last_price_date" and actual_text
            else actual_text
        )
        target = (
            _date(wanted_text)
            if field == "last_price_date" and wanted_text
            else wanted_text
        )
        if actual != target:
            return False
    return True


def _require_exact_prior_terminal_price(
    prices: pd.DataFrame,
    *,
    security_id: str,
    expected: Mapping[str, Any],
    label: str,
    terminal: bool,
) -> dict[str, Any]:
    required = {
        "security_id", "session", "open", "high", "low", "close", "volume",
        "currency", "source", "retrieved_at", "source_hash",
    }
    missing = sorted(required - set(prices.columns))
    if missing:
        raise RuntimeError(f"Exact prior terminal prices lack columns: {missing}")
    own = prices.loc[prices["security_id"].astype(str).eq(security_id)]
    own_sessions = pd.to_datetime(own["session"], errors="coerce")
    if own.empty or own_sessions.isna().any():
        raise RuntimeError(f"Exact prior terminal price path changed: {label}")
    session = _date(expected["session"])
    rows = own.loc[own_sessions.dt.date.astype(str).eq(session)]
    if len(rows) != 1:
        raise RuntimeError(
            f"Exact prior terminal price boundary is missing or duplicated: {label}"
        )
    row = rows.iloc[0].to_dict()
    actual_ohlcv = tuple(
        float(row[field]) for field in ("open", "high", "low", "close", "volume")
    )
    if any(
        abs(actual - float(wanted)) > 1e-9
        for actual, wanted in zip(actual_ohlcv, expected["ohlcv"], strict=True)
    ):
        raise RuntimeError(f"Exact prior terminal OHLCV changed: {label}")
    for field, wanted in {
        "currency": "USD",
        "source": "eodhd_eod",
        "retrieved_at": expected["retrieved_at"],
        "source_hash": expected["source_hash"],
    }.items():
        if _text(row.get(field)) != _text(wanted):
            raise RuntimeError(
                f"Exact prior terminal price provenance changed: {label}/{field}"
            )
    if terminal and own_sessions.max().date().isoformat() != session:
        raise RuntimeError(f"Exact prior terminal last session changed: {label}")
    return row


def _preserved_exact_prior_terminal_transitions(
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
    candidates: Iterable[LifecycleCandidate],
) -> tuple[
    dict[str, dict[str, Any]], tuple[str, ...], pd.DataFrame, pd.DataFrame
]:
    """Restore four reviewed resolutions and drop only four exact duplicates."""

    candidate_values = tuple(candidates)
    exact_keys = set(EXACT_PRIOR_TERMINAL_TRANSITIONS)
    known_event_ids = {
        _text(event_id)
        for spec in EXACT_PRIOR_TERMINAL_TRANSITIONS.values()
        for event_id in (
            spec["action"]["event_id"],
            spec["superseded_action"]["event_id"],
        )
    }
    candidate_relevant = any(
        _key(candidate.security_id, candidate.last_price_date) in exact_keys
        for candidate in candidate_values
    )
    action_relevant = _frame_contains_exact_value(
        frames.get("corporate_actions"), "event_id", known_event_ids
    )
    resolution_relevant = _frame_contains_exact_value(
        frames.get("lifecycle_resolutions"), "event_id", known_event_ids
    )
    if not (candidate_relevant or action_relevant or resolution_relevant):
        return (
            {},
            (),
            frames.get("corporate_actions", pd.DataFrame()).copy(deep=True),
            frames.get("source_archive", pd.DataFrame()).copy(deep=True),
        )

    required = {
        "security_master", "symbol_history", "daily_price_raw",
        "corporate_actions", "source_archive", "lifecycle_resolutions",
    }
    missing = sorted(required - set(frames))
    if missing:
        raise RuntimeError(
            "Exact prior terminal preservation inputs are missing: "
            + ", ".join(missing)
        )
    if _date(release.completed_session) != "2026-07-15":
        raise RuntimeError("Exact prior terminal release binding changed.")

    candidate_rows: dict[str, list[LifecycleCandidate]] = {}
    for candidate in candidate_values:
        candidate_rows.setdefault(
            _key(candidate.security_id, candidate.last_price_date), []
        ).append(candidate)
    actions = frames["corporate_actions"]
    normalized_archive = frames["source_archive"].copy(deep=True)
    resolution_frame = frames["lifecycle_resolutions"]
    lifecycle_types = {
        "cash_merger", "stock_merger", "spinoff", "ticker_change", "delisting",
    }
    restored: dict[str, dict[str, Any]] = {}
    markers: list[str] = []
    remove_event_ids: set[str] = set()

    for key, spec in EXACT_PRIOR_TERMINAL_TRANSITIONS.items():
        corrected_id = _text(spec["action"]["event_id"])
        superseded_id = _text(spec["superseded_action"]["event_id"])
        relevant = bool(candidate_rows.get(key)) or _frame_contains_exact_value(
            actions, "event_id", (corrected_id, superseded_id)
        ) or _frame_contains_exact_value(
            resolution_frame, "event_id", (corrected_id, superseded_id)
        )
        if not relevant:
            continue
        rows = candidate_rows.get(key, [])
        if len(rows) != 1:
            raise RuntimeError(
                "Exact prior terminal candidate is missing or duplicated: " + key
            )
        candidate = rows[0]
        expected_candidate = spec["candidate"]
        candidate_values = {
            "security_id": candidate.security_id,
            "symbol": candidate.symbol,
            "name": candidate.name,
            "exchange": candidate.exchange,
            "last_price_date": _date(candidate.last_price_date),
            "active_to": _date(candidate.active_to),
            "index_remove_dates": tuple(candidate.index_remove_dates),
        }
        if any(
            candidate_values[field] != wanted
            for field, wanted in expected_candidate.items()
        ):
            raise RuntimeError(
                f"Exact prior terminal candidate changed: {expected_candidate['symbol']}"
            )

        symbol = _text(expected_candidate["symbol"])
        corrected = _require_exact_prior_terminal_action(
            actions, spec["action"], label=f"{symbol}/reviewed"
        )
        superseded_expected = {
            **spec["action"],
            **spec["superseded_action"],
        }
        superseded_rows = actions.loc[
            actions["event_id"].astype(str).eq(superseded_id)
        ]
        if not superseded_rows.empty:
            _require_exact_prior_terminal_action(
                actions,
                superseded_expected,
                label=f"{symbol}/superseded",
            )
            remove_event_ids.add(superseded_id)
        own_lifecycle = actions.loc[
            actions["security_id"].astype(str).eq(candidate.security_id)
            & actions["action_type"].astype(str).str.lower().isin(lifecycle_types)
        ]
        allowed_ids = {corrected_id} | (
            {superseded_id} if not superseded_rows.empty else set()
        )
        if (
            len(own_lifecycle) != len(allowed_ids)
            or set(own_lifecycle["event_id"].astype(str)) != allowed_ids
        ):
            raise RuntimeError(
                f"Exact prior terminal action inventory changed: {symbol}"
            )

        _require_exact_identity_row(
            frames["security_master"],
            security_id=candidate.security_id,
            expected=spec["target_master"],
            label=f"{symbol} terminal master",
        )
        _require_exact_history_row(
            frames["symbol_history"],
            security_id=candidate.security_id,
            symbol=symbol,
            expected=spec["target_history"],
            label=f"{symbol} terminal",
        )
        successor_id = _text(spec["action"]["new_security_id"])
        _require_exact_identity_row(
            frames["security_master"],
            security_id=successor_id,
            expected=spec["successor_master"],
            label=f"{symbol} successor master",
        )
        terminal_price = _require_exact_prior_terminal_price(
            frames["daily_price_raw"],
            security_id=candidate.security_id,
            expected=spec["target_price"],
            label=symbol,
            terminal=True,
        )
        successor_price = _require_exact_prior_terminal_price(
            frames["daily_price_raw"],
            security_id=successor_id,
            expected=spec["successor_price"],
            label=f"{symbol} successor",
            terminal=False,
        )
        old_close = float(terminal_price["close"])
        implied = (
            float(successor_price["close"])
            if _text(corrected["action_type"]).lower() == "ticker_change"
            else float(corrected.get("ratio") or 0.0)
            * float(successor_price["close"])
            + float(corrected.get("cash_amount") or 0.0)
        )
        if abs(old_close - implied) / max(abs(old_close), abs(implied), 1e-12) > 0.20:
            raise RuntimeError(
                f"Exact prior terminal economic boundary changed: {symbol}"
            )

        source_url = _text(spec["action"]["source_url"])
        source_hash = _text(spec["action"]["source_hash"]).lower()
        content, archive_row = _archive_pair_content(
            repository,
            normalized_archive,
            source_url=source_url,
            source_hash=source_hash,
            source="sec_edgar_filing",
        )
        expected_archive = {
            "archive_id": source_hash,
            "dataset": "sec_edgar_filing",
            "object_path": f"archives/2026-07-15/{source_hash}.txt.gz",
            "content_type": "text/plain",
            "effective_date": "2026-07-15",
            "source": "sec_edgar_filing",
            "source_url": source_url,
            "source_hash": source_hash,
        }
        for field, wanted in expected_archive.items():
            actual_text = _text(archive_row.get(field))
            actual = (
                _date(actual_text)
                if field == "effective_date" and actual_text
                else actual_text
            )
            if actual != _text(wanted):
                raise RuntimeError(
                    f"Exact prior terminal archive row changed: {symbol}/{field}"
                )
        archive_retrieved_at = _text(archive_row.get("retrieved_at"))
        allowed_archive_retrievals = {
            _text(other["archive_retrieved_at"])
            for other in EXACT_PRIOR_TERMINAL_TRANSITIONS.values()
            if _text(other["action"]["source_hash"]).lower() == source_hash
        } | {
            _text(other["superseded_action"]["retrieved_at"])
            for other in EXACT_PRIOR_TERMINAL_TRANSITIONS.values()
            if _text(other["action"]["source_hash"]).lower() == source_hash
        }
        if archive_retrieved_at not in allowed_archive_retrievals:
            raise RuntimeError(
                f"Exact prior terminal archive row changed: {symbol}/retrieved_at"
            )
        archive_mask = (
            normalized_archive["archive_id"].astype(str).str.lower().eq(source_hash)
            & normalized_archive["source_url"].astype(str).eq(source_url)
            & normalized_archive["source"].astype(str).eq("sec_edgar_filing")
        )
        if int(archive_mask.sum()) != 1:
            raise RuntimeError(
                f"Exact prior terminal archive inventory changed: {symbol}"
            )
        normalized_archive.loc[
            archive_mask, "retrieved_at"
        ] = spec["archive_retrieved_at"]
        if len(content) != int(spec["archive_bytes"]):
            raise RuntimeError(
                f"Exact prior terminal archive bytes changed: {symbol}"
            )

        resolution_rows = resolution_frame.loc[
            resolution_frame["security_id"].astype(str).eq(candidate.security_id)
        ]
        if len(resolution_rows) != 1:
            raise RuntimeError(
                f"Exact prior terminal resolution is missing or duplicated: {symbol}"
            )
        resolution_row = resolution_rows.iloc[0].to_dict()
        canonical_resolution = dict(spec["resolution"])
        superseded_resolution = {
            **canonical_resolution,
            **spec["superseded_resolution"],
        }
        if not (
            _prior_terminal_resolution_matches(
                resolution_row, canonical_resolution
            )
            or _prior_terminal_resolution_matches(
                resolution_row, superseded_resolution
            )
        ):
            raise RuntimeError(
                f"Exact prior terminal resolution changed: {symbol}"
            )
        restored[key] = canonical_resolution
        markers.append(symbol)

    normalized_actions = actions.loc[
        ~actions["event_id"].astype(str).isin(remove_event_ids)
    ].reset_index(drop=True)
    return (
        restored,
        tuple(sorted(markers)),
        normalized_actions,
        normalized_archive.reset_index(drop=True),
    )


def _preserve_exact_reviewed_nbl_resolution(
    candidate: LifecycleCandidate,
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    expected = TRUSTED_OPERATIONAL_NBL_TERMINAL_STATE
    if not (
        candidate.security_id == expected["security_id"]
        and candidate.symbol.upper() == expected["symbol"]
        and candidate.name == "Noble Energy Inc"
        and candidate.exchange.upper() == expected["identity_exchange"]
        and _date(candidate.last_price_date) == expected["last_real_session"]
        and _date(candidate.active_to) == expected["last_real_session"]
        and tuple(candidate.index_remove_dates)
        == (expected["next_remove_effective_date"],)
    ):
        raise RuntimeError("Exact reviewed NBL lifecycle candidate changed.")
    reviewed = reviewed_operational_index_identity_gap_fingerprints(
        _FrameRepository(frames)
    )
    if reviewed != (expected["fingerprint"],):
        raise RuntimeError(
            "Exact reviewed NBL operational terminal state changed."
        )
    return _require_exact_repaired_resolution(
        frames["lifecycle_resolutions"],
        expected={
            "candidate_id": expected["candidate_id"],
            "security_id": expected["security_id"],
            "symbol": expected["symbol"],
            "last_price_date": expected["last_real_session"],
            "resolution": "applied",
            "event_id": expected["event_id"],
            "exception_code": "",
            "exception_reason": "",
            "reviewed_by": expected["resolution_reviewer"],
            "reviewed_at": expected["repair_reviewed_at"],
            "recheck_after": "",
            "successor_security_id": expected["successor_security_id"],
            "successor_symbol": expected["successor_symbol"],
            "source_url": expected["official_source_url"],
            "source": expected["resolution_source"],
            "retrieved_at": expected["repair_reviewed_at"],
            "source_hash": expected["official_source_hash"],
        },
    )


def _preserved_exact_repair_resolution(
    candidate: LifecycleCandidate,
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any] | None:
    key = _key(candidate.security_id, candidate.last_price_date)
    market_date_spec = EXACT_REVIEWED_MARKET_DATE_TRANSITIONS.get(key)
    if market_date_spec is not None:
        relevant = (
            _frame_contains_exact_value(
                frames.get("corporate_actions"),
                "event_id",
                (market_date_spec["event_id"],),
            )
            or _frame_contains_exact_value(
                frames.get("lifecycle_resolutions"),
                "event_id",
                (market_date_spec["event_id"],),
            )
        )
        return (
            _preserve_exact_market_date_transition_resolution(
                candidate, repository, release, frames
            )
            if relevant
            else None
        )
    short_terminal_spec = EXACT_SHORT_TERMINAL_MARKET_TRANSITIONS.get(key)
    if short_terminal_spec is not None:
        relevant = (
            _frame_contains_exact_value(
                frames.get("corporate_actions"),
                "event_id",
                (short_terminal_spec["event_id"],),
            )
            or _frame_contains_exact_value(
                frames.get("lifecycle_resolutions"),
                "event_id",
                (short_terminal_spec["event_id"],),
            )
        )
        return (
            _preserve_exact_short_terminal_market_resolution(
                candidate, repository, release, frames
            )
            if relevant
            else None
        )
    nbl = TRUSTED_OPERATIONAL_NBL_TERMINAL_STATE
    if key == _key(nbl["security_id"], nbl["last_real_session"]):
        relevant = (
            _frame_contains_exact_value(
                frames.get("corporate_actions"),
                "event_id",
                (nbl["event_id"],),
            )
            or _frame_contains_exact_value(
                frames.get("lifecycle_resolutions"),
                "event_id",
                (nbl["event_id"],),
            )
        )
        return (
            _preserve_exact_reviewed_nbl_resolution(candidate, frames)
            if relevant
            else None
        )
    if key == _key(AVP_EXACT_SECURITY_ID, AVP_EXACT_LAST_SESSION):
        # There is no fallback exception for this exact candidate.  Once the
        # reviewed action exists, every finalization must independently prove
        # the legal close, next-session successor price, and identity boundary.
        return _preserve_exact_avp_resolution(
            candidate, repository, release, frames
        )
    if key == _key(NTCO_EXACT_SECURITY_ID, NTCO_EXACT_LAST_SESSION):
        relevant = (
            frames["corporate_actions"]["event_id"]
            .astype(str)
            .isin({NTCO_EXACT_TICKER_EVENT_ID, NTCO_EXACT_TERMINAL_EVENT_ID})
            .any()
            or frames["lifecycle_resolutions"]["event_id"]
            .astype(str)
            .eq(NTCO_EXACT_TERMINAL_EVENT_ID)
            .any()
        )
        return (
            _preserve_exact_ntco_resolution(
                candidate, repository, release, frames
            )
            if relevant
            else None
        )
    if key == _key(SIVB_EXACT_SECURITY_ID, SIVB_EXACT_LAST_SESSION):
        relevant = (
            frames["corporate_actions"]["event_id"]
            .astype(str)
            .isin({SIVB_EXACT_TICKER_EVENT_ID, SIVB_EXACT_MARKET_EXIT_EVENT_ID})
            .any()
            or frames["lifecycle_resolutions"]["event_id"]
            .astype(str)
            .eq(SIVB_EXACT_MARKET_EXIT_EVENT_ID)
            .any()
        )
        return (
            _preserve_exact_sivb_resolution(
                candidate, repository, release, frames
            )
            if relevant
            else None
        )
    if key == _key(CELG_EXACT_SECURITY_ID, CELG_EXACT_LAST_SESSION):
        relevant = (
            frames["corporate_actions"]["event_id"]
            .astype(str)
            .isin({CELG_EXACT_DISTRIBUTION_EVENT_ID, CELG_EXACT_MERGER_EVENT_ID})
            .any()
            or frames["lifecycle_resolutions"]["event_id"]
            .astype(str)
            .eq(CELG_EXACT_MERGER_EVENT_ID)
            .any()
        )
        return (
            _preserve_exact_celg_resolution(
                candidate, repository, release, frames
            )
            if relevant
            else None
        )
    if key == _key(
        CELG_OFFICIAL_EXIT_SECURITY_ID, CELG_EXACT_EFFECTIVE_DATE
    ):
        relevant = frames["corporate_actions"]["event_id"].astype(str).eq(
            CELG_OFFICIAL_EXIT_EVENT_ID
        ).any()
        return (
            _preserve_exact_celg_resolution(
                candidate, repository, release, frames
            )
            if relevant
            else None
        )
    if key == _key(ABMD_EXACT_SECURITY_ID, ABMD_EXACT_LAST_SESSION):
        relevant = (
            frames["corporate_actions"]["event_id"]
            .astype(str)
            .eq(ABMD_EXACT_EVENT_ID)
            .any()
            or frames["lifecycle_resolutions"]["event_id"]
            .astype(str)
            .eq(ABMD_EXACT_EVENT_ID)
            .any()
        )
        return (
            _preserve_exact_abmd_resolution(candidate, repository, release, frames)
            if relevant
            else None
        )
    return None


def _validate_identity_bound_terminal_event(
    candidate: LifecycleCandidate,
    event: Mapping[str, Any],
    successor_security_id: str,
) -> None:
    key = _key(candidate.security_id, candidate.last_price_date)
    expected = IDENTITY_BOUND_TERMINAL_EVENTS.get(key)
    symbol = candidate.symbol.upper()
    if expected is None:
        if symbol in IDENTITY_BOUND_TERMINAL_SYMBOLS:
            raise RuntimeError(
                "Lifecycle event has no reviewed exact identity/date binding for a "
                f"reused terminal symbol: {key}/{symbol}"
            )
        return
    actual_signature = (
        symbol,
        _text(event.get("action_type")).lower(),
        _date(event.get("effective_date")),
        _text(event.get("new_symbol")).upper(),
        _text(successor_security_id),
    )
    expected_signature = (
        str(expected["symbol"]).upper(),
        str(expected["action_type"]),
        str(expected["effective_date"]),
        str(expected["new_symbol"]).upper(),
        str(expected["successor_security_id"]),
    )
    source_matches = (
        not expected.get("source_url")
        or _text(event.get("source_url")) == _text(expected.get("source_url"))
    ) and (
        not expected.get("source_hash")
        or _text(event.get("source_hash")).lower()
        == _text(expected.get("source_hash")).lower()
    )
    if actual_signature != expected_signature or not _same_optional_number(
        event.get("cash_amount"), expected["cash_amount"]
    ) or not _same_optional_number(event.get("ratio"), expected["ratio"]) or not source_matches:
        raise RuntimeError(
            "Lifecycle event differs from its reviewed exact identity/date binding: "
            f"{key}; actual={actual_signature!r}; expected={expected_signature!r}"
        )


def _reuse_identity_bound_existing_action(
    candidate: LifecycleCandidate,
    existing: pd.DataFrame,
) -> dict[str, Any] | None:
    expected_event = IDENTITY_BOUND_TERMINAL_EVENTS.get(
        _key(candidate.security_id, candidate.last_price_date)
    )
    if expected_event is None or not bool(expected_event.get("reuse_existing_action")):
        return None
    rows = existing.loc[
        existing["security_id"].astype(str).eq(candidate.security_id)
        & existing["action_type"].astype(str).str.lower().eq("stock_merger")
        & pd.to_datetime(existing["effective_date"], errors="coerce")
        .dt.date.astype(str)
        .eq("2017-09-01")
    ]
    if len(rows) != 1:
        raise RuntimeError(
            "DD finalization requires exactly one pre-existing reviewed action: "
            f"matches={len(rows)}"
        )
    row = rows.iloc[0].to_dict()
    numeric_ok = _same_optional_number(row.get("cash_amount"), None) and (
        _same_optional_number(row.get("ratio"), float(DD_EXISTING_ACTION["ratio"]))
    )
    text_fields = (
        "event_id",
        "security_id",
        "action_type",
        "effective_date",
        "ex_date",
        "currency",
        "new_security_id",
        "new_symbol",
        "source_url",
        "source_kind",
        "source",
        "source_hash",
    )
    text_ok = all(
        (_date(row.get(field)) if field in {"effective_date", "ex_date"} else _text(row.get(field)))
        == str(DD_EXISTING_ACTION[field])
        for field in text_fields
    )
    official = row.get("official")
    official_ok = isinstance(official, (bool, np.bool_)) and bool(official)
    if not numeric_ok or not text_ok or not official_ok:
        raise RuntimeError(
            "DD pre-existing reviewed action differs from the exact event/source binding."
        )
    return row


def _event_from_record(record: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    verified = record.get("verified_event")
    if verified is not None:
        if not isinstance(verified, dict):
            raise RuntimeError("verified_event must be an object.")
        event = dict(verified)
        event.setdefault("confidence", "high")
        event.setdefault("filing_date", (record.get("filing") or {}).get("filing_date", ""))
        return event, True
    parsed = record.get("parsed")
    if not isinstance(parsed, dict):
        raise RuntimeError("Applied record has no parsed lifecycle event.")
    event = dict(parsed)
    event.update(
        {
            "filing_date": _text((record.get("filing") or {}).get("filing_date")),
            "source_url": _text(record.get("source_url")),
            "source_hash": _text(record.get("source_hash")),
        }
    )
    return event, False


def _reviewed_cross_basis_ineligible_record(
    candidate: LifecycleCandidate,
    record: Mapping[str, Any],
) -> bool:
    key = _key(candidate.security_id, candidate.last_price_date)
    spec = REVIEWED_CROSS_BASIS_TERMINAL_PRICE_PROVENANCE.get(key)
    if spec is None:
        return False
    if bool(record.get("eligible_for_apply")) or "verified_event" in record:
        return False
    if (
        bool(record.get("manual_review"))
        or _text(record.get("manual_review_reason"))
        or _text(record.get("error"))
    ):
        raise RuntimeError(
            f"Reviewed cross-basis record is not a clean collector result: {key}"
        )
    event, override = _event_from_record(record)
    if override or _text(event.get("confidence")).lower() != "high":
        raise RuntimeError(
            f"Reviewed cross-basis record lacks one parsed high-confidence event: {key}"
        )
    supplied_successor = _text(record.get("successor_security_id"))
    if supplied_successor != str(spec["successor_security_id"]):
        raise RuntimeError(
            f"Reviewed cross-basis record has the wrong collector successor: {key}"
        )
    _validate_identity_bound_terminal_event(candidate, event, supplied_successor)
    crosscheck = record.get("crosscheck")
    if not isinstance(crosscheck, Mapping):
        raise RuntimeError(f"Reviewed cross-basis record lacks collector crosscheck: {key}")
    deviation = pd.to_numeric(crosscheck.get("relative_deviation"), errors="coerce")
    expected_failed_level_check = (
        crosscheck.get("basis") == "eodhd_terminal_price"
        and crosscheck.get("passed") is False
        and crosscheck.get("date_passed") is True
        and crosscheck.get("economic_terms_passed") is False
        and not pd.isna(deviation)
        and math.isfinite(float(deviation))
        and float(deviation) > 0.20
    )
    if not expected_failed_level_check:
        raise RuntimeError(
            "Reviewed cross-basis entry is allowed only for the collector's raw-level "
            f"economic mismatch: {key}"
        )
    return True


def _validate_applied_record(
    record: Mapping[str, Any],
    *,
    override: bool,
    reviewed_cross_basis: bool = False,
) -> None:
    if _text(record.get("manual_review_reason")) or bool(record.get("manual_review")):
        raise RuntimeError("Manual-review lifecycle evidence cannot be applied.")
    if reviewed_cross_basis:
        if override or bool(record.get("eligible_for_apply")):
            raise RuntimeError("Reviewed cross-basis entry path received the wrong record state.")
        return
    if not override:
        if not bool(record.get("eligible_for_apply")):
            raise RuntimeError("Unapproved lifecycle evidence cannot be applied.")
        crosscheck = record.get("crosscheck") or {}
        if not all(bool(crosscheck.get(key)) for key in ("passed", "date_passed", "economic_terms_passed")):
            raise RuntimeError("Reported lifecycle date/economic crosscheck is incomplete.")


def _canonical_action_id(
    security_id: str,
    action_type: str,
    effective_date: str,
    *,
    cash_amount: Any = None,
) -> str:
    if action_type in {"cash_merger", "stock_merger", "ticker_change", "delisting"}:
        return canonical_lifecycle_event_id(security_id, action_type, effective_date)
    payload = json.dumps(
        {
            "security_id": security_id,
            "action_type": action_type,
            "effective_date": effective_date,
            "cash_amount": None if cash_amount is None else float(cash_amount),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256_bytes(payload)


def _action_record(
    candidate: LifecycleCandidate,
    event: Mapping[str, Any],
    artifact: SourceArtifact,
    successor_security_id: str,
) -> dict[str, Any]:
    action_type = _text(event.get("action_type")).lower()
    effective = _date(event.get("effective_date"))
    cash_amount = event.get("cash_amount")
    return {
        "event_id": _canonical_action_id(
            candidate.security_id,
            action_type,
            effective,
            cash_amount=cash_amount,
        ),
        "security_id": candidate.security_id,
        "action_type": action_type,
        "effective_date": effective,
        "ex_date": effective,
        "announcement_date": _date(event.get("filing_date") or effective),
        "record_date": "",
        "payment_date": "",
        "cash_amount": cash_amount,
        "ratio": event.get("ratio"),
        "currency": "USD",
        "new_security_id": successor_security_id,
        "new_symbol": _text(event.get("new_symbol")).upper(),
        "official": True,
        "source": "sec_edgar+stored_price_crosscheck",
        "source_url": artifact.source_url,
        "source_kind": "official_crosscheck",
        "retrieved_at": artifact.retrieved_at,
        "source_hash": artifact.source_hash,
    }


def _jwn_special_dividends(
    candidate: LifecycleCandidate,
    artifact: SourceArtifact,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for amount in (0.25, 0.1462):
        effective = "2025-05-19"
        rows.append(
            {
                "event_id": _canonical_action_id(
                    candidate.security_id,
                    "special_dividend",
                    effective,
                    cash_amount=amount,
                ),
                "security_id": candidate.security_id,
                "action_type": "special_dividend",
                "effective_date": effective,
                "ex_date": effective,
                "announcement_date": "2025-05-20",
                "record_date": effective,
                "payment_date": "2025-05-27",
                "cash_amount": amount,
                "ratio": None,
                "currency": "USD",
                "new_security_id": "",
                "new_symbol": "",
                "official": True,
                "source": "sec_edgar",
                "source_url": artifact.source_url,
                "source_kind": "official_crosscheck",
                "retrieved_at": artifact.retrieved_at,
                "source_hash": artifact.source_hash,
            }
        )
    return rows


def _action_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    action_type = _text(row.get("action_type")).lower()
    security_id = _text(row.get("security_id"))
    effective = _date(row.get("effective_date"))
    if action_type in {"cash_merger", "stock_merger", "ticker_change", "delisting"}:
        return ("lifecycle", security_id, action_type, effective)
    if action_type in {"cash_dividend", "special_dividend"}:
        amount = pd.to_numeric(row.get("cash_amount"), errors="coerce")
        return ("distribution", security_id, action_type, effective, str(float(amount)))
    return ("event", _text(row.get("event_id")))


def _economic_signature(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        _text(row.get(column)).upper() if column == "new_symbol" else _text(row.get(column))
        for column in (
            "security_id",
            "action_type",
            "effective_date",
            "cash_amount",
            "ratio",
            "currency",
            "new_security_id",
            "new_symbol",
        )
    )


def merge_canonical_actions(
    existing: pd.DataFrame,
    additions: Iterable[dict[str, Any]],
) -> pd.DataFrame:
    existing_rows = existing.to_dict(orient="records")
    by_key = {_action_key(row): row for row in existing_rows}
    for addition in additions:
        key = _action_key(addition)
        prior = by_key.get(key)
        if prior is not None and _economic_signature(prior) != _economic_signature(addition):
            raise RuntimeError(f"Conflicting canonical corporate action: {key}")
        by_key[key] = addition
    rows = list(by_key.values())
    rows.sort(
        key=lambda row: (
            _date(row.get("effective_date")),
            _text(row.get("security_id")),
            _text(row.get("action_type")),
            _text(row.get("cash_amount")),
        )
    )
    output = pd.DataFrame(rows, columns=existing.columns if len(existing.columns) else None)
    if output.empty:
        return pd.DataFrame(columns=dataset_spec("corporate_actions").required_columns)
    duplicates = output.duplicated("event_id", keep=False)
    if duplicates.any():
        grouped = output.loc[duplicates].groupby("event_id")
        if any(len({_economic_signature(row) for row in group.to_dict(orient="records")}) > 1 for _, group in grouped):
            raise RuntimeError("Conflicting corporate actions share an event_id.")
        output = output.drop_duplicates("event_id", keep="last")
    return output.reset_index(drop=True)


def _resolution_applied(
    candidate: LifecycleCandidate,
    action: Mapping[str, Any],
    artifact: SourceArtifact,
) -> dict[str, Any]:
    return {
        "candidate_id": lifecycle_candidate_id(candidate.security_id, candidate.last_price_date),
        "security_id": candidate.security_id,
        "symbol": candidate.symbol,
        "last_price_date": _date(candidate.last_price_date),
        "resolution": "applied",
        "event_id": _text(action.get("event_id")),
        "exception_code": "",
        "exception_reason": "",
        "reviewed_by": REVIEWED_BY,
        "reviewed_at": REVIEWED_AT,
        "recheck_after": "",
        "successor_security_id": _text(action.get("new_security_id")),
        "successor_symbol": _text(action.get("new_symbol")),
        "source_url": artifact.source_url,
        "source": "lifecycle_finalizer",
        "retrieved_at": artifact.retrieved_at,
        "source_hash": artifact.source_hash,
    }


def _resolution_exception(
    candidate: LifecycleCandidate,
    spec: ExceptionSpec,
    document: ReportDocument,
    retrieved_at: str,
    artifact: SourceArtifact | None = None,
) -> dict[str, Any]:
    if str(spec.code) in PERMANENT_EXCEPTION_CODES and artifact is None:
        raise RuntimeError(
            "Permanent lifecycle exception cannot use the self-authored lifecycle "
            f"report as evidence: {candidate.security_id}/{candidate.last_price_date}"
        )
    source_url = (
        artifact.source_url
        if artifact is not None
        else f"archive://source_archive/{document.sha256}"
    )
    source = artifact.source if artifact is not None else "lifecycle_evidence_report"
    source_retrieved_at = artifact.retrieved_at if artifact is not None else retrieved_at
    source_hash = artifact.source_hash if artifact is not None else document.sha256
    return {
        "candidate_id": lifecycle_candidate_id(candidate.security_id, candidate.last_price_date),
        "security_id": candidate.security_id,
        "symbol": candidate.symbol,
        "last_price_date": _date(candidate.last_price_date),
        "resolution": "exception",
        "event_id": "",
        "exception_code": str(spec.code),
        "exception_reason": spec.reason,
        "reviewed_by": spec.reviewed_by,
        "reviewed_at": spec.reviewed_at,
        "recheck_after": spec.recheck_after,
        "successor_security_id": "",
        "successor_symbol": "",
        "source_url": source_url,
        "source": source,
        "retrieved_at": source_retrieved_at,
        "source_hash": source_hash,
    }


def _exception_for(
    candidate: LifecycleCandidate,
    mapping: Mapping[str, ExceptionSpec],
    official_evidence_specs: Mapping[
        str, OfficialLifecycleExceptionEvidenceSpec
    ] | None = None,
) -> ExceptionSpec | None:
    exact_key = _key(candidate.security_id, candidate.last_price_date)
    symbol_key = _symbol_key(candidate.symbol, candidate.last_price_date)
    exact = mapping.get(exact_key)
    symbol_fallback = mapping.get(symbol_key)
    if exact is not None:
        if symbol_fallback is not None and symbol_fallback != exact:
            raise RuntimeError(
                "Exact lifecycle exception conflicts with a symbol fallback: "
                f"{exact_key}/{symbol_key}"
            )
        return exact
    if symbol_fallback is not None:
        if candidate.symbol.upper() in IDENTITY_BOUND_TERMINAL_SYMBOLS:
            raise RuntimeError(
                "Symbol lifecycle exception fallback is forbidden for an "
                f"identity-bound terminal symbol: {exact_key}/{symbol_key}"
            )
        return symbol_fallback
    matches = tuple(
        evidence
        for evidence in matching_official_exception_specs(
            candidate, official_evidence_specs or {}
        )
        if evidence.resolution_kind == "exception"
    )
    if len(matches) > 1:
        raise RuntimeError(
            "Current lifecycle candidate matches multiple official exception evidence "
            f"bindings: {candidate.security_id}/{[item.evidence_id for item in matches]}"
        )
    if not matches:
        return None
    evidence = matches[0]
    return ExceptionSpec(
        code=evidence.exception_code,
        reason=evidence.claim,
        require_official_provenance=True,
        evidence_id=evidence.evidence_id,
    )


def _archive_path(effective_date: str, artifact: SourceArtifact) -> str:
    content_type = artifact.content_type.lower()
    extension = (
        "json" if "json" in content_type else "html" if "html" in content_type else "pdf" if "pdf" in content_type else "txt"
    )
    return f"archives/{effective_date}/{artifact.source_hash}.{extension}.gz"


def _archive_rows(
    artifacts: Iterable[ArchivedArtifact],
    effective_date: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "archive_id": item.artifact.source_hash,
                "dataset": item.artifact.source,
                "object_path": item.object_path,
                "content_type": item.artifact.content_type,
                "effective_date": effective_date,
                "source": item.artifact.source,
                "source_url": item.artifact.source_url,
                "retrieved_at": item.artifact.retrieved_at,
                "source_hash": item.artifact.source_hash,
            }
            for item in artifacts
        ],
        columns=tuple(
            dict.fromkeys(
                (*dataset_spec("source_archive").required_columns, "source_url")
            )
        ),
    )


class _FrameRepository:
    def __init__(self, frames: Mapping[str, pd.DataFrame]):
        self.frames = frames

    def current_manifest(self, dataset: str):
        return object() if dataset in self.frames else None

    def read_frame(self, dataset: str):
        return self.frames[dataset].copy()


def _validate_all_frames(
    frames: Mapping[str, pd.DataFrame],
    completed_session: str,
) -> tuple[str, ...]:
    warnings: list[str] = []
    for dataset, frame in frames.items():
        report = validate_dataset(
            dataset,
            frame,
            incomplete_action_policy="block",
            completed_session=completed_session,
        )
        report.raise_for_errors()
        warnings.extend(issue.message for issue in report.issues if issue.severity != "error")
    cross = validate_operational_repository_snapshot(_FrameRepository(frames))
    cross.raise_for_errors()
    warnings.extend(issue.message for issue in cross.issues if issue.severity != "error")
    return tuple(dict.fromkeys(warnings))


def _assert_release_unchanged(
    repository: LocalDatasetRepository,
    release: DataRelease,
    release_etag: str | None,
) -> None:
    current, current_etag = repository.current_release()
    if current is None or current.to_bytes() != release.to_bytes() or current_etag != release_etag:
        raise RuntimeError("Current release changed after lifecycle finalization began.")


def _new_planned_versions(release: DataRelease) -> dict[str, str]:
    token = uuid.uuid4().hex
    session = release.completed_session.replace("-", "")
    return {
        dataset: f"lifecycle-finalizer-{session}-{token}-{dataset}"
        for dataset in WRITE_DATASETS
    }


def _adjustment_source_version(
    daily_price_version: str,
    corporate_actions_version: str,
) -> str:
    if not daily_price_version or not corporate_actions_version:
        raise RuntimeError(
            "Adjustment factors require exact daily-price and corporate-action versions."
        )
    return f"{daily_price_version}+{corporate_actions_version}"


def _manifest_adjustment_source_version(
    metadata: Mapping[str, Any],
    daily_price_version: str,
    corporate_actions_version: str,
) -> str:
    """Return the exact lineage format owned by the factor-table writer."""

    if _text(metadata.get("operation")) == EARLY_TERMINAL_HISTORY_FACTOR_OPERATION:
        digest = sha256_bytes(
            f"{daily_price_version}|{corporate_actions_version}".encode()
        )
        return f"{EARLY_TERMINAL_HISTORY_FACTOR_PREFIX}{digest}"
    return _adjustment_source_version(
        daily_price_version,
        corporate_actions_version,
    )


def _assert_adjustment_source_version(
    factors: pd.DataFrame,
    expected: str,
) -> None:
    if "source_version" not in factors:
        raise RuntimeError("Adjustment factors are missing source_version.")
    observed = set(factors["source_version"].dropna().astype(str))
    if not factors.empty and observed != {expected}:
        raise RuntimeError(
            "Adjustment factor source_version does not match the frozen inputs: "
            f"expected={expected!r}, observed={sorted(observed)!r}."
        )


def _assert_adjustment_output_lineage(
    factors: pd.DataFrame,
    expected: str,
) -> None:
    """Require planned/applied factor rows to bind the exact new input pair."""

    _assert_adjustment_source_version(factors, expected)
    required = {"source", "source_hash"}
    missing = sorted(required - set(factors.columns))
    if missing:
        raise RuntimeError(
            "Adjustment-factor output lineage lacks columns: " + ", ".join(missing)
        )
    if not factors.empty and (
        factors["source"].astype(str).ne("derived").any()
        or factors["source_hash"].astype(str).ne(expected).any()
    ):
        raise RuntimeError(
            "Adjustment-factor output rows do not bind the planned input versions."
        )


def _action_lineage_signatures(
    frame: pd.DataFrame,
) -> tuple[dict[str, tuple[Any, ...]], dict[str, tuple[Any, ...]]]:
    """Return exact economic and full signatures keyed by unique event_id."""

    def optional_number(value: Any) -> float | None:
        parsed = pd.to_numeric(value, errors="coerce")
        return None if pd.isna(parsed) else float(parsed)

    def optional_metadata(value: Any) -> str:
        return "" if not _text(value) else _canonical_metadata_text(value)

    def optional_date(value: Any) -> str:
        return "" if not _text(value) else _date(value)

    required = {
        "event_id",
        "security_id",
        "action_type",
        "effective_date",
        "ex_date",
        "announcement_date",
        "record_date",
        "payment_date",
        "cash_amount",
        "ratio",
        "currency",
        "new_security_id",
        "new_symbol",
        "official",
        "source_url",
        "source_kind",
        "source",
        "retrieved_at",
        "source_hash",
        "metadata",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            "Adjustment-lineage corporate actions lack columns: "
            + ", ".join(missing)
        )
    if frame["event_id"].astype(str).duplicated().any():
        raise RuntimeError("Adjustment-lineage corporate actions duplicate event_id.")
    economic: dict[str, tuple[Any, ...]] = {}
    full: dict[str, tuple[Any, ...]] = {}
    for row in frame.to_dict(orient="records"):
        event_id = _text(row.get("event_id"))
        if not event_id:
            raise RuntimeError("Adjustment-lineage corporate action has no event_id.")
        core = (
            event_id,
            _text(row.get("security_id")),
            _text(row.get("action_type")).lower(),
            optional_date(row.get("effective_date")),
            optional_date(row.get("ex_date")),
            optional_date(row.get("announcement_date")),
            optional_date(row.get("record_date")),
            optional_date(row.get("payment_date")),
            optional_number(row.get("cash_amount")),
            optional_number(row.get("ratio")),
            _text(row.get("currency")).upper(),
            _text(row.get("new_security_id")),
            _text(row.get("new_symbol")).upper(),
            _text(row.get("official")).lower(),
        )
        provenance = (
            _text(row.get("source_url")),
            _text(row.get("source_kind")),
            _text(row.get("source")),
            _text(row.get("retrieved_at")),
            _text(row.get("source_hash")).lower(),
            optional_metadata(row.get("metadata")),
        )
        economic[event_id] = core
        full[event_id] = (*core, *provenance)
    return economic, full


def _require_exact_sivb_occ_action(actions: pd.DataFrame) -> None:
    _require_exact_repaired_action(
        actions,
        event_id=SIVB_EXACT_TICKER_EVENT_ID,
        expected_text={
            "event_id": SIVB_EXACT_TICKER_EVENT_ID,
            "security_id": SIVB_EXACT_SECURITY_ID,
            "action_type": "ticker_change",
            "effective_date": SIVB_EXACT_OTC_START,
            "ex_date": SIVB_EXACT_OTC_START,
            "announcement_date": "2023-03-27",
            "record_date": "",
            "payment_date": "",
            "currency": "USD",
            "new_security_id": SIVB_EXACT_SECURITY_ID,
            "new_symbol": "SIVBQ",
            "source_url": SIVB_EXACT_OCC_URL,
            "source_kind": "official_crosscheck",
            "source": "occ_information_memo",
            "retrieved_at": "2026-07-18T18:20:45Z",
            "source_hash": SIVB_EXACT_OCC_PDF_SHA256,
        },
        cash_amount=None,
        ratio=None,
        metadata_sha256=SIVB_EXACT_TICKER_METADATA_SHA256,
    )


def _exact_non_security_partition(
    frame: pd.DataFrame,
    security_id: str,
) -> pd.DataFrame:
    if "security_id" not in frame.columns:
        raise RuntimeError("Exact lineage partition lacks security_id.")
    return frame.loc[
        frame["security_id"].astype(str).ne(security_id)
    ].reset_index(drop=True)


def _validate_exact_ntco_mixed_adjustment_lineage(
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
    factor_manifest: Any,
    *,
    source_price_version: str,
    source_action_version: str,
    source_version: str,
) -> str:
    """Validate the one reviewed per-security NTCO factor rebuild bridge."""

    prior_action_version = EXACT_PROVENANCE_BRIDGE_FACTOR_ACTION_VERSION
    prior_price_version = prior_action_version.replace(
        "-corporate_actions", "-daily_price_raw"
    )
    prior_factor_version = prior_action_version.replace(
        "-corporate_actions", "-adjustment_factors"
    )
    prior_source_version = _adjustment_source_version(
        prior_price_version, prior_action_version
    )
    current_source_version = _adjustment_source_version(
        NTCO_EXACT_MIXED_PRICE_VERSION,
        NTCO_EXACT_MIXED_ACTION_VERSION,
    )
    versions = release.dataset_versions
    if (
        _text(versions.get("daily_price_raw")) != NTCO_EXACT_MIXED_PRICE_VERSION
        or _text(versions.get("corporate_actions"))
        != NTCO_EXACT_MIXED_ACTION_VERSION
        or _text(versions.get("adjustment_factors"))
        != NTCO_EXACT_MIXED_FACTOR_VERSION
        or _text(getattr(factor_manifest, "version", ""))
        != NTCO_EXACT_MIXED_FACTOR_VERSION
    ):
        raise RuntimeError("Exact NTCO mixed-lineage release versions changed.")
    metadata = getattr(factor_manifest, "metadata", None)
    if not isinstance(metadata, Mapping) or any(
        _text(metadata.get(field)) != wanted
        for field, wanted in {
            "source_daily_price_version": prior_price_version,
            "source_corporate_actions_version": prior_action_version,
            "source_version": prior_source_version,
            "daily_price_version": NTCO_EXACT_MIXED_PRICE_VERSION,
            "corporate_action_version": NTCO_EXACT_MIXED_ACTION_VERSION,
            "operation": "repair_us_ntco_ntcoy_transition",
        }.items()
    ):
        raise RuntimeError("Exact NTCO mixed-lineage manifest contract changed.")
    if (
        source_price_version != prior_price_version
        or source_action_version != prior_action_version
        or source_version != prior_source_version
    ):
        raise RuntimeError("Exact NTCO mixed-lineage source contract changed.")

    required_frames = {
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "corporate_actions",
        "adjustment_factors",
        "source_archive",
        "lifecycle_resolutions",
    }
    missing = sorted(required_frames - set(frames))
    if missing:
        raise RuntimeError(
            "Exact NTCO mixed-lineage inputs are missing: " + ", ".join(missing)
        )
    factors = frames["adjustment_factors"]
    factor_columns = {"security_id", "source_version", "source_hash", "source"}
    missing_factor_columns = sorted(factor_columns - set(factors.columns))
    if missing_factor_columns:
        raise RuntimeError(
            "Exact NTCO mixed-lineage factors lack columns: "
            + ", ".join(missing_factor_columns)
        )
    target_mask = factors["security_id"].astype(str).eq(NTCO_EXACT_SECURITY_ID)
    target_factors = factors.loc[target_mask]
    non_target_factors = factors.loc[~target_mask]
    if target_factors.empty or non_target_factors.empty:
        raise RuntimeError("Exact NTCO mixed-lineage factor partition is incomplete.")
    if (
        target_factors["source_version"].astype(str).ne(current_source_version).any()
        or target_factors["source_hash"].astype(str).ne(current_source_version).any()
        or target_factors["source"].astype(str).ne("derived").any()
        or non_target_factors["source_version"].astype(str).ne(prior_source_version).any()
        or non_target_factors["source_hash"].astype(str).ne(prior_source_version).any()
        or non_target_factors["source"].astype(str).ne("derived").any()
    ):
        raise RuntimeError("Exact NTCO mixed-lineage factor row provenance changed.")

    try:
        prior_factors = repository.read_frame(
            "adjustment_factors", prior_factor_version
        )
    except Exception as exc:
        raise RuntimeError(
            "Exact NTCO prior adjustment-factor version is unavailable."
        ) from exc
    _assert_adjustment_source_version(prior_factors, prior_source_version)
    if (
        "source_hash" not in prior_factors.columns
        or "source" not in prior_factors.columns
        or prior_factors["source_hash"].astype(str).ne(prior_source_version).any()
        or prior_factors["source"].astype(str).ne("derived").any()
    ):
        raise RuntimeError("Exact NTCO prior factor lineage is inconsistent.")
    if not _exact_non_security_partition(
        factors, NTCO_EXACT_SECURITY_ID
    ).equals(
        _exact_non_security_partition(prior_factors, NTCO_EXACT_SECURITY_ID)
    ):
        raise RuntimeError(
            "Exact NTCO bridge changed a non-NTCO adjustment-factor row."
        )

    try:
        prior_prices = repository.read_frame(
            "daily_price_raw", prior_price_version
        )
    except Exception as exc:
        raise RuntimeError("Exact NTCO prior daily-price version is unavailable.") from exc
    if not _exact_non_security_partition(
        frames["daily_price_raw"], NTCO_EXACT_SECURITY_ID
    ).equals(
        _exact_non_security_partition(prior_prices, NTCO_EXACT_SECURITY_ID)
    ):
        raise RuntimeError("Exact NTCO bridge changed a non-NTCO price row.")

    try:
        prior_actions = repository.read_frame(
            "corporate_actions", prior_action_version
        )
    except Exception as exc:
        raise RuntimeError(
            "Exact NTCO prior corporate-action version is unavailable."
        ) from exc
    current_economic, current_full = _action_lineage_signatures(
        frames["corporate_actions"]
    )
    prior_economic, prior_full = _action_lineage_signatures(prior_actions)
    event_ids = set(current_full) | set(prior_full)
    economic_changed = {
        event_id
        for event_id in event_ids
        if current_economic.get(event_id) != prior_economic.get(event_id)
    }
    full_changed = {
        event_id
        for event_id in event_ids
        if current_full.get(event_id) != prior_full.get(event_id)
    }
    ntco_additions = {
        NTCO_EXACT_TICKER_EVENT_ID,
        NTCO_EXACT_TERMINAL_EVENT_ID,
    }
    expected_full_changed = {
        SIVB_EXACT_TICKER_EVENT_ID,
        FRC_EXACT_EVENT_ID,
        *ntco_additions,
    }
    if (
        economic_changed != ntco_additions
        or full_changed != expected_full_changed
        or set(current_full) - set(prior_full) != ntco_additions
        or set(prior_full) - set(current_full)
    ):
        raise RuntimeError("Exact NTCO bridge corporate-action delta changed.")
    _require_exact_frc_occ_action(frames["corporate_actions"])
    _require_exact_sivb_occ_action(frames["corporate_actions"])

    # This existing invariant independently binds the target master/history,
    # action rows, raw archives, EOD payload, factor economics, and resolution.
    _preserve_exact_ntco_resolution(
        LifecycleCandidate(
            NTCO_EXACT_SECURITY_ID,
            NTCO_EXACT_NEW_SYMBOL,
            "Natura &Co Holding S.A.",
            "OTC",
            NTCO_EXACT_LAST_SESSION,
            NTCO_EXACT_TERMINAL_DATE,
        ),
        repository,
        release,
        frames,
    )
    return prior_source_version


def _validate_input_adjustment_lineage_for_refinalization(
    repository: LocalDatasetRepository,
    release: DataRelease,
    frames: Mapping[str, pd.DataFrame],
) -> str:
    """Validate current lineage or the one code-pinned provenance-only bridge."""

    versions = release.dataset_versions
    price_version = _text(versions.get("daily_price_raw"))
    current_action_version = _text(versions.get("corporate_actions"))
    factor_version = _text(versions.get("adjustment_factors"))
    if not price_version or not current_action_version or not factor_version:
        raise RuntimeError("Input adjustment-factor release pointers are incomplete.")
    try:
        factor_manifest = repository.manifest_for_version(
            "adjustment_factors", factor_version
        )
    except Exception as exc:
        raise RuntimeError(
            "Current adjustment-factor manifest is unavailable."
        ) from exc
    metadata = getattr(factor_manifest, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise RuntimeError("Adjustment-factor lineage metadata is missing.")
    source_price_version = _text(metadata.get("source_daily_price_version"))
    source_action_version = _text(metadata.get("source_corporate_actions_version"))
    source_version = _text(metadata.get("source_version"))
    expected_source_version = _manifest_adjustment_source_version(
        metadata,
        source_price_version,
        source_action_version,
    )
    if (
        not source_price_version
        or not source_action_version
        or source_version != expected_source_version
    ):
        raise RuntimeError(
            "Exact BMYRT adjustment-factor manifest lineage is stale or "
            "internally inconsistent."
        )
    if factor_version == NTCO_EXACT_MIXED_FACTOR_VERSION:
        return _validate_exact_ntco_mixed_adjustment_lineage(
            repository,
            release,
            frames,
            factor_manifest,
            source_price_version=source_price_version,
            source_action_version=source_action_version,
            source_version=source_version,
        )
    if source_price_version != price_version:
        raise RuntimeError(
            "Exact BMYRT adjustment-factor manifest lineage is stale or "
            "internally inconsistent."
        )
    factors = frames["adjustment_factors"]
    _assert_adjustment_source_version(factors, source_version)
    if (
        "source_hash" not in factors.columns
        or (not factors.empty and factors["source_hash"].astype(str).ne(source_version).any())
        or (
            "source" in factors.columns
            and not factors.empty
            and factors["source"].astype(str).ne("derived").any()
        )
    ):
        raise RuntimeError("Adjustment-factor row lineage is internally inconsistent.")
    if source_action_version == current_action_version:
        return source_version

    if (
        source_action_version != EXACT_PROVENANCE_BRIDGE_FACTOR_ACTION_VERSION
        or current_action_version != EXACT_PROVENANCE_BRIDGE_CURRENT_ACTION_VERSION
    ):
        raise RuntimeError(
            "Adjustment-factor action lineage is stale outside the reviewed bridge."
        )
    try:
        prior_actions = repository.read_frame(
            "corporate_actions", source_action_version
        )
    except Exception as exc:
        raise RuntimeError(
            "Reviewed provenance bridge action version is unavailable."
        ) from exc
    current_economic, current_full = _action_lineage_signatures(
        frames["corporate_actions"]
    )
    prior_economic, prior_full = _action_lineage_signatures(prior_actions)
    if current_economic != prior_economic:
        raise RuntimeError(
            "Reviewed adjustment-lineage bridge changed action economics."
        )
    changed_event_ids = {
        event_id
        for event_id in current_full
        if current_full[event_id] != prior_full[event_id]
    }
    expected_changed = {SIVB_EXACT_TICKER_EVENT_ID, FRC_EXACT_EVENT_ID}
    if changed_event_ids != expected_changed:
        raise RuntimeError(
            "Reviewed adjustment-lineage bridge changed an unapproved action row."
        )
    _require_exact_frc_occ_action(frames["corporate_actions"])
    _require_exact_sivb_occ_action(frames["corporate_actions"])
    return source_version


def prepare_finalization(
    repository: LocalDatasetRepository,
    release: DataRelease,
    release_etag: str | None,
    document: ReportDocument,
    *,
    sec_cache: Path,
    exception_mapping: Mapping[str, ExceptionSpec] = EXPLICIT_EXCEPTION_MAPPING,
    official_evidence_specs: Mapping[
        str, OfficialLifecycleExceptionEvidenceSpec
    ] | None = None,
    candidates: Iterable[LifecycleCandidate] | None = None,
    hints_path: Path = DEFAULT_HINTS,
) -> PreparedFinalization:
    candidate_values = tuple(candidates) if candidates is not None else (
        include_bound_official_applied_event_candidates(
            build_lifecycle_candidates(repository, release=release),
            repository,
            release,
            official_evidence_specs or {},
        )
    )
    records = _validate_full_report(
        document,
        release,
        candidate_values,
        hints_path=hints_path,
    )
    input_versions = dict(sorted(release.dataset_versions.items()))
    planned_versions = _new_planned_versions(release)
    required = {
        "security_master",
        "symbol_history",
        "daily_price_raw",
        "corporate_actions",
        "adjustment_factors",
        "source_archive",
        "index_constituent_anchors",
        "index_membership_events",
    }
    missing_versions = sorted(required - set(input_versions))
    if missing_versions:
        raise RuntimeError(f"Release is missing finalizer input datasets: {missing_versions}")
    frames = {
        dataset: repository.read_frame(dataset, version)
        for dataset, version in input_versions.items()
    }
    (
        exact_prior_terminal_resolutions,
        exact_prior_terminal_markers,
        normalized_prior_terminal_actions,
        normalized_prior_terminal_archive,
    ) = _preserved_exact_prior_terminal_transitions(
        repository,
        release,
        frames,
        candidate_values,
    )
    frames["corporate_actions"] = normalized_prior_terminal_actions
    frames["source_archive"] = normalized_prior_terminal_archive
    exact_frc_para_resolutions, exact_frc_para_markers = (
        _preserved_exact_frc_para_repairs(
            repository,
            release,
            frames,
            candidate_values,
        )
    )
    master = frames["security_master"]
    history = frames["symbol_history"]
    histories = _price_histories(frames["daily_price_raw"])
    report_candidate_artifacts = {
        candidate: frozenset(
            (
                _text(item.get("source_url")),
                _text(item.get("source_hash")).lower(),
            )
            for item in records[candidate.security_id].get("artifacts") or ()
            if isinstance(item, dict)
            and _text(item.get("source_url"))
            and _text(item.get("source_hash"))
        )
        for candidate in candidate_values
    }
    report_candidate_urls = {
        candidate: tuple(source_url for source_url, _source_hash in artifacts)
        for candidate, artifacts in report_candidate_artifacts.items()
    }
    cache = _ArtifactCache(
        sec_cache,
        archive_replay_factory=lambda: _CurrentReleaseSecArchiveReplay(
            repository,
            release,
            report_candidate_urls,
        ),
        archive_authorized_artifacts=report_candidate_artifacts,
    )
    report_retrieved_at = _text(document.value.get("generated_at")) or release.created_at
    report_artifact = SourceArtifact(
        source="lifecycle_evidence_report",
        source_url=f"file://{document.path}",
        retrieved_at=report_retrieved_at,
        content=document.content,
        content_type="application/json",
    )
    accepted: dict[str, SourceArtifact] = {report_artifact.source_hash: report_artifact}
    additions: list[dict[str, Any]] = []
    resolutions: list[dict[str, Any]] = []
    preserved_exact_repairs: list[str] = [
        *exact_frc_para_markers,
        *exact_prior_terminal_markers,
    ]

    for candidate in candidate_values:
        exact_prior_terminal_resolution = exact_prior_terminal_resolutions.get(
            _key(candidate.security_id, candidate.last_price_date)
        )
        if exact_prior_terminal_resolution is not None:
            resolutions.append(exact_prior_terminal_resolution)
            continue
        frc_para_resolution = exact_frc_para_resolutions.get(
            _key(candidate.security_id, candidate.last_price_date)
        )
        if frc_para_resolution is not None:
            resolutions.append(frc_para_resolution)
            continue
        short_terminal_reviewed = (
            _preserve_exact_short_terminal_reviewed_resolution(
                candidate,
                frames,
            )
        )
        preserved = _preserved_exact_repair_resolution(
            candidate,
            repository,
            release,
            frames,
        )
        if short_terminal_reviewed is not None and preserved is not None:
            raise RuntimeError(
                "Exact short-terminal review overlaps another preservation path."
            )
        if preserved is not None:
            resolutions.append(preserved)
            preserved_exact_repairs.append(candidate.symbol.upper())
            continue
        record = records[candidate.security_id]
        override_present = isinstance(record.get("verified_event"), dict)
        normal_apply = bool(record.get("eligible_for_apply")) or override_present
        if short_terminal_reviewed is not None and not normal_apply:
            raise RuntimeError(
                "Exact short-terminal review cannot bypass the normal "
                "applied lifecycle path."
            )
        reviewed_cross_basis = (
            _reviewed_cross_basis_ineligible_record(candidate, record)
            if not normal_apply
            else False
        )
        should_apply = normal_apply or reviewed_cross_basis
        if should_apply:
            try:
                event, override = _event_from_record(record)
                _validate_applied_record(
                    record,
                    override=override,
                    reviewed_cross_basis=reviewed_cross_basis,
                )
                if _text(event.get("confidence")).lower() != "high":
                    raise RuntimeError("Only high-confidence lifecycle evidence can be applied.")
                successor_id = _successor_for_event(event, master, history)
                _validate_identity_bound_terminal_event(
                    candidate,
                    event,
                    successor_id,
                )
                artifact = _artifact_from_event(
                    event,
                    record,
                    cache,
                    trusted_override=override,
                    candidate=candidate,
                )
                _crosscheck_event(
                    candidate,
                    event,
                    successor_id,
                    histories,
                    repository=repository,
                    prices=frames["daily_price_raw"],
                    source_archive=frames["source_archive"],
                )
                action = _reuse_identity_bound_existing_action(
                    candidate,
                    frames["corporate_actions"],
                ) or _action_record(candidate, event, artifact, successor_id)

                # SWN's immediate successor traded as CHK for one session.  Do
                # not apply the first leg unless the cached official second leg
                # and the EXE successor both validate.
                if candidate.symbol.upper() == "SWN" and _text(event.get("new_symbol")).upper() == "CHK":
                    chain_event = dict(CHK_EXE_EVIDENCE)
                    chain_event["successor_security_id"] = ""
                    chain_successor = _successor_for_event(chain_event, master, history)
                    chain_candidate = LifecycleCandidate(
                        security_id=successor_id,
                        symbol="CHK",
                        name="Chesapeake Energy / Expand Energy",
                        exchange="NASDAQ",
                        last_price_date="2024-10-01",
                        active_to="2024-10-01",
                    )
                    chain_artifact = _artifact_from_event(
                        chain_event,
                        {},
                        cache,
                        trusted_override=True,
                        candidate=chain_candidate,
                    )
                    _crosscheck_event(
                        chain_candidate,
                        chain_event,
                        chain_successor,
                        histories,
                        repository=repository,
                        prices=frames["daily_price_raw"],
                        source_archive=frames["source_archive"],
                    )
                    additions.append(
                        _action_record(
                            chain_candidate,
                            chain_event,
                            chain_artifact,
                            chain_successor,
                        )
                    )
                    accepted[chain_artifact.source_hash] = chain_artifact

                additions.append(action)
                if candidate.security_id == JWN_SECURITY_ID and action["action_type"] == "cash_merger":
                    if action["effective_date"] != "2025-05-20" or float(action["cash_amount"]) != 24.25:
                        raise RuntimeError("JWN merger terms do not match the verified official filing.")
                    additions.extend(_jwn_special_dividends(candidate, artifact))
                accepted[artifact.source_hash] = artifact
                generated_resolution = _resolution_applied(
                    candidate,
                    action,
                    artifact,
                )
                if short_terminal_reviewed is not None:
                    generated_resolution = (
                        _restore_exact_short_terminal_reviewed_resolution(
                            short_terminal_reviewed,
                            generated_resolution,
                        )
                    )
                    preserved_exact_repairs.append(candidate.symbol.upper())
                resolutions.append(generated_resolution)
                continue
            except (FileNotFoundError, RuntimeError, ValueError):
                if short_terminal_reviewed is not None:
                    raise
                # Only SWN has an explicitly reviewed incomplete-chain fallback.
                # Every other claimed applied/override record fails closed.
                if candidate.symbol.upper() != "SWN":
                    raise
        spec = _exception_for(
            candidate,
            exception_mapping,
            official_evidence_specs,
        )
        if spec is None:
            raise RuntimeError(
                "Unclassified lifecycle candidate has no explicit exception: "
                f"{candidate.security_id}/{candidate.symbol}/{candidate.last_price_date}"
            )
        exception_artifact = _artifact_from_exception(
            spec,
            record,
            cache,
            official_evidence_specs,
            candidate=candidate,
        )
        if exception_artifact is not None:
            accepted[exception_artifact.source_hash] = exception_artifact
        resolutions.append(
            _resolution_exception(
                candidate,
                spec,
                document,
                report_retrieved_at,
                exception_artifact,
            )
        )

    actions = merge_canonical_actions(frames["corporate_actions"], additions)
    if "FRC/FRCB" in exact_frc_para_markers:
        _require_exact_frc_occ_action(actions)
    adjustment_source_version = _adjustment_source_version(
        input_versions["daily_price_raw"],
        planned_versions["corporate_actions"],
    )
    factors = build_adjustment_factors(
        frames["daily_price_raw"],
        actions,
        source_version=adjustment_source_version,
    )
    _assert_adjustment_output_lineage(factors, adjustment_source_version)
    resolution_frame = pd.DataFrame(
        resolutions,
        columns=dataset_spec("lifecycle_resolutions").required_columns,
    )
    accepted_values = tuple(accepted[key] for key in sorted(accepted))
    archived = tuple(
        ArchivedArtifact(
            artifact=item,
            object_path=_archive_path(release.completed_session, item),
        )
        for item in accepted_values
    )
    archive_delta = _archive_rows(archived, release.completed_session)
    source_archive = pd.concat(
        [frames["source_archive"], archive_delta],
        ignore_index=True,
        sort=False,
    ).drop_duplicates("archive_id", keep="last")

    output_frames = dict(frames)
    output_frames["corporate_actions"] = actions
    output_frames["adjustment_factors"] = factors
    output_frames["lifecycle_resolutions"] = resolution_frame
    output_frames["source_archive"] = source_archive.reset_index(drop=True)
    warnings = _validate_all_frames(output_frames, release.completed_session)
    coverage = validate_lifecycle_coverage(
        _candidate_frame(candidate_values),
        resolution_frame,
        actions,
        completed_session=release.completed_session,
    )
    if not coverage.valid or coverage.open_count != 0:
        details = "; ".join(issue.code for issue in coverage.issues)
        raise RuntimeError(f"Lifecycle coverage did not close: {details}")
    if document.sha256 not in set(source_archive["archive_id"].astype(str)):
        raise RuntimeError("The exact evidence report was not added to source_archive.")
    lifecycle_metadata = {
        "operation": "finalize_us_lifecycle_coverage",
        "finalizer_version": FINALIZER_VERSION,
        "input_release_version": release.version,
        "input_versions": input_versions,
        "output_versions": planned_versions,
        "adjustment_source_version": adjustment_source_version,
        "evidence_report_sha256": document.sha256,
        "preserved_exact_repairs": sorted(preserved_exact_repairs),
        **coverage.manifest_metadata(),
    }
    pointer_etags = {
        dataset: repository.current_pointer(dataset)[1]
        for dataset in WRITE_DATASETS
    }
    _assert_release_unchanged(repository, release, release_etag)
    summary = {
        "status": "validated_plan",
        "mode": "plan",
        "network_accessed": False,
        "writes_performed": False,
        "release_version": release.version,
        "completed_session": release.completed_session,
        "input_versions": input_versions,
        "planned_versions": planned_versions,
        "adjustment_source_version": adjustment_source_version,
        "evidence_report_sha256": document.sha256,
        "coverage": coverage.manifest_metadata(),
        "actions": {
            "existing_count": len(frames["corporate_actions"]),
            "added_count": len(additions),
            "total_count": len(actions),
            "jwn_special_dividend_count": sum(
                row["security_id"] == JWN_SECURITY_ID and row["action_type"] == "special_dividend"
                for row in additions
            ),
            "chk_exe_action_count": sum(
                row["action_type"] == "ticker_change"
                and row["effective_date"] == "2024-10-02"
                and row["new_symbol"] == "EXE"
                for row in additions
            ),
        },
        "archive_artifact_count": len(archived),
        "preserved_exact_repairs": sorted(preserved_exact_repairs),
        "warnings": list(warnings),
    }
    return PreparedFinalization(
        release=release,
        release_etag=release_etag,
        pointer_etags=pointer_etags,
        planned_versions=planned_versions,
        input_versions=input_versions,
        frames=output_frames,
        artifacts=archived,
        coverage_report=coverage,
        evidence_report_sha256=document.sha256,
        lifecycle_metadata=lifecycle_metadata,
        warnings=warnings,
        summary=summary,
    )


def _transaction_record(path: Path, value: Mapping[str, Any]) -> None:
    write_atomic(
        path,
        (
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n"
        ).encode(),
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
        recovery_root = repository.root / "recovery/lifecycle-finalizer"
        pending = tuple(recovery_root.glob("*.json")) if recovery_root.exists() else ()
        if pending:
            raise RuntimeError(
                "Lifecycle-finalizer recovery marker blocks writes: "
                + ", ".join(str(path) for path in pending)
            )
        transaction_root = repository.root / "transactions/lifecycle-finalizer"
        interrupted: list[Path] = []
        if transaction_root.exists():
            for path in transaction_root.glob("*.json"):
                try:
                    status = str(json.loads(path.read_bytes()).get("status") or "")
                except Exception:
                    status = "unreadable"
                if status in {"prepared", "rollback_failed", "unreadable"}:
                    interrupted.append(path)
        if interrupted:
            raise RuntimeError(
                "Interrupted lifecycle-finalizer transaction requires recovery: "
                + ", ".join(str(path) for path in interrupted)
            )
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _persist_archive_payloads(
    repository: LocalDatasetRepository,
    artifacts: Iterable[ArchivedArtifact],
) -> None:
    for item in artifacts:
        destination = repository.root / item.object_path
        if destination.is_file():
            try:
                current = gzip.decompress(destination.read_bytes())
            except Exception as exc:
                raise RuntimeError(
                    f"Existing archive payload is invalid: {destination}"
                ) from exc
            if current != item.artifact.content:
                raise RuntimeError(f"Existing archive payload conflicts: {destination}")
            continue
        write_atomic(destination, gzip.compress(item.artifact.content, mtime=0))
        if gzip.decompress(destination.read_bytes()) != item.artifact.content:
            raise RuntimeError(f"Archive payload verification failed: {destination}")


def _delete_new_pointer(
    repository: LocalDatasetRepository,
    dataset: str,
    planned_version: str,
) -> None:
    key = repository.current_key(dataset)
    current = repository.objects.get(key)
    pointer = CurrentPointer.from_bytes(current.data)
    if pointer.version != planned_version:
        raise RuntimeError(
            f"unexpected pointer version during rollback: {pointer.version}"
        )
    root = repository.root.resolve()
    path = (repository.root / key).resolve()
    if root not in path.parents:
        raise RuntimeError(f"pointer path escapes repository: {path}")
    if repository.objects.get(key).etag != current.etag:
        raise RuntimeError(f"pointer changed during rollback: {dataset}")
    path.unlink()
    try:
        repository.objects.get(key)
    except ObjectNotFound:
        return
    raise RuntimeError(f"new pointer deletion verification failed: {dataset}")


def _restore_transaction_state(
    repository: LocalDatasetRepository,
    *,
    old_release_bytes: bytes,
    old_pointer_bytes: Mapping[str, bytes | None],
    planned_versions: Mapping[str, str],
    committed_release_version: str,
) -> tuple[str, ...]:
    errors: list[str] = []
    release_key = "releases/current.json"
    try:
        current = repository.objects.get(release_key)
        if current.data != old_release_bytes:
            observed = DataRelease.from_bytes(current.data)
            is_transaction_release = (
                bool(committed_release_version)
                and observed.version == committed_release_version
            ) or all(
                observed.dataset_versions.get(dataset) == version
                for dataset, version in planned_versions.items()
            )
            if not is_transaction_release:
                raise RuntimeError(
                    f"unexpected release during rollback: {observed.version}"
                )
            repository.objects.put(
                release_key,
                old_release_bytes,
                if_match=current.etag,
            )
        if repository.objects.get(release_key).data != old_release_bytes:
            raise RuntimeError("release preimage verification failed")
    except Exception as exc:
        errors.append(f"{release_key}: {type(exc).__name__}: {exc}")

    for dataset in reversed(WRITE_DATASETS):
        key = repository.current_key(dataset)
        old_bytes = old_pointer_bytes[dataset]
        try:
            try:
                current = repository.objects.get(key)
            except ObjectNotFound:
                current = None
            if old_bytes is None:
                if current is not None:
                    _delete_new_pointer(
                        repository,
                        dataset,
                        planned_versions[dataset],
                    )
                continue
            if current is None:
                raise RuntimeError("current pointer disappeared during rollback")
            if current.data != old_bytes:
                pointer = CurrentPointer.from_bytes(current.data)
                if pointer.version != planned_versions[dataset]:
                    raise RuntimeError(
                        f"unexpected pointer version during rollback: {pointer.version}"
                    )
                repository.objects.put(key, old_bytes, if_match=current.etag)
            if repository.objects.get(key).data != old_bytes:
                raise RuntimeError("pointer preimage verification failed")
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _assert_applied_release_invariant(
    repository: LocalDatasetRepository,
    release: DataRelease,
) -> None:
    current, _etag = repository.current_release()
    if current is None or current.to_bytes() != release.to_bytes():
        raise RuntimeError("Committed lifecycle-finalizer release is not current.")
    for dataset, version in release.dataset_versions.items():
        pointer, _pointer_etag = repository.current_pointer(dataset)
        if pointer is None or pointer.version != version:
            actual = pointer.version if pointer is not None else "missing"
            raise RuntimeError(
                f"Applied release pointer mismatch for {dataset}: "
                f"expected={version}, actual={actual}."
            )
    expected_adjustment_source = _adjustment_source_version(
        release.dataset_versions.get("daily_price_raw", ""),
        release.dataset_versions.get("corporate_actions", ""),
    )
    factor_version = release.dataset_versions.get("adjustment_factors", "")
    if not factor_version:
        raise RuntimeError("Applied release is missing adjustment_factors.")
    factor_manifest = repository.manifest_for_version(
        "adjustment_factors",
        factor_version,
    )
    if factor_manifest.metadata.get("source_version") != expected_adjustment_source:
        raise RuntimeError(
            "Applied adjustment manifest does not name the release action inputs."
        )
    _assert_adjustment_output_lineage(
        repository.read_frame("adjustment_factors", factor_version),
        expected_adjustment_source,
    )
    post = validate_operational_repository_snapshot(repository)
    post.raise_for_errors()


def apply_finalization(
    repository: LocalDatasetRepository,
    prepared: PreparedFinalization,
    *,
    failure_injector: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    inject = failure_injector or (lambda _stage: None)
    with _exclusive_repository_lock(repository):
        _assert_release_unchanged(repository, prepared.release, prepared.release_etag)
        planned_versions = dict(prepared.planned_versions)
        if set(planned_versions) != set(WRITE_DATASETS) or any(
            not value for value in planned_versions.values()
        ):
            raise RuntimeError("Prepared finalization has incomplete planned versions.")
        if len(set(planned_versions.values())) != len(WRITE_DATASETS):
            raise RuntimeError("Prepared finalization dataset versions are not unique.")
        adjustment_source_version = _adjustment_source_version(
            prepared.release.dataset_versions.get("daily_price_raw", ""),
            planned_versions["corporate_actions"],
        )
        _assert_adjustment_output_lineage(
            prepared.frames["adjustment_factors"],
            adjustment_source_version,
        )
        old_release_value = repository.objects.get("releases/current.json")
        old_pointer_bytes: dict[str, bytes | None] = {}
        for dataset in WRITE_DATASETS:
            expected_etag = prepared.pointer_etags[dataset]
            pointer, actual_etag = repository.current_pointer(dataset)
            expected_version = prepared.release.dataset_versions.get(dataset)
            if actual_etag != expected_etag:
                raise RuntimeError(f"{dataset} pointer changed before apply.")
            if expected_version:
                if pointer is None or pointer.version != expected_version:
                    raise RuntimeError(
                        f"{dataset} pointer does not match the frozen release."
                    )
                old_pointer_bytes[dataset] = repository.objects.get(
                    repository.current_key(dataset)
                ).data
            else:
                if pointer is not None:
                    raise RuntimeError(
                        f"{dataset} has an uncommitted pointer outside the frozen release."
                    )
                old_pointer_bytes[dataset] = None

        transaction_id = uuid.uuid4().hex
        journal_path = (
            repository.root
            / "transactions/lifecycle-finalizer"
            / f"{transaction_id}.json"
        )
        journal: dict[str, Any] = {
            "transaction_id": transaction_id,
            "status": "prepared",
            "base_release_version": prepared.release.version,
            "old_release_base64": base64.b64encode(old_release_value.data).decode("ascii"),
            "old_pointer_base64": {
                dataset: (
                    base64.b64encode(value).decode("ascii")
                    if value is not None
                    else None
                )
                for dataset, value in old_pointer_bytes.items()
            },
            "planned_versions": planned_versions,
            "created_at": utc_now_iso(),
        }
        _transaction_record(journal_path, journal)

        committed_release: DataRelease | None = None
        try:
            _persist_archive_payloads(repository, prepared.artifacts)
            inject("after_archive_payloads")
            versions = dict(prepared.release.dataset_versions)
            metadata = {
                "operation": "finalize_us_lifecycle_coverage",
                "input_release_version": prepared.release.version,
                "input_versions": prepared.input_versions,
            }
            adjustment_metadata = {
                **metadata,
                "source_version": adjustment_source_version,
                "source_daily_price_version": prepared.release.dataset_versions[
                    "daily_price_raw"
                ],
                "source_corporate_actions_version": planned_versions[
                    "corporate_actions"
                ],
            }
            for dataset in WRITE_DATASETS:
                result = repository.write_frame(
                    dataset,
                    prepared.frames[dataset],
                    completed_session=prepared.release.completed_session,
                    incomplete_action_policy="block",
                    metadata=(
                        prepared.lifecycle_metadata
                        if dataset == "lifecycle_resolutions"
                        else adjustment_metadata
                        if dataset == "adjustment_factors"
                        else metadata
                    ),
                    expected_pointer_etag=prepared.pointer_etags[dataset],
                    version=planned_versions[dataset],
                )
                if result.conflict:
                    raise RuntimeError(
                        f"{dataset} write conflicted: {result.conflict_path}"
                    )
                versions[dataset] = result.manifest.version
                inject(f"after_{dataset}")

            post = validate_operational_repository_snapshot(repository)
            post.raise_for_errors()
            warnings = tuple(
                dict.fromkeys((*prepared.release.warnings, *prepared.warnings))
            )
            committed_release = repository.commit_release(
                prepared.release.completed_session,
                versions,
                quality=DataQuality.DEGRADED if warnings else DataQuality.VALID,
                warnings=warnings,
                expected_etag=prepared.release_etag,
            )
            inject("after_release_commit")
            _assert_applied_release_invariant(repository, committed_release)
            journal["status"] = "committed"
            journal["committed_release_version"] = committed_release.version
            journal["completed_at"] = utc_now_iso()
            _transaction_record(journal_path, journal)
            return {
                **prepared.summary,
                "status": "applied",
                "mode": "apply",
                "writes_performed": True,
                "new_release_version": committed_release.version,
                "new_dataset_versions": versions,
                "quality": committed_release.quality,
                "transaction_id": transaction_id,
            }
        except BaseException as original:
            rollback_errors = _restore_transaction_state(
                repository,
                old_release_bytes=old_release_value.data,
                old_pointer_bytes=old_pointer_bytes,
                planned_versions=planned_versions,
                committed_release_version=(
                    committed_release.version if committed_release is not None else ""
                ),
            )
            journal["status"] = "rollback_failed" if rollback_errors else "rolled_back"
            journal["original_error"] = f"{type(original).__name__}: {original}"
            journal["rollback_errors"] = list(rollback_errors)
            journal["completed_at"] = utc_now_iso()
            _transaction_record(journal_path, journal)
            if rollback_errors:
                recovery_path = (
                    repository.root
                    / "recovery/lifecycle-finalizer"
                    / f"{transaction_id}.json"
                )
                _transaction_record(recovery_path, journal)
                raise RuntimeError(
                    "Lifecycle-finalizer rollback failed; recovery marker blocks "
                    f"further writes: {recovery_path}; errors={rollback_errors}"
                ) from original
            raise


def run(
    args: argparse.Namespace,
    *,
    repository_factory: Callable[[str | Path], LocalDatasetRepository] = LocalDatasetRepository,
    candidates: Iterable[LifecycleCandidate] | None = None,
    exception_mapping: Mapping[str, ExceptionSpec] = EXPLICIT_EXCEPTION_MAPPING,
) -> dict[str, Any]:
    repository = repository_factory(args.cache_root)
    release, release_etag = repository.current_release()
    if release is None:
        raise RuntimeError("A current local release is required.")
    document = load_report_document(Path(args.report))
    official_evidence_specs = load_official_lifecycle_exception_evidence(
        Path(getattr(args, "hints", DEFAULT_HINTS))
    )
    prepared = prepare_finalization(
        repository,
        release,
        release_etag,
        document,
        sec_cache=Path(args.sec_cache),
        exception_mapping=exception_mapping,
        official_evidence_specs=official_evidence_specs,
        candidates=candidates,
        hints_path=Path(getattr(args, "hints", DEFAULT_HINTS)),
    )
    return apply_finalization(repository, prepared) if bool(args.apply) else prepared.summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
