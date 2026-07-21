from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import URLError

import exchange_calendars as xcals
import pandas as pd
import yaml

from supertrend_quant.market_store.cross_validation import (
    TRUSTED_PINNED_EXTERNAL_OVERLAPS,
    TRUSTED_REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_SPECS,
    TRUSTED_NTCO_EVIDENCE_BINDING_EVENT_IDS,
    TRUSTED_NTCO_EVIDENCE_BINDINGS_SHA256,
    TRUSTED_SIVB_EVIDENCE_BINDING_EVENT_IDS,
    TRUSTED_SIVB_EVIDENCE_BINDINGS_SHA256,
    _check_report_rows,
    reviewed_identity_bound_hint_sha256,
    reviewed_no_data_successor_chain_sha256,
    reviewed_terminal_event_gate_mismatches,
    terminal_event_gate_action_semantics,
    terminal_event_gate_archive_semantics,
    terminal_event_gate_report_semantics,
    terminal_event_gate_resolution_semantics,
    trusted_sivb_evidence_binding_diagnostic,
    trusted_sivb_evidence_binding_inventory_sha256,
    trusted_sivb_evidence_bindings,
    trusted_sivb_report_diagnostic_passed,
    trusted_ntco_evidence_binding_diagnostic,
    trusted_ntco_evidence_binding_inventory_sha256,
    trusted_ntco_evidence_bindings,
    trusted_ntco_report_diagnostic_passed,
)
from supertrend_quant.market_store.yahoo_chart import (
    YahooChartCache as RawYahooChartCache,
    normalize_yahoo_symbol,
    parse_yahoo_chart_json,
    parse_yahoo_chart_no_data_evidence,
)


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts/validate_us_lifecycle_cross_sources.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_us_lifecycle_cross_sources", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot import {SCRIPT_PATH}")
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)

IDENTITY_SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts/collect_us_index_identity_repairs.py"
)
IDENTITY_SPEC = importlib.util.spec_from_file_location(
    "collect_us_index_identity_repairs_for_cross_validation_test",
    IDENTITY_SCRIPT_PATH,
)
if IDENTITY_SPEC is None or IDENTITY_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot import {IDENTITY_SCRIPT_PATH}")
identity_script = importlib.util.module_from_spec(IDENTITY_SPEC)
sys.modules[IDENTITY_SPEC.name] = identity_script
IDENTITY_SPEC.loader.exec_module(identity_script)


POLICY_PATH = Path(__file__).parents[1] / "configs/us_cross_validation.yaml"
ROOT = Path(__file__).parents[2]
PRODUCTION_YAHOO_CACHE = ROOT / "data/cache/state/us_cross_validation/yahoo_chart"
SYMC_NLOK_NO_DATA_REQUESTS = (
    (
        "SYMC",
        1420070400,
        1572652800,
        "https://query1.finance.yahoo.com/v8/finance/chart/SYMC?period1=1420070400&period2=1572652800&events=history&includeAdjustedClose=true&interval=1d",
        "f0b5f9feeac9f1fc93c31b1edc03fbc1fda4acb3ec5e233ebb39d6f8b66e1769",
        "d0a3dcdb139b2664e528c5de4d35759bb6e548fa87c338e6b8780b96a1d0b245",
    ),
    (
        "NLOK",
        1572825600,
        1667865600,
        "https://query1.finance.yahoo.com/v8/finance/chart/NLOK?period1=1572825600&period2=1667865600&events=history&includeAdjustedClose=true&interval=1d",
        "f0b5f9feeac9f1fc93c31b1edc03fbc1fda4acb3ec5e233ebb39d6f8b66e1769",
        "afe8c6a11f3f9e285e0a646aefed2374a8fb662c935c27dcb667b9a68a0441b7",
    ),
)

REVIEWED_NO_DATA_PROMOTIONS = (
    (
        "HFC",
        "US:EODHD:234e6b6a-3fdb-53df-8c0e-e6de98a8563a",
        "2015-01-01",
        "2022-03-14",
        "2022-03-15",
        "US:EODHD:636ef90b-6f62-589e-8cbe-368e89552f16",
        "DINO",
        "b7b1b1bfc538b5fc8fc282256792e9af04153ec2fc328410af1f02480647d144",
        "ticker_change",
        "35cde40250334ff984829ae59905a8a8976bba97cb5896ccd04bb6ee0361b7a8",
    ),
    (
        "TMK",
        "US:EODHD:2dd21653-9767-51df-af08-07ce749ea5d6",
        "2015-01-01",
        "2019-08-08",
        "2019-08-09",
        "US:EODHD:f1ac2066-e4b0-5e28-8248-be51a504be4b",
        "GL",
        "65162bd127f7c8fc1f223a7689bf4b034bc8d78cef0e029c97f6c5869a742059",
        "ticker_change",
        "8f58a0fae11ce7117d0a70b40ae112438863d6cbce857b4888aabda8feb52b1d",
    ),
    (
        "GPS",
        "US:EODHD:3e7acffc-054a-5173-bcb3-af5b1bf93c93",
        "2015-01-01",
        "2024-08-21",
        "2024-08-22",
        "US:EODHD:6fe06809-dfb5-5ab5-a411-d769c24a645b",
        "GAP",
        "b52e115810345757fc45f42472220f3295798de3349a05c50249243045e73859",
        "ticker_change",
        "8f3d6c07846b118cd7e9386a2277cc06f02bf1fad0201c04dba01a58d43e6249",
    ),
    (
        "ADS",
        "US:EODHD:527e931f-3364-53a2-963a-2755a59461cb",
        "2015-01-01",
        "2022-04-01",
        "2022-04-04",
        "US:EODHD:dcbd9086-2735-585b-8fb0-1b0fc480ab6c",
        "BFH",
        "dd380edb815ab420f84c7959e09b529a4da0b297d6a5eb0ba434dbc434232d7f",
        "ticker_change",
        "4e76aa6df5f73a43596e69e0cc08ef74932794ebf73dcae1c8c26253aa7d24d8",
    ),
    (
        "MYL",
        "US:EODHD:74df9527-7797-51a0-8a2c-48e4a2ba91dd",
        "2015-01-01",
        "2020-11-13",
        "2020-11-16",
        "US:EODHD:174743d5-32ab-5169-bbfa-7c4fd2fc9739",
        "VTRS",
        "c9adf310af860b0c79d54f87874783da7bdc4df393fd79ec8caf10878d8dc220",
        "stock_merger",
        "4798f65a592588a122fd6f384180ab748550a0a1b0bfb8c64dcd78005a7f05d0",
    ),
    (
        "MMC",
        "US:EODHD:7f097426-7ca9-571f-acba-1c1ad87568ad",
        "2015-01-01",
        "2026-01-13",
        "2026-01-14",
        "US:EODHD:52caaf4b-4064-554f-8547-5cb604115917",
        "MRSH",
        "2c399d5446bf31396fbb94bbf7bd2a417ad10cefaa9a4fbefda26c8d136e9a5c",
        "ticker_change",
        "0c513f492b70b4a03d035024ded9893a16a779c4abbf13ee0ef0fb0426b88e1a",
    ),
    (
        "PKI",
        "US:EODHD:9a968d54-1ad6-5daf-9edd-ae838a9569b3",
        "2015-01-01",
        "2023-05-15",
        "2023-05-16",
        "US:EODHD:9dd2300f-f254-560c-804d-952867f0a126",
        "RVTY",
        "88ede6dc475ca5af3d7fd1387338e0067ec0fb5254f4bfd26852e5c040e5e8d0",
        "ticker_change",
        "d9975667b18a468e9f4a52e8858f6a935e2bf0abfb2f33d3cb036229cd479156",
    ),
    (
        "PX",
        "US:EODHD:f2724822-1b78-5cac-9dda-67144f42a664",
        "2015-01-01",
        "2018-10-30",
        "2018-10-31",
        "US:EODHD:4d6e6bce-789a-539f-b19d-ad03863c0ac0",
        "LIN",
        "5f92af750cd080a5ea16563063377d1439d4f0439e7d9ff9575351061ae99b0d",
        "stock_merger",
        "df5989da85287bbaaf65cfa35348e01a9f8faa99870db9e7c37908d2dbc7af64",
    ),
    (
        "CTL",
        "US:EODHD:fc00ff9c-3a71-5995-968e-bc351f950cb4",
        "2015-01-01",
        "2020-09-17",
        "2020-09-18",
        "US:EODHD:3d809999-79d0-52f9-b937-a51185539c4d",
        "LUMN",
        "67c829d4b6057105578c6d20fbf6231b5b22716cd68e0f637f4a80ccb894ef15",
        "ticker_change",
        "aec54fbabeb8e22fa9b34af648d5caa491f51bccdf0c4780f603f97e511836b1",
    ),
)


def _policy():
    return script.load_policy(POLICY_PATH)


def _policy_without_terminal_event_gates(*event_ids: str):
    """Keep synthetic legacy-policy tests independent of release-bound gates."""

    value = copy.deepcopy(_policy().value)
    excluded = set(event_ids)
    value["events"]["reviewed_terminal_event_gates"] = [
        item
        for item in value["events"]["reviewed_terminal_event_gates"]
        if item["event_id"] not in excluded
    ]
    return script.Policy(value)


def _archive_row(
    source_hash: str,
    source_url: str,
    *,
    source: str = "sec_edgar_filing",
) -> dict[str, str]:
    return {
        "archive_id": source_hash,
        "dataset": source,
        "source": source,
        "source_hash": source_hash,
        "source_url": source_url,
    }


def _chart_json(
    frame: pd.DataFrame,
    *,
    symbol: str = "SEC",
    currency: str = "USD",
    adjclose: list[float] | None = None,
    error=None,
) -> bytes:
    if error is not None:
        return json.dumps({"chart": {"result": None, "error": error}}).encode()
    sessions = pd.to_datetime(frame.get("session", pd.Series(dtype="datetime64[ns]")))
    timestamps = [
        int((pd.Timestamp(value).tz_localize("UTC") + pd.Timedelta(hours=16)).timestamp())
        for value in sessions
    ]
    quote = {
        field: [float(value) for value in frame.get(field, pd.Series(dtype="float64"))]
        for field in ("open", "high", "low", "close", "volume")
    }
    indicators = {"quote": [quote]}
    if adjclose is not None:
        indicators["adjclose"] = [{"adjclose": list(adjclose)}]
    return json.dumps(
        {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "symbol": symbol,
                            "currency": currency,
                            "instrumentType": "EQUITY",
                            "exchangeName": "NMS",
                            "exchangeTimezoneName": "America/New_York",
                            "dataGranularity": "1d",
                        },
                        "timestamp": timestamps,
                        "indicators": indicators,
                    }
                ],
                "error": None,
            }
        },
        separators=(",", ":"),
    ).encode()


def _retired_yhd_placeholder_json(symbol: str) -> bytes:
    return json.dumps(
        {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "symbol": symbol,
                            "currency": None,
                            "instrumentType": "MUTUALFUND",
                            "exchangeName": "YHD",
                            "fullExchangeName": "YHD",
                            "exchangeTimezoneName": "America/New_York",
                            "dataGranularity": "1d",
                            "range": "",
                        },
                        "indicators": {"quote": [{}], "adjclose": [{}]},
                    }
                ],
                "error": None,
            }
        },
        separators=(",", ":"),
    ).encode()


def _request_periods(start: str, end: str) -> tuple[int, int]:
    start_day = pd.Timestamp(start, tz="UTC")
    end_day = pd.Timestamp(end, tz="UTC")
    return (
        int(start_day.timestamp()),
        int((end_day + pd.Timedelta(days=1)).timestamp()),
    )


def _source_url(
    symbol: str,
    start: str = "2024-01-02",
    end: str = "2024-02-29",
) -> str:
    period1, period2 = _request_periods(start, end)
    return RawYahooChartCache(Path("unused")).url(
        symbol,
        period1=period1,
        period2=period2,
    )


def _response(
    symbol: str,
    payload: bytes,
    *,
    start: str,
    end: str,
    http_status: int = 200,
    wrapper_hash: str = "",
) -> script.CachedResponse:
    period1, period2 = _request_periods(start, end)
    return script.CachedResponse(
        symbol=symbol,
        source_url=_source_url(symbol, start, end),
        retrieved_at="2026-01-01T00:00:00Z",
        content=payload,
        content_type="application/json",
        http_status=http_status,
        wrapper_hash=wrapper_hash,
        request_period1=period1,
        request_period2=period2,
    )


def _bars(sessions, *, scale: float = 1.0) -> pd.DataFrame:
    close = pd.Series(range(100, 100 + len(sessions)), dtype="float64") * scale
    return pd.DataFrame(
        {
            "session": pd.to_datetime(sessions),
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": 1000.0,
        }
    )


class EventCrossValidationTest(unittest.TestCase):
    def _fixtures(self):
        evidence = "a" * 64
        action = pd.DataFrame(
            [
                {
                    "event_id": "event-1",
                    "security_id": "OLD",
                    "action_type": "stock_merger",
                    "effective_date": "2024-01-05",
                    "cash_amount": 2.5,
                    "ratio": 0.4,
                    "new_symbol": "NEW",
                    "new_security_id": "NEW-ID",
                    "currency": "USD",
                    "official": True,
                    "source_kind": "official_crosscheck",
                    "source_url": "https://www.sec.gov/Archives/event.txt",
                    "source_hash": evidence,
                }
            ]
        )
        resolutions = pd.DataFrame(
            [
                {
                    "candidate_id": "candidate-1",
                    "security_id": "OLD",
                    "symbol": "OLD",
                    "resolution": "applied",
                    "event_id": "event-1",
                    "successor_security_id": "NEW-ID",
                }
            ]
        )
        report = {
            "records": {
                "OLD": {
                    "verified_event": {
                        "action_type": "stock_merger",
                        "effective_date": "2024-01-05",
                        "cash_amount": 2.5,
                        "ratio": 0.4,
                        "new_symbol": "NEW",
                        "source_url": "https://www.sec.gov/Archives/event.txt",
                        "source_hash": evidence,
                    }
                }
            }
        }
        return action, resolutions, report, evidence

    def test_every_applied_event_requires_exact_official_terms_and_hash(self):
        actions, resolutions, report, evidence = self._fixtures()
        archive = pd.DataFrame(
            [_archive_row(evidence, "https://www.sec.gov/Archives/event.txt")]
        )
        checks = script.build_event_checks(
            actions, resolutions, report, archive, _policy()
        )
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]["status"], "passed")
        self.assertTrue(checks[0]["date_match"])
        self.assertTrue(checks[0]["terms_match"])
        self.assertTrue(checks[0]["official_original"])

        actions.loc[0, "ratio"] = 0.5
        failed = script.build_event_checks(
            actions, resolutions, report, archive, _policy()
        )
        self.assertEqual(failed[0]["status"], "mismatch")
        self.assertFalse(failed[0]["terms_match"])

    def test_lila_terminal_overrides_only_collector_price_heuristic(self):
        source_url = (
            "https://www.sec.gov/Archives/edgar/data/1570585/"
            "000157058517000401/ex991split-offrecordanddis.htm"
        )
        source_hash = (
            "0efad7b02b77a0daefab021c58fdbbb40f03955f069f42eac3e24d403f2813e4"
        )
        rows = [
            (
                "52e8663611264e84d2b91d4c2eb5fd8346f001086987649d097950a420e66c05",
                "US:EODHD:5c946c06-0214-5b7b-8e7c-31f91485a215",
                "US:EODHD:1b6b9beb-42b0-5a06-81f3-23a49627565f",
                "LILA",
            ),
            (
                "94b7da742aa70fd546532862fdab23dd9bcc15b0c48efb7efdbde1f66d378630",
                "US:EODHD:24bfb026-6327-5e04-9e32-15589dcb45ba",
                "US:EODHD:7fda02a3-10dd-51a3-96cb-41695fcff341",
                "LILAK",
            ),
        ]
        actions = pd.DataFrame(
            [
                {
                    "event_id": event_id,
                    "security_id": security_id,
                    "action_type": "stock_merger",
                    "effective_date": "2018-01-02",
                    "cash_amount": None,
                    "ratio": 1.0,
                    "new_symbol": symbol,
                    "new_security_id": successor_id,
                    "currency": "USD",
                    "official": True,
                    "source_kind": "official_crosscheck",
                    "source_url": source_url,
                    "source_hash": source_hash,
                }
                for event_id, security_id, successor_id, symbol in rows
            ]
        )
        resolutions = pd.DataFrame(
            [
                {
                    "candidate_id": f"candidate-{symbol.lower()}",
                    "security_id": security_id,
                    "symbol": symbol,
                    "resolution": "applied",
                    "event_id": event_id,
                    "successor_security_id": successor_id,
                }
                for event_id, security_id, successor_id, symbol in rows
            ]
        )
        report = {
            "records": {
                security_id: {
                    "candidate": {"security_id": security_id},
                    "eligible_for_apply": False,
                    "manual_review_reason": "",
                    "crosscheck": {
                        "passed": False,
                        "date_passed": True,
                        "economic_terms_passed": False,
                    },
                    "parsed": {
                        "action_type": "stock_merger",
                        "effective_date": "2018-01-02",
                        "cash_amount": None,
                        "ratio": 1.0,
                        "new_symbol": symbol,
                    },
                    "source_url": source_url,
                    "source_hash": source_hash,
                    "successor_security_id": successor_id,
                }
                for _, security_id, successor_id, symbol in rows
            }
        }
        archive = pd.DataFrame([_archive_row(source_hash, source_url)])

        checks = script.build_event_checks(
            actions, resolutions, report, archive, _policy()
        )
        self.assertEqual({item["status"] for item in checks}, {"passed"})
        self.assertTrue(
            all(not item["lifecycle_report_collector_approved"] for item in checks)
        )
        self.assertTrue(
            all(item["reviewed_terminal_override_applied"] for item in checks)
        )
        self.assertTrue(
            all(item["reviewed_terminal_override_match"] for item in checks)
        )
        self.assertTrue(
            all(len(item["reviewed_terminal_override_sha256"]) == 64 for item in checks)
        )

        mutations = {
            "action_ratio": lambda a, r, p: a.loc.__setitem__(
                (0, "ratio"), 1.0000000000001
            ),
            "action_official": lambda a, r, p: a.loc.__setitem__(
                (0, "official"), False
            ),
            "parsed_date": lambda a, r, p: p["records"][rows[0][1]][
                "parsed"
            ].__setitem__("effective_date", "2018-01-03"),
            "parsed_ratio_exact": lambda a, r, p: p["records"][rows[0][1]][
                "parsed"
            ].__setitem__("ratio", 1.0000000000001),
            "report_source": lambda a, r, p: p["records"][rows[0][1]].__setitem__(
                "source_url", source_url + "?changed=1"
            ),
            "report_successor": lambda a, r, p: p["records"][rows[0][1]].__setitem__(
                "successor_security_id", "DIFFERENT"
            ),
            "report_security": lambda a, r, p: p["records"][rows[0][1]][
                "candidate"
            ].__setitem__("security_id", "DIFFERENT"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                changed_actions = actions.copy(deep=True)
                changed_resolutions = resolutions.copy(deep=True)
                changed_report = copy.deepcopy(report)
                mutate(changed_actions, changed_resolutions, changed_report)
                result = script.build_event_checks(
                    changed_actions,
                    changed_resolutions,
                    changed_report,
                    archive,
                    _policy(),
                )
                by_id = {item["event_id"]: item for item in result}
                self.assertEqual(by_id[rows[0][0]]["status"], "mismatch")
                self.assertFalse(
                    by_id[rows[0][0]]["reviewed_terminal_override_applied"]
                )

    def test_terminal_override_inventory_is_code_pinned(self):
        policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            set(script.reviewed_terminal_overrides(policy["events"])),
            set(script.TRUSTED_REVIEWED_TERMINAL_OVERRIDE_EVENT_IDS),
        )
        self.assertEqual(
            script.reviewed_terminal_override_inventory_sha256(policy["events"]),
            script.TRUSTED_REVIEWED_TERMINAL_OVERRIDES_SHA256,
        )

        policy["events"]["reviewed_terminal_overrides"][0]["ratio"] = 1.0001
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.yaml"
            path.write_text(yaml.safe_dump(policy), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not code-pinned"):
                script.load_policy(path)

    def test_terminal_event_gate_inventory_is_code_pinned_and_stale_routes_removed(self):
        policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
        gates = script.reviewed_terminal_event_gates(policy["events"])
        self.assertEqual(len(gates), 18)
        self.assertEqual(
            set(gates),
            set(script.TRUSTED_REVIEWED_TERMINAL_EVENT_GATE_EVENT_IDS),
        )
        self.assertEqual(
            script.reviewed_terminal_event_gate_inventory_sha256(
                policy["events"]
            ),
            script.TRUSTED_REVIEWED_TERMINAL_EVENT_GATES_SHA256,
        )

        by_symbol = {item["symbol"]: item for item in gates.values()}
        self.assertEqual(
            {
                symbol: by_symbol[symbol]["policy_code"]
                for symbol in (
                    "HRS",
                    "LLL",
                    "ALXN",
                    "CXO",
                    "NLOK",
                    "NLSN",
                    "AVP",
                )
            },
            {
                "HRS": "terminal_close_before_legal_completion/v1",
                "LLL": "terminal_close_before_legal_completion/v1",
                "ALXN": "terminal_close_before_legal_completion/v1",
                "CXO": "provider_tail_market_transition/v1",
                "NLOK": "provider_tail_market_transition/v1",
                "NLSN": "provider_tail_market_transition/v1",
                "AVP": (
                    "legal_completion_before_market_transition_"
                    "missing_report_successor/v1"
                ),
            },
        )
        self.assertEqual(
            by_symbol["BMYRT"]["policy_code"],
            "bmyrt_official_exit_mark/v1",
        )
        self.assertEqual(
            by_symbol["ECA"]["archive_ids"],
            [
                "3958dc0304e1449a9fd3e33d538877eddd70053c897b61e0fb6666555e05967c",
                "5550528f1a63b94d35e6880c9c046756a64390f6675a65e2bf728dc560166474",
                "58d199861b620211b63c846e3184baf1ff7982adb124e085c5f726e2fd06af59",
            ],
        )
        self.assertEqual(
            set(script.reviewed_terminal_market_date_corrections(policy["events"])),
            {
                "39caca00fbedb7b23e8b5294cdf3bdc5aba9014f98bf72b6a6da341602651192",
                "951960822646a8f972002508d162bd37a3ab3aaec42349fbff37caa1919b7b51",
                "ea791beeaa569aaf19b1df04e59df3710d45382cd70e5d82078d7247286272c6",
            },
        )
        self.assertTrue(
            {"WFM", "AVP", "SIVBQ"}.isdisjoint(
                item["symbol"]
                for item in script.reviewed_terminal_market_date_corrections(
                    policy["events"]
                ).values()
            )
        )

        sivbq = by_symbol["SIVBQ"]
        self.assertEqual(
            reviewed_identity_bound_hint_sha256(sivbq["hint_key"]),
            sivbq["hint_sha256"],
        )
        for field in (
            "candidate_id",
            "action_sha256",
            "resolution_sha256",
            "report_semantic_sha256",
            "archive_binding_sha256",
            "lifecycle_evidence_report_sha256",
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                mutated = copy.deepcopy(policy)
                current = mutated["events"]["reviewed_terminal_event_gates"][0][
                    field
                ]
                mutated["events"]["reviewed_terminal_event_gates"][0][field] = (
                    "0" * 64 if current != "0" * 64 else "1" * 64
                )
                path = Path(directory) / "policy.yaml"
                path.write_text(yaml.safe_dump(mutated), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "not code-pinned"):
                    script.load_policy(path)

    def test_terminal_event_gate_binds_action_resolution_report_archive_and_dates(self):
        event_id = (
            "25bce725b19ce21cebac0fa09351a30e5b89479256f7d1ed25f9218b557754c4"
        )
        candidate_id = (
            "98d3b9997c6c9cab63831a97574cb9126c7db0cfeff661b447755caa60d3d97c"
        )
        security_id = "US:EODHD:c24f6a80-3a51-56f7-9c55-68916e553fad"
        source_url = "https://www.sec.gov/Archives/edgar/data/example.txt"
        archive_id = "6" * 64
        lifecycle_hash = "a" * 64
        action = {
            "event_id": event_id,
            "security_id": security_id,
            "action_type": "cash_merger",
            "effective_date": "2017-08-28",
            "ex_date": "2017-08-28",
            "announcement_date": "2017-08-28",
            "record_date": "",
            "payment_date": "",
            "cash_amount": 42.0,
            "ratio": None,
            "new_symbol": "",
            "new_security_id": "",
            "currency": "USD",
            "official": True,
            "source_kind": "official_crosscheck",
            "source": "sec_edgar_filing",
            "source_url": source_url,
            "source_hash": archive_id,
            "retrieved_at": "2026-07-18T00:00:00Z",
            "metadata": "{}",
        }
        resolution = {
            "candidate_id": candidate_id,
            "security_id": security_id,
            "symbol": "WFM",
            "last_price_date": "2017-08-25",
            "resolution": "applied",
            "event_id": event_id,
            "exception_code": "",
            "exception_reason": "",
            "reviewed_by": "codex",
            "reviewed_at": "2026-07-18T00:00:00Z",
            "recheck_after": "",
            "successor_security_id": "",
            "successor_symbol": "",
            "source_url": source_url,
            "source": "sec_edgar_filing",
            "retrieved_at": "2026-07-18T00:00:00Z",
            "source_hash": archive_id,
        }
        record = {
            "candidate": {
                "candidate_id": candidate_id,
                "security_id": security_id,
                "symbol": "WFM",
                "last_price_date": "2017-08-25",
                "active_to": "2017-08-25",
            },
            "eligible_for_apply": True,
            "manual_review": False,
            "manual_review_reason": "",
            "crosscheck": {
                "passed": True,
                "date_passed": True,
                "economic_terms_passed": True,
            },
            "parsed": {
                "action_type": "cash_merger",
                "effective_date": "2017-08-28",
                "cash_amount": 42.0,
                "ratio": None,
                "new_symbol": "",
                "source_url": source_url,
                "source_hash": archive_id,
            },
            "source_url": source_url,
            "source_hash": archive_id,
            "successor_security_id": "",
            "filing": {
                "accession_number": "0000000000-00-000000",
                "filing_date": "2017-08-28",
            },
        }
        archive = pd.DataFrame(
            [
                {
                    "archive_id": archive_id,
                    "dataset": "sec_edgar_filing",
                    "object_path": "archives/example.txt.gz",
                    "content_type": "text/plain",
                    "effective_date": "2017-08-28",
                    "source": "sec_edgar_filing",
                    "retrieved_at": "2026-07-18T00:00:00Z",
                    "source_hash": archive_id,
                    "source_url": source_url,
                }
            ]
        )
        gate = copy.deepcopy(
            script.reviewed_terminal_event_gates(_policy().events)[event_id]
        )
        gate.update(
            {
                "action_sha256": script.canonical_json_sha256(
                    terminal_event_gate_action_semantics(action)
                ),
                "resolution_sha256": script.canonical_json_sha256(
                    terminal_event_gate_resolution_semantics(resolution)
                ),
                "report_semantic_sha256": script.canonical_json_sha256(
                    terminal_event_gate_report_semantics(record)
                ),
                "archive_ids": [archive_id],
                "archive_binding_sha256": script.canonical_json_sha256(
                    terminal_event_gate_archive_semantics(archive, [archive_id])
                ),
                "lifecycle_evidence_report_sha256": lifecycle_hash,
            }
        )

        def mismatches(
            changed_action=action,
            changed_resolution=resolution,
            changed_record=record,
            changed_archive=archive,
            changed_lifecycle_hash=lifecycle_hash,
        ):
            return reviewed_terminal_event_gate_mismatches(
                changed_action,
                changed_resolution,
                changed_record,
                changed_archive,
                gate,
                changed_lifecycle_hash,
            )

        self.assertEqual(mismatches(), ())

        changed_action = copy.deepcopy(action)
        changed_action["effective_date"] = "2017-08-29"
        self.assertIn("action_sha256", mismatches(changed_action=changed_action))

        changed_resolution = copy.deepcopy(resolution)
        changed_resolution["successor_security_id"] = "DIFFERENT"
        self.assertIn(
            "resolution_sha256",
            mismatches(changed_resolution=changed_resolution),
        )

        changed_resolution = copy.deepcopy(resolution)
        changed_resolution["candidate_id"] = "0" * 64
        self.assertIn(
            "candidate_id",
            mismatches(changed_resolution=changed_resolution),
        )

        changed_resolution = copy.deepcopy(resolution)
        changed_resolution["last_price_date"] = "2017-08-24"
        self.assertIn(
            "canonical_candidate_id",
            mismatches(changed_resolution=changed_resolution),
        )

        changed_record = copy.deepcopy(record)
        changed_record["parsed"]["effective_date"] = "2017-08-23"
        self.assertIn(
            "report_semantic_sha256",
            mismatches(changed_record=changed_record),
        )

        changed_archive = archive.copy(deep=True)
        changed_archive.loc[0, "object_path"] = "archives/changed.txt.gz"
        self.assertIn(
            "archive_binding_sha256",
            mismatches(changed_archive=changed_archive),
        )
        missing_archive_columns = archive.drop(columns=["source_url"])
        self.assertIn(
            "action_archive_pair",
            mismatches(changed_archive=missing_archive_columns),
        )
        self.assertIn(
            "lifecycle_evidence_report_sha256",
            mismatches(changed_lifecycle_hash="f" * 64),
        )

    def test_terminal_market_date_correction_inventory_is_code_pinned(self):
        policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
        corrections = script.reviewed_terminal_market_date_corrections(
            policy["events"]
        )
        self.assertEqual(
            set(corrections),
            set(script.TRUSTED_REVIEWED_TERMINAL_MARKET_DATE_CORRECTION_EVENT_IDS),
        )
        self.assertEqual(
            script.reviewed_terminal_market_date_correction_inventory_sha256(
                policy["events"]
            ),
            script.TRUSTED_REVIEWED_TERMINAL_MARKET_DATE_CORRECTIONS_SHA256,
        )
        self.assertFalse(
            set(corrections)
            & set(script.reviewed_terminal_overrides(policy["events"]))
        )
        self.assertFalse(
            set(corrections)
            & set(script.reviewed_nonterminal_extractions(policy["events"]))
        )

        kors = corrections[
            "951960822646a8f972002508d162bd37a3ab3aaec42349fbff37caa1919b7b51"
        ]
        self.assertEqual(
            {
                "superseded_event_id": kors["superseded_event_id"],
                "candidate_id": kors["candidate_id"],
                "report_effective_date": kors["report_effective_date"],
                "official_completion_date": kors["official_completion_date"],
                "effective_date": kors["effective_date"],
                "date_relation": kors["date_relation"],
                "source_hash": kors["source_hash"],
                "report_source_hash": kors["report_source_hash"],
            },
            {
                "superseded_event_id": (
                    "51ffb3cf286e69bdcc2d66a6945a33cc8cc3deb41661920b5ab4a7fa7b327f36"
                ),
                "candidate_id": (
                    "2a6ac318cc5c6e6c42b4788a75c400278ac09cc59abab518aff6ca44bb2b8512"
                ),
                "report_effective_date": "2018-12-31",
                "official_completion_date": "2018-12-31",
                "effective_date": "2019-01-02",
                "date_relation": "next_xnys_session_after_terminal_close",
                "source_hash": (
                    "ff4732e714524028c56a66c96e6ac8c50a401a36a4a46037cb80b01bb8454d25"
                ),
                "report_source_hash": (
                    "ff4732e714524028c56a66c96e6ac8c50a401a36a4a46037cb80b01bb8454d25"
                ),
            },
        )
        self.assertEqual(
            script.reviewed_terminal_market_date_correction_sha256(kors),
            "2f7fcd592d9daa876e264d505a8fdb85b2c764cc71d4de7c3b2b1c55faad1373",
        )

        policy["events"]["reviewed_terminal_market_date_corrections"][0][
            "ratio"
        ] = 0.99
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.yaml"
            path.write_text(yaml.safe_dump(policy), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not code-pinned"):
                script.load_policy(path)

    def test_utx_ticker_event_is_exact_nonterminal_provenance_only(self):
        policy = _policy()
        event_id = (
            "1eebcc8f193de474779068560d90da76961560ffd1fe459dabc10d3c1085374b"
        )
        extraction = script.reviewed_nonterminal_extractions(policy.events)[event_id]
        self.assertEqual(
            script.reviewed_nonterminal_extraction_sha256(extraction),
            "4b43b89b4fb242784fef1361eeb449a45017e4ea6369cef65e9c1666c8718126",
        )
        action = {**extraction, "official": True}
        archive = pd.DataFrame(
            [_archive_row(extraction["source_hash"], extraction["source_url"])]
        )
        result = script.build_event_checks(
            pd.DataFrame([action]),
            pd.DataFrame(columns=["resolution", "event_id"]),
            {"records": {}},
            archive,
            policy,
        )[0]
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["validation_kind"], script.NONTERMINAL_EVENT_VALIDATION)
        self.assertTrue(result["reviewed_extraction_match"])
        self.assertFalse(result["lifecycle_report_extraction_approved"])

        exception = script.trusted_permanent_exception_specs()[
            "utx_2020_carr_otis_distributions"
        ]
        self.assertEqual(exception.resolution_kind, "exception")
        self.assertEqual(exception.exception_code, "unsupported_consideration")
        self.assertEqual(exception.candidate_last_price_dates, ("2020-04-02",))
        self.assertIn("CARR 1.0-share and OTIS 0.5-share", exception.claim)
        self.assertNotEqual(exception.source_sha256, extraction["source_hash"])

        changed = {**action, "new_symbol": "CARR"}
        mismatch = script.build_event_checks(
            pd.DataFrame([changed]),
            pd.DataFrame(columns=["resolution", "event_id"]),
            {"records": {}},
            archive,
            policy,
        )[0]
        self.assertEqual(mismatch["status"], "mismatch")
        self.assertFalse(mismatch["reviewed_extraction_match"])
        self.assertIn(
            "reviewed nonterminal extraction differs: new_symbol",
            mismatch["reasons"],
        )

    def test_sivbq_verified_hint_is_exactly_identity_bound(self):
        event_id = (
            "f8cbecc851776e882ac85872972ac5c1680a672138276c1a9301d532c2d8ad3f"
        )
        gate = script.reviewed_terminal_event_gates(_policy().events)[event_id]
        self.assertEqual(
            gate["hint_key"],
            "US:EODHD:181d58e2-2b82-5f5d-8c33-311c5d7b5129|2024-11-07",
        )
        self.assertEqual(
            reviewed_identity_bound_hint_sha256(gate["hint_key"]),
            gate["hint_sha256"],
        )

        hints_path = Path(__file__).parents[1] / "configs/us_lifecycle_hints.yaml"
        payload = yaml.safe_load(hints_path.read_text(encoding="utf-8"))
        payload["identity_bound_hints"][gate["hint_key"]]["verified_event"][
            "effective_date"
        ] = "2024-11-08"
        with tempfile.TemporaryDirectory() as directory:
            changed_path = Path(directory) / "hints.yaml"
            changed_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            self.assertNotEqual(
                reviewed_identity_bound_hint_sha256(
                    gate["hint_key"], hints_path=changed_path
                ),
                gate["hint_sha256"],
            )

    def test_sivb_raw_occ_trust_inventory_is_code_pinned_and_narrow(self):
        bindings = trusted_sivb_evidence_bindings()
        self.assertEqual(
            set(bindings),
            set(TRUSTED_SIVB_EVIDENCE_BINDING_EVENT_IDS),
        )
        self.assertEqual(
            trusted_sivb_evidence_binding_inventory_sha256(),
            TRUSTED_SIVB_EVIDENCE_BINDINGS_SHA256,
        )
        policy = _policy().events
        disjoint_event_ids = (
            set(script.reviewed_nonterminal_extractions(policy))
            | set(script.reviewed_terminal_overrides(policy))
            | set(script.reviewed_terminal_policy_exceptions(policy))
            | set(script.reviewed_terminal_price_tail_corrections(policy))
        )
        self.assertFalse(set(bindings) & disjoint_event_ids)
        sivbq_event_id = (
            "f8cbecc851776e882ac85872972ac5c1680a672138276c1a9301d532c2d8ad3f"
        )
        self.assertFalse(
            set(bindings)
            & set(script.reviewed_terminal_market_date_corrections(policy))
        )
        self.assertIn(
            sivbq_event_id,
            script.reviewed_terminal_event_gates(policy),
        )

    def test_sivb_raw_occ_diagnostic_binds_action_archive_and_tamper(self):
        metadata_by_event = {
            "01419d978e03e608512e4e898e695fdb39953278b08dc8138d97e0d0e21e4caa": {
                "cusip": "78486Q101",
                "evidence_binding_schema": "occ_information_memo_binding/v1",
                "memo_number": "52179",
                "nasdaq_halt_date": "2023-03-10",
                "nasdaq_suspension_date": "2023-03-28",
                "occ_disclaimer_role": "unofficial_corporate_event_summary",
                "occ_legacy_reviewed_extraction_sha256": "11e986df7ea010021dd343353e24da8009673053b555b32cce95d4aae1598c6f",
                "occ_official_origin_confirmed": True,
                "occ_raw_pdf_bytes": 566940,
                "occ_raw_pdf_extracted_text_sha256": "cb3bde780b9935d56d7d69105609b9eb4f90d8024489e0b9df2da925b6445673",
                "occ_raw_pdf_object_path": "archives/2026-07-15/28f25f761b61e7f898fa7d1c237520a1a4fed97e88af0b788ef52256608c6035.pdf.gz",
                "occ_raw_pdf_page_count": 2,
                "occ_raw_pdf_reviewed_at": "2026-07-18T18:20:45Z",
                "occ_raw_pdf_reviewed_by": "codex-independent-pdf-review",
                "occ_raw_pdf_sha256": "28f25f761b61e7f898fa7d1c237520a1a4fed97e88af0b788ef52256608c6035",
                "occ_reviewed_extraction_hash": "11e986df7ea010021dd343353e24da8009673053b555b32cce95d4aae1598c6f",
                "otc_open_date": "2023-03-28",
                "policy": "legal_cancellation_and_market_transition_separation/v1",
                "ratio": 1.0,
                "same_common_share_identity": True,
                "sec_market_source_hash": "69f3b20dfab4c9c43641a3c38a99f288129665af40e5ae3e6993ec36ccf4fcef",
            },
            "f8cbecc851776e882ac85872972ac5c1680a672138276c1a9301d532c2d8ad3f": {
                "date_relation": "first_xnys_session_after_last_otc_close",
                "engine_terminal_session": "2024-11-08",
                "last_observed_otc_price_session": "2024-11-07",
                "legal_cancellation_date": "2024-11-07",
                "legal_zero_distribution_preserved": True,
                "original_event_id": "1f4a23cffdf2decb8c26be93d94318d6d5a2be7fc045c33ff9e5abd4e9c69c82",
                "otc_price_source_hash": "038c5a1ab7a5b439835a12507ebacc8bd8342ba73005479a0c57acc60ff04a1f",
                "policy": "legal_cancellation_and_market_transition_separation/v1",
            },
        }
        for event_id, spec in trusted_sivb_evidence_bindings().items():
            with self.subTest(event_id=event_id):
                action = {
                    "event_id": event_id,
                    "security_id": spec["security_id"],
                    "action_type": spec["action_type"],
                    "effective_date": spec["effective_date"],
                    "ex_date": spec["ex_date"],
                    "announcement_date": spec["announcement_date"],
                    "payment_date": spec["payment_date"],
                    "cash_amount": spec["cash_amount"],
                    "ratio": spec["ratio"],
                    "currency": spec["currency"],
                    "new_security_id": spec["new_security_id"],
                    "new_symbol": spec["new_symbol"],
                    "official": True,
                    "source_kind": spec["action_source_kind"],
                    "source": spec["action_source"],
                    "source_url": spec["action_source_url"],
                    "source_hash": spec["action_source_hash"],
                    "metadata": json.dumps(
                        metadata_by_event[event_id],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
                archive = pd.DataFrame(
                    [
                        {
                            "archive_id": evidence["source_hash"],
                            "source_hash": evidence["source_hash"],
                            "source_url": evidence["source_url"],
                            "dataset": evidence["dataset"],
                            "source": evidence["source"],
                            "content_type": evidence["content_type"],
                            "object_path": evidence["object_path"],
                            "effective_date": evidence["effective_date"],
                        }
                        for evidence in spec["evidence"]
                    ]
                )
                exact = trusted_sivb_evidence_binding_diagnostic(
                    action, archive
                )
                self.assertIsNotNone(exact)
                assert exact is not None
                self.assertEqual(exact["status"], "trusted")
                self.assertTrue(exact["action_binding_exact"])
                self.assertTrue(exact["sec_raw_archived"])
                self.assertTrue(exact["eodhd_raw_archived"])
                self.assertTrue(exact["occ_raw_pdf_archived"])
                self.assertTrue(exact["legacy_extraction_archived"])
                self.assertFalse(exact["legacy_extraction_authoritative"])

                tampered_action = dict(action)
                tampered_action["source_hash"] = "0" * 64
                tampered = trusted_sivb_evidence_binding_diagnostic(
                    tampered_action, archive
                )
                self.assertIsNotNone(tampered)
                assert tampered is not None
                self.assertFalse(tampered["action_binding_exact"])
                self.assertEqual(tampered["status"], "blocked")

                tampered_archive = archive.copy(deep=True)
                target = tampered_archive["dataset"].eq("eodhd_eod")
                tampered_archive.loc[target, "object_path"] = "archives/changed.json.gz"
                tampered = trusted_sivb_evidence_binding_diagnostic(
                    action, tampered_archive
                )
                self.assertIsNotNone(tampered)
                assert tampered is not None
                self.assertFalse(
                    tampered["evidence_archive_bindings"][
                        "eodhd_otc_prices_raw"
                    ]
                )
                self.assertEqual(tampered["status"], "blocked")

    def test_sivb_ticker_report_uses_only_exact_raw_occ_trust_path(self):
        event_id = (
            "01419d978e03e608512e4e898e695fdb39953278b08dc8138d97e0d0e21e4caa"
        )
        spec = trusted_sivb_evidence_bindings()[event_id]
        metadata = {
            "cusip": "78486Q101",
            "evidence_binding_schema": "occ_information_memo_binding/v1",
            "memo_number": "52179",
            "nasdaq_halt_date": "2023-03-10",
            "nasdaq_suspension_date": "2023-03-28",
            "occ_disclaimer_role": "unofficial_corporate_event_summary",
            "occ_legacy_reviewed_extraction_sha256": "11e986df7ea010021dd343353e24da8009673053b555b32cce95d4aae1598c6f",
            "occ_official_origin_confirmed": True,
            "occ_raw_pdf_bytes": 566940,
            "occ_raw_pdf_extracted_text_sha256": "cb3bde780b9935d56d7d69105609b9eb4f90d8024489e0b9df2da925b6445673",
            "occ_raw_pdf_object_path": "archives/2026-07-15/28f25f761b61e7f898fa7d1c237520a1a4fed97e88af0b788ef52256608c6035.pdf.gz",
            "occ_raw_pdf_page_count": 2,
            "occ_raw_pdf_reviewed_at": "2026-07-18T18:20:45Z",
            "occ_raw_pdf_reviewed_by": "codex-independent-pdf-review",
            "occ_raw_pdf_sha256": "28f25f761b61e7f898fa7d1c237520a1a4fed97e88af0b788ef52256608c6035",
            "occ_reviewed_extraction_hash": "11e986df7ea010021dd343353e24da8009673053b555b32cce95d4aae1598c6f",
            "otc_open_date": "2023-03-28",
            "policy": "legal_cancellation_and_market_transition_separation/v1",
            "ratio": 1.0,
            "same_common_share_identity": True,
            "sec_market_source_hash": "69f3b20dfab4c9c43641a3c38a99f288129665af40e5ae3e6993ec36ccf4fcef",
        }
        action = {
            "event_id": event_id,
            "security_id": spec["security_id"],
            "action_type": "ticker_change",
            "effective_date": "2023-03-28",
            "ex_date": "2023-03-28",
            "announcement_date": "2023-03-27",
            "payment_date": "",
            "cash_amount": None,
            "ratio": None,
            "new_security_id": spec["security_id"],
            "new_symbol": "SIVBQ",
            "currency": "USD",
            "official": True,
            "source_kind": spec["action_source_kind"],
            "source": spec["action_source"],
            "source_url": spec["action_source_url"],
            "source_hash": spec["action_source_hash"],
            "metadata": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        }
        archive = pd.DataFrame(
            [
                {
                    "archive_id": evidence["source_hash"],
                    "source_hash": evidence["source_hash"],
                    "source_url": evidence["source_url"],
                    "dataset": evidence["dataset"],
                    "source": evidence["source"],
                    "content_type": evidence["content_type"],
                    "object_path": evidence["object_path"],
                    "effective_date": evidence["effective_date"],
                }
                for evidence in spec["evidence"]
            ]
        )
        result = script.build_event_checks(
            pd.DataFrame([action]),
            pd.DataFrame(columns=["resolution", "event_id"]),
            {"records": {}},
            archive,
            _policy(),
        )[0]
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reasons"], [])
        diagnostic = result["trusted_sivb_evidence_binding"]
        self.assertEqual(diagnostic["status"], "trusted")
        self.assertTrue(diagnostic["sec_raw_archived"])
        self.assertTrue(diagnostic["eodhd_raw_archived"])
        self.assertTrue(diagnostic["occ_raw_pdf_archived"])
        self.assertFalse(result["reviewed_extraction_match"])

        self.assertTrue(trusted_sivb_report_diagnostic_passed(result))
        counts = _check_report_rows(
            {"events": [result], "permanent_exceptions": [], "prices": []}
        )
        self.assertEqual(counts["event_mismatch_count"], 0)
        self.assertEqual(counts["nonterminal_event_count"], 1)
        self.assertEqual(counts["reviewed_nonterminal_event_count"], 1)

        tampered = copy.deepcopy(result)
        tampered["trusted_sivb_evidence_binding"]["occ_raw_pdf_archived"] = False
        self.assertFalse(trusted_sivb_report_diagnostic_passed(tampered))
        counts = _check_report_rows(
            {"events": [tampered], "permanent_exceptions": [], "prices": []}
        )
        self.assertEqual(counts["event_mismatch_count"], 1)
        self.assertEqual(counts["reviewed_nonterminal_event_count"], 0)

    def test_ntco_transition_trust_inventory_and_policy_are_exact(self):
        bindings = trusted_ntco_evidence_bindings()
        self.assertEqual(
            set(bindings), set(TRUSTED_NTCO_EVIDENCE_BINDING_EVENT_IDS)
        )
        self.assertEqual(
            trusted_ntco_evidence_binding_inventory_sha256(),
            TRUSTED_NTCO_EVIDENCE_BINDINGS_SHA256,
        )
        policy = _policy().events
        self.assertEqual(
            set(policy["reviewed_ntco_transition_event_ids"]),
            set(TRUSTED_NTCO_EVIDENCE_BINDING_EVENT_IDS),
        )
        other_reviewed = (
            set(script.reviewed_nonterminal_extractions(policy))
            | set(script.reviewed_terminal_overrides(policy))
            | set(script.reviewed_terminal_market_date_corrections(policy))
            | set(script.reviewed_terminal_policy_exceptions(policy))
            | set(script.reviewed_terminal_price_tail_corrections(policy))
            | set(trusted_sivb_evidence_bindings())
        )
        self.assertFalse(set(bindings) & other_reviewed)

        mutated = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
        mutated["events"]["reviewed_ntco_transition_event_ids"] = [
            next(iter(TRUSTED_NTCO_EVIDENCE_BINDING_EVENT_IDS))
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.yaml"
            path.write_text(yaml.safe_dump(mutated), encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, "exact isolated policy set"
            ):
                script.load_policy(path)

    def test_ntco_transition_checks_bind_both_actions_and_all_raw_evidence(self):
        identity_url = "https://infomemo.theocc.com/infomemos?number=54105"
        cash_url = (
            "https://www.adrbny.com/content/dam/adr/documents/"
            "corporate-actions-dr/files/ad1145447.pdf"
        )
        metadata_by_event = {
            "0bd231d216016d33c8d6eb1ec9bb85450a21b9298c4abda34a3399727e9b2c00": {
                "canonical_exchange": "OTC",
                "cboe_source_url": (
                    "https://cdn.cboe.com/resources/product_restriction/2024/"
                    "Cboe-Options-Exchanges-Restrictions-on-Transactions-in-"
                    "Options-on-Natura-Co-Holding-S-A.pdf"
                ),
                "cusip": "63884N108",
                "deliverable": "100 American Depositary Shares",
                "occ_source_url": identity_url,
                "official_destination_market": "Other-OTC",
            },
            "d1eeadf8f4779212ff4b3162af1731bf7bfc804c10756cc4bb8074c307e1f746": {
                "ads_to_underlying_ratio": "1:2",
                "cancellation_fee_per_ads": "0",
                "gross_rate_per_ads": "5.043659",
                "mandatory_exchange": True,
                "net_rate_per_ads": "5.043659",
            },
        }
        actions: list[dict] = []
        archive_rows: list[dict] = []
        for event_id, spec in trusted_ntco_evidence_bindings().items():
            actions.append(
                {
                    "event_id": event_id,
                    "security_id": spec["security_id"],
                    "action_type": spec["action_type"],
                    "effective_date": spec["effective_date"],
                    "ex_date": spec["ex_date"],
                    "announcement_date": spec["announcement_date"],
                    "record_date": spec["record_date"],
                    "payment_date": spec["payment_date"],
                    "cash_amount": spec["cash_amount"],
                    "ratio": spec["ratio"],
                    "currency": spec["currency"],
                    "new_security_id": spec["new_security_id"],
                    "new_symbol": spec["new_symbol"],
                    "official": spec["official"],
                    "source_kind": spec["action_source_kind"],
                    "source": spec["action_source"],
                    "source_url": spec["action_source_url"],
                    "source_hash": spec["action_source_hash"],
                    "retrieved_at": spec["action_retrieved_at"],
                    "metadata": json.dumps(
                        metadata_by_event[event_id],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
            archive_rows.extend(
                {
                    "archive_id": evidence["archive_id"],
                    "source_hash": evidence["source_hash"],
                    "source_url": evidence["source_url"],
                    "dataset": evidence["dataset"],
                    "source": evidence["source"],
                    "content_type": evidence["content_type"],
                    "object_path": evidence["object_path"],
                    "effective_date": evidence["effective_date"],
                    "retrieved_at": evidence["retrieved_at"],
                }
                for evidence in spec["evidence"]
            )
        action_frame = pd.DataFrame(actions)
        archive = pd.DataFrame(archive_rows)
        terminal_id = (
            "d1eeadf8f4779212ff4b3162af1731bf7bfc804c10756cc4bb8074c307e1f746"
        )
        security_id = actions[0]["security_id"]
        resolutions = pd.DataFrame(
            [
                {
                    "candidate_id": "c725a22e1244939304cc2bb5f81afceb011109ff48afe22bcad10aec629d818c",
                    "security_id": security_id,
                    "symbol": "NTCOY",
                    "resolution": "applied",
                    "event_id": terminal_id,
                    "successor_security_id": "",
                }
            ]
        )
        report = {
            "records": {
                security_id: {
                    "candidate": {
                        "security_id": security_id,
                        "symbol": "NTCOY",
                        "active_to": "2024-09-04",
                        "last_price_date": "2024-08-07",
                    },
                    "verified_event": {
                        "action_type": "delisting",
                        "effective_date": "2024-09-04",
                        "new_symbol": "",
                        "ratio": None,
                        "cash_amount": 5.043659,
                        "source_url": cash_url,
                        "source_hash": (
                            "3be830009ee6942d4ca604aafdc19730a693f13e62b046641a638e9a1ee112c1"
                        ),
                    },
                    "successor_security_id": "",
                    "manual_review": False,
                    "manual_review_reason": "",
                }
            }
        }
        checks = script.build_event_checks(
            action_frame,
            resolutions,
            report,
            archive,
            _policy(),
        )
        self.assertEqual({item["status"] for item in checks}, {"passed"})
        self.assertTrue(all(item["reasons"] == [] for item in checks))
        for item in checks:
            self.assertTrue(trusted_ntco_report_diagnostic_passed(item))
            self.assertEqual(
                item["trusted_ntco_evidence_binding"]["status"], "trusted"
            )
        counts = _check_report_rows(
            {"events": checks, "permanent_exceptions": [], "prices": []}
        )
        self.assertEqual(counts["event_mismatch_count"], 0)
        self.assertEqual(counts["nonterminal_event_count"], 1)
        self.assertEqual(counts["reviewed_nonterminal_event_count"], 1)

        ticker_id = (
            "0bd231d216016d33c8d6eb1ec9bb85450a21b9298c4abda34a3399727e9b2c00"
        )
        exact = trusted_ntco_evidence_binding_diagnostic(
            next(item for item in actions if item["event_id"] == ticker_id),
            archive,
        )
        self.assertIsNotNone(exact)
        assert exact is not None
        self.assertEqual(exact["status"], "trusted")

        tampered_archive = archive.copy(deep=True)
        target = tampered_archive["source"].eq("official_occ")
        tampered_archive.loc[target, "object_path"] = "archives/changed.bin.gz"
        tampered_checks = script.build_event_checks(
            action_frame,
            resolutions,
            report,
            tampered_archive,
            _policy(),
        )
        tampered_by_id = {item["event_id"]: item for item in tampered_checks}
        self.assertEqual(tampered_by_id[ticker_id]["status"], "mismatch")
        self.assertIn(
            "occ_memo_54105_raw_pdf",
            " ".join(tampered_by_id[ticker_id]["reasons"]),
        )

    def test_terminal_price_tail_corrections_attest_exact_candidate_rows(self):
        policy = _policy()
        corrections = script.reviewed_terminal_price_tail_corrections(
            policy.events
        )
        validation_policy = _policy_without_terminal_event_gates(*corrections)
        actions: list[dict] = []
        resolutions: list[dict] = []
        records: dict[str, dict] = {}
        archive_rows: list[dict] = []

        for correction in corrections.values():
            actions.append(
                {
                    "event_id": correction["event_id"],
                    "security_id": correction["security_id"],
                    "action_type": correction["action_type"],
                    "effective_date": correction["market_transition_session"],
                    "ex_date": correction["market_transition_session"],
                    "announcement_date": correction[
                        "official_completion_date"
                    ],
                    "record_date": "",
                    "payment_date": "",
                    "cash_amount": correction["cash_amount"],
                    "ratio": correction["ratio"],
                    "new_symbol": correction["new_symbol"],
                    "new_security_id": correction["new_security_id"],
                    "currency": "USD",
                    "official": True,
                    "source_kind": "official_crosscheck",
                    "source_url": correction["official_source_url"],
                    "source_hash": correction["official_source_hash"],
                }
            )
            resolutions.append(
                {
                    "candidate_id": correction["candidate_id"],
                    "security_id": correction["security_id"],
                    "symbol": correction["symbol"],
                    "last_price_date": correction["last_real_session"],
                    "resolution": "applied",
                    "event_id": correction["event_id"],
                    "successor_security_id": correction["new_security_id"],
                    "successor_symbol": correction["new_symbol"],
                    "source_url": correction["official_source_url"],
                    "source_hash": correction["official_source_hash"],
                }
            )
            records[correction["security_id"]] = {
                "candidate": {
                    "candidate_id": correction["old_candidate_id"],
                    "security_id": correction["security_id"],
                    "symbol": correction["symbol"],
                    "last_price_date": correction[
                        "report_candidate_last_price_date"
                    ],
                    "active_to": correction["report_candidate_active_to"],
                    "index_remove_dates": [
                        item["effective_date"]
                        for item in correction["index_removals_observed"]
                    ],
                },
                "eligible_for_apply": False,
                "manual_review": False,
                "manual_review_reason": "",
                "crosscheck": {
                    "passed": False,
                    "date_passed": False,
                    "economic_terms_passed": True,
                    "old_price_session": correction[
                        "report_crosscheck_old_price_session"
                    ],
                },
                "parsed": {
                    "action_type": correction["action_type"],
                    "effective_date": correction["report_effective_date"],
                    "cash_amount": correction["cash_amount"],
                    "ratio": correction["ratio"],
                    "new_symbol": correction["new_symbol"],
                    "source_url": correction["official_source_url"],
                    "source_hash": correction["official_source_hash"],
                },
                "source_url": correction["official_source_url"],
                "source_hash": correction["official_source_hash"],
                "successor_security_id": correction["new_security_id"],
                "filing": {
                    "accession_number": correction[
                        "filing_accession_number"
                    ],
                    "filing_date": correction["official_completion_date"],
                },
            }
            archive_rows.extend(
                [
                    _archive_row(
                        correction["official_source_hash"],
                        correction["official_source_url"],
                    ),
                    _archive_row(
                        correction["raw_source_hash"],
                        correction["raw_source_url"],
                        source="eodhd_eod",
                    ),
                ]
            )
            if correction["successor_source_hash"]:
                archive_rows.append(
                    _archive_row(
                        correction["successor_source_hash"],
                        "",
                        source="eodhd_eod",
                    )
                )

        action_frame = pd.DataFrame(actions)
        resolution_frame = pd.DataFrame(resolutions)
        archive = pd.DataFrame(archive_rows).drop_duplicates(
            subset=["archive_id"], keep="first"
        )

        def build(
            changed_actions=action_frame,
            changed_resolutions=resolution_frame,
            changed_records=records,
            changed_archive=archive,
        ):
            checks_by_symbol: dict[str, dict] = {}
            report_hashes = {
                correction["lifecycle_evidence_report_sha256"]
                for correction in corrections.values()
            }
            for report_hash in report_hashes:
                group = {
                    event_id: correction
                    for event_id, correction in corrections.items()
                    if correction["lifecycle_evidence_report_sha256"]
                    == report_hash
                }
                event_ids = set(group)
                security_ids = {
                    correction["security_id"] for correction in group.values()
                }
                checks = script.build_event_checks(
                    changed_actions.loc[
                        changed_actions["event_id"].isin(event_ids)
                    ],
                    changed_resolutions.loc[
                        changed_resolutions["event_id"].isin(event_ids)
                    ],
                    {
                        "records": {
                            security_id: changed_records[security_id]
                            for security_id in security_ids
                        }
                    },
                    changed_archive,
                    validation_policy,
                    lifecycle_report_sha256=report_hash,
                )
                checks_by_symbol.update(
                    {item["symbol"]: item for item in checks}
                )
            return checks_by_symbol

        passed = build()
        self.assertEqual(set(passed), {"NBL", "XLNX", "CXO", "NLSN", "NLOK"})
        self.assertTrue(all(item["status"] == "passed" for item in passed.values()))
        self.assertTrue(
            all(
                item["reviewed_terminal_price_tail_correction_applied"]
                and item["reviewed_terminal_price_tail_correction_match"]
                and len(item["reviewed_terminal_price_tail_correction_sha256"])
                == 64
                for item in passed.values()
            )
        )

        by_symbol = {
            correction["symbol"]: correction
            for correction in corrections.values()
        }
        changed_records = copy.deepcopy(records)
        nbl = by_symbol["NBL"]
        changed_records[nbl["security_id"]]["parsed"]["effective_date"] = (
            "2020-10-06"
        )
        self.assertFalse(
            build(changed_records=changed_records)["NBL"][
                "reviewed_terminal_price_tail_correction_applied"
            ]
        )

        changed_archive = archive.copy(deep=True)
        xlnx = by_symbol["XLNX"]
        changed_archive.loc[
            changed_archive["archive_id"].eq(xlnx["raw_source_hash"]),
            "source_url",
        ] = xlnx["raw_source_url"] + "&changed=1"
        self.assertFalse(
            build(changed_archive=changed_archive)["XLNX"][
                "reviewed_terminal_price_tail_correction_applied"
            ]
        )

        changed_actions = action_frame.copy(deep=True)
        cxo = by_symbol["CXO"]
        changed_actions.loc[
            changed_actions["event_id"].eq(cxo["event_id"]),
            "effective_date",
        ] = cxo["official_completion_date"]
        self.assertFalse(
            build(changed_actions=changed_actions)["CXO"][
                "reviewed_terminal_price_tail_correction_applied"
            ]
        )

        changed_actions = action_frame.copy(deep=True)
        nlsn = by_symbol["NLSN"]
        changed_actions.loc[
            changed_actions["event_id"].eq(nlsn["event_id"]),
            "cash_amount",
        ] = 27.99
        self.assertFalse(
            build(changed_actions=changed_actions)["NLSN"][
                "reviewed_terminal_price_tail_correction_applied"
            ]
        )

        changed_actions = action_frame.copy(deep=True)
        nlok = by_symbol["NLOK"]
        changed_actions.loc[
            changed_actions["event_id"].eq(nlok["event_id"]),
            "new_symbol",
        ] = "FORGED"
        self.assertFalse(
            build(changed_actions=changed_actions)["NLOK"][
                "reviewed_terminal_price_tail_correction_applied"
            ]
        )

        changed_records = copy.deepcopy(records)
        changed_records[nlok["security_id"]]["candidate"][
            "index_remove_dates"
        ] = ["2022-11-08"]
        self.assertFalse(
            build(changed_records=changed_records)["NLOK"][
                "reviewed_terminal_price_tail_correction_applied"
            ]
        )

    def test_terminal_price_tail_inventory_equals_repair_registry_hash(self):
        policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
        corrections = script.reviewed_terminal_price_tail_corrections(
            policy["events"]
        )
        self.assertEqual(
            set(corrections),
            set(
                script.TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTION_EVENT_IDS
            ),
        )
        self.assertEqual(
            script.reviewed_terminal_price_tail_correction_inventory_sha256(
                policy["events"]
            ),
            script.TRUSTED_REVIEWED_TERMINAL_PRICE_TAIL_CORRECTIONS_SHA256,
        )

        policy["events"]["reviewed_terminal_price_tail_corrections"][0][
            "removed_tail_count"
        ] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.yaml"
            path.write_text(yaml.safe_dump(policy), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "registry item hash"):
                script.load_policy(path)

    def test_reviewed_market_transitions_keep_exact_event_validation_passed(self):
        policy = _policy()
        corrections = script.reviewed_terminal_market_date_corrections(
            policy.events
        )
        target_ids = {
            "39caca00fbedb7b23e8b5294cdf3bdc5aba9014f98bf72b6a6da341602651192",
            "951960822646a8f972002508d162bd37a3ab3aaec42349fbff37caa1919b7b51",
            "ea791beeaa569aaf19b1df04e59df3710d45382cd70e5d82078d7247286272c6",
        }
        validation_policy = _policy_without_terminal_event_gates(*target_ids)
        self.assertEqual(target_ids, target_ids & set(corrections))

        for event_id in sorted(target_ids):
            with self.subTest(event_id=event_id):
                correction = corrections[event_id]
                action = {
                    key: correction[key]
                    for key in (
                        "event_id",
                        "security_id",
                        "action_type",
                        "effective_date",
                        "ex_date",
                        "announcement_date",
                        "payment_date",
                        "cash_amount",
                        "ratio",
                        "new_symbol",
                        "new_security_id",
                        "currency",
                        "source_kind",
                        "source_url",
                        "source_hash",
                    )
                }
                action["official"] = True
                resolution = {
                    "candidate_id": correction["candidate_id"],
                    "security_id": correction["security_id"],
                    "symbol": correction["symbol"],
                    "last_price_date": correction["last_price_date"],
                    "resolution": "applied",
                    "event_id": event_id,
                    "successor_security_id": correction["new_security_id"],
                }
                record = {
                    "candidate": {
                        "candidate_id": correction["candidate_id"],
                        "security_id": correction["security_id"],
                        "symbol": correction["symbol"],
                        "last_price_date": correction["last_price_date"],
                        "active_to": correction["last_price_date"],
                    },
                    "eligible_for_apply": True,
                    "manual_review": False,
                    "manual_review_reason": "",
                    "crosscheck": {
                        "passed": True,
                        "date_passed": True,
                        "economic_terms_passed": True,
                    },
                    "parsed": {
                        "action_type": correction["action_type"],
                        "effective_date": correction["report_effective_date"],
                        "cash_amount": correction["cash_amount"],
                        "ratio": correction["ratio"],
                        "new_symbol": correction["new_symbol"],
                        "source_url": correction["report_source_url"],
                        "source_hash": correction["report_source_hash"],
                    },
                    "source_url": correction["report_source_url"],
                    "source_hash": correction["report_source_hash"],
                    "successor_security_id": correction["new_security_id"],
                    "filing": {
                        "accession_number": correction[
                            "filing_accession_number"
                        ],
                        "filing_date": correction["filing_date"],
                    },
                }
                archive = pd.DataFrame(
                    [
                        _archive_row(
                            correction["source_hash"], correction["source_url"]
                        )
                    ]
                )

                result = script.build_event_checks(
                    pd.DataFrame([action]),
                    pd.DataFrame([resolution]),
                    {"records": {correction["security_id"]: record}},
                    archive,
                    validation_policy,
                    lifecycle_report_sha256=correction[
                        "lifecycle_evidence_report_sha256"
                    ],
                )[0]

                self.assertEqual(result["status"], "passed")
                self.assertTrue(result["date_match"])
                self.assertTrue(
                    result[
                        "reviewed_terminal_market_date_correction_applied"
                    ]
                )
                self.assertEqual(
                    result["official_completion_date"],
                    correction["official_completion_date"],
                )
                self.assertEqual(
                    result["terminal_market_date_relation"],
                    correction["date_relation"],
                )

                changed_record = copy.deepcopy(record)
                changed_record["parsed"]["source_hash"] = "f" * 64
                mismatch = script.build_event_checks(
                    pd.DataFrame([action]),
                    pd.DataFrame([resolution]),
                    {"records": {correction["security_id"]: changed_record}},
                    archive,
                    validation_policy,
                    lifecycle_report_sha256=correction[
                        "lifecycle_evidence_report_sha256"
                    ],
                )[0]
                self.assertEqual(mismatch["status"], "mismatch")
                self.assertFalse(
                    mismatch[
                        "reviewed_terminal_market_date_correction_applied"
                    ]
                )

    def test_yahoo_no_data_terminal_action_allowlist_is_exact(self):
        policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
        expected = {
            "delisting",
            "cash_merger",
            "stock_merger",
            "ticker_change",
        }
        self.assertEqual(
            set(policy["prices"]["no_data_terminal_action_types"]), expected
        )
        self.assertEqual(
            set(script.YAHOO_NO_DATA_TERMINAL_ACTION_TYPES), expected
        )

        mutations = (
            ["delisting", "cash_merger", "stock_merger"],
            [*sorted(expected), "spinoff"],
            ["delisting", "delisting", "stock_merger", "ticker_change"],
        )
        for changed in mutations:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                mutated = copy.deepcopy(policy)
                mutated["prices"]["no_data_terminal_action_types"] = changed
                path = Path(directory) / "policy.yaml"
                path.write_text(yaml.safe_dump(mutated), encoding="utf-8")
                with self.assertRaisesRegex(
                    RuntimeError, "terminal action allowlist must be exact"
                ):
                    script.load_policy(path)

    def test_yahoo_no_data_date_and_successor_policy_is_exact(self):
        policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            set(policy["prices"]["no_data_terminal_date_relations"]),
            set(script.YAHOO_NO_DATA_TERMINAL_DATE_RELATIONS),
        )
        self.assertEqual(
            policy["prices"]["no_data_successor_validation_basis"],
            script.YAHOO_NO_DATA_SUCCESSOR_VALIDATION_BASIS,
        )
        mutations = (
            {"no_data_terminal_date_relations": ["event_on_terminal_session"]},
            {"no_data_successor_validation_basis": "any_successor_target"},
        )
        for changed in mutations:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                mutated = copy.deepcopy(policy)
                mutated["prices"].update(changed)
                path = Path(directory) / "policy.yaml"
                path.write_text(yaml.safe_dump(mutated), encoding="utf-8")
                with self.assertRaisesRegex(
                    RuntimeError, "date/successor policy must be exact"
                ):
                    script.load_policy(path)

    def test_terminal_original_filing_kind_requires_explicit_policy_allowlist(self):
        actions, resolutions, report, evidence = self._fixtures()
        actions.loc[0, "source_kind"] = "official_filing"
        archive = pd.DataFrame(
            [_archive_row(evidence, "https://www.sec.gov/Archives/event.txt")]
        )

        default_policy = _policy()
        default_policy.value["events"].pop(
            "terminal_official_source_kinds", None
        )
        default = script.build_event_checks(
            actions, resolutions, report, archive, default_policy
        )
        self.assertEqual(default[0]["status"], "mismatch")

        policy = _policy()
        policy.value["events"]["terminal_official_source_kinds"] = [
            "official_crosscheck",
            "official_filing",
        ]
        allowed = script.build_event_checks(
            actions, resolutions, report, archive, policy
        )
        self.assertEqual(allowed[0]["status"], "passed")
        self.assertTrue(allowed[0]["official_original"])

    def test_nonterminal_action_uses_official_provenance_without_terminal_report(self):
        actions, resolutions, report, evidence = self._fixtures()
        unchecked = actions.iloc[0].copy()
        unchecked["event_id"] = "event-unchecked"
        unchecked["action_type"] = "ticker_change"
        unchecked["new_security_id"] = "OLD"
        unchecked["new_symbol"] = "RENAMED"
        unchecked["cash_amount"] = None
        unchecked["ratio"] = None
        actions = pd.concat([actions, pd.DataFrame([unchecked])], ignore_index=True)
        archive = pd.DataFrame(
            [_archive_row(evidence, "https://www.sec.gov/Archives/event.txt")]
        )

        policy = _policy()
        policy.value["events"]["reviewed_nonterminal_extractions"] = [
            {
                "event_id": "event-unchecked",
                "security_id": "OLD",
                "action_type": "ticker_change",
                "effective_date": "2024-01-05",
                "new_security_id": "OLD",
                "new_symbol": "RENAMED",
                "ratio": None,
                "cash_amount": None,
                "currency": "USD",
                "source_kind": "official_crosscheck",
                "source_url": "https://www.sec.gov/Archives/event.txt",
                "source_hash": evidence,
            }
        ]
        checks = script.build_event_checks(
            actions, resolutions, report, archive, policy
        )

        self.assertEqual({item["event_id"] for item in checks}, {
            "event-1",
            "event-unchecked",
        })
        by_id = {item["event_id"]: item for item in checks}
        self.assertEqual(by_id["event-1"]["status"], "passed")
        self.assertEqual(by_id["event-1"]["validation_kind"], script.TERMINAL_EVENT_VALIDATION)
        self.assertEqual(by_id["event-unchecked"]["status"], "passed")
        self.assertEqual(
            by_id["event-unchecked"]["validation_kind"],
            script.NONTERMINAL_EVENT_VALIDATION,
        )
        self.assertFalse(
            by_id["event-unchecked"]["lifecycle_report_extraction_approved"]
        )
        self.assertTrue(by_id["event-unchecked"]["official_provenance_passed"])
        self.assertTrue(by_id["event-unchecked"]["reviewed_extraction_match"])
        self.assertEqual(
            len(by_id["event-unchecked"]["reviewed_extraction_sha256"]), 64
        )

        actions.loc[
            actions["event_id"].eq("event-unchecked"), "new_security_id"
        ] = "DIFFERENT"
        semantic_mismatch = script.build_event_checks(
            actions, resolutions, report, archive, policy
        )
        mismatch_by_id = {
            item["event_id"]: item for item in semantic_mismatch
        }
        self.assertEqual(mismatch_by_id["event-unchecked"]["status"], "mismatch")
        self.assertFalse(
            mismatch_by_id["event-unchecked"]["reviewed_extraction_match"]
        )
        self.assertTrue(
            any(
                "new_security_id" in reason
                for reason in mismatch_by_id["event-unchecked"]["reasons"]
            )
        )

        wrong_url = archive.copy()
        wrong_url.loc[0, "source_url"] = "https://www.sec.gov/Archives/other.txt"
        mismatched = script.build_event_checks(
            actions.iloc[:1], resolutions, report, wrong_url, _policy()
        )
        self.assertEqual(mismatched[0]["status"], "mismatch")
        self.assertIn("exact archived URL/hash pair", mismatched[0]["reasons"])

    def test_reviewed_nonterminal_extraction_compares_every_field_exactly(self):
        extraction = {
            "event_id": "event-reviewed",
            "security_id": "OLD",
            "action_type": "stock_merger",
            "effective_date": "2024-01-05",
            "new_security_id": "NEW-ID",
            "new_symbol": "NEW",
            "ratio": 0.4,
            "cash_amount": 2.5,
            "currency": "USD",
            "source_kind": "official_filing",
            "source_url": "https://www.sec.gov/Archives/event.txt",
            "source_hash": "a" * 64,
        }
        self.assertEqual(
            script.reviewed_nonterminal_extraction_mismatches(
                dict(extraction), extraction
            ),
            (),
        )
        mutations = {
            "event_id": "different-event",
            "security_id": "DIFFERENT-ID",
            "action_type": "ticker_change",
            "effective_date": "2024-01-06",
            "new_security_id": "DIFFERENT-SUCCESSOR",
            "new_symbol": "OTHER",
            "ratio": 0.4000000000001,
            "cash_amount": 2.5000000000001,
            "currency": "EUR",
            "source_kind": "official_crosscheck",
            "source_url": "https://www.sec.gov/Archives/other.txt",
            "source_hash": "b" * 64,
        }
        for field, changed in mutations.items():
            with self.subTest(field=field):
                action = {**extraction, field: changed}
                mismatches = script.reviewed_nonterminal_extraction_mismatches(
                    action, extraction
                )
                self.assertIn(field, mismatches)


class PriceTargetConstructionTest(unittest.TestCase):
    def test_applied_resolution_and_lifecycle_action_build_real_targets(self):
        master = pd.DataFrame(
            [
                {
                    "security_id": "OLD-ID",
                    "primary_symbol": "OLD",
                    "active_from": "2015-01-01",
                    "active_to": "2024-01-04",
                },
                {
                    "security_id": "NEW-ID",
                    "primary_symbol": "NEW",
                    "active_from": "2024-01-05",
                    "active_to": "",
                },
            ]
        )
        history = pd.DataFrame(
            [
                {
                    "security_id": "OLD-ID",
                    "symbol": "OLD",
                    "effective_from": "2015-01-01",
                    "effective_to": "2024-01-04",
                },
                {
                    "security_id": "NEW-ID",
                    "symbol": "NEW",
                    "effective_from": "2024-01-05",
                    "effective_to": "",
                },
            ]
        )
        actions = pd.DataFrame(
            [
                {
                    "event_id": "event-1",
                    "security_id": "OLD-ID",
                    "action_type": "stock_merger",
                    "effective_date": "2024-01-05",
                    "new_security_id": "NEW-ID",
                    "new_symbol": "NEW",
                }
            ]
        )
        resolutions = pd.DataFrame(
            [
                {
                    "security_id": "OLD-ID",
                    "symbol": "OLD",
                    "resolution": "applied",
                    "event_id": "event-1",
                    "successor_security_id": "NEW-ID",
                    "successor_symbol": "NEW",
                }
            ]
        )

        targets = script.build_price_targets(master, history, actions, resolutions)

        self.assertIsInstance(targets, list)
        self.assertEqual([item.security_id for item in targets], ["NEW-ID", "OLD-ID"])
        by_id = {item.security_id: item for item in targets}
        self.assertIn("applied_resolution_source", by_id["OLD-ID"].origins)
        self.assertIn("lifecycle_action_source", by_id["OLD-ID"].origins)
        self.assertEqual(by_id["OLD-ID"].terminal_event_id, "event-1")
        self.assertIn("applied_resolution_successor", by_id["NEW-ID"].origins)
        self.assertIn("lifecycle_action_successor", by_id["NEW-ID"].origins)
        self.assertEqual(by_id["NEW-ID"].active_from, "2024-01-05")

    def test_provider_supplement_and_reused_ticker_peer_are_always_targets(self):
        master = pd.DataFrame(
            [
                {
                    "security_id": "OLD-LILA",
                    "primary_symbol": "LILA",
                    "active_from": "2015-07-02",
                    "active_to": "2017-12-29",
                },
                {
                    "security_id": "CURRENT-LILA",
                    "primary_symbol": "LILA",
                    "active_from": "2018-01-02",
                    "active_to": "",
                },
            ]
        )
        history = pd.DataFrame(
            columns=("security_id", "symbol", "effective_from", "effective_to")
        )
        actions = pd.DataFrame(
            columns=(
                "security_id",
                "action_type",
                "effective_date",
                "new_security_id",
                "new_symbol",
            )
        )
        resolutions = pd.DataFrame(
            columns=(
                "security_id",
                "symbol",
                "resolution",
                "event_id",
                "successor_security_id",
                "successor_symbol",
            )
        )
        prices = pd.DataFrame(
            [
                {
                    "security_id": "OLD-LILA",
                    "session": "2017-12-29",
                    "source": "identity_repair_supplement",
                    "source_url": _source_url("LILA"),
                },
                {
                    "security_id": "CURRENT-LILA",
                    "session": "2024-02-29",
                    "source": "eodhd",
                    "source_url": "https://eodhd.com/api/eod/LILA.US",
                },
            ]
        )

        targets = script.build_price_targets(
            master, history, actions, resolutions, prices
        )

        by_id = {item.security_id: item for item in targets}
        self.assertEqual(set(by_id), {"OLD-LILA", "CURRENT-LILA"})
        self.assertIn(
            "independent_provider_internal_source", by_id["OLD-LILA"].origins
        )
        self.assertIn(
            "independent_provider_reused_symbol_peer",
            by_id["CURRENT-LILA"].origins,
        )

    def test_same_security_id_builds_one_price_target_per_symbol_interval(self):
        master = pd.DataFrame(
            [
                {
                    "security_id": "SAME-ID",
                    "primary_symbol": "NEW",
                    "active_from": "2015-01-02",
                    "active_to": "",
                }
            ]
        )
        history = pd.DataFrame(
            [
                {
                    "security_id": "SAME-ID",
                    "symbol": "OLD",
                    "effective_from": "2015-01-02",
                    "effective_to": "2019-12-31",
                },
                {
                    "security_id": "SAME-ID",
                    "symbol": "NEW",
                    "effective_from": "2020-01-02",
                    "effective_to": "",
                },
            ]
        )
        actions = pd.DataFrame(
            [
                {
                    "event_id": "ticker-change",
                    "security_id": "SAME-ID",
                    "action_type": "ticker_change",
                    "effective_date": "2020-01-02",
                    "new_security_id": "SAME-ID",
                    "new_symbol": "NEW",
                }
            ]
        )
        resolutions = pd.DataFrame(
            columns=(
                "security_id",
                "symbol",
                "resolution",
                "event_id",
                "successor_security_id",
                "successor_symbol",
            )
        )

        targets = script.build_price_targets(
            master, history, actions, resolutions
        )

        self.assertEqual(
            [(item.symbol, item.active_from, item.active_to) for item in targets],
            [
                ("OLD", "2015-01-02", "2019-12-31"),
                ("NEW", "2020-01-02", ""),
            ],
        )
        by_symbol = {item.symbol: item for item in targets}
        self.assertEqual(by_symbol["OLD"].terminal_event_id, "ticker-change")
        self.assertEqual(by_symbol["OLD"].successor_security_id, "SAME-ID")
        self.assertEqual(by_symbol["NEW"].terminal_event_id, "")

    def test_intermediate_segment_rejects_nonadjacent_or_ambiguous_ticker_action(self):
        master = pd.DataFrame(
            [{"security_id": "SAME-ID", "primary_symbol": "NEW"}]
        )
        history = pd.DataFrame(
            [
                {
                    "security_id": "SAME-ID",
                    "symbol": "OLD",
                    "effective_from": "2023-01-03",
                    "effective_to": "2024-01-05",
                },
                {
                    "security_id": "SAME-ID",
                    "symbol": "NEW",
                    "effective_from": "2024-01-10",
                    "effective_to": "",
                },
            ]
        )
        base = {
            "security_id": "SAME-ID",
            "action_type": "ticker_change",
            "effective_date": "2024-01-10",
            "new_security_id": "SAME-ID",
            "new_symbol": "NEW",
        }
        resolutions = pd.DataFrame(
            columns=(
                "security_id",
                "symbol",
                "resolution",
                "event_id",
                "successor_security_id",
                "successor_symbol",
            )
        )
        nonadjacent = script.build_price_targets(
            master,
            history,
            pd.DataFrame([{**base, "event_id": "late-event"}]),
            resolutions,
        )
        self.assertEqual(nonadjacent[0].terminal_event_id, "")

        adjacent_history = history.copy()
        adjacent_history.loc[0, "effective_to"] = "2024-01-09"
        ambiguous = script.build_price_targets(
            master,
            adjacent_history,
            pd.DataFrame(
                [
                    {**base, "event_id": "event-a"},
                    {**base, "event_id": "event-b"},
                ]
            ),
            resolutions,
        )
        self.assertEqual(ambiguous[0].terminal_event_id, "")

class IdentityBoundaryEvidenceTest(unittest.TestCase):
    def test_unpinned_yahoo_primary_bundle_is_rejected(self):
        columns = ("security_id", "session", "close", "source")
        primary = identity_script.FetchedHistories(
            prices=pd.DataFrame(
                [["OLD-LILA", "2015-07-02", 10.0, "eodhd_eod"]],
                columns=columns,
            ),
            crosscheck_prices=pd.DataFrame(),
            corporate_actions=pd.DataFrame(),
            artifacts=(),
            role_codes={"old_lila": "LILAV"},
            http_attempts=32,
        )
        supplement = identity_script.FetchedHistories(
            prices=pd.DataFrame(
                [
                    ["OLD-LILA", "2015-07-02", 99.0, "yahoo_chart_json"],
                    ["OLD-LILA", "2015-07-03", 11.0, "yahoo_chart_json"],
                ],
                columns=columns,
            ),
            crosscheck_prices=pd.DataFrame(),
            corporate_actions=pd.DataFrame(),
            artifacts=(),
            role_codes={
                "old_lila_regular_way_independent_crosscheck": "YAHOO_CHART:LILA"
            },
            http_attempts=0,
        )

        with self.assertRaisesRegex(ValueError, "role binding changed"):
            identity_script.merge_fetched_histories(
                primary, supplement, ids=SimpleNamespace()
            )

    def test_identity_repair_requests_lila_when_issued_history_from_legal_start(self):
        class RecordingClient:
            def __init__(self):
                self.calls = []
                self.attempt_count = 0

            def get_json(self, path, *, params):
                self.attempt_count += 1
                self.calls.append((path, dict(params)))
                return []

            def safe_url(self, path, *, params):
                return f"https://eodhd.test/{path}?from={params['from']}&to={params['to']}"

        client = RecordingClient()
        source = identity_script.CappedIdentityHistorySource(client)
        role_ids = {
            "wyn": "WYN-ID",
            "spectra": "SE-ID",
            "old_fox": "FOX-ID",
            "old_foxa": "FOXA-ID",
            "old_lila": "LILA-ID",
            "old_lilak": "LILAK-ID",
            "old_aa": "AA-ID",
            "tnl": "TNL-ID",
            "valaris": "VAL-ID",
            "bhi": "BHI-ID",
        }

        source.fetch(role_ids, completed_session="2026-07-17")

        requests = dict(client.calls)
        self.assertEqual(requests["eod/LILAV.US"]["from"], "2015-06-22")
        self.assertEqual(requests["eod/LILKV.US"]["from"], "2015-06-22")
        self.assertEqual(
            requests["eod/VALPQ.US"],
            {"from": "2019-07-31", "to": "2021-04-27"},
        )
        self.assertNotIn("eod/VAL_old.US", requests)

    def test_identity_repair_archive_id_is_the_exact_payload_hash(self):
        artifact = identity_script.SourceArtifact(
            source="official_identity_evidence",
            source_url="https://www.sec.gov/Archives/example.htm",
            retrieved_at="2026-07-18T00:00:00Z",
            content=b"official raw payload",
            content_type="text/html",
        )

        archive = identity_script.append_source_archive(
            pd.DataFrame(),
            [artifact],
            completed_session="2026-07-17",
        )

        self.assertEqual(str(archive.iloc[0]["archive_id"]), artifact.source_hash)
        self.assertEqual(str(archive.iloc[0]["source_hash"]), artifact.source_hash)

    def test_old_and_current_lila_boundaries_bind_to_exact_archived_urls_and_hashes(self):
        nasdaq_payload = b"nasdaq ETA2015-79 payload"
        sec_payload = b"SEC split-off exhibit payload"
        nasdaq_hash = script.sha256_bytes(nasdaq_payload)
        sec_hash = script.sha256_bytes(sec_payload)
        nasdaq_url = (
            "https://www.nasdaqtrader.com/TraderNews.aspx?id=ETA2015-79"
        )
        sec_url = (
            "https://www.sec.gov/Archives/edgar/data/1570585/"
            "000157058517000401/ex991split-offrecordanddis.htm"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for digest, payload, url in (
                (nasdaq_hash, nasdaq_payload, nasdaq_url),
                (sec_hash, sec_payload, sec_url),
            ):
                object_path = f"archives/{digest}.bin.gz"
                path = root / object_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(gzip.compress(payload, mtime=0))
                rows.append(
                    {
                        "archive_id": digest,
                        "object_path": object_path,
                        "source_url": url,
                    }
                )
            targets = [
                script.PriceTarget(
                    "OLD-LILA", "LILA", ("test",), "2015-07-02", "2017-12-29"
                ),
                script.PriceTarget(
                    "CURRENT-LILA", "LILA", ("test",), "2018-01-02", ""
                ),
                script.PriceTarget(
                    "OLD-LILAK", "LILAK", ("test",), "2015-07-02", "2017-12-29"
                ),
                script.PriceTarget(
                    "CURRENT-LILAK", "LILAK", ("test",), "2018-01-02", ""
                ),
            ]

            evidence = script.resolve_identity_boundary_evidence(
                SimpleNamespace(root=root),
                targets,
                pd.DataFrame(rows),
                _policy(),
            )

        by_identity = {
            (target.security_id, target.active_from, target.active_to): evidence[
                target.target_id
            ]
            for target in targets
        }
        old_lila = {
            item["boundary"]: item
            for item in by_identity[("OLD-LILA", "2015-07-02", "2017-12-29")]
        }
        current_lila = by_identity[("CURRENT-LILA", "2018-01-02", "")]
        old_lilak = {
            item["boundary"]: item
            for item in by_identity[("OLD-LILAK", "2015-07-02", "2017-12-29")]
        }
        current_lilak = by_identity[("CURRENT-LILAK", "2018-01-02", "")]
        self.assertEqual(old_lila["active_from"]["evidence_sha256"], nasdaq_hash)
        self.assertEqual(old_lila["active_to"]["evidence_sha256"], sec_hash)
        self.assertEqual(old_lilak["active_from"]["evidence_sha256"], nasdaq_hash)
        self.assertEqual(old_lilak["active_to"]["evidence_sha256"], sec_hash)
        self.assertEqual(current_lila[0]["date"], "2018-01-02")
        self.assertEqual(current_lila[0]["source_url"], sec_url)
        self.assertEqual(current_lilak[0]["date"], "2018-01-02")
        self.assertEqual(current_lilak[0]["evidence_sha256"], sec_hash)


class YahooChartPriceComparisonTest(unittest.TestCase):
    def test_valid_no_data_is_distinct_from_html_challenge(self):
        parsed = parse_yahoo_chart_json(
            _chart_json(pd.DataFrame(), symbol="OLD"), "OLD"
        )
        self.assertTrue(parsed.bars.empty)
        with self.assertRaisesRegex(ValueError, "HTML or a verification challenge"):
            parse_yahoo_chart_json(b"<html>challenge</html>", "OLD")

    def test_exact_empty_retired_yhd_placeholder_is_no_data_not_prices(self):
        payload = _retired_yhd_placeholder_json("OLD")
        evidence = parse_yahoo_chart_no_data_evidence(
            payload,
            "OLD",
            http_status=200,
        )
        self.assertEqual(
            evidence.kind, "http_200_empty_retired_yhd_placeholder"
        )
        with self.assertRaisesRegex(ValueError, "currency must be USD"):
            parse_yahoo_chart_json(payload, "OLD")

        mutations = {
            "wrong_symbol": ("OTHER", "OLD"),
            "wrong_exchange": ("OLD", "OLD"),
            "nonempty_quote": ("OLD", "OLD"),
        }
        for label, (payload_symbol, expected_symbol) in mutations.items():
            with self.subTest(label=label):
                changed = json.loads(
                    _retired_yhd_placeholder_json(payload_symbol)
                )
                item = changed["chart"]["result"][0]
                if label == "wrong_exchange":
                    item["meta"]["exchangeName"] = "NMS"
                if label == "nonempty_quote":
                    item["timestamp"] = [1_700_000_000]
                    item["indicators"]["quote"] = [
                        {
                            "open": [1.0],
                            "high": [1.0],
                            "low": [1.0],
                            "close": [1.0],
                            "volume": [1.0],
                        }
                    ]
                with self.assertRaises(ValueError):
                    parse_yahoo_chart_no_data_evidence(
                        json.dumps(changed).encode(),
                        expected_symbol,
                        http_status=200,
                    )

    def test_http_400_no_data_requires_exact_echoed_request_epochs(self):
        period1, period2 = _request_periods("2024-01-02", "2024-05-31")
        description = (
            f"Data doesn't exist for startDate = {period1}, endDate = {period2}"
        )
        payload = json.dumps(
            {
                "chart": {
                    "result": None,
                    "error": {
                        "code": "Bad Request",
                        "description": description,
                    },
                }
            },
            separators=(",", ":"),
        ).encode()
        evidence = parse_yahoo_chart_no_data_evidence(
            payload,
            "OLD",
            http_status=400,
            request_period1=period1,
            request_period2=period2,
        )
        self.assertEqual(evidence.kind, "http_400_bounded_history_not_found")

        for label, overrides in {
            "wrong_start": {"request_period1": period1 + 1},
            "wrong_end": {"request_period2": period2 + 1},
            "missing_bounds": {
                "request_period1": None,
                "request_period2": None,
            },
        }.items():
            with self.subTest(label=label):
                kwargs = {
                    "request_period1": period1,
                    "request_period2": period2,
                    **overrides,
                }
                with self.assertRaises(ValueError):
                    parse_yahoo_chart_no_data_evidence(
                        payload,
                        "OLD",
                        http_status=400,
                        **kwargs,
                    )

        changed = json.loads(payload)
        changed["chart"]["extra"] = True
        with self.assertRaises(ValueError):
            parse_yahoo_chart_no_data_evidence(
                json.dumps(changed).encode(),
                "OLD",
                http_status=400,
                request_period1=period1,
                request_period2=period2,
            )

    def test_api_error_wrong_symbol_and_wrong_currency_are_rejected(self):
        bars = _bars(pd.bdate_range("2024-01-02", periods=3))
        with self.assertRaisesRegex(ValueError, "chart.error"):
            parse_yahoo_chart_json(
                _chart_json(
                    pd.DataFrame(),
                    error={"code": "Not Found", "description": "No data"},
                ),
                "SEC",
            )
        with self.assertRaisesRegex(ValueError, "symbol mismatch"):
            parse_yahoo_chart_json(_chart_json(bars, symbol="OTHER"), "SEC")
        with self.assertRaisesRegex(ValueError, "currency must be USD"):
            parse_yahoo_chart_json(
                _chart_json(bars, symbol="SEC", currency="EUR"), "SEC"
            )

    def test_non_equity_or_non_us_exchange_metadata_is_rejected(self):
        bars = _bars(pd.bdate_range("2024-01-02", periods=3))
        mutual_fund = json.loads(_chart_json(bars, symbol="REUSED"))
        mutual_fund["chart"]["result"][0]["meta"]["instrumentType"] = (
            "MUTUALFUND"
        )
        with self.assertRaisesRegex(ValueError, "instrument type must be EQUITY"):
            parse_yahoo_chart_json(json.dumps(mutual_fund).encode(), "REUSED")

        foreign = json.loads(_chart_json(bars, symbol="REUSED"))
        foreign["chart"]["result"][0]["meta"]["exchangeName"] = "LSE"
        with self.assertRaisesRegex(ValueError, "exchange is not an allowed US exchange"):
            parse_yahoo_chart_json(json.dumps(foreign).encode(), "REUSED")

        wrong_timezone = json.loads(_chart_json(bars, symbol="REUSED"))
        wrong_timezone["chart"]["result"][0]["meta"][
            "exchangeTimezoneName"
        ] = "Europe/London"
        with self.assertRaisesRegex(ValueError, "exchange timezone"):
            parse_yahoo_chart_json(json.dumps(wrong_timezone).encode(), "REUSED")

    def test_non_daily_data_granularity_is_rejected(self):
        bars = _bars(pd.bdate_range("2024-01-02", periods=3))
        payload = json.loads(_chart_json(bars, symbol="SEC"))
        payload["chart"]["result"][0]["meta"]["dataGranularity"] = "3mo"
        with self.assertRaisesRegex(ValueError, "dataGranularity must be exactly 1d"):
            parse_yahoo_chart_json(json.dumps(payload).encode(), "SEC")

    def test_http_error_response_cannot_be_a_price_check(self):
        sessions = pd.bdate_range("2024-01-02", periods=30)
        prices = _bars(sessions)
        prices["security_id"] = "SEC"
        prices["currency"] = "USD"
        start = sessions[0].date().isoformat()
        end = sessions[-1].date().isoformat()
        target = script.PriceTarget(
            "SEC", "SEC", ("test",), request_start=start, request_end=end
        )
        response = _response(
            "SEC",
            _chart_json(prices, symbol="SEC"),
            start=start,
            end=end,
            http_status=404,
        )
        checks = script.build_price_checks(
            [target],
            {target.target_id: response},
            prices,
            pd.DataFrame(columns=["action_type", "security_id", "effective_date"]),
            [],
            _policy(),
        )
        self.assertEqual(checks[0]["status"], "mismatch")
        self.assertIn("HTTP 404", checks[0]["reason"])

    def test_raw_quote_close_is_used_instead_of_adjusted_close(self):
        bars = _bars(pd.bdate_range("2024-01-02", periods=3))
        parsed = parse_yahoo_chart_json(
            _chart_json(bars, symbol="SEC", adjclose=[1.0, 1.0, 1.0]), "SEC"
        )
        self.assertEqual(parsed.adjustment_basis, "raw_quote_ohlcv")
        self.assertEqual(parsed.bars["close"].tolist(), bars["close"].tolist())
        self.assertNotEqual(parsed.bars["close"].tolist(), [1.0, 1.0, 1.0])

    def test_misaligned_timestamps_and_invalid_ohlcv_are_rejected(self):
        bars = _bars(pd.bdate_range("2024-01-02", periods=3))
        payload = json.loads(_chart_json(bars, symbol="SEC"))
        payload["chart"]["result"][0]["indicators"]["quote"][0]["close"].pop()
        with self.assertRaisesRegex(ValueError, "misaligned"):
            parse_yahoo_chart_json(json.dumps(payload).encode(), "SEC")
        bars.loc[0, "high"] = 1.0
        with self.assertRaisesRegex(ValueError, "invalid OHLCV"):
            parse_yahoo_chart_json(_chart_json(bars, symbol="SEC"), "SEC")

    def test_all_overlap_sessions_are_checked_after_stable_split_regimes(self):
        sessions = pd.bdate_range("2024-01-02", periods=30)
        eod = _bars(sessions)
        eod["security_id"] = "SEC"
        eod["currency"] = "USD"
        yahoo = _bars(sessions)
        boundary = sessions[15]
        columns = ["open", "high", "low", "close"]
        yahoo.loc[yahoo["session"].lt(boundary), columns] *= 0.5
        target = script.PriceTarget("SEC", "SEC", ("test",))

        result = script.compare_price_history(
            target,
            eod,
            yahoo,
            [boundary.date().isoformat()],
            _policy(),
            "USD",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["overlap_session_count"], 30)
        self.assertTrue(result["all_overlap_sessions_compared"])
        self.assertEqual(len(result["regimes"]), 2)
        self.assertTrue(result["scale_stability_passed"])
        self.assertEqual(sum(result["mismatch_counts"].values()), 0)

    def test_unstable_ratio_is_not_hidden_by_regime_median(self):
        sessions = pd.bdate_range("2024-01-02", periods=30)
        eod = _bars(sessions)
        eod["security_id"] = "SEC"
        eod["currency"] = "USD"
        yahoo = _bars(sessions)
        yahoo.loc[8, "close"] *= 1.25
        target = script.PriceTarget("SEC", "SEC", ("test",))

        result = script.compare_price_history(
            target, eod, yahoo, [], _policy(), "USD"
        )
        self.assertEqual(result["status"], "mismatch")
        self.assertFalse(result["scale_stability_passed"])

    def test_short_common_tail_cannot_pass_full_internal_history_coverage(self):
        sessions = pd.bdate_range("2015-01-02", periods=1_000)
        eod = _bars(sessions)
        eod["security_id"] = "SAME-ID"
        eod["currency"] = "USD"
        yahoo = eod.tail(20)[
            ["session", "open", "high", "low", "close", "volume"]
        ].copy()
        target = script.PriceTarget(
            "SAME-ID", "NEW", ("ticker_change",), active_from="2015-01-02"
        )

        result = script.compare_price_history(
            target, eod, yahoo, [], _policy(), "USD"
        )

        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(result["overlap_session_count"], 20)
        self.assertEqual(result["eodhd_history_session_count"], 1_000)
        self.assertAlmostEqual(result["eodhd_full_history_overlap_ratio"], 0.02)
        self.assertFalse(result["session_coverage_passed"])

    def test_yahoo_supplement_cannot_validate_against_itself(self):
        sessions = pd.bdate_range("2024-01-02", periods=30)
        prices = _bars(sessions)
        prices["security_id"] = "SEC"
        prices["currency"] = "USD"
        prices["source"] = "yahoo_chart_json"
        start = sessions[0].date().isoformat()
        end = sessions[-1].date().isoformat()
        target = script.PriceTarget(
            "SEC", "SEC", ("test",), request_start=start, request_end=end
        )
        response = _response(
            "SEC", _chart_json(prices, symbol="SEC"), start=start, end=end
        )
        checks = script.build_price_checks(
            [target],
            {target.target_id: response},
            prices,
            pd.DataFrame(columns=["action_type", "security_id", "effective_date"]),
            [],
            _policy(),
        )

        self.assertEqual(checks[0]["status"], "mismatch")
        self.assertEqual(checks[0]["independent_internal_price_rows"], 0)
        self.assertEqual(checks[0]["self_source_rows_excluded"], 30)

    def test_cov_yahoo_fallback_is_not_independent_cross_validation(self):
        # The lifecycle successor collector's exact COV window is reproduced
        # here without importing its execution module a second time.
        sessions = pd.to_datetime(
            [
                "2015-01-02", "2015-01-05", "2015-01-06", "2015-01-07",
                "2015-01-08", "2015-01-09", "2015-01-12", "2015-01-13",
                "2015-01-14", "2015-01-15", "2015-01-16", "2015-01-20",
                "2015-01-21", "2015-01-22", "2015-01-23", "2015-01-26",
            ]
        )
        prices = _bars(sessions)
        prices["security_id"] = "COV-ID"
        prices["currency"] = "USD"
        prices["source"] = "yahoo_chart_json"
        start = sessions[0].date().isoformat()
        end = sessions[-1].date().isoformat()
        target = script.PriceTarget(
            "COV-ID",
            "COV",
            ("independent_provider_internal_source",),
            request_start=start,
            request_end=end,
        )
        response = _response(
            "COV", _chart_json(prices, symbol="COV"), start=start, end=end
        )
        checks = script.build_price_checks(
            [target],
            {target.target_id: response},
            prices,
            pd.DataFrame(columns=["action_type", "security_id", "effective_date"]),
            [],
            _policy(),
        )

        self.assertEqual(checks[0]["status"], "mismatch")
        self.assertEqual(checks[0]["independent_internal_price_rows"], 0)
        self.assertEqual(checks[0]["self_source_rows_excluded"], 16)

    def test_sparse_auto_aggregated_history_fails_xnys_inventory(self):
        calendar = xcals.get_calendar("XNYS")
        sessions = pd.DatetimeIndex(
            calendar.sessions_in_range("2024-01-02", "2024-04-30")[:60]
        ).tz_localize(None)
        prices = _bars(sessions)
        prices["security_id"] = "SEC"
        prices["currency"] = "USD"
        sparse = prices.iloc[::10].copy()
        start = sessions[0].date().isoformat()
        end = sessions[-1].date().isoformat()
        target = script.PriceTarget(
            "SEC",
            "SEC",
            ("test",),
            active_from=start,
            active_to=end,
            request_start=start,
            request_end=end,
        )
        response = _response(
            "SEC", _chart_json(sparse, symbol="SEC"), start=start, end=end
        )

        checks = script.build_price_checks(
            [target],
            {target.target_id: response},
            prices,
            pd.DataFrame(columns=["action_type", "security_id", "effective_date"]),
            [],
            _policy(),
        )

        self.assertEqual(checks[0]["status"], "mismatch")
        self.assertFalse(checks[0]["provider_request_inventory_passed"])
        self.assertLess(checks[0]["provider_request_xnys_coverage_ratio"], 0.2)

    def test_reused_ticker_series_outside_identity_window_is_rejected(self):
        current_sessions = pd.bdate_range("2018-01-02", periods=30)
        eod = _bars(current_sessions)
        eod["security_id"] = "CURRENT-LILAK"
        eod["currency"] = "USD"
        old_sessions = pd.bdate_range("2015-07-02", periods=30)
        yahoo = pd.concat(
            [_bars(old_sessions), _bars(current_sessions)], ignore_index=True
        )
        target = script.PriceTarget(
            "CURRENT-LILAK",
            "LILAK",
            ("lifecycle_action_successor",),
            active_from="2018-01-01",
        )

        result = script.compare_price_history(
            target, eod, yahoo, [], _policy(), "USD"
        )
        self.assertEqual(result["status"], "mismatch")
        self.assertFalse(result["identity_boundary_passed"])
        self.assertEqual(result["provider_sessions_before_identity"], 30)

        official_boundary = {
            "boundary": "active_from",
            "date": "2018-01-01",
            "source_url": "https://www.sec.gov/Archives/boundary.txt",
            "source_kind": "sec_filing",
            "evidence_sha256": "c" * 64,
            "official_original": True,
        }
        segmented = script.compare_price_history(
            target, eod, yahoo, [], _policy(), "USD", [official_boundary]
        )
        self.assertEqual(segmented["status"], "passed")
        self.assertTrue(segmented["identity_boundary_passed"])

    def test_open_history_terminal_boundary_is_derived_only_from_exact_event(self):
        calendar = xcals.get_calendar("XNYS")
        sessions = pd.DatetimeIndex(
            calendar.sessions_in_range("2024-01-02", "2024-06-30")[-60:]
        ).tz_localize(None)
        prices = _bars(sessions)
        prices["security_id"] = "OLD"
        prices["currency"] = "USD"
        terminal = sessions[-1].date().isoformat()
        effective = pd.Timestamp(
            calendar.next_session(pd.Timestamp(terminal))
        ).tz_localize(None).date().isoformat()
        target = script.PriceTarget(
            "OLD",
            "OLD",
            ("applied_resolution_source",),
            active_from=sessions[0].date().isoformat(),
            active_to="",
            terminal_event_id="event-1",
            request_end=terminal,
        )
        event = {
            "event_id": "event-1",
            "security_id": "OLD",
            "action_type": "cash_merger",
            "effective_date": effective,
            "status": "passed",
            "evidence_sha256": "b" * 64,
        }
        actions = pd.DataFrame(
            columns=["action_type", "security_id", "effective_date"]
        )

        period1, period2 = _request_periods(target.active_from, terminal)
        http_400_payload = json.dumps(
            {
                "chart": {
                    "result": None,
                    "error": {
                        "code": "Bad Request",
                        "description": (
                            "Data doesn't exist for startDate = "
                            f"{period1}, endDate = {period2}"
                        ),
                    },
                }
            },
            separators=(",", ":"),
        ).encode()
        responses = (
            (
                _response(
                    "OLD",
                    _retired_yhd_placeholder_json("OLD"),
                    start=target.active_from,
                    end=terminal,
                ),
                "http_200_empty_retired_yhd_placeholder",
            ),
            (
                _response(
                    "OLD",
                    http_400_payload,
                    start=target.active_from,
                    end=terminal,
                    http_status=400,
                ),
                "http_400_bounded_history_not_found",
            ),
        )
        for response, expected_kind in responses:
            with self.subTest(expected_kind=expected_kind):
                check = script.build_price_checks(
                    [target],
                    {target.target_id: response},
                    prices,
                    actions,
                    [event],
                    _policy(),
                )[0]
                self.assertEqual(check["status"], "explicit_exception")
                self.assertEqual(check["no_data_evidence_kind"], expected_kind)
                self.assertEqual(
                    check["exception"]["identity_date_basis"],
                    "derived_local_terminal_session",
                )
                self.assertEqual(
                    check["exception"]["derived_identity_active_to"], terminal
                )

        second_session = pd.Timestamp(
            calendar.next_session(pd.Timestamp(effective))
        ).tz_localize(None).date().isoformat()
        rejected_event = script.build_price_checks(
            [target],
            {target.target_id: responses[0][0]},
            prices,
            actions,
            [{**event, "effective_date": second_session}],
            _policy(),
        )[0]
        self.assertEqual(rejected_event["status"], "mismatch")
        self.assertFalse(rejected_event["exception"]["identity_date_match"])

        incomplete_calendar = script.build_price_checks(
            [target],
            {target.target_id: responses[0][0]},
            prices.iloc[1:].copy(),
            actions,
            [event],
            _policy(),
        )[0]
        self.assertEqual(incomplete_calendar["status"], "mismatch")
        self.assertFalse(
            incomplete_calendar["exception"]["terminal_calendar_complete"]
        )
        self.assertFalse(incomplete_calendar["exception"]["identity_date_match"])

    def test_terminal_date_binding_accepts_only_event_adjacent_market_boundaries(self):
        cases = (
            # Exact legal completion during the sessionless weekend gap.
            ("", "2016-05-27", "2016-05-28", True, "derived_local_terminal_session"),
            # Stored identity boundary is Sunday; the new identity starts Monday.
            (
                "2017-02-26",
                "2017-02-24",
                "2017-02-27",
                True,
                "stored_identity_active_to",
            ),
            # Completion is an XNYS session on which the retired identity did not trade.
            (
                "2025-08-07",
                "2025-08-06",
                "2025-08-07",
                True,
                "stored_identity_active_to",
            ),
        )
        for active_to, terminal, effective, complete, expected_basis in cases:
            with self.subTest(active_to=active_to, effective=effective):
                matched, basis, derived = script._terminal_event_date_binding(
                    active_to,
                    terminal,
                    effective,
                    terminal_calendar_complete=complete,
                )
                self.assertTrue(matched)
                self.assertEqual(basis, expected_basis)
                self.assertEqual(
                    derived,
                    terminal if not active_to else "",
                )

        rejected = (
            # Official event precedes a stored price tail.
            ("", "2021-05-17", "2021-05-14", True),
            # Multiple unexplained exchange sessions remain after the event.
            ("", "2024-05-24", "2024-03-25", True),
            # A derived boundary cannot use an incomplete terminal calendar.
            ("", "2016-05-27", "2016-05-28", False),
            # Stored boundary skips a live XNYS session before completion.
            ("2020-05-11", "2020-05-08", "2020-05-12", True),
        )
        for active_to, terminal, effective, complete in rejected:
            with self.subTest(rejected=(active_to, terminal, effective)):
                self.assertFalse(
                    script._terminal_event_date_binding(
                        active_to,
                        terminal,
                        effective,
                        terminal_calendar_complete=complete,
                    )[0]
                )

    def test_no_data_needs_official_calendar_and_exact_successor_pass(self):
        calendar = xcals.get_calendar("XNYS")
        sessions = calendar.sessions_in_range("2024-01-02", "2024-05-31")[-60:]
        sessions = pd.DatetimeIndex(sessions).tz_localize(None)
        old = _bars(sessions)
        old["security_id"] = "OLD"
        old["currency"] = "USD"
        successor = _bars(sessions)
        successor["security_id"] = "NEW"
        successor["currency"] = "USD"
        prices = pd.concat([old, successor], ignore_index=True)
        old_target = script.PriceTarget(
            "OLD",
            "OLD",
            ("applied_resolution_source",),
            active_from=sessions[0].date().isoformat(),
            active_to=sessions[-1].date().isoformat(),
            terminal_event_id="event-1",
            successor_security_id="NEW",
        )
        new_target = script.PriceTarget(
            "NEW",
            "NEW",
            ("applied_resolution_successor",),
            request_start=sessions[0].date().isoformat(),
            request_end=sessions[-1].date().isoformat(),
        )
        no_data = _response(
            "OLD",
            _chart_json(pd.DataFrame(), symbol="OLD"),
            start=old_target.active_from,
            end=old_target.active_to,
        )
        new_response = _response(
            "NEW",
            _chart_json(successor, symbol="NEW"),
            start=new_target.request_start,
            end=new_target.request_end,
        )
        evidence = "b" * 64
        delisting_date = pd.Timestamp(
            calendar.next_session(pd.Timestamp(sessions[-1]))
        ).tz_localize(None).date().isoformat()
        checks = script.build_price_checks(
            [old_target, new_target],
            {
                old_target.target_id: no_data,
                new_target.target_id: new_response,
            },
            prices,
            pd.DataFrame(columns=["action_type", "security_id", "effective_date"]),
            [
                {
                    "event_id": "event-1",
                    "security_id": "OLD",
                    "action_type": "stock_merger",
                    "effective_date": delisting_date,
                    "new_security_id": "NEW",
                    "new_symbol": "NEW",
                    "status": "passed",
                    "evidence_sha256": evidence,
                }
            ],
            _policy(),
        )
        by_id = {item["security_id"]: item for item in checks}
        self.assertEqual(by_id["NEW"]["status"], "passed")
        self.assertEqual(by_id["OLD"]["status"], "explicit_exception")
        self.assertTrue(by_id["OLD"]["exception"]["terminal_calendar_complete"])
        self.assertTrue(by_id["OLD"]["exception"]["successor_requirement_passed"])
        self.assertEqual(
            by_id["OLD"]["exception"]["successor_validation"]["target_id"],
            new_target.target_id,
        )
        self.assertEqual(
            by_id["OLD"]["exception"]["successor_validation"]["status"],
            "passed",
        )
        self.assertEqual(
            by_id["OLD"]["exception"]["official_action_type"], "stock_merger"
        )

        missing_successor = script.build_price_checks(
            [old_target],
            {old_target.target_id: no_data},
            prices,
            pd.DataFrame(columns=["action_type", "security_id", "effective_date"]),
            [
                {
                    "event_id": "event-1",
                    "security_id": "OLD",
                    "action_type": "stock_merger",
                    "effective_date": delisting_date,
                    "new_security_id": "NEW",
                    "new_symbol": "NEW",
                    "status": "passed",
                    "evidence_sha256": evidence,
                }
            ],
            _policy(),
        )
        self.assertEqual(missing_successor[0]["status"], "mismatch")
        self.assertEqual(
            missing_successor[0]["successor_failure"]["status"],
            "successor_target_not_unique",
        )
        self.assertEqual(
            missing_successor[0]["successor_failure"]["candidate_count"], 0
        )

    def test_real_yahoo_not_found_requires_exact_official_terminal_identity_date(self):
        calendar = xcals.get_calendar("XNYS")
        sessions = calendar.sessions_in_range("2024-01-02", "2024-05-31")[-60:]
        sessions = pd.DatetimeIndex(sessions).tz_localize(None)
        prices = _bars(sessions)
        prices["security_id"] = "OLD"
        prices["currency"] = "USD"
        terminal_session = sessions[-1].date().isoformat()
        delisting_date = pd.Timestamp(
            calendar.next_session(pd.Timestamp(terminal_session))
        ).tz_localize(None).date().isoformat()
        target = script.PriceTarget(
            "OLD",
            "OLD",
            ("applied_resolution_source",),
            active_from=sessions[0].date().isoformat(),
            active_to=terminal_session,
            terminal_event_id="event-1",
        )
        not_found_payload = json.dumps(
            {
                "chart": {
                    "result": None,
                    "error": {
                        "code": "Not Found",
                        "description": "No data found, symbol may be delisted",
                    },
                }
            },
            separators=(",", ":"),
        ).encode()
        response = _response(
            "OLD",
            not_found_payload,
            start=target.active_from,
            end=target.active_to,
            http_status=404,
            wrapper_hash="f" * 64,
        )
        event = {
            "event_id": "event-1",
            "security_id": "OLD",
            "action_type": "delisting",
            "effective_date": delisting_date,
            "status": "passed",
            "evidence_sha256": "b" * 64,
        }

        checks = script.build_price_checks(
            [target],
            {target.target_id: response},
            prices,
            pd.DataFrame(columns=["action_type", "security_id", "effective_date"]),
            [event],
            _policy(),
        )

        self.assertEqual(checks[0]["status"], "explicit_exception")
        self.assertEqual(checks[0]["provider_support"], "no_data")
        self.assertTrue(checks[0]["exception"]["identity_event_match"])
        self.assertTrue(checks[0]["exception"]["identity_date_match"])
        self.assertEqual(
            checks[0]["exception"]["official_action_type"], "delisting"
        )

        for action_type in sorted(script.YAHOO_NO_DATA_TERMINAL_ACTION_TYPES):
            with self.subTest(allowed_action_type=action_type):
                allowed = script.build_price_checks(
                    [target],
                    {target.target_id: response},
                    prices,
                    pd.DataFrame(
                        columns=["action_type", "security_id", "effective_date"]
                    ),
                    [{**event, "action_type": action_type}],
                    _policy(),
                )
                self.assertEqual(allowed[0]["status"], "explicit_exception")
                self.assertEqual(
                    allowed[0]["exception"]["official_action_type"], action_type
                )

        wrong_type = [{**event, "action_type": "spinoff"}]
        rejected = script.build_price_checks(
            [target],
            {target.target_id: response},
            prices,
            pd.DataFrame(columns=["action_type", "security_id", "effective_date"]),
            wrong_type,
            _policy(),
        )
        self.assertEqual(rejected[0]["status"], "mismatch")

        mutations = {
            "event_id": {**event, "event_id": "different"},
            "security_id": {**event, "security_id": "different"},
            "status": {**event, "status": "mismatch"},
            "effective_date": {
                **event,
                "effective_date": sessions[-2].date().isoformat(),
            },
        }
        for label, changed_event in mutations.items():
            with self.subTest(rejected_event_binding=label):
                changed = script.build_price_checks(
                    [target],
                    {target.target_id: response},
                    prices,
                    pd.DataFrame(
                        columns=["action_type", "security_id", "effective_date"]
                    ),
                    [changed_event],
                    _policy(),
                )
                self.assertEqual(changed[0]["status"], "mismatch")

        missing_calendar_session = script.build_price_checks(
            [target],
            {target.target_id: response},
            prices.iloc[1:].copy(),
            pd.DataFrame(columns=["action_type", "security_id", "effective_date"]),
            [event],
            _policy(),
        )
        self.assertEqual(missing_calendar_session[0]["status"], "mismatch")

        wrong_response = _response(
            "OTHER",
            not_found_payload,
            start=target.active_from,
            end=target.active_to,
            http_status=404,
            wrapper_hash="e" * 64,
        )
        wrong_cache_identity = script.build_price_checks(
            [target],
            {target.target_id: wrong_response},
            prices,
            pd.DataFrame(columns=["action_type", "security_id", "effective_date"]),
            [event],
            _policy(),
        )
        self.assertEqual(wrong_cache_identity[0]["status"], "mismatch")

    def test_reviewed_119_no_data_population_promotes_exact_nine_targets(self):
        calendar = xcals.get_calendar("XNYS")
        targets = []
        responses = {}
        price_frames = []
        event_checks = []
        expected_ids = set()
        not_found = json.dumps(
            {
                "chart": {
                    "result": None,
                    "error": {
                        "code": "Not Found",
                        "description": "No data found, symbol may be delisted",
                    },
                }
            },
            separators=(",", ":"),
        ).encode()

        for (
            symbol,
            security_id,
            active_from,
            active_to,
            effective_date,
            successor_id,
            successor_symbol,
            event_id,
            action_type,
            expected_target_id,
        ) in REVIEWED_NO_DATA_PROMOTIONS:
            sessions = pd.DatetimeIndex(
                calendar.sessions_in_range(
                    pd.Timestamp(active_to) - pd.Timedelta(days=300), active_to
                )[-60:]
            ).tz_localize(None)
            self.assertEqual(sessions[-1].date().isoformat(), active_to)
            target = script.PriceTarget(
                security_id,
                symbol,
                ("reviewed_no_data_terminal",),
                active_from=active_from,
                active_to=active_to,
                terminal_event_id=event_id,
                successor_security_id=successor_id,
            )
            self.assertEqual(target.target_id, expected_target_id)
            expected_ids.add(expected_target_id)
            targets.append(target)
            responses[target.target_id] = _response(
                symbol,
                not_found,
                start=active_from,
                end=active_to,
                http_status=404,
                wrapper_hash=expected_target_id,
            )
            old = _bars(sessions)
            old["security_id"] = security_id
            old["currency"] = "USD"
            price_frames.append(old)
            event_checks.append(
                {
                    "event_id": event_id,
                    "security_id": security_id,
                    "action_type": action_type,
                    "effective_date": effective_date,
                    "new_security_id": successor_id,
                    "new_symbol": successor_symbol,
                    "status": "passed",
                    "evidence_sha256": "a" * 64,
                }
            )

            successor_start = sessions[0].date().isoformat()
            successor_end = sessions[-1].date().isoformat()
            successor_target = script.PriceTarget(
                successor_id,
                successor_symbol,
                ("reviewed_no_data_successor",),
                request_start=successor_start,
                request_end=successor_end,
            )
            targets.append(successor_target)
            successor = _bars(sessions)
            successor["security_id"] = successor_id
            successor["currency"] = "USD"
            price_frames.append(successor)
            responses[successor_target.target_id] = _response(
                successor_symbol,
                _chart_json(successor, symbol=successor_symbol),
                start=successor_start,
                end=successor_end,
            )

        negative_sessions = pd.DatetimeIndex(
            calendar.sessions_in_range("2024-01-02", "2024-06-30")[-60:]
        ).tz_localize(None)
        negative_start = negative_sessions[0].date().isoformat()
        negative_end = negative_sessions[-1].date().isoformat()
        negative_effective = pd.Timestamp(
            calendar.next_session(pd.Timestamp(negative_end))
        ).tz_localize(None).date().isoformat()
        for index in range(110):
            security_id = f"NEGATIVE-{index:03d}"
            symbol = f"Z{index:03d}"
            event_id = f"negative-event-{index:03d}"
            target = script.PriceTarget(
                security_id,
                symbol,
                ("reviewed_no_data_negative_control",),
                active_from=negative_start,
                active_to=negative_end,
                terminal_event_id=event_id,
            )
            targets.append(target)
            responses[target.target_id] = _response(
                symbol,
                not_found,
                start=negative_start,
                end=negative_end,
                http_status=404,
                wrapper_hash=f"{index:064x}",
            )
            frame = _bars(negative_sessions)
            frame["security_id"] = security_id
            frame["currency"] = "USD"
            price_frames.append(frame)
            event_checks.append(
                {
                    "event_id": event_id,
                    "security_id": security_id,
                    "action_type": "spinoff",
                    "effective_date": negative_effective,
                    "status": "passed",
                    "evidence_sha256": "b" * 64,
                }
            )

        checks = script.build_price_checks(
            targets,
            responses,
            pd.concat(price_frames, ignore_index=True),
            pd.DataFrame(columns=["action_type", "security_id", "effective_date"]),
            event_checks,
            _policy(),
        )
        no_data_checks = [
            item for item in checks if item.get("provider_support") == "no_data"
        ]
        promoted = {
            item["target_id"]
            for item in no_data_checks
            if item["status"] == "explicit_exception"
        }
        self.assertEqual(len(no_data_checks), 119)
        self.assertEqual(promoted, expected_ids)
        self.assertEqual(len(promoted), 9)
        self.assertEqual(
            sum(item["status"] == "mismatch" for item in no_data_checks), 110
        )
        self.assertEqual(206 - len(promoted), 197)
        self.assertEqual(2 + len(promoted), 11)


class ReviewedNoDataSuccessorChainTest(unittest.TestCase):
    @staticmethod
    def _binding_fixture():
        policy = _policy()
        registry = script.reviewed_no_data_successor_chains(policy.prices)
        spec = next(
            value
            for value in registry.values()
            if value["nodes"][0]["provider_symbol"] == "QRTEA"
        )
        price_checks = []
        event_checks = []
        for node in spec["nodes"]:
            price_checks.append(
                {
                    "target_id": node["target_id"],
                    "security_id": node["security_id"],
                    "provider_symbol": node["provider_symbol"],
                    "identity_active_from": "2020-01-01",
                    "identity_active_to": "",
                    "terminal_event_id": node["event_id"],
                    "provider_support": "no_data",
                    "status": "mismatch",
                    "source_sha256": node["source_sha256"],
                    "cache_wrapper_sha256": node["cache_wrapper_sha256"],
                    "exception": {
                        "official_event_verified": True,
                        "identity_event_match": True,
                        "identity_date_match": True,
                        "terminal_calendar_complete": True,
                        "response_identity_match": True,
                        "no_data_evidence_validated": True,
                        "official_event_id": node["event_id"],
                        "official_evidence_sha256": node[
                            "official_evidence_sha256"
                        ],
                        "successor_security_id": node["successor_security_id"],
                    },
                }
            )
            event_checks.append(
                {
                    "event_id": node["event_id"],
                    "security_id": node["security_id"],
                    "action_type": "ticker_change",
                    "effective_date": "2024-01-02",
                    "new_security_id": node["successor_security_id"],
                    "new_symbol": node["successor_symbol"],
                    "status": "passed",
                    "evidence_sha256": node["official_evidence_sha256"],
                }
            )
        final = spec["final"]
        price_checks.append(
            {
                "target_id": final["target_id"],
                "security_id": final["security_id"],
                "provider_symbol": final["provider_symbol"],
                "identity_active_from": "2020-01-01",
                "identity_active_to": "",
                "status": "passed",
                "source_sha256": final["source_sha256"],
                "cache_wrapper_sha256": final["cache_wrapper_sha256"],
                "reviewed_price_evidence_applied": True,
                "reviewed_price_evidence_sha256": final[
                    "reviewed_price_evidence_sha256"
                ],
            }
        )
        return policy, registry, spec, price_checks, event_checks

    @staticmethod
    def _symc_binding_fixture():
        policy = _policy()
        registry = script.reviewed_no_data_successor_chains(policy.prices)
        spec = registry[
            "76cfddc97b878414119dfd9db08e356216cffc4ddc2839188451df534e11296f"
        ]
        intervals = (
            ("2015-01-01", "2019-11-01", "2019-11-04"),
            ("2019-11-04", "2022-11-07", "2022-11-08"),
        )
        price_checks = []
        event_checks = []
        for node, (active_from, active_to, effective_date) in zip(
            spec["nodes"], intervals, strict=True
        ):
            price_checks.append(
                {
                    "target_id": node["target_id"],
                    "security_id": node["security_id"],
                    "provider_symbol": node["provider_symbol"],
                    "identity_active_from": active_from,
                    "identity_active_to": active_to,
                    "terminal_event_id": node["event_id"],
                    "provider_support": "no_data",
                    "status": "explicit_exception",
                    "source_url": node["source_url"],
                    "expected_source_url": node["source_url"],
                    "request_period1": node["request_period1"],
                    "request_period2": node["request_period2"],
                    "http_status": node["http_status"],
                    "no_data_evidence_kind": node["no_data_evidence_kind"],
                    "source_sha256": node["source_sha256"],
                    "cache_wrapper_sha256": node["cache_wrapper_sha256"],
                    "exception": {
                        "official_event_verified": True,
                        "identity_event_match": True,
                        "identity_date_match": True,
                        "terminal_calendar_complete": True,
                        "response_identity_match": True,
                        "no_data_evidence_validated": True,
                        "official_event_id": node["event_id"],
                        "official_evidence_sha256": node[
                            "official_evidence_sha256"
                        ],
                        "successor_security_id": node["successor_security_id"],
                    },
                }
            )
            event_checks.append(
                {
                    "event_id": node["event_id"],
                    "security_id": node["security_id"],
                    "action_type": "ticker_change",
                    "effective_date": effective_date,
                    "new_security_id": node["successor_security_id"],
                    "new_symbol": node["successor_symbol"],
                    "status": "passed",
                    "evidence_sha256": node["official_evidence_sha256"],
                }
            )
        final = spec["final"]
        price_checks.append(
            {
                "target_id": final["target_id"],
                "security_id": final["security_id"],
                "provider_symbol": final["provider_symbol"],
                "identity_active_from": "2015-01-01",
                "identity_active_to": "",
                "provider_support": "price_history",
                "status": "passed",
                "source_sha256": final["source_sha256"],
                "cache_wrapper_sha256": final["cache_wrapper_sha256"],
                "reviewed_price_evidence_applied": False,
                "reviewed_price_evidence_sha256": "",
            }
        )
        return policy, registry, spec, price_checks, event_checks

    def test_symc_nlok_chain_binds_exact_requests_and_rejects_self_validation(self):
        _, registry, spec, price_checks, event_checks = self._symc_binding_fixture()

        def bind(prices, events):
            return script.successor_price_check_binding(
                prices,
                events[0],
                source_target_id=spec["root_target_id"],
                expected_successor_security_id=spec["nodes"][0][
                    "successor_security_id"
                ],
                reviewed_successor_chains=registry,
                event_checks=events,
            )

        exact = bind(price_checks, event_checks)
        self.assertTrue(exact["passed"])
        self.assertEqual(exact["target_id"], spec["nodes"][1]["target_id"])
        self.assertEqual(
            exact["chain_sha256"],
            reviewed_no_data_successor_chain_sha256(spec),
        )

        mutations = {
            "wrong_symbol": lambda prices, events: prices[0].update(
                {"provider_symbol": "OTHER"}
            ),
            "wrong_url": lambda prices, events: prices[0].update(
                {"source_url": prices[0]["source_url"].replace("SYMC", "NLOK")}
            ),
            "wrong_period": lambda prices, events: prices[0].update(
                {"request_period2": prices[0]["request_period2"] + 86400}
            ),
            "wrong_http_status": lambda prices, events: prices[0].update(
                {"http_status": 200}
            ),
            "wrong_body_hash": lambda prices, events: prices[0].update(
                {"source_sha256": "0" * 64}
            ),
            "wrong_wrapper_hash": lambda prices, events: prices[0].update(
                {"cache_wrapper_sha256": "0" * 64}
            ),
            "broken_chain": lambda prices, events: events[1].update(
                {"new_symbol": "OTHER"}
            ),
            "final_not_passed": lambda prices, events: prices[-1].update(
                {"status": "explicit_exception"}
            ),
            "provider_self_validation": lambda prices, events: prices[-1].update(
                {"provider_support": "no_data"}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                changed_prices = copy.deepcopy(price_checks)
                changed_events = copy.deepcopy(event_checks)
                mutate(changed_prices, changed_events)
                self.assertFalse(bind(changed_prices, changed_events)["passed"])

    def test_symc_same_sid_binding_forbids_terminal_resolution_semantics(self):
        policy = _policy()
        extractions = script.reviewed_nonterminal_extractions(policy.events)
        event_id = (
            "fc556f24050c3205150b7934f431b72d6348ab5fbfad3e85bfbb149c7b9781bd"
        )
        extraction = extractions[event_id]
        target = {
            "target_id": (
                "76cfddc97b878414119dfd9db08e356216cffc4ddc2839188451df534e11296f"
            ),
            "security_id": extraction["security_id"],
            "provider_symbol": "SYMC",
            "active_from": "2015-01-01",
            "active_to": "2019-11-01",
            "terminal_event_id": event_id,
            "successor_security_id": extraction["security_id"],
        }
        event = {
            **extraction,
            "status": "passed",
            "validation_kind": script.NONTERMINAL_EVENT_VALIDATION,
            "candidate_id": "",
            "lifecycle_report_extraction_approved": False,
            "reviewed_extraction_match": True,
            "reviewed_extraction_sha256": (
                script.reviewed_nonterminal_extraction_sha256(extraction)
            ),
            "evidence_sha256": extraction["source_hash"],
        }
        binding = script.reviewed_nonterminal_same_sid_no_data_binding(
            target, event, extractions
        )
        self.assertIsNotNone(binding)
        self.assertFalse(binding["terminal_resolution_required"])
        self.assertTrue(binding["terminal_resolution_forbidden"])

        mutations = (
            {"validation_kind": script.TERMINAL_EVENT_VALIDATION},
            {"candidate_id": "forged-terminal-candidate"},
            {"lifecycle_report_extraction_approved": True},
        )
        for changed in mutations:
            with self.subTest(changed=changed):
                self.assertIsNone(
                    script.reviewed_nonterminal_same_sid_no_data_binding(
                        target, {**event, **changed}, extractions
                    )
                )

    def test_reviewed_same_sid_no_data_cohort_is_exact_and_binds_policy(self):
        new_event_ids = {
            "2df5c4c0298e5ff531aaa785146a20cba98d22080c970eabbd841b802ec60e7e",
            "31281f82fe09566d70782ba37514ea57e1da1a6915b68ed14c56a9569832a53e",
            "350960af29b81ec304e10cc318837f2e24c70ce2f89983bad95df38ad7f66cda",
            "3df08f0e3e4593c773a5cddf9d7ff1abc46b017110516d1fb1dc65b3d89dbd43",
            "47235ed0f22108df208fefcab63d0bb9118c5ecb58345387b1a50431e7bc388c",
            "4a662e7caca7ed147c918e5907187b6890397ca72b9b8a2e06e7ee411cedbd7c",
            "5c67b30c00cf201d6248706eabb50a89f50312ff11b445e4a612af76168d4cbf",
            "6cdae488bbfad53d85b79752afcb2c54c5b19b41b9f55800cb8ab4db51901d50",
            "8f6dd7b99d5cc344bb60449f3536979a54dd1737f35e9b16b99f795b1d271dc5",
            "958ed869cc179ffda932c0012af35439ea22b21b07691fb6c5e221844cb0a0ed",
            "a066f9db433eb3bce0365744b09de62e7c10a64d9d89eabed22b3ec359963718",
        }
        self.assertEqual(
            len(TRUSTED_REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_SPECS), 13
        )
        self.assertTrue(
            new_event_ids.issubset(
                TRUSTED_REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_SPECS
            )
        )

        extractions = script.reviewed_nonterminal_extractions(_policy().events)
        for event_id in sorted(new_event_ids):
            with self.subTest(event_id=event_id):
                spec = TRUSTED_REVIEWED_NONTERMINAL_SAME_SID_NO_DATA_SPECS[
                    event_id
                ]
                extraction = extractions[event_id]
                target = {
                    "target_id": spec["source_target_id"],
                    "security_id": spec["security_id"],
                    "provider_symbol": spec["old_symbol"],
                    "active_from": spec["old_active_from"],
                    "active_to": spec["old_active_to"],
                    "terminal_event_id": event_id,
                    "successor_security_id": spec["security_id"],
                }
                event = {
                    **extraction,
                    "status": "passed",
                    "validation_kind": script.NONTERMINAL_EVENT_VALIDATION,
                    "candidate_id": "",
                    "lifecycle_report_extraction_approved": False,
                    "reviewed_extraction_match": True,
                    "reviewed_extraction_sha256": (
                        script.reviewed_nonterminal_extraction_sha256(extraction)
                    ),
                    "evidence_sha256": extraction["source_hash"],
                }
                binding = script.reviewed_nonterminal_same_sid_no_data_binding(
                    target, event, extractions
                )
                self.assertIsNotNone(binding)
                self.assertEqual(
                    binding["successor_target_id"], spec["successor_target_id"]
                )
                self.assertTrue(binding["same_security_id_continuation"])
                self.assertTrue(binding["terminal_resolution_forbidden"])

    def test_collector_promotes_only_the_exact_code_pinned_finite_chain(self):
        _, registry, spec, price_checks, event_checks = self._binding_fixture()
        binding = script.successor_price_check_binding(
            price_checks,
            event_checks[0],
            source_target_id=spec["root_target_id"],
            expected_successor_security_id=spec["nodes"][0][
                "successor_security_id"
            ],
            reviewed_successor_chains=registry,
            event_checks=event_checks,
        )
        self.assertTrue(binding["passed"])
        self.assertEqual(
            binding["validation_basis"],
            script.REVIEWED_NO_DATA_SUCCESSOR_CHAIN_BASIS,
        )
        self.assertEqual(
            binding["chain_target_ids"],
            [node["target_id"] for node in spec["nodes"]]
            + [spec["final"]["target_id"]],
        )

        for mutate in (
            lambda prices, events: prices[-1].update({"source_sha256": "0" * 64}),
            lambda prices, events: events[-1].update(
                {"evidence_sha256": "0" * 64}
            ),
        ):
            with self.subTest(mutate=mutate):
                changed_prices = copy.deepcopy(price_checks)
                changed_events = copy.deepcopy(event_checks)
                mutate(changed_prices, changed_events)
                rejected = script.successor_price_check_binding(
                    changed_prices,
                    changed_events[0],
                    source_target_id=spec["root_target_id"],
                    expected_successor_security_id=spec["nodes"][0][
                        "successor_security_id"
                    ],
                    reviewed_successor_chains=registry,
                    event_checks=changed_events,
                )
                self.assertFalse(rejected["passed"])

    def test_new_finite_chain_roots_bind_only_exact_archived_nodes(self):
        registry = script.reviewed_no_data_successor_chains(_policy().prices)
        expected = {
            "0c7ccbea602b6ae66d806f0f13edfc3034b14fd7ab49b98bf8e7667b6d0be110": (
                "DOW",
                "DD",
            ),
            "4ab52d92c2c23f0103bd7b20979c943223107b3ff72d5ca3d42e5717ccf5bb10": (
                "COG",
                "DVN",
            ),
            "cd1e97410c98f59bfb065a2f3642cb602b77241b1bc5dd0428631f0f0ff80e31": (
                "XEC",
                "DVN",
            ),
            "db0b71658e5be84e59ce757b46b9c150d8d8af4e768b3fbb37e2d2f7191d3204": (
                "HCP",
                "DOC",
            ),
        }
        self.assertEqual(len(registry), 16)
        self.assertTrue(set(expected).issubset(registry))

        for root_target_id, (root_symbol, final_symbol) in expected.items():
            with self.subTest(root_symbol=root_symbol):
                spec = registry[root_target_id]
                self.assertEqual(spec["nodes"][0]["provider_symbol"], root_symbol)
                self.assertEqual(spec["final"]["provider_symbol"], final_symbol)
                price_checks = []
                event_checks = []
                for node in spec["nodes"]:
                    price_checks.append(
                        {
                            "target_id": node["target_id"],
                            "security_id": node["security_id"],
                            "provider_symbol": node["provider_symbol"],
                            "identity_active_from": "2010-01-01",
                            "identity_active_to": "",
                            "terminal_event_id": node["event_id"],
                            "provider_support": "no_data",
                            "status": "explicit_exception",
                            "source_url": node["source_url"],
                            "expected_source_url": node["source_url"],
                            "request_period1": node["request_period1"],
                            "request_period2": node["request_period2"],
                            "http_status": node["http_status"],
                            "no_data_evidence_kind": node[
                                "no_data_evidence_kind"
                            ],
                            "source_sha256": node["source_sha256"],
                            "cache_wrapper_sha256": node[
                                "cache_wrapper_sha256"
                            ],
                            "exception": {
                                "official_event_verified": True,
                                "identity_event_match": True,
                                "identity_date_match": True,
                                "terminal_calendar_complete": True,
                                "response_identity_match": True,
                                "no_data_evidence_validated": True,
                                "official_event_id": node["event_id"],
                                "official_evidence_sha256": node[
                                    "official_evidence_sha256"
                                ],
                                "successor_security_id": node[
                                    "successor_security_id"
                                ],
                            },
                        }
                    )
                    event_checks.append(
                        {
                            "event_id": node["event_id"],
                            "security_id": node["security_id"],
                            "action_type": "ticker_change",
                            "effective_date": "2024-01-02",
                            "new_security_id": node["successor_security_id"],
                            "new_symbol": node["successor_symbol"],
                            "status": "passed",
                            "evidence_sha256": node[
                                "official_evidence_sha256"
                            ],
                        }
                    )
                final = spec["final"]
                price_checks.append(
                    {
                        "target_id": final["target_id"],
                        "security_id": final["security_id"],
                        "provider_symbol": final["provider_symbol"],
                        "identity_active_from": "2010-01-01",
                        "identity_active_to": "",
                        "provider_support": "price_history",
                        "status": "passed",
                        "source_sha256": final["source_sha256"],
                        "cache_wrapper_sha256": final[
                            "cache_wrapper_sha256"
                        ],
                        "reviewed_price_evidence_applied": False,
                        "reviewed_price_evidence_sha256": "",
                    }
                )

                binding = script.successor_price_check_binding(
                    price_checks,
                    event_checks[0],
                    source_target_id=root_target_id,
                    expected_successor_security_id=spec["nodes"][0][
                        "successor_security_id"
                    ],
                    reviewed_successor_chains=registry,
                    event_checks=event_checks,
                )
                self.assertTrue(binding["passed"])
                self.assertEqual(binding["chain_root_target_id"], root_target_id)
                self.assertEqual(binding["final_target_id"], final["target_id"])

                price_checks[0]["cache_wrapper_sha256"] = "0" * 64
                rejected = script.successor_price_check_binding(
                    price_checks,
                    event_checks[0],
                    source_target_id=root_target_id,
                    expected_successor_security_id=spec["nodes"][0][
                        "successor_security_id"
                    ],
                    reviewed_successor_chains=registry,
                    event_checks=event_checks,
                )
                self.assertFalse(rejected["passed"])

    def test_collector_rejects_a_rehashed_chain_policy_tamper(self):
        value = copy.deepcopy(_policy().value)
        value["prices"]["reviewed_no_data_successor_chains"][0]["final"][
            "source_sha256"
        ] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.yaml"
            path.write_text(yaml.safe_dump(value), encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, "successor-chain inventory is not code-pinned"
            ):
                script.load_policy(path)

    def test_collector_rejects_a_rehashed_unsupported_path_policy_tamper(self):
        value = copy.deepcopy(_policy().value)
        value["prices"]["reviewed_no_data_unsupported_paths"][0][
            "official_evidence_sha256"
        ] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.yaml"
            path.write_text(yaml.safe_dump(value), encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError,
                "unsupported-path inventory is not the exact isolated code-pinned set",
            ):
                script.load_policy(path)

    def test_collector_permanent_exception_no_data_allowlist_is_finite(self):
        target = {
            "target_id": (
                "a1a391246a313a058341bfae9a17b5cabb6c94d2d930672e8a06414c869f3ec5"
            ),
            "security_id": "US:EODHD:9398e16f-425d-5a51-8720-35fba7433f28",
            "provider_symbol": "TFCFA",
            "active_to": "2019-03-19",
            "terminal_event_id": "",
            "successor_security_id": "",
        }
        passed_check = {
            "status": "passed",
            "validation_kind": (
                "permanent_lifecycle_exception_official_provenance"
            ),
            "security_id": target["security_id"],
            "symbol": "TFCFA",
            "last_price_date": "2019-03-19",
            "exception_code": "unsupported_consideration",
            "identity_date_bound": True,
            "registry_binding_passed": True,
            "reviewer_pin_passed": True,
            "official_original": True,
            "exact_archive_pair": True,
            "archive_payload_verified": True,
            "candidate_id": (
                "3e701ee6402e494c072e4c1efa03cdd7728e28b52338f0e6802e59fbfb4f7667"
            ),
            "evidence_id": "tfcfa_2019_disney_proration",
            "evidence_sha256": (
                "08ba720b0e5326b652fb94cde8ba44c45bcac09a81b77d70f006e934e9d36d93"
            ),
            "exception_reason": "cash-or-stock election and proration",
            "source_url": (
                "https://www.sec.gov/Archives/edgar/data/1308161/"
                "000119312519079716/d710665dex991.htm"
            ),
        }
        binding = script.permanent_exception_no_data_binding(
            target,
            "2019-03-19",
            [passed_check],
        )
        self.assertIsNotNone(binding)
        self.assertEqual(
            binding["validation_basis"],
            script.REVIEWED_PERMANENT_EXCEPTION_NO_DATA_BASIS,
        )
        self.assertFalse(binding["generic_date_tolerance"])

        # AABA and UTX are the two additional reviewed permanent limitations.
        # They remain explicit no-price-path exceptions, never accuracy passes.
        aaba_target = {
            **target,
            "target_id": (
                "8e4019d9e30a1697a498ab984b1dfc65cac75f8023e7fd30e5fb91a2fff8865d"
            ),
            "security_id": "US:EODHD:9b1bbdaa-839c-5d59-8bda-99b2087022e6",
            "provider_symbol": "AABA",
            "active_to": "2019-10-02",
        }
        aaba_check = {
            **passed_check,
            "security_id": aaba_target["security_id"],
            "symbol": "AABA",
            "last_price_date": "2019-10-02",
            "candidate_id": (
                "6b4e657e672273e482ceb480d6a9391e97ef88285082f87bfbe9513b8b5c22c1"
            ),
            "evidence_id": "aaba_2019_liquidation_distributions",
            "evidence_sha256": (
                "01cea25a16cb222ab50254d9c9758a0a7cb0751e045e0696e05d203464f9682d"
            ),
        }
        aaba_binding = script.permanent_exception_no_data_binding(
            aaba_target,
            "2019-10-02",
            [aaba_check],
        )
        self.assertIsNotNone(aaba_binding)
        self.assertFalse(aaba_binding["price_history_supported"])
        self.assertFalse(aaba_binding["generic_date_tolerance"])

        utx_target = {
            **target,
            "target_id": (
                "7a86baed1c01c95f89ac3235093e04947d5af4c362692237ecc51da57fcfc046"
            ),
            "security_id": "US:EODHD:aefd1dd7-529d-5b6a-80e9-65d0e14102a6",
            "provider_symbol": "UTX",
            "active_to": "2020-04-02",
        }
        utx_check = {
            **passed_check,
            "security_id": utx_target["security_id"],
            "symbol": "UTX",
            "last_price_date": "2020-04-02",
            "candidate_id": (
                "a8bcc6524b2ed8772255bdf856e45162a3a2e8a590fb3b0786486c9c030ab9c2"
            ),
            "evidence_id": "utx_2020_carr_otis_distributions",
            "evidence_sha256": (
                "8b3131e8bf46b322c0c7c9e37e32c624c05336e3f3acaddf86c777ce17f7d6a2"
            ),
        }
        utx_binding = script.permanent_exception_no_data_binding(
            utx_target,
            "2020-04-02",
            [utx_check],
        )
        self.assertIsNotNone(utx_binding)
        self.assertFalse(utx_binding["price_history_supported"])

        unreviewed_target = {**aaba_target, "target_id": "0" * 64}
        self.assertIsNone(
            script.permanent_exception_no_data_binding(
                unreviewed_target,
                "2019-10-02",
                [aaba_check],
            )
        )


class PinnedExternalOverlapTest(unittest.TestCase):
    def test_identity_collector_and_publication_gate_pin_the_same_boris_files(self):
        for symbol in ("LILA", "LILAK"):
            collector = identity_script.BORIS_KAGGLE_FILES[symbol]
            gate = TRUSTED_PINNED_EXTERNAL_OVERLAPS[symbol]
            self.assertEqual(collector["url"], gate["external_source_url"])
            self.assertEqual(collector["sha256"], gate["external_source_sha256"])
            self.assertEqual(collector["segment_start"], gate["overlap_start"])
            self.assertEqual(collector["segment_end"], gate["overlap_end"])
            self.assertEqual(collector["segment_rows"], gate["overlap_sessions"])

    def test_policy_rejects_substituting_a_different_external_payload_hash(self):
        value = json.loads(json.dumps(_policy().value))
        value["prices"]["pinned_external_overlaps"][0][
            "external_source_sha256"
        ] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.yaml"
            path.write_text(yaml.safe_dump(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "controls changed"):
                script.load_policy(path)

    def test_old_lila_yahoo_primary_is_checked_against_archived_boris_overlap(self):
        policy_value = json.loads(json.dumps(_policy().value))
        spec = policy_value["prices"]["pinned_external_overlaps"][0]
        primary_sessions = script._xnys_sessions(
            spec["active_from"], spec["active_to"]
        )
        overlap_sessions = script._xnys_sessions(
            spec["overlap_start"], spec["overlap_end"]
        )
        primary_close = {
            session: float(100 + index)
            for index, session in enumerate(primary_sessions)
        }
        raw_sessions = ["2015-06-22", "2015-06-23", *overlap_sessions]
        raw_rows = []
        for index, session in enumerate(raw_sessions):
            close = (
                primary_close[session] / 2.0
                if session in primary_close
                else float(40 + index)
            )
            raw_rows.append(
                {
                    "Date": session,
                    "Open": close * 0.99,
                    "High": close * 1.01,
                    "Low": close * 0.98,
                    "Close": close,
                    "Volume": 1000,
                    "OpenInt": 0,
                }
            )
        external_payload = pd.DataFrame(raw_rows).to_csv(index=False).encode()
        external_hash = script.sha256_bytes(external_payload)
        spec["external_source_sha256"] = external_hash
        policy = script.Policy(policy_value)

        primary_bars = pd.DataFrame(
            {
                "session": pd.to_datetime(primary_sessions),
                "open": [primary_close[value] * 0.99 for value in primary_sessions],
                "high": [primary_close[value] * 1.01 for value in primary_sessions],
                "low": [primary_close[value] * 0.98 for value in primary_sessions],
                "close": [primary_close[value] for value in primary_sessions],
                "volume": [1000.0] * len(primary_sessions),
            }
        )
        primary_payload = _chart_json(primary_bars, symbol="LILA")
        primary_hash = script.sha256_bytes(primary_payload)
        primary = primary_bars.copy()
        primary["session"] = primary["session"].dt.date.astype(str)
        primary["security_id"] = "OLD-LILA"
        primary["source"] = spec["primary_source"]
        primary["source_url"] = spec["primary_source_url"]
        primary["source_hash"] = primary_hash
        primary["currency"] = "USD"
        target = script.PriceTarget(
            "OLD-LILA",
            "LILA",
            ("independent_provider_internal_source",),
            spec["active_from"],
            spec["active_to"],
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_rows = []
            for digest, payload, url in (
                (primary_hash, primary_payload, spec["primary_source_url"]),
                (external_hash, external_payload, spec["external_source_url"]),
            ):
                object_path = f"archives/{digest}.bin.gz"
                destination = root / object_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(gzip.compress(payload, mtime=0))
                archive_rows.append(
                    {
                        "archive_id": digest,
                        "source_hash": digest,
                        "source_url": url,
                        "object_path": object_path,
                        "retrieved_at": "2026-07-18T00:00:00Z",
                    }
                )
            evidence = script.resolve_pinned_overlap_evidence(
                SimpleNamespace(root=root),
                [target],
                primary,
                pd.DataFrame(archive_rows),
                policy,
            )

        checks = script.build_price_checks(
            [target],
            {},
            primary,
            pd.DataFrame(columns=["action_type", "security_id", "effective_date"]),
            [],
            policy,
            {},
            evidence,
        )

        self.assertEqual(checks[0]["status"], "passed")
        self.assertEqual(
            checks[0]["validation_basis"], script.PINNED_EXTERNAL_OVERLAP_VALIDATION
        )
        self.assertEqual(checks[0]["overlap_session_count"], 597)
        self.assertEqual(checks[0]["internal_history_session_count"], 630)
        self.assertEqual(checks[0]["uncrosschecked_tail_sessions"], 33)
        self.assertFalse(checks[0]["independent_provider_claimed"])


class PermanentExceptionCrossValidationTest(unittest.TestCase):
    def _fixtures(self, root: Path):
        payload = b"official permanent lifecycle exception filing"
        digest = script.sha256_bytes(payload)
        source_url = (
            "https://www.sec.gov/Archives/edgar/data/1/"
            "000000000124000001/permanent-exception.txt"
        )
        object_path = f"archives/2026-07-18/{digest}.txt.gz"
        path = root / object_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(payload, mtime=0))
        last_price_date = "2024-01-05"
        security_id = "OLD-ID"
        resolutions = pd.DataFrame(
            [
                {
                    "candidate_id": script.lifecycle_candidate_id(
                        security_id, last_price_date
                    ),
                    "security_id": security_id,
                    "symbol": "OLD",
                    "last_price_date": last_price_date,
                    "resolution": "exception",
                    "exception_code": "unsupported_consideration",
                    "exception_reason": "CVR terms are not representable.",
                    "recheck_after": "",
                    "source_url": source_url,
                    "source_hash": digest,
                }
            ]
        )
        archive = pd.DataFrame(
            [
                {
                    "archive_id": digest,
                    "source_hash": digest,
                    "source_url": source_url,
                    "object_path": object_path,
                }
            ]
        )
        template = next(iter(script.trusted_permanent_exception_specs().values()))
        official_spec = replace(
            template,
            evidence_id="fixture_permanent_exception",
            candidate_symbols=("OLD",),
            candidate_name_contains=("old",),
            candidate_security_ids=(security_id,),
            candidate_last_price_dates=(last_price_date,),
            effective_date=last_price_date,
            resolution_kind="exception",
            exception_code="unsupported_consideration",
            action_type="",
            cash_amount=None,
            claim="CVR terms are not representable.",
            source_url=source_url,
            source_sha256=digest,
        )
        return (
            SimpleNamespace(root=root),
            resolutions,
            archive,
            {official_spec.evidence_id: official_spec},
        )

    def test_exact_official_url_hash_archive_and_identity_date_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, resolutions, archive, specs = self._fixtures(Path(directory))
            checks = script.build_permanent_exception_checks(
                repository, resolutions, archive, specs
            )

        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]["status"], "passed")
        self.assertTrue(checks[0]["identity_date_bound"])
        self.assertTrue(checks[0]["official_original"])
        self.assertTrue(checks[0]["exact_archive_pair"])
        self.assertTrue(checks[0]["archive_payload_verified"])

    def test_self_authored_report_hash_cannot_replace_official_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, resolutions, archive, specs = self._fixtures(Path(directory))
            resolutions.loc[0, "source_url"] = "archive://lifecycle/evidence-report"
            archive.loc[0, "source_url"] = "archive://lifecycle/evidence-report"
            checks = script.build_permanent_exception_checks(
                repository, resolutions, archive, specs
            )

        self.assertEqual(checks[0]["status"], "mismatch")
        self.assertFalse(checks[0]["official_original"])
        self.assertIn("official SEC/FDIC URL", checks[0]["reasons"])

    def test_wrong_identity_date_or_missing_bytes_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, resolutions, archive, specs = self._fixtures(Path(directory))
            resolutions.loc[0, "last_price_date"] = "2024-01-08"
            archive.loc[0, "object_path"] = "archives/missing-official.bin.gz"
            checks = script.build_permanent_exception_checks(
                repository, resolutions, archive, specs
            )

        self.assertEqual(checks[0]["status"], "mismatch")
        self.assertFalse(checks[0]["identity_date_bound"])
        self.assertFalse(checks[0]["archive_payload_verified"])
        self.assertIn("candidate identity/date binding", checks[0]["reasons"])
        self.assertIn("archived official payload bytes", checks[0]["reasons"])

    def test_different_official_sec_document_cannot_replace_exact_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, resolutions, archive, specs = self._fixtures(Path(directory))
            other_url = (
                "https://www.sec.gov/Archives/edgar/data/2/"
                "000000000224000001/other-official.txt"
            )
            resolutions.loc[0, "source_url"] = other_url
            archive.loc[0, "source_url"] = other_url
            checks = script.build_permanent_exception_checks(
                repository, resolutions, archive, specs
            )

        self.assertTrue(checks[0]["official_original"])
        self.assertTrue(checks[0]["exact_archive_pair"])
        self.assertFalse(checks[0]["reviewer_pin_passed"])
        self.assertEqual(checks[0]["status"], "mismatch")


class CrossValidationPreflightTest(unittest.TestCase):
    def test_temporary_lifecycle_exception_fails_before_yahoo_access(self):
        versions = {
            name: f"{name}-v1"
            for name in (*script.VALIDATED_DATASETS, "source_archive")
        }
        release = SimpleNamespace(dataset_versions=versions)
        frames = {name: pd.DataFrame() for name in versions}
        frames["lifecycle_resolutions"] = pd.DataFrame(
            [
                {
                    "resolution": "exception",
                    "exception_code": "temporary_review",
                    "recheck_after": "2026-10-31",
                }
            ]
        )
        repository = SimpleNamespace(
            current_release=Mock(return_value=(release, None)),
            read_frame=Mock(side_effect=lambda name, _version: frames[name].copy()),
        )
        cache = Mock()

        with self.assertRaisesRegex(RuntimeError, "temporary exceptions must be zero"):
            script.prepare_cross_validation(
                repository,
                _policy(),
                cache,
                fetch_missing=True,
            )

        cache.fill_missing.assert_not_called()
        cache.get.assert_not_called()

    def test_offline_apply_plan_uses_cache_get_and_never_fetches(self):
        versions = {
            name: f"{name}-v1"
            for name in (*script.VALIDATED_DATASETS, "source_archive")
        }
        release = SimpleNamespace(dataset_versions=versions)
        frames = {name: pd.DataFrame() for name in versions}
        frames["lifecycle_resolutions"] = pd.DataFrame(
            [
                {
                    "resolution": "applied",
                    "exception_code": "",
                    "recheck_after": "",
                }
            ]
        )
        frames["source_archive"] = pd.DataFrame(columns=["archive_id"])
        repository = SimpleNamespace(
            current_release=Mock(return_value=(release, None)),
            read_frame=Mock(side_effect=lambda name, _version: frames[name].copy()),
        )
        cache = Mock()
        cache.get.side_effect = RuntimeError("offline cache read reached")
        target = script.PriceTarget("SEC", "SEC", ("test",))
        policy = _policy()
        market_date_ids = set(
            script.reviewed_terminal_market_date_corrections(policy.events)
        )
        policy_exception_ids = set(
            script.reviewed_terminal_policy_exceptions(policy.events)
        )
        tail_ids = set(
            script.reviewed_terminal_price_tail_corrections(policy.events)
        )
        gate_checks = []
        for event_id in script.TRUSTED_REVIEWED_TERMINAL_EVENT_GATE_EVENT_IDS:
            gate_checks.append(
                {
                    "event_id": event_id,
                    "reviewed_terminal_event_gate_applied": True,
                    "reviewed_terminal_market_date_correction_applied": (
                        event_id in market_date_ids
                    ),
                    "reviewed_terminal_policy_exception_applied": (
                        event_id in policy_exception_ids
                    ),
                    "reviewed_terminal_price_tail_correction_applied": (
                        event_id in tail_ids
                    ),
                }
            )
        with (
            patch.object(script, "_lifecycle_evidence_report", return_value=({}, "a" * 64)),
            patch.object(script, "build_event_checks", return_value=gate_checks),
            patch.object(script, "build_price_targets", return_value=[target]),
        ):
            with self.assertRaisesRegex(RuntimeError, "offline cache read reached"):
                script.prepare_cross_validation(
                    repository,
                    policy,
                    cache,
                    fetch_missing=False,
                )
        cache.get.assert_called_once_with(target)
        cache.fill_missing.assert_not_called()


class FrozenWikiPriceOnlyCollectorTest(unittest.TestCase):
    def _diagnostic(self):
        return {
            "validation_basis": script.REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_BASIS,
            "policy_spec_sha256": "1" * 64,
            "policy_registry_sha256": (
                script.TRUSTED_REVIEWED_SOURCE_ARCHIVE_PRICE_ONLY_SHA256
            ),
            "extract_sha256": script.WIKI_EXTRACT_SHA256,
            "provenance_sha256": script.WIKI_PROVENANCE_SHA256,
            "target_id": (
                "ed969b35974af909d34adab11ace79a964e9fc06d70e543d52ef573576cfd994"
            ),
            "security_id": "US:EODHD:dbd287da-b35f-5aaa-873c-76f941b4d93b",
            "symbol": "BBBY",
            "target_provider_symbol": "BBBY",
            "identity_bound_provider_symbol": "BBBY_old.US",
            "identity_active_from": "2015-01-01",
            "identity_active_to": "2023-05-02",
            "terminal_event_id": (
                "7d150e99cfe15587e4e9994dfaebde08942117f970a7d11ce94fd05b84bc85f5"
            ),
            "raw_price_source_sha256": "2" * 64,
            "identity_source_sha256": "3" * 64,
            "overlap_session_count": 650,
            "overlap_start": "2015-01-02",
            "overlap_end": "2018-03-07",
            "relation_sha256": "4" * 64,
            "triple_supertrend_field_differences": {
                column: 0
                for column in (
                    "TripleST1_Trend",
                    "TripleST2_Trend",
                    "TripleST3_Trend",
                    "TripleAllUp",
                    "TripleDownCount",
                    "TripleBuySignal",
                    "TripleSellSignal",
                )
            },
            "current_signal_sha256": "5" * 64,
            "substituted_signal_sha256": "5" * 64,
            "action_factor_status": "incomplete_not_rewritten",
            "wiki_dividends_missing_from_current": [],
            "current_dividends_missing_from_wiki": [],
            "raw_price_rewritten": False,
            "corporate_actions_rewritten": False,
            "adjustment_factors_rewritten": False,
            "generic_ticker_reuse_allowed": False,
            "yahoo_symbol_only_identity_reuse_allowed": False,
            "price_only_pass_must_not_imply_action_factor_pass": True,
            "limitation": "price only; action/factor coverage remains incomplete",
            "projection_sha256": "6" * 64,
        }

    def test_collector_promotes_only_explicit_price_only_exception_without_yahoo(self):
        diagnostic = self._diagnostic()
        target = script.PriceTarget(
            diagnostic["security_id"],
            "BBBY",
            ("test",),
            active_from="2015-01-01",
            active_to="2023-05-02",
            terminal_event_id=diagnostic["terminal_event_id"],
        )
        self.assertEqual(target.target_id, diagnostic["target_id"])
        prices = pd.DataFrame(
            [
                {
                    "security_id": diagnostic["security_id"],
                    "session": "2015-01-02",
                }
            ]
        )
        checks = script.build_price_checks(
            [target],
            {},
            prices,
            pd.DataFrame(columns=["security_id", "action_type", "effective_date"]),
            [],
            _policy(),
            source_archive_price_only_evidence={target.target_id: diagnostic},
        )
        self.assertEqual(len(checks), 1)
        item = checks[0]
        self.assertEqual(item["status"], "explicit_exception")
        self.assertTrue(item["price_only_arbitration_passed"])
        self.assertFalse(item["corporate_actions_validated"])
        self.assertFalse(item["adjustment_factors_validated"])
        self.assertFalse(item["generic_ticker_reuse_allowed"])
        self.assertEqual(item["source_sha256"], script.WIKI_EXTRACT_SHA256)
        self.assertEqual(item["provenance_sha256"], script.WIKI_PROVENANCE_SHA256)

        computed = _check_report_rows(
            {"events": [], "permanent_exceptions": [], "prices": checks}
        )
        self.assertEqual(computed["price_exception_count"], 1)
        self.assertEqual(computed["overlap_session_count"], 650)

    def test_price_only_exception_tamper_is_rejected_by_report_gate(self):
        diagnostic = self._diagnostic()
        target = script.PriceTarget(
            diagnostic["security_id"],
            "BBBY",
            ("test",),
            active_from="2015-01-01",
            active_to="2023-05-02",
            terminal_event_id=diagnostic["terminal_event_id"],
        )
        item = script.build_price_checks(
            [target],
            {},
            pd.DataFrame(
                [{"security_id": diagnostic["security_id"], "session": "2015-01-02"}]
            ),
            pd.DataFrame(columns=["security_id", "action_type", "effective_date"]),
            [],
            _policy(),
            source_archive_price_only_evidence={target.target_id: diagnostic},
        )[0]
        item["reviewed_source_archive_price_only_evidence"][
            "generic_ticker_reuse_allowed"
        ] = True
        with self.assertRaisesRegex(RuntimeError, "price-only exception is incomplete"):
            _check_report_rows(
                {"events": [], "permanent_exceptions": [], "prices": [item]}
            )


@unittest.skipUnless(
    PRODUCTION_YAHOO_CACHE.is_dir(),
    "local frozen Yahoo cache is required for exact SYMC/NLOK replay",
)
class SymcNlokFrozenYahooReplayTest(unittest.TestCase):
    def test_exact_no_data_wrappers_replay_offline(self):
        cache = RawYahooChartCache(PRODUCTION_YAHOO_CACHE)
        requests = [
            (symbol, period1, period2)
            for symbol, period1, period2, *_rest
            in SYMC_NLOK_NO_DATA_REQUESTS
        ]
        with patch(
            "supertrend_quant.market_store.yahoo_chart.urlopen",
            side_effect=AssertionError("network called during frozen replay"),
        ):
            responses = cache.fill_missing(requests)
        self.assertEqual(cache.http_attempts, 0)
        self.assertEqual(set(responses), set(requests))
        for (
            symbol,
            period1,
            period2,
            source_url,
            source_sha256,
            wrapper_sha256,
        ) in SYMC_NLOK_NO_DATA_REQUESTS:
            with self.subTest(symbol=symbol):
                response = responses[(symbol, period1, period2)]
                self.assertEqual(response.symbol, symbol)
                self.assertEqual(response.source_url, source_url)
                self.assertEqual(response.request_period1, period1)
                self.assertEqual(response.request_period2, period2)
                self.assertEqual(response.http_status, 404)
                self.assertEqual(response.source_hash, source_sha256)
                self.assertEqual(response.wrapper_hash, wrapper_sha256)
                evidence = parse_yahoo_chart_no_data_evidence(
                    response.content,
                    symbol,
                    http_status=response.http_status,
                    request_period1=period1,
                    request_period2=period2,
                )
                self.assertEqual(evidence.kind, "chart_not_found")


class YahooChartCacheSafetyTest(unittest.TestCase):
    def test_reviewed_cache_lookup_keeps_the_original_bounded_request(self):
        payload = _chart_json(
            _bars(pd.bdate_range("2024-01-02", periods=2)), symbol="AA"
        )
        period1, period2 = _request_periods("2024-01-02", "2024-01-31")
        class Response:
            status = 200
            headers = {"Content-Type": "application/json"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return payload

        with tempfile.TemporaryDirectory() as directory:
            cache = RawYahooChartCache(Path(directory), max_http_attempts=1)
            with patch(
                "supertrend_quant.market_store.yahoo_chart.urlopen",
                return_value=Response(),
            ):
                response = cache.fill_missing([("AA", period1, period2)])["AA", period1, period2]
            reviewed = cache.get_by_wrapper_hash("AA", response.wrapper_hash)

        self.assertIsNotNone(reviewed)
        self.assertEqual(reviewed.request_period1, period1)
        self.assertEqual(reviewed.request_period2, period2)
        self.assertEqual(reviewed.source_hash, script.sha256_bytes(payload))

    def test_archive_artifact_hash_is_the_exact_response_hash(self):
        payload = _chart_json(
            _bars(pd.bdate_range("2024-01-02", periods=2)), symbol="AA"
        )
        artifact = script.ArchiveArtifact(
            source="yahoo_chart_json",
            source_url=_source_url("AA"),
            retrieved_at="2026-07-18T00:00:00Z",
            content=payload,
            content_type="application/json",
            object_path="archives/example.json.gz",
        )
        self.assertEqual(artifact.source_hash, script.sha256_bytes(payload))

    def test_request_set_over_cap_fails_before_http(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = script.YahooChartCache(Path(directory), _policy())
            targets = [
                script.PriceTarget(
                    f"SEC-{index}",
                    f"S{index}",
                    ("test",),
                    request_start="2024-01-02",
                    request_end="2024-02-29",
                )
                for index in range(401)
            ]
            with self.assertRaisesRegex(RuntimeError, "before any HTTP call"):
                cache.fill_missing(targets)
            self.assertEqual(cache.http_attempts, 0)

    def test_same_symbol_windows_have_distinct_bounded_urls_and_cache_keys(self):
        first = _request_periods("2015-01-02", "2017-08-31")
        second = _request_periods("2019-06-03", "2026-07-17")
        cache = RawYahooChartCache(Path("unused"))
        first_url = cache.url("DD", period1=first[0], period2=first[1])
        second_url = cache.url("DD", period1=second[0], period2=second[1])

        self.assertNotEqual(first_url, second_url)
        self.assertNotEqual(
            cache.path("DD", period1=first[0], period2=first[1]),
            cache.path("DD", period1=second[0], period2=second[1]),
        )
        self.assertNotIn("range=max", first_url)
        self.assertIn(f"period1={first[0]}", first_url)
        self.assertIn(f"period2={first[1]}", first_url)

    def test_network_failure_is_attempted_once_without_retry(self):
        period1, period2 = _request_periods("2024-01-02", "2024-02-29")
        with tempfile.TemporaryDirectory() as directory:
            cache = RawYahooChartCache(Path(directory), max_http_attempts=1)
            with patch(
                "supertrend_quant.market_store.yahoo_chart.urlopen",
                side_effect=URLError("offline"),
            ) as opener:
                with self.assertRaisesRegex(RuntimeError, "single HTTP attempt failed"):
                    cache.fetch("AA", period1=period1, period2=period2)
                with self.assertRaisesRegex(RuntimeError, "attempt cap reached"):
                    cache.fetch("AA", period1=period1, period2=period2)
            opener.assert_called_once()
            self.assertEqual(cache.http_attempts, 1)

    def test_reusable_cache_preserves_exact_bytes_and_second_run_is_offline(self):
        payload = _chart_json(_bars(pd.bdate_range("2024-01-02", periods=2)), symbol="LILA")
        period1, period2 = _request_periods("2024-01-02", "2024-02-29")
        request = ("LILA", period1, period2)

        class Response:
            status = 200
            headers = {"Content-Type": "application/json"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return payload

        self.assertEqual(normalize_yahoo_symbol("BRK.B.US"), "BRK-B")
        with tempfile.TemporaryDirectory() as directory:
            first = RawYahooChartCache(Path(directory), max_http_attempts=1)
            with patch(
                "supertrend_quant.market_store.yahoo_chart.urlopen",
                return_value=Response(),
            ) as opener:
                fetched = first.fill_missing([("LILA.US", period1, period2)])
            self.assertEqual(first.http_attempts, 1)
            self.assertEqual(fetched[request].content, payload)
            self.assertEqual(fetched[request].source_hash, script.sha256_bytes(payload))
            self.assertEqual(len(fetched[request].wrapper_hash), 64)
            self.assertNotIn("crumb", fetched[request].source_url.lower())
            self.assertNotIn("token", fetched[request].source_url.lower())
            self.assertNotIn("range=max", fetched[request].source_url)
            self.assertEqual(fetched[request].request_period1, period1)
            self.assertEqual(fetched[request].request_period2, period2)
            opener.assert_called_once()

            second = RawYahooChartCache(Path(directory), max_http_attempts=1)
            with patch(
                "supertrend_quant.market_store.yahoo_chart.urlopen",
                side_effect=AssertionError("network called"),
            ):
                cached = second.fill_missing([("LILA.US", period1, period2)])
            self.assertEqual(second.http_attempts, 0)
            self.assertEqual(cached[request].content, payload)

    def test_cache_wrapper_and_content_tampering_are_rejected(self):
        payload = _chart_json(_bars(pd.bdate_range("2024-01-02", periods=2)), symbol="AA")
        period1, period2 = _request_periods("2024-01-02", "2024-02-29")

        class Response:
            status = 200
            headers = {"Content-Type": "application/json"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return payload

        with tempfile.TemporaryDirectory() as directory:
            cache = RawYahooChartCache(Path(directory), max_http_attempts=1)
            with patch(
                "supertrend_quant.market_store.yahoo_chart.urlopen",
                return_value=Response(),
            ):
                cache.fetch("AA", period1=period1, period2=period2)
            path = cache.path("AA", period1=period1, period2=period2)
            envelope = json.loads(gzip.decompress(path.read_bytes()))
            envelope["retrieved_at"] = "tampered"
            path.write_bytes(
                gzip.compress(
                    (json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n").encode(),
                    mtime=0,
                )
            )
            with self.assertRaisesRegex(RuntimeError, "wrapper hash mismatch"):
                cache.get("AA", period1=period1, period2=period2)

            with patch(
                "supertrend_quant.market_store.yahoo_chart.urlopen",
                return_value=Response(),
            ):
                fresh = RawYahooChartCache(Path(directory) / "fresh", max_http_attempts=1)
                fresh.fetch("AA", period1=period1, period2=period2)
            fresh_path = fresh.path("AA", period1=period1, period2=period2)
            envelope = json.loads(gzip.decompress(fresh_path.read_bytes()))
            envelope["content_base64"] = "e30="
            unhashed = dict(envelope)
            unhashed.pop("wrapper_sha256")
            envelope["wrapper_sha256"] = script.sha256_bytes(
                (json.dumps(unhashed, sort_keys=True, separators=(",", ":")) + "\n").encode()
            )
            fresh_path.write_bytes(
                gzip.compress(
                    (json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n").encode(),
                    mtime=0,
                )
            )
            with self.assertRaisesRegex(RuntimeError, "content hash mismatch"):
                fresh.get("AA", period1=period1, period2=period2)


if __name__ == "__main__":
    unittest.main()
