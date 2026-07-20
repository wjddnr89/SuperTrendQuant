from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_us_gpn_price_arbitration.py"
)
SPEC = importlib.util.spec_from_file_location("audit_us_gpn_price_arbitration", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def test_field_arbitration_is_fail_closed() -> None:
    decision = script.arbitrate_field(
        session="2019-08-19",
        field="low",
        internal=158.54,
        yahoo=154.0,
        supporters={"eikon": 158.54},
    )
    assert decision["decision"] == "retain_eodhd_internal"
    assert decision["raw_price_repair_required"] is False

    with pytest.raises(RuntimeError, match="independent support"):
        script.arbitrate_field(
            session="2019-08-19",
            field="low",
            internal=158.54,
            yahoo=154.0,
            supporters={},
        )
    with pytest.raises(RuntimeError, match="do not agree"):
        script.arbitrate_field(
            session="2019-08-19",
            field="low",
            internal=158.54,
            yahoo=154.0,
            supporters={"eikon": 158.53},
        )


def test_signal_diff_reports_exact_sessions() -> None:
    import pandas as pd

    index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    baseline = pd.DataFrame(
        {
            "TripleST1_Trend": [1, 1, 1],
            "TripleST2_Trend": [1, 1, 1],
            "TripleST3_Trend": [1, 1, 1],
            "TripleAllUp": [True, True, True],
            "TripleDownCount": [0, 0, 0],
            "TripleBuySignal": [True, False, False],
            "TripleSellSignal": [False, False, False],
        },
        index=index,
    )
    alternate = baseline.copy()
    alternate.loc[index[1], "TripleST3_Trend"] = -1
    alternate.loc[index[1], "TripleAllUp"] = False
    alternate.loc[index[1], "TripleDownCount"] = 1
    diff = script._signal_diff(baseline, alternate)
    assert diff["TripleST3_Trend"] == {"count": 1, "sessions": ["2020-01-03"]}
    assert diff["TripleAllUp"]["count"] == 1
    assert diff["TripleBuySignal"]["count"] == 0
    assert diff["three_bar_confirmed_exit_state"]["count"] == 0


_INTEGRATION_PATHS = (
    script.DEFAULT_RELEASE,
    script.DEFAULT_WIKI_ZIP,
    script.DEFAULT_EIKON_CSV,
    script.DEFAULT_EIKON_README,
    script.DEFAULT_EIKON_TREE,
)


@pytest.mark.skipif(
    not all(path.is_file() for path in _INTEGRATION_PATHS),
    reason="Exact local GPN arbitration artifacts are not installed.",
)
def test_current_release_gpn_arbitration_replays_exactly() -> None:
    report = script.build_report()
    assert report["status"] == "passed"
    assert report["decision"] == "retain_current_eodhd_prices"
    assert report["raw_price_repair_required"] is False
    assert report["action_factor_diagnosis"]["classification"] == (
        "not_a_missing_adjustment_or_action"
    )
    assert report["field_decisions"][0]["independent_support"] == {
        "frozen_quandl_wiki_raw": 109.99,
        "eikon_split_normalized": 109.99,
    }
    assert report["field_decisions"][1]["independent_support"] == {
        "eikon_raw": 158.54
    }
    results = report["strategy_sensitivity"]["results"]
    for mode in ("raw", "total_return_adjusted"):
        assert results[mode]["yahoo_2015_high_only"]["TripleST3_Trend"]["count"] == 9
        assert results[mode]["yahoo_2015_high_only"]["TripleBuySignal"]["count"] == 2
        assert all(
            item["count"] == 0
            for item in results[mode]["yahoo_2019_low_only"].values()
        )
        assert all(
            item["count"] == 0
            for item in results[mode]["accepted_eikon_fields"].values()
        )
    assert report["external_request_accounting"]["total_external_requests"] == 3
    assert report["external_request_accounting"]["eodhd_calls"] == 0
    assert len(report["body_sha256"]) == 64
    assert all(character in "0123456789abcdef" for character in report["body_sha256"])
