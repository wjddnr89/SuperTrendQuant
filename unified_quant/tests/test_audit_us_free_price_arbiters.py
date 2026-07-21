from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_us_free_price_arbiters.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_us_free_price_arbiters", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def test_target_inventory_caps_and_signal_profiles_are_exact() -> None:
    assert script.SYMBOLS == ("APC", "HOT", "IR", "LB", "PCL", "POM", "SPLS")
    assert tuple(target.symbol for target in script.TARGETS) == script.SYMBOLS
    assert script.MAX_STOOQ_HTTP_ATTEMPTS == 7
    assert script.MAX_BORIS_HTTP_ATTEMPTS == 7
    assert script.MAX_TOTAL_HTTP_ATTEMPTS == 14
    assert set(script.EXPECTED_BASELINE_SIGNAL_SHA256) == set(script.SYMBOLS)
    assert set(script.EXPECTED_WIKI_ADJUSTED_DIFF_COUNTS) == set(script.SYMBOLS)
    assert set(script.EXPECTED_BORIS_ADJUSTED_DIFF_COUNTS) == {"APC", "IR", "LB"}
    assert all(
        len(counts) == len(script.SIGNAL_COLUMNS)
        for counts in script.EXPECTED_WIKI_ADJUSTED_DIFF_COUNTS.values()
    )


def test_boris_http_error_body_is_cached_after_exactly_one_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b'{"code":404,"message":"Dataset not found"}'
    headers = Message()
    headers["Content-Type"] = "application/json"
    calls = 0

    def fake_urlopen(request: object, timeout: float) -> object:
        nonlocal calls
        calls += 1
        raise HTTPError(
            url=str(getattr(request, "full_url")),
            code=404,
            msg="Not Found",
            hdrs=headers,
            fp=io.BytesIO(body),
        )

    monkeypatch.setattr(script, "urlopen", fake_urlopen)
    cache = script.BorisPriceAuditCache(
        tmp_path,
        max_http_attempts=1,
        permit_initial_unpinned_capture=True,
    )
    response = cache.fetch("PCL")

    assert calls == 1
    assert cache.http_attempts == 1
    assert response.http_status == 404
    assert response.content_type == "application/json"
    assert response.content == body
    assert response.source_hash == hashlib.sha256(body).hexdigest()
    assert cache.path("PCL").is_file()

    replay = script.BorisPriceAuditCache(tmp_path, max_http_attempts=1).get("PCL")
    assert replay is not None
    assert replay.content == body
    assert calls == 1

    with pytest.raises(RuntimeError, match="attempt cap reached"):
        cache.fetch("PCL")
    assert calls == 1


_STOOQ = script.StooqHistoricalCache(
    script.DEFAULT_STOOQ_CACHE,
    max_http_attempts=script.MAX_STOOQ_HTTP_ATTEMPTS,
)
_BORIS = script.BorisPriceAuditCache(
    script.DEFAULT_BORIS_CACHE,
    max_http_attempts=script.MAX_BORIS_HTTP_ATTEMPTS,
)
_YAHOO = script.YahooChartCache(script.DEFAULT_YAHOO_CACHE)
_INTEGRATION_PATHS = (
    script.DEFAULT_RELEASE,
    script.DEFAULT_CROSSVALIDATION_REPORT,
    script.DEFAULT_WIKI_ZIP,
    script.DEFAULT_ATTEMPT_LEDGER,
    *(_STOOQ.path(symbol) for symbol in script.SYMBOLS),
    *(_BORIS.path(symbol) for symbol in script.SYMBOLS if symbol != "HOT"),
    *(
        _YAHOO.path(
            target.symbol,
            period1=target.yahoo_period1,
            period2=target.yahoo_period2,
        )
        for target in script.TARGETS
    ),
)


@pytest.mark.skipif(
    not all(path.is_file() for path in _INTEGRATION_PATHS),
    reason="Exact local free-price arbitration artifacts are not installed.",
)
def test_current_release_free_price_arbitration_replays_fail_closed() -> None:
    result = script.run_audit()

    assert result["summary"] == {
        "target_count": 7,
        "stooq_valid_price_targets": 0,
        "boris_valid_price_targets": 3,
        "boris_strict_wiki_adjusted_stability_targets": ["APC"],
        "promoted_crossvalidation_passes": 0,
        "remaining_fail_closed": 7,
        "candidate_eodhd_raw_close_anomaly_needing_independent_confirmation": ["APC"],
    }
    assert result["controls"]["network_http_attempts_for_acquisition"] == 14
    assert result["controls"]["network_retries"] == 0
    assert "do not reconstruct transport retry history" in result["controls"][
        "acquisition_transport_count_evidence_limit"
    ]
    assert result["controls"]["cached_raw_responses"] == 13
    assert result["controls"]["hot_boris_404_uncached_failures"] == 1
    assert result["controls"]["audit_run_http_attempts"] == 0
    assert result["controls"]["eodhd_api_calls"] == 0
    assert result["controls"]["r2_calls"] == 0
    assert result["controls"]["release_apply_calls"] == 0
    assert result["controls"]["dataset_mutations"] == 0
    assert result["controls"]["generic_exceptions_added"] == 0

    by_symbol = {item["symbol"]: item for item in result["targets"]}
    assert set(by_symbol) == set(script.SYMBOLS)
    for symbol, item in by_symbol.items():
        assert item["providers"]["eodhd"][
            "parquet_exactly_reproduces_retained_archived_raw"
        ] is True
        assert item["providers"]["yahoo"]["identity_accepted"] is False
        assert item["providers"]["stooq"]["status"] == "rejected_html_challenge"
        assert item["disposition"]["status"] == (
            "fail_closed_no_crossvalidation_pass"
        )
        assert item["disposition"]["generic_exception_allowed"] is False
        assert item["disposition"]["release_repair_allowed"] is False
        differences = item["triple_supertrend"]["wiki_raw_substitution"][
            "total_return_adjusted"
        ]
        assert script._signal_difference_counts(differences) == (
            script.EXPECTED_WIKI_ADJUSTED_DIFF_COUNTS[symbol]
        )

    assert by_symbol["HOT"]["providers"]["boris"]["status"] == (
        "rejected_404_uncached_no_retry"
    )
    assert by_symbol["APC"]["providers"]["boris"][
        "strict_long_scale_return_stability_passed"
    ] is True
    assert by_symbol["APC"]["eodhd_wiki_raw_relation"][
        "maximum_close_disagreement"
    ]["date"] == "2015-11-10"

    ledger = json.loads(
        script.DEFAULT_ATTEMPT_LEDGER.read_text(encoding="utf-8")
    )
    hot_failures = [
        item
        for item in ledger["failures"]
        if item.get("provider") == "boris" and item.get("symbol") == "HOT"
    ]
    assert len(hot_failures) == 1
    hot_failure = hot_failures[0]
    assert hot_failure["cache_written"] is False
    assert hot_failure["raw_bytes_available"] is False
    assert hot_failure["http_status"] == 404
    assert hot_failure["content_type"] == "application/json"
    assert hot_failure["source_url"] == script.BORIS_URL_TEMPLATE.format(
        symbol="hot"
    )
    assert "Retrying this URL is forbidden" in hot_failure["note"]

    rendered = (
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    assert hashlib.sha256(rendered).hexdigest() == (
        "2db2a0dce3dde3e096f7686c69c15bd7048322cc8087510440fae689f1416bc7"
    )
