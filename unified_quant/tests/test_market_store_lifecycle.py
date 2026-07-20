from __future__ import annotations

from types import SimpleNamespace

import pytest
import pandas as pd

from supertrend_quant.market_store.lifecycle import (
    LifecycleCandidate,
    SecFiling,
    build_lifecycle_candidates,
    parse_sec_lifecycle_filing,
    resolve_new_security_id,
)


def _candidate(symbol: str, *, last_price_date: str = "2022-04-01") -> LifecycleCandidate:
    return LifecycleCandidate(
        security_id=f"US:TEST:{symbol}",
        symbol=symbol,
        name=f"{symbol} Company",
        exchange="US",
        last_price_date=last_price_date,
        active_to=last_price_date,
    )


def _filing(symbol: str, filing_date: str = "2022-04-01") -> SecFiling:
    return SecFiling(
        cik="1",
        accession_number="0000000000-22-000001",
        filing_date=filing_date,
        form="8-K",
        items=("2.01", "3.01"),
        display_name=f"{symbol} Company ({symbol})",
        score=20.0,
    )


def test_lifecycle_candidates_follow_successor_chain_without_self_authorizing_exceptions() -> None:
    securities = (
        ("INDEXED", "IDX", "2026-07-15", ""),
        ("MERGED", "MRG", "2020-01-02", "2020-01-02"),
        ("RENAMED", "RNM", "2021-02-03", "2021-02-03"),
        ("SPINOFF", "SPN", "2022-03-04", "2022-03-04"),
        ("UNRELATED", "UNR", "2019-04-05", "2019-04-05"),
    )
    frames = {
        "security_master": pd.DataFrame(
            [
                {
                    "security_id": security_id,
                    "primary_symbol": symbol,
                    "name": f"{symbol} Company",
                    "exchange": "NYSE",
                    "active_to": active_to,
                }
                for security_id, symbol, _session, active_to in securities
            ]
        ),
        "daily_price_raw": pd.DataFrame(
            [
                {"security_id": security_id, "session": session}
                for security_id, _symbol, session, _active_to in securities
            ]
        ),
        "index_constituent_anchors": pd.DataFrame(
            [{"security_id": "INDEXED"}]
        ),
        "index_membership_events": pd.DataFrame(
            columns=("security_id", "operation", "effective_date")
        ),
        "corporate_actions": pd.DataFrame(
            [
                {
                    "security_id": "INDEXED",
                    "action_type": "stock_merger",
                    "new_security_id": "MERGED",
                },
                {
                    "security_id": "MERGED",
                    "action_type": "ticker_change",
                    "new_security_id": "RENAMED",
                },
                {
                    "security_id": "RENAMED",
                    "action_type": "ticker_change",
                    "new_security_id": "INDEXED",
                },
                {
                    "security_id": "INDEXED",
                    "action_type": "spinoff",
                    "new_security_id": "SPINOFF",
                },
            ]
        ),
        # A reviewed exception for an old, disconnected identity must not seed
        # the candidate graph.  The builder intentionally never reads this frame.
        "lifecycle_resolutions": pd.DataFrame(
            [
                {
                    "security_id": "UNRELATED",
                    "resolution": "exception",
                }
            ]
        ),
    }

    class Repository:
        def read_frame(self, dataset, _version=None):
            return frames[dataset].copy()

    release = SimpleNamespace(
        completed_session="2026-07-15",
        dataset_versions={dataset: "v1" for dataset in frames},
    )

    candidates = build_lifecycle_candidates(Repository(), release=release)

    assert [candidate.security_id for candidate in candidates] == [
        "MERGED",
        "RENAMED",
        "SPINOFF",
    ]
    assert all(candidate.index_remove_dates == () for candidate in candidates)


@pytest.mark.parametrize(
    ("symbol", "successor", "text", "ratio", "cash"),
    (
        (
            "PBCT",
            "MTB",
            "On April 1, 2022, the Company completed the merger. Each share of "
            "People's United Common Stock, par value $0.50 per share, was converted "
            "into the right to receive 0.118 of a share of M&T Common Stock "
            "(the Exchange Ratio).",
            0.118,
            None,
        ),
        (
            "CAM",
            "SLB",
            "On April 1, 2022, the Company completed the merger. Each share of "
            "Company Common Stock, par value $0.01 per share, was converted into the "
            "right to receive (a) $14.44 in cash and (b) 0.716 shares of Schlumberger "
            "common stock. Each employee award was converted into 2.0 shares of "
            "common stock.",
            0.716,
            14.44,
        ),
        (
            "DFS",
            "COF",
            "On April 1, 2022, the Company completed the merger. Each share of common "
            "stock, par value $0.01 per share, was converted into the right to receive "
            "1.0192 shares (the Exchange Ratio) of common stock of Capital One.",
            1.0192,
            None,
        ),
        (
            "ANSS",
            "SNPS",
            "On April 1, 2022, the Company completed the merger. Each share of common "
            "stock was converted into the right to receive (i) 0.3399 (the Exchange "
            "Ratio) of a share of Synopsys common stock and (ii) $199.91 in cash.",
            0.3399,
            199.91,
        ),
        (
            "LLTC",
            "ADI",
            "On April 1, 2022, the Company completed the merger. Each outstanding "
            "share of Company common stock was automatically converted into the right "
            "to receive the following consideration: $46.00 in cash; and 0.2321 (the "
            "Exchange Ratio) shares of common stock of Analog Devices.",
            0.2321,
            46.0,
        ),
        (
            "ESRX",
            "CI",
            "On April 1, 2022, the merger was completed. Each outstanding "
            "share of Express Scripts common stock was converted into (1) 0.2434 "
            "of a share of New Cigna common stock and (2) the right to receive "
            "$48.75 in cash, without interest.",
            0.2434,
            48.75,
        ),
    ),
)
def test_stock_merger_terms_come_from_common_share_consideration(
    symbol: str,
    successor: str,
    text: str,
    ratio: float,
    cash: float | None,
) -> None:
    parsed = parse_sec_lifecycle_filing(
        text,
        candidate=_candidate(symbol),
        filing=_filing(symbol),
        preferred_symbols=(successor,),
    )

    assert parsed is not None
    assert parsed.action_type == "stock_merger"
    assert parsed.ratio == pytest.approx(ratio)
    assert parsed.cash_amount == cash
    assert parsed.new_symbol == successor
    assert parsed.confidence == "high"


def test_employee_award_ratio_is_not_a_common_share_exchange_ratio() -> None:
    text = (
        "On April 1, 2022, the Company completed the merger. Each outstanding stock "
        "option was converted into an award covering 2.0 shares of Buyer common "
        "stock. The common stock will no longer be listed."
    )
    parsed = parse_sec_lifecycle_filing(
        text,
        candidate=_candidate("OLD"),
        filing=_filing("OLD"),
        preferred_symbols=("NEW",),
    )

    assert parsed is not None
    assert parsed.action_type == "delisting"
    assert parsed.ratio is None


def test_par_value_is_not_treated_as_cash_consideration() -> None:
    text = (
        "On April 1, 2022, the Company completed the merger. Each share of common "
        "stock was converted into the right to receive 0.118 of a share of Buyer "
        "common stock, par value $0.50 per share."
    )
    parsed = parse_sec_lifecycle_filing(
        text,
        candidate=_candidate("OLD"),
        filing=_filing("OLD"),
        preferred_symbols=("NEW",),
    )

    assert parsed is not None
    assert parsed.action_type == "stock_merger"
    assert parsed.cash_amount is None


def test_explicit_wfm_completion_beats_closer_earliest_event_effective_time() -> None:
    text = (
        "Date of Report (Date of Earliest Event Reported): August 23, 2017. "
        "At the shareholder meeting, the Effective Time was approved on "
        "August 23, 2017. On August 28, 2017, Amazon.com completed its "
        "previously announced acquisition of Whole Foods Market. Merger Sub "
        "merged with and into Whole Foods Market on August 28, 2017. Each "
        "share was converted into the right to receive $42.00 in cash. "
        "NASDAQ suspended trading prior to market open on August 28, 2017."
    )

    parsed = parse_sec_lifecycle_filing(
        text,
        candidate=_candidate("WFM", last_price_date="2017-08-25"),
        filing=_filing("WFM", filing_date="2017-08-28"),
        preferred_symbols=(),
    )

    assert parsed is not None
    assert parsed.action_type == "cash_merger"
    assert parsed.effective_date == "2017-08-28"
    assert parsed.cash_amount == 42.0
    assert parsed.confidence == "high"


@pytest.mark.parametrize(
    ("symbol", "successor", "text", "ratio", "cash"),
    (
        (
            "AET",
            "CVS",
            "On April 1, 2022, the Company completed the merger. Each issued and "
            "outstanding share of AET common stock was converted into the right to "
            "receive $145.00 in cash and 0.8378 of a share of CVS common stock. "
            "Item 2.01 Completion of Acquisition.",
            0.8378,
            145.0,
        ),
        (
            "RAI",
            "BTI",
            "On April 1, 2022, the Company completed the merger. Each outstanding "
            "share of RAI common stock was converted into the right to receive "
            "0.5260 of a share of BTI common stock and $29.44 in cash. The buyer's "
            "4 percent preferred stock was not merger consideration.",
            0.526,
            29.44,
        ),
        (
            "FLIR",
            "TDY",
            "On April 1, 2022, the Company completed the merger. Each outstanding "
            "share of FLIR common stock was converted into the right to receive "
            "$28.00 in cash and 0.0718 shares of Teledyne common stock. Item 1.01 "
            "and Item 2.01 follow.",
            0.0718,
            28.0,
        ),
        (
            "CELG",
            "BMY",
            "On April 1, 2022, the Company completed the merger. Each outstanding "
            "share of CELG common stock was converted into the right to receive one "
            "(1) share of Bristol-Myers common stock, $50.00 in cash and one CVR.",
            1.0,
            50.0,
        ),
        (
            "ETFC",
            "MS",
            "On April 1, 2022, the Company completed the merger. Each outstanding "
            "share of ETFC common stock was converted into the right to receive "
            "1.0432 shares of Morgan Stanley common stock. Section 2 describes "
            "preferred stock with a liquidation preference of $1,000.",
            1.0432,
            None,
        ),
        (
            "CCE",
            "CCEP",
            "On April 1, 2022, the Company completed the merger. Each outstanding "
            "share of CCE common stock was converted into the right to receive one "
            "(1) share of CCEP common stock and $14.50 in cash.",
            1.0,
            14.5,
        ),
        (
            "KSU",
            "CP",
            "On April 1, 2022, the Company completed the merger. Each outstanding "
            "share of KSU common stock was converted into the right to receive "
            "2.884 shares of Canadian Pacific common stock and $90.00 in cash. "
            "Page 4 discusses the 4 percent preferred shares.",
            2.884,
            90.0,
        ),
        (
            "AGN",
            "ACT",
            "On April 1, 2022, the Company completed the merger. Each holder of a "
            "share of AGN common stock issued and outstanding immediately before "
            "the merger had the right to receive a combination of (1) 0.3683 of "
            "an Actavis ordinary share and (2) $129.22 in cash, without interest.",
            0.3683,
            129.22,
        ),
    ),
)
def test_ratio_parser_ignores_enumeration_awards_and_preferred_terms(
    symbol: str,
    successor: str,
    text: str,
    ratio: float,
    cash: float | None,
) -> None:
    parsed = parse_sec_lifecycle_filing(
        text,
        candidate=_candidate(symbol),
        filing=_filing(symbol),
        preferred_symbols=(successor,),
    )

    assert parsed is not None
    assert parsed.action_type == "stock_merger"
    assert parsed.ratio == pytest.approx(ratio)
    assert parsed.cash_amount == cash
    assert parsed.new_symbol == successor


def test_target_company_ratio_beats_buyers_one_for_one_reorganization() -> None:
    text = (
        "On April 1, 2022, the merger was completed. Each outstanding share of "
        "Cigna common stock was converted into one share of New Cigna common stock. "
        "Each outstanding share of ESRX Company common stock was converted into "
        "0.2434 of a share of New Cigna common stock and the right to receive "
        "$48.75 in cash."
    )
    parsed = parse_sec_lifecycle_filing(
        text,
        candidate=_candidate("ESRX"),
        filing=_filing("ESRX"),
        preferred_symbols=("CI",),
    )

    assert parsed is not None
    assert parsed.ratio == pytest.approx(0.2434)
    assert parsed.cash_amount == 48.75


def test_successor_resolution_prefers_historical_alias_over_reused_ticker() -> None:
    master = pd.DataFrame(
        [
            {
                "security_id": "ACTAVIS",
                "primary_symbol": "AGN",
                "provider_symbol": "AGN.US",
                "active_from": "2015-01-01",
                "active_to": "2020-05-08",
            },
            {
                "security_id": "ENACT",
                "primary_symbol": "ACT",
                "provider_symbol": "ACT.US",
                "active_from": "2021-09-16",
                "active_to": "",
            },
        ]
    )
    history = pd.DataFrame(
        [
            {
                "security_id": "ACTAVIS",
                "symbol": "ACT",
                "effective_from": "2015-01-01",
                "effective_to": "2015-06-14",
            },
            {
                "security_id": "ACTAVIS",
                "symbol": "AGN",
                "effective_from": "2015-06-15",
                "effective_to": "2020-05-08",
            },
            {
                "security_id": "ENACT",
                "symbol": "ACT",
                "effective_from": "2021-09-16",
                "effective_to": "",
            },
        ]
    )

    resolved = resolve_new_security_id(
        master,
        new_symbol="ACT",
        effective_date="2015-03-17",
        symbol_history=history,
    )

    assert resolved == "ACTAVIS"
