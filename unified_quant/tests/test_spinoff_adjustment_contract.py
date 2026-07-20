from __future__ import annotations

import json

import pandas as pd
import pytest

from supertrend_quant.market_store.adjustments import (
    SPINOFF_PRICE_ADJUSTMENT_CONTRACT,
    build_adjustment_factors,
)


SEC_HASH = "a" * 64
BASIS_HASH = "b" * 64


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"security_id": "DD", "session": "2015-06-30", "close": 63.95},
            {"security_id": "DD", "session": "2015-07-01", "close": 61.43},
            {"security_id": "DD", "session": "2015-07-02", "close": 59.99},
        ]
    )


def _metadata(**updates: object) -> str:
    value: dict[str, object] = {
        "price_adjustment_contract": SPINOFF_PRICE_ADJUSTMENT_CONTRACT,
        "distribution_ratio": 0.2,
        "child_fair_market_value_per_share": 16.21,
        "distributed_value_per_parent_share": 3.242,
        "cost_basis_fraction": 0.05085,
        "parent_cost_basis_fraction": 0.94915,
        "parent_fair_market_value_per_share": 60.51,
        "terms_source_hash": SEC_HASH,
        "terms_source_url": "https://www.sec.gov/example.htm",
        "basis_source_hash": BASIS_HASH,
        "basis_source_url": "https://issuer.example/basis.pdf",
    }
    value.update(updates)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _actions(metadata: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "chemours-spin",
                "security_id": "DD",
                "action_type": "spinoff",
                "effective_date": "2015-07-01",
                "ex_date": "2015-07-01",
                "cash_amount": None,
                "ratio": 0.2,
                "metadata": metadata,
            }
        ]
    )


def test_exact_spinoff_contract_adjusts_total_return_only() -> None:
    factors = build_adjustment_factors(
        _prices(), _actions(_metadata()), source_version="candidate"
    )

    expected = (63.95 - 3.242) / 63.95
    assert list(factors["split_factor"]) == [1.0, 1.0, 1.0]
    assert factors.iloc[0]["total_return_factor"] == pytest.approx(expected)
    assert list(factors.iloc[1:]["total_return_factor"]) == [1.0, 1.0]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"distribution_ratio": 0.25}, "ratio disagrees"),
        ({"distributed_value_per_parent_share": 3.2}, "internally inconsistent"),
        ({"parent_cost_basis_fraction": 0.9}, "do not sum"),
        (
            {"cost_basis_fraction": 0.2, "parent_cost_basis_fraction": 0.8},
            "basis fraction disagrees",
        ),
        ({"terms_source_hash": "not-a-sha"}, "provenance is not exact"),
    ],
)
def test_exact_spinoff_contract_rejects_inconsistent_terms(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_adjustment_factors(
            _prices(), _actions(_metadata(**updates)), source_version="candidate"
        )


def test_legacy_basis_only_spinoff_does_not_guess_a_price_adjustment() -> None:
    legacy = json.dumps({"cost_basis_fraction": 0.05085})
    factors = build_adjustment_factors(
        _prices(), _actions(legacy), source_version="candidate"
    )

    assert list(factors["split_factor"]) == [1.0, 1.0, 1.0]
    assert list(factors["total_return_factor"]) == [1.0, 1.0, 1.0]


def test_spinoff_contract_is_not_taxed_like_a_cash_dividend() -> None:
    gross = build_adjustment_factors(
        _prices(), _actions(_metadata()), source_version="candidate"
    )
    taxed = build_adjustment_factors(
        _prices(),
        _actions(_metadata()),
        source_version="candidate",
        dividend_tax_rate=0.5,
    )

    assert list(gross["total_return_factor"]) == list(
        taxed["total_return_factor"]
    )
