from __future__ import annotations

import pandas as pd
import pytest

from supertrend_quant.market_store.operational_validation import (
    TRUSTED_OPERATIONAL_NBL_TERMINAL_STATE,
)
from supertrend_quant.universe import _index_security_member


EXPECTED = TRUSTED_OPERATIONAL_NBL_TERMINAL_STATE


def _history(*, effective_from: str = "2015-01-01") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "security_id": EXPECTED["security_id"],
                "symbol": EXPECTED["symbol"],
                "exchange": EXPECTED["identity_exchange"],
                "effective_from": effective_from,
                "effective_to": EXPECTED["last_real_session"],
            }
        ]
    )


def _master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "security_id": EXPECTED["security_id"],
                "name": "Noble Energy Inc",
                "exchange": EXPECTED["identity_exchange"],
                "asset_type": "STOCK",
            }
        ]
    )


def _events(**updates: str) -> pd.DataFrame:
    row = {
        "event_id": EXPECTED["next_remove_event_id"],
        "index_id": EXPECTED["index_id"],
        "effective_date": EXPECTED["next_remove_effective_date"],
        "operation": "REMOVE",
        "security_id": EXPECTED["security_id"],
        "source": EXPECTED["next_remove_source"],
        "source_hash": EXPECTED["next_remove_source_hash"],
    }
    row.update(updates)
    return pd.DataFrame([row])


def _resolve(
    *,
    profiles: tuple[str, ...] = ("sp500",),
    events: pd.DataFrame | None = None,
    fingerprints: tuple[str, ...] = (EXPECTED["fingerprint"],),
    history: pd.DataFrame | None = None,
):
    return _index_security_member(
        EXPECTED["security_id"],
        pd.Timestamp(EXPECTED["replay_date"]),
        profiles,
        _history() if history is None else history,
        _master(),
        _events() if events is None else events,
        fingerprints,
    )


def test_exact_reviewed_gap_uses_final_expired_alias_as_display_identity() -> None:
    member = _resolve()

    assert member.symbol == "NBL"
    assert member.security_id == EXPECTED["security_id"]
    assert member.profiles == ("sp500",)


@pytest.mark.parametrize(
    ("events", "profiles", "fingerprints"),
    (
        (_events(source_hash="f" * 64), ("sp500",), (EXPECTED["fingerprint"],)),
        (_events(event_id="different-remove"), ("sp500",), (EXPECTED["fingerprint"],)),
        (_events(source="different-source"), ("sp500",), (EXPECTED["fingerprint"],)),
        (_events(effective_date="2020-10-13"), ("sp500",), (EXPECTED["fingerprint"],)),
        (_events(), ("nasdaq100",), (EXPECTED["fingerprint"],)),
        (_events(), ("sp500", "nasdaq100"), (EXPECTED["fingerprint"],)),
        (_events(), ("sp500",), ()),
    ),
)
def test_unreviewed_or_mutated_gap_remains_fail_closed(
    events: pd.DataFrame,
    profiles: tuple[str, ...],
    fingerprints: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="No active symbol"):
        _resolve(
            events=events,
            profiles=profiles,
            fingerprints=fingerprints,
        )


def test_reviewed_gap_never_selects_a_future_alias() -> None:
    with pytest.raises(ValueError, match="No historical symbol"):
        _resolve(history=_history(effective_from="2020-10-08"))
