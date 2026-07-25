"""Exact constants for the reviewed SYMC -> NLOK identity canonicalization.

This module intentionally contains no I/O.  The repair command, lifecycle
finalizer integration and cross-provider validator can share one finite set of
identity/event pins without importing one another.  Expanding this inventory
requires a code review; there is no symbol or date fallback.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


OLD_SECURITY_ID = "US:EODHD:e92a676b-43f3-5d6b-9b7a-5d314fd6135f"
CANONICAL_SECURITY_ID = "US:EODHD:e9eea478-61d8-5762-9f5b-fbdfd69a02a3"
OLD_SYMBOL = "SYMC"
CANONICAL_SYMBOL = "NLOK"
OLD_SYMBOL_FROM = "2015-01-01"
OLD_SYMBOL_TO = "2019-11-01"
CANONICAL_SYMBOL_FROM = "2019-11-04"
CANONICAL_SYMBOL_TO = "2022-11-07"
TRANSITION_DATE = "2019-11-04"

OLD_EVENT_ID = "1b19b589542dfaf2e0e07c11188c59beab3db1b9e1aaab1b96570cc54d49a1cc"
CANONICAL_EVENT_ID = (
    "fc556f24050c3205150b7934f431b72d6348ab5fbfad3e85bfbb149c7b9781bd"
)
OLD_CANDIDATE_ID = (
    "021de8091979121c3d68fc49b4d13682499e6ef3384a6f2f8b670774ec0c0a88"
)
OFFICIAL_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/849399/"
    "000110465919059239/0001104659-19-059239.txt"
)
OFFICIAL_SOURCE_HASH = (
    "87a584813a438f76e5cee9ae800678771bc2df5ed6f3e50461273c1849026e18"
)

SP500_SWAP_REMOVE_EVENT_ID = (
    "bf239cd42cbe4375570653d830c953c28f902243c6416015db8a416bacf1a92c"
)
SP500_SWAP_ADD_EVENT_ID = (
    "d211aff53be0340f3197da6b034a0e7db5e8e5faafd423214bcec59f3a8fd902"
)

# These are the canonical target identities after the repair.  They are not a
# permission to reuse arbitrary Yahoo responses; response/cache hashes remain
# separate policy pins.
SYMC_PRICE_TARGET_ID = (
    "76cfddc97b878414119dfd9db08e356216cffc4ddc2839188451df534e11296f"
)
NLOK_PRICE_TARGET_ID = (
    "9648613a55f30697b2d3bb6893a3526b7d582169fc2dca21b6e2e4c9e481e1b6"
)
GEN_PRICE_TARGET_ID = (
    "088a66b2d0065fea58f5c99be2836df32174d12891ad2bf4e58e309cc97c8fb7"
)
NLOK_TO_GEN_EVENT_ID = (
    "d82975bc819ca47d10c7b2e2ca963422629980682933a4ee13b355fe564e6344"
)
NLOK_TO_GEN_OFFICIAL_HASH = (
    "a4732aaa030033aebda1d508bed1742e237694dc97fdb1a71f9af02f20d95d83"
)
GEN_SECURITY_ID = "US:EODHD:cb0b8e57-3e09-542c-adf8-fe2c98d97b55"


def _canonical_json_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def reviewed_nonterminal_extraction() -> dict[str, Any]:
    """Return the exact twelve-field reviewed action extraction."""

    return {
        "event_id": CANONICAL_EVENT_ID,
        "security_id": CANONICAL_SECURITY_ID,
        "action_type": "ticker_change",
        "effective_date": TRANSITION_DATE,
        "new_security_id": CANONICAL_SECURITY_ID,
        "new_symbol": CANONICAL_SYMBOL,
        "ratio": None,
        "cash_amount": None,
        "currency": "USD",
        "source_kind": "official_crosscheck",
        "source_url": OFFICIAL_SOURCE_URL,
        "source_hash": OFFICIAL_SOURCE_HASH,
    }


REVIEWED_NONTERMINAL_EXTRACTION_SHA256 = (
    "2f2cf8effa9cc40fdac0f51edad9f3857af1892ba232b05f6132f8228749715e"
)


def reviewed_same_sid_no_data_spec() -> dict[str, str]:
    """Return the exact cross-validation handoff for the closed SYMC interval."""

    return {
        "event_id": CANONICAL_EVENT_ID,
        "source_target_id": SYMC_PRICE_TARGET_ID,
        "successor_target_id": NLOK_PRICE_TARGET_ID,
        "security_id": CANONICAL_SECURITY_ID,
        "old_symbol": OLD_SYMBOL,
        "successor_symbol": CANONICAL_SYMBOL,
        "old_active_from": OLD_SYMBOL_FROM,
        "old_active_to": OLD_SYMBOL_TO,
        "successor_active_from": CANONICAL_SYMBOL_FROM,
        "effective_date": TRANSITION_DATE,
        "official_source_hash": OFFICIAL_SOURCE_HASH,
        "reviewed_extraction_sha256": REVIEWED_NONTERMINAL_EXTRACTION_SHA256,
    }


def exact_nonterminal_binding_inputs(
    target: Mapping[str, Any],
    event: Mapping[str, Any],
    extraction: Mapping[str, Any],
) -> bool:
    """Fail-closed structural verifier used by the integration tests/draft.

    The production cross-validation module additionally verifies provider raw
    response/cache hashes and the finite NLOK -> GEN price chain.  This helper
    deliberately does not approve a price result by itself.
    """

    expected_extraction = reviewed_nonterminal_extraction()
    if dict(extraction) != expected_extraction:
        return False
    if _canonical_json_sha256(expected_extraction) != REVIEWED_NONTERMINAL_EXTRACTION_SHA256:
        return False
    spec = reviewed_same_sid_no_data_spec()
    target_exact = (
        str(target.get("target_id", "")).strip().lower() == SYMC_PRICE_TARGET_ID
        and str(target.get("security_id", "")).strip() == CANONICAL_SECURITY_ID
        and str(target.get("provider_symbol") or target.get("symbol") or "").strip().upper()
        == OLD_SYMBOL
        and str(target.get("identity_active_from") or target.get("active_from") or "").strip()
        == OLD_SYMBOL_FROM
        and str(target.get("identity_active_to") or target.get("active_to") or "").strip()
        == OLD_SYMBOL_TO
        and str(target.get("terminal_event_id", "")).strip().lower()
        == CANONICAL_EVENT_ID
        and str(target.get("successor_security_id", "")).strip()
        == CANONICAL_SECURITY_ID
    )
    event_exact = (
        str(event.get("event_id", "")).strip().lower() == CANONICAL_EVENT_ID
        and str(event.get("security_id", "")).strip() == CANONICAL_SECURITY_ID
        and str(event.get("action_type", "")).strip().lower() == "ticker_change"
        and str(event.get("effective_date", ""))[:10] == TRANSITION_DATE
        and str(event.get("new_security_id", "")).strip() == CANONICAL_SECURITY_ID
        and str(event.get("new_symbol", "")).strip().upper() == CANONICAL_SYMBOL
        and str(event.get("evidence_sha256", "")).strip().lower()
        == OFFICIAL_SOURCE_HASH
        and event.get("status") == "passed"
        and event.get("reviewed_extraction_match") is True
        and not str(event.get("candidate_id", "")).strip()
    )
    return bool(spec and target_exact and event_exact)
