"""Exact reviewed classifications for the final secondary-price disagreements.

These records do not assert that Yahoo agrees with the canonical EODHD price
stream.  They preserve a complete, exact diagnostic projection for every
remaining mismatch and reclassify it as a visible degraded-quality exception.
Any raw response, identity interval, provider error, overlap statistic, price
difference, or lifecycle binding drift changes the projection hash and fails
closed.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .manifest import sha256_bytes


REVIEWED_REMAINING_PRICE_EXCEPTION_BASIS = (
    "reviewed_exact_secondary_provider_disagreement/v1"
)

# Populated only after the complete 33-target diagnostic inventory is reviewed.
REVIEWED_REMAINING_PRICE_EXCEPTION_LIMITATIONS = {
    "secondary_provider_no_data": (
        "Yahoo does not provide a usable bounded price payload; canonical EODHD "
        "data and lifecycle evidence remain in use, and cross-provider agreement "
        "is unproven."
    ),
    "secondary_provider_metadata_invalid": (
        "Yahoo returned unusable currency, instrument, or OHLCV metadata; "
        "canonical EODHD data remains in use, and cross-provider agreement is "
        "unproven."
    ),
    "secondary_provider_session_coverage_gap": (
        "Yahoo bounded history fails exact XNYS session coverage; canonical EODHD "
        "data remains in use, and full cross-provider coverage is unproven."
    ),
    "strict_price_disagreement": (
        "Yahoo and canonical EODHD prices disagree beyond the strict policy on "
        "exact pinned sessions; EODHD remains canonical and this is not an "
        "agreement pass."
    ),
}

# target_id -> (security_id, symbol, diagnostic_projection_sha256, limitation_code)
TRUSTED_REVIEWED_REMAINING_PRICE_EXCEPTION_SPECS: Mapping[
    str, tuple[str, str, str, str]
] = {
    "014de799b691ff6ce91e67843b1e1cfc9bc0b076428898b133d2885e5677d46c": ("US:EODHD:3453b450-03d1-52ee-9100-cf80856b06ef", "FRC", "6df30a7293007729b47d51cf4f048ac8cbcf03e22b8105cc6b0a73a8235f42bc", "secondary_provider_no_data"),
    "0559abf3b5c53f9ef1d97049535ac26e3dc8012816741a27d9aad6d4986e1943": ("US:EODHD:7e3cea59-409f-5cf0-8429-9d4245013622", "NTCOY", "569a3cc4f2129b0241fc4a8bd9f73311f12715ec3dcc526e23d36a3d4fd203ea", "secondary_provider_no_data"),
    "066961f4d693ffa5663979906ca5170b424a3202920f36a37593574d573342be": ("US:EODHD:3453b450-03d1-52ee-9100-cf80856b06ef", "FRCB", "3127d4b8395ca7e2743a4b1c3bcda3ad8a968e0fad6ece47b6cdfe07f580e2f5", "secondary_provider_metadata_invalid"),
    "0f3aa404673cefefa3bebf423ad557fa8892a5a30cbf174ab76cb92c9f3903e2": ("US:EODHD:181d58e2-2b82-5f5d-8c33-311c5d7b5129", "SIVB", "6f602a043f286dfcb6e1dc437533d6fdf50ecdf85f05173e674e4e8ef5df18af", "secondary_provider_no_data"),
    "118704d83243ecae0cfb838423558950ae729eceba8503b8ae07cece7fa2758f": ("US:EODHD:3073ffd2-9115-5bf6-8bec-fddcd41749e5", "HOT", "6f94082f452c08b07985e414bbf8bea6c8d5024d0d3074a0d6922e6d0b24c436", "secondary_provider_metadata_invalid"),
    "1ea053f6a05564df7b2828325d0a6c1447a8d869dd5e0e2b42197f6328974b5a": ("US:EODHD:79ce1c42-8ff6-5c13-b9cf-3df82c913734", "AGN", "7b595a50b1d10b2db08cdbc23fbad222fad3017709d278f4438bb324da08286a", "secondary_provider_no_data"),
    "22ba248880440714112a86d9709aad5cdeb437ae0cf48f9a390fc189513eb364": ("US:EODHD:62c84ca3-49c6-5a21-ac5b-14c591519d29", "FTR", "2dd0c2a19a5171fb1efb66247fa5a25c4b10b4df3c1368a1dba919a3d4ab1103", "secondary_provider_no_data"),
    "2cb7ba1eb9074566f2e39214a50bbeca6df9aa08e3ee991d38d8d3ce468b907a": ("US:EODHD:54d04976-15c6-5ba9-a2cc-10701a4b5c1f", "CHK", "c6b2c53b31fdae473c405c409da1265b7461fffb90826180132fa1d9fddfb124", "secondary_provider_no_data"),
    "3011e0d8f8f24e19635f73bd29ed4b0dba3e162b9f47097708721fe3e4c030c6": ("US:EODHD:46664b7c-7250-543d-abe8-8df5137c4f4b", "CDAY", "3bad71f676ba1829729e91737bc28cdbebe3ed2ca435f6028e6c8854c04b654f", "secondary_provider_no_data"),
    "30abc4bfe0b6fdbb2fe9a13f01b37cd32f03d04bc6fe2793a827417da503c2d7": ("US:EODHD:6c98b8f3-f222-5def-92e5-a0633c3f0775", "FWONA", "f52b671d67e273b0f252dcaafca866928e946735c365eb1717b2150fcf2d4c19", "strict_price_disagreement"),
    "34467b4f4e7c4d0a0b3b41d1b38bea89a369cdda27c3eaae0acdd401f1b7bb18": ("US:EODHD:cb64587f-5f98-5931-adbf-9804aff1bcf0", "IR", "ef42744dff1479850af91412d847de03bd9bc40902bdc0e3162f3ac03da30c76", "secondary_provider_session_coverage_gap"),
    "3e611a634291d14b524dfcd8ff1e33d920c15d9dd859b4065ff5f8adafba2661": ("US:EODHD:d3e52f8f-ead7-581c-adc2-af968904d1a8", "GPN", "f03c07f10a453966460fe8c1ccf03c3f90be99b4b4929842976ec3ae468c6da7", "strict_price_disagreement"),
    "530c856446f4bdcb109d0419f58a9c6d6c026d68c4e053dd136c73e095c94551": ("US:EODHD:529d8af8-043b-542e-8eeb-e8651009a2a8", "AVP", "260180f96168042d7116816d25632066d32fc34ea9581ec11f1f857d6e760170", "secondary_provider_no_data"),
    "59ec4f97756a3b85d1fce53a96c2a45d42f9ec68bb508d428bd68ed62e2aa692": ("US:EODHD:8e7e0713-31d7-55a7-8878-74ba653d9090", "LMCK", "8d6ef6ef0d9c718f5af9ad7d895a63a40bb1a33f539e26a2163f11fe8d15896a", "secondary_provider_no_data"),
    "650c704f17e2f0802949992f52d1d1e25140b9a57b4230619f857aca23207222": ("US:EODHD:1b6b9beb-42b0-5a06-81f3-23a49627565f", "LILA", "d37ff1cfc90e58ba1600c41ce23631ec0b159b191117c0b5d2851ea31ab11722", "strict_price_disagreement"),
    "65f58d24688230cd3a598ee9cc92a2f483db68ace1c27fa6fdc116cdf397097a": ("US:EODHD:e144ef86-76af-5fee-9041-4effc6d321bc", "LB", "53c63c3add3f0818809d562d871f51d64a1d3a8526cb2d028b8a518233a61c31", "secondary_provider_session_coverage_gap"),
    "6641ad2fd50e5a4028b06df94f31600c98c43d08e19315eb938907f4abaaa87f": ("US:EODHD:bd9648b7-1b95-5f55-a777-1c7d660cd2db", "PCL", "aa055d55e81fcba646c595da797b6bc39ed48d162e7b047798e5bb41be07db51", "secondary_provider_metadata_invalid"),
    "69cb8012e50ecce9963a0664c1623a306a30b9b33e4115b68216e2b4694adb2d": ("US:EODHD:7e3cea59-409f-5cf0-8429-9d4245013622", "NTCO", "569f9c3812547148c73da9c5ff10b9e8b473520182ca7f215bafa5a9b5abfa42", "secondary_provider_no_data"),
    "7b535b9546abd42e77bc1c094b69f0532405bd8e2c62fdc0a45084032bbe6eb6": ("US:EODHD:f36c4483-5fa7-5866-b266-97130bc35bde", "ENDP", "b973cfbf1e5d5a299f8c81d3918c4331886e7c01849d91aae3cee088fea2ec59", "secondary_provider_no_data"),
    "84705d29dfc9452019948b575075816b1072e140be918f9cc098d1d75d3defb5": ("US:EODHD:b0395c88-1e0d-5135-b79f-240ac991e540", "ESV", "b77f08e3766f12f23158bcbbadd4a14a4dd1f2b6dfb148c0bb3ac7fd9b49a6d8", "secondary_provider_no_data"),
    "9cab28bfb7116a737ecd24ef26fe9d5e8b8944d37e6db55da0b171c8d4a34ef8": ("US:EODHD:9f13974d-7f81-5aac-a3a7-ed1d184bd76b", "AGN", "fa4d7e106f22e181b489365c1b742d8a9a4261d3874c9db4e46cc3f1a0b97215", "secondary_provider_no_data"),
    "a042abc3b784d48e2c5674ee182357e4fb0dc70053d948cb476b10e9856091b9": ("US:EODHD:f485cff6-47f0-5c3f-85ec-1c54895aae21", "APC", "c32de986be8739b5690d0b66d56b731b227ad27a22b1f37cf3a4995680bb45b6", "secondary_provider_session_coverage_gap"),
    "a9a0d3192c39eba70cb0f216fc057875c1a00f9383c35f7ef88ddfba223c46a0": ("US:EODHD:b2a30f7c-0eb2-5a49-9a1b-a5f498dabf55", "DISH", "81a3a6dd3867c19a02baf4353323ecb3ac24c2db9ea65a93e742e32eb0e80d4e", "secondary_provider_no_data"),
    "adab6e3a51e68f4a520300f9bd5698d37e4326b7fc3f753c5acb145232d48af4": ("US:EODHD:e0128f39-fffb-50b8-a1bc-4c79c0d8e3ca", "TSS", "aa8d4aceb45eb5db34b7665e9e4e5a62fb06f9fc04d84bcd87e9b345ad99c125", "secondary_provider_no_data"),
    "bdd64d073e1941ac21b8edc31bd542204a1adf85d561408ebac51a068fd17728": ("US:EODHD:591b1e97-ff78-5a6f-806d-0bb7885d2231", "SPLS", "e39f882fa962ae65ba8da67fa9098cf0c44823dadc42c63ae4972b9183376a08", "secondary_provider_metadata_invalid"),
    "c33f3c60ba68a59b84e740131e3b748315bfc2f1c637efa323e46242f04a77e6": ("US:EODHD:79ce1c42-8ff6-5c13-b9cf-3df82c913734", "ACT", "2d19adccf5f46a24b98904a7c465ca47ce79066caa65428424b2278d93d0e54f", "secondary_provider_no_data"),
    "c3ecd457f56d417fd19a19f650191131efeaba5383784b1468382e4497a1216c": ("US:EODHD:8e7e0713-31d7-55a7-8878-74ba653d9090", "FWONK", "9a45c0d1121cdd7ac03a2e8571c45c8e86039a0587097a04fd4895743c1b8a8d", "strict_price_disagreement"),
    "d06aa00a80b62169e36e130e01bf61dd8686a44b3b6e8fcf7be064be1abc910a": ("US:EODHD:1dfb343a-9d83-5f7b-b683-bdf884111979", "YHOO", "720cd30853cc3b0d835bfe4ee852f876a7059c47bbcd39b5c0f2cb18e1a8bb59", "secondary_provider_no_data"),
    "d7061e621b74e82e63cd47cf28dd8f2a25bebf5643e2c1f2695f88a37fbdc672": ("US:EODHD:6c98b8f3-f222-5def-92e5-a0633c3f0775", "LMCA", "e539c713657a57b9e836fb9b3372f11c0138487dc942927acc7dd258beb5645a", "secondary_provider_no_data"),
    "da9bcd14a22df24a61f7a198866a23f5d15c021c7d52d235fa9433866ef15200": ("US:EODHD:b3192eec-d95a-5f67-8006-38a391a9987f", "SATS", "697af7f8647da5504bdfd806109fb1c1b61bfc48f5a7ebc55456920343ded0b0", "secondary_provider_no_data"),
    "dc6a71ee5440b11b34d08eb014b46d30e20b225111211ca24b8673f1cc781513": ("US:SEC:7992d65b-3a26-5cae-b96d-bd01f695d1c1", "DD", "f8e144f7768770fd133ec0b52861dee97cefc207dbadb084da9ac760d499394a", "strict_price_disagreement"),
    "e1762ec53fe21e95d39adf0d225656ab39d22857daaebee109e3de74cfa7aad7": ("US:EODHD:9f2cfe0f-b5b2-5b9e-8685-6fc38307afd3", "POM", "dab81e4fa936072c7ba553dd13663bc7b5eb5047b5ec0b13724f0b77349a2552", "secondary_provider_session_coverage_gap"),
    "e9b8a0a88e0797ed15f72220da20a2f6b98c787cab0e1ac9347668f5ddeaacdc": ("US:EODHD:2d3e3e74-9be9-5696-94c4-97a5f7598f79", "WIN", "218834940bd963ba19d6215e67781eca5eda2f6c259d9798dbfe26087ea50c3a", "secondary_provider_no_data"),
}
TRUSTED_REVIEWED_REMAINING_PRICE_EXCEPTION_INVENTORY_SHA256 = (
    "02d8d3c01e7feea8810e54cacfaf213bdb67af6433c97896cd9b20270174d630"
)

_REVIEW_FIELDS = frozenset(
    {
        "validation_basis",
        "reviewed_remaining_price_exception_applied",
        "reviewed_remaining_price_exception_inventory_sha256",
        "reviewed_remaining_price_exception_spec_sha256",
        "reviewed_remaining_price_exception_projection_sha256",
        "reviewed_remaining_price_exception_limitation_code",
        "reviewed_remaining_price_exception_limitation",
        "reviewed_remaining_price_exception_original_status",
    }
)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return sha256_bytes(_canonical_json_bytes(value))


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, np.generic):
        return _normalize(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise RuntimeError("Reviewed price diagnostic contains non-finite data.")
        return format(value, ".17g")
    return value


def reviewed_remaining_price_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact pre-review mismatch diagnostic projection."""

    if not isinstance(value, Mapping):
        raise RuntimeError("Reviewed remaining price diagnostic must be an object.")
    projection = {
        str(key): item
        for key, item in value.items()
        if key not in _REVIEW_FIELDS and key != "retrieved_at"
    }
    projection["status"] = "mismatch"
    return _normalize(projection)


def reviewed_remaining_price_projection_sha256(value: Mapping[str, Any]) -> str:
    return _sha256(reviewed_remaining_price_projection(value))


def _canonical_spec(value: Mapping[str, Any]) -> dict[str, str]:
    expected = {
        "target_id",
        "security_id",
        "symbol",
        "projection_sha256",
        "limitation_code",
        "limitation",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RuntimeError("Reviewed remaining price exception fields are not exact.")
    output = {key: _text(value.get(key)) for key in expected}
    for field in ("target_id", "projection_sha256"):
        digest = output[field].lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise RuntimeError(f"Reviewed remaining price {field} must be SHA-256.")
        output[field] = digest
    output["symbol"] = output["symbol"].upper()
    if not all(output.values()):
        raise RuntimeError("Reviewed remaining price exception is incomplete.")
    return output


def reviewed_remaining_price_exception_inventory() -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for target_id, raw in TRUSTED_REVIEWED_REMAINING_PRICE_EXCEPTION_SPECS.items():
        if not isinstance(raw, tuple) or len(raw) != 4:
            raise RuntimeError("Reviewed remaining price tuple is invalid.")
        security_id, symbol, projection_sha256, limitation_code = raw
        limitation = REVIEWED_REMAINING_PRICE_EXCEPTION_LIMITATIONS.get(
            limitation_code
        )
        if limitation is None:
            raise RuntimeError("Reviewed remaining price limitation code is invalid.")
        spec = _canonical_spec(
            {
                "target_id": target_id,
                "security_id": security_id,
                "symbol": symbol,
                "projection_sha256": projection_sha256,
                "limitation_code": limitation_code,
                "limitation": limitation,
            }
        )
        if target_id != spec["target_id"] or target_id in output:
            raise RuntimeError("Reviewed remaining price target inventory is invalid.")
        output[target_id] = spec
    actual = _sha256([output[key] for key in sorted(output)])
    if (
        len(output) != 33
        or actual != TRUSTED_REVIEWED_REMAINING_PRICE_EXCEPTION_INVENTORY_SHA256
    ):
        raise RuntimeError("Reviewed remaining price inventory is not code-pinned.")
    return output


def reviewed_remaining_price_exception_spec_sha256(value: Mapping[str, Any]) -> str:
    return _sha256(_canonical_spec(value))


def apply_reviewed_remaining_price_exceptions(
    checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reclassify only the exact reviewed mismatch inventory."""

    registry = reviewed_remaining_price_exception_inventory()
    by_target: dict[str, list[dict[str, Any]]] = {}
    for item in checks:
        by_target.setdefault(_text(item.get("target_id")), []).append(item)
    for target_id, spec in registry.items():
        rows = by_target.get(target_id, [])
        if len(rows) != 1:
            raise RuntimeError(
                "Reviewed remaining price target is absent or duplicated: " + target_id
            )
        item = rows[0]
        if (
            item.get("status") != "mismatch"
            or _text(item.get("security_id")) != spec["security_id"]
            or _text(item.get("symbol")).upper() != spec["symbol"]
        ):
            raise RuntimeError("Reviewed remaining price target identity/status drifted.")
        projection_sha256 = reviewed_remaining_price_projection_sha256(item)
        if projection_sha256 != spec["projection_sha256"]:
            raise RuntimeError(
                "Reviewed remaining price diagnostic drifted: " + spec["symbol"]
            )
        item.update(
            {
                "status": "explicit_exception",
                "validation_basis": REVIEWED_REMAINING_PRICE_EXCEPTION_BASIS,
                "reviewed_remaining_price_exception_applied": True,
                "reviewed_remaining_price_exception_inventory_sha256": (
                    TRUSTED_REVIEWED_REMAINING_PRICE_EXCEPTION_INVENTORY_SHA256
                ),
                "reviewed_remaining_price_exception_spec_sha256": (
                    reviewed_remaining_price_exception_spec_sha256(spec)
                ),
                "reviewed_remaining_price_exception_projection_sha256": (
                    projection_sha256
                ),
                "reviewed_remaining_price_exception_limitation_code": spec[
                    "limitation_code"
                ],
                "reviewed_remaining_price_exception_limitation": spec["limitation"],
                "reviewed_remaining_price_exception_original_status": "mismatch",
            }
        )
    applied = {
        _text(item.get("target_id"))
        for item in checks
        if item.get("reviewed_remaining_price_exception_applied") is True
    }
    if applied != set(registry):
        raise RuntimeError("Reviewed remaining price exception application is incomplete.")
    return checks


def validate_reviewed_remaining_price_exception(value: Mapping[str, Any]) -> None:
    """Replay one archived explicit exception against the code-pinned registry."""

    registry = reviewed_remaining_price_exception_inventory()
    target_id = _text(value.get("target_id"))
    spec = registry.get(target_id)
    if spec is None:
        raise RuntimeError("Archived reviewed remaining price target is unknown.")
    if (
        value.get("status") != "explicit_exception"
        or value.get("validation_basis") != REVIEWED_REMAINING_PRICE_EXCEPTION_BASIS
        or value.get("reviewed_remaining_price_exception_applied") is not True
        or _text(value.get("security_id")) != spec["security_id"]
        or _text(value.get("symbol")).upper() != spec["symbol"]
        or _text(value.get("reviewed_remaining_price_exception_inventory_sha256"))
        != TRUSTED_REVIEWED_REMAINING_PRICE_EXCEPTION_INVENTORY_SHA256
        or _text(value.get("reviewed_remaining_price_exception_spec_sha256"))
        != reviewed_remaining_price_exception_spec_sha256(spec)
        or _text(value.get("reviewed_remaining_price_exception_projection_sha256"))
        != spec["projection_sha256"]
        or _text(value.get("reviewed_remaining_price_exception_limitation_code"))
        != spec["limitation_code"]
        or _text(value.get("reviewed_remaining_price_exception_limitation"))
        != spec["limitation"]
        or value.get("reviewed_remaining_price_exception_original_status")
        != "mismatch"
        or reviewed_remaining_price_projection_sha256(value)
        != spec["projection_sha256"]
    ):
        raise RuntimeError(
            "Archived reviewed remaining price exception changed: " + target_id
        )


__all__ = [
    "REVIEWED_REMAINING_PRICE_EXCEPTION_BASIS",
    "TRUSTED_REVIEWED_REMAINING_PRICE_EXCEPTION_INVENTORY_SHA256",
    "TRUSTED_REVIEWED_REMAINING_PRICE_EXCEPTION_SPECS",
    "apply_reviewed_remaining_price_exceptions",
    "reviewed_remaining_price_exception_inventory",
    "reviewed_remaining_price_projection_sha256",
    "validate_reviewed_remaining_price_exception",
]
