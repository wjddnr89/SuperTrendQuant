from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
import re


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, order=True)
class ReviewedProvenanceArchiveId:
    """One code-reviewed non-content ``source_archive`` primary key."""

    archive_id: str
    source: str
    source_url: str
    source_hash: str


# These include the complete provenance-qualified IDs in the reviewed
# 2026-07-15 release plus the one code-reviewed NTCOY empty-splits ID staged for
# its next transactional repair.  Five distinguish identical ``[]`` payloads
# returned by different endpoints.  The remaining ten are retained as exact
# legacy/reviewed IDs; no formula-derived ID outside this inventory is trusted
# automatically.
_REVIEWED_PROVENANCE_ARCHIVE_IDS = (
    ReviewedProvenanceArchiveId(
        archive_id="3958dc0304e1449a9fd3e33d538877eddd70053c897b61e0fb6666555e05967c",
        source="sec_rule_provision_notice",
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/876661/"
            "000087666120000056/ruleprovisionnotice.htm"
        ),
        source_hash="58d199861b620211b63c846e3184baf1ff7982adb124e085c5f726e2fd06af59",
    ),
    ReviewedProvenanceArchiveId(
        archive_id="5550528f1a63b94d35e6880c9c046756a64390f6675a65e2bf728dc560166474",
        source="ovintiv_issuer_reorganization",
        source_url=(
            "https://investor.ovintiv.com/2020-01-24-Encana-Completes-"
            "Reorganization-and-Establishes-Corporate-Domicile-in-the-U-S"
        ),
        source_hash="cb6cdb670b3a30d38f0529d242f4ea470052c04204e3101537627f7df3955bef",
    ),
    ReviewedProvenanceArchiveId(
        archive_id="57dd1d48a3580b65a68633138740f0dc261654bfd0fea4b25509dcf29ca7397b",
        source="qvc_issuer_stock_cost_basis",
        source_url="https://investors.qvcgrp.com/investors/stock-cost-basis",
        source_hash="55829c9064eee534b6f79027648172494a507f8b9be16e9598dc57cdd58c165b",
    ),
    ReviewedProvenanceArchiveId(
        archive_id="5de7d46d9fd1d7a1f2674a41826f4592d6786a59bb0e9891143aadc6974588c1",
        source="frcb_reviewed_ohlcv_envelope_correction",
        source_url=(
            "https://eodhd.com/api/eod/FRCB.US?"
            "from=2023-05-03&to=2026-07-15"
        ),
        source_hash="53eed10c5d6a7ccc262215b7848d30efa606a1621d2e793ca21b6002f8a5c298",
    ),
    ReviewedProvenanceArchiveId(
        archive_id="62b1df178614e3b8c5a9bc0f4cac8438946cc8c297c985ecd7be0e6ea14bc423",
        source="eodhd_div",
        source_url=(
            "https://eodhd.com/api/div/FRCB.US?"
            "from=2023-05-03&to=2026-07-15"
        ),
        source_hash="4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    ),
    ReviewedProvenanceArchiveId(
        archive_id="6a09ccaafcdf8ad57177fd1be2146ce912c84c4269cdc11ce736c7b4faad4461",
        source="eodhd_splits",
        source_url=(
            "https://eodhd.com/api/splits/NTCOY.US?"
            "from=2024-02-12&to=2024-09-03"
        ),
        source_hash="4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    ),
    ReviewedProvenanceArchiveId(
        archive_id="84811457cb719ddcf2756efd41922aace51c682a0a9abc52a997e1cada86fdaf",
        source="eodhd_ovv_div",
        source_url=(
            "https://eodhd.com/api/div/OVV.US?"
            "from=2020-01-27&to=2026-07-15"
        ),
        source_hash="63d125e117f9eeb8dcfb65833216553b46010f91abec2992ccc5e28c290f7fa6",
    ),
    ReviewedProvenanceArchiveId(
        archive_id="89d477dee0ed285cb326e4737b42fe4b685e875ac1bac4019c2df22bea9f3c8c",
        source="eodhd_ovv_eod",
        source_url=(
            "https://eodhd.com/api/eod/OVV.US?"
            "from=2020-01-27&to=2026-07-15"
        ),
        source_hash="2911e9b1eb3e59f3649f1a7ccef3b3a62b6b2667ed910aca8e335001afceafca",
    ),
    ReviewedProvenanceArchiveId(
        archive_id="8e3667ac5f9a407a63bca8e9ac134f11523530051111571d5a47ba9e2e1a78d1",
        source="eodhd_eod",
        source_url=(
            "https://eodhd.com/api/eod/FRCB.US?"
            "from=2023-05-03&to=2026-07-15"
        ),
        source_hash="3c96be232fb5f567e77a94fe315b67aa61e520d1611842620c04680fb5df6ab3",
    ),
    ReviewedProvenanceArchiveId(
        archive_id="954f03350a7b7f7342286c80c058b171e2776ac9b4dadca8de842008debf0e4e",
        source="eodhd_qvcaq_splits",
        source_url=(
            "https://eodhd.com/api/splits/QVCAQ.US?"
            "from=2026-04-24&to=2026-07-15"
        ),
        source_hash="4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    ),
    ReviewedProvenanceArchiveId(
        archive_id="985e539ff612e98c26f81b0dd4ac05387cfcaa91191de5eb4544a24e7dcfd22d",
        source="eodhd_qvcaq_div",
        source_url=(
            "https://eodhd.com/api/div/QVCAQ.US?"
            "from=2026-04-24&to=2026-07-15"
        ),
        source_hash="4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    ),
    ReviewedProvenanceArchiveId(
        archive_id="b40ae5d1ee136ea2c15dbdcfeb1a443ea28286916aaedb6d3a4c7d1960b303de",
        source="eodhd_ovv_splits",
        source_url=(
            "https://eodhd.com/api/splits/OVV.US?"
            "from=2020-01-27&to=2026-07-15"
        ),
        source_hash="195def5749f8d07f7311576b9470a2cf2c22a8b866bb2e263763311e828793b5",
    ),
    ReviewedProvenanceArchiveId(
        archive_id="c38253725941d3e78dec01244016d2769d048e5d544c739869fc434eb2bbad29",
        source="eodhd_splits",
        source_url=(
            "https://eodhd.com/api/splits/FRCB.US?"
            "from=2023-05-03&to=2026-07-15"
        ),
        source_hash="4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    ),
    ReviewedProvenanceArchiveId(
        archive_id="c568a6ac21ddc05d3c5821c228b94b7bd7e52a602a96b1cfb2f5f08ee24af658",
        source="occ_reviewed_memo_extraction",
        source_url="https://infomemo.theocc.com/infomemos?number=52352",
        source_hash="377bcc0663eb9666b9f639edaf541b0ec729fd4b84ac876e345baaa9bf413668",
    ),
    ReviewedProvenanceArchiveId(
        archive_id="cdb868a6b94fd620ac01a66f5bd44ca56ecb357a3f9e3d47851a6898d388453d",
        source="eodhd_qvcaq_eod",
        source_url=(
            "https://eodhd.com/api/eod/QVCAQ.US?"
            "from=2026-04-24&to=2026-07-15"
        ),
        source_hash="66a03be49bab3e158b6133fb2e49897008e90acbd4629ff11c812d6ee46f76aa",
    ),
)

TRUSTED_PROVENANCE_ARCHIVE_ID_INVENTORY_SHA256 = (
    "bf548fc10a3640ea93e52ecaa5434b2daa157382f908dddb6b789a2e5b93be1a"
)


def _canonical_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or value != value.lower()
        or not _SHA256_RE.fullmatch(value)
    ):
        raise ValueError(f"source_archive {field} must be one canonical lowercase SHA-256.")
    return value


def _canonical_provenance_text(
    value: object,
    *,
    field: str,
    allow_empty: bool,
) -> str:
    if allow_empty and (
        value is None
        or (isinstance(value, float) and math.isnan(value))
    ):
        return ""
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(
            f"source_archive {field} must be canonical text without outer whitespace."
        )
    if not allow_empty and not value:
        raise ValueError(f"source_archive {field} must not be empty.")
    return value


def _qualified_formula_candidates(row: ReviewedProvenanceArchiveId) -> frozenset[str]:
    values = {
        hashlib.sha256(
            f"{row.source}|{row.source_hash}".encode("utf-8")
        ).hexdigest()
    }
    if row.source_url:
        values.add(
            hashlib.sha256(
                f"{row.source}|{row.source_url}|{row.source_hash}".encode("utf-8")
            ).hexdigest()
        )
    return frozenset(values)


@lru_cache(maxsize=1)
def reviewed_provenance_archive_ids() -> tuple[ReviewedProvenanceArchiveId, ...]:
    """Return the complete code-pinned non-content archive-ID inventory."""

    rows = tuple(sorted(_REVIEWED_PROVENANCE_ARCHIVE_IDS))
    payload = json.dumps(
        [asdict(row) for row in rows],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != (
        TRUSTED_PROVENANCE_ARCHIVE_ID_INVENTORY_SHA256
    ):
        raise RuntimeError(
            "Reviewed provenance-qualified source_archive inventory is not code-pinned."
        )
    archive_ids: set[str] = set()
    provenance: set[tuple[str, str, str]] = set()
    for row in rows:
        _canonical_sha256(row.archive_id, field="archive_id")
        _canonical_sha256(row.source_hash, field="source_hash")
        _canonical_provenance_text(
            row.source, field="source", allow_empty=False
        )
        _canonical_provenance_text(
            row.source_url, field="source_url", allow_empty=True
        )
        if row.archive_id == row.source_hash:
            raise RuntimeError(
                "Reviewed provenance-qualified archive ID duplicates its content hash."
            )
        if row.archive_id not in _qualified_formula_candidates(row):
            raise RuntimeError(
                "Reviewed source_archive ID is not provenance-qualified: "
                + row.archive_id
            )
        key = (row.source, row.source_url, row.source_hash)
        if row.archive_id in archive_ids or key in provenance:
            raise RuntimeError(
                "Reviewed provenance-qualified source_archive inventory is duplicated."
            )
        archive_ids.add(row.archive_id)
        provenance.add(key)
    return rows


def source_archive_id_candidates(
    *,
    source: str,
    source_url: str,
    source_hash: str,
) -> frozenset[str]:
    """Return only the content ID and exact code-reviewed qualified ID, if any."""

    digest = _canonical_sha256(source_hash, field="source_hash")
    source_value = _canonical_provenance_text(
        source, field="source", allow_empty=False
    )
    url_value = _canonical_provenance_text(
        source_url, field="source_url", allow_empty=True
    )
    candidates = {digest}
    candidates.update(
        row.archive_id
        for row in reviewed_provenance_archive_ids()
        if (
            row.source == source_value
            and row.source_url == url_value
            and row.source_hash == digest
        )
    )
    return frozenset(candidates)


def validate_source_archive_id(
    archive_id: str,
    *,
    source: str,
    source_url: str,
    source_hash: str,
) -> str:
    """Validate one canonical content ID or exact code-reviewed qualified ID."""

    value = _canonical_sha256(archive_id, field="archive_id")
    candidates = source_archive_id_candidates(
        source=source,
        source_url=source_url,
        source_hash=source_hash,
    )
    if value not in candidates:
        raise ValueError(
            "source_archive archive_id is neither its content hash nor one exact "
            "code-reviewed provenance-qualified ID."
        )
    return value
