#!/usr/bin/env python3
"""Cache the two official SEC documents needed by the Kraft dividend repair.

This collector is intentionally separate from the offline repair.  It performs
at most one HTTP request per pinned SEC URL, never calls EODHD or R2, validates
the reviewed terms before persisting exact bytes, and reuses a hash-verified
cache on later runs.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from supertrend_quant.market_store.manifest import write_atomic


DEFAULT_CACHE_ROOT = Path("data/cache")
USER_AGENT = "SuperTrendQuant personal research jh@example.com"
MAX_RESPONSE_BYTES = 8_000_000

SOURCES = (
    {
        "label": "kraft_special_dividend_declaration",
        "source_url": (
            "https://www.sec.gov/Archives/edgar/data/1545158/"
            "000119312515230632/d947291d425.htm"
        ),
        "required_text_groups": (
            ("June 22, 2015",),
            ("special cash dividend in the amount of $16.50 per share",),
            ("conditioned upon the closing of the proposed merger",),
            ("payable to Kraft shareholders of record immediately prior",),
        ),
    },
    {
        "label": "kraft_special_dividend_completion_payment",
        "source_url": (
            "https://www.sec.gov/Archives/edgar/data/1637459/"
            "000163745915000021/khc10q62815.htm"
        ),
        "required_text_groups": (
            ("consummated on July 2, 2015",),
            ("on a one-for-one basis",),
            ("Upon the completion of the 2015 Merger",),
            ("received a special cash dividend of $16.50 per share",),
        ),
    },
)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _normalized_text(content: bytes) -> str:
    decoded = content.decode("utf-8", errors="replace")
    decoded = html.unescape(re.sub(r"<[^>]+>", " ", decoded))
    return re.sub(r"\s+", " ", decoded).strip().casefold()


def _verify_terms(content: bytes, source: dict[str, Any]) -> None:
    if not content or len(content) > MAX_RESPONSE_BYTES:
        raise ValueError(
            f"{source['label']} response size is outside the reviewed envelope."
        )
    text = _normalized_text(content)
    for alternatives in source["required_text_groups"]:
        if not any(value.casefold() in text for value in alternatives):
            raise ValueError(
                f"{source['label']} lacks reviewed official term: "
                + " | ".join(alternatives)
            )


def _fetch(source: dict[str, Any]) -> bytes:
    request = urllib.request.Request(
        source["source_url"],
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(
                f"{source['label']} returned HTTP {response.status}."
            )
        content = response.read(MAX_RESPONSE_BYTES + 1)
    _verify_terms(content, source)
    return content


def collect(cache_root: Path) -> dict[str, Any]:
    evidence_dir = cache_root / "state/issuer_lifecycle"
    report_path = evidence_dir / "kraft_special_dividend_evidence.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = report.get("evidence", [])
        if len(rows) != len(SOURCES):
            raise ValueError("Cached Kraft evidence report has an unexpected inventory.")
        for source, row in zip(SOURCES, rows, strict=True):
            if row.get("source_url") != source["source_url"]:
                raise ValueError("Cached Kraft evidence URL changed.")
            path = evidence_dir / str(row.get("filename") or "")
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != row.get("source_hash") or len(content) != row.get("size"):
                raise ValueError("Cached Kraft evidence hash/size verification failed.")
            _verify_terms(content, source)
        return {**report, "status": "cache_verified", "http_attempts_this_run": 0}

    evidence_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in SOURCES:
        content = _fetch(source)
        digest = hashlib.sha256(content).hexdigest()
        filename = f"{digest}.html"
        path = evidence_dir / filename
        if path.is_file() and path.read_bytes() != content:
            raise RuntimeError(f"Immutable evidence collision at {path}.")
        if not path.is_file():
            write_atomic(path, content)
        rows.append(
            {
                "label": source["label"],
                "source_url": source["source_url"],
                "source_hash": digest,
                "size": len(content),
                "filename": filename,
                "retrieved_at": _now(),
            }
        )
    report = {
        "schema": "us_kraft_special_dividend_evidence/v1",
        "status": "collected",
        "evidence": rows,
        "http_attempts_this_run": len(SOURCES),
        "eodhd_calls": 0,
        "r2_accessed": False,
    }
    write_atomic(
        report_path,
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cache exact official SEC evidence for the Kraft dividend repair."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(collect(args.cache_root), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
