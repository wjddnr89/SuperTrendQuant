#!/usr/bin/env python3
"""Collect one post-transition SEC filing for the FI -> FISV review.

The existing 2025 Form 8-K remains the scheduling evidence for the exact
2025-11-11 market-open transition.  This collector is deliberately limited to
one later Form 10-Q URL whose cover page confirms that the resulting security
is registered on Nasdaq under FISV.  It never calls EODHD or R2.

The default mode is a read-only offline plan.  ``--fetch`` is the only network
entry point, permits exactly one HTTP attempt to the code-pinned SEC URL, and
caches the exact response bytes plus their SHA-256 for later offline review.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

from supertrend_quant.market_store.manifest import write_atomic


DEFAULT_CACHE_ROOT = Path("data/cache")
EVIDENCE_SUBDIR = Path("state/issuer_lifecycle/fisv_confirmation")
REPORT_FILENAME = "fisv_confirmation_evidence.json"
SCHEMA = "us_fisv_confirmation_evidence/v1"
MAX_HTTP_ATTEMPTS = 1
MAX_RESPONSE_BYTES = 8_000_000
TIMEOUT_SECONDS = 30

CONFIRMATION_SOURCE: dict[str, Any] = {
    "label": "fisv_nasdaq_post_transition_confirmation",
    "source_url": (
        "https://www.sec.gov/Archives/edgar/data/798354/"
        "000079835426000018/fisv-20260331.htm"
    ),
    "form": "10-Q",
    "period_end": "2026-03-31",
    "required_text_groups": (
        ("Fiserv, Inc.",),
        ("March 31, 2026",),
        ("Trading Symbol", "Trading Symbol(s)"),
        ("FISV",),
        ("The Nasdaq Stock Market LLC", "Nasdaq Stock Market LLC"),
    ),
}

Fetcher = Callable[[str, str], bytes]


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _normalized_text(content: bytes) -> str:
    decoded = content.decode("utf-8", errors="replace")
    decoded = html.unescape(re.sub(r"<[^>]+>", " ", decoded))
    return re.sub(r"\s+", " ", decoded).strip().casefold()


def _require_exact_sec_url(url: str) -> None:
    if url != CONFIRMATION_SOURCE["source_url"]:
        raise ValueError("FISV confirmation collector URL is not the code-pinned URL.")
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() != "www.sec.gov"
        or parsed.query
        or parsed.fragment
        or parsed.path
        != "/Archives/edgar/data/798354/000079835426000018/fisv-20260331.htm"
    ):
        raise ValueError("FISV confirmation collector URL envelope changed.")


def verify_content(content: bytes) -> None:
    if not content or len(content) > MAX_RESPONSE_BYTES:
        raise ValueError("FISV confirmation response size is outside the reviewed envelope.")
    text = _normalized_text(content)
    for alternatives in CONFIRMATION_SOURCE["required_text_groups"]:
        if not any(str(value).casefold() in text for value in alternatives):
            raise ValueError(
                "FISV confirmation lacks reviewed official term: "
                + " | ".join(str(value) for value in alternatives)
            )


def _validate_user_agent(value: str) -> str:
    user_agent = value.strip()
    if not user_agent or "@" not in user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT with a contact email is required for the one-shot fetch."
        )
    return user_agent


def _fetch_once(url: str, user_agent: str) -> bytes:
    _require_exact_sec_url(url)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": _validate_user_agent(user_agent),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        status = int(getattr(response, "status", response.getcode()))
        final_url = str(response.geturl())
        if status != 200:
            raise RuntimeError(f"FISV confirmation returned HTTP {status}.")
        if final_url != url:
            raise RuntimeError(
                "FISV confirmation request redirected outside the exact URL: "
                + final_url
            )
        content = response.read(MAX_RESPONSE_BYTES + 1)
    verify_content(content)
    return content


def _report_path(cache_root: Path) -> Path:
    return cache_root / EVIDENCE_SUBDIR / REPORT_FILENAME


def _payload_path(cache_root: Path, filename: str) -> Path:
    base = (cache_root / EVIDENCE_SUBDIR).resolve()
    path = (base / filename).resolve()
    if path == base or base not in path.parents:
        raise ValueError("FISV confirmation payload path escapes the evidence directory.")
    return path


def verify_cached_evidence(cache_root: Path) -> dict[str, Any] | None:
    report_path = _report_path(cache_root)
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Cached FISV confirmation report is unreadable.") from exc
    if set(report) != {
        "schema",
        "status",
        "evidence",
        "http_attempts_total",
        "eodhd_calls",
        "r2_accessed",
    }:
        raise ValueError("Cached FISV confirmation report fields are not exact.")
    if (
        report.get("schema") != SCHEMA
        or report.get("status") != "collected"
        or report.get("http_attempts_total") != 1
        or report.get("eodhd_calls") != 0
        or report.get("r2_accessed") is not False
    ):
        raise ValueError("Cached FISV confirmation report contract changed.")
    evidence = report.get("evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "label",
        "source_url",
        "source_hash",
        "size",
        "filename",
        "retrieved_at",
        "form",
        "period_end",
    }:
        raise ValueError("Cached FISV confirmation evidence fields are not exact.")
    if (
        evidence.get("label") != CONFIRMATION_SOURCE["label"]
        or evidence.get("source_url") != CONFIRMATION_SOURCE["source_url"]
        or evidence.get("form") != CONFIRMATION_SOURCE["form"]
        or evidence.get("period_end") != CONFIRMATION_SOURCE["period_end"]
    ):
        raise ValueError("Cached FISV confirmation identity changed.")
    _require_exact_sec_url(str(evidence["source_url"]))
    digest = str(evidence.get("source_hash") or "")
    filename = str(evidence.get("filename") or "")
    size = evidence.get("size")
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or filename != f"{digest}.html"
        or not isinstance(size, int)
        or size <= 0
        or size > MAX_RESPONSE_BYTES
        or not str(evidence.get("retrieved_at") or "").endswith("Z")
    ):
        raise ValueError("Cached FISV confirmation hash/size metadata is invalid.")
    path = _payload_path(cache_root, filename)
    if not path.is_file():
        raise FileNotFoundError(f"Cached FISV confirmation payload is missing: {path}.")
    content = path.read_bytes()
    if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Cached FISV confirmation hash/size verification failed.")
    verify_content(content)
    return {**report, "payload_path": str(path)}


def offline_plan(cache_root: Path) -> dict[str, Any]:
    cached = verify_cached_evidence(cache_root)
    if cached is not None:
        return {
            **cached,
            "status": "cache_verified",
            "mode": "offline_plan",
            "http_attempts_this_run": 0,
            "writes_performed": False,
            "network_accessed": False,
        }
    _require_exact_sec_url(str(CONFIRMATION_SOURCE["source_url"]))
    return {
        "schema": SCHEMA,
        "status": "ready_for_authorized_fetch",
        "mode": "offline_plan",
        "source_url": CONFIRMATION_SOURCE["source_url"],
        "source_role": "post_transition_exchange_and_ticker_confirmation",
        "max_http_attempts": MAX_HTTP_ATTEMPTS,
        "http_attempts_this_run": 0,
        "writes_performed": False,
        "network_accessed": False,
        "eodhd_calls": 0,
        "r2_accessed": False,
    }


def collect(
    cache_root: Path,
    *,
    user_agent: str,
    fetcher: Fetcher = _fetch_once,
) -> dict[str, Any]:
    cached = verify_cached_evidence(cache_root)
    if cached is not None:
        return {
            **cached,
            "status": "cache_verified",
            "mode": "fetch",
            "http_attempts_this_run": 0,
            "writes_performed": False,
            "network_accessed": False,
        }
    url = str(CONFIRMATION_SOURCE["source_url"])
    _require_exact_sec_url(url)
    content = fetcher(url, _validate_user_agent(user_agent))
    verify_content(content)
    digest = hashlib.sha256(content).hexdigest()
    filename = f"{digest}.html"
    path = _payload_path(cache_root, filename)
    report_path = _report_path(cache_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() != content:
        raise RuntimeError(f"Immutable FISV confirmation collision at {path}.")
    if not path.is_file():
        write_atomic(path, content)
    report = {
        "schema": SCHEMA,
        "status": "collected",
        "evidence": {
            "label": CONFIRMATION_SOURCE["label"],
            "source_url": url,
            "source_hash": digest,
            "size": len(content),
            "filename": filename,
            "retrieved_at": _now(),
            "form": CONFIRMATION_SOURCE["form"],
            "period_end": CONFIRMATION_SOURCE["period_end"],
        },
        "http_attempts_total": MAX_HTTP_ATTEMPTS,
        "eodhd_calls": 0,
        "r2_accessed": False,
    }
    write_atomic(
        report_path,
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )
    return {
        **report,
        "status": "collected",
        "mode": "fetch",
        "payload_path": str(path),
        "http_attempts_this_run": MAX_HTTP_ATTEMPTS,
        "writes_performed": True,
        "network_accessed": True,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or perform the one-URL FISV SEC confirmation fetch."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--fetch", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = (
        collect(
            args.cache_root,
            user_agent=os.getenv("SEC_USER_AGENT", ""),
        )
        if args.fetch
        else offline_plan(args.cache_root)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
