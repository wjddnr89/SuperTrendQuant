#!/usr/bin/env python3
"""Collect hashed OpenDART lifecycle evidence for unresolved KR candidates."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from supertrend_quant.env import load_env
from supertrend_quant.market_store.kr_pipeline import KrCheckpointStore
from supertrend_quant.market_store.kr_providers import artifact_from_payload
from supertrend_quant.market_store.manifest import write_atomic


DART_API_BASE_URL = "https://opendart.fss.or.kr/api"
DART_LIFECYCLE_ENDPOINTS = (
    "cmpMgDecsn.json",
    "stkExtrDecsn.json",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        default="data/cache/markets/KR",
        help="KR local dataset repository root.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.15,
        help="Delay between OpenDART requests.",
    )
    args = parser.parse_args()
    load_env()
    api_key = os.getenv("DART_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DART_API_KEY is required.")

    root = Path(args.cache_root)
    candidate_path = root / "state" / "lifecycle" / "candidates.json"
    catalog_path = root / "state" / "bootstrap" / "identity_catalog.parquet"
    if not candidate_path.is_file():
        raise SystemExit(f"Lifecycle candidate checkpoint is missing: {candidate_path}")
    if not catalog_path.is_file():
        raise SystemExit(f"KR identity catalog checkpoint is missing: {catalog_path}")

    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidates = [
        row
        for row in candidate_payload.get("candidates", ())
        if not str(row.get("resolution") or "").strip()
    ]
    catalog = pd.read_parquet(catalog_path)
    dart_by_symbol = (
        catalog.loc[
            catalog["dart_corp_code"].fillna("").astype(str).str.strip().ne("")
        ]
        .assign(
            primary_symbol=lambda frame: frame["primary_symbol"].astype(str),
            dart_corp_code=lambda frame: frame["dart_corp_code"]
            .astype(str)
            .str.zfill(8),
        )
        .groupby("primary_symbol")["dart_corp_code"]
        .agg(lambda values: tuple(sorted(set(values))))
        .to_dict()
    )
    checkpoint = KrCheckpointStore(root / "state" / "lifecycle")
    session = requests.Session()
    rows: list[dict] = []
    for candidate in candidates:
        symbol = str(candidate["symbol"])
        corp_codes = dart_by_symbol.get(symbol, ())
        if len(corp_codes) != 1:
            rows.append(
                {
                    "symbol": symbol,
                    "security_id": str(candidate["security_id"]),
                    "endpoint": "",
                    "status": "identity_error",
                    "message": (
                        "Expected one DART corp code; found "
                        f"{len(corp_codes)}."
                    ),
                    "result_count": 0,
                    "results": [],
                    "source_url": "",
                    "source_hash": "",
                    "retrieved_at": "",
                }
            )
            continue
        corp_code = corp_codes[0]
        last_price = pd.Timestamp(candidate["last_price_date"])
        active_to = pd.to_datetime(
            candidate.get("active_to"), errors="coerce"
        )
        start = max(pd.Timestamp("2015-01-01"), last_price - pd.DateOffset(years=2))
        boundary = (
            pd.Timestamp(active_to)
            if pd.notna(active_to)
            else pd.Timestamp(candidate_payload["completed_session"])
        )
        end = boundary + pd.DateOffset(years=1)
        request_record = {
            "corp_code": corp_code,
            "bgn_de": start.strftime("%Y%m%d"),
            "end_de": end.strftime("%Y%m%d"),
        }
        for endpoint in DART_LIFECYCLE_ENDPOINTS:
            source_url = f"{DART_API_BASE_URL}/{endpoint}"
            response = _request_json(
                session,
                source_url,
                {
                    "crtfc_key": api_key,
                    **request_record,
                },
            )
            artifact = artifact_from_payload(
                f"opendart_{endpoint.removesuffix('.json')}",
                source_url,
                {
                    "request": {
                        "endpoint": endpoint,
                        **request_record,
                    },
                    "response": response,
                },
            )
            checkpoint.save_local_artifact(artifact, scope="opendart")
            results = response.get("list")
            if not isinstance(results, list):
                results = []
            rows.append(
                {
                    "symbol": symbol,
                    "security_id": str(candidate["security_id"]),
                    "dart_corp_code": corp_code,
                    "endpoint": endpoint,
                    "status": str(response.get("status") or ""),
                    "message": str(response.get("message") or ""),
                    "result_count": len(results),
                    "results": results,
                    "source_url": source_url,
                    "source_hash": artifact.source_hash,
                    "retrieved_at": artifact.retrieved_at,
                }
            )
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    output = {
        "schema_version": 1,
        "market": "KR",
        "completed_session": candidate_payload["completed_session"],
        "candidate_count": len(candidates),
        "request_count": len(rows),
        "successful_result_count": sum(
            int(row.get("result_count") or 0) for row in rows
        ),
        "rows": rows,
    }
    output_path = root / "state" / "lifecycle" / "opendart_summary.json"
    write_atomic(
        output_path,
        (
            json.dumps(
                output,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "candidate_count": output["candidate_count"],
                "request_count": output["request_count"],
                "successful_result_count": output[
                    "successful_result_count"
                ],
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _request_json(
    session: requests.Session,
    url: str,
    params: dict[str, str],
) -> dict:
    last_error = ""
    for attempt in range(3):
        try:
            response = session.get(url, params=params, timeout=45)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("OpenDART response is not an object.")
            status = str(payload.get("status") or "")
            if status not in {"000", "013"}:
                raise RuntimeError(
                    f"OpenDART status={status}: {payload.get('message')}"
                )
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = type(exc).__name__
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"OpenDART request failed after retries: {last_error}")


if __name__ == "__main__":
    raise SystemExit(main())
