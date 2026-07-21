#!/usr/bin/env python3
"""Read-only, exact-source arbitration of the two GPN Yahoo mismatches.

The audit binds the current immutable release to its archived EODHD bytes,
the exact Yahoo cache envelope, a frozen Quandl WIKI extract and a
commit-pinned Eikon CSV.  It does not fetch, repair, publish, or update any
release pointer.  The Eikon and WIKI inputs are evidence only: their licensing
does not permit this project to redistribute the underlying files.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from supertrend_quant.indicators import add_triple_supertrend
from supertrend_quant.market_store.adjustments import apply_adjustment_factors
from supertrend_quant.market_store.repository import LocalDatasetRepository
from supertrend_quant.market_store.yahoo_chart import (
    YahooChartCache,
    parse_yahoo_chart_json,
)


DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_RELEASE = Path(
    "data/cache/releases/20260715-20260718T230255094849Z.json"
)
DEFAULT_YAHOO_CACHE = Path("data/cache/state/us_cross_validation/yahoo_chart")
DEFAULT_WIKI_ZIP = Path("/tmp/marketneutral-quandl-wiki-prices.zip")
DEFAULT_EIKON_CSV = Path("/tmp/gpn-eikon-9a09c265.csv")
DEFAULT_EIKON_README = Path("/tmp/alpha_readme.md")
DEFAULT_EIKON_TREE = Path("/tmp/alpha_tree.json")

SECURITY_ID = "US:EODHD:d3e52f8f-ead7-581c-adc2-af968904d1a8"
TARGET_ID = "3e611a634291d14b524dfcd8ff1e33d920c15d9dd859b4065ff5f8adafba2661"
SYMBOL = "GPN"
SPLIT_DATE = "2015-11-03"
SPLIT_EVENT_ID = "70e76757e18bf60f0a851a1dc2e31fc56fa733254e4138717af3380d4222b0be"
EODHD_EOD_SHA256 = "39235d1f822263c250a69557ff9d9cd8310a0b7487ad68cc9a5dfec39fb2a46c"
EODHD_SPLIT_SHA256 = "15e91028d4409088c66b6524691ad84c71630d7a714ddd54d46f31e1c26291fc"
EODHD_DIVIDEND_SHA256 = "6a03086487f6f12059d3d31209e002bfa54ad0a0920254ed978fb292941dce1f"
YAHOO_SOURCE_SHA256 = "be071582774ed528eff679d4e5d5630f191ef3f0519635ebc5627e8789ecda11"
YAHOO_WRAPPER_SHA256 = "fce7693dcd1bad37d8fcc1f8d6825851651831eba90df0b5ac7e6dc69170ff98"
YAHOO_PERIOD1 = 1_420_070_400
YAHOO_PERIOD2 = 1_784_160_000
YAHOO_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/GPN?"
    "period1=1420070400&period2=1784160000&events=history&"
    "includeAdjustedClose=true&interval=1d"
)
EIKON_URL = (
    "https://raw.githubusercontent.com/cuicanyeah/Alpha-Generation/"
    "9a09c2658e7bc3bf9fb437548cc0e4471c9b5fdb/AlphaEvolve/raw_data/"
    "eikon_data/price_data_nyse/GPN.csv"
)
EIKON_TREE_SHA = "9a09c2658e7bc3bf9fb437548cc0e4471c9b5fdb"
EIKON_BLOB_SHA = "c347a2850467651e7e191c48a11c3fd06113bde9"
EIKON_BLOB_PATH = "AlphaEvolve/raw_data/eikon_data/price_data_nyse/GPN.csv"
WIKI_MEMBER = "WIKI_PRICES.csv"

TRIPLE_SETTINGS = ((10, 1.0), (11, 2.0), (12, 3.0))
SIGNAL_COLUMNS = (
    "TripleST1_Trend",
    "TripleST2_Trend",
    "TripleST3_Trend",
    "TripleAllUp",
    "TripleDownCount",
    "TripleBuySignal",
    "TripleSellSignal",
)


@dataclass(frozen=True)
class EvidencePins:
    release_version: str = "20260715-20260718T230255094849Z"
    daily_price_version: str = (
        "early-terminal-history-2026-07-15-566e79bcc7ac4e268c4cc304e14b700e-"
        "daily_price_raw"
    )
    action_version: str = (
        "early-terminal-history-2026-07-15-566e79bcc7ac4e268c4cc304e14b700e-"
        "corporate_actions"
    )
    factor_version: str = (
        "early-terminal-history-2026-07-15-566e79bcc7ac4e268c4cc304e14b700e-"
        "adjustment_factors"
    )
    source_archive_version: str = (
        "wiki-price-arbitration-20260715-301b7adc38334f65a4012a095993dce9-"
        "source_archive"
    )
    wiki_zip_sha256: str = (
        "36c667bbecf42c43e5b9e8e4e5d9a1268522705bc40a4bac9671d62c1a20cbae"
    )
    wiki_zip_size: int = 463_184_323
    wiki_member_size: int = 1_797_003_576
    wiki_member_crc32: int = 0x946874CE
    wiki_extract_sha256: str = (
        "9b60c9c5bdb6b8de302828807b42103441cb8adf8d73e1c3f3d74d27be8fc839"
    )
    wiki_extract_size: int = 552_613
    wiki_rows: int = 4_325
    eikon_sha256: str = (
        "5004f8b8c7c76eafde90bb0002669a460f70de8967bce7462e6287d92825f90d"
    )
    eikon_size: int = 126_043
    eikon_rows: int = 2_524
    eikon_first_date: str = "2013-01-02"
    eikon_last_date: str = "2023-01-10"
    eikon_readme_sha256: str = (
        "1fae87bffadd50184b54b0afdff41ca74ef259fc9cddb0f5136ef3b4d3354a60"
    )
    eikon_readme_size: int = 4_197
    eikon_tree_sha256: str = (
        "83a3cd52a064adfd8fa9271a56a787de6348d52f5889e328311aa0e72cca51fb"
    )
    eikon_tree_size: int = 5_073_068


DEFAULT_PINS = EvidencePins()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _number(value: Any) -> float:
    result = float(value)
    _require(math.isfinite(result), "Price evidence contains a non-finite number.")
    return result


def _date(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    _require(not pd.isna(parsed), "Price evidence contains an invalid date.")
    return pd.Timestamp(parsed).date().isoformat()


def _safe_archived_payload(
    repository: LocalDatasetRepository,
    archive: pd.DataFrame,
    *,
    source_hash: str,
    dataset: str,
    source_url: str,
) -> bytes:
    rows = archive.loc[archive["source_hash"].astype(str).eq(source_hash)]
    _require(len(rows) == 1, f"Expected one archived {dataset} object for GPN.")
    row = rows.iloc[0]
    _require(str(row["dataset"]) == dataset, "Archived GPN dataset changed.")
    _require(str(row["source_url"]) == source_url, "Archived GPN source URL changed.")
    root = repository.root.resolve()
    path = (root / str(row["object_path"])).resolve()
    _require(root in path.parents and path.is_file(), "Archived GPN payload is missing.")
    try:
        payload = gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise RuntimeError("Archived GPN payload is not valid gzip.") from exc
    _require(_sha256_bytes(payload) == source_hash, "Archived GPN payload hash changed.")
    return payload


def _project_frame(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    rows: list[dict[str, str]] = []
    for row in frame.loc[:, list(columns)].to_dict(orient="records"):
        rows.append(
            {
                key: _date(value) if key == "session" else format(_number(value), ".17g")
                for key, value in row.items()
            }
        )
    return _sha256_bytes(_canonical_json_bytes(rows))


def _load_wiki(path: Path, pins: EvidencePins) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require(path.is_file(), "Frozen WIKI ZIP is missing.")
    _require(path.stat().st_size == pins.wiki_zip_size, "Frozen WIKI ZIP size changed.")
    _require(_sha256_file(path) == pins.wiki_zip_sha256, "Frozen WIKI ZIP hash changed.")
    lines: list[bytes] = []
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        _require(len(infos) == 1 and infos[0].filename == WIKI_MEMBER, "WIKI member changed.")
        info = infos[0]
        _require(info.file_size == pins.wiki_member_size, "WIKI member size changed.")
        _require(info.CRC == pins.wiki_member_crc32, "WIKI member CRC changed.")
        with archive.open(info) as handle:
            header = handle.readline()
            _require(header.startswith(b"ticker,date,open,high,low,close,volume,"), "WIKI header changed.")
            lines.append(header)
            for line in handle:
                if line.startswith(b"GPN,"):
                    lines.append(line)
    extract = b"".join(lines)
    _require(len(lines) - 1 == pins.wiki_rows, "WIKI GPN row inventory changed.")
    _require(len(extract) == pins.wiki_extract_size, "WIKI GPN extract size changed.")
    _require(_sha256_bytes(extract) == pins.wiki_extract_sha256, "WIKI GPN extract hash changed.")
    frame = pd.read_csv(io.BytesIO(extract))
    frame["session"] = pd.to_datetime(frame.pop("date"), errors="coerce").dt.normalize()
    _require(not frame["session"].isna().any(), "WIKI GPN dates are invalid.")
    return frame, {
        "download_sha256": pins.wiki_zip_sha256,
        "download_size": pins.wiki_zip_size,
        "member": WIKI_MEMBER,
        "member_size": pins.wiki_member_size,
        "member_crc32": f"0x{pins.wiki_member_crc32:08x}",
        "gpn_extract_sha256": pins.wiki_extract_sha256,
        "gpn_extract_size": pins.wiki_extract_size,
        "gpn_rows": pins.wiki_rows,
        "license": "Unknown",
        "redistribution_allowed": False,
    }


def _load_eikon(
    csv_path: Path,
    readme_path: Path,
    tree_path: Path,
    pins: EvidencePins,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    for path, size, digest, label in (
        (csv_path, pins.eikon_size, pins.eikon_sha256, "Eikon GPN CSV"),
        (
            readme_path,
            pins.eikon_readme_size,
            pins.eikon_readme_sha256,
            "Eikon repository README",
        ),
        (tree_path, pins.eikon_tree_size, pins.eikon_tree_sha256, "Eikon repository tree"),
    ):
        _require(path.is_file(), f"{label} is missing.")
        _require(path.stat().st_size == size, f"{label} size changed.")
        _require(_sha256_file(path) == digest, f"{label} hash changed.")

    readme = readme_path.read_text(encoding="utf-8")
    _require(
        "downloaded from Eikon" in readme
        and "ten years of U.S. stocks from 2013 to 2022" in readme,
        "Eikon provenance statement changed.",
    )
    tree = json.loads(tree_path.read_bytes())
    _require(tree.get("sha") == EIKON_TREE_SHA, "Eikon repository tree SHA changed.")
    entries = tree.get("tree")
    _require(isinstance(entries, list), "Eikon repository tree is invalid.")
    matches = [item for item in entries if item.get("path") == EIKON_BLOB_PATH]
    _require(
        len(matches) == 1
        and matches[0].get("sha") == EIKON_BLOB_SHA
        and matches[0].get("size") == pins.eikon_size,
        "Commit-pinned Eikon GPN blob identity changed.",
    )
    license_paths = [
        str(item.get("path", ""))
        for item in entries
        if Path(str(item.get("path", ""))).name.lower()
        in {"license", "license.txt", "license.md", "copying"}
    ]
    _require(not license_paths, "Eikon repository licensing inventory changed.")

    frame = pd.read_csv(csv_path)
    _require(
        list(frame.columns) == ["Date", "HIGH", "CLOSE", "LOW", "OPEN", "COUNT", "VOLUME"],
        "Eikon GPN CSV columns changed.",
    )
    frame = frame.rename(
        columns={
            "Date": "session",
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "VOLUME": "volume",
        }
    )
    frame["session"] = pd.to_datetime(frame["session"], errors="coerce").dt.normalize()
    _require(
        len(frame) == pins.eikon_rows
        and _date(frame["session"].min()) == pins.eikon_first_date
        and _date(frame["session"].max()) == pins.eikon_last_date
        and not frame["session"].isna().any(),
        "Eikon GPN date/row inventory changed.",
    )
    return frame, {
        "source_url": EIKON_URL,
        "source_sha256": pins.eikon_sha256,
        "source_size": pins.eikon_size,
        "row_count": pins.eikon_rows,
        "first_date": pins.eikon_first_date,
        "last_date": pins.eikon_last_date,
        "repository_tree_sha": EIKON_TREE_SHA,
        "repository_tree_sha256": pins.eikon_tree_sha256,
        "gpn_blob_sha": EIKON_BLOB_SHA,
        "readme_sha256": pins.eikon_readme_sha256,
        "provider_attested_by_readme": "Eikon",
        "license_file_present": False,
        "license_status": "No repository license grant; Eikon-origin data",
        "use_scope": "private_internal_validation_only",
        "redistribution_allowed": False,
        "public_publication_allowed": False,
    }


def _regime_join(
    internal: pd.DataFrame,
    provider: pd.DataFrame,
    *,
    boundary: str = SPLIT_DATE,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    columns = ["session", "open", "high", "low", "close", "volume"]
    left = internal.loc[:, columns].copy()
    right = provider.loc[:, columns].copy()
    left["session"] = pd.to_datetime(left["session"]).dt.normalize()
    right["session"] = pd.to_datetime(right["session"]).dt.normalize()
    joined = left.merge(right, on="session", suffixes=("_internal", "_provider"), validate="one_to_one")
    _require(not joined.empty, "Provider has no GPN overlap.")
    split = pd.Timestamp(boundary)
    joined["regime"] = joined["session"].ge(split).astype(int)
    regimes: list[dict[str, Any]] = []
    scales: dict[int, float] = {}
    for regime, group in joined.groupby("regime", sort=True):
        ratio = pd.to_numeric(group["close_provider"]) / pd.to_numeric(group["close_internal"])
        scale = float(ratio.median())
        deviation = float(((ratio / scale) - 1.0).abs().max())
        _require(math.isfinite(scale) and scale > 0 and deviation <= 0.01, "Provider scale regime is unstable.")
        scales[int(regime)] = scale
        regimes.append(
            {
                "regime": int(regime),
                "start": _date(group["session"].min()),
                "end": _date(group["session"].max()),
                "session_count": len(group),
                "median_provider_to_internal_close_scale": scale,
                "maximum_close_scale_deviation": deviation,
            }
        )
    joined["median_scale"] = joined["regime"].map(scales)
    return joined, regimes


def _yahoo_mismatches(joined: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for field in ("open", "high", "low", "close"):
        internal = pd.to_numeric(joined[f"{field}_internal"])
        provider = pd.to_numeric(joined[f"{field}_provider"])
        normalized = provider / joined["median_scale"]
        relative = 0.005 if field == "close" else 0.01
        tolerance = internal.abs().mul(relative).clip(lower=0.02)
        failed = normalized.sub(internal).abs().gt(tolerance)
        for index in joined.index[failed]:
            output.append(
                {
                    "session": _date(joined.loc[index, "session"]),
                    "field": field,
                    "internal": _number(internal.loc[index]),
                    "provider": _number(provider.loc[index]),
                    "normalized_provider": _number(normalized.loc[index]),
                    "median_scale": _number(joined.loc[index, "median_scale"]),
                }
            )
    output.sort(key=lambda item: (item["session"], item["field"]))
    _require(
        [(item["session"], item["field"]) for item in output]
        == [("2015-08-24", "high"), ("2019-08-19", "low")],
        "Yahoo GPN mismatch inventory changed.",
    )
    return output


def _one_session(frame: pd.DataFrame, session: str) -> pd.Series:
    rows = frame.loc[pd.to_datetime(frame["session"]).dt.normalize().eq(pd.Timestamp(session))]
    _require(len(rows) == 1, f"Expected one GPN row on {session}.")
    return rows.iloc[0]


def arbitrate_field(
    *,
    session: str,
    field: str,
    internal: float,
    yahoo: float,
    supporters: Mapping[str, float],
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Retain a value only when every named independent supporter agrees."""

    _require(bool(supporters), "A disputed field requires independent support.")
    _require(
        all(math.isclose(value, internal, rel_tol=0.0, abs_tol=tolerance) for value in supporters.values()),
        "Independent GPN arbitration sources do not agree with the internal value.",
    )
    _require(
        not math.isclose(yahoo, internal, rel_tol=0.0, abs_tol=tolerance),
        "The purported Yahoo disagreement is no longer a disagreement.",
    )
    return {
        "session": session,
        "field": field,
        "internal_value": internal,
        "yahoo_normalized_value": yahoo,
        "independent_support": dict(supporters),
        "decision": "retain_eodhd_internal",
        "raw_price_repair_required": False,
    }


def _signal_frame(raw: pd.DataFrame, factors: pd.DataFrame, mode: str) -> pd.DataFrame:
    adjusted = (
        raw.copy()
        if mode == "raw"
        else apply_adjustment_factors(raw, factors, mode=mode)
    )
    values = adjusted.set_index("session")[["high", "low", "close"]].rename(
        columns={"high": "High", "low": "Low", "close": "Close"}
    )
    return add_triple_supertrend(
        values,
        settings=TRIPLE_SETTINGS,
        atr_method="wilder",
        exit_down_count=2,
    )


def _signal_diff(baseline: pd.DataFrame, alternate: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for column in SIGNAL_COLUMNS:
        changed = baseline[column].ne(alternate[column])
        output[column] = {
            "count": int(changed.sum()),
            "sessions": [_date(value) for value in baseline.index[changed]],
        }
    baseline_confirmed = baseline["TripleDownCount"].ge(2).rolling(3).sum().eq(3)
    alternate_confirmed = alternate["TripleDownCount"].ge(2).rolling(3).sum().eq(3)
    changed = baseline_confirmed.ne(alternate_confirmed)
    output["three_bar_confirmed_exit_state"] = {
        "count": int(changed.sum()),
        "sessions": [_date(value) for value in baseline.index[changed]],
    }
    return output


def _strategy_impact(
    raw: pd.DataFrame,
    factors: pd.DataFrame,
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    replacements = {
        "yahoo_2015_high_only": {("2015-08-24", "high"): mismatches[0]["normalized_provider"]},
        "yahoo_2019_low_only": {("2019-08-19", "low"): mismatches[1]["normalized_provider"]},
        "both_yahoo_fields": {
            ("2015-08-24", "high"): mismatches[0]["normalized_provider"],
            ("2019-08-19", "low"): mismatches[1]["normalized_provider"],
        },
        "accepted_eikon_fields": {
            ("2015-08-24", "high"): 109.99,
            ("2019-08-19", "low"): 158.54,
        },
    }
    output: dict[str, Any] = {}
    for mode in ("raw", "total_return_adjusted"):
        baseline = _signal_frame(raw, factors, mode)
        mode_output: dict[str, Any] = {}
        for label, changes in replacements.items():
            candidate = raw.copy()
            for (session, field), value in changes.items():
                mask = pd.to_datetime(candidate["session"]).dt.normalize().eq(pd.Timestamp(session))
                _require(int(mask.sum()) == 1, "Strategy replacement session inventory changed.")
                candidate.loc[mask, field] = value
            mode_output[label] = _signal_diff(baseline, _signal_frame(candidate, factors, mode))
        output[mode] = mode_output
    return output


def _comparison_summary(
    joined: pd.DataFrame,
    *,
    fields: Iterable[str],
    normalized: bool,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in fields:
        internal = pd.to_numeric(joined[f"{field}_internal"])
        provider = pd.to_numeric(joined[f"{field}_provider"])
        if normalized:
            provider = provider / joined["median_scale"]
        delta = provider.sub(internal).abs()
        output[field] = {
            "exact_count": int(delta.le(1e-10).sum()),
            "session_count": len(delta),
            "maximum_absolute_difference": float(delta.max()),
            "p99_absolute_difference": float(delta.quantile(0.99)),
        }
    return output


def build_report(
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    release_path: Path = DEFAULT_RELEASE,
    yahoo_cache_root: Path = DEFAULT_YAHOO_CACHE,
    wiki_zip_path: Path = DEFAULT_WIKI_ZIP,
    eikon_csv_path: Path = DEFAULT_EIKON_CSV,
    eikon_readme_path: Path = DEFAULT_EIKON_README,
    eikon_tree_path: Path = DEFAULT_EIKON_TREE,
    pins: EvidencePins = DEFAULT_PINS,
) -> dict[str, Any]:
    repository = LocalDatasetRepository(cache_root)
    release = json.loads(release_path.read_bytes())
    _require(release.get("version") == pins.release_version, "GPN release version changed.")
    versions = release.get("dataset_versions") or {}
    expected_versions = {
        "daily_price_raw": pins.daily_price_version,
        "corporate_actions": pins.action_version,
        "adjustment_factors": pins.factor_version,
        "source_archive": pins.source_archive_version,
    }
    _require(
        all(versions.get(name) == version for name, version in expected_versions.items()),
        "GPN release dataset lineage changed.",
    )

    prices = repository.read_frame("daily_price_raw", pins.daily_price_version)
    actions = repository.read_frame("corporate_actions", pins.action_version)
    factors = repository.read_frame("adjustment_factors", pins.factor_version)
    archive = repository.read_frame("source_archive", pins.source_archive_version)
    own_prices = prices.loc[prices["security_id"].astype(str).eq(SECURITY_ID)].copy()
    own_prices["session"] = pd.to_datetime(own_prices["session"]).dt.normalize()
    own_prices = own_prices.sort_values("session", kind="stable").reset_index(drop=True)
    own_factors = factors.loc[factors["security_id"].astype(str).eq(SECURITY_ID)].copy()
    own_factors["session"] = pd.to_datetime(own_factors["session"]).dt.normalize()
    _require(
        len(own_prices) == 2_899
        and _date(own_prices["session"].min()) == "2015-01-02"
        and _date(own_prices["session"].max()) == "2026-07-15",
        "Current GPN price inventory changed.",
    )
    _require(
        len(own_factors) == len(own_prices)
        and set(own_factors["session"]) == set(own_prices["session"]),
        "Current GPN factor coverage changed.",
    )

    eodhd_url = "https://eodhd.com/api/eod/GPN.US?from=2015-01-01&to=2026-07-15"
    eodhd_payload = _safe_archived_payload(
        repository,
        archive,
        source_hash=EODHD_EOD_SHA256,
        dataset="eodhd_eod",
        source_url=eodhd_url,
    )
    raw_rows = json.loads(eodhd_payload)
    eodhd_frame = pd.DataFrame(raw_rows).rename(columns={"date": "session"})
    eodhd_frame["session"] = pd.to_datetime(eodhd_frame["session"]).dt.normalize()
    raw_join = own_prices.merge(
        eodhd_frame[["session", "open", "high", "low", "close", "volume"]],
        on="session",
        suffixes=("_release", "_archive"),
        validate="one_to_one",
    )
    _require(len(raw_join) == len(own_prices), "Archived EODHD GPN sessions changed.")
    for field in ("open", "high", "low", "close", "volume"):
        _require(
            pd.to_numeric(raw_join[f"{field}_release"]).eq(
                pd.to_numeric(raw_join[f"{field}_archive"])
            ).all(),
            f"Release GPN {field} differs from archived EODHD bytes.",
        )

    split_payload = _safe_archived_payload(
        repository,
        archive,
        source_hash=EODHD_SPLIT_SHA256,
        dataset="eodhd_splits",
        source_url="https://eodhd.com/api/splits/GPN.US?from=2015-01-01&to=2026-07-15",
    )
    _safe_archived_payload(
        repository,
        archive,
        source_hash=EODHD_DIVIDEND_SHA256,
        dataset="eodhd_div",
        source_url="https://eodhd.com/api/div/GPN.US?from=2015-01-01&to=2026-07-15",
    )
    split_rows = json.loads(split_payload)
    _require(
        len(split_rows) == 1
        and _date(split_rows[0].get("date")) == SPLIT_DATE
        and math.isclose(float(str(split_rows[0].get("split")).split("/")[0]), 2.0),
        "Archived GPN split payload changed.",
    )
    split_action = actions.loc[actions["event_id"].astype(str).eq(SPLIT_EVENT_ID)]
    _require(
        len(split_action) == 1
        and str(split_action.iloc[0]["security_id"]) == SECURITY_ID
        and str(split_action.iloc[0]["action_type"]) == "split"
        and _date(split_action.iloc[0]["effective_date"]) == SPLIT_DATE
        and math.isclose(_number(split_action.iloc[0]["ratio"]), 2.0)
        and str(split_action.iloc[0]["source_hash"]) == EODHD_SPLIT_SHA256,
        "Current GPN split action changed.",
    )
    factor_counts = {
        format(float(value), ".17g"): int(count)
        for value, count in own_factors.groupby("split_factor").size().items()
    }
    _require(factor_counts == {"0.5": 211, "1": 2_688}, "GPN split factors changed.")

    yahoo_cache = YahooChartCache(yahoo_cache_root, max_http_attempts=1)
    yahoo_response = yahoo_cache.get(SYMBOL, period1=YAHOO_PERIOD1, period2=YAHOO_PERIOD2)
    _require(yahoo_response is not None, "Exact GPN Yahoo cache is missing.")
    _require(
        yahoo_response.http_status == 200
        and yahoo_response.source_url == YAHOO_URL
        and yahoo_response.source_hash == YAHOO_SOURCE_SHA256
        and yahoo_response.wrapper_hash == YAHOO_WRAPPER_SHA256,
        "Exact GPN Yahoo cache identity changed.",
    )
    yahoo = parse_yahoo_chart_json(yahoo_response.content, SYMBOL).bars
    _require(len(yahoo) == 2_899, "Yahoo GPN row inventory changed.")
    yahoo_join, yahoo_regimes = _regime_join(own_prices, yahoo)
    _require(len(yahoo_join) == 2_899, "Yahoo GPN overlap changed.")
    mismatches = _yahoo_mismatches(yahoo_join)

    wiki, wiki_evidence = _load_wiki(wiki_zip_path, pins)
    wiki_provider = wiki[["session", "open", "high", "low", "close"]].copy()
    wiki_provider["volume"] = pd.to_numeric(wiki["adj_volume"])
    wiki_provider = wiki_provider.loc[wiki_provider["session"].ge(pd.Timestamp("2015-01-01"))]
    wiki_join, wiki_regimes = _regime_join(own_prices, wiki_provider)
    _require(
        len(wiki_join) == 813
        and _date(wiki_join["session"].min()) == "2015-01-02"
        and _date(wiki_join["session"].max()) == "2018-03-27",
        "WIKI GPN overlap changed.",
    )

    eikon, eikon_evidence = _load_eikon(
        eikon_csv_path, eikon_readme_path, eikon_tree_path, pins
    )
    eikon_provider = eikon[["session", "open", "high", "low", "close", "volume"]]
    eikon_join, eikon_regimes = _regime_join(own_prices, eikon_provider)
    _require(
        len(eikon_join) == 2_020
        and _date(eikon_join["session"].min()) == "2015-01-02"
        and _date(eikon_join["session"].max()) == "2023-01-10",
        "Eikon GPN overlap changed.",
    )

    internal_2015 = _one_session(own_prices, "2015-08-24")
    yahoo_2015 = yahoo_join.loc[yahoo_join["session"].eq(pd.Timestamp("2015-08-24"))].iloc[0]
    wiki_2015 = _one_session(wiki, "2015-08-24")
    eikon_2015 = eikon_join.loc[eikon_join["session"].eq(pd.Timestamp("2015-08-24"))].iloc[0]
    internal_2019 = _one_session(own_prices, "2019-08-19")
    yahoo_2019 = yahoo_join.loc[yahoo_join["session"].eq(pd.Timestamp("2019-08-19"))].iloc[0]
    eikon_2019 = eikon_join.loc[eikon_join["session"].eq(pd.Timestamp("2019-08-19"))].iloc[0]

    decisions = [
        arbitrate_field(
            session="2015-08-24",
            field="high",
            internal=_number(internal_2015["high"]),
            yahoo=_number(yahoo_2015["high_provider"] / yahoo_2015["median_scale"]),
            supporters={
                "frozen_quandl_wiki_raw": _number(wiki_2015["high"]),
                "eikon_split_normalized": _number(
                    eikon_2015["high_provider"] / eikon_2015["median_scale"]
                ),
            },
        ),
        arbitrate_field(
            session="2019-08-19",
            field="low",
            internal=_number(internal_2019["low"]),
            yahoo=_number(yahoo_2019["low_provider"] / yahoo_2019["median_scale"]),
            supporters={"eikon_raw": _number(eikon_2019["low_provider"])},
        ),
    ]
    _require(
        decisions[0]["internal_value"] == 109.99
        and decisions[1]["internal_value"] == 158.54,
        "GPN arbitration target values changed.",
    )

    strategy = _strategy_impact(own_prices, own_factors, mismatches)
    for mode in ("raw", "total_return_adjusted"):
        _require(
            strategy[mode]["yahoo_2015_high_only"]["TripleST3_Trend"]["count"] == 9
            and strategy[mode]["yahoo_2015_high_only"]["TripleAllUp"]["count"] == 9
            and strategy[mode]["yahoo_2015_high_only"]["TripleBuySignal"]["count"] == 2,
            "GPN 2015 Triple Supertrend sensitivity changed.",
        )
        _require(
            all(
                item["count"] == 0
                for item in strategy[mode]["yahoo_2019_low_only"].values()
            ),
            "GPN 2019 low now changes Triple Supertrend.",
        )
        _require(
            all(
                item["count"] == 0
                for item in strategy[mode]["accepted_eikon_fields"].values()
            ),
            "Accepted Eikon-supported GPN values change Triple Supertrend.",
        )

    evidence_archived = {
        "eikon_csv": int(archive["source_hash"].astype(str).eq(pins.eikon_sha256).sum()),
        "wiki_gpn_extract": int(
            archive["source_hash"].astype(str).eq(pins.wiki_extract_sha256).sum()
        ),
    }
    _require(evidence_archived == {"eikon_csv": 0, "wiki_gpn_extract": 0}, "GPN evidence archive state changed.")

    report: dict[str, Any] = {
        "schema": "gpn_exact_price_arbitration/v1",
        "status": "passed",
        "target_id": TARGET_ID,
        "security_id": SECURITY_ID,
        "symbol": SYMBOL,
        "release_version": pins.release_version,
        "validated_versions": expected_versions,
        "decision": "retain_current_eodhd_prices",
        "raw_price_repair_required": False,
        "corporate_action_repair_required": False,
        "adjustment_factor_repair_required": False,
        "backtest_release_change_required": False,
        "publication_basis_state": "plan_only_external_evidence_not_archived",
        "publication_ready": False,
        "publication_blocker": (
            "Archive the exact Eikon CSV and GPN-only WIKI extract, add a code-pinned "
            "review basis, then replay the publication gate."
        ),
        "primary_source": {
            "provider": "EODHD",
            "source_url": eodhd_url,
            "source_sha256": EODHD_EOD_SHA256,
            "raw_bytes": len(eodhd_payload),
            "row_count": len(eodhd_frame),
            "release_rows_equal_archived_raw": True,
            "release_ohlcv_sha256": _project_frame(
                own_prices, ("session", "open", "high", "low", "close", "volume")
            ),
        },
        "comparison_source": {
            "provider": "Yahoo chart",
            "source_url": YAHOO_URL,
            "source_sha256": YAHOO_SOURCE_SHA256,
            "cache_wrapper_sha256": YAHOO_WRAPPER_SHA256,
            "row_count": len(yahoo),
            "regimes": yahoo_regimes,
            "mismatches": mismatches,
        },
        "third_source": {
            **eikon_evidence,
            "regimes": eikon_regimes,
            "overlap_comparison": _comparison_summary(
                eikon_join,
                fields=("open", "high", "low", "close"),
                normalized=True,
            ),
        },
        "frozen_wiki": {
            **wiki_evidence,
            "regimes": wiki_regimes,
            "overlap_comparison": _comparison_summary(
                wiki_join,
                fields=("open", "high", "low", "close", "volume"),
                normalized=False,
            ),
        },
        "field_decisions": decisions,
        "action_factor_diagnosis": {
            "classification": "not_a_missing_adjustment_or_action",
            "split_event_id": SPLIT_EVENT_ID,
            "split_effective_date": SPLIT_DATE,
            "split_ratio": 2.0,
            "split_source_sha256": EODHD_SPLIT_SHA256,
            "dividend_source_sha256": EODHD_DIVIDEND_SHA256,
            "split_factor_session_counts": factor_counts,
            "factor_session_coverage": 1.0,
            "split_factor_values": [0.5, 1.0],
        },
        "strategy_sensitivity": {
            "settings": [
                {"period": period, "multiplier": multiplier}
                for period, multiplier in TRIPLE_SETTINGS
            ],
            "atr_method": "wilder",
            "exit_down_count": 2,
            "exit_confirm_bars": 3,
            "signal_price_modes": ["raw", "total_return_adjusted"],
            "results": strategy,
            "portfolio_interpretation": (
                "The accepted fields equal the current release, so the portfolio backtest "
                "does not change. The Yahoo 2015 high is a rejected counterfactual that "
                "would alter GPN eligibility for nine sessions and two buy-signal dates."
            ),
        },
        "license_and_redistribution": {
            "eikon": "No repository license grant; private internal validation only.",
            "wiki": "Kaggle/Quandl WIKI license is Unknown; private internal validation only.",
            "raw_evidence_redistribution_allowed": False,
            "public_publication_allowed": False,
        },
        "external_request_accounting": {
            "this_offline_audit_network_calls": 0,
            "eodhd_calls": 0,
            "preceding_arbitration_requests": [
                {
                    "provider": "Nasdaq historical API",
                    "result": "HTTP 200 error payload; invalid fromdate",
                    "raw_bytes": 165,
                    "raw_sha256": "e7286b2c32d902a3826b1ae117df2ceba3bec299aaaf99d15bbf729fbee9c7f7",
                    "usable_price_evidence": False,
                },
                {
                    "provider": "Stooq",
                    "result": "HTTP 200 JavaScript verification challenge",
                    "raw_bytes": 796,
                    "raw_sha256": "f896ac5e5a468f4593f32c60597024b789062946d0a73a6039a9c154a067b933",
                    "usable_price_evidence": False,
                },
                {
                    "provider": "Eikon via commit-pinned GitHub raw",
                    "result": "HTTP 200 exact CSV",
                    "raw_bytes": pins.eikon_size,
                    "raw_sha256": pins.eikon_sha256,
                    "usable_price_evidence": True,
                    "retries": 0,
                },
            ],
            "total_external_requests": 3,
        },
        "source_archive_plan": {
            "current_presence_count": evidence_archived,
            "rows_to_add": 2,
            "datasets_to_rewrite": ["source_archive"],
            "daily_price_rows_to_change": 0,
            "corporate_action_rows_to_change": 0,
            "adjustment_factor_rows_to_change": 0,
            "release_apply_performed": False,
            "r2_accessed": False,
        },
    }
    report["body_sha256"] = _sha256_bytes(_canonical_json_bytes(report))
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--yahoo-cache", type=Path, default=DEFAULT_YAHOO_CACHE)
    parser.add_argument("--wiki-zip", type=Path, default=DEFAULT_WIKI_ZIP)
    parser.add_argument("--eikon-csv", type=Path, default=DEFAULT_EIKON_CSV)
    parser.add_argument("--eikon-readme", type=Path, default=DEFAULT_EIKON_README)
    parser.add_argument("--eikon-tree", type=Path, default=DEFAULT_EIKON_TREE)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_report(
        cache_root=args.cache_root,
        release_path=args.release,
        yahoo_cache_root=args.yahoo_cache,
        wiki_zip_path=args.wiki_zip,
        eikon_csv_path=args.eikon_csv,
        eikon_readme_path=args.eikon_readme,
        eikon_tree_path=args.eikon_tree,
    )
    payload = _canonical_json_bytes(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(payload.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
