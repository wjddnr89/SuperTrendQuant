from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "collect_us_fisv_confirmation_evidence.py"
)
SPEC = importlib.util.spec_from_file_location(
    "collect_us_fisv_confirmation_evidence", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def _content() -> bytes:
    return (
        "<html><body>Fiserv, Inc. quarter ended March 31, 2026 "
        "Trading Symbol(s) FISV The Nasdaq Stock Market LLC</body></html>"
    ).encode()


def _tree(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_default_offline_plan_is_read_only_and_one_url_limited(tmp_path: Path):
    before = _tree(tmp_path)

    result = script.offline_plan(tmp_path)

    assert result["status"] == "ready_for_authorized_fetch"
    assert result["source_url"] == script.CONFIRMATION_SOURCE["source_url"]
    assert result["max_http_attempts"] == 1
    assert result["http_attempts_this_run"] == 0
    assert result["writes_performed"] is False
    assert result["network_accessed"] is False
    assert result["eodhd_calls"] == 0
    assert result["r2_accessed"] is False
    assert _tree(tmp_path) == before


def test_fetch_uses_exactly_one_url_then_replays_without_network(tmp_path: Path):
    calls: list[tuple[str, str]] = []

    def fetcher(url: str, user_agent: str) -> bytes:
        calls.append((url, user_agent))
        return _content()

    first = script.collect(
        tmp_path,
        user_agent="Researcher researcher@example.com",
        fetcher=fetcher,
    )
    second = script.collect(
        tmp_path,
        user_agent="Researcher researcher@example.com",
        fetcher=lambda *_args: pytest.fail("cache replay attempted network"),
    )

    assert calls == [
        (
            script.CONFIRMATION_SOURCE["source_url"],
            "Researcher researcher@example.com",
        )
    ]
    assert first["http_attempts_this_run"] == 1
    assert first["network_accessed"] is True
    assert second["status"] == "cache_verified"
    assert second["http_attempts_this_run"] == 0
    assert second["network_accessed"] is False
    evidence = first["evidence"]
    payload = Path(first["payload_path"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == evidence["source_hash"]
    assert len(payload) == evidence["size"]


def test_fetch_requires_contact_user_agent_before_attempt(tmp_path: Path):
    called = False

    def fetcher(_url: str, _user_agent: str) -> bytes:
        nonlocal called
        called = True
        return _content()

    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        script.collect(tmp_path, user_agent="missing-contact", fetcher=fetcher)
    assert called is False
    assert _tree(tmp_path) == {}


@pytest.mark.parametrize(
    "missing_phrase",
    [
        "Fiserv, Inc.",
        "March 31, 2026",
        "Trading Symbol(s)",
        "FISV",
        "The Nasdaq Stock Market LLC",
    ],
)
def test_fetch_rejects_missing_reviewed_term_without_writes(
    tmp_path: Path, missing_phrase: str
):
    content = _content().replace(missing_phrase.encode(), b"removed")

    with pytest.raises(ValueError, match="lacks reviewed official term"):
        script.collect(
            tmp_path,
            user_agent="Researcher researcher@example.com",
            fetcher=lambda *_args: content,
        )

    assert _tree(tmp_path) == {}


def test_cached_payload_and_report_tampering_are_rejected(tmp_path: Path):
    result = script.collect(
        tmp_path,
        user_agent="Researcher researcher@example.com",
        fetcher=lambda *_args: _content(),
    )
    payload_path = Path(result["payload_path"])
    payload_path.write_bytes(payload_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="hash/size verification failed"):
        script.offline_plan(tmp_path)

    payload_path.write_bytes(_content())
    report_path = tmp_path / script.EVIDENCE_SUBDIR / script.REPORT_FILENAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["evidence"]["source_url"] = "https://www.sec.gov/Archives/wrong.htm"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="identity changed"):
        script.offline_plan(tmp_path)


def test_hardcoded_url_envelope_rejects_any_other_target(monkeypatch):
    changed = dict(script.CONFIRMATION_SOURCE)
    changed["source_url"] = (
        "https://www.sec.gov/Archives/edgar/data/798354/"
        "000079835426000018/another.htm"
    )
    monkeypatch.setattr(script, "CONFIRMATION_SOURCE", changed)

    with pytest.raises(ValueError, match="URL envelope changed"):
        script.offline_plan(Path("unused"))
