from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
ADAPTER = RUNTIME / "scoring_adapter_v1/sitecustomize.py"
AMENDMENT = RUNTIME / "qwen38_27b_llamacpp_ternary_controls_scoring_amendment_v2.json"
RESUME = RUNTIME / "resume_qwen38_27b_llamacpp_ternary_controls_after_peg_v1.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_adapter():
    spec = importlib.util.spec_from_file_location("qwen38_scoring_adapter_v1", ADAPTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def http_error(body: bytes, status: int = 500) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://127.0.0.1:8110/v1/chat/completions",
        status,
        "Internal Server Error",
        {},
        io.BytesIO(body),
    )


def test_exact_peg_native_failure_is_scored_incorrect_and_audited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_adapter()
    body = json.dumps(
        {
            "error": {
                "code": 500,
                "message": module.EXPECTED_MESSAGE,
                "type": "server_error",
            }
        },
        separators=(",", ":"),
    ).encode()

    def fail(*_args, **_kwargs):
        raise http_error(body)

    events = tmp_path / "events.jsonl"
    monkeypatch.setattr(module, "_original_urlopen", fail)
    monkeypatch.setenv("CLARK_PEG_ERROR_EVENTS", str(events))
    request = urllib.request.Request(
        "http://127.0.0.1:8110/v1/chat/completions",
        data=b'{"messages":[]}',
        method="POST",
    )
    with module.resilient_urlopen(request) as response:
        result = json.load(response)

    assert result["choices"][0]["message"]["content"] == module.SENTINEL
    assert result["choices"][0]["finish_reason"] == "server_format_error"
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["scored_as_incorrect"] is True
    assert rows[0]["upstream_http_status"] == 500
    assert rows[0]["request_body_sha256"] == hashlib.sha256(b'{"messages":[]}').hexdigest()


def test_unmatched_http_error_is_reraised_with_body_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_adapter()
    body = b'{"error":{"code":500,"message":"out of memory","type":"server_error"}}'

    def fail(*_args, **_kwargs):
        raise http_error(body)

    monkeypatch.setattr(module, "_original_urlopen", fail)
    with pytest.raises(urllib.error.HTTPError) as caught:
        module.resilient_urlopen("http://127.0.0.1:8110/health")
    assert caught.value.read() == body


def test_post_observation_amendment_freezes_conservative_adapter() -> None:
    payload = json.loads(AMENDMENT.read_text())
    assert payload["model"] == "Qwen/Qwen3.8-27B"
    assert payload["post_observation_amendment"] is True
    assert payload["observation"]["first_attempt"]["infrastructure_invalid"] is False
    assert payload["observation"]["first_attempt"]["terminal_suite_result"] is False
    assert payload["scoring_clarification"]["promotability_increased"] is False
    assert payload["scoring_clarification"]["thresholds_changed"] is False
    assert payload["continuation"]["rerun_tq1_conversion"] is False
    for descriptor in payload["frozen_sources"].values():
        path = REPO / descriptor["path"]
        assert path.is_file()
        assert sha256(path) == descriptor["sha256"]


def test_resume_preserves_attempt_and_reuses_frozen_runner() -> None:
    script = RESUME.read_text()
    assert "ATTEMPT_ARCHIVED" in script
    assert 'require_sha256 "$tq1_model" "$tq1_sha"' in script
    assert "CLARK_SCORE_PEG_NATIVE_HTTP_500_AS_INCORRECT=1" in script
    assert 'bash "$original_runner"' in script
    assert "rerun_tq1_conversion" not in script
