from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_trained_compact_http_canary_preregistration_v65.json"
RUNNER = RUNTIME / "run_qwen38_27b_trained_compact_http_canary_v65.sh"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def test_v65_preserves_v64_and_changes_only_transport() -> None:
    p = json.loads(PREREG.read_text())
    parent = p["frozen_parent"]
    assert parent["v64_receipt_sha256"] == sha256(
        RUNTIME / "receipts/qwen38_27b_trained_compact_model_graph_v64/receipt.json"
    )
    assert parent["v64_adjudication_sha256"] == sha256(
        RUNTIME / "receipts/qwen38_27b_trained_compact_model_graph_v64/adjudication.json"
    )
    assert parent["v64_terminal_negative_must_remain_unchanged"]
    assert parent["v64_printed_generation_tokens_per_second_is_inadmissible"]
    assert p["unchanged_candidate"]["candidate_sha256"] == (
        "506505935f82bade12ab4b51bd1ca9b6e1b5cc11eeae99a6e230b9a7b9044ffc"
    )

def test_v65_is_bounded_semantic_http_canary() -> None:
    p = json.loads(PREREG.read_text())
    protocol, bounds, gate = p["http_protocol"], p["resource_bounds"], p["pass_gate"]
    assert protocol["endpoint"] == "/v1/chat/completions"
    assert protocol["max_tokens"] == 8
    assert protocol["enable_thinking"] is False
    assert protocol["stream"] is False
    assert bounds["memory_max_bytes"] == 30000000000
    assert bounds["response_max_bytes"] == 1048576
    assert gate["assistant_content_case_insensitively_contains"] == "Paris"
    assert gate["response_contains_no_timing_or_tokens_per_second_fields"]

def test_v65_runner_enforces_protocol_and_resource_bounds() -> None:
    source = RUNNER.read_text()
    assert f"expected_prereg={sha256(PREREG)}" in source
    assert "QWEN38_AFF3_LAYERS=0-63" in source
    assert "QWEN38_RESIDUAL_DIR" in source
    assert "MemoryMax=30000000000" in source
    assert "MemorySwapMax=0" in source
    assert "RuntimeMaxSec=420" in source
    assert "--log-disable" in source
    assert '"chat_template_kwargs": {"enable_thinking": False}' in source
    assert '"paris" in content.casefold()' in source
    assert '"tokens_per_second" in key' in source
