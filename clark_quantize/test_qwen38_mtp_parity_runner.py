from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_mtp_parity_preflight_v1.json"
RUNNER = RUNTIME / "run_qwen38_27b_mtp_parity_v1.py"


def test_mtp_preflight_is_qwen38_cpu_only_and_hash_pinned() -> None:
    payload = json.loads(PREREG.read_text())
    assert payload["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert payload["scope"]["qwen_4b_compute_forbidden"] is True
    assert payload["scope"]["cpu_only"] is True
    assert len(payload["target"]["sha256"]) == 64
    assert len(payload["draft"]["sha256"]) == 64
    assert len(payload["runtime"]["llama_server_sha256"]) == 64


def test_mtp_preflight_requires_exact_ids_and_counts_only_accepted_output() -> None:
    payload = json.loads(PREREG.read_text())
    assert payload["gate"]["exact_committed_token_id_parity_on_every_prompt"] is True
    assert payload["gate"]["target_only_control_required"] is True
    assert payload["gate"]["speedup_does_not_override_parity_failure"] is True
    assert payload["generation"]["return_tokens"] is True
    assert payload["claim_boundary"].startswith("Four-prompt development preflight")


def test_mtp_runner_is_restartable_and_terminates_its_server() -> None:
    source = RUNNER.read_text()
    assert "completed_response" in source
    assert '"return_tokens": True' in source
    assert '"ignore_eos": True' in source
    assert "start_new_session=True" in source
    assert "os.killpg(process.pid, signal.SIGTERM)" in source
    assert 'return 0 if summary["passed"] else 2' in source


def test_mtp_runner_reports_parity_acceptance_and_weighted_speed() -> None:
    source = RUNNER.read_text()
    assert '"exact_token_parity": config_parity' in source
    assert '"draft_n_accepted": draft_accepted' in source
    assert '"accepted_tokens_per_second_weighted": weighted_tps' in source
    assert '"accepted_decode_speedup_vs_target"' in source
    assert 'int(timing["predicted_n"]) - 1' in source
    assert "target_calls = accepted_decode_tokens - draft_accepted" in source
    assert '"mean_committed_tokens_per_target_call"' in source
    assert '"minimum_weight_bandwidth_bytes_per_second_at_100_tps"' in source
