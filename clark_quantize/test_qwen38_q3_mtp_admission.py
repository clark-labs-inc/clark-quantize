from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_q3_mtp_admission_v1.json"
RUNNER = RUNTIME / "run_qwen38_27b_q3_mtp_admission_v1.sh"


def test_q3_mtp_admission_is_qwen38_cpu_only_and_quality_gated() -> None:
    payload = json.loads(PREREG.read_text())
    assert payload["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert payload["scope"]["qwen_4b_compute_forbidden"] is True
    assert payload["scope"]["cpu_only"] is True
    assert payload["prerequisites"]["q3_baseline_terminal_and_passed"] is True
    assert payload["experiment"]["mtp_max_widths"] == [3, 4, 6, 8]
    assert payload["promotion_gate"]["exact_committed_token_id_parity_on_every_prompt_and_width"] is True
    assert payload["promotion_gate"]["maximum_best_combined_weight_bytes_per_accepted_token_lower_bound"] == 7_000_000_000


def test_q3_mtp_runner_authenticates_and_traces_the_exact_models() -> None:
    source = RUNNER.read_text()
    assert 'require_sha256 "$target" "$expected_target_sha"' in source
    assert 'require_sha256 "$draft" "$mtp_sha"' in source
    assert 'require_sha256 "$server" "$server_sha"' in source
    assert "--widths 0,3,4,6,8" in source
    assert "--trace-speculative" in source
    assert "--server-verbosity 5" in source
    assert "--draft-p-min 0.0" in source
    assert "another Qwen3.8 workload is active" in source


def test_q3_mtp_runner_preserves_scientific_rejection_and_terminal_receipt() -> None:
    source = RUNNER.read_text()
    assert '"$runner_status" -ne 0 && "$runner_status" -ne 2' in source
    assert 'best["mean_committed_tokens_per_target_call"] >= 3.0' in source
    assert 'best["combined_weight_bytes_per_accepted_token_lower_bound"] <= 7_000_000_000' in source
    assert "payload = {" in source
    assert 'printf \'terminal\\n\' > "$state_dir/ADMISSION_TERMINAL"' in source
