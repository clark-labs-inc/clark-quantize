from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
POLICY = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "qwen38_27b_materiality_priority_amendment_v4.json"
)


def test_materiality_policy_requires_whole_qwen38_text_graph() -> None:
    payload = json.loads(POLICY.read_text())
    target = payload["admission_policy_after_current_gate"]["required_full_target"]
    assert payload["model"] == "Qwen/Qwen3.8-27B"
    assert target["parameters"] == 26_895_998_464
    assert target["tensors"] == 851
    assert target["qwen_4b_compute_forbidden"] is True
    assert payload["primary_big_bet"]["falsification_gate"]["scope"].endswith(
        "authenticated BF16 parent"
    )


def test_materiality_policy_blocks_small_standalone_lanes() -> None:
    payload = json.loads(POLICY.read_text())
    blocked = payload["admission_policy_after_current_gate"][
        "forbidden_as_standalone_next_steps"
    ]
    assert "sampled weight rows" in blocked
    assert "synthetic-only microkernels" in blocked
    assert "new hash routing variants below complete-block scope" in blocked
    assert payload["runtime_big_bet"]["minimum_paired_end_to_end_speedup"] == 1.15
    assert payload["primary_big_bet"]["falsification_gate"][
        "maximum_projected_complete_text_file_bits_per_parameter"
    ] <= 1.70


def test_twla_is_only_an_audited_starting_point() -> None:
    payload = json.loads(POLICY.read_text())
    upstream = payload["primary_big_bet"]["external_starting_point"]
    assert upstream["arxiv"] == "2606.13054v2"
    assert upstream["audited_commit"] == "805cbb9887c66fb894f88935e2bcfd5399f384a5"
    assert len(upstream["not_accepted_as_evidence"]) == 4
    assert payload["conditional_original_big_bet"]["admission_condition"].startswith(
        "run only if"
    )
