from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
MODULE_PATH = RUNTIME / "run_qwen38_27b_imatrix_cq3_hybrid_quality_v81.py"
PREREG = RUNTIME / "qwen38_27b_imatrix_cq3_hybrid_quality_preregistration_v81.json"
LAUNCHER = RUNTIME / "launch_qwen38_27b_imatrix_cq3_hybrid_quality_v81.sh"
SPEC = importlib.util.spec_from_file_location("imatrix_cq3_quality_v81", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v81_freezes_complete_model_quality_before_observation() -> None:
    data = json.loads(PREREG.read_text())
    assert data["frozen_before_first_v80_generation_or_perplexity_observation"] is True
    assert data["candidate"]["sha256"] == MODULE.EXPECTED_CANDIDATE_SHA256
    assert data["candidate"]["bytes"] == MODULE.EXPECTED_CANDIDATE_BYTES
    assert data["performance_is_forbidden_inside_v81"] is True


def test_v81_uses_all_required_capability_surfaces_and_strict_gates() -> None:
    data = json.loads(PREREG.read_text())
    assert data["evaluation_order"] == [
        "CUTE Quick 20 free generation",
        "MATH-500 8 plus code-trace 6 free generation",
        "WikiText-2 four by 512 perplexity",
        "MMLU five-shot 64 only if every quick gate passes",
    ]
    assert data["quick_gates"] == {
        "minimum_cute_correct_of_20": 8,
        "minimum_math500_correct_of_8": 4,
        "minimum_code_trace_correct_of_6": 4,
        "maximum_wikitext_ppl_ratio_vs_q4": 1.05,
        "q4_control_wikitext_ppl": 5.7847,
    }
    assert data["mmlu_gate"]["minimum_correct_of_64"] == 50


def test_v81_marks_cute_lineage_and_forbids_adaptive_rescue() -> None:
    data = json.loads(PREREG.read_text())
    assert data["data_status"]["cute_is_adaptive_lineage_evidence"] is True
    assert data["data_status"]["promotion_or_novelty_claim_from_v81_alone_forbidden"] is True
    assert "do not tune" in data["promotion"]["if_any_quick_gate_fails"]


def test_v81_runner_is_restartable_and_enforces_denominators() -> None:
    source = MODULE_PATH.read_text()
    assert 'if marker_path.is_file() and final_path.is_file()' in source
    assert 'len(cute.get("records", [])) != 20' in source
    assert 'len(math_code.get("records", [])) != 14' in source
    assert '"--checkpoint-every", "4"' in source
    assert 'command.append("--resume")' in source
    launcher = LAUNCHER.read_text()
    assert "AllowedCPUs=0,2,4,6" in launcher
    assert "MemoryMax=30000000000" in launcher
    assert "MemorySwapMax=0" in launcher
