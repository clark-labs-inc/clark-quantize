from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime/summarize_qwen38_27b_cross_build_token_parity_v35.py"
)
SPEC = importlib.util.spec_from_file_location("v35_cross_build", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_response(
    root: Path, arm: str, mode: str, prompt: int, tokens: list[int]
) -> None:
    path = root / arm / mode / f"prompt-{prompt:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "terminal": True,
                "prompt_index": prompt,
                "prompt_sha256": f"prompt-{prompt}",
                "configuration": mode,
                "n_predict": 96,
                "response": {"tokens": tokens},
            }
        )
    )


def test_first_mismatch_covers_content_and_length() -> None:
    assert MODULE.first_mismatch([1, 2], [1, 2]) is None
    assert MODULE.first_mismatch([1, 2], [1, 3]) == 1
    assert MODULE.first_mismatch([1], [1, 2]) == 1


def test_cross_build_diagnostic_exposes_acceptance_confound(tmp_path: Path) -> None:
    for prompt in MODULE.PROMPTS:
        parent = [prompt] * 96
        candidate = parent.copy()
        if prompt != 0:
            candidate[40 + prompt] = 100 + prompt
        for arm in MODULE.ARMS:
            tokens = candidate if arm.startswith("candidate") else parent
            for mode in MODULE.MODES:
                write_response(tmp_path, arm, mode, prompt, tokens)

    result = MODULE.summarize(tmp_path)
    assert result["aggregate"] == {
        "within_build_target_mtp3_exact_pairs": 16,
        "within_build_target_mtp3_pair_denominator": 16,
        "cross_build_candidate_parent_exact_prompts": 1,
        "cross_build_candidate_parent_prompt_denominator": 4,
        "parent_ab_repeat_exact_every_prompt": True,
        "candidate_ab_repeat_exact_every_prompt": True,
    }
    assert result["prompts"][1][
        "candidate_parent_first_mismatch_token_index"
    ] == 41
    assert "sequence_confounded" in result["classification"]
