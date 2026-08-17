import json
from pathlib import Path

from assess_llama_cpp_mmlu_replication import assess, canonical_digest


def write_report(path: Path, label: str, correct: set[int]) -> None:
    contract = {
        "manifest_content_sha256": "manifest",
        "model": f"/{label}.gguf",
        "model_label": label,
        "model_sha256": f"{label}-sha",
        "server": f"http://127.0.0.1:{label}",
        "workers": 4,
        "evaluator_sha256": "evaluator",
        "completion": {"method": "exact-choice"},
    }
    records = [
        {
            "sample_id": f"mmlu:{index}",
            "dataset_index": index,
            "subject": f"subject-{index % 4}",
            "row_sha256": f"row-{index}",
            "gold": "A",
            "prediction": "A" if index in correct else "B",
            "correct": index in correct,
        }
        for index in range(64)
    ]
    payload = {
        "schema": "llama_cpp_mmlu_exact_choice_v1",
        "status": "completed",
        "valid": True,
        "contract": contract,
        "summary": {"n": 64, "correct": len(correct)},
        "records": records,
    }
    payload["report_content_sha256"] = canonical_digest(payload)
    path.write_text(json.dumps(payload))


def test_replication_passes_at_frozen_boundary(tmp_path: Path) -> None:
    bf16 = tmp_path / "bf16.json"
    ptq = tmp_path / "ptq.json"
    candidate = tmp_path / "candidate.json"
    write_report(bf16, "bf16", set(range(50)))
    write_report(ptq, "ptq", set(range(47)))
    write_report(candidate, "candidate", set(range(47)))

    result = assess(bf16, ptq, candidate)

    assert result["pass"] is True
    assert result["decision"] == "authorize_preregistered_two_layer_experiment"
    assert result["broad_depth_expansion_authorized"] is False


def test_replication_rejects_candidate_below_direct_ptq(tmp_path: Path) -> None:
    bf16 = tmp_path / "bf16.json"
    ptq = tmp_path / "ptq.json"
    candidate = tmp_path / "candidate.json"
    write_report(bf16, "bf16", set(range(50)))
    write_report(ptq, "ptq", set(range(49)))
    write_report(candidate, "candidate", set(range(48)))

    result = assess(bf16, ptq, candidate)

    assert result["pass"] is False
    assert result["gates"]["candidate_at_least_direct_ptq"] is False
