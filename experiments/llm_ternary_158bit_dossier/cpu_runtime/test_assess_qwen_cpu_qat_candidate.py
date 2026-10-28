import json
from pathlib import Path

from assess_qwen_cpu_qat_candidate import assess, canonical_digest


def write_report(path: Path, payload: dict, digest_key: str) -> None:
    payload[digest_key] = canonical_digest(payload)
    path.write_text(json.dumps(payload))


def test_assessment_passes_only_with_bound_complete_reports(tmp_path: Path) -> None:
    model_sha = "model-sha"
    reconstruction = {
        "candidate_patch_sha256": "patch-sha",
        "candidate_exact_ternary": True,
        "splits": {
            "test": {
                "positions": 231,
                "candidate_relative_improvement_vs_direct_ptq_percent": 27.5,
            }
        },
    }
    scope = {
        "valid": True,
        "target_count": 3,
        "non_target_count": 438,
        "models": {"candidate": {"sha256": model_sha}},
    }
    cute = {
        "model_sha256": model_sha,
        "summaries": [{"task": "cute_quick_20", "correct": 7, "n": 20}],
    }
    math_code = {
        "model_sha256": model_sha,
        "summaries": [
            {"task": "math500", "correct": 5, "n": 8},
            {"task": "code_trace", "correct": 4, "n": 6},
        ],
    }
    ppl = {
        "model_sha256": model_sha,
        "valid": True,
        "status": "completed",
        "final_ppl": 8.67,
        "final_uncertainty": 0.73,
    }
    paths = {
        "reconstruction": tmp_path / "reconstruction.json",
        "scope": tmp_path / "scope.json",
        "cute": tmp_path / "cute.json",
        "math_code": tmp_path / "math-code.json",
        "ppl": tmp_path / "ppl.json",
    }
    write_report(paths["reconstruction"], reconstruction, "report_content_sha256")
    write_report(paths["scope"], scope, "receipt_content_sha256")
    write_report(paths["cute"], cute, "report_content_sha256")
    write_report(paths["math_code"], math_code, "report_content_sha256")
    write_report(paths["ppl"], ppl, "report_content_sha256")

    result = assess(
        reconstruction_path=paths["reconstruction"],
        scope_path=paths["scope"],
        cute_path=paths["cute"],
        math_code_path=paths["math_code"],
        ppl_path=paths["ppl"],
        min_reconstruction_improvement=10.0,
        min_cute_correct=7,
        min_math_correct=5,
        min_code_correct=4,
        max_ppl=8.6975,
    )

    assert result["pass"] is True
    assert result["depth_expansion_authorized"] is False
    assert all(result["gates"].values())


def test_assessment_rejects_valid_but_high_ppl(tmp_path: Path) -> None:
    model_sha = "model-sha"
    payloads = {
        "reconstruction": {
            "candidate_patch_sha256": "patch-sha",
            "candidate_exact_ternary": True,
            "splits": {"test": {"positions": 1, "candidate_relative_improvement_vs_direct_ptq_percent": 11.0}},
        },
        "scope": {"valid": True, "target_count": 3, "non_target_count": 438, "models": {"candidate": {"sha256": model_sha}}},
        "cute": {"model_sha256": model_sha, "summaries": [{"task": "cute_quick_20", "correct": 7, "n": 20}]},
        "math_code": {"model_sha256": model_sha, "summaries": [{"task": "math500", "correct": 5, "n": 8}, {"task": "code_trace", "correct": 4, "n": 6}]},
        "ppl": {"model_sha256": model_sha, "valid": True, "status": "completed", "final_ppl": 8.8, "final_uncertainty": 0.7},
    }
    paths = {name: tmp_path / f"{name}.json" for name in payloads}
    for name, payload in payloads.items():
        write_report(
            paths[name],
            payload,
            "receipt_content_sha256" if name == "scope" else "report_content_sha256",
        )

    result = assess(
        reconstruction_path=paths["reconstruction"],
        scope_path=paths["scope"],
        cute_path=paths["cute"],
        math_code_path=paths["math_code"],
        ppl_path=paths["ppl"],
        min_reconstruction_improvement=10.0,
        min_cute_correct=7,
        min_math_correct=5,
        min_code_correct=4,
        max_ppl=8.6975,
    )

    assert result["pass"] is False
    assert result["gates"]["wikitext"] is False
