from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]
RUNTIME = ROOT / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
MODULE_PATH = RUNTIME / "summarize_qwen38_27b_imatrix_q4_k_native_x4_composition_v23.py"
SPEC = importlib.util.spec_from_file_location("v23_summary", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PREREG = RUNTIME / "qwen38_27b_imatrix_q4_k_native_x4_composition_v23.json"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def bench_row(nanoseconds: int) -> list[dict[str, object]]:
    return [
        {
            "n_prompt": 4,
            "n_gen": 0,
            "samples_ns": [nanoseconds] * 7,
        }
    ]


def test_three_arm_graph_uses_rotation_paired_medians(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    frequency = tmp_path / "frequency.log"
    output = tmp_path / "graph.json"
    write_json(artifact, {"passed": True})
    frequency.write_text("1 2400.0\n2 2400.0\n")
    for rotation in range(1, 7):
        write_json(tmp_path / f"graph-rotation-{rotation:02d}-A.json", bench_row(125))
        write_json(tmp_path / f"graph-rotation-{rotation:02d}-B.json", bench_row(100))
        write_json(tmp_path / f"graph-rotation-{rotation:02d}-C.json", bench_row(80))

    status = MODULE.summarize_graph(
        SimpleNamespace(
            preregistration=PREREG,
            artifact_decision=artifact,
            state=tmp_path,
            frequency_log=frequency,
            out=output,
        )
    )

    result = json.loads(output.read_text())
    assert status == 0
    assert result["passed"] is True
    assert result["denominator"]["samples_per_arm"] == 42
    assert result["ratios"]["b_over_a_median"] == 1.25
    assert result["ratios"]["c_over_b_median"] == 1.25
    assert result["ratios"]["c_over_a_median"] == 1.5625


def mtp_summary(target_tps: float, mtp3_tps: float) -> dict[str, object]:
    def row(label: str, tps: float) -> dict[str, object]:
        return {
            "configuration": label,
            "accepted_tokens_per_second_weighted": tps,
            "output_tokens_including_prompt_eval_token": 384,
            "draft_n": 300 if label == "mtp3" else 0,
            "draft_n_accepted": 200 if label == "mtp3" else 0,
        }

    return {
        "terminal": True,
        "generation": {"prompt_count": 4, "tokens_per_prompt": 96},
        "configurations": [row("target", target_tps), row("mtp3", mtp3_tps)],
    }


def write_mtp(root: Path, target_tps: float, mtp3_tps: float, token_offset: int) -> None:
    write_json(root / "summary.json", mtp_summary(target_tps, mtp3_tps))
    tokens = list(range(token_offset, token_offset + 96))
    for label in ("target", "mtp3"):
        for prompt in range(4):
            write_json(
                root / label / f"prompt-{prompt:02d}.json",
                {"response": {"tokens": tokens}},
            )


def test_system_gate_requires_candidate_own_target_parity_not_cross_model_identity(
    tmp_path: Path,
) -> None:
    graph = tmp_path / "graph.json"
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    control_server = tmp_path / "control-server"
    candidate_server = tmp_path / "candidate-server"
    output = tmp_path / "system.json"
    write_json(graph, {"passed": True})
    write_mtp(control, 2.0, 3.0, token_offset=1000)
    write_mtp(candidate, 2.4, 3.9, token_offset=2000)
    control_server.write_bytes(b"control")
    candidate_server.write_bytes(b"candidate")

    status = MODULE.summarize_system(
        SimpleNamespace(
            preregistration=PREREG,
            graph_decision=graph,
            control_mtp=control,
            candidate_mtp=candidate,
            control_server=control_server,
            candidate_server=candidate_server,
            out=output,
        )
    )

    result = json.loads(output.read_text())
    assert status == 0
    assert result["passed"] is True
    assert result["target"]["candidate_over_control"] == 1.2
    assert result["mtp3"]["candidate_over_control"] == 1.3
    assert result["mtp3"]["candidate_exact_parity_with_own_target"] is True
