from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

import torch

from clark_quantize.qwen38_layer0_qat import RowShardTrainingConfig


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "run_qwen38_27b_mixed_train_blocks_v1.py"
)


def load_module():
    runtime = SCRIPT.parent
    sys.path.insert(0, str(runtime))
    try:
        spec = importlib.util.spec_from_file_location("mixed_train_blocks", SCRIPT)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(runtime))


def test_mixed_training_runner_is_qwen38_only_and_has_no_holdout_argument() -> None:
    module = load_module()
    assert module.MODEL_ID == "Qwen/Qwen3.8-27B"
    assert module.TARGET_LAYERS == (0, 32, 63)
    signature = inspect.signature(module.train_cq3_tensor)
    assert "validation" not in " ".join(signature.parameters)
    assert "confirmation" not in " ".join(signature.parameters)
    source = SCRIPT.read_text()
    assert "Qwen3.5-4B" not in source
    assert 'cache["validation_flat_rows"]' not in source
    assert "confirmation-cache.pt" not in source


def test_mixed_training_runner_writes_and_reuses_one_terminal_cq3_tensor(
    tmp_path: Path,
) -> None:
    module = load_module()
    torch.manual_seed(29)
    source = torch.randn(128, 256, dtype=torch.bfloat16) * 0.02
    candidate = torch.randn(4, 256, dtype=torch.bfloat16)
    teacher = candidate + 0.01 * torch.randn_like(candidate)
    target = {
        "state_name": "mlp.gate_proj.weight",
        "gguf_name": "blk.0.ffn_gate.weight",
        "shape": (128, 256),
    }
    arguments = dict(
        source=source,
        candidate_inputs=candidate,
        teacher_inputs=teacher,
        target=target,
        layer_index=0,
        tensor_root=tmp_path,
        repository_commit="a" * 40,
        preregistration_sha256="b" * 64,
        training_amendment_sha256="c" * 64,
        training_cache_sha256="d" * 64,
        expected_source_sha256=module.tensor_sha256(source),
        config=RowShardTrainingConfig(steps=1),
        row_shard=128,
    )
    first = module.train_cq3_tensor(**arguments)
    second = module.train_cq3_tensor(**arguments)
    assert first == second
    assert first["schema"] == module.TENSOR_SCHEMA
    assert first["payload"]["bytes"] == 128 * 68
    assert first["all_pack_unpack_mappings_exact"] is True
    assert first["complete"] is True
    assert first["terminal"] is True
    assert not (tmp_path / "active-shards").exists()


def test_mixed_training_runner_has_frozen_sequence_and_terminal_gate() -> None:
    source = SCRIPT.read_text()
    assert 'pre_execution_amendment["execution_state_at_freeze"]' in source
    assert '"mixed_training_started"' in source
    gate_position = source.index("for target in targets[:2]")
    down_position = source.index("candidate_down_input =")
    aff3_position = source.index("calibration_only_hard_forward_trust_region(")
    confirmation_boundary = source.index("fresh confirmation remains sealed")
    assert gate_position < down_position < aff3_position < confirmation_boundary
    assert 'raise SystemExit(0 if gate["passed"] else 2)' in source
    assert "minimum_mean_relative_calibration_nmse_reduction_vs_direct_mixed" in source
    assert "maximum_weighted_code_churn_fraction" in source
    assert "storage_projection_within_maximum" in source
    assert "expected_source_sha256" in source
    assert "exact BF16 layer no longer reproduces the training cache" in source
