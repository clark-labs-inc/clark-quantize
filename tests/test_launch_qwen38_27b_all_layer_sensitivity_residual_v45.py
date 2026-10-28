from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / (
    "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
    "launch_qwen38_27b_all_layer_sensitivity_residual_v45.sh"
)


def test_v45_launcher_is_parent_serialized_and_resource_bounded() -> None:
    source = SCRIPT.read_text()

    assert "refusing to overlap v45 residual fitting with active v44 training" in source
    assert "MemoryMax=31000000000" in source
    assert "MemorySwapMax=0" in source
    assert "AllowedCPUs=0,2,4,6" in source
    assert "CPUQuota=400%" in source
    assert "RestartPreventExitStatus=2" in source
    assert "StartLimitBurst=3" in source


def test_v45_launcher_binds_frozen_runner_and_complete_parent_denominator() -> None:
    source = SCRIPT.read_text()

    assert 'frozen.get("runner_repository_commit") != head' in source
    assert 'frozen.get("runner_sha256") != sha256(runner)' in source
    assert 'frozen.get("training_receipt_sha256") != sha256(training_receipt_path)' in source
    assert 'len(layers) != 64' in source
    assert 'list(range(64))' in source
    assert 'training_receipt.get("gate", {}).get("passed") is not True' in source
    assert '"event": "v45_launch_identity_passed"' in source


def test_v45_launcher_remains_qwen38_only_and_uses_the_frozen_runner() -> None:
    source = SCRIPT.read_text()

    assert "Qwen/Qwen3.8-27B" in source
    assert "all_layer_sensitivity_residual_v45.py" in source
    assert "--training-receipt" in source
    assert "--threads 4" in source
    assert "qwen35|qwen3[.]5|4b" in source
