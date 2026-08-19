from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "launch_qwen38_27b_all_layer_mixed_train_v44.sh"
)


def test_launcher_requires_passed_frozen_parent_and_exact_runner() -> None:
    source = SCRIPT.read_text()

    assert "frozen_runner.get(\"repository_commit\") != head" in source
    assert "frozen_runner.get(\"sha256\") != sha256(runner)" in source
    assert 'direct_receipt.get("gate", {}).get("passed") is not True' in source
    assert 'len(direct_receipt.get("layers", [])) != 64' in source
    assert "refusing to overlap v44 training with active v43 parent" in source


def test_launcher_pins_physical_cores_memory_and_restart_budget() -> None:
    source = SCRIPT.read_text()

    assert "--property=CPUQuota=400%" in source
    assert "--property=AllowedCPUs=0,2,4,6" in source
    assert "--property=MemoryMax=31000000000" in source
    assert "--property=MemorySwapMax=0" in source
    assert "--property=Restart=on-failure" in source
    assert "--property=RestartPreventExitStatus=2" in source
    assert "--property=StartLimitBurst=3" in source
    assert "/usr/bin/taskset -c 0,2,4,6" in source


def test_launcher_passes_explicit_durable_paths() -> None:
    source = SCRIPT.read_text()

    assert '--preregistration "$preregistration"' in source
    assert '--direct-receipt "$direct_receipt"' in source
    assert '--full-context-receipt "$full_context_receipt"' in source
    assert '--artifact-root "$artifact_root"' in source
    assert '--receipt "$receipt"' in source
    assert '--completion-marker "$completion_marker"' in source
    assert "--threads 4" in source
