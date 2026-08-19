from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
RUNNER = RUNTIME / "run_qwen38_27b_trained_compact_materialization_v62.sh"
LAUNCHER = RUNTIME / "launch_qwen38_27b_trained_compact_materialization_v62.sh"


def test_v62_materialization_runner_is_fail_closed_and_restartable() -> None:
    source = RUNNER.read_text()
    assert "QWEN38_RESIDUAL_DIR" in source
    assert "v62 residual-free candidate refuses residual environment variables" in source
    assert "expected_training=1f33a04c" in source
    assert "expected_direct=3a3179b1" in source
    assert "sha256sum -c \"$template_manifest\"" in source
    assert "existing v62 CQ3 materialization changed" in source
    assert "layer_count\") != 64" in source
    assert '"residual_payload_bytes": 0' in source
    assert "partial_then_atomic_rename_required" not in source


def test_v62_materialization_launcher_bounds_resources_and_overlap() -> None:
    source = LAUNCHER.read_text()
    assert "another Qwen experiment is active" in source
    assert "--property=CPUQuota=400%" in source
    assert "--property=AllowedCPUs=0,2,4,6" in source
    assert "--property=MemoryMax=20000000000" in source
    assert "--property=MemorySwapMax=0" in source
    assert "--property=Restart=on-failure" in source
    assert "--property=RestartPreventExitStatus=2" in source
