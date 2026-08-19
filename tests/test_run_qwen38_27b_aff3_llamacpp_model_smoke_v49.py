from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "run_qwen38_27b_aff3_llamacpp_model_smoke_v49.sh"
)


def test_model_smoke_binds_exact_runtime_model_and_sidecar() -> None:
    source = SCRIPT.read_text()

    assert "1bd2b6d3d5f0b1b988508bb6682cd9af896b40da46422200a47b887a7c4ae600" in source
    assert "dc4860ba93c9e2df68e001aa7dba159b062edc7297555b80ad67651e978bd83f" in source
    assert "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34" in source
    assert "a4cec7372ac21a0093d047c1452277508583ebca1e3f4f66c73a8a74142ccf48" in source
    assert "QWEN38_AFF3_LAYERS=0" in source
    assert "layer-00.aff3x16" in source


def test_model_smoke_forbids_overlap_and_performance_timing() -> None:
    source = SCRIPT.read_text()

    assert "all-layer-mixed-direct-v43.service" in source
    assert "all-layer-mixed-trained-v44.service" in source
    assert "--property=AllowedCPUs=0,2,4,6" in source
    assert "--property=MemoryMax=30000000000" in source
    assert "--no-warmup" in source
    assert "--no-perf" in source
    assert '"performance_timing_ran": performance_timing_ran' in source
    assert '"tokens per second"' in source


def test_model_smoke_is_bounded_deterministic_free_generation() -> None:
    source = SCRIPT.read_text()

    assert "--n-predict 8" in source
    assert "--seed 20260819" in source
    assert "--temp 0" in source
    assert "--ignore-eos" in source
    assert '"bounded_free_generation_nonempty": bool(stdout_bytes.strip())' in source
    assert "This development smoke can establish only" in source
