from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / (
    "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
    "run_qwen38_27b_aff3_llamacpp_model_smoke_v50.sh"
)


def test_actual_model_repair_smoke_is_hash_bound_and_untimed() -> None:
    source = SCRIPT.read_text()

    assert "qwen38_27b_aff3_llamacpp_model_smoke_v50" in source
    assert "qwen38_27b_aff3_llamacpp_v50_repair_preregistration.json" in source
    assert "6953d033c467029db896efa528b6261eb04999b94d3e731952b75f4ffa5e36d5" in source
    assert "8205a610522110f839787bdc82b36402ce85582929f076f9e52a6e45908231d6" in source
    assert "6300e5ab7c8e6fec83b720dff9ed685986158d47c9ebb9aa3d50f8efc0fe7088" in source
    assert "--n-predict 8" in source
    assert "--no-warmup" in source
    assert "--no-perf" in source
    assert '"performance_timing_ran": performance_timing_ran' in source


def test_actual_model_repair_smoke_preserves_exclusivity_and_failure_receipt() -> None:
    source = SCRIPT.read_text()

    assert "all-layer-mixed-direct-v43" in source
    assert "all-layer-mixed-trained-v44" in source
    assert "MemoryMax=30000000000" in source
    assert "MemorySwapMax=0" in source
    assert "AllowedCPUs=0,2,4,6" in source
    assert 'if [[ -s "$receipt" ]]' in source
    assert "existing v50 model-smoke receipt changed" in source
    assert 'raise SystemExit(0 if receipt.get("passed") is True else 2)' in source


def test_actual_model_repair_smoke_binds_all_runtime_libraries() -> None:
    source = SCRIPT.read_text()

    for name in (
        "libllama.so",
        "libggml-cpu.so",
        "libggml-base.so",
        "libggml.so",
        "libllama-cli-impl.so",
        "libllama-common.so",
        "libllama-server-impl.so",
        "libmtmd.so",
    ):
        assert f'require_sha256 "$runtime_lib_dir/{name}"' in source
