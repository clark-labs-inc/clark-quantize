from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
LLAMA_RUNTIME = RUNTIME / "llama_cpp"
PREPARE = LLAMA_RUNTIME / "prepare_llama_cpp_qwen38_aff3_v49.py"
IMPLEMENTATION = LLAMA_RUNTIME / "qwen38-aff3-runtime.cpp"
SMOKE = LLAMA_RUNTIME / "qwen38-aff3-runtime-smoke.cpp"
PREREG = RUNTIME / "qwen38_27b_aff3_llamacpp_v49_development_preregistration.json"


def load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_aff3_v49", PREPARE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preregistration_forbids_performance_and_defers_real_model() -> None:
    prereg = json.loads(PREREG.read_text())

    assert prereg["model"] == "Qwen/Qwen3.8-27B"
    assert prereg["development_gate"]["transformed_llama_tree_must_compile"] is True
    assert prereg["development_gate"]["one_and_eight_token_output_bit_exact_vs_v47"] is True
    assert prereg["development_gate"]["performance_timing_permitted"] is False
    assert prereg["development_gate"][
        "actual_qwen_model_load_and_graph_execution_deferred_until_v43_releases_memory"
    ] is True
    assert prereg["integration_contract"]["persistent_dense_weight_bytes"] == 0
    assert prereg["integration_contract"]["decoded_group_scale_matrix_bytes"] == 0


def test_transform_freezes_revision_and_source_local_isa_flags() -> None:
    module = load_prepare_module()
    assert module.LLAMA_REVISION == "4df29be4f4c3673f428170fda944a5b19f743bb8"
    assert set(module.FROZEN_FILES) == {
        "src/CMakeLists.txt",
        "src/models/models.h",
        "src/models/qwen35.cpp",
    }
    source = 'file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")\n\nadd_library(llama'
    transformed = module.transform_cmake(source)
    assert transformed.count("set_source_files_properties") == 1
    assert "models/qwen38-aff3-runtime.cpp" in transformed
    for flag in ("-mavx2", "-mf16c", "-mfma", "-mssse3"):
        assert flag in transformed


def test_qwen_transform_is_fail_closed_and_replaces_both_projection_types() -> None:
    source = PREPARE.read_text()

    assert source.count("aff3_images[il] ? TENSOR_SKIP : flags") >= 2
    assert "model.layers[il].wo" in source
    assert "model.layers[il].ssm_out" in source
    assert source.count("qwen38_aff3_build_projection") >= 2
    assert "does not silently bypass LoRA" in source
    assert "if seam_count != 2 or tensor_skip_count != 2:" in source
    assert "hparams.n_layer() != 64 || hparams.n_embd != 5120" in source
    assert "source_must_be_clean" not in source
    assert 'status = subprocess.check_output' in source
    assert 'if revision != LLAMA_REVISION or status:' in source


def test_runtime_authenticates_mmap_and_partitions_token_tile_product() -> None:
    source = IMPLEMENTATION.read_text()

    assert "PROT_READ, MAP_PRIVATE" in source
    assert "PROT_WRITE" not in source
    assert "runtime root SHA-256 mismatch" in source
    assert "reserved ternary code 3" in source
    assert "stored pairing is not a perfect matching" in source
    assert "constexpr int kGraphTasks = 4" in source
    assert "const int64_t total_jobs = static_cast<int64_t>(tokens) * tiles" in source
    assert "ggml_map_custom2" in source
    assert "std::vector<float> cosines" in source
    assert "std::vector<float> sines" in source
    assert "std::vector<float> weights" not in source
    assert "std::vector<float> scales" in source  # per-task A8 activation scales only


def test_smoke_contract_observes_four_tasks_without_timing() -> None:
    source = SMOKE.read_text()

    assert "QWEN38_AFF3_DEVELOPMENT_TESTING" in source
    assert "observed_task_mask" in source
    assert "task_mask != UINT32_C(15)" in source
    assert "output_bit_exact_vs_direct_v47" in source
    assert 'options.tokens != 1 && options.tokens != 8' in source
    assert "performance_timing_ran\\\":false" in source
