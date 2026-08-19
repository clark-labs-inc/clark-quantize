from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / (
    "experiments/llm_ternary_158bit_dossier/cpu_runtime/llama_cpp/"
    "prepare_llama_cpp_qwen38_cq3_aff3_residual_v57.py"
)
RUNTIME = SOURCE.parent / "qwen38-residual-runtime.cpp"
SPEC = importlib.util.spec_from_file_location("prepare_residual_v57", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def qwen_fixture() -> str:
    return """class qwen38_aff3_graph_owner final : public llm_graph_input_i {
public:
    explicit qwen38_aff3_graph_owner(
            std::vector<qwen38_aff3_node_state_ptr> states_value)
        : states(std::move(states_value)) {}

    void set_input(const llama_ubatch *) override {}

    bool can_reuse(const llm_graph_params &) override {
        return true;
    }

private:
    std::vector<qwen38_aff3_node_state_ptr> states;
};

    if (qwen38_aff3_environment_requested()) {
        if (hparams.n_layer() != 64 || hparams.n_embd != 5120) {
            throw std::runtime_error("QWEN38_AFF3 is valid only for Qwen3.8-27B");
        }
        aff3_images = qwen38_aff3_load_from_environment(64, 5120, 6144);
    } else {
        aff3_images.assign(hparams.n_layer(), nullptr);
    }
    if (qwen38_aff3_any_enabled(qwen_model.aff3_images) && !model.loras.empty()) {
        throw std::runtime_error("Qwen3.8 AFF3 projection does not silently bypass LoRA");
    }
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        cur = build_cvec(cur, il);
    if (!aff3_node_states.empty()) {
        res->add_input(std::make_unique<qwen38_aff3_graph_owner>(
            std::move(aff3_node_states)));
    }

    ggml_build_forward_expand(gf, cur);
"""


def test_v57_places_residual_after_complete_block_before_control_vector() -> None:
    transformed = MODULE.transform_qwen_source(qwen_fixture())

    post_ffn = transformed.index('cb(cur, "post_ffn", il);')
    residual = transformed.index("qwen38_residual_apply(")
    control = transformed.index("cur = build_cvec(cur, il);")
    assert post_ffn < residual < control
    assert "ggml_tensor * block_input = inpL;" in transformed
    assert "block_input = inpSA;" in transformed
    assert "residual_node_states" in transformed
    assert "qwen38_residual_graph_owner" in transformed


def test_v57_model_header_owns_bundle_and_callback_lifetime() -> None:
    fixture = """#include "qwen38-aff3-runtime.h"
    std::vector<qwen38_aff3_image_ptr> aff3_images;
        std::vector<qwen38_aff3_node_state_ptr> aff3_node_states;
        const llama_model & model;
"""
    transformed = MODULE.transform_models_header(fixture)

    assert '#include "qwen38-residual-runtime.h"' in transformed
    assert "qwen38_residual_bundle_ptr residual_bundle;" in transformed
    assert "std::vector<qwen38_residual_node_state_ptr> residual_node_states;" in transformed


def test_v57_runtime_requires_parent_hashes_and_compact_native_algebra() -> None:
    source = RUNTIME.read_text()

    for variable in (
        "QWEN38_RESIDUAL_TRAINING_SHA256",
        "QWEN38_RESIDUAL_RECEIPT_SHA256",
        "QWEN38_RESIDUAL_PREREG_SHA256",
    ):
        assert variable in source
    assert "mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE" in source
    assert "cq3_calibration_residual_f32_avx2_trusted" in source
    assert "float scratch[kRank]" in source
    assert "std::vector<float>" not in source
    assert "callback_abort" in source


def test_v57_cmake_compiles_native_c_and_runtime_with_amd_flags() -> None:
    fixture = """file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(models/qwen38-aff3-runtime.cpp PROPERTIES
        COMPILE_OPTIONS "-mavx2;-mf16c;-mfma;-mssse3")
endif()
"""
    transformed = MODULE.transform_cmake(fixture)

    assert "qwen38_calibration_residual_cpu.c" in transformed
    assert "models/qwen38-residual-runtime.cpp" in transformed
    assert 'COMPILE_OPTIONS "-mavx2;-mf16c;-mfma;-mssse3"' in transformed
