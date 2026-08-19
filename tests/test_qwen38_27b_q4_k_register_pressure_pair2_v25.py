import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "experiments" / "llm_ternary_158bit_dossier" / "cpu_runtime"
PREPARE_PATH = RUNTIME / "kernels" / "prepare_llama_cpp_q4_k_register_pressure_pair2_v25.py"
PREREG_PATH = RUNTIME / "qwen38_27b_q4_k_avx2_register_pressure_pair2_v25.json"


def load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_pair2_v25", PREPARE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def synthetic_parent(module) -> str:
    fp_zeros = (
        "_mm256_setzero_ps(), _mm256_setzero_ps(), "
        "_mm256_setzero_ps(), _mm256_setzero_ps(),"
    )
    int_zeros = (
        "_mm256_setzero_si256(), _mm256_setzero_si256(), "
        "_mm256_setzero_si256(), _mm256_setzero_si256(),"
    )
    return f"""prefix
{module.FUNCTION_HEADER}
    assert(nr > 0 && nr <= 4);

    // Decode each Q4_K weight block once and reuse it across all ragged
    __m256 acc_rows[4] = {{ {fp_zeros} }};
    __m256 acc_min_rows[4] = {{ {fp_zeros} }};
    __m256i iacc_b[4] = {{ {int_zeros} }};
    __m256i iacc_min_b[4] = {{ {int_zeros} }};
    __m256i q8s[4];
}}

{module.NEXT_FUNCTION_HEADER}int n) {{}}
"""


def test_source_transform_specializes_only_the_new_helper():
    module = load_prepare_module()
    candidate = module.transform(synthetic_parent(module))

    assert candidate.count(module.FUNCTION_HEADER) == 1
    assert candidate.count(module.HELPER_NAME) == 3
    helper, public = candidate.split(module.FUNCTION_HEADER, maxsplit=1)
    for declaration in ("acc_rows", "acc_min_rows", "iacc_b", "iacc_min_b", "q8s"):
        assert f"{declaration}[2]" in helper
        assert f"{declaration}[4]" in public
    assert "constexpr int nr = 2;" in helper
    assert "constexpr int nc = 8;" in helper
    assert "if (nr == 4)" in public
    assert "a_ptr_start + 2 * b_nb" in public
    assert "s + 2 * bs + x * 8" in public


def test_preregistration_freezes_bit_exact_and_measured_gates():
    prereg = json.loads(PREREG_PATH.read_text())
    contract = prereg["implementation_contract"]
    gates = prereg["sequential_gates"]

    assert contract["integer_dot_order_per_row_unchanged"] is True
    assert contract["floating_accumulation_order_per_row_unchanged"] is True
    assert contract["no_approximation_or_weight_change"] is True
    assert gates["gate_0_source_property_backend_and_codegen"]["candidate_outputs_must_be_bit_exact_to_parent"] is True
    assert gates["gate_0_source_property_backend_and_codegen"]["maximum_pair2_specialization_stack_frame_bytes"] == 768
    assert gates["gate_1_paired_native_microkernel"]["measured_pairs_per_shape"] == 31
    assert gates["gate_1_paired_native_microkernel"]["minimum_geometric_mean_parent_over_candidate_latency_ratio"] == 1.08
    assert gates["gate_2_paired_real_model_four_row_graph"]["minimum_parent_over_candidate_latency_ratio"] == 1.05
    assert gates["gate_3_target_and_mtp3_system"]["minimum_candidate_over_parent_mtp3_accepted_tps"] == 1.04
