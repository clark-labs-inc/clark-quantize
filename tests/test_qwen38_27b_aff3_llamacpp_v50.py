from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREPARE = RUNTIME / "llama_cpp/prepare_llama_cpp_qwen38_aff3_v50.py"
PREREG = RUNTIME / "qwen38_27b_aff3_llamacpp_v50_repair_preregistration.json"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_aff3_v50", PREPARE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transformed_fixture() -> str:
    return """#include <stdexcept>
void llama_model_qwen35::load_arch_hparams(llama_model_loader & ml) {
}
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
"""


def test_preregistration_freezes_failure_and_ownership_only_repair() -> None:
    prereg = json.loads(PREREG.read_text())

    assert prereg["model"] == "Qwen/Qwen3.8-27B"
    assert prereg["parent_failure"]["classification"] == (
        "callback_userdata_lifetime_violation"
    )
    assert prereg["repair_contract"]["numerical_code_changes_permitted"] is False
    assert prereg["repair_contract"]["callback_state_owner"].startswith(
        "llm_graph_result"
    )
    assert prereg["development_gate"]["performance_timing_permitted"] is False


def test_repair_transfers_state_into_graph_result_owner() -> None:
    module = load_module()
    transformed = module.repair_transformed_qwen_source(transformed_fixture())

    assert "class qwen38_aff3_graph_owner final" in transformed
    assert "public llm_graph_input_i" in transformed
    assert "void set_input(const llama_ubatch *) override {}" in transformed
    assert "bool can_reuse(const llm_graph_params &) override" in transformed
    assert "res->add_input(std::make_unique<qwen38_aff3_graph_owner>" in transformed
    assert "std::move(aff3_node_states)" in transformed
    assert "#include <utility>" in transformed


def test_repair_is_fail_closed_when_transfer_seam_changes() -> None:
    module = load_module()

    with pytest.raises(RuntimeError, match="ownership transfer"):
        module.repair_transformed_qwen_source(
            transformed_fixture().replace("res->t_logits = cur;", "")
        )


def test_successor_keeps_exact_v49_revision_and_frozen_file_scope() -> None:
    module = load_module()

    assert module.LLAMA_REVISION == "4df29be4f4c3673f428170fda944a5b19f743bb8"
    assert set(module.FROZEN_FILES) == {
        "src/CMakeLists.txt",
        "src/models/models.h",
        "src/models/qwen35.cpp",
    }
