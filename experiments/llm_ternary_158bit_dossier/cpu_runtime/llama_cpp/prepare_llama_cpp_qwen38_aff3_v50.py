#!/usr/bin/env python3
"""Apply the v50 graph-result lifetime repair after the frozen v49 transform."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


HERE = Path(__file__).resolve().parent
V49_PATH = HERE / "prepare_llama_cpp_qwen38_aff3_v49.py"


def load_v49() -> ModuleType:
    spec = importlib.util.spec_from_file_location("qwen38_aff3_v49", V49_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen v49 transformer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V49 = load_v49()
LLAMA_REVISION = V49.LLAMA_REVISION
FROZEN_FILES = V49.FROZEN_FILES


def repair_transformed_qwen_source(text: str) -> str:
    """Move callback state from the temporary builder into the graph result."""
    text = V49.replace_exact(
        text,
        "#include <stdexcept>\n",
        "#include <stdexcept>\n#include <utility>\n",
        "v50 utility include",
    )
    owner = """namespace {

class qwen38_aff3_graph_owner final : public llm_graph_input_i {
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

}  // namespace

"""
    text = V49.replace_exact(
        text,
        "void llama_model_qwen35::load_arch_hparams(llama_model_loader & ml) {\n",
        owner
        + "void llama_model_qwen35::load_arch_hparams(llama_model_loader & ml) {\n",
        "v50 graph-result owner",
    )
    old_tail = """    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
"""
    new_tail = """    cb(cur, "result_output", -1);
    res->t_logits = cur;

    if (!aff3_node_states.empty()) {
        res->add_input(std::make_unique<qwen38_aff3_graph_owner>(
            std::move(aff3_node_states)));
    }

    ggml_build_forward_expand(gf, cur);
}
"""
    return V49.replace_exact(
        text,
        old_tail,
        new_tail,
        "v50 graph-result ownership transfer",
    )


def transform_qwen_source(text: str) -> str:
    return repair_transformed_qwen_source(V49.transform_qwen_source(text))


def main() -> None:
    # Reuse the exact v49 revision/hash/copy machinery while replacing only
    # the Qwen source transform with the versioned lifetime repair.
    V49.transform_qwen_source = transform_qwen_source
    V49.main()


if __name__ == "__main__":
    main()
