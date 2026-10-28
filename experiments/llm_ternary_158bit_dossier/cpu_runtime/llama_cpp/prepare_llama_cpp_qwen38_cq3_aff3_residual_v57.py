#!/usr/bin/env python3
"""Attach the compact v45 post-block residual to the combined v54 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CPU_RUNTIME = HERE.parent
V54_PREPARE = HERE / "prepare_llama_cpp_qwen38_cq3_aff3_v54.py"
LLAMA_REVISION = "4df29be4f4c3673f428170fda944a5b19f743bb8"
FROZEN_V54_FILES = {
    "src/CMakeLists.txt": "8d24ff0114fe15a585499f07434c136d3ba652e0d4ece1c3904d1866847f516e",
    "src/models/models.h": "67724d24225496ed52a36ea530307798d810fc2a60080870b0bca1dcef53e4ab",
    "src/models/qwen35.cpp": "6953d033c467029db896efa528b6261eb04999b94d3e731952b75f4ffa5e36d5",
}
RUNTIME_FILES = {
    HERE / "qwen38-residual-runtime.h": (
        "9eb335e357346c77b69be6bbd15ab7103444a236bcef0f119ddb70c1a555aa75"
    ),
    HERE / "qwen38-residual-runtime.cpp": (
        "eae93366e9b2b23fc33ea353a580a0aec6c23ab028452d59beffe628e0674169"
    ),
    REPO / "clark_quantize/qwen38_calibration_residual_cpu.h": (
        "d53f8c2d4c14f306da57c35f3efc3d93d27bc5a8969bf472ad8dbf8e50d1a4af"
    ),
    REPO / "clark_quantize/qwen38_calibration_residual_cpu.c": (
        "dc99b3899c8c9946d4a9e38fe07bafd31161fc1df6ca62ebd99989aaf748137a"
    ),
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, observed {count}")
    return text.replace(old, new)


def transform_cmake(text: str) -> str:
    text = replace_exact(
        text,
        'file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")\n\n',
        'file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")\n'
        'list(APPEND LLAMA_MODELS_SOURCES\n'
        '    "models/qwen38-residual-kernels/qwen38_calibration_residual_cpu.c")\n\n',
        "residual native source",
    )
    return replace_exact(
        text,
        "    set_source_files_properties(models/qwen38-aff3-runtime.cpp PROPERTIES\n"
        '        COMPILE_OPTIONS "-mavx2;-mf16c;-mfma;-mssse3")\n',
        "    set_source_files_properties(\n"
        "        models/qwen38-aff3-runtime.cpp\n"
        "        models/qwen38-residual-runtime.cpp\n"
        "        models/qwen38-residual-kernels/qwen38_calibration_residual_cpu.c\n"
        "        PROPERTIES COMPILE_OPTIONS \"-mavx2;-mf16c;-mfma;-mssse3\")\n",
        "residual ISA flags",
    )


def transform_models_header(text: str) -> str:
    text = replace_exact(
        text,
        '#include "qwen38-aff3-runtime.h"\n',
        '#include "qwen38-aff3-runtime.h"\n#include "qwen38-residual-runtime.h"\n',
        "residual runtime include",
    )
    text = replace_exact(
        text,
        "    std::vector<qwen38_aff3_image_ptr> aff3_images;\n",
        "    std::vector<qwen38_aff3_image_ptr> aff3_images;\n"
        "    qwen38_residual_bundle_ptr residual_bundle;\n",
        "residual model ownership",
    )
    return replace_exact(
        text,
        "        std::vector<qwen38_aff3_node_state_ptr> aff3_node_states;\n"
        "        const llama_model & model;\n",
        "        std::vector<qwen38_aff3_node_state_ptr> aff3_node_states;\n"
        "        std::vector<qwen38_residual_node_state_ptr> residual_node_states;\n"
        "        const llama_model & model;\n",
        "residual graph ownership",
    )


def transform_qwen_source(text: str) -> str:
    owner_anchor = """class qwen38_aff3_graph_owner final : public llm_graph_input_i {
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

"""
    residual_owner = """class qwen38_residual_graph_owner final : public llm_graph_input_i {
public:
    explicit qwen38_residual_graph_owner(
            std::vector<qwen38_residual_node_state_ptr> states_value)
        : states(std::move(states_value)) {}

    void set_input(const llama_ubatch *) override {}

    bool can_reuse(const llm_graph_params &) override {
        return true;
    }

private:
    std::vector<qwen38_residual_node_state_ptr> states;
};

"""
    text = replace_exact(
        text, owner_anchor, owner_anchor + residual_owner, "residual graph owner"
    )
    aff3_load = """    if (qwen38_aff3_environment_requested()) {
        if (hparams.n_layer() != 64 || hparams.n_embd != 5120) {
            throw std::runtime_error("QWEN38_AFF3 is valid only for Qwen3.8-27B");
        }
        aff3_images = qwen38_aff3_load_from_environment(64, 5120, 6144);
    } else {
        aff3_images.assign(hparams.n_layer(), nullptr);
    }
"""
    residual_load = """

    if (qwen38_residual_environment_requested()) {
        residual_bundle = qwen38_residual_load_from_environment(64, 5120, 5120);
    } else {
        residual_bundle.reset();
    }
"""
    text = replace_exact(
        text, aff3_load, aff3_load + residual_load, "residual bundle loading"
    )
    text = replace_exact(
        text,
        "    if (qwen38_aff3_any_enabled(qwen_model.aff3_images) && !model.loras.empty()) {\n"
        '        throw std::runtime_error("Qwen3.8 AFF3 projection does not silently bypass LoRA");\n'
        "    }\n",
        "    if ((qwen38_aff3_any_enabled(qwen_model.aff3_images) ||\n"
        "         qwen38_residual_any_enabled(qwen_model.residual_bundle)) &&\n"
        "        !model.loras.empty()) {\n"
        '        throw std::runtime_error("Qwen3.8 compact runtime does not silently bypass LoRA");\n'
        "    }\n",
        "combined LoRA guard",
    )
    text = replace_exact(
        text,
        "        res->t_layer_inp[il] = inpL;\n\n"
        "        ggml_tensor * inpSA = inpL;\n",
        "        res->t_layer_inp[il] = inpL;\n\n"
        "        ggml_tensor * block_input = inpL;\n"
        "        ggml_tensor * inpSA = inpL;\n",
        "saved complete-block input",
    )
    text = replace_exact(
        text,
        "            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);\n"
        "            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);\n",
        "            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);\n"
        "            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);\n"
        "            block_input = inpSA;\n",
        "masked final-layer residual input",
    )
    post_ffn = """        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

"""
    post_residual = """        if (qwen38_residual_layer_enabled(qwen_model.residual_bundle, il)) {
            cur = qwen38_residual_apply(
                ctx0, cur, block_input, qwen_model.residual_bundle, il,
                residual_node_states);
            cb(cur, "post_block_residual", il);
        }

"""
    text = replace_exact(
        text,
        post_ffn + "        cur = build_cvec(cur, il);\n",
        post_ffn + post_residual + "        cur = build_cvec(cur, il);\n",
        "post-block residual placement",
    )
    aff3_tail = """    if (!aff3_node_states.empty()) {
        res->add_input(std::make_unique<qwen38_aff3_graph_owner>(
            std::move(aff3_node_states)));
    }

"""
    residual_tail = """    if (!residual_node_states.empty()) {
        res->add_input(std::make_unique<qwen38_residual_graph_owner>(
            std::move(residual_node_states)));
    }

"""
    return replace_exact(
        text,
        aff3_tail + "    ggml_build_forward_expand(gf, cur);\n",
        aff3_tail + residual_tail + "    ggml_build_forward_expand(gf, cur);\n",
        "residual graph-result ownership transfer",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-source", type=Path, required=True)
    args = parser.parse_args()
    source = args.llama_source.resolve()
    for path, expected in RUNTIME_FILES.items():
        if sha256_path(path) != expected:
            raise RuntimeError(f"frozen residual runtime input changed: {path.name}")

    v54_output = subprocess.check_output(
        [sys.executable, str(V54_PREPARE), "--llama-source", str(source)],
        text=True,
    )
    v54_receipt = json.loads(v54_output)
    if not v54_receipt.get("complete") or not v54_receipt.get("terminal"):
        raise RuntimeError("v54 parent transform did not complete")
    for relative, expected in FROZEN_V54_FILES.items():
        if sha256_path(source / relative) != expected:
            raise RuntimeError(f"frozen v54 source changed: {relative}")

    transformed = {
        source / "src/CMakeLists.txt": transform_cmake(
            (source / "src/CMakeLists.txt").read_text()
        ),
        source / "src/models/models.h": transform_models_header(
            (source / "src/models/models.h").read_text()
        ),
        source / "src/models/qwen35.cpp": transform_qwen_source(
            (source / "src/models/qwen35.cpp").read_text()
        ),
    }
    for path, content in transformed.items():
        temporary = path.with_suffix(path.suffix + ".v57.tmp")
        temporary.write_text(content)
        os.replace(temporary, path)

    copied: list[Path] = []
    for name in ("qwen38-residual-runtime.h", "qwen38-residual-runtime.cpp"):
        target = source / "src/models" / name
        if target.exists():
            raise FileExistsError(target)
        shutil.copyfile(HERE / name, target)
        copied.append(target)
    kernel_root = source / "src/models/qwen38-residual-kernels"
    kernel_root.mkdir()
    for name in (
        "qwen38_calibration_residual_cpu.h",
        "qwen38_calibration_residual_cpu.c",
    ):
        target = kernel_root / name
        shutil.copyfile(REPO / "clark_quantize" / name, target)
        copied.append(target)

    subprocess.check_call(["git", "diff", "--check"], cwd=source)
    tracked_diff = subprocess.check_output(
        ["git", "diff", "--binary"], cwd=source
    )
    result = {
        "schema": "qwen38_27b_cq3_aff3_residual_llamacpp_transform_v57",
        "model": "Qwen/Qwen3.8-27B",
        "llama_revision": LLAMA_REVISION,
        "parent_v54": v54_receipt,
        "transformed_sha256": {
            str(path.relative_to(source)): sha256_path(path)
            for path in (*transformed.keys(), *copied)
        },
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "placement": "after_complete_block_before_control_vector",
        "active_layer_nodes_only": True,
        "persistent_dense_residual_bytes": 0,
        "complete": True,
        "terminal": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
