#!/usr/bin/env python3
"""Apply the frozen Qwen3.8 AFF3 sidecar integration to llama.cpp 4df29be."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


LLAMA_REVISION = "4df29be4f4c3673f428170fda944a5b19f743bb8"
FROZEN_FILES = {
    "src/CMakeLists.txt": "3f9c9046c1bd19df660f644b164c81aa9f643ebe2d62a0174b7bb39af2af7170",
    "src/models/models.h": "f56211fd74dc21fc35294ce93103ad3b4cbd29daf2aeea5462b9d762f6d6179b",
    "src/models/qwen35.cpp": "b61c5813fc80c54269d45cbbdbbc34b090500e8017dc190483605b37249d007c",
}
RUNTIME_FILES = (
    "qwen38-aff3-runtime.h",
    "qwen38-aff3-runtime.cpp",
)
KERNEL_FILES = (
    "qwen38_affine_avx2_x16_fused_v17.cpp",
    "qwen38_affine_avx2_register_lookup_v18.cpp",
    "qwen38_aff3_rotated_avx2_v47.cpp",
)


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


def transform_models_header(text: str) -> str:
    text = replace_exact(
        text,
        '#include "llama-model-loader.h"\n',
        '#include "llama-model-loader.h"\n#include "qwen38-aff3-runtime.h"\n',
        "runtime header include",
    )
    start = text.index("struct llama_model_qwen35 : public llama_model_base {")
    stop = text.index("struct llama_model_qwen35moe : public llama_model_base {", start)
    before, block, after = text[:start], text[start:stop], text[stop:]
    block = replace_exact(
        block,
        "    void load_arch_tensors(llama_model_loader & ml) override;\n\n"
        "    struct graph : public llm_build_delta_net_base {",
        "    void load_arch_tensors(llama_model_loader & ml) override;\n\n"
        "    std::vector<qwen38_aff3_image_ptr> aff3_images;\n\n"
        "    struct graph : public llm_build_delta_net_base {",
        "Qwen model image ownership",
    )
    block = replace_exact(
        block,
        "        const llama_model & model;\n",
        "        std::vector<qwen38_aff3_node_state_ptr> aff3_node_states;\n"
        "        const llama_model & model;\n",
        "Qwen graph callback ownership",
    )
    return before + block + after


def transform_cmake(text: str) -> str:
    return replace_exact(
        text,
        'file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")\n\nadd_library(llama',
        'file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")\n\n'
        'if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")\n'
        '    set_source_files_properties(models/qwen38-aff3-runtime.cpp PROPERTIES\n'
        '        COMPILE_OPTIONS "-mavx2;-mf16c;-mfma;-mssse3")\n'
        'endif()\n\n'
        'add_library(llama',
        "AFF3 per-source ISA flags",
    )


def transform_qwen_source(text: str) -> str:
    text = replace_exact(
        text,
        '#include "llama-memory-recurrent.h"\n',
        '#include "llama-memory-recurrent.h"\n\n#include <stdexcept>\n',
        "stdexcept include",
    )
    switch = """    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_8B : LLM_TYPE_2B; break;
        case 32: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_9B; break;
        case 64: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
"""
    switch_with_runtime = switch + """
    if (qwen38_aff3_environment_requested()) {
        if (hparams.n_layer() != 64 || hparams.n_embd != 5120) {
            throw std::runtime_error("QWEN38_AFF3 is valid only for Qwen3.8-27B");
        }
        aff3_images = qwen38_aff3_load_from_environment(64, 5120, 6144);
    } else {
        aff3_images.assign(hparams.n_layer(), nullptr);
    }
"""
    text = replace_exact(text, switch, switch_with_runtime, "AFF3 model loading")
    text = replace_exact(
        text,
        '            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);',
        '            const int projection_flags = aff3_images[il] ? TENSOR_SKIP : flags;\n'
        '            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, projection_flags);',
        "full-attention tensor skip",
    )
    text = replace_exact(
        text,
        '            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);',
        '            const int projection_flags = aff3_images[il] ? TENSOR_SKIP : flags;\n'
        '            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, projection_flags);',
        "linear-attention tensor skip",
    )
    constructor = """    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
"""
    constructor_with_guard = """    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const auto & qwen_model = static_cast<const llama_model_qwen35 &>(model);
    if (qwen38_aff3_any_enabled(qwen_model.aff3_images) && !model.loras.empty()) {
        throw std::runtime_error("Qwen3.8 AFF3 projection does not silently bypass LoRA");
    }

    int sections[4];
"""
    text = replace_exact(
        text, constructor, constructor_with_guard, "LoRA fail-closed guard"
    )
    text = replace_exact(
        text,
        "    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);\n"
        "    cb(cur, \"attn_output\", il);",
        "    const auto & qwen_model = static_cast<const llama_model_qwen35 &>(model);\n"
        "    if (qwen_model.aff3_images[il]) {\n"
        "        cur = qwen38_aff3_build_projection(\n"
        "            ctx0, cur, qwen_model.aff3_images[il], aff3_node_states);\n"
        "    } else {\n"
        "        cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);\n"
        "    }\n"
        "    cb(cur, \"attn_output\", il);",
        "full-attention graph seam",
    )
    text = replace_exact(
        text,
        "    // Output projection\n"
        "    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);\n"
        "    cb(cur, \"linear_attn_out\", il);\n\n"
        "    // Reshape back to original dimensions\n"
        "    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);",
        "    // Output projection\n"
        "    const auto & qwen_model = static_cast<const llama_model_qwen35 &>(model);\n"
        "    if (qwen_model.aff3_images[il]) {\n"
        "        cur = qwen38_aff3_build_projection(\n"
        "            ctx0, final_output, qwen_model.aff3_images[il], aff3_node_states);\n"
        "    } else {\n"
        "        cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);\n"
        "        cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);\n"
        "    }\n"
        "    cb(cur, \"linear_attn_out\", il);",
        "linear-attention graph seam",
    )
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-source", type=Path, required=True)
    parser.add_argument(
        "--runtime-source", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument(
        "--kernel-source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "kernels",
    )
    args = parser.parse_args()
    llama = args.llama_source.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(llama), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(llama), "status", "--porcelain"], text=True
    )
    if revision != LLAMA_REVISION or status:
        raise RuntimeError("llama.cpp source revision or cleanliness changed")
    for relative, expected in FROZEN_FILES.items():
        observed = sha256_path(llama / relative)
        if observed != expected:
            raise RuntimeError(f"frozen llama.cpp input changed: {relative}")

    cmake_source = llama / "src/CMakeLists.txt"
    models_header = llama / "src/models/models.h"
    qwen_source = llama / "src/models/qwen35.cpp"
    transformed = {
        cmake_source: transform_cmake(cmake_source.read_text()),
        models_header: transform_models_header(models_header.read_text()),
        qwen_source: transform_qwen_source(qwen_source.read_text()),
    }
    for path, content in transformed.items():
        temporary = path.with_suffix(path.suffix + ".v49.tmp")
        temporary.write_text(content)
        os.replace(temporary, path)

    copied: list[Path] = []
    for name in RUNTIME_FILES:
        source = args.runtime_source / name
        target = llama / "src/models" / name
        if target.exists():
            raise FileExistsError(target)
        shutil.copyfile(source, target)
        copied.append(target)
    kernel_target = llama / "src/models/qwen38-aff3-kernels"
    kernel_target.mkdir()
    for name in KERNEL_FILES:
        source = args.kernel_source / name
        target = kernel_target / name
        shutil.copyfile(source, target)
        copied.append(target)

    seam_count = qwen_source.read_text().count("qwen38_aff3_build_projection")
    tensor_skip_count = qwen_source.read_text().count(
        "aff3_images[il] ? TENSOR_SKIP : flags"
    )
    if seam_count != 2 or tensor_skip_count != 2:
        raise RuntimeError("transformed Qwen projection seam count changed")

    result = {
        "schema": "qwen38_27b_aff3_llamacpp_transform_v49",
        "llama_revision": revision,
        "base_sha256": FROZEN_FILES,
        "transformed_sha256": {
            str(path.relative_to(llama)): sha256_path(path)
            for path in (*transformed.keys(), *copied)
        },
        "projection_seam_count": seam_count,
        "tensor_skip_count": tensor_skip_count,
        "complete": True,
        "terminal": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
