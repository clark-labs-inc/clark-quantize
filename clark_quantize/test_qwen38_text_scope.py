from __future__ import annotations

import json
from pathlib import Path

import pytest

from clark_quantize.qwen38_storage_budget import TensorSpec, canonical_inventory_digest
from clark_quantize.qwen38_text_scope import (
    is_qwen38_text_inference_tensor,
    qwen38_checkpoint_scope_partition,
    qwen38_text_inference_tensors,
    tensor_scope_summary,
)


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"


def test_text_scope_matches_language_model_and_untied_head_only() -> None:
    assert is_qwen38_text_inference_tensor(
        "model.language_model.layers.63.mlp.down_proj.weight"
    )
    assert is_qwen38_text_inference_tensor("model.language_model.embed_tokens.weight")
    assert is_qwen38_text_inference_tensor("lm_head.weight")
    assert not is_qwen38_text_inference_tensor("model.visual.blocks.0.attn.qkv.weight")
    assert not is_qwen38_text_inference_tensor("mtp.layers.0.mlp.up_proj.weight")
    with pytest.raises(ValueError, match="nonempty"):
        is_qwen38_text_inference_tensor("")


def test_text_scope_filter_is_deterministic() -> None:
    tensors = [
        TensorSpec("model.visual.proj.weight", (2, 2), "BF16"),
        TensorSpec("lm_head.weight", (3, 2), "BF16"),
        TensorSpec("model.language_model.norm.weight", (2,), "BF16"),
        TensorSpec("mtp.fc.weight", (2, 4), "BF16"),
    ]
    selected = qwen38_text_inference_tensors(tensors)
    assert [tensor.name for tensor in selected] == [
        "lm_head.weight",
        "model.language_model.norm.weight",
    ]
    assert canonical_inventory_digest(selected) == canonical_inventory_digest(
        list(selected)
    )


def test_scope_partition_and_summary_are_exhaustive() -> None:
    tensors = [
        TensorSpec("model.language_model.proj.weight", (2, 256), "BF16"),
        TensorSpec("model.language_model.norm.weight", (2,), "BF16"),
        TensorSpec("model.visual.proj.weight", (3, 128), "BF16"),
        TensorSpec("mtp.fc.weight", (4, 64), "BF16"),
    ]
    partition = qwen38_checkpoint_scope_partition(tensors)
    assert {name: len(values) for name, values in partition.items()} == {
        "text": 2,
        "vision": 1,
        "mtp_nextn": 1,
    }
    text = tensor_scope_summary(partition["text"])
    assert text == {
        "tensors": 2,
        "parameters": 514,
        "rank_ge_2_tensors": 1,
        "rank_ge_2_parameters": 512,
        "retained_exact_tensors": 1,
        "retained_exact_parameters": 2,
        "output_rows": 2,
        "group_128_scales": 4,
    }


def test_scope_reconciliation_has_exact_parameter_partition() -> None:
    payload = json.loads(
        (RUNTIME / "qwen38_27b_text_graph_scope_amendment_v1.json").read_text()
    )
    hf = payload["observed_checkpoint_scope"]
    text = payload["selected_text_graph_scope"]
    excluded = payload["excluded_non_text_scopes"]
    assert text["parameters"] == 26_895_998_464
    assert text["parameters"] == payload["llama_cpp_control_scope"]["parameters"]
    assert hf["parameters"] == text["parameters"] + sum(
        row["parameters"] for row in excluded.values()
    )
    assert hf["tensors"] == text["tensors"] + sum(
        row["tensors"] for row in excluded.values()
    )
    assert text["mapped_rank_ge_2_tensors"] + text["supplemental_rank_ge_2_tensors"] == text["rank_ge_2_tensors"]
    assert text["projected_complete_bundle_bpw"] <= 1.64


def test_text_row_scale_gate_uses_the_reconciled_scope() -> None:
    preregistration = json.loads(
        (
            RUNTIME
            / "qwen38_27b_text_row_normalized_scale_codebook4_preregistration_v1.json"
        ).read_text()
    )
    scope = preregistration["scope"]
    supplemental = preregistration["supplemental_fit"]
    assert scope["text_graph_parameters"] == 26_895_998_464
    assert scope["rank_ge_2_tensors"] == 546
    assert scope["group_128_scales"] == 210_595_840
    assert supplemental["tensors"] == 144
    assert supplemental["parameters"] == 25_559_040
    runner = (
        RUNTIME / "run_qwen38_27b_text_row_normalized_scale_codebook4_v1.py"
    ).read_text()
    watcher = (RUNTIME / "watch_qwen38_27b_row_scale_pipeline_v2.sh").read_text()
    assert "EXPECTED_TEXT_PARAMETERS = 26_895_998_464" in runner
    assert "EXPECTED_TOTAL_TENSORS = 546" in runner
    assert "qwen38_text_inference_tensors(checkpoint_tensors)" in runner
    assert "qwen38-27b-text-row-normalized-scale-codebook4-v1.json" in watcher
    assert "qwen38_27b_text_graph_scope_amendment_v1.json" in watcher
    assert 'mapped_watch=${MAPPED_WATCH:-$state/mapped-watch-receipt.json}' in watcher
