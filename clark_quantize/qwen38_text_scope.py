"""Canonical deployable text-inference scope for Qwen3.8-27B."""
from __future__ import annotations

import math
from typing import Any, Iterable, TypeVar


_T = TypeVar("_T")


def is_qwen38_text_inference_tensor(name: str) -> bool:
    """Match the exact HF tensors retained by llama.cpp ``--no-nextn``.

    Qwen3.8-27B's Hugging Face repository also carries a vision encoder and an
    MTP/NextN prediction head.  Neither participates in the text-only graph
    measured by the CPU baselines.  The language model and untied LM head sum
    to the exact 26,895,998,464 parameters reported by llama.cpp.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("a nonempty tensor name is required")
    return name.startswith("model.language_model.") or name == "lm_head.weight"


def qwen38_text_inference_tensors(tensors: Iterable[_T]) -> list[_T]:
    result = [
        tensor
        for tensor in tensors
        if is_qwen38_text_inference_tensor(getattr(tensor, "name"))
    ]
    if not result:
        raise ValueError("Qwen3.8 text-inference tensor scope is empty")
    return result


def tensor_scope_summary(tensors: Iterable[Any]) -> dict[str, int]:
    values = list(tensors)
    quantized = [tensor for tensor in values if tensor.quantized]
    retained = [tensor for tensor in values if not tensor.quantized]
    return {
        "tensors": len(values),
        "parameters": sum(tensor.parameters for tensor in values),
        "rank_ge_2_tensors": len(quantized),
        "rank_ge_2_parameters": sum(tensor.parameters for tensor in quantized),
        "retained_exact_tensors": len(retained),
        "retained_exact_parameters": sum(tensor.parameters for tensor in retained),
        "output_rows": sum(tensor.rows for tensor in quantized),
        "group_128_scales": sum(
            tensor.rows * math.ceil(tensor.columns / 128) for tensor in quantized
        ),
    }


def qwen38_checkpoint_scope_partition(tensors: Iterable[_T]) -> dict[str, list[_T]]:
    values = list(tensors)
    result = {
        "text": qwen38_text_inference_tensors(values),
        "vision": [
            tensor
            for tensor in values
            if getattr(tensor, "name").startswith("model.visual.")
        ],
        "mtp_nextn": [
            tensor for tensor in values if getattr(tensor, "name").startswith("mtp.")
        ],
    }
    names = [getattr(tensor, "name") for group in result.values() for tensor in group]
    if len(names) != len(values) or len(set(names)) != len(values):
        raise ValueError("Qwen3.8 checkpoint has unclassified or duplicate tensor scope")
    return result
