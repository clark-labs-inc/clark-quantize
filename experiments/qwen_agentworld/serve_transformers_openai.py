"""Minimal OpenAI-compatible Transformers server for AgentWorld smoke runs.

This server is for single-node smoke testing. Full official 262k-context
deployment should use Qwen's documented vLLM/SGLang tensor-parallel setup.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import transformers
import uvicorn

from fake_quant_policy import apply_fake_quant_policy
from packed_manifest_loader import apply_packed_manifest
from packed_runtime_linear import apply_packed_runtime_linears, collect_packed_runtime_stats
from weight_patch import apply_weight_patch


MODEL_LOADER_CANDIDATES = (
    "AutoModelForImageTextToText",
    "AutoModelForMultimodalLM",
    "AutoModelForCausalLM",
    "AutoModelForVision2Seq",
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.6


def build_app(
    model_path: str,
    max_model_len: int,
    *,
    enable_thinking: bool = False,
    weight_patch: str | None = None,
    weight_patch_report: str | None = None,
    fake_quant_policy: str | None = None,
    fake_quant_max_groups: int | None = None,
    fake_quant_include_experts: bool = False,
    fake_quant_rates: str | None = None,
    fake_quant_buckets: str | None = None,
    fake_quant_report: str | None = None,
    packed_manifest: str | None = None,
    packed_max_groups: int | None = None,
    packed_include_experts: bool = False,
    packed_rates: str | None = None,
    packed_buckets: str | None = None,
    packed_report: str | None = None,
    packed_runtime_manifest: str | None = None,
    packed_runtime_max_groups: int | None = None,
    packed_runtime_include_experts: bool = False,
    packed_runtime_rates: str | None = None,
    packed_runtime_buckets: str | None = None,
    packed_runtime_cache: str = "cpu",
    packed_runtime_active_cache_limit: int = 32,
    packed_runtime_active_cpu_cache_limit: int = 0,
    packed_runtime_report: str | None = None,
) -> FastAPI:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    load_errors = []
    model = None
    for loader_name in MODEL_LOADER_CANDIDATES:
        loader = getattr(transformers, loader_name, None)
        if loader is None:
            continue
        try:
            model = loader.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            print(f"loaded_model loader={loader_name} class={model.__class__.__name__}", flush=True)
            break
        except Exception as exc:
            load_errors.append(f"{loader_name}: {type(exc).__name__}: {exc}")
    if model is None:
        raise RuntimeError("could not load model with any supported Transformers loader:\n" + "\n".join(load_errors))
    model.eval()
    weight_patch_result = None
    fake_quant_result = None
    packed_result = None
    packed_runtime_result = None
    if weight_patch:
        weight_patch_result = apply_weight_patch(
            model,
            weight_patch,
            out_path=weight_patch_report,
        )
        print(
            "weight_patch_applied "
            f"count={weight_patch_result['applied_count']} "
            f"missing={weight_patch_result['missing_count']} "
            f"report={weight_patch_report}",
            flush=True,
        )
    if fake_quant_policy:
        fake_quant_result = apply_fake_quant_policy(
            model,
            fake_quant_policy,
            max_groups=fake_quant_max_groups,
            include_experts=fake_quant_include_experts,
            rates=fake_quant_rates,
            buckets=fake_quant_buckets,
            out_path=fake_quant_report,
        )
        print(
            "fake_quant_applied "
            f"count={fake_quant_result['applied_count']} "
            f"missing={fake_quant_result['missing_count']} "
            f"report={fake_quant_report}",
            flush=True,
        )
    if packed_manifest:
        packed_result = apply_packed_manifest(
            model,
            packed_manifest,
            max_groups=packed_max_groups,
            include_experts=packed_include_experts,
            rates=packed_rates,
            buckets=packed_buckets,
            out_path=packed_report,
        )
        print(
            "packed_manifest_applied "
            f"count={packed_result['applied_count']} "
            f"missing={packed_result['missing_count']} "
            f"report={packed_report}",
            flush=True,
        )
    if packed_runtime_manifest:
        packed_runtime_result = apply_packed_runtime_linears(
            model,
            packed_runtime_manifest,
            max_groups=packed_runtime_max_groups,
            include_experts=packed_runtime_include_experts,
            rates=packed_runtime_rates,
            buckets=packed_runtime_buckets,
            cache=packed_runtime_cache,
            active_cache_limit=packed_runtime_active_cache_limit,
            active_cpu_cache_limit=packed_runtime_active_cpu_cache_limit,
            out_path=packed_runtime_report,
        )
        print(
            "packed_runtime_modules_replaced "
            f"count={packed_runtime_result['replaced_count']} "
            f"expert_count={packed_runtime_result['replaced_expert_count']} "
            f"skipped={packed_runtime_result['skipped_count']} "
            f"removed_parameter_bytes={packed_runtime_result['removed_parameter_bytes']} "
            f"report={packed_runtime_report}",
            flush=True,
        )

    app = FastAPI()

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        meta = {"id": model_path, "object": "model"}
        if weight_patch_result is not None:
            meta["weight_patch"] = {
                "applied_count": weight_patch_result["applied_count"],
                "missing_count": weight_patch_result["missing_count"],
            }
        if fake_quant_result is not None:
            meta["fake_quant"] = {
                "applied_count": fake_quant_result["applied_count"],
                "missing_count": fake_quant_result["missing_count"],
            }
        if packed_result is not None:
            meta["packed_manifest"] = {
                "applied_count": packed_result["applied_count"],
                "missing_count": packed_result["missing_count"],
            }
        if packed_runtime_result is not None:
            meta["packed_runtime_linears"] = {
                "replaced_count": packed_runtime_result["replaced_count"],
                "replaced_expert_count": packed_runtime_result["replaced_expert_count"],
                "skipped_count": packed_runtime_result["skipped_count"],
                "removed_parameter_bytes": packed_runtime_result["removed_parameter_bytes"],
                "runtime_stats": collect_packed_runtime_stats(model),
            }
        return {"object": "list", "data": [meta]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatCompletionRequest) -> dict[str, Any]:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=max_model_len)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=req.temperature > 0,
                temperature=max(req.temperature, 1e-5),
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[-1] :]
        content = tokenizer.decode(new_tokens, skip_special_tokens=True)
        now = int(time.time())
        return {
            "id": f"chatcmpl-{now}",
            "object": "chat.completion",
            "created": now,
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/models/Qwen-AgentWorld-35B-A3B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--weight-patch")
    parser.add_argument("--weight-patch-report")
    parser.add_argument("--fake-quant-policy")
    parser.add_argument("--fake-quant-max-groups", type=int)
    parser.add_argument("--fake-quant-include-experts", action="store_true")
    parser.add_argument("--fake-quant-rates")
    parser.add_argument("--fake-quant-buckets")
    parser.add_argument("--fake-quant-report")
    parser.add_argument("--packed-manifest")
    parser.add_argument("--packed-max-groups", type=int)
    parser.add_argument("--packed-include-experts", action="store_true")
    parser.add_argument("--packed-rates")
    parser.add_argument("--packed-buckets")
    parser.add_argument("--packed-report")
    parser.add_argument("--packed-runtime-manifest")
    parser.add_argument("--packed-runtime-max-groups", type=int)
    parser.add_argument("--packed-runtime-include-experts", action="store_true")
    parser.add_argument("--packed-runtime-rates")
    parser.add_argument("--packed-runtime-buckets")
    parser.add_argument("--packed-runtime-cache", default="cpu", choices=["none", "cpu", "gpu", "active", "active_lazy"])
    parser.add_argument("--packed-runtime-active-cache-limit", type=int, default=32)
    parser.add_argument("--packed-runtime-active-cpu-cache-limit", type=int, default=0)
    parser.add_argument("--packed-runtime-report")
    args = parser.parse_args()
    uvicorn.run(
        build_app(
            args.model_path,
            args.max_model_len,
            enable_thinking=args.enable_thinking,
            weight_patch=args.weight_patch,
            weight_patch_report=args.weight_patch_report,
            fake_quant_policy=args.fake_quant_policy,
            fake_quant_max_groups=args.fake_quant_max_groups,
            fake_quant_include_experts=args.fake_quant_include_experts,
            fake_quant_rates=args.fake_quant_rates,
            fake_quant_buckets=args.fake_quant_buckets,
            fake_quant_report=args.fake_quant_report,
            packed_manifest=args.packed_manifest,
            packed_max_groups=args.packed_max_groups,
            packed_include_experts=args.packed_include_experts,
            packed_rates=args.packed_rates,
            packed_buckets=args.packed_buckets,
            packed_report=args.packed_report,
            packed_runtime_manifest=args.packed_runtime_manifest,
            packed_runtime_max_groups=args.packed_runtime_max_groups,
            packed_runtime_include_experts=args.packed_runtime_include_experts,
            packed_runtime_rates=args.packed_runtime_rates,
            packed_runtime_buckets=args.packed_runtime_buckets,
            packed_runtime_cache=args.packed_runtime_cache,
            packed_runtime_active_cache_limit=args.packed_runtime_active_cache_limit,
            packed_runtime_active_cpu_cache_limit=args.packed_runtime_active_cpu_cache_limit,
            packed_runtime_report=args.packed_runtime_report,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
