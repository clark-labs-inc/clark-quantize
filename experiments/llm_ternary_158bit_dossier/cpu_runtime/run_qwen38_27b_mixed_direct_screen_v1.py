#!/usr/bin/env python3
"""Build and score the frozen step-zero Qwen3.8-27B mixed pilot."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
RUNTIME = Path(__file__).resolve().parent
import run_qwen38_27b_affine_butterfly_block_qat_v1 as qat

from clark_quantize.qwen38_affine_artifact import sha256_file
from clark_quantize.qwen38_mixed_cq3_runtime import (
    StreamedCq3Linear,
    tensor_sha256,
    write_step_zero_cq3_tensor,
)


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TARGET_LAYERS = (0, 32, 63)
SCHEMA = "qwen38_27b_calibration_ranked_mixed_direct_screen_v1"
LAYER_SCHEMA = "qwen38_27b_calibration_ranked_mixed_direct_layer_v1"


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def mlp_targets(layer_index: int) -> tuple[dict[str, Any], ...]:
    return (
        {
            "state_name": "mlp.gate_proj.weight",
            "gguf_name": f"blk.{layer_index}.ffn_gate.weight",
            "shape": (17408, 5120),
        },
        {
            "state_name": "mlp.up_proj.weight",
            "gguf_name": f"blk.{layer_index}.ffn_up.weight",
            "shape": (17408, 5120),
        },
        {
            "state_name": "mlp.down_proj.weight",
            "gguf_name": f"blk.{layer_index}.ffn_down.weight",
            "shape": (5120, 17408),
        },
    )


def validate_or_write_direct_payload(
    *,
    source: torch.Tensor,
    target: dict[str, Any],
    payload_path: Path,
    descriptor_path: Path,
    layer_index: int,
    repository_commit: str,
    preregistration_sha256: str,
    amendment_sha256: str,
    training_cache_sha256: str,
) -> dict[str, Any]:
    source_sha256 = tensor_sha256(source)
    if payload_path.exists() or descriptor_path.exists():
        if not payload_path.is_file() or not descriptor_path.is_file():
            raise RuntimeError("partial direct CQ3 payload requires inspection")
        descriptor = json.loads(descriptor_path.read_text())
        if (
            descriptor.get("schema") != "qwen38_27b_mixed_direct_cq3_tensor_v1"
            or descriptor.get("layer") != layer_index
            or descriptor.get("state_tensor") != target["state_name"]
            or descriptor.get("gguf_tensor") != target["gguf_name"]
            or tuple(descriptor.get("shape", [])) != target["shape"]
            or descriptor.get("source_sha256") != source_sha256
            or descriptor.get("repository_commit") != repository_commit
            or descriptor.get("preregistration_sha256") != preregistration_sha256
            or descriptor.get("amendment_sha256") != amendment_sha256
            or descriptor.get("training_cache_sha256") != training_cache_sha256
            or descriptor.get("payload", {}).get("sha256") != sha256_file(payload_path)
            or descriptor.get("payload", {}).get("bytes") != payload_path.stat().st_size
            or descriptor.get("complete") is not True
            or descriptor.get("terminal") is not True
        ):
            raise ValueError("existing direct CQ3 payload identity changed")
        return descriptor
    if tuple(source.shape) != target["shape"] or source.dtype != torch.bfloat16:
        raise ValueError("direct CQ3 source tensor changed")
    payload = write_step_zero_cq3_tensor(source, payload_path, row_shard=128)
    descriptor = {
        "schema": "qwen38_27b_mixed_direct_cq3_tensor_v1",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": repository_commit,
        "preregistration_sha256": preregistration_sha256,
        "amendment_sha256": amendment_sha256,
        "training_cache_sha256": training_cache_sha256,
        "layer": layer_index,
        "state_tensor": target["state_name"],
        "gguf_tensor": target["gguf_name"],
        "shape": list(target["shape"]),
        "source_dtype": str(source.dtype),
        "source_sha256": source_sha256,
        "payload": payload,
        "complete": True,
        "terminal": True,
    }
    atomic_json(descriptor, descriptor_path)
    return descriptor


class SelectedRowsCq3Mlp(torch.nn.Module):
    """Compute the mixed MLP only for the frozen calibration rows."""

    def __init__(
        self,
        gate: StreamedCq3Linear,
        up: StreamedCq3Linear,
        down: StreamedCq3Linear,
        selected_rows: torch.Tensor,
    ):
        super().__init__()
        self.gate = gate
        self.up = up
        self.down = down
        self.register_buffer(
            "selected_rows", selected_rows.long().contiguous(), persistent=False
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        flat = hidden.reshape(-1, hidden.shape[-1])
        selected = flat.index_select(0, self.selected_rows)
        gate = self.gate(selected)
        up = self.up(selected)
        mixed = torch.nn.functional.silu(gate.float()) * up.float()
        selected_output = self.down(mixed.to(hidden.dtype))
        output = torch.zeros(
            flat.shape[0],
            selected_output.shape[-1],
            dtype=selected_output.dtype,
            device=selected_output.device,
        )
        output.index_copy_(0, self.selected_rows, selected_output)
        return output.reshape(*hidden.shape[:-1], selected_output.shape[-1])


def build_direct_layer(
    *,
    config: Any,
    layer_index: int,
    state: dict[str, torch.Tensor],
    parent_artifact_root: Path,
    payload_root: Path,
    selected_rows: torch.Tensor,
    repository_commit: str,
    preregistration_sha256: str,
    amendment_sha256: str,
    training_cache_sha256: str,
) -> tuple[qat.TrainableBlock, list[dict[str, Any]]]:
    block = qat.build_trainable_block(
        config=config,
        layer_index=layer_index,
        state=state,
        parent_artifact_root=parent_artifact_root,
        selected_sites={"attention_output"},
    )
    modules = []
    descriptors = []
    for target in mlp_targets(layer_index):
        source = state[target["state_name"]]
        stem = target["gguf_name"]
        payload_path = payload_root / f"{stem}.cq3"
        descriptor_path = payload_root / f"{stem}.json"
        descriptor = validate_or_write_direct_payload(
            source=source,
            target=target,
            payload_path=payload_path,
            descriptor_path=descriptor_path,
            layer_index=layer_index,
            repository_commit=repository_commit,
            preregistration_sha256=preregistration_sha256,
            amendment_sha256=amendment_sha256,
            training_cache_sha256=training_cache_sha256,
        )
        modules.append(
            StreamedCq3Linear.from_payload(
                payload_path,
                rows=target["shape"][0],
                columns=target["shape"][1],
                row_shard=128,
                output_dtype=torch.bfloat16,
            )
        )
        descriptors.append(descriptor)
    qat.set_child_module(
        block.layer,
        "mlp",
        SelectedRowsCq3Mlp(*modules, selected_rows=selected_rows),
    )
    # ``assign=True`` makes the original MLP parameters share these tensors with
    # ``state``.  Once the complete MLP has been replaced, retaining the state
    # entries would keep a second, unused BF16 MLP resident during scoring.
    for target in mlp_targets(layer_index):
        state.pop(target["state_name"])
    return block, descriptors


def validate_existing_layer(
    path: Path,
    *,
    repository_commit: str,
    preregistration_sha256: str,
    amendment_sha256: str,
    training_cache_sha256: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if (
        result.get("schema") != LAYER_SCHEMA
        or result.get("repository_commit") != repository_commit
        or result.get("preregistration_sha256") != preregistration_sha256
        or result.get("amendment_sha256") != amendment_sha256
        or result.get("training_cache_sha256") != training_cache_sha256
        or result.get("complete") is not True
        or result.get("terminal") is not True
    ):
        raise ValueError("existing direct mixed layer result changed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=RUNTIME / "qwen38_27b_calibration_ranked_aff3_cq3_mixed_v1.json",
    )
    parser.add_argument(
        "--amendment",
        type=Path,
        default=RUNTIME
        / "qwen38_27b_calibration_ranked_aff3_cq3_mixed_pre_execution_amendment_v1.json",
    )
    parser.add_argument(
        "--training-cache",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-affine-butterfly-blocks-v1/block-cache.pt"
        ),
    )
    parser.add_argument(
        "--parent-artifact-root",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-affine-butterfly-blocks-v1/artifact"
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-calibration-ranked-mixed-v1/direct"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-calibration-ranked-mixed-direct-screen-v1.json"
        ),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads != 4:
        raise ValueError("the frozen direct mixed screen uses four CPU threads")
    qat.resource_guard(Path("/workspace"))
    torch.set_num_threads(args.threads)

    preregistration_sha256 = sha256_file(args.preregistration)
    amendment_sha256 = sha256_file(args.amendment)
    prereg = json.loads(args.preregistration.read_text())
    amendment = json.loads(args.amendment.read_text())
    repository_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    training_cache_sha256 = sha256_file(args.training_cache)
    index_path = args.model_dir / "model.safetensors.index.json"
    if (
        prereg.get("status") != "frozen_after_terminal_qat_before_mixed_execution"
        or prereg["scope"]["active_model_only"] != MODEL_ID
        or prereg["scope"]["revision"] != MODEL_REVISION
        or tuple(prereg["scope"]["pilot_layers"]) != TARGET_LAYERS
        or prereg["scope"]["qwen_4b_compute_forbidden"] is not True
        or amendment.get("status")
        != "frozen_before_any_mixed_training_or_confirmation_score"
        or amendment.get("parent_preregistration_sha256")
        != preregistration_sha256
        or training_cache_sha256 != prereg["data"]["training_cache"]["sha256"]
        or sha256_file(index_path) != qat.INDEX_SHA256
    ):
        raise ValueError("direct mixed screen identity changed")
    if args.out.exists():
        existing = json.loads(args.out.read_text())
        if (
            existing.get("schema") != SCHEMA
            or existing.get("repository_commit") != repository_commit
            or existing.get("preregistration_sha256") != preregistration_sha256
            or existing.get("amendment_sha256") != amendment_sha256
            or existing.get("complete") is not True
            or existing.get("terminal") is not True
        ):
            raise ValueError("existing direct mixed screen changed")
        print(json.dumps(existing, indent=2, sort_keys=True))
        raise SystemExit(0 if existing["gate"]["passed"] else 2)

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    cache = torch.load(args.training_cache, map_location="cpu", weights_only=False)
    attention_mask = cache["attention_mask"].long()[:8]
    position_ids = cache["text_position_ids"].long()[:8]
    selected_rows = cache["calibration_flat_rows"].long()
    if selected_rows.numel() != 64 or int(selected_rows.max()) >= 8 * 64:
        raise ValueError("direct mixed calibration denominator changed")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    threshold = float(
        prereg["sequential_gates"]["gate_0_identity_and_direct_screen"][
            "maximum_direct_mixed_calibration_complete_block_nmse_per_layer"
        ]
    )
    started = time.monotonic()
    results = []
    for layer_index in TARGET_LAYERS:
        terminal_path = (
            args.artifact_root / "layers" / str(layer_index) / "terminal-result.json"
        )
        existing_layer = validate_existing_layer(
            terminal_path,
            repository_commit=repository_commit,
            preregistration_sha256=preregistration_sha256,
            amendment_sha256=amendment_sha256,
            training_cache_sha256=training_cache_sha256,
        )
        if existing_layer is not None:
            results.append(existing_layer)
            print(
                json.dumps(
                    {"event": "mixed_direct_layer_reused", "layer": layer_index},
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        layer_started = time.monotonic()
        print(
            json.dumps(
                {"event": "mixed_direct_layer_started", "layer": layer_index},
                sort_keys=True,
            ),
            flush=True,
        )
        state = qat.layer_state(args.model_dir, weight_map, layer_index)
        payload_root = args.artifact_root / "layers" / str(layer_index) / "payloads"
        block, descriptors = build_direct_layer(
            config=config,
            layer_index=layer_index,
            state=state,
            parent_artifact_root=args.parent_artifact_root,
            payload_root=payload_root,
            selected_rows=selected_rows,
            repository_commit=repository_commit,
            preregistration_sha256=preregistration_sha256,
            amendment_sha256=amendment_sha256,
            training_cache_sha256=training_cache_sha256,
        )
        hidden = cache["layers"][str(layer_index)]["input"][:8]
        parent = cache["layers"][str(layer_index)]["output"][:8]
        position_embeddings, masks = qat.make_context(
            config, hidden, attention_mask, position_ids
        )
        with torch.no_grad():
            output = qat.forward_with_grad(
                block.layer,
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=masks[config.layer_types[layer_index]],
                position_ids=position_ids,
            )
        calibration_nmse = qat.selected_nmse(output, parent, selected_rows)
        result = {
            "schema": LAYER_SCHEMA,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "repository_commit": repository_commit,
            "preregistration_sha256": preregistration_sha256,
            "amendment_sha256": amendment_sha256,
            "training_cache_sha256": training_cache_sha256,
            "layer": layer_index,
            "type": config.layer_types[layer_index],
            "calibration_rows": selected_rows.numel(),
            "direct_mixed_calibration_complete_block_nmse": calibration_nmse,
            "maximum_allowed_nmse": threshold,
            "cq3_payloads": [
                {
                    "state_tensor": item["state_tensor"],
                    "gguf_tensor": item["gguf_tensor"],
                    "source_sha256": item["source_sha256"],
                    "payload": item["payload"],
                }
                for item in descriptors
            ],
            "attention_output_mode": "authenticated parent AFF3 PTQ payload",
            "passed": calibration_nmse <= threshold,
            "elapsed_seconds": time.monotonic() - layer_started,
            "complete": True,
            "terminal": True,
        }
        atomic_json(result, terminal_path)
        results.append(result)
        print(
            json.dumps({"event": "mixed_direct_layer_terminal", **result}, sort_keys=True),
            flush=True,
        )
        del output, block, state
        gc.collect()

    gate = {
        "calibration_rows_per_layer": 64,
        "maximum_direct_mixed_calibration_complete_block_nmse_per_layer": threshold,
        "all_layers_terminal": all(item["terminal"] for item in results),
        "all_values_finite": all(
            torch.isfinite(
                torch.tensor(item["direct_mixed_calibration_complete_block_nmse"])
            ).item()
            for item in results
        ),
        "all_layers_below_maximum": all(item["passed"] for item in results),
    }
    gate["passed"] = bool(
        gate["all_layers_terminal"]
        and gate["all_values_finite"]
        and gate["all_layers_below_maximum"]
    )
    receipt = {
        "schema": SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": repository_commit,
        "preregistration_sha256": preregistration_sha256,
        "amendment_sha256": amendment_sha256,
        "training_cache_sha256": training_cache_sha256,
        "layers": results,
        "gate": gate,
        "elapsed_seconds": time.monotonic() - started,
        "complete": True,
        "terminal": True,
        "claim_boundary": "terminal three-block step-zero mixed calibration screen only; no training, fresh-confirmation score, complete-model artifact, capability result, CPU runtime result, or breakthrough",
    }
    atomic_json(receipt, args.out)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if gate["passed"] else 2)


if __name__ == "__main__":
    main()
