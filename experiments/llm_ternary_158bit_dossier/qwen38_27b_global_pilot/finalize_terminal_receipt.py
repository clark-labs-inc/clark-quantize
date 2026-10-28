#!/usr/bin/env python3
"""Seal terminal integrity, retention, and machine-state evidence."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys

from safetensors import safe_open

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments" / "large_scale_ternary"))
from gate_b_protocol import digest_json, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--patch-sha256", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    names = ["capability_manifest.json", "eval_bf16.json", "eval_base3_ptq.json", "train_pilot.json", "eval_trained.json", "forgetting_comparison.json"]
    documents = {name: json.loads((args.state_root / name).read_text()) for name in names}
    trained = documents["eval_trained.json"]
    comparison = documents["forgetting_comparison.json"]
    training = documents["train_pilot.json"]
    if trained["transform"]["patch_sha256"] != args.patch_sha256:
        raise ValueError("trained-eval patch identity differs from independently supplied SHA-256")
    if trained["transform"]["applied_count"] != 704:
        raise ValueError("trained eval did not apply all 704 patch tensors")
    if training["history"][-1]["step"] != 224 or training.get("early_stop") is not None:
        raise ValueError("training did not reach the preregistered terminal step")
    if comparison["promotion"]["passed"] is not False:
        raise ValueError("failed pilot must not be marked promoted")

    with safe_open(args.patch, framework="pt", device="cpu") as handle:
        patch_tensor_count = len(list(handle.keys()))
    if patch_tensor_count != trained["transform"]["applied_count"]:
        raise ValueError("patch tensor count differs from applied count")

    legacy = sorted(glob.glob("/workspace/quant-runs/**/checkpoint-*", recursive=True))
    legacy = [path for path in legacy if Path(path).is_dir()]
    new_patches = sorted(str(path) for path in args.patch.parent.glob("*.safetensors"))
    retention_passed = len(legacy) == 1 and len(new_patches) == 1 and new_patches[0] == str(args.patch)
    if not retention_passed:
        raise ValueError(f"checkpoint retention violation: legacy={legacy}, new={new_patches}")

    power = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=power.limit,power.default_limit,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().split(", ")
    power_state = {
        "power_limit_watts": float(power[0]),
        "default_power_limit_watts": float(power[1]),
        "memory_used_mib": int(power[2]),
        "utilization_percent": int(power[3]),
        "temperature_c": int(power[4]),
    }
    if power_state["power_limit_watts"] != power_state["default_power_limit_watts"]:
        raise ValueError(f"GPU power limit not restored: {power_state}")

    disk = shutil.disk_usage("/workspace")
    sources = [
        REPO / "experiments/qwen_agentworld/train_all2d_ternary_qat.py",
        REPO / "experiments/qwen_agentworld/fake_quant_policy.py",
        REPO / "experiments/qwen_agentworld/weight_patch.py",
        Path(__file__).with_name("prepare_capability_manifest.py"),
        Path(__file__).with_name("evaluate_capabilities.py"),
        Path(__file__).with_name("compare_forgetting.py"),
        Path(__file__),
    ]
    receipt = {
        "schema": "qwen38_global_hard_ternary_pilot_terminal_v1",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "terminal_state": "completed_failed_hypothesis_not_promoted",
        "model": {
            "path": str(args.model_dir),
            "revision": trained["model"]["revision"],
            "config_sha256": sha256_file(args.model_dir / "config.json"),
            "index_sha256": sha256_file(args.model_dir / "model.safetensors.index.json"),
            "evaluation_config_identity": trained["model"]["config_sha256"],
        },
        "patch": {
            "path": str(args.patch),
            "bytes": args.patch.stat().st_size,
            "sha256": args.patch_sha256,
            "sha256_verification": "matched independent sha256sum and evaluator full-file hash",
            "tensor_count": patch_tensor_count,
            "applied_parameters": trained["transform"]["applied_parameters"],
            "storage_form": "dense_bfloat16_evaluation_patch_not_packed_release",
        },
        "training": {
            "steps": training["history"][-1]["step"],
            "all_quantized_modules": training["all_quantized_modules"],
            "trained_norm_tensors": training["other_params"],
            "peak_accelerator_memory_allocated_bytes": training["peak_accelerator_memory_allocated_bytes"],
            "peak_accelerator_memory_reserved_bytes": training["peak_accelerator_memory_reserved_bytes"],
            "early_stop": training["early_stop"],
            "final_code_changed_frac": training["history"][-1]["code_changed_frac"],
            "final_code_persistence_frac": training["history"][-1]["code_persistence_frac"],
        },
        "evaluation": {
            "manifest_content_sha256": comparison["manifest_content_sha256"],
            "aggregate": comparison["aggregate"],
            "tasks": comparison["tasks"],
            "promotion": comparison["promotion"],
        },
        "retention": {
            "passed": retention_passed,
            "legacy_checkpoint_directories": legacy,
            "new_checkpoint_files": new_patches,
        },
        "machine_terminal_state": {
            "gpu": power_state,
            "workspace_disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        },
        "artifact_boundary": {
            "raw_artifacts_remain_on_a100": True,
            "remote_run_root": "/workspace/quant-runs/qwen38-global-pilot-v1",
            "remote_state_root": str(args.state_root),
            "unrelated_client_infrastructure_used": False,
        },
        "documents": {name: {"path": str(args.state_root / name), "sha256": sha256_file(args.state_root / name)} for name in names},
        "source_sha256": {str(path.relative_to(REPO)): sha256_file(path) for path in sources},
    }
    receipt["content_sha256"] = digest_json(receipt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
