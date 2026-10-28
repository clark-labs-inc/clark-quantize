#!/usr/bin/env python3
"""Prove exact next-step replay through the global Qwen QAT checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes
from clark_quantize.qwen_global_qat import (
    capture_replay_state,
    canonical_json_sha256,
    join_model_state,
    load_global_qat_checkpoint,
    restore_process_rng,
    save_global_qat_checkpoint,
)
from clark_quantize.qwen_qat import state_tree_equal
from clark_quantize.ternary_kd import BonsaiMuonFast, KLBanditMixer, MovementServo


class ToyLayer(torch.nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight_master = torch.nn.Parameter(torch.randn(size, size))


class ToyModel(torch.nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.layer = ToyLayer(size)
        self.norm = torch.nn.Parameter(torch.ones(size))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return (values @ self.layer.weight_master.T) * self.norm


def optimizer_for(model: ToyModel) -> BonsaiMuonFast:
    return BonsaiMuonFast([{
        "params": [model.layer.weight_master],
        "lr": 0.01,
        "role": "muon",
        "state_dtype": torch.bfloat16,
        "ns_dtype": torch.bfloat16,
    }])


def snapshot_replay(
    arms: dict[str, np.random.Generator],
    bandit: KLBanditMixer,
    servo: MovementServo,
    history: list[tuple[float, float]],
) -> dict:
    return capture_replay_state(
        arm_rng_states={name: generator.bit_generator.state for name, generator in arms.items()},
        bandit_kl=bandit.kl,
        bandit_rng_state=bandit._rng.getstate(),
        servo_state=servo.__dict__,
        kl_history=history,
    )


def restore_replay(
    replay: dict,
    arms: dict[str, np.random.Generator],
    bandit: KLBanditMixer,
    servo: MovementServo,
) -> list[tuple[float, float]]:
    restore_process_rng(replay, cuda_devices=0)
    for name, generator in arms.items():
        generator.bit_generator.state = replay["arm_rng_states"][name]
    bandit.kl = dict(replay["bandit_kl"])
    bandit._rng.setstate(replay["bandit_rng_state"])
    servo.__dict__.update(replay["servo_state"])
    return [tuple(item) for item in replay["kl_history"]]


def training_step(
    model: ToyModel,
    optimizer: BonsaiMuonFast,
    arms: dict[str, np.random.Generator],
    bandit: KLBanditMixer,
    servo: MovementServo,
    history: list[tuple[float, float]],
    step: int,
) -> dict:
    arm = bandit.sample()
    numpy_values = arms[arm].normal(size=model.norm.numel()).astype(np.float32)
    python_value = random.random()
    torch_values = torch.rand(model.norm.numel())
    inputs = torch.from_numpy(numpy_values) + torch_values + python_value
    optimizer.zero_grad(set_to_none=True)
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.param_groups[0]["lr"] = 0.01 * servo.lr_mult
    optimizer.step()
    loss_value = float(loss.detach())
    bandit.update(arm, loss_value)
    history.append((step / 10.0, loss_value))
    servo.update(
        {"churn": min(0.02 * step, 0.3), "sign_inherit": 0.95},
        step / 10.0,
    )
    return {
        "arm": arm,
        "numpy_values_sha256": hashlib.sha256(numpy_values.tobytes()).hexdigest(),
        "python_value": python_value,
        "torch_values_sha256": hashlib.sha256(torch_values.numpy().tobytes()).hexdigest(),
        "loss": loss_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, default=Path(__file__).with_name("qwen_global_qat_resume_preregistration_v1.json"))
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint_root.is_absolute() or not args.receipt.is_absolute():
        raise ValueError("checkpoint root and receipt must be absolute")
    if args.checkpoint_root.exists() and any(args.checkpoint_root.iterdir()):
        raise ValueError("resume smoke checkpoint root must start empty")

    prereg_content = args.preregistration.read_bytes()
    prereg = json.loads(prereg_content)
    if prereg.get("schema") != "clark_qwen_global_qat_resume_preregistration_v1":
        raise ValueError("resume preregistration schema mismatch")
    seed = prereg["seed"]
    arm_names = prereg["arms"]
    size = prereg["matrix_shape"][0]
    run_identity = {
        "schema": "clark_qwen_global_qat_resume_smoke_identity_v1",
        "preregistration_sha256": hashlib.sha256(prereg_content).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_source_sha256": hashlib.sha256(
            (REPO / "clark_quantize/qwen_global_qat.py").read_bytes()
        ).hexdigest(),
        "optimizer_source_sha256": hashlib.sha256(
            (REPO / "clark_quantize/ternary_kd.py").read_bytes()
        ).hexdigest(),
        "torch_version": torch.__version__,
    }

    torch.manual_seed(seed)
    np.random.seed(seed + 1)
    random.seed(seed + 2)
    model = ToyModel(size)
    optimizer = optimizer_for(model)
    arms = {
        name: np.random.default_rng(seed + 100 + index)
        for index, name in enumerate(arm_names)
    }
    bandit = KLBanditMixer(arm_names, seed=seed + 200)
    servo = MovementServo()
    history: list[tuple[float, float]] = []
    observations = []
    for step in range(1, prereg["checkpoint_after_step"] + 1):
        observations.append(training_step(
            model, optimizer, arms, bandit, servo, history, step
        ))
    replay = snapshot_replay(arms, bandit, servo, history)
    checkpoint = save_global_qat_checkpoint(
        args.checkpoint_root,
        step=prereg["checkpoint_after_step"],
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        replay_state=replay,
        run_identity=run_identity,
    )
    expected = training_step(
        model, optimizer, arms, bandit, servo, history, prereg["comparison_step"]
    )
    expected_model = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    expected_optimizer = optimizer.state_dict()
    expected_bandit = dict(bandit.kl)
    expected_servo = dict(servo.__dict__)
    expected_history = list(history)

    torch.manual_seed(seed + 999)
    resumed_model = ToyModel(size)
    resumed_optimizer = optimizer_for(resumed_model)
    resumed_arms = {
        name: np.random.default_rng(seed + 999 + index)
        for index, name in enumerate(arm_names)
    }
    resumed_bandit = KLBanditMixer(arm_names, seed=seed + 999)
    resumed_servo = MovementServo()
    loaded, manifest = load_global_qat_checkpoint(
        checkpoint,
        run_identity=run_identity,
        arms=arm_names,
        cuda_devices=0,
    )
    resumed_model.load_state_dict(join_model_state(loaded), strict=True)
    resumed_optimizer.load_state_dict(loaded["optimizer"])
    optimizer_load_exact = state_tree_equal(
        resumed_optimizer.state_dict(), loaded["optimizer"]
    )
    resumed_history = restore_replay(
        loaded["replay"], resumed_arms, resumed_bandit, resumed_servo
    )
    actual = training_step(
        resumed_model,
        resumed_optimizer,
        resumed_arms,
        resumed_bandit,
        resumed_servo,
        resumed_history,
        prereg["comparison_step"],
    )
    model_exact = state_tree_equal(resumed_model.state_dict(), expected_model)
    optimizer_exact = state_tree_equal(
        resumed_optimizer.state_dict(), expected_optimizer
    )
    observation_exact = actual == expected
    controller_exact = (
        resumed_bandit.kl == expected_bandit
        and resumed_servo.__dict__ == expected_servo
        and resumed_history == expected_history
    )
    passed = all((optimizer_load_exact, model_exact, optimizer_exact, observation_exact, controller_exact))
    report = {
        "schema": "clark_qwen_global_qat_resume_smoke_v1",
        "created_at_unix": time.time(),
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": hashlib.sha256(prereg_content).hexdigest(),
            "rules": prereg,
        },
        "run_identity": run_identity,
        "run_identity_sha256": canonical_json_sha256(run_identity),
        "checkpoint": {
            "path": str(checkpoint),
            "state_sha256": manifest["state_sha256"],
            "manifest": manifest,
        },
        "prefix_observations": observations,
        "expected_next_step": expected,
        "resumed_next_step": actual,
        "checks": {
            "optimizer_load_exact": optimizer_load_exact,
            "model_after_next_step_exact": model_exact,
            "optimizer_after_next_step_exact": optimizer_exact,
            "observation_exact": observation_exact,
            "controller_and_history_exact": controller_exact,
        },
        "promotion": {
            "passed": passed,
            "scope": "restart harness only; no model-quality claim",
        },
    }
    report["report_content_sha256"] = canonical_json_sha256(report)
    atomic_bytes(args.receipt, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
