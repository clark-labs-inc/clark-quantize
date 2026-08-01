#!/usr/bin/env python3
"""Distill the exact fused browser video student from Swin2SR and ground truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import Swin2SRForImageSuperResolution

STUDY_ROOT = Path(__file__).resolve().parent
SWIN_ROOT = STUDY_ROOT.parent
sys.path.insert(0, str(SWIN_ROOT))

from data_pipeline import degrade, discover_images, load_rgb, to_tensor, ui_crop
from realtime_student import RealtimeLumaStudent, RealtimeStudentConfig
from realtime_student import initialize_rgb_from_luma
from realtime_student import initialize_wider_rgb_from_rgb

MODEL_ID = "caidas/swin2SR-lightweight-x2-64"
MODEL_REVISION = "07e8e61492ce6daf4f34c651622960faf110c17f"
DEFAULT_DATA = (
    Path("/Users/stan/.cache/clark-quantize/swin2sr-hf-100k-v1"),
    Path("/Users/stan/.cache/clark-quantize/swin2sr-flores200-text-102k-v1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="append", type=Path)
    parser.add_argument("--out", type=Path, default=STUDY_ROOT / "training")
    parser.add_argument("--steps", type=int, default=26_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--color-mode", choices=("luma", "rgb"), default="luma")
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--teacher-weight", type=float, default=0.25)
    parser.add_argument("--edge-weight", type=float, default=0.20)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-images-per-source", type=int, default=16)
    parser.add_argument("--validation-text-per-language", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    return parser.parse_args()


def charbonnier(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((left - right).square() + 1e-6).mean()


def image_gradients(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    horizontal = values[..., :, 1:] - values[..., :, :-1]
    vertical = values[..., 1:, :] - values[..., :-1, :]
    return horizontal, vertical


def edge_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_x, left_y = image_gradients(left)
    right_x, right_y = image_gradients(right)
    return charbonnier(left_x, right_x) + charbonnier(left_y, right_y)


def psnr(left: torch.Tensor, right: torch.Tensor) -> float:
    mse = F.mse_loss(left.float(), right.float()).clamp_min(1e-12)
    return float((-10 * torch.log10(mse)).item())


def sample_batch(
    groups: list[tuple[str, list[Path], float]],
    batch_size: int,
    crop_size: int,
    device: str,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    low_values = []
    high_values = []
    while len(low_values) < batch_size:
        group = rng.choices(groups, weights=[item[2] for item in groups], k=1)[0]
        paths = group[1]
        path = paths[rng.randrange(len(paths))]
        try:
            high_image = ui_crop(load_rgb(path), crop_size, rng)
            low_image = degrade(high_image, rng)
        except Exception:
            continue
        low_values.append(to_tensor(low_image, device))
        high_values.append(to_tensor(high_image, device))
    return torch.cat(low_values), torch.cat(high_values)


def grouped_paths(paths: list[Path], roots: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    resolved_roots = [root.resolve() for root in roots]
    for path in paths:
        resolved = path.resolve()
        for root in resolved_roots:
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            prefix = "flores" if "flores200" in root.name else "image"
            group = f"{prefix}/{relative.parts[0]}"
            groups.setdefault(group, []).append(path)
            break
        else:
            raise ValueError(f"image is outside declared roots: {path}")
    return groups


def split_stratified(
    groups: dict[str, list[Path]],
    images_per_source: int,
    text_per_language: int,
    seed: int,
) -> tuple[dict[str, list[Path]], list[tuple[str, Path]]]:
    training = {}
    validation = []
    for name, paths in sorted(groups.items()):
        ordered = sorted(
            paths,
            key=lambda path: hashlib.sha256(
                f"{seed}:{path}".encode()
            ).digest(),
        )
        requested = text_per_language if name.startswith("flores/") else images_per_source
        count = min(requested, max(1, len(ordered) // 100))
        validation.extend((name, path) for path in ordered[:count])
        training[name] = ordered[count:]
    return training, validation


def sampling_groups(groups: dict[str, list[Path]]) -> list[tuple[str, list[Path], float]]:
    flores = [name for name in groups if name.startswith("flores/")]
    fixed_weights = {
        "image/rico": 0.35,
        "image/aria_ui": 0.15,
        "image/websight": 0.20,
        "image/pd12m": 0.10,
    }
    weighted = []
    for name, paths in sorted(groups.items()):
        if name.startswith("flores/"):
            weight = 0.20 / len(flores)
        else:
            weight = fixed_weights.get(name, 0.0)
        if weight > 0 and paths:
            weighted.append((name, paths, weight))
    total = sum(item[2] for item in weighted)
    if total <= 0:
        raise ValueError("no supported training source groups")
    return [(name, paths, weight / total) for name, paths, weight in weighted]


@torch.no_grad()
def validate(
    student: RealtimeLumaStudent,
    teacher: Swin2SRForImageSuperResolution,
    samples: list[tuple[str, Path]],
    crop_size: int,
    device: str,
    seed: int,
) -> dict[str, float]:
    student.eval()
    by_source: dict[str, dict[str, list[float]]] = {}
    for index, (group, path) in enumerate(samples):
        rng = random.Random(seed + index)
        high_image = ui_crop(load_rgb(path), crop_size, rng)
        low_image = degrade(high_image, rng)
        low = to_tensor(low_image, device)
        high = to_tensor(high_image, device)
        output = student(low)
        teacher_output = teacher(low).reconstruction.clamp(0, 1)
        source = "flores" if group.startswith("flores/") else group.split("/", 1)[1]
        scores = by_source.setdefault(source, {"ground_truth": [], "teacher": []})
        scores["ground_truth"].append(psnr(output, high))
        scores["teacher"].append(psnr(output, teacher_output))
    student.train()
    source_metrics = {
        source: {
            "samples": len(scores["ground_truth"]),
            "psnr_vs_ground_truth": float(np.mean(scores["ground_truth"])),
            "psnr_vs_teacher": float(np.mean(scores["teacher"])),
        }
        for source, scores in sorted(by_source.items())
    }
    source_weights = {
        "aria_ui": 0.15,
        "flores": 0.20,
        "pd12m": 0.10,
        "rico": 0.35,
        "websight": 0.20,
    }
    weighted_ground_truth = sum(
        source_metrics[source]["psnr_vs_ground_truth"] * source_weights[source]
        for source in source_metrics
    ) / sum(source_weights[source] for source in source_metrics)
    weighted_teacher = sum(
        source_metrics[source]["psnr_vs_teacher"] * source_weights[source]
        for source in source_metrics
    ) / sum(source_weights[source] for source in source_metrics)
    return {
        "product_weighted_psnr_vs_ground_truth": weighted_ground_truth,
        "product_weighted_psnr_vs_teacher": weighted_teacher,
        "sample_weighted_psnr_vs_ground_truth": float(np.mean([
            value for scores in by_source.values() for value in scores["ground_truth"]
        ])),
        "sample_weighted_psnr_vs_teacher": float(np.mean([
            value for scores in by_source.values() for value in scores["teacher"]
        ])),
        "by_source": source_metrics,
    }


def main() -> None:
    args = parse_args()
    if args.crop_size % 4:
        raise ValueError("crop-size must be divisible by four")
    if not 0 <= args.teacher_weight <= 1 or not 0 <= args.edge_weight <= 1:
        raise ValueError("loss weights must be between zero and one")
    ground_truth_weight = 1 - args.teacher_weight - args.edge_weight
    if ground_truth_weight <= 0:
        raise ValueError("teacher-weight plus edge-weight must be below one")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    data_roots = args.data or list(DEFAULT_DATA)
    paths = discover_images(data_roots)
    groups = grouped_paths(paths, data_roots)
    training_by_group, validation_paths = split_stratified(
        groups,
        args.validation_images_per_source,
        args.validation_text_per_language,
        args.seed,
    )
    training_groups = sampling_groups(training_by_group)
    args.out.mkdir(parents=True, exist_ok=True)

    teacher = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    ).to(args.device).eval()
    teacher.requires_grad_(False)
    student = RealtimeLumaStudent(
        RealtimeStudentConfig(
            channels=args.channels,
            blocks=args.blocks,
            color_mode=args.color_mode,
        )
    ).to(args.device).train()
    initialization = "random"
    if args.initialize_from:
        parent = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
        if parent.get("format") != "backdropkit-realtime-student-checkpoint-v1":
            raise ValueError("unsupported initialization checkpoint")
        parent_student = parent["manifest"]["student"]
        if args.color_mode == "rgb" and parent_student["color_mode"] == "luma":
            initialize_rgb_from_luma(student, parent["model"])
            initialization = "exact-luma-to-rgb"
        elif (
            args.color_mode == "rgb"
            and parent_student["color_mode"] == "rgb"
            and args.channels > int(parent_student["channels"])
        ):
            initialize_wider_rgb_from_rgb(student, parent["model"])
            initialization = "exact-rgb-width-expansion"
        else:
            student.load_state_dict(parent["model"], strict=True)
            initialization = "checkpoint"
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
    started = time.time()
    manifest = {
        "format": "backdropkit-realtime-student-training-v1",
        "teacher": {"model": MODEL_ID, "revision": MODEL_REVISION},
        "student": {
            "channels": args.channels,
            "blocks": args.blocks,
            "parameters": student.parameter_count(),
            "color_mode": args.color_mode,
        },
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "crop_size": args.crop_size,
            "learning_rate": args.learning_rate,
            "loss_weights": {
                "ground_truth": ground_truth_weight,
                "teacher": args.teacher_weight,
                "edge": args.edge_weight,
            },
            "seed": args.seed,
            "device": args.device,
        },
        "initialization": {
            "method": initialization,
            "checkpoint": str(args.initialize_from.resolve()) if args.initialize_from else None,
            "checkpoint_sha256": (
                hashlib.sha256(args.initialize_from.read_bytes()).hexdigest()
                if args.initialize_from else None
            ),
        },
        "data_roots": [str(path) for path in data_roots],
        "training_images": sum(len(paths) for paths in training_by_group.values()),
        "validation_images": len(validation_paths),
        "validation_groups": len(groups),
        "sampling": [
            {"group": name, "images": len(paths), "weight": weight}
            for name, paths, weight in training_groups
        ],
    }
    baseline = validate(
        student,
        teacher,
        validation_paths,
        args.crop_size,
        args.device,
        args.seed + 1_000_000,
    )
    manifest["baseline_validation"] = baseline
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    with (args.out / "train.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            low, high = sample_batch(
                training_groups,
                args.batch_size,
                args.crop_size,
                args.device,
                rng,
            )
            with torch.no_grad():
                teacher_output = teacher(low).reconstruction.clamp(0, 1)
            output = student(low)
            ground_truth_loss = charbonnier(output, high)
            teacher_loss = charbonnier(output, teacher_output)
            edges = edge_loss(output, high)
            loss = (
                ground_truth_weight * ground_truth_loss
                + args.teacher_weight * teacher_loss
                + args.edge_weight * edges
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            record = {
                "step": step,
                "loss": float(loss.item()),
                "ground_truth_loss": float(ground_truth_loss.item()),
                "teacher_loss": float(teacher_loss.item()),
                "edge_loss": float(edges.item()),
                "elapsed_seconds": time.time() - started,
            }
            if step % args.validate_every == 0 or step == args.steps:
                record["validation"] = validate(
                    student,
                    teacher,
                    validation_paths,
                    args.crop_size,
                    args.device,
                    args.seed + 1_000_000,
                )
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)

    torch.save(
        {
            "format": "backdropkit-realtime-student-checkpoint-v1",
            "model": student.state_dict(),
            "manifest": manifest,
        },
        args.out / "checkpoint-final.pt",
    )


if __name__ == "__main__":
    main()
