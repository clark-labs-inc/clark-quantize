from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


def load_index(root: Path) -> dict[str, str]:
    return json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]


def tensor_from(root: Path, index: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(root / index[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def kind(name: str) -> str:
    if name == "model.embed_tokens.weight":
        return "embed"
    if name == "lm_head.weight":
        return "lm_head"
    for key, label in [
        ("self_attn.q_proj.weight", "q_proj"),
        ("self_attn.k_proj.weight", "k_proj"),
        ("self_attn.v_proj.weight", "v_proj"),
        ("self_attn.o_proj.weight", "o_proj"),
        ("mlp.gate_proj.weight", "gate_proj"),
        ("mlp.up_proj.weight", "up_proj"),
        ("mlp.down_proj.weight", "down_proj"),
    ]:
        if name.endswith(key):
            return label
    if name.endswith(".weight") and ".norm" in name or "layernorm" in name:
        return "norm"
    return "other"


def layer(name: str) -> str:
    match = re.search(r"model\.layers\.(\d+)\.", name)
    return match.group(1) if match else "non_layer"


def optimal_ternary_codes(base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = base.reshape(-1, 128).astype(np.float32)
    abs_groups = np.abs(groups)
    order = np.argsort(-abs_groups, axis=1)
    sorted_abs = np.take_along_axis(abs_groups, order, axis=1)
    csum = np.cumsum(sorted_abs, axis=1)
    ks = np.arange(1, 129, dtype=np.float32).reshape(1, -1)
    score = csum * csum / ks
    best = np.argmax(score, axis=1)
    support = np.zeros_like(groups, dtype=bool)
    row = np.arange(groups.shape[0])[:, None]
    max_k = int(best.max()) + 1
    ranks = np.arange(max_k)[None, :]
    support[row, order[:, :max_k]] = ranks <= best[:, None]
    codes = np.zeros_like(groups, dtype=np.int8)
    codes[support] = np.sign(groups[support]).astype(np.int8)
    sums = np.take_along_axis(csum, best[:, None], axis=1).reshape(-1)
    scales = sums / (best.astype(np.float32) + 1.0)
    return codes.reshape(-1), scales


def trained_ternary_codes(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    groups = weight.reshape(-1, 128).astype(np.float32)
    scales = np.max(np.abs(groups), axis=1)
    denom = np.maximum(scales[:, None], 1e-12)
    ratio = groups / denom
    codes = np.rint(ratio).clip(-1, 1).astype(np.int8)
    recon = codes.astype(np.float32) * scales[:, None]
    err = np.max(np.abs(recon - groups)) if groups.size else 0.0
    return codes.reshape(-1), scales, float(err)


def sample_group_indices(n_groups: int, max_groups: int, seed: int) -> np.ndarray:
    if n_groups <= max_groups:
        return np.arange(n_groups)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_groups, size=max_groups, replace=False))


def analyze_tensor(base: torch.Tensor, trained: torch.Tensor, *, max_groups: int, seed: int) -> dict[str, Any]:
    b = base.detach().float().numpy()
    t = trained.detach().float().numpy()
    if b.shape != t.shape:
        if b.ndim != t.ndim:
            raise ValueError(f"rank mismatch: {b.shape} vs {t.shape}")
        slices = tuple(slice(0, min(a, c)) for a, c in zip(b.shape, t.shape))
        b = b[slices]
        t = t[slices]
    if b.ndim != 2:
        delta = t.astype(np.float32) - b.astype(np.float32)
        return {
            "n_elements": int(b.size),
            "sampled_elements": int(b.size),
            "norm_delta": float(np.linalg.norm(delta.reshape(-1)) / max(np.linalg.norm(b.reshape(-1)), 1e-12)),
            "is_2d": False,
        }
    n_groups = b.size // 128
    group_idx = sample_group_indices(n_groups, max_groups=max_groups, seed=seed)
    flat_idx = (group_idx[:, None] * 128 + np.arange(128)[None, :]).reshape(-1)
    b_sample = b.reshape(-1)[flat_idx]
    t_sample = t.reshape(-1)[flat_idx]
    ptq_code, ptq_scale = optimal_ternary_codes(b_sample)
    trained_code, trained_scale, ternary_err = trained_ternary_codes(t_sample)

    same = ptq_code == trained_code
    ptq_support = ptq_code != 0
    trained_support = trained_code != 0
    support_same = ptq_support == trained_support
    sign_flip = (ptq_code * trained_code) == -1
    activation = (~ptq_support) & trained_support
    deactivation = ptq_support & (~trained_support)
    preserved_nonzero = (ptq_code == trained_code) & trained_support
    scale_ratio = trained_scale / np.maximum(ptq_scale, 1e-12)
    return {
        "n_elements": int(b.size),
        "sampled_groups": int(len(group_idx)),
        "sampled_elements": int(b_sample.size),
        "is_2d": True,
        "ternary_recon_max_err": ternary_err,
        "ptq_zero": float(np.mean(ptq_code == 0)),
        "trained_zero": float(np.mean(trained_code == 0)),
        "code_agreement": float(np.mean(same)),
        "support_agreement": float(np.mean(support_same)),
        "preserved_nonzero": float(np.mean(preserved_nonzero)),
        "sign_flip": float(np.mean(sign_flip)),
        "activation": float(np.mean(activation)),
        "deactivation": float(np.mean(deactivation)),
        "scale_ratio_mean": float(np.mean(scale_ratio)),
        "scale_ratio_p10": float(np.quantile(scale_ratio, 0.10)),
        "scale_ratio_p50": float(np.quantile(scale_ratio, 0.50)),
        "scale_ratio_p90": float(np.quantile(scale_ratio, 0.90)),
        "base_std": float(np.std(b_sample)),
        "trained_std": float(np.std(t_sample)),
        "std_ratio": float(np.std(t_sample) / max(np.std(b_sample), 1e-12)),
    }


def weighted_average(rows: list[dict[str, Any]], key: str) -> float:
    num = sum(row[key] * row["sampled_elements"] for row in rows if key in row)
    den = sum(row["sampled_elements"] for row in rows if key in row)
    return num / den if den else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-groups-per-tensor", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    base_index = load_index(args.base)
    trained_index = load_index(args.trained)
    names = sorted(set(base_index) & set(trained_index))
    rows = []
    for i, name in enumerate(names):
        if not name.endswith(".weight"):
            continue
        base = tensor_from(args.base, base_index, name)
        trained = tensor_from(args.trained, trained_index, name)
        row = {
            "name": name,
            "kind": kind(name),
            "layer": layer(name),
            **analyze_tensor(base, trained, max_groups=args.max_groups_per_tensor, seed=args.seed + i),
        }
        rows.append(row)
        if row.get("is_2d"):
            print(json.dumps({k: row[k] for k in ["name", "kind", "trained_zero", "code_agreement", "support_agreement", "scale_ratio_p50", "ternary_recon_max_err"]}), flush=True)

    two_d = [row for row in rows if row.get("is_2d")]
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in two_d:
        by_kind[row["kind"]].append(row)
        by_layer[row["layer"]].append(row)
    keys = [
        "ptq_zero",
        "trained_zero",
        "code_agreement",
        "support_agreement",
        "preserved_nonzero",
        "sign_flip",
        "activation",
        "deactivation",
        "scale_ratio_mean",
        "scale_ratio_p50",
        "std_ratio",
    ]
    report = {
        "rows": rows,
        "summary": {key: weighted_average(two_d, key) for key in keys},
        "by_kind": {
            k: {"params_sampled": sum(r["sampled_elements"] for r in v), **{key: weighted_average(v, key) for key in keys}}
            for k, v in sorted(by_kind.items())
        },
        "by_layer": {
            k: {"params_sampled": sum(r["sampled_elements"] for r in v), **{key: weighted_average(v, key) for key in keys}}
            for k, v in sorted(by_layer.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else -1)
        },
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
