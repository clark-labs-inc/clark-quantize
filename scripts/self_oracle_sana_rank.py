"""Teacher-native self-oracle ranking for Sana quantized checkpoints.

This offline pilot uses the frozen FP16 Sana transformer as a transition,
representation, and prompt-contrastive oracle. It is intentionally a ranker,
not a training loss:

- run FP16 hard prompts and cache transformer trajectories;
- load each quantized checkpoint, run the same prompts, and cache student states;
- restore FP16 weights and judge the student states with the teacher;
- report path divergence, one-step teacher correction, and token-geometry CKA.
- optionally score final latents with flow-compatible counterfactual energy:
  the teacher should explain a candidate better under the true prompt than under
  minimal "missing object / wrong relation" prompts.
- optionally score counterfactual intervention grounding maps:
  the teacher's spatial response to removing/changing a prompt concept should
  land in the same places on the candidate trajectory as it does on FP16.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import SanaPipeline

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clark_quantize.sana_ckpt import load_compact_checkpoint, validate_quant_names


def is_trunk_lin(n: str, m: nn.Module) -> bool:
    return (
        isinstance(m, nn.Linear)
        and n.startswith("transformer_blocks")
        and (".attn1." in n or ".attn2." in n)
        and any(s in n for s in ("to_q", "to_k", "to_v", "to_out.0"))
    )


def is_trunk_conv(n: str, m: nn.Module) -> bool:
    return isinstance(m, nn.Conv2d) and n.startswith("transformer_blocks") and (
        ".ff.conv_inverted" in n or ".ff.conv_point" in n
    )


def is_quantized_trunk(n: str, m: nn.Module) -> bool:
    return is_trunk_lin(n, m) or is_trunk_conv(n, m)


def parse_candidate(value: str) -> tuple[str, str]:
    if "=" not in value:
        path = value
        return Path(path).stem, path
    label, path = value.split("=", 1)
    return label, path


def tensor_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in kwargs.items():
        if k in ("hidden_states", "encoder_hidden_states", "timestep"):
            continue
        if torch.is_tensor(v):
            out[k] = v.detach()
        else:
            out[k] = v
    return out


def guided(out: torch.Tensor, guidance: float) -> torch.Tensor:
    if out.shape[0] >= 2:
        vu, vc = out[:1].float(), out[1:2].float()
        return vu + guidance * (vc - vu)
    return out[:1].float()


def model_call(tf: nn.Module, rec: dict[str, Any]) -> torch.Tensor:
    out = tf(
        hidden_states=rec["hs"].to(next(tf.parameters()).dtype),
        encoder_hidden_states=rec["ehs"],
        timestep=rec["ts"],
        **rec["extra"],
    )
    return out[0] if isinstance(out, (tuple, list)) else out.sample


def capture_rollout(pipe: SanaPipeline, prompt: str, seed: int, steps: int, guidance: float) -> dict[str, Any]:
    tf = pipe.transformer
    records: list[dict[str, Any]] = []

    def hook(_m, args, kwargs, output):
        out = output[0] if isinstance(output, (tuple, list)) else output.sample
        hs = kwargs.get("hidden_states", args[0] if args else None)
        ehs = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
        ts = kwargs.get("timestep", args[2] if len(args) > 2 else None)
        records.append(
            {
                "hs": hs.detach(),
                "ehs": ehs.detach() if torch.is_tensor(ehs) else ehs,
                "ts": ts.detach() if torch.is_tensor(ts) else ts,
                "extra": tensor_kwargs(kwargs),
                "out": out.detach(),
            }
        )

    handle = tf.register_forward_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            out = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                height=512,
                width=512,
                guidance_scale=guidance,
                output_type="latent",
                generator=torch.Generator("cuda").manual_seed(seed),
            )
    finally:
        handle.remove()
    return {"records": records, "latent": out.images.detach()[:1]}


def capture_trajectory(pipe: SanaPipeline, prompt: str, seed: int, steps: int, guidance: float) -> list[dict[str, Any]]:
    return capture_rollout(pipe, prompt=prompt, seed=seed, steps=steps, guidance=guidance)["records"]


def capture_prompt_context(pipe: SanaPipeline, prompt: str, seed: int, steps: int, guidance: float) -> dict[str, Any]:
    rollout = capture_rollout(pipe, prompt=prompt, seed=seed, steps=steps, guidance=guidance)
    if not rollout["records"]:
        raise RuntimeError(f"no transformer records captured for prompt {prompt!r}")
    rec = rollout["records"][0]
    return {"ehs": rec["ehs"], "extra": rec["extra"]}


def dequant(code: torch.Tensor, log_scales: torch.Tensor, in_features: int) -> torch.Tensor:
    code = code.float()
    scales = torch.exp(log_scales.float()).unsqueeze(-1)
    out, n_groups, group_size = code.shape
    return (code * scales).reshape(out, n_groups * group_size)[:, :in_features]


@torch.no_grad()
def save_fp16_state(qmods: list[tuple[str, nn.Module]], island_params: list[tuple[str, nn.Parameter]]) -> dict[str, Any]:
    return {
        "trunk": {n: m.weight.detach().clone() for n, m in qmods},
        "islands": {n: p.detach().clone() for n, p in island_params},
    }


@torch.no_grad()
def restore_fp16_state(state: dict[str, Any], qmods: list[tuple[str, nn.Module]], island_params: list[tuple[str, nn.Parameter]]) -> None:
    for n, m in qmods:
        m.weight.data.copy_(state["trunk"][n])
    for n, p in island_params:
        p.data.copy_(state["islands"][n])


@torch.no_grad()
def load_checkpoint_into_transformer(
    *,
    path: str,
    qmods: list[tuple[str, nn.Module]],
    island_params: list[tuple[str, nn.Parameter]],
    device: str,
) -> int:
    ck = load_compact_checkpoint(path)
    validate_quant_names(ck, [n for n, _ in qmods])
    for (n, module), code, log_scales, shape in zip(qmods, ck["codes"], ck["log_scales"], ck["shapes"]):
        q = dequant(code, log_scales, int(shape[1])).reshape(module.weight.shape)
        module.weight.data.copy_(q.to(device=device, dtype=module.weight.dtype))
    islands = ck.get("islands", {})
    if isinstance(islands, dict):
        copied = 0
        for n, p in island_params:
            if n in islands:
                p.data.copy_(islands[n].to(device=p.device, dtype=p.dtype))
                copied += 1
        print(f"[load] {Path(path).name} islands_copied={copied}", flush=True)
    else:
        vals = list(islands)
        for (n, p), src in zip(island_params, vals):
            p.data.copy_(src.to(device=p.device, dtype=p.dtype))
    return int(ck.get("step", 0))


def centered_cka_from_tokens(
    a: torch.Tensor,
    b: torch.Tensor,
    projection: torch.Tensor,
    max_tokens: int,
) -> torch.Tensor:
    # Use the first CFG row. Sana hidden states are usually [B, H, W, C];
    # some code paths may already expose [B, tokens, C]. Flatten every spatial
    # token axis and keep the final channel axis.
    za = a[:1].float().reshape(-1, a.shape[-1])
    zb = b[:1].float().reshape(-1, b.shape[-1])
    n = min(za.shape[0], zb.shape[0])
    if n > max_tokens:
        idx = torch.linspace(0, n - 1, max_tokens, device=za.device).long()
        za = za.index_select(0, idx)
        zb = zb.index_select(0, idx)
    else:
        za = za[:n]
        zb = zb[:n]
    proj = projection[: za.shape[-1]].to(za.device, za.dtype)
    za = F.normalize(F.layer_norm(za, za.shape[-1:]) @ proj, dim=-1)
    zb = F.normalize(F.layer_norm(zb, zb.shape[-1:]) @ proj, dim=-1)
    ga = za @ za.T
    gb = zb @ zb.T
    ga = ga - ga.mean(dim=0, keepdim=True) - ga.mean(dim=1, keepdim=True) + ga.mean()
    gb = gb - gb.mean(dim=0, keepdim=True) - gb.mean(dim=1, keepdim=True) + gb.mean()
    return (ga * gb).sum() / (ga.square().sum().sqrt() * gb.square().sum().sqrt() + 1e-8)


def load_counterfactual_specs(path: str | None, prompts: list[str]) -> list[dict[str, Any] | None]:
    if not path:
        return [None for _ in prompts]
    data = json.loads(Path(path).read_text())
    raw_items = data.get("items", data.get("prompts", data if isinstance(data, list) else []))
    if not isinstance(raw_items, list):
        raise ValueError(f"counterfactual file {path} must contain an items array")
    by_prompt = {str(item.get("prompt", "")): item for item in raw_items if isinstance(item, dict)}
    out: list[dict[str, Any] | None] = []
    for i, prompt in enumerate(prompts):
        item = by_prompt.get(prompt)
        if item is None and i < len(raw_items) and isinstance(raw_items[i], dict):
            item = raw_items[i]
        out.append(item)
    return out


def make_energy_samples(
    *,
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    x0_shape: torch.Size,
    count: int,
    noise_per_timestep: int,
    seed: int,
    device: str,
) -> list[dict[str, torch.Tensor]]:
    if count <= 0 or noise_per_timestep <= 0:
        return []
    n_steps = min(len(timesteps), max(1, len(sigmas) - 1))
    # Bias toward mid/low noise; high noise is less useful for the object/detail
    # failures this oracle is meant to catch.
    lo = max(0, int(0.35 * (n_steps - 1)))
    hi = n_steps - 1
    idxs = torch.linspace(lo, hi, count).round().long().clamp(0, n_steps - 1).unique().tolist()
    g = torch.Generator(device=device).manual_seed(seed)
    sigma_max = sigmas[:n_steps].float().amax().clamp_min(1e-8)
    samples: list[dict[str, torch.Tensor]] = []
    for idx in idxs:
        sigma = (sigmas[idx].float() / sigma_max).clamp(0.0, 1.0)
        timestep = timesteps[idx].detach()
        for _ in range(noise_per_timestep):
            samples.append(
                {
                    "idx": torch.tensor(idx, device=device),
                    "sigma": sigma.to(device),
                    "timestep": timestep.to(device),
                    "noise": torch.randn(x0_shape, generator=g, device=device, dtype=torch.float32),
                }
            )
    return samples


def flow_xt_target(x0: torch.Tensor, noise: torch.Tensor, sigma01: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    x0f = x0.float()
    nf = noise.float()
    t = sigma01.float().reshape(1, *([1] * (x0f.ndim - 1)))
    if mode == "rectified-sigma":
        xt = (1.0 - t) * x0f + t * nf
        target_velocity = nf - x0f
        return xt, target_velocity
    raise ValueError(f"unsupported energy noising mode {mode!r}")


def timestep_for_batch(timestep: torch.Tensor, batch: int) -> torch.Tensor:
    ts = timestep.detach()
    if ts.ndim == 0:
        return ts.reshape(1).repeat(batch)
    if ts.shape[0] == batch:
        return ts
    if ts.numel() == 1:
        return ts.reshape(1).repeat(batch)
    return ts[:batch]


@torch.no_grad()
def flow_conditional_energy(
    *,
    tf: nn.Module,
    x0: torch.Tensor,
    context: dict[str, Any],
    samples: list[dict[str, torch.Tensor]],
    guidance: float,
    noising_mode: str,
) -> float:
    if not samples:
        return 0.0
    p0 = next(tf.parameters())
    ehs = context["ehs"]
    batch = int(ehs.shape[0]) if torch.is_tensor(ehs) and ehs.ndim > 0 else 1
    values = []
    for sample in samples:
        xt, target_velocity = flow_xt_target(x0, sample["noise"], sample["sigma"], noising_mode)
        lat = xt.repeat(batch, *([1] * (xt.ndim - 1))).to(device=p0.device, dtype=p0.dtype)
        out = tf(
            hidden_states=lat,
            encoder_hidden_states=ehs,
            timestep=timestep_for_batch(sample["timestep"], batch),
            **context["extra"],
        )
        pred = out[0] if isinstance(out, (tuple, list)) else out.sample
        pred_g = guided(pred, guidance)
        err = (pred_g.float() - target_velocity.to(pred_g.device).float()).square().flatten(1).mean(1)
        values.append(err.mean())
    return float(torch.stack(values).mean().item())


@torch.no_grad()
def trajectory_counterfactual_margin(
    *,
    tf: nn.Module,
    records: list[dict[str, Any]],
    positive_context: dict[str, Any],
    negative_contexts: list[dict[str, Any]],
    guidance: float,
    max_records: int,
) -> tuple[float, float, list[float]]:
    if not records or not negative_contexts:
        return 0.0, 0.0, []
    n = len(records)
    lo = max(0, int(0.35 * (n - 1)))
    hi = n - 1
    if max_records > 0:
        idxs = torch.linspace(lo, hi, max_records).round().long().clamp(0, n - 1).unique().tolist()
    else:
        idxs = list(range(lo, hi + 1))
    p0 = next(tf.parameters())
    neg_accum = [0.0 for _ in negative_contexts]
    denom_count = 0
    for idx in idxs:
        rec = records[int(idx)]
        hs = rec["hs"].to(device=p0.device, dtype=p0.dtype)
        ts = rec["ts"]
        pos_out = tf(
            hidden_states=hs,
            encoder_hidden_states=positive_context["ehs"],
            timestep=ts,
            **positive_context["extra"],
        )
        pos = guided(pos_out[0] if isinstance(pos_out, (tuple, list)) else pos_out.sample, guidance)
        denom = pos.float().square().mean().clamp_min(1e-8)
        for j, neg_context in enumerate(negative_contexts):
            neg_out = tf(
                hidden_states=hs,
                encoder_hidden_states=neg_context["ehs"],
                timestep=ts,
                **neg_context["extra"],
            )
            neg = guided(neg_out[0] if isinstance(neg_out, (tuple, list)) else neg_out.sample, guidance)
            neg_accum[j] += float(((neg.float() - pos.float()).square().mean() / denom).item())
        denom_count += 1
    neg_values = [v / max(1, denom_count) for v in neg_accum]
    hardest = min(neg_values) if neg_values else 0.0
    mean_neg = sum(neg_values) / max(1, len(neg_values))
    return hardest, mean_neg, neg_values


def selected_record_indices(n: int, count: int) -> list[int]:
    if n <= 0 or count <= 0:
        return []
    lo = max(0, int(0.35 * (n - 1)))
    hi = n - 1
    return torch.linspace(lo, hi, count).round().long().clamp(0, n - 1).unique().tolist()


def output_mean_dot(
    x: torch.Tensor,
    y: torch.Tensor,
    inv_var: torch.Tensor | None = None,
) -> torch.Tensor:
    x = x.float()
    y = y.float()
    if inv_var is not None:
        x = x * inv_var
    return (x * y).flatten(1).mean(dim=1)


def coefficient_cvar(values: list[float], frac: float) -> float:
    if not values:
        return 0.0
    k = max(1, math.ceil(len(values) * max(0.0, min(1.0, frac))))
    return float(sum(sorted(values)[:k]) / k)


def coefficient_softmin(values: list[float], temp: float) -> float:
    if not values:
        return 0.0
    if temp <= 0:
        return float(min(values))
    x = torch.tensor(values, dtype=torch.float32)
    return float((-temp * torch.logsumexp(-x / temp, dim=0) + temp * math.log(float(len(values)))).item())


def assertion_specs(spec: dict[str, Any] | None, positive_override: str | None = None) -> list[dict[str, str]]:
    if not spec:
        return []
    # The positive branch must match the prompt that generated the rollout
    # state. Otherwise even FP16 does not calibrate to coefficient=1.
    base_positive = str(positive_override or spec.get("positive") or spec.get("prompt") or "").strip()
    out = []
    for i, item in enumerate(spec.get("assertions", []) or []):
        positive = base_positive
        negative = str(item.get("negative") or "").strip()
        if positive and negative:
            out.append(
                {
                    "id": str(item.get("id") or f"assertion_{i}"),
                    "positive": positive,
                    "negative": negative,
                }
            )
    if out:
        return out
    positive = str(spec.get("positive") or spec.get("prompt") or "").strip()
    for i, negative in enumerate(str(v).strip() for v in spec.get("negatives", []) if str(v).strip()):
        if positive and negative:
            out.append({"id": f"negative_{i}", "positive": positive, "negative": negative})
    return out


@torch.no_grad()
def score_cvf_coefficients(
    *,
    pipe: SanaPipeline,
    student_records: list[dict[str, Any]],
    spec: dict[str, Any] | None,
    rollout_prompt: str,
    context_cache: dict[str, dict[str, Any]],
    steps: int,
    guidance: float,
    max_records: int,
    threshold: float,
    sep_min: float,
    cvar_frac: float,
    softmin_temp: float,
) -> dict[str, Any] | None:
    assertions = assertion_specs(spec, positive_override=rollout_prompt)
    if not assertions or max_records <= 0:
        return None
    idxs = selected_record_indices(len(student_records), max_records)
    if not idxs:
        return None
    rows = []
    all_coeffs = []
    all_valid_coeffs = []
    for assertion in assertions:
        positive = assertion["positive"]
        negative = assertion["negative"]
        if positive not in context_cache:
            context_cache[positive] = capture_prompt_context(pipe, positive, seed=75000 + len(context_cache), steps=steps, guidance=guidance)
        if negative not in context_cache:
            context_cache[negative] = capture_prompt_context(pipe, negative, seed=75000 + len(context_cache), steps=steps, guidance=guidance)
        pos_context = context_cache[positive]
        neg_context = context_cache[negative]
        coeffs = []
        valid_coeffs = []
        separabilities = []
        p0 = next(pipe.transformer.parameters())
        for idx in idxs:
            rec = student_records[int(idx)]
            hs = rec["hs"].to(device=p0.device, dtype=p0.dtype)
            ts = rec["ts"]
            pos_out = pipe.transformer(
                hidden_states=hs,
                encoder_hidden_states=pos_context["ehs"],
                timestep=ts,
                **pos_context["extra"],
            )
            neg_out = pipe.transformer(
                hidden_states=hs,
                encoder_hidden_states=neg_context["ehs"],
                timestep=ts,
                **neg_context["extra"],
            )
            v_student = guided(rec["out"], guidance)
            v_pos = guided(pos_out[0] if isinstance(pos_out, (tuple, list)) else pos_out.sample, guidance)
            v_neg = guided(neg_out[0] if isinstance(neg_out, (tuple, list)) else neg_out.sample, guidance)
            direction = v_pos.detach().float() - v_neg.detach().float()
            relative = v_student.float() - v_neg.detach().float()
            denom = output_mean_dot(direction, direction).clamp_min(1e-12)
            coeff = output_mean_dot(relative, direction) / denom
            sep = denom.sqrt()
            coeff_value = float(coeff.mean().item())
            sep_value = float(sep.mean().item())
            coeffs.append(coeff_value)
            separabilities.append(sep_value)
            if sep_value >= sep_min:
                valid_coeffs.append(coeff_value)
        usable = len(valid_coeffs) > 0
        cvar = coefficient_cvar(valid_coeffs, cvar_frac) if usable else 0.0
        softmin = coefficient_softmin(valid_coeffs, softmin_temp) if usable else 0.0
        passed = (not usable) or cvar >= threshold
        all_coeffs.extend(coeffs)
        all_valid_coeffs.extend(valid_coeffs)
        rows.append(
            {
                "id": assertion["id"],
                "positive": positive,
                "negative": negative,
                "coefficients": coeffs,
                "valid_coefficients": valid_coeffs,
                "separabilities": separabilities,
                "mean_coefficient": float(sum(valid_coeffs) / len(valid_coeffs)) if usable else 0.0,
                "cvar": cvar,
                "softmin": softmin,
                "mean_separability": float(sum(separabilities) / len(separabilities)) if separabilities else 0.0,
                "usable": usable,
                "pass": passed,
            }
        )
    valid_rows = [r for r in rows if r["usable"]]
    failed_rows = [r for r in valid_rows if not r["pass"]]
    valid_cvars = [float(r["cvar"]) for r in valid_rows]
    valid_softmins = [float(r["softmin"]) for r in valid_rows]
    return {
        "prompt_id": spec.get("prompt_id", "prompt") if spec else "prompt",
        "threshold": threshold,
        "sep_min": sep_min,
        "cvar_frac": cvar_frac,
        "assertions": rows,
        "usable_assertions": len(valid_rows),
        "failed_assertions": len(failed_rows),
        "mean_cvar": float(sum(valid_cvars) / len(valid_cvars)) if valid_cvars else 0.0,
        "min_cvar": float(min(valid_cvars)) if valid_cvars else 0.0,
        "mean_softmin": float(sum(valid_softmins) / len(valid_softmins)) if valid_softmins else 0.0,
        "min_softmin": float(min(valid_softmins)) if valid_softmins else 0.0,
        "mean_coefficient": float(sum(all_valid_coeffs) / len(all_valid_coeffs)) if all_valid_coeffs else 0.0,
        "failed": failed_rows,
    }


def spatial_delta_map(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """Normalize a teacher output-delta map to a spatial probability map."""
    delta = (pos.float() - neg.float()).square()
    if delta.ndim == 4:
        # Diffusers transformer outputs are latent-like [B, C, H, W] here.
        if delta.shape[1] <= delta.shape[-1]:
            delta = delta.mean(dim=1)
        else:
            delta = delta.mean(dim=-1)
    elif delta.ndim == 3:
        delta = delta.mean(dim=-1)
    else:
        delta = delta.flatten(1)
    delta = delta[:1].reshape(1, -1).sqrt()
    return delta / delta.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def js_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(1, -1)
    b = b.float().reshape(1, -1)
    n = min(a.shape[-1], b.shape[-1])
    if a.shape[-1] != n:
        a = F.interpolate(a.reshape(1, 1, 1, -1), size=(1, n), mode="nearest").reshape(1, n)
    if b.shape[-1] != n:
        b = F.interpolate(b.reshape(1, 1, 1, -1), size=(1, n), mode="nearest").reshape(1, n)
    a = a.clamp_min(1e-8)
    b = b.clamp_min(1e-8)
    a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    b = b / b.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    m = 0.5 * (a + b)
    js = 0.5 * ((a * (a.log() - m.log())).sum(dim=-1) + (b * (b.log() - m.log())).sum(dim=-1))
    # JS with natural logs is bounded by ln(2). Convert to a 0..1 similarity.
    sim = 1.0 - float((js / math.log(2.0)).clamp(0.0, 1.0).mean().item())
    return max(0.0, min(1.0, sim))


@torch.no_grad()
def teacher_counterfactual_maps(
    *,
    tf: nn.Module,
    rec: dict[str, Any],
    positive_context: dict[str, Any],
    negative_contexts: list[dict[str, Any]],
    guidance: float,
) -> list[torch.Tensor]:
    if not negative_contexts:
        return []
    p0 = next(tf.parameters())
    hs = rec["hs"].to(device=p0.device, dtype=p0.dtype)
    ts = rec["ts"]
    pos_out = tf(
        hidden_states=hs,
        encoder_hidden_states=positive_context["ehs"],
        timestep=ts,
        **positive_context["extra"],
    )
    pos = guided(pos_out[0] if isinstance(pos_out, (tuple, list)) else pos_out.sample, guidance)
    maps = []
    for neg_context in negative_contexts:
        neg_out = tf(
            hidden_states=hs,
            encoder_hidden_states=neg_context["ehs"],
            timestep=ts,
            **neg_context["extra"],
        )
        neg = guided(neg_out[0] if isinstance(neg_out, (tuple, list)) else neg_out.sample, guidance)
        maps.append(spatial_delta_map(pos, neg))
    return maps


@torch.no_grad()
def score_grounding_maps(
    *,
    pipe: SanaPipeline,
    fp16_records: list[dict[str, Any]],
    student_records: list[dict[str, Any]],
    spec: dict[str, Any] | None,
    context_cache: dict[str, dict[str, Any]],
    steps: int,
    guidance: float,
    max_records: int,
    critical_threshold: float,
) -> dict[str, Any] | None:
    if not spec or max_records <= 0:
        return None
    positive = str(spec.get("positive") or spec.get("prompt") or "")
    negatives = [str(v) for v in spec.get("negatives", []) if str(v).strip()]
    if not positive or not negatives:
        return None
    if positive not in context_cache:
        context_cache[positive] = capture_prompt_context(pipe, positive, seed=73000 + len(context_cache), steps=steps, guidance=guidance)
    neg_contexts = []
    for neg in negatives:
        if neg not in context_cache:
            context_cache[neg] = capture_prompt_context(pipe, neg, seed=73000 + len(context_cache), steps=steps, guidance=guidance)
        neg_contexts.append(context_cache[neg])

    n = min(len(fp16_records), len(student_records))
    lo = max(0, int(0.35 * (n - 1)))
    hi = n - 1
    idxs = torch.linspace(lo, hi, max_records).round().long().clamp(0, n - 1).unique().tolist()
    sims_by_negative = [0.0 for _ in neg_contexts]
    counts_by_negative = [0 for _ in neg_contexts]
    prompt_min = 1.0
    for idx in idxs:
        fp16_maps = teacher_counterfactual_maps(
            tf=pipe.transformer,
            rec=fp16_records[int(idx)],
            positive_context=context_cache[positive],
            negative_contexts=neg_contexts,
            guidance=guidance,
        )
        student_maps = teacher_counterfactual_maps(
            tf=pipe.transformer,
            rec=student_records[int(idx)],
            positive_context=context_cache[positive],
            negative_contexts=neg_contexts,
            guidance=guidance,
        )
        for j, (fp16_map, student_map) in enumerate(zip(fp16_maps, student_maps)):
            sim = js_similarity(student_map, fp16_map)
            sims_by_negative[j] += sim
            counts_by_negative[j] += 1
            prompt_min = min(prompt_min, sim)
    negative_rows = []
    all_sims = []
    for neg, total, count in zip(negatives, sims_by_negative, counts_by_negative):
        mean_sim = total / max(1, count)
        negative_rows.append({"negative": neg, "map_similarity": mean_sim})
        all_sims.append(mean_sim)
    mean_sim = sum(all_sims) / max(1, len(all_sims))
    return {
        "prompt_id": spec.get("prompt_id", "prompt"),
        "positive": positive,
        "mean_similarity": mean_sim,
        "min_similarity": prompt_min,
        "head": mean_sim,
        "critical_failure": prompt_min < critical_threshold,
        "negatives": negative_rows,
    }


def sigmoid_head(x: float, temp: float) -> float:
    temp = max(temp, 1e-8)
    z = max(-60.0, min(60.0, x / temp))
    return 1.0 / (1.0 + math.exp(-z))


@torch.no_grad()
def score_energy_margins(
    *,
    pipe: SanaPipeline,
    x0: torch.Tensor,
    records: list[dict[str, Any]],
    prompt_specs: list[dict[str, Any] | None],
    context_cache: dict[str, dict[str, Any]],
    samples: list[dict[str, torch.Tensor]],
    steps: int,
    guidance: float,
    noising_mode: str,
    energy_temp: float,
) -> dict[str, Any]:
    margins = []
    prompt_rows = []
    for prompt_index, spec in enumerate(prompt_specs):
        if not spec:
            continue
        positive = str(spec.get("positive") or spec.get("prompt") or "")
        negatives = [str(v) for v in spec.get("negatives", []) if str(v).strip()]
        if not positive or not negatives:
            continue
        if positive not in context_cache:
            context_cache[positive] = capture_prompt_context(pipe, positive, seed=71000 + len(context_cache), steps=steps, guidance=guidance)
        neg_contexts = []
        for neg in negatives:
            if neg not in context_cache:
                context_cache[neg] = capture_prompt_context(pipe, neg, seed=71000 + len(context_cache), steps=steps, guidance=guidance)
            neg_contexts.append(context_cache[neg])
        if noising_mode == "trajectory-margin":
            hardest_negative, mean_negative, neg_energies = trajectory_counterfactual_margin(
                tf=pipe.transformer,
                records=records,
                positive_context=context_cache[positive],
                negative_contexts=neg_contexts,
                guidance=guidance,
                max_records=len(samples),
            )
            e_pos = 0.0
            margin = hardest_negative
        else:
            e_pos = flow_conditional_energy(
                tf=pipe.transformer,
                x0=x0,
                context=context_cache[positive],
                samples=samples,
                guidance=guidance,
                noising_mode=noising_mode,
            )
            neg_energies = [
                flow_conditional_energy(
                    tf=pipe.transformer,
                    x0=x0,
                    context=neg_context,
                    samples=samples,
                    guidance=guidance,
                    noising_mode=noising_mode,
                )
                for neg_context in neg_contexts
            ]
            hardest_negative = min(neg_energies)
            mean_negative = sum(neg_energies) / max(1, len(neg_energies))
            margin = hardest_negative - e_pos
        margins.append(margin)
        prompt_rows.append(
            {
                "prompt_index": prompt_index,
                "prompt_id": spec.get("prompt_id", f"prompt_{prompt_index}"),
                "positive": positive,
                "e_positive": e_pos,
                "e_hardest_negative": hardest_negative,
                "e_mean_negative": mean_negative,
                "margin": margin,
                "head": sigmoid_head(margin, energy_temp),
            }
        )
    if not margins:
        return {"enabled": False, "mean_margin": 0.0, "min_margin": 0.0, "head": 0.0, "prompts": []}
    mean_margin = float(sum(margins) / len(margins))
    min_margin = float(min(margins))
    return {
        "enabled": True,
        "mean_margin": mean_margin,
        "min_margin": min_margin,
        "head": sigmoid_head(mean_margin, energy_temp),
        "min_head": sigmoid_head(min_margin, energy_temp),
        "prompts": prompt_rows,
    }


@torch.no_grad()
def score_candidate(
    *,
    pipe: SanaPipeline,
    fp16_refs: list[dict[str, Any]],
    student_refs: list[dict[str, Any]],
    prompts: list[str],
    seeds: list[int],
    sigmas: torch.Tensor,
    projection: torch.Tensor,
    steps: int,
    guidance: float,
    step_stride: int,
    max_tokens: int,
    energy_specs: list[dict[str, Any] | None],
    energy_context_cache: dict[str, dict[str, Any]],
    energy_samples: list[dict[str, torch.Tensor]],
    energy_noising: str,
    energy_temp: float,
    grounding_records: int,
    grounding_threshold: float,
    cvf_records: int,
    cvf_threshold: float,
    cvf_sep_min: float,
    cvf_cvar_frac: float,
    cvf_softmin_temp: float,
) -> dict[str, Any]:
    # The caller restores FP16 weights before this function computes teacher outputs.
    tf = pipe.transformer
    path_losses = []
    fixed_losses = []
    cka_scores = []
    energy_rows = []
    grounding_rows = []
    cvf_rows = []
    for prompt_index, (fp16_pack, student_pack) in enumerate(zip(fp16_refs, student_refs)):
        fp16_traj = fp16_pack["records"]
        student_traj = student_pack["records"]
        n_steps = min(len(fp16_traj), len(student_traj), steps)
        for k in range(0, n_steps, step_stride):
            srec = student_traj[k]
            tref = fp16_traj[k]
            out_teacher = model_call(tf, srec)
            v_teacher = guided(out_teacher, guidance)
            v_student = guided(srec["out"], guidance)
            denom = v_teacher.square().mean().clamp_min(1e-8)
            path_losses.append(((v_student - v_teacher).square().mean() / denom).item())
            cka = centered_cka_from_tokens(srec["hs"], tref["hs"], projection, max_tokens=max_tokens)
            cka_scores.append(float(cka.item()))
            if k + 1 < n_steps:
                dt = sigmas[k + 1] - sigmas[k]
                teacher_next = srec["hs"][:1].float() + dt * v_teacher
                student_next = student_traj[k + 1]["hs"][:1].float()
                fixed_losses.append(
                    ((student_next - teacher_next).square().mean() / student_next.square().mean().clamp_min(1e-8)).item()
                )
        if any(energy_specs) and energy_samples:
            prompt_energy = score_energy_margins(
                pipe=pipe,
                x0=student_pack["latent"],
                records=student_traj,
                prompt_specs=[energy_specs[prompt_index]],
                context_cache=energy_context_cache,
                samples=energy_samples,
                steps=steps,
                guidance=guidance,
                noising_mode=energy_noising,
                energy_temp=energy_temp,
            )
            if prompt_energy.get("enabled"):
                energy_rows.extend(prompt_energy["prompts"])
        if any(energy_specs) and grounding_records > 0:
            grounding = score_grounding_maps(
                pipe=pipe,
                fp16_records=fp16_traj,
                student_records=student_traj,
                spec=energy_specs[prompt_index],
                context_cache=energy_context_cache,
                steps=steps,
                guidance=guidance,
                max_records=grounding_records,
                critical_threshold=grounding_threshold,
            )
            if grounding:
                grounding_rows.append(grounding)
        if any(energy_specs) and cvf_records > 0:
            cvf = score_cvf_coefficients(
                pipe=pipe,
                student_records=student_traj,
                spec=energy_specs[prompt_index],
                rollout_prompt=prompts[prompt_index],
                context_cache=energy_context_cache,
                steps=steps,
                guidance=guidance,
                max_records=cvf_records,
                threshold=cvf_threshold,
                sep_min=cvf_sep_min,
                cvar_frac=cvf_cvar_frac,
                softmin_temp=cvf_softmin_temp,
            )
            if cvf:
                cvf_rows.append(cvf)
    path = float(sum(path_losses) / max(1, len(path_losses)))
    fixed = float(sum(fixed_losses) / max(1, len(fixed_losses)))
    cka = float(sum(cka_scores) / max(1, len(cka_scores)))
    geometry_loss = 1.0 - cka
    if energy_rows:
        energy_margins = [float(r["margin"]) for r in energy_rows]
        energy = {
            "enabled": True,
            "mean_margin": float(sum(energy_margins) / len(energy_margins)),
            "min_margin": float(min(energy_margins)),
            "head": sigmoid_head(float(sum(energy_margins) / len(energy_margins)), energy_temp),
            "min_head": sigmoid_head(float(min(energy_margins)), energy_temp),
            "prompts": energy_rows,
        }
    else:
        energy = {"enabled": False, "mean_margin": 0.0, "min_margin": 0.0, "head": 0.0, "min_head": 0.0, "prompts": []}
    if grounding_rows:
        grounding_mean = float(sum(float(r["mean_similarity"]) for r in grounding_rows) / len(grounding_rows))
        grounding_min = float(min(float(r["min_similarity"]) for r in grounding_rows))
        grounding = {
            "enabled": True,
            "mean_similarity": grounding_mean,
            "min_similarity": grounding_min,
            "head": grounding_mean,
            "min_head": grounding_min,
            "prompts": grounding_rows,
        }
    else:
        grounding = {"enabled": False, "mean_similarity": 0.0, "min_similarity": 0.0, "head": 0.0, "min_head": 0.0, "prompts": []}
    if cvf_rows:
        usable_rows = [r for r in cvf_rows if int(r.get("usable_assertions", 0)) > 0]
        failed_assertions = [a for r in cvf_rows for a in r.get("failed", [])]
        min_cvars = [float(r["min_cvar"]) for r in usable_rows]
        mean_cvars = [float(r["mean_cvar"]) for r in usable_rows]
        cvf = {
            "enabled": True,
            "threshold": cvf_threshold,
            "sep_min": cvf_sep_min,
            "cvar_frac": cvf_cvar_frac,
            "softmin_temp": cvf_softmin_temp,
            "mean_cvar": float(sum(mean_cvars) / len(mean_cvars)) if mean_cvars else 0.0,
            "min_cvar": float(min(min_cvars)) if min_cvars else 0.0,
            "failed_assertions": len(failed_assertions),
            "prompts": cvf_rows,
        }
    else:
        cvf = {
            "enabled": False,
            "threshold": cvf_threshold,
            "sep_min": cvf_sep_min,
            "cvar_frac": cvf_cvar_frac,
            "softmin_temp": cvf_softmin_temp,
            "mean_cvar": 0.0,
            "min_cvar": 0.0,
            "failed_assertions": 0,
            "prompts": [],
        }
    heads = {
        "path": math.exp(-path / 0.08),
        "fixed": math.exp(-fixed / 0.08),
        "geometry": max(0.0, min(1.0, cka)),
        "teacher": math.exp(-(0.65 * path + 0.35 * fixed) / 0.08),
        "composition": math.exp(-(0.45 * path + 0.25 * fixed + 0.30 * geometry_loss) / 0.10),
    }
    if energy["enabled"]:
        heads["energy"] = energy["head"]
        heads["energy_min"] = energy["min_head"]
    if grounding["enabled"]:
        heads["grounding"] = grounding["head"]
        heads["grounding_min"] = grounding["min_head"]
    if cvf["enabled"]:
        heads["cvf"] = max(0.0, min(1.0, cvf["mean_cvar"]))
        heads["cvf_min"] = cvf["min_cvar"]
    if energy["enabled"] and grounding["enabled"]:
        score = 100.0 * (
            0.22 * heads["path"]
            + 0.16 * heads["fixed"]
            + 0.17 * heads["geometry"]
            + 0.10 * heads["composition"]
            + 0.15 * heads["energy_min"]
            + 0.20 * heads["grounding_min"]
        )
    elif energy["enabled"]:
        score = 100.0 * (
            0.28 * heads["path"]
            + 0.20 * heads["fixed"]
            + 0.20 * heads["geometry"]
            + 0.12 * heads["composition"]
            + 0.20 * heads["energy_min"]
        )
    elif grounding["enabled"]:
        score = 100.0 * (
            0.28 * heads["path"]
            + 0.20 * heads["fixed"]
            + 0.20 * heads["geometry"]
            + 0.12 * heads["composition"]
            + 0.20 * heads["grounding_min"]
        )
    else:
        score = 100.0 * (
            0.34 * heads["path"]
            + 0.24 * heads["fixed"]
            + 0.24 * heads["geometry"]
            + 0.18 * heads["composition"]
        )
    return {
        "losses": {
            "path": path,
            "fixed": fixed,
            "geometry": geometry_loss,
            "cka": cka,
            "energy_margin_mean": energy["mean_margin"],
            "energy_margin_min": energy["min_margin"],
            "grounding_mean": grounding["mean_similarity"],
            "grounding_min": grounding["min_similarity"],
            "cvf_mean_cvar": cvf["mean_cvar"],
            "cvf_min_cvar": cvf["min_cvar"],
        },
        "head_mean": heads,
        "energy": energy,
        "grounding": grounding,
        "cvf": cvf,
        "critical_failures": [
            *[r for r in energy.get("prompts", []) if float(r["margin"]) <= 0.0],
            *[r for r in grounding.get("prompts", []) if bool(r.get("critical_failure"))],
            *[
                {"type": "cvf", "prompt_id": r.get("prompt_id"), **a}
                for r in cvf.get("prompts", [])
                for a in r.get("failed", [])
            ],
        ],
        "score": score,
        "samples": {
            "path": len(path_losses),
            "fixed": len(fixed_losses),
            "geometry": len(cka_scores),
            "energy": len(energy_rows),
            "grounding": len(grounding_rows),
            "cvf": len(cvf_rows),
        },
    }


def write_markdown(rows: list[dict[str, Any]], path: str) -> None:
    lines = [
        "| rank | score | label | path | fixed | geometry | energy_min | grounding_min | cvf_min | failures | ckpt |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, row in enumerate(rows, 1):
        losses = row["losses"]
        heads = row["head_mean"]
        lines.append(
            f"| {i} | {row['score']:.2f} | {row['label']} | {heads['path']:.3f} | "
            f"{heads['fixed']:.3f} | {heads['geometry']:.3f} | {heads.get('energy_min', 0.0):.3f} | "
            f"{heads.get('grounding_min', 0.0):.3f} | "
            f"{heads.get('cvf_min', 0.0):.3f} | "
            f"{len(row.get('critical_failures', []))} | `{row['path']}` |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", required=True, help="label=/path/to/ckpt.pt; repeatable")
    ap.add_argument("--prompts", default="shared_hard.json")
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--model", default="models/Sana_1600M_512px")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--seed-base", type=int, default=9000, help="Base seed for prompt rollouts; prompt i uses seed-base+i")
    ap.add_argument("--step-stride", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--proj-dim", type=int, default=64)
    ap.add_argument("--projection-seed", type=int, default=1234)
    ap.add_argument("--counterfactuals", default="", help="JSON file with positive/negative prompts for energy margins")
    ap.add_argument("--energy-timesteps", type=int, default=0, help="Enable contrastive energy with this many mid/low-noise timesteps")
    ap.add_argument("--energy-noise", type=int, default=2, help="Noise draws per energy timestep")
    ap.add_argument("--energy-seed", type=int, default=4242)
    ap.add_argument("--energy-temp", type=float, default=0.02)
    ap.add_argument("--energy-noising", default="trajectory-margin", choices=["trajectory-margin", "rectified-sigma"])
    ap.add_argument("--grounding-records", type=int, default=0, help="Enable counterfactual intervention maps with this many mid/low-noise records")
    ap.add_argument("--grounding-threshold", type=float, default=0.60, help="Prompt-level critical failure threshold for min intervention-map similarity")
    ap.add_argument("--cvf-records", type=int, default=0, help="Enable counterfactual vector-field coefficients with this many mid/low-noise records")
    ap.add_argument("--cvf-threshold", type=float, default=0.50, help="CVaR coefficient threshold for each usable semantic assertion")
    ap.add_argument("--cvf-sep-min", type=float, default=1e-4, help="Minimum teacher branch separability for a CVF assertion observation")
    ap.add_argument("--cvf-cvar-frac", type=float, default=0.20, help="Worst-fraction aggregate for CVF coefficients")
    ap.add_argument("--cvf-softmin-temp", type=float, default=0.10)
    ap.add_argument("--out-json", default="self_oracle_rank.json")
    ap.add_argument("--out-md", default="self_oracle_rank.md")
    args = ap.parse_args()

    device = "cuda"
    prompts = json.load(open(args.prompts))[: args.n_prompts]
    seeds = [args.seed_base + i for i in range(len(prompts))]
    pipe = SanaPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe.scheduler.set_timesteps(args.steps, device=device)
    sigmas = pipe.scheduler.sigmas.float().to(device)
    tf = pipe.transformer
    qmods = [(n, m) for n, m in tf.named_modules() if is_quantized_trunk(n, m)]
    island_params = [
        (n, p)
        for n, p in tf.named_parameters()
        if any(k in n for k in ("patch_embed", "time_embed", "caption_projection", "caption_norm", "scale_shift_table", "norm_out", "proj_out", "conv_depth"))
    ]
    fp16_state = save_fp16_state(qmods, island_params)
    print(f"[init] qmods={len(qmods)} islands={len(island_params)} prompts={len(prompts)}", flush=True)

    print("[fp16] capturing reference trajectories", flush=True)
    fp16_refs = [
        capture_rollout(pipe, prompt=p, seed=seed, steps=args.steps, guidance=args.guidance)
        for p, seed in zip(prompts, seeds)
    ]
    hidden_dim = int(fp16_refs[0]["records"][0]["hs"].shape[-1])
    gen = torch.Generator(device=device).manual_seed(args.projection_seed)
    projection = torch.empty(hidden_dim, args.proj_dim, device=device).bernoulli_(0.5, generator=gen)
    projection.mul_(2).sub_(1).div_(math.sqrt(args.proj_dim))
    energy_specs = load_counterfactual_specs(args.counterfactuals or None, prompts)
    energy_context_cache: dict[str, dict[str, Any]] = {}
    energy_samples = []
    if args.energy_timesteps > 0 and any(energy_specs):
        energy_samples = make_energy_samples(
            sigmas=sigmas,
            timesteps=pipe.scheduler.timesteps,
            x0_shape=fp16_refs[0]["latent"].shape,
            count=args.energy_timesteps,
            noise_per_timestep=args.energy_noise,
            seed=args.energy_seed,
            device=device,
        )
        print(
            f"[energy] specs={sum(1 for s in energy_specs if s)} samples={len(energy_samples)} "
            f"mode={args.energy_noising}",
            flush=True,
        )

    rows = []
    for raw in args.candidate:
        label, ckpt_path = parse_candidate(raw)
        print(f"[candidate] {label} {ckpt_path}", flush=True)
        if ckpt_path.lower() in ("fp16", "teacher", "reference", "-"):
            restore_fp16_state(fp16_state, qmods, island_params)
            step = 0
            student_pack = fp16_refs
        else:
            restore_fp16_state(fp16_state, qmods, island_params)
            step = load_checkpoint_into_transformer(path=ckpt_path, qmods=qmods, island_params=island_params, device=device)
            student_pack = [
                capture_rollout(pipe, prompt=p, seed=seed, steps=args.steps, guidance=args.guidance)
                for p, seed in zip(prompts, seeds)
            ]
            restore_fp16_state(fp16_state, qmods, island_params)
        scored = score_candidate(
            pipe=pipe,
            fp16_refs=fp16_refs,
            student_refs=student_pack,
            prompts=prompts,
            seeds=seeds,
            sigmas=sigmas,
            projection=projection,
            steps=args.steps,
            guidance=args.guidance,
            step_stride=args.step_stride,
            max_tokens=args.max_tokens,
            energy_specs=energy_specs,
            energy_context_cache=energy_context_cache,
            energy_samples=energy_samples,
            energy_noising=args.energy_noising,
            energy_temp=args.energy_temp,
            grounding_records=args.grounding_records,
            grounding_threshold=args.grounding_threshold,
            cvf_records=args.cvf_records,
            cvf_threshold=args.cvf_threshold,
            cvf_sep_min=args.cvf_sep_min,
            cvf_cvar_frac=args.cvf_cvar_frac,
            cvf_softmin_temp=args.cvf_softmin_temp,
        )
        rows.append({"label": label, "path": ckpt_path, "step": step, **scored})
        print(
            f"[score] {label} score={scored['score']:.2f} "
            f"path={scored['losses']['path']:.4f} fixed={scored['losses']['fixed']:.4f} "
            f"cka={scored['losses']['cka']:.4f} energy_min={scored['losses']['energy_margin_min']:.4g} "
            f"grounding_min={scored['losses']['grounding_min']:.4f} "
            f"cvf_min={scored['losses']['cvf_min_cvar']:.4f} "
            f"failures={len(scored.get('critical_failures', []))}",
            flush=True,
        )

    rows.sort(key=lambda r: (len(r.get("critical_failures", [])) == 0, -len(r.get("critical_failures", [])), r["score"]), reverse=True)
    payload = {
        "description": "FP16 Sana self-oracle checkpoint ranker: path, fixed correction, hidden-token CKA, prompt-contrastive flow energy, and counterfactual intervention grounding.",
        "warning": "Offline ranker only. Energy and grounding use common random numbers/context where possible and should be validated against held-out visual rankings before training use.",
        "heads": ["path", "fixed", "geometry", "teacher", "composition", "energy", "grounding", "cvf"],
        "counterfactuals": args.counterfactuals,
        "energy": {
            "enabled": bool(args.energy_timesteps > 0 and any(energy_specs)),
            "timesteps": args.energy_timesteps,
            "noise_per_timestep": args.energy_noise,
            "samples": len(energy_samples),
            "seed": args.energy_seed,
            "temp": args.energy_temp,
            "noising": args.energy_noising,
        },
        "seed_base": args.seed_base,
        "projection_seed": args.projection_seed,
        "grounding": {
            "enabled": bool(args.grounding_records > 0 and any(energy_specs)),
            "records": args.grounding_records,
            "threshold": args.grounding_threshold,
        },
        "cvf": {
            "enabled": bool(args.cvf_records > 0 and any(energy_specs)),
            "records": args.cvf_records,
            "threshold": args.cvf_threshold,
            "sep_min": args.cvf_sep_min,
            "cvar_frac": args.cvf_cvar_frac,
            "softmin_temp": args.cvf_softmin_temp,
        },
        "prompts": prompts,
        "rows": rows,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(rows, args.out_md)
    print(f"[done] {args.out_json} {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
