"""Full-coverage forensic diff of a Prism ternary 'unpacked' model vs its FP16 base.

Goes beyond compare_bonsai_ternary_to_qwen.py (sampled, MSE-PTQ only):
  - tests BOTH PTQ hypotheses on the base weights (MSE-optimal sort rule vs
    BitNet-style absmean+round rule) against the trained codes, full coverage
  - sign inheritance: do trained nonzero codes keep the base weight's sign?
  - support predictability: is the trained support exactly top-k(|W_base|)
    within each group (k taken from the trained group)?
  - movement: relative Frobenius delta + cosine per tensor (2D and 1D islands)
  - scale ratios: trained group scale vs absmean(base) and vs MSE-scale(base)
  - zero structure: per-row / per-column zero stats (structured sparsity, dead
    rows/cols), per-group k histogram
  - embed/lm_head: per-token row stats + pruned-tail row norms
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


def load_index(root: Path) -> dict[str, str]:
    return json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]


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
    if "norm" in name or "layernorm" in name:
        return "norm"
    return "other"


def layer(name: str) -> str:
    m = re.search(r"model\.layers\.(\d+)\.", name)
    return m.group(1) if m else "non_layer"


class Acc:
    """Scalar accumulators for one tensor."""

    def __init__(self) -> None:
        self.n = 0
        self.groups = 0
        self.recon_err_max = 0.0
        # code agreement
        self.agree_absmean = 0
        self.agree_mse = 0
        self.zero_trained = 0
        self.zero_absmean = 0
        self.zero_mse = 0
        # sign / support
        self.nz_trained = 0
        self.sign_match = 0
        self.topk_overlap = 0
        # support transitions vs absmean-PTQ (act/deact/signflip)
        self.ab_act = 0
        self.ab_deact = 0
        self.ab_signflip = 0
        # movement
        self.d2 = 0.0
        self.b2 = 0.0
        self.t2 = 0.0
        self.dot = 0.0
        # scale ratios (log-space mean/var)
        self.log_r_ab_sum = 0.0
        self.log_r_ab_sq = 0.0
        self.log_r_mse_sum = 0.0
        self.log_r_mse_sq = 0.0
        # k histogram (129 bins)
        self.k_hist = torch.zeros(129, dtype=torch.long)


def analyze_2d(
    b_full: torch.Tensor,
    t_full: torch.Tensor,
    *,
    row_chunk: int,
    want_row_stats: bool,
) -> dict[str, Any]:
    rows = min(b_full.shape[0], t_full.shape[0])
    cols = min(b_full.shape[1], t_full.shape[1])
    assert cols % 128 == 0, f"cols {cols} not multiple of 128"
    acc = Acc()
    col_zero = torch.zeros(cols, dtype=torch.long)
    row_zero_fracs = [] if want_row_stats else None
    karange = torch.arange(1, 129, dtype=torch.float32).unsqueeze(0)

    for r0 in range(0, rows, row_chunk):
        r1 = min(rows, r0 + row_chunk)
        wb = b_full[r0:r1, :cols].float()
        wt = t_full[r0:r1, :cols].float()
        r = r1 - r0
        b = wb.reshape(-1, 128)
        t = wt.reshape(-1, 128)
        g = b.shape[0]

        # trained codes + scales (exact by construction of 'unpacked')
        s_t = t.abs().amax(dim=1)
        c_t = torch.round(t / s_t.clamp_min(1e-12).unsqueeze(1)).clamp(-1, 1)
        recon = c_t * s_t.unsqueeze(1)
        acc.recon_err_max = max(acc.recon_err_max, float((recon - t).abs().max()))
        nz = c_t != 0
        k_t = nz.sum(dim=1)

        # absmean(BitNet)-PTQ on base
        s_ab = b.abs().mean(dim=1)
        c_ab = torch.round(b / s_ab.clamp_min(1e-12).unsqueeze(1)).clamp(-1, 1)

        # MSE-optimal PTQ on base (sort rule)
        sa, _ = torch.sort(b.abs(), dim=1, descending=True)
        csum = sa.cumsum(dim=1)
        score = csum.square() / karange
        kstar = score.argmax(dim=1)  # k*-1
        thresh = sa.gather(1, kstar.unsqueeze(1))
        c_mse = torch.sign(b) * (b.abs() >= thresh)
        s_mse = csum.gather(1, kstar.unsqueeze(1)).squeeze(1) / (kstar.float() + 1.0)

        acc.n += b.numel()
        acc.groups += g
        acc.agree_absmean += int((c_t == c_ab).sum())
        acc.agree_mse += int((c_t == c_mse).sum())
        acc.zero_trained += int((c_t == 0).sum())
        acc.zero_absmean += int((c_ab == 0).sum())
        acc.zero_mse += int((c_mse == 0).sum())

        acc.nz_trained += int(nz.sum())
        acc.sign_match += int(((torch.sign(b) == c_t) & nz).sum())

        # top-k(|base|) overlap at k = k_t
        kth = sa.gather(1, (k_t - 1).clamp_min(0).unsqueeze(1))
        topk_mask = b.abs() >= kth
        valid = (k_t > 0).unsqueeze(1)
        acc.topk_overlap += int((topk_mask & nz & valid).sum())

        # transitions vs absmean-PTQ
        ab_nz = c_ab != 0
        acc.ab_act += int(((~ab_nz) & nz).sum())
        acc.ab_deact += int((ab_nz & (~nz)).sum())
        acc.ab_signflip += int(((c_ab * c_t) == -1).sum())

        # movement
        d = wt - wb
        acc.d2 += float(d.square().sum())
        acc.b2 += float(wb.square().sum())
        acc.t2 += float(wt.square().sum())
        acc.dot += float((wb * wt).sum())

        # scale ratios (log space)
        lr_ab = (s_t.clamp_min(1e-12) / s_ab.clamp_min(1e-12)).log()
        lr_mse = (s_t.clamp_min(1e-12) / s_mse.clamp_min(1e-12)).log()
        acc.log_r_ab_sum += float(lr_ab.sum())
        acc.log_r_ab_sq += float(lr_ab.square().sum())
        acc.log_r_mse_sum += float(lr_mse.sum())
        acc.log_r_mse_sq += float(lr_mse.square().sum())

        acc.k_hist += torch.bincount(k_t, minlength=129)

        # zero structure
        zeros_2d = (c_t == 0).reshape(r, cols)
        col_zero += zeros_2d.sum(dim=0)
        if row_zero_fracs is not None:
            row_zero_fracs.append(zeros_2d.float().mean(dim=1))

    n = max(acc.n, 1)
    g = max(acc.groups, 1)
    nz = max(acc.nz_trained, 1)
    out: dict[str, Any] = {
        "rows": rows,
        "cols": cols,
        "n": acc.n,
        "recon_err_max": acc.recon_err_max,
        "zero_trained": acc.zero_trained / n,
        "zero_ptq_absmean": acc.zero_absmean / n,
        "zero_ptq_mse": acc.zero_mse / n,
        "agree_absmean": acc.agree_absmean / n,
        "agree_mse": acc.agree_mse / n,
        "sign_inherit": acc.sign_match / nz,
        "topk_overlap": acc.topk_overlap / nz,
        "ab_activation": acc.ab_act / n,
        "ab_deactivation": acc.ab_deact / n,
        "ab_signflip": acc.ab_signflip / n,
        "rel_movement": (acc.d2 / max(acc.b2, 1e-12)) ** 0.5,
        "cosine": acc.dot / max((acc.b2 ** 0.5) * (acc.t2 ** 0.5), 1e-12),
        "std_ratio": (acc.t2 / max(acc.b2, 1e-12)) ** 0.5,
        "scale_over_absmean_logmean": acc.log_r_ab_sum / g,
        "scale_over_absmean_logstd": max(acc.log_r_ab_sq / g - (acc.log_r_ab_sum / g) ** 2, 0.0) ** 0.5,
        "scale_over_mse_logmean": acc.log_r_mse_sum / g,
        "scale_over_mse_logstd": max(acc.log_r_mse_sq / g - (acc.log_r_mse_sum / g) ** 2, 0.0) ** 0.5,
        "k_hist": acc.k_hist.tolist(),
    }
    # column structure
    col_frac = col_zero.float() / rows
    out["col_zero_p50"] = float(col_frac.median())
    out["col_zero_p99"] = float(col_frac.quantile(0.99))
    out["col_zero_max"] = float(col_frac.max())
    out["cols_over_90pct_zero"] = int((col_frac > 0.9).sum())
    if row_zero_fracs is not None:
        rf = torch.cat(row_zero_fracs)
        out["row_zero_p50"] = float(rf.median())
        out["row_zero_p99"] = float(rf.quantile(0.99))
        out["row_zero_max"] = float(rf.max())
        out["rows_over_90pct_zero"] = int((rf > 0.9).sum())
        top = torch.topk(rf, k=min(20, rf.numel()))
        out["top_zero_rows"] = [[int(i), round(float(v), 4)] for v, i in zip(top.values, top.indices)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--trained", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--row-chunk", type=int, default=4096)
    args = ap.parse_args()

    bi = load_index(args.base)
    ti = load_index(args.trained)
    names = sorted(set(bi) & set(ti))
    only_base = sorted(set(bi) - set(ti))
    only_trained = sorted(set(ti) - set(bi))

    rows_out: list[dict[str, Any]] = []
    t_start = time.time()
    for i, name in enumerate(names):
        with safe_open(args.base / bi[name], framework="pt", device="cpu") as f:
            wb = f.get_tensor(name)
        with safe_open(args.trained / ti[name], framework="pt", device="cpu") as f:
            wt = f.get_tensor(name)
        rec: dict[str, Any] = {"name": name, "kind": kind(name), "layer": layer(name)}
        if wb.ndim == 2 and wb.shape[1] % 128 == 0 and wt.ndim == 2:
            want_rows = kind(name) in ("embed", "lm_head")
            rec.update(analyze_2d(wb, wt, row_chunk=args.row_chunk, want_row_stats=want_rows))
            rec["is_2d"] = True
            # pruned vocab tail on base side
            if wb.shape[0] != wt.shape[0]:
                tail = wb[min(wb.shape[0], wt.shape[0]):].float()
                kept = wb[: min(wb.shape[0], wt.shape[0])].float()
                rec["shape_mismatch"] = [list(wb.shape), list(wt.shape)]
                if tail.numel():
                    rec["pruned_tail_rows"] = int(tail.shape[0])
                    rec["pruned_tail_rownorm_mean"] = float(tail.norm(dim=1).mean())
                    rec["kept_rownorm_mean"] = float(kept.norm(dim=1).mean())
        else:
            b = wb.float().reshape(-1)
            t = wt.float().reshape(-1)
            m = min(b.numel(), t.numel())
            b, t = b[:m], t[:m]
            d = t - b
            rec.update(
                is_2d=False,
                n=m,
                rel_movement=float(d.norm() / b.norm().clamp_min(1e-12)),
                cosine=float((b * t).sum() / (b.norm() * t.norm()).clamp_min(1e-12)),
                mean_abs_base=float(b.abs().mean()),
                mean_abs_trained=float(t.abs().mean()),
            )
        rows_out.append(rec)
        if rec.get("is_2d"):
            print(
                json.dumps(
                    {
                        k: rec[k]
                        for k in [
                            "name", "zero_trained", "zero_ptq_absmean", "agree_absmean",
                            "agree_mse", "sign_inherit", "topk_overlap", "rel_movement",
                        ]
                    }
                ),
                flush=True,
            )
        else:
            print(json.dumps({k: rec[k] for k in ["name", "rel_movement", "cosine"]}), flush=True)

    # aggregates over 2D
    two_d = [r for r in rows_out if r.get("is_2d")]
    def wavg(rs: list[dict[str, Any]], key: str, wkey: str = "n") -> float:
        num = sum(r[key] * r[wkey] for r in rs if key in r)
        den = sum(r[wkey] for r in rs if key in r)
        return num / den if den else float("nan")

    keys = [
        "zero_trained", "zero_ptq_absmean", "zero_ptq_mse", "agree_absmean", "agree_mse",
        "sign_inherit", "topk_overlap", "ab_activation", "ab_deactivation", "ab_signflip",
        "rel_movement", "cosine", "std_ratio",
    ]
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in two_d:
        by_kind[r["kind"]].append(r)
        by_layer[r["layer"]].append(r)
    report = {
        "elapsed_sec": time.time() - t_start,
        "only_base_tensors": only_base,
        "only_trained_tensors": only_trained,
        "summary": {k: wavg(two_d, k) for k in keys},
        "by_kind": {k: {kk: wavg(v, kk) for kk in keys} for k, v in sorted(by_kind.items())},
        "by_layer": {
            k: {kk: wavg(v, kk) for kk in keys}
            for k, v in sorted(by_layer.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else -1)
        },
        "norm_islands": [
            {k: r[k] for k in ["name", "rel_movement", "cosine", "mean_abs_base", "mean_abs_trained"]}
            for r in rows_out
            if not r.get("is_2d")
        ],
        "rows": rows_out,
    }
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"elapsed: {report['elapsed_sec']:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
