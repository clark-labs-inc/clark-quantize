"""Ground-truth check: our absmean quantizer must reproduce Prism's shipped codes.

Compares absmean-PTQ of the Qwen3-8B embedding against the codes actually in
Ternary-Bonsai-8B-unpacked (proven frozen-PTQ in bonsai_ternary_deep_forensics.md).
Expected agreement ≥ 0.9999 (residual = bf16 tie-rounding). Requires the local
artifacts under .analysis/bonsai_ternary (present on the Mac, not the A100).

  python scripts/verify_absmean_vs_prism.py [--rows 32768]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clark_quantize.ternary_kd import absmean_quantize

NAME = "model.embed_tokens.weight"


def load_rows(root: Path, rows: int) -> torch.Tensor:
    idx = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(root / idx[NAME], framework="pt", device="cpu") as f:
        return f.get_tensor(NAME)[:rows].float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path(".analysis/bonsai_ternary/Qwen3-8B"))
    ap.add_argument("--trained", type=Path,
                    default=Path(".analysis/bonsai_ternary/Ternary-Bonsai-8B-unpacked"))
    ap.add_argument("--rows", type=int, default=32768)
    args = ap.parse_args()

    base = load_rows(args.base, args.rows)
    trained = load_rows(args.trained, args.rows)
    ours, our_scales = absmean_quantize(base)

    tg = trained.reshape(trained.shape[0], -1, 128)
    s = tg.abs().amax(dim=-1).clamp_min(1e-12)
    prism = torch.round(tg / s.unsqueeze(-1)).clamp_(-1, 1).to(torch.int8)

    agree = float((ours == prism).float().mean())
    scale_ratio = (our_scales / s.clamp_min(1e-12)).log()
    print(f"rows={args.rows}  code agreement={agree:.6f}  "
          f"scale log-ratio mean={float(scale_ratio.mean()):+.2e} std={float(scale_ratio.std()):.2e}")
    assert agree >= 0.9999, "quantizer does NOT match Prism's shipped codes"
    print("OK: absmean quantizer reproduces Prism's shipped embedding codes")


if __name__ == "__main__":
    main()
