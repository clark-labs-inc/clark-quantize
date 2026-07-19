"""Precompute teacher top-K logits for the whole corpus (grid windows), so the
training run needs NO teacher on the GPU (-9 GB VRAM at 4B, +25-30% tok/s).

Windows follow the trainer's grid: window w of arm covers tokens
[PROBE_TOKENS + w*seq_len, +seq_len). Probe windows (the first PROBE_TOKENS)
are cached separately by the trainer itself at first probe.

Disk: n_tokens * K * 6 bytes  (fp16 value + int32 idx).
  2.4B-token corpus, K=16  -> ~230 GB;  K=8 -> ~115 GB.

  python scripts/cache_teacher_topk.py --model Qwen/Qwen3.5-4B \
      --data-root /workspace/kd_arms --out /workspace/kd_teacher_k16 \
      --k 16 --seq-len 2048 --batch 8

Then train with:  --teacher-cache /workspace/kd_teacher_k16 --kd-topk 16 --anchor-frac 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clark_quantize.ternary_kd import hidden_forward, teacher_topk_chunked
from train_qwen35_ternary_kd import ARMS, PROBE_TOKENS, load_models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    teacher, _, _ = load_models(args.model, args.device, want_teacher=False)
    teacher.eval().requires_grad_(False)
    teacher.config.use_cache = False
    head = teacher.get_output_embeddings()

    for arm in args.arms:
        tokens = np.memmap(args.data_root / f"{arm}.bin", dtype=np.uint32, mode="r")
        n_windows = (len(tokens) - PROBE_TOKENS) // args.seq_len
        tv = np.memmap(args.out / f"{arm}.tv.bin", dtype=np.float16, mode="w+",
                       shape=(n_windows, args.seq_len, args.k))
        ti = np.memmap(args.out / f"{arm}.ti.bin", dtype=np.int32, mode="w+",
                       shape=(n_windows, args.seq_len, args.k))
        t0 = time.time()
        with torch.no_grad():
            for w0 in range(0, n_windows, args.batch):
                ws = range(w0, min(w0 + args.batch, n_windows))
                rows = np.stack([
                    tokens[PROBE_TOKENS + w * args.seq_len : PROBE_TOKENS + (w + 1) * args.seq_len]
                    .astype(np.int64) for w in ws
                ])
                ids = torch.from_numpy(rows).to(args.device)
                h, _ = hidden_forward(teacher, ids)
                v, i = teacher_topk_chunked(h, head, k=args.k)
                tv[w0 : w0 + len(rows)] = v.to(torch.float16).cpu().numpy()
                ti[w0 : w0 + len(rows)] = i.to(torch.int32).cpu().numpy()
                if (w0 // args.batch) % 50 == 0:
                    done = (w0 + len(rows)) * args.seq_len
                    rate = done / max(time.time() - t0, 1e-6)
                    print(f"[{arm}] {w0+len(rows)}/{n_windows} windows ({rate:.0f} tok/s)", flush=True)
        tv.flush(); ti.flush()
        (args.out / f"{arm}.meta.json").write_text(
            json.dumps(dict(n_windows=n_windows, seq_len=args.seq_len, k=args.k)))
        print(f"[{arm}] DONE {n_windows} windows -> {args.out}")


if __name__ == "__main__":
    main()
