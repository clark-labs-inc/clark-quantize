"""Throughput/VRAM ladder for the ternary KD trainer — the go/no-go gate.

Runs real training steps (teacher fwd + student fwd/bwd + Muon) on random
tokens across a ladder of (seq_len, micro_batch, grad_checkpoint) configs and
reports tok/s + peak VRAM for each. No data files needed.

Acceptance for the 5B-token phase-1 run: >= 2500 tok/s at some config.
(Roofline: ~30 GFLOPs/token total on a ~125 TFLOP/s effective A100 -> ~4.2k.)

  python scripts/throughput_ladder.py --model Qwen/Qwen3.5-4B
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clark_quantize.ternary_kd import (
    BonsaiMuonFast,
    anchor_layer_indices,
    anchored_hidden_loss,
    collect_param_groups,
    hidden_forward,
    student_kd_ce_chunked,
    teacher_topk_chunked,
    ternarize_qwen35,
)

LADDER = [
    # (seq_len, micro_batch, grad_checkpoint)
    (1024, 1, True),
    (2048, 1, True),
    (2048, 2, True),
    (2048, 4, True),
    (2048, 2, False),
    (4096, 1, True),
    (4096, 2, True),
    (4096, 1, False),
]


def one_config(student, teacher, optimizer, vocab, anchors, heads, *, seq, mb, ckpt,
               accum, device, warmup=2, timed=3, anchor_w=1.0):
    s_head, t_head = heads
    try:
        if ckpt:
            student.gradient_checkpointing_enable()
        else:
            student.gradient_checkpointing_disable()
    except (AttributeError, ValueError):
        pass
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def step():
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            ids = torch.randint(0, vocab, (mb, seq), device=device)
            need_h = anchor_w > 0
            with torch.no_grad():
                h_t, th = hidden_forward(teacher, ids, need_hidden=need_h)
                tv, ti = teacher_topk_chunked(h_t, t_head, k=64)
            h_s, sh = hidden_forward(student, ids, need_hidden=need_h)
            loss = student_kd_ce_chunked(h_s, s_head, tv, ti, ids, ce_weight=0.05)
            if need_h:
                loss = loss + anchor_w * anchored_hidden_loss(sh, th, anchors)
            (loss / accum).backward()
        optimizer.step()

    try:
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(timed):
            step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / timed
        toks = mb * seq * accum
        return dict(ok=True, tok_s=round(toks / dt), step_s=round(dt, 2),
                    peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
    except torch.cuda.OutOfMemoryError:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return dict(ok=False, tok_s=0, step_s=None, peak_gb=None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-anchors", action="store_true", help="ladder for the post-20%% regime")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_qwen35_ternary_kd import load_models

    student, teacher, text_cfg = load_models(args.model, args.device)
    surgery = ternarize_qwen35(student)
    print(f"ternarized {len(surgery['ternarized'])} trunk, tied={not surgery['head']}")
    for m in (student, teacher):
        if hasattr(m, "config"):
            m.config.use_cache = False
    anchors = anchor_layer_indices(text_cfg.layer_types)
    vocab = int(getattr(text_cfg, "vocab_size", 151936))
    heads = (student.get_output_embeddings(), teacher.get_output_embeddings())
    optimizer = BonsaiMuonFast(collect_param_groups(student, master_lr=1e-4))
    anchor_w = 0.0 if args.no_anchors else 1.0

    results = []
    print(f"{'seq':>5} {'mb':>3} {'ckpt':>5} {'accum':>5} | {'tok/s':>7} {'s/step':>7} {'peakGB':>7}")
    for seq, mb, ckpt in LADDER:
        r = one_config(student, teacher, optimizer, vocab, anchors, heads, seq=seq, mb=mb,
                       ckpt=ckpt, accum=args.grad_accum, device=args.device,
                       anchor_w=anchor_w)
        results.append(dict(seq=seq, mb=mb, ckpt=ckpt, **r))
        print(f"{seq:>5} {mb:>3} {str(ckpt):>5} {args.grad_accum:>5} | "
              + (f"{r['tok_s']:>7} {r['step_s']:>7} {r['peak_gb']:>7}" if r["ok"] else "    OOM"))
    best = max((r for r in results if r["ok"]), key=lambda r: r["tok_s"], default=None)
    print()
    if best:
        days_5b = 5e9 / max(best["tok_s"], 1) / 86400
        print(f"BEST: seq={best['seq']} mb={best['mb']} ckpt={best['ckpt']} -> "
              f"{best['tok_s']} tok/s, {best['peak_gb']} GB | 5B tokens ≈ {days_5b:.1f} days")
        print("GATE:", "PASS (>=2500 tok/s) — launch phase-1" if best["tok_s"] >= 2500
              else "FAIL — kernels still missing? check setup_a100_speed.sh output")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
