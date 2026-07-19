"""Per-phase profiler for the ternary KD step: says exactly where the time goes.

Reports (CUDA-event timed): teacher forward, teacher top-k head, student
forward, student head+loss, backward, Muon step, quantizer refresh — plus a
kernel-dispatch audit (is flash-linear-attention actually being used?).

  python scripts/profile_components.py --model Qwen/Qwen3.5-4B --seq-len 2048 --micro-batch 2
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clark_quantize.ternary_kd import (
    AbsmeanTernaryLinear,
    BonsaiMuonFast,
    collect_param_groups,
    hidden_forward,
    student_kd_ce_chunked,
    teacher_topk_chunked,
    ternarize_qwen35,
)
from train_qwen35_ternary_kd import load_models


def timed(fn, *, reps=3, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def dispatch_audit(model) -> None:
    print("== dispatch audit ==")
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        print("  fla.ops.gated_delta_rule: importable ✓")
    except Exception as e:
        print(f"  fla: MISSING ({e}) — DeltaNet layers are on the slow torch fallback!")
    try:
        import causal_conv1d  # noqa: F401
        print("  causal_conv1d: importable ✓")
    except Exception as e:
        print(f"  causal_conv1d: missing ({e}) — conv fallback (minor)")
    kinds = {}
    for name, m in model.named_modules():
        if name.endswith("linear_attn") or name.endswith("self_attn"):
            kinds.setdefault(type(m).__name__, 0)
            kinds[type(m).__name__] += 1
    print(f"  mixer module classes: {kinds}")
    n_q = sum(isinstance(m, AbsmeanTernaryLinear) for m in model.modules())
    print(f"  AbsmeanTernaryLinear modules: {n_q}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--kd-topk", type=int, default=64)
    ap.add_argument("--vocab-chunk", type=int, default=16384)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    student, teacher, text_cfg = load_models(args.model, args.device)
    ternarize_qwen35(student)
    for m in (student, teacher):
        m.config.use_cache = False
    student.gradient_checkpointing_enable()
    dispatch_audit(student)

    vocab = int(getattr(text_cfg, "vocab_size", 151936))
    ids = torch.randint(0, vocab, (args.micro_batch, args.seq_len), device=args.device)
    s_head = student.get_output_embeddings()
    t_head = teacher.get_output_embeddings()
    optimizer = BonsaiMuonFast(collect_param_groups(student, master_lr=1e-4))

    with torch.no_grad():
        t_fwd = timed(lambda: hidden_forward(teacher, ids))
        h_t, _ = hidden_forward(teacher, ids)
        t_topk = timed(lambda: teacher_topk_chunked(h_t, t_head, k=args.kd_topk, chunk=args.vocab_chunk))
        tv, ti = teacher_topk_chunked(h_t, t_head, k=args.kd_topk, chunk=args.vocab_chunk)

    def fwd_loss():
        h_s, _ = hidden_forward(student, ids)
        return student_kd_ce_chunked(h_s, s_head, tv, ti, ids, ce_weight=0.05, chunk=args.vocab_chunk)

    s_fwd_loss = timed(fwd_loss)

    def full_bwd():
        optimizer.zero_grad(set_to_none=True)
        fwd_loss().backward()

    fb = timed(full_bwd, reps=2)
    opt_ms = timed(lambda: optimizer.step(), reps=2)

    def quant_refresh():
        for m in student.modules():
            if isinstance(m, AbsmeanTernaryLinear):
                m._qcache = None
                m._cached_dequant(torch.bfloat16)

    q_ms = timed(quant_refresh, reps=2)

    bwd = fb - s_fwd_loss
    tok = args.micro_batch * args.seq_len
    total = t_fwd + t_topk + fb + (opt_ms + q_ms) / max(1, 1)  # per-micro view; opt+quant amortize over accum
    print("\n== per-micro-batch phase times (ms) ==")
    for k, v in [("teacher fwd", t_fwd), ("teacher topk head", t_topk),
                 ("student fwd+loss", s_fwd_loss), ("backward", bwd),
                 ("muon step (per opt step)", opt_ms), ("quantizer refresh (per opt step)", q_ms)]:
        print(f"  {k:34s} {v:9.1f}")
    per_micro = t_fwd + t_topk + fb
    print(f"\n  micro-batch tokens: {tok}; steady-state ≈ {tok / (per_micro/1000):.0f} tok/s "
          f"(+ {opt_ms + q_ms:.0f} ms per optimizer step amortized over grad-accum)")
    print(f"  peak VRAM: {torch.cuda.max_memory_allocated()/2**30:.1f} GB")


if __name__ == "__main__":
    main()
