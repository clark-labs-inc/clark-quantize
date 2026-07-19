"""Reference trainer: signature-steered ternary KD of Qwen3.5 (4B/9B).

Speed architecture (see A100_RUNBOOK.md):
  - DeltaNet layers dispatch to flash-linear-attention Triton kernels
    (scripts/setup_a100_speed.sh installs + verifies).
  - Ternary masters quantize through a fused Triton kernel, cached per
    optimizer step (version-counter invalidation).
  - Head losses are vocab-chunked: the [seq, 248320] logits are never
    materialized (auto-verified against the reference path at startup).
  - Muon runs shape-batched Newton-Schulz in bf16 (BonsaiMuonFast).
  - Probes cache the frozen teacher top-K; with --teacher-cache the teacher
    is removed from the GPU entirely.

Data layout (per arm): pre-tokenized uint32 memmap at {data_root}/{arm}.bin;
first PROBE_TOKENS tokens are the frozen probe slice.

Usage sketch (single A100, 4B):
  python scripts/train_qwen35_ternary_kd.py \
      --model Qwen/Qwen3.5-4B --data-root /workspace/kd_arms \
      --out /workspace/ckpt/qwen35-4b-ternary --budget-tokens 5e9 \
      --seq-len 2048 --micro-batch 2 --grad-accum 32
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clark_quantize.bonsai_muon import BonsaiMuon
from clark_quantize.ternary_kd import (
    BonsaiMuonFast,
    KLBanditMixer,
    MovementServo,
    SignatureMeter,
    anchor_layer_indices,
    anchored_hidden_loss,
    chunked_ce_loss,
    collect_param_groups,
    export_unpacked,
    hidden_forward,
    should_stop,
    student_kd_ce_chunked,
    teacher_topk,
    teacher_topk_chunked,
    ternarize_qwen35,
    topk_kd_from_cache,
    topk_kd_loss,
)

ARMS = ["knowledge", "math", "code", "instruct", "tools", "multilingual"]
PROBE_TOKENS = 65_536  # per-arm frozen probe prefix


class ArmData:
    def __init__(self, root: Path, arm: str, seq_len: int, seed: int) -> None:
        self.tokens = np.memmap(root / f"{arm}.bin", dtype=np.uint32, mode="r")
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        if len(self.tokens) < PROBE_TOKENS + seq_len + 1:
            raise ValueError(f"arm {arm} too small")
        self.n_windows = (len(self.tokens) - PROBE_TOKENS) // seq_len  # teacher-cache grid

    def _rows(self, starts, device) -> torch.Tensor:
        rows = np.stack([self.tokens[s : s + self.seq_len].astype(np.int64) for s in starts])
        t = torch.from_numpy(rows)
        if device != "cpu":
            t = t.pin_memory()
        return t.to(device, non_blocking=True)

    def train_batch(self, micro_batch: int, device) -> torch.Tensor:
        starts = self.rng.integers(PROBE_TOKENS, len(self.tokens) - self.seq_len - 1, size=micro_batch)
        return self._rows(starts, device)

    def grid_batch(self, micro_batch: int, device):
        """Windows on the fixed grid (teacher-cache mode). Returns (ids, window_idx)."""
        w = self.rng.integers(0, self.n_windows, size=micro_batch)
        starts = PROBE_TOKENS + w * self.seq_len
        return self._rows(starts, device), w

    def probe_batch(self, device, n_windows: int = 4) -> torch.Tensor:
        rows = [
            self.tokens[i * self.seq_len : (i + 1) * self.seq_len].astype(np.int64)
            for i in range(n_windows)
        ]
        return torch.from_numpy(np.stack(rows)).to(device)


class TeacherCache:
    """Memmapped per-window teacher top-K built by scripts/cache_teacher_topk.py."""

    def __init__(self, root: Path, arm: str) -> None:
        meta = json.loads((root / f"{arm}.meta.json").read_text())
        self.k, self.seq = meta["k"], meta["seq_len"]
        n = meta["n_windows"]
        self.tv = np.memmap(root / f"{arm}.tv.bin", dtype=np.float16, mode="r", shape=(n, self.seq, self.k))
        self.ti = np.memmap(root / f"{arm}.ti.bin", dtype=np.int32, mode="r", shape=(n, self.seq, self.k))

    def get(self, windows, device):
        tv = torch.from_numpy(np.ascontiguousarray(self.tv[windows])).to(device).float()
        ti = torch.from_numpy(np.ascontiguousarray(self.ti[windows]).astype(np.int64)).to(device)
        return tv, ti


def load_models(name: str, device: str, dtype=torch.bfloat16, attn_impl: str | None = "sdpa",
                want_teacher: bool = True):
    from transformers import AutoConfig, AutoModelForCausalLM

    kwargs = dict(torch_dtype=dtype, trust_remote_code=True)
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    def load():
        try:
            return AutoModelForCausalLM.from_pretrained(name, **kwargs)
        except ValueError:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText.from_pretrained(name, **kwargs)

    student = load().to(device)
    teacher = None
    if want_teacher:
        teacher = load().to(device)
        teacher.eval().requires_grad_(False)
    cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
    return student, teacher, getattr(cfg, "text_config", cfg)


def verify_chunked_path(student, teacher, s_head, t_head, vocab, device, k) -> bool:
    """Compare chunked-head KD against the reference logits path on one batch."""
    ids = torch.randint(0, vocab, (1, 256), device=device)
    with torch.no_grad():
        ref = topk_kd_loss(student(input_ids=ids).logits, teacher(input_ids=ids).logits, k=k)
        h_t, _ = hidden_forward(teacher, ids)
        tv, ti, tlse = teacher_topk_chunked(
            h_t, t_head, k=k, chunk=8192, return_lse=True
        )
        h_s, _ = hidden_forward(student, ids)
        alt = student_kd_ce_chunked(
            h_s, s_head, tv, ti, ids, ce_weight=0.0, chunk=8192, teacher_lse=tlse
        )
    rel = abs(float(ref - alt)) / max(abs(float(ref)), 1e-6)
    print(f"chunked-head verification: ref={float(ref):.5f} chunked={float(alt):.5f} rel={rel:.2e}")
    return rel < 1e-2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget-tokens", type=float, default=30e9)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=64)
    ap.add_argument("--master-lr", type=float, default=1e-4)
    ap.add_argument("--train-norms", action="store_true",
                    help="train continuous text RMSNorm weights as the compensation channel")
    ap.add_argument("--norm-lr", type=float, default=1e-4)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--warmup-frac", type=float, default=0.01)
    ap.add_argument("--anchor-frac", type=float, default=0.2)
    ap.add_argument("--ce-weight", type=float, default=0.05)
    ap.add_argument("--kd-topk", type=int, default=128)
    ap.add_argument("--kd-temperature", type=float, default=2.0,
                    help="soften teacher distribution; >1 penalizes the tail, harder to mimic")
    ap.add_argument("--hidden-weight", type=float, default=2.0,
                    help="weight on all-layer hidden-state distillation (the reorganization driver)")
    ap.add_argument("--hidden-layers", choices=["full", "half", "all"], default="all",
                    help="which layer residual streams to distill: full-attn(8) | every-other(16) | all(32)")
    ap.add_argument("--hidden-anneal-floor", type=float, default=1.0,
                    help="1.0 = never anneal the hidden loss (default); <1 anneals toward this floor over anchor-frac")
    ap.add_argument("--vocab-chunk", type=int, default=16384)
    ap.add_argument("--probe-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    ap.add_argument("--no-chunked-head", action="store_true")
    ap.add_argument("--optimizer", choices=["muon-fast", "muon"], default="muon-fast")
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--muon-fp32", action="store_true")
    ap.add_argument("--teacher-cache", type=Path, default=None,
                    help="dir from cache_teacher_topk.py; removes teacher from GPU (forces anchors off)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    use_cache_mode = args.teacher_cache is not None
    if use_cache_mode:
        args.anchor_frac = 0.0

    student, teacher, text_cfg = load_models(
        args.model, args.device,
        attn_impl=None if args.attn_impl == "none" else args.attn_impl,
        want_teacher=not use_cache_mode,
    )
    surgery = ternarize_qwen35(student, train_norms=args.train_norms)
    tied = not surgery["head"]
    print(f"ternarized {len(surgery['ternarized'])} trunk + {len(surgery['head'])} head "
          f"(tied embeddings: {tied}); frozen {len(surgery['frozen'])} tensors")
    print(f"trainable continuous norms: {len(surgery['trainable_norms'])}")
    for m in (student, teacher):
        if m is not None and hasattr(m, "config"):
            m.config.use_cache = False
    if not args.no_grad_checkpoint:
        try:
            student.gradient_checkpointing_enable()
        except (AttributeError, ValueError) as e:
            print(f"gradient checkpointing unavailable: {e}")
    n_layers = len(text_cfg.layer_types)
    if args.hidden_layers == "all":
        anchors = list(range(n_layers))
    elif args.hidden_layers == "half":
        anchors = list(range(1, n_layers, 2))
    else:
        anchors = anchor_layer_indices(text_cfg.layer_types)
    print(f"hidden distillation: {len(anchors)} layers, weight {args.hidden_weight}, "
          f"anneal-floor {args.hidden_anneal_floor}; KD top-{args.kd_topk} T={args.kd_temperature}")
    vocab = int(getattr(text_cfg, "vocab_size", 151936))
    s_head = student.get_output_embeddings()
    t_head = teacher.get_output_embeddings() if teacher is not None else None

    chunked = not args.no_chunked_head
    if chunked and teacher is not None:
        if not verify_chunked_path(student, teacher, s_head, t_head, vocab, args.device, args.kd_topk):
            print("chunked-head verification FAILED — falling back to reference path")
            chunked = False
    if args.compile:
        try:
            student = torch.compile(student, dynamic=False)
        except Exception as e:
            print(f"torch.compile failed, continuing eager: {e}")

    meter = SignatureMeter(student, base_sample_frac=0.25)
    servo = MovementServo()
    bandit = KLBanditMixer(ARMS)
    data = {arm: ArmData(args.data_root, arm, args.seq_len, seed=i) for i, arm in enumerate(ARMS)}
    tcache = {arm: TeacherCache(args.teacher_cache, arm) for arm in ARMS} if use_cache_mode else None
    probe_teacher_cache: dict[str, tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = {}

    mdtype = torch.float32 if args.muon_fp32 else torch.bfloat16
    groups = collect_param_groups(
        student,
        master_lr=args.master_lr,
        norm_lr=args.norm_lr if args.train_norms else None,
        state_dtype=mdtype,
        ns_dtype=mdtype,
    )
    optimizer = (BonsaiMuonFast if args.optimizer == "muon-fast" else BonsaiMuon)(groups)
    base_lrs = [g["lr"] for g in groups]
    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        student.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        del ckpt
        gc.collect()
        if args.device != "cpu":
            torch.cuda.empty_cache()
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr
        print(f"resumed {args.resume} at step {start_step}; lr reset to CLI base values")

    tokens_per_step = args.micro_batch * args.grad_accum * args.seq_len
    total_steps = int(args.budget_tokens / tokens_per_step)
    kl_history: list[tuple[float, float]] = []
    gates = {int(g / tokens_per_step) for g in (2e9, 8e9, 20e9)}
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train_log.jsonl"
    t_last, tok_last = time.time(), 0

    def micro_step(arm: str, hidden_w: float) -> None:
        need_h = hidden_w > 0
        T = args.kd_temperature
        if use_cache_mode:
            ids, w = data[arm].grid_batch(args.micro_batch, args.device)
            tv, ti = tcache[arm].get(w, args.device)
            h_s, _ = hidden_forward(student, ids)
            loss = student_kd_ce_chunked(h_s, s_head, tv, ti, ids,
                                         ce_weight=args.ce_weight, chunk=args.vocab_chunk, temperature=T)
        elif chunked:
            ids = data[arm].train_batch(args.micro_batch, args.device)
            with torch.no_grad():
                h_t, th = hidden_forward(teacher, ids, need_hidden=need_h)
                tv, ti, tlse = teacher_topk_chunked(
                    h_t, t_head, k=args.kd_topk, chunk=args.vocab_chunk,
                    temperature=T, return_lse=True,
                )
            h_s, sh = hidden_forward(student, ids, need_hidden=need_h)
            loss = student_kd_ce_chunked(h_s, s_head, tv, ti, ids,
                                         ce_weight=args.ce_weight, chunk=args.vocab_chunk,
                                         temperature=T, teacher_lse=tlse)
            if need_h:
                loss = loss + hidden_w * anchored_hidden_loss(sh, th, anchors)
        else:
            ids = data[arm].train_batch(args.micro_batch, args.device)
            with torch.no_grad():
                t_out = teacher(input_ids=ids, output_hidden_states=need_h)
            s_out = student(input_ids=ids, output_hidden_states=need_h)
            loss = topk_kd_loss(s_out.logits, t_out.logits, k=args.kd_topk, temperature=T)
            if args.ce_weight > 0:
                loss = loss + args.ce_weight * chunked_ce_loss(s_out.logits, ids)
            if need_h:
                loss = loss + hidden_w * anchored_hidden_loss(
                    s_out.hidden_states, t_out.hidden_states, anchors)
        (loss / args.grad_accum).backward()

    def probe_kl(arm: str) -> tuple[float, dict[str, float]]:
        ids = data[arm].probe_batch(args.device)
        if arm not in probe_teacher_cache:
            if use_cache_mode:
                n = ids.shape[0]
                tv, ti = tcache[arm].get(np.arange(n), args.device)  # grid window i == probe window i
                tlse = None
            else:
                with torch.no_grad():
                    if chunked:
                        h_t, _ = hidden_forward(teacher, ids)
                        tv, ti, tlse = teacher_topk_chunked(
                            h_t, t_head, k=args.kd_topk, chunk=args.vocab_chunk,
                            temperature=1.0, return_lse=True,
                        )
                    else:
                        teacher_logits = teacher(input_ids=ids).logits
                        tv, ti = teacher_topk(teacher_logits, k=args.kd_topk)
                        tlse = torch.logsumexp(teacher_logits.float(), dim=-1)
            probe_teacher_cache[arm] = (tv, ti, tlse)
        tv, ti, tlse = probe_teacher_cache[arm]
        with torch.no_grad():
            if chunked or use_cache_mode:
                h_s, _ = hidden_forward(student, ids)
                loss, metrics = student_kd_ce_chunked(
                    h_s, s_head, tv, ti, ids, ce_weight=0.0,
                    chunk=args.vocab_chunk, temperature=1.0, teacher_lse=tlse,
                    return_metrics=True,
                )
                return float(loss), {k: float(v) for k, v in metrics.items()}
            logits = student(input_ids=ids).logits
            loss = topk_kd_from_cache(logits, tv, ti, teacher_lse=tlse)
            top1 = (logits.argmax(dim=-1) == ti[..., 0]).float().mean()
            student_mass = torch.gather(logits.softmax(dim=-1), -1, ti).sum(dim=-1).mean()
            teacher_mass = (tv - tlse.unsqueeze(-1)).exp().sum(dim=-1).mean()
            return float(loss), {
                "top1_agreement": float(top1),
                "teacher_topk_mass": float(teacher_mass),
                "student_topk_mass": float(student_mass),
            }

    all_params = [p for g in optimizer.param_groups for p in g["params"]]
    last_grad_norm = 0.0

    for step in range(start_step + 1, total_steps + 1):
        u = step / total_steps
        warm = min(step / max(int(total_steps * args.warmup_frac), 1), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * u)) * 0.9 + 0.1
        if args.hidden_anneal_floor >= 1.0:
            hidden_w = args.hidden_weight  # never anneal — hold the reorganization signal on
        else:
            frac = max(args.hidden_anneal_floor, 1.0 - u / args.anchor_frac) if args.anchor_frac > 0 else 1.0
            hidden_w = args.hidden_weight * frac

        optimizer.zero_grad(set_to_none=True)
        arm = bandit.sample()
        for _ in range(args.grad_accum):
            micro_step(arm, hidden_w)
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
        last_grad_norm = float(grad_norm.detach().cpu())
        for g, base in zip(optimizer.param_groups, base_lrs):
            servo_mult = servo.lr_mult if g.get("role", "muon") == "muon" else 1.0
            g["lr"] = base * warm * cosine * servo_mult
        optimizer.step()

        if step % args.probe_every == 0 or step in gates:
            probe_results = {a: probe_kl(a) for a in ARMS}
            probe_kls = {a: result[0] for a, result in probe_results.items()}
            probe_metrics = {a: result[1] for a, result in probe_results.items()}
            for a, v in probe_kls.items():
                bandit.update(a, v)
            mean_kl = sum(probe_kls.values()) / len(probe_kls)
            kl_history.append((u, mean_kl))
            sig = meter.measure()
            servo.update(sig, u)
            now = time.time()
            tok_s = (step * tokens_per_step - tok_last) / max(now - t_last, 1e-6)
            t_last, tok_last = now, step * tokens_per_step
            rec = dict(step=step, u=round(u, 4), tokens=step * tokens_per_step,
                       tok_per_s=round(tok_s), probe_kl=probe_kls, mean_kl=mean_kl,
                       probe_metrics=probe_metrics,
                       mean_top1=sum(m["top1_agreement"] for m in probe_metrics.values()) / len(probe_metrics),
                       mean_student_topk_mass=sum(m["student_topk_mass"] for m in probe_metrics.values()) / len(probe_metrics),
                       lr_mult=servo.lr_mult,
                       servo_signal=servo.signal,
                       servo_ref=servo.last_ref,
                       servo_err=servo.last_err,
                       base_lr=args.master_lr,
                       effective_lr=max(g["lr"] for g in optimizer.param_groups),
                       norm_lr=next((g["lr"] for g in optimizer.param_groups
                                    if g.get("tag") == "norms"), None),
                       hidden_w=round(hidden_w, 3),
                       grad_norm=last_grad_norm,
                       grad_clip=args.grad_clip,
                       mix=bandit.probs(),
                       peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1)
                       if args.device != "cpu" else None,
                       **{k: sig[k] for k in
                          ("movement", "churn", "zeros", "sign_inherit", "distance", "in_band")})
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            stop, reason = should_stop(kl_history, sig, budget_frac=u)
            if step in gates:
                print(f"[gate {step*tokens_per_step/1e9:.0f}B] {reason}; "
                      f"signature distance {sig['distance']:.2f}")
            if stop:
                print(f"stopping: {reason}")
                break

        if step % args.ckpt_every == 0:
            torch.save({"step": step, "model": student.state_dict(),
                        "optimizer": optimizer.state_dict()}, args.out / f"ckpt_{step}.pt")

    export_unpacked(student, args.out / "unpacked")
    print(f"exported Prism-convention unpacked checkpoint to {args.out / 'unpacked'}")


if __name__ == "__main__":
    main()
