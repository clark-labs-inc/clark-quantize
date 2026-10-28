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
import importlib.metadata
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clark_quantize.bonsai_muon import BonsaiMuon
from clark_quantize.kd_data_manifest import file_sha256, verify_kd_arms
from clark_quantize.qwen_global_qat import (
    canonical_json_sha256,
    capture_replay_state,
    join_model_state,
    load_global_qat_checkpoint,
    restore_process_rng,
    save_global_qat_checkpoint,
)
from clark_quantize.qwen_qat import state_tree_equal
from clark_quantize.qwen_quality_gate import (
    atomic_write_json,
    make_gate_request,
    validate_continuation_authorization,
)
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
from scripts.spot.prune_qat_checkpoints import prune as prune_qat_checkpoints

ARMS = ["knowledge", "math", "code", "instruct", "tools", "multilingual"]
PROBE_TOKENS = 65_536  # per-arm frozen probe prefix


def run_identity(
    args: argparse.Namespace,
    *,
    data_manifest_sha256: str,
    resolved_chunked_head: bool,
    resolved_compile: bool,
) -> dict:
    teacher_cache = None
    if args.teacher_cache is not None:
        cache_root = args.teacher_cache.resolve()
        teacher_cache = {
            "root": str(cache_root),
            "metadata": {
                arm: {
                    "sha256": file_sha256(cache_root / f"{arm}.meta.json"),
                    "tv_bytes": (cache_root / f"{arm}.tv.bin").stat().st_size,
                    "ti_bytes": (cache_root / f"{arm}.ti.bin").stat().st_size,
                }
                for arm in ARMS
            },
        }
    return {
        "schema": "clark_qwen35_global_qat_run_identity_v1",
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "data_manifest_sha256": data_manifest_sha256,
        "teacher_cache": teacher_cache,
        "budget_tokens": args.budget_tokens,
        "seq_len": args.seq_len,
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "master_lr": args.master_lr,
        "train_norms": args.train_norms,
        "norm_lr": args.norm_lr,
        "warmup_frac": args.warmup_frac,
        "anchor_frac": args.anchor_frac,
        "ce_weight": args.ce_weight,
        "kd_topk": args.kd_topk,
        "kd_temperature": args.kd_temperature,
        "hidden_weight": args.hidden_weight,
        "hidden_layers": args.hidden_layers,
        "hidden_anneal_floor": args.hidden_anneal_floor,
        "vocab_chunk": args.vocab_chunk,
        "probe_every": args.probe_every,
        "grad_clip": args.grad_clip,
        "gradient_checkpointing": not args.no_grad_checkpoint,
        "chunked_head": not args.no_chunked_head,
        "resolved_chunked_head": resolved_chunked_head,
        "optimizer": args.optimizer,
        "attn_impl": args.attn_impl,
        "compile": args.compile,
        "resolved_compile": resolved_compile,
        "muon_fp32": args.muon_fp32,
        "allow_unsealed_teacher_cache": args.allow_unsealed_teacher_cache,
        "trainer_source_sha256": file_sha256(Path(__file__)),
        "checkpoint_source_sha256": file_sha256(
            Path(__file__).resolve().parents[1] / "clark_quantize/qwen_global_qat.py"
        ),
        "data_manifest_source_sha256": file_sha256(
            Path(__file__).resolve().parents[1] / "clark_quantize/kd_data_manifest.py"
        ),
        "quality_gate_source_sha256": file_sha256(
            Path(__file__).resolve().parents[1] / "clark_quantize/qwen_quality_gate.py"
        ),
        "ternary_kd_source_sha256": file_sha256(
            Path(__file__).resolve().parents[1] / "clark_quantize/ternary_kd.py"
        ),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_devices": [
            {
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
        "transformers_version": importlib.metadata.version("transformers"),
        "python_version": sys.version,
    }


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
                want_teacher: bool = True, revision: str | None = None):
    from transformers import AutoConfig, AutoModelForCausalLM

    kwargs = dict(torch_dtype=dtype, trust_remote_code=True, revision=revision)
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
    cfg = AutoConfig.from_pretrained(name, trust_remote_code=True, revision=revision)
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
    ap.add_argument("--revision", required=True,
                    help="immutable Hugging Face commit or equivalent local model identity")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--data-manifest", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget-tokens", type=float, default=30e9)
    ap.add_argument("--stop-after-tokens", type=float, default=None,
                    help="operational stage boundary; does not change the full-budget schedule")
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=64)
    ap.add_argument("--master-lr", type=float, default=1e-4)
    ap.add_argument("--train-norms", action="store_true",
                    help="train continuous text RMSNorm weights as the compensation channel")
    ap.add_argument("--norm-lr", type=float, default=1e-4)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--capability-gate-receipt", type=Path, default=None,
                    help="checkpoint-bound authorization required after a planned stage boundary")
    ap.add_argument("--allow-legacy-resume", action="store_true",
                    help="allow old non-exact ckpt_*.pt resume; forbidden for promoted runs")
    ap.add_argument("--checkpoint-root", type=Path, default=None)
    ap.add_argument("--checkpoint-keep", type=int, choices=[1, 2], default=2)
    ap.add_argument("--checkpoint-reserve-gb", type=float, default=10.0)
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
    ap.add_argument("--allow-unsealed-teacher-cache", action="store_true",
                    help="explicitly permit legacy unhashed teacher caches; run is non-promotable")
    ap.add_argument("--min-free-gb", type=float, default=65.0,
                    help="minimum free accelerator memory before model allocation")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not args.out.is_absolute():
        raise SystemExit("--out must be absolute for restart-safe QAT")
    checkpoint_root = args.checkpoint_root or (args.out / "checkpoints")
    if not checkpoint_root.is_absolute():
        raise SystemExit("--checkpoint-root must be absolute")
    if args.resume is not None and not args.resume.is_absolute():
        raise SystemExit("--resume must be absolute")
    if args.checkpoint_reserve_gb < 0:
        raise SystemExit("--checkpoint-reserve-gb must be nonnegative")
    if args.budget_tokens <= 0:
        raise SystemExit("--budget-tokens must be positive")
    if args.stop_after_tokens is not None and args.stop_after_tokens <= 0:
        raise SystemExit("--stop-after-tokens must be positive")
    if args.min_free_gb < 0:
        raise SystemExit("--min-free-gb must be nonnegative")
    if args.capability_gate_receipt is not None and args.resume is None:
        raise SystemExit("--capability-gate-receipt only applies to --resume")

    data_root = args.data_root.resolve()
    data_manifest_path = (
        args.data_manifest.resolve() if args.data_manifest else data_root / "manifest.json"
    )
    data_manifest = verify_kd_arms(
        data_manifest_path,
        root=data_root,
        arms=ARMS,
        model=args.model,
        revision=args.revision,
        probe_tokens=PROBE_TOKENS,
        rehash=True,
    )
    data_manifest_sha256 = canonical_json_sha256(data_manifest)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + 3)

    use_cache_mode = args.teacher_cache is not None
    if use_cache_mode:
        if not args.allow_unsealed_teacher_cache:
            raise SystemExit(
                "teacher-cache files are not atomically sealed or content-hashed; "
                "pass --allow-unsealed-teacher-cache only for a non-promotable run"
            )
        print(
            "WARNING: unsealed teacher-cache mode is explicitly non-promotable",
            flush=True,
        )
        args.anchor_frac = 0.0
        args.hidden_weight = 0.0

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(f"{args.device} requested but CUDA is unavailable")
        free_bytes, total_bytes = torch.cuda.mem_get_info(args.device)
        free_gb, total_gb = free_bytes / 2**30, total_bytes / 2**30
        print(
            f"accelerator preflight: {free_gb:.1f}/{total_gb:.1f} GiB free; "
            f"required {args.min_free_gb:.1f} GiB",
            flush=True,
        )
        if free_gb < args.min_free_gb:
            raise SystemExit(
                f"insufficient free accelerator memory before model load: "
                f"{free_gb:.1f} GiB < {args.min_free_gb:.1f} GiB"
            )

    prevalidated_gate_step: int | None = None
    if args.resume is not None and args.resume.is_dir():
        resume_manifest_path = args.resume / "checkpoint.json"
        if not resume_manifest_path.is_file():
            raise SystemExit(f"resume checkpoint manifest missing: {resume_manifest_path}")
        resume_manifest = json.loads(resume_manifest_path.read_text())
        resume_step = resume_manifest.get("step")
        if not isinstance(resume_step, int) or resume_step <= 0:
            raise SystemExit("resume checkpoint manifest has an invalid step")
        gate_request_path = (
            args.out / "capability_gates" / f"step-{resume_step}-required.json"
        )
        if gate_request_path.exists():
            if args.capability_gate_receipt is None:
                raise SystemExit(
                    f"checkpoint step {resume_step} ended a planned stage; "
                    f"run the frozen quick MMLU/GSM8K gate in {gate_request_path} "
                    "and pass --capability-gate-receipt before continuing"
                )
            request = json.loads(gate_request_path.read_text())
            if Path(request["checkpoint_path"]).resolve() != args.resume.resolve():
                raise SystemExit("capability gate request points to a different checkpoint")
            authorization = json.loads(args.capability_gate_receipt.read_text())
            validate_continuation_authorization(authorization, request)
            prevalidated_gate_step = resume_step
            print(
                f"prevalidated checkpoint-bound capability authorization "
                f"{args.capability_gate_receipt} before model allocation",
                flush=True,
            )
        elif args.capability_gate_receipt is not None:
            raise SystemExit(
                "--capability-gate-receipt was supplied but this checkpoint has "
                "no planned-stage gate request"
            )
    elif args.capability_gate_receipt is not None:
        raise SystemExit(
            "--capability-gate-receipt requires an atomic checkpoint-directory resume"
        )

    student, teacher, text_cfg = load_models(
        args.model, args.device,
        attn_impl=None if args.attn_impl == "none" else args.attn_impl,
        want_teacher=not use_cache_mode,
        revision=args.revision,
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
    compile_active = False
    if args.compile:
        try:
            student = torch.compile(student, dynamic=False)
            compile_active = True
        except Exception as e:
            print(f"torch.compile failed, continuing eager: {e}")

    identity = run_identity(
        args,
        data_manifest_sha256=data_manifest_sha256,
        resolved_chunked_head=chunked,
        resolved_compile=compile_active,
    )
    print(
        json.dumps(
            {
                "run_identity_sha256": canonical_json_sha256(identity),
                "data_manifest": str(data_manifest_path),
                "data_manifest_sha256": data_manifest_sha256,
                "checkpoint_root": str(checkpoint_root),
                "resolved_chunked_head": chunked,
                "resolved_compile": compile_active,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    meter = SignatureMeter(student, base_sample_frac=0.25, seed=args.seed + 3000)
    servo = MovementServo()
    bandit = KLBanditMixer(ARMS, seed=args.seed + 2000)
    data = {
        arm: ArmData(data_root, arm, args.seq_len, seed=args.seed + 1000 + i)
        for i, arm in enumerate(ARMS)
    }
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
    kl_history: list[tuple[float, float]] = []
    if args.resume is not None:
        if args.resume.is_dir():
            cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
            checkpoint, manifest = load_global_qat_checkpoint(
                args.resume,
                run_identity=identity,
                arms=ARMS,
                cuda_devices=cuda_devices,
            )
            checkpoint_model_state = join_model_state(checkpoint)
            student.load_state_dict(checkpoint_model_state, strict=True)
            if not state_tree_equal(student.state_dict(), checkpoint_model_state):
                raise RuntimeError(
                    "model state changed dtype or value during resume; refusing continuation"
                )
            optimizer.load_state_dict(checkpoint["optimizer"])
            if not state_tree_equal(optimizer.state_dict(), checkpoint["optimizer"]):
                raise RuntimeError(
                    "optimizer state changed dtype or value during resume; refusing continuation"
                )
            replay = checkpoint["replay"]
            for arm in ARMS:
                data[arm].rng.bit_generator.state = replay["arm_rng_states"][arm]
            bandit.kl = dict(replay["bandit_kl"])
            bandit._rng.setstate(replay["bandit_rng_state"])
            for name, value in replay["servo_state"].items():
                if not hasattr(servo, name):
                    raise ValueError(f"checkpoint has unknown servo field: {name}")
                setattr(servo, name, value)
            kl_history = [tuple(item) for item in replay["kl_history"]]
            restore_process_rng(replay, cuda_devices=cuda_devices)
            start_step = int(checkpoint["step"])
            print(
                f"exactly resumed {args.resume} at step {start_step}; "
                f"state_sha256={manifest['state_sha256']}",
                flush=True,
            )
            del checkpoint, replay
        else:
            if not args.allow_legacy_resume:
                raise SystemExit(
                    "legacy ckpt_*.pt resume is not exact; pass --allow-legacy-resume "
                    "only for a non-promotable recovery run"
                )
            checkpoint = torch.load(args.resume, map_location="cpu")
            student.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = int(checkpoint.get("step", 0))
            print(
                f"WARNING: non-exact legacy resume {args.resume} at step {start_step}; "
                "RNG, data mixer, servo, and KL history were not restored",
                flush=True,
            )
            del checkpoint

        if prevalidated_gate_step is not None and start_step != prevalidated_gate_step:
            raise RuntimeError("prevalidated capability gate step changed during checkpoint load")
        gc.collect()
        if args.device != "cpu":
            torch.cuda.empty_cache()

    tokens_per_step = args.micro_batch * args.grad_accum * args.seq_len
    total_steps = int(args.budget_tokens / tokens_per_step)
    if total_steps <= start_step:
        raise SystemExit(
            f"budget permits {total_steps} steps but checkpoint is already at {start_step}"
        )
    run_until_step = total_steps
    if args.stop_after_tokens is not None:
        run_until_step = min(
            total_steps, int(args.stop_after_tokens / tokens_per_step)
        )
        if run_until_step <= start_step:
            raise SystemExit(
                f"operational stage ends at step {run_until_step}, not after "
                f"resumed step {start_step}"
            )
    print(
        f"training schedule: total_steps={total_steps}, "
        f"this_stage_ends={run_until_step}, tokens_per_step={tokens_per_step}",
        flush=True,
    )
    gates = {int(g / tokens_per_step) for g in (2e9, 8e9, 20e9)}
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train_log.jsonl"
    t_last, tok_last = time.time(), start_step * tokens_per_step

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
    stop_signal: dict[str, int | None] = {"number": None}

    def request_stop(signum, _frame) -> None:
        if stop_signal["number"] is None:
            stop_signal["number"] = int(signum)
            print(
                f"received signal {signum}; will checkpoint after the current optimizer step",
                flush=True,
            )

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def checkpoint(step: int) -> Path:
        # Leave one known-good generation before beginning the next atomic save.
        # This bounds steady-state storage to two complete checkpoints plus the
        # one hidden in-progress directory created by save_qat_checkpoint().
        prune_qat_checkpoints(checkpoint_root, keep=1)
        replay = capture_replay_state(
            arm_rng_states={
                arm: data[arm].rng.bit_generator.state for arm in ARMS
            },
            bandit_kl=bandit.kl,
            bandit_rng_state=bandit._rng.getstate(),
            servo_state=servo.__dict__,
            kl_history=kl_history,
        )
        path = save_global_qat_checkpoint(
            checkpoint_root,
            step=step,
            model_state=student.state_dict(),
            optimizer_state=optimizer.state_dict(),
            replay_state=replay,
            run_identity=identity,
            reserve_bytes=int(args.checkpoint_reserve_gb * 2**30),
        )
        retention = prune_qat_checkpoints(
            checkpoint_root, keep=args.checkpoint_keep
        )
        print(
            json.dumps(
                {
                    "checkpoint": str(path),
                    "step": step,
                    "retained": retention["retained"],
                    "removed": retention["removed"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return path

    interrupted = False
    last_checkpoint_path: Path | None = None
    completed_step = start_step

    for step in range(start_step + 1, run_until_step + 1):
        completed_step = step
        stop_reason: str | None = None
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
                stop_reason = reason

        signal_number = stop_signal["number"]
        if (
            step % args.ckpt_every == 0
            or step == run_until_step
            or stop_reason is not None
            or signal_number is not None
        ):
            last_checkpoint_path = checkpoint(step)
        if stop_reason is not None:
            print(f"stopping: {stop_reason}", flush=True)
            break
        if signal_number is not None:
            interrupted = True
            print(
                f"checkpointed signal {signal_number} at step {step}; exiting without export",
                flush=True,
            )
            break

    if interrupted:
        return
    if last_checkpoint_path is None:
        raise RuntimeError("completed stage has no durable checkpoint")
    checkpoint_manifest_path = last_checkpoint_path / "checkpoint.json"
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text())
    gate_request = make_gate_request(
        step=completed_step,
        tokens=completed_step * tokens_per_step,
        checkpoint_path=last_checkpoint_path,
        checkpoint_state_sha256=checkpoint_manifest["state_sha256"],
        checkpoint_manifest_sha256=file_sha256(checkpoint_manifest_path),
    )
    gate_request_path = (
        args.out / "capability_gates" / f"step-{completed_step}-required.json"
    )
    atomic_write_json(gate_request_path, gate_request)
    print(
        f"wrote mandatory checkpoint-bound capability gate request "
        f"{gate_request_path}",
        flush=True,
    )
    if completed_step < total_steps:
        print(
            "stage boundary reached; evaluate the exact checkpoint before continuation",
            flush=True,
        )
        return
    export_unpacked(student, args.out / "unpacked")
    print(f"exported Prism-convention unpacked checkpoint to {args.out / 'unpacked'}")


if __name__ == "__main__":
    main()
