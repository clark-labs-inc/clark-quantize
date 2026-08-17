# A100 Runbook — Qwen3.5-4B Ternary KD (signature-steered, max-speed)

The single source of truth for the single-A100 run. Supersedes the scattered
instructions in earlier notes; recipe rationale lives in
`TERNARY_QWEN35_RECIPE.md`, forensic evidence in `bonsai_ternary_deep_forensics.md`.

## 0. What's running and why it's fast now

| bottleneck (136 tok/s pilot) | fix | where |
|---|---|---|
| DeltaNet torch fallback (sequential recurrence, huge activations) | flash-linear-attention Triton kernels + causal-conv1d | `setup_a100_speed.sh` |
| 3.6B masters re-quantized every micro-forward (~80 GB HBM traffic) | fused Triton absmean kernel + `_version`-counter cache: quantize once per optimizer step | `ternary_kd.py::AbsmeanTernaryLinear` |
| [seq, 248320] logits materialized ×2 models (~2 GB/µbatch each) | vocab-chunked head losses, per-chunk grad recompute (~130 MB peak) | `student_kd_ce_chunked`, `teacher_topk_chunked` |
| ~100 sequential fp32 Newton-Schulz loops per step | shape-batched bf16 NS (one bmm chain per shape bucket) | `BonsaiMuonFast` |
| fp32 momentum (14.3 GB) | bf16 momentum (7.2 GB) | `collect_param_groups` defaults |
| teacher re-run on frozen probes | teacher top-K cached at first probe | trainer |
| allocator fragmentation OOMs | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | trainer (env default) |
| teacher forward at all (optional, +25-30%) | offline top-K corpus cache; teacher leaves the GPU | `cache_teacher_topk.py` + `--teacher-cache` |

Startup self-checks: chunked-head KD is verified against the reference path on a
real batch (falls back with a warning if mismatched); the Triton quantizer
verifies against the torch reference in `setup_a100_speed.sh`.

## 1. One-time setup

```bash
cd /workspace/clark-quantize
bash scripts/setup_a100_speed.sh          # installs triton/fla/causal-conv1d, verifies dispatch
```
Must print `fla gated_delta_rule: OK` and quantizer match ≥ 0.9999. If fla is
missing, DO NOT proceed — everything else is noise next to that fallback.

**Gate trap (hit 2026-07-03):** transformers' Qwen3.5 `is_fast_path_available`
requires `causal_conv1d` AND fla — if causal-conv1d fails to build (no nvcc on
the box, CUDA 13 setup.py bug), transformers silently disables the fla kernels
too and falls back to the 100×-slower torch DeltaNet. Fix in place: a torch
shim package at `/workspace/causal_conv1d_shim` (exact `causal_conv1d_fn` /
`causal_conv1d_update` semantics via cuDNN depthwise conv, verified bit-exact)
is pip-installed to satisfy the gate. Check for the log line "fast path is not
available" — it must NOT appear. flash-attn note: PyPI tops out at 2.8.3 (no
cu130 wheel); FA3/FA4 are Hopper/Blackwell-gen and sm80 caps flash backward at
head_dim 192 < Qwen3.5's 256 — SDPA auto-dispatch is correct on A100; revisit
FA4 only on rented H100s.

Pin Qwen3.5-4B to revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Every scientific launch must
pass this value with `--revision`; floating `main` is rejected.

## 2. Measure before training

```bash
python scripts/profile_components.py --model Qwen/Qwen3.5-4B --seq-len 2048 --micro-batch 2
python scripts/throughput_ladder.py   --model Qwen/Qwen3.5-4B                  # anchors on (first 20% regime)
python scripts/throughput_ladder.py   --model Qwen/Qwen3.5-4B --no-anchors     # steady-state regime
```
- Profiler tells you where remaining ms go (teacher fwd / student fwd / bwd / muon / quantizer).
- Ladder prints tok/s + peak GB per (seq, mb, ckpt) and a PASS/FAIL gate.
- **Gate: ≥ 2500 tok/s** somewhere on the ladder (roofline ≈ 4.2k at 40% MFU; with
  `--teacher-cache` later, ≈ 5.5k). Below gate → re-check dispatch audit; suspect
  attn-impl (`--attn-impl flash_attention_2` if installed) and ckpt-off rungs.

## 3. Data

```bash
# pilot data (validates loop only)
python scripts/prepare_kd_arms.py --out /workspace/kd_arms_synth --synthetic 5e6 --model Qwen/Qwen3.5-4B
# real corpus (~2.4B tokens across 6 arms; needs `huggingface-cli login` for gated sets)
python scripts/prepare_kd_arms.py --out /workspace/kd_arms --model Qwen/Qwen3.5-4B
```

Seal the completed token files before loading a training model:

```bash
python scripts/seal_kd_arms.py \
  --data-root /workspace/kd_arms \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

The trainer rehashes every arm against this manifest on fresh launch and
resume. Missing, shortened, changed, symlinked, or unpinned data fails before
model allocation.

## 4. Phase 0 — bounded stage and mandatory capability gate

```bash
DATA_ROOT=/workspace/kd_arms \
OUT_ROOT=/workspace/ckpt/qwen35-4b-ternary \
STOP_AFTER_TOKENS=100000000 \
scripts/spot/run_qwen35_4b_qat_stage.sh
```
The full schedule remains 5B tokens; `STOP_AFTER_TOKENS` is only a restart-safe
operational boundary. Pass criteria: tok/s within ~15% of ladder,
`chunked-head verification` printed OK, probe KL falling, no OOM at steady state,
and a passing checkpoint-bound MMLU/GSM8K authorization receipt. The exact eval
and resume commands are in `QWEN35_4B_QAT_STAGED_LAUNCH.md`. The trainer refuses
to continue a planned stage checkpoint without that receipt.

## 5. Phase 1 — staged continuation toward 5B tokens

```bash
DATA_ROOT=/workspace/kd_arms \
OUT_ROOT=/workspace/ckpt/qwen35-4b-ternary \
STOP_AFTER_TOKENS=500000000 \
RESUME_CHECKPOINT=/workspace/ckpt/qwen35-4b-ternary/checkpoints/checkpoint-<N> \
CAPABILITY_GATE_RECEIPT=/workspace/evals/qwen35-4b-step-<N>-authorization.json \
scripts/spot/run_qwen35_4b_qat_stage.sh
```
At 4k tok/s ≈ 14 days; at 3k ≈ 19 days. Checkpoints are temporary-directory
writes followed by an atomic rename and SHA-256 manifest. They retain the
complete split model state without duplicating masters, optimizer state,
Python/NumPy/Torch RNG, every arm RNG, KL-bandit RNG and estimates, movement
servo, and KL history. At most two complete checkpoints are retained.

Resume from the directory, not directly from `state.pt`:

```bash
python scripts/train_qwen35_ternary_kd.py <same exact arguments> \
  --resume /workspace/ckpt/qwen35-4b-ternary/checkpoints/checkpoint-1000
```

Run identity includes the immutable model revision, sealed data hash, all
training-affecting flags, source hashes, and framework versions. Drift fails
closed. Old `ckpt_*.pt` files lack replay state and are non-promotable; they
require the explicit `--allow-legacy-resume` escape hatch.

SIGTERM/SIGINT requests are handled at the optimizer-step boundary: the
trainer writes one complete checkpoint and exits without attempting export.

**Legacy non-promotable throughput experiment:** teacher-cache files built by
the current helper are not atomic or content-hash sealed. The trainer therefore
rejects them unless `--allow-unsealed-teacher-cache` is passed, and any such run
is explicitly non-promotable. A future sealed-cache format may restore this lane.

For forensic reproduction only:
```bash
python scripts/cache_teacher_topk.py --model Qwen/Qwen3.5-4B \
  --data-root /workspace/kd_arms --out /workspace/kd_teacher_k16 --k 16 --seq-len <best>
# disk: ~230 GB at K=16 for 2.4B tokens; K=8 halves it
python scripts/train_qwen35_ternary_kd.py ... --teacher-cache /workspace/kd_teacher_k16 \
  --allow-unsealed-teacher-cache --kd-topk 16 --anchor-frac 0
```
Trade-off: K=16 (vs 64) truncates the teacher distribution tail; acceptable for
the KD term (≥97% of teacher mass), and the freed VRAM usually buys a bigger
micro-batch on top of the teacher removal.

## 6. Steering (read the log, don't babysit)

Every probe line prints: `tok_per_s, peak_gb, mean_kl, probe_kl per arm, mix,
lr_mult, movement, churn, zeros, sign_inherit, distance, in_band`.

| signal | healthy | if not |
|---|---|---|
| movement | tracks √(0.51²+0.49²·u) → 0.71 | servo handles it; if lr_mult pinned at 3.0 raise `--master-lr` |
| churn | → ~0.2 at 5B, ~0.30 at parity | flat ⇒ LR too low; >0.40 ⇒ servo cuts LR automatically |
| sign_inherit | 0.88–0.93 | below ⇒ updates too violent (servo ×0.7 kicks in; check grad clip) |
| zeros | drifting 0.32 → 0.37–0.40 | ≫0.41 ⇒ masters collapsing; check loss scale |
| mix | shifts toward lagging arms, then rebalances | one arm pinned ⇒ its probe/probe data is broken |
| mean_kl | monotone down, plateaus late | flat from start ⇒ data or dispatch problem, stop early |

Stop rule is automatic (`should_stop`): KL plateau ∧ signature in band; gates at
2B (sanity), 8B, 20B print advisory status.

## 7. Failure playbook

- **OOM at launch**: drop micro-batch first, then seq (ladder already told you the
  frontier); `expandable_segments` is on by default; last resort `--muon-fp32` OFF
  is already default-off (bf16), so check that `--no-chunked-head` wasn't passed.
- **Slow (<50% of ladder)**: `profile_components.py`; usually fla missing (dispatch
  audit) or grad checkpointing double-recompute (try the ckpt-off rung if VRAM allows).
- **`chunked-head verification FAILED`**: trainer already fell back to the exact
  reference path (slower, correct). File the printed values; likely a transformers
  version changed hidden-state conventions.
- **Loss spikes / NaN**: bf16 NS is the only reduced-precision state; retry the step
  with `--muon-fp32`. The recurrence is fp32 by config (`mamba_ssm_dtype`).
- **Probe KL rises on one arm only**: bandit will chase it; if persistent, that
  arm's probe slice may overlap training data (rebuild arm with fresh seed).

## 8. Export & pack

Training ends with `export_unpacked()` → Prism-convention bf16 ternary
safetensors at `<out>/unpacked` (embeddings already exactly ternary; verified
bit-losslessly packable). Then the existing Clark pipeline:
unpacked → pack40-log6-fp8 (1.585+ε bits/w, beats Prism's Q2_0 2.125) and/or
GGUF Q2_0 g128 for llama.cpp comparison. Benchmarks: EvalScope suite matching
Prism's table (MMLU-Redux, MuSR, GSM8K, HE+, IFEval, BFCLv3) — scoreboard
target: Ternary-Bonsai-4B = 70.7 avg @ 1.07 GB.

## 8.5 Mid-run checkpoint eval (`scripts/eval_ternary_ckpt.py`)

Turns "probe-KL keeps dropping" into an actual capability number. Evaluates the
**deployed bf16 dequant** of an atomic QAT checkpoint directory (reconstructed
on CPU: dequant every `weight_master`, copy frozen tensors, drop vision/MTP for
text eval, and require an exact assignment for every base-model tensor). Legacy
`ckpt_*.pt` inputs remain readable only for forensics and are non-promotable.
The sealed Gate B manifest fixes the exact GSM8K generation and subject-balanced
MMLU log-probability denominators.

- **GPU preflight + self-cap**: aborts if free < `--min-free-gb`, caps the
  process to the free slack, so it cannot OOM a co-resident training run. On an
  80 GB card with training at ~71 GB the ~9 GB text model is too tight and the
  preflight *will* abort (by design) when a probe step transiently grows.
- **CPU path (reliable, zero GPU contention)**: `CUDA_VISIBLE_DEVICES="" --device cpu`.
  MMLU is single-forward — ~100 questions in tens of minutes on the box's vCPUs,
  fully decoupled from training. GSM8K needs generation (too slow on CPU); run it
  on the GPU only when training frees the card, or on a second host.

Create the immutable Gate B sample manifest once, using exact dataset revisions,
and reuse the same file for the BF16 control and every checkpoint. The task names,
sample counts, shots, dataset fingerprints, and row hashes live in this manifest;
the evaluator intentionally has no `--tasks` or `--limit` escape hatch.

```bash
python experiments/large_scale_ternary/prepare_gate_b_manifest.py \
  --tasks mmlu \
  --mmlu-revision <exact-mmlu-commit> \
  --mmlu-limit 100 \
  --out /workspace/evals/qwen35-4b-mmlu-quick.json
```

```bash
# MMLU now, CPU-isolated (safe alongside training):
CUDA_VISIBLE_DEVICES="" python scripts/eval_ternary_ckpt.py --model Qwen/Qwen3.5-4B \
  --model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --arm qat_ternary \
  --ckpt <run>/checkpoints/checkpoint-1500 \
  --manifest /workspace/evals/qwen35-4b-mmlu-quick.json \
  --device cpu --out eval_1500.json

# BF16 parent control, with the identical sealed denominator:
CUDA_VISIBLE_DEVICES="" python scripts/eval_ternary_ckpt.py --model Qwen/Qwen3.5-4B \
  --model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --arm bf16 \
  --manifest /workspace/evals/qwen35-4b-mmlu-quick.json \
  --device cpu --out eval_bf16.json
```
Read the number against Prism's 4B (70.7 avg) and the base Qwen3.5-4B teacher
(the `--arm bf16` receipt). Rising churn only matters if capability also holds.

The Gate B slice is useful during training, but it does not replace the mandatory
promotion tripwire in `QUALITY_PREFLIGHT.md`. Before extending a checkpoint,
expanding ternary depth, or moving to 27B, run the repository's frozen quick
WikiText perplexity plus MATH-500 and CUTE-Quick-20 character-level suite against
the exact BF16 parent. A pinned subject-balanced MMLU slice may replace CUTE only
when the candidate runtime cannot generate characters. MATH-500 alone is the
minimum emergency tripwire. Loss, KL, teacher-forced top-1, and kernel parity
never authorize a model-changing continuation by themselves.

## 9. Known headroom not yet taken (in order of value/effort)

1. `--compile` (regional inductor fusion) — try once stable; graph breaks possible with fla.
2. flash-attn v2 for the 8 full-attention layers (source build ~30 min) — a few %.
3. int8 teacher (torchao W8A16) — ~1.7× teacher fwd, KD-fidelity risk; only if
   teacher-cache is infeasible.
4. CUDA graphs — not worth the pain with HF + fla + checkpointing.
