# Signature-Steered Ternary Distillation for Qwen3.5-9B

**The problem:** forensics (`bonsai_ternary_deep_forensics.md`) recovered Prism's quantizer, freeze map, init behavior, and convergence signatures — but not their KD data mix, token count, or optimizer. Those are unrecoverable from weights.

**The solution:** don't guess the inputs — control toward the outputs. We know the *fixed point* a successful run lands on. So every unknown becomes a measured, closed-loop control variable:

| unknown | classical treatment | here |
|---|---|---|
| data mix | hand-tuned corpus ratios | **KL-bandit**: batch mix ∝ measured student↔teacher divergence per capability arm |
| token count | pick a number, pray | **budget-as-output**: stop on KL plateau ∧ signature band; √-diffusion reference curve |
| learning rate / optimizer | grid search | **movement servo**: PI controller on LR tracking the forensic movement trajectory; Muon retained (full-rank updates match the observed full-rank delta) |

One knob class replaces all of them: *targets*, and the targets are measured facts, not guesses.

Implementation: `clark_quantize/ternary_kd.py` (module) + `scripts/train_qwen35_ternary_kd.py` (reference trainer).

---

## 1. The fixed constants (from forensics — not tunable)

- Quantizer: per-group-128 (input dim) **absmean**: `s = mean|w|`, `code = clip(round(w/s))`, STE, `w_q = code·s`. Scales recomputed from the live master every forward. No learned scales, no thresholds, no flip constraints, no sparsity losses.
- Freeze map (Qwen3.5-9B): train `mlp.{gate,up,down}_proj` (32L), `self_attn.{q,k,v,o}_proj` (8 full-attn layers; q carries the fused output gate), `linear_attn.{in_proj_qkv, in_proj_z, out_proj}` (24L), `lm_head` at 0.25× LR. Freeze: `embed_tokens` (absmean-PTQ at export), every norm, `conv1d`, `A_log`, `dt_bias`, `in_proj_a/b`, all of `mtp.*`, all of `visual.*` (BF16 in v1). Recurrence stays fp32 (`mamba_ssm_dtype`).
- Init: masters = base BF16 weights promoted to fp32. At step 0 the quantized forward *is* the PTQ model — training only has to move the ~30% of codes that must change.

## 2. Unknown #1 — data mix → the KL-bandit

Capability arms mirror Prism's own eval axes (they optimized for these, so divergence on them is the right currency): `knowledge` (FineWeb-Edu/DCLM slice), `math` (OpenWebMath/MathPile), `code` (StarCoder2/StackV2 slice), `instruct` (SmolTalk/Tulu-style chats rendered through the Qwen3.5 chat template), `tools` (function-calling traces, BFCL-style), `multilingual` (Qwen's strength — don't let it silently die), optional `vision-text` arm later.

Every `probe_every` steps, evaluate mean per-token KL(teacher‖student) on a small frozen probe set per arm. Batch sampling probability = `floor + softmax(KL/τ)`. Broken capabilities automatically attract budget; healed ones release it. The mix is never chosen — it is *observed*. Teacher-forced KD only (student never samples during training): stable, cheap, and exactly what the movement evidence implies.

Loss: `KL(teacher ‖ student)` on top-K=64 renormalized logits (vocab 248,320 makes full-vocab KL memory-silly; top-64 holds ≫99% of teacher mass at T=1), plus optional 0.1× CE on true tokens as an anti-drift anchor. For the hybrid's recurrent path: **anchor distillation** — add `1 − cos(h_s, h_t)` on the residual stream after each of the 8 full-attention layers (indices 3,7,…,31), annealed to zero over the first 20% of budget. The 3:1 linear:full pattern means these are natural checkpoints: the DeltaNet blocks between anchors float freely while cumulative drift is clamped every 4th layer. This is our invention (not forensically observed) — it exists to keep early training stable through the fp32 recurrence, and it removes itself.

## 3. Unknown #2 — token count → budget-as-output

Prior for planning: code churn must flip ~30% of 8.9B code decisions under SGD noise; O(10–100) useful gradient signals per flipped decision puts the run at 10¹⁰–10¹¹ token-scale. Plan **30B tokens**, gates at **2B / 8B / 20B**, extend only if KL is still falling.

Reference trajectory (diffusion scaling): learned movement grows like √tokens. Dequant movement vs base starts at the PTQ floor 0.51 and should track
`m_ref(u) = sqrt(0.51² + 0.49²·u)`, u = tokens/budget → 0.71 at u=1.

Stop when: (a) probe-KL improvement < 2% over the last 15% of budget, AND (b) signature in band — movement 0.65–0.75, churn 0.28–0.38, zeros 0.36–0.41, sign-inheritance 0.88–0.93. If (a) fires while KL still falls steeply at u=1, extend the budget 1.5× and re-arm. Token count is the *output* of this rule.

Cost honesty: 30B tokens ≈ 6·N·D student + 2·N·D teacher ≈ 2.2e21 FLOPs ≈ ~16 days on one 8×H100 node (40% MFU), ~2 days on 8 nodes. The 8B-scale forensics say there is no shortcut (full-rank delta, 71% movement) — anyone promising a 100× cheaper path is fighting the measured evidence.

## 4. Unknown #3 — optimizer → Muon + movement servo

Keep `BonsaiMuon` for the 2D masters — with the falsified parts deleted (no scale param group, no prox pull, no flip trust region). Justification beyond incumbency: Muon's orthogonalized updates maximize descent per spectral norm, i.e. they *prefer* dense, full-rank weight motion — precisely the delta structure observed (stable rank 164–469, top-64 dirs = 10% energy). AdamW works too; Muon is the better-matched geometry.

LR is not chosen; it's *servoed*. Every probe cycle, a PI controller in log-LR-space corrects toward `m_ref(u)`:
- movement lagging → LR up; overshooting → LR down (multiplier clamped [0.1, 3]).
- Guardrails from forensics: sign-inheritance < 0.88 → LR ×0.7 (updates are violent enough to destroy base sign structure — a regime Prism's weights say never happens); churn > 0.40 early → LR ×0.8.
- Base peak LR 1e-4-ish (AdamW-equivalent RMS after Muon rms_match); warmup 1%, cosine to 10%; the servo's multiplier rides on top. lm_head group at 0.25× (its 85% PTQ residue says it barely trains).

Two phases fall out naturally: bulk (anchors on, servo active) and polish (last ~10%: anchors off, LR cosine tail — code churn rate decays on its own; no code-freezing gadget needed unless export-time churn > 0.1%/1k steps).

## 5. Dashboard: steer by Prism's numbers

`SignatureMeter` emits, every probe cycle: dequant movement, churn-vs-PTQ-init, zero fraction, sign inheritance, per-layer-depth profile (expect the U: ends quiet, middle hot; early-MLP dead columns appearing), per-kind zeros (down/v sparsest; gate/up most retrained). Divergence from the profile is diagnostic, not merely cosmetic — e.g. flat-by-depth movement ⇒ LR too low; sign inheritance collapsing ⇒ LR too high; zeros ≫ 41% ⇒ masters collapsing toward zero (check loss scale / grad clip).

## 6. Runbook

1. Tokenize per-arm corpora → uint32 memmaps + frozen probe slices (`scripts/train_qwen35_ternary_kd.py --prepare` stubs).
2. Launch: teacher BF16 (frozen) + student (surgery via `ternarize_qwen35`), FSDP/DeepSpeed wiring is infra-specific.
3. Watch gates at 2B/8B/20B; the meter + servo do the steering.
4. Export: `export_unpacked()` → Prism-convention dequantized safetensors (embed PTQ'd here) → existing Clark pack pipeline → GGUF Q2_0 g128 (bit-lossless by construction) / gemlite / MLX.

v2 options (out of scope, evidenced-adjacent): waterfilled binary/ternary per-tensor mix via `stein_qat.choose_binary_ternary_rates` to go sub-2-bit; ternary vision tower with image-KD; MTP ternary for speculative decoding on-device.

## 7. Single-A100 plan: run Qwen3.5-4B first

One 80 GB A100 cannot hold the 9B run (masters 35.6 + Muon momentum 35.6 + teacher 18.8 GB > 80 GB) and could not finish it anyway (~1.6k tok/s → 7+ months for 30B tokens). The 4B fits with ~30 GB headroom (masters 14.3 + momentum 14.3 + teacher 9.1 + frozen 2.1 + activations ~6 GB) at ~3.5k tok/s ≈ 300M tokens/day.

**Qwen3.5-4B recipe delta:** `tie_word_embeddings: true` — the head IS the embedding. Surgery detects the tie, skips wrapping lm_head, and PTQ-freezes the shared matrix in place before training (matching the frozen-PTQ signature of Prism's own tied 4B: token_embd zeros 0.3095 ≡ the proven-frozen 8B embed). Trainable = trunk only (~3.6B params). Same hybrid layout as the 9B: 24 linear_attn + 8 full_attention, hidden 2560, inter 9216.

Phases: **P0** synthetic-data pilot (hours — validates loop/VRAM/tok-s); **P1** 4–5B tokens real data (~2.5 weeks on the single A100) → v0 artifact + scoreboard vs Ternary-Bonsai-4B (70.7 avg @ ~1.07 GB Q2_0), expect partial signature (movement ~0.6, churn ~0.2); **P2** parity run (30B tokens) and/or the 9B on rented multi-GPU with the identical controllers — the A100 keeps serving as the eval/pack box.

### 7.1 Speed stack (after the 136 tok/s pilot)

The pilot's slowness was (a) the DeltaNet **torch fallback** — the 24 linear_attention layers need `flash-linear-attention` (Triton gated-delta-rule kernels) + `causal-conv1d`, without which the recurrence runs sequentially and materializes per-token state (also the 68 GB @ seq 512); and (b) the reference quantizer re-quantizing all 3.6B masters **every micro-forward** (~80+ GB HBM traffic each).

Fixes, in `scripts/setup_a100_speed.sh` + code:
- install `flash-linear-attention`, `causal-conv1d`, triton ≥3.1; full-attention layers use SDPA (flash-attn v2 / HF `kernels` hub optional);
- fused Triton absmean-dequant kernel (one fp32 read → one bf16 write, ~21 GB/refresh) with a **version-counter cache**: in-place optimizer updates bump the master's `_version`, so quantization happens exactly once per optimizer step and is reused across grad-accum micro-steps and probes;
- bf16 Muon momentum + bf16 Newton-Schulz (halves optimizer memory, NS on tensor cores; `--muon-fp32` to revert);
- frozen-probe teacher top-K cached at first probe (probes become student-only);
- TF32 on, pinned non-blocking H2D, `--attn-impl`, `--compile` flags.

Gate before phase-1: `python scripts/throughput_ladder.py --model Qwen/Qwen3.5-4B` must report **≥2500 tok/s** at some config (roofline ~4.2k at 40% MFU; 5B tokens ≈ 14 days at 4k). If it fails, the setup script's dispatch report says which kernel is still missing.
