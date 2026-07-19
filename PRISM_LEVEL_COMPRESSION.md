# Achieving Prism-Level Compression: A Mechanistic Deep Dive

**Date:** 2026-07-06. Grounds: our reverse-engineering of Ternary-Bonsai
(`bonsai_ternary_deep_forensics.md`, `prism_ml_researchers_reconstruction.md`),
the Clark pack pipeline, and live diagnostics from the running Qwen3.5-4B KD job
at u=0.118.

---

## 0. The reframe: compression is two problems, not one

> **compression = (bits per weight) × (quality retained at that bit-rate)**

"Prism-level compression" means matching their point on the (bits, quality)
Pareto frontier: **4B at ~2.0–2.1 bpw scoring 70.7 avg** (89% of the FP16 base).
Those are *independent* axes and they have *opposite* status for us:

| axis | Prism | Clark today | verdict |
|---|---|---|---|
| **bits/weight** (packing) | 2.125 (Q2_0) | **1.65** (pack40-log6) | **we already win by 29%** |
| **quality @ bits** (training) | 70.7 | unmeasured; weights show a *gentler* fixed point | **the real gap** |

The compression *ratio* is a solved, even-won problem. The hard part — and the
only part worth calling "Prism-level" — is retaining quality at that bit-rate,
which is a **training** problem, not a packing problem. The rest of this doc is
mostly about that.

---

## 1. Bits: we already beat Q2_0 (the packing math)

Ternary carries log₂3 = **1.585 bits** of information. How each scheme spends bits:

| scheme | bits/weight | efficiency vs 1.585 | 4B trunk |
|---|---|---|---|
| FP16 | 16 | — | 8.04 GB |
| Prism **Q2_0 g128** | 2 + 16/128 = **2.125** | 74.6% | 1.068 GB |
| Clark **pack40 + log6** | 64/40 + 6/128 = **1.647** | **96.2%** | **0.828 GB** |
| theoretical floor | 1.585 | 100% | 0.797 GB |

Prism wastes 0.54 bits: the 2-bit code has a 4th codepoint that ternary never
uses (their own README calls Q2_0 "larger than a tight ternary packing"). Clark's
pack40 stuffs **40 trits into one 64-bit word** (3⁴⁰ = 1.2×10¹⁹ < 2⁶⁴ = 1.8×10¹⁹)
at 1.6 bits/trit — 96% efficient, near the entropy floor. **At identical ternary
weights, pack40 ships 29% smaller than Prism.** This is not aspirational; the pack
pipeline exists (`scripts/pack_sana_8x.py`, ~8× on Sana).

Headroom beyond pack40 (all on the bits axis, all additive to quality work):
- **Mixed precision by sensitivity.** Not every tensor needs ternary. The
  Fisher/damage waterfill in `stein_qat.choose_binary_ternary_rates` can spend
  1 bit on robust tensors (down_proj) and 2–3 on sensitive ones (attention out,
  first/last blocks), minimizing damage at a target average bit-rate. This is the
  *right* frontier move and Prism doesn't do it (uniform ternary).
- **Scale sharing / smaller scale.** log6 already beats fp16 scales; a shared
  per-tensor scale + 4-bit per-group deltas trims the 0.047 bpw scale overhead.

**Bottom line on bits:** ship pack40, and the Clark artifact is *strictly smaller*
than Prism's at equal weights. Do not spend effort matching 2.125 — we're past it.

---

## 2. Why ternary retains quality at all (the theory license)

Three facts make a 1.58-bit model viable, and they're worth stating because they
tell you *what the training has to accomplish*:

1. **Overparameterization.** A 4B model's function lives on a low-dimensional
   manifold; you're not compressing the *function*, you're finding a nearby
   *ternary-representable* function. QAT searches for it.
2. **Group-wise scale factorization.** `w = scale_g · code`. The ternary code
   carries **direction+support** (per 128-group); the fp16/log6 scale carries
   **magnitude**. Binary throws away the zero state; ternary's zero is *free
   expressivity* — the model chooses which connections to drop (Optimal Brain
   Surgeon reborn — and OBS is literally Hassibi's 1993 paper; "Bonsai" = a pruned
   tree is their in-joke).
3. **The one-bit license (their own theory).** Akhtiamov–Ghane–Hassibi, *One-Bit
   Quantization for Random Features* (2510.16250): quantizing all layers *except
   the last* incurs asymptotically **zero** generalization loss. This is the
   theoretical permission slip for the whole program — and it's why the LM head
   is treated gently and embeddings are frozen-PTQ.

So the training's job is: **find the ternary support + scales that parametrize a
function matching the FP16 teacher.** Prism's weight signature (37.5% zeros, 30%
churn, 91% sign-inheritance, 70% movement) is the *evidence* that their training
did this thoroughly. Ours, so far, has not — and the diagnostics say exactly why.

---

## 3. Where our run actually is: the mechanistic gap

Three live measurements, and each is a different kind of "not migrating":

**(a) Movement is flat at the PTQ floor — the deployed weights have barely left
their starting point.**

| step | u | movement | learned-fraction of budget |
|---|---|---|---|
| 1500 | 0.039 | 0.51875 | 0.02% |
| 4500 | 0.118 | 0.51907 | **0.17%** |

At 12% of budget, the dequantized weights have realized **0.17%** of the
floor→Prism learned-movement budget. They are *still at PTQ*. And critically, this
is **not** the √t reference trajectory (which would already be at 0.545) — the
curve is nearly flat, not slowly rising. The run is converging to the
**minimal-movement solution**.

This is not a bug — it's the mirror-descent implicit bias their own theory proves.
Matrix SMD (Akhtiamov–Pooladzandi–Hassibi, 2602.18997) converges to the
interpolator **minimizing Bregman divergence from initialization**. STE-on-master
*is* mirror descent (master = dual variable, quantizer = link). So the optimizer
is doing exactly what the math says: finding the ternary function *closest to PTQ*
that matches the teacher. The surprise is that such a close solution *exists* — and
that tells us the distillation task is **too easy**.

**(b) The distribution shape is unchanged — no sparsification.**

`zeros` is a within-group kurtosis proxy: `P(|w| < 0.5·mean|w|)`. Gaussian → ~0.31.
Ours: 0.312 → 0.3134 over 3000 steps (**flat**). Prism: 0.375 → their weights
became *more kurtotic*, concentrating magnitude into fewer connections. Our
network is **not concentrating magnitude** — the migration is "horizontal" (swap
which weights are active, at constant density) not "vertical" (reshape toward
sparsity). Note: **absmean is scale-invariant**, so weight decay does *nothing* to
zeros (shrink all weights → scale shrinks → codes identical). Reaching 0.375 needs
a genuine *shape* change, and nothing in the current recipe drives one.

**(c) Signs are frozen** (sign_inherit ~1.0 vs Prism 0.91). Sign flips need a
weight to cross zero — the hardest boundary — and only happen after large drift.
With ~zero drift (a), signs can't flip. This is a *symptom* of (a), not separate.

**Synthesis:** our run satisfies the KD objective by adapting *scales* and a thin
~2% of *support*, while leaving the weight cloud essentially at PTQ. It found the
Bregman-closest ternary function and stopped. KL fell 0.5→0.365 almost entirely
from **scale refinement + minimal support churn**, not reorganization.

---

## 4. The deepest lever: reorganization is driven by task difficulty

Here is the crux, and it reframes "how to get Prism-level":

> The amount a QAT model *must* reorganize is set by **how hard the distillation
> task is relative to the minimal-movement solution's capacity.** If a
> near-PTQ ternary function can match the teacher on your data, the optimizer
> will find it and never reorganize. Prism's weights reorganized 70% because
> their training made the minimal solution *insufficient* — vastly more tokens,
> broader/harder distribution, and (plausibly) a signal that constrains internal
> representations, not just output logits.

Our probe KL 0.365 with ~zero movement is the proof: on our 5B / logit-only KD,
the easy solution *is good enough* to match the teacher's top-64 logits. So the
model never pays the reorganization cost. **Token count alone won't fix this** —
more tokens of an easy task still admit the easy solution. You have to make the
task hard enough that the PTQ-adjacent function *fails*, forcing the weights to
move. That's a signal-design problem, and it's the lever the √t token-budget story
misses.

---

## 5. The three levers to close the quality gap, ranked

**Lever 1 — Harder distillation signal (highest leverage, most underrated).**
Logit-only top-64 KD is too weak; the minimal solution satisfies it. Add signal
that constrains *internal computation*, which a near-PTQ function can't fake:
- **Feature/hidden-state distillation** on every block's residual stream (we have
  `anchored_hidden_loss` — currently only on the 8 full-attention layers and
  annealed *off*; instead keep it *on*, all layers, un-annealed). Matching the
  teacher's intermediate representations forces the ternary weights to reproduce
  the teacher's internal geometry — a much stronger constraint than matching
  outputs, and it keeps gradients alive after logit-KL saturates.
- **Attention-map / relation distillation** (MiniLM-style): match Q·Kᵀ patterns.
- **Full-vocab KL** or higher temperature: top-64 lets the tail off the hook;
  the tail is where a lossy ternary function differs most from the teacher.

**Lever 2 — Explicit sparsification pressure (needed for the zeros; it does NOT
emerge).** Empirically flat zeros ⇒ reintroduce a *gentle* shape pressure, despite
the forensic read that Prism's sparsity was "emergent" (weight-forensics can't see
their objective; ours proves it doesn't emerge at our scale). Options, mildest first:
- Periodic **magnitude pruning of masters** toward the group threshold (push the
  smallest |w| below 0.5·s each N steps) — directly increases zeros without a loss
  term.
- A small **L1 on the master/scale ratio** `|w|/s` (scale-invariant-safe, unlike
  weight decay) — pulls mass toward the zero band, raising kurtosis.
- A soft **target-zero-fraction** term per group toward 0.375 (the mask-logit idea
  we dropped — revisit it, but as a *soft prior*, not a hard mask).

**Lever 3 — Token budget (necessary but not sufficient).** 5B was always the
"partial-signature v0." The 30–50B parity run gives cumulative drift and lets
Levers 1–2 act over enough steps. But budget *alone*, on the current easy signal,
mostly buys a lower-KL version of the same minimal-movement solution. Budget is the
substrate; Levers 1–2 are what make the substrate reorganize.

Do **not** reach for "raise the LR cap further" — the run is already pinned at
lr_mult 3.0 and the KL got volatile there. More LR on an easy task is churn without
quality; it fights the objective. The fix is a harder objective, not a hotter one.

---

## 6. The strategic insight: you may not need Prism's signature

The signature (37.5/30/0.91/0.71) is a **means**; benchmark quality is the **end**.
We inferred "must reorganize like Prism" from *their* endpoint — it is not proven
that the aggressive fixed point is *necessary* for 70.7. Our gentle, denser
(31%-zero), sign-preserved ternary might already score well: **denser ternary
throws away less information than Prism's sparser one.** It is entirely possible
our minimal-movement solution is *Pareto-competitive at 10× less training compute* —
which, for a compute-efficient lab, is a *better* story than "we spent Prism-scale
to match Prism."

**This is exactly why the eval was deferred, and why it should now run.** The
deferred benchmark on `ckpt_4500` (rented GPU-hour: GSM8K + MMLU, ternary +
baseline teacher) is the pivot:
- If the gentle ternary scores ~68+ → **stop early, ship pack40, claim the
  efficiency win.** We don't need the parity run.
- If it scores ~55 → the minimal solution is insufficient → commit the parity run
  **with Levers 1–2** (feature distillation + sparsity pressure), not just more
  tokens.

Measuring first sizes the entire effort. Chasing Prism's signature blind risks
spending a parity budget to reproduce a fixed point we may not need.

---

## 7. Concrete recipe for the Prism-level (or better) 4B

If the eval says we need it, the parity run is:

1. **Base:** Qwen3.5-4B (tied-embed → PTQ-frozen head, as now).
2. **Quantizer:** absmean g128 + STE (unchanged — it's the proven, near-optimal
   forward; keep it).
3. **Signal (Lever 1):** logit top-128 KL (T=2) **+ un-annealed all-layer hidden
   distillation** + optional attention-relation loss. This is the change that
   forces reorganization.
4. **Sparsity (Lever 2):** L1 on |w|/s ramped to target zeros→0.375, or periodic
   threshold-pruning every 500 steps.
5. **Budget (Lever 3):** 30–50B tokens; cosine LR to 0.05×; churn-servo repurposed
   to steer on the *movement* signal once it clears the floor (not churn, which
   saturates).
6. **Pack:** pack40-log6-fp8 → **0.83 GB, 29% under Prism's 1.07 GB** at equal
   quality; mixed-precision waterfill if pushing further.
7. **Eval gate:** benchmark every ~10B tokens; stop when the quality plateau meets
   or beats 70.7, don't run budget for its own sake.

**The headline:** Prism-level *bits* are already beaten. Prism-level *quality* is a
harder-signal + sparsity-pressure + budget problem — and the single highest-value
next action is the deferred eval, which tells us whether we even need to fight for
the aggressive signature or can ship the efficient gentle one today.
