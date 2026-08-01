# Clark Quantize optimizer and training-mechanism audit

Status date: 2026-07-29

## Scope and evidence rule

This audit separates code, completed experimental receipts, and inference.
A run name or forensic resemblance is not evidence that an optimizer was used.

Evidence grades:

- **A — reproduced artifact:** source, checkpoint/result, and numerical or
  visual receipt are present.
- **B — historical artifact:** repository history and a release/result receipt
  agree, but the original training log or checkpoint is not local.
- **C — executable candidate:** implementation exists, but no matched completed
  run establishes quality.
- **D — hypothesis:** inferred from another model's weights or from literature.

No Qwen model was loaded or evaluated during this audit.

## Executive finding

The strongest Clark Quantize breakthrough is not a particular named optimizer.
It is the training representation:

```text
continuous master weight
  -> hard binary/ternary forward
  -> straight-through gradient to the master
  -> support changes only when the master crosses a code boundary
```

On Sana, the winning historical optimizer path was **AdamW8bit**, followed by a
scheduled proximal pull toward the active codebook and periodic closed-form
scale refresh. The release manifest tag contains the word `muon`, but the exact
trainer at release commit `56d77dd` constructs only
`bitsandbytes.optim.AdamW8bit`. Muon therefore must not be described as the
released optimizer.

## Evidence matrix

| Mechanism | Grade | What the repository establishes | Remaining question |
| --- | --- | --- | --- |
| MSE-optimal ternary PTQ | A | Closed-form top-k support/scale beats absmean on Sana PTQ and is used as the QAT initializer | How much it helps after long, matched QAT |
| Activation-aware GPTQ + rotation | B | Historical Sana measurements report a large PSNR gain at the same rate | Reproduce on the current pinned harness |
| Hard-forward master-weight QAT | A/B | TinyDiT artifacts and Sana release history show coherent recovery; current Swin2SR hard QAT is improving on held-out data | Replicate across seeds and model families |
| AdamW8bit + proximal pull | B | This is the actual optimizer path in the released Sana trainer | Separate the contribution of AdamW, prox, and scale refresh |
| Periodic closed-form scale refresh | B | Present in the released MW-QAT trainer and paired with learned scales | Determine optimal cadence and whether learned scales should pause around refresh |
| Protected continuous islands | A/B | Required for image models; their bytes are included in the ledger | Learn allocation by measured distortion per packed bit |
| BonsaiQAT8 | C | Executable low-state hybrid optimizer with int8 first moment and factored second moment | Quality, peak transient memory, and resume equivalence |
| Muon / batched Muon | C | Executable Newton-Schulz matrix update; historical code says about 3x slower than AdamW | Any matched quality-per-compute win |
| Movement/churn servo | C/D | Executable controller around a forensic target | Whether target matching improves behavior rather than imitation |
| Prism/Bonsai optimizer identity | D | Public method is undisclosed; weight geometry is compatible with several preconditioners | Cannot be promoted from inference without direct evidence |

## What actually won on Sana

The sequence in repository history matters:

1. `b2ff7be` introduced BonsaiMuon, hard STE masters, learned scales, proximal
   pull, and an optional flip trust region.
2. `51da544` disabled the flip trust region by default because its per-step
   search was about 15 times slower.
3. `8584e9c` made AdamW8bit the default because Muon was about three times
   slower, retaining proximal pull as an explicit post-update operation.
4. The explicit-mask branches—PrismLock, CTD, and Mirror-CTD—either froze the
   useful support or moved it destructively. The surviving abstraction returned
   to continuous masters with derived support.
5. `ba6eb21`, `20e390c`, and `e834f72` established the MW-QAT v2 recipe:
   differentiated island learning rates, frozen output projection, correct
   per-module group sizes, scale refresh, hard endpoint evaluation, and
   forensics.
6. Release commit `56d77dd` identifies `scripts/train_sana_mwqat.py` at step
   31,000. That exact file uses AdamW8bit. Its release tag
   `sana_t112_science_fresh_muon_g112` is historical naming, not an optimizer
   receipt.

The visible repository artifacts agree with this interpretation:

- CTD round 6 is collapsed over most prompts.
- Mirror-CTD at step 500 is highly distorted.
- MW-QAT v2 at step 1,500 is coherent, though still semantically different from
  its teacher.
- The released step-31,000 comparison is coherent and compact, but it does not
  establish zero quality loss; text and composition still visibly differ.

## BonsaiQAT8 code audit

`clark_quantize/bonsai_qat8.py` uses:

- one block-int8 first moment for each matrix;
- factored float32 row and column second moments;
- full AdamW for scales and continuous islands;
- optional threshold-local update amplification;
- optional proximal pull, scale refresh, sign-flip prevention, and periodic
  Newton-Schulz Muon.

The factored denominator matches the Adafactor-style reconstruction:

```text
V_hat ~= outer(v_row, v_col) / mean(v_row)
denom = sqrt(V_hat)
```

Persistent matrix optimizer state is roughly:

```text
1 byte * numel + 4 bytes * (rows + columns)
```

before framework and block-scale overhead. This is materially smaller than two
full optimizer moments. However, each matrix step still constructs several
float32 transient tensors (`m`, `g`, `g^2`, denominator/update, proposal), so
persistent-state compression does not by itself guarantee a low peak-memory
training run.

### Defect found and repaired

`global_step` controlled Muon, proximal, and refresh intervals but was not part
of the optimizer state dictionary. A resumed run silently restarted those
schedules at zero. The optimizer now:

- serializes `bonsai_global_step`;
- restores it on load;
- infers it from per-parameter steps for legacy checkpoints.

Four CPU tests now cover block-int8 error bounds, low-state matrix updates,
exact schedule-step resume, and legacy-state recovery.

## Muon conclusion

Muon is a useful preregistered arm, not a confirmed secret.

The weight-forensics reports correctly observe unusually uniform singular
spectra and broad update participation. Those observations are compatible with
orthogonalized or matrix-preconditioned optimization, but they do not identify
Muon uniquely. The repository's own researcher reconstruction says the
proprietary optimizer is undisclosed and describes PSGD/Kron or dual-space
preconditioning as an educated hypothesis.

Therefore:

- do not call Muon "confirmed";
- do not use forensic signature matching as a success criterion;
- compare Muon on held-out behavior per unit of time and memory;
- retain it only if it beats the AdamW control under matched exposure.

### Primary-literature check

The current literature supports testing efficient Muon variants, but still does
not establish Muon as a ternary-conversion optimizer:

- [8-bit Muon](https://arxiv.org/abs/2509.23106) reports optimizer-state
  quantization through 2.7B pretraining and up to 62% optimizer-state reduction.
  This supports the low-state direction in BonsaiQAT8, not its QAT quality.
- [Newton-Muon](https://arxiv.org/abs/2604.01472) derives a right-preconditioned
  update using layer-input second moments and reports a modest pretraining
  speedup over standard Muon. This is more directly compatible with Clark's
  activation-aware evidence than blindly orthogonalizing raw gradients.
- [Hierarchical Muon](https://arxiv.org/abs/2606.27216) replaces full-matrix
  Newton-Schulz with scheduled tiles. It reports lower update cost with small
  quality differences in single-seed transformer runs; it is a scaling/runtime
  arm after full Muon earns a quality signal.
- [TWLA](https://arxiv.org/abs/2606.13054) emphasizes calibration-metric
  relocation, Kronecker orthogonal shaping, and inter-layer activation-bit
  allocation. That strengthens the case for activation geometry and shaping as
  independent factors, not for a particular optimizer.

Prism's current public [Bonsai demo](https://github.com/PrismML-Eng/Bonsai-demo)
now exposes binary and ternary families through 27B and increasingly upstreamed
runtime formats. It still provides deployment artifacts rather than a
reproducible conversion optimizer. Scale is evidence that the target is
possible, not evidence for Clark to label Muon as their method.

## Stable negative results

These should not be restarted without a new falsifiable mechanism:

| Branch | Observed failure |
| --- | --- |
| Explicit density/mask objectives | Global density moved without reliably preserving useful support |
| PrismLock | Mask parameterization froze or misdirected support |
| CTD | Forced discrete transitions produced severe visual collapse |
| Mirror-CTD | Soft mask continuation hardened into distorted images |
| Replay/perceptual-only objectives | Broad structure survived but detail became soft or averaged |
| Bellman/value shaping | Learned trajectory closeness, not visible semantic correctness |
| PTD | Stable scalar plateau without a render champion |
| Stein/Jeffreys | Stable and non-collapsing, but no visual breakthrough |
| Self-oracle/CVF promotion | Failed stability/inversion gates |

The general lesson is to optimize teacher behavior and ground-truth task loss,
not proxy similarity to another checkpoint's code statistics.

## Current Swin2SR control

The active browser super-resolution run is a valuable clean control:

- optimizer: ordinary PyTorch AdamW;
- quantizer: groupwise MSE-optimal ternary;
- schedule: attention first, MLP second, then fully hard;
- data: 104,500 licensed UI/natural images plus 102,000 balanced
  FLORES-200 text canvases;
- held-out pilot: step 8,500 improved `+1.0575 dB` ground-truth PSNR over the
  old step-3,000 reference.

This run must finish and receive the full 3,040-sample held-out evaluation
before optimizer comparisons begin. Changing its optimizer mid-run would
destroy the control.

## Preregistered matched optimizer study

Use the same Swin2SR initialization, quantizer, source order, examples, hard
schedule, loss, and validation samples for every arm:

1. **AdamW control** — current recipe.
2. **AdamW + proximal/refresh** — isolate the Sana mechanism.
3. **BonsaiQAT8** — low-state first/factored-second moment, Muon disabled.
4. **BonsaiQAT8 + periodic Muon** — identical to arm 3 except the registered
   Muon interval.

Primary outcomes:

- paired ground-truth, teacher, and edge PSNR;
- worst-decile and per-script deltas;
- OCR accuracy on deterministic rendered text;
- wall time, examples/second, peak resident memory, checkpoint size;
- code churn, zero fraction, and scale statistics as diagnostics only.

Promotion requires a replicated held-out quality gain or equal quality with a
material memory/time advantage. Forensic resemblance alone cannot promote an
arm.

## Implication for the large-scale 1.58-bit program

The next large run should inherit only the validated invariants:

- optimal PTQ initialization as a control;
- hard forward from a continuous master;
- teacher-behavior loss plus task loss;
- explicit bit and island ledger;
- exact resume and immutable data order;
- optimizer chosen by matched evidence.

It should not inherit:

- a Muon assumption;
- target zero/churn/sign statistics as objectives;
- explicit mask editing;
- a claim of lossless Q4-to-Q1 conversion.
