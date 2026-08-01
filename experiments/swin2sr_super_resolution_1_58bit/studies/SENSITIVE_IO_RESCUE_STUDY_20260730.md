# Sensitive input-convolution rescue study

Date preregistered: 2026-07-30
Selection source: development only
Status: waiting for the sealed control to reach the comparison checkpoints

## Motivation

The sealed `sensitive_io_v2` control exposed a new failure when
`swin2sr.first_convolution` began transitioning:

- step 15,000: 39.0241 dB blended versus 20.7513 dB hard-started;
- step 16,000: 36.3558 dB blended versus 22.8325 dB hard-started.

A deterministic group-size probe at step 16,000 found that optimal-MSE ternary
group size 2 raises the hard-started endpoint to 30.7220 dB. Its weighted
storage cost is only +0.01031 bits across the model's 977,604 quantized
weights relative to the group-size-9 control.

## Hypothesis

The input convolution needs a finer scale grid and a larger continuous-master
update than the residual convolutions. Ternary codes with two weights per FP16
scale can preserve enough input color/edge structure for hard-shadow training
to recover without materially changing the whole-model bit budget.

## Frozen source and initialization

- Resume from the sealed control's step-14,000 checkpoint, before the input
  convolution starts at schedule progress 0.62.
- Preserve the exact continuous masters, optimizer state, data epoch/cursor,
  augmentation RNG, seed, losses, hard-shadow cadence, schedule, teacher,
  validation files, and already-started surface configurations.
- Change only the input-convolution group size and the preregistered
  convolution learning-rate multiplier.
- Keep the output convolution at the sealed group-size-9 configuration.
- Do not open the glyph-safe final heldout.

## Arms

All learning-rate multipliers are relative to the global `1e-6` learning rate.

| Arm | Input group | Input method | Convolution LR multiplier |
|---|---:|---|---:|
| sealed control | 9 | optimal MSE | 0.25 |
| io-g2-lr025 | 2 | optimal MSE | 0.25 |
| io-g2-lr1 | 2 | optimal MSE | 1.0 |
| io-g2-lr4 | 2 | optimal MSE | 4.0 |

The sealed control is already running and is not restarted or modified.

## Checkpoints and metrics

- Record blended and hard-started development metrics at steps 15,000,
  16,000, 17,000, 18,000, and 19,000.
- Step 19,000 is the primary input-layer comparison: the input transition has
  ended at step 18,960 and the output transition does not begin until step
  19,200.
- Record per-surface code churn, zero fraction, sign inheritance, and relative
  master movement at steps 16,000 and 19,000.
- Evaluate the best two arms once on the 3,040-sample development bundle,
  preserving per-source and paired-sample results.

## Development selection rule

An arm is eligible only when:

1. its step-19,000 hard-started ground-truth and teacher PSNR are both higher
   than the sealed control;
2. its blended ground-truth PSNR is no more than 0.10 dB below the control;
3. no development source has a paired ground-truth PSNR delta below
   -0.10 dB versus the best eligible arm;
4. training, checkpoint resume, and materialization remain finite and
   provenance-complete.

Among eligible arms, select the smallest convolution LR multiplier within
0.10 dB of the best mean development ground-truth PSNR. This recipe choice
does not promote a model and does not authorize opening the final heldout.

## Output-convolution boundary

Do not transfer the input group-size result to `upsample.conv`. When the
sealed control reaches the output transition, run an independent group-size
probe for that 6,480-weight surface and preregister its rescue before training
an all-hard candidate.

## Protocol amendment — matched restart control

Amended 2026-07-30 after the first step-14,500 rescue checkpoint and before
any arm reached an input-transition checkpoint.

The `io-g2-lr025` arm scored 38.9394 dB at step 14,500, while the historical
uninterrupted control scored 38.6667 dB. The input convolution does not begin
transitioning until step 14,880, so this 0.2727 dB difference cannot be caused
by its group size. It reveals restart and/or MPS execution drift large enough
to confound comparison with the uninterrupted control.

A matched `restart-g9-lr025` control is therefore added. It forks the same
step-14,000 checkpoint, preserves the same model, optimizer moments, RNG,
data cursor, and source contract, and records an explicit group-size-9 input
surface plus the unchanged 0.25x convolution LR. At step 14,001 it reproduced
the group-2 0.25x arm exactly at 38.948371 dB, confirming equality before the
input transition.

The matched restart control is the primary comparator for causal group-size
and LR conclusions. The uninterrupted control remains an external trajectory
reference. This amendment does not change arm outcomes, thresholds, the
step-19,000 boundary, or the locked final heldout.

## Sealed full-development gate

The step-19 checkpoint selector may shortlist at most two arms by the frozen
four-image checkpoint rule. The matched restarted group-9 control and both
shortlisted group-2 arms are then evaluated with already-started surfaces hard
and the future output surface dense on the complete 3,040-sample development
bundle.

For that full-bundle gate:

1. identify the eligible arm with the highest development ground-truth PSNR;
2. retain another arm only when its overall mean and every source mean are no
   more than 0.10 dB below that best arm;
3. select the smallest convolution learning-rate multiplier among retained
   arms.

Paired bootstrap intervals, edge quality, teacher quality, and comparisons to
the matched control and q4 are reported, but they do not replace the
preregistered selection rule above.

`development-gate-manifest-v2.json` seals the evaluator, selector, studies,
bundle, q4 reference, margins, seed, and source files before any step-19 rescue
result existed. The earlier v1 gate is retained with an explicit invalidation
receipt because it accidentally imposed a different selection rule.
