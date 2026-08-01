# Output-convolution rescue study

Date preregistered: 2026-07-30
Selection source: development only
Status: waiting for the sensitive-input development winner

## Motivation

The external group-size-9 control reached step 20,000 with all 204 surfaces
started and only `upsample.conv` still transitioning. Forcing every started
surface hard scored 22.7661 dB ground-truth PSNR.

The pre-outcome 40-point projection grid showed:

| Method | Group size | GT PSNR | Whole-model bit delta vs g9 |
|---|---:|---:|---:|
| optimal MSE | 1 | 32.1937 dB | +0.09427 |
| optimal MSE | 2 | 29.2690 dB | +0.04124 |
| optimal MSE | 3 | 28.7411 dB | +0.02357 |
| optimal MSE | 4 | 26.2052 dB | +0.01473 |
| optimal MSE | 9 | 22.7661 dB | baseline |

Group size 1 is excluded because one FP16 scale per weight is not a meaningful
compressed ternary surface. Group sizes 2 and 3 are the only primary rescue
candidates: both materially reduce the early hard-endpoint corruption, and
group 3 uses 1,080 fewer FP16 scales than group 2.

## Frozen parent and controls

- Wait for the sensitive-input full-development gate to select one exact
  step-19,000 checkpoint.
- Preserve its model, optimizer moments, RNG, data epoch/cursor, input group
  size, convolution learning-rate multiplier, and all other training settings.
- Fork every output arm from that exact checkpoint before `upsample.conv`
  starts at step 19,200.
- Keep `swin2sr.first_convolution` fixed to the selected input recipe.
- Keep the final glyph-safe heldout closed.

## Arms

All arms use optimal-MSE ternary projection for `upsample.conv`.

| Arm | Output group size | Role |
|---|---:|---|
| matched output control | 9 | Current kernel-grouped baseline |
| output-g3 | 3 | Lower-scale-overhead rescue |
| output-g2 | 2 | Highest-quality shared-scale rescue |

Group size 1 remains an external diagnostic only and is never trained or
eligible.

## Training and checkpoints

- Continue each arm from step 19,000 through step 24,000.
- Record blended and hard-started fixed-crop metrics at steps 20,000, 22,000,
  and 24,000.
- Record target and whole-model ternary signatures at the same boundaries.
- At step 24,000 the 24,000-step schedule is complete; every one of the 204
  quantized surfaces must be fully hard and finite.
- Materialize one canonical hard state per arm. Never independently reproject
  continuous masters on different backends.

## Full-development selection

Evaluate all three canonical step-24 states over the complete 3,040-sample
development bundle, preserving paired samples and per-source results.

1. A rescue arm must beat the matched group-9 control on overall
   ground-truth and teacher PSNR.
2. Identify the rescue arm with the highest overall development
   ground-truth PSNR.
3. Retain another rescue arm only if its overall mean and every source mean
   are no more than 0.10 dB below that best arm.
4. Among retained arms, select the largest group size, minimizing scale
   overhead.
5. If neither rescue arm beats control or survives the source gate, select no
   output recipe and revise training.

Bootstrap intervals, edge quality, q4 deltas, target signatures, and artifact
accounting are reported but do not replace this rule.

## Evidence boundary

- The step-20 external probe used the uninterrupted input-group-9 control. It
  shortlists output representations but cannot select the output recipe.
- Only matched training from the selected input checkpoint and the full
  development gate can select g2 or g3.
- The result chooses a recipe for all-hard recovery and the optimizer-transfer
  study. It does not promote a model or authorize final heldout access.
