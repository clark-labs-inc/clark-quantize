# Swin2SR Sensitive Hard-Shadow Study

Research dossier: `experiments/swin2sr_super_resolution_1_58bit/`

## Scope

This study applies lessons from Clark Quantize's LLM ternary work only to
BackdropKit's tiny Swin2SR super-resolution model. It starts from the
non-quantized `caidas/swin2SR-lightweight-x2-64` teacher. It does not start
from the shipped q4 artifact and does not pursue an unrelated language model.

The sealed run is `sensitive_hardshadow_20260730`. Selection uses only the
development bundle. The final glyph-safe held-out bundle remains locked until
a development candidate passes the preregistered quality gate.

## What transferred from the LLM experiments

1. Keep continuous master weights. Hard ternary values are used in the
   forward path and gradients reach the masters through a straight-through
   estimator.
2. Recompute group scales from current masters. Do not learn scales or train
   directly against churn, sparsity, sign inheritance, or codebook statistics.
3. Keep trainable continuous compensation parameters. For Swin2SR these are
   convolution/linear biases and LayerNorm parameters.
4. Use BonsaiQAT8 with periodic Muon geometry for 2D matrix masters. Small
   corrections and convolution masters retain the lower-rate AdamW role until
   a matched convolution optimizer study justifies changing it.
5. Preserve internal behavior with feature distillation. Swin2SR already
   anchors normalized hidden features as well as teacher pixels, ground-truth
   pixels, and image gradients.
6. Train the fully hard endpoint during the blend. Every second transition
   step, a second backward pass makes all already-started surfaces hard while
   leaving future surfaces dense. Its weight rises from 0.05 to 0.25.

## Failure corrected

The rejected full-surface run optimized only the blended branch. Its image
input and output convolutions both entered ternary at progress 0.70 and
finished at 0.92. Quality collapsed when those two sensitive endpoints moved
together, and the optimizer did not see the fully hard branch until the end.

The new schedule separates the image-facing layers:

| Surface | Start | End |
|---|---:|---:|
| Attention linears | 0.03 | 0.15 |
| Other core linears | 0.10 | 0.30 |
| Residual body convolutions | 0.28 | 0.50 |
| Position-bias linears | 0.48 | 0.65 |
| Input convolution | 0.62 | 0.79 |
| Upsampling/output convolution | 0.80 | 0.98 |

For the 24,000-step schedule this gives the input convolution 4,080 transition
steps and the output convolution 4,320 steps, without overlap.

## Sealed run

- Total steps: 30,000
- Ternary schedule: 24,000 steps
- Fully hard recovery: 6,000 steps
- Training images: 206,500 per balanced epoch
- Multilingual synthetic UI images: 102,000 across 204 languages
- Matrix quantizer: absmean, group 60, Hadamard-shaped core
- Convolution quantizer: optimal-MSE, kernel-aligned groups
- Optimizer: BonsaiQAT8/Muon for matrix masters; lower-rate AdamW roles for
  convolution masters and continuous corrections
- Loss: teacher 0.55, ground truth 0.20, edge 0.15, hidden feature 0.10
- Hard shadow: 0.05 to 0.25, every 2 transition steps

## Acceptance gates

The candidate is rejected unless it is statistically non-inferior or better
than the shipped q4 model on overall, edge, teacher, source, language, and
script slices. Passing quality is necessary but not sufficient: the packed
artifact must also be materially smaller and the native ternary runner must be
viable on WebGPU, WASM, and CPU. No BackdropKit integration follows from a
training checkpoint alone.
