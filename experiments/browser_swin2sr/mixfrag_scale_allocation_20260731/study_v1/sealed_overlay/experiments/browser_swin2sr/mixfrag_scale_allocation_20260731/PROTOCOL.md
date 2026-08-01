# Output-fragility scale allocation protocol

## Source and transfer boundary

MixFrag (`arXiv:2607.28589v1`, submitted 2026-07-30) independently quantizes
each Vision Transformer component at candidate bit widths, measures output
fragility, and solves a Multiple-Choice Knapsack Problem under a global bit
budget. Its reported task is classification/detection, its fragility is KL
divergence over output probabilities, and its backend is multi-bit AdaLog PTQ.

This study transfers only the isolation-plus-budget principle. It does not
claim that classification KL, ImageNet allocations, or 3–6-bit activation
results apply to Swin2SR ternary super-resolution.

Primary source: https://arxiv.org/abs/2607.28589v1

## Swin2SR adaptation

Starting authority is the selected all-hard ternary checkpoint with dense
FP16 per-group scales. For each of 204 weight surfaces and each scale choice,
only that surface's magnitude channel changes; all ternary codes and every
other surface remain fixed.

Candidate choices are:

- q1, q2, q4, and q8 learned/fitted FP16-center scale codebooks;
- dense FP16 per-group scales as the zero-compression reference.

Calibration uses 64 deterministic, source-balanced examples from training
corpora, never development or final-heldout samples. The reference output is
cached once. Fragility is the mean normalized output MSE plus 0.25 times mean
normalized Sobel-edge MSE between the isolated candidate and reference.
Ground truth is not used for allocation, preventing the calibration metric
from becoming a hidden development selection surface.

The allocator selects exactly one choice per surface and minimizes summed
isolated fragility under exact browser scale-storage bytes. Cost includes
FP16 centers, packed indices, four-byte alignment, or dense FP16 scales. Ternary
code bytes and raw non-quantized tensors are constant and excluded from the
optimization but restored in whole-artifact accounting.

## Budgets and gates

The preregistered scale-storage budgets are:

1. the completed sensitive-bridge scale storage;
2. two times that storage;
3. the current dense-FP16 scale storage.

An allocation is only a hypothesis. It must be progressively trained from the
same hard parent and evaluated on the sealed 3,040-sample development bundle.
No final-heldout evaluation is authorized unless an allocation independently
passes the existing q4 noninferiority, artifact-size, robustness, and runtime
gates. Because isolated fragilities are not guaranteed additive, measured
whole-model quality always overrides the knapsack prediction.

## Frozen cautions

- MixFrag v1 was submitted one day before this protocol and is not treated as
  settled evidence.
- Weight RMSE is diagnostic only; output fragility drives allocation.
- Layer choices may interact, so the MCKP objective is an approximation.
- Activation quantization is outside this study.
- Retained FP16 weight islands are tested separately; scale allocation cannot
  recover precision already lost by ternary codes.
