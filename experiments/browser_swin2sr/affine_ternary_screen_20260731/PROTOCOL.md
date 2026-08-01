# Affine ternary weight-screen protocol

## Hypothesis

The remaining symmetric-ternary reconstruction error is partly caused by
non-zero group means. Keeping the existing ternary code map byte-identical but
fitting FP16 shift and positive-scale values per group will reduce aggregate
weight squared error by at least 10% relative to the selected refined model's
current hard projection.

## Intervention

For each of the 204 quantized weight surfaces, use its exact current group size
and current hard ternary codes. Fit the least-squares representation
`weight = shift + scale * code`, with `code` in `{-1, 0, +1}` and non-negative
scale. Roundtrip both continuous values through FP16 before scoring. No code,
bias, normalization value, activation, model output, or data sample changes in
this screen.

A second alternating-fit result may reassign ternary codes and is reported only
as an exploratory upper bound. It cannot advance the fixed-code hypothesis.

## Advancement gate

Advance to a separately sealed, 100-record development screen only if all are
true:

1. fixed-code FP16 affine values reduce aggregate weight squared error by at
   least 10%;
2. no surface category worsens by more than 1%;
3. exact extra shift storage remains within the existing 2,838,021-byte artifact
   ceiling after later packing;
4. the runtime design explicitly accounts for the input-sum correction term;
5. the screen has not read training, development, or final-heldout data.

Weight error is not an output-quality proxy. Passing this gate only authorizes a
small output screen, not training, final-heldout access, runtime implementation,
or product promotion.

## Research transfer boundary

TWLA's E2M-ATQ uses row-wise affine ternary levels
`{shift-scale, shift, shift+scale}` and output-aware calibration for LLMs. This
study transfers only the affine alphabet to Swin2SR's existing group geometry.
It does not import LLM activations, Kronecker rotation, activation quantization,
or TWLA's quality claims.

Primary source: https://arxiv.org/abs/2606.13054v1
