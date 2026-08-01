# Output-convolution group-size probe

Status: external step-20 diagnostic complete; the primary probe waits for the
selected input-convolution recipe.

## Question

Which ternary scale grouping preserves color, texture, and small UI detail for
the 6,480-weight `upsample.conv` surface without erasing the whole-model size
advantage?

The input-convolution group-size result is not transferred automatically.
`upsample.conv` has shape `(12, 60, 3, 3)`, a different channel geometry and
four times as many weights.

## Schedule boundary

- The output convolution begins transitioning at schedule progress `0.80`,
  step 19,200 of the 24,000-step schedule.
- It finishes at progress `0.98`, step 23,520.
- A step-20,000 probe is an early hard-projection diagnostic.
- Step 22,000 and step 24,000 probes measure adaptation later in and after the
  transition.

Every probe forces already-started surfaces hard while leaving no future
surface artificially quantized.

## Grid

Evaluate both optimal-MSE and absmean ternary projection over group sizes:

`1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 45, 60, 72, 90, 120, 180, 360, 720, 1620, 6480`

Report:

- ground-truth and teacher PSNR on the fixed product-UI development crops;
- per-layer scale count and effective layer bits;
- weighted whole-model bit delta versus group size 9;
- target code churn, zero fraction, and master movement;
- visual side-by-side renders for the leading non-degenerate groups.

Group size 1 is a diagnostic upper bound, not a promotion route. A candidate
must use shared scales and remain materially smaller than q4 after the complete
artifact ledger.

## Evidence boundary

The uninterrupted group-9 control may be probed at steps 20,000, 22,000, and
24,000 as an external trajectory diagnostic. It cannot select the output
recipe because its input convolution still uses group size 9.

The primary output recipe is selected only after:

1. the sensitive-I/O development gate chooses the input group and convolution
   learning-rate multiplier;
2. that exact checkpoint is continued past step 19,200 with preserved
   optimizer moments and data cursor;
3. the output grid is repeated on that selected-input trajectory;
4. the leading shared-scale choices are trained from the pre-output step-19
   checkpoint and compared on the full development bundle.

No result from this probe authorizes final heldout access or model promotion.
