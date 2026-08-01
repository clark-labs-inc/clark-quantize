# Scientific journal

## 2026-07-31 — Preregistration

The selected refined checkpoint uses symmetric ternary weight levels with one
positive magnitude per group. TWLA suggests that a group offset can be a useful
second continuous degree of freedom. This is plausible for biased groups but is
not free: a linear group needs an additional `shift * sum(input_group)` term,
and a convolutional kernel group needs the corresponding patch-sum term.

The first experiment is therefore weight-only and data-free. It freezes the
current ternary codes to isolate the value of the affine magnitude channel. A
10% aggregate squared-error reduction is required before spending evaluator or
runtime effort. Reassigned-code results are diagnostic and cannot satisfy the
gate. No training, development, or final-heldout samples are authorized.

### Preflight correction

The v1 command failed before checkpoint loading because its `PYTHONPATH` omitted
the repository root required by `clark_air`. No output was written. The failed
manifest is preserved; v2 changes only `PYTHONPATH` from
`experiments/browser_swin2sr` to `experiments/browser_swin2sr:.`. Hypothesis,
checkpoint, source hashes, thresholds, and data prohibitions are unchanged.

## 2026-07-31 — Weight-only result

### Fixed-code result

On all 977,604 quantized weights, fitting FP16-roundtripped affine shift and
scale values while preserving the current ternary code map reduced squared
error to the continuous masters from 3,107.849342 to 2,436.834206, a 21.59098%
reduction. Relative RMSE moved from 0.534243 to 0.473066. Every category
improved: core linears 20.73550%, position-bias linears 24.50080%, residual-body
convolutions 12.10478%, and image-I/O convolutions 83.09602%. The added FP16
shift channel costs exactly 75,356 data bytes across 37,678 groups before
container metadata.

The preregistered 10% weight-error gate passes. This is not an output-quality
result and does not authorize training, final-heldout access, runtime work, or
promotion. The report SHA-256 is
`d69decf81b43f06adadbb34185396e854ae700e38db843c6f6c3a2030fa63f89`;
the corrected v2 manifest SHA-256 is
`4b4c63a964ef9c594cc72036dcc3a7c564dd80905c18a1010255de4603081c39`.

### Dense-side-channel caveat and narrowed candidate

The image-I/O category uses group size 2. Two FP16 continuous values per
two-weight group can nearly encode those weights directly, so its large error
reduction is not meaningful compression evidence. The same concern applies to
the two group-2 position-bias surfaces and, more weakly, one group-3 core
surface.

A cleaner candidate retains only group-60 core and position-bias linears. Its
189 surfaces and 783,856 weights capture 641.387619 squared-error units, or
95.58467% of the full affine improvement, while adding only 26,208 FP16 shift
bytes. Within that subset, error falls 21.93662%. This avoids convolutional
patch-sum kernels and dense group-2 side channels. It is the only affine variant
eligible for the separately sealed 100-record output screen.

### Exploratory code reassignment

Allowing the alternating solver to reassign ternary codes reduced aggregate
weight error by 41.99504%. This result is an upper-bound diagnostic only. It
changes the discrete intervention and cannot satisfy the fixed-code hypothesis
or select a model without a separate preregistration.

## 2026-07-31 — Output screen sealed

Before reading any affine model output, a 100-record screen was sealed for the
group-60 linear-only candidate. It uses the same first 100 records and exact
control report as the earlier continuous-master falsification probe. Advancement
requires a positive lower 95% paired ground-truth PSNR delta and non-negative
mean edge and teacher deltas. Failure on any condition rejects static affine
restoration; it does not authorize a post hoc island or code-map adjustment.

### Runtime-layout caveat discovered before output

All 143 selected core linears are row-aligned because their input widths are 60
or 120; their group-60 shifts capture 384.048531 squared-error units for 22,920
added bytes. The 46 position-bias linears have shapes `(6, 512)` or `(512, 2)`,
so flat groups of 60 cross output-row boundaries. They capture another
257.339088 units for 3,288 bytes, but cannot use a simple per-row input-sum
correction kernel.

The sealed output screen remains unchanged because this layout was inspected
before any affine output and tests the static mathematical intervention. If it
passes, a later runtime study must either ablate to the row-aligned core subset
or preregister a row-aligned position-bias representation. Dense unpacking is
not accepted as proof of a fast native affine kernel.

## 2026-08-01 — Final disposition

This affine screen is closed as a diagnostic study. No writer is active and no
affine candidate was promoted. The later G3 mixed-precision artifact is the
selected still-image release; its exact evidence is recorded in
`../joint_bonsai_recovery_20260801/SCIENTIFIC_JOURNAL.md`. The row-alignment
caveat remains a requirement if an affine correction is studied again.
