# Full-surface optimizer-transfer study

Status: preregistered design; launch blocked on the sensitive-I/O development
decision, not on final heldout evidence.

## Question

Does the three-seed core-QAT AdamW result transfer to the complete 977,604
weight Swin2SR ternary surface once the image-facing convolution recipe is
fixed?

The optimizer replication selected AdamW, but that experiment did not expose
the staged image-I/O convolutions. Treating it as proof for full-surface
training would overstate the evidence.

## Locked inputs

- FP32 teacher and revision already recorded in the program journal.
- Balanced 206,500-image training epoch and exact sampler/data-resume contract.
- `sensitive_io_v2` 24,000-step schedule and 30,000 total steps.
- Hard-shadow objective and validation cadence used by the sealed control.
- First-convolution group size and convolution learning-rate multiplier chosen
  only by the preregistered sensitive-I/O development gates.
- Same seed, data order, crop size, loss weights, quantizers, rotation, and
  validation material across optimizer arms.

The final glyph-safe heldout remains closed.

## Arms

1. **QAT8 + periodic Muon control**
   - Continue the sensitive-I/O development winner into the preregistered
     output-convolution study.
   - After that study selects an output group, continue the exact selected
     output branch to step 30,000 with its optimizer moments and exact data
     cursor.
   - The first 14,000 steps came from the sealed parent before the input
     convolution began its transition, so the later group-size override does
     not alter an already-active input surface.
2. **AdamW transfer**
   - Start from the non-quantized FP32 teacher at step zero.
   - Use explicit AdamW beta values `(0.9, 0.999)`, epsilon `1e-8`, and the
     same sensitive matrix/convolution/correction learning-rate groups.
   - Do not initialize from a ternary or q4 checkpoint.

Each arm must have a sealed source snapshot, command, dataset receipts,
optimizer identity, per-surface group map, and checkpoint hashes.

The AdamW arm uses the same selected input and output group-size map from step
zero. The QAT8 arm may reuse its selected output-study trajectory because it is
the exact matched QAT8 training recipe, not an initialization shortcut from a
different optimizer or model. Exposure is compared at the same 30,000 global
steps and schedule boundaries.

## Checkpoint and development gates

1. Evaluate blended and hard-started diagnostics at matched schedule
   boundaries. Training-log loss is not selection evidence.
2. At step 30,000, require every one of the 204 surfaces to be fully hard and
   materialize one canonical ternary state per arm.
3. Evaluate both canonical states on the full development bundle with paired
   overall, edge, teacher, worst-decile, source, language, and script deltas.
4. Apply the existing full-surface development ranking order:
   worst-source PSNR delta, overall PSNR delta, edge PSNR delta, then teacher
   PSNR delta.
5. Advance only a development winner that satisfies the preregistered q4
   non-inferiority margins and artifact accounting. If neither arm passes,
   select neither and keep the final heldout closed.

This study chooses a training recipe only. It cannot promote a browser model.

## Required implementation receipts

- full-surface CLI accepts an explicit optimizer and per-surface group map;
- resume rejects optimizer or sensitive learning-rate drift;
- AdamW preserves distinct matrix, convolution, and correction groups;
- exact next-step AdamW resume parity passes;
- future study seals hash the CLI and full-surface optimizer implementation.
