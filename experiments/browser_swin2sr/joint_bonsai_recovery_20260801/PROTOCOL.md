# Joint Bonsai Recovery Protocol

## Question

Can the four RSTB fusion convolutions recover quality when trained jointly on
top of the released mixed-precision ternary Swin2SR model, while the released
image-I/O FP16 island and every other hard-ternary surface remain unchanged?

## Immutable parent

- Checkpoint: `../prism_bonsai_transfer_20260731/retained_precision_study_v2_20260731/image_io_fp16__fp16__seed20260733/run/checkpoint-island-4000.pt`
- SHA-256: `cb5455bba3154a46eb9f744a6b5aee03497c0807413e632fd7eea070a21a1c87`
- Released development score: `34.098155400` dB on 3,040 locked samples
- Released structure: 202 ternary matrix/conv tensors; first and final
  image-I/O convolutions retained in FP16; 96.4347% ternary by parameters
- Released native browser package: 330,933 bytes

The loader must reproduce the released dense-hard state exactly before any
new optimization. The preflight receipt is zero tensor differences across 484
tensors and `0.0` maximum absolute error.

## Preregistered discovery arm

- Island: `rstb_fusion_fp16`
- Modules: `swin2sr.encoder.stages.{0,1,2,3}.conv`
- Trainable values: 129,600
- Precision during recovery: FP16 round-trip STE
- Seed: 20260801
- Optimizer: AdamW
- Learning rate: 4e-6
- Weight decay: 1e-4
- Gradient clip: 1.0
- Steps: 4,000
- Physical batch: 8
- Crop: 128
- Validation/checkpoint interval: 500 steps
- Device: MPS
- Training corpus: 206,500 balanced product/UI/text samples, including 204
  language buckets

Only the declared RSTB weights are trainable. The inherited image-I/O FP16
island must remain bit-identical unless a later experiment explicitly tests it.

## Causal control

If the FP16 arm has finite gradients and improves over its start by step 500,
run the same seed/data order/objective/optimizer with the RSTB island retained
as hard ternary. This matched arm isolates the effect of the precision island.
No second expensive arm is started when the discovery trajectory is already
non-viable.

## Compression boundary

FP16 is a recovery instrument, not the intended transport. The final prior
boundary decision selected int5-per-kernel as the smallest tested RSTB
representation retaining at least 90% of every measured FP16 effect. It used
55,080 incremental bytes over ternary and retained 96.5% of ground-truth gain,
97.6% of edge gain, and 90.5% of teacher gain. A promising checkpoint therefore
advances first to that confirmed boundary; int7-per-output is a higher-fidelity
fallback, not the storage-optimal prior winner.

## Gates

The candidate is not promotable merely because a small validation set rises.
It must:

1. exceed `34.098155400` dB on the complete locked 3,040-sample development
   bundle;
2. have no harmful aggregate edge, text, or language/script regression;
3. survive materialization and reload without tensor drift;
4. satisfy the browser package, execution-parity, memory, and latency gates;
5. remain within a justified storage budget after RSTB compression; and
6. leave the final heldout set unopened until one candidate is frozen.

Failure on the complete development gate rejects the branch. Results are never
composed in weight space from separately trained islands; that approach already
failed because the gains were non-additive.
