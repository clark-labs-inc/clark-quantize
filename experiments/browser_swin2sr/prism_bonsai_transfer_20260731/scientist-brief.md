# Frozen Scientist brief: heterogeneous precision for tiny Swin2SR

Propose exactly one maximally discriminating next experiment for BackdropKit's
tiny browser Swin2SR. Do not propose a roadmap and do not treat this brief or
model reasoning as evidence.

## Objective

Reach q4-noninferior glyph-safe super-resolution quality with a materially
smaller, browser-native model. Preserve cross-browser WebGPU/WASM/CPU runtime
correctness. Start from the exact non-quantized teacher lineage and use gradual
training into the chosen low-bit representation.

## Current measured state

- q4 reference: 5,676,042 bytes and 34.813515 dB mean ground-truth PSNR.
- Best scientifically selected ternary checkpoint: refined surface-group g2,
  33.002482 dB, 36.576722 dB edge PSNR, and +4.900888 dB over bicubic.
- It is still -1.921164 dB paired mean versus q4; promotion is unauthorized.
- Swin2SR has 1,005,348 parameters. Deployment ternarizes 977,604 values
  (97.24036%) and retains 27,744 raw FP16 values.
- The retained magnitude channel dominates: selected ternary codes cost
  1.600943 bits/quantized weight and FP16 per-group scales cost 0.581831,
  totaling 2.182773 trunk bits/weight and 2.56408 whole-model bits/parameter.
- The two image-I/O convolutions contain only 8,100 weights but use group size
  2 because they are highly sensitive. Retaining all 8,100 as FP16 costs only
  16,200 raw bytes before container overhead.
- Abrupt post-training scale compression failed. The best static q1-core,
  q4-image-I/O, q2-position, q2-residual map reached 1.679699 trunk bits/weight
  but lost 1.93448 dB versus the refined FP16-scale state.
- A gradual scale-codebook QAT study is active; do not duplicate it or inspect
  final heldout data.

## Exact Prisma evidence

- PrismML's Ternary Bonsai Image packs 94.96734% of 3.876B transformer
  parameters into ternary linears and retains 5.03266% in BF16.
- The BF16 island is role-specific: modulation streams, context/time/input
  embedders, final norm/projection, and attention norms.
- Ternary codes use group size 128. Activations remain BF16 and sensitive
  accumulation remains FP32.
- The released Gemlite checkpoint is 3.179852 effective bits/logical parameter
  after runtime packing and BF16 islands, not 1.58 whole-model bits.
- All FP32 zero-points equal negative scales; every scale is exactly
  BF16/FP16-representable. A derived-zero/BF16-scale format is lossless for the
  released checkpoint and would reduce it to 2.823293 bits/parameter.
- The empirical ternary entropy is 1.582676 bits/weight and the fourth INT2 code
  is unused, so code entropy offers almost no gain below log2(3).
- PrismML does not disclose a reproducible training curriculum in the released
  whitepaper. Do not invent one.

## Frozen scientific boundaries

- Use only development data for selection; final heldout remains locked.
- Compare exact paired samples with 10,000 bootstrap resamples.
- q4 gates: ground-truth noninferiority -0.10 dB, edge -0.15 dB, teacher -0.25
  dB, plus frozen p10, source, script, and language gates.
- Candidate artifact must be below 50% of q4 bytes.
- Minimum three seeded replications for a promotion-bearing conclusion.
- Preserve source, dataset, objective, optimizer, sampler, checkpoint, and
  runtime provenance. Change only the preregistered precision intervention.
- Separate hypothesis generation from deterministic execution and evaluation.

## Desired proposal shape

Choose one intervention that distinguishes among these explanations:

1. the image-I/O signs themselves need higher precision;
2. ternary signs are adequate but their magnitude side channel must remain
   dense;
3. a different small structural island—not simply first/last layers—causes the
   remaining q4 gap.

Return an executable falsification test, expected effect, exact intervention
object, at least three replications, assumptions, limitations, and protected
metric risks.
