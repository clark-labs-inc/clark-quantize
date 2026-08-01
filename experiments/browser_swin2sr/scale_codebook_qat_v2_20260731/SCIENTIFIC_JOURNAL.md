# Scientific journal

## 2026-07-31 — Sensitive bridge completed

### Representation

The arm uses one-bit learned scale codebooks for core linears, four-bit
codebooks for image-I/O convolutions, and two-bit codebooks for position-bias
linears and residual-body convolutions. All 204 surfaces completed their
gradual transition.

The exported artifact is 400,201 bytes: 261,332 binary bytes plus 138,869
manifest bytes. It stores 97.24036% of model parameters in the quantized trunk,
uses 1.600943 packed code bits and 0.081931 scale bits per quantized weight,
and reaches 2.079535 whole-model data bits per parameter. This is an honest
whole-model accounting result, not a `1.58-bit` label applied only to the
weight alphabet.

### Development quality

On all 3,040 development samples, the arm scores:

- ground-truth PSNR: 32.911420 dB;
- edge PSNR: 36.693982 dB;
- teacher PSNR: 38.005486 dB;
- gain over bicubic: 4.809826 dB.

The paired comparison with the exact selected ternary parent is -0.091062 dB
ground-truth PSNR, 95% interval `[-0.115804, -0.066733]`, while edge PSNR
improves by +0.117261 dB, interval `[0.093964, 0.140560]`. Teacher PSNR is
-0.129068 dB. The scale-codebook representation therefore preserves nearly
all parent reconstruction quality and improves edge fidelity at a much lower
scale-channel cost.

The paired comparison against the q4 report on the **same development
bundle** is -2.012226 dB ground-truth PSNR, interval
`[-2.081650, -1.943312]`, -1.739237 dB edge PSNR, and -8.927314 dB teacher
PSNR. A prior arithmetic subtraction using the published q4 aggregate mixed
different bundles and is rejected. Only this paired same-sample result is
valid for development interpretation.

### Decision status

This arm is a strong ultra-small functional model but is not a q4 replacement.
It remains in the sealed study until the uniform one-bit-scale arm completes;
no final-heldout evaluation or product promotion is authorized.

### Endpoint error localization

A deterministic endpoint analysis toggles only each trained scale-codebook
projection while keeping that surface's trained hard ternary weight fixed. It
therefore measures scale approximation, not output sensitivity.

- core linears contribute 60.95% of total squared scale error because they
  contain most weights; their aggregate relative weight RMSE is 11.22%;
- residual-body convolutions contribute 27.30% at 14.27% relative RMSE;
- position-bias linears contribute 11.52% at 8.76% relative RMSE;
- image I/O contributes only 0.22% at 6.20% relative RMSE, validating the
  four-bit image-I/O bridge as more precision than the other categories.

The four RSTB fusion convolutions are the top four convolutional error
surfaces and together contribute 23.44% of whole-model scale error. This is
independent support for testing them as a retained-precision island. It does
not prove causal output sensitivity; only the matched FP16-versus-ternary
study can do that. The analysis SHA-256 is
`844a74fa576e00ec1139e78888e718b01d3f45083a543807d6a2cc297034ee26`.

### Receipts

- checkpoint SHA-256:
  `31c0b3e8bbac123965512e37aec44181543a84b70943c88449e89171c90f08f9`
- materialization SHA-256:
  `e572a1b452b1aaa2b6c769f29ba2db120c8e6f920820fd8d558c06b57fa6034f`
- development report SHA-256:
  `707d403cc1c7b47bca299d54ceabf59a6fd569d807e108b0257eabda3e789740`
- parent comparison SHA-256:
  `94a58a08d9a8a1088cc337b1d3c7df6813bb0c30159992b6eae86e28f11f02a8`
- q4 comparison SHA-256:
  `6e6e7d95a845a6855f8505add11e4861920a262744eb30d2b93ba9330ff165b2`
- binary SHA-256:
  `5a7967bc6982310c9d9606c5f552485b07fc2992cde0dca6f57f8dce316b5fe7`
- manifest SHA-256:
  `f190ec6b1ae0b231e4b951cc89c5da6a16b44bbe190a94e5e4416fb0b5cc0b25`

## 2026-07-31 — Uniform one-bit scale endpoint completed and rejected

### Representation

The uniform arm completed all 4,000 preregistered steps with all 204 surfaces
fully projected into one-bit learned scale codebooks. The exported artifact is
395,666 bytes: 256,837 binary bytes plus 138,829 manifest bytes. It stores
97.24036% of model parameters in the quantized trunk, uses 1.600943 packed
ternary-code bits and 0.045867 scale bits per quantized weight, and reaches
2.043766 whole-model data bits per parameter. It passes both frozen size
conditions: 93.03405% smaller than the declared q4 artifact and 1.646810
packed trunk bits per quantized weight, below the 1.66-bit endpoint ceiling.

### Development quality

On the exact 3,040-sample development bundle, the materialized hard state
scores:

- ground-truth PSNR: 31.272979 dB;
- edge PSNR: 35.184447 dB;
- teacher PSNR: 34.981080 dB;
- gain over bicubic: 3.171385 dB.

The paired comparison against q4 is decisively negative:

- ground-truth PSNR: -3.650667 dB, 95% interval
  `[-3.763752, -3.543160]`;
- edge PSNR: -3.248772 dB, 95% interval
  `[-3.349586, -3.148330]`;
- teacher PSNR: -11.951719 dB, 95% interval
  `[-12.079056, -11.828389]`.

Both source strata and every script stratum fail their frozen
non-inferiority conditions. The uniform endpoint therefore meets the size and
bit-accounting goal but does not meet the quality goal. It is rejected as a
q4 replacement and is not eligible for final-heldout access, runtime
promotion, or product promotion.

### Study decision

The sealed decision ranks the arms `sensitive_bridge`, `image_io_bridge`,
then `uniform_q1`. No approximately-1.58-bit endpoint is eligible. The
sensitive bridge remains the next-stage parent because it is the highest
quality completed arm, not because it passes the q4 gate. The final heldout
set remains locked.

### Receipts

- checkpoint SHA-256:
  `ba57b22a9981c77aabfa3b1b2843987126ed229338acda0c0673d8445afc5901`
- materialization SHA-256:
  `cd595414c33853f36182cf17d8df1102d5c75a5558288b09e5e6585fd175f9fa`
- development report SHA-256:
  `a74fad4b451d24565f713ce96773e1b7fc816484aeec925f2ee637bf7ad0996e`
- binary SHA-256:
  `b6d1d71c89edbb7da707cbf61e001d80b65f385b25e136ed5b893eb545cbced2`
- manifest SHA-256:
  `713b152c439c1a3ce2b413cb815702350f169621959fb5af236414bdf14688aa`
- sealed decision SHA-256:
  `eac79dfc2e58aa8cc2267dee0a3daf368bc26e778d5157955f8f5f99e8611d2c`

## 2026-08-01 — Final disposition

The uniform one-bit scale endpoint remains rejected and no v2 codebook writer
is active. The sensitive bridge is retained as negative and mechanistic
evidence, not as a selected release. G3 supersedes this lane for product use;
any new scale-codebook experiment requires a fresh protocol from exact G3.
