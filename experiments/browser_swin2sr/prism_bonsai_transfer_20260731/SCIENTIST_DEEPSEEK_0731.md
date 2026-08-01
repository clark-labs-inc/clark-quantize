# Clark Scientist receipt: DeepSeek Flash 07/31

## Authority and execution

- Provider: OpenRouter
- Model: `deepseek/deepseek-v4-flash-0731`
- Clark Scientist mode: one read-only `model.reason` effect
- File tools: none
- Evaluator tools: none
- Training authority: none
- Attempts: 1
- Recoveries: 0
- Input tokens: 3,520
- Output tokens: 20,169
- Recorded cost: USD 0.00614012
- Changed paths reported by the run: none
- Completed or failed tools reported by the run: none

The proposal is a hypothesis-generation artifact, not scientific evidence.

## Frozen run artifacts

- Run root:
  `/Users/stan/Documents/git/clark-scientist/runs/20260731T180026Z-deepseek-swin-scientist`
- `proposal.json`
  - SHA-256: `11c2ea03c1b697873ee65102fb3c176e26c92ed0b60239c347e900064a509ba5`
- `responses.jsonl`
  - SHA-256: `920f4352d4d62b8027045f9085f48e02e14c2875322ec9c93af7963a5e2a9076`
- `worker.json`
  - SHA-256: `16b46f9d26f0e17bde51e044a7a8e8e9db008f937b561d49d9f4a183ee5642e2`

## Proposed discriminating hypothesis

With the refined dense FP16 scale channel held fixed, a gradually trained q4
island on the four intra-RSTB residual-fusion convolutions should recover at
least 0.1 dB of the paired lower-95 q4 gap, while the same treatment on the two
image-I/O convolutions should not. The proposal calls for three matched seeds,
identical data order and optimization, deterministic paired evaluation, and no
final-heldout access before artifact precommit.

## Deterministic verification of model assumptions

The exact pinned Swin2SR teacher resolves the proposed RSTB island to:

- `swin2sr.encoder.stages.0.conv`: 32,400 weights
- `swin2sr.encoder.stages.1.conv`: 32,400 weights
- `swin2sr.encoder.stages.2.conv`: 32,400 weights
- `swin2sr.encoder.stages.3.conv`: 32,400 weights

Total proposed RSTB q4 island: 129,600 weights. The image-I/O control contains
8,100 weights. The broader `residual_body_conv` category contains 180,000
weights, so the experiment must use the four explicit module paths rather than
that category label.

The shipped q4 reference does not define q4 convolutions. Its ONNX graph has
144 `MatMulNBits` linear operators with four-bit, block-size-32 weights, while
all 12 convolutions remain floating point. The current native Clark artifact
supports raw FP16 and packed ternary tensors, not packed q4 tensors. The
proposal's q4-convolution arm therefore has no existing deployment geometry
and is rejected as written rather than silently inventing one.

The executable first test is the same exact tensor-location contrast using raw
FP16 islands, which the native artifact already supports. The four RSTB
convolutions contain 259,200 FP16 bytes. Their current kernel-grouped ternary
representation uses 25,920 code bytes plus 28,800 FP16 scale bytes, so the
replacement adds 204,480 data bytes before alignment and metadata. This still
fits comfortably below the frozen 2,838,021-byte ceiling. If the FP16 RSTB arm
earns a material quality effect, q8/q4 compression of that proven island can be
preregistered as a separate representation and runtime experiment.

## Scientific disposition

The tensor-location hypothesis is accepted as a lead, not a finding. The q4
intervention is rejected as underspecified and unsupported by the current
convolution runtime. The FP16 version does not duplicate the active
scale-codebook study because it changes the stored precision of an explicit
weight island while leaving the remaining scales unchanged. It must not launch
until its tensor manifest, continuous-master hash, three seeds, optimizer
contract, and futility boundary are frozen. Prior continuous
compensation-island studies are related evidence but are not the same
intervention as retaining those weights in FP16 at deployment.
