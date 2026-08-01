# Scientific journal

## 2026-07-31 — Exact Bonsai Image transformer inventory

### Question

Does PrismML's released Ternary Bonsai Image support the hypothesis that an
honest low-bit image model should retain a small, role-specific high-precision
island rather than forcing every tensor into one precision?

### Frozen upstream identity

- Repository: `prism-ml/bonsai-image-ternary-4B-gemlite-2bit`
- Revision: `bf047aa80569f193d2c4960033af6ba2629988fa`
- Transformer file: `transformer-gemlite-int2/state_dict.pt`
- File bytes: 1,540,457,482
- SHA-256: `a3a7df8a90374fea24afce3b36f00b4c728d0254717143d61f912a7b3070e7ac`
- Loading: `torch.load(..., weights_only=True, mmap=True)`; no model execution

### Verified representation

- The checkpoint contains 100 packed matrix-heavy linears covering
  3,680,501,760 logical parameters.
- It retains 195,042,816 logical parameters in BF16. That is 5.03266% of the
  3,875,544,576 logical transformer parameters.
- Retained BF16 storage is 390,085,632 bytes. The three modulation matrices
  consume 283,115,520 bytes, or 72.58% of that island.
- Other material BF16 islands are `context_embedder` (47,185,920 bytes),
  `norm_out` (37,748,736), and `time_guidance_embed` (20,447,232).
- The packed ternary histogram is 1,177,251,744 code 0; 1,325,136,885 code 1;
  1,178,113,131 code 2; and zero code 3. Its empirical Shannon entropy is
  1.582676 bits/weight, very close to `log2(3)` because the states are nearly
  balanced.
- Packed values occupy two-bit slots. The fourth value is unused.
- The 28,753,920 group scales and zero-points are both stored as FP32 in the
  Gemlite checkpoint. Every zero-point is exactly the negative of its scale.
  Every scale round-trips exactly through BF16 and FP16.

### Honest storage accounting

- Exact checkpoint: 3.179852 bits per logical parameter.
- Exact tensor payload: 3.179420 bits per logical parameter.
- INT2 codes plus derived zero-points and BF16 scales, preserving the BF16
  island: 2.823293 bits per logical parameter. This is a lossless format change
  for the released values, not a quality claim about a different model.
- Ideal ternary entropy plus BF16 scales, preserving the BF16 island:
  2.429131 bits per logical parameter.
- The `1.58-bit` label correctly describes the ternary alphabet, not the whole
  deployed transformer. Whole-model claims must include high-precision islands,
  scales, metadata, packing, and runtime alignment.

### Comparison with current Swin2SR

- Swin2SR has 1,005,348 parameters. The selected refined state ternarizes
  977,604 values across 204 weight tensors and retains 27,744 raw values for
  FP16 deployment, giving 97.24036% ternary coverage.
- Unlike Bonsai Image, Swin's dominant retained precision is in dense FP16
  per-group scales. Its sensitive image I/O tensors use group size 2, which is
  effectively a dense magnitude side channel even though their signs remain
  ternary.
- The selected refined FP16-scale artifact costs 1.600943 packed code bits and
  0.581831 scale bits per quantized weight: 2.182773 trunk bits/weight and
  2.56408 whole-model data bits/parameter.
- A static scale-codebook map reached 1.679699 trunk bits/weight but regressed
  1.93448 dB versus the refined FP16-scale state. This rejects abrupt
  post-training scale collapse, not gradual heterogeneous training.

### Interpretation

The transferable Bonsai lesson is not its exact exception list. FLUX
modulation streams have no direct Swin equivalent. The transferable pattern is
to separate three precision decisions:

1. weight alphabet: ternary or retained FP16/BF16;
2. magnitude channel: group density and scale/codebook precision;
3. arithmetic: stored weights versus activation and accumulation precision.

For Swin2SR, promoting the 8,100 image-I/O weights to FP16 would cost only
16,200 raw bytes before container overhead. This should be tested directly
against the current group-2 ternary representation because the size delta is
small and those surfaces are already known to be sensitive. It must not be
assumed beneficial without a paired development evaluation.

### Limitations

- PrismML's whitepaper documents representation and deployment, but does not
  disclose enough training detail to reproduce its ternarization curriculum.
- Weight inspection cannot establish why an upstream tensor was retained or
  whether a different low-bit representation would preserve quality.
- The released Gemlite storage overhead is runtime-specific; it is not a lower
  bound for a browser-native format.
- No Bonsai result is evidence about Swin2SR quality until tested through the
  sealed glyph-safe evaluator.

### Next discriminating gate

Preregister a matched heterogeneous-precision study from one exact continuous
master checkpoint. Compare the current group-2 ternary image-I/O path against
FP16 image-I/O islands and progressively trained q8/q4 alternatives while
holding the core ternary map, data order, optimizer state, objective, and
evaluation bundle fixed. The final heldout remains locked.

## 2026-07-31 — Bounded Clark Scientist hypothesis pass

### Receipt

One read-only Clark Scientist pass used
`deepseek/deepseek-v4-flash-0731` through OpenRouter. It consumed 3,520 input
tokens and 20,169 output tokens at a recorded cost of USD 0.00614012. It called
no tools, changed no paths, and had no training or evaluator authority. The raw
proposal SHA-256 is
`11c2ea03c1b697873ee65102fb3c176e26c92ed0b60239c347e900064a509ba5`.

### Proposal and verification

The model proposed a matched q4-island contrast: the four intra-RSTB
residual-fusion convolutions versus the two image-I/O convolutions, with dense
scales held byte-identical and three matched seeds. Direct inspection of the
pinned Swin2SR resolves the RSTB set to four 32,400-weight tensors, or 129,600
weights total. This is narrower than the 180,000-weight
`residual_body_conv` category and must be expressed by exact module path.

Direct ONNX inspection shows that the shipped q4 reference applies four-bit,
block-size-32 `MatMulNBits` to 144 linear operators and leaves all convolutions
floating point. It therefore supplies no q4-convolution geometry. The model's
q4 arm is rejected as written rather than inventing a representation after the
hypothesis was proposed.

The first executable location test will retain the candidate island in raw
FP16, which the native artifact already supports. The RSTB island costs 259,200
FP16 bytes versus 25,920 current ternary code bytes and 28,800 kernel-grouped
FP16 scale bytes, a 204,480-byte data delta before alignment and metadata. The
frozen 2,838,021-byte ceiling is therefore not the likely failure mode. Exact
packing remains a protected metric.

### Scientific interpretation

The tensor-location hypothesis is plausible and non-duplicative with the active
scale-codebook study, but it is not a finding. The FP16 location contrast must
first establish that retained precision in the RSTB fusion tensors has causal
value. Only then should q8/q4 compression and a new browser runtime path be
considered. The preregistration must also distinguish retained deployment
precision from earlier continuous compensation-island training studies whose
final weights were still materialized as ternary.

## 2026-07-31 — Continuous-master restoration falsified

### Question

Can a retained FP16 island be created by directly restoring the latent
continuous masters behind the selected hard-ternary checkpoint?

### Development-only screen

Three states were evaluated on the same first 100 records of the sealed
development bundle. Every tensor outside the declared island was identical to
the canonical hard state. The final heldout remained locked.

| State | PSNR vs ground truth | Edge PSNR | PSNR vs teacher |
| --- | ---: | ---: | ---: |
| canonical hard ternary | 31.918438 | 35.643640 | 35.511217 |
| latent-master image I/O in FP16 | 29.438515 | 34.373244 | 31.628472 |
| latent-master RSTB fusion in FP16 | 30.291923 | 34.401878 | 32.485400 |

Against the hard state, the image-I/O probe lost 2.479923 dB ground-truth
PSNR with paired-bootstrap 95% interval `[-2.785376, -2.193796]` and zero
wins. The RSTB probe lost 1.626516 dB with interval
`[-1.877754, -1.383490]` and a 1% win rate. Its smaller loss than image I/O
does not make it usable: both intervals decisively exclude parity.

### Finding

The continuous STE masters are optimization state, not deployable weights.
Directly restoring them creates a large representation discontinuity and is
rejected. This result does **not** show that retained FP16 precision is harmful;
it shows that FP16 training must begin from the deployed hard weight, not from
the latent master. The bounded DeepSeek proposal's continuous-master
initialization is therefore falsified while its tensor-location hypothesis
remains open.

## 2026-07-31 — Matched retained-precision study sealed

### Corrected initialization

The selected tensor is first projected to the exact hard ternary value. An
FP16 arm copies that value into a new master and applies an IEEE-FP16
roundtrip STE on every forward pass. Its initialization receipt proves zero
difference from the hard value after the same FP16 roundtrip. A ternary arm
copies the same hard value into the existing ternary master and retains the
hard ternary projection within a recorded four-epsilon FP32 roundoff bound.
Only the declared island weights require gradients;
biases, normalization parameters, and every other surface are frozen.

### Causal control and execution rule

For each island and seed, FP16 is paired with a same-location ternary control.
Both arms use the same 206,500-image source/language-balanced corpus, data
order, augmentation seed, 4,000 steps, balanced loss, PSNR-aligned pixel
objective, AdamW, 4e-6 learning rate, and 1e-4 weight decay. This control is
required because comparing two FP16 locations alone would confound precision
with which tensors received additional training.

AdamW is not an untested default here. The earlier nine-optimizer,
three-learning-rate diagnostic tournament promoted four recipes into three
matched replicate seeds. AdamW at the promoted learning-rate multiplier had
the best aggregate ground-truth and teacher PSNR; Mirror SGD, Factored
Adafactor, and Bonsai QAT8 plus Muon did not establish non-inferiority under
the frozen replicate rule. In the later 3,040-sample transfer study, AdamW
also beat Bonsai QAT8 by +0.412509 dB ground-truth PSNR, 95% interval
`[0.388157, 0.438157]`, although neither arm passed q4 quality. Those results
justify holding AdamW fixed for the location experiment; they do not rule out
a separately preregistered optimizer study on a location that first proves a
causal precision benefit. The optimizer replicate decision SHA-256 is
`4e7886819f59e96690afca6ffa00bc0daa30442cf95df26e0d86e418f123b84e`;
the transfer decision SHA-256 is
`98d28ce8a1ec43c8d6a592ae341b8f06276e1c769f69d199c028c11ffbc1a7ca`.

Seed `20260731` is discovery. Seeds `20260732` and `20260733` are gated
confirmations: they run only if FP16 minus ternary has a positive lower 95%
paired-bootstrap bound for ground-truth PSNR and non-negative mean edge and
teacher deltas in discovery. Optimizer variants remain a separate later study
so precision and optimizer effects are not mixed.

The initial v1 seal failed its immutable RSTB control inspection before
training because optimal-MSE reprojection changed a weight by
`7.450580596923828e-09`. That seal and failure receipt are preserved. The
corrected preregistered manifest is
`retained_precision_study_v2_20260731/study-manifest.json`, SHA-256
`7e164fdc5e5f69ea806e24cf0fe9c919849a9189ede853d861e7bcd4e6c453b8`.
It freezes four discovery arms, eight gated confirmation arms, immutable
source overlays, parent identity, data identities, q4 reference, development
bundle, and final-heldout prohibition before training results exist. Eight
targeted unit tests pass. All four immutable real-model inspections pass; the
FP16 arms have zero post-roundtrip initialization delta and the RSTB ternary
arms record the 7.45e-9 delta against a 4.77e-7 bound. A one-step sealed smoke
also produced finite gradients for exactly 8,100 values, reloaded and
materialized 484 tensors, and resumed exactly from an interrupted checkpoint.

Before any discovery comparison existed, the confirmation gate was also
materialized as executable code and sealed in
`retained_precision_study_v2_20260731/discovery-decision-protocol.json`,
SHA-256
`3356b25679b39517c1f1afd7c71e66353071480b226e94fdab9a8828523276ed`.
It verifies the paired report identities and all 3,040 sample counts, then
authorizes only the already preregistered confirmation arms when the
ground-truth lower 95% bound is strictly positive and both edge and teacher
mean deltas are non-negative. Five targeted tests pass. The decision cannot
open final heldout or authorize model promotion.

The final three-seed rule was likewise sealed before discovery results in
`retained_precision_study_v2_20260731/confirmation-decision-protocol.json`,
SHA-256
`e93d7be8cfeadc6523f2fe79f0f4278680118594e56f35a5fa66da51531af632`.
An island is development-supported only if every one of seeds `20260731`,
`20260732`, and `20260733` has a strictly positive ground-truth lower 95%
bound and non-negative edge and teacher means. If both islands pass, ranking
uses their three-seed mean ground-truth, teacher, and edge deltas in that
order. Three confirmation tests pass; eight discovery-plus-confirmation gate
tests pass together. A surviving island is authorized only for a separately
preregistered compression/runtime study, never direct promotion.

### Broad-convolution follow-up, not yet authorized

The shipped q4 ONNX leaves all 12 convolution operators floating point while
quantizing 144 matrix multiplications. Swin's current ternary model covers
188,100 convolution weights: 8,100 image-I/O, 129,600 RSTB fusion, 32,400
post-body fusion, and 18,000 patch-embedding weights. Retaining all of them in
FP16 would replace 37,620 packed ternary-code bytes plus 44,700 FP16-scale
bytes with 376,200 weight bytes, a net data increase of 293,880 bytes. Even
added to the 400,201-byte sensitive-codebook artifact, the rough result remains
well below the 2,838,021-byte promotion ceiling.

This makes all-convolution FP16 a plausible follow-up if the narrower RSTB
island is positive or if it localizes insufficient recovery. It is deliberately
not added to the already sealed discovery, because changing the hypothesis set
after seeing training results would invalidate the preregistration. A later
study must pair it with a same-location ternary control and account for the
mixed-precision browser kernel before promotion.

## 2026-07-31 — Image-I/O ternary control result; parent attribution invalidated

The discovery-seed image-I/O ternary control completed its sealed 4,000-step
run and exact 3,040-sample development evaluation. It trains only the 8,100
weights in `swin2sr.first_convolution` and `upsample.conv`; every other tensor
remains frozen. The hard materialized state scores 33.997176 dB ground-truth
PSNR, 37.631534 dB edge PSNR, 40.706004 dB teacher PSNR, and gains 5.895582
dB over bicubic.

An initial post-hoc comparison against the earlier refined-parent report
produced the following apparent improvements:

- ground-truth PSNR by +0.994693 dB, 95% interval
  `[0.952582, 1.038797]`, with a 96.94% win fraction;
- edge PSNR by +1.054813 dB, 95% interval `[1.006281, 1.104253]`, with a
  97.66% win fraction;
- teacher PSNR by +2.571450 dB, 95% interval `[2.512125, 2.635019]`, with a
  99.87% win fraction.

Against q4, the remaining paired gaps are -0.926470 dB ground truth,
-0.801685 dB edges, and -6.226795 dB teacher. These q4 comparisons are
descriptive and do not alter the sealed FP16-versus-ternary gate.

That causal interpretation is invalid. Exact-state inspection later proved the
parent report used an MPS hard projection while the retained-precision arm used
a CPU hard projection; 200 tensors and 218,256 values differed outside the
trained island. The numeric comparison remains a receipt of those two states,
but it cannot attribute the apparent gain to focused image-I/O training. The
under-recovery claim is withdrawn pending evaluation of the resealed,
same-device CPU parent. The matched FP16-versus-ternary comparison remains
valid because those two arms use the identical CPU materialization path.

### Receipts

- checkpoint SHA-256:
  `a154b6a9bf94c90a0a68397d3eb5bace29b0240224937da6caa9f34a946464ea`
- materialization SHA-256:
  `64eca69e010b64503fdf2447a4db0957739f44f1219e828a174a06e30ed26355`
- development report SHA-256:
  `efb0504682aed61a12323b62c3449bf6219aee479cc048695a2a37aa6e4dd43e`
- q4 descriptive comparison SHA-256:
  `0d69d2a7ffc8c44005a2d3c7ba533cb1501c7f4d21a293261be8a716f82f97c6`
- cross-device-confounded refined-parent comparison SHA-256:
  `7d574c983bbe9902da935ef7d05d10c6683920afd3e884582678c29a198ac196`

## 2026-07-31 — FTerViT transfer boundary

### Verified upstream method

FTerViT (`arXiv:2605.21171v1`) fully ternarizes a DeiT-family classifier's
matrix weights, patch-embedding convolution, and LayerNorm affine parameters.
Its patch convolution uses one absmean scale per output channel. Biases and
LayerNorm statistics remain FP32. Training initializes the ternary student from
the FP32 teacher, uses same-architecture quantization-aware distillation, then
restarts AdamW at a lower learning rate for a short recovery phase. The reported
ImageNet-1K result is strong compression with a remaining 2.42 percentage-point
gap at 384-pixel resolution and a 3.44-point gap at 224 pixels; it is not a
lossless-ternarization result.

Primary source: https://arxiv.org/abs/2605.21171v1

### What transfers to Swin2SR

1. A low-learning-rate recovery restart after a genuinely saturated hard model
   is a credible later intervention. It should be compared with an equal-step,
   no-restart continuation so restart and extra compute are not confounded.
2. Taylor first-order and Hessian-trace component importance can serve as an
   independent sensitivity diagnostic alongside the preregistered output-
   fragility allocation. It cannot replace paired output evaluation.
3. Keeping numerically destructive statistics and small correction parameters
   out of the ternary weight alphabet is compatible with honest whole-artifact
   accounting; their bytes must still be counted.

### What does not transfer directly

FTerViT's one-scale-per-output-channel convolution geometry is coarser than the
current Swin2SR kernel grouping. For a 3x3 convolution, Swin stores one scale
per nine-weight input/output kernel; FTerViT stores one scale across all input
channels and spatial weights of an output filter. Adopting the latter might
save scale bytes, but it is not a quality-recovery mechanism and must not replace
the current geometry without a matched storage-quality experiment.

The classification-only KL objective also does not transfer to super-resolution
pixels. The existing frozen-teacher, ground-truth, edge, and feature objective
is task-specific and remains the active contract. No FTerViT-inspired arm is
added to the already sealed retained-precision study after the fact.

## 2026-07-31 — Preregistered disjoint hard-ternary composition

The image-I/O hard-ternary control initially appeared to recover roughly one dB
over the refined full-surface parent. The later cross-device audit invalidated
that attribution, so it is now only a hypothesis: independently focused recovery
of disjoint sensitive tensor groups may be composable. The same-device CPU
parent and five frozen comparisons below are the test of that hypothesis.

Before the RSTB-fusion discovery arm started, I sealed a deterministic union of
the image-I/O control's two tensors and the RSTB-fusion control's four tensors.
The composer checks that both sources are hard-ternary arms with the same seed
and parent, verifies their file and tensor hashes, and rejects either source if
it changed any tensor outside its declared island. It starts from the canonical
parent and copies only the six declared tensors. Therefore it adds no parameters,
FP16 islands, scale tables, or bytes.

An exact preflight exposed that the prior canonical parent had been hard-projected
on MPS while all retained-precision arms are materialized on CPU. Those two
projection paths differed in 200 tensors and 218,256 exact values even though the
underlying non-island masters were frozen. The contamination check correctly
rejected this mismatch. I therefore materialized the unchanged parent through
the same sealed CPU path used by the arms and resealed against that state. The
image-I/O source then differed in exactly its two declared tensors and 6,081
values, with no undeclared changes. The CPU parent must receive its own 3,040-
sample report, and the decision script requires that report's state hash to match
the sealed parent file.

The candidate can become the next *ternary research parent* only if all five
paired 3,040-sample development comparisons pass the same frozen gate:

1. image-I/O control versus parent;
2. RSTB-fusion control versus parent;
3. composition versus parent;
4. composition versus image-I/O alone; and
5. composition versus RSTB-fusion alone.

Each comparison requires a strictly positive lower 95% paired-bootstrap bound
for ground-truth PSNR plus nonnegative mean edge and teacher PSNR. Failure of any
one rejects the composition. The protocol cannot authorize final-heldout access
or product promotion.

### Seal receipts

- protocol:
  `retained_precision_study_v2_20260731/disjoint-ternary-composition-protocol.json`
- protocol SHA-256:
  `e717b3c6bf343e152b9f1e032bde13860c4cbd6b532d2e674c82971a63026628`
- composer SHA-256:
  `b039c9e38f16bc01eec5840d4160803712925b93afaa13fcc1733d7da9c5306b`
- decision script SHA-256:
  `29d4b26a55c664e9b09c175367164117eedd351c5a4cc1286843c4069f987158`
- test result: seven passed
- same-device CPU parent state SHA-256:
  `e8de7e5abd67022462064dfc5c9c841b511bc6bcc8c5ee4051acde1d62863e6b`
- timing receipt: created `2026-07-31T20:13:45Z`, hash-enforced at
  `2026-07-31T20:20:11Z`, and same-device resealed at
  `2026-07-31T20:34:00Z`, with no RSTB discovery training log present at
  any of those points.

The sealed composer subsequently accepted both completed sources. Relative to
the same-device CPU parent, the image source changes exactly its two declared
tensors, the RSTB source changes exactly its four declared tensors, and the
composition changes exactly their six-name union. The composed dense state has
file SHA-256
`4aba8a395a42f0496aebf4f37fe9a1068d28b26f68cb3be09a18133e66654952`
and composition-receipt SHA-256
`e66be324dc2e7741922ec6a037db1bbee0791d199adab85aa54a9c86783e1fb2`.
It is ready for evaluation after the active MPS training queue ends.

## 2026-07-31 — Image-I/O FP16 discovery passes

The matched image-I/O FP16 arm completed 4,000 steps from an exact FP16
roundtrip of the hard-started weights. Its 3,040-sample development result is:

- ground-truth PSNR: `34.094595681` dB;
- edge PSNR: `37.719748795` dB;
- teacher PSNR: `40.992753114` dB; and
- gain over bicubic: `5.993001574` dB.

Against the same-seed, same-data-order, same-optimizer hard-ternary control,
FP16 improves:

- ground-truth PSNR by `+0.097419916` dB, paired-bootstrap 95% CI
  `[+0.093602237, +0.101352265]`, with `89.64%` sample wins;
- edge PSNR by `+0.088214700` dB, with `91.12%` wins; and
- teacher PSNR by `+0.286748790` dB, with `99.28%` wins.

Every preregistered discovery check passes. This authorizes the two sealed
confirmation seeds for `image_io_fp16`; it does not authorize promotion or
final-heldout access.

Only the matched attribution is currently valid: retaining the 8,100 trained
image-I/O weights in FP16 contributes `+0.097419916` dB over the same-seed
hard-ternary control, with an extremely tight positive interval. The apparent
`+1.092113292` dB versus the older refined-parent report and the ternary
control's apparent `+0.994693376` dB share are cross-device-confounded and must
not be interpreted as optimization-allocation gains. The resealed CPU parent
will determine that contribution. The valid FP16 increment costs only
`+6,480` projected bytes.

The candidate remains below the shipped q4 baseline by `-0.829050372` dB
ground-truth, `-0.713470296` dB edge, and `-5.940046442` dB teacher PSNR.
Closing that residual gap still requires additional independent gains.

### Receipts

- checkpoint SHA-256:
  `2aa37e2ee7795a16e7569e9e6da066c7e26d217566e23c84e676d52f646ed983`
- materialization SHA-256:
  `134cb4203a3e64d5ddb2e9491b0b07e080bc5302d0c70bfa66c10f28c8902b56`
- dense-state file SHA-256:
  `1e0580308d98244d7418c316f034125ebc4590f8927f20623dc38ccf587ab820`
- development report SHA-256:
  `a5c5d81aef86060dad7513fd5bf0facad6e796f192070ebcbf34658915c0e02c`
- sealed FP16-versus-ternary comparison SHA-256:
  `7cb852b0834f8273cce4d89c719b488e9e39aa0672a0864a76890b101ee1434d`
- q4 descriptive comparison SHA-256:
  `bce0575ba4e967b9b3ab0516cd52afb4ed0ca7f3574c76430a7780760f438e73`
- cross-device-confounded refined-parent comparison SHA-256:
  `f3da8aef0c22e1a8f19c812e0c4a2ffd44d0862b95c3d6ecfcea08705019e704`

## 2026-07-31 — Preregistered image-I/O FP16 sub-tensor ablation

After the joint image-I/O FP16 discovery passed, but before evaluating either
sub-tensor, I froze two deterministic ablations. Both begin from the matched
hard-ternary control. `input_only` copies only the trained FP16
`swin2sr.first_convolution.weight` (1,620 values); `reconstruction_only` copies
only `upsample.conv.weight` (6,480 values). Exact-state checks prove that the
paired FP16 and ternary arms differ in only these two tensors and that each
ablation changes exactly one.

With the sealed group-two ternary geometry, the projected data-byte increments
are:

- input only: `+1,296` bytes;
- reconstruction only: `+5,184` bytes; and
- both tensors: `+6,480` bytes.

Across the exact 1,005,348-parameter model, those increments are only
`+0.010313`, `+0.041251`, and `+0.051564` whole-model data bits per parameter,
respectively.

A single tensor may replace the joint island only if its paired ground-truth
lower 95% bound is strictly positive, edge and teacher means are nonnegative,
and it preserves at least 90% of the joint gain on *each* of ground-truth,
edge, and teacher PSNR. If both pass, the smaller input-only island wins. If
neither passes, the joint island remains the research candidate. This is a
development-only attribution and cannot authorize promotion or final-heldout
access.

### Seal receipts

- protocol:
  `image_io_fp16_ablation_20260731/protocol.json`
- protocol SHA-256:
  `f5288764387e22ca30bb93c3551042a1d446b39af78daf1fb35a21c3548205e1`
- materializer SHA-256:
  `e5b2900641a027af13ef9a12a15340e96126ccfb128b567378211ccb37bbd764`
- decision script SHA-256:
  `828625ea3b8fc205d3714dd6b4972fcc4caac6b0b74be71b1d0500060cd49e50`
- tests: eight passed
- timing receipt: created `2026-07-31T20:44:46Z`, with the joint source
  reports and comparison hash-pinned and resealed at `2026-07-31T20:56:16Z`;
  neither single-tensor report existed at either point.

## 2026-07-31 — RSTB ternary control and preregistered mixed precision

The same-seed RSTB-fusion hard-ternary control completed 4,000 steps and its
3,040-sample development result is:

- ground-truth PSNR: `33.406174589` dB;
- edge PSNR: `36.968834375` dB;
- teacher PSNR: `39.210346068` dB; and
- gain over bicubic: `5.304580482` dB.

It remains meaningfully below the shipped browser q4 baseline: `-1.517471464`
dB ground-truth, `-1.464384716` dB edge, and `-7.722453488` dB teacher PSNR.
The control is still functionally useful, but it identifies the RSTB-fusion
convolutions as a plausible source of the remaining q4 gap.

Direct inspection of the exact 5,675,089-byte browser q4 ONNX explains why
this comparison matters. The graph has 144 `MatMulNBits` nodes using four-bit,
block-32 weights, but all 12 convolution nodes retain float32 weights. Those
convolutions occupy 752,400 bytes, including 518,400 bytes for the four RSTB
fusion convolutions alone. In other words, the shipped artifact called q4 is
heterogeneous: its transformer matrices are q4 while its sensitive
convolutions are FP32. This independently agrees with the DeepSeek 07/31
scientist's recommendation to treat the RSTB convolutions as a precision
island, but it also establishes a much smaller experiment than simply copying
the shipped format.

The FP16 RSTB arm showed a large early matched-training signal at step 500:
`37.224017913` dB ground-truth and `39.668051332` dB teacher PSNR, versus
`36.179832341` and `38.265706185` for ternary at the same step. This is not a
selection result. Before the FP16 3,040-sample report existed, I sealed three
deterministic post-training compression arms:

- signed int4 with one FP16 scale per output channel: `+10,560` bytes over
  the current ternary RSTB representation;
- signed int4 with one FP16 scale per 3x3 input/output kernel: `+38,880`
  bytes; and
- signed int8 with one FP16 scale per output channel: `+75,360` bytes.

Every arm starts from the final trained FP16 state, performs deployment-exact
scale roundtrips, and may alter only the four declared RSTB tensors. The
smallest arm wins only if its paired ground-truth lower 95% bound is strictly
positive versus ternary, its edge and teacher means are nonnegative, and it
retains at least 90% of the full FP16 effect on each of the three metrics.
Otherwise selection falls through to the next storage tier and finally to
FP16. The protocol remains development-only and cannot authorize promotion or
final-heldout access.

As a non-selecting diagnostic, applying the frozen quantizers to the shipped
q4 model's original float32 RSTB weights gives weight-domain SNRs of `7.878` dB
for group-nine ternary, `14.635` dB for int4/output, `21.823` dB for
int4/kernel, and `39.808` dB for int8/output. This predicts that per-kernel
int4 or per-output int8 may be the useful region, but the image-space paired
gate remains authoritative; weight reconstruction error cannot select an arm.

### Receipts

- RSTB ternary checkpoint SHA-256:
  `ddd99a89ef006641afa04fb909fe53c579a1b528cf8dc3b4291d0389db7864bb`
- RSTB ternary dense-state SHA-256:
  `ab5c3487c7c2dc8dc5d3397a513d7fd85efd63c0d6afdbf0d7de5e0dab1e8268`
- RSTB ternary development report SHA-256:
  `12ba42331c8ce4091c8b70ae698ebe8953398f13ef367d0267ca661fd5c219c5`
- q4 comparison SHA-256:
  `862ae2810d4a439da2469a017de49e75f64a2fb72a7a60f2775d5112d574715e`
- exact browser q4 ONNX SHA-256:
  `b75c09757a7159f32ad279919a77fb185c3389676674cf226a467eaf2f7928f1`
- mixed-precision protocol:
  `rstb_mixed_precision_20260731/protocol.json`
- mixed-precision protocol SHA-256:
  `d19c40ceeabad3991151737dbaee3d07c3f3a437d31b0782dabf1ea7b764d6b0`
- materializer SHA-256:
  `b8c8d011bc450b6434e06676667f242fe050f67bc8ba7298afcd0b94d28b4dbf`
- decision script SHA-256:
  `8d099f45b84bfd529c9b6755bbb92d75d33c085ea7051103789e1eeee2070dbe`
- tests: ten passed
- timing receipt: protocol sealed at `2026-07-31T21:21:23Z`, after the
  step-500 health signal but before the RSTB FP16 3,040-sample result and before
  any low-bit materialization existed.

After the FP16 training checkpoint was fixed and CPU-materialized—but while
its 3,040-sample evaluation was still running—all three deterministic low-bit
states materialized successfully. Each changes exactly the four declared RSTB
tensors and records exact FP16-scale roundtrips. On the trained weights, their
weight-domain SNRs are `17.030` dB for int4/output, `26.984` dB for
int4/kernel, and `39.482` dB for int8/output. The int4/kernel reconstruction is
substantially easier after task recovery than on the original shipped weights
(`21.823` dB), a useful hypothesis for progressive train-then-compress work;
the paired image-space evaluation remains the selection authority.

Materialized dense-state file SHA-256 receipts:

- int4/output:
  `3532953fc37fd705d07cc3d15765197eb34e28d5925723613766cb8b6571b050`;
- int4/kernel:
  `a552475cf647b530ee21cbcf7bb8294fec21f1afc988fe3723232d31b71cd476`;
- int8/output:
  `838e7c0b8a7415582e230e4e9db5897b08ec7613c313c3e76a3a4d3ab00bbcf6`.

## 2026-07-31 — RSTB FP16 discovery passes

The matched RSTB FP16 arm completed 4,000 steps and its exact 3,040-sample
development result is:

- ground-truth PSNR: `33.903041052` dB;
- edge PSNR: `37.453255877` dB;
- teacher PSNR: `40.488772689` dB; and
- gain over bicubic: `5.801446945` dB.

Against the same-seed, same-data-order, same-optimizer hard-ternary control,
FP16 improves:

- ground-truth PSNR by `+0.496866463` dB, paired-bootstrap 95% CI
  `[+0.479648212, +0.514719783]`, with `96.78%` sample wins;
- edge PSNR by `+0.484421503` dB, with `97.14%` wins; and
- teacher PSNR by `+1.278426621` dB, with `99.84%` wins.

Every preregistered discovery check passes, and the sealed decision therefore
authorizes both confirmation seeds for this island. The result establishes
that the four RSTB fusion convolutions are genuinely precision-sensitive under
the current recovery recipe. It does not establish that FP16 is the minimum
precision; the frozen int4/int8 ladder now tests that narrower claim.

The FP16 arm remains below the shipped q4 baseline by `-1.020605001` dB
ground-truth, `-0.979963213` dB edge, and `-6.444026867` dB teacher PSNR.
Therefore this is an attribution breakthrough, not yet a q4-quality candidate.

### Receipts

- checkpoint SHA-256:
  `3d9c50c1c8510b377ea7086143fae8468489fc61d580c8289d65c9efe11e5bde`
- materialization SHA-256:
  `aef96351f7f13c0042287a44452ac4816b04996beee430be1ad3334957afb9ca`
- dense-state file SHA-256:
  `7fae1af778c958d803c9dbf982743192dfe90c79de7984cdac2fe5c2d49e1208`
- development report SHA-256:
  `f0d78fa8033bda589453eef205b5c83dd8fd701a9641f27c897afaaccae50fbd`
- FP16-versus-ternary comparison SHA-256:
  `dc4c75c76e4764f94cc9adf969a7141531e557b8e047ea24c4204e91b1d37e21`
- q4 descriptive comparison SHA-256:
  `8ceea1120692270d569837a9fae371020f30d9ea7198b26c1dc9fdf8e5341a13`
- sealed discovery decision SHA-256:
  `8eac45d84864506852f7b4253f307416e895bee370c8533c6836f5bc015f0db2`

## 2026-07-31 — Same-device composition and image ablation decisions

The same-device CPU parent measures `32.979605543` dB ground-truth,
`36.566228100` dB edge, and `38.053167751` dB teacher PSNR. This resolves the
earlier projection confound. Relative to that parent, focused image-I/O
ternary recovery contributes `+1.017570222` dB ground-truth and focused RSTB
ternary recovery contributes `+0.426569046` dB; both have strictly positive
paired intervals and pass every frozen gate.

The zero-byte six-tensor composition nevertheless fails. It improves the
parent by `+0.805839934` dB and the RSTB-only control by `+0.379270888` dB, but
regresses against the stronger image-only control by `-0.211730288` dB
ground-truth, 95% CI `[-0.227054182, -0.197137367]`, `-0.144721720` dB edge,
and `-0.443420838` dB teacher PSNR. Independent recovery gains are therefore
not additive in weight space; the composition is rejected as the next ternary
research parent.

The image-I/O FP16 sub-tensor ablation also rejects both smaller subsets. The
input convolution alone is harmful versus the ternary control on all three
axes. The reconstruction convolution alone is beneficial, but retains only
`54.55%` of ground-truth, `45.59%` of edge, and `41.48%` of teacher gain,
below the frozen 90% rule. The two image-I/O convolutions form a positive
interaction and the joint `+6,480`-byte FP16 island remains selected.

### Receipts

- same-device parent report SHA-256:
  `80f5f21c709cb5b46f24a8cdbc222607d0ecd901e76a268f2c91de46957d6563`
- composition report SHA-256:
  `0f67a16118106ce71dbee11a9b1d2f5ffa2ec2de41131744225bae21f510066b`
- rejected composition decision SHA-256:
  `d23469c98bd2d24c4bbdce691284eda514ac307b38e356bb3ff4fc133b600083`
- input-only report SHA-256:
  `cfe4659d6f41dee37bfc14157596f0c30ecf6a9c37cd2b0fe20ed7d30efdf2df`
- reconstruction-only report SHA-256:
  `e195f279dfcd91196b32cf9599314dd1879ce735ebabe7744b62a72a159d6b01`
- joint-selection decision SHA-256:
  `68e0f9e85f563b1c6ec1ba644cce86049ab72a7887f49f629a961e8810e21996`

## 2026-07-31 — Int8 RSTB beats the FP16 endpoint

The preregistered mixed-precision ladder identifies int8/output as the first
low-bit RSTB representation that preserves the FP16 discovery effect. Against
the same ternary control it improves:

- ground-truth PSNR by `+0.503550690` dB, or `101.35%` of the FP16 effect;
- edge PSNR by `+0.489057408` dB, or `100.96%`; and
- teacher PSNR by `+1.278820204` dB, or `100.03%`.

All paired intervals are strictly positive. It is slightly better than FP16
on every mean while reducing the island's incremental data from `204,480` to
`75,360` bytes, a `63.15%` reduction. This is evidence that quantization noise
acts as a mild regularizer after FP16 task recovery, not evidence that all
weights should be int8.

Int4/output retains only `6.46%` to `21.43%` of the FP16 effects. Int4/kernel
is much stronger—`73.02%` ground-truth and `78.18%` edge—but retains only
`48.05%` of teacher agreement, so it fails the 90% rule. The next phase is a
sealed search over all 5/6/7-bit representations smaller than int8/output.

### Receipts

- mixed-precision decision SHA-256:
  `5f7ff513a73adda531fc03ce08bb788c6e2b2aea72a9c61a0071bb7909377847`
- int4/output report SHA-256:
  `119102ec8c775fa7829139a7d7eec045d92abcfc26f67d76f467beea20c66bf5`
- int4/kernel report SHA-256:
  `ad4dcabe5446218cc0b927ff16fbb3596a270a5a5b3d36699cafd307b99c8d39`
- int8/output report SHA-256:
  `8d61f62c512ac8af626b0fe02ebab251cc66d7fe57b11aa3737abc32599d0489`
- int8/output comparison SHA-256:
  `19106b41fd03b8a524b81ae7b791ecbfea76ad4f3ed637e4bd82196ba76b212f`
- boundary-search protocol SHA-256:
  `57f6ff13ecda04cb1606a85a249278cac32248bd1d8572656b2567671bdceaad`
- boundary-search tests: five passed

## 2026-07-31 — Five-bit per-kernel RSTB boundary selected

The complete preregistered 5/6/7-bit boundary ladder found that signed int5
with one FP16 scale per 3x3 input/output kernel is the smallest representation
that preserves at least 90% of every FP16-over-ternary discovery gain. Against
the matched ternary control it improves:

- ground-truth PSNR by `+0.479549188` dB, paired-bootstrap 95% CI
  `[+0.462779463, +0.496591462]`, retaining `96.51%` of the FP16 effect;
- edge PSNR by `+0.472908107` dB, retaining `97.62%`; and
- teacher PSNR by `+1.157281645` dB, retaining `90.52%`.

The selected island adds `55,080` bytes over its ternary representation,
equivalent to `+0.438295` whole-model data bits per parameter. The int7/output
and int6/kernel arms also pass but require `59,160` and `71,280` incremental
bytes, respectively, so the frozen smallest-passing rule rejects them.
Int6/output narrowly misses because it retains only `78.01%` of teacher gain;
int5/output retains only `31.83%` of teacher gain.

This is the first evidence that the sensitive RSTB island does not require the
shipped browser artifact's `518,400` FP32 bytes or the experiment's `204,480`
FP16 increment. It remains a discovery-seed result: the full model is still
below q4 quality, and the geometry is not promotion-ready until it reproduces
on both preregistered confirmation seeds.

Before either confirmation seed began training, a stricter confirmation
contract was sealed. Each seed must first independently reproduce the FP16
benefit; the int5 state must then preserve at least 90% of that same seed's own
FP16 gain on all three quality axes. Both seeds must pass, and averaging cannot
rescue a failed seed. Final heldout remains locked.

### Receipts

- boundary decision SHA-256:
  `253238705ab064415c099d6eca06df0ddd65880165ab328a1b5b571a5fceba13`
- int5/kernel report SHA-256:
  `22e0089f3b9a46d87e5eab6e0c1111c4c75cfb3b23319120cc1b24c2533b9af3`
- int5/kernel comparison SHA-256:
  `56210c01a71f8d81b2a1a5e0a1a2f686523b8cfb484f7e1b53e1a78f567d63b2`
- boundary protocol SHA-256:
  `57f6ff13ecda04cb1606a85a249278cac32248bd1d8572656b2567671bdceaad`
- confirmation protocol:
  `rstb_int5_confirmation_20260731/protocol.json`
- confirmation protocol SHA-256:
  `285bd81a84d2b8182f5151950b411938f099407ac8a19f8e4013aac6544b5f02`
- confirmation tests: five passed
- timing receipt: protocol sealed at `2026-07-31T22:22:29Z`; all sixteen
  declared seed-32/33 future outputs were absent and all thirteen immutable
  source hashes verified at seal time.

## 2026-07-31 — Int5/kernel does not reproduce on both confirmation seeds

Both held-back confirmation seeds independently reproduce the RSTB FP16
effect with strictly positive paired-bootstrap lower bounds. Int5/kernel also
retains the FP16 effect strongly on ground-truth and edge quality:

- seed `20260732`: `+0.481324295` dB ground-truth (`96.412%` retention),
  `+0.474347014` dB edge (`97.310%`), and `+1.136157300` dB teacher
  (`90.141%`); and
- seed `20260733`: `+0.478613830` dB ground-truth (`95.631%` retention),
  `+0.470779047` dB edge (`96.492%`), and `+1.140518643` dB teacher
  (`89.851%`).

The second seed misses the preregistered teacher-retention floor by `0.149`
percentage points. The rule is per seed and per axis, so averaging or rounding
cannot rescue it. Int5/kernel is rejected as the reliable default even though
its mean behavior is excellent.

The fallback contract was sealed after this failure and before any fallback
evaluation. It tests only the two discovery-passing larger representations:
int7/output (`+59,160` bytes over ternary) and int6/kernel (`+71,280` bytes).
The smallest representation passing the same per-seed 90% rule on both seeds
wins; otherwise the island remains FP16.

### Receipts

- int5 confirmation decision SHA-256:
  `60bfc212b08cfd2f2e7a942b5322355db6c6a8dfdc5bc2609507295112d2dc52`
- fallback protocol SHA-256:
  `261a8da37a3588dfaa2250620d844ef58c0ae67da20e77980d45ba332442ba09`
- fallback protocol tests: four passed
- all four fallback dense states were materialized only after the seal; their
  SHA-256 receipts are pinned in the fallback protocol outputs.

## 2026-07-31 — Maximum-stable-batch study sealed

The current retained-precision confirmations remain frozen at physical batch
eight. Changing their batch would invalidate the matched causal comparison.
For subsequent training, a separate Apple M5 capacity study was sealed before
any probe output existed.

The study runs the exact hard-ternary student, FP32 teacher, 128 px crop, and
training loss in a fresh process for every candidate. It searches geometrically
from batch 8 through 1,024 and then binary-refines in multiples of eight. A
candidate requires five measured finite updates and peak MPS driver allocation
no greater than 75% of the process-recommended maximum. The 25% headroom is
intentional: the scientific target is the largest stable usable batch, not the
first batch at the OOM cliff.

Capacity does not select the new default by itself. The capacity winner must
next face batch 8 at equal images processed under fixed, square-root-scaled,
and linearly scaled learning rates. Final selection requires both higher
throughput and non-inferior development quality.

### Receipts

- protocol SHA-256:
  `becfea1bb9bf165d1246126f3425daca9e88f1dd1220283d9ea36bf73f6d52e4`
- immutable source hashes verified: four
- immutable parent/data hashes verified: five
- future capacity outputs at seal time: zero
- unit tests: five passed

## 2026-07-31 — Both precision islands reproduce across three seeds

The two preregistered confirmation seeds independently reproduce both
discovery effects. Across discovery plus confirmation, FP16 over the matched
ternary location averages:

- image I/O: `+0.098183132` dB ground-truth, `+0.087901151` dB edge,
  and `+0.287084236` dB teacher agreement; and
- RSTB fusion: `+0.498859990` dB ground-truth, `+0.486591970` dB edge,
  and `+1.269394977` dB teacher agreement.

Every seed has a strictly positive ground-truth paired-bootstrap lower 95%
bound and nonnegative edge and teacher means. Both precision locations are
therefore supported. The larger RSTB effect remains the selected location for
compression research. This is not model-promotion authorization; final
heldout remains locked.

### Receipt

- three-seed confirmation decision SHA-256:
  `ce7ec83408affff905eb6ff71a25c023958d7a5133bee7c148b58d597bc7d1fb`

## 2026-07-31 — Largest safe M5 physical batch is 56, but batch 8 is fastest

The sealed capacity search completed in fresh MPS processes. Batch 56 is the
largest multiple of eight passing the finite-update and 25% memory-headroom
gates. It peaks at `34.2481` GiB driver allocation, `66.065%` of MPS's
recommended maximum. Batch 64 completes finite updates but peaks at `39.2696`
GiB, `75.752%`, and is rejected because it crosses the frozen 75% ceiling.

Capacity is not throughput. Measured compute throughput declines as batch
grows:

- batch 8: `19.0772` images/s, `5.2852` GiB peak driver allocation;
- batch 16: `18.7645` images/s, `11.1485` GiB;
- batch 32: `16.5374` images/s, `20.1856` GiB;
- batch 48: `12.9701` images/s, `28.2325` GiB; and
- batch 56: `12.9051` images/s, `34.2481` GiB.

The largest safe batch is therefore `32.36%` slower than batch 8 on compute
throughput and uses about `6.48x` the peak driver memory. Batch 8 also wins the
serial end-to-end estimate (`15.2436` versus `10.9770` images/s). Because the
capacity winner already fails the preregistered throughput prerequisite, it
does not warrant expensive equal-images quality/LR training as a default
candidate. Batch 56 remains available for experiments whose scientific goal
specifically requires lower gradient noise, but it should not be used as a
speed optimization.

### Receipts

- capacity decision SHA-256:
  `1c7044c68f8a827738336defe4b10e0eaba4d04395ca8fa7dfde072170b24301`
- capacity protocol SHA-256:
  `becfea1bb9bf165d1246126f3425daca9e88f1dd1220283d9ea36bf73f6d52e4`
- tested physical batches: `8`, `16`, `32`, `48`, `56`, and `64`
- selection: largest safe physical batch `56`; first rejected batch `64`

## 2026-07-31 — Int7/output is the confirmed RSTB precision boundary

The sealed fallback study completed after int5/kernel's seed-33 teacher miss.
Both remaining discovery-passing representations reproduce on both
confirmation seeds, and the frozen smallest-passing rule selects int7 with one
FP16 scale per output channel.

Int7/output retains the same seed's FP16-over-ternary effect as follows:

- seed `20260732`: `98.341%` ground-truth, `97.796%` edge, and `100.312%`
  teacher retention; and
- seed `20260733`: `99.280%` ground-truth, `98.691%` edge, and `100.601%`
  teacher retention.

Every ground-truth paired-bootstrap lower bound is strictly positive. The
selected island adds `59,160` data bytes over ternary, only `4,080` bytes more
than the rejected int5/kernel representation and `145,320` bytes less than
the FP16 island. Int6/kernel also passes with roughly `99.5–99.7%` retention,
but its `71,280`-byte increment is larger and is rejected by the frozen size
ordering.

This authorizes int7/output as the development-supported RSTB precision island
for a separately preregistered joint-training study. It does not authorize
model promotion or final-heldout access.

### Receipt

- fallback decision SHA-256:
  `b2175cddcf64c5ea51bfec59a793dec914f27309e2f2b95fe18ce7223c671fbe`

## 2026-07-31 — Current candidates already exceed released Ternary Lite

The production Ternary Lite artifact is the all-hard g2 model at
`32.896689950` dB ground-truth PSNR over 3,040 development samples, a
`4.795095843` dB gain over bicubic. Later sealed training produced stronger
states on the identical bundle:

- the fully ternary image-I/O retraining control averages `33.996990961` dB
  across three seeds, about `+1.100301011` dB over the released model; and
- retaining the same 8,100 image-I/O values in FP16 averages `34.095174093`
  dB, about `+1.198484143` dB over release.

The fully ternary arm is the cleanest prospective Ternary Lite v2 because all
204 quantized surfaces remain hard ternary. The FP16 arm is the current higher
quality mixed-precision state. These are development findings, not release
authorization: neither has yet completed browser artifact export, execution
parity, robustness, and final-heldout promotion gates.

In parallel, the MixFrag-inspired per-surface scale study selected a genuinely
new 10,012-byte allocation after its combined error stayed 94.8% below uniform
q2 on both allocation and disjoint training-only shards. That map now advances
to QAT; it does not supersede the stronger image-I/O checkpoints until trained
and evaluated on the sealed 3,040-sample development bundle.

### Receipts

- released artifact manifest SHA-256 is pinned by the BackdropKit repository;
- retained-precision confirmation decision SHA-256:
  `ce7ec83408affff905eb6ff71a25c023958d7a5133bee7c148b58d597bc7d1fb`
- MixFrag interaction result SHA-256:
  `141e55aad9b9b3a98778b4b6d62696f126ac42453f55a794ecd56d60d4724b2a`

## 2026-07-31 — The 34.000 dB fully ternary leader is now a real browser artifact

The strongest fully ternary confirmation seed, `20260733`, has now crossed
the artifact and first runtime boundaries. Its exact evaluated dense-hard
state was reconstructed without tensor drift, packed into the browser-v2
format, and compared against the same 64-to-128 fixture in Python, native
Rust, SIMD WebAssembly, and scalar WebAssembly.

The artifact is `440,503` bytes total (`326,480` binary bytes). It stores
`97.240%` of the model's parameters as ternary code plus scales, with
`1.600943` physical code bits per quantized weight and `2.597946` whole-model
data bits per parameter after raw tensors and scales are included. The packed
Python path reaches `77.2472` dB against the exact evaluated state. Native
Rust, browser SIMD WASM, and browser scalar WASM all remain below `0.000759`
maximum absolute error on the sealed fixture.

The real in-app-browser single-run fixture timings were `2.6292` seconds for
SIMD WASM and `3.1851` seconds for scalar WASM. These are conformance timings,
not throughput claims. This updates the earlier status: browser artifact
export and generic WASM execution are now proven for the fully ternary leader.
Safari, Firefox, mobile, WebGPU, product integration, robustness, and final
heldout gates remain open, so promotion is still unauthorized.

### Receipts

- browser-export study manifest SHA-256:
  `2af5c1cb5dad70949a6409092da3b02dc12000e238bdd150878b95ece32db5f5`
- artifact manifest SHA-256:
  `5424487a6c7a5c7f81f677d94d081cdb0e8a614d39d5ec428d9a086f5b4e087c`
- artifact binary SHA-256:
  `aaee28988ab651cb5be67ce940e69d77d6a07ad9ea659edab2431140dab27bd1`
- runtime conformance receipt:
  `retained_precision_study_v2_20260731/image_io_fp16__ternary__seed20260733/browser_export_v2/runtime-conformance-receipt.json`

## 2026-07-31 — Bonsai-style mixed precision becomes the release leader

The release target is no longer an artificial “exactly 1.58-bit” number. The
Prism ML Bonsai Image implementation provides the more useful engineering
analogy: keep the matrix-heavy trunk ternary and preserve a small set of
precision-sensitive supporting tensors. Its public model card reports 100
ternary matrix-heavy linears with FP16 group scales, while under 5% of model
parameters remain FP16. It also distinguishes the 1.21 GB model-data estimate
from the 1.43 GB aligned MLX package. We now make the equivalent distinctions
for Swin2SR: ternary code entropy, whole-model data, and actual lossless
transport size are separate quantities.

The strongest confirmed image-I/O FP16 seed, `20260733`, is the new release
leader. Its exact evaluated state is `34.098155400` dB over the locked 3,040
development samples, `+1.201465450` dB over the previously released Ternary
Lite model. It keeps 202 weight tensors and 969,504 weight values ternary, while
only the first and final image-I/O convolution weights—8,100 values—remain
FP16. This is 96.4347% ternary by all model parameters.

The complete model data occupies `2.649510418` bits per model parameter. The
lossless native browser package is 330,933 bytes, or `2.633380680` bits per
parameter including its manifest and precision islands. The portable ONNX
transport is 700,584 bytes. This is therefore honestly a mixed-precision
ternary release, not an exact 1.585-bit model.

All implemented execution paths passed the release fixture: packed Python,
native Rust, portable ONNX on Chromium WebGPU and WASM, and the independent
packed runtime on SIMD and scalar WebAssembly. On the 64-to-128 release
fixture, warm portable-ONNX inference measured about 29.9 ms on WebGPU and
399.2 ms on WASM; packed WASM measured about 1.86 seconds with SIMD and 3.03
seconds scalar. These are single-machine conformance observations, not broad
performance claims.

The Hugging Face model package and Space were restaged with the exact release
weights, 331 KB native package, 701 KB portable download, regenerated Set5
comparisons, and truthful mixed-precision language. The Transformers package
reloaded with zero tensor error against the evaluated state; all four published
artifact hashes match the release ledger. Public upload remains gated only by
the absence of Hugging Face credentials in this environment.

The competing MixFrag scale-QAT branch is rejected as a replacement. Its full
3,040-sample result is `32.582926` dB and its sealed scale allocation is not
smaller than the 10,012-byte control, so it fails both the quality and storage
reasons for advancement. Its allocation mechanism should not be transferred
to the release model without a revised hypothesis.

### Receipts

- exact release checkpoint SHA-256:
  `cb5455bba3154a46eb9f744a6b5aee03497c0807413e632fd7eea070a21a1c87`
- dense evaluated state SHA-256:
  `09cced7f7540bccd0f2d0dd3dae4387e18ddc25bf469fec605471f5889c81aa3`
- native package SHA-256:
  `d2b1ad86600b485c66ff81bad52ffb4893f6fa899368edd038116b5dcb830cd0`
- portable ONNX gzip SHA-256:
  `b6308e886a0ef07a2b433e2aa83f6a2234d2dbcdbf1427160abedb4e3629aa54`
- browser conformance receipt:
  `retained_precision_study_v2_20260731/image_io_fp16__fp16__seed20260733/browser_export_v2/browser-conformance-receipt.json`
- Hugging Face package receipt:
  `huggingface/clark-swin2sr-lightweight-x2-1.58bit/technical/quantization.json`
- MixFrag rejection decision:
  `mixfrag_scale_allocation_20260731/qat_v1/decision_gate_v2/decision.json`

## 2026-07-31 — Mixed-precision ternary v2 is published

The Bonsai-style mixed-precision leader is now the public Clark Swin2SR release.
The Hugging Face model repository was published at commit
`a7be3269cb695b3eb6d05f0f33044aee3971f1ee`; the static browser Space was
published at commit `39b80d2538996c520d0b89902b44cfb9af8fb30d`.

Release closure downloaded `model.safetensors`, the portable ONNX gzip, and the
native packed bundle back from the immutable public model commit. Every byte
count and SHA-256 matched `technical/quantization.json`. The public Space file
tree contains only its static application and images; the obsolete server-side
`app.py` and `requirements.txt` were removed in the Space commit and remain
recoverable from repository history.

The live public Space reported WebGPU ready, upscaled the bundled 128×128
butterfly to 256×256, and enabled the full-resolution PNG download. The forced
`?backend=wasm` route independently reported WASM ready, completed the same
upscale, and enabled the download. ONNX Runtime emitted only its known execution
provider assignment warning in the WASM path; there was no functional failure.

### Public receipts

- model commit:
  `https://huggingface.co/clark-labs/clark-swin2sr-lightweight-x2-1.58bit/commit/a7be3269cb695b3eb6d05f0f33044aee3971f1ee`
- Space commit:
  `https://huggingface.co/spaces/clark-labs/clark-swin2sr-lightweight-x2-1.58bit/commit/39b80d2538996c520d0b89902b44cfb9af8fb30d`
- live static Space:
  `https://clark-labs-clark-swin2sr-lightweight-x2-1-58bit.static.hf.space`

## 2026-08-01 — v2 superseded by G3

The mixed-precision v2 publication remains immutable historical evidence, but
it is no longer the selected public payload. G3 improves the previous ternary
release by 0.209351 dB ground-truth, 0.150167 dB edge, and 0.721767 dB teacher
PSNR on the same 3,040-sample development bundle. The current Hugging Face
model commit is `b357304af178a6e31a5482958f1adf702f99be82`; the non-blocking
Space repair is `9896ce694616a8294fc566cd68f3c4a873a33d8c`. No writer in this v2
folder is active.
