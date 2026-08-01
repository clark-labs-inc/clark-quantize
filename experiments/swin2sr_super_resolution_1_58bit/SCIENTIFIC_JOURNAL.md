# Scientific journal

This journal is append-only in spirit: correct an error with a dated
correction rather than silently rewriting an earlier conclusion. Paths are
relative to `experiments/swin2sr_super_resolution_1_58bit/` unless stated
otherwise.

## 2026-07-29 — Scope and acceptance contract

**Verified**

- The teacher is `caidas/swin2SR-lightweight-x2-64`; the shipped comparison is
  `Xenova/swin2SR-lightweight-x2-64` q4 at revision
  `54ccc3f9ed678912a9c47dc14f6c478855380a2b`.
- The q4 glyph-safe baseline is 5,676,042 bytes and scores 34.8135 dB against
  ground truth, 38.3047 edge dB, and 46.8731 dB against the teacher over 3,040
  heldout samples.
- The heldout combines 1,000 RICO examples and 2,040 multilingual examples:
  204 languages, 29 scripts, ten images per language, zero training-image
  overlap, zero raw-sentence overlap, and zero unknown glyphs.

**Decision**

- Clark Quantize is a source of mechanisms and negative results, not an
  independent objective. This study is only about BackdropKit's tiny Swin2SR.
- Begin with FP32 teacher weights. Keep continuous masters and introduce hard
  ternary forward paths gradually; do not ternarize first and attempt to repair
  afterward.
- Final acceptance requires slice-wise statistical non-inferiority or better
  quality than q4, a materially smaller browser artifact, and runtime viability
  on WebGPU, WASM, and CPU.

## 2026-07-30 — Corpus and evaluation isolation

**Verified**

- Balanced training epoch: 206,500 images.
- Multilingual synthetic UI subset: 102,000 images, 500 per language across
  204 languages, generated with glyph-safe font fallback.
- The final glyph-safe bundle is sealed separately from training and
  development selection.

**Decision**

- Optimizer, schedule, quantizer, and sensitive-layer choices use only
  development evidence. No optimizer arm in the current study has opened the
  final candidate heldout.

## 2026-07-30 — Browser artifact feasibility

**Verified infrastructure evidence**

- A full-surface, untrained hard-state artifact covered 204 ternary tensors and
  977,604 values at 1.8675 effective weight bits/value including scales.
- Expanded portable ONNX size was 4,807,898 bytes, 15.29% below the shipped q4
  artifact.
- One Chromium receipt passed WebGPU and WASM numerical parity. At 128px input,
  warm WebGPU inference was 118–170 ms and WASM was 1.41–1.46 s.
- A compressed infrastructure package was 650,126 bytes. On a Chromium surface
  without a GPU adapter, WebGPU failed cleanly while WASM passed at roughly
  374 ms warm inference for 64px input.

**Interpretation**

- True browser deployment is technically feasible, and WebGPU must remain an
  optimization rather than a prerequisite.
- These artifacts came from infrastructure-only hard states. They prove graph,
  packaging, and fallback behavior—not visual quality or promotion readiness.
- Safari, Firefox, iOS Safari, and Android Chrome still need direct receipts
  for the eventual quality winner.

## 2026-07-30 — Canonical materialization across devices

**Observed failure**

- Reloading the step-14,000 continuous-master checkpoint and projecting it on
  MPS reproduced the live 39.3275 dB development validation, while projecting
  the same masters on CPU produced 38.9418 dB.
- Exactly 13 of 977,604 ternary codes differed. Every difference was in an
  optimal-MSE residual convolution whose competing support errors differed by
  roughly 10^-8; absmean surfaces had no code differences.

**Verified deployment boundary**

- The same state was projected once on MPS, materialized into ordinary dense
  hard tensors, and loaded unchanged on MPS and CPU.
- Across four fixed UI samples, maximum output error was 5.96e-7, minimum
  output PSNR was 99 dB, and mean ground-truth PSNR differed by 5.86e-7 dB.
- Receipt:
  `../browser_swin2sr/sensitive_hardshadow_20260730/materialized-device-parity-step-14000.json`.

**Decision**

- A continuous-master checkpoint is not the deployable model identity and
  must not be independently reprojected by each backend.
- Export must materialize one canonical hard state once, hash it, and feed
  those exact codes/scales or expanded weights to every WebGPU, WASM, and CPU
  evaluator. Backend parity is evaluated after that boundary.
- The step-14,000 parity receipt is diagnostic only: forcing the two untrained
  image-I/O surfaces hard at this intermediate point has poor visual quality.

**Canonical step-14,000 infrastructure artifact**

- One MPS materialization produced a hashed dense hard state and a hashed
  runtime hard state covering all 204 quantized surfaces.
- The packed browser artifact is 429,885 bytes in total. Its packed trunk is
  2.1312 bits per quantized weight, and verification against the canonical
  dense hard state passed at 78.3363 dB output PSNR.
- The portable ONNX package is 677,510 bytes compressed. A Node
  ONNX Runtime WASM receipt passed with maximum output error 0.0003047 and
  498–543 ms inference at 64px input.
- The same hashed compressed package passed in Chromium through both real
  browser backends. WebGPU warm inference was 37.5–59.1 ms and WASM warm
  inference was 490.1–505.4 ms at 64px. Both matched the canonical dense hard
  state within maximum absolute error 0.0003047 and mean error 0.0000576.
- These are identity, packing, and runtime-path receipts for an intermediate
  checkpoint. They are not quality or promotion evidence.

## 2026-07-30 — Sensitive hard-shadow curriculum

**Verified design**

- Continuous masters with hard ternary STE forward.
- Closed-form current-master scales rather than learned scale drift.
- Staged surfaces: attention, remaining core linears, residual-body
  convolutions, position-bias linears, input convolution, then
  upsampling/output convolution.
- A concurrent hard-shadow branch trains every already-started surface at its
  fully hard endpoint every second transition step.
- Sealed control: 30,000 total steps, 24,000-step schedule, 6,000 hard recovery
  steps, absmean group-60 Hadamard-shaped matrices, kernel-grouped optimal-MSE
  convolutions, and continuous bias/normalization compensation.

**Observed trajectory**

| Step | Blended GT dB | Hard-started GT dB | Interpretation |
|---:|---:|---:|---|
| 6,000 | 43.1257 | 41.6922 | Core branch remained recoverable |
| 7,000 | 43.0242 | 33.1409 | Residual category entered together and hard branch collapsed |
| 8,000 | 42.3596 | 34.4507 | Partial recovery |
| 10,000 | 40.4355 | 36.4446 | Hard branch continued recovering |
| 12,000 | 38.6499 | 38.6059 | 154 surfaces hard; forcing 48 started position-bias surfaces hard added little extra loss |
| 14,000 | 39.3275 | 39.3020 | MPS projection; hard-started gap remained only 0.0255 dB |
| 15,000 | 39.0241 | 20.7513 | First input convolution had just started; forcing it hard exposed severe sensitivity |
| 16,000 | 36.3558 | 22.8325 | Input-convolution hard endpoint recovered 2.08 dB but remained 13.52 dB below blended |
| 17,000 | 35.1861 | 25.5379 | Hard endpoint recovered another 2.71 dB; blended-to-hard gap remained 9.65 dB |
| 18,000 | 34.0745 | 30.1491 | Hard endpoint recovered another 4.61 dB; blended-to-hard gap narrowed to 3.93 dB |

At step 12,000, overall code churn was 1.4396%, zero fraction 34.6049%,
sign inheritance 100%, and relative master movement 2.2206%. These signatures
are diagnostic only and are not optimization targets.

At step 14,000, overall code churn was 1.5912%, zero fraction 34.6108%,
sign inheritance remained 100%, and relative master movement was 2.4464%.
The table uses the training-device MPS projection; the canonical
materialization entry above explains why CPU reprojection of continuous
masters is not equivalent.

At step 15,000, 203 of 204 surfaces had started, but the input convolution was
only beginning its transition. Forcing every started surface hard reduced
ground-truth PSNR by 18.2729 dB relative to the blended branch. The input/output
category had moved only 1.0401% relative to its FP32 initialization, with
1.1235% code churn and no direct sign flips. The immediately following blended
step-15,500 validation also fell to 37.0299 dB.

**Interpretation**

- Blended metrics hid a real residual hard-endpoint collapse at 7,000 steps.
- Hard-shadow training subsequently recovered much of the gap.
- The input convolution is substantially more sensitive than the already-hard
  core and residual surfaces. Step 15,000 is an early-transition diagnostic,
  not evidence that the sealed curriculum has failed; recovery must be
  measured at subsequent hard-started checkpoints before changing the study.
- The control is still not a promotion candidate: the input surface has not
  completed its transition, the output surface has not started, final all-hard
  recovery is incomplete, and heldout quality is unmeasured.

At step 18,000, 203 surfaces had started and 202 were fully hard. The input
convolution remained the only transitioning surface. Overall code churn was
1.8391%, zero fraction was 34.6292%, sign inheritance remained 100%, and
relative master movement was 2.8214%. The hard endpoint has recovered
monotonically from 20.7513 dB at step 15,000 to 30.1491 dB at step 18,000,
but it remains below the shipped q4 development baseline.

The step-18,000 evaluator initially failed closed because the control's sealed
source folder contained only one of the 17 manifest-pinned files. The missing
sources were recovered without weakening verification: unchanged files came
from exact-hash copies, and four edited files were reconstructed by reversing
known later patches. Every resulting byte stream matches the SHA-256 already
recorded before training. The reconstruction receipt is retained with the
sealed source snapshot.

## 2026-07-30 — Sensitive input-convolution scale groups

**Protocol**

- Development-only evaluation at the step-16,000 MPS checkpoint.
- Every already-started surface was forced hard; the future output convolution
  remained dense.
- The 1,620-weight input convolution was evaluated with optimal-MSE and absmean
  ternary projections across coarse and refined group-size grids.
- Storage cost was expressed as the weighted change in bits across all 977,604
  quantized weights, relative to the sealed group-size-9 control.

**Verified**

- The sealed optimal-MSE group-size-9 endpoint scored 22.8325 dB.
- Optimal-MSE group size 2 scored 30.7220 dB: a 7.8895 dB improvement for only
  +0.01031 whole-model bits per quantized weight.
- Group size 1 scored 30.1808 dB and was worse than group size 2 despite
  carrying more scale information. The result is therefore not explained by
  a trivial per-weight FP16-scale escape hatch.
- A refined grid covering group sizes 1–10 and several larger controls
  confirmed group size 2 as the deterministic winner.
- At step 17,000, the same optimal-MSE group-size-2 projection improved to
  32.0031 dB versus 25.5379 dB for group size 9. The advantage persisted at
  6.4652 dB as the control masters continued adapting.
- Side-by-side renders over all four fixed product-UI crops confirmed the
  metric. Group size 9 introduced a strong brown cast; group size 2 largely
  removed it and restored neutral dark backgrounds and clearer text
  structure. Per-image gains ranged from 5.52 to 10.41 dB.

**Decision**

- Preserve the sealed 30,000-step control unchanged.
- Preregister a separate sensitive-I/O rescue using optimal-MSE group size 2
  for the input convolution. This remains a ternary-code surface and changes
  the weighted model budget by roughly one-hundredth of a bit.
- Do not assume the same choice for the 6,480-weight output convolution. Probe
  that surface independently once it begins its scheduled transition.

## 2026-07-30 — Optimizer tournament

**Protocol**

- Nine Clark-developed optimizer paths crossed with 0.25x, 1x, and 4x learning
  rates: 27 arms.
- Teacher revision, initialization, data order, seed, quantizer, rotation,
  schedule, crop size, loss, and validation were held fixed.
- Every optimizer also passed CPU and MPS finite-step checks, 4D convolution
  updates, checkpoint serialization, and exact next-step resume parity.

**Verified 200-step development ranking**

| Rank | Optimizer | LR | GT dB | Optimizer state |
|---:|---|---:|---:|---:|
| 1 | Mirror SGD | 8e-5 | 42.2311 | 0 bytes |
| 2 | BonsaiQAT8 + periodic Muon | 8e-5 | 42.1793 | 1,006,080 bytes |
| 3 | Factored Adafactor | 8e-5 | 41.5711 | 149,760 bytes |
| 4 | BonsaiQAT8 | 8e-5 | 41.4566 | 1,006,080 bytes |
| 5 | AdamW | 8e-5 | 41.0866 | 5,669,376 bytes |

All 27 arms completed, with no crashes, non-finite metrics, or sealed-source
drift.

**Negative results**

- Sana-style proximal AdamW collapsed to roughly 30.8 dB at all three learning
  rates.
- Direct Muon and batched direct Muon matched numerically but peaked near
  35.11 dB. Periodic Muon helped QAT8; replacing every matrix update with Muon
  did not.
- Group AdaGrad damped mirror recovery and trailed stateless mirror SGD.

**Decision**

- The 0.0518 dB mirror/QAT8+Muon difference is too small to call a winner.
- Advance mirror SGD, QAT8+periodic Muon, factored Adafactor, and AdamW control
  into 1,000-step paired replications across seeds 20260730–20260732.
- Rank replicated arms by paired confidence intervals. Do not use runtime from
  contended runs as a quality tie-breaker.

**Preregistered replication decision rule**

Before any 1,000-step arm completed, the development-only selector was fixed in
`../browser_swin2sr/decide_optimizer_replicates.py`:

1. Identify the empirical best mean ground-truth PSNR arm.
2. Admit another arm only when the upper bound of its paired 95% confidence
   interval versus that best is at most 0.10 dB for both ground-truth and
   teacher PSNR.
3. Within that statistically non-inferior set, minimize maximum optimizer-state
   bytes across seeds, then maximize ground-truth PSNR and teacher PSNR.
4. Treat the result only as a recipe choice for the next full-surface
   development run. It cannot promote a model or open the final heldout.

**Completed three-seed replication**

| Rank | Optimizer | Mean GT dB | Mean teacher dB | Optimizer state |
|---:|---|---:|---:|---:|
| 1 | AdamW | 43.5756 | 49.3211 | 5,669,376 bytes |
| 2 | Mirror SGD | 43.3204 | 48.0068 | 0 bytes |
| 3 | Factored Adafactor | 42.8741 | 47.9990 | 149,760 bytes |
| 4 | QAT8 + periodic Muon | 42.7508 | 47.7290 | 1,006,080 bytes |

All four arms completed 1,000 steps on seeds 20260730–20260732 with finite
metrics and exact run receipts. AdamW was the empirical quality best. No
lower-state arm's paired 95% confidence upper bound was within the frozen
0.10 dB margin on both ground-truth and teacher PSNR, so AdamW was the only
eligible arm and is selected for the next development recipe.

This is a negative result for the current low-state alternatives, not proof
that AdamW is universally superior. With only three seeds, the paired
intervals are wide: Mirror SGD's AdamW-minus-mirror ground-truth interval was
-0.7715 to +1.2818 dB, and its teacher interval was -1.1979 to +3.8265 dB.
The preregistered rule correctly refuses to trade quality for state when
non-inferiority is not established.

## Active as of 2026-07-30

- `../browser_swin2sr/sensitive_hardshadow_20260730/` — sealed 30,000-step
  full-surface control; active writer.
- `../browser_swin2sr/optimizer_replicates_20260730/` — completed paired
  optimizer decision; AdamW selected for the next development recipe.
- `../browser_swin2sr/sensitive_io_rescue_20260730/` — active matched
  sensitive-I/O rescue arms.

## 2026-07-30 — Sensitive-I/O rescue execution

**Verified provenance**

- A fail-closed checkpoint fork now supports exact per-surface group sizes.
  The trainer, hard-started evaluator, signature analyzer, checkpoint writer,
  and resume validator all reproduce the override.
- All three rescue arms fork the sealed step-14,000 checkpoint with parent
  SHA-256 `4f4abd928f11dfd1d110487232e6cbc5ddda9e11e8ae7a99a58dd1b2b766c45a`.
- Model, optimizer moments, run RNG, data epoch, and data cursor are unchanged.
  Every fork sets only `swin2sr.first_convolution` to group size 2. The 1x and
  4x arms additionally change the canonical convolution optimizer group's
  learning-rate multiplier from 0.25 to 1 or 4.
- Every arm has a hashed fork receipt, an exact command, and a sealed source
  snapshot under
  `../browser_swin2sr/sensitive_io_rescue_20260730/`.
- Inspect-only resume verified 204 surfaces, 977,604 ternary values, exact
  step 14,000 data resume, group size 2 only on the first convolution, and the
  intended convolution learning-rate multiplier.

**Execution**

- `io-g2-lr025`, `io-g2-lr1`, and `io-g2-lr4` are active matched MPS writers.
- Their first pre-update losses at step 14,001 were identical, as required by
  the shared parent state and data cursor. Their post-update validation values
  already diverged by learning rate; those one-step values are diagnostics,
  not selection evidence.
- The preregistered step-19,000 rule remains authoritative. No arm has been
  selected and final heldout remains closed.

**Matched-control amendment**

- At step 14,500, `io-g2-lr025` was 0.2727 dB above the uninterrupted control
  even though the input surface had not started. The historical trajectory is
  therefore not a sufficiently matched causal control after restart.
- A `restart-g9-lr025` arm was added before any rescue arm reached the input
  transition. It shares the exact step-14,000 state and reproduced
  `io-g2-lr025` exactly at step 14,001: 38.948371 dB ground-truth and
  42.079459 dB teacher PSNR.
- Step-19,000 decisions will use the matched restarted group-9 arm as the
  primary comparator and retain the uninterrupted control as an external
  trajectory reference.

## 2026-07-30 — Research-folder consolidation

**Verified**

- `experiments/swin2sr_super_resolution_1_58bit/` is the semantic home for
  this program and contains its scope, append-only scientific journal,
  artifact index, method map, preregistered studies, and promotion-policy
  amendments.
- `../browser_swin2sr/` is still the canonical implementation and artifact
  root. At the time of this entry it has five live MPS writers: the sealed
  30,000-step hard-shadow control plus four sensitive-I/O rescue arms.
- The LLM and Sana programs have separate archived dossiers with their own
  scientific journals and artifact indexes. Neither is an active BackdropKit
  training objective.

**Decision**

- Do not rename, move, or duplicate live outputs, sealed source snapshots, or
  immutable receipts merely to make the tree look uniform. Their paths and
  hashes are scientific evidence.
- New Swin2SR discoveries are recorded here. New studies and policy changes
  are created under this semantic folder rather than as loose repository-root
  notes.
- Once every writer under `../browser_swin2sr/` has stopped, classify each
  path as sealed-and-pinned or unsealed-and-movable. Only the latter may move,
  with an old-path/new-path/SHA-256 migration receipt and updated imports,
  tests, and documentation.

## 2026-07-30 — Full-surface optimizer transfer boundary

**Verified implementation**

- The full-surface trainer now accepts explicit `adamw` or
  `bonsai_qat8_muon` recipes instead of silently hardcoding the latter.
- AdamW uses explicit `(0.9, 0.999)` betas and `1e-8` epsilon while retaining
  separate matrix, convolution, and correction learning-rate groups.
- Fresh FP32 runs accept an exact per-surface group-size map. Resume validation
  rejects optimizer, convolution multiplier, correction multiplier, or
  surface-map drift.
- A real inspect-only load instantiated all 204 surfaces and 977,604 ternary
  values with AdamW and group size 2 on
  `swin2sr.first_convolution`. Targeted optimizer, resume, CLI, sealer, and
  surface tests passed.

**Decision**

- The core-only optimizer replication is not sufficient proof that AdamW wins
  on image-facing convolutions. Preregister a matched full-surface transfer
  study before launching it:
  `studies/FULL_SURFACE_OPTIMIZER_TRANSFER_STUDY_20260730.md`.
- Do not launch that study until the sensitive-I/O rescue fixes the
  first-convolution group size and convolution learning-rate multiplier.

## 2026-07-30 — Sensitive-I/O step-15 and sealed stage-two gate

**Verified diagnostics**

- The first group-2 checkpoints reached step 15,000, just after the input
  convolution began transitioning.
- Hard-started ground-truth/teacher PSNR was:
  - 0.25x convolution LR: 27.3439 / 27.8258 dB;
  - 1x convolution LR: 27.2700 / 27.7543 dB;
  - 4x convolution LR: 28.1207 / 28.6929 dB.
- The historical uninterrupted group-9 step-15 checkpoint was 20.7513 /
  20.8906 dB. This large gap remains non-causal until the matched restarted
  group-9 checkpoint reaches the same boundary.
- At 0.25x LR, group size 2 changed only three first-convolution codes at this
  boundary, but its first-convolution zero fraction was 24.38% versus 41.42%
  for historical group size 9. The gain is primarily a better support/scale
  representation, not broad master-weight motion.
- The uninterrupted group-9 control reached step 19,000 with 203 started
  surfaces fully hard and the future output surface dense: 33.5324 dB
  ground-truth and 34.4240 dB teacher PSNR. This is an external trajectory,
  not the matched causal control.

**Provenance repair**

- Resumed training checkpoints retained the sealed fork path but did not copy
  the fork-parent object into later checkpoints. The hard-started evaluator now
  reconstructs that lineage only after verifying the fork checkpoint, fork
  receipt, parent hash, and study-manifest hashes. The preregistered selector
  remains fail-closed.

**Pre-outcome protocol correction**

- A first stage-two gate manifest accidentally imposed confidence-bound
  improvement versus control and maximum-quality ranking, conflicting with the
  original smallest-LR-within-0.10-dB preregistration.
- It was invalidated before any step-19 rescue checkpoint, hard-started
  receipt, or full-development report existed. The immutable invalidation
  receipt preserves its hash and reason.
- `development-gate-manifest-v2.json` now seals the original rule: evaluate
  the frozen top two over all 3,040 development samples, require overall and
  every-source means within 0.10 dB of the best arm, then select the smallest
  convolution LR multiplier. Model promotion and final heldout access remain
  unauthorized.

## 2026-07-30 — Output-convolution boundary preregistered

**Verified**

- `upsample.conv` contains 6,480 weights with shape `(12, 60, 3, 3)`.
- Its transition begins at step 19,200 and ends at step 23,520 under the
  sealed 24,000-step schedule.
- The existing sensitive group-size probe already targets this surface and
  reports whole-model scale overhead.

**Decision**

- Do not copy the input-convolution group size onto the output convolution.
- Use the uninterrupted control at steps 20,000, 22,000, and 24,000 only for
  external diagnostics.
- Select an output recipe only after repeating the probe and matched training
  on the input-recipe winner. The preregistration is
  `studies/OUTPUT_CONV_GROUP_PROBE_20260730.md`.

## 2026-07-30 — Research-folder organization

**Verified**

- Stable Swin2SR methods, study definitions, policy amendments, findings, and
  evidence references now have one semantic home:
  `experiments/swin2sr_super_resolution_1_58bit/`.
- The active implementation and sealed evidence remain under
  `experiments/browser_swin2sr/`.

**Decision**

- Do not rename active writers, command roots, sealed manifests, checkpoints,
  or receipts for cosmetic consistency.
- Treat `ARTIFACT_INDEX.md` as the canonical bridge until the active runs
  finish. Any later move requires a path-and-hash migration receipt as defined
  in `FOLDER_MIGRATION.md`.

**Latest live receipt**

- Mirror SGD completed 1,000 steps on all three seeds with final checkpoints:
  43.2396/48.0896 dB GT/teacher for seed 20260730,
  43.8106/48.1688 for seed 20260731, and
  42.9111/47.7621 for seed 20260732.
- Every runner then advanced automatically to QAT8+periodic Muon.
- These are development observations from one completed arm, not a winner or
  promotion result. The preregistered selector remains closed until all four
  matched arms finish.

## Remaining gates

The earlier optimizer-replication gate is complete. AdamW is the
preregistered development winner for the later full-surface transfer study;
it is not a promoted model.

1. Finish the matched sensitive-I/O rescue and its full 3,040-sample
   development gate.
2. Apply the selected recipe to the full sensitive-surface curriculum with
   hard-shadow endpoint training.
3. Complete the all-hard recovery phase.
4. Pass development overall, edge, teacher, source, language, and script gates
   against q4.
5. Open the final glyph-safe heldout exactly once for an eligible candidate.
6. Export the actual winner and prove size, numerical parity, memory, and
   latency across WebGPU, WASM, CPU, and required browser/device surfaces.
7. Integrate into BackdropKit only if every gate passes.

## 2026-07-30 — Matched step-15 sensitive-I/O result

**Verified**

- The restarted group-size-9, 0.25x control reached the same step-15,000
  boundary as all three group-size-2 arms.
- At identical 0.25x convolution LR, group size 2 exceeded the matched
  group-size-9 control by 0.3969 dB blended ground-truth PSNR and 6.5344 dB
  hard-started ground-truth PSNR.
- The hard-started teacher-PSNR advantage was 6.8772 dB.
- Relative continuous-master movement was 0.5455% for group size 2 and
  0.5450% for group size 9. The first-convolution zero fractions were 24.38%
  and 41.42%, respectively.

**Interpretation**

- The matched result removes the restart confound present in the
  uninterrupted-control comparison.
- The large hard-endpoint advantage with nearly identical master movement
  supports a representational explanation: group size 2 gives the sensitive
  input convolution a materially better ternary support/scale basis.
- No learning-rate arm is selected from step 15. The preregistered step-19
  selector and full 3,040-sample development gate remain authoritative.

## 2026-07-30 — Folder standard applied across research programs

**Verified**

- Swin2SR, LLM ternary, and Sana image-generation research now each have a
  semantic program folder containing a README, scientific journal, and
  artifact index.
- Stable Swin2SR methods, study preregistrations, and policy amendments have
  been physically consolidated into this folder.
- LLM and Sana historical files remain linked at their authoritative paths;
  moving them would rewrite old commands, imports, release references, or
  provenance without improving reproducibility.

**Decision**

- `experiments/README.md` is the repository experiment registry.
- New scientific programs begin in their own semantic folders.
- New discoveries are appended to the program journal, and new run evidence is
  added to its artifact index.
- Live and sealed artifacts move only after the safe consolidation gate in
  `FOLDER_MIGRATION.md`; cosmetic uniformity never overrides provenance.

## 2026-07-30 — Output-probe receipt completed before observation

**Observed implementation gap**

- The preregistered output-convolution probe required code churn, zero
  fraction, sign inheritance, direct sign flips, and relative master movement
  for every method/group-size point.
- The initial implementation emitted quality and storage metrics but omitted
  those per-projection signatures.

**Correction**

- Before a step-20,000 output-convolution checkpoint existed, the probe was
  extended to compute the exact target-layer signatures for every
  optimal-MSE and absmean group-size point.
- Each receipt now hashes both the probe and shared signature implementation.
- The visual renderer now accepts either sensitive I/O surface, either
  quantizer method, and an arbitrary ordered list of shared-scale group sizes;
  it is no longer hardcoded to input-convolution groups 9 and 2.
- Five targeted tests passed, and a real step-16,000 CPU smoke reproduced the
  known input-convolution ranking while emitting complete signatures.
- Six renderer/probe tests and a four-image visual smoke also passed.

**Evidence boundary**

- The smoke used an already-observed input-convolution checkpoint and exists
  only to validate the receipt machinery.
- No output-convolution result had been observed, and no output recipe was
  selected. The preregistered step-20,000, 22,000, and 24,000 boundaries
  remain unchanged.

## 2026-07-30 — External step-20 output-convolution probe

**Verified**

- The uninterrupted control reached step 20,000 with all 204 quantized
  surfaces started, 203 fully hard, and only `upsample.conv` transitioning.
- Blended ground-truth PSNR was 33.3630 dB. Forcing the output surface hard
  reduced it to 22.7661 dB.
- The preregistered 40-point grid completed with full quality, storage, and
  signature receipts.
- Group size 1 reached 32.1937 dB but used one FP16 scale per weight and is
  excluded as a diagnostic escape hatch.
- Among meaningful shared-scale choices, optimal-MSE group size 2 scored
  29.2690 dB and group size 3 scored 28.7411 dB. They improved over group 9
  by 6.5030 and 5.9750 dB, respectively.
- Group size 2 added 0.04124 whole-model bits per quantized weight versus g9;
  group size 3 added 0.02357. Group size 4 fell to 26.2052 dB.
- Group size 2 beat group size 3 on every fixed crop by 0.30–0.80 dB.
  Visual montages showed that both remove most of g9's green/checkerboard
  corruption, while neither yet matches the ground truth.

**Decision**

- Shortlist optimal-MSE g2 and g3 for matched training from the eventual
  selected input-convolution checkpoint.
- Keep g9 as the matched output control and exclude g1 from training.
- Preregister the exact parent, arms, checkpoints, and full-development rule
  in `studies/OUTPUT_CONV_RESCUE_STUDY_20260730.md`.
- Implement the fail-closed full-development selector before any matched
  output arm exists. It requires exact g9/g3/g2 coverage, one shared fork
  parent and non-output surface map, 204 fully hard surfaces at step 24,000,
  complete paired samples, and the frozen overall/source margins. Three
  synthetic decision-path tests pass.

**Evidence boundary**

- This checkpoint still uses the external input-group-9 trajectory. The probe
  identifies candidate output representations but cannot select one.
- Output selection waits for the sensitive-input winner, matched g9/g3/g2
  training through step 24,000, and the complete development bundle.

## 2026-07-30 — External control paused at a durable boundary

**Verified**

- Five concurrent MPS training contexts had pushed macOS unified-memory swap
  usage above 31 GB.
- The uninterrupted group-9 control had already produced a complete,
  hash-verified step-20,000 checkpoint, hard-started evaluation, signatures,
  output-group grid, and visual comparison.
- Its step-20 checkpoint SHA-256 is
  `54789cbdf282548f80b3b4b30c3236ca777004e0e27c699171732360641e4588`.

**Decision**

- Stop only the superseded external control writer at this durable boundary,
  leaving the four causally matched sensitive-input arms active.
- Preserve the checkpoint and exact command so optional step-22/24 external
  diagnostics can resume later.
- Prioritize memory and MPS scheduling for the matched study that can actually
  select the input recipe.

This is an execution-scheduling change, not an outcome-dependent recipe
change. No checkpoint, selection threshold, data order, or matched arm was
modified.

## 2026-07-30 — Matched sensitive-input step-16 boundary

**Verified**

| Arm | Blended GT | Hard GT | Hard teacher |
|---|---:|---:|---:|
| restarted g9, 0.25x | 36.2838 | 22.8598 | 23.0630 |
| g2, 0.25x | 38.8923 | 29.5366 | 30.2229 |
| g2, 1x | 38.7233 | 30.2363 | 31.0109 |
| g2, 4x | 40.2086 | 33.8791 | 35.3022 |

- At matched 0.25x LR, g2 exceeds g9 by 2.6085 dB blended ground-truth,
  6.6768 dB hard ground-truth, and 7.1599 dB hard teacher PSNR.
- The g2 4x arm exceeds matched g9 by 3.9248, 11.0193, and 12.2392 dB on the
  same metrics.
- First-convolution relative master movement was 0.5646% for matched g9,
  0.5662% for g2 0.25x, 0.6784% for g2 1x, and 0.8690% for g2 4x.
- Only 2–5 first-convolution codes changed from initialization. No arm had a
  direct sign flip.

**Interpretation**

- The near-identical 0.25x master movement and 6.68 dB hard advantage provide
  a second matched boundary supporting the finer g2 representation.
- The additional 4x gain occurs with less than 0.9% input-master movement and
  five changed codes. Finer scales, continuous scale recalibration, and
  downstream compensation are more plausible explanations than broad code
  rewriting.
- The 4x arm is the current development leader, not a selected recipe.
  Step 19 and the complete 3,040-sample development rule remain authoritative.

## 2026-07-30 — Output-development gate implementation frozen before arms

**Verified implementation boundary**

- The output-specific stratified evaluator is separate from the already sealed
  sensitive-input evaluator and verifies its own immutable gate format.
- A fail-closed gate sealer now requires exactly three matched study manifests:
  output group sizes 9, 3, and 2.
- It requires one shared parent checkpoint, training state, data fingerprints,
  bundle, q4 reference, and non-output surface map.
- Every arm must end at step 24,000 with the 24,000-step hard schedule.
- The resulting gate freezes source hashes, paired-development inputs,
  10,000 bootstrap samples, the 0.10 dB overall/source margins, and the rule
  that selects the largest eligible group.
- The manifest cannot authorize promotion and keeps the final heldout locked.
- An output-specific arm sealer requires the selected sensitive-input decision,
  the exact selected step-19 parent hash, one explicit g9/g3/g2 output fork,
  unchanged convolution learning rate, and preserved non-output surface map.
- Future fork receipts persist the exact surface/group change in addition to
  the pre/post training-state hashes, removing ambiguity from the arm record.

**Validation**

- Fourteen targeted fork, arm-sealer, gate-sealer, evaluator, and selector
  tests pass.
- The complete browser-Swin2SR test suite passes: 239 tests.

**Evidence boundary**

- No matched output arm exists yet, so no gate manifest has been materialized
  and no output outcome has been observed.
- After the input recipe is selected, the g9/g3/g2 fork and study manifests
  must be created and then sealed by this implementation before training.

## 2026-07-30 — Step-19 exact-arm coverage tightened before outcome

**Observed protocol gap**

- The sealed step-19 selector validates each supplied arm but does not itself
  require exactly the preregistered 0.25x, 1x, and 4x group-2 set.
- Invoking it with a missing arm could therefore produce a syntactically valid
  but scientifically incomplete shortlist.

**Correction**

- Preserve the already sealed selector and thresholds unchanged.
- Run a fail-closed preflight before that selector. It requires one matched
  group-9 0.25x control, exactly three group-2 arms at 0.25x/1x/4x, one shared
  fork parent, distinct study manifests, identical validation identity, and
  the exact step-19 boundary.
- The preflight hashes every input receipt plus its own implementation and the
  sealed selector. It cannot select a recipe, authorize promotion, or open the
  final heldout.

This is a preregistration-preserving completeness check, not an outcome or
threshold change. Step-19 results had not been observed when it was added.
Three targeted tests and the complete 242-test Swin2SR suite pass.

## 2026-07-30 — Older full-surface runner is not optimizer-transfer evidence

**Verified**

- `run_full_surface_comparison.py` defines `fp32` and `recovered_core` lanes.
- Its scientific question is initialization-path comparison; it does not
  create the preregistered QAT8-continuation versus fresh-teacher AdamW arms.
- The existing optimizer implementation does provide explicit AdamW matrix,
  convolution, and correction groups with betas `(0.9, 0.999)`, epsilon
  `1e-8`, and an exact next-step resume test.

**Decision**

- Reuse the proven optimizer, materialization, paired-evaluation, and artifact
  primitives where their contracts match.
- Do not reuse the older comparison summary as evidence for optimizer
  transfer.
- Build and seal a dedicated optimizer-transfer study after the input and
  output group maps are development-selected. It must compare the exact
  continued QAT8 trajectory with a fresh FP32-teacher AdamW trajectory at
  matched 30,000-step exposure.

## 2026-07-30 — Matched sensitive-input step-17 boundary

**Verified**

| Arm | Blended GT | Hard GT | Hard teacher |
|---|---:|---:|---:|
| restarted g9, 0.25x | 35.3581 | 25.6919 | 26.0348 |
| g2, 0.25x | 38.2242 | 31.6791 | 32.6111 |
| g2, 1x | 39.1545 | 33.0700 | 34.3398 |
| g2, 4x | 40.8341 | 37.0069 | 39.3391 |

- At matched 0.25x LR, g2 exceeds g9 by 2.8661 dB blended ground-truth,
  5.9872 dB hard ground-truth, and 6.5764 dB hard teacher PSNR.
- The g2 4x arm exceeds matched g9 by 5.4760, 11.3150, and 13.3044 dB.
- From step 16 to 17, hard ground-truth PSNR improved by 2.1424 dB for
  g2/0.25x, 2.8337 dB for g2/1x, 3.1278 dB for g2/4x, and 2.8321 dB for
  matched g9.
- First-convolution relative master movement was 0.6047% for g9, 0.6109% for
  g2/0.25x, 0.7900% for g2/1x, and 1.1308% for g2/4x.
- The input convolution changed only 2, 3, 6, and 6 codes respectively, with
  no direct sign flips.

**Interpretation**

- The group-2 representation remains strongly superior under matched
  training, and greater convolution LR is accelerating hard-endpoint recovery
  without broad support churn.
- Step 17 still cannot select an arm. Step 19 and the complete 3,040-sample
  paired development rule remain authoritative.

**Execution boundary**

- Four stopped MPS processes retained enough unified-memory allocation to push
  swap near 25 GB.
- Each process was terminated only after its exact step-17 checkpoint,
  hard-started receipt, and signature receipt existed.
- Termination handlers also emitted later interrupted checkpoints at steps
  17,010–17,019. Those are explicitly excluded. Continuation uses only the
  hash-identified step-17,000 checkpoints listed in the receipts.
- The exact states will continue sequentially to step 19, preserving optimizer
  state, RNG, sampler cursor, data order, and continuous masters while holding
  only one MPS context.

## 2026-07-30 — Step-17 continuation uses the sealed trainer

**Observed provenance risk**

- Before continuation, the exact step-17 checkpoint and study-manifest hashes
  all matched their receipts.
- Five current repository sources differ from the arm-sealed snapshot:
  `surface_model_state.py`, `surface_quantization.py`,
  `surface_run_state.py`, `train_swin2sr_full_surface.py`, and the fork tool.
- The current trainer contains later optimizer/output support and is therefore
  not the byte stream that produced the matched step-17 trajectories.

**Correction**

- A fail-closed resume wrapper now executes the trainer from each arm's exact
  sealed source root rather than the current repository trainer.
- It verifies all 18 manifest-pinned files, preloads the sealed BonsaiQAT8 and
  optimal-ternary modules, and hashes the complete local import closure.
- The observed closure contains 32 local modules. Sixteen unsealed fallback
  dependencies are allowed only because their filesystem mtimes predate the
  original study seal; every path, mtime, and SHA-256 is written into the
  precommit receipt.
- The sealed trainer SHA-256 is
  `bdd267cd4e82b1035146f2aac0123e1c239815cf5c328ad76a150dfff240f513`.
- The wrapper preserves the original direct fork path required by the frozen
  evaluator while adding the exact step-17 checkpoint path and hash as an
  explicit `resume_chain`. No model, optimizer, RNG, sampler, or master-weight
  state is modified.

**Validation and execution**

- Four targeted sealed-resume tests and the complete 246-test Swin2SR suite
  pass.
- A real preflight over the g2/4x arm resolved the full closure and emitted the
  correct step-17 hash and lineage policy.
- Sequential continuation started with the g2/4x arm. Only one MPS context is
  active, and the first resumed validation occurred at exact step 17,001.

## 2026-07-30 — Optimizer-transfer arm sealer implemented before selection

**Verified implementation boundary**

- The sealer requires successful input and output development decisions whose
  checkpoint hashes form one exact parent chain.
- The selected output checkpoint must be the decision's exact step-24 hash,
  contain all 204 quantized surfaces, carry the selected input/output group
  map, use the 24,000-step schedule, and identify QAT8 plus periodic Muon.
- The QAT8 arm continues that exact checkpoint and optimizer state to step
  30,000.
- The AdamW arm starts from the non-quantized FP32 teacher at step zero with
  explicit betas `(0.9, 0.999)`, epsilon `1e-8`, and the same matrix,
  convolution, correction, schedule, data, seed, loss, and surface-map
  controls.
- The sealer snapshots 22 training sources and keeps the final heldout locked.

**Evidence boundary**

- Three targeted contract tests and the complete 249-test Swin2SR suite pass.
- No input or output recipe has yet been selected, so no optimizer-transfer
  manifest has been materialized and neither optimizer arm has started.
- A dedicated full-development evaluator and selector are still required
  before this study can choose a training recipe.

## 2026-07-30 — Research dossier consolidation

**Verified organization**

- `experiments/README.md` now defines one semantic home per scientific
  program and forbids new loose experiment outputs.
- This folder owns the active BackdropKit Swin2SR program and contains its
  journal, current-state boundary, acceptance ledger, artifact index, methods,
  preregistered studies, promotion-policy amendments, and migration record.
- Stable Swin2SR method, study, and policy documents exist only in their
  semantic `methods/`, `studies/`, and `policies/` locations.
- Separate archived dossiers now organize the LLM and Sana 1.58-bit work
  without making either lane part of the active BackdropKit objective.
- All 102 local Markdown links across the registry and three dossiers resolve.

**Provenance decision**

- Active writers, sealed manifests, hashed source snapshots, checkpoints,
  immutable receipts, and historical commands remain at their recorded legacy
  paths. Moving those bytes during training would change experiment identity
  or break reproducibility.
- `../browser_swin2sr/README.md` points back to this semantic home so the
  implementation root cannot be mistaken for a second research program.
- Future synthesis belongs in the semantic dossier. Physical relocation of
  unsealed implementation files waits for the explicit safe-consolidation
  gate in `FOLDER_MIGRATION.md`.

## 2026-07-30 — First exact step-19 input arm and canonical optimizer gate

**Verified input-arm result**

- The group-2, 4x convolution-LR arm resumed from the exact hashed step-17
  state and completed step 19.
- The independent resume verifier confirms exact data/RNG restoration,
  baseline reproduction, the full 32-module import closure, and the original
  fork lineage.
- At step 18 its scheduled model scored 38.5494 dB ground-truth PSNR while
  the hard-started endpoint scored 38.9651 dB.
- At step 19 the input convolution was fully hard. Scheduled and hard-started
  output became identical at 39.1359 dB ground-truth and 42.4191 dB teacher
  PSNR.
- Overall code churn was 2.1990%, zero fraction 34.5998%, relative
  continuous-master movement 3.0198%, and direct sign flips remained zero.

This is one arm, not a recipe selection. The matched g2/1x, g2/0.25x, and
g9/0.25x control must complete before the sealed selector runs.

**Optimizer-transfer protocol correction before outcome**

- Optimizer evaluation now requires a checkpoint-bound canonical
  materialization. Continuous masters are projected once on MPS; the hashed
  dense-hard state is the quality identity and the paired runtime state is the
  later export identity.
- The evaluator rejects a changed checkpoint, materialization, quantization
  contract, training contract, surface count, dense state, or runtime state.
- The QAT8-versus-AdamW study now seals materialization and evaluation
  commands plus its recursively audited 39-file local source closure.
- Its development selector applies the final v7 paired and stratified quality
  margins to the separate development bundle. Neither optimizer is eligible
  merely for lower state or compute cost; it must pass the quality screen.
- The selector records paired evidence against q4 and the other optimizer,
  ranks the exact canonical identities, and keeps final heldout locked.

**Validation**

- Fifteen targeted canonical-state, sealer, evaluator, and selector tests
  pass.
- The complete browser-Swin2SR suite passes: 260 tests.

**Pre-outcome source-closure amendment**

- A recursive local-import audit found that the v2 input development gate
  hashed 16 direct sources but omitted 13 executed dependencies.
- `development-gate-manifest-v3.json` preserves every v2 study identity,
  dataset, q4 reference, bootstrap count, seed, margin, and selection rule.
  Only the creation timestamp and source-hash map differ.
- V3 seals all 30 files in its recursive local closure; no source is missing.
- The not-yet-sealed output training study now covers its complete 34-file
  local closure, its output development gate covers all 31 files including
  the gate builder itself, and the optimizer-transfer study covers all 39.
- These amendments were completed before any full 3,040-sample input
  evaluation, output-arm training, or optimizer-transfer training. They do not
  change an outcome threshold or use final heldout data.

## 2026-07-30 — Second exact step-19 input arm and final handoff closure

**Verified input-arm result**

- The group-2, 1x convolution-LR arm resumed from its exact hashed step-17
  state and completed step 19.
- The independent verifier confirms exact data/RNG restoration, baseline
  reproduction, the 32-module local import closure, and the original fork
  lineage.
- At step 18 its scheduled model scored 36.4004 dB ground-truth PSNR and the
  hard-started endpoint scored 37.2489 dB.
- At step 19 the input convolution was fully hard. Scheduled and hard-started
  output became identical at 38.5289 dB ground-truth and 41.3010 dB teacher
  PSNR.
- Overall code churn was 1.9511%, zero fraction 34.5990%, relative
  continuous-master movement 2.8716%, and direct sign flips remained zero.
- The exact step-19 checkpoint SHA-256 is
  `9f89ae594fa6d101beddee2ad4fe71a4aed8842e80b262d1edb4b9af18ed80fb`.

This is the second of four arms, not a recipe selection. Group-2/0.25x is now
the only MPS writer; the matched group-9/0.25x control follows it. The final
heldout remains closed.

**Final-candidate protocol completed before candidate selection**

- The optimizer-selected artifact preparer revalidates the exact optimizer
  decision, checkpoint, canonical materialization, dense hard state, and
  runtime state before producing a deterministic portable ONNX/gzip identity.
- Its artifact precommit seals the optimizer decision, promotion policy, base
  ONNX, materialization, and all 32 files in the recursive local source
  closure before export.
- The artifact must be no more than 50% of the shipped q4 byte size and must
  already have passed the development quality screen before it becomes
  eligible for one final evaluation.
- The final evaluator rejects development/final data reuse, revalidates every
  selected artifact identity, and seals the exact candidate, final bundle, q4
  reference, policy, and all 18 files in its evaluation closure before
  invoking the final bundle.
- Final metrics do not authorize promotion. The existing promotion gate still
  requires paired and stratified non-inferiority, the material-size gate, and
  all required WebGPU/WASM browser receipts.
- A final authorizer now binds that promotion result to the exact candidate
  manifest, final-evaluation receipt and precommit, q4 reference, promotion
  policy, candidate-specific runtime fixture, and runtime-receipt bundle. It
  verifies the fixture's hard state, expanded ONNX, compressed package, input
  shape, and hash before accepting browser receipts. Partial success cannot
  authorize the model. A failed final candidate is rejected without tuning
  against the final result.

**Validation**

- Seventeen focused old/new final-pipeline tests pass.
- The complete browser-Swin2SR suite passes: 270 tests.
- Recursive import audits report exact closure coverage: 32/32 files for
  artifact preparation, 18/18 for final evaluation, 31/31 for the canonical
  runtime-fixture generator, and 34/34 for the final promotion authorizer.

## 2026-07-30 — Runtime fixture identity policy v8

**Observed acceptance gap**

- V7 bound runtime receipts to the compressed package, expanded ONNX graph,
  browser/backend surface, and harness files.
- It did not hash the expected-output fixture or prove that the fixture's
  expected tensor came from the selected canonical dense hard state.

**Pre-outcome amendment**

- `promotion_policy_glyphsafe_v8.json` supersedes v7 without changing quality,
  q4 identity, final data, artifact-size, runtime-surface, input-shape,
  numerical, compression, or latency requirements.
- The browser harness now hashes the complete fixture and records its embedded
  canonical hard-state SHA-256 on every success and failure receipt.
- Runtime merging rejects mixed fixture or hard-state identities. Promotion
  rejects missing fixture hashes or a fixture derived from another hard state.
- No optimizer-transfer manifest, final candidate, final evaluation, or final
  runtime result existed when this amendment was registered.

**Validation**

- Twenty-eight focused promotion, policy, runtime-merge, and optimizer-gate
  tests pass.
- The complete browser-Swin2SR suite passes: 276 tests.
- Five JavaScript runtime-surface and gzip-decompression tests pass.
- A real Chromium infrastructure run emitted the new fixture SHA-256
  `d1a12e5f57f4c1fd8910782f6eb4305686d187ae886c5d6cb3a20f9e697fa381`
  and canonical hard-state SHA-256
  `417551d993913ee2d6ebe721f637da12fb7b7c599ba5e9a0efa6f604c398b0ea`
  on both WebGPU and WASM receipts.
- WebGPU passed with maximum error 9.56e-5, 3.43 s session creation, and
  100.9 ms worst warm inference. WASM passed with the same maximum error,
  3.78 s session creation, and 1.48 s worst warm inference at
  `[1,3,128,128]`.
- That browser run used the older 4,807,898-byte uncompressed infrastructure
  model. It validates the v8 harness and fixture binding only, not candidate
  quality, material size, compressed deployment, or promotion.

## 2026-07-30 — Third exact step-19 input arm

**Verified**

- The group-2, 0.25x convolution-LR arm resumed from its exact step-17 state
  and completed step 19.
- The independent verifier confirms exact data/RNG restoration, baseline
  reproduction, the 32-module import closure, and original fork lineage.
- At step 18, scheduled ground-truth PSNR was 36.8138 dB and the hard-started
  endpoint was 34.8254 dB.
- At step 19 the input surface was fully hard. Scheduled and hard-started
  outputs became identical at 36.7314 dB ground-truth and 38.6603 dB teacher
  PSNR.
- Overall code churn was 1.8859%, zero fraction 34.6001%, relative master
  movement 2.8877%, and direct sign flips remained zero.
- The exact checkpoint SHA-256 is
  `c5906da5b3f982f567796adf36824e5e6f8252ec9f1a46d71b543e26333c25ac`.

**Interpretation**

- On the four-image checkpoint diagnostic, 0.25x trails 1x by 1.7975 dB
  ground-truth and 2.6407 dB teacher PSNR, and trails 4x by 2.4045 dB and
  3.7588 dB respectively.
- This diagnostic does not select the recipe. The matched group-9 control and
  full 3,040-sample development gate remain authoritative.

## 2026-07-30 — Exact step-19 input gate closed

**Verified**

- The matched restarted group-9, 0.25x control completed its sealed exact
  continuation from step 17,000 to step 19,000.
- The independent verifier confirms exact data/RNG restoration, baseline
  reproduction, the 32-file local import closure, and the shared fork lineage.
- At step 18 the control's scheduled ground-truth PSNR was 34.0816 dB and its
  hard-started endpoint was 29.9998 dB.
- At step 19 the input surface was fully hard. Scheduled and hard-started
  output became identical at 33.4394 dB ground-truth and 34.3131 dB teacher
  PSNR.
- Overall code churn was 1.9035%, zero fraction 34.6189%, relative
  continuous-master movement 2.9116%, and direct sign flips remained zero.
- The exact control checkpoint SHA-256 is
  `eca1f2e9dec3937dc206964392b2c287a9ea4957463df059abe2cc55d28cae9f`.
- `step19-exact-arm-coverage.json` verifies exact step-19 coverage for all
  three group-size-2 arms against the matched control.
- The preregistered checkpoint selector shortlisted only group size 2 with
  the 4x convolution-learning-rate multiplier. Its fully hard 39.1359 dB
  ground-truth and 42.4191 dB teacher PSNR exceeded the control by 5.6965 dB
  and 8.1060 dB, respectively. The 1x and 0.25x arms were outside the frozen
  0.10 dB margin of the best hard ground-truth result.

**Decision**

- Only the 4x arm proceeds to the sealed 3,040-sample development comparison
  against the matched group-9 control.
- This is checkpoint eligibility, not final input-recipe selection or model
  promotion. The paired overall and per-source development gate remains open,
  and the final heldout remains locked.

## 2026-07-30 — Sensitive-input development recipe selected

**Verified**

- The matched group-9 control completed all 3,040 development samples at
  30.1769 dB mean ground-truth PSNR, 35.0410 dB edge PSNR, and 33.2290 dB
  teacher PSNR.
- The checkpoint-eligible group-2/4x arm completed the identical 3,040 samples
  at 33.2596 dB mean ground-truth PSNR, 37.0443 dB edge PSNR, and 39.1035 dB
  teacher PSNR.
- Its paired ground-truth improvement over the matched control was
  +3.0826 dB with a 95% bootstrap interval of
  `[+3.0036, +3.1621]` dB. Its worst source-level improvement was
  +2.7765 dB.
- The frozen overall and every-source selection gates passed. Group size 2
  with a 4x convolution-learning-rate multiplier is the selected input
  recipe, bound to checkpoint SHA-256
  `07ebb2fcc6e2fc12cefcca323f94e00f559876c6991adcf3314487a6ad04365f`.
- Against shipped q4, the selected input state remains 1.6641 dB worse in
  paired mean ground-truth PSNR; the 95% interval is
  `[-1.7189, -1.6093]` dB and its worst source deficit is 2.0560 dB.

**Interpretation**

- The sensitive-input representation recovered a large, stable degradation
  across every development source, but it did not close the q4 quality gap.
- This is a successful surface-recipe selection, not evidence of overall
  non-inferiority.

**Protocol correction**

- The generated input-decision receipt carries a stale prose
  `required_next_gate` value that names optimizer transfer directly.
- The preregistered output-convolution rescue study predates the outcome and
  requires output g9/g3/g2 training before optimizer transfer. That study is
  the authoritative next gate. The stale prose does not alter metrics,
  hashes, thresholds, or selection.

**Next gate**

- The output g9/g3/g2 studies are sealed from the exact selected input
  checkpoint. All three share the same input decision, corpus, development
  bundle, q4 reference, and 34-file source closure; a read-only audit found
  zero source drift.
- Train the three arms serially through step 24,000, then apply the frozen
  full-development output selector. The final heldout remains locked.

**Pre-outcome diagnostics amendment**

- The original output-study document requires hard-started fixed-crop metrics
  and whole-model ternary signatures at steps 20,000, 22,000, and 24,000, but
  the sealed arm commands only train/checkpoint and then run the frozen
  step-24 full-development selector.
- Before the first arm reached step 20,000, a separate diagnostics plan sealed
  the evaluator, signature analyzer, validation files, arm manifests, steps,
  and execution boundary.
- These diagnostics will run serially after the frozen output decision and
  before optimizer-transfer training. They are diagnostic-only and cannot
  alter the already-frozen output selector.

## 2026-07-30 — Output-g9 control reached frozen step 22,000

**Verified interim trajectory**

- The matched output-group-9 arm reached its preregistered step-22,000
  checkpoint with input group size 2, output group size 9, the exact selected
  step-19 parent, and the 24,000-step sensitive-I/O schedule intact.
- The checkpoint SHA-256 is
  `f4517be9f317bdc34f7122f4c5b35898ddfeef41d498fe9a44d87511881872fc`;
  its identity fields and all logged losses were reloaded and verified.
- At 64.81% output-surface transition, scheduled fixed-crop ground-truth PSNR
  was 37.3898 dB and teacher PSNR was 39.7939 dB.
- Hard-shadow ground-truth loss improved from 0.0487 at step 20,000 to
  0.0202 at step 22,000. The change is not monotonic at every 500-step
  interval, but the hard branch improved materially over the preregistered
  2,000-step window.

**Interpretation**

- The hard-shadow objective is adapting the forced-hard branch while the live
  blended path absorbs the transition.
- This is trajectory health evidence only. The output control remains
  partially blended, and no group-size recipe or q4 comparison can be selected
  until all three arms are fully hard and evaluated over all 3,040 development
  samples.

## 2026-07-30 — Output-g9 control completed all-hard development evaluation

**Verified**

- The matched output-group-9 arm reached step 24,000 with all 204 quantized
  surfaces fully hard, zero transitioning surfaces, and the selected
  input-group-2 recipe intact.
- Its step-24 checkpoint SHA-256 is
  `73dfe645337ea1f715b3a797f2c65a9b6c235333386d6272b01bad517251111f`.
  The evaluator's canonical state SHA-256 is
  `645a776ccc706f69a7d198a36d7ae0fb3adf55edbcc450344d8cdcf7f4d137a1`.
- On all 3,040 development samples, the all-hard control scored 31.7858 dB
  ground-truth PSNR, 35.1817 dB edge PSNR, and 35.9669 dB teacher PSNR.
- Against the exact frozen q4 development reference, the deficits are
  3.1379 dB ground-truth, 3.2515 dB edge, and 10.9659 dB teacher PSNR.
- The report covers the complete bundle, remains development-only, does not
  authorize promotion, and keeps the final heldout locked.

**Interpretation**

- Output group size 9 is a confirmed all-hard quality bottleneck after
  matched training, not merely a transient blended-versus-hard discrepancy.
- It is the required control, not an eligible recipe. Output g3 and g2 must
  beat this exact state and then survive the frozen overall and every-source
  comparison.

**Next gate**

- The output-g3 rescue arm started automatically from the same selected input
  checkpoint and sealed training state. It is the only active MPS writer.

## 2026-07-30 — Output-g3 reached frozen step 20,000

**Verified interim trajectory**

- Output-g3 reached its preregistered step-20 checkpoint from the exact same
  parent, optimizer state, sampler path, schedule, and transition fraction as
  output-g9. Its checkpoint SHA-256 is
  `3733ee958c0d407cdd5e48636ad1a06a92c41df7cd9f4a87c738d84a3e3ae05e`.
- At the matched step-19,500 boundary, g3 exceeded g9 by 0.7235 dB scheduled
  ground-truth and 1.1255 dB teacher PSNR. Its hard-shadow ground-truth,
  teacher, and edge losses were lower by 0.01985, 0.02063, and 0.03964.
- At step 20,000 and 18.52% output transition, g3's scheduled score was
  38.7352 dB ground-truth and 42.1272 dB teacher PSNR, trailing the g9
  scheduled path by 0.6071 and 0.7743 dB.
- The forced-hard branch still favored g3: ground-truth, teacher, and edge
  losses were lower than g9 by 0.01372, 0.01583, and 0.02853.

**Interpretation**

- G3 improves the representation being optimized—the forced-hard branch—but
  its blended path is not uniformly better during early transition.
- This is neither a win nor a failure. The study selects only from fully hard,
  complete-bundle evidence at step 24,000.

## 2026-07-30 — Output-g3 reached frozen step 22,000

**Verified interim trajectory**

- Output-g3 reached its preregistered step-22 checkpoint with the matched
  lineage and group mapping intact. The checkpoint SHA-256 is
  `2647f5203d2627661cb8068cb3dc3cb117c7be4e7c08508bba69f2383565f83e`.
- At 64.81% output transition, g3 scored 38.9969 dB scheduled ground-truth
  PSNR and 42.3239 dB teacher PSNR.
- Against matched g9 at the same step, those scheduled scores are higher by
  1.6071 and 2.5300 dB.
- The forced-hard ground-truth losses are effectively tied: g3 is 0.00076
  worse. G3's forced-hard teacher loss is 0.00002 better and its edge loss is
  0.00578 better.

**Interpretation**

- G3 retains a large blended-path advantage through the preregistered
  step-22 boundary while its forced-hard branch is approximately tied on
  ground-truth and teacher loss and better on edges.
- The comparison is still diagnostic. The surface remains partially blended,
  and only fully hard complete-bundle reports can select the output recipe.

## 2026-07-30 — Data-scale hypothesis recorded without changing sealed runs

**Verified**

- The current training roots contain 206,500 distinct image files: 104,500
  UI/general images and 102,000 glyph-safe multilingual UI/text images.
- The full 30,000-step schedule at batch size 8 consumes 240,000 draws, about
  1.16 balanced epochs. Merely adding one million files without increasing or
  redesigning the sampling budget would leave most of them unseen.
- The current online degradation is bicubic x2 plus optional JPEG. It does not
  cover many browser, display, image-codec, or product-video degradations.
- Swin2SR's paper reports competitive results using DIV2K and Flickr2K even
  against a one-million-image challenge entry, while a binary super-resolution
  study reports one million training iterations over the much smaller
  DIV2K/Flickr2K corpus. These results distinguish training-pair count from
  unique-image count.
- PrismML's public Ternary Bonsai announcement and whitepaper do not disclose
  a dataset size or training recipe sufficient to verify that a large corpus,
  rather than another method choice, caused its reported quality.

**Decision**

- Preserve the current sealed output and optimizer studies exactly.
- Treat a one-million-pair corpus as a high-value next experiment, not as an
  assumed cure. Isolate clean-data scale from degradation diversity with a
  four-arm development-only ablation after the current model recipe is frozen.
- Keep screenshots, product surfaces, multilingual typography, and
  product-demo video frames dominant. Portraits are a bounded robustness slice
  because BackdropKit is not portrait-first.
- Prefer deterministic online low-quality generation from licensed,
  high-quality sources. Add realistic still-image, browser-resampling, and
  video-codec degradations while retaining a clean bicubic anchor.
- Keep development/final isolation, exact and near-duplicate filtering,
  source revisions, licenses, hashes, sampler state, and repeated-seed evidence
  as hard gates.

**Study**

- `studies/DATA_SCALE_AND_DEGRADATION_STUDY_20260730.md`

## 2026-07-30 — Output-g3 completed all-hard development evaluation

**Verified**

- Output-g3 reached step 24,000 with all 204 quantized surfaces fully hard,
  zero transitioning surfaces, input group size 2, and output group size 3.
- The evaluated final-state SHA-256 is
  `597632ba40db57bae5f7b8b710c89e3488a89746a561598154e1ebc93506e69c`.
- On all 3,040 development samples, g3 scored 32.8170 dB ground-truth,
  36.3902 dB edge, and 38.1838 dB teacher PSNR.
- Against the matched all-hard output-g9 control, g3 improves ground-truth by
  1.0312 dB, edge by 1.2085 dB, and teacher PSNR by 2.2170 dB.
- Against the exact frozen q4 development reference, g3 still trails by
  2.1067 dB ground-truth, 2.0430 dB edge, and 8.7490 dB teacher PSNR.
- The report is complete-bundle, development-only, and keeps the final heldout
  locked.

**Interpretation**

- Smaller output groups materially recover all-hard quality, validating the
  sensitive-output hypothesis, but g3 alone does not close the q4 gap.
- No output recipe is selected yet. Output-g2 has started from the exact same
  parent and must finish before the frozen paired overall/every-source selector
  runs.

## 2026-07-30 — Output-g2 reached frozen step 20,000

**Verified interim trajectory**

- Output-g2 reached its preregistered step-20 checkpoint with input and output
  group sizes 2, convolution learning-rate multiplier 4, seed 20260730,
  surface step 20,000, and exact sampler cursor 160,000.
- The checkpoint SHA-256 is
  `fedd5384748ce320feb8c1a8e93a59511860db4be1b76492614b3c2310df4ed4`.
- At the matched 18.52% output transition, g2 exceeds g3 by 0.7390 dB
  scheduled ground-truth and 0.9678 dB teacher PSNR.
- G2's forced-hard ground-truth, teacher, and edge losses are lower than g3
  by 0.00548, 0.00588, and 0.00492. Against g9, they are lower by 0.01921,
  0.02172, and 0.03345.

**Interpretation**

- G2 improves both the blended and forced-hard paths at the first sealed
  output-transition checkpoint. This supports continuing the arm.
- The result remains trajectory evidence only. The study selects from the
  all-hard 3,040-sample reports and may prefer g3 when its quality is within
  the frozen 0.10 dB overall/every-source margin because g3 has fewer scales.

## 2026-07-30 — Output-g2 reached frozen step 22,000

**Verified interim trajectory**

- Output-g2 reached its preregistered step-22 checkpoint with input/output
  group sizes 2, convolution learning-rate multiplier 4, seed 20260730,
  exact sampler cursor 176,000, and the 24,000-step schedule intact.
- The checkpoint SHA-256 is
  `b1a6ab4d49280c6edf93770937cfbb0de7026024a09a0330a0aaf3a8972276a2`.
- At the matched 64.81% output transition, g2 trails g3 by 0.5664 dB
  scheduled ground-truth and 0.6752 dB teacher PSNR.
- G2's forced-hard ground-truth, teacher, and edge losses are lower than g3
  by 0.00502, 0.00527, and 0.00625. Against g9, they are lower by 0.00426,
  0.00529, and 0.01202.

**Interpretation**

- The blended path remains schedule-sensitive, while the forced-hard branch
  consistently favors g2. This is exactly why the study does not select from
  transitional scheduled PSNR.
- Continue unchanged to the fully hard step-24 checkpoint and complete
  3,040-sample development report. No output recipe is selected yet.

## 2026-07-30 — Output-g2 selected by frozen development gate

**Verified**

- Output-g2 reached step 24,000 with all 204 quantized surfaces fully hard,
  zero transitioning surfaces, input and output group sizes 2, sampler cursor
  192,000, and the sealed convolution learning-rate multiplier of 4.
- The selected final checkpoint SHA-256 is
  `ad5e76db8c378c16da689a4cf37d96eb6e204f7b9de543199efc103c0d371469`.
- On all 3,040 development samples, g2 scored 32.8967 dB ground-truth,
  36.5558 dB edge, and 38.2194 dB teacher PSNR.
- Against the matched all-hard output-g9 control, g2 improves ground-truth by
  1.1109 dB, edge by 1.3741 dB, and teacher PSNR by 2.2525 dB. The paired
  bootstrap 95% interval for ground-truth improvement is
  `[+1.0666, +1.1561]` dB.
- G2 exceeds g3 by 0.0797 dB ground-truth, 0.1656 dB edge, and 0.0355 dB
  teacher PSNR. G3 failed the frozen every-source 0.10 dB margin because its
  multilingual synthetic-UI source was 0.1069 dB below g2.
- Against the exact shipped-q4 development reference, g2 remains behind by
  2.0270 dB ground-truth, 1.8774 dB edge, and 8.7134 dB teacher PSNR. Its
  paired ground-truth 95% interval is `[-2.0935, -1.9618]` dB.
- The frozen selector therefore chose output group size 2 and preserved the
  final heldout lock. It did not authorize model promotion.

**Decision**

- Freeze input/output group size 2 as the representation recipe for the next
  study. Do not reopen the g9/g3/g2 selection from later diagnostics.
- Proceed to the preregistered full-surface QAT8-versus-AdamW optimizer
  transfer study.
- The remaining q4 deficit makes the proposed data-scale/degradation ablation
  scientifically relevant if optimizer transfer does not close the gap. It
  does not prove that corpus scale alone is causal.

## 2026-07-31 — Selected g2 shipped as an explicit Ternary Lite tier

**Verified**

- The user explicitly authorized the selected all-hard g2 development
  candidate as a product tier despite its frozen q4 deficit. BackdropKit names
  it **Ternary Lite** and keeps the shipped q4 model available as **Quality**.
- The product package is 683,547 bytes with SHA-256
  `d58ba816c4ac5cd3db7edd3c61068619695d233f7c4d5dd2984da52b7bde2de9`.
  It expands locally to a 4,846,332-byte ONNX graph with SHA-256
  `8776ce3a6a4f1da8dcea1bba443ff5798bf1e0aeeca906f7135b045f5d2f334e`.
- A canonical 128x128 WASM run reproduced the materialized CPU reference with
  maximum absolute error 0.0002941 and mean absolute error 0.00003950. The
  measured single-thread inference times were 1.7124, 1.7119, and 1.7117
  seconds after a 5.381-second session creation.
- BackdropKit commits `18dbfc065` and `9a996e648` implement integrity-checked
  compressed loading, WebGPU with SIMD/threaded-WASM fallback, pure pixel
  preprocessing without an OffscreenCanvas dependency, silent artifact and
  session warming, bounded input sizes, worker reclamation, and Cloudflare
  cross-origin isolation headers.
- The exact archived commit passed deterministic tests, production build,
  Cloudflare routing tests, real browser WebGPU inference, and deliberately
  forced CPU inference before deployment.
- Cloudflare version `9e77edd4-74f2-4d07-b67d-fdb7ee8ca3a3` is live. Three
  repeated public downloads returned 683,547 bytes and the exact package hash.
  The production document returned
  `Cross-Origin-Embedder-Policy: credentialless`,
  `Cross-Origin-Opener-Policy: same-origin`, and
  `Origin-Agent-Cluster: ?1`.
- Live post-prewarm product runs completed in 1,249 ms on WebGPU and 720 ms
  with WebGPU deliberately disabled, both rendering the expected sharpened
  focus result.

**Interpretation**

- This is a useful, explicitly labeled size/speed tier, not scientific
  promotion over q4. The q4 non-inferiority gate remains open and q4 remains
  the quality tier.
- Compressed ternary weights plus local graph expansion give a practical
  browser delivery path now. They do not yet prove a custom packed ternary
  arithmetic kernel faster than dense ONNX execution.

## 2026-07-31 — QAT8 optimizer continuation regressed

**Verified**

- The sealed QAT8 continuation reached step 30,000 and completed the full
  3,040-sample development evaluation.
- It scored 32.6162 dB ground-truth, 36.1223 dB edge, and 37.6409 dB teacher
  PSNR, with a 4.5147 dB mean gain over bicubic.
- This is below the selected step-24,000 g2 result of 32.8967 dB
  ground-truth, 36.5558 dB edge, and 38.2194 dB teacher PSNR.
- The matched fresh AdamW arm started automatically under the sealed study.

**Decision**

- Do not promote the QAT8 continuation.
- Complete the sealed AdamW arm and use the preregistered optimizer decision
  instead of selecting from transient fixed-crop peaks.

## 2026-07-31 — QAT8 regression localized; PSNR-aligned objective prepared

**Verified**

- Exact sample identities align across selected g2, the step-30,000 QAT8
  continuation, and shipped q4 for all 3,040 development samples.
- QAT8 versus selected g2 changed paired mean ground-truth PSNR by
  -0.2804 dB, edge PSNR by -0.4335 dB, and teacher PSNR by -0.5785 dB.
- Only 25.3% of samples improved in ground-truth PSNR. The highest-quality
  selected-g2 decile lost 0.7710 dB, while the lowest-quality decile lost
  0.0867 dB.
- The largest edge regressions cluster in Ol Chiki, Thai, Tibetan, Myanmar,
  and Arabic scripts. RICO lost 0.1154 dB ground-truth; multilingual
  synthetic UI lost 0.3614 dB.
- Selected g2 versus q4 remains weakest on RICO (-2.4672 dB) and on several
  text-heavy scripts, including Ol Chiki, Thai, Traditional Han, Lao,
  Hebrew, Tifinagh, Tibetan, and Devanagari.
- Mean per-sample PSNR is exactly an affine transform of negative mean
  per-sample log MSE. The current Charbonnier teacher and ground-truth terms
  optimize a different robust pixel objective.
- The q4 development reference has geometric ground-truth MSE
  0.00032184 and geometric teacher MSE 0.00002026. These values motivate
  frozen, rounded stability floors of 0.00032 and 0.00002 for a bounded
  log-MSE probe.
- A `psnr_aligned` pixel objective is now implemented but not selected. It
  uses `sqrt(floor) / log(2) * log1p(per_sample_mse / floor)`, matching the
  RMSE scale at the frozen floor and balancing gradient norms for reciprocal
  error scales around that floor.
- Forty-two targeted objective, decision, checkpoint-fork, curriculum, recovery, and
  resume-contract tests pass. Legacy Charbonnier checkpoints retain exact
  resume semantics.
- A dry fork of the exact selected g2 step-24,000 checkpoint preserved hashes
  for model, optimizer, sampler/run state, and quantization across all 204
  surfaces while changing only the declared pixel objective. This gives the
  next study a matched one-variable starting point.

**Interpretation**

- The evidence rejects another undirected continuation of the same QAT8
  objective. It does not yet prove that objective mismatch caused the
  regression.
- The stabilized log-MSE path is a causal probe, not a new default. Its q4
  derived floors are frozen from development evidence; final heldout remains
  closed.

**Decision**

- Let the already sealed AdamW arm finish unchanged.
- Next run a matched, development-only objective study from the same selected
  all-hard g2 state, with identical optimizer, sampler, schedule, and budget.
  Compare the existing Charbonnier control against the stabilized
  PSNR-aligned objective; do not combine this change with a new optimizer.

**Preregistration**

- The two-arm probe is sealed at
  `experiments/browser_swin2sr/pixel_objective_probe_20260731` with manifest
  SHA-256
  `d4df42f8b1022458afc680b9ad00cf1c957eb4954e502fabc916414e05208d03`.
- Both arms fork selected g2 checkpoint
  `ad5e76db8c378c16da689a4cf37d96eb6e204f7b9de543199efc103c0d371469`
  at step 24,000 and continue to step 26,000 with the exact same QAT8
  optimizer state, sampler state, source weights, schedule, seed, batch size,
  and 206,500-image corpus.
- The explicit Charbonnier fork SHA-256 is
  `8ea7d8d7bdb9797ac9993c94ec8963fe4106ca2cb0183c3046c2755f9d9e36c9`;
  the PSNR-aligned fork SHA-256 is
  `c3cf4cff5313af26b0354372ad3603c42ad2704ccf6d4e4f8c1a920fd18d6ce2`.
  Their model, optimizer, run-state, and quantization hashes are identical.
- Forty-three training/evaluation source files are copied and hashed under
  the sealed source root with zero missing or drifted files.
- Both exact sealed training commands passed CPU preflight: each resumes at
  surface step 24,000 with 204 surfaces, 977,604 ternary values, the same
  group-size-2 input/output map, QAT8 optimizer, and sensitive-I/O schedule.
- PSNR alignment is selected only if the paired ground-truth 95% bootstrap
  lower bound is above 0 dB, mean edge and teacher deltas are each at least
  -0.10 dB, and every source is at least -0.10 dB. Otherwise the
  Charbonnier control remains selected.
- This probe has not started. It remains queued behind the already running
  sealed AdamW study so concurrent MPS load cannot contaminate timing or
  trajectory evidence.

## 2026-07-31 — Surface-allocation diagnostics preregistered

**Verified negative results**

- Restoring the selected g2 checkpoint's continuous master weights is not a
  dense quality upper bound. On the two-sample diagnostic, the complete
  continuous-master state was 10.0128 dB below its hard ternary projection
  against ground truth, 9.1338 dB lower on edges, and 15.0786 dB lower against
  the teacher. A separate parametrized-versus-plain state check agreed within
  0.00000262 maximum absolute weight error. The collapse is therefore a real
  property of STE-adapted masters, not a materialization error.
- A first-order task-gradient estimate was compared with exact one-surface
  group-2 changes for all 204 surfaces on the same eight development samples.
  Its correlation with measured ground-truth PSNR change was -0.6167 for the
  Charbonnier objective and -0.5893 for the PSNR-aligned objective. The five
  surfaces ranked highest by the PSNR-aligned gradient all regressed in the
  exact intervention, with a mean change of -0.4756 dB.
- Do not use continuous-master restoration or the current first-order gradient
  score for sensitive-layer allocation. The discrete projection changes are
  too large and nonlinear for those shortcuts at this checkpoint.

**Measured screen and frozen confirmation**

- Exact one-at-a-time group-2 interventions found 82 of 204 surfaces with a
  positive eight-sample ground-truth change and 19 above +0.02 dB. These are
  exploratory development effects, not additive gains or promotion evidence.
- A deterministic screen retained only surfaces with positive ground-truth and
  edge changes, teacher change at least -0.10 dB, and positive scale cost. It
  ranked by ground-truth dB per added KiB and greedily selected 17 surfaces
  using 64,616 of a frozen 65,536-byte group-2 budget.
- The larger confirmation is sealed at
  `experiments/browser_swin2sr/surface_group_confirmation_20260731` with
  manifest SHA-256
  `b43abda5c05fcdbc617cde940eebe5bb1556f4c04f6533baf975d96afc75adb5`.
  It compares group sizes 2, 3, and 6 on 64 balanced development samples with
  5,000 deterministic paired bootstrap resamples per intervention.
- An individual surface/group passes only when ground-truth lower-95% change
  is above 0 dB, mean edge change is at least -0.02 dB, and mean teacher change
  is at least -0.10 dB. Any resulting surface map must then be evaluated
  jointly under the same byte ceiling before recovery training. Final heldout
  remains locked and this diagnostic cannot authorize promotion.

**Concurrent optimizer receipt**

- The sealed fresh AdamW arm remains healthy and unchanged. It reached step
  6,500/30,000 with exact sampler/config logging and checkpoints through step
  6,000. Its fixed validation ground-truth PSNR was 44.3338 dB at step 6,500;
  no optimizer decision has been made.

## 2026-07-31 — Confirmed small mixed-group map admitted to recovery

**Frozen individual confirmation**

- The preregistered 17-surface confirmation completed on 64 balanced
  development samples for group sizes 2, 3, and 6. The eight-sample effects
  did not broadly replicate: only two group-2, three group-3, and one group-6
  interventions passed the frozen individual gate.
- The deterministic efficiency selector chose three unique surfaces using
  only 4,256 added scale bytes: stage-3/layer-2 attention output at group size
  3, and the first continuous-position-bias MLP in stage-1/layers 0 and 1 at
  group size 2.
- The surface decision SHA-256 is
  `0f878354cfe172e60a923821b9b2df5f1c9bcc0e5f20a12b794fd7291ecdec70`.

**Disjoint joint confirmation**

- The exact three-surface map was evaluated jointly on 256 balanced
  development samples after excluding every one of the 64 individual
  confirmation samples.
- It improved mean ground-truth PSNR by 0.02217 dB with paired 95% interval
  `[+0.01485, +0.02967]` and 169/256 sample wins. Edge PSNR improved by
  0.03112 dB with interval `[+0.02407, +0.03837]`. Teacher PSNR changed by
  -0.00651 dB, inside the frozen -0.10 dB guard.
- The joint report passed every preregistered gate and has SHA-256
  `c7f4568c7e44e7feb0cabe72b4f9e27ed0b859787a1f0132d4ab6319c5cfd89f`.
  This admits a recovery study only; it is not a q4 comparison or promotion.

**Next matched study**

- A control-versus-refined recovery protocol was frozen before the optimizer
  or pixel-objective winner exists. It uses a q4-eligible optimizer winner if
  one emerges; otherwise it uses the frozen pixel-objective winner. Both arms
  continue the same parent for exactly 2,000 steps, differing only by the
  selected three-surface group map.
- Protocol SHA-256 is
  `7baf52450360012a7e1227693e1be87e475013a255c77443e8b793c8bd850c72`.
  It freezes 46 source files, full 3,040-sample development evaluation,
  10,000 paired bootstrap resamples, edge/teacher/every-source guards, and the
  65,536-byte ceiling. It is queued behind the active MPS studies.
- Thirty-three focused tests for the diagnostic, bootstrap, selection,
  checkpoint-fork, recovery-sealing, and quantization contracts pass.
- AdamW continued independently to step 8,000/30,000 with checkpoints and no
  runtime error. No upstream optimizer or objective recipe is selected yet.

## 2026-07-31 — Broader surface map rejected by full teacher guard

**Broader development search**

- A second intervention screen covered all 204 surfaces at group sizes 3 and
  6 while excluding the 320 samples used by the earlier screen and
  confirmation. Forty candidates advanced to a frozen 64-sample confirmation.
- Eleven new surfaces passed their individual gates. Combined with the
  previously selected three surfaces, the expanded map covered 14 surfaces and
  added 17,810 scale bytes. Its exact map SHA-256 is
  `5dcbaa9da2de7d776f5f4968563505c9bf2a97124df1a287be7c27eea6a51ce9`.
- On 256 further disjoint development samples, the expanded map improved g2 by
  0.13881 dB ground-truth with paired lower-95% bound +0.09641 dB and by
  0.20004 dB on edges. Teacher agreement changed by -0.09218 dB, inside the
  joint-screen mean guard. This admitted a full-development check only.

**Frozen full-development decision**

- The exact canonical expanded hard state completed all 3,040 development
  samples. It scored 33.0338 dB ground-truth, 36.7574 dB edge, and 38.1021 dB
  teacher PSNR.
- Relative to selected g2, paired ground-truth changed by +0.13710 dB with 95%
  interval `[+0.12358, +0.15132]`; edge changed by +0.20162 dB. Both RICO
  (+0.04861 dB) and multilingual synthetic UI (+0.18048 dB) improved.
- Teacher agreement changed by -0.11725 dB. This fails the preregistered
  -0.10 dB mean guard even though ground-truth, edge, per-source, and storage
  gates passed. The decision receipt SHA-256 is
  `34a6f2a47128c6674398c250e37818590bfe3f5f4222135ac5cdc3525136374b`.
- The expanded map still trails shipped q4 by 1.88985 dB mean ground-truth
  PSNR, with paired 95% interval `[-1.95784, -1.82322]`.

**Decision**

- Do not relax the teacher guard and do not use the expanded map in recovery.
  Retain it as diagnostic evidence of nonlinear surface interactions.
- Use the previously confirmed three-surface map in the already frozen matched
  recovery protocol. A persistent runner now waits for the optimizer and
  pixel-objective decisions, then executes matched control and refined +2,000
  step arms, canonical materialization, full development evaluation, and the
  frozen paired decision.
- The fresh AdamW arm remained healthy and reached step 18,000/30,000 with an
  exact step-18,000 checkpoint. Final heldout remains locked.

## 2026-07-31 — Zero-parameter residual gain rejected

**Protocol**

- A development-only CPU probe tested whether scaling the ternary model's
  learned residual over bicubic could recover quality without new weights or a
  larger browser artifact. Gains `0.90`, `0.95`, `1.00`, `1.025`, `1.05`,
  `1.075`, `1.10`, and `1.15` were frozen before results.
- The protocol used 64 samples per source for selection and a disjoint 128 per
  source for confirmation, with only two CPU threads. Its SHA-256 is
  `224752636b68280d6fe79f3fab6f27ec19d7bf91d76922ce8e274214d44383b1`.

**Verified result**

- No non-identity gain achieved a positive ground-truth lower-95% bound on the
  128-sample screen. The frozen selector retained gain `1.0`.
- Gain `0.95` improved teacher agreement by 0.05000 dB but reduced
  ground-truth by 0.02252 dB. Gain `1.025` changed ground-truth by -0.00454 dB
  and split by source: +0.01255 dB on multilingual synthetic UI but -0.02162
  dB on RICO. Larger gains increasingly damaged RICO and teacher agreement.
- The identity confirmation remained 2.18022 dB behind q4 on its 256 samples.
  The result receipt SHA-256 is
  `3a09cb00deabc5744a4879055462c4cd988e0f8c1ea8496c4795d445de971a60`.

**Decision**

- Do not add residual-gain post-processing and do not introduce a source or
  content classifier to exploit the contradictory slice effects. Continue
  improving the actual ternary training trajectory.
- Final heldout remains locked.

## 2026-07-31 — Post-training scale codebooks rejected

**Honest storage boundary**

- The selected g2 packed tensor trunk contains 977,604 ternary values. Packing
  20 trits per 32-bit word costs 1.60094 bits/value, but its 35,550 FP16
  per-group scales add 0.58183 bits/value. The honest packed trunk is therefore
  2.18277 bits/value before raw biases and normalization tensors, not 1.58.
- A deterministic weighted one-dimensional codebook compressed each tensor's
  scales independently. The fit weighted every scale by the number of nonzero
  ternary weights it controls and stored codebook centers as the FP16 values a
  browser runtime would actually decode.
- The protocol excluded all 384 development samples used by the residual-gain
  study, then froze a 64-sample balanced screen and a separate 128-sample
  confirmation. Protocol SHA-256 is
  `0617df0520e6547fe19c25786402f1a0dbf758ee3359e04a5f809e42c97aeef0`.

**Measured result**

- One-bit scale indices reached 1.64465 packed-trunk bits/value and reduced
  scale storage from 71,100 to 5,341 bytes, but lost 12.8844 dB ground-truth,
  9.8778 dB edge, and 17.6145 dB teacher PSNR against the exact FP16-scale g2
  browser baseline.
- Two-, three-, and four-bit scale indices reached 1.68713, 1.73731, and
  1.79979 bits/value but still lost 7.7426, 2.1568, and 1.7805 dB mean
  ground-truth PSNR respectively.
- Eight-bit indices were closest at -0.03457 dB ground-truth, -0.04542 dB
  edge, and -0.09099 dB teacher PSNR, but failed every frozen quality guard and
  reduced scale storage by only 13.5%, below the required 50%.
- No candidate entered confirmation. The selector retained FP16 scales. Result
  SHA-256 is
  `4291ffd004a202450855606d7c2e7e0a0edb12f40c8d1f9496445f42598af77d`.

**Runtime receipt and next implication**

- The independently packed FP16-scale g2 artifact is 322,224 binary bytes plus
  a 113,989-byte manifest. The native Rust scalar runner reproduced a 16x16
  dense-state fixture in 138.5 ms with maximum absolute error 0.000738 and mean
  error 0.000108.
- On a 64x64 fixture, speed-optimized native Rust ran at a 1.6469-second median.
  Browser scalar WASM ran at a 2.3472-second median, while an optional SIMD128
  build ran at 1.8760 seconds, a 20.1% improvement. Both stayed within
  0.000743 maximum absolute error. The adaptive loader attempts SIMD and falls
  back to scalar WASM when SIMD initialization fails; a forced-scalar browser
  receipt verified that path. The independent runner is still slower than the
  shipped ONNX Runtime path and is not the product default.
- Do not post-quantize scale values independently and do not advertise the
  current trunk as honest 1.58-bit storage. Scale compression must become part
  of the training geometry, with a gradual codebook schedule and sensitive
  scale exceptions, then repeat this preregistered image gate.
- AdamW continued independently to step 20,000/30,000 with exact checkpoints;
  no optimizer selection has been made. Final heldout remains locked.

## 2026-07-31 — Gradual scale-codebook QAT and a native packed v3 contract

**Training implementation and bounded smoke**

- The post-training scale result above is treated as a falsification, not a
  reason to retain FP16 scales indefinitely. A second QAT parametrization now
  keeps every hard-ternary continuous master trainable while progressively
  moving its group scales onto learned per-tensor codebooks. Core matrices,
  residual convolutions, position-bias matrices, and image I/O convolutions
  enter the scale transition in separate frozen windows.
- The codebook forward uses the exact FP16-rounded centers the browser artifact
  will store, with straight-through gradients to both the continuous masters
  and positive learned centers. A forced-final scale hard shadow runs during
  transition. The q1 configuration has 204 codebooks and 408 learned centers.
- The trainer now persists and exactly resumes model, optimizer, data cursor,
  augmentation RNG, NumPy/Torch/MPS RNG, elapsed time, scale schedule position,
  and parent identity. An inspect-only receipt restored the selected g2 parent
  exactly against the 206,500-image balanced corpus. A one-step CPU smoke wrote,
  reloaded, and materialized a fully hard q1 checkpoint. Its 22.13 dB fixed-crop
  score is deliberately not quality evidence: the smoke jumps every category
  to q1 in one step and exists only to exercise the state boundary.

**Honest browser storage and execution boundary**

- Browser artifact v3 stores FP16 codebook centers plus LSB-first bit-packed
  scale indices instead of silently re-expanding learned scales to FP16 per
  group. The q1 smoke contains all 204 ternary tensors in that representation:
  5,341 scale bytes, 256,573 binary bytes, and 1.64465 packed-trunk
  bits/quantized weight. With the uncompressed JSON manifest it totals 395,474
  bytes; this is infrastructure evidence, not a promotion artifact.
- The independent Rust loader validates v3 codebook widths, level counts,
  index sizes, byte ranges, and out-of-range codes, then decodes scales once at
  artifact load. Native scalar execution completed the 16x16 full Swin2SR
  graph in 107.7 ms. In an actual browser, the rebuilt SIMD128 WASM package ran
  the same unpromoted smoke in 174.9 ms and the forced scalar-WASM fallback in
  207.3 ms. The large error against the g2 fixture is the expected untrained-q1
  quality loss, not a backend discrepancy.
- Thirteen focused Python tests and 30 Rust tests pass, including unaligned
  scale bitpacking, FP16 storage-aware QAT, v3 artifact decode, invalid-code
  rejection, and all existing v1/v2 compatibility paths.
- A gradient receipt on the first bounded smoke exposed that the shared surface
  parameter-group helper froze the newly attached center parameters along with
  the rest of the model. No long scale run had started. The scale parameter
  group now explicitly re-enables those parameters and a regression test pins
  the boundary. The repeated smoke reports all 408 center parameters with
  finite gradients, L2 norm 0.17496 and maximum absolute gradient 0.11613,
  alongside live continuous-master gradients.

**Decision**

- The new trainer and runtime boundary are eligible infrastructure for a
  preregistered scale-QAT study, but no q1 quality candidate is selected. Freeze
  the study only after the already sealed optimizer, objective, and matched
  surface-recovery chain chooses its parent. Final heldout remains locked.

## 2026-07-31 — Scale sensitivity is concentrated in image I/O

**Preregistered mechanism probe**

- Before observing category-specific image results, eight mappings were frozen:
  uniform q1 and q2; four q1 maps that individually retain q4 scales for image
  I/O, position bias, residual-body, or core tensors; and two low-overhead
  sensitive combinations. The probe reused the already sealed, balanced
  64-sample development screen and remained diagnostic-only. Protocol SHA-256
  is `f9a1e23727538be1187f456eb9dae6b5c7d3ca9b6014ff28d7c4043f622219d8`.

**Result**

- Image I/O is the decisive scale-sensitive category. Raising only its two
  tensors from q1 to q4 moves the trunk from 1.64465 to just 1.65753 bits per
  quantized weight but reduces the post-training ground-truth loss from
  12.8844 to 3.4368 dB. Edge and teacher losses improve from 9.8778 and
  17.6145 dB to 2.7264 and 6.2106 dB.
- Giving q4 scales only to position-bias or core tensors does not recover the
  q1 image loss. Residual-body q4 alone recovers only 0.593 dB. This also shows
  that global weight RMSE is a poor selection surrogate: several maps reduce
  RMSE while leaving product-image PSNR essentially unchanged.
- The best frozen low-bit mechanism map uses q4 image I/O, q2 residual and
  position scales, and q1 core scales. It reaches 1.67970 trunk bits and loses
  1.9345 dB ground-truth, 1.6082 dB edge, and 3.9441 dB teacher PSNR versus the
  FP16-scale parent before any recovery training. Uniform q2 is larger at
  1.68713 bits yet loses 7.7426 dB, demonstrating the value of sensitive-layer
  allocation rather than uniform precision.
- Result SHA-256 is
  `cfb7bef6ce7c344e2813f40fbd3f515a868feae2c2a43b262e2437ed5adc943d`.

**Decision**

- The scale-QAT study must include both honest uniform q1 and the sensitive
  q1-core/q4-image/q2-residual stepping map. The latter is not the final
  approximately-1.58-bit objective; it is the scientifically justified bridge
  for a later staged collapse to q1. Do not spend primary training budget on
  uniform q2 or on extra position/core precision. Final heldout remains locked.
- A proposed shortcut that copied a hard bridge back into a fresh first-stage
  ternary master was rejected by an exactness test: reapplying the original
  quantizer changed hard weights by 0.17122 maximum. That implementation was
  removed. A test-pinned chained projection instead preserves the bridge output
  bit-exactly at collapse amount zero, freezes its centers, appends the lower-bit
  projection, and propagates gradients to both the new centers and the original
  continuous masters. This is the eligible staged-collapse mechanism if the
  sensitive bridge advances.

## 2026-07-31 — AdamW improves recovery but fails the q4 gate

- The matched AdamW arm completed step 30,000 and all 3,040 development
  samples. It scores 33.02876 dB ground-truth, 36.81346 dB edge, and 38.25982
  dB teacher PSNR. Relative to q4, the paired means remain -1.89489, -1.61976,
  and -8.67298 dB; the worst source is -2.39221 dB.
- AdamW is materially better than the matched QAT8 continuation, which remains
  -2.30740 dB ground-truth and -2.31090 dB edge versus q4, but neither arm
  passes the frozen development screen. No optimizer recipe is selected and
  final heldout remains locked. Decision SHA-256 is
  `98d28ce8a1ec43c8d6a592ae341b8f06276e1c769f69d199c028c11ffbc1a7ca`.
- The first decision invocation correctly stopped because the live
  `full_surface_cli.py` no longer matched the frozen study hash. The completed
  artifacts were then evaluated by the study's own unchanged sealed source
  tree, whose source hashes match the manifest. No live source was reverted and
  no provenance check was weakened.
- With no passing optimizer arm, the already sealed Charbonnier versus
  PSNR-aligned objective study now runs from the fixed step-24,000 parent; its
  Charbonnier control started successfully. Surface recovery and scale-QAT stay
  queued behind that decision.

## 2026-07-31 — Exact chained q1 collapse is implementation-ready

**Mechanism**

- A mixed-bit recovered scale bridge can now be loaded as an immutable hard
  projection and followed by a new gradual q1 projection. At collapse amount
  zero the effective weights are bit-exact to the bridge; bridge centers are
  frozen, while the original continuous masters, ordinary correction
  parameters, and new q1 centers remain trainable.
- The staged trainer persists exact bridge identity, both scale-codebook
  lineages, optimizer state, schedule position, sampler cursor, augmentation
  and Torch/NumPy/MPS RNG state, and elapsed time. Strict checkpoint reload and
  hard materialization retain the browser-v3 FP16-center plus packed-index
  contract without pretending that the bridge itself is the final endpoint.
- Two sensitive-layer schedules are explicit: the existing late image-I/O
  transition and a gentler transition that moves the two decisive image-I/O
  tensors over the final half of training. A prepared matched study varies
  schedule first and hard-shadow pressure second at equal 4,000-step compute;
  it can seal only from the prior scale-QAT decision and only when that decision
  has not already produced an eligible q1 endpoint.

**Verification and boundary**

- Eleven focused Python tests pass across chained exactness, frozen bridge
  centers, live q1 gradients, strict state reload, hard materialization,
  schedule behavior, v3 scale bitpacking, and existing scale-QAT behavior.
- This is implementation evidence, not image-quality evidence. The follow-on
  study is intentionally not sealed until the queued objective, recovery, and
  first scale-QAT decisions identify its exact parent. Final heldout remains
  locked.

## 2026-07-31 — PSNR-aligned objective selected

- The matched PSNR-aligned arm completed step 26,000 and all 3,040 development
  samples at 32.89727 dB ground-truth, 36.64522 dB edge, and 38.28674 dB
  teacher PSNR. The Charbonnier control scored 32.60076, 36.27028, and
  37.73528 dB on the exact same samples.
- Paired PSNR-aligned minus Charbonnier ground-truth is +0.29651 dB with 95%
  interval `[+0.28233, +0.31010]` and 86.41% sample wins. Edge improves
  +0.37494 dB and teacher agreement +0.55146 dB. Both sealed source strata
  improve: +0.31317 dB multilingual synthetic UI and +0.26251 dB RICO.
- Every frozen objective gate passes, so PSNR-aligned is selected for the
  already sealed surface-recovery comparison. It remains 2.02638 dB below q4
  on development and therefore does not authorize promotion or final-heldout
  access. Decision SHA-256 is
  `f4cda91488de7c058c85336019015c8577ef5174dfa0abadce5506598f177cc0`.
- The matched three-surface recovery study sealed from checkpoint SHA-256
  `6653ad2dd9d65f5c9c4290b2f85be37f59f67a84f8750b3415ddf03c5d30994a`
  and began its first arm. Its manifest SHA-256 is
  `c822d1b00b63ba86419f720a2987625fce114b4dca717656b827a7105a76e7bb`.

## 2026-07-31 — Single-file v3 browser transport prepared

- Browser-v3's exact `model.json` and `model.bin` bytes can now be wrapped in a
  length-delimited bundle and deterministically gzip-compressed into one
  transport artifact. Its ledger binds both component hashes, the expanded
  bundle hash, compressed package hash, sizes, compression mode, and measured
  packed-trunk bits.
- The browser loader accepts either the expanded bundle or gzip bytes, validates
  the magic and exact component lengths, and returns the original manifest and
  binary byte-for-byte to the existing adaptive SIMD-to-scalar WASM runtime.
  This avoids treating two loosely related HTTP objects as one promotion
  identity and matches the frozen promotion policy's gzip transport contract.
- Three Python and three JavaScript transport tests pass, including deterministic
  gzip, exact component recovery, ledger mismatch rejection, and truncation
  rejection. No model-quality claim follows from this packaging work.

## 2026-07-31 — Packed-q1 final gate is format-native

- A development decision can precommit a final candidate only when it names an
  endpoint-eligible uniform-q1 arm. The precommit revalidates the exact study,
  checkpoint, hard materialization, browser-v3 manifest and binary, paired q4
  development gate, packed-trunk ceiling, gzip artifact size, and source
  closure before it can mark final evaluation eligible.
- A separate evaluator opens the final bundle only after that precommit. Its
  result is bound to the canonical hard state and the exact packed transport.
  A scale-specific promotion decision then applies the existing paired,
  stratified, artifact, and six-browser runtime policy without routing the q1
  artifact through the older optimizer/ONNX-only manifest shape.
- A 128x128 full-graph fixture generator binds the canonical hard state,
  deterministic inputs and expected output, compressed/expanded artifact
  hashes, browser-v3 component hashes, and runtime harness sources. Nine Python
  gate tests and three JavaScript transport tests pass, including modified
  package rejection. Final heldout remains locked because no eligible image
  candidate exists yet.

## 2026-08-01 — G3 becomes the selected deployed ternary artifact

**Verified.** The joint Bonsai recovery study selected a single canonical G3
state with SHA-256
`308b7fc0de513a4983563d519219c100ff8f13bb0a5143d5cd3e1d4204d2a612`.
Of 1,005,348 parameters, 839,904 are in 198 ternary tensors, 129,600 are in
four signed-int7 RSTB residual-group convolutions, and 8,100 image-I/O weights
remain FP16. Its packed native transport is 380,107 bytes, or 3.02468 lossless
bits per model parameter; the portable ONNX gzip is 808,002 bytes and hashes
to `a5e32c6b853f32e6c2fd39615ff27f7780e076fdac504dc3f2be1e822e3e085c`.

On the fixed 3,040-sample development bundle, G3 reaches 34.307506 dB
ground-truth, 37.873079 dB edge, and 41.703679 dB teacher PSNR. Relative to the
previous deployed ternary g2 artifact, those are gains of 0.209351, 0.150167,
and 0.721767 dB respectively, with no tested group regression. Set5 RGB PSNR
is 35.302292 dB, 3.515532 dB above bicubic.

The exact portable graph passed Chromium WebGPU and forced-WASM conformance.
The 64-to-128 fixture measured 30.4 ms warm WebGPU and 377.3 ms forced WASM;
both matched the sealed expected output within 0.000143 maximum error. These
receipts establish artifact identity and numerical execution, not production-
resolution throughput or six-browser closure.

The artifact is public at Hugging Face model commit
`b357304af178a6e31a5482958f1adf702f99be82` and BackdropKit Cloudflare Worker
version `a559a467-dc53-4b91-82e5-10a2f628c40c`. The Space main-thread freeze
was subsequently repaired by moving ONNX session creation and tiled inference
into a module worker with heartbeat, cancel, and timeout behavior at Space
commit `9896ce694616a8294fc566cd68f3c4a873a33d8c`.

**Interpretation.** Mixed precision is not an accounting loophole here. G3 is
83.5436% ternary by parameter count, and its remaining precision is localized
to surfaces that repeated ablations identified as sensitive. Its 3.02468
lossless bits per model parameter are comparable to the honest Bonsai-style
target the program adopted, but it must not be described as a globally exact
1.58-bit model. It is a meaningful replacement for the older ternary tier,
not yet a quality-equivalent replacement for q4.

**Decision.** G3 is the selected still-image ternary product artifact. The q4
model remains the scientific quality reference until a single candidate passes
the full q4 non-inferiority ledger.

## 2026-08-01 — Scalar corrections to G3 are rejected

**Verified.** Fixed-int7 scale training with mask 7 at learning rate 2e-4,
mask 0 at 2e-4, and mask 7 at 5e-5 all regressed at their sealed step-250
gates. A separate 240-value RSTB-bias-only arm also regressed at step 250.
Every trainer stopped under its registered rule; none is eligible to replace
G3.

**Interpretation.** G3 is locally saturated under corrections that only move
existing scales or biases. A stronger candidate probably needs new
representational capacity while preserving exact G3 behavior at initialization.

**Decision.** Do not continue tuning fixed codes through scalar scale or bias
updates. The next still-image hypothesis is a zero-initialized tiny output
residual adapter. It is not started and must pass the full 3,040-sample
subgroup screen before any packaging work.

## 2026-08-01 — Realtime RGB video path selected; first temporal arm rejected

**Verified.** The selected 12-channel RGB realtime student has 8,496
parameters and a 16,992-byte FP16 browser payload. On Apple M5 Chrome 150,
actual 1920x1080-to-3840x2160 WebGPU execution measured 8.9 ms median and
10.6 ms p95 over 120 frames. With H.264 enqueue, it measured 9.4 ms median,
11.0 ms p95, emitted 120/120 chunks, and reached queue depth two.

On a 120-frame Seedance action fixture, the student improved RGB PSNR by
0.703365 dB and edge PSNR by 2.672525 dB over bicubic, while temporal-delta
PSNR regressed by 0.921971 dB. A zero-regression 14-channel temporal arm
improved its registered temporal metrics at step 250 but regressed static-
delta error at step 500, so its trainer stopped and the arm is rejected.

**Decision.** The 12-channel RGB model remains the realtime video default on
eligible WebGPU `shader-f16` devices. The next temporal study must improve both
temporal-delta and static-region consistency without surrendering the current
spatial and edge gains.

## 2026-08-01 — Video data scale is separated from target quality

**Verified.** The pinned Markov metadata index contains 48,477 identified
screen recordings and approximately 12,281 total hours. The balanced initial
ingest list contains 4,282 videos and 1,001.506 hours. Downloaded Markov
samples are 640x360, while the sampled Humyn workflows are approximately
382/384x848/864. OpenVidHD remains quarantined because its metadata does not
close item-level source rights.

**Decision.** Markov and Humyn can contribute motion priors and source-disjoint
evaluation, but the sampled files are not honest 2K/4K reconstruction targets.
Raw video blobs and the generated 22.8 MB index remain outside Git; committed
manifests, hashes, split rules, and receipts preserve the scientific record.

## 2026-08-01 — Program status review

No still-image or video training process is active on this machine. G3 and the
12-channel RGB model are the current selected artifacts. LLM and Sana folders
remain archived scientific references. The next open work is hypothesis-led:
the zero-initialized still-image residual adapter, temporal consistency without
static regression, broader runtime coverage, and product-scale OCR/flicker
measurement.

## 2026-08-01 — Runtime policy fork preserves historical V9

The test suite detected that the current scalar and SIMD WebAssembly payloads
no longer matched the immutable V9 policy hashes after signed-int7 v4 decoding
was added for G3. V9 was not edited. V10 supersedes it for future candidates,
changes only those two WebAssembly identities, and pins the current runtime
before any new evaluation. The q4 reference, quality gates, dataset, artifact
size, required browsers, numerical limits, and JavaScript harness identities
are unchanged.
