# Qwen3.8-27B ideal base-3 packing journal

## 2026-08-15 — Reference arithmetic-code gate

### Preregistration

- Hypothesis: a streaming uniform arithmetic codec can losslessly pack real
  optimal-MSE Qwen ternary codes within 0.001 payload bits/trit of `log2(3)`.
- Primary metrics: exact code equality and payload bits per trit.
- Protected metrics: CRC corruption detection and complete-model effective-bit
  accounting including scales and retained high-precision tensors.
- Budget/stop: one million real codes or the first correctness/rate failure; no
  full-checkpoint export.

### Result

- Status: passed.
- Model revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Input: 1,000,000 optimal-MSE ternary codes from two representative Qwen
  attention matrices; code SHA-256
  `5b5631d6c459da7af12952375421a3321c794c3f8a9814a38f799658b0b6c3fb`.
- Histogram: 266,670 negative, 466,137 zero, 267,193 positive.
- Exact arithmetic payload: 1,584,964 bits, or 1.584964 bits/trit.
- Uniform ternary limit: 1.584962500721156 bits/trit.
- Five-trits-per-byte control: 1.6 bits/trit.
- Aligned two-bit control: 2.0 bits/trit.
- Framed artifact: 198,146 bytes including the 25-byte header and CRC.
- Reference Python encoding: 0.9175 seconds for one million trits. Projected
  serial time for 26,869,760,000 mapped ternary parameters is 6.85 hours.
- Projected ternary-code payload: 5,323,445,251 bytes. Complete model remains
  7,566,621,155 bytes and 2.1789006 effective bits/parameter after FP16 scales
  and retained islands; this is not an end-to-end 1.58-bit claim.

### Interpretation and next gate

The mathematical packing treatment is promoted as a reference codec only.
The next gate is a bounded-memory, independently decodable chunk format with
parallel encoding, per-chunk hashes, an atomic manifest, and exact resume. A
full checkpoint pack is forbidden until that implementation survives forced
interruption and demonstrates useful throughput on this spot node.

## 2026-08-15 — Parallel chunk/resume gate

### Preregistration

- Hypothesis: a bounded parallel chunk container can resume after forced
  process termination with exact source identity, independently decodable
  chunks, and aggregate payload no more than 0.001 bits/trit above `log2(3)`.
- Input: 32,000,000 deterministic real Qwen optimal-MSE trits, in fixed tensor
  and row order, split into one-million-trit chunks.
- Primary metrics: reused committed prefix, exact full decode, source SHA-256,
  payload bits/trit, and effective end-to-end throughput.
- Protected metrics: model/quantizer/source identity, raw and frame SHA-256 per
  chunk, frame CRC32, bounded in-flight work, and atomic manifest replacement.
- Stop: 32 million real trits or first identity, decode, integrity, or rate
  failure. A first uninterrupted completion does not pass: at least one chunk
  must be recovered from an actually terminated process.

### Full-pack gate (held behind resume promotion)

If and only if the interruption/resume gate passes, export all 26,869,760,000
mapped trits in deterministic shard/tensor/row order. Each row batch has an
independent arithmetic frame and FP16 scale block. Promotion requires exact
decode, regenerated code and scale SHA-256, exact mapped/scale counts, payload
within 0.001 bits/trit of `log2(3)`, bounded row memory and process queues, and
an honest complete-model ledger including retained FP16 islands. Behavioral
quality and duplication of retained source tensors remain explicitly separate
gates.

## 2026-08-15 — Real provider eviction

- Prime Intellect history records the 1 x RTX Pro 6000 96 GB Verda spot node
  created at 19:13 and terminated at 20:05 PDT, billed $0.87.
- SSH timed out immediately afterward and the active instance count fell from
  six to five.
- Prime's Storage tab reported no shared disks. The root-disk receipts, model
  cache, and reference artifact must therefore be treated as lost even though
  the important reference numbers had already been independently recorded.
- Harness correction: use local atomic state and strict resume validation.
  Off-node mirroring is disabled until a project-approved destination exists.

## 2026-08-15 — Offline full-artifact integration gate

- Built a synthetic Qwen-format safetensors index containing three mapped
  matrix types plus retained vector and excluded visual weights.
- The production full-pack entry point ran on CPU with bounded two-process
  encoding across nine row batches and promoted its exact bit ledger.
- The artifact reader independently decoded all trits and FP16 scales and
  reconstructed every mapped tensor batch exactly.
- An identical second run reused all nine committed batches and wrote none.
- After one scale block was changed by one bit, strict resume rejected the
  artifact. This closes the end-to-end pack/scale/unpack test gap but does not
  substitute for the 32-million-real-trit GPU interruption gate.
- The terminated IP now presents a different ED25519 host key. Strict SSH host
  checking prevented connection to the reassigned address; the old known-host
  entry must only be replaced after Prime supplies a new instance and address.

## 2026-08-15 — Storage-boundary correction

- An off-node mirror was found to target infrastructure outside this project's
  approved boundary. The mirror was stopped, its service and source credential
  were removed, the restricted destination key was revoked, and the exact
  copied experiment directory was deleted.
- No off-node backup target is configured now. The non-spot A100 retains the
  pinned 18-shard model and local experiment state. Future QAT runs are capped
  at the newest two local checkpoints to prevent disk growth.

## 2026-08-15 — Complete-rate correction and 1.60-bpp target

The active group-128 artifact is an intentionally conservative quality anchor,
not a complete 1.58-bit checkpoint. Its exact preflight ledger is:

- 27,781,427,952 total parameters;
- 26,869,760,000 mapped ternary parameters;
- 209,920,000 FP16 group scales;
- 911,667,952 retained BF16 parameters;
- 1.532950793 ideal code bits per original parameter;
- 0.120898033 scale bits per original parameter;
- 0.525051745 retained-island bits per original parameter;
- 2.178900571 complete effective bits per original parameter.

The retained BF16 total is dominated by 424,699,392 MTP parameters and
460,730,096 vision parameters. Another 23,592,960 parameters are the excluded
linear-attention A/B projections, 1,966,080 are convolution projections, and
679,424 are other currently unmapped tensors. Calling only the arithmetic code
stream “the model” would therefore be misleading.

Header-only inventory of the pinned checkpoint establishes a technically
plausible next target without changing the active experiment: ternarize every
rank-2 parameter, retain only the 4,850,416 rank-not-equal-to-2 parameters in
BF16, and store one FP16 scale per output row. A full retained-dtype audit found
all 4,850,416 parameters across 582 rank-not-2 tensors are BF16; the target gate
now seals and verifies that complete dtype ledger rather than relying on one
convolution control. The resulting exact projected
ledger is 1.590214265 effective bits per original parameter. Alternatives with
FP8 group scales project to 1.595360135 bpp at group 1024 and 1.592019816 bpp at
group 2048.

These are storage projections, not quality results. The next rung is sealed in
`target_158_preregistration.json`: finish the conservative anchor first, then
compare fixed reconstruction samples spanning language, vision, MTP,
convolution, and linear-attention projections. A second full pack is forbidden
until that receipt selects a candidate, and no complete-model quality claim is
allowed without separate text, multimodal, and MTP-aware behavioral evidence.

## 2026-08-15 — QAT continuation contract correction

The original ten-step script correctly measured eligibility for a 100-step
selected-layer rung but had no executable continuation. That gap is closed
before any QAT outcome exists. `one_layer_qat_100.py` requires a promoting
ten-step receipt, authenticates and reloads checkpoint 10, reconstructs the
same packed model and selected hard-ternary wrappers, restores master weights,
optimizer global step, and RNG state, and proves checkpoint-10 held-out metric
reproduction before step 11.

The continuation evaluates held-out teacher KL at steps 20 through 100, stops
after two consecutive regressions, writes one atomic terminal checkpoint, and
keeps at most checkpoint 10 plus that terminal state. Its result has a distinct
receipt and may be a valid scientific non-promotion. The completion auditor
requires this receipt only when the ten-step rung actually promotes.

A prelaunch resume audit found that PyTorch's generic optimizer loader cast
BonsaiQAT8's block-int8 first-moment buffers to the FP32 parameter dtype. The
integer values survived, but the low-state memory contract did not: a resumed
372M-parameter layer could allocate roughly 1.1 GB more optimizer memory. The
loader now restores the declared int8/FP32 state dtypes explicitly. Promotion
also requires exact comparison of the complete nested optimizer state and a
complete CPU/CUDA RNG payload; the CUDA resume probe passes on the A100.

The behavioral gate's prelaunch audit also exposed two evidence gaps. Parameter
counts alone did not prove that the evaluated artifact was the artifact named by
the full-pack receipt, and greedy decoding was described as deterministic
without a replay. The gate now binds the exact artifact path, manifest hash,
code-source hash, scale-source hash, model revision, model-index hash, and trit
count before loading the model. Both BF16 and ternary generation controls are
replayed byte-for-byte and their canonical hashes are recorded. The final audit
requires the same artifact identity chain.

The radix-5 performance receipt is no longer accepted through its promotion
boolean alone. Its 10-million-real-trit sample count and source subtotal, exact
round-trip, 1.6-bit payload rate, delta from `log2(3)`, 50 Mtrit/s encode/decode
floors, and full-model byte/rate projection are recomputed by the final audit
from the pinned preflight ledger. The numerical thresholds are also immutable
in the executable preregistration check.

The retained export previously copied the ideal preflight rate into its
complete-bundle receipt even though the actual arithmetic frames have a small,
measured overhead. It now inherits code-frame, FP16-scale, retained-byte, total
payload, and effective-rate values from the promoted full-pack ledger and
cross-checks their arithmetic. At final audit the safetensors bundle is reopened
and every tensor name, metadata field, shape, dtype, parameter count, byte count,
and SHA-256 is independently checked against the retained manifest.

The layer-sensitivity and teacher-cache handoff is now sealed end to end. The
candidate order `[0, 3, 31, 63]`, each mapped tensor/parameter/restore-byte
inventory, and the exact selection rule must match the QAT preregistration.
Sensitivity promotion requires a strictly positive teacher-KL recovery. The
eight literal calibration texts, 128-token cap, top-64 width, model revision,
and selected layer are authenticated before caching and again before either QAT
rung consumes the cache.

The conditional continuation and independent target-1.59 reconstruction gate
are chained by a separate post-pipeline systemd unit. This leaves the active
pipeline byte-for-byte untouched while making the new stages reboot-safe. The
postlude refuses malformed or ambiguous receipts, never reruns a terminal
scientific non-promotion, reapplies the two-checkpoint cap, and ends with the
fail-closed completion audit.

## 2026-08-15 — Clark compaction dependency refresh staged

The canonical committed `clark-agent-compaction` source advanced from
`e5f4a4b7d122f73adc362c4a5395355ac95e540c` to
`865f4f64b2e8617a0350e0fcefa2ef099248650a`. The delta contains CI changes and
rustfmt-only source/test formatting; the local repository has no dirty tracked
files. A complete versioned Git bundle and clean source archive were transferred
directly to the A100 and verified at SHA-256
`484e15965498e034d1dadab32d8bd0477e80f40dc4d068a30343ed4a19ce1392` and
`105bfeceac8d647ff30973782c6fc45637f50d6b54199e3ef98f6bb84340c9aa`,
respectively. The remote dependency worktree was clean and fast-forwarded to the
new revision; the previous bundle remains available for diagnosis.

The installed Clark worker remains the previously verified 0.1.155 binary while
the CPU-bound pack is active. Provenance now requires the new dependency
revision, so the receipt-gated pipeline will rebuild and atomically install the
worker after the pack and behavioral-quality stage instead of competing with
the live pack for CPU.

Clark completion verification now retains and authenticates the source
transport itself. Provenance includes exact local paths for every clean archive
and dependency bundle. The final audit rehashes them, checks all staged source
directories and live Git heads where available, requires the install receipt to
match Clark, Desktop, Autoresearch, and Compaction revisions, and validates the
worker version, protocol, specialist count, catalog hash, binary hash, and full
self-test equality.

## 2026-08-15 — Final framed-ledger verification correction

The completion audit formerly authenticated the finalized manifest and checked
aggregate trit/decode counts, but it did not independently recompute every
complete-pack byte subtotal from the files still present on disk. That left a
receipt-authoring bug capable of preserving a plausible effective-rate claim.

The audit now requires the manifest identity hash, exact mapped-source and
zero-based chunk inventories, contiguous row coverage for every matrix, exact
chunk and scale filename sets, group-128 scale shapes, and encoder round-trip
evidence. In production mode it reads every frame and scale block, verifies
their SHA-256 values, parses each `CQ3A1` header, checks CRC32 and zero padding,
then recomputes payload bits, frame/header bytes, FP16 scale counts/bytes,
retained BF16 bytes, artifact metadata, complete payload/transport bytes, and
both effective rates. Deliberate frame-content and receipt-ledger corruption
are rejected by the test suite. Local and A100 suites pass 71 tests after this
correction; the running full pack was not restarted.

## 2026-08-16 — Full pack completion and stage-handoff repair

The bounded full pack completed all 34,344 row batches and independently
decoded all 26,869,760,000 ternary values before publishing its atomic receipt.
The finalized manifest SHA-256 is
`c98d878d0c5e3ea92be27c5ba59da2c9d7d0154c839dd89a17d0e94eb18b75b4`.
The arithmetic payload is 1.5849638042542993 bits per trit, only
1.3035331432043051e-6 above `log2(3)`. The honest complete-model payload,
including FP16 group scales and referenced BF16 islands, is
2.179156193432515 bits per original parameter. The independent decode took
3,748.570 seconds and covered exactly the full mapped trit ledger.

The retained-tensor export then completed and promoted, but the immediate next
job exposed a harness race. The heartbeat shell's child `sleep` inherited the
exclusive `job.lock` descriptor; although the retained-export command had
exited, that helper could hold the lock long enough for the next five-second
acquisition to fail. The main pipeline therefore stopped before behavioral
quality. Because the post systemd unit only had an ordering dependency, it then
started after the failed main unit and ran the independently defined target
reconstruction before detecting the absent QAT chain. Its terminal target
receipt is preserved as scientific evidence but does not establish pipeline or
experiment completion.

The harness now closes the lock descriptor in the scientific child, output
`tee`, and heartbeat subshell, terminates and reaps the heartbeat's child sleep,
and has a Linux regression proving two immediate sequential jobs both acquire
the lock. The postlude now requires promoting receipts for every upstream gate
and a schema-valid terminal QAT10 receipt before doing any post-pipeline work;
its systemd relation is `Requires=` rather than `Wants=`. Seven focused tests
pass on the A100, including the Linux lock regression and both fail-closed
postlude cases. The receipt-gated main and post services were then re-queued;
the main pipeline reused the completed pack and retained receipts and entered
behavioral quality without repeating either stage.

The first behavioral-quality attempt exposed a separate performance defect
before producing a receipt. Artifact application decoded all 34,344 frames in
one Python process. Its measured file-read cadence matched one worker from the
full verifier, projecting roughly 13--15 hours solely to reconstruct the model.
The receipt-less attempt was stopped cleanly. Artifact record authentication
and decode are now exposed as a process-safe top-level function, and quality
uses a bounded 14-worker, 28-in-flight decoder while preserving ordered model
mutation in the parent process. Codes and FP16 scales, rather than full model
state, cross the worker boundary; reconstruction math and GPU copies remain in
the parent. Synthetic exact-equality and corruption tests pass locally and on
the A100. On the real artifact all 14 workers spawned successfully and sustained
roughly 82--87 percent CPU each with the BF16 model resident in about 52 GB of
VRAM. This changes execution throughput only; quality prompts, thresholds,
artifact identity, reconstruction values, and promotion criteria are unchanged.

## 2026-08-16 — PTQ non-promotion, rescue routing, and native decode

The completed behavioral receipt authenticated all 34,344 artifact batches,
402 mapped tensors, and 26,869,760,000 applied parameters against manifest
SHA-256 `c98d878d0c5e3ea92be27c5ba59da2c9d7d0154c839dd89a17d0e94eb18b75b4`.
Both BF16 and ternary generations were deterministic and nonempty, but the
data-free group-128 PTQ hypothesis failed decisively: BF16 perplexity was
16.134834700092938, ternary perplexity was 1,896.376609965226, the ratio was
117.53306713171983 against a 1.5 limit, and mean teacher KL was
5.310865478515625 against a 1.0 limit. The failed promotion is retained as the
terminal observation; thresholds and receipt contents were not changed.

The pipeline had treated any non-promoting quality receipt as absent, which
would rerun a 4,433-second completed experiment forever. It also contradicted
the sealed QAT entry contract, which requires finite quality metrics and
deterministic generations rather than PTQ promotion. Terminal-quality
validation now independently requires the exact artifact and model identity,
full applied-parameter count, finite BF16/ternary metrics, deterministic
nonempty controls, and an authenticated local teacher cache. Only then may a
failed PTQ observation enter the preregistered layer-sensitivity/QAT rescue
rung. PTQ promotion remains false and is reported separately in the final
audit. Terminal scientific jobs are schema-gated against automatic reruns;
selection and teacher-cache stages must still promote before training starts.

The Clark Scientist/Engineer worker was refreshed from the pinned source set,
self-tested as version 0.1.155, and installed with SHA-256
`e4019d5349c4914457510185b7ce649d3115dfb0cb4a9f3a768860cf463edb82`.
The independent radix-5 gate then passed an exact 10,000,000-trit round trip at
107.90096973691992 million trits/s encode and 329.16519133045114 million
trits/s decode. Its code payload is exactly 1.6 bits/trit, while the honest
complete-model projection is 2.1934446042617153 bits per original parameter.

The ideal arithmetic decoder remained the dominant repeated-stage cost. An
optional C11 decoder now implements the same 32-bit uniform arithmetic state
machine behind the unchanged `CQ3A1` frame and Python fallback. Every artifact
record still verifies frame SHA-256, header/CRC, decoded raw-code SHA-256,
scale SHA-256, and shape before model mutation. On a real 655,360-trit Qwen
frame, the A100 host measured 54.9476 million trits/s natively versus 0.4839
million trits/s in Python (about 113.6x), and both decoders produced the same
raw SHA-256 `edcb0e8713c4a9160942c654e2e8cf4235797a68a9eb08b631c575cff69c850b`.
The already-running initial sensitivity reconstruction retains its original
workers; newly spawned restoration and QAT workers use the verified native
decoder.

## 2026-08-16 — Sensitivity selection, bounded QAT result, and terminal audit

The native-decoder sensitivity gate completed in 371.788 seconds. Against the
all-ternary mean teacher KL of 5.310865478515625, restoring layer 3 produced the
only positive candidate recovery: 1.0158368937174478 KL, or
1.46509433685446 KL/GiB across 372,244,480 parameters. Layers 0, 31, and 63
recovered -1.3130224609374999, -0.014266255696614216, and
-0.21728851318359332 KL respectively. Exact reversion returned to the baseline
with zero measured KL error. The sealed eight-row layer-3 teacher cache has
SHA-256 `2c3c4ea43720436859787c0039cc3cbcd57c2d46a6314737f12d729e69551271`.

Two implementation defects were found without changing the preregistered
training rule. First, the QAT launcher compared PyTorch's returned pre-clipping
gradient norm to the post-clipping limit; it now measures both values and gates
the actual post-clipping norm. Second, terminal evaluation attempted an in-place
copy into leaf master tensors outside `no_grad`. The completed authenticated
step-10 checkpoint was not overwritten or retrained; a dedicated finalization
path loaded it, verified optimizer and RNG state, and published the receipt.
The retained checkpoint state is
`f651d286a058e079ad54ece508b6654a24e3a70adb5ffb9cea02586e6e4946c9`, and
`checkpoint-10` is the sole retained checkpoint.

The ten-step layer-3 QAT probe reduced mean teacher KL from
5.310865478515625 to 5.247605387369791 and mean NLL from
7.5477002970377605 to 7.4962471516927085. Relative KL improvement was
0.011911446712733288, below the preregistered 0.05 requirement, so promotion is
false and the 100-step rung was correctly skipped. This is a valid bounded
non-promotion, not an infrastructure failure. The projection changed 13,827 of
372,244,480 codes (3.7144943022392165e-5 churn) and used 56,125,011,968 peak
allocated GPU bytes. The terminal QAT receipt SHA-256 is
`15b7a37cadc3bfc1ff7c8f413c80b6e960ce36d4a4b1b8bf09ef1cdb6b78dd85`.

The separate storage-only reconstruction selected per-row FP16 scales for all
rank-2 tensors at 1.590214265345321 complete effective bits per parameter. Its
receipt explicitly does not authorize a full pack or make a quality claim.
The final audit initially exposed a system-service provenance bug: root could
not invoke Git in repositories owned by the experiment user. Revision checks
now drop only the read-only `rev-parse` command to the registered repository
owner, preserving Git's ownership protection and avoiding global safe-directory
exceptions. The rerun passed all 54 checks. Completion-audit SHA-256 is
`861b0711c901ac893b58e49543779199b6fff48e9e2c01fc78b112f8438c2ecb`;
the terminal classifications are `ptq_valid_nonpromotion` and
`ten_step_valid_nonpromotion`.
