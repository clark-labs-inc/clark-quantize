# Scientific Journal — Joint Bonsai Recovery

## 2026-08-01 — Study opened

Training was not running when the lane was resumed. The active product image
model is the released 331 KB mixed-precision ternary Swin2SR leader, not the
newer 16.6 KiB FP16 realtime-video student. The earlier MixFrag scale-QAT
candidate remains rejected at 32.582926 dB and is not a parent for this study.

The scientific reason to test RSTB fusion is strong but bounded. The previous
three-seed retained-precision study measured about +0.499 dB from restoring
these four convolutions. The subsequent final boundary decision selected
int5-per-kernel, not int7-per-output: it retained 96.5% of ground-truth gain,
97.6% of edge gain, and 90.5% of teacher gain for 55,080 incremental bytes.
However, directly composing separately trained image-I/O and RSTB weights did
not work. This study therefore trains the RSTB island jointly from the exact
released image-I/O FP16 checkpoint.

The generic retained-island loader and materializer were extended to preserve
multiple islands in checkpoint lineage. Twenty-one focused unit tests passed.
An independent parity reconstruction of the released parent compared 484
tensors with zero differences and maximum absolute error 0.0.

The discovery preflight resolved exactly 129,600 trainable RSTB values and no
others. Every selected convolution was an exact FP16 round-trip from its hard
parent (`max_abs = 0.0`). The projected temporary FP16 island adds 204,480
bytes relative to the current ternary-code-plus-scale representation; it is a
training state, not a release artifact.

### Decision

Launch `rstb_fp16_seed20260801` for 4,000 steps on the 206,500-sample balanced
corpus. Inspect the step-1 gradient and step-500 validation receipt before
authorizing the matched ternary control. Keep final heldout access locked.

## 2026-08-01 — Step 1 health receipt

The live MPS process started successfully. The exact parent scored 40.560209 dB
on the small live validation set. Step 1 scored 40.562498 dB. Gradients were
finite across all 129,600 intended trainable values (`L2 = 0.006157`, maximum
absolute gradient `0.0001451`), with no evidence of an immediate numerical or
checkpoint-lineage failure. Training continues to the predeclared step-500
gate.

## 2026-08-01 — Step 500 authorizes the matched control

The FP16 arm reached 41.178582 dB at step 500, +0.618373 dB over its
40.560209 dB start on the small live validation set. Its gradients remained
finite across all 129,600 intended values. The checkpoint records both
effective precision islands in order: inherited `image_io_fp16` and current
`rstb_fusion_fp16`; its parent SHA-256 still matches the released checkpoint.

This passes the predeclared health gate, so the matched RSTB ternary arm was
started with the same parent, seed, data order, batch, objective, optimizer,
learning rate, and schedule. Its step-1 score was 40.560438 dB—effectively flat
from the 40.560209 dB baseline—while its gradient receipt remained finite and
closely matched the FP16 arm. Both runs continue; the step-500 paired result is
the first interpretable precision-effect comparison.

## 2026-08-01 — First paired precision effect is positive

At matched step 500, the FP16 arm scored 41.178582 dB and the hard-ternary
control scored 40.769044 dB. The observed FP16-minus-ternary effect is therefore
+0.409538 dB. The ternary control itself improved +0.208835 dB from the shared
40.560209 dB start, so the comparison separates a real general training effect
from the larger precision-enabled recovery.

The FP16 arm subsequently reached 41.649626 dB at step 1,500, +1.089417 dB
from start, with finite gradients. No early peak or instability is visible, so
both arms continue to their preregistered 4,000-step endpoints.

The step-500 checkpoints were independently materialized through the new
multi-island path. Both dense states contain 484 tensors and preserve the
inherited image-I/O FP16 island. Comparing the two states found exactly four
different tensors, precisely the declared RSTB fusion convolution weights, and
no undeclared difference. This is a packaging smoke receipt, not a promotion
evaluation.

At matched step 1,000, FP16 scored 41.467946 dB and ternary scored 40.910541
dB, widening the precision effect to +0.557405 dB. FP16 reached 41.809012 dB
at step 2,000, +1.248804 dB over its start, with finite gradients. The live
trajectory remains monotonic at every 500-step receipt.

At matched step 2,000, FP16 scored 41.809012 dB and ternary scored 41.141457
dB, widening the precision effect to +0.667556 dB. FP16 reached 41.844940 dB
at step 3,000, but only +0.003408 dB above step 2,500; the live curve is
saturating without regressing. Preserve and later compare the 2,500, 3,000,
and final checkpoints on the complete development bundle rather than selecting
solely from this small live set.

## 2026-08-01 — Joint FP16 endpoint beats the released model

The FP16 arm completed all 4,000 steps at a live-validation best of 41.965174
dB, +1.404965 dB from start. Checkpoint SHA-256 is
`4927fc3a5d6cd81c651b0bc50cee05fe25bd4bac1196bbe050e63b0aa94bcc6f`.
Materialization produced 484 tensors and tensor-state SHA-256
`9121af19699e44050be43b95d75a3dc50d63066181f0bced43e2860d451dd26a`.

On the complete locked 3,040-sample development bundle, the candidate scored
34.319350 dB versus the released parent's 34.098155 dB: +0.221195 dB. Its edge
score rose from 37.722911 to 37.883304 dB (+0.160392 dB), and teacher agreement
rose +0.755023 dB. The paired-bootstrap 95% interval for ground-truth gain is
[+0.210669, +0.230958] dB; the candidate wins 87.47% of samples.

There are zero negative group means across both sources, all three splits, all
29 scripts, and all 204 languages for ground-truth PSNR, edge PSNR, and teacher
agreement. The smallest language ground-truth gain is +0.041242 dB on
`fao_Latn`; the smallest script gain is +0.064466 dB on Malayalam.

The development evaluation shared MPS with control training, so its measured
inference time is deliberately excluded from any latency claim. Remeasure
performance alone after training ends. The FP16 state passes the quality gate
but remains a training/recovery artifact; the next promotion gate is the
predeclared int5-per-kernel RSTB boundary.

## 2026-08-01 — Int5 rejected; int7 passes the mixed-precision gate

The int5-per-kernel candidate scored 34.305860 dB. Relative to the matched
ternary control, it retained 91.45% of FP16 ground-truth gain and 96.69% of
edge gain, but only 75.84% of teacher-agreement gain. It therefore fails the
predeclared 90%-on-every-metric compression gate despite remaining +0.207705
dB better than the released model. The gate was not relaxed after seeing the
result.

A sealed fallback protocol then tested int7-per-output, only 4,080 bytes larger
than int5-per-kernel. It scored 34.307506 dB, retaining 92.50% of FP16
ground-truth gain, 92.15% of edge gain, and 92.71% of teacher gain relative to
the matched ternary control. All three clear the 90% threshold.

Against the released model, int7 gains +0.209351 dB ground truth, +0.150167 dB
edge, and +0.721767 dB teacher agreement. The ground-truth paired-bootstrap
95% interval is [+0.199630, +0.218457] dB. It has zero negative group means
across both sources, all three splits, all 29 scripts, and all 204 languages.

The RSTB int7 representation occupies 113,880 bytes, 59,160 bytes above the
prior ternary RSTB data and only 4,080 bytes above int5-per-kernel. This is the
scientific quality/storage winner. Promotion remains unauthorized until the
native browser artifact, 7-bit decode/kernel path, execution parity, and
uncontended latency gates pass.

## 2026-08-01 — Packed browser execution passes; continuation begins

The selected mixed candidate was exported as a browser-v4 artifact with 198
ternary surfaces, four int7-per-output RSTB fusion weights, and the inherited
FP16 image-I/O island. The binary is 392,128 bytes; its gzip transport bundle
is 380,107 bytes, only 49,174 bytes larger than the current release.

The real in-app browser executed both runtime fallbacks from the packed
artifact. WebAssembly SIMD produced maximum error 0.000689328 and mean error
0.000094055 in 2,478.2 ms for the 64-to-128 conformance fixture. Forced scalar
WebAssembly produced maximum error 0.000689715 and mean error 0.000094053 in
3,142.4 ms. Both returned the expected `[1, 3, 128, 128]` tensor and neither
emitted a browser warning or error. These are correctness receipts, not yet
production-size latency claims.

Training had stopped after the matched 4,000-step arms completed. A new phase
was therefore sealed before its outputs: continue the winning FP16 recovery
parent for 4,000 more steps at `1e-6`, physical batch 8, and seed 20260802,
then recompress at the same int7-per-output boundary. Its initialization
reproduced all 129,600 RSTB values with zero FP16 round-trip drift. The exact
image-I/O FP16 island and all hard-ternary surfaces remain frozen. Step 500 is
an early health gate; the complete locked development bundle, not the small
live set, decides whether the continuation is actually better.

The continuation was stopped at its sealed step-500 gate. Its live score was
41.964961 dB versus the exact 41.965174 dB start, a −0.000213 dB change, and
teacher agreement was also lower. The difference is tiny, but the gate was
defined before seeing it; spending another 3,500 steps on the same saturated
FP16 trajectory would not be evidence-driven. The checkpoint was retained with
SHA-256 `6c2f18590c9cfe6685181b46b3935b0366988cc1fe762c6e4dfb30a5135b46e5`.

The next arm moves to the actual remaining loss: int7 quantization-aware
recovery. A deployment-exact signed-int7/per-output/FP16-scale STE was added to
the retained-island loader, training path, materializer, and checkpoint loader.
Twenty-four focused tests pass. Its initialization exactly reproduces all four
weights in the already evaluated int7 boundary: zero differing tensors and
maximum absolute error 0.0. Thus step 0 is the known 34.307506 dB candidate,
not a new or approximate baseline. Only those four convolution masters are
trainable; the 198 hard-ternary surfaces and image-I/O FP16 island remain
frozen.

The plain int7-QAT arm was also stopped at its sealed step-500 gate. It moved
from 41.948387 to 41.927384 dB (−0.021003 dB), with finite gradients. This is a
useful optimizer result: the deployment boundary is differentiable through the
STE, but ordinary AdamW treats continuous master-weight movement as equally
valuable even when only movement across half-integer int7 code boundaries can
change the shipped model.

A matched two-arm boundary-aware race was sealed next. It preserves AdamW's
direction and moments but boosts gradients smoothly near int7 transition
boundaries, based on distance to the nearest half-integer normalized code. The
first arm uses base learning rate `2.5e-7`, maximum boost 4, and the identical
seed/data order as the failed plain arm. Twenty-five focused tests pass,
including exact int7 initialization, checkpoint recreation, materialization
contracts, and selective boundary gain. A second, differently scaled arm is
authorized only if the first fails its step-500 gate.

Both boundary-aware arms failed their matched step-500 gate. Boost-4 at
`2.5e-7` reached 41.923874 dB and boost-2 at `5e-7` reached 41.925791 dB,
both below the 41.948387 dB deployed-int7 start. The first arm accidentally
continued past its gate while artifact validation was running; those later
steps are excluded from selection. The gradient-amplification hypothesis is
rejected.

Changing the scale estimator produced the first new positive result. Four
Lloyd-style per-output scale updates, constrained not to exceed max-range
scaling and rounded to FP16, raised the live score to 41.968884 dB before any
training. On the complete 3,040-sample development bundle it added +0.001283
dB ground-truth PSNR, +0.001455 dB edge PSNR, and +0.001471 dB teacher PSNR
over the selected max-range int7 candidate. All three paired-bootstrap 95%
intervals exclude zero.

The full Lloyd endpoint is not selected because some small subgroup means are
negative relative to max-range int7. A three-arm conservative blend race was
sealed at 25%, 50%, and 75% of the Lloyd scale movement. The max-range model
remains the leader unless one blend has positive overall ground-truth gain and
no negative source, split, script, or language means on all three quality
metrics.

## 2026-08-01 — Global scale blending rejected; retain max-range int7

All three sealed scale blends completed the same locked 3,040-sample
development evaluation. Relative to max-range int7, the 25% arm changed
ground-truth, edge, and teacher PSNR by +0.000488, +0.001117, and -0.002607
dB. The 50% arm changed them by -0.000554, +0.000087, and -0.002438 dB. The
75% arm changed them by +0.001727, +0.002123, and -0.001057 dB.

None passes the sealed rule because every arm loses teacher agreement and has
negative means in source, split, script, and language slices. The 75% arm is
the strongest blend on ground truth and edges, but its teacher-loss bootstrap
interval is entirely below zero, [-0.001724, -0.000457] dB. The result rejects
one global Lloyd-movement coefficient for all four RSTB fusion layers. The
selected max-range int7 model remains unchanged.

The next experiment must localize the scale change per RSTB layer. There are
only four int7 fusion tensors, so an exhaustive pre-registered 16-mask
max-range-versus-Lloyd sensitivity map is feasible. This tests whether a
subset captures the ground-truth and edge gains without perturbing the
teacher-sensitive layers, while preserving exactly the same packed runtime
format and byte count.

## 2026-08-01 — Portable G3 artifact passes the exact browser path

The selected max-range int7 candidate's 808,002-byte gzip portable ONNX was
executed in the real Chromium browser through the same ONNX Runtime WebGPU and
WASM surfaces used by BackdropKit. The initial smoke exposed a fixture-path
bug: relative package paths resolved below the harness directory and fetched
the wrong bytes. `prepare_full_onnx_smoke.py` now seals resolved model paths,
and the regenerated fixture identifies the exact compressed and expanded
artifacts before inference.

WebGPU passed with maximum error 0.000142574 and mean error 0.000022182; its
three fixture runs were 97.2, 30.4, and 28.3 ms. Forced WASM passed with
maximum error 0.000142813 and mean error 0.000022179; its runs were 390.6,
367.0, and 377.3 ms. These are 64-to-128 correctness/relative-latency
receipts, not production-resolution throughput claims. ONNX Runtime emitted
only its expected notice that shape-only operations can be assigned to CPU.

This closes the exact BackdropKit technical gate for the already selected G3
replacement. External Hugging Face publication and BackdropKit promotion
remain separate release actions.

## 2026-08-01 — G3 is staged for both release surfaces

A reproducible Hugging Face package was built from the exact selected dense
state (`308b7fc0...`). It contains a load-tested 4,082,952-byte safetensors
state, the 808,002-byte portable ONNX gzip, the 380,107-byte packed native
runtime, fresh Set5 before/after cards, and the browser-conformance receipt.
The standard Transformers loader accepted all 484 tensors and produced the
expected two-times output shape. Set5 full-image RGB PSNR is 35.302292 dB,
+3.515532 dB over bicubic.

BackdropKit now points the stable saved-project ID `ternary-g2` at a new
cache-safe `model-v3.bin` URL while preserving `model-v2.bin` and `model.bin`
for existing clients. Artifact identity tests, runtime capability tests, the
production build, and the Cloudflare worker tests all pass. This is a local
release candidate; no external upload or production deployment was performed.

## 2026-08-01 — Mask-7 fixed-code scale training rejected

The first frozen-code arm trained exactly 240 FP16 scale deltas while keeping
all signed-int7 codes, ternary tensors, biases, and image-I/O weights fixed.
Its pre-registered step-250 gate failed decisively: live ground-truth PSNR was
-0.088591 dB and teacher PSNR was -0.185560 dB from the exact step-zero
baseline. The trainer emitted checkpoints rather than enforcing the scientific
gate and continued while packaging was validated; every result after step 250
is excluded from selection. The process was stopped and the selected G3 state
remains unchanged. The sealed mask-0 fallback is now running with the identical
optimizer, data order, and gate.

The mask-0 fallback also failed at step 250: ground-truth PSNR was -0.077951
dB and teacher PSNR was -0.164704 dB from its exact start. Because both code
estimators fail similarly, the rejected variable is the `2e-4` scale learning
rate, not the Lloyd mask. The trainer now enforces its sealed step-250 and
step-500 gates directly so future arms cannot overshoot while other artifacts
are being verified. Seventeen focused fixed-scale tests pass.

The pre-registered next arm changes only the learning rate to `5e-5`, restores
the primary mask-7 initialization, and retains seed 20260807, batch 16, the
same 206,500-image sampler, 240 trainable scales, and every frozen deployed
code. G3 remains the release candidate unless this lower-rate arm passes both
live gates and the full 3,040-sample subgroup gate.

## 2026-08-01 — Frozen scale and RSTB-bias recovery are rejected

The lower-rate fixed-scale arm also failed its enforced step-250 gate. Relative
to the exact G3 start, ground-truth PSNR fell 0.053476 dB and teacher PSNR fell
0.106510 dB. A separate arm then trained only the 240 existing RSTB bias
values, preserving every deployed weight and code. It likewise failed at step
250, at -0.027322 dB ground-truth PSNR and -0.033256 dB teacher PSNR. Both
trainers stopped at their sealed gates; later overshoot states are ineligible.

G3 appears locally saturated under scalar corrections to existing scale and
bias parameters. The next still-image hypothesis must add a genuinely new but
zero-initialized output-space basis, such as a tiny depthwise residual filter,
so its step-zero output is exactly G3 and any measured gain is attributable to
new representational capacity rather than perturbing the selected trunk.

## 2026-08-01 — G3 published and deployed

The verified G3 package was published to the existing Hugging Face model at
commit `b357304af178a6e31a5482958f1adf702f99be82`. The public Space was updated
and then pinned to that immutable model revision at Space commit
`22dd81eb6ff705c8724f2ee33f3f7bb576e49b9d`, preventing cached `main` downloads
from retaining the prior artifact. Immutable download verification reproduced
the local hashes for safetensors, portable ONNX gzip, and packed native bundle.

BackdropKit Cloudflare Worker version
`a559a467-dc53-4b91-82e5-10a2f628c40c` now serves the exact same
808,002-byte portable graph from `model-v3.bin`. The file downloaded back from
the public apex hashed to `a5e32c6b853f32e6c2fd39615ff27f7780e076fdac504dc3f2be1e822e3e085c`,
identical to the immutable Hugging Face artifact. The stable product model ID
and both older cache-compatibility artifacts remain intact.

## 2026-08-01 — Hugging Face Space main-thread freeze repaired

The deployed Space reproduced a misleading freeze on ordinary screenshots.
Model initialization and every ONNX tile ran on the page's main JavaScript
thread. The 128x128 benchmark could block the interface for seconds; a normal
multi-tile screenshot left the iframe unable to repaint progress or accept
Cancel until inference yielded.

The Space now creates the ONNX Runtime session and executes every tile in a
dedicated module worker. Planar input and reconstruction buffers transfer
between the page and worker without structured-clone copies. WebGPU remains the
preferred execution provider and WASM remains the automatic fallback. The page
adds an immediate per-tile status, a one-second heartbeat, responsive
cancel-after-current-tile behavior, and a 180-second request watchdog so a bad
driver cannot wait forever. Versioned `app.js` and worker URLs prevent returning
browsers from retaining the old main-thread runtime.

The cache-safe repair is public at Space commit
`9896ce694616a8294fc566cd68f3c4a873a33d8c`. On the live static origin, the
WASM benchmark completed and enabled the Clark PNG download. A 582x720 product
UI screenshot entered a nine-tile job, updated its heartbeat while tile one was
running, accepted Cancel in approximately 0.5 seconds, and stopped cleanly after
that tile. Browser console verification found no warning or error. The G3 model
revision and artifact hashes are unchanged.
