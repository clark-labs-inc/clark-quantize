# Scientific journal

## 2026-07-31 — Allocation-method audit

### Question

Is isolated output fragility plus an exact storage-budget optimizer a defensible
way to choose heterogeneous scale-codebook precision for the 204 Swin2SR
surfaces?

### Primary-source evidence

MixFrag (`arXiv:2607.28589v1`) measures each vision-transformer component at
each candidate precision while every other component remains at the reference
precision, then minimizes summed output fragility with a Multiple-Choice
Knapsack Problem. This directly supports the sealed study's isolation and exact
budget structure. Its calibration output is a classifier probability
distribution, whereas Swin2SR produces pixels; the sealed adaptation therefore
uses normalized output and Sobel-edge error rather than classification KL.

Primary source: https://arxiv.org/abs/2607.28589v1

LampQ (`arXiv:2511.10004v2`, AAAI 2026) predates MixFrag and identifies three
failure modes of metric-based mixed precision: coarse granularity, incomparable
metric scale across component types, and quantization-unaware allocation. The
exact 1,687,945-byte arXiv source archive inspected here has SHA-256
`e36ef100b0965abf30754794b136caf9c6153d11ae69c3ac93478b4f59be13af`.
LampQ initializes allocation from type-scaled Fisher traces and an integer
program. It then repeatedly exchanges one bit from the layer with the smallest
estimated degradation to the layer with the largest estimated gain, accepting
swaps only while calibration accuracy improves. This is independent warning
that a one-pass additive allocation can be wrong when surfaces interact.

Primary source: https://arxiv.org/abs/2511.10004v2

### Current design disposition

The sealed v1 calibration already addresses granularity by operating on every
weight surface, metric scale by normalizing output and edge MSE to the reference
energy, and quantization awareness by measuring each actual q1/q2/q4/q8/FP16
scale representation rather than using parameter count as a proxy. It does not
solve interactions between simultaneously changed surfaces.

Consequently, its MCKP output remains a candidate, never a finding. The combined
allocation must be materialized and evaluated as a whole model. If measured
fragility materially disagrees with the additive prediction, v1 is falsified as
a sufficient allocator. A later, separately sealed study may use exact-byte
precision swaps around the selected mixed-precision state, accepting a swap
only when the reference-output proxy improves on a disjoint calibration shard.
Unlike LampQ, it must not tune against BackdropKit ground truth or development
metrics. That change is not added to v1 after sealing.

### Evidence boundary

MixFrag v1 was submitted on 2026-07-30 and is not treated as settled evidence.
LampQ's exact v2 TeX source was inspected, but its classification accuracy
acceptance rule is not imported into this ground-truth-protected study. Neither
paper studies ternary super-resolution scale channels or provides evidence that
a chosen allocation will close BackdropKit's q4 quality gap.

## 2026-07-31 — Combined interactions confirm both heterogeneous allocations

The per-surface allocations were materialized together only after a separate
two-shard protocol was sealed. The first shard exactly reproduces the 64
training-only records used for allocation; the second contains 64 deterministic,
source-balanced training records with no path overlap. Development and final
heldout remained untouched.

The 10,012-byte allocation's combined fragility is `0.009976330` on the
allocation shard and `0.010665895` on disjoint confirmation. That is `94.817%`
and `94.795%` below uniform q2. Its measured-to-summed-isolated ratio is
`0.966596`, so the additive predictor is accurate within about `3.34%` rather
than hiding a destructive interaction.

The 20,024-byte allocation also passes: combined fragility is `0.001448983`
and `0.001490167`, respectively `86.496%` and `87.267%` below uniform q4.
Its interaction ratio is `1.310570`; the additive estimate is optimistic by
about `31.1%`, but the whole-model candidate still clears the frozen 10% gate
by a wide margin.

The smallest-passing rule therefore selects the 10,012-byte allocation for a
separately sealed progressive-QAT study. It covers all 204 surfaces and its
calibration storage and physical browser-aligned span are both exactly 10,012
bytes. It is genuinely different from the existing equal-byte category bridge:
73 surfaces change precision (50 q2-to-q1, 19 q1-to-q2, two q2-to-q4, one
q1-to-q4, and one q4-to-q8).

### Storage correction

The completed calibration exposed a four-byte preflight error for uniform q8:
65,232 observed calibration bytes versus 65,228 estimated. Its physical
separately aligned browser span is 65,548 bytes. q8 is diagnostic and does not
affect selection; both advancing allocations were recomputed from their actual
per-surface codebooks and pass physical byte accounting.

### Receipts

- interaction protocol SHA-256:
  `608e408a7f992dc529979b1ce903001acfea99f7e22bb01b71553c7b8169936e`
- interaction result SHA-256:
  `141e55aad9b9b3a98778b4b6d62696f126ac42453f55a794ecd56d60d4724b2a`
- selected map surface-receipt SHA-256:
  `c7a12a9cbb5bc0e34d4c8ba0bfd83e7f6209a7bb034973c184fc18f0530233bc`
- result: `mixfrag_10012` selected for QAT; no model promotion or final-heldout
  access authorized.

## 2026-07-31 — Per-surface progressive QAT sealed and launched

The selected MixFrag allocation now has a versioned 204-surface target-map
contract and a resumable scale-codebook training path. A sealed dry run proved
exact recovery from parent SHA-256
`0b155f9ada8b747544ba1abb83596e160927fdc91a4c33bf36879ff3f77ce00f`,
exact continuation of the 206,500-record balanced sampler, and the intended
`174 q1 / 25 q2 / 4 q4 / 1 q8` surface count. The stored scale budget remains
exactly 10,012 bytes.

The initial mechanism control is frozen at 4,000 MPS steps, batch size eight,
AdamW, master learning rate `1e-6`, center learning rate `1e-5`, progressive
category scheduling, and a periodic hard-shadow loss. It is not an optimizer
selection study and cannot authorize promotion.

The step-one receipt is numerically healthy: all matrix, convolution,
correction, and 768 scale-center gradients are finite. This only establishes
that the new per-surface QAT mechanism is live. It is not quality evidence;
development evaluation, exact artifact export, browser parity, robustness, and
final-heldout remain pending or locked as preregistered.

### Receipts

- study manifest SHA-256:
  `071ded64168e5db7c08a3791369809b96375e082af9778561790bb2ed3c4adfc`
- target map SHA-256:
  `6b03989221024733de40460ca6611371fd9e13e4256dd7fbc66e4a98135a52f4`
- status: training active; no MixFrag-QAT result or model promotion exists yet.

Before the first step-500 checkpoint, a separate decision protocol was sealed
at SHA-256
`8c0d402d959a08a8215442436bf8fba6afafd626efb7b5273654bcbdd0c6d057`.
It prevents three distinct claims from being conflated. Mechanism advancement
requires exact 10,012-byte scale storage, official paired and stratified
noninferiority versus the refined parent, and positive mean ground-truth PSNR
versus both the released Ternary Lite and the prior sensitive bridge. Becoming
the quality leader additionally requires positive ground-truth and edge-PSNR
deltas versus the independently confirmed three-seed all-ternary mean. Official
q4 gates remain separate and unchanged. Failure of any stronger gate cannot be
rebranded as promotion evidence.

### Step-500 trajectory and strongest-parent transfer preflight

The first major checkpoint remained finite across every parameter group. Promo
validation rose from the 35.827333 dB start to 37.428807 dB at step 500; the
forced-hard shadow loss was 0.021536 and its transition fraction was 0.3125.
These four screenshots remain a trajectory diagnostic and do not alter the
sealed 3,040-sample decision rule.

In parallel, the parent loader was extended to accept the confirmed all-ternary
image-I/O recovery checkpoint without flattening its training provenance. A CPU
roundtrip of the seed-20260733 leader reconstructed all 484 tensors and
1,005,348 values exactly: zero differing tensors, maximum absolute difference
zero, and matching tensor SHA-256
`839190220e78790a5a25d79957ee812b5689f299a2767ad71eef9757434b61b1`.
An inspect-only scale-QAT preflight then proved exact sampler resume and exact
204-surface target-map coverage from retained-parent checkpoint SHA-256
`22350f29385e06097500ecbf62094f3ac6c20c39e3add6839051a5fd1531d824`.
This enables, but does not yet authorize, transferring a successful MixFrag
mechanism to the current all-ternary quality leader.

## 2026-07-31 — Strongest-parent transfer frozen before the causal result

At mechanism step 2,000, before any full-development result exists, a
conditional transfer protocol was frozen for the seed-20260733 fully ternary
leader. The protocol preserves the exact 204-surface map, 10,012-byte scale
budget, 4,000-step schedule, batch size eight, learning rates, hard-shadow
schedule, 206,500-image data contract, and 3,040-sample development bundle.
It permits no post-result surface reselection or hyperparameter tuning.

The transfer remains unauthorized unless the already sealed causal decision
exists, matches decision-protocol SHA-256
`8c0d402d959a08a8215442436bf8fba6afafd626efb7b5273654bcbdd0c6d057`,
and sets `mechanism_advances` to true. This keeps the current refined-parent
run responsible for establishing the representation mechanism; the 34.000 dB
parent can only test whether that frozen mechanism composes with the stronger
all-ternary state.

### Receipt

- conditional transfer protocol SHA-256:
  `ace1d0d2129ba2f69fff34c5a1395cd8c0f98802cbe04922c3c297a5912e0084`
- status: preregistered only; no transfer study is sealed or launched.

## 2026-08-01 — Final disposition

No MixFrag transfer writer is active. The allocation study remains useful
representation evidence, but it did not become the selected product artifact.
G3 supersedes it as the current still-image release. Any future scale-allocation
study must start in a new dated folder and compare from the exact G3 state;
this preregistration is not silently reused.
