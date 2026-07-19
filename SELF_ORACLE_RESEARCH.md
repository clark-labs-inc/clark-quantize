# Self-Oracle Quantization Research

## Current Read

The VLM judging path is not ready for automation. It was useful as a prompt
for rubrics, but the first bakeoff failed the acceptance gate because the cheap
judges were too position-sensitive or too tie-heavy.

The teacher-native self-oracle path passed its first feasibility gate. The
existing heads:

- `path`: teacher/student transition agreement on student states
- `fixed`: one-step teacher correction consistency
- `geometry`: hidden-token CKA/relational similarity

ranked the science-fresh branch above the old champion. The weakness is that
these heads mostly ask whether a candidate lies on a trajectory the FP16 teacher
recognizes. That can leave known visual failures too close to strong checkpoints.

The missing heads are:

- prompt-contrastive semantics
- object/relationship grounding
- deterministic text/OCR or text-region scoring

## Experiment Results: 2026-06-25

Live training was stopped cleanly at `s43679`. The latest render before stop was
`/workspace/release_pool_clark_air/render_s43500.png`; it remains a strong
science-fresh checkpoint but the horse/person prompt is still wrong enough that
the oracle must not be reduced to a single trajectory score.

Three offline self-oracle experiments were run:

1. `rectified-sigma` final-latent energy failed the FP16 gate. FP16 produced
   semantic critical failures, so this noising convention is invalid for the
   current Sana ranker.
2. `trajectory-margin` energy passed the FP16 gate and downranked bad replay,
   but kept `bellman_final` too close to science-fresh checkpoints.
3. Counterfactual intervention grounding passed the FP16 gate and improved the
   known ranking:

```text
FP16              94.78
science_s42750    67.06
science_s40000    67.00
science_s43679    66.57
bellman_final     66.18
science_s43500    66.17
qat8_latest       65.37
old_s5000         64.13
bad_replay_q1     62.71
bad_replay_q2     59.78
```

The important result is not the exact scalar. It is that adding grounding moves
`bellman_final` below the stronger science-fresh checkpoints instead of tying
them. The limitation is that `grounding_min` is still compressed into a narrow
band (`bad_replay_q2=0.8666`, `science_s43679=0.8966`, `FP16=1.0`), so this is
a ranking head, not yet a reliable hard-failure detector for glass/person/harp
mistakes.

Stability validation across three independent rollout seed bases failed the
autonomous-selection gate:

```text
mean rank / ranks across seeds

FP16              1.00   1,1,1
science_s42750    3.00   2,5,2
science_s40000    3.67   6,2,3
bellman_final     3.67   3,3,5
science_s43500    4.67   4,4,6
science_s43679    5.67   7,6,4
qat8_latest       6.67   5,8,7
old_s5000         7.67   8,7,8
bad_replay_q1     9.00   9,9,9
bad_replay_q2    10.00   10,10,10
```

Bellman/science inversion rate:

```text
7 / 12 = 58.33%
```

Decision: the current teacher-native oracle is useful as a supporting offline
diagnostic, but it failed as an autonomous promotion or rate-allocation gate.
Do not use it to select binary spillover modules yet. The missing signal is
still prompt-specific visible semantics: exact text/OCR, glass presence,
person/horse relation, and harp-string structure.

## Experiment Results: Counterfactual Vector-Field Oracle

Implemented `Counterfactual Vector-Field` coefficients in
`scripts/self_oracle_sana_rank.py` behind:

```bash
--cvf-records 4
--cvf-threshold 0.50
--cvf-sep-min 1e-4
--cvf-cvar-frac 0.20
```

Also added a CVF-specific prompt suite:

```text
data/sana_cvf_hard_prompts.json
```

Important calibration fix: the CVF positive branch must be the exact rollout
prompt. Earlier smoke tests failed because assertion-level positive prompts did
not match `/workspace/shared_hard.json`, and that made FP16 fail its own
coefficient sanity check. The implementation now uses the actual prompt string
as `c+` and only uses the JSON assertions for counterfactual negatives.

After calibration, FP16 passes:

```text
FP16 cvf_min = 1.0000
FP16 failures = 0
```

Full archive stability with `cvf_threshold=0.50`:

```text
Bellman/science inversion rate: 4 / 12 = 33.33%
```

This is better than the previous scalar oracle's `7 / 12 = 58.33%`, but it
still fails the requested gate (`<= 1 / 12`). A threshold sweep found one
apparent rescue at `0.55`, but only when scalar path score is used as a
tie-breaker, and that setting promotes `bad_replay_q1` as the top non-FP16
candidate on one seed. With stricter ordering:

```text
failure count -> CVF aggregate -> scalar quality
```

the sweep still fails:

```text
threshold 0.50: Bellman/science inversions = 4 / 12
threshold 0.55: Bellman/science inversions = 3 / 12
threshold 0.60: Bellman/science inversions = 4 / 12
```

Decision: CVF is calibrated and directionally better, but the current assertion
set and coefficient aggregation do not pass the offline gate. Do not launch
CVF-QAT yet.

Artifacts:

```text
/workspace/self_oracle_experiments/energy_trajectory_rank.md
/workspace/self_oracle_experiments/grounding_full_rank.md
/workspace/self_oracle_experiments/grounding_full_rank.json
/workspace/self_oracle_experiments/stability/summary.md
/workspace/self_oracle_experiments/stability/grounding_seed_9000.json
/workspace/self_oracle_experiments/stability/grounding_seed_19000.json
/workspace/self_oracle_experiments/stability/grounding_seed_29000.json
/workspace/self_oracle_experiments/cvf/stability/cvf_seed_9000.json
/workspace/self_oracle_experiments/cvf/stability/cvf_seed_19000.json
/workspace/self_oracle_experiments/cvf/stability/cvf_seed_29000.json
/workspace/self_oracle_experiments/cvf/stability/threshold_sweep.md
```

## New Head 1: Flow Conditional Energy

Implemented in `scripts/self_oracle_sana_rank.py` behind:

```bash
--counterfactuals data/sana_self_oracle_counterfactuals.json
--energy-timesteps 8
--energy-noise 2
--energy-noising trajectory-margin
```

Two modes exist. The only currently valid mode is `trajectory-margin`, which
judges the candidate on its own denoising states and compares the FP16 teacher's
positive-prompt velocity against counterfactual-prompt velocities.

The older `rectified-sigma` mode asks the FP16 teacher whether a candidate final
latent `x0` is explained better under the true prompt than under minimal
counterfactual prompts:

```text
x_t = (1 - t) x0 + t noise
v*  = noise - x0
```

That mode is retained for research, but it failed the FP16 sanity gate in the
current pilot and must not be used for ranking until the Sana scheduler
convention is fixed.

The reported value is a conservative margin:

```text
min_j E_T(x0, negative_j) - E_T(x0, positive)
```

Margins less than or equal to zero are treated as critical semantic failures.
The ranker still outputs the old continuous heads so failures do not get hidden
inside one scalar.

## New Head 2: Intervention Grounding

Implemented in `scripts/self_oracle_sana_rank.py` behind:

```bash
--grounding-records 5
--grounding-threshold 0.85
```

The implementation uses output-delta maps:

```text
G_k(p) = ||v_T(x_t, full_prompt)_p - v_T(x_t, ablated_prompt)_p||
```

This asks where the FP16 teacher computation changes when a required phrase is
removed or replaced by a counterfactual. Candidate maps are compared against
FP16 maps with Jensen-Shannon map similarity.

It currently acts as a weak object/relation grounding prior for:

- wine glass separate from bottle
- person separate from horse
- harp strings inside the frame
- text/glyph region on the yellow wall

## Validation Gate

Before using the oracle for training or a critic, rerank this archive:

- FP16
- science_s29750
- science_s32000
- science_s40000
- science_s42750
- old_s5000
- bellman_final
- QAT8 composition drift
- q256 binary
- bad replay

Required gates:

```text
pairwise accuracy >= 85%
critical inversion rate <= 5%
Spearman correlation >= 0.75
```

Explicit checks:

- Bellman must lose to science on known object/relation defects.
- No-glass candidates must lose the wine object head.
- Merged horse/person candidates must lose the relation head.
- Generic vertical texture must not win the harp-string head.

Run three independent rollout seed bases. The current ranker changes materially
and fails Bellman/science inversion checks, so downstream rate allocation is
blocked until stronger visible-semantics heads are added.

## Product Use

Use the validated oracle first for binary/ternary rate allocation, not QAT
gradients:

```text
rho_j = Delta reward_j / Delta bits_j
```

Convert only low-damage modules/groups to binary, with hard constraints on
glass, person/horse relation, harp strings, and text.

Do not train a critic or start another QAT branch until the offline oracle
passes the critical-inversion tests.

## Caveat

Some early CTD negatives are confounded by alpha-schedule deadlock and
scale-gradient starvation. They are useful visual negatives, but not clean
causal evidence against every CTD-style idea.

## Stein-Jeffreys Rate-Distortion QAT

Decision after the CVF sweep: discard the current self-oracle as a scalar
selector. Its calibrated heads still inspect a finite semantic subspace, so
orthogonal visual failures can survive. The next experiment targets the
distributional object directly:

```text
low-bit conditional denoising path distribution ~= FP16 teacher path distribution
```

Implemented behind opt-in trainer flags in `scripts/train_sana_mwqat.py`:

```bash
JEFFREYS_ENABLE=1
STEIN_ENABLE=1
TRANSFORMER_GRAD_CKPT=1
```

Reusable helpers live in `clark_quantize/stein_qat.py`.

### Components

1. Symmetric path side:

```text
teacher-state path: existing velocity/m8 terms
student-state path: FP16 teacher judges states generated by the quantized student
```

The new `student_state_path_loss()` samples a student rollout state, runs the
frozen FP16 teacher on that off-bank state, and penalizes
`||v_Q(x_t^Q) - v_T(x_t^Q)||^2`.

2. Low-noise Stein transport:

For several student particles from the same prompt, compute the FP16 teacher
flow score using Sana's flow-sigma convention:

```text
x_sigma = (1 - sigma) x0 + sigma eps
v = eps - x0
score = -(x + (1 - sigma) v) / sigma
```

Then build an IMQ-kernel Stein/SVGD witness field with a fixed Rademacher/JL
projection and use it as a stop-gradient transport direction.

3. Codebook geometry:

Keep the existing small codebook attraction term active:

```text
CODEBOOK_LAMBDA=0.001
```

### Engineering Notes

Holding a second full Sana pipeline on A100 caused OOM. The trainer now loads
only the teacher transformer onto GPU and deletes the unused teacher pipeline
components. The student transformer also needs gradient checkpointing for this
branch:

```text
TRANSFORMER_GRAD_CKPT=1
```

Smoke test from `/workspace/release_pool_clark_air/ckpt_s42750.pt` passed with:

```text
Ljeff active
Lstein active
no OOM after gradient checkpointing
```

Current launched probe:

```text
TAG=sana_t112_stein_jeff_s42750_probe1
RESUME=/workspace/release_pool_clark_air/ckpt_s42750.pt
STEPS=500
PROX_RHO=0.02
RENDER_EVERY=100
SAVE_STEPS=1
FULL_CKPT_EVERY=250
HARDFILE=data/sana_cvf_hard_prompts.json
```

Pass gate:

- wall/text must not regress
- wine glass remains present and separate
- harp does not become a texture silhouette
- horse/person relation does not further degrade
- at least one hard-detail category improves

If it passes, estimate empirical Fisher from the same quality objective and use
rate-distortion water filling for binary/ternary group assignment. Do not use
CVF/Bellman scalar rank as the 8x selector.
